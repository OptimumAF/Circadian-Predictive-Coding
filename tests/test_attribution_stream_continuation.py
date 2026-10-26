"""Whole owner-local stream continuation with native and fake learner ports.

Native pairs compare uninterrupted six-wake runs against two real reconstructions.
Refusal controls use explicit fakes; detached observations never grant authority.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import build_stream, replay_policy
from src.app.coupled_attribution import AttributionStream
from src.app.stream_rehearsal import StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.core.shared_replay_schedule import SharedReplayBuffer
from src.shared.torch_runtime import require_torch


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def assert_owned_values(actual, expected, torch=None):
    if torch is not None and torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert actual.shape == expected.shape and torch.equal(actual, expected)
    elif type(expected) is np.ndarray:
        assert type(actual) is np.ndarray and actual.dtype == expected.dtype
        assert actual.shape == expected.shape and actual.tobytes() == expected.tobytes()
    elif type(expected) is SharedReplayBuffer:
        assert type(actual) is SharedReplayBuffer
        assert_owned_values(vars(actual), vars(expected), torch)
    elif is_dataclass(expected) and not isinstance(expected, type):
        assert type(actual) is type(expected)
        for field in fields(expected):
            assert_owned_values(getattr(actual, field.name), getattr(expected, field.name), torch)
    elif type(expected) is dict:
        assert type(actual) is dict and tuple(actual) == tuple(expected)
        for key in expected:
            assert_owned_values(actual[key], expected[key], torch)
    elif type(expected) in (tuple, list):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_owned_values(left, right, torch)
    else:
        assert type(actual) is type(expected) and actual == expected


def make_session(source, *, enabled=False, now=None, observe=None):
    now = [0.0] if now is None else now
    clock = lambda: now[0]
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=6, max_wall_seconds=60), clock)
    replay = None
    rehearsal = None
    if enabled:
        replay = ToyBudgetSession(
            ToyExecutionBudget(max_replay_examples=3, max_wall_seconds=60), clock
        )
        rehearsal = StreamRehearsal(
            source, wake, replay_policy(), replay_budget=replay, total_updates=6, input_dim=2
        )
    training, heldout = build_stream()
    stream = AttributionStream(
        source,
        training,
        heldout,
        checkpoint_updates=(0, 2, 4, 6),
        label_delay=1,
        budget=wake,
        observe_checkpoint=observe,
        rehearsal=rehearsal,
    )
    return stream, wake, replay, rehearsal, (training, heldout)


def native(method):
    return TorchFeatureLearner(TorchFeaturePolicy(method, 2, 3, 17, 0.03, 2, 0.04, 0.9))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_should_resume_the_complete_native_stream_at_two_boundaries_without_duplicate_work(
    monkeypatch, enabled, method
):
    torch = require_torch()
    global_rng = torch.get_rng_state().clone()
    numpy_rng = pickle.dumps(np.random.get_state())
    reference_source, paused_source = native(method), native(method)
    original_reference, original_paused = (
        reference_source.snapshot_state(),
        paused_source.snapshot_state(),
    )
    expected_states: dict[int, Any] = {}
    actual_states: dict[int, Any] = {}

    def observe_into(destination):
        def observe(updates, state, arrived):
            assert (arrived is None) == (updates == 0)
            if arrived is not None:
                assert arrived.role == "train"
            destination[updates] = deepcopy(state)

        return observe

    control, control_budget, control_replay, control_helper, control_data = make_session(
        reference_source, enabled=enabled, observe=observe_into(expected_states)
    )
    resumed, budget, replay, helper, data = make_session(
        paused_source, enabled=enabled, observe=observe_into(actual_states)
    )
    data_before = pickle.dumps((control_data, data))
    finish_allowed = [False]
    predict = TorchFeatureLearner.predict

    def final_only(self, features):
        assert finish_allowed[0], "final prediction was released before finish"
        return predict(self, features)

    monkeypatch.setattr(TorchFeatureLearner, "predict", final_only)
    control.run_until(6)
    expected_cursor = control._runtime._inbox.capture_cursor()
    expected_consolidation = control._runtime.capture_consolidation_cursor()
    expected_candidate = control.candidate_snapshot()
    finish_allowed[0] = True
    expected_report = control.finish()
    finish_allowed[0] = False
    runtime, actor, clock = resumed._runtime, resumed._runtime.actor, resumed._clock
    actor_before = actor.snapshot_state().state
    budget_limits = budget.budget
    replay_limits = None if replay is None else replay.budget
    for point in (2, 4):
        resumed.run_until(point)
        assert resumed.completed_updates == point and resumed.next_tick == point + 1
        candidate, inbox = runtime._candidate, runtime._inbox
        head = candidate._head
        buffer = None if helper is None else helper._buffer
        token = resumed.pause()
        observed = token.snapshot_state()
        assert {
            "candidate_state",
            "inbox_cursor",
            "rehearsal_state",
            "next_tick",
            "completed_updates",
            "checkpoint_states",
        }.issubset(observed)
        assert_owned_values(observed["candidate_state"], expected_states[point], torch)
        cursor = observed["inbox_cursor"]
        assert len(cursor.experiences) == point + 1
        assert len(cursor.labels) == len(cursor.applied) == point
        assert cursor.completed_updates == point
        assert observed["completed_updates"] == point and observed["next_tick"] == point + 1
        saved_observation = token.snapshot_state()
        observed["candidate_state"].head_state["weight_feature_hidden"].zero_()
        observed["inbox_cursor"].experiences[0].features.fill(999)
        assert_owned_values(token.snapshot_state(), saved_observation, torch)
        started_at, counters = (
            budget.started_at,
            (budget.updates_completed, budget.replay_examples_completed),
        )
        replay_count = None if replay is None else replay.replay_examples_completed
        resumed.resume(token)
        assert resumed._runtime is runtime and runtime.actor is actor and resumed._clock is clock
        assert resumed._budget is budget and budget.budget is budget_limits
        assert (
            budget.started_at == started_at
            and (budget.updates_completed, budget.replay_examples_completed) == counters
        )
        assert runtime._candidate is not candidate and runtime._candidate._head is not head
        assert runtime._inbox is not inbox and runtime._inbox._learner is runtime._candidate
        assert_owned_values(
            resumed.candidate_snapshot(), saved_observation["candidate_state"], torch
        )
        assert_owned_values(
            runtime._inbox.capture_cursor(), saved_observation["inbox_cursor"], torch
        )
        if enabled:
            assert helper is resumed._rehearsal and helper._buffer is not buffer
            assert replay is not None and replay.budget is replay_limits
            assert replay.replay_examples_completed == replay_count == point // 2
        else:
            assert observed["rehearsal_state"] is None
        assert_owned_values(actor.snapshot_state().state, actor_before, torch)
    resumed.run_until(6)
    assert_owned_values(resumed.candidate_snapshot(), expected_candidate, torch)
    assert_owned_values(runtime._inbox.capture_cursor(), expected_cursor, torch)
    # Each successful ownership reconstruction advances the runtime revision once;
    # every consumed attempt and consolidation receipt must otherwise stay exact.
    assert_owned_values(
        runtime.capture_consolidation_cursor(),
        replace(expected_consolidation, revision=expected_consolidation.revision + 2),
        torch,
    )
    assert set(actual_states) == set(expected_states) == {0, 2, 4, 6}
    for point in expected_states:
        assert_owned_values(actual_states[point], expected_states[point], torch)
    if enabled:
        assert control_helper is not None and helper is not None
        assert helper.report == control_helper.report
        assert helper.retention == control_helper.retention
        assert helper._buffer.rotation_cursor == control_helper._buffer.rotation_cursor
        assert replay is not None and control_replay is not None
        assert replay.replay_examples_completed == control_replay.replay_examples_completed == 3
    assert budget.updates_completed == control_budget.updates_completed == 6
    finish_allowed[0] = True
    actual_report = resumed.finish()
    assert actual_report == expected_report
    assert actual_report["completed_updates"] == 6 and actual_report["score_forward_calls"] == 8
    assert_owned_values(paused_source.snapshot_state(), original_paused, torch)
    assert_owned_values(reference_source.snapshot_state(), original_reference, torch)
    assert_owned_values(actor.snapshot_state().state, actor_before, torch)
    assert pickle.dumps((control_data, data)) == data_before
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng


class FakeLearner:
    def __init__(self, trace=None, state=None, flags=None):
        self.trace = [] if trace is None else trace
        self.state: dict[str, Any] = (
            {"updates": 0, "value": np.zeros(1)} if state is None else deepcopy(state)
        )
        self.flags = {} if flags is None else flags

    def fork(self):
        self.trace.append(("fork",))
        return FakeLearner(self.trace, self.state, self.flags)

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.trace.append(("restore",))
        self.state = deepcopy(state)
        if self.flags.get("restore_failure"):
            raise RuntimeError("declared restore failure")
        if self.flags.get("restore_corruption"):
            self.state["value"] += 1

    def train_batch(self, features, targets):
        self.trace.append(("train", len(features)))
        callback = self.flags.get("reenter")
        if callback is not None:
            callback()
        self.state["updates"] += 1
        self.state["value"] += float(features.sum() + targets.sum())
        if self.flags.get("train_failure") or (
            len(features) == 1 and self.flags.get("replay_failure")
        ):
            raise RuntimeError("declared training failure")
        return TrainingDiagnostic("continuation_fake_count", float(self.state["updates"]))

    def predict(self, features):
        assert self.flags.get("score_allowed"), "heldout scoring released early"
        self.trace.append(("predict", self.state["updates"]))
        return np.full((len(features), 1), 0.5 + self.state["updates"] / 100)


def fake_setup(*, enabled=True, point=2):
    source = FakeLearner()
    seconds = [0.0]
    session, budget, replay, helper, _ = make_session(source, enabled=enabled, now=seconds)
    if point:
        session.run_until(point)
    return session, source, budget, replay, helper, seconds


def train_count(source):
    return sum(event[0] == "train" for event in source.trace)


def test_should_keep_detached_pause_observation_data_separate_from_the_original_capability():
    stream, source, budget, replay, helper, _ = fake_setup()
    token = stream.pause()
    before = token.snapshot_state()
    observation = token.snapshot_state()
    observation["completed_updates"] = 99
    observation["candidate_state"]["value"].fill(999)
    observation["inbox_cursor"].labels[0].targets.fill(999)
    assert_owned_values(token.snapshot_state(), before)
    with pytest.raises(ValueError):
        stream.resume(observation)
    assert budget.updates_completed == 2 and replay.replay_examples_completed == 1
    assert train_count(source) == 3


@pytest.mark.parametrize("target", [True, 0, -1, 7, 1.5])
def test_should_refuse_invalid_absolute_wake_target_before_training(target):
    stream, source, _, _, _, _ = fake_setup(point=0)
    before = stream.candidate_snapshot()
    with pytest.raises(ValueError):
        stream.run_until(target)
    assert train_count(source) == 0
    assert_owned_values(stream.candidate_snapshot(), before)


def test_should_refuse_early_or_repeated_finish_without_releasing_scoring():
    stream, source, _, _, _, _ = fake_setup(enabled=False, point=0)
    with pytest.raises(ValueError):
        stream.finish()
    assert not any(event[0] == "predict" for event in source.trace)
    # Use a fresh owner after the refused request; no failed capability is renewed.
    stream, source, _, _, _, _ = fake_setup(enabled=False, point=6)
    source.flags["score_allowed"] = True
    stream.finish()
    count = len(source.trace)
    with pytest.raises(ValueError):
        stream.finish()
    assert len(source.trace) == count


def test_should_refuse_foreign_and_consumed_pause_tokens_before_further_training():
    first, source, _, _, _, _ = fake_setup()
    second, other, _, _, _, _ = fake_setup()
    token, foreign = first.pause(), second.pause()
    before = train_count(source), train_count(other)
    with pytest.raises(ValueError):
        first.resume(foreign)
    assert (train_count(source), train_count(other)) == before
    # A separate valid owner establishes consumption independently of the refusal.
    third, own, _, _, _, _ = fake_setup()
    valid = third.pause()
    third.resume(valid)
    count = train_count(own)
    with pytest.raises(ValueError):
        third.resume(valid)
    assert train_count(own) == count
    assert token is not foreign


@pytest.mark.parametrize("field", ["completed", "native", "inbox"])
def test_should_refuse_changed_private_token_data_without_using_it_as_restore_authority(field):
    stream, source, budget, replay, _, _ = fake_setup()
    token = stream.pause()
    if field == "completed":
        token._snapshot["completed_updates"] += 1
    elif field == "native":
        token._snapshot["candidate_state"]["value"] += 1
    else:
        token._snapshot["inbox_cursor"].experiences[0].features.fill(999)
    before = train_count(source)
    with pytest.raises(ValueError):
        stream.resume(token)
    assert train_count(source) == before
    assert budget.updates_completed == 2 and replay.replay_examples_completed == 1


@pytest.mark.parametrize(
    "fault",
    [
        "budget_identity",
        "limits",
        "counter",
        "started",
        "clock",
        "revision",
        "native",
        "replay_counter",
        "replay_limits",
    ],
)
def test_should_refuse_changed_live_state_or_renewed_original_budget_before_any_more_training(
    fault,
):
    stream, source, budget, replay, _, _ = fake_setup()
    token = stream.pause()
    if fault == "budget_identity":
        stream._budget = ToyBudgetSession(deepcopy(budget.budget), budget.clock)
    elif fault == "limits":
        object.__setattr__(budget.budget, "max_training_updates", 7)
    elif fault == "counter":
        budget.updates_completed = 0
    elif fault == "started":
        budget.started_at -= 1
    elif fault == "clock":
        budget.clock = lambda: 0.0
    elif fault == "revision":
        stream._runtime._revision += 1
    elif fault == "native":
        stream._runtime._candidate.state["value"] += 1
    elif fault == "replay_counter":
        replay.replay_examples_completed = 0
    else:
        object.__setattr__(replay.budget, "max_replay_examples", 4)
    count = train_count(source)
    with pytest.raises(ValueError):
        stream.resume(token)
    assert train_count(source) == count
    assert not any(event[0] == "predict" for event in source.trace)


def test_should_keep_the_original_deadline_spent_when_resuming():
    stream, source, budget, replay, _, seconds = fake_setup()
    token = stream.pause()
    started = budget.started_at
    seconds[0] = 60.0
    count = train_count(source)
    with pytest.raises(RuntimeError):
        stream.resume(token)
    assert budget.started_at == started
    assert budget.updates_completed == 2 and replay.replay_examples_completed == 1
    assert train_count(source) == count


@pytest.mark.parametrize("boundary", ["advance", "finish"])
def test_should_refuse_reset_wake_work_without_rehearsal_before_training_or_scoring(boundary):
    point = 2 if boundary == "advance" else 6
    stream, source, budget, replay, helper, _ = fake_setup(enabled=False, point=point)
    assert replay is helper is None
    assert budget.updates_completed == point
    source.flags["score_allowed"] = True
    count = train_count(source)
    candidate = stream._runtime._candidate.snapshot_state()
    budget.updates_completed = 0
    with pytest.raises(ValueError):
        if boundary == "advance":
            stream.run_until(6)
        else:
            stream.finish()
    assert train_count(source) == count
    assert_owned_values(stream._runtime._candidate.snapshot_state(), candidate)
    assert not any(event[0] == "predict" for event in source.trace)
    with pytest.raises(ValueError):
        stream.finish()


@pytest.mark.parametrize("failure", ["raise", "corrupt_return"])
def test_should_stop_after_private_restore_failure_without_training_or_resetting_allowance(
    failure,
):
    stream, source, budget, replay, helper, _ = fake_setup()
    token = stream.pause()
    before = train_count(source)
    candidate, inbox, buffer = stream._runtime._candidate, stream._runtime._inbox, helper._buffer
    native_before, cursor_before = candidate.snapshot_state(), inbox.capture_cursor()
    buffer_before = pickle.dumps(buffer)
    source.flags["restore_failure" if failure == "raise" else "restore_corruption"] = True
    expected_error = RuntimeError if failure == "raise" else ValueError
    with pytest.raises(expected_error):
        stream.resume(token)
    assert stream._runtime._candidate is candidate and stream._runtime._inbox is inbox
    assert helper._buffer is buffer and pickle.dumps(buffer) == buffer_before
    assert_owned_values(candidate.snapshot_state(), native_before)
    assert_owned_values(inbox.capture_cursor(), cursor_before)
    assert train_count(source) == before
    assert budget.updates_completed == 2 and replay.replay_examples_completed == 1
    assert not any(event[0] == "predict" for event in source.trace)
    with pytest.raises(ValueError):
        stream.run_until(4)


@pytest.mark.parametrize("failure", ["wake", "replay", "reentrant"])
def test_should_stop_uncertain_native_work_and_never_release_final_scores(failure):
    stream, source, budget, replay, _, _ = fake_setup(point=0)
    if failure == "wake":
        source.flags["train_failure"] = True
    elif failure == "replay":
        source.flags["replay_failure"] = True
    else:
        source.flags["reenter"] = stream.pause
    with pytest.raises((ValueError, RuntimeError)):
        stream.run_until(2)
    count = train_count(source)
    with pytest.raises(ValueError):
        stream.run_until(4)
    with pytest.raises(ValueError):
        stream.pause()
    with pytest.raises(ValueError):
        stream.finish()
    assert train_count(source) == count
    assert not any(event[0] == "predict" for event in source.trace)
    if failure == "replay":
        assert budget.updates_completed == 2 and replay.replay_examples_completed == 1
    else:
        assert budget.updates_completed == 0
