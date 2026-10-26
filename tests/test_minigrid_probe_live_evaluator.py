"""Typed count endpoints through scripted ports/games, never native execution.

Literal complete count snapshots exercise admission and full-state isolation.
Scripted probabilities/endings test routing and paid-prefix behavior, not count
prediction, physical game correctness, adaptation or scientific benefit.
"""

import builtins
from copy import deepcopy
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryEntry, ActionMemoryPolicy, ActionMemorySnapshot
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID


def snapshot(native=16, *, cap=64, payload=155136, rows=None, fifo=False):
    rows = 9 * native if rows is None else rows
    entry = ActionMemoryEntry((-1.0,) * 296, (rows, 0, 0, 0, 0, 0, 0))
    native_state: ActionMemorySnapshot | FifoActionMemorySnapshot
    if fifo:
        fifo_policy = FifoActionMemoryPolicy(
            feature_dim=296, num_classes=7, max_entries=cap, max_payload_bytes=payload
        )
        native_state = FifoActionMemorySnapshot(fifo_policy, (entry,), native, rows, 0, 0)
    else:
        exact_policy = ActionMemoryPolicy(
            feature_dim=296, num_classes=7, max_entries=cap, max_payload_bytes=payload
        )
        native_state = ActionMemorySnapshot(exact_policy, (entry,), native, rows)
    return InvariantMemorySnapshot(InvariantMemoryPolicy(native_state.policy), native_state)


class ScriptedPort:
    def __init__(self, state, work, *, child=False):
        self.value = copy_invariant_memory_snapshot(state)
        self.work = work
        self.child = child

    def snapshot_state(self):
        hook = self.work.get("on_snapshot")
        if hook is not None:
            hook(self)
        return deepcopy(self.value)

    def fork(self):
        self.work.setdefault("forks", []).append(self)
        if self.work.get("identity_fork"):
            return self
        child = ScriptedPort(self.value, self.work, child=True)
        self.work["scorer"] = child
        return child

    def restore_state(self, state):
        self.value = copy_invariant_memory_snapshot(state)
        self.work.setdefault("restores", []).append(self)
        hook = self.work.get("on_restore")
        if hook is not None:
            hook(self)

    def predict(self, features):
        assert features.dtype == np.float64 and features.shape == (1, 296)
        self.work.setdefault("attempts", []).append(features.copy())
        if self.work.get("prediction_error"):
            raise RuntimeError("scripted prediction failed")
        result = np.full((1, 7), 1 / 7, dtype=np.float64)
        self.work.setdefault("returns", []).append(features.copy())
        hook = self.work.get("on_predict")
        if hook is not None:
            hook(self)
        if self.work.get("bad_probabilities"):
            result[0, 0] = np.nan
        return result

    def train_batch(self, *args):
        pytest.fail("live evaluator reached training")


class ScriptedEnvironment:
    work: dict[str, Any] = {}

    def __init__(self, layout, cue, *, seed, max_steps, role, episode_id):
        assert seed == 17 and max_steps == 16
        self.layout, self.cue, self.role = layout, cue, role
        self.identity = episode_id
        self.steps = self.attempts = 0
        self.transitions = []
        self.closed = False
        self.work.setdefault("environments", []).append(self)

    @property
    def observation(self):
        # Valid normalized public codes; each game retains its own cue frame.
        values = [-1.0] * 148
        values[0] = 0.2  # object6
        values[1] = -1.0 if self.cue == "blue" else -0.6
        values[3] = -0.8 if self.layout.context == "A" else -0.4
        values[-1] = 2 * (self.steps % 4) / 3 - 1
        return tuple(values)

    @property
    def legal_actions(self):
        return tuple(range(7))

    @property
    def terminal(self):
        return self.steps == 2

    def step(self, action):
        self.attempts += 1
        if self.work.get("step_error"):
            raise RuntimeError("scripted physics failed")
        assert action == 0  # uniform probabilities, first argmax; no repair
        self.transitions.append(dict(step=self.steps, action=action, legal=True))
        self.steps += 1

    def close(self):
        self.closed = True
        if self.work.get("close_error"):
            raise RuntimeError("scripted close failed")

    @property
    def report(self):
        if self.closed and self.work.get("readback_error"):
            raise RuntimeError("scripted report readback failed")
        return dict(
            status="complete" if self.terminal else "failed",
            terminated=self.terminal,
            truncated=False,
            total_return=0.5 if self.terminal else 0.0,
            environment_steps_attempted=self.attempts,
            environment_steps=self.steps,
            failure_work_may_be_unobserved=self.attempts > self.steps,
            transitions=deepcopy(self.transitions),
        )


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    monkeypatch.setattr(ScriptedEnvironment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", ScriptedEnvironment)

    def forbidden(*args, **options):
        pytest.fail("scripted engineering test reached native work or Torch comparator")

    for kind in (ActionMemoryLearner, FifoActionMemoryLearner, InvariantActionMemoryLearner):
        monkeypatch.setattr(kind, "predict", forbidden)
        monkeypatch.setattr(kind, "train_batch", forbidden)
    monkeypatch.setattr(evaluation, "_same_state", forbidden)
    original_import = builtins.__import__

    def no_torch(name, *args, **options):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("memory endpoint evaluation imported Torch")
        return original_import(name, *args, **options)

    monkeypatch.setattr(builtins, "__import__", no_torch)

    def build(source_updates=0, phase="after_probe"):
        native = source_updates + (16 if phase == "after_probe" else 32)
        state = snapshot(native)
        source = ScriptedPort(state, work)
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=0, max_replay_examples=0), lambda: 10.0
        )
        result = SimpleNamespace(
            work=work,
            source=source,
            state=state,
            budget=budget,
            source_updates=source_updates,
            phase=phase,
        )

        def boundary():
            work["boundaries"] = work.get("boundaries", 0) + 1
            hook = work.get("on_boundary")
            if hook is not None:
                hook(result)

        result.boundary = boundary
        return result

    return build


def evaluate(current, **changes):
    options = dict(
        source_updates=current.source_updates,
        phase=current.phase,
        budget=current.budget,
        check_boundary=current.boundary,
    )
    options.update(changes)
    return evaluation.evaluate_memory_probe_checkpoint(current.source, current.state, **options)


@pytest.mark.parametrize("origin", [0, 128])
@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
def test_should_use_all_four_actual_endpoints_and_only_fixed_a_f_games(case, origin, phase):
    current = case(origin, phase)
    budget = current.budget
    before = pickle.dumps(
        (
            current.state,
            current.source.value,
            budget.budget,
            budget.started_at,
            budget.updates_completed,
            budget.replay_examples_completed,
        )
    )
    clock = budget.clock
    report = evaluate(current)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    relative = 16 if phase == "after_probe" else 32
    assert (report["source_updates"], report["relative_updates"], report["phase"]) == (
        origin,
        relative,
        phase,
    )
    assert report["updates"] == origin + relative and report["native_training_examples"] == 9 * (
        origin + relative
    )
    assert report["model_encoding"] == INVARIANT_HISTORY_ENCODING_ID
    assert report["method"] == "public_invariant_observation_memory" and report["scores"] == []
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
        ("F-blue", "blue", "probe_test"),
        ("F-red", "red", "probe_test"),
    ]
    assert all(slot["status"] == "complete" and slot["success"] for slot in report["episodes"])
    assert report["work"]["live_prediction_calls_returned"] == 8
    assert report["work"]["environment_steps_returned"] == 8
    assert report["work"]["internal_projection_array_bytes_returned"] == 8 * 2368
    assert len(current.work["forks"]) == len(current.work["restores"]) == 1
    assert len(current.work["environments"]) == 4 and all(
        env.closed for env in current.work["environments"]
    )
    assert budget.clock is clock
    assert (
        pickle.dumps(
            (
                current.state,
                current.source.value,
                budget.budget,
                budget.started_at,
                budget.updates_completed,
                budget.replay_examples_completed,
            )
        )
        == before
    )
    for slot in report["episodes"]:
        first = slot["predictions"][0]["observation"]
        for decision in slot["predictions"]:
            assert decision["features"][:148] == first
            assert decision["features"][148:] == decision["observation"]
            assert decision["action"] == 0


class TextSubclass(str):
    pass


@pytest.mark.parametrize(
    "changes",
    [
        {"source_updates": True},
        {"source_updates": np.int64(0)},
        {"source_updates": 32},
        {"phase": "after_probe_extra"},
        {"phase": TextSubclass("after_probe")},
        {"phase": None},
        {"check_boundary": None},
    ],
)
def test_should_refuse_invalid_labels_before_guards_fork_or_environment(case, changes):
    current = case()
    with pytest.raises(ValueError):
        evaluate(current, **changes)
    assert current.work == {}


@pytest.mark.parametrize(
    "fault", ["raw", "fifo", "cap", "payload", "clock", "rows", "extra", "encoding", "noncanonical"]
)
def test_should_bind_full_exact_memory_snapshots_before_resources(case, fault):
    current = case()
    if fault == "raw":
        current.state = current.state.native_state
    elif fault == "fifo":
        current.state = snapshot(fifo=True)
    elif fault == "cap":
        current.state = snapshot(cap=63)
    elif fault == "payload":
        current.state = snapshot(payload=155135)
    elif fault == "clock":
        current.state = snapshot(32)
    elif fault == "rows":
        current.state = snapshot(rows=145)
    elif fault == "extra":
        object.__setattr__(current.state, "extra", 1)
    elif fault == "encoding":
        object.__setattr__(current.state.policy, "encoding_id", "unbound")
    else:
        entry = current.state.native_state.entries[0]
        object.__setattr__(entry, "features", (*entry.features[:-1], 1.0))
    with pytest.raises(ValueError):
        evaluate(current)
    assert current.work == {}


@pytest.mark.parametrize("training,replay", [(1, 0), (0, None), (0, 1), (0, True)])
def test_should_require_zero_update_and_replay_authority_before_work(case, training, replay):
    current = case()
    object.__setattr__(current.budget.budget, "max_training_updates", training)
    object.__setattr__(current.budget.budget, "max_replay_examples", replay)
    with pytest.raises(ValueError):
        evaluate(current)
    assert current.work == {}


@pytest.mark.parametrize(
    "fault",
    [
        "identity_fork",
        "prediction_error",
        "bad_probabilities",
        "step_error",
        "close_error",
        "readback_error",
    ],
)
def test_should_retain_partial_paid_work_and_unreached_slots(case, fault):
    current = case()
    current.work[fault] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current)
    report = caught.value.report
    assert report["status"] == "failed" and report["phase"] == "after_probe"
    assert len(report["episodes"]) == 4
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert all(slot["status"] == "not_run" for slot in report["episodes"][1:])
    if fault == "identity_fork":
        assert not current.work.get("environments")
        assert report["work"]["live_prediction_calls_attempted"] == 0
    else:
        assert len(current.work["environments"]) == 1 and current.work["environments"][0].closed
        assert report["episodes"][0]["status"] == "failed"
        assert report["work"]["live_prediction_calls_attempted"] >= 1
        assert report["work"]["live_prediction_calls_returned"] == (
            0
            if fault == "prediction_error"
            else 1
            if fault in ("bad_probabilities", "step_error")
            else 2
        )


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint", "budget", "source_clock"])
def test_should_refuse_changes_after_a_returned_prediction_before_physics(case, target):
    current = case()

    def mutate(scorer):
        if target == "budget":
            current.budget.updates_completed = 1
        elif target == "source_clock":
            current.source.value = snapshot(32)
        else:
            state = (
                current.source.value
                if target == "source"
                else scorer.value
                if target == "scorer"
                else current.state
            )
            entry = state.native_state.entries[0]
            counts = entry.counts
            object.__setattr__(entry, "counts", (counts[0] - 1, 1, *counts[2:]))

    current.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current)
    report = caught.value.report
    assert report["work"]["live_prediction_calls_attempted"] == 1
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert report["episodes"][0]["prediction_calls_returned"] == 1
    assert all(slot["status"] == "not_run" for slot in report["episodes"][1:])


def test_should_recheck_the_parent_after_the_last_snapshot_callback(case):
    current = case()
    changed = [False]

    def late_snapshot(port):
        if port.child and current.work.get("returns"):
            changed[0] = True

    def parent_boundary(owner):
        if changed[0]:
            raise ValueError("preceding paid parent report changed")

    current.work.update(on_snapshot=late_snapshot, on_boundary=parent_boundary)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current)
    assert caught.value.report["work"]["live_prediction_calls_returned"] == 1
    assert caught.value.report["work"]["environment_steps_returned"] == 0


def test_should_reject_an_incompatible_source_port_before_fork_or_games(case):
    current = case()
    current.source.value = snapshot(32)
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(current)
    assert current.work.get("boundaries", 0) > 0
    assert not current.work.get("forks") and not current.work.get("environments")


@pytest.mark.parametrize("updates", [16, 144, 160])
def test_should_leave_the_previous_representation_clock_contract_unchanged(case, updates):
    current = case()
    with pytest.raises(ValueError):
        evaluation.evaluate_representation_checkpoint(
            current.source,
            current.state,
            updates=updates,
            budget=current.budget,
            check_boundary=current.boundary,
        )
    assert current.work == {}
