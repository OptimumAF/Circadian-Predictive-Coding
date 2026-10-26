"""Owned public-cue schedules on scripted native ports, never game evidence."""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.adapters import torch_train_weighted_learner as weighted
from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters.minigrid_learning_pilot import _batch
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingUpdateFailed
from test_public_cue_recurrent_learner import ScriptedNative, policy
from test_public_cue_weighted_train import packet, scripted as weighted_native_fixture

scripted = weighted_native_fixture


@pytest.fixture
def scripted_public(scripted, monkeypatch):
    monkeypatch.setattr(weighted, "_same_state", _same_value)


def schedule():
    return tuple(
        replace(packet(cue), episode_id=f"weighted-{i}-{cue}")
        for i, cue in enumerate(("blue", "red", "blue"))
    )


def owner():
    return weighted.TorchTrainWeightedLearner(policy(), schedule())


def test_should_apply_original_delayed_stream_with_full_owned_heads_actor_and_receipts(
    scripted_public,
):
    source = owner()
    before = source.snapshot_state()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=3, max_wall_seconds=40), lambda: 0.0
    )
    stream = AttributionStream(
        source,
        tuple(_batch(p) for p in schedule()),
        (),
        training_only=True,
        label_delay=1,
        checkpoint_updates=(0, 1, 3),
        budget=budget,
    )
    stream.run_until(1)
    first = stream.candidate_snapshot()
    assert first.completed_updates == 1 and [h.completed_updates for h in first.heads] == [1, 0]
    stream.run_until(3)
    state = stream.candidate_snapshot()
    result = stream.finish()
    assert [h.completed_updates for h in state.heads] == [2, 1]
    assert [h.training_examples for h in state.heads] == [8, 4]
    assert [h.optimizer_state["momentum"][0] for h in state.heads] == [4, 2]
    assert len(result["applied_updates"]) == 3 and result["score_forward_calls"] == 0
    assert budget.updates_completed == 3 and budget.replay_examples_completed == 0
    assert _same_value(source.snapshot_state(), before)
    assert _same_value(stream._runtime.actor.snapshot_state().state, before)
    assert [c[0] for c in ScriptedNative.calls].count("weighted") == 3


@pytest.mark.parametrize(
    "fault", ["final_test", "probe_train", "probe_test", "missing", "mixed", "next", "extra"]
)
def test_should_refuse_invalid_whole_schedule_before_native_construction(
    scripted_public, monkeypatch, fault
):
    episodes = list(schedule())
    if fault in {"final_test", "probe_train", "probe_test"}:
        episodes[-1] = packet(role=fault)
    elif fault == "missing":
        object.__setattr__(episodes[-1].transitions[0], "observation", tuple([-1.0] * 148))
    elif fault == "mixed":
        red = packet("red").transitions[0].observation
        object.__setattr__(episodes[-1].transitions[0], "next_observation", red)
        object.__setattr__(episodes[-1].transitions[1], "observation", red)
    elif fault == "next":
        row = list(episodes[-1].transitions[-1].next_observation)
        row[0] = 0.123
        object.__setattr__(episodes[-1].transitions[-1], "next_observation", tuple(row))
    else:
        object.__setattr__(episodes[-1], "extra", True)
    monkeypatch.setattr(
        weighted.public_recurrent,
        "PublicCueRecurrentLearner",
        lambda *a: pytest.fail("native constructed"),
    )
    with pytest.raises(ValueError):
        weighted.TorchTrainWeightedLearner(policy(), tuple(episodes))


@pytest.mark.parametrize("fault", ["features", "labels", "wrong_next", "dtype"])
def test_should_refuse_nonmatching_next_arrays_with_no_paid_work(scripted_public, fault):
    source = owner()
    before = source.snapshot_state()
    x, y = episode_arrays(schedule()[0 if fault != "wrong_next" else 1])
    if fault == "features":
        x[1, -1] = 0
    elif fault == "labels":
        y[0, 0] = 1
    elif fault == "dtype":
        x = x.astype("float32")
    with pytest.raises(ValueError):
        source.train_batch(x, y)
    assert not ScriptedNative.calls and _same_value(source.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["schedule", "owner", "parameters", "gradients", "momentum", "heads"]
)
def test_should_refuse_replaced_or_corrupt_prior_evidence_before_work(scripted_public, fault):
    source = owner()
    if fault == "schedule":
        source._episodes = tuple(deepcopy(p) for p in source._episodes)
    elif fault == "owner":
        source._learner = source._learner.fork()
    elif fault == "heads":
        source.train_batch(*episode_arrays(schedule()[0]))
        state = source._learner._state
        source._learner._state = replace(state, heads=(state.heads[1], state.heads[0]))
    else:
        field = {
            "parameters": "parameters",
            "gradients": "gradients",
            "momentum": "optimizer_state",
        }[fault]
        mapping = getattr(source._learner._state.heads[0], field)
        next(iter(mapping.values()))[0] += 1
    calls = len(ScriptedNative.calls)
    with pytest.raises(ValueError):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert len(ScriptedNative.calls) == calls


@pytest.mark.parametrize("fault", ["lost", "definition", "input", "owner", "schedule"])
def test_should_keep_paid_failure_and_refuse_every_continuation_without_rollback(
    scripted_public, monkeypatch, fault
):
    source = owner()
    source.train_batch(*episode_arrays(schedule()[0]))
    before = source.snapshot_state()
    x, y = episode_arrays(schedule()[1])
    original = public.PublicCueRecurrentLearner.train_class_balanced_episode
    if fault in {"lost", "definition"}:
        native_update = getattr(ScriptedNative, "train_weighted_episode")

        def fail_native(self, episode, weights):
            result = native_update(self, episode, weights)
            if fault == "lost":
                raise RuntimeError("lost weighted return")
            return replace(result, definition="wrong")

        monkeypatch.setattr(ScriptedNative, "train_weighted_episode", fail_native)
    else:

        def fail_public(self, episode):
            result = original(self, episode)
            if fault == "input":
                x[1, -1] = 0
            elif fault == "owner":
                source._learner = self.fork()
            else:
                source._episodes = tuple(deepcopy(p) for p in source._episodes)
            return result

        monkeypatch.setattr(
            public.PublicCueRecurrentLearner, "train_class_balanced_episode", fail_public
        )
    with pytest.raises(TrainingUpdateFailed) as failure:
        source.train_batch(x, y)
    paid = deepcopy(failure.value.work)
    assert paid["failure_work_may_be_unobserved"] is True
    if fault in {"lost", "definition"}:
        state = source.snapshot_state()
        assert _same_value(state.heads, before.heads) and state.failed_work is not None
        assert paid["native_calls_attempted"] == 1
        assert paid["native_calls_returned"] == int(fault != "lost")
        assert paid["failed_state_read_verified"] is True
    else:
        assert paid["adapter_calls_returned"] == 1 and paid["failed_state_read_verified"] is False
        with pytest.raises(ValueError):
            source.snapshot_state()
    calls = len(ScriptedNative.calls)
    for run in (
        lambda: source.train_batch(*episode_arrays(schedule()[1])),
        lambda: source.predict(episode_arrays(schedule()[0])[0]),
        source.start_episode,
        source.fork,
        lambda: source.restore_state(before),
    ):
        with pytest.raises(TrainingUpdateFailed) as stopped:
            run()
        assert stopped.value.work == paid
    assert len(ScriptedNative.calls) == calls


def test_should_restore_exact_prefix_fork_and_keep_frozen_sessions(scripted_public):
    source = owner()
    source.train_batch(*episode_arrays(schedule()[0]))
    first = source.snapshot_state()
    child = deepcopy(source)
    child.train_batch(*episode_arrays(schedule()[1]))
    source.restore_state(child.snapshot_state())
    source.train_batch(*episode_arrays(schedule()[2]))
    assert source.completed_updates == 3
    source.restore_state(first)
    assert source.completed_updates == 1
    session = source.start_episode()
    assert _same_value(session.snapshot_state(), first)
    source.train_batch(*episode_arrays(schedule()[1]))
    assert _same_value(session.snapshot_state(), first)


def test_should_refuse_wrong_head_prefix_and_failed_snapshot_import(scripted_public):
    source = owner()
    source.train_batch(*episode_arrays(schedule()[0]))
    first = source.snapshot_state()
    foreign = replace(first, heads=(first.heads[1], first.heads[0]))
    with pytest.raises(ValueError):
        source.restore_state(foreign)
    failed = replace(first, failed_work=public.FailureWork("train", 1))
    with pytest.raises(ValueError):
        source.restore_state(failed)
    assert _same_value(source.snapshot_state(), first)


@pytest.mark.parametrize("operation", ["predict", "restore"])
def test_should_keep_read_or_restore_failure_and_block_any_healthy_export(
    scripted_public, monkeypatch, operation
):
    source = owner()
    before = source.snapshot_state()
    if operation == "predict":

        def lost(self, rows):
            raise RuntimeError("lost prediction return")

        monkeypatch.setattr(ScriptedNative, "predict", lost)
        run = lambda: source.predict(episode_arrays(schedule()[0])[0])
    else:
        original = public.PublicCueRecurrentLearner.restore_state

        def lost_restore(self, state):
            original(self, state)
            raise RuntimeError("lost readonly restore return")

        monkeypatch.setattr(public.PublicCueRecurrentLearner, "restore_state", lost_restore)
        run = lambda: source.restore_state(before)
    with pytest.raises(TrainingUpdateFailed):
        run()
    if operation == "predict":
        assert source.snapshot_state().failed_work is not None
    else:
        with pytest.raises(ValueError):
            source.snapshot_state()
    with pytest.raises(TrainingUpdateFailed):
        source.fork()


def test_should_preserve_committed_prefix_receipts_and_failure_under_original_stream_budget(
    scripted_public, monkeypatch
):
    source = owner()
    initial = source.snapshot_state()
    episodes = schedule()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=3, max_wall_seconds=40), lambda: 0.0
    )
    stream = AttributionStream(
        source,
        tuple(_batch(p) for p in episodes),
        (),
        training_only=True,
        label_delay=1,
        checkpoint_updates=(0, 1, 3),
        budget=budget,
    )
    update = getattr(ScriptedNative, "train_weighted_episode")

    def lost(self, episode, weights):
        result = update(self, episode, weights)
        if episode.episode_id == episodes[1].episode_id:
            raise RuntimeError("lost second stream update")
        return result

    monkeypatch.setattr(ScriptedNative, "train_weighted_episode", lost)
    with pytest.raises(TrainingUpdateFailed) as failure:
        stream.run_until(3)
    assert failure.value.work["native_calls_attempted"] == 1
    assert failure.value.work["native_calls_returned"] == 0
    assert len(stream._runtime.applied_updates) == 1
    assert budget.updates_completed == 1 and budget.replay_examples_completed == 0
    assert _same_value(source.snapshot_state(), initial)
    assert _same_value(stream._runtime.actor.snapshot_state().state, initial)
    calls = len(ScriptedNative.calls)
    with pytest.raises(ValueError, match="failed or finished"):
        stream.run_until(3)
    assert len(ScriptedNative.calls) == calls


def test_should_keep_all32_failure_counters_without_masking_or_admitting_more_work(
    scripted_public, monkeypatch
):
    source = owner()
    before = source.snapshot_state()
    counters = {f"native_paid_{index}": index for index in range(32)}

    def lost(self, state):
        raise TrainingUpdateFailed("lost restore proof", "bounded_native_failure", counters)

    monkeypatch.setattr(public.PublicCueRecurrentLearner, "restore_state", lost)
    with pytest.raises(TrainingUpdateFailed) as original:
        source.restore_state(before)
    assert original.value.work == counters
    original.value.work["native_paid_31"] = 999
    with pytest.raises(TrainingUpdateFailed) as repeated:
        source.train_batch(*episode_arrays(schedule()[0]))
    assert repeated.value.work == counters
    assert not ScriptedNative.calls
    with pytest.raises(ValueError):
        source.snapshot_state()
