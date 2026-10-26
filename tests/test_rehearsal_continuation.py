"""Owner-local replay continuation with actual runtime gates and fake learners."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any, Callable

import numpy as np
import pytest

import src.app.stream_rehearsal as module
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.stream_rehearsal import (
    RecentStreamReplayPolicy,
    StreamRehearsal,
    StreamReplayPolicy,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.core.shared_replay_schedule import ReplayAllocation, ReplayRotationCursor


class FakeLearner:
    def __init__(self, trace: list[Any], value: float = 0.0, *, fail_replay: bool = False):
        self.value = np.array([value])
        self.trace = trace
        self.fail_replay = fail_replay

    def fork(self):
        return FakeLearner(self.trace, float(self.value[0]), fail_replay=self.fail_replay)

    def snapshot_state(self):
        return self.value.copy()

    def restore_state(self, state):
        self.value = state.copy()

    def train_batch(self, features, targets):
        self.trace.append((len(features), float(self.value[0]), features.copy(), targets.copy()))
        self.value += 1
        if self.fail_replay and len(features) == 1:
            raise RuntimeError("fake native replay failed after work")
        return TrainingDiagnostic("fake_continuation_v1", float(self.value[0]))

    def predict(self, features):
        return np.full((len(features), 1), 0.4 + float(self.value[0]) / 100)


def _setup(mode="rotating", *, fail_replay=False):
    seconds = [0.0]
    clock = lambda: seconds[0]
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4, max_wall_seconds=100), clock)
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=4, max_wall_seconds=100), clock
    )
    source = FakeLearner([], fail_replay=fail_replay)
    policy = (
        RecentStreamReplayPolicy(2, 2, ReplayRetentionBudget(4, 96))
        if mode == "recent"
        else StreamReplayPolicy(
            2,
            2,
            ReplayRetentionBudget(4, 96),
            ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48)),
            53,
        )
    )
    helper = StreamRehearsal(
        source, wake, policy, replay_budget=replay, total_updates=4, input_dim=2
    )
    return source, wake, replay, helper, seconds


def _batches():
    training = tuple(
        AttributionBatch(
            "train",
            "A" if k < 2 else "B",
            (f"train-{k}-0", f"train-{k}-1"),
            np.array([[float(k), 0.25], [k + 0.5, -0.25]]),
            np.full((2, 1), float(k % 2)),
        )
        for k in range(4)
    )
    final = tuple(
        AttributionBatch(
            "final_test",
            context,
            (f"final-{context}",),
            np.array([[9.0, float(k)]]),
            np.array([[float(k)]]),
        )
        for k, context in enumerate(("A", "B"))
    )
    return training, final


Hook = Callable[[StreamRehearsal, ToyBudgetSession, ToyBudgetSession, list[float]], None]


def _run(mode="rotating", hook: Hook | None = None, *, fail_replay=False):
    source, wake, replay, helper, seconds = _setup(mode, fail_replay=fail_replay)
    training, final = _batches()

    def observe(updates, state, batch):
        if updates == 2 and hook is not None:
            runtime = helper._runtime
            assert runtime is not None
            # This is the actual caller lease used by stream pause/resume. Any
            # public runtime reader in the helper would fail by reentering it.
            with runtime._exclusive():
                hook(helper, wake, replay, seconds)

    result = run_attribution_stream(
        source,
        training,
        final,
        checkpoint_updates=(0, 2, 4),
        label_delay=2,
        budget=wake,
        rehearsal=helper,
        observe_checkpoint=observe,
    )
    return result, source, wake, replay, helper


@pytest.mark.parametrize("mode", ["rotating", "recent"])
def test_owned_roundtrip_preserves_future_selection_receipts_and_scores(mode):
    baseline = _run(mode)

    def restore(helper, wake, replay, seconds):
        buffer, events, policy = helper._buffer, helper._events, helper._policy
        snapshot = helper._capture_continuation()
        prepared = helper._prepare_continuation(snapshot)
        assert prepared is not snapshot
        assert prepared.buffer is not buffer and prepared.events is not events
        assert prepared.buffer.rotation_cursor == buffer.rotation_cursor
        for key, rows in buffer._rows.items():
            assert not np.shares_memory(rows[0], snapshot.buffer._rows[key][0])
            assert not np.shares_memory(rows[0], prepared.buffer._rows[key][0])
            assert not np.shares_memory(
                snapshot.buffer._rows[key][0], prepared.buffer._rows[key][0]
            )
        # A caller's observation stays separate even after preparation.
        next(iter(snapshot.buffer._rows.values()))[0].fill(999)
        snapshot.events.clear()
        helper._commit_continuation(prepared)
        assert helper._buffer is prepared.buffer and helper._events is prepared.events
        assert helper._policy is policy
        assert wake.updates_completed == 2 and replay.replay_examples_completed == 2

    restored = _run(mode, restore)
    assert restored[0] == baseline[0]
    assert restored[4].report == baseline[4].report
    assert restored[2].updates_completed == 4
    assert restored[3].replay_examples_completed == 4
    np.testing.assert_array_equal(restored[1].snapshot_state(), [0])
    assert len(restored[1].trace) == 8


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "missing",
        "policy",
        "observed",
        "stopped",
        "reason",
        "binding",
        "row",
        "dtype",
        "row_shape",
        "events",
        "cursor",
        "partition",
    ],
)
def test_modified_snapshot_refuses_without_changing_live_helper_or_charges(change):
    def attempt(helper, wake, replay, seconds):
        snapshot = helper._capture_continuation()
        before = pickle.dumps(helper.report)
        calls = len(helper._learner.trace)
        if change == "version":
            object.__setattr__(snapshot, "format_version", True)
        elif change == "missing":
            object.__delattr__(snapshot, "reason")
        elif change == "policy":
            object.__setattr__(snapshot.policy, "seed", 17)
        elif change in {"observed", "stopped", "reason", "binding"}:
            value = {"observed": 1, "stopped": True, "reason": "changed", "binding": ()}[change]
            object.__setattr__(snapshot, change, value)
        elif change == "row":
            next(iter(snapshot.buffer._rows.values()))[0].fill(999)
        elif change in {"dtype", "row_shape"}:
            key = next(iter(snapshot.buffer._rows))
            x, y = snapshot.buffer._rows[key]
            snapshot.buffer._rows[key] = (
                x.astype(np.float32) if change == "dtype" else np.zeros((2, 2)),
                y,
            )
        elif change == "events":
            snapshot.events[0]["selected_ids"].reverse()
        elif change == "cursor":
            cursor = snapshot.buffer.rotation_cursor
            snapshot.buffer._rotation_cursor = ReplayRotationCursor(
                1 - cursor.next_partition, cursor.reservoir_after
            )
        else:
            snapshot.buffer._recent.clear()
        with pytest.raises(ValueError, match="continuation"):
            helper._prepare_continuation(snapshot)
        assert pickle.dumps(helper.report) == before
        assert len(helper._learner.trace) == calls
        assert wake.updates_completed == replay.replay_examples_completed == 2

    _run(hook=attempt)


@pytest.mark.parametrize(
    "change",
    [
        "clock",
        "started",
        "limits",
        "wake",
        "replay",
        "source",
        "runtime",
        "buffer",
        "events",
        "geometry",
    ],
)
def test_changed_current_binding_refuses_without_publishing(change):
    def attempt(helper, wake, replay, seconds):
        snapshot = helper._capture_continuation()
        value: object
        if change == "clock":
            target, name, value = wake, "clock", lambda: 0.0
        elif change == "started":
            target, name, value = wake, "started_at", -1.0
        elif change == "limits":
            target, name, value = wake, "budget", replace(wake.budget, max_training_updates=5)
        elif change == "wake":
            target, name, value = wake, "updates_completed", 3
        elif change == "replay":
            target, name, value = replay, "replay_examples_completed", 3
        elif change == "source":
            target, name, value = helper, "_learner", FakeLearner([])
        elif change == "runtime":
            target, name, value = helper, "_runtime", None
        elif change == "buffer":
            target, name, value = helper, "_buffer", deepcopy(helper._buffer)
        elif change == "events":
            target, name, value = helper, "_events", deepcopy(helper._events)
        else:
            target, name, value = helper, "_input_dim", 3
        original = getattr(target, name)
        setattr(target, name, value)
        try:
            with pytest.raises(ValueError, match="continuation"):
                helper._prepare_continuation(snapshot)
        finally:
            setattr(target, name, original)

    _run(hook=attempt)


def test_original_clock_checks_can_advance_without_restoring_telemetry():
    def restore(helper, wake, replay, seconds):
        snapshot = helper._capture_continuation()
        seconds[0] = 0.5
        wake.before_final()
        replay.before_final()
        prepared = helper._prepare_continuation(snapshot)
        helper._commit_continuation(prepared)
        assert wake.started_at == replay.started_at == 0
        assert wake.last_clock == replay.last_clock == 0.5
        assert wake.updates_completed == replay.replay_examples_completed == 2

    _run(hook=restore)


def test_unbound_and_foreign_helper_refuse_without_fake_work():
    source, wake, replay, helper, _ = _setup()
    with pytest.raises(ValueError, match="bound open"):
        helper._capture_continuation()
    helper.bind(source, wake, 4, 2)
    snapshot = helper._capture_continuation()
    other_source, other_wake, _, other, _ = _setup()
    other.bind(other_source, other_wake, 4, 2)
    with pytest.raises(ValueError, match="continuation"):
        other._prepare_continuation(snapshot)
    assert not source.trace and not other_source.trace
    assert wake.updates_completed == replay.replay_examples_completed == 0


def test_stopped_helper_cannot_capture_or_prepare():
    def attempt(helper, wake, replay, seconds):
        snapshot = helper._capture_continuation()
        helper._stopped = True
        try:
            with pytest.raises(ValueError, match="bound open"):
                helper._capture_continuation()
            with pytest.raises(ValueError, match="bound open"):
                helper._prepare_continuation(snapshot)
        finally:
            helper._stopped = False

    _run(hook=attempt)


def test_prepare_copy_failure_leaves_all_live_work_and_state_unchanged(monkeypatch):
    failure = RuntimeError("prepared replay data copy failed")
    original_copy = module.deepcopy

    def attempt(helper, wake, replay, seconds):
        snapshot = helper._capture_continuation()
        before = pickle.dumps(helper.report)

        def fail_final_copy(value):
            if value is snapshot:
                raise failure
            return original_copy(value)

        with monkeypatch.context() as patch:
            patch.setattr(module, "deepcopy", fail_final_copy)
            with pytest.raises(RuntimeError) as caught:
                helper._prepare_continuation(snapshot)
        assert caught.value is failure
        assert pickle.dumps(helper.report) == before
        assert wake.updates_completed == replay.replay_examples_completed == 2

    _run(hook=attempt)


def test_commit_only_assigns_prepared_data_without_callbacks(monkeypatch):
    def restore(helper, wake, replay, seconds):
        prepared = helper._prepare_continuation(helper._capture_continuation())

        def forbidden(*args, **kwargs):
            raise AssertionError("commit must not invoke preparation or callbacks")

        with monkeypatch.context() as patch:
            patch.setattr(module, "deepcopy", forbidden)
            patch.setattr(module.pickle, "dumps", forbidden)
            patch.setattr(helper, "_continuation_binding", forbidden)
            helper._commit_continuation(prepared)
        assert helper._buffer is prepared.buffer
        assert helper._events is prepared.events

    _run(hook=restore)
