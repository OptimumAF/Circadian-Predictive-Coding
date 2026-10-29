"""Real shared replay/inbox phases using the existing counted scalar learner."""

from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest

from src.app.resource_sharing import ResourceSharedRuntime
from src.app.toy_execution_budget import ToyExecutionStopped
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.resource_sharing import SharingLimits
from test_categorical_replay_action import ScalarLearner, ready, setup


def combined(*, wakes=1, action=None, **options):
    choice = action or AdaptationAction("combined", wakes, 3)
    envelope = AdaptationEnvelope(
        (
            AdaptationAction("wait"),
            AdaptationAction("wake", 1),
            AdaptationAction("replay", replay_examples=3),
            choice,
        ),
        20,
        4,
        6,
        1,
        1.0,
        ("unchanged", "neutral"),
    )
    return setup(action_envelope=envelope, **options)


def test_should_replay_then_wake_with_one_decision_and_distinct_permanent_charges():
    w, r, replay, source, clock, wake, bank, gate, _ = combined()
    before = source.snapshot_state()
    budget_bindings = (
        r._budget,
        wake.budget,
        wake.clock,
        wake.started_at,
        bank.budget,
        bank.clock,
        bank.started_at,
    )
    ready(r, clock)
    poll = w.train_action("combined")
    p = poll.progress
    assert source.calls == [3, 2]
    assert [u.sample_id for u in poll.training.updates] == ["s"]
    assert poll.replay_receipt == r.applied_consolidations[0]
    assert poll.requested_wake_updates == 1 and poll.requested_replay_examples == 3
    assert p.decisions_attempted == p.decisions_returned == 1
    assert p.replay_admissions == p.replay_commits == p.wake_admissions == 1
    assert p.native_attempts == p.native_completions == gate.snapshot().admitted_updates == 2
    assert p.poll_updates_returned == 1 and not p.unreturned_work_unknown
    assert wake.updates_completed == 1 and wake.replay_examples_completed == 0
    assert bank.updates_completed == 0 and bank.replay_examples_completed == 3
    assert r.candidate_snapshot().state.completed_updates == 2
    assert r.candidate_snapshot().state.training_examples == 5
    assert r.actor.snapshot_state().state == source.snapshot_state() == before
    assert budget_bindings == (
        r._budget,
        wake.budget,
        wake.clock,
        wake.started_at,
        bank.budget,
        bank.clock,
        bank.started_at,
    )
    other, runtime, _, control, c, _, _, _, _ = combined()
    ready(runtime, c)
    other.train_action("replay")
    other.train_action("wake")
    assert control.calls == source.calls
    assert runtime.candidate_snapshot().state == r.candidate_snapshot().state
    assert runtime.applied_updates == r.applied_updates
    assert runtime.applied_consolidations == r.applied_consolidations
    assert not replay.stopped


def test_should_keep_fifo_and_original_poll_cap_for_multiple_wakes():
    w, r, _, source, c, wake, bank, _, _ = combined(wakes=3)
    for name in ("first", "second", "third"):
        ready(r, c, name)
    p = w.train_action("combined")
    assert p.requested_wake_updates == 3
    assert [u.sample_id for u in p.training.updates] == ["first", "second"]
    assert source.calls == [3, 2, 2] and wake.updates_completed == 2
    assert bank.replay_examples_completed == 3
    assert [u.sample_id for u in w.train_action("wake").training.updates] == ["third"]
    assert source.calls == [3, 2, 2, 2]


@pytest.mark.parametrize("future", [False, True])
def test_should_keep_absent_or_future_labels_unconsumed_after_replay(future):
    w, r, _, source, clock, wake, bank, _, _ = combined()
    if future:
        r.record_experience(
            Experience(
                "s",
                "train",
                0,
                "a0",
                np.ones((2, 7)),
                "train",
                ExperiencePermissions(training=True),
            )
        )
        r.record_label(LabelArrival("label-s", "s", "train", 2, "a0", np.zeros((2, 1))))
    p = w.train_action("combined")
    assert p.replay_receipt is not None and p.training.updates == ()
    assert p.training.deferred_reason is None
    assert wake.updates_completed == 0 and bank.replay_examples_completed == 3
    assert source.calls == [3]
    if future:
        clock.advance_to(2)
    else:
        ready(r, clock)
    assert len(w.train_action("wake").training.updates) == 1
    assert source.calls == [3, 2] and bank.replay_examples_completed == 3


@pytest.mark.parametrize(
    "reason", ["paused", "serving_active", "resource_unavailable", "work_quota"]
)
def test_should_defer_both_phases_without_any_replay_or_wake_work(reason, monkeypatch):
    limits = SharingLimits(1, 0, 1) if reason == "work_quota" else None
    w, r, _, source, clock, wake, bank, gate, _ = combined(
        limits=limits, available=lambda: reason != "resource_unavailable"
    )
    ready(r, clock)
    if reason == "paused":
        gate.pause()
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("fork after initial deferral"))
    with gate.serving() if reason == "serving_active" else nullcontext():
        p = w.train_action("combined")
    assert p.training.deferred_reason == reason and p.replay_receipt is None
    assert p.training.updates == () and p.progress.decisions_returned == 1
    assert source.calls == [] and bank.replay_examples_completed == wake.updates_completed == 0
    assert gate.snapshot().admitted_updates == 0


@pytest.mark.parametrize(
    "reason", ["paused", "serving_active", "resource_unavailable", "work_quota"]
)
def test_should_return_paid_replay_when_the_separate_wake_admission_defers(reason, monkeypatch):
    limits = SharingLimits(1, 1, 2) if reason == "work_quota" else None
    available = [True]
    w, r, replay, source, clock, wake, bank, gate, _ = combined(
        limits=limits, available=lambda: available[0]
    )
    ready(r, clock)
    execute = replay.execute
    serving = None

    def between(*args):
        nonlocal serving
        receipt = execute(*args)
        if reason == "paused":
            gate.pause()
        elif reason == "resource_unavailable":
            available[0] = False
        elif reason == "serving_active":
            serving = gate.serving()
            serving.__enter__()
        return receipt

    monkeypatch.setattr(replay, "execute", between)
    try:
        p = w.train_action("combined")
    finally:
        if serving is not None:
            serving.__exit__(None, None, None)
    assert p.replay_receipt == r.applied_consolidations[0]
    assert p.training.updates == () and p.training.deferred_reason == reason
    assert p.progress.replay_commits == 1 and p.progress.wake_admissions == 0
    assert source.calls == [3] and bank.replay_examples_completed == 3
    assert wake.updates_completed == 0 and gate.snapshot().admitted_updates == 1
    if reason != "work_quota":
        gate.resume()
        available[0] = True
        assert len(w.train_action("wake").training.updates) == 1
        assert source.calls == [3, 2] and bank.replay_examples_completed == 3


@pytest.mark.parametrize("exhausted", ["wake", "replay"])
def test_should_check_both_cumulative_envelopes_before_any_composite_admission(
    exhausted, monkeypatch
):
    w, r, _, source, clock, _, _, gate, _ = combined()
    for n in range(4 if exhausted == "wake" else 2):
        if exhausted == "wake":
            ready(r, clock, str(n))
        w.train_action(exhausted)
    before = len(source.calls), gate.snapshot().admitted_updates
    monkeypatch.setattr(gate, "_resource", lambda: pytest.fail("probe after exhausted envelope"))
    with pytest.raises(ValueError, match=f"cumulative {exhausted} envelope"):
        w.train_action("combined")
    assert (len(source.calls), gate.snapshot().admitted_updates) == before


@pytest.mark.parametrize(
    "change", [{"learning_rate_multiplier": 0.5}, {"gate_setting": "neutral"}, {"sleep_updates": 1}]
)
def test_should_refuse_other_mixed_effects_before_resource_or_payload(change, monkeypatch):
    w, _, _, source, _, _, bank, gate, _ = combined(
        action=AdaptationAction("combined", 1, 3, **change)
    )
    monkeypatch.setattr(gate, "_resource", lambda: pytest.fail("unsupported resource probe"))
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("unsupported payload"))
    with pytest.raises(ValueError, match="unsupported"):
        w.train_action("combined")
    assert source.calls == [] and bank.replay_examples_completed == 0


def test_should_refuse_composite_without_original_replay_executor(monkeypatch):
    _, r, _, source, _, _, _, gate, _ = combined()
    e = AdaptationEnvelope((AdaptationAction("combined", 1, 3),), 1, 1, 3)
    w = ResourceSharedRuntime(r, gate, action_envelope=e)
    monkeypatch.setattr(gate, "_resource", lambda: pytest.fail("missing executor resource probe"))
    with pytest.raises(ValueError, match="unsupported"):
        w.train_action("combined")
    assert source.calls == []


@pytest.mark.parametrize("fault,returned", [("raise_replay", 0), ("wrong_replay", 1)])
def test_should_keep_replay_failure_paid_and_never_start_wake(fault, returned):
    w, r, replay, source, clock, wake, bank, _, _ = combined()
    ready(r, clock)
    source.faults[fault] = True
    with pytest.raises((RuntimeError, ValueError)):
        w.train_action("combined")
    p = w.adaptation_snapshot().progress
    assert source.calls == [3] and bank.replay_examples_completed == 3
    assert p.replay_native_attempts == 1 and p.replay_native_completions == returned
    assert p.wake_admissions == wake.updates_completed == 0
    assert p.replay_commits == 0 and replay.stopped
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")


def test_should_stop_before_wake_when_replay_restore_is_uncertain(monkeypatch):
    w, r, replay, source, c, wake, bank, _, _ = combined()
    ready(r, c)

    def fail(state):
        r._candidate.state = replace(state, weight=999.0)
        raise RuntimeError("lost replay restore")

    monkeypatch.setattr(r._candidate, "restore_state", fail)
    with pytest.raises(RuntimeError, match="lost replay restore"):
        w.train_action("combined")
    p = w.adaptation_snapshot().progress
    assert p.replay_publication_unknown and p.unreturned_work_unknown and replay.stopped
    assert source.calls == [3] and bank.replay_examples_completed == 3
    assert wake.updates_completed == p.wake_admissions == 0


@pytest.mark.parametrize("fault", ["receipt", "candidate", "source", "clock"])
def test_should_revalidate_replay_success_before_admitting_wake(fault, monkeypatch):
    w, r, replay, source, c, wake, bank, gate, _ = combined()
    ready(r, c)
    execute = replay.execute

    def corrupt(*args):
        receipt = execute(*args)
        if fault == "receipt":
            r._consolidations[0] = replace(r._consolidations[0], event_id="replaced")
        elif fault == "candidate":
            r._candidate.state = replace(r._candidate.state, weight=999.0)
        elif fault == "source":
            source.state = replace(source.state, weight=999.0)
        else:
            wake.clock = lambda: 0.0
        return receipt

    monkeypatch.setattr(replay, "execute", corrupt)
    with pytest.raises(ValueError):
        w.train_action("combined")
    assert source.calls == [3] and bank.replay_examples_completed == 3
    assert wake.updates_completed == w._action_progress.wake_admissions == 0
    assert gate.snapshot().admitted_updates == 1 and replay.stopped


@pytest.mark.parametrize("phase", ["replay", "wake"])
def test_should_preserve_known_commits_and_original_clock_after_late_budget_stop(
    phase, monkeypatch
):
    w, r, replay, source, c, wake, bank, _, wall = combined()
    ready(r, c)
    before = wake.before_final

    def stop():
        # The budget callback already runs under the runtime lease.
        if r._consolidations and (phase == "replay" or r._inbox.applied_updates):
            wall[0] = 10.0
        before()

    monkeypatch.setattr(wake, "before_final", stop)
    with pytest.raises(ToyExecutionStopped):
        w.train_action("combined")
    p = w.adaptation_snapshot().progress
    assert bank.replay_examples_completed == 3 and p.replay_commits == 1
    assert not p.replay_publication_unknown
    assert wake.updates_completed == len(r.applied_updates) == int(phase == "wake")
    assert source.calls == ([3] if phase == "replay" else [3, 2])
    assert p.decisions_refused == 1 and p.poll_updates_returned == 0
    assert source.snapshot_state() == r.actor.snapshot_state().state


def test_should_keep_replay_committed_when_wake_native_return_is_lost(monkeypatch):
    w, r, replay, source, c, wake, bank, gate, _ = combined()
    ready(r, c)
    train = ScalarLearner.train_batch

    def lose(owner, features, targets):
        result = train(owner, features, targets)
        if len(features) == 2:
            raise RuntimeError("lost wake return")
        return result

    monkeypatch.setattr(ScalarLearner, "train_batch", lose)
    with pytest.raises(RuntimeError, match="lost wake return"):
        w.train_action("combined")
    p = w.adaptation_snapshot().progress
    assert source.calls == [3, 2] and len(r.applied_consolidations) == 1
    assert p.replay_commits == 1 and bank.replay_examples_completed == 3
    assert p.native_attempts == 2 and p.native_completions == 1 and p.unreturned_work_unknown
    assert replay.stopped and wake.updates_completed == 0
    assert p.wake_admissions == 1 and gate.snapshot().admitted_updates == 2
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")
