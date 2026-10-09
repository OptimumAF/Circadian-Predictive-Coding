"""Cooperative admission at native boundaries, with deterministic contention."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event
import pickle

import numpy as np
import pytest

from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.core.resource_sharing import SharingLimits
from test_actor_shadow import PausableLearner, queue, setup


def shared(source=None, *, limits=None, resource=None):
    runtime, clock, original, budget = setup(source=source)
    gate = ServingPriorityGate(
        limits or SharingLimits(2, 8, 2), resource_available=resource or (lambda: True)
    )
    return ResourceSharedRuntime(runtime, gate), runtime, clock, gate, original, budget


def ready(runtime, clock):
    queue(runtime)
    queue(runtime, "s2", [9, 9])
    clock.advance_to(3)


def test_should_defer_before_native_work_when_serving_is_admitted_then_resume_in_order():
    wrapper, runtime, clock, gate, _, budget = shared()
    ready(runtime, clock)
    before = runtime.candidate_snapshot()
    with gate.serving():
        poll = wrapper.train_ready()
        assert poll.updates == () and poll.deferred_reason == "serving_active"
        assert runtime.candidate_snapshot() == before and budget.updates_completed == 0
    poll = wrapper.train_ready()
    assert [u.sample_id for u in poll.updates] == ["s1", "s2"]
    assert poll.deferred_reason is None and gate.snapshot().admitted_updates == 2
    assert wrapper.train_ready().updates == ()
    assert gate.snapshot().admitted_updates == 2  # empty polls consume no admission


def test_should_pause_exactly_between_ready_native_updates_without_replay():
    wrapper, runtime, clock, gate, _, _ = shared()
    ready(runtime, clock)
    original = runtime._candidate.train_batch

    def train(features, targets):
        diagnostic = original(features, targets)
        gate.pause()
        return diagnostic

    runtime._candidate.train_batch = train
    first = wrapper.train_ready()
    assert [u.sample_id for u in first.updates] == ["s1"]
    assert first.deferred_reason == "paused" and gate.snapshot().admitted_updates == 1
    runtime._candidate.train_batch = original
    gate.resume()
    second = wrapper.train_ready()
    assert [u.sample_id for u in second.updates] == ["s2"]
    assert [u.update_number for u in runtime.applied_updates] == [1, 2]


def test_should_bound_each_poll_and_preserve_remaining_ready_cursor():
    wrapper, runtime, clock, gate, _, _ = shared(limits=SharingLimits(2, 8, 1))
    ready(runtime, clock)
    assert [u.sample_id for u in wrapper.train_ready().updates] == ["s1"]
    assert [u.sample_id for u in wrapper.train_ready().updates] == ["s2"]
    assert wrapper.train_ready().updates == () and gate.snapshot().admitted_updates == 2


@pytest.mark.parametrize("contention", ["serving", "pause"])
def test_should_prioritize_serving_after_resource_probe_race_without_native_work(contention):
    entered, release = Event(), Event()

    def resource():
        entered.set()
        assert release.wait(3)
        return True

    wrapper, runtime, clock, gate, _, _ = shared(resource=resource)
    ready(runtime, clock)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(wrapper.train_ready)
        try:
            assert entered.wait(3)
            with gate.serving():
                if contention == "pause":
                    gate.pause()
                release.set()
                result = pending.result(timeout=3)
                assert result.deferred_reason == (
                    "paused" if contention == "pause" else "serving_active"
                )
        finally:
            release.set()
    assert gate.snapshot().admitted_updates == 0


def test_should_serve_actual_actor_while_native_candidate_is_partially_mutated_and_defer_next():
    entered, release = Event(), Event()
    wrapper, runtime, clock, gate, _, _ = shared(
        PausableLearner(entered=entered, release=release, phase="train")
    )
    ready(runtime, clock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(wrapper.train_ready)
        try:
            assert entered.wait(3)
            output = pool.submit(wrapper.predict, ["x"]).result(timeout=3)
            assert output.actor_version == "actor-0" and output.prediction == [2, 2]
            gate.pause()  # cooperative: already active native work is not cancelled
        finally:
            release.set()
        result = pending.result(timeout=3)
    assert [u.sample_id for u in result.updates] == ["s1"]
    assert result.deferred_reason == "paused" and runtime.candidate_snapshot().state == [8, 8]
    assert not gate.snapshot().training_active


def test_should_bound_actual_waiting_serving_calls_before_copying_overload_payload():
    wrapper, runtime, _, gate, _, _ = shared(limits=SharingLimits(2, 8, 1))
    entered, release, queued = Event(), Event(), Event()
    original_serving = gate.serving

    @contextmanager
    def serving():
        with original_serving():
            if gate.snapshot().active_serving_requests == 2:
                queued.set()
            yield

    gate.serving = serving
    original = runtime.actor._learner.predict

    def predict(features):
        entered.set()
        assert release.wait(3)
        return original(features)

    runtime.actor._learner.predict = predict

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("overload payload accessed")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(wrapper.predict, ["x"])
        assert entered.wait(3)
        second = pool.submit(wrapper.predict, ["x"])
        try:
            assert queued.wait(3) and not second.done()
            with pytest.raises(ValueError, match="serving.*capacity"):
                wrapper.predict(CannotCopy())
            assert gate.snapshot().active_serving_requests == 2
        finally:
            release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert gate.snapshot().active_serving_requests == 0


def test_should_defer_concurrent_training_without_spending_another_admission():
    _, _, _, gate, _, _ = shared()
    with gate.training() as active:
        assert active.allowed
        with gate.training() as nested:
            assert not nested.allowed and nested.reason == "training_active"
    assert gate.snapshot().admitted_updates == 1 and not gate.snapshot().training_active


def test_should_never_refund_admitted_failed_attempts_and_release_all_leases():
    _, _, _, gate, _, _ = shared(limits=SharingLimits(2, 1, 1))
    with pytest.raises(RuntimeError):
        with gate.training() as decision:
            assert decision.allowed
            raise RuntimeError("native failed")
    assert gate.snapshot().admitted_updates == 1 and not gate.snapshot().training_active
    with gate.training() as denied:
        assert denied.reason == "work_quota"
    with pytest.raises(KeyboardInterrupt):
        with gate.serving():
            raise KeyboardInterrupt()
    assert gate.snapshot().active_serving_requests == 0


@pytest.mark.parametrize("value", [False, 0, None, "yes"])
def test_should_refuse_false_or_malformed_resource_probe_without_native_work(value):
    wrapper, runtime, clock, gate, _, _ = shared(resource=lambda: value)
    ready(runtime, clock)
    before = runtime.candidate_snapshot()
    if value is False:
        assert wrapper.train_ready().deferred_reason == "resource_unavailable"
    else:
        with pytest.raises(ValueError, match="exact boolean"):
            wrapper.train_ready()
    assert runtime.candidate_snapshot() == before and gate.snapshot().admitted_updates == 0


def test_should_propagate_resource_probe_failure_without_lease_or_native_work():
    def fail():
        raise RuntimeError("resource unavailable")

    wrapper, runtime, clock, gate, _, _ = shared(resource=fail)
    ready(runtime, clock)
    with pytest.raises(RuntimeError, match="resource unavailable"):
        wrapper.train_ready()
    assert gate.snapshot().admitted_updates == 0 and not gate.snapshot().training_active


def test_should_not_probe_resources_or_copy_payload_when_known_paused():
    def fail():
        raise AssertionError("unneeded resource probe")

    wrapper, runtime, clock, gate, _, _ = shared(resource=fail)
    ready(runtime, clock)
    gate.pause()
    assert wrapper.train_ready().deferred_reason == "paused"
    assert wrapper.predict(["x"]).prediction == [2, 2]


def test_should_return_detached_priority_snapshots_and_actor_outputs():
    wrapper, _, _, gate, _, _ = shared()
    snapshot = gate.snapshot()
    object.__setattr__(snapshot, "admitted_updates", 99)
    object.__setattr__(snapshot.limits, "max_admitted_updates", 99)
    result = wrapper.predict(["x"])
    result.prediction[:] = [99, 99]
    assert wrapper.predict(["x"]).prediction == [2, 2]
    assert gate.snapshot().admitted_updates == 0
    assert gate.snapshot().limits.max_admitted_updates == 8


def test_should_defer_before_copying_pending_native_payload():
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("pending native payload copied")

    wrapper, runtime, clock, gate, _, budget = shared()
    queue(runtime)
    clock.advance_to(3)
    source = runtime._inbox._experiences[("e1", "s1")]
    runtime._inbox._experiences[source.key] = replace(source, features=CannotCopy())
    gate.pause()
    assert wrapper.train_ready().deferred_reason == "paused"
    assert budget.updates_completed == 0 and runtime.applied_updates == ()
    assert gate.snapshot().admitted_updates == 0


def test_should_release_admission_and_stop_candidate_after_actual_native_cancellation():
    class Cancelled(PausableLearner):
        def fork(self):
            return Cancelled(self.value)

        def train_batch(self, features, targets):
            self.value[0] = 99
            raise KeyboardInterrupt()

    wrapper, runtime, clock, gate, _, _ = shared(Cancelled())
    queue(runtime)
    clock.advance_to(3)
    with pytest.raises(KeyboardInterrupt):
        wrapper.train_ready()
    assert not gate.snapshot().training_active and gate.snapshot().admitted_updates == 1
    assert runtime.applied_updates == ()
    with pytest.raises(ValueError, match="stopped"):
        runtime.candidate_snapshot()
    assert wrapper.predict(["x"]).prediction == [2, 2]


def test_should_preserve_future_arrivals_and_default_full_drain_behavior():
    wrapper, runtime, clock, gate, _, _ = shared()
    queue(runtime)
    clock.advance_to(2)
    assert wrapper.train_ready().updates == () and gate.snapshot().admitted_updates == 0
    queue(runtime, "s2", [9, 9])
    clock.advance_to(3)
    assert len(runtime.train_ready()) == 2  # original ungated API still works as configured


def test_should_preserve_completed_receipt_when_native_post_budget_check_stops():
    wrapper, runtime, clock, gate, _, budget = shared()
    ready(runtime, clock)
    original = runtime._candidate.train_batch
    elapsed = [0.0]
    budget.clock = lambda: elapsed[0]
    budget.budget = replace(budget.budget, max_wall_seconds=1.0)

    def train(features, targets):
        diagnostic = original(features, targets)
        elapsed[0] = 2.0
        return diagnostic

    runtime._candidate.train_batch = train
    from src.app.toy_execution_budget import ToyExecutionStopped

    with pytest.raises(ToyExecutionStopped):
        wrapper.train_ready()
    assert [u.sample_id for u in runtime.applied_updates] == ["s1"]
    assert budget.updates_completed == 1 and gate.snapshot().admitted_updates == 1
    assert not gate.snapshot().training_active


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_should_refuse_invalid_partial_drain_before_event_clock_or_payload_access(limit):
    _, runtime, _, _, _, _ = shared()
    with pytest.raises(ValueError, match="max_updates"):
        runtime.train_ready(max_updates=limit)


def test_should_refuse_malformed_per_update_context_result_without_training():
    _, runtime, clock, _, _, budget = shared()
    ready(runtime, clock)

    @contextmanager
    def admit():
        yield 1

    with pytest.raises(ValueError, match="exact boolean"):
        runtime.train_ready(before_each_update=admit)
    assert budget.updates_completed == 0


@pytest.mark.parametrize("limits", [(0, 1, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1), (1, True, 1)])
def test_should_refuse_invalid_sharing_limits(limits):
    with pytest.raises(ValueError):
        SharingLimits(*limits)


def test_should_support_zero_work_quota_and_serving_after_quota_exhaustion():
    wrapper, runtime, clock, gate, _, _ = shared(limits=SharingLimits(1, 0, 1))
    ready(runtime, clock)
    assert wrapper.train_ready().deferred_reason == "work_quota"
    assert wrapper.predict(["x"]).prediction == [2, 2]
    assert gate.snapshot().admitted_updates == 0


def test_should_compose_actual_cached_serving_and_release_shared_admission():
    from test_serving_promotion import setup as promotion_setup

    actor, runtime, _ = promotion_setup()
    gate = ServingPriorityGate(SharingLimits(1, 1, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    assert not wrapper.serve([0], now=2).cache_hit
    frame = wrapper.serve([0], now=2)
    assert frame.cache_hit and frame.prediction == [0.5] and frame.generation == 0
    frame.prediction[:] = [99]
    assert actor.serve([0], now=2).prediction == [0.5]
    assert gate.snapshot().active_serving_requests == 0


def test_should_refuse_cached_serving_for_fixed_actor_without_admission():
    wrapper, _, _, gate, _, _ = shared()
    with pytest.raises(ValueError, match="PromotableActor"):
        wrapper.serve(["x"], now=2)
    assert gate.snapshot().active_serving_requests == 0


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_match_complete_native_state_after_partial_poll_pause_and_resume(kind):
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
    from test_experience_inbox import make_native_pair

    _, source = make_native_pair(kind)
    direct = source.fork()
    features = np.array([[0.3, -0.2], [-0.5, 0.4]])
    targets = np.array([[1.0], [0.0]])
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0),
    )
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    for index, target in enumerate((targets, 1 - targets)):
        key = f"s{index}"
        runtime.record_experience(
            Experience(
                key, "train", 1, "actor-0", features, "train", ExperiencePermissions(training=True)
            )
        )
        runtime.record_label(LabelArrival(f"label-{index}", key, "train", 2, "actor-0", target))
    clock.advance_to(2)
    before = pickle.dumps(runtime.candidate_snapshot().state)
    gate.pause()
    assert wrapper.train_ready().deferred_reason == "paused"
    assert pickle.dumps(runtime.candidate_snapshot().state) == before
    assert wrapper.predict(features).actor_version == "actor-0"  # one new native prediction/kind
    gate.resume()
    first = wrapper.train_ready()
    assert [u.sample_id for u in first.updates] == ["s0"]
    direct.train_batch(features, targets)
    assert pickle.dumps(runtime.candidate_snapshot().state) == pickle.dumps(direct.snapshot_state())
    second = wrapper.train_ready()
    assert [u.sample_id for u in second.updates] == ["s1"]
    direct.train_batch(features, 1 - targets)
    assert pickle.dumps(runtime.candidate_snapshot().state) == pickle.dumps(direct.snapshot_state())
    assert wrapper.train_ready().updates == () and gate.snapshot().admitted_updates == 2
