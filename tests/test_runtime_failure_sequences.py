"""Finite repeated fault sequences preserve actor availability and spent authority."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
from time import monotonic
import tracemalloc

import numpy as np
import pytest

from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from test_actor_shadow import PausableLearner, queue
from test_candidate_checkpoint import controller, digest
from test_managed_experience import declaration
from test_promotion_guard_evaluation import policy as promotion_policy
from test_resource_sharing import shared
from test_retained_payload_budget import setup as bytes_setup
from test_serving_promotion import setup as promotion_setup, prepare

ROUNDS = 64
STAGE = Path("artifacts/runs/r37a-inprocess-20261007")


def test_should_bound_rejected_sleep_attempts_across_repeated_faults_and_handoff():
    wrapper, old, _, gate, source, budget = shared()
    calls = []

    def reject(state):
        calls.append(1)
        raise RuntimeError("rejected transform")

    for i in range(ROUNDS):
        with pytest.raises(RuntimeError if i < 2 else ValueError):
            old.consolidate("sleep-" + str(i), reject)
        assert wrapper.predict([0]).prediction == [2, 2]
    assert (
        len(calls) == len(old._attempted_ids) == 2
        and old._consolidations == []
        and budget.updates_completed == 0
    )
    gate.pause()
    control = controller(wrapper)
    new = control.restore(control.capture())
    for i in range(ROUNDS):
        with pytest.raises(ValueError, match="quota"):
            new.consolidate("new-" + str(i), reject)
    assert (
        len(calls) == 2
        and new._attempted_ids == old._attempted_ids
        and new._budget is budget
        and old._retired
    )
    assert source.value == [2, 2] and wrapper.predict([0]).prediction == [2, 2]


def test_should_bound_repeated_rejected_promotions_without_retained_growth_or_serving_change():
    actor, runtime, control = promotion_setup(
        declared=replace(promotion_policy(), min_new_utility=0.9)
    )
    actor.serve([0], now=1)
    before = pickle.dumps(actor.serving_snapshot())
    candidate = runtime.candidate_snapshot()
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        for _ in range(ROUNDS):
            with pytest.raises(ValueError, match="rejected"):
                prepare(control, runtime)
            assert (
                actor.serve([0], now=1).cache_hit
                and control._pending == {}
                and control._latest is None
            )
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 4 * 1024 * 1024
    finally:
        tracemalloc.stop()
    assert (
        pickle.dumps(actor.serving_snapshot()) == before
        and runtime.candidate_snapshot() == candidate
    )


def test_should_refuse_repeated_corrupt_checkpoint_without_native_preparation_or_owner_change():
    wrapper, old, _, gate, _, budget = shared()
    gate.pause()
    control = controller(wrapper)
    token = control.capture()
    before = old.candidate_snapshot()
    control._pending[token].view.state[0] = 99
    for _ in range(ROUNDS):
        with pytest.raises(ValueError, match="changed"):
            control.restore(token)
        assert wrapper.predict([0]).prediction == [2, 2]
    assert (
        control._attempts == 0
        and control._models == []
        and wrapper._runtime is old
        and not old._retired
    )
    assert old.candidate_snapshot() == before and budget.updates_completed == 0
    control.discard(token)
    assert control._pending == {}


def test_should_exhaust_failed_preparation_once_without_infinite_retry_or_native_growth():
    wrapper, old, _, gate, _, budget = shared()
    gate.pause()
    calls = []

    def fail(state):
        calls.append(1)
        raise RuntimeError("builder failed")

    control = controller(wrapper, build_learner=fail, max_prepare_attempts=4)
    token = control.capture()
    before = old.candidate_snapshot()
    for i in range(ROUNDS):
        with pytest.raises(RuntimeError if i < 4 else ValueError):
            control.restore(token)
        assert wrapper.predict([0]).prediction == [2, 2]
    assert (
        len(calls) == control._attempts == 4
        and control._models == []
        and len(control._pending) == 1
    )
    assert (
        wrapper._runtime is old
        and old.candidate_snapshot() == before
        and budget.updates_completed == 0
    )
    control.discard(token)


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, KeyboardInterrupt])
def test_should_keep_actor_available_after_partial_failed_update_and_stopped_handoff(error_type):
    class PartialFailure(PausableLearner):
        def __init__(self, value=None):
            super().__init__(value)
            self.calls = 0

        def fork(self):
            return PartialFailure(self.value)

        def train_batch(self, features, targets):
            self.calls += 1
            self.value[0] = 99
            raise error_type("partial failure")

    wrapper, old, clock, gate, source, budget = shared(source=PartialFailure())
    queue(old)
    clock.advance_to(3)
    with pytest.raises(error_type):
        wrapper.train_ready()
    assert old._stopped and old._candidate.calls == 1 and old._candidate.value == [99, 2]
    for _ in range(ROUNDS):
        with pytest.raises(ValueError, match="stopped"):
            wrapper.train_ready()
        assert wrapper.predict([0]).prediction == [2, 2]
    gate.pause()
    control = controller(wrapper)
    new = control.restore(control.capture())
    gate.resume()
    for _ in range(ROUNDS):
        with pytest.raises(ValueError, match="stopped"):
            wrapper.train_ready()
        assert wrapper.predict([0]).prediction == [2, 2]
    assert new._stopped and new._budget is budget and old._candidate.calls == 1
    assert (
        budget.updates_completed == 0
        and gate.snapshot().admitted_updates == 1
        and source.value == [2, 2]
    )


def test_should_refuse_repeated_payload_growth_without_refunding_capacity_after_purge():
    owner, cleanup, wrapper, runtime, _, gate, _, budget = bytes_setup(limit=16)
    owner.declare(declaration())
    source = Experience(
        "s1", "e1", 1, "actor-0", [1, 2], "train", ExperiencePermissions(True, True)
    )
    owner.record_experience(source)
    for i in range(ROUNDS):
        with pytest.raises(ValueError, match="byte"):
            owner.record_label(LabelArrival("large-" + str(i), "s1", "e1", 3, "actor-0", [0] * 128))
        assert wrapper.predict([0]).prediction == [2, 2]
    assert (
        runtime._inbox._labels == {}
        and cleanup.payload_byte_snapshot().charged_bytes == 16
        and cleanup.admitted_payload_bytes == 16
    )
    gate.pause()
    cleanup.delete((("e1", "s1"),))
    for _ in range(ROUNDS):
        with pytest.raises(ValueError, match="revoked"):
            owner.record_experience(source)
    assert (
        cleanup.payload_byte_snapshot().observed_retained_bytes == 0
        and cleanup.payload_byte_snapshot().charged_bytes == 16
    )
    assert budget.updates_completed == gate.snapshot().admitted_updates == 0


def test_should_refuse_repeated_stale_labels_and_cache_time_without_payload_copy_or_serving_change():
    actor, runtime, control = promotion_setup()
    actor.serve([0], now=10)
    before = pickle.dumps(actor.serving_snapshot())
    runtime.record_experience(
        Experience("s1", "e1", 1, "actor-0", [0], "train", ExperiencePermissions(True))
    )

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("stale payload copied")

    for i in range(ROUNDS):
        with pytest.raises(ValueError, match="version"):
            runtime.record_label(
                LabelArrival("stale-" + str(i), "s1", "e1", 3, "stale-actor", CannotCopy())
            )
        with pytest.raises(ValueError, match="backwards"):
            actor.serve([0], now=9)
        assert actor.serve([0], now=10).cache_hit
    assert (
        runtime._inbox._labels == {}
        and runtime._budget.updates_completed == 0
        and control._pending == {}
    )
    assert pickle.dumps(actor.serving_snapshot()) == before


@pytest.mark.parametrize("method", ["backprop", "circadian"])
def test_should_bound_eight_native_stream_cycles_with_original_authority_and_observed_memory(
    method,
):
    from src.adapters.numpy_learners import ManagedNumpyBuilder, make_managed_data_lifecycle
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.candidate_checkpoint import CandidateCheckpointController
    from src.app.managed_experience import ManagedExperienceOwner
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.serving_promotion import PromotableActor
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.data_lifecycle import LifecycleLimits
    from src.core.data_retention import DataRetentionPolicy
    from src.core.experience import LogicalClock
    from src.core.payload_bytes import PayloadCopyLimits
    from src.core.payload_ownership import PayloadOwnershipLimits
    from src.core.resource_sharing import SharingLimits
    from src.core.serving_ports import ServingConfiguration
    from src.shared.process_memory import ProcessRssSampler
    from test_experience_inbox import make_native_pair
    from test_managed_data_lifecycle import native_without_replay

    _, source = make_native_pair(method)
    source_before = pickle.dumps(source.snapshot_state())
    clock = LogicalClock()
    with ProcessRssSampler() as sampler:
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=8, max_process_rss_bytes=512 * 1024 * 1024),
            monotonic,
        )
        budget.process_rss_sampler = sampler
        actor = PromotableActor(
            source,
            version="actor-0",
            configuration=ServingConfiguration(3, 2),
            feature_digest=digest,
            metadata={},
        )
        old = ActorShadowRuntime(
            source,
            actor_version="actor-0",
            candidate_version="candidate-0",
            clock=clock,
            budget=budget,
            actor=actor,
            max_experiences=8,
        )
        gate = ServingPriorityGate(SharingLimits(2, 8, 1), resource_available=lambda: True)
        wrapper = ResourceSharedRuntime(old, gate)
        owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(8, 8, allow_synthetic=True))
        cleanup = make_managed_data_lifecycle(
            owner,
            policy=DataRetentionPolicy(
                2048,
                100,
                PayloadOwnershipLimits(16, 32),
                owned_payload_copies=PayloadCopyLimits(8192),
            ),
        )
        control = CandidateCheckpointController(
            wrapper,
            build_learner=ManagedNumpyBuilder(source),
            state_digest=digest,
            policy_digest=lambda m: digest("fixed"),
        )
        snapshots = []
        baseline = None
        retired = []
        features = np.array([[0.3, -0.2], [-0.5, 0.4]])
        targets = np.array([[1.0], [0.0]])
        assert not tracemalloc.is_tracing()
        tracemalloc.start()
        try:
            for i in range(8):
                sample = "s" + str(i)
                grant = declaration(sample)
                owner.declare(replace(grant, provenance=replace(grant.provenance, synthetic=True)))
                owner.record_experience(
                    Experience(
                        sample,
                        "e1",
                        i,
                        "actor-0",
                        features,
                        "train",
                        ExperiencePermissions(True, True),
                    )
                )
                owner.record_label(
                    LabelArrival("label-" + sample, sample, "e1", i, "actor-0", targets)
                )
                clock.advance_to(i)
                assert len(wrapper.train_ready().updates) == 1
                prediction = wrapper.serve(features, now=i).prediction
                if baseline is None:
                    baseline = prediction.copy()
                assert np.array_equal(prediction, baseline)
                if i in (0, 3):
                    gate.pause()
                    retired.append(wrapper._runtime)
                    wrapper_runtime = control.restore(control.capture())
                    gate.resume()
                    assert (
                        wrapper_runtime._budget is budget and wrapper_runtime._inbox._clock is clock
                    )
                model = wrapper._runtime._candidate
                before = native_without_replay(model)
                gate.pause()
                cleanup.delete((("e1", sample),))
                gate.resume()
                snapshot = cleanup.payload_byte_snapshot()
                assert snapshot.observed_retained_bytes == 0 and snapshot.charged_bytes <= 8192
                assert (
                    native_without_replay(model) == before
                    and budget.updates_completed == gate.snapshot().admitted_updates == i + 1
                )
                snapshots.append(asdict(snapshot))
                sampler.sample()
            for i in range(ROUNDS):
                with pytest.raises(ValueError, match="quota"):
                    owner.declare(declaration("extra-" + str(i)))
                with pytest.raises(ValueError, match="duplicate"):
                    owner.declare(declaration("s0"))
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 8 * 1024 * 1024
        finally:
            tracemalloc.stop()
        observed = sampler.snapshot()
        assert (
            observed.peak_bytes <= 512 * 1024 * 1024
            and observed.peak_bytes - observed.start_bytes <= 32 * 1024 * 1024
        )
        assert [r._inbox.capture_cursor().completed_updates for r in retired] == [1, 4]
        assert (
            control._attempts == 2
            and len(owner.declarations) == 8
            and len(wrapper._runtime.applied_updates) == 8
        )
        assert (
            cleanup.admitted_payload_bytes == 384
            and pickle.dumps(source.snapshot_state()) == source_before
        )
    assert sampler._thread is None or not sampler._thread.is_alive()
    receipt = {
        "method": method,
        "cycles": 8,
        "updates": 8,
        "native_predict_calls": 8,
        "sleep": 0,
        "rss": asdict(observed),
        "tracked_peak_bytes": peak,
        "byte_snapshots": snapshots,
        "retired_work": [1, 4],
        "ingress_bytes": 384,
        "spent_checkpoint_attempts": 2,
        "final_completed": budget.updates_completed,
        "source_unchanged": True,
        "process_restart_claim": False,
    }
    with (STAGE / ("native-summary-" + method + ".json")).open("x", encoding="utf8") as f:
        json.dump(receipt, f, indent=2)
