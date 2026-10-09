"""Owned metadata age is independent of its measured numeric array bytes."""

from dataclasses import replace
import pickle
import numpy as np
import pytest

from src.adapters.numpy_learners import _managed_auxiliary_bytes
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.retention_expiry import RetentionExpiryDriver
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import LifecycleLimits
from src.core.experience import LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.resource_sharing import SharingLimits
from src.core.retention_driver import RetentionDriverLimits
from src.core.serving_ports import CachedPrediction, ServingConfiguration
from test_managed_data_lifecycle import policy
from test_promotion_guard_evaluation import make_evaluator, policy as promotion_policy
from test_retention_expiry import Clock
from test_serving_promotion import Source, digest, prepare


def setup(metadata=None, *, cache=None, copy_policy=True):
    source = Source([0.5, 0.8])
    wall = Clock()
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata=metadata or {},
    )
    if cache is not None:
        actor._slot.bundle.cache.update(cache)
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), wall),
        actor=actor,
    )
    runtime._candidate.restore_state([0.75, 0.8])
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4))
    cleanup = ManagedDataLifecycle(
        owner,
        policy=replace(
            policy(),
            max_retention_seconds=2,
            owned_payload_copies=PayloadCopyLimits(128) if copy_policy else None,
        ),
        measure_payload_bytes=_managed_auxiliary_bytes,
        measure_auxiliary_bytes=_managed_auxiliary_bytes,
        native_footprint=lambda m: ReplayPayloadErasure(0, 0, 0),
        native_erase=lambda m: ReplayPayloadErasure(0, 0, 0),
        measure_checkpoint_bytes=lambda v: 0,
        native_growth_bytes=lambda m, f, t: 0,
        prepare_model_bytes=lambda b, s: 0,
        prediction_cache_bytes=lambda m, f: 0,
    )
    evaluator = make_evaluator()
    control = ServingPromotionController(
        actor,
        policy=promotion_policy(),
        evaluator=evaluator,
        build_learner=evaluator._build,
        state_digest=digest,
    )
    driver = RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, 100, 2))
    return owner, cleanup, driver, actor, runtime, control, wall, gate


@pytest.mark.parametrize(
    "metadata",
    [
        {"raw": "text"},
        {"raw": None},
        {"raw": False},
        {"raw": 0},
        {"raw": []},
        {"raw": {}},
        {"raw": np.empty((0, 2))},
    ],
)
def test_should_expire_nonempty_initial_metadata_with_zero_array_bytes(metadata):
    owner, cleanup, driver, actor, runtime, _, wall, gate = setup(metadata)
    assert cleanup.payload_byte_snapshot().charged_bytes == 0
    assert cleanup._auxiliary_started_at == 0.0
    wall.value = 1.99
    assert driver.poll_once().outcome == "idle"
    wall.value = 2
    with pytest.raises(ValueError, match="expired"):
        actor.serving_snapshot()
    before = pickle.dumps(runtime._candidate.snapshot_state())
    result = driver.poll_once()
    assert result.outcome == "purged" and actor._slot.bundle.metadata == {}
    assert cleanup._auxiliary_started_at is None and owner.declarations == ()
    assert (
        cleanup.payload_byte_snapshot().charged_bytes == 0
        and runtime._budget.updates_completed == 0
        and not gate.snapshot().paused
    )
    assert pickle.dumps(runtime._candidate.snapshot_state()) == before


@pytest.mark.parametrize("prediction", [0, "text", [], np.empty((0,))])
def test_should_expire_initial_zero_array_cache_entries(prediction):
    _, cleanup, driver, actor, _, _, wall, _ = setup(
        cache={digest([0]): CachedPrediction(100, prediction)}
    )
    assert cleanup._auxiliary_started_at == 0 and cleanup.payload_byte_snapshot().charged_bytes == 0
    wall.value = 2
    assert driver.poll_once().outcome == "purged" and actor._slot.bundle.cache == {}


def test_should_leave_empty_owned_containers_without_a_deadline_until_first_copy():
    _, cleanup, driver, actor, _, _, wall, _ = setup()
    wall.value = 20
    assert driver.poll_once().outcome == "idle" and cleanup._auxiliary_started_at is None
    actor.serve([0], now=0)
    assert cleanup._auxiliary_started_at == 20 and actor._slot.bundle.cache
    wall.value = 21
    actor.serve([1], now=1)
    assert cleanup._auxiliary_started_at == 20
    wall.value = 22
    assert driver.poll_once().outcome == "purged" and actor._slot.bundle.cache == {}


def test_should_anchor_prepared_scalar_metadata_without_renewing_on_discard_or_new_copy():
    _, cleanup, driver, actor, runtime, control, wall, _ = setup()
    wall.value = 1
    ticket = prepare(control, runtime, metadata={"raw": "first"})
    assert (
        cleanup._auxiliary_started_at == 1
        and control._pending
        and cleanup.payload_byte_snapshot().charged_bytes == 0
    )
    control.discard(ticket)
    wall.value = 2
    ticket = prepare(control, runtime, metadata={"raw": None})
    control.commit(runtime, ticket)
    assert cleanup._auxiliary_started_at == 1 and actor._slot.bundle.metadata == {"raw": None}
    wall.value = 3
    assert (
        driver.poll_once().outcome == "purged"
        and actor._slot.bundle.metadata == {}
        and control._latest is None
    )
    wall.value = 4
    actor.serve([0], now=0)
    assert cleanup._auxiliary_started_at == 4
    wall.value = 6
    assert driver.poll_once().outcome == "purged" and actor._slot.bundle.cache == {}


def test_should_keep_first_auxiliary_age_when_promotion_is_rejected():
    _, cleanup, driver, actor, runtime, control, wall, _ = setup()
    wall.value = 1
    declared = replace(control._policy, min_new_utility=0.99)
    control._policy = declared
    with pytest.raises(ValueError, match="rejected"):
        prepare(control, runtime, metadata={"raw": "attempt"})
    assert (
        cleanup._auxiliary_started_at == 1
        and control._pending == {}
        and actor._slot.bundle.metadata == {}
    )
    wall.value = 3
    assert driver.poll_once().outcome == "purged" and cleanup._auxiliary_started_at is None


def test_should_anchor_scalar_metadata_without_optional_array_byte_policy():
    _, cleanup, driver, actor, _, _, wall, _ = setup({"raw": "text"}, copy_policy=False)
    wall.value = 2
    assert driver.poll_once().outcome == "purged" and actor._slot.bundle.metadata == {}
    assert cleanup._copy_budget is None


def test_should_purge_scalar_metadata_using_actual_bounded_worker():
    _, _, driver, actor, _, _, wall, _ = setup({"raw": "text"})
    driver.start()
    try:
        wall.value = 2
        driver.wake()
        assert driver.wait_for_purge(1.0) and actor._slot.bundle.metadata == {}
    finally:
        assert driver.stop()
    assert not driver.snapshot().alive


def test_should_validate_every_initial_graph_without_short_circuiting_at_first_array():
    source = Source([0.5, 0.8])
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"raw": np.ones(1)},
    )
    actor._slot.bundle.cache["bad"] = CachedPrediction(100, object())
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), Clock()),
        actor=actor,
    )
    owner = ManagedExperienceOwner(
        ResourceSharedRuntime(
            runtime, ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
        ),
        limits=LifecycleLimits(1, 1),
    )
    with pytest.raises(ValueError, match="unsupported"):
        ManagedDataLifecycle(
            owner,
            policy=replace(policy(), max_retention_seconds=2),
            measure_payload_bytes=_managed_auxiliary_bytes,
            measure_auxiliary_bytes=_managed_auxiliary_bytes,
            native_footprint=lambda m: ReplayPayloadErasure(0, 0, 0),
            native_erase=lambda m: ReplayPayloadErasure(0, 0, 0),
        )
    assert owner._lifecycle is None and actor._payload_registry._lifecycle is None


@pytest.mark.parametrize("method", ["backprop", "circadian"])
def test_should_expire_scalar_metadata_before_sample_age_across_native_handoff(method):
    from src.adapters.numpy_learners import ManagedNumpyBuilder, make_managed_data_lifecycle
    from src.app.candidate_checkpoint import CandidateCheckpointController
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival
    from test_candidate_checkpoint import digest
    from test_experience_inbox import make_native_pair
    from test_managed_experience import declaration
    from test_managed_data_lifecycle import native_without_replay

    _, source = make_native_pair(method)
    wall = Clock()
    logical = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), wall)
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"raw": "scalar-only"},
    )
    old = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=logical,
        budget=budget,
        actor=actor,
    )
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(old, gate)
    owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4, allow_synthetic=True))
    cleanup = make_managed_data_lifecycle(
        owner,
        policy=replace(
            policy(byte_limit=512, age=100),
            max_retention_seconds=2,
            owned_payload_copies=PayloadCopyLimits(1024),
        ),
    )
    driver = RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, 100, 2))
    source_before = pickle.dumps(source.snapshot_state())

    def enqueue(sample, at):
        grant = declaration(sample)
        owner.declare(replace(grant, provenance=replace(grant.provenance, synthetic=True)))
        owner.record_experience(
            Experience(
                sample,
                "e1",
                1,
                "actor-0",
                np.array([[0.3, -0.2], [-0.5, 0.4]]),
                "train",
                ExperiencePermissions(True, True),
            )
        )
        owner.record_label(
            LabelArrival("label-" + sample, sample, "e1", at, "actor-0", np.array([[1.0], [0.0]]))
        )

    wall.value = 1
    enqueue("s1", 3)
    logical.advance_to(3)
    shared.train_ready()
    gate.pause()
    control = CandidateCheckpointController(
        shared,
        build_learner=ManagedNumpyBuilder(source),
        state_digest=digest,
        policy_digest=lambda m: digest("fixed"),
    )
    new = control.restore(control.capture())
    token = control.capture()
    gate.resume()
    before = {
        id(m): native_without_replay(m)
        for m in (old._candidate, new._candidate, actor._slot.bundle.learner)
    }
    charged = cleanup.payload_byte_snapshot().charged_bytes
    assert cleanup._auxiliary_started_at == 0 and owner._declaration_seconds[("e1", "s1")] == 1
    driver.start()
    try:
        wall.value = 2
        driver.wake()
        assert driver.wait_for_purge(1)
        assert (
            actor._slot.bundle.metadata == {}
            and old._inbox._experiences == new._inbox._experiences == {}
            and control._pending == {}
        )
        for m in (old._candidate, new._candidate, actor._slot.bundle.learner):
            assert (
                native_without_replay(m) == before[id(m)]
                and cleanup._describe(m).payload_bytes == 0
            )
        assert (
            cleanup.payload_byte_snapshot().observed_retained_bytes == 0
            and cleanup.payload_byte_snapshot().charged_bytes == charged
        )
        with pytest.raises(ValueError, match="checkpoint"):
            control.restore(token)
        enqueue("s2", 5)
        logical.advance_to(5)
        shared.train_ready()
        assert (
            budget.updates_completed == gate.snapshot().admitted_updates == 2
            and old._inbox.capture_cursor().completed_updates == 1
        )
    finally:
        assert driver.stop()
    assert not driver.snapshot().alive and new._inbox._experiences == {}
    assert (
        cleanup.admitted_payload_bytes == 96
        and pickle.dumps(source.snapshot_state()) == source_before
    )
