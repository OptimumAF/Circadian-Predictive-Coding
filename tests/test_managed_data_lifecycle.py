"""Original-authority conservative cleanup, quotas, expiry and failure."""

from dataclasses import replace
import pickle

import pytest

from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.core.data_retention import DataRetentionPolicy
from src.core.data_erasure import ReplayPayloadErasure
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.data_lifecycle import LifecycleLimits
from test_managed_experience import declaration, record
from test_resource_sharing import shared
from test_candidate_checkpoint import controller


def policy(byte_limit=1024, age=10):
    return DataRetentionPolicy(byte_limit, age, PayloadOwnershipLimits(16, 64))


def install(wrapper, *, declared=None, erase=None):
    def measure(value):
        return len(value) * 8

    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))
    cleanup = ManagedDataLifecycle(
        owner,
        policy=declared or policy(),
        measure_payload_bytes=measure,
        native_footprint=lambda model: ReplayPayloadErasure(0, 0, 0),
        native_erase=erase or (lambda model: ReplayPayloadErasure(0, 0, 0)),
    )
    return owner, cleanup


def setup(**kwargs):
    wrapper, runtime, clock, gate, source, budget = shared()
    owner, cleanup = install(wrapper, **kwargs)
    return owner, cleanup, wrapper, runtime, clock, gate, source, budget


def test_should_delete_all_delivered_copies_and_invalidate_checkpoint_without_unlearning():
    owner, cleanup, wrapper, runtime, clock, gate, _, budget = setup()
    owner.declare(declaration())
    record(owner)
    clock.advance_to(3)
    wrapper.train_ready()
    before = runtime.candidate_snapshot().state
    control = controller(wrapper)
    gate.pause()
    token = control.capture()
    external = control.inspect(token)
    report = cleanup.delete((("e1", "s1"),))
    assert report.reason == "deleted" and report.revoked_keys == (("e1", "s1"),)
    assert runtime._inbox._experiences == runtime._inbox._labels == {}
    assert runtime.candidate_snapshot().state == before and budget.updates_completed == 1
    assert control._pending == {} and external.inbox.experiences  # caller copy is outside deletion
    with pytest.raises(ValueError, match="checkpoint"):
        control.restore(token)
    with pytest.raises(ValueError, match="revoked"):
        record(owner)
    gate.resume()
    assert wrapper.train_ready().updates == ()


def test_should_clear_retired_inbox_after_live_budget_has_advanced():
    owner, cleanup, wrapper, old, clock, gate, _, budget = setup()
    owner.declare(declaration())
    record(owner)
    clock.advance_to(3)
    wrapper.train_ready()
    gate.pause()
    control = controller(wrapper)
    new = control.restore(control.capture())
    gate.resume()
    owner.declare(declaration("s2"))
    record(owner, "s2")
    wrapper.train_ready()
    assert budget.updates_completed == 2 and len(old.applied_updates) == 1
    gate.pause()
    cleanup.delete((("e1", "s1"),))
    assert (
        old._inbox._experiences
        == old._inbox._labels
        == new._inbox._experiences
        == new._inbox._labels
        == {}
    )
    assert old._inbox.capture_cursor().completed_updates == 1
    assert new._inbox.capture_cursor().completed_updates == 2 and new._budget is budget
    assert owner._revoked_keys == {("e1", "s1"), ("e1", "s2")}
    assert old._inbox._historical_completed_updates == 1


def test_should_revoke_requested_undelivered_key_without_fabricating_inbox_payload():
    owner, cleanup, _, runtime, _, gate, _, _ = setup()
    owner.declare(declaration())
    gate.pause()
    report = cleanup.delete((("e1", "s1"),))
    assert report.revoked_keys == (("e1", "s1"),) and runtime._inbox._erased == {}
    with pytest.raises(ValueError, match="revoked"):
        record(owner)


def test_should_integrate_subject_opt_out_with_cleanup_and_future_declaration_refusal():
    owner, cleanup, wrapper, runtime, clock, gate, _, budget = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    owner.opt_out("person")
    assert runtime._inbox._experiences == runtime._inbox._labels == {}
    with pytest.raises(ValueError, match="opted out"):
        owner.declare(declaration("s2"))
    clock.advance_to(3)
    gate.resume()
    assert wrapper.train_ready().updates == () and budget.updates_completed == 0


def test_should_block_expired_training_and_snapshots_then_purge_at_expire_boundary():
    owner, cleanup, wrapper, runtime, clock, gate, _, budget = setup(declared=policy(age=3))
    owner.declare(declaration())
    record(owner)
    clock.advance_to(3)
    for operation in (
        runtime.train_ready,
        runtime.candidate_snapshot,
        runtime.actor.snapshot_state,
    ):
        with pytest.raises(ValueError, match="expired"):
            operation()
    gate.pause()
    report = cleanup.expire()
    assert (
        report.reason == "expired" and runtime._inbox._experiences == runtime._inbox._labels == {}
    )
    gate.resume()
    assert wrapper.train_ready().updates == () and budget.updates_completed == 0


def test_should_not_refund_lifetime_ingress_bytes_after_cleanup():
    owner, cleanup, _, runtime, _, gate, _, _ = setup(declared=policy(byte_limit=24))
    owner.declare(declaration())
    record(owner)
    assert cleanup.admitted_payload_bytes == 24
    gate.pause()
    cleanup.delete((("e1", "s1"),))
    owner.declare(declaration("s2", "another"))
    with pytest.raises(ValueError, match="byte"):
        record(owner, "s2")
    assert cleanup.admitted_payload_bytes == 24 and runtime._inbox._experiences == {}


@pytest.mark.parametrize("kind", ["unpaused", "candidate", "actor", "controller", "serving"])
def test_should_refuse_busy_cleanup_without_partial_revocation_or_erasure(kind):
    owner, cleanup, wrapper, runtime, _, gate, _, _ = setup()
    owner.declare(declaration())
    record(owner)
    control = controller(wrapper)
    if kind != "unpaused":
        gate.pause()
    context = {
        "candidate": runtime._exclusive,
        "actor": runtime.actor._payload_exclusive,
        "controller": control._exclusive,
        "serving": gate.serving,
    }.get(kind)
    from contextlib import nullcontext

    with context() if context else nullcontext(), pytest.raises(ValueError, match="paused|busy"):
        cleanup.delete((("e1", "s1"),))
    assert owner._revoked_keys == set() and runtime._inbox._experiences and runtime._inbox._labels


def test_should_fail_closed_on_partial_native_cleanup_and_retry_without_budget_reset():
    fail = [True]

    def erase(model):
        if fail[0]:
            raise RuntimeError("injected cleanup failure")
        return ReplayPayloadErasure(0, 0, 0)

    owner, cleanup, wrapper, runtime, _, gate, _, budget = setup(erase=erase)
    owner.declare(declaration())
    record(owner)
    control = controller(wrapper)
    gate.pause()
    token = control.capture()
    with pytest.raises(RuntimeError):
        cleanup.delete((("e1", "s1"),))
    assert owner._revoked_keys == {("e1", "s1")} and control._pending == {} and runtime._stopped
    with pytest.raises(ValueError, match="failed|stopped"):
        runtime.train_ready()
    fail[0] = False
    cleanup.retry_cleanup()
    assert (
        runtime._inbox._experiences == runtime._inbox._labels == {}
        and budget.updates_completed == 0
    )
    with pytest.raises(ValueError, match="stopped"):
        runtime.train_ready()
    with pytest.raises(ValueError, match="checkpoint"):
        control.restore(token)


def test_should_recover_after_second_model_failure_without_reopening_stopped_owner():
    owner, cleanup, _, runtime, _, gate, _, budget = setup()
    owner.declare(declaration())
    record(owner)
    buffers = {id(runtime.actor._learner): [1], id(runtime._candidate): [2]}
    seen = []

    def footprint(model):
        return (
            ReplayPayloadErasure(1, 1, 8) if buffers[id(model)] else ReplayPayloadErasure(0, 0, 0)
        )

    def erase(model):
        seen.append(id(model))
        if len(seen) == 2:
            raise RuntimeError("second model failed")
        before = footprint(model)
        buffers[id(model)].clear()
        return before

    cleanup._footprint = footprint
    cleanup._erase = erase
    gate.pause()
    with pytest.raises(RuntimeError, match="second"):
        cleanup.delete((("e1", "s1"),))
    assert (
        not buffers[id(runtime.actor._learner)]
        and buffers[id(runtime._candidate)]
        and runtime._stopped
    )
    assert owner._revoked_keys == {("e1", "s1")} and runtime._inbox._experiences
    cleanup.retry_cleanup()
    assert all(not buffer for buffer in buffers.values()) and runtime._inbox._experiences == {}
    assert runtime._stopped and budget.updates_completed == 0


@pytest.mark.parametrize("boundary", ["manager", "incomplete", "footprint"])
def test_should_refuse_failed_preflight_without_mutation(boundary):
    from contextlib import nullcontext

    owner, cleanup, _, runtime, _, gate, _, _ = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    if boundary == "incomplete":
        runtime._payload_ready = False
    if boundary == "footprint":
        cleanup._footprint = lambda model: None
    with owner._operation() if boundary == "manager" else nullcontext(), pytest.raises(ValueError):
        cleanup.delete((("e1", "s1"),))
    assert not owner._revoked_keys and not runtime._stopped and runtime._inbox._experiences


def test_should_keep_age_anchors_bytes_and_original_authority_through_handoff():
    owner, cleanup, wrapper, runtime, clock, gate, _, budget = setup(declared=policy(age=4))
    owner.declare(declaration())
    record(owner)
    gate.pause()
    clock.advance_to(2)
    control = controller(wrapper)
    new = control.restore(control.capture())
    assert new._inbox._clock is clock and new._budget is budget and wrapper._sharing is gate
    assert owner._declaration_ticks == {("e1", "s1"): 0} and cleanup.admitted_payload_bytes == 24
    clock.advance_to(4)
    with pytest.raises(ValueError, match="expired"):
        new.candidate_snapshot()
    cleanup.expire()
    assert owner._revoked_keys == {("e1", "s1")} and cleanup.admitted_payload_bytes == 24
    assert (
        new._inbox._registration_guard.__self__ is owner
        and new._inbox._training_guard.__self__ is owner
    )


def test_should_refuse_unsupported_default_native_before_binding_policy():
    from src.core.data_lifecycle import LifecycleLimits

    wrapper, runtime, *_ = shared()
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))
    with pytest.raises(ValueError, match="supported"):
        ManagedDataLifecycle(owner, policy=policy())
    assert owner._lifecycle is None and runtime.actor._payload_registry._lifecycle is None


def test_should_preserve_original_lineage_and_refuse_unmanaged_new_candidate_or_large_controller():
    from src.app.actor_shadow import ActorShadowRuntime
    from src.core.experience import LogicalClock
    from test_actor_shadow import PausableLearner

    owner, cleanup, wrapper, runtime, _, _, _, budget = setup()
    with pytest.raises(ValueError, match="lineage"):
        ActorShadowRuntime(
            PausableLearner(),
            actor_version="actor-0",
            candidate_version="other",
            clock=LogicalClock(),
            budget=budget,
            actor=runtime.actor,
        )
    with pytest.raises(ValueError, match="capacity"):
        controller(wrapper, max_pending=99)
    assert cleanup.admitted_payload_bytes == 0


def test_should_refuse_unknown_deletion_or_backwards_clock_before_mutation():
    owner, cleanup, _, runtime, clock, gate, _, _ = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    with pytest.raises(ValueError, match="unknown"):
        cleanup.delete((("e1", "unknown"),))
    clock.advance_to(3)
    runtime._inbox._read_clock()
    clock._time = 0
    with pytest.raises(ValueError, match="backwards"):
        cleanup.delete((("e1", "s1"),))
    assert not owner._revoked_keys and runtime._inbox._experiences


def test_should_block_expired_checkpoint_inspection_and_restore_before_native_work():
    owner, cleanup, wrapper, runtime, clock, gate, _, _ = setup(declared=policy(age=3))
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper)
    token = control.capture()
    clock.advance_to(3)

    def no_native(*args):
        raise AssertionError("expired access touched native model")

    runtime._candidate.snapshot_state = no_native
    for operation in (
        lambda: control.inspect(token),
        lambda: control.restore(token),
        control.capture,
    ):
        with pytest.raises(ValueError, match="expired"):
            operation()
    assert control._attempts == 0 and token in control._pending
    cleanup.expire()
    assert control._pending == {}


@pytest.mark.parametrize(
    "change",
    ["budget", "sharing", "clock", "resource", "progress", "budget_policy", "sharing_limits"],
)
def test_should_refuse_replaced_original_authority_before_cleanup_or_admission(change):
    owner, cleanup, wrapper, runtime, clock, gate, _, _ = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    if change == "budget":
        runtime._budget = object()
    if change == "sharing":
        wrapper._sharing = gate.__class__(gate._limits, resource_available=lambda: True)
        wrapper._sharing.pause()
    if change == "clock":
        runtime._inbox._clock = clock.__class__()
    if change == "resource":
        gate._resource = lambda: True
    if change == "progress":
        runtime._budget.progress = object()
    if change == "budget_policy":
        runtime._budget.budget = replace(runtime._budget.budget, max_training_updates=999)
    if change == "sharing_limits":
        gate._limits = replace(gate._limits, max_admitted_updates=999)
    for operation in (
        lambda: cleanup.delete((("e1", "s1"),)),
        lambda: owner.declare(declaration("s2")),
    ):
        with pytest.raises(ValueError, match="authority"):
            operation()
    assert (
        not owner._revoked_keys
        and ("e1", "s2") not in owner._catalog
        and runtime._inbox._experiences
    )


def test_should_clear_failed_preparation_model_and_keep_spent_attempts():
    from test_actor_shadow import PausableLearner

    seen = []

    def erase(model):
        seen.append(model)
        return ReplayPayloadErasure(0, 0, 0)

    owner, cleanup, wrapper, runtime, _, gate, _, budget = setup(erase=erase)
    owner.declare(declaration())
    record(owner)

    def build(state):
        model = PausableLearner(state)
        model.phase = "restore_error"
        return model

    control = controller(wrapper, build_learner=build)
    gate.pause()
    token = control.capture()
    with pytest.raises(RuntimeError):
        control.restore(token)
    failed = control._models[0]
    cleanup.delete((("e1", "s1"),))
    assert any(m is failed for m in seen) and control._attempts == 1 and control._pending == {}
    assert budget.updates_completed == 0 and wrapper._runtime is runtime


def promotion_setup():
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.serving_promotion import PromotableActor, ServingPromotionController
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import LogicalClock
    from src.core.resource_sharing import SharingLimits
    from src.core.serving_ports import ServingConfiguration
    from test_serving_promotion import Source, digest
    from test_promotion_guard_evaluation import make_evaluator, policy as promotion_policy

    source = Source([0.5, 0.8])
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"payload": [1]},
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0),
        actor=actor,
    )
    runtime._candidate.restore_state([0.75, 0.8])  # deterministic fixture, no training/sleep
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    seen = []

    def erase(model):
        seen.append(model)
        return ReplayPayloadErasure(0, 0, 0)

    owner, cleanup = install(wrapper, erase=erase)
    evaluator = make_evaluator()
    control = ServingPromotionController(
        actor,
        policy=promotion_policy(),
        evaluator=evaluator,
        build_learner=evaluator._build,
        state_digest=digest,
    )
    return owner, cleanup, runtime, actor, control, gate, seen


def test_should_clear_promotion_current_previous_pending_auxiliary_and_disable_rollback():
    from test_serving_promotion import prepare

    owner, cleanup, runtime, actor, control, gate, seen = promotion_setup()
    owner.declare(declaration())
    actor.serve([0], now=1)
    first = prepare(control, runtime, metadata={"payload": [2]})
    second = prepare(control, runtime, metadata={"payload": [3]})
    pending = control._pending[second].bundle
    receipt = control.commit(runtime, first)
    current, previous = actor._slot.bundle, actor._slot.previous
    assert previous is not None
    actor.serve([0], now=2)
    gate.pause()
    before = actor.snapshot_state().state
    report = cleanup.delete((("e1", "s1"),))
    assert (
        report.promotions_invalidated == 1
        and actor._slot.previous is None
        and control._pending == {}
        and control._latest is None
    )
    for bundle in (current, previous, pending):
        assert bundle.metadata == bundle.cache == {} and any(m is bundle.learner for m in seen)
    assert actor.snapshot_state().state == before
    with pytest.raises(ValueError, match="ticket"):
        control.commit(runtime, second)
    with pytest.raises(ValueError, match="receipt"):
        control.rollback(receipt)


def test_should_prevalidate_auxiliary_before_revocation_or_native_erasure():
    owner, cleanup, _, actor, _, gate, seen = promotion_setup()
    owner.declare(declaration())
    gate.pause()
    actor._slot = replace(actor._slot, bundle=replace(actor._slot.bundle, metadata=[]))
    with pytest.raises(ValueError, match="auxiliary"):
        cleanup.delete((("e1", "s1"),))
    assert not owner._revoked_keys and seen == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_lifetime_ingress_bytes", True),
        ("max_retention_ticks", 0),
        ("max_checkpoint_pending", 0),
        ("max_checkpoint_preparations", -1),
        ("max_promotion_pending", False),
        ("holders", None),
    ],
)
def test_should_refuse_invalid_retention_policy(field, value):
    with pytest.raises(ValueError):
        replace(policy(), **{field: value})


@pytest.mark.parametrize(
    "keys", [(["e1", "s1"],), (("e1",),), (("e1", 1),), (("e1", "s1"), ("e1", "s1"))]
)
def test_should_refuse_malformed_cleanup_reports(keys):
    from src.core.data_retention import DataCleanupReport

    with pytest.raises(ValueError):
        DataCleanupReport("deleted", keys, (), 0, 0, 0, 0)


def test_should_deny_invalid_measurement_and_duplicate_registration_without_refund():
    owner, cleanup, _, runtime, _, gate, _, _ = setup()
    owner.declare(declaration())
    cleanup._measure = lambda value: True
    with pytest.raises(ValueError):
        record(owner)
    assert cleanup.admitted_payload_bytes == 0 and runtime._inbox._experiences == {}
    cleanup._measure = lambda value: len(value) * 8
    record(owner)
    with pytest.raises(ValueError, match="duplicate"):
        record(owner)
    assert cleanup.admitted_payload_bytes == 24
    gate.pause()
    cleanup.delete((("e1", "s1"),))
    with pytest.raises(ValueError, match="revoked"):
        record(owner)


def native_without_replay(model):
    snapshot = model.snapshot_state()
    snapshot.state.pop("_replay_memory", None)
    return pickle.dumps(snapshot)


@pytest.mark.parametrize("method", ["backprop", "circadian"])
def test_should_clean_real_owned_native_copies_and_continue_three_fixed_wakes(method):
    import numpy as np
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.data_lifecycle import LifecycleLimits
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
    from src.core.resource_sharing import SharingLimits
    from test_experience_inbox import make_native_pair
    from test_candidate_checkpoint import digest
    from src.app.candidate_checkpoint import CandidateCheckpointController

    _, source = make_native_pair(method)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source, actor_version="actor-0", candidate_version="candidate-0", clock=clock, budget=budget
    )
    gate = ServingPriorityGate(SharingLimits(2, 3, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4, allow_synthetic=True))
    from src.adapters.numpy_learners import make_managed_data_lifecycle

    cleanup = make_managed_data_lifecycle(owner, policy=policy(byte_limit=512, age=100))
    source_before = pickle.dumps(source.snapshot_state())

    def enqueue(sample, at):
        owner.declare(
            replace(
                declaration(sample),
                provenance=replace(declaration(sample).provenance, synthetic=True),
            )
        )
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

    enqueue("s1", 3)
    clock.advance_to(3)
    wrapper.train_ready()
    old = runtime
    gate.pause()
    control = CandidateCheckpointController(
        wrapper,
        build_learner=lambda state: source.fork(),
        state_digest=digest,
        policy_digest=lambda learner: digest("fixed"),
    )
    runtime = control.restore(control.capture())
    gate.resume()
    enqueue("s2", 5)
    clock.advance_to(5)
    wrapper.train_ready()
    gate.pause()
    token = control.capture()
    with runtime.actor._payload_registry._lease() as groups:
        models = cleanup._models(groups)
    before = {id(model): native_without_replay(model) for model in models}
    cleanup.delete((("e1", "s1"),))
    assert all(
        native_without_replay(model) == before[id(model)]
        and model.replay_payload_footprint().payload_bytes == 0
        for model in models
    )
    assert (
        cleanup._describe(old._candidate).payload_bytes
        == cleanup._describe(runtime._candidate).payload_bytes
        == 0
    )
    assert (
        old._inbox.capture_cursor().completed_updates == 1
        and runtime._inbox.capture_cursor().completed_updates == 2
    )
    with pytest.raises(ValueError):
        control.restore(token)
    gate.resume()
    enqueue("s3", 7)
    clock.advance_to(7)
    wrapper.train_ready()
    gate.pause()
    cleanup.delete((("e1", "s3"),))
    assert budget.updates_completed == gate.snapshot().admitted_updates == 3
    assert (
        cleanup.admitted_payload_bytes == 144
        and pickle.dumps(source.snapshot_state()) == source_before
    )
