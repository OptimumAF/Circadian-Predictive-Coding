"""Shared copy allowances, exact retained counts and pre-copy denial."""

from dataclasses import replace
from typing import Any
import pytest

from src.app.payload_copy_budget import PayloadCopyBudget
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.core.payload_bytes import PayloadCopyLimits
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import LifecycleLimits
from test_managed_data_lifecycle import policy
from test_managed_experience import declaration, record
from test_resource_sharing import shared
from test_candidate_checkpoint import controller


def graph(value):
    from src.core.serving_ports import CachedPrediction

    if type(value) is CachedPrediction:
        return graph(value.prediction)
    if type(value) is dict:
        return sum(graph(v) for v in value.values())
    if type(value) is tuple:
        return sum(graph(v) for v in value)
    if type(value) is list:
        return len(value) * 8
    if value is None or type(value) in (str, int, float, bool):
        return 0
    raise ValueError("unsupported fake payload graph")


def view_bytes(view):
    return sum(graph(s.features) for s in view.inbox.experiences) + sum(
        graph(t.targets) for t in view.inbox.labels
    )


def setup(limit=256, growth=0, prepare=None):
    wrapper, runtime, clock, gate, source, budget = shared()
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))
    cleanup = ManagedDataLifecycle(
        owner,
        policy=replace(policy(), owned_payload_copies=PayloadCopyLimits(limit)),
        measure_payload_bytes=graph,
        native_footprint=lambda m: ReplayPayloadErasure(0, 0, 0),
        native_erase=lambda m: ReplayPayloadErasure(0, 0, 0),
        measure_auxiliary_bytes=graph,
        measure_checkpoint_bytes=view_bytes,
        native_growth_bytes=lambda m, f, t: growth,
        prepare_model_bytes=prepare or (lambda b, s: 0),
        prediction_cache_bytes=lambda m, f: graph(f),
    )
    return owner, cleanup, wrapper, runtime, clock, gate, source, budget


@pytest.mark.parametrize("value", [True, -1, 1.5, None])
def test_should_reject_invalid_payload_copy_limits(value):
    with pytest.raises(ValueError):
        PayloadCopyLimits(value)


def test_should_charge_attempts_and_refuse_over_quota_without_refund():
    budget = PayloadCopyBudget(PayloadCopyLimits(24))
    budget.reserve(8)
    budget.reserve(16)
    with pytest.raises(ValueError, match="byte"):
        budget.reserve(1)
    assert budget.snapshot(24).charged_bytes == 24
    assert budget.snapshot(0).charged_bytes == 24
    with pytest.raises(ValueError):
        budget.snapshot(25)


def test_should_deny_ingress_before_opaque_copy_and_keep_original_budgets():
    owner, cleanup, _, runtime, _, _, _, budget = setup(7)
    owner.declare(declaration())
    with pytest.raises(ValueError, match="byte"):
        record(owner)
    assert (
        runtime._inbox._experiences == {}
        and cleanup.admitted_payload_bytes == 0
        and budget.updates_completed == 0
    )
    assert cleanup.payload_byte_snapshot().charged_bytes == 0


def test_should_deny_native_growth_before_native_work_and_keep_attempt_charge():
    owner, cleanup, wrapper, runtime, clock, gate, _, budget = setup(24, growth=24)
    owner.declare(declaration())
    record(owner)
    clock.advance_to(3)
    with pytest.raises(ValueError, match="byte"):
        wrapper.train_ready()
    assert (
        budget.updates_completed == 0
        and not runtime._stopped
        and gate.snapshot().admitted_updates == 1
    )
    assert runtime.candidate_snapshot().state == [2, 2]
    assert cleanup.payload_byte_snapshot().observed_retained_bytes == 24


def test_should_deny_checkpoint_before_snapshot_copy_and_keep_existing_payload():
    owner, cleanup, wrapper, runtime, _, gate, _, _ = setup(24)
    owner.declare(declaration())
    record(owner)
    gate.pause()

    def no_snapshot():
        raise AssertionError("quota denial copied native snapshot")

    runtime._candidate.snapshot_state = no_snapshot
    control = controller(wrapper)
    with pytest.raises(ValueError, match="byte"):
        control.capture()
    assert control._pending == {} and runtime._inbox._experiences
    assert cleanup.payload_byte_snapshot().charged_bytes == 24


def test_should_count_pending_views_retired_inboxes_and_keep_ledger_across_cleanup():
    owner, cleanup, wrapper, old, _, gate, _, budget = setup(72)
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper)
    token = control.capture()
    assert cleanup.payload_byte_snapshot().observed_retained_bytes == 48
    new = control.restore(token)
    report = cleanup.payload_byte_snapshot()
    assert report.charged_bytes == 72 and report.observed_retained_bytes == 48
    assert new._budget is budget and new._inbox._payload_copy_guard.__self__ is cleanup
    cleanup.delete((("e1", "s1"),))
    assert (
        cleanup.payload_byte_snapshot().observed_retained_bytes == 0
        and cleanup.payload_byte_snapshot().charged_bytes == 72
    )
    assert old._inbox._experiences == new._inbox._experiences == {}
    owner.declare(declaration("s2"))
    with pytest.raises(ValueError, match="byte"):
        record(owner, "s2")


def test_should_deny_preparation_before_builder_and_keep_checkpoint_attempt_spent():
    def unexpected(state):
        raise AssertionError("quota denial called builder")

    owner, cleanup, wrapper, runtime, _, gate, _, _ = setup(48, prepare=lambda b, s: 8)
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper, build_learner=unexpected)
    token = control.capture()
    with pytest.raises(ValueError, match="byte"):
        control.restore(token)
    assert control._attempts == 1 and token in control._pending and wrapper._runtime is runtime
    assert cleanup.payload_byte_snapshot().charged_bytes == 48


def test_should_preserve_failed_preparation_reservation():
    def fail(state):
        raise RuntimeError("builder failure")

    owner, cleanup, wrapper, _, _, gate, _, _ = setup(80, prepare=lambda b, s: 8)
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper, build_learner=fail)
    token = control.capture()
    with pytest.raises(RuntimeError, match="builder"):
        control.restore(token)
    assert cleanup.payload_byte_snapshot().charged_bytes == 80 and control._attempts == 1


def test_should_refuse_unsupported_graph_before_checkpoint_or_admission_copy():
    owner, cleanup, _, runtime, _, gate, _, _ = setup()
    owner.declare(declaration())
    from src.core.experience import Experience, ExperiencePermissions

    class Opaque:
        def __deepcopy__(self, memo):
            raise AssertionError("opaque copied")

    with pytest.raises(ValueError, match="unsupported"):
        owner.record_experience(
            Experience(
                "s1", "e1", 0, "actor-0", Opaque(), "train", ExperiencePermissions(True, True)
            )
        )
    assert runtime._inbox._experiences == {} and cleanup.payload_byte_snapshot().charged_bytes == 0


def promotion_setup(limit):
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
    runtime._candidate.restore_state([0.75, 0.8])  # deterministic learned-parameter fixture
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))
    cleanup = ManagedDataLifecycle(
        owner,
        policy=replace(policy(), owned_payload_copies=PayloadCopyLimits(limit)),
        measure_payload_bytes=graph,
        native_footprint=lambda m: ReplayPayloadErasure(0, 0, 0),
        native_erase=lambda m: ReplayPayloadErasure(0, 0, 0),
        measure_auxiliary_bytes=graph,
        measure_checkpoint_bytes=view_bytes,
        native_growth_bytes=lambda m, f, t: 0,
        prepare_model_bytes=lambda b, s: 0,
        prediction_cache_bytes=lambda m, f: graph(f),
    )
    evaluator = make_evaluator()
    control = ServingPromotionController(
        actor,
        policy=promotion_policy(),
        evaluator=evaluator,
        build_learner=evaluator._build,
        state_digest=digest,
    )
    return owner, cleanup, runtime, actor, control, gate


def test_should_count_current_previous_pending_promotion_and_cached_outputs():
    from test_serving_promotion import prepare

    owner, cleanup, runtime, actor, control, gate = promotion_setup(40)
    owner.declare(declaration())
    assert cleanup.payload_byte_snapshot().observed_retained_bytes == 8
    actor.serve([0], now=1)
    first = prepare(control, runtime, metadata={"payload": [2]})
    second = prepare(control, runtime, metadata={"payload": [3]})
    control.commit(runtime, first)
    actor.serve([0], now=2)
    before = cleanup.payload_byte_snapshot()
    assert before.charged_bytes == before.observed_retained_bytes == 40

    def no_guard(*args, **kwargs):
        raise AssertionError("byte denial evaluated guard")

    control._evaluator.evaluate = no_guard
    with pytest.raises(ValueError, match="byte"):
        prepare(control, runtime, metadata={"payload": [4]})
    assert second in control._pending and cleanup.payload_byte_snapshot() == before
    gate.pause()
    cleanup.delete((("e1", "s1"),))
    assert (
        cleanup.payload_byte_snapshot().observed_retained_bytes == 0
        and cleanup.payload_byte_snapshot().charged_bytes == 40
    )


def test_should_deny_cache_before_native_prediction_and_not_charge_hits_twice():
    _, cleanup, _, actor, _, _ = promotion_setup(16)
    actor.serve([0], now=1)
    assert actor.serve([0], now=2).cache_hit
    assert cleanup.payload_byte_snapshot().charged_bytes == 16

    def no_predict(features):
        raise AssertionError("cache denial called native prediction")

    actor._slot.bundle.learner.predict = no_predict
    with pytest.raises(ValueError, match="byte"):
        actor.serve([1], now=2)
    assert cleanup.payload_byte_snapshot().observed_retained_bytes == 16


def test_should_refuse_opaque_promotion_metadata_before_evaluation_or_copy():
    from test_serving_promotion import prepare

    _, cleanup, runtime, _, control, _ = promotion_setup(128)

    class Opaque:
        def __deepcopy__(self, memo):
            raise AssertionError("metadata copied")

    def no_guard(*args, **kwargs):
        raise AssertionError("unsupported metadata evaluated guard")

    control._evaluator.evaluate = no_guard
    with pytest.raises(ValueError, match="unsupported"):
        prepare(control, runtime, metadata={"raw": Opaque()})
    assert cleanup.payload_byte_snapshot().charged_bytes == 8


@pytest.mark.parametrize("kind", ["cycle", "opaque", "object_array", "too_wide", "too_deep"])
def test_should_refuse_unsupported_numpy_graphs_without_opaque_hooks(kind):
    import numpy as np
    from src.adapters.numpy_learners import _managed_auxiliary_bytes

    class Opaque:
        def __iter__(self):
            raise AssertionError("opaque iterated")

        def __deepcopy__(self, memo):
            raise AssertionError("opaque copied")

    value: Any
    if kind == "cycle":
        value = []
        value.append(value)
    if kind == "opaque":
        value = Opaque()
    if kind == "object_array":
        value = np.array([Opaque()], dtype=object)
    if kind == "too_wide":
        value = [None] * 4097
    if kind == "too_deep":
        value = []
        for _ in range(33):
            value = [value]
    with pytest.raises(ValueError, match="unsupported|bounds"):
        _managed_auxiliary_bytes(value)


def test_should_count_aliases_once_within_each_copied_numpy_graph():
    import numpy as np
    from src.adapters.numpy_learners import _managed_auxiliary_bytes

    array = np.arange(3, dtype=np.float64)
    nested = {"a": [array, array], "b": array}
    assert _managed_auxiliary_bytes(nested) == 24
    assert _managed_auxiliary_bytes({"a": array.copy(), "b": array.copy()}) == 48


def test_should_refuse_busy_byte_budget_without_consuming_capacity():
    budget = PayloadCopyBudget(PayloadCopyLimits(8))
    with budget._gate, pytest.raises(ValueError, match="busy"):
        budget.reserve(8)
    assert budget.snapshot(0).charged_bytes == 0


def test_should_preserve_byte_policy_and_age_anchor_through_refused_restore():
    owner, cleanup, wrapper, runtime, clock, gate, _, budget = setup(48)
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper)
    token = control.capture()
    clock.advance_to(2)
    with pytest.raises(ValueError, match="byte"):
        control.restore(token)
    assert cleanup.payload_byte_snapshot().charged_bytes == 48 and owner._declaration_ticks == {
        ("e1", "s1"): 0
    }
    assert runtime._budget is budget and wrapper._sharing is gate


def test_should_refuse_checkpoint_controller_on_alternate_sharing_wrapper():
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate

    owner, cleanup, wrapper, runtime, _, gate, _, _ = setup()
    second = ResourceSharedRuntime(
        runtime, ServingPriorityGate(gate._limits, resource_available=lambda: True)
    )
    with pytest.raises(ValueError, match="original|lineage"):
        controller(second)
    assert (
        wrapper._runtime is runtime
        and cleanup.payload_byte_snapshot().charged_bytes == 0
        and owner.declarations == ()
    )


@pytest.mark.parametrize("value", [-8, True, 1.5, None])
def test_should_reject_invalid_preparation_counts_before_aggregation(value):
    owner, cleanup, wrapper, runtime, _, gate, _, _ = setup(256, prepare=lambda b, s: value)
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper)
    token = control.capture()
    with pytest.raises(ValueError):
        control.restore(token)
    assert (
        wrapper._runtime is runtime
        and control._attempts == 1
        and cleanup.payload_byte_snapshot().charged_bytes == 48
    )


def test_should_charge_failed_ingress_copy_without_refunding_lifetime_ledgers():
    owner, cleanup, _, runtime, _, _, _, _ = setup(8)
    owner.declare(declaration())
    from src.core.experience import Experience, ExperiencePermissions

    class FailCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("copy failed")

    cleanup._measure = lambda value: 8
    with pytest.raises(RuntimeError, match="copy failed"):
        owner.record_experience(
            Experience(
                "s1", "e1", 0, "actor-0", FailCopy(), "train", ExperiencePermissions(True, True)
            )
        )
    assert (
        cleanup.admitted_payload_bytes == 8
        and cleanup.payload_byte_snapshot().charged_bytes == 8
        and runtime._inbox._experiences == {}
    )


def test_should_refuse_opaque_default_builder_before_invocation():
    from src.adapters.numpy_learners import BackpropLearner, make_managed_data_lifecycle
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.backprop_mlp import BackpropMLP
    from src.core.experience import LogicalClock
    from src.core.resource_sharing import SharingLimits
    from src.app.candidate_checkpoint import CandidateCheckpointController
    from test_candidate_checkpoint import digest

    source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
    )
    gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(1, 1))
    cleanup = make_managed_data_lifecycle(
        owner, policy=replace(policy(), owned_payload_copies=PayloadCopyLimits(128))
    )

    def opaque(state):
        raise AssertionError("opaque builder invoked")

    control = CandidateCheckpointController(
        wrapper, build_learner=opaque, state_digest=digest, policy_digest=lambda m: digest("fixed")
    )
    gate.pause()
    token = control.capture()
    with pytest.raises(ValueError, match="inspectable"):
        control.restore(token)
    assert (
        control._attempts == 1
        and cleanup.payload_byte_snapshot().charged_bytes == 0
        and control._models == []
    )


@pytest.mark.parametrize("method", ["backprop", "circadian"])
def test_should_bound_real_owned_copies_across_handoff_cache_and_two_fixed_wakes(method):
    import numpy as np
    import pickle
    from src.adapters.numpy_learners import ManagedNumpyBuilder, make_managed_data_lifecycle
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.serving_promotion import PromotableActor
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
    from src.core.resource_sharing import SharingLimits
    from src.core.serving_ports import ServingConfiguration
    from test_experience_inbox import make_native_pair
    from test_candidate_checkpoint import digest
    from src.app.candidate_checkpoint import CandidateCheckpointController

    _, source = make_native_pair(method)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(10, 2),
        feature_digest=digest,
        metadata={"raw": np.array([1.0])},
    )
    old = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        actor=actor,
    )
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(old, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4, allow_synthetic=True))
    cleanup = make_managed_data_lifecycle(
        owner,
        policy=replace(
            policy(byte_limit=512, age=100), owned_payload_copies=PayloadCopyLimits(1024)
        ),
    )
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

    enqueue("s1", 3)
    clock.advance_to(3)
    wrapper.train_ready()
    actor.serve(np.array([[0.3, -0.2], [-0.5, 0.4]]), now=3)  # sole NEW feedforward call per method
    gate.pause()
    builder = ManagedNumpyBuilder(source)
    control = CandidateCheckpointController(
        wrapper, build_learner=builder, state_digest=digest, policy_digest=lambda m: digest("fixed")
    )
    token = control.capture()
    pending = cleanup.payload_byte_snapshot()
    assert pending.observed_retained_bytes == (120 if method == "backprop" else 216)
    new = control.restore(token)
    after = cleanup.payload_byte_snapshot()
    assert (
        after.observed_retained_bytes == pending.observed_retained_bytes
        and after.charged_bytes == (168 if method == "backprop" else 312)
    )
    cleanup.delete((("e1", "s1"),))
    assert cleanup.payload_byte_snapshot().observed_retained_bytes == 0
    assert cleanup.payload_byte_snapshot().charged_bytes == after.charged_bytes
    gate.resume()
    enqueue("s2", 5)
    clock.advance_to(5)
    wrapper.train_ready()
    gate.pause()
    final = cleanup.payload_byte_snapshot()
    assert final.observed_retained_bytes == (48 if method == "backprop" else 96)
    assert (
        budget.updates_completed == gate.snapshot().admitted_updates == 2
        and new._budget is budget
        and old._inbox.capture_cursor().completed_updates == 1
    )
    assert (
        pickle.dumps(source.snapshot_state()) == source_before
        and cleanup.admitted_payload_bytes == 96
    )
