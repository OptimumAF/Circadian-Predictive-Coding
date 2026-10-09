"""Original-clock elapsed purge, quiescent retries and bounded automatic worker."""

from dataclasses import replace
import pytest

from src.app.retention_expiry import RetentionExpiryDriver
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.core.retention_driver import RetentionDriverLimits
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import LifecycleLimits
from test_managed_data_lifecycle import policy
from test_managed_experience import declaration, record
from test_resource_sharing import shared
from test_retained_payload_budget import graph
from test_candidate_checkpoint import controller


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def setup(*, polls=20, erase=None):
    wrapper, runtime, logical, gate, source, budget = shared()
    wall = Clock()
    budget.clock = wall
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))
    cleanup = ManagedDataLifecycle(
        owner,
        policy=replace(policy(), max_retention_seconds=2.0),
        measure_payload_bytes=graph,
        native_footprint=lambda model: ReplayPayloadErasure(0, 0, 0),
        native_erase=erase or (lambda model: ReplayPayloadErasure(0, 0, 0)),
        measure_auxiliary_bytes=graph,
    )
    driver = RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, polls, 2.0))
    return owner, cleanup, driver, wrapper, runtime, logical, wall, gate, budget


def test_should_anchor_elapsed_deadline_without_resetting_logical_age_or_budget():
    owner, cleanup, driver, _, runtime, logical, wall, gate, budget = setup()
    owner.declare(declaration())
    record(owner)
    wall.value = 1.99
    assert driver.poll_once().outcome == "idle" and runtime._inbox._experiences
    wall.value = 2.0
    with pytest.raises(ValueError, match="expired"):
        runtime.candidate_snapshot()
    result = driver.poll_once()
    assert (
        result.outcome == "purged" and runtime._inbox._experiences == runtime._inbox._labels == {}
    )
    assert (
        not gate.snapshot().paused
        and budget.updates_completed == 0
        and owner._declaration_ticks == {("e1", "s1"): 0}
    )
    assert owner._declaration_seconds == {("e1", "s1"): 0.0} and logical.now() == 0
    assert cleanup.admitted_payload_bytes == 24


@pytest.mark.parametrize(
    "kind", ["candidate", "actor", "manager", "serving", "checkpoint", "registry"]
)
def test_should_retry_busy_purge_with_independent_pause_and_overdue_access_refusal(kind):
    owner, _, driver, wrapper, runtime, _, wall, gate, budget = setup()
    owner.declare(declaration())
    record(owner)
    control = controller(wrapper)
    wall.value = 2.0
    contexts = {
        "candidate": runtime._exclusive,
        "actor": runtime.actor._payload_exclusive,
        "manager": owner._operation,
        "serving": gate.serving,
        "checkpoint": control._exclusive,
        "registry": lambda: runtime.actor._payload_registry._gate,
    }
    with contexts[kind]():
        assert driver.poll_once().outcome == "busy" and runtime._inbox._experiences
        gate.resume()
        assert gate.snapshot().paused
    assert driver.poll_once().outcome == "purged" and not gate.snapshot().paused
    assert not runtime._inbox._experiences and budget.updates_completed == 0


def test_should_preserve_manual_pause_after_automatic_cleanup():
    owner, _, driver, _, _, _, wall, gate, _ = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    wall.value = 2
    assert driver.poll_once().outcome == "purged" and gate.snapshot().paused
    gate.resume()
    assert not gate.snapshot().paused


def test_should_clean_retired_and_pending_copies_under_original_deadline():
    owner, cleanup, driver, wrapper, old, _, wall, gate, budget = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper)
    new = control.restore(control.capture())
    token = control.capture()
    gate.resume()
    wall.value = 2
    assert driver.poll_once().outcome == "purged"
    assert old._inbox._experiences == new._inbox._experiences == {} and new._budget is budget
    with pytest.raises(ValueError, match="checkpoint"):
        control.restore(token)
    assert cleanup.admitted_payload_bytes == 24 and owner._declaration_seconds[("e1", "s1")] == 0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_should_fail_closed_on_invalid_original_elapsed_clock(value):
    owner, _, driver, _, runtime, _, wall, gate, _ = setup()
    owner.declare(declaration())
    record(owner)
    wall.value = value
    assert driver.poll_once().outcome == "failed" and gate.snapshot().paused
    with pytest.raises(ValueError, match="clock|retention"):
        runtime.candidate_snapshot()
    assert runtime._inbox._experiences


def test_should_refuse_backwards_or_replaced_original_clock():
    owner, _, driver, _, runtime, _, wall, gate, budget = setup()
    owner.declare(declaration())
    record(owner)
    wall.value = 1
    driver.poll_once()
    wall.value = 0.5
    assert driver.poll_once().outcome == "failed" and gate.snapshot().paused
    with pytest.raises(ValueError):
        runtime.candidate_snapshot()
    budget.clock = Clock()
    with pytest.raises(ValueError):
        runtime.actor.snapshot_state()


def test_should_fail_closed_after_partial_erasure_and_recover_without_reopening_training():
    fail = [True]

    def erase(model):
        if fail[0]:
            raise RuntimeError("injected failure")
        return ReplayPayloadErasure(0, 0, 0)

    owner, _, driver, _, runtime, _, wall, gate, budget = setup(erase=erase)
    owner.declare(declaration())
    record(owner)
    wall.value = 2
    assert driver.poll_once().outcome == "failed" and runtime._stopped and gate.snapshot().paused
    fail[0] = False
    assert driver.finish_cleanup().outcome == "purged"
    assert runtime._inbox._experiences == {} and runtime._stopped and budget.updates_completed == 0
    with pytest.raises(ValueError, match="stopped"):
        runtime.train_ready()


def test_should_purge_on_bounded_exhaustion_and_refuse_driver_or_allowance_renewal():
    owner, cleanup, driver, _, runtime, _, _, gate, _ = setup(polls=1)
    owner.declare(declaration())
    record(owner)
    assert driver.poll_once().outcome == "purged" and driver.snapshot().state == "exhausted"
    assert (
        runtime._inbox._experiences == {}
        and not gate.snapshot().paused
        and cleanup.admitted_payload_bytes == 24
    )
    with pytest.raises(ValueError, match="driver|retention"):
        owner.declare(declaration("s2"))
    with pytest.raises(ValueError, match="original|driver"):
        RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, 20, 2))
    with pytest.raises(ValueError):
        driver.start()


def test_should_stop_with_busy_data_unfinished_then_finish_cleanup_without_native_access():
    owner, _, driver, _, runtime, _, _, gate, _ = setup()
    owner.declare(declaration())
    record(owner)
    with runtime._exclusive():
        assert not driver.stop()
    assert (
        driver.snapshot().pending_cleanup and gate.snapshot().paused and runtime._inbox._experiences
    )
    with pytest.raises(ValueError, match="retention|driver"):
        runtime.candidate_snapshot()
    assert (
        driver.finish_cleanup().outcome == "purged"
        and not gate.snapshot().paused
        and runtime._inbox._experiences == {}
    )


def test_should_run_real_bounded_worker_and_join_after_automatic_purge():
    owner, _, driver, _, runtime, _, wall, gate, _ = setup(polls=100)
    driver.start()
    try:
        owner.declare(declaration())
        record(owner)
        wall.value = 2.0
        driver.wake()
        assert (
            driver.wait_for_purge(1.0)
            and runtime._inbox._experiences == runtime._inbox._labels == {}
        )
        assert not gate.snapshot().paused
    finally:
        assert driver.stop()
    assert not driver.snapshot().alive and driver.snapshot().state == "stopped"


def test_should_use_logical_deadline_even_when_elapsed_clock_does_not_advance():
    owner, _, driver, _, runtime, logical, _, _, _ = setup()
    owner.declare(declaration())
    record(owner)
    logical.advance_to(10)
    assert driver.poll_once().outcome == "purged" and runtime._inbox._experiences == {}


def test_should_revalidate_deadline_after_checkpoint_builder_before_restore_copy():
    from test_actor_shadow import PausableLearner

    owner, _, driver, wrapper, runtime, _, wall, gate, _ = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()

    def build(state):
        model = PausableLearner(state)
        wall.value = 2
        setattr(
            model,
            "restore_state",
            lambda state: (_ for _ in ()).throw(AssertionError("expired restore copied state")),
        )
        return model

    control = controller(wrapper, build_learner=build)
    token = control.capture()
    with pytest.raises(ValueError, match="expired"):
        control.restore(token)
    assert control._attempts == 1 and len(control._models) == 1 and wrapper._runtime is runtime
    assert driver.poll_once().outcome == "purged" and control._pending == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("poll_interval_seconds", 0),
        ("max_run_seconds", float("nan")),
        ("max_polls", True),
        ("max_polls", 0),
        ("join_timeout_seconds", -1),
    ],
)
def test_should_reject_invalid_driver_limits(field, value):
    with pytest.raises(ValueError):
        replace(RetentionDriverLimits(0.01, 20, 2), **{field: value})


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_should_reject_invalid_elapsed_policy(value):
    with pytest.raises(ValueError):
        replace(policy(), max_retention_seconds=value)


def test_should_bound_actual_worker_under_driver_operation_contention():
    owner, _, driver, _, runtime, _, _, gate, _ = setup(polls=100)
    owner.declare(declaration())
    record(owner)
    driver._limits = replace(driver._limits, max_run_seconds=0.05)
    driver.start()
    with driver._operation_gate:
        driver._thread.join(0.3)
        assert (
            not driver.snapshot().alive
            and driver.snapshot().state == "exhausted"
            and gate.snapshot().paused
        )
    assert driver.finish_cleanup().outcome == "purged" and runtime._inbox._experiences == {}


def promotion_fixture():
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.serving_promotion import PromotableActor, ServingPromotionController
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import LogicalClock
    from src.core.resource_sharing import SharingLimits
    from src.core.serving_ports import ServingConfiguration
    from test_serving_promotion import Source, digest
    from test_promotion_guard_evaluation import make_evaluator, policy as promotion_policy

    source = Source([0.5, 0.8])
    wall = Clock()
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"raw": [1]},
    )
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
    wrapper = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))
    cleanup = ManagedDataLifecycle(
        owner,
        policy=replace(policy(), max_retention_seconds=2),
        measure_payload_bytes=graph,
        measure_auxiliary_bytes=graph,
        native_footprint=lambda m: ReplayPayloadErasure(0, 0, 0),
        native_erase=lambda m: ReplayPayloadErasure(0, 0, 0),
    )
    evaluator = make_evaluator()
    control = ServingPromotionController(
        actor,
        policy=promotion_policy(),
        evaluator=evaluator,
        build_learner=evaluator._build,
        state_digest=digest,
    )
    driver = RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, 20, 2))
    return actor, runtime, control, wall, driver


def test_should_refuse_cache_publication_when_prediction_crosses_deadline():
    actor, _, _, wall, driver = promotion_fixture()

    def predict(features):
        wall.value = 2
        return [0.5]

    actor._slot.bundle.learner.predict = predict
    with pytest.raises(ValueError, match="expired"):
        actor.serve([0], now=0)
    assert actor._slot.bundle.cache == {}
    assert driver.poll_once().outcome == "purged"


def test_should_refuse_checkpoint_publication_when_probe_crosses_deadline():
    owner, _, driver, wrapper, _, _, wall, gate, _ = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper)

    def probe(model):
        wall.value = 2
        return "0" * 64

    control._policy = probe
    with pytest.raises(ValueError, match="expired"):
        control.capture()
    assert control._pending == {} and driver.poll_once().outcome == "purged"


def test_should_refuse_serving_commit_when_actor_probe_crosses_deadline():
    from test_serving_promotion import prepare, digest

    actor, runtime, control, wall, driver = promotion_fixture()
    ticket = prepare(control, runtime)

    def probe(state):
        if state == [0.5, 0.8]:
            wall.value = 2
        return digest(state)

    control._digest = probe
    with pytest.raises(ValueError, match="expired"):
        control.commit(runtime, ticket)
    assert actor._slot.generation == 0 and actor._slot.bundle.version == "actor-0"
    assert driver.poll_once().outcome == "purged" and control._pending == {}


def test_should_return_unfinished_when_trusted_cleanup_callback_blocks_join_bound():
    from threading import Event

    entered, release = Event(), Event()

    def erase(model):
        entered.set()
        assert release.wait(1.0)
        return ReplayPayloadErasure(0, 0, 0)

    owner, _, driver, _, runtime, _, wall, gate, _ = setup(erase=erase)
    owner.declare(declaration())
    record(owner)
    driver._limits = replace(driver._limits, join_timeout_seconds=0.02)
    driver.start()
    wall.value = 2
    driver.wake()
    try:
        assert entered.wait(0.5)
        assert not driver.stop() and driver.snapshot().alive and gate.snapshot().paused
    finally:
        release.set()
        driver._thread.join(1.0)
    assert not driver.snapshot().alive and runtime._inbox._experiences == {}
    assert driver.stop()


def test_should_cap_terminal_cleanup_attempts_without_restarting_monitor():
    owner, _, driver, _, runtime, _, _, _, _ = setup(polls=1)
    owner.declare(declaration())
    record(owner)
    with runtime._exclusive():
        assert driver.poll_once().outcome == "busy"
        assert driver.finish_cleanup().outcome == "busy"
        assert driver.finish_cleanup().outcome == "busy"
        assert driver.finish_cleanup().outcome == "failed"
    assert driver.snapshot().cleanup_attempts == 3 and driver.snapshot().polls == 1
    with pytest.raises(ValueError):
        driver.start()


def test_should_not_retain_error_tracebacks_or_raw_exception_messages():
    def erase(model):
        raise RuntimeError("private raw payload in message")

    owner, _, driver, _, _, _, wall, _, _ = setup(erase=erase)
    owner.declare(declaration())
    record(owner)
    wall.value = 2
    result = driver.poll_once()
    assert result.outcome == "failed"
    assert driver.snapshot().error_type == "RuntimeError"
    assert all(not isinstance(value, BaseException) for value in driver.__dict__.values())


def test_should_purge_auxiliary_arrays_without_any_declared_sample():
    import numpy as np
    from src.adapters.numpy_learners import BackpropLearner, make_managed_data_lifecycle
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.serving_promotion import PromotableActor
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.backprop_mlp import BackpropMLP
    from src.core.experience import LogicalClock
    from src.core.serving_ports import ServingConfiguration
    from src.core.resource_sharing import SharingLimits
    from test_candidate_checkpoint import digest

    source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
    wall = Clock()
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"raw": np.array([1.0])},
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), wall),
        actor=actor,
    )
    gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(1, 1))
    cleanup = make_managed_data_lifecycle(owner, policy=replace(policy(), max_retention_seconds=2))
    driver = RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, 20, 2))
    wall.value = 2
    result = driver.poll_once()
    assert (
        result.outcome == "purged"
        and result.report is not None
        and result.report.requested_keys == ()
    )
    assert actor._slot.bundle.metadata == {} and cleanup._auxiliary_started_at is None


@pytest.mark.parametrize("method", ["backprop", "circadian"])
def test_should_automatically_expire_real_handoff_and_stop_after_two_fixed_wakes(method):
    import pickle
    import numpy as np
    from src.adapters.numpy_learners import ManagedNumpyBuilder, make_managed_data_lifecycle
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
    from src.core.resource_sharing import SharingLimits
    from src.core.payload_bytes import PayloadCopyLimits
    from src.app.candidate_checkpoint import CandidateCheckpointController
    from test_candidate_checkpoint import digest
    from test_experience_inbox import make_native_pair
    from test_managed_data_lifecycle import native_without_replay

    _, source = make_native_pair(method)
    wall = Clock()
    logical = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), wall)
    old = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=logical,
        budget=budget,
    )
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(old, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4, allow_synthetic=True))
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
    driver.start()

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

    try:
        enqueue("s1", 3)
        logical.advance_to(3)
        wrapper.train_ready()
        gate.pause()
        control = CandidateCheckpointController(
            wrapper,
            build_learner=ManagedNumpyBuilder(source),
            state_digest=digest,
            policy_digest=lambda m: digest("fixed"),
        )
        new = control.restore(control.capture())
        token = control.capture()
        gate.resume()
        before = {
            id(m): native_without_replay(m)
            for m in (old._candidate, new._candidate, old.actor._learner)
        }
        charged = cleanup.payload_byte_snapshot().charged_bytes
        wall.value = 2
        driver.wake()
        assert driver.wait_for_purge(1.0)
        assert old._inbox._experiences == new._inbox._experiences == {} and control._pending == {}
        for model in (old._candidate, new._candidate, old.actor._learner):
            assert (
                native_without_replay(model) == before[id(model)]
                and cleanup._describe(model).payload_bytes == 0
            )
        assert (
            cleanup.payload_byte_snapshot().charged_bytes == charged
            and cleanup.payload_byte_snapshot().observed_retained_bytes == 0
        )
        with pytest.raises(ValueError, match="checkpoint"):
            control.restore(token)
        enqueue("s2", 5)
        logical.advance_to(5)
        wrapper.train_ready()
        assert (
            budget.updates_completed == gate.snapshot().admitted_updates == 2
            and new._budget is budget
            and old._inbox.capture_cursor().completed_updates == 1
        )
    finally:
        assert driver.stop()
    assert not driver.snapshot().alive and new._inbox._experiences == {}
    assert (
        cleanup.admitted_payload_bytes == 96
        and pickle.dumps(source.snapshot_state()) == source_before
    )
