"""Fresh budgeted zero-update actual lifecycle and driver concurrency controls."""

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
import json
import pytest
import numpy as np

from src.adapters.numpy_learners import BackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_lifecycle_capture import capture_managed_lifecycle
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.retention_expiry import RetentionExpiryDriver
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LogicalClock
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits
from src.core.retention_driver import RetentionDriverLimits
from src.core.promotion_guard import PromotionPolicy
from src.core.serving_ports import ServingConfiguration

LIMITS = LifecycleCaptureLimits(64, 128)


class Clock:
    def __init__(self):
        self.value = 0.0
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.value


def declaration(name="one", subject="person"):
    return LifecycleDeclaration(
        ("episode", name),
        DataProvenance("local", subject, True, False),
        DataConsent(True, True),
        "replay",
    )


@pytest.fixture(scope="module")
def ledger(request):
    counts = dict(
        graphs=0,
        driver_workers=0,
        contenders=0,
        updates=0,
        predictions=0,
        snapshots=0,
        restores=0,
        sleeps=0,
        structural=0,
        live_threads=0,
    )
    yield counts
    assert counts["graphs"] <= 64
    assert counts["driver_workers"] <= 16 and counts["contenders"] <= 16
    assert counts["live_threads"] == 0
    path = Path(request.config.option.basetemp) / "capture-work-ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, indent=2), encoding="utf8")


@pytest.fixture
def graph_factory(monkeypatch, ledger):
    graphs = []
    for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
        monkeypatch.setattr(BackpropLearner, name, lambda *a, **k: pytest.fail("native work"))

    def build(*, copies=True, elapsed=True, driver=True, polls=64, duration=0.5, promotable=False):
        ledger["graphs"] += 1
        assert ledger["graphs"] <= 64
        wall, logical = Clock(), LogicalClock(0)
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock=wall)
        source = BackpropLearner(BackpropMLP(3, 2, seed=23), learning_rate=0.01)
        actor = (
            PromotableActor(
                source,
                version="actor",
                configuration=ServingConfiguration(2, 2),
                feature_digest=lambda x: "unused",
                metadata={"origin": "owned"},
            )
            if promotable
            else None
        )
        runtime = ActorShadowRuntime(
            source,
            actor_version="actor",
            candidate_version="candidate",
            clock=logical,
            budget=budget,
            max_experiences=4,
            actor=actor,
        )
        sharing = ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True)
        owner = ManagedExperienceOwner(
            ResourceSharedRuntime(runtime, sharing), limits=LifecycleLimits(4, 4)
        )
        policy = DataRetentionPolicy(
            4096,
            20,
            PayloadOwnershipLimits(8, 12),
            owned_payload_copies=PayloadCopyLimits(4096) if copies else None,
            max_retention_seconds=2.0 if elapsed else None,
        )

        def measure(value: object) -> int:
            if not isinstance(value, np.ndarray):
                raise ValueError("fixture only admits supported NumPy payloads")
            return value.nbytes

        lifecycle = ManagedDataLifecycle(
            owner,
            policy=policy,
            measure_payload_bytes=measure,
            native_footprint=lambda x: ReplayPayloadErasure(0, 0, 0),
            native_erase=lambda x: ReplayPayloadErasure(0, 0, 0),
            measure_auxiliary_bytes=lambda x: 0,
            measure_checkpoint_bytes=lambda x: 0,
            native_growth_bytes=lambda *a: 0,
            prepare_model_bytes=lambda *a: 0,
            prediction_cache_bytes=lambda *a: 0,
        )
        retention = (
            RetentionExpiryDriver(lifecycle, limits=RetentionDriverLimits(0.01, polls, duration))
            if driver
            else None
        )
        result = owner, lifecycle, retention, runtime, wall, logical, sharing, budget
        graphs.append(result)
        return result

    yield build
    for _, lifecycle, driver, _, _, _, _, budget in graphs:
        if driver is not None and driver._thread is not None:
            driver.stop()
            driver._thread.join(1.0)
            assert not driver._thread.is_alive()
        assert budget.updates_completed == 0
        if lifecycle._copy_budget is not None:
            assert lifecycle._copy_budget._charged <= 4096


def capture(owner):
    return capture_managed_lifecycle(owner, limits=LIMITS)


def record(owner):
    owner.record_experience(
        Experience(
            sample_id="one",
            episode_id="episode",
            observed_at=0,
            model_version="actor",
            features=np.ones((1, 3)),
            role="train",
            permissions=ExperiencePermissions(True, True),
        )
    )


def test_should_capture_complete_actual_history_without_reading_ports_or_renewing_authority(
    graph_factory, monkeypatch
):
    owner, life, driver, runtime, wall, logical, sharing, budget = graph_factory()
    owner.declare(declaration())
    owner.declare(declaration("two", "other"))
    record(owner)
    sharing.pause()
    owner.opt_out("other")
    before = (
        wall.reads,
        life._last_tick,
        life._last_seconds,
        life._copy_budget._charged,
        life._admitted_bytes,
        logical.now(),
        budget.updates_completed,
        driver._created_at,
    )
    for name in (
        "_measure",
        "_footprint",
        "_erase",
        "_auxiliary_bytes",
        "_checkpoint_bytes",
        "_growth_bytes",
        "_prepare_bytes",
        "_prediction_bytes",
    ):
        monkeypatch.setattr(life, name, lambda *a: pytest.fail("capture invoked port"))
    result = capture(owner)
    assert before == (
        wall.reads,
        life._last_tick,
        life._last_seconds,
        life._copy_budget._charged,
        life._admitted_bytes,
        logical.now(),
        budget.updates_completed,
        driver._created_at,
    )
    state = result.metadata
    assert state.owner.catalog == tuple(owner._catalog.values())
    assert state.owner.catalog[0] is not owner._catalog[declaration().key]
    assert state.owner.opted_out == ("other",)
    assert state.owner.revoked_keys == (("episode", "one"), ("episode", "two"))
    assert state.owner.declaration_ticks == tuple(owner._declaration_ticks.items())
    assert state.owner.declaration_seconds == tuple(owner._declaration_seconds.items())
    assert state.copy.charged_bytes == state.lifecycle.admitted_bytes == 24
    assert state.registry.limits is state.lifecycle.policy.holders
    assert state.copy.limits is state.lifecycle.policy.owned_payload_copies
    refs = {item.path: item.value for item in result.authority}
    assert refs["root.owner"] is owner and refs["root.lifecycle"] is life
    assert refs["registry._holders"] is life._registry._holders
    assert refs["driver._token"] is driver._token
    assert refs["driver._state_gate"] is driver._state_gate
    assert refs["lifecycle._budget"] is budget and refs["lifecycle._clock"] is logical
    assert refs["sharing._gate"] is sharing._gate
    assert runtime._inbox._experiences == {}


@pytest.mark.parametrize(
    "copies,elapsed,driver",
    [(False, False, False), (True, False, False), (False, True, False), (True, True, True)],
)
def test_should_capture_only_actual_configured_optional_owners(
    graph_factory, copies, elapsed, driver
):
    owner, life, retention, *_ = graph_factory(copies=copies, elapsed=elapsed, driver=driver)
    state = capture(owner).metadata
    assert (state.copy is not None) == copies and (state.driver is not None) == driver
    assert (state.lifecycle.last_seconds is not None) == elapsed
    assert life._retention_driver is retention


@pytest.mark.parametrize(
    "gate", ["manager", "registry", "actor", "candidate", "time", "copy", "sharing", "driver"]
)
def test_should_refuse_busy_original_lease_without_any_state_change_and_release_prior_leases(
    graph_factory, gate
):
    owner, life, driver, runtime, *_ = graph_factory()
    locks = dict(
        manager=owner._gate,
        registry=life._registry._gate,
        actor=runtime.actor._read_gate,
        candidate=runtime._write_gate,
        time=life._time_gate,
        copy=life._copy_budget._gate,
        sharing=life._sharing._gate,
        driver=driver._operation_gate,
    )
    before = capture(owner)
    with locks[gate]:
        with pytest.raises(PayloadOwnershipBusy):
            capture(owner)
    after = capture(owner)
    assert after.metadata == before.metadata
    assert all(a.value is b.value for a, b in zip(before.authority, after.authority))


def test_should_preserve_dead_retained_weak_entries_without_pruning(graph_factory):
    owner, life, _, runtime, *_ = graph_factory()
    control = CandidateCheckpointController(
        owner._shared,
        build_learner=lambda x: pytest.fail("build"),
        state_digest=lambda x: "unused",
        policy_digest=lambda x: "unused",
    )
    reference = life._registry._holders[3][1]
    del control
    assert reference() is None
    result = capture(owner)
    assert result.metadata.registry.holders[-1].ready is None
    assert 3 in life._registry._holders and result.metadata.registry.total_enrollments == 3
    assert runtime._payload_ready


def test_should_capture_all_actual_supported_holder_kinds_and_original_auxiliary_epoch(
    graph_factory,
):
    owner, life, _, runtime, *_ = graph_factory(promotable=True)

    def build(state):
        pytest.fail("capture invoked native builder")

    control = CandidateCheckpointController(
        owner._shared,
        build_learner=build,
        state_digest=lambda x: "unused",
        policy_digest=lambda x: "unused",
    )
    evaluator = PromotionGuardEvaluator(
        build_learner=build,
        utility=lambda p, t: 0.0,
        actions=lambda p: ("allow",),
        state_valid=lambda s: True,
        prediction_valid=lambda p: True,
        resource_bytes=lambda s: 0,
        state_digest=lambda s: "unused",
        clock=lambda: 0.0,
    )
    promotion = ServingPromotionController(
        runtime.actor,
        policy=PromotionPolicy("utility", "bytes", 0.0, 0.0, 0.0, 1.0, 4096, ("allow",)),
        evaluator=evaluator,
        build_learner=build,
        state_digest=lambda x: "unused",
    )
    state = capture(owner).metadata
    assert tuple(h.kind for h in state.registry.holders) == (
        "actor",
        "candidate",
        "checkpoint",
        "promotion",
    )
    assert state.registry.total_enrollments == 4 and all(h.ready for h in state.registry.holders)
    assert state.lifecycle.auxiliary_started_at == life._auxiliary_started_at == 0.0
    with control._exclusive():
        with pytest.raises(PayloadOwnershipBusy):
            capture(owner)
    with promotion._exclusive():
        with pytest.raises(PayloadOwnershipBusy):
            capture(owner)
    assert capture(owner).metadata == state


@pytest.mark.parametrize("field", ["_lifecycle", "_gate", "_catalog"])
def test_should_refuse_missing_owner_schema_before_reading_incomplete_roots(
    graph_factory, monkeypatch, field
):
    owner, *_ = graph_factory()
    with monkeypatch.context() as patch:
        patch.delattr(owner, field)
        with pytest.raises(ValueError):
            capture(owner)
    assert capture(owner).metadata.registry.total_enrollments == 2


def test_should_refuse_incomplete_live_holder_without_dropping_its_enrollment(
    graph_factory, monkeypatch
):
    owner, life, _, runtime, *_ = graph_factory()
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_payload_ready", False)
        with pytest.raises(ValueError, match="incomplete"):
            capture(owner)
    assert life._registry._total == len(life._registry._holders) == 2
    assert capture(owner).metadata.registry.total_enrollments == 2


@pytest.mark.parametrize(
    "change",
    ["budget_clock", "lineage", "policy", "unknown", "catalog_key", "bounds", "string", "holder"],
)
def test_should_refuse_corrupt_or_changed_original_authority_before_detachment(
    graph_factory, monkeypatch, change
):
    owner, life, _, runtime, *_ = graph_factory()
    owner.declare(declaration())
    if change == "budget_clock":
        monkeypatch.setattr(life._budget, "clock", lambda: 0)
    elif change == "lineage":
        monkeypatch.setattr(runtime, "_payload_lineage", object())
    elif change == "policy":
        monkeypatch.setattr(life._registry, "_limits", replace(life._policy.holders))
    elif change == "unknown":
        monkeypatch.setattr(life, "_future", 1, raising=False)
    elif change == "catalog_key":
        monkeypatch.setitem(owner._catalog, ("episode", "one"), declaration("different"))
    elif change == "string":
        monkeypatch.setitem(
            owner._catalog,
            ("episode", "one"),
            replace(declaration(), provenance=DataProvenance("x" * 129, "person", True, False)),
        )
    elif change == "holder":
        monkeypatch.delitem(life._registry._holders, 1)
    limits = replace(LIMITS, max_records=0) if change == "bounds" else LIMITS
    before = (life._copy_budget._charged, life._admitted_bytes, life._last_tick, life._last_seconds)
    with pytest.raises(ValueError):
        capture_managed_lifecycle(owner, limits=limits)
    assert before == (
        life._copy_budget._charged,
        life._admitted_bytes,
        life._last_tick,
        life._last_seconds,
    )


def test_should_capture_pending_hold_and_terminal_cleanup_without_refunding_copies(graph_factory):
    owner, life, driver, runtime, wall, *_ = graph_factory()
    owner.declare(declaration())
    record(owner)
    wall.value = 2.0
    with runtime._exclusive():
        assert driver.poll_once().outcome == "busy"
    pending = capture(owner)
    assert pending.metadata.driver.held and pending.metadata.driver.pending
    assert {r.path: r.value for r in pending.authority}["sharing._retention_hold"] is driver._token
    charged = life._copy_budget._charged
    assert driver.stop()
    stopped = capture(owner).metadata
    assert (
        stopped.driver.state == "stopped" and stopped.driver.stop_set and stopped.driver.purged_set
    )
    assert stopped.driver.created_at == pending.metadata.driver.created_at
    assert stopped.copy.charged_bytes == charged and stopped.lifecycle.admitted_bytes == 24


def test_should_capture_partial_failure_and_bounded_retry_history_without_reopening_runtime(
    graph_factory, monkeypatch
):
    owner, life, driver, runtime, wall, *_ = graph_factory()
    owner.declare(declaration())
    record(owner)
    original = life._erase
    calls = 0

    def fail_second(model):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("bounded partial cleanup fixture")
        return original(model)

    monkeypatch.setattr(life, "_erase", fail_second)
    wall.value = 2
    assert driver.poll_once().outcome == "failed"
    state = capture(owner).metadata
    assert state.lifecycle.failed and state.lifecycle.retention_fault
    assert state.driver.state == "failed" and state.driver.error_type == "RuntimeError"
    assert state.driver.cleanup_attempts == 1 and state.driver.purges == 0
    assert runtime._stopped
    monkeypatch.setattr(life, "_erase", original)
    assert driver.finish_cleanup().outcome == "purged"
    state = capture(owner).metadata
    assert not state.lifecycle.failed and not state.lifecycle.retention_fault
    assert state.driver.state == "failed" and state.driver.cleanup_attempts == 2
    assert state.copy.charged_bytes == 24 and runtime._stopped


def start_worker(driver, ledger):
    ledger["driver_workers"] += 1
    assert ledger["driver_workers"] <= 16
    driver.start()


def contender(target, ledger):
    ledger["contenders"] += 1
    assert ledger["contenders"] <= 16
    errors = []

    def run():
        try:
            target()
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=run, daemon=False)
    thread.start()
    return thread, errors


def test_should_refuse_busy_state_gate_and_serialize_wake_at_original_capture_lease(
    graph_factory, ledger
):
    owner, _, driver, *_ = graph_factory()
    entered, release = Event(), Event()

    def hold():
        with driver._state_gate:
            entered.set()
            assert release.wait(1)

    thread, errors = contender(hold, ledger)
    try:
        assert entered.wait(1)
        with pytest.raises(PayloadOwnershipBusy):
            capture(owner)
    finally:
        release.set()
        thread.join(1)
    assert not thread.is_alive() and not errors
    driver.wake()
    assert capture(owner).metadata.driver.wake_set


def test_should_stop_during_callback_without_waiting_for_state_lock_and_refuse_reentrant_capture(
    graph_factory, ledger, monkeypatch
):
    owner, life, driver, *_ = graph_factory()
    entered, release, stop_signaled = Event(), Event(), Event()

    def due(self):
        with pytest.raises(PayloadOwnershipBusy):
            capture(owner)
        entered.set()
        assert release.wait(1)
        return (), False

    monkeypatch.setattr(ManagedDataLifecycle, "_due", due)
    start_worker(driver, ledger)
    stop_thread = None
    errors = []
    try:
        assert entered.wait(1)

        def stop():
            # stop's event signals before it waits for the worker's callback.
            driver.stop()
            stop_signaled.set()

        stop_thread, errors = contender(stop, ledger)
        assert driver._stop.wait(1)
        assert driver.snapshot().state == "stopping"
        with pytest.raises(PayloadOwnershipBusy):
            capture(owner)
    finally:
        release.set()
        if stop_thread is not None:
            stop_thread.join(1)
        driver._thread.join(1)
    assert not driver._thread.is_alive() and not stop_thread.is_alive() and not errors
    assert stop_signaled.is_set()
    state = capture(owner).metadata.driver
    assert state.state == "stopped" and not state.thread_alive and state.thread_present


def test_should_exhaust_worker_even_when_original_operation_gate_is_busy(graph_factory, ledger):
    owner, life, driver, *_ = graph_factory(duration=0.05)
    start_worker(driver, ledger)
    with driver._operation_gate:
        driver._thread.join(1)
        assert not driver._thread.is_alive()
    state = capture(owner).metadata
    assert state.driver.state == "exhausted" and state.driver.held and state.driver.pending
    assert life._sharing._retention_hold is driver._token
    assert driver.finish_cleanup().outcome == "purged"
    assert not capture(owner).metadata.driver.pending


def test_should_wake_purge_and_join_actual_worker_with_original_epoch_and_thread(
    graph_factory, ledger
):
    owner, life, driver, _, wall, *_ = graph_factory()
    owner.declare(declaration())
    record(owner)
    epoch = driver._created_at
    start_worker(driver, ledger)
    wall.value = 2
    driver.wake()
    assert driver.wait_for_purge(1)
    assert driver.stop()
    result = capture(owner)
    assert result.metadata.driver.created_at == epoch
    assert result.metadata.driver.state == "stopped" and not result.metadata.driver.thread_alive
    assert result.metadata.copy.charged_bytes == 24
    assert life._owner._revoked_keys == {("episode", "one")}
    assert {r.path: r.value for r in result.authority}["driver._thread"] is driver._thread


def test_should_capture_worker_failure_and_preserve_original_token_and_fault(
    graph_factory, ledger, monkeypatch
):
    owner, life, driver, *_ = graph_factory()

    def fail(self):
        raise RuntimeError("bounded worker error")

    monkeypatch.setattr(ManagedDataLifecycle, "_due", fail)
    start_worker(driver, ledger)
    driver._thread.join(1)
    assert not driver._thread.is_alive()
    state = capture(owner).metadata
    assert state.driver.state == "failed" and state.driver.held and state.driver.pending
    assert state.driver.error_type == "RuntimeError" and state.lifecycle.retention_fault


def test_should_capture_outer_worker_exception_transition_under_state_gate(
    graph_factory, ledger, monkeypatch
):
    owner, life, driver, *_ = graph_factory()

    def fail(self):
        raise RuntimeError("bounded outer worker error")

    monkeypatch.setattr(RetentionExpiryDriver, "poll_once", fail)
    start_worker(driver, ledger)
    driver._thread.join(1)
    assert not driver._thread.is_alive()
    state = capture(owner).metadata
    assert state.driver.state == "failed" and state.driver.polls == 0
    assert state.driver.cleanup_attempts == 0 and state.driver.held and state.driver.pending
    assert state.lifecycle.retention_fault and life._sharing._retention_hold is driver._token


def test_should_exclude_stop_and_wake_mutations_during_metadata_detachment(
    graph_factory, ledger, monkeypatch
):
    import src.app.managed_lifecycle_capture as module

    owner, _, driver, *_ = graph_factory()
    entered, release = Event(), Event()
    wake_done, stop_done = Event(), Event()
    original = module.detach_lifecycle_record
    captures = []

    def detach(state, limits):
        entered.set()
        assert release.wait(1)
        return original(state, limits)

    monkeypatch.setattr(module, "detach_lifecycle_record", detach)
    capturing, capture_errors = contender(lambda: captures.append(capture(owner)), ledger)
    waking = stopping = None
    try:
        assert entered.wait(1)

        def wake():
            driver.wake()
            wake_done.set()

        def stop():
            driver.stop()
            stop_done.set()

        waking, wake_errors = contender(wake, ledger)
        stopping, stop_errors = contender(stop, ledger)
        assert not wake_done.wait(0.01) and not stop_done.wait(0.01)
        with pytest.raises(PayloadOwnershipBusy):
            capture(owner)
    finally:
        release.set()
        for thread in (capturing, waking, stopping):
            if thread is not None:
                thread.join(1)
                assert not thread.is_alive()
    assert not capture_errors and not wake_errors and not stop_errors
    assert wake_done.is_set() and stop_done.is_set()
    assert captures[0].metadata.driver.state == "ready"
    assert not captures[0].metadata.driver.stop_set and not captures[0].metadata.driver.wake_set
    monkeypatch.setattr(module, "detach_lifecycle_record", original)
    assert capture(owner).metadata.driver.state == "stopped"
