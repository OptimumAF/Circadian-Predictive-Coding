"""Fresh zero-update actual captures and pure synthetic complete histories."""

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import pytest

import src.app.managed_record_capture as module
from src.adapters.numpy_learners import BackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_lifecycle_capture import capture_managed_lifecycle
from src.app.managed_record_capture import capture_managed_records, INBOX_FIELDS, RUNTIME_FIELDS
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.retention_expiry import RetentionExpiryDriver
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import AppliedConsolidation
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
from src.core.learner_ports import TrainingDiagnostic
from src.core.managed_lifecycle_state import AuthorityReference, LifecycleCaptureLimits
from src.core.managed_record_state import (
    ManagedRecordCapture,
    ManagedRecordMetadata,
    RuntimeRecordObservation,
    RECORD_AUTHORITY_PATHS,
    validate_managed_record_capture,
)
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits
from src.core.retention_driver import RetentionDriverLimits
from src.core.serving_ports import ServingConfiguration
from test_managed_lifecycle_state import sample  # Pure metadata factory, no inherited fixtures.

LIMITS = LifecycleCaptureLimits(64, 128)


class Port:
    def __init__(self, value):
        self.value, self.calls, self.blocked = value, 0, False

    def __call__(self, *args):
        if self.blocked:
            pytest.fail("capture invoked original port")
        self.calls += 1
        return self.value(*args) if callable(self.value) else self.value


def make_owner(variant):
    wall = Port(8.0)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock=wall)
    learner = BackpropLearner(BackpropMLP(3, 2, seed=23), learning_rate=0.01)
    actor = None
    if variant == 1:
        actor = PromotableActor(
            learner,
            version="actor",
            configuration=ServingConfiguration(2, 2),
            feature_digest=lambda x: "unused",
            metadata={"origin": "owned"},
        )
    runtime = ActorShadowRuntime(
        learner,
        actor_version="actor",
        candidate_version="candidate",
        clock=LogicalClock(0),
        budget=budget,
        max_experiences=4,
        max_consolidations=4,
        actor=actor,
    )
    owner = ManagedExperienceOwner(
        ResourceSharedRuntime(
            runtime, ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True)
        ),
        limits=LifecycleLimits(4, 4),
    )
    ports = [
        Port(lambda value: int(value.nbytes)),
        Port(ReplayPayloadErasure(0, 0, 0)),
        *[Port(0) for _ in range(5)],
    ]
    life = ManagedDataLifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096,
            20,
            PayloadOwnershipLimits(8, 12),
            owned_payload_copies=None if variant == 2 else PayloadCopyLimits(4096),
            max_retention_seconds=None if variant == 2 else 2.0,
        ),
        measure_payload_bytes=ports[0],
        native_footprint=ports[1],
        native_erase=lambda x: pytest.fail("native erase"),
        measure_auxiliary_bytes=ports[2],
        measure_checkpoint_bytes=ports[3],
        native_growth_bytes=ports[4],
        prepare_model_bytes=ports[5],
        prediction_cache_bytes=ports[6],
    )
    driver = (
        None
        if variant == 2
        else RetentionExpiryDriver(life, limits=RetentionDriverLimits(0.01, 64, 0.5))
    )
    provenance, consent = DataProvenance("local", "person", True, False), DataConsent(True, True)
    for name in ("one", "two"):
        owner.declare(LifecycleDeclaration(("episode", name), provenance, consent, "replay"))
    owner.record_experience(
        Experience(
            "one",
            "episode",
            0,
            "actor",
            np.ones((1, 3)),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    revision = runtime._revision
    with pytest.raises(ValueError, match="arbitrary consolidation"):
        runtime.consolidate("refused", lambda *a: pytest.fail("transform"))
    assert runtime._revision == revision + 1
    assert runtime._attempted_ids == set() and runtime._consolidations == []
    for port in [wall, *ports]:
        port.blocked = True
    return owner, life, runtime, budget, driver, [wall, *ports]


@pytest.fixture(scope="module")
def actual(request):
    with pytest.MonkeyPatch.context() as patch:
        for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
            patch.setattr(BackpropLearner, name, lambda *a, **k: pytest.fail("native operation"))
        owners = [make_owner(index) for index in range(3)]
        calls = [[port.calls for port in row[-1]] for row in owners]
        patch.setattr(LogicalClock, "now", lambda *a: pytest.fail("clock read"))
        yield owners
        for row, before in zip(owners, calls):
            _, life, runtime, budget, driver, ports = row
            assert budget.updates_completed == 0 and runtime._revision == 5
            assert life._admitted_bytes == 24
            assert driver is None or (driver._thread is None and driver._polls == 0)
            assert life._copy_budget is None or life._copy_budget._charged == 24
            assert [port.calls for port in ports] == before
    folder = Path(request.config.option.basetemp)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "paired-native-work.json").write_text(
        json.dumps(
            dict(
                graphs=3,
                managed_refusals=3,
                updates=0,
                predictions=0,
                snapshots=0,
                restores=0,
                sleeps=0,
                structural=0,
                workers=0,
                threads=0,
            ),
            indent=2,
        ),
        encoding="utf8",
    )


@pytest.mark.parametrize("index", range(3))
def test_should_capture_complete_actual_pair_without_renewing_original_authority(actual, index):
    owner, life, runtime, budget, driver, _ = actual[index]
    state = capture_managed_records(owner, limits=LIMITS)
    validate_managed_record_capture(state, LIMITS)
    assert state.metadata.consolidation == runtime.capture_consolidation_cursor()
    assert state.metadata.lifecycle == capture_managed_lifecycle(owner, limits=LIMITS).metadata
    obs = state.metadata.owner
    assert obs.revision == 5 and obs.budget_updates == obs.inbox_completed_updates == 0
    assert obs.inbox_last_tick == 0 and obs.enrollment == 2
    refs = {item.path: item.value for item in state.authority}
    assert set(refs) == RECORD_AUTHORITY_PATHS and len(refs) == 59
    assert refs["runtime.root"] is runtime and refs["runtime.budget"] is budget
    assert refs["inbox.clock"] is life._clock and refs["root.driver"] is driver
    assert (
        state.metadata.lifecycle.owner.catalog[0].consent
        is state.metadata.lifecycle.owner.catalog[1].consent
    )
    assert (
        state.metadata.lifecycle.owner.catalog[0].consent
        is not owner._catalog[("episode", "one")].consent
    )
    if driver is not None:
        assert isinstance(driver, RetentionExpiryDriver)
        assert state.metadata.lifecycle.driver is not None
        assert state.metadata.lifecycle.driver.created_at == driver._created_at
    if life._copy_budget is not None:
        assert state.metadata.lifecycle.copy is not None
        assert state.metadata.lifecycle.copy.charged_bytes == 24
        assert (
            state.metadata.lifecycle.copy.limits
            is state.metadata.lifecycle.lifecycle.policy.owned_payload_copies
        )


def original_gates(row):
    owner, life, runtime, _, driver, _ = row
    gates = {
        "manager": owner._gate,
        "registry": life._registry._gate,
        "candidate": runtime._write_gate,
        "actor": runtime.actor._read_gate,
        "time": life._time_gate,
        "sharing": life._sharing._gate,
    }
    if life._copy_budget is not None:
        gates["copy"] = life._copy_budget._gate
    if driver is not None:
        gates["driver"] = driver._operation_gate
        # State is an RLock, which intentionally permits same-thread reentry.
    return gates


def test_should_hold_original_common_interval_through_read_validation_and_single_copy(
    actual, monkeypatch
):
    row = actual[0]
    owner, _, runtime, _, _, _ = row
    read, copy, validate = (
        ActorShadowRuntime._read_consolidation_cursor,
        module.deepcopy,
        module.validate_managed_record_capture,
    )
    seen = []

    def check_interval(where):
        for gate in original_gates(row).values():
            assert not gate.acquire(blocking=False)
        with pytest.raises((ValueError, PayloadOwnershipBusy)):
            capture_managed_records(owner, limits=LIMITS)
        with pytest.raises(ValueError):
            runtime.capture_consolidation_cursor()
        seen.append(where)

    def read_inside(value):
        check_interval("read")
        return read(value)

    def copy_inside(value):
        assert type(value) is ManagedRecordMetadata
        check_interval("copy")
        return copy(value)

    def validate_inside(value, limits):
        check_interval("validate")
        return validate(value, limits)

    monkeypatch.setattr(ActorShadowRuntime, "_read_consolidation_cursor", read_inside)
    monkeypatch.setattr(module, "deepcopy", copy_inside)
    monkeypatch.setattr(module, "validate_managed_record_capture", validate_inside)
    capture_managed_records(owner, limits=LIMITS)
    assert seen == ["read", "validate", "copy", "validate"]
    for gate in original_gates(row).values():
        assert gate.acquire(blocking=False)
        gate.release()


@pytest.mark.parametrize(
    "name", ("driver", "manager", "registry", "actor", "candidate", "time", "copy", "sharing")
)
def test_should_refuse_busy_original_gate_without_copy_and_release_prior_leases(
    actual, monkeypatch, name
):
    row = actual[0]
    before = capture_managed_records(row[0], limits=LIMITS)
    monkeypatch.setattr(module, "deepcopy", lambda *a: pytest.fail("copy during contention"))
    gates = original_gates(row)
    assert gates[name].acquire(blocking=False)
    try:
        with pytest.raises((ValueError, PayloadOwnershipBusy)):
            capture_managed_records(row[0], limits=LIMITS)
    finally:
        gates[name].release()
    monkeypatch.undo()
    after = capture_managed_records(row[0], limits=LIMITS)
    assert after.metadata == before.metadata
    assert all(a.value is b.value for a, b in zip(before.authority, after.authority))


SOURCE_CASES = [(0, name) for name in sorted(RUNTIME_FIELDS)] + [
    (1, name) for name in sorted(INBOX_FIELDS)
]


@pytest.mark.parametrize("which,name", SOURCE_CASES)
def test_should_refuse_every_missing_runtime_or_inbox_source_field_before_read_or_copy(
    actual, monkeypatch, which, name
):
    owner, _, runtime, _, _, _ = actual[0]
    target = runtime if which == 0 else runtime._inbox
    monkeypatch.delattr(target, name)
    monkeypatch.setattr(module, "deepcopy", lambda *a: pytest.fail("copy"))
    with pytest.raises(ValueError, match="schema"):
        capture_managed_records(owner, limits=LIMITS)


@pytest.mark.parametrize("which", (0, 1))
def test_should_refuse_unknown_original_source_fields(actual, monkeypatch, which):
    runtime = actual[0][2]
    monkeypatch.setattr(runtime if which == 0 else runtime._inbox, "_future", True, raising=False)
    with pytest.raises(ValueError, match="schema"):
        capture_managed_records(actual[0][0], limits=LIMITS)


@pytest.mark.parametrize(
    "which", ("learner", "budget", "version", "draining", "lineage", "clock", "actor")
)
def test_should_refuse_mixed_live_owner_relationships_without_copy(actual, monkeypatch, which):
    owner, _, runtime, _, _, _ = actual[0]
    other = actual[1][2]
    changes = {
        "learner": (runtime._inbox, "_learner", other._candidate),
        "budget": (runtime._inbox, "_budget", other._budget),
        "version": (runtime._inbox, "_learner_version", "other"),
        "draining": (runtime._inbox, "_draining", True),
        "lineage": (runtime, "_payload_lineage", other._payload_lineage),
        "clock": (runtime._inbox, "_clock", other._inbox._clock),
        "actor": (runtime, "_actor", other.actor),
    }
    monkeypatch.setattr(*changes[which])
    monkeypatch.setattr(module, "deepcopy", lambda *a: pytest.fail("copy"))
    with pytest.raises(ValueError):
        capture_managed_records(owner, limits=LIMITS)


@pytest.mark.parametrize("bound", (0, 7, 8))
def test_should_bound_aggregate_original_histories_before_cursor_read_or_copy(
    actual, monkeypatch, bound
):
    monkeypatch.setattr(
        ActorShadowRuntime, "_read_consolidation_cursor", lambda *a: pytest.fail("cursor read")
    )
    monkeypatch.setattr(module, "deepcopy", lambda *a: pytest.fail("copy"))
    with pytest.raises(ValueError):
        capture_managed_records(actual[0][0], limits=replace(LIMITS, max_records=bound))


@pytest.mark.parametrize(
    "name,value", (("_base_actor_version", "é" * 129), ("_candidate_version", "x" * 129))
)
def test_should_bound_source_strings_before_cursor_read(actual, monkeypatch, name, value):
    monkeypatch.setattr(actual[0][2], name, value)
    monkeypatch.setattr(
        ActorShadowRuntime, "_read_consolidation_cursor", lambda *a: pytest.fail("read")
    )
    with pytest.raises(ValueError):
        capture_managed_records(actual[0][0], limits=LIMITS)


def synthetic():
    life = sample()
    refs = {item.path: item.value for item in life.authority}
    candidate, inbox, root, gate = object(), object(), object(), object()
    additions = {
        "runtime.root": root,
        "runtime.actor": refs["lifecycle._actor"],
        "runtime.candidate": candidate,
        "runtime.inbox": inbox,
        "runtime.budget": refs["lifecycle._budget"],
        "runtime.lineage": refs["lifecycle._lineage"],
        "runtime.gate": gate,
        "inbox.learner": candidate,
        "inbox.clock": refs["lifecycle._clock"],
        "inbox.budget": refs["lifecycle._budget"],
        "shared.runtime": root,
    }
    diagnostic = TrainingDiagnostic("native_loss", -0.0)
    from src.core.consolidation_cursor import ConsolidationCursor

    cursor = ConsolidationCursor(
        1,
        "base",
        "candidate",
        4,
        ("a", "b", "c"),
        (
            AppliedConsolidation("a", "base", "candidate", 1, diagnostic),
            AppliedConsolidation("c", "base", "candidate", 3, diagnostic),
        ),
        True,
        True,
        11,
        False,
    )
    observation = RuntimeRecordObservation(
        "base", "promoted", "candidate", 11, 4, True, True, False, 7, 5, 12, True, 2
    )
    return ManagedRecordCapture(
        ManagedRecordMetadata(1, observation, cursor, life.metadata),
        tuple(
            sorted(
                life.authority + tuple(AuthorityReference(k, v) for k, v in additions.items()),
                key=lambda x: x.path,
            )
        ),
    )


def test_should_preserve_complete_synthetic_closed_histories_and_original_reference_graph():
    state = synthetic()
    validate_managed_record_capture(state, LIMITS)
    detached = ManagedRecordCapture(deepcopy(state.metadata), state.authority)
    validate_managed_record_capture(detached, LIMITS)
    assert detached.metadata == state.metadata
    assert detached.authority is state.authority
    assert (
        detached.metadata.consolidation.consolidations[0].diagnostic
        is detached.metadata.consolidation.consolidations[1].diagnostic
    )
    assert (
        detached.metadata.owner.serving_actor_version != detached.metadata.owner.base_actor_version
    )
    assert detached.metadata.owner.inbox_last_tick > detached.metadata.lifecycle.lifecycle.last_tick


@pytest.mark.parametrize("path", sorted(RECORD_AUTHORITY_PATHS))
def test_should_require_every_original_paired_reference_slot(path):
    state = synthetic()
    broken = replace(state, authority=tuple(item for item in state.authority if item.path != path))
    with pytest.raises(ValueError):
        validate_managed_record_capture(broken, LIMITS)


@pytest.mark.parametrize(
    "path",
    (
        "runtime.actor",
        "runtime.budget",
        "runtime.lineage",
        "inbox.clock",
        "inbox.budget",
        "inbox.learner",
        "shared.runtime",
    ),
)
def test_should_refuse_mixed_original_reference_aliases_without_foreign_equality(path):
    state = synthetic()
    broken = replace(
        state,
        authority=tuple(
            AuthorityReference(item.path, object()) if item.path == path else item
            for item in state.authority
        ),
    )
    with pytest.raises(ValueError):
        validate_managed_record_capture(broken, LIMITS)


@pytest.mark.parametrize(
    "name,value",
    (
        ("base_actor_version", "other"),
        ("learner_version", "other"),
        ("revision", 12),
        ("consolidation_limit", 5),
        ("stopped", False),
        ("retired", False),
        ("payload_ready", True),
        ("budget_updates", 4),
        ("enrollment", 1),
        ("inbox_last_tick", -2),
        ("inbox_last_tick", True),
        ("inbox_completed_updates", True),
        ("revision", 2**63),
        ("stopped", 1),
    ),
)
def test_should_refuse_mixed_revision_version_budget_flag_enrollment_and_tick_records(name, value):
    state = synthetic()
    broken = replace(
        state,
        metadata=replace(state.metadata, owner=replace(state.metadata.owner, **{name: value})),
    )
    with pytest.raises(ValueError):
        validate_managed_record_capture(broken, LIMITS)


NATIVE_RECORD_CASES = [
    (kind, item.name)
    for kind in (ManagedRecordCapture, ManagedRecordMetadata, RuntimeRecordObservation)
    for item in fields(kind)
]


@pytest.mark.parametrize("kind,name", NATIVE_RECORD_CASES)
def test_should_refuse_every_missing_paired_native_field(kind, name):
    state = synthetic()
    targets = {
        ManagedRecordCapture: state,
        ManagedRecordMetadata: state.metadata,
        RuntimeRecordObservation: state.metadata.owner,
    }
    object.__delattr__(targets[kind], name)
    with pytest.raises(ValueError):
        validate_managed_record_capture(state, LIMITS)


@pytest.mark.parametrize(
    "kind", (ManagedRecordCapture, ManagedRecordMetadata, RuntimeRecordObservation)
)
def test_should_refuse_unknown_paired_native_field(kind):
    state = synthetic()
    targets = {
        ManagedRecordCapture: state,
        ManagedRecordMetadata: state.metadata,
        RuntimeRecordObservation: state.metadata.owner,
    }
    object.__setattr__(targets[kind], "future", True)
    with pytest.raises(ValueError):
        validate_managed_record_capture(state, LIMITS)


def test_should_require_stop_for_original_cleanup_failure_or_inbox_uncertainty():
    state = synthetic()
    broken = replace(
        state,
        metadata=replace(
            state.metadata,
            owner=replace(state.metadata.owner, stopped=False),
            consolidation=replace(state.metadata.consolidation, stopped=False),
        ),
    )
    with pytest.raises(ValueError, match="uncertain-stop"):
        validate_managed_record_capture(broken, LIMITS)


def test_should_refuse_joint_capacity_and_utf8_before_detachment(actual, monkeypatch):
    monkeypatch.setattr(module, "deepcopy", lambda *a: pytest.fail("copy"))
    with pytest.raises(ValueError):
        capture_managed_records(actual[0][0], limits=replace(LIMITS, max_identifier_bytes=2))


@pytest.mark.parametrize("value", (-0.0, -2.5, 2**100, 0, 0.125))
def test_should_preserve_native_diagnostic_numeric_type_value_and_shared_identity(value):
    state = synthetic()
    diagnostic = TrainingDiagnostic("native_loss", value)
    cursor = replace(
        state.metadata.consolidation,
        consolidations=tuple(
            replace(receipt, diagnostic=diagnostic)
            for receipt in state.metadata.consolidation.consolidations
        ),
    )
    state = replace(state, metadata=replace(state.metadata, consolidation=cursor))
    validate_managed_record_capture(state, LIMITS)
    detached = deepcopy(state.metadata)
    a, b = detached.consolidation.consolidations
    assert a.diagnostic is b.diagnostic and type(a.diagnostic.value) is type(value)
    assert json.dumps(a.diagnostic.value) == json.dumps(value)


@pytest.mark.parametrize("which", ("unknown", "duplicate", "empty_runtime"))
def test_should_refuse_unknown_duplicate_or_empty_runtime_reference(which):
    state = synthetic()
    refs = list(state.authority)
    if which == "unknown":
        refs[0] = AuthorityReference("unknown", object())
    elif which == "duplicate":
        refs[0] = refs[1]
    else:
        refs = [
            AuthorityReference(item.path, None) if item.path == "runtime.inbox" else item
            for item in refs
        ]
    with pytest.raises(ValueError):
        validate_managed_record_capture(replace(state, authority=tuple(refs)), LIMITS)


@pytest.mark.parametrize(
    "name",
    (
        "format_version",
        "actor_version",
        "learner_version",
        "consolidation_limit",
        "attempted_ids",
        "consolidations",
        "stopped",
        "retired",
        "revision",
        "payload_ready",
    ),
)
def test_should_refuse_every_missing_consolidation_component_field(name):
    state = synthetic()
    object.__delattr__(state.metadata.consolidation, name)
    with pytest.raises(ValueError):
        validate_managed_record_capture(state, LIMITS)


@pytest.mark.parametrize(
    "name", ("event_id", "actor_version", "learner_version", "attempt_number", "diagnostic")
)
def test_should_refuse_every_missing_consolidation_receipt_field(name):
    state = synthetic()
    object.__delattr__(state.metadata.consolidation.consolidations[0], name)
    with pytest.raises(ValueError):
        validate_managed_record_capture(state, LIMITS)


@pytest.mark.parametrize("name", ("definition", "value"))
def test_should_refuse_every_missing_native_diagnostic_field(name):
    state = synthetic()
    object.__delattr__(state.metadata.consolidation.consolidations[0].diagnostic, name)
    with pytest.raises(ValueError):
        validate_managed_record_capture(state, LIMITS)


def test_should_allow_original_inbox_initial_sentinel_and_independent_observed_tick_lag():
    state = synthetic()
    state = replace(
        state,
        metadata=replace(state.metadata, owner=replace(state.metadata.owner, inbox_last_tick=-1)),
    )
    validate_managed_record_capture(state, LIMITS)


def test_should_bound_complete_synthetic_joint_histories():
    state = synthetic()
    with pytest.raises(ValueError, match="aggregate"):
        validate_managed_record_capture(state, replace(LIMITS, max_records=15))


@pytest.mark.parametrize("failure,inbox", ((True, False), (False, True)))
def test_should_independently_require_runtime_stop_for_each_uncertainty(failure, inbox):
    state = synthetic()
    life = replace(
        state.metadata.lifecycle,
        lifecycle=replace(state.metadata.lifecycle.lifecycle, failed=failure),
    )
    broken = replace(
        state,
        metadata=replace(
            state.metadata,
            lifecycle=life,
            owner=replace(state.metadata.owner, stopped=False, inbox_stopped=inbox),
            consolidation=replace(state.metadata.consolidation, stopped=False),
        ),
    )
    with pytest.raises(ValueError, match="uncertain-stop"):
        validate_managed_record_capture(broken, LIMITS)
