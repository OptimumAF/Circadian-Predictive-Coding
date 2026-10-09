"""Complete lifecycle contract without executing capture,native work or a worker."""

from dataclasses import fields, replace
import json
from typing import Any

import pytest

from src.app.managed_lifecycle_schema import (
    SOURCE_FIELDS,
    detach_lifecycle_record,
    require_lifecycle_source_schema,
)
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.managed_lifecycle_state import (
    AUTHORITY_PATHS,
    DRIVER_REFERENCES,
    AuthorityReference,
    CopyBudgetState,
    LifecycleAccountingState,
    LifecycleCaptureLimits,
    LifecycleMetadata,
    ManagedLifecycleCapture,
    ManagedOwnerState,
    OwnershipEnrollment,
    OwnershipRegistryState,
    RetentionDriverRecord,
)
from src.core.managed_lifecycle_validation import validate_lifecycle_capture
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.retention_driver import RetentionDriverLimits

LIMITS = LifecycleCaptureLimits(64, 128)


class PoisonReference:
    def __deepcopy__(self, memo):
        pytest.fail("live authority was deepcopied")

    def __eq__(self, other):
        pytest.fail("live authority comparison invoked foreign equality")


class PoisonPort(PoisonReference):
    def __call__(self, *args, **kwargs):
        pytest.fail("live authority port was invoked")


def sample(*, elapsed=True, copies=True, driver=True, state="failed", held=True, thread=True):
    if driver and not elapsed:
        raise AssertionError("fixture driver requires elapsed policy")
    holders = PayloadOwnershipLimits(8, 16)
    copy_limits = PayloadCopyLimits(2048) if copies else None
    policy = DataRetentionPolicy(
        2048,
        10,
        holders,
        owned_payload_copies=copy_limits,
        max_retention_seconds=10.0 if elapsed else None,
    )
    declarations = tuple(
        LifecycleDeclaration(
            ("episode", key),
            DataProvenance("source", "subject-" + key, True, False),
            DataConsent(True, True),
            "replay",
        )
        for key in ("b", "a")
    )
    ticks = tuple((item.key, index + 2) for index, item in enumerate(declarations))
    seconds = tuple((key, float(tick)) for key, tick in ticks) if elapsed else ()
    metadata = LifecycleMetadata(
        1,
        ManagedOwnerState(
            LifecycleLimits(4, 4), declarations, ("subject-b",), (("episode", "b"),), ticks, seconds
        ),
        LifecycleAccountingState(
            policy, 64, 8, True, 8.0 if elapsed else None, 3.0 if elapsed else None, True
        ),
        OwnershipRegistryState(
            holders,
            4,
            (
                OwnershipEnrollment(1, "actor", True),
                OwnershipEnrollment(2, "candidate", False),
                OwnershipEnrollment(4, "checkpoint", None),
            ),
        ),
        CopyBudgetState(copy_limits, 128) if copy_limits is not None else None,
        RetentionDriverRecord(
            RetentionDriverLimits(1.0, 4, 8.0),
            state,
            2,
            1,
            3,
            11.0,
            held,
            held,
            "RuntimeError" if state == "failed" else None,
            True,
            False,
            True,
            thread,
            thread,
        )
        if driver
        else None,
    )
    refs: dict[str, object] = {path: PoisonReference() for path in AUTHORITY_PATHS}
    for left, right in (
        ("owner._lifecycle", "root.lifecycle"),
        ("registry._lifecycle", "root.lifecycle"),
        ("lifecycle._owner", "root.owner"),
        ("lifecycle._registry", "root.registry"),
        ("owner._shared", "lifecycle._shared"),
    ):
        refs[left] = refs[right]
    refs["owner._issuing"] = None
    refs["lifecycle._progress"] = refs["lifecycle._sampler"] = None
    for name in (
        "_measure",
        "_footprint",
        "_erase",
        "_wall_clock",
        "_resource",
        "_auxiliary_bytes",
        "_checkpoint_bytes",
        "_growth_bytes",
        "_prepare_bytes",
        "_prediction_bytes",
    ):
        refs["lifecycle." + name] = PoisonPort()
    if not copies:
        refs["root.copy"] = refs["copy._gate"] = None
    refs["lifecycle._copy_budget"] = refs["root.copy"]
    if not driver:
        refs["root.driver"] = None
        for name in DRIVER_REFERENCES:
            refs["driver." + name] = None
    else:
        refs["driver._lifecycle"] = refs["root.lifecycle"]
        if not thread:
            refs["driver._thread"] = None
    refs["lifecycle._retention_driver"] = refs["root.driver"]
    refs["sharing._retention_hold"] = refs["driver._token"] if driver and held else None
    return ManagedLifecycleCapture(
        metadata, tuple(AuthorityReference(path, refs[path]) for path in sorted(refs))
    )


@pytest.fixture(scope="session", autouse=True)
def zero_native_ledger(tmp_path_factory):
    path = tmp_path_factory.mktemp("native-work") / "native-work.json"
    path.write_text(
        json.dumps(
            dict(
                updates=0, predictions=0, snapshots=0, restores=0, sleeps=0, structural=0, workers=0
            )
        )
    )


@pytest.mark.parametrize(
    "elapsed,copies,driver",
    [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True) if a or not c],
)
def test_should_preserve_complete_metadata_original_policy_aliases_and_every_live_reference(
    elapsed, copies, driver
):
    original = sample(elapsed=elapsed, copies=copies, driver=driver)
    detached = detach_lifecycle_record(original, LIMITS)
    assert detached.metadata == original.metadata and detached.metadata is not original.metadata
    assert detached.authority is original.authority
    assert len(detached.authority) == 48
    assert detached.metadata.owner.catalog[0].key == ("episode", "b")
    assert detached.metadata.owner.catalog[0] is not original.metadata.owner.catalog[0]
    assert detached.metadata.registry.limits is detached.metadata.lifecycle.policy.holders
    assert detached.metadata.registry.limits is not original.metadata.registry.limits
    if detached.metadata.copy is not None:
        assert (
            detached.metadata.copy.limits is detached.metadata.lifecycle.policy.owned_payload_copies
        )
    assert detached.metadata.registry.holders[-1].ready is None  # actual dead retained weak entry


@pytest.mark.parametrize(
    "state", ["ready", "running", "stopping", "stopped", "exhausted", "failed"]
)
@pytest.mark.parametrize("held,thread", [(a, b) for a in (False, True) for b in (False, True)])
def test_should_retain_every_driver_epoch_hold_event_thread_and_terminal_observation(
    state, held, thread
):
    original = sample(state=state, held=held, thread=thread)
    assert detach_lifecycle_record(original, LIMITS).metadata == original.metadata


def record_at(capture, path):
    value: Any = capture
    for item in path.split("."):
        value = value[int(item)] if item.isdigit() else getattr(value, item)
    return value


RECORDS = [
    ("", ManagedLifecycleCapture),
    ("metadata", LifecycleMetadata),
    ("metadata.owner", ManagedOwnerState),
    ("metadata.owner.limits", LifecycleLimits),
    ("metadata.owner.catalog.0", LifecycleDeclaration),
    ("metadata.owner.catalog.0.provenance", DataProvenance),
    ("metadata.owner.catalog.0.consent", DataConsent),
    ("metadata.lifecycle", LifecycleAccountingState),
    ("metadata.lifecycle.policy", DataRetentionPolicy),
    ("metadata.registry", OwnershipRegistryState),
    ("metadata.registry.limits", PayloadOwnershipLimits),
    ("metadata.registry.holders.0", OwnershipEnrollment),
    ("metadata.copy", CopyBudgetState),
    ("metadata.copy.limits", PayloadCopyLimits),
    ("metadata.driver", RetentionDriverRecord),
    ("metadata.driver.limits", RetentionDriverLimits),
    ("authority.0", AuthorityReference),
]
ALL_FIELDS = [(path, field.name) for path, kind in RECORDS for field in fields(kind)]


@pytest.mark.parametrize("path,name", ALL_FIELDS)
def test_should_refuse_every_missing_complete_metadata_or_authority_field(path, name):
    capture = sample()
    record = capture if not path else record_at(capture, path)
    object.__delattr__(record, name)
    with pytest.raises(ValueError):
        detach_lifecycle_record(capture, LIMITS)


@pytest.mark.parametrize("path,kind", RECORDS)
def test_should_refuse_unknown_nested_metadata_or_authority_fields(path, kind):
    capture = sample()
    record = capture if not path else record_at(capture, path)
    object.__setattr__(record, "future", 1)
    with pytest.raises(ValueError):
        detach_lifecycle_record(capture, LIMITS)


BAD_VALUES = [
    ("metadata", "format_version", True),
    ("metadata", "format_version", 2),
    ("metadata.owner", "catalog", []),
    ("metadata.owner", "opted_out", ("foreign",)),
    ("metadata.owner", "revoked_keys", (("episode", "foreign"),)),
    ("metadata.owner", "declaration_ticks", ()),
    ("metadata.owner", "declaration_seconds", ()),
    ("metadata.lifecycle", "admitted_bytes", 2049),
    ("metadata.lifecycle", "last_tick", 1),
    ("metadata.lifecycle", "last_seconds", 1.0),
    ("metadata.lifecycle", "auxiliary_started_at", 9.0),
    ("metadata.lifecycle", "failed", 1),
    ("metadata.lifecycle", "retention_fault", 0),
    ("metadata.lifecycle", "last_seconds", float("nan")),
    ("metadata.registry", "total_enrollments", 3),
    ("metadata.registry.holders.0", "enrollment", 0),
    ("metadata.registry.holders.0", "kind", "foreign"),
    ("metadata.registry.holders.0", "ready", 1),
    ("metadata.copy", "charged_bytes", 63),
    ("metadata.copy", "charged_bytes", 2049),
    ("metadata.driver", "state", "foreign"),
    ("metadata.driver", "polls", 5),
    ("metadata.driver", "purges", 4),
    ("metadata.driver", "cleanup_attempts", 7),
    ("metadata.driver", "created_at", -1.0),
    ("metadata.driver", "pending", False),
    ("metadata.driver", "thread_present", False),
    ("metadata.driver", "stop_set", 1),
    ("metadata.driver", "error_type", object()),
]


@pytest.mark.parametrize("path,name,value", BAD_VALUES)
def test_should_refuse_inconsistent_original_epoch_identity_count_or_type(path, name, value):
    capture = sample()
    object.__setattr__(record_at(capture, path), name, value)
    with pytest.raises(ValueError):
        detach_lifecycle_record(capture, LIMITS)


@pytest.mark.parametrize("path", sorted(AUTHORITY_PATHS))
def test_should_refuse_each_missing_original_live_reference(path):
    capture = sample()
    incomplete = replace(
        capture, authority=tuple(item for item in capture.authority if item.path != path)
    )
    with pytest.raises(ValueError):
        detach_lifecycle_record(incomplete, LIMITS)


@pytest.mark.parametrize(
    "path",
    [
        "owner._lifecycle",
        "registry._lifecycle",
        "lifecycle._owner",
        "lifecycle._registry",
        "lifecycle._copy_budget",
        "lifecycle._retention_driver",
        "owner._shared",
        "driver._lifecycle",
        "sharing._retention_hold",
    ],
)
def test_should_refuse_changed_original_authority_aliases(path):
    capture = sample()
    altered = replace(
        capture,
        authority=tuple(
            AuthorityReference(item.path, PoisonReference()) if item.path == path else item
            for item in capture.authority
        ),
    )
    with pytest.raises(ValueError):
        detach_lifecycle_record(altered, LIMITS)


def test_should_refuse_equal_but_distinct_original_holder_or_copy_policies():
    for path in ("metadata.registry", "metadata.copy"):
        capture = sample()
        record = record_at(capture, path)
        object.__setattr__(record, "limits", replace(record.limits))
        with pytest.raises(ValueError):
            detach_lifecycle_record(capture, LIMITS)


def test_should_enforce_independent_aggregate_record_and_exact_utf8_bounds():
    capture = sample()
    owner = capture.metadata.owner
    size = sum(
        len(items)
        for items in (
            owner.catalog,
            owner.opted_out,
            owner.revoked_keys,
            owner.declaration_ticks,
            owner.declaration_seconds,
            capture.metadata.registry.holders,
        )
    )
    assert (
        detach_lifecycle_record(capture, replace(LIMITS, max_records=size)).metadata
        == capture.metadata
    )
    with pytest.raises(ValueError):
        detach_lifecycle_record(capture, replace(LIMITS, max_records=size - 1))
    object.__setattr__(owner.catalog[0].provenance, "source_id", "é" * 64)
    validate_lifecycle_capture(capture, LIMITS)
    object.__setattr__(owner.catalog[0].provenance, "source_id", "é" * 65)
    with pytest.raises(ValueError):
        detach_lifecycle_record(capture, LIMITS)


@pytest.fixture(scope="module")
def native_sources():
    from src.adapters.numpy_learners import BackpropLearner
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.managed_data_lifecycle import ManagedDataLifecycle
    from src.app.managed_experience import ManagedExperienceOwner
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.retention_expiry import RetentionExpiryDriver
    from src.app.toy_execution_budget import ToyExecutionBudget, ToyBudgetSession
    from src.core.backprop_mlp import BackpropMLP
    from src.core.data_erasure import ReplayPayloadErasure
    from src.core.experience import LogicalClock
    from src.core.resource_sharing import SharingLimits

    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 8.0)
    learner = BackpropLearner(BackpropMLP(3, 2, seed=23), learning_rate=0.03)
    runtime = ActorShadowRuntime(
        learner,
        actor_version="actor",
        candidate_version="learner",
        clock=LogicalClock(3),
        budget=budget,
    )
    sharing = ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True)
    owner = ManagedExperienceOwner(
        ResourceSharedRuntime(runtime, sharing), limits=LifecycleLimits(4, 4)
    )
    lifecycle = ManagedDataLifecycle(
        owner,
        policy=sample().metadata.lifecycle.policy,
        measure_payload_bytes=lambda value: 0,
        native_footprint=lambda value: ReplayPayloadErasure(0, 0, 0),
        native_erase=lambda value: pytest.fail("native erase"),
        measure_auxiliary_bytes=lambda value: 0,
        measure_checkpoint_bytes=lambda value: 0,
        native_growth_bytes=lambda *a: 0,
        prepare_model_bytes=lambda *a: 0,
        prediction_cache_bytes=lambda *a: 0,
    )
    driver = RetentionExpiryDriver(lifecycle, limits=RetentionDriverLimits(1.0, 4, 8.0))
    with pytest.MonkeyPatch.context() as patch:
        for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
            patch.setattr(BackpropLearner, name, lambda *a, **k: pytest.fail("native operation"))
        yield owner, lifecycle, lifecycle._registry, lifecycle._copy_budget, driver
    assert budget.updates_completed == 0 and driver._thread is None


def source_guard(sources):
    owner, lifecycle, registry, copy, driver = sources
    require_lifecycle_source_schema(owner, lifecycle, registry, copy=copy, driver=driver)


def test_should_cover_exact_actual_all70_source_fields_without_invoking_any_port(native_sources):
    source_guard(native_sources)
    assert sum(len(vars(value)) for value in native_sources) == 70
    assert sum(len(names) for names in SOURCE_FIELDS.values()) == 70


SOURCE_CASES = [(kind, name) for kind, names in SOURCE_FIELDS.items() for name in sorted(names)]


@pytest.mark.parametrize("kind,name", SOURCE_CASES)
def test_should_refuse_each_missing_current_source_field(native_sources, monkeypatch, kind, name):
    source = next(value for value in native_sources if type(value) is kind)
    monkeypatch.delattr(source, name)
    with pytest.raises(ValueError):
        source_guard(native_sources)


@pytest.mark.parametrize("kind", list(SOURCE_FIELDS))
def test_should_refuse_unknown_future_native_source_fields(native_sources, monkeypatch, kind):
    source = next(value for value in native_sources if type(value) is kind)
    monkeypatch.setattr(source, "_future", True, raising=False)
    with pytest.raises(ValueError):
        source_guard(native_sources)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_records", True),
        ("max_records", -1),
        ("max_records", 2**63),
        ("max_identifier_bytes", 0),
        ("max_identifier_bytes", True),
        ("max_identifier_bytes", 2**63),
    ],
)
def test_should_refuse_invalid_independently_supplied_capture_limits(field, value):
    with pytest.raises(ValueError):
        detach_lifecycle_record(sample(), replace(LIMITS, **{field: value}))


def test_should_refuse_unknown_capture_limit_fields():
    limits = replace(LIMITS)
    object.__setattr__(limits, "future", 1)
    with pytest.raises(ValueError):
        detach_lifecycle_record(sample(), limits)


@pytest.mark.parametrize(
    "path",
    [
        "owner._gate",
        "lifecycle._actor",
        "lifecycle._clock",
        "lifecycle._budget",
        "lifecycle._lineage",
        "lifecycle._time_gate",
        "registry._gate",
        "registry._holders",
        "sharing._gate",
        "copy._gate",
        "driver._token",
        "driver._operation_gate",
        "driver._state_gate",
        "driver._stop",
        "driver._wake",
        "driver._purged",
        "lifecycle._measure",
        "lifecycle._footprint",
        "lifecycle._erase",
        "lifecycle._resource",
        "lifecycle._wall_clock",
        "lifecycle._auxiliary_bytes",
        "lifecycle._checkpoint_bytes",
        "lifecycle._growth_bytes",
        "lifecycle._prepare_bytes",
        "lifecycle._prediction_bytes",
    ],
)
def test_should_refuse_missing_required_lock_token_event_or_callable_port(path):
    capture = sample()
    changed = replace(
        capture,
        authority=tuple(
            AuthorityReference(item.path, None) if item.path == path else item
            for item in capture.authority
        ),
    )
    with pytest.raises(ValueError):
        detach_lifecycle_record(changed, LIMITS)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_reference",
        "unknown_reference",
        "inflight_issue",
        "duplicate_catalog",
        "duplicate_ticks",
        "duplicate_revocation",
        "duplicate_holder",
        "unconfigured_driver",
        "copy_omitted",
    ],
)
def test_should_refuse_duplicate_unknown_or_silently_omitted_authority_and_history(mutation):
    capture = sample()
    if mutation == "duplicate_reference":
        capture = replace(capture, authority=capture.authority[:-1] + (capture.authority[0],))
    elif mutation == "unknown_reference":
        capture = replace(
            capture,
            authority=capture.authority[:-1] + (AuthorityReference("future", PoisonReference()),),
        )
    elif mutation == "inflight_issue":
        capture = replace(
            capture,
            authority=tuple(
                AuthorityReference(item.path, PoisonReference())
                if item.path == "owner._issuing"
                else item
                for item in capture.authority
            ),
        )
    elif mutation == "duplicate_catalog":
        object.__setattr__(capture.metadata.owner, "catalog", capture.metadata.owner.catalog * 2)
    elif mutation == "duplicate_ticks":
        object.__setattr__(
            capture.metadata.owner,
            "declaration_ticks",
            capture.metadata.owner.declaration_ticks * 2,
        )
    elif mutation == "duplicate_revocation":
        object.__setattr__(
            capture.metadata.owner, "revoked_keys", capture.metadata.owner.revoked_keys * 2
        )
    elif mutation == "duplicate_holder":
        object.__setattr__(
            capture.metadata.registry, "holders", capture.metadata.registry.holders * 2
        )
    elif mutation == "unconfigured_driver":
        object.__setattr__(capture.metadata, "driver", None)
    else:
        object.__setattr__(capture.metadata, "copy", None)
    with pytest.raises(ValueError):
        detach_lifecycle_record(capture, LIMITS)


def test_should_refuse_omitting_actual_driver_or_copy_source_roots(native_sources):
    owner, lifecycle, registry, copy, driver = native_sources
    for kwargs in ({"driver": driver}, {"copy": copy}):
        with pytest.raises(ValueError):
            require_lifecycle_source_schema(owner, lifecycle, registry, **kwargs)
