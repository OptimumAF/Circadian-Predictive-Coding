"""Typed pre-payload relationships only; fixtures are not durable authority."""

from dataclasses import replace

import pytest

from src.core.recovery_admission import (
    RecoveryFence,
    RecoveryLimits,
    RecoveryManifest,
    RecoveryMetadata,
    RecoveryUsage,
    validate_recovery_admission,
)


def fixture():
    manifest = RecoveryManifest(*[f"{n:064x}" for n in range(1, 10)])
    limits = RecoveryLimits(10, 1000, 100, 10, 4, 1024)
    usage = RecoveryUsage(3, 3, 40, 3, 2, 512)
    metadata = RecoveryMetadata(
        1,
        "session",
        "boot-epoch",
        "owner-a",
        7,
        4,
        100,
        400,
        12,
        manifest,
        limits,
        usage,
        False,
        False,
    )
    fence = RecoveryFence(metadata, "owner-b", 5, "boot-epoch", 600, 640, True, True)
    return metadata, fence


def test_should_count_downtime_and_preserve_original_remaining_work():
    metadata, fence = fixture()
    admission = validate_recovery_admission(metadata, fence)
    assert admission.elapsed_ns == 500
    assert admission.peak_rss_bytes == 640
    assert admission.remaining_updates == 7
    assert admission.remaining_copy_bytes == 60
    assert admission.owner_epoch == 5 and admission.owner_id == "owner-b"
    assert metadata == fence.expected  # No mutation of saved counters/limits.


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("clock_epoch", "other-boot"),
        ("owner_id", "other-owner"),
        ("sequence", 6),
        ("owner_epoch", 3),
        ("started_ns", 200),
        ("observed_ns", 300),
        ("event_tick", 11),
    ],
)
def test_should_refuse_metadata_rollback_or_changed_original_authority(field, value):
    metadata, fence = fixture()
    with pytest.raises(ValueError, match="authoritative"):
        validate_recovery_admission(replace(metadata, **{field: value}), fence)


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "policy_sha256",
        "payload_sha256",
        "native_sha256",
        "inbox_sha256",
        "consolidation_sha256",
        "lifecycle_sha256",
        "actor_sha256",
        "sharing_sha256",
    ],
)
def test_should_refuse_changed_complete_component_or_source_binding(field):
    metadata, fence = fixture()
    changed = replace(metadata.manifest, **{field: "f" * 64})
    with pytest.raises(ValueError, match="authoritative"):
        validate_recovery_admission(replace(metadata, manifest=changed), fence)


@pytest.mark.parametrize(
    "field",
    [
        "max_updates",
        "max_elapsed_ns",
        "max_copy_bytes",
        "max_grants",
        "max_checkpoint_attempts",
        "max_rss_bytes",
    ],
)
def test_should_refuse_renewed_or_changed_original_limit(field):
    metadata, fence = fixture()
    changed = replace(metadata.limits, **{field: getattr(metadata.limits, field) + 1})
    with pytest.raises(ValueError, match="authoritative"):
        validate_recovery_admission(replace(metadata, limits=changed), fence)


@pytest.mark.parametrize(
    "field",
    [
        "updates_admitted",
        "updates_completed",
        "copied_bytes",
        "grants",
        "checkpoint_attempts",
        "peak_rss_bytes",
    ],
)
def test_should_refuse_refunded_or_rolled_back_spent_resources(field):
    metadata, fence = fixture()
    changed = replace(metadata.usage, **{field: getattr(metadata.usage, field) - 1})
    with pytest.raises(ValueError, match="authoritative"):
        validate_recovery_admission(replace(metadata, usage=changed), fence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_id", "owner-a"),
        ("owner_epoch", 4),
        ("owner_epoch", 6),
        ("clock_epoch", "after-reboot"),
        ("now_ns", 399),
        ("previous_owner_ended", False),
        ("lease_live", False),
    ],
)
def test_should_refuse_unsupported_clock_or_ownership_fence(field, value):
    metadata, fence = fixture()
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, replace(fence, **{field: value}))


@pytest.mark.parametrize(
    "changes",
    [
        {"now_ns": 1100},
        {"now_ns": 1101},
        {"rss_bytes": 1025},
    ],
)
def test_should_refuse_expired_time_or_current_resource_cap(changes):
    metadata, fence = fixture()
    with pytest.raises(ValueError, match="exhausted"):
        validate_recovery_admission(metadata, replace(fence, **changes))


@pytest.mark.parametrize(
    "field,value",
    [
        ("stopped", True),
        ("uncertain_work", True),
    ],
)
def test_should_refuse_stopped_or_uncertain_work_even_if_authority_agrees(field, value):
    metadata, fence = fixture()
    metadata = replace(metadata, **{field: value})
    with pytest.raises(ValueError, match="stopped|uncertain"):
        validate_recovery_admission(metadata, replace(fence, expected=metadata))


@pytest.mark.parametrize(
    "field,value",
    [
        ("updates_admitted", 4),
        ("updates_completed", 4),
        ("updates_admitted", 11),
        ("copied_bytes", 101),
        ("grants", 11),
        ("checkpoint_attempts", 5),
        ("peak_rss_bytes", 1025),
    ],
)
def test_should_refuse_inflight_or_over_cap_work_even_if_authority_agrees(field, value):
    metadata, fence = fixture()
    metadata = replace(metadata, usage=replace(metadata.usage, **{field: value}))
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, replace(fence, expected=metadata))


def test_should_keep_exhausted_work_and_copy_quotas_without_renewal():
    metadata, fence = fixture()
    usage = replace(metadata.usage, updates_admitted=10, updates_completed=10, copied_bytes=100)
    metadata = replace(metadata, usage=usage)
    admission = validate_recovery_admission(metadata, replace(fence, expected=metadata))
    assert admission.remaining_updates == 0 and admission.remaining_copy_bytes == 0


def test_should_preserve_previous_absolute_rss_peak_and_accept_exact_cap():
    metadata, fence = fixture()
    metadata = replace(metadata, usage=replace(metadata.usage, peak_rss_bytes=1024))
    admission = validate_recovery_admission(metadata, replace(fence, expected=metadata))
    assert admission.peak_rss_bytes == 1024


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("sequence", -1),
        ("observed_ns", 2**63),
        ("session_id", " session"),
        ("session_id", "s" * 129),
        ("stopped", 0),
        ("uncertain_work", None),
    ],
)
def test_should_revalidate_manually_corrupted_saved_metadata(field, value):
    metadata, fence = fixture()
    object.__setattr__(metadata, field, value)
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)


@pytest.mark.parametrize(
    "part,field,value",
    [
        ("manifest", "lifecycle_sha256", "ABC"),
        ("limits", "max_rss_bytes", 0),
        ("limits", "max_elapsed_ns", False),
        ("usage", "copied_bytes", -1),
        ("usage", "peak_rss_bytes", 0),
    ],
)
def test_should_revalidate_nested_corruption(part, field, value):
    metadata, fence = fixture()
    object.__setattr__(getattr(metadata, part), field, value)
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)


def test_should_refuse_untyped_inputs_without_accessing_opaque_state():
    class Opaque:
        def __getattr__(self, name):
            raise AssertionError("opaque payload access")

        def __deepcopy__(self, memo):
            raise AssertionError("opaque payload copied")

    metadata, fence = fixture()
    with pytest.raises(ValueError):
        validate_recovery_admission(Opaque(), fence)
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, Opaque())
    object.__setattr__(metadata.manifest, "native_sha256", Opaque())
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)


def test_should_refuse_corrupt_independent_authority_before_comparison():
    metadata, fence = fixture()
    independent, _ = fixture()
    object.__setattr__(independent.usage, "grants", False)
    object.__setattr__(fence, "expected", independent)
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_epoch", True),
        ("now_ns", -1),
        ("rss_bytes", 0),
        ("previous_owner_ended", 1),
        ("lease_live", None),
        ("expected", None),
    ],
)
def test_should_revalidate_manually_corrupted_fence(field, value):
    metadata, fence = fixture()
    object.__setattr__(fence, field, value)
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)


def test_should_refuse_changed_registry_without_skipping_required_component_fields():
    metadata, fence = fixture()
    object.__setattr__(metadata.manifest, "__dataclass_fields__", {})
    object.__setattr__(metadata.manifest, "lifecycle_sha256", None)
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)


def test_should_refuse_record_subclasses_before_their_hooks():
    class Unsupported(RecoveryUsage):
        def __post_init__(self):
            raise AssertionError("unsupported record hook called")

    metadata, fence = fixture()
    # Bypass subclass construction deliberately to probe the boundary.
    object.__setattr__(metadata, "usage", object.__new__(Unsupported))
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)


def test_should_refuse_original_observation_before_start():
    metadata, fence = fixture()
    object.__setattr__(metadata, "observed_ns", 99)
    with pytest.raises(ValueError):
        validate_recovery_admission(metadata, fence)
