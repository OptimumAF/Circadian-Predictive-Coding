"""Decode and preflight complete prospective bundles through an inner file-proof port.

Inputs are the fixed design, immutable expected file spec and verifying reader.
Outputs bind all file observations to the full role inspector and retain canonical
UTC/owner/resource/repeat declarations. Actual admission remains unavailable.
No filesystem imports, source/array construction, RNG, models or scoring belong
here; the outer reader must verify whole bytes and recheck after consumption.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import json
import re
from typing import Any

from src.app.continual_confirmation_json import object_fields, same_json
from src.app.prospective_confirmation_design import validate_prospective_design
from src.app.prospective_role_requests import inspect_prospective_role_request
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_request_bundles import (
    BundlePreflight,
    BundleSnapshot,
    CodeFileBinding,
    ProspectiveBundleReader,
    ProspectiveBundleSpec,
    validate_bundle_spec,
)
from src.core.prospective_role_requests import (
    ProspectiveRoleDeclaration,
    ProspectiveRoleRequest,
    ProspectiveSourceDeclaration,
)
from src.core.seed_stream_screening import (
    DerivedSeedStream,
    EvidenceIdentity,
    validate_evidence_identity,
)
from src.core.seed_usage import strict_seed_metadata_json


def _rows(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("prospective bundle ordered JSON fields must be complete arrays")
    return value


def _identity(body: Any) -> EvidenceIdentity:
    result = EvidenceIdentity(**object_fields(body, {"byte_count", "sha256"}, "whole identity"))
    validate_evidence_identity(result)
    return result


def _slot(body: Any) -> ReplicaSlot:
    return ReplicaSlot(**object_fields(body, {field.name for field in fields(ReplicaSlot)}, "slot"))


def _source(body: Any) -> ProspectiveSourceDeclaration:
    row = object_fields(
        body, {field.name for field in fields(ProspectiveSourceDeclaration)}, "source"
    )
    return ProspectiveSourceDeclaration(
        row["source_group"],
        row["phase"],
        row["base_seed"],
        _identity(row["code_identity"]),
        _identity(row["generator_configuration_identity"]),
    )


def _role(body: Any) -> ProspectiveRoleDeclaration:
    row = object_fields(body, {field.name for field in fields(ProspectiveRoleDeclaration)}, "role")
    return ProspectiveRoleDeclaration(
        _slot(row["slot"]),
        row["phase"],
        row["role"],
        row["expected_count"],
        tuple(_rows(row["sample_ids"])),
        row["source_available_at"],
        row["labels_available_at"],
        row["allowed_use"],
        None if row["array_identity"] is None else _identity(row["array_identity"]),
        row["final_released"],
    )


def _role_request(body: Any) -> ProspectiveRoleRequest:
    row = object_fields(
        body, {field.name for field in fields(ProspectiveRoleRequest)}, "role request"
    )
    bindings = [
        object_fields(item, {"slot", "base_seed"}, "binding") for item in _rows(row["bindings"])
    ]
    streams = [
        object_fields(item, {"base_seed", "name", "value"}, "stream")
        for item in _rows(row["streams"])
    ]
    return ProspectiveRoleRequest(
        _identity(row["design_identity"]),
        tuple(ReplicaSeedBinding(_slot(item["slot"]), item["base_seed"]) for item in bindings),
        tuple(DerivedSeedStream(**item) for item in streams),
        tuple(_source(item) for item in _rows(row["sources"])),
        tuple(_role(item) for item in _rows(row["roles"])),
        _identity(row["source_map_identity"]),
        _identity(row["request_identity"]),
        tuple(_rows(row["required_actual_bindings"])),
        row["schema_id"],
        row["execution_requested"],
    )


def decode_prospective_code_manifest(
    body: Any, spec: ProspectiveBundleSpec
) -> tuple[CodeFileBinding, ...]:
    """Validate the entire closed declared membership; physical proof belongs to the port."""
    validate_bundle_spec(spec)
    manifest = object_fields(body, {"schema_id", "files"}, "code manifest")
    same_json(manifest["schema_id"], "p67_closed_prospective_code_files_v1", "code manifest schema")
    rows = [
        object_fields(row, {"path", "identity"}, "code file") for row in _rows(manifest["files"])
    ]
    same_json(
        [row["path"] for row in rows], list(spec.code_paths), "complete expected code membership"
    )
    return tuple(CodeFileBinding(row["path"], _identity(row["identity"])) for row in rows)


def decode_prospective_bundle(
    body: Any,
    spec: ProspectiveBundleSpec,
    code_files: tuple[CodeFileBinding, ...],
    observed_utc: str,
) -> BundleSnapshot:
    """Decode exact schemas; callers must verify whole files before returning a snapshot."""
    row = object_fields(
        body,
        {
            "schema_id",
            "role_request",
            "prospective_utc",
            "owner_id",
            "resource_envelope",
            "independent_repeat_envelope",
        },
        "request bundle",
    )
    same_json(
        row["schema_id"], "p67_complete_prospective_request_bundle_v1", "request bundle schema"
    )
    return BundleSnapshot(
        spec,
        _role_request(row["role_request"]),
        code_files,
        row["prospective_utc"],
        row["owner_id"],
        json.dumps(
            row["resource_envelope"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        json.dumps(
            row["independent_repeat_envelope"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        observed_utc,
    )


def _utc(value: str) -> datetime:
    if type(value) is not str:
        raise ValueError("prospective bundle UTC must be a canonical aware UTC string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("prospective bundle UTC is invalid") from error
    if parsed.tzinfo != timezone.utc or parsed.isoformat() != value:
        raise ValueError("prospective bundle UTC must be canonical and explicitly UTC")
    return parsed


def _validate_snapshot(snapshot: BundleSnapshot, spec: ProspectiveBundleSpec) -> None:
    if type(snapshot) is not BundleSnapshot:
        raise ValueError("prospective bundle port returned a foreign snapshot schema")
    validate_bundle_spec(snapshot.spec)
    if snapshot.spec != spec or type(snapshot.code_files) is not tuple:
        raise ValueError("prospective bundle port detached its snapshot or code membership")
    for row in snapshot.code_files:
        if type(row) is not CodeFileBinding or type(row.path) is not str:
            raise ValueError("prospective bundle port returned foreign code binding types")
        validate_evidence_identity(row.identity)
    if tuple(row.path for row in snapshot.code_files) != spec.code_paths:
        raise ValueError("prospective bundle port omitted or reordered code membership")
    if _utc(snapshot.prospective_utc) > _utc(snapshot.observed_utc):
        raise ValueError("prospective request UTC is later than the file observation")
    if (
        type(snapshot.owner_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", snapshot.owner_id) is None
    ):
        raise ValueError("prospective bundle requires a canonical owner declaration")
    if (
        type(snapshot.resource_envelope_json) is not str
        or type(snapshot.repeat_envelope_json) is not str
    ):
        raise ValueError("prospective bundle port must retain immutable envelope JSON")


def _validate_envelopes(design: dict[str, Any], snapshot: BundleSnapshot) -> None:
    same_json(
        strict_seed_metadata_json(snapshot.resource_envelope_json),
        design["resource_envelope"],
        "complete unchanged resource envelope",
    )
    request = snapshot.role_request
    repeat = {
        "required": True,
        "design_identity": {
            "byte_count": request.design_identity.byte_count,
            "sha256": request.design_identity.sha256,
        },
        "source_map_identity": {
            "byte_count": request.source_map_identity.byte_count,
            "sha256": request.source_map_identity.sha256,
        },
        "role_request_identity": {
            "byte_count": request.request_identity.byte_count,
            "sha256": request.request_identity.sha256,
        },
        "caps_identical": True,
        "success_failure_and_every_attempt_charged": True,
        "actual_independent_repeat_verified": False,
    }
    same_json(
        strict_seed_metadata_json(snapshot.repeat_envelope_json),
        repeat,
        "complete same-request repeat declaration",
    )


def preflight_prospective_request_bundle(
    design: dict[str, Any],
    spec: ProspectiveBundleSpec,
    reader: ProspectiveBundleReader,
) -> BundlePreflight:
    """Bind complete physical metadata without granting any actual admission authority."""
    validate_prospective_design(design)
    validate_bundle_spec(spec)
    snapshot = reader.read_bundle(spec)
    _validate_snapshot(snapshot, spec)
    role = inspect_prospective_role_request(
        design, spec.code_manifest_identity, snapshot.role_request
    )
    if snapshot.role_request.source_map_identity != spec.source_map_identity:
        raise ValueError("role request is detached from the physical whole source metadata map")
    _validate_envelopes(design, snapshot)
    # Why this: code/request/map drift after app validation must reject before
    # returning a result. Neither an owner string nor this check is a live lease.
    reader.recheck_bundle(snapshot)
    return BundlePreflight(snapshot, role, role.required_actual_bindings)
