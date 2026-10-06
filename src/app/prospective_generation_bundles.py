"""Decode and preflight whole V2 generation files through an inner proof port.

Inputs are the full fixed design, immutable expected files and verifying reader.
Outputs bind every physical observation to g's complete recipes and preserve all
actual admission obligations. Canonical UTC/owner/resource/repeat declarations
cannot prove actual chronology, source arrival, lease or execution. No filesystem,
array/source/RNG/model/final/scoring work belongs in this application module.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import json
import re
from typing import Any

from src.app.continual_confirmation_json import object_fields, same_json
from src.app.prospective_confirmation_design import validate_prospective_design
from src.app.prospective_generation_requests import inspect_prospective_generation_request
from src.core.prospective_generation_bundles import (
    GenerationBundlePreflight,
    GenerationBundleSnapshot,
    ProspectiveGenerationBundleReader,
)
from src.core.prospective_generation_requests import (
    ProspectiveGenerationRequest,
    RoleAssignmentRecipe,
)
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_request_bundles import (
    CodeFileBinding,
    ProspectiveBundleSpec,
    validate_bundle_spec,
)
from src.core.prospective_role_requests import ProspectiveSourceDeclaration
from src.core.seed_stream_screening import (
    DerivedSeedStream,
    EvidenceIdentity,
    validate_evidence_identity,
)
from src.core.seed_usage import strict_seed_metadata_json


def _rows(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("generation bundle ordered JSON fields require complete arrays")
    return value


def _identity(body: Any) -> EvidenceIdentity:
    value = EvidenceIdentity(**object_fields(body, {"byte_count", "sha256"}, "whole identity"))
    validate_evidence_identity(value)
    return value


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


def _recipe(body: Any) -> RoleAssignmentRecipe:
    row = dict(
        object_fields(body, {field.name for field in fields(RoleAssignmentRecipe)}, "role recipe")
    )
    row["slot"] = _slot(row["slot"])
    if row["declared_sample_ids"] is not None:
        row["declared_sample_ids"] = tuple(_rows(row["declared_sample_ids"]))
    if row["array_identity"] is not None:
        row["array_identity"] = _identity(row["array_identity"])
    return RoleAssignmentRecipe(**row)


def _generation_request(body: Any) -> ProspectiveGenerationRequest:
    row = object_fields(
        body, {field.name for field in fields(ProspectiveGenerationRequest)}, "generation request"
    )
    bindings = [
        object_fields(item, {"slot", "base_seed"}, "binding") for item in _rows(row["bindings"])
    ]
    streams = [
        object_fields(item, {"base_seed", "name", "value"}, "stream")
        for item in _rows(row["streams"])
    ]
    return ProspectiveGenerationRequest(
        _identity(row["design_identity"]),
        tuple(ReplicaSeedBinding(_slot(item["slot"]), item["base_seed"]) for item in bindings),
        tuple(DerivedSeedStream(**item) for item in streams),
        tuple(_source(item) for item in _rows(row["sources"])),
        tuple(_recipe(item) for item in _rows(row["role_recipes"])),
        _identity(row["source_map_identity"]),
        _identity(row["request_identity"]),
        tuple(_rows(row["required_actual_bindings"])),
        row["schema_id"],
        row["execution_requested"],
    )


def decode_prospective_generation_bundle(
    body: Any,
    spec: ProspectiveBundleSpec,
    code_files: tuple[CodeFileBinding, ...],
    observed_utc: str,
) -> GenerationBundleSnapshot:
    """Decode exact V2 schemas; the reader must bind all whole physical bytes."""
    row = object_fields(
        body,
        {
            "schema_id",
            "generation_request",
            "prospective_utc",
            "owner_id",
            "resource_envelope",
            "independent_repeat_envelope",
        },
        "generation bundle",
    )
    same_json(
        row["schema_id"],
        "p67_complete_prospective_generation_bundle_v2",
        "generation bundle schema",
    )
    return GenerationBundleSnapshot(
        spec,
        _generation_request(row["generation_request"]),
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
        raise ValueError("generation bundle UTC must be a canonical aware UTC string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("generation bundle UTC is invalid") from error
    if parsed.tzinfo != timezone.utc or parsed.isoformat() != value:
        raise ValueError("generation bundle UTC must be canonical and explicitly UTC")
    return parsed


def _validate_snapshot(snapshot: GenerationBundleSnapshot, spec: ProspectiveBundleSpec) -> None:
    if type(snapshot) is not GenerationBundleSnapshot:
        raise ValueError("generation bundle port returned a foreign V2 snapshot schema")
    validate_bundle_spec(snapshot.spec)
    if snapshot.spec != spec or type(snapshot.code_files) is not tuple:
        raise ValueError("generation bundle port detached its snapshot or code membership")
    for row in snapshot.code_files:
        if type(row) is not CodeFileBinding or type(row.path) is not str:
            raise ValueError("generation bundle port returned foreign code binding types")
        validate_evidence_identity(row.identity)
    if tuple(row.path for row in snapshot.code_files) != spec.code_paths:
        raise ValueError("generation bundle port omitted or reordered code membership")
    if _utc(snapshot.prospective_utc) > _utc(snapshot.observed_utc):
        raise ValueError("prospective generation UTC is later than file observation")
    if (
        type(snapshot.owner_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", snapshot.owner_id) is None
    ):
        raise ValueError("generation bundle requires a canonical owner declaration")
    if (
        type(snapshot.resource_envelope_json) is not str
        or type(snapshot.repeat_envelope_json) is not str
    ):
        raise ValueError("generation bundle port must retain immutable envelope JSON")


def _validate_envelopes(design: dict[str, Any], snapshot: GenerationBundleSnapshot) -> None:
    same_json(
        strict_seed_metadata_json(snapshot.resource_envelope_json),
        design["resource_envelope"],
        "complete unchanged resource envelope",
    )
    request = snapshot.generation_request
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
        "generation_request_identity": {
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
        "complete same-generation-request repeat declaration",
    )


def preflight_prospective_generation_bundle(
    design: dict[str, Any], spec: ProspectiveBundleSpec, reader: ProspectiveGenerationBundleReader
) -> GenerationBundlePreflight:
    """Bind the full physical V2 metadata scope with actual admission still denied."""
    validate_prospective_design(design)
    validate_bundle_spec(spec)
    snapshot = reader.read_bundle(spec)
    _validate_snapshot(snapshot, spec)
    generation = inspect_prospective_generation_request(
        design, spec.code_manifest_identity, snapshot.generation_request
    )
    if snapshot.generation_request.source_map_identity != spec.source_map_identity:
        raise ValueError("generation request is detached from its whole physical source map")
    _validate_envelopes(design, snapshot)
    # Why this: late byte/path drift must fail before any metadata preflight can
    # be consumed. The owner declaration remains separate from a live lease.
    reader.recheck_bundle(snapshot)
    return GenerationBundlePreflight(snapshot, generation, generation.required_actual_bindings)
