"""Inspect the complete fixed prospective role/source/request metadata envelope.

Inputs are the full design, an expected code declaration and all ordered source,
stream and role declarations. Outputs preserve every unresolved actual-proof
obligation. Checksums establish metadata integrity; they do not establish actual
code, data, seeded partitions, release order or independence. No IO, RNG, source,
array, model, metric selection or execution belongs here.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from src.app.prospective_confirmation_design import (
    declare_prospective_replica_streams,
    validate_prospective_design,
)
from src.app.prospective_stream_declarations import validate_prospective_stream_declarations
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_role_requests import (
    ROLE_REQUEST_SCHEMA,
    ProspectiveRoleDeclaration,
    ProspectiveRoleRequest,
    ProspectiveSourceDeclaration,
    RoleRequestInspection,
)
from src.core.seed_stream_screening import (
    EvidenceIdentity,
    confirmation_seed_streams,
    validate_evidence_identity,
)


def _metadata_identity(body: Any) -> EvidenceIdentity:
    raw = (json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return EvidenceIdentity(len(raw), sha256(raw).hexdigest())


def prospective_source_map_identity(
    sources: tuple[ProspectiveSourceDeclaration, ...],
) -> EvidenceIdentity:
    """Encode every metadata row; resealing an invalid declaration cannot validate it."""
    return _metadata_identity([asdict(row) for row in sources])


def prospective_role_request_identity(request: ProspectiveRoleRequest) -> EvidenceIdentity:
    """Bind the entire metadata request except its own identity, never actual sources."""
    body = asdict(request)
    del body["request_identity"]
    return _metadata_identity(body)


def _validate_code_identity(identity: EvidenceIdentity) -> None:
    validate_evidence_identity(identity)
    if identity.byte_count == 0:
        raise ValueError("expected code identity must declare a nonempty whole source")


def _generator_configuration(source: dict[str, Any], phase: str, base_seed: int) -> dict[str, Any]:
    training = source["training"]
    streams = {row.name: row.value for row in confirmation_seed_streams(base_seed)}
    sample_count = training[f"sample_count_phase_{phase}"]
    effective_count = 2 * (sample_count // 2)
    development_count = int((1.0 - training["test_ratio"]) * effective_count)
    retained_count = (
        min(development_count, max(8, int(development_count * training["phase_b_train_fraction"])))
        if phase == "b"
        else development_count
    )
    return {
        "generator": "generate_two_cluster_dataset_with_transform",
        "sample_count": sample_count,
        "noise_scale": training[f"phase_{phase}_noise_scale"],
        "test_ratio": training["test_ratio"],
        "rotation_degrees": training["phase_b_rotation_degrees"] if phase == "b" else 0.0,
        "translation_x": training["phase_b_translation_x"] if phase == "b" else 0.0,
        "translation_y": training["phase_b_translation_y"] if phase == "b" else 0.0,
        "source_seed": streams[f"phase_{phase}_source"],
        "role_split_seed": streams[f"phase_{phase}_roles"],
        "exposure_seed": streams["phase_b_exposure"] if phase == "b" else None,
        "exposure_fraction": training["phase_b_train_fraction"] if phase == "b" else 1.0,
        "original_development_count": development_count,
        "retained_development_count": retained_count,
        "final_count": effective_count - development_count,
        "inner_guard_fraction": source["inner_guard_fraction"],
        "outer_selection_fraction": source["outer_selection_fraction"],
    }


def _source_declarations(
    design: dict[str, Any],
    bindings: tuple[ReplicaSeedBinding, ...],
    code_identity: EvidenceIdentity,
) -> tuple[ProspectiveSourceDeclaration, ...]:
    sources = {
        row["name"]: row["configuration_without_historical_reservations"]["source"]
        for row in design["family_templates"]
    }
    declared: dict[tuple[str, str], ProspectiveSourceDeclaration] = {}
    for binding in bindings:
        for phase in ("a", "b"):
            configuration = _generator_configuration(
                sources[binding.slot.family], phase, binding.base_seed
            )
            value = ProspectiveSourceDeclaration(
                binding.slot.source_group,
                phase,
                binding.base_seed,
                code_identity,
                _metadata_identity(configuration),
            )
            key = (value.source_group, phase)
            # Why this: coupled gating/replay views must share data arguments,
            # although their model/sleep settings differ. A group name is not proof.
            if declared.setdefault(key, value) != value:
                raise ValueError("shared source group has different data-generator declarations")
    return tuple(declared.values())


def declare_prospective_sources(
    design: dict[str, Any],
    bindings: tuple[ReplicaSeedBinding, ...],
    expected_code_identity: EvidenceIdentity,
) -> tuple[ProspectiveSourceDeclaration, ...]:
    """Derive all ordered configuration declarations from validated fixed inputs."""
    _validate_code_identity(expected_code_identity)
    declare_prospective_replica_streams(design, bindings)
    return _source_declarations(design, bindings, expected_code_identity)


def _validate_sources(
    claimed: tuple[ProspectiveSourceDeclaration, ...],
    expected: tuple[ProspectiveSourceDeclaration, ...],
) -> None:
    if type(claimed) is not tuple or len(claimed) != len(expected):
        raise ValueError("sources require the complete ordered immutable phase-source map")
    for position, (actual, required) in enumerate(zip(claimed, expected, strict=True)):
        if (
            type(actual) is not ProspectiveSourceDeclaration
            or type(actual.source_group) is not str
            or type(actual.phase) is not str
            or type(actual.base_seed) is not int
        ):
            raise ValueError(f"source declaration at position {position} has a foreign schema/type")
        validate_evidence_identity(actual.code_identity)
        validate_evidence_identity(actual.generator_configuration_identity)
        if actual != required:
            raise ValueError(f"source declaration at position {position} differs from fixed inputs")


def _validate_role_scope(actual: ProspectiveRoleDeclaration, required: dict[str, Any]) -> None:
    if (
        type(actual) is not ProspectiveRoleDeclaration
        or type(actual.slot) is not ReplicaSlot
        or type(actual.slot.family) is not str
        or type(actual.slot.source_group) is not str
        or type(actual.slot.ordinal) is not int
        or actual.slot != ReplicaSlot(**required["slot"])
        or type(actual.expected_count) is not int
        or actual.expected_count != required["expected_count"]
        or actual.array_identity is not None
        or actual.final_released is not False
    ):
        raise ValueError(
            "role declaration has a foreign scope/type or claims array/release authority"
        )
    for name in ("phase", "role", "source_available_at", "labels_available_at", "allowed_use"):
        if type(getattr(actual, name)) is not str or getattr(actual, name) != required[name]:
            raise ValueError(f"role declaration differs in fixed {name}")


def _sample_positions(role: ProspectiveRoleDeclaration, base_seed: int) -> tuple[int, ...]:
    ids = role.sample_ids
    if type(ids) is not tuple or len(ids) != role.expected_count:
        raise ValueError("role sample IDs require the exact immutable count")
    final = role.role == "final_test"
    prefix = f"phase_{role.phase}/seed_{base_seed}/{'final' if final else 'development'}/"
    positions = []
    for value in ids:
        if type(value) is not str or not value.startswith(prefix):
            raise ValueError("role sample ID has a foreign type/source/phase/namespace")
        suffix = value[len(prefix) :]
        if not suffix.isascii() or not suffix.isdecimal() or len(suffix) > 3:
            raise ValueError("role sample position is not a canonical bounded decimal")
        position = int(suffix)
        if str(position) != suffix or position >= (role.expected_count if final else 120):
            raise ValueError("role sample position differs from the original source universe")
        positions.append(position)
    result = tuple(positions)
    if result != tuple(sorted(set(result))):
        raise ValueError("role sample positions must be distinct and increasing")
    if final and result != tuple(range(role.expected_count)):
        raise ValueError("final sample IDs require the complete canonical source positions")
    return result


def _validate_roles(
    design: dict[str, Any],
    request: ProspectiveRoleRequest,
) -> None:
    required = design["role_requirements"]
    if type(request.roles) is not tuple or len(request.roles) != len(required):
        raise ValueError("roles require the complete ordered immutable role layout")
    seeds = {row.slot: row.base_seed for row in request.bindings}
    partitions: dict[tuple[ReplicaSlot, str], set[int]] = {}
    shared: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for actual, expected in zip(request.roles, required, strict=True):
        _validate_role_scope(actual, expected)
        positions = _sample_positions(actual, seeds[actual.slot])
        shared_key = (actual.slot.source_group, actual.phase, actual.role)
        if shared.setdefault(shared_key, actual.sample_ids) != actual.sample_ids:
            raise ValueError("planned shared source views declare different role sample IDs")
        if actual.role == "final_test":
            continue
        used = partitions.setdefault((actual.slot, actual.phase), set())
        if used.intersection(positions):
            raise ValueError("development role sample positions overlap")
        used.update(positions)
    for (_, phase), partition_positions in partitions.items():
        # Why this: B retains 60 original partition_positions from 120, not dense new IDs.
        if len(partition_positions) != (120 if phase == "a" else 60):
            raise ValueError("development partition does not retain the fixed complete count")


def _validate_envelope(design: dict[str, Any], request: ProspectiveRoleRequest) -> None:
    if (
        type(request) is not ProspectiveRoleRequest
        or type(request.schema_id) is not str
        or request.schema_id != ROLE_REQUEST_SCHEMA
        or request.execution_requested is not False
    ):
        raise ValueError("role request requires the exact metadata schema and no execution request")
    obligations = request.required_actual_bindings
    if (
        type(obligations) is not tuple
        or any(type(name) is not str for name in obligations)
        or obligations != tuple(design["required_actual_request_bindings"])
    ):
        raise ValueError("role request must retain every ordered actual-proof obligation")
    validate_evidence_identity(request.source_map_identity)
    validate_evidence_identity(request.request_identity)


def inspect_prospective_role_request(
    design: dict[str, Any],
    expected_code_identity: EvidenceIdentity,
    request: ProspectiveRoleRequest,
) -> RoleRequestInspection:
    """Check all declarations and identities while leaving actual proof unavailable."""
    _validate_code_identity(expected_code_identity)
    validate_prospective_design(design)
    _validate_envelope(design, request)
    validate_prospective_stream_declarations(
        design, request.bindings, request.design_identity, request.streams
    )
    expected = _source_declarations(design, request.bindings, expected_code_identity)
    _validate_sources(request.sources, expected)
    _validate_roles(design, request)
    if request.source_map_identity != prospective_source_map_identity(request.sources):
        raise ValueError("role request is detached from its whole source metadata map")
    if request.request_identity != prospective_role_request_identity(request):
        raise ValueError("role request is detached from its whole metadata body")
    return RoleRequestInspection(
        request.sources,
        request.roles,
        request.source_map_identity,
        request.request_identity,
        request.required_actual_bindings,
    )
