"""Freeze full assignment recipes, then inspect separate concrete role claims.

Inputs are the fixed whole design, code/seed/stream declarations and optionally
all concrete role ID declarations. Outputs bind metadata only. The live splitter
needs arrived class labels; this module neither predicts them nor runs its RNG.
Actual source/arrival/seeded-assignment/runtime/owner/prior/resource/repeat proofs
remain required. No IO, array, source, model, scoring or execution belongs here.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass, replace
from hashlib import sha256
import json
from typing import Any

from src.app.prospective_confirmation_design import prospective_design_identity
from src.app.prospective_role_requests import (
    declare_prospective_sources,
    inspect_prospective_role_request,
    prospective_role_request_identity,
    prospective_source_map_identity,
)
from src.app.prospective_stream_declarations import validate_prospective_stream_declarations
from src.core.prospective_generation_requests import (
    ArrivedRoleDeclarationInspection,
    GenerationRequestInspection,
    ProspectiveGenerationRequest,
    RoleAssignmentRecipe,
)
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_role_requests import ProspectiveRoleDeclaration, ProspectiveRoleRequest
from src.core.seed_stream_screening import (
    DerivedSeedStream,
    EvidenceIdentity,
    confirmation_seed_streams,
)


def prospective_generation_request_identity(
    request: ProspectiveGenerationRequest,
) -> EvidenceIdentity:
    """Bind every recipe/declaration except the self pin; encoding is not proof."""
    body = asdict(request)
    del body["request_identity"]
    raw = (json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return EvidenceIdentity(len(raw), sha256(raw).hexdigest())


def _development_geometry(source: dict[str, Any], phase: str) -> tuple[int, int]:
    training = source["training"]
    effective_count = 2 * (training[f"sample_count_phase_{phase}"] // 2)
    original_count = int((1.0 - training["test_ratio"]) * effective_count)
    retained_count = (
        min(original_count, max(8, int(original_count * training["phase_b_train_fraction"])))
        if phase == "b"
        else original_count
    )
    return original_count, retained_count


def _role_recipe(
    required: dict[str, Any], source: dict[str, Any], base_seed: int
) -> RoleAssignmentRecipe:
    phase, role = required["phase"], required["role"]
    final = role == "final_test"
    original_count, retained_count = _development_geometry(source, phase)
    streams = {row.name: row.value for row in confirmation_seed_streams(base_seed)}
    prefix = f"phase_{phase}/seed_{base_seed}/{'final' if final else 'development'}"
    # Why this: stratification/exposure select from arrived labels. Only the final
    # index geometry can be declared before data without executing source RNG.
    return RoleAssignmentRecipe(
        slot=ReplicaSlot(**required["slot"]),
        phase=phase,
        role=role,
        expected_count=required["expected_count"],
        source_index_count=required["expected_count"] if final else original_count,
        retained_source_count=required["expected_count"] if final else retained_count,
        sample_id_prefix=prefix,
        declared_sample_ids=tuple(f"{prefix}/{i}" for i in range(required["expected_count"]))
        if final
        else None,
        sample_ids_available_at="before_first_source" if final else f"phase_{phase}_arrival",
        assignment_policy="canonical_final_source_order_v1"
        if final
        else "arrived_class_stratified_roles_v1",
        assignment_seed=None if final else streams[f"phase_{phase}_roles"],
        exposure_policy="not_applied_to_final_test"
        if final
        else "class_balanced_original_source_positions_v1"
        if phase == "b"
        else "identity_original_source_positions_v1",
        exposure_seed=streams["phase_b_exposure"] if phase == "b" and not final else None,
        inner_guard_fraction=None if final else source["inner_guard_fraction"],
        outer_selection_fraction=None if final else source["outer_selection_fraction"],
        source_available_at=required["source_available_at"],
        labels_available_at=required["labels_available_at"],
        allowed_use=required["allowed_use"],
    )


def _role_recipes(
    design: dict[str, Any], bindings: tuple[ReplicaSeedBinding, ...]
) -> tuple[RoleAssignmentRecipe, ...]:
    sources = {
        row["name"]: row["configuration_without_historical_reservations"]["source"]
        for row in design["family_templates"]
    }
    bases = {row.slot: row.base_seed for row in bindings}
    return tuple(
        _role_recipe(
            required, sources[required["slot"]["family"]], bases[ReplicaSlot(**required["slot"])]
        )
        for required in design["role_requirements"]
    )


def freeze_prospective_generation_request(
    design: dict[str, Any],
    bindings: tuple[ReplicaSeedBinding, ...],
    expected_code_identity: EvidenceIdentity,
    claimed_streams: tuple[DerivedSeedStream, ...],
) -> ProspectiveGenerationRequest:
    """Derive every recipe from the complete validated fixed declarations."""
    design_identity = EvidenceIdentity(**prospective_design_identity(design))
    validate_prospective_stream_declarations(design, bindings, design_identity, claimed_streams)
    sources = declare_prospective_sources(design, bindings, expected_code_identity)
    request = ProspectiveGenerationRequest(
        design_identity,
        bindings,
        claimed_streams,
        sources,
        _role_recipes(design, bindings),
        prospective_source_map_identity(sources),
        EvidenceIdentity(0, "0" * 64),
        tuple(design["required_actual_request_bindings"]),
    )
    return replace(request, request_identity=prospective_generation_request_identity(request))


def _require_exact_value(actual: object, expected: object, path: str) -> None:
    # Why this: equality alone accepts bool/int/float aliases and schema subclasses.
    # Check the complete recursively immutable record, including its whole identities.
    if type(actual) is not type(expected):
        raise ValueError(f"generation request has a foreign schema/type at {path}")
    if is_dataclass(expected):
        for field in fields(expected):
            _require_exact_value(
                getattr(actual, field.name), getattr(expected, field.name), f"{path}.{field.name}"
            )
    elif isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(actual) != len(expected):
            raise ValueError(f"generation request has incomplete ordered membership at {path}")
        for position, (value, required) in enumerate(zip(actual, expected, strict=True)):
            _require_exact_value(value, required, f"{path}[{position}]")
    elif actual != expected:
        raise ValueError(f"generation request differs from the frozen rule/value at {path}")


def inspect_prospective_generation_request(
    design: dict[str, Any],
    expected_code_identity: EvidenceIdentity,
    request: ProspectiveGenerationRequest,
) -> GenerationRequestInspection:
    """Check the entire V2 request; leave actual IDs, provenance and authority unset."""
    if type(request) is not ProspectiveGenerationRequest:
        raise ValueError("generation request requires the exact versioned record")
    expected = freeze_prospective_generation_request(
        design, request.bindings, expected_code_identity, request.streams
    )
    _require_exact_value(request, expected, "request")
    return GenerationRequestInspection(
        request.sources,
        request.role_recipes,
        request.source_map_identity,
        request.request_identity,
        request.required_actual_bindings,
    )


def _validate_concrete_role_schema(
    roles: tuple[ProspectiveRoleDeclaration, ...], recipes: tuple[RoleAssignmentRecipe, ...]
) -> None:
    """Reject foreign payloads before canonical encoding can invoke deepcopy."""
    if type(roles) is not tuple or len(roles) != len(recipes):
        raise ValueError("concrete roles require the complete ordered immutable layout")
    for position, (role, recipe) in enumerate(zip(roles, recipes, strict=True)):
        if (
            type(role) is not ProspectiveRoleDeclaration
            or type(role.sample_ids) is not tuple
            or any(type(value) is not str for value in role.sample_ids)
        ):
            raise ValueError(f"concrete role has a foreign record/ID type at position {position}")
        expected = ProspectiveRoleDeclaration(
            recipe.slot,
            recipe.phase,
            recipe.role,
            recipe.expected_count,
            role.sample_ids,
            recipe.source_available_at,
            recipe.labels_available_at,
            recipe.allowed_use,
        )
        _require_exact_value(role, expected, f"concrete_roles[{position}]")


def inspect_arrived_role_declarations(
    design: dict[str, Any],
    expected_code_identity: EvidenceIdentity,
    generation_request: ProspectiveGenerationRequest,
    roles: tuple[ProspectiveRoleDeclaration, ...],
) -> ArrivedRoleDeclarationInspection:
    """Bind full concrete ID claims separately; actual arrival/assignment is unproved.

    A later outer arrival observer must prove these are the frozen recipe's real
    outputs at permitted events. This bridge only composes existing complete
    metadata partition/shared-view checks; it never mutates the pre-source request.
    """
    inspected = inspect_prospective_generation_request(
        design, expected_code_identity, generation_request
    )
    _validate_concrete_role_schema(roles, inspected.role_recipes)
    concrete = ProspectiveRoleRequest(
        generation_request.design_identity,
        generation_request.bindings,
        generation_request.streams,
        generation_request.sources,
        roles,
        generation_request.source_map_identity,
        EvidenceIdentity(0, "0" * 64),
        generation_request.required_actual_bindings,
    )
    concrete = replace(concrete, request_identity=prospective_role_request_identity(concrete))
    result = inspect_prospective_role_request(design, expected_code_identity, concrete)
    return ArrivedRoleDeclarationInspection(inspected.request_identity, result)
