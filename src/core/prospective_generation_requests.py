"""Immutable pre-source recipes and separate concrete-role declaration results.

Inputs freeze assignment rules without development arrays, labels or realized
IDs. Final ID declarations describe index geometry only. Outputs preserve all
actual-proof obligations; no record proves source arrival, assignment execution,
physical code closure, independence or release authority. App owns validation;
IO and scientific work belong outside these records.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_role_requests import ProspectiveSourceDeclaration, RoleRequestInspection
from src.core.seed_stream_screening import DerivedSeedStream, EvidenceIdentity

GENERATION_REQUEST_SCHEMA = "p67_complete_prospective_generation_request_v2"


@dataclass(frozen=True)
class RoleAssignmentRecipe:
    slot: ReplicaSlot
    phase: str
    role: str
    expected_count: int
    source_index_count: int
    retained_source_count: int
    sample_id_prefix: str
    declared_sample_ids: tuple[str, ...] | None
    sample_ids_available_at: str
    assignment_policy: str
    assignment_seed: int | None
    exposure_policy: str
    exposure_seed: int | None
    inner_guard_fraction: float | None
    outer_selection_fraction: float | None
    source_available_at: str
    labels_available_at: str
    allowed_use: str
    array_identity: EvidenceIdentity | None = None
    final_released: bool = False


@dataclass(frozen=True)
class ProspectiveGenerationRequest:
    design_identity: EvidenceIdentity
    bindings: tuple[ReplicaSeedBinding, ...]
    streams: tuple[DerivedSeedStream, ...]
    sources: tuple[ProspectiveSourceDeclaration, ...]
    role_recipes: tuple[RoleAssignmentRecipe, ...]
    source_map_identity: EvidenceIdentity
    request_identity: EvidenceIdentity
    required_actual_bindings: tuple[str, ...]
    schema_id: str = GENERATION_REQUEST_SCHEMA
    execution_requested: bool = False


@dataclass(frozen=True)
class GenerationRequestInspection:
    sources: tuple[ProspectiveSourceDeclaration, ...]
    role_recipes: tuple[RoleAssignmentRecipe, ...]
    source_map_identity: EvidenceIdentity
    request_identity: EvidenceIdentity
    required_actual_bindings: tuple[str, ...]
    final_id_declarations_complete: bool = True
    development_sample_ids_bound: bool = False
    independent_source_replications: None = None
    actual_sources_verified: bool = False
    actual_roles_verified: bool = False
    actual_code_verified: bool = False
    chronology_verified: bool = False
    fresh_roles_authorized: bool = False
    execution_authorized: bool = False
    precision_certified: bool = False


@dataclass(frozen=True)
class ArrivedRoleDeclarationInspection:
    generation_request_identity: EvidenceIdentity
    concrete_role_inspection: RoleRequestInspection
    actual_arrival_verified: bool = False
    assignment_execution_verified: bool = False
