"""Immutable declarations for the complete prospective source and role layout.

Inputs describe expected metadata, never arrays or physical provenance. Inspected
outputs retain the actual-proof obligations and grant no execution or release
authority. Validation orchestration belongs in app; IO and science do not belong
in these records.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.seed_stream_screening import DerivedSeedStream, EvidenceIdentity

ROLE_REQUEST_SCHEMA = "p67_complete_prospective_role_source_request_v1"


@dataclass(frozen=True)
class ProspectiveSourceDeclaration:
    source_group: str
    phase: str
    base_seed: int
    code_identity: EvidenceIdentity
    generator_configuration_identity: EvidenceIdentity


@dataclass(frozen=True)
class ProspectiveRoleDeclaration:
    slot: ReplicaSlot
    phase: str
    role: str
    expected_count: int
    sample_ids: tuple[str, ...]
    source_available_at: str
    labels_available_at: str
    allowed_use: str
    array_identity: EvidenceIdentity | None = None
    final_released: bool = False


@dataclass(frozen=True)
class ProspectiveRoleRequest:
    design_identity: EvidenceIdentity
    bindings: tuple[ReplicaSeedBinding, ...]
    streams: tuple[DerivedSeedStream, ...]
    sources: tuple[ProspectiveSourceDeclaration, ...]
    roles: tuple[ProspectiveRoleDeclaration, ...]
    source_map_identity: EvidenceIdentity
    request_identity: EvidenceIdentity
    required_actual_bindings: tuple[str, ...]
    schema_id: str = ROLE_REQUEST_SCHEMA
    execution_requested: bool = False


@dataclass(frozen=True)
class RoleRequestInspection:
    sources: tuple[ProspectiveSourceDeclaration, ...]
    roles: tuple[ProspectiveRoleDeclaration, ...]
    source_map_identity: EvidenceIdentity
    request_identity: EvidenceIdentity
    required_actual_bindings: tuple[str, ...]
    independent_source_replications: None = None
    actual_sources_verified: bool = False
    actual_roles_verified: bool = False
    actual_code_verified: bool = False
    chronology_verified: bool = False
    fresh_roles_authorized: bool = False
    execution_authorized: bool = False
    precision_certified: bool = False
