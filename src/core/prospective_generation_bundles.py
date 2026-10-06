"""Immutable V2 physical metadata snapshots, proof port and denied preflight.

Inputs use the existing closed expected-file spec and frozen generation recipes.
Outputs retain physical file observations and all actual-proof obligations. Core
performs no IO or scientific work; matching files do not establish runtime code,
exclusive ownership, before-source chronology, arrival, assignment or freshness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.prospective_generation_requests import (
    GenerationRequestInspection,
    ProspectiveGenerationRequest,
)
from src.core.prospective_request_bundles import CodeFileBinding, ProspectiveBundleSpec


@dataclass(frozen=True)
class GenerationBundleSnapshot:
    spec: ProspectiveBundleSpec
    generation_request: ProspectiveGenerationRequest
    code_files: tuple[CodeFileBinding, ...]
    prospective_utc: str
    owner_id: str
    resource_envelope_json: str
    repeat_envelope_json: str
    observed_utc: str


class ProspectiveGenerationBundleReader(Protocol):
    """Read/recheck all pinned bytes without constructing a scientific source."""

    def read_bundle(self, spec: ProspectiveBundleSpec) -> GenerationBundleSnapshot: ...

    def recheck_bundle(self, snapshot: GenerationBundleSnapshot) -> None: ...


@dataclass(frozen=True)
class GenerationBundlePreflight:
    snapshot: GenerationBundleSnapshot
    generation_inspection: GenerationRequestInspection
    required_actual_bindings: tuple[str, ...]
    physical_files_verified: bool = True
    runtime_code_closure_verified: bool = False
    exclusive_owner_verified: bool = False
    chronology_verified: bool = False
    actual_sources_verified: bool = False
    actual_roles_verified: bool = False
    independent_source_replications: None = None
    fresh_roles_authorized: bool = False
    execution_authorized: bool = False
    precision_certified: bool = False
