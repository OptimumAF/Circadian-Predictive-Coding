"""Immutable live runtime observations and ports; no process inspection or IO.

Inputs are inspected full V2 snapshots and ordered actual entrypoint names.
Outputs bind the process observed at that time. Source-version attestation,
source/role arrival, independence, resource/repeat and execution are separate.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from src.core.prospective_generation_bundles import GenerationBundleSnapshot
from src.core.prospective_generation_ownership import (
    GenerationOwnershipScope,
    ObservedGenerationBundleOwnership,
)
from src.core.seed_stream_screening import EvidenceIdentity


@dataclass(frozen=True)
class RuntimeCodeObservation:
    scope: GenerationOwnershipScope
    process_id: int
    lease_nonce: str
    sequence: int
    observed_utc: str
    entrypoints: tuple[str, ...]
    runtime_json: str
    runtime_identity: EvidenceIdentity
    complete_process_membership_observed: bool = True
    runtime_source_version_attested: bool = False


class LiveRuntimeCodeLease(Protocol):
    def observe(self, snapshot: GenerationBundleSnapshot) -> RuntimeCodeObservation: ...


class GenerationRuntimeObserver(Protocol):
    def freeze(
        self, snapshot: GenerationBundleSnapshot, entrypoints: tuple[str, ...]
    ) -> AbstractContextManager[LiveRuntimeCodeLease]: ...


@dataclass(frozen=True)
class ObservedGenerationRuntime:
    ownership: ObservedGenerationBundleOwnership
    runtime_at_entry: RuntimeCodeObservation
    required_actual_bindings: tuple[str, ...]
    runtime_process_observed: bool = True
    runtime_source_version_attested: bool = False
    actual_arrival_verified: bool = False
    chronology_verified: bool = False
    independent_source_replications: None = None
    fresh_roles_authorized: bool = False
    execution_authorized: bool = False
    precision_certified: bool = False
