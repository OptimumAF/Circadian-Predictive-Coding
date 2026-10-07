"""Immutable complete ownership scope, time observations and live lease ports.

Inputs bind already inspected V2 file metadata. Outputs describe ownership at an
observed time, never a future execution capability. Core performs no IO/locking,
scientific work or runtime/source/freshness admission.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Protocol

from src.core.prospective_generation_bundles import (
    GenerationBundlePreflight,
    GenerationBundleSnapshot,
)
from src.core.prospective_generation_requests import ProspectiveGenerationRequest
from src.core.prospective_request_bundles import (
    CodeFileBinding,
    ProspectiveBundleSpec,
    validate_bundle_spec,
)
from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity


@dataclass(frozen=True)
class GenerationOwnershipScope:
    spec: ProspectiveBundleSpec
    generation_request_identity: EvidenceIdentity
    design_identity: EvidenceIdentity
    source_map_identity: EvidenceIdentity
    code_manifest_identity: EvidenceIdentity
    owner_id: str


def validate_generation_ownership_scope(scope: GenerationOwnershipScope) -> None:
    if type(scope) is not GenerationOwnershipScope:
        raise ValueError("generation owner requires its exact immutable scope")
    validate_bundle_spec(scope.spec)
    for value in (
        scope.generation_request_identity,
        scope.design_identity,
        scope.source_map_identity,
        scope.code_manifest_identity,
    ):
        validate_evidence_identity(value)
        if value.byte_count == 0:
            raise ValueError("generation owner requires complete nonempty identities")
    if (
        scope.source_map_identity != scope.spec.source_map_identity
        or scope.code_manifest_identity != scope.spec.code_manifest_identity
    ):
        raise ValueError("generation owner scope is detached from physical metadata")
    if type(scope.owner_id) is not str or re.fullmatch(r"[0-9a-f]{32}", scope.owner_id) is None:
        raise ValueError("generation owner requires a canonical owner declaration")


def generation_ownership_scope(snapshot: GenerationBundleSnapshot) -> GenerationOwnershipScope:
    if (
        type(snapshot) is not GenerationBundleSnapshot
        or type(snapshot.generation_request) is not ProspectiveGenerationRequest
    ):
        raise ValueError("generation owner requires an exact immutable V2 snapshot")
    validate_bundle_spec(snapshot.spec)
    if generation_ownership_utc(snapshot.prospective_utc) > generation_ownership_utc(
        snapshot.observed_utc
    ):
        raise ValueError("generation owner snapshot UTC is detached")
    if type(snapshot.code_files) is not tuple:
        raise ValueError("generation owner requires complete immutable code membership")
    for row in snapshot.code_files:
        if type(row) is not CodeFileBinding or type(row.path) is not str:
            raise ValueError("generation owner received foreign code membership")
        validate_evidence_identity(row.identity)
    if tuple(row.path for row in snapshot.code_files) != snapshot.spec.code_paths:
        raise ValueError("generation owner code membership is detached")
    request = snapshot.generation_request
    scope = GenerationOwnershipScope(
        snapshot.spec,
        request.request_identity,
        request.design_identity,
        request.source_map_identity,
        snapshot.spec.code_manifest_identity,
        snapshot.owner_id,
    )
    validate_generation_ownership_scope(scope)
    return scope


def generation_owner_lock_name(scope: GenerationOwnershipScope) -> str:
    validate_generation_ownership_scope(scope)
    # Why this: outer owner/file copies of the same full generation request must
    # contend in one configured registry; owner strings cannot choose lock keys.
    return f".p67-generation-{scope.generation_request_identity.sha256}.owner.lock"


def generation_ownership_utc(value: str) -> datetime:
    if type(value) is not str:
        raise ValueError("generation ownership UTC must be a canonical aware string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("generation ownership UTC is invalid") from error
    if parsed.tzinfo != timezone.utc or parsed.isoformat() != value:
        raise ValueError("generation ownership UTC must be canonical and explicitly UTC")
    return parsed


@dataclass(frozen=True)
class GenerationOwnershipObservation:
    scope: GenerationOwnershipScope
    registry_root: str
    lock_file_name: str
    physical_device: int
    physical_inode: int
    lease_nonce: str
    acquired_utc: str
    observed_utc: str
    sequence: int
    native_lock_observed: bool = True


class LiveGenerationRequestLease(Protocol):
    """Observe only while held, rechecking its scope and native file identity."""

    def observe(self, snapshot: GenerationBundleSnapshot) -> GenerationOwnershipObservation: ...


class GenerationRequestOwner(Protocol):
    def claim(
        self, snapshot: GenerationBundleSnapshot
    ) -> AbstractContextManager[LiveGenerationRequestLease]: ...


@dataclass(frozen=True)
class ObservedGenerationBundleOwnership:
    bundle: GenerationBundlePreflight
    ownership_at_entry: GenerationOwnershipObservation
    required_actual_bindings: tuple[str, ...]
    runtime_code_closure_verified: bool = False
    actual_arrival_verified: bool = False
    chronology_verified: bool = False
    independent_source_replications: None = None
    fresh_roles_authorized: bool = False
    execution_authorized: bool = False
    precision_certified: bool = False
