"""Immutable file descriptors and a port for complete prospective request bundles.

Inputs bind a closed expected code membership and three whole metadata files.
Snapshots carry decoded declarations and observed file identities; preflight
results retain every actual admission proof obligation. IO implementations belong
outside core. Physical integrity is not runtime closure, ownership, provenance,
chronology, independent replication, resource acceptance or execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.prospective_role_requests import ProspectiveRoleRequest, RoleRequestInspection
from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity


@dataclass(frozen=True)
class ProspectiveBundleSpec:
    request_path: str
    request_identity: EvidenceIdentity
    source_map_path: str
    source_map_identity: EvidenceIdentity
    code_manifest_path: str
    code_manifest_identity: EvidenceIdentity
    code_paths: tuple[str, ...]


@dataclass(frozen=True)
class CodeFileBinding:
    path: str
    identity: EvidenceIdentity


@dataclass(frozen=True)
class BundleSnapshot:
    spec: ProspectiveBundleSpec
    role_request: ProspectiveRoleRequest
    code_files: tuple[CodeFileBinding, ...]
    prospective_utc: str
    owner_id: str
    resource_envelope_json: str
    repeat_envelope_json: str
    observed_utc: str


class ProspectiveBundleReader(Protocol):
    """Implementations verify every whole file and path before returning/rechecking."""

    def read_bundle(self, spec: ProspectiveBundleSpec) -> BundleSnapshot: ...

    def recheck_bundle(self, snapshot: BundleSnapshot) -> None: ...


@dataclass(frozen=True)
class BundlePreflight:
    snapshot: BundleSnapshot
    role_inspection: RoleRequestInspection
    required_actual_bindings: tuple[str, ...]
    physical_files_verified: bool = True
    runtime_code_closure_verified: bool = False
    exclusive_owner_verified: bool = False
    chronology_verified: bool = False
    actual_sources_verified: bool = False
    independent_source_replications: None = None
    fresh_roles_authorized: bool = False
    execution_authorized: bool = False
    precision_certified: bool = False


def validate_bundle_path(name: str) -> None:
    if (
        type(name) is not str
        or not name
        or any(character in name for character in ("\\", ":", "\0"))
        or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in name.split("/"))
    ):
        raise ValueError("prospective bundle paths must be canonical and root-relative")


def validate_bundle_spec(spec: ProspectiveBundleSpec) -> None:
    if type(spec) is not ProspectiveBundleSpec:
        raise ValueError("prospective bundle requires the exact immutable expected-file spec")
    metadata = (spec.request_path, spec.source_map_path, spec.code_manifest_path)
    if type(spec.code_paths) is not tuple or not spec.code_paths:
        raise ValueError("prospective bundle requires complete immutable expected code membership")
    for name in (*metadata, *spec.code_paths):
        validate_bundle_path(name)
    if spec.code_paths != tuple(sorted(spec.code_paths)) or len(
        set(name.casefold() for name in (*metadata, *spec.code_paths))
    ) != len(metadata) + len(spec.code_paths):
        raise ValueError("prospective bundle paths must be distinct and code membership ordered")
    for identity in (spec.request_identity, spec.source_map_identity, spec.code_manifest_identity):
        validate_evidence_identity(identity)
        if identity.byte_count == 0:
            raise ValueError("prospective bundle requires nonempty whole metadata file identities")
