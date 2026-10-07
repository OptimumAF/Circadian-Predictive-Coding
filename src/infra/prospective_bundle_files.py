"""Shared whole-file IO for separately typed prospective bundle versions.

Inputs are a root/clock, closed file spec and version-specific decoder/source
accessor/recheck callbacks. Outputs are the decoder's immutable snapshot after
all physical files are checked. This module owns path/alias/publication/canonical
JSON/byte checks; it does not select schemas, inspect scientific rules, prove
runtime closure/ownership or execute code/data/models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, TypeVar

from src.app.continual_confirmation_json import same_json
from src.app.prospective_request_bundles import decode_prospective_code_manifest
from src.core.prospective_request_bundles import (
    CodeFileBinding,
    ProspectiveBundleSpec,
    validate_bundle_spec,
)
from src.core.prospective_role_requests import ProspectiveSourceDeclaration
from src.core.seed_stream_screening import EvidenceIdentity
from src.core.seed_usage import strict_seed_metadata_json

SnapshotType = TypeVar("SnapshotType")


class WholeProspectiveBundleFiles:
    """Use one physical integrity policy for V1 and V2; callers own typed schemas."""

    def __init__(self, root: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._root = root.resolve(strict=True)
        self._clock = clock if clock is not None else lambda: datetime.now(timezone.utc)

    def _path(self, name: str) -> Path:
        path = self._root / name
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self._root) or not path.is_file():
                raise ValueError("prospective bundle file is outside its root or nonregular")
            for part in (path, *path.parents):
                if part.is_relative_to(self._root) and (
                    part.is_symlink() or getattr(part, "is_junction", lambda: False)()
                ):
                    raise ValueError("prospective bundle paths cannot use symlinks or junctions")
            if any(
                marker.exists()
                for marker in (path.with_suffix(".claim"), path.with_suffix(".failure.json"))
            ):
                raise ValueError("prospective bundle file has a pending or failed publication")
            return path
        except OSError as error:
            raise ValueError(f"prospective bundle file is unavailable: {name}") from error

    def _paths(self, spec: ProspectiveBundleSpec) -> dict[str, Path]:
        paths = {
            name: self._path(name)
            for name in (
                spec.request_path,
                spec.source_map_path,
                spec.code_manifest_path,
                *spec.code_paths,
            )
        }
        physical = {(path.stat().st_dev, path.stat().st_ino) for path in paths.values()}
        if len(physical) != len(paths):
            raise ValueError("prospective bundle logical paths alias the same physical file")
        return paths

    def _read(self, path: Path, expected: EvidenceIdentity) -> bytes:
        try:
            if path.stat().st_size != expected.byte_count:
                raise ValueError(f"prospective bundle whole byte count differs: {path.name}")
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(f"prospective bundle file could not be read: {path.name}") from error
        if EvidenceIdentity(len(raw), sha256(raw).hexdigest()) != expected:
            raise ValueError(f"prospective bundle whole identity differs: {path.name}")
        return raw

    def _json(self, path: Path, expected: EvidenceIdentity) -> Any:
        raw = self._read(path, expected)
        body = strict_seed_metadata_json(raw)
        canonical = (
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if raw != canonical:
            raise ValueError("prospective bundle metadata must use complete canonical JSON bytes")
        return body

    def _read_code(self, files: tuple[CodeFileBinding, ...], paths: dict[str, Path]) -> None:
        for row in files:
            self._read(paths[row.path], row.identity)

    def read_snapshot(
        self,
        spec: ProspectiveBundleSpec,
        decoder: Callable[
            [Any, ProspectiveBundleSpec, tuple[CodeFileBinding, ...], str], SnapshotType
        ],
        sources: Callable[[SnapshotType], tuple[ProspectiveSourceDeclaration, ...]],
        recheck: Callable[[SnapshotType], None],
    ) -> SnapshotType:
        validate_bundle_spec(spec)
        paths = self._paths(spec)
        body = self._json(paths[spec.request_path], spec.request_identity)
        source_map = self._json(paths[spec.source_map_path], spec.source_map_identity)
        manifest = self._json(paths[spec.code_manifest_path], spec.code_manifest_identity)
        files = decode_prospective_code_manifest(manifest, spec)
        now = self._clock()
        if type(now) is not datetime or now.tzinfo != timezone.utc:
            raise ValueError("prospective bundle clock must return an aware UTC datetime")
        snapshot = decoder(body, spec, files, now.isoformat())
        same_json(
            source_map,
            [asdict(row) for row in sources(snapshot)],
            "entire physical source metadata map",
        )
        self._read_code(files, paths)
        # Why this: keep the versioned reader's read/recheck contract while sharing
        # identical physical checks. Its snapshot type remains outside this store.
        recheck(snapshot)
        return snapshot

    def recheck_snapshot(
        self,
        snapshot: SnapshotType,
        spec: ProspectiveBundleSpec,
        code_files: tuple[CodeFileBinding, ...],
        observed_utc: str,
        decoder: Callable[
            [Any, ProspectiveBundleSpec, tuple[CodeFileBinding, ...], str], SnapshotType
        ],
        sources: Callable[[SnapshotType], tuple[ProspectiveSourceDeclaration, ...]],
    ) -> None:
        validate_bundle_spec(spec)
        paths = self._paths(spec)
        body = self._json(paths[spec.request_path], spec.request_identity)
        source_map = self._json(paths[spec.source_map_path], spec.source_map_identity)
        manifest = self._json(paths[spec.code_manifest_path], spec.code_manifest_identity)
        files = decode_prospective_code_manifest(manifest, spec)
        if files != code_files:
            raise ValueError("prospective bundle code membership changed after inspection")
        rebuilt = decoder(body, spec, files, observed_utc)
        if rebuilt != snapshot:
            raise ValueError("prospective bundle snapshot changed after whole-file read")
        same_json(
            source_map,
            [asdict(row) for row in sources(snapshot)],
            "entire rechecked source metadata map",
        )
        self._read_code(files, paths)
        if self._paths(spec) != paths:
            raise ValueError("prospective bundle paths changed during whole-file recheck")
