"""Verify V1 whole-file request metadata with the shared physical IO policy.

The public root/clock/read/recheck API and V1 typed schema are preserved. Shared
outer IO owns canonical whole bytes, paths, aliases and publication/recheck checks.
App owns full role inspection; this reader does not prove runtime code closure,
actual sources, before-source chronology, live ownership or execution authority.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from src.app.prospective_request_bundles import decode_prospective_bundle
from src.core.prospective_request_bundles import BundleSnapshot, ProspectiveBundleSpec
from src.infra.prospective_bundle_files import WholeProspectiveBundleFiles


class FileProspectiveBundleReader:
    """Read/recheck every V1 pinned file through the same physical integrity rules."""

    def __init__(self, root: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._files = WholeProspectiveBundleFiles(root, clock=clock)

    def read_bundle(self, spec: ProspectiveBundleSpec) -> BundleSnapshot:
        return self._files.read_snapshot(
            spec,
            decode_prospective_bundle,
            lambda row: row.role_request.sources,
            self.recheck_bundle,
        )

    def recheck_bundle(self, snapshot: BundleSnapshot) -> None:
        if type(snapshot) is not BundleSnapshot:
            raise ValueError("prospective bundle requires its exact immutable snapshot")
        self._files.recheck_snapshot(
            snapshot,
            snapshot.spec,
            snapshot.code_files,
            snapshot.observed_utc,
            decode_prospective_bundle,
            lambda row: row.role_request.sources,
        )
