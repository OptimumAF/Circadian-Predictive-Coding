"""Implement V2 generation read/recheck with the shared whole-file IO policy.

Inputs are a root/UTC clock and the existing immutable expected-file spec. Outputs
are typed V2 snapshots after all physical file checks. App owns full recipe/rule
inspection. This reader does not construct source/labels/arrays, prove actual
arrival/runtime closure/live ownership or grant scientific execution authority.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from src.app.prospective_generation_bundles import decode_prospective_generation_bundle
from src.core.prospective_generation_bundles import GenerationBundleSnapshot
from src.core.prospective_request_bundles import ProspectiveBundleSpec
from src.core.prospective_role_requests import ProspectiveSourceDeclaration
from src.infra.prospective_bundle_files import WholeProspectiveBundleFiles


def _snapshot_sources(
    snapshot: GenerationBundleSnapshot,
) -> tuple[ProspectiveSourceDeclaration, ...]:
    # Why this: an error traceback retains the selector passed to file recheck.
    # A module function avoids creating a new callable during a live freeze.
    return snapshot.generation_request.sources


class FileProspectiveGenerationBundleReader:
    """Keep V2 snapshots separate while sharing all closed whole-file guarantees."""

    def __init__(self, root: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._files = WholeProspectiveBundleFiles(root, clock=clock)

    def read_bundle(self, spec: ProspectiveBundleSpec) -> GenerationBundleSnapshot:
        return self._files.read_snapshot(
            spec,
            decode_prospective_generation_bundle,
            _snapshot_sources,
            self.recheck_bundle,
        )

    def recheck_bundle(self, snapshot: GenerationBundleSnapshot) -> None:
        if type(snapshot) is not GenerationBundleSnapshot:
            raise ValueError("generation bundle requires its exact immutable V2 snapshot")
        self._files.recheck_snapshot(
            snapshot,
            snapshot.spec,
            snapshot.code_files,
            snapshot.observed_utc,
            decode_prospective_generation_bundle,
            _snapshot_sources,
        )
