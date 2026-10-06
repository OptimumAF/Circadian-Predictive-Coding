"""Compose complete V2 file inspection with a live request-owner proof port.

Inputs are the unchanged fixed design/spec and trusted reader/owner ports. Outputs
retain point-in-time native ownership evidence and every actual admission proof.
App performs no filesystem/locking/scientific work; runtime closure, chronology,
arrival, source independence/resource/repeat/b3 and execution stay unverified.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import PurePath
import re
from typing import Any

from src.app.continual_confirmation_json import same_json
from src.app.prospective_generation_bundles import preflight_prospective_generation_bundle
from src.core.prospective_generation_bundles import (
    GenerationBundleSnapshot,
    ProspectiveGenerationBundleReader,
)
from src.core.prospective_generation_ownership import (
    GenerationOwnershipObservation,
    GenerationRequestOwner,
    ObservedGenerationBundleOwnership,
    generation_owner_lock_name,
    generation_ownership_scope,
    generation_ownership_utc,
    validate_generation_ownership_scope,
)
from src.core.prospective_request_bundles import ProspectiveBundleSpec


def _validate_observation(
    snapshot: GenerationBundleSnapshot,
    observation: GenerationOwnershipObservation,
    previous: GenerationOwnershipObservation | None = None,
) -> None:
    if type(observation) is not GenerationOwnershipObservation:
        raise ValueError("generation owner port returned a foreign observation")
    validate_generation_ownership_scope(observation.scope)
    same_json(
        asdict(observation.scope),
        asdict(generation_ownership_scope(snapshot)),
        "complete observed ownership scope",
    )
    if observation.native_lock_observed is not True:
        raise ValueError("generation owner port did not observe a held native lock")
    if (
        type(observation.registry_root) is not str
        or not PurePath(observation.registry_root).is_absolute()
        or PurePath(observation.registry_root).as_posix() != observation.registry_root
        or ".." in PurePath(observation.registry_root).parts
    ):
        raise ValueError("generation owner registry must be canonical and absolute")
    if type(
        observation.lock_file_name
    ) is not str or observation.lock_file_name != generation_owner_lock_name(observation.scope):
        raise ValueError("generation owner lock name is detached from its full request")
    for value, minimum in (
        (observation.physical_device, 0),
        (observation.physical_inode, 1),
        (observation.sequence, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError("generation owner physical/sequence values require exact integers")
    if (
        type(observation.lease_nonce) is not str
        or re.fullmatch(r"[0-9a-f]{32}", observation.lease_nonce) is None
    ):
        raise ValueError("generation owner requires a canonical live lease nonce")
    acquired = generation_ownership_utc(observation.acquired_utc)
    observed = generation_ownership_utc(observation.observed_utc)
    if (
        not generation_ownership_utc(snapshot.prospective_utc) <= acquired <= observed
        or generation_ownership_utc(snapshot.observed_utc) > observed
    ):
        raise ValueError("generation owner times are detached from its file observation")
    if previous is None:
        if observation.sequence != 0:
            raise ValueError("generation owner initial observation sequence differs")
    else:
        _validate_continuity(observation, previous)


def _validate_continuity(
    current: GenerationOwnershipObservation, previous: GenerationOwnershipObservation
) -> None:
    fields = (
        "scope",
        "registry_root",
        "lock_file_name",
        "physical_device",
        "physical_inode",
        "lease_nonce",
        "acquired_utc",
    )
    for name in fields:
        if getattr(current, name) != getattr(previous, name):
            raise ValueError("generation owner scope or native handle changed while held")
    if current.sequence != previous.sequence + 1 or generation_ownership_utc(
        current.observed_utc
    ) < generation_ownership_utc(previous.observed_utc):
        raise ValueError("generation owner observation sequence/time did not advance monotonically")


@contextmanager
def claim_prospective_generation_bundle(
    design: dict[str, Any],
    spec: ProspectiveBundleSpec,
    reader: ProspectiveGenerationBundleReader,
    owner: GenerationRequestOwner,
) -> Iterator[ObservedGenerationBundleOwnership]:
    """Keep all full file/owner checks inside a native lease; grant no execution."""
    initial = preflight_prospective_generation_bundle(design, spec, reader)
    with owner.claim(initial.snapshot) as lease:
        first = lease.observe(initial.snapshot)
        _validate_observation(initial.snapshot, first)
        current = preflight_prospective_generation_bundle(design, spec, reader)
        entry = lease.observe(current.snapshot)
        _validate_observation(current.snapshot, entry, first)
        reader.recheck_bundle(current.snapshot)
        try:
            yield ObservedGenerationBundleOwnership(
                current, entry, current.required_actual_bindings
            )
        finally:
            # Why this: failed consumers still owe complete late physical/owner
            # checks. The native port releases its handle even if these fail.
            try:
                reader.recheck_bundle(current.snapshot)
            finally:
                final = lease.observe(current.snapshot)
                _validate_observation(current.snapshot, final, entry)
