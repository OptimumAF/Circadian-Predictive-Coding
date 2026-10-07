"""Compose full V2/native owner and a trusted live process runtime proof port.

No process/filesystem/scientific operations occur in app. A whole runtime
observation is not source-version, prior-freshness or execution attestation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
import re
from typing import Any

from src.app.continual_confirmation_json import same_json
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.core.prospective_generation_bundles import (
    GenerationBundleSnapshot,
    ProspectiveGenerationBundleReader,
)
from src.core.prospective_generation_ownership import (
    GenerationRequestOwner,
    generation_ownership_scope,
    generation_ownership_utc,
    validate_generation_ownership_scope,
)
from src.core.prospective_request_bundles import ProspectiveBundleSpec
from src.core.prospective_runtime_closure import (
    GenerationRuntimeObserver,
    ObservedGenerationRuntime,
    RuntimeCodeObservation,
)
from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity
from src.core.runtime_payload_schema import validate_runtime_payload


def validate_runtime_entrypoints(names: tuple[str, ...]) -> None:
    if type(names) is not tuple or not names or any(type(name) is not str for name in names):
        raise ValueError("runtime requires unique ordered immutable entrypoints")
    if len(set(names)) != len(names):
        raise ValueError("runtime requires unique ordered immutable entrypoints")
    pattern = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    if any(
        type(name) is not str or re.fullmatch(pattern, name, flags=re.ASCII) is None
        for name in names
    ):
        raise ValueError("runtime entrypoint requires a canonical module:attribute path")


def _validate_runtime_header(
    snapshot: GenerationBundleSnapshot,
    observation: Any,
) -> None:
    if type(observation) is not RuntimeCodeObservation:
        raise ValueError("runtime port returned a foreign immutable observation")
    validate_generation_ownership_scope(observation.scope)
    same_json(
        asdict(observation.scope), asdict(generation_ownership_scope(snapshot)), "runtime scope"
    )
    validate_runtime_entrypoints(observation.entrypoints)
    if (
        type(observation.process_id) is not int
        or observation.process_id <= 0
        or type(observation.sequence) is not int
        or observation.sequence < 0
        or type(observation.lease_nonce) is not str
        or re.fullmatch(r"[0-9a-f]{32}", observation.lease_nonce) is None
        or observation.complete_process_membership_observed is not True
        or observation.runtime_source_version_attested is not False
    ):
        raise ValueError("runtime observation has invalid process/lifetime/proof values")
    if generation_ownership_utc(observation.observed_utc) < generation_ownership_utc(
        snapshot.observed_utc
    ):
        raise ValueError("runtime observation predates its full file observation")
    validate_evidence_identity(observation.runtime_identity)
    if type(observation.runtime_json) is not str:
        raise ValueError("runtime body must be immutable canonical JSON")


def validate_runtime_observation(
    snapshot: GenerationBundleSnapshot,
    observation: Any,
    previous: RuntimeCodeObservation | None = None,
) -> None:
    _validate_runtime_header(snapshot, observation)
    try:
        raw = observation.runtime_json.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("runtime body must contain valid UTF-8 JSON") from error
    if EvidenceIdentity(len(raw), sha256(raw).hexdigest()) != observation.runtime_identity:
        raise ValueError("runtime whole observation identity differs")
    try:
        body = json.loads(raw)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("runtime observation JSON is invalid") from error
    if (
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        != observation.runtime_json
    ):
        raise ValueError("runtime observation must retain its whole canonical JSON")
    validate_runtime_payload(body, observation.entrypoints)
    if previous is None:
        if observation.sequence != 0:
            raise ValueError("runtime initial observation sequence differs")
    else:
        # Why this: a bad prior header cannot become valid through Python's
        # bool/int equality. The equal immutable body has just been checked in
        # full, so reparsing it again would add cost without additional coverage.
        _validate_runtime_header(snapshot, previous)
        for name in (
            "scope",
            "process_id",
            "lease_nonce",
            "entrypoints",
            "runtime_json",
            "runtime_identity",
        ):
            if getattr(observation, name) != getattr(previous, name):
                raise ValueError("runtime code/object/process membership drifted while held")
        if observation.sequence != previous.sequence + 1 or generation_ownership_utc(
            observation.observed_utc
        ) < generation_ownership_utc(previous.observed_utc):
            raise ValueError("runtime observation sequence/time moved backward")


@contextmanager
def observe_prospective_generation_runtime(
    design: dict[str, Any],
    spec: ProspectiveBundleSpec,
    reader: ProspectiveGenerationBundleReader,
    owner: GenerationRequestOwner,
    runtime: GenerationRuntimeObserver,
    entrypoints: tuple[str, ...],
) -> Iterator[ObservedGenerationRuntime]:
    validate_runtime_entrypoints(entrypoints)
    with claim_prospective_generation_bundle(design, spec, reader, owner) as ownership:
        snapshot = ownership.bundle.snapshot
        with runtime.freeze(snapshot, entrypoints) as lease:
            previous: RuntimeCodeObservation | None = None
            try:
                first = lease.observe(snapshot)
                validate_runtime_observation(snapshot, first)
                previous = first
                if first.entrypoints != entrypoints:
                    raise ValueError("runtime port detached requested actual entrypoints")
                reader.recheck_bundle(snapshot)
                entry = lease.observe(snapshot)
                validate_runtime_observation(snapshot, entry, first)
                previous = entry
                yield ObservedGenerationRuntime(
                    ownership, entry, ownership.required_actual_bindings
                )
            finally:
                try:
                    reader.recheck_bundle(snapshot)
                finally:
                    final = lease.observe(snapshot)
                    validate_runtime_observation(snapshot, final, previous)
