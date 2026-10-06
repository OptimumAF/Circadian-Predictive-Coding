"""Verify complete caller-supplied numeric streams against the fixed study.

Inputs are the whole prospective design, all ordered replica bindings, its exact
canonical identity and an immutable external stream tuple. Return the existing
checked replica declaration; actual independence remains unknown and fresh
authority false. This is not source/request/history/role provenance, an RNG
domain proof or an execution gate. No IO, source, RNG, model or score belongs here.
"""

from __future__ import annotations

from typing import Any

from src.app.prospective_confirmation_design import (
    declare_prospective_replica_streams,
    prospective_design_identity,
)
from src.core.prospective_replications import DeclaredReplicaStreams, ReplicaSeedBinding
from src.core.seed_stream_screening import (
    DerivedSeedStream,
    EvidenceIdentity,
    validate_evidence_identity,
)


def _validate_external_streams(
    claimed: tuple[DerivedSeedStream, ...], expected: tuple[DerivedSeedStream, ...]
) -> None:
    if type(claimed) is not tuple or len(claimed) != len(expected):
        raise ValueError("stream declarations require the complete ordered immutable tuple")
    for position, (actual, required) in enumerate(zip(claimed, expected, strict=True)):
        # Why this: dataclass equality alone permits True==1 and integer-valued
        # floats; a caller's record must have the exact current schema and types.
        if (
            type(actual) is not DerivedSeedStream
            or type(actual.base_seed) is not int
            or type(actual.name) is not str
            or type(actual.value) is not int
            or actual != required
        ):
            raise ValueError(f"stream declaration at position {position} differs in type or value")


def validate_prospective_stream_declarations(
    design: dict[str, Any],
    bindings: tuple[ReplicaSeedBinding, ...],
    design_identity: EvidenceIdentity,
    claimed_streams: tuple[DerivedSeedStream, ...],
) -> DeclaredReplicaStreams:
    """Check the whole design and every claimed stream before any scientific use.

    Reuse the public full-layout declaration checks for fixed scope, ordered slots,
    shared groups and all within/between group collisions. No caller-supplied stream
    value, schema extension or numeric separation substitutes for those checks.
    """
    validate_evidence_identity(design_identity)
    required_identity = EvidenceIdentity(**prospective_design_identity(design))
    if design_identity != required_identity:
        raise ValueError(
            "stream declarations are detached from the whole prospective design identity"
        )
    declaration = declare_prospective_replica_streams(design, bindings)
    _validate_external_streams(claimed_streams, declaration.streams)
    return declaration
