"""Screen declared confirmation streams without constructing or admitting sources.

Inputs are exact seed identities and conservative occurrence/uncertainty counts.
Outputs retain ordered streams, possible numeric collisions and open admission
requirements. Declarations are not execution, chronology or independence proof.
No IO, RNG, configuration selection, model or role release belongs here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceIdentity:
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class DerivedSeedStream:
    base_seed: int
    name: str
    value: int


@dataclass(frozen=True)
class SeedStreamCollision:
    stream: DerivedSeedStream
    prior_occurrences: int


@dataclass(frozen=True)
class ProposedStreamCollision:
    first: DerivedSeedStream
    second: DerivedSeedStream


@dataclass(frozen=True)
class SeedRoleScreen:
    proposed_base_seeds: tuple[int, ...]
    streams: tuple[DerivedSeedStream, ...]
    collisions: tuple[SeedStreamCollision, ...]
    proposal_collisions: tuple[ProposedStreamCollision, ...]
    uncertainty_counts: tuple[tuple[str, int], ...]
    blockers: tuple[str, ...]
    fresh_roles_authorized: bool = False


_OFFSETS = (
    ("phase_a_source", 0),
    ("phase_b_source", 101),
    ("phase_a_roles", 17),
    ("phase_b_roles", 138),
    ("phase_b_exposure", 118),
    ("model_initialization", 1001),
    ("parent_selection", 5001),
    ("circadian_local_rng", 11002),
)


def validate_evidence_identity(identity: EvidenceIdentity) -> None:
    if (
        type(identity) is not EvidenceIdentity
        or type(identity.byte_count) is not int
        or identity.byte_count < 0
        or type(identity.sha256) is not str
        or len(identity.sha256) != 64
        or any(character not in "0123456789abcdef" for character in identity.sha256)
    ):
        raise ValueError("seed evidence requires an exact whole identity")


def confirmation_seed_streams(base_seed: int) -> tuple[DerivedSeedStream, ...]:
    """Declare every current confirmation stream, including local noise RNG."""
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("confirmation stream seed must be a nonnegative Python integer")
    # Why this: checking only a base misses cross-stream reuse. The separate
    # local circadian RNG is base+1001+10001, as bound by checkpoint validation.
    return tuple(
        DerivedSeedStream(base_seed, name, base_seed + offset) for name, offset in _OFFSETS
    )


def screen_seed_roles(
    proposed_base_seeds: tuple[int, ...],
    prior_occurrences: Mapping[int, int],
    uncertainty_counts: Mapping[str, int],
) -> SeedRoleScreen:
    """A collision screen never upgrades declaration evidence to admission."""
    if type(proposed_base_seeds) is not tuple:
        raise ValueError("proposed seeds must be an ordered immutable tuple")
    streams = tuple(
        stream for seed in proposed_base_seeds for stream in confirmation_seed_streams(seed)
    )
    if len(set(proposed_base_seeds)) != len(proposed_base_seeds):
        raise ValueError("proposed seeds must be unique")
    if any(
        type(seed) is not int or seed < 0 or type(count) is not int or count <= 0
        for seed, count in prior_occurrences.items()
    ):
        raise ValueError("prior seed occurrence identities/counts must be exact and positive")
    if any(
        type(name) is not str or not name or type(count) is not int or count <= 0
        for name, count in uncertainty_counts.items()
    ):
        raise ValueError("seed uncertainty counts must have names and exact positive counts")
    collisions = tuple(
        SeedStreamCollision(stream, prior_occurrences[stream.value])
        for stream in streams
        if stream.value in prior_occurrences
    )
    seen: dict[int, list[DerivedSeedStream]] = {}
    proposal_collisions: list[ProposedStreamCollision] = []
    for stream in streams:
        proposal_collisions.extend(
            ProposedStreamCollision(previous, stream) for previous in seen.get(stream.value, ())
        )
        seen.setdefault(stream.value, []).append(stream)
    # Why this: no caller-supplied boolean, absence of collisions or static
    # literal can replace verified chronology and the full prospective gates.
    return SeedRoleScreen(
        proposed_base_seeds,
        streams,
        collisions,
        tuple(proposal_collisions),
        tuple(sorted(uncertainty_counts.items())),
        (
            "actual_execution_and_release_chronology_unverified",
            "complete_prior_usage_resource_gate_unaccepted",
            "prospective_role_and_full_execution_contract_unfinished",
        ),
    )
