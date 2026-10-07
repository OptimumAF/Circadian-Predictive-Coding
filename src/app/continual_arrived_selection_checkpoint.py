"""Typed v7 candidate-manifest checkpoint and source-free header checks.

Inputs are an ordered candidate manifest, completed unscored candidate
records, and an optional active v6 cursor. Outputs are local checkpoint
values. This module does not train, read files, or release final roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.app.continual_arrived_benchmark import ContinualArrivedRolesConfig
from src.app.continual_arrived_checkpoint import ArrivedRunnerCheckpoint, ArrivedUnscoredSeed

if TYPE_CHECKING:
    from src.app.continual_arrived_selection import ArrivedOuterTrial, ArrivedSelectionFreeze


ARRIVED_SELECTION_CHECKPOINT_FORMAT = 8


@dataclass(frozen=True)
class CompletedSelectionCandidate:
    """One fully trained candidate, still without any final-test value."""

    candidate_id: str
    unscored_seeds: tuple[ArrivedUnscoredSeed, ...]
    trials: tuple[ArrivedOuterTrial, ...]
    trial_digest: str
    sleep_history_digest: str


@dataclass(frozen=True)
class ArrivedSelectionCheckpoint:
    """One atomic run-level transaction around format-6 candidate progress."""

    format_version: int
    manifest_digest: str
    candidate_manifest: tuple[tuple[str, ContinualArrivedRolesConfig], ...]
    candidate_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    candidate_index: int
    stage: str
    completed_candidates: tuple[CompletedSelectionCandidate, ...]
    active_v6: ArrivedRunnerCheckpoint | None = None
    freeze: ArrivedSelectionFreeze | None = None


class ArrivedSelectionCheckpointStore(Protocol):
    """App port for a trusted replaceable local v7 transaction."""

    def load(self) -> ArrivedSelectionCheckpoint: ...

    def save(self, checkpoint: ArrivedSelectionCheckpoint) -> None: ...


def validate_selection_checkpoint_header(
    checkpoint: ArrivedSelectionCheckpoint,
    *,
    manifest_digest: str,
    candidate_manifest: tuple[tuple[str, ContinualArrivedRolesConfig], ...],
    candidate_ids: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    """Reject wrong run and impossible outer cursor before source access."""
    if (
        type(checkpoint) is not ArrivedSelectionCheckpoint
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != ARRIVED_SELECTION_CHECKPOINT_FORMAT
        or checkpoint.manifest_digest != manifest_digest
        or checkpoint.candidate_manifest != candidate_manifest
        or checkpoint.candidate_ids != candidate_ids
        or checkpoint.seeds != seeds
    ):
        raise ValueError("incompatible v7 candidate manifest or checkpoint format")
    index = checkpoint.candidate_index
    if (
        type(index) is not int
        or not 0 <= index <= len(candidate_ids)
        or type(checkpoint.completed_candidates) is not tuple
        or len(checkpoint.completed_candidates) != index
        or any(
            type(record) is not CompletedSelectionCandidate
            or record.candidate_id != candidate_ids[position]
            or type(record.unscored_seeds) is not tuple
            or len(record.unscored_seeds) != len(seeds)
            or type(record.trials) is not tuple
            or type(vars(record).get("sleep_history_digest")) is not str
            or len(vars(record).get("sleep_history_digest", "")) != 64
            for position, record in enumerate(checkpoint.completed_candidates)
        )
    ):
        raise ValueError("incompatible v7 completed candidate cursor")
    if checkpoint.stage == "frozen":
        if (
            index != len(candidate_ids)
            or checkpoint.active_v6 is not None
            or checkpoint.freeze is None
            or type(vars(checkpoint.freeze).get("sleep_history_digest")) is not str
            or len(vars(checkpoint.freeze).get("sleep_history_digest", "")) != 64
        ):
            raise ValueError("incompatible v7 frozen selection cursor")
    elif checkpoint.stage == "training":
        if checkpoint.freeze is not None or (
            checkpoint.active_v6 is not None
            and (
                index == len(candidate_ids)
                or type(checkpoint.active_v6) is not ArrivedRunnerCheckpoint
            )
        ):
            raise ValueError("incompatible v7 active candidate cursor")
    else:
        raise ValueError("incompatible v7 selection stage")
