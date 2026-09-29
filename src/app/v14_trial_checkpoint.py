"""Validate a trusted local prefix of complete, unscored v14 trials.

Inputs are the fixed manifest, source/protocol identity, and completed trials.
Outputs are a typed format-10 cursor. This module does not read files,
train models, or release final roles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Protocol

from src.app.comparison_scope import (
    NUMPY_BACKPROP_ALGORITHM_ID,
    NUMPY_CIRCADIAN_ALGORITHM_ID,
    NUMPY_PC_ALGORITHM_ID,
)
from src.app.continual_checkpoint import continual_config_digest
from src.app.continual_trigger_replay_outcomes import TRIGGER_REPLAY_OUTCOMES_PROTOCOL
from src.app.continual_trigger_replay_runner import (
    TRIGGER_REPLAY_TRAINING_PROTOCOL,
    TriggerReplayTrainingResult,
)
from src.app.continual_trigger_replay_schedule import (
    TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL,
    TriggerReplayOpportunityManifest,
    TriggerReplayScheduleSession,
    _validate_manifest,
)
from src.app.continual_trigger_replay_training_study import (
    TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
    TriggerReplayTrainingStudy,
    _validate_matched_seed,
    _validate_matched_seed_prefix,
    _validate_roles,
    _validate_trial,
)
from src.app.wake_diagnostic import validate_wake_diagnostic_trials


V14_TRIAL_CHECKPOINT_FORMAT = 10


@dataclass(frozen=True)
class V14TrialPrefixCheckpoint:
    """Completed Cartesian prefix; the next trial starts from its beginning."""

    format_version: int
    source_workspace_sha256: str
    config_sha256: str
    protocol_sha256: str
    capture_wake_diagnostics: bool
    next_trial_index: int
    next_cell: tuple[int, str] | None
    trials: tuple[TriggerReplayTrainingResult, ...]


class V14TrialCheckpointStore(Protocol):
    """App port for an immutable trusted local format-10 file."""

    def load(self) -> V14TrialPrefixCheckpoint: ...

    def save(self, checkpoint: V14TrialPrefixCheckpoint) -> None: ...


def v14_resume_protocol_sha256() -> str:
    """Bind the code-level v14 protocol and algorithm identifiers."""
    values = {
        "checkpoint_format": V14_TRIAL_CHECKPOINT_FORMAT,
        "source": TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL,
        "trial": TRIGGER_REPLAY_TRAINING_PROTOCOL,
        "training": TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
        "outcomes": TRIGGER_REPLAY_OUTCOMES_PROTOCOL,
        "backprop": NUMPY_BACKPROP_ALGORITHM_ID,
        "predictive_coding": NUMPY_PC_ALGORITHM_ID,
        "circadian_predictive_coding": NUMPY_CIRCADIAN_ALGORITHM_ID,
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def expected_v14_cells(manifest: TriggerReplayOpportunityManifest) -> tuple[tuple[int, str], ...]:
    _validate_manifest(manifest)
    return tuple((seed, arm) for seed in manifest.seeds for arm in manifest.arms)


def build_v14_trial_checkpoint(
    manifest: TriggerReplayOpportunityManifest,
    source_workspace_sha256: str,
    capture_wake_diagnostics: bool,
    trials: tuple[TriggerReplayTrainingResult, ...],
) -> V14TrialPrefixCheckpoint:
    """Construct and preflight a committed prefix before persistence."""
    cells = expected_v14_cells(manifest)
    index = len(trials)
    # Why this: the arrived role carries a deferred source object. It must
    # not bring final-test arrays into an unscored checkpoint on disk.
    sealed = tuple(_without_deferred_sources(trial) for trial in trials)
    checkpoint = V14TrialPrefixCheckpoint(
        V14_TRIAL_CHECKPOINT_FORMAT,
        source_workspace_sha256,
        continual_config_digest(manifest, manifest.seeds),
        v14_resume_protocol_sha256(),
        capture_wake_diagnostics,
        index,
        cells[index] if index < len(cells) else None,
        sealed,
    )
    validate_v14_trial_checkpoint(
        checkpoint, manifest, source_workspace_sha256, capture_wake_diagnostics
    )
    return checkpoint


def validate_v14_trial_checkpoint(
    checkpoint: V14TrialPrefixCheckpoint,
    manifest: TriggerReplayOpportunityManifest,
    source_workspace_sha256: str,
    capture_wake_diagnostics: bool,
) -> None:
    """Reject identity drift and malformed facts before further training."""
    cells = expected_v14_cells(manifest)
    if (
        type(checkpoint) is not V14TrialPrefixCheckpoint
        or type(source_workspace_sha256) is not str
        or len(source_workspace_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_workspace_sha256)
        or checkpoint.format_version != V14_TRIAL_CHECKPOINT_FORMAT
        or checkpoint.source_workspace_sha256 != source_workspace_sha256
        or checkpoint.config_sha256 != continual_config_digest(manifest, manifest.seeds)
        or checkpoint.protocol_sha256 != v14_resume_protocol_sha256()
        or type(capture_wake_diagnostics) is not bool
        or checkpoint.capture_wake_diagnostics is not capture_wake_diagnostics
    ):
        raise ValueError("v14 checkpoint source, config, protocol, or capture identity differs")
    index = checkpoint.next_trial_index
    if (
        type(index) is not int
        or not 1 <= index <= len(cells)
        or type(checkpoint.trials) is not tuple
        or len(checkpoint.trials) != index
        or checkpoint.next_cell != (cells[index] if index < len(cells) else None)
        or tuple((trial.seed, trial.arm) for trial in checkpoint.trials) != cells[:index]
    ):
        raise ValueError("v14 checkpoint trial prefix or next cell differs")
    for trial in checkpoint.trials:
        if trial.pending.phase_a._source is not None or trial.pending.phase_b._source is not None:
            raise ValueError("v14 checkpoint contains a deferred final source")
        _validate_trial(manifest, trial)
        if bool(trial.wake_diagnostics) is not capture_wake_diagnostics:
            raise ValueError("v14 checkpoint wake capture mode differs")
    for seed in manifest.seeds:
        matched = tuple(trial for trial in checkpoint.trials if trial.seed == seed)
        if len(matched) == len(manifest.arms):
            _validate_matched_seed(matched)
        elif matched:
            _validate_matched_seed_prefix(matched)
    if capture_wake_diagnostics:
        validate_wake_diagnostic_trials(checkpoint.trials)


def study_from_v14_trial_checkpoint(
    checkpoint: V14TrialPrefixCheckpoint, manifest: TriggerReplayOpportunityManifest
) -> TriggerReplayTrainingStudy:
    """Build only a full unscored study for the existing global gate."""
    if checkpoint.next_trial_index != len(expected_v14_cells(manifest)):
        raise ValueError("v14 checkpoint does not contain all unscored trials")
    # Recreate fixed sources only after the full unscored prefix has passed.
    # The caller still runs the global preflight before any final release.
    rebound = tuple(_rebind_deferred_sources(manifest, trial) for trial in checkpoint.trials)
    return TriggerReplayTrainingStudy(
        TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
        manifest,
        checkpoint.config_sha256,
        rebound,
    )


def _without_deferred_sources(trial: TriggerReplayTrainingResult) -> TriggerReplayTrainingResult:
    pending = trial.pending
    return replace(
        trial,
        pending=replace(
            pending,
            phase_a=replace(pending.phase_a, _source=None),
            phase_b=replace(pending.phase_b, _source=None),
        ),
    )


def _rebind_deferred_sources(
    manifest: TriggerReplayOpportunityManifest, trial: TriggerReplayTrainingResult
) -> TriggerReplayTrainingResult:
    session = TriggerReplayScheduleSession(manifest, seed=trial.seed)
    phase_a = session._roles
    _validate_roles(trial.pending.phase_a, phase_a)
    for _ in range(manifest.arrived.training.phase_a_epochs):
        session.complete_wake_epoch(manifest, source_role="train")
    session.arrive_phase_b(manifest)
    phase_b = session._roles
    _validate_roles(trial.pending.phase_b, phase_b)
    pending = trial.pending
    return replace(
        trial,
        pending=replace(
            pending,
            phase_a=replace(pending.phase_a, _source=phase_a._source),
            phase_b=replace(pending.phase_b, _source=phase_b._source),
        ),
    )
