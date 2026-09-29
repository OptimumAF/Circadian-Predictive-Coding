"""Resume v7 candidate training and selection through one run-level cursor.

Inputs are a predeclared candidate manifest and a trusted checkpoint port.
Outputs are unscored candidate states, complete outer trials, and a frozen
choice. This module does not read files or release final-test fields.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_arrived_selection as selection
from src.app import continual_shift_benchmark as base
from src.app.continual_arrived_checkpoint import (
    ARRIVED_CHECKPOINT_FORMAT,
    ArrivedRunnerCheckpoint,
    arrived_config_digest,
    validate_arrived_checkpoint_header,
)
from src.app.continual_arrived_selection_checkpoint import (
    ARRIVED_SELECTION_CHECKPOINT_FORMAT,
    ArrivedSelectionCheckpoint,
    ArrivedSelectionCheckpointStore,
    CompletedSelectionCandidate,
    validate_selection_checkpoint_header,
)


@dataclass
class _CandidateStoreAdapter:
    """Embed every v6 transaction in the current v7 candidate cursor."""

    store: ArrivedSelectionCheckpointStore
    checkpoint: ArrivedSelectionCheckpoint

    def load(self) -> ArrivedRunnerCheckpoint:
        if self.checkpoint.active_v6 is None:
            raise ValueError("v7 active candidate has no v6 transaction")
        return self.checkpoint.active_v6

    def save(self, active: ArrivedRunnerCheckpoint) -> None:
        self.checkpoint = replace(self.checkpoint, active_v6=deepcopy(active))
        self.store.save(self.checkpoint)


def run_or_resume_selection(
    candidates: tuple[selection.ArrivedSelectionCandidate, ...],
    seeds: list[int],
    store: ArrivedSelectionCheckpointStore,
    resume: bool,
) -> tuple[
    dict[tuple[str, int], arrived._PendingSeed],
    tuple[selection.ArrivedOuterTrial, ...],
    selection.ArrivedSelectionFreeze,
]:
    """Validate prior candidates before any update, then freeze before final use."""
    ordered_seeds = tuple(seeds)
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    candidate_manifest = tuple(
        (candidate.candidate_id, candidate.config) for candidate in candidates
    )
    manifest_digest = selection._candidate_manifest_digest(candidates, ordered_seeds)
    if resume:
        checkpoint = store.load()
        validate_selection_checkpoint_header(
            checkpoint,
            manifest_digest=manifest_digest,
            candidate_manifest=candidate_manifest,
            candidate_ids=candidate_ids,
            seeds=ordered_seeds,
        )
        _validate_embedded_headers(checkpoint, candidates, ordered_seeds)
    else:
        checkpoint = ArrivedSelectionCheckpoint(
            ARRIVED_SELECTION_CHECKPOINT_FORMAT,
            manifest_digest,
            candidate_manifest,
            candidate_ids,
            ordered_seeds,
            0,
            "training",
            (),
        )
        store.save(checkpoint)
    pending, trials = _restore_completed_candidates(checkpoint, candidates, ordered_seeds)
    if checkpoint.stage == "frozen":
        expected = selection._freeze_selection(candidates, ordered_seeds, tuple(trials))
        if (
            checkpoint.freeze != expected
            or checkpoint.freeze is None
            or checkpoint.freeze.sleep_history_digest != expected.sleep_history_digest
        ):
            raise ValueError("incompatible v7 frozen candidate choice")
        return pending, tuple(trials), expected
    checkpoint = _complete_remaining_candidates(
        checkpoint, candidates, seeds, store, pending, trials
    )
    selection._validate_shared_roles(candidates, seeds, pending)
    frozen = selection._freeze_selection(candidates, ordered_seeds, tuple(trials))
    store.save(replace(checkpoint, stage="frozen", freeze=frozen))
    return pending, tuple(trials), frozen


def _validate_embedded_headers(
    checkpoint: ArrivedSelectionCheckpoint,
    candidates: tuple[selection.ArrivedSelectionCandidate, ...],
    seeds: tuple[int, ...],
) -> None:
    for index, record in enumerate(checkpoint.completed_candidates):
        candidate = candidates[index]
        terminal = ArrivedRunnerCheckpoint(
            format_version=ARRIVED_CHECKPOINT_FORMAT,
            config_digest=arrived_config_digest(candidate.config, seeds),
            seeds=seeds,
            seed_index=len(seeds) - 1,
            phase="seed_complete",
            phase_epoch_completed=candidate.config.training.phase_b_epochs,
            stage="after_sleep",
            next_model_index=0,
            unscored_seeds=record.unscored_seeds,
        )
        _validate_v6_header(terminal, candidate.config, seeds)
    if checkpoint.active_v6 is not None:
        _validate_v6_header(
            checkpoint.active_v6, candidates[checkpoint.candidate_index].config, seeds
        )


def _validate_v6_header(
    checkpoint: ArrivedRunnerCheckpoint,
    config: arrived.ContinualArrivedRolesConfig,
    seeds: tuple[int, ...],
) -> None:
    validate_arrived_checkpoint_header(
        checkpoint,
        config_digest=arrived_config_digest(config, seeds),
        seeds=seeds,
        phase_a_epochs=config.training.phase_a_epochs,
        phase_b_epochs=config.training.phase_b_epochs,
        model_order=config.training.model_order,
    )


def _restore_completed_candidates(
    checkpoint: ArrivedSelectionCheckpoint,
    candidates: tuple[selection.ArrivedSelectionCandidate, ...],
    seeds: tuple[int, ...],
) -> tuple[dict[tuple[str, int], arrived._PendingSeed], list[selection.ArrivedOuterTrial]]:
    pending: dict[tuple[str, int], arrived._PendingSeed] = {}
    trials: list[selection.ArrivedOuterTrial] = []
    for index, record in enumerate(checkpoint.completed_candidates):
        candidate = candidates[index]
        rows = tuple(
            arrived._rehydrate_unscored_arrived_seed(saved, candidate.config, seed)
            for saved, seed in zip(record.unscored_seeds, seeds, strict=True)
        )
        expected_trials = _candidate_trials(candidate.candidate_id, rows)
        if (
            len(record.trials) != len(expected_trials)
            or any(
                type(saved) is not selection.ArrivedOuterTrial
                or vars(saved).get("sleep_events") != expected.sleep_events
                or vars(saved).get("sleep_history_digest") != expected.sleep_history_digest
                for saved, expected in zip(record.trials, expected_trials, strict=True)
            )
            or record.trials != expected_trials
            or record.trial_digest != selection._trial_digest(expected_trials)
            or record.sleep_history_digest != selection._sleep_history_digest(expected_trials)
        ):
            raise ValueError("incompatible v7 completed outer trials or exposure cursor")
        for row in rows:
            pending[candidate.candidate_id, row.seed] = row
        trials.extend(expected_trials)
    return pending, trials


def _complete_remaining_candidates(
    checkpoint: ArrivedSelectionCheckpoint,
    candidates: tuple[selection.ArrivedSelectionCandidate, ...],
    seeds: list[int],
    store: ArrivedSelectionCheckpointStore,
    pending: dict[tuple[str, int], arrived._PendingSeed],
    trials: list[selection.ArrivedOuterTrial],
) -> ArrivedSelectionCheckpoint:
    for index in range(checkpoint.candidate_index, len(candidates)):
        candidate = candidates[index]
        adapter = _CandidateStoreAdapter(store, checkpoint)
        rows = arrived._run_completed_seed_checkpoints(
            candidate.config, seeds, adapter, checkpoint.active_v6 is not None
        )
        terminal = adapter.checkpoint.active_v6
        if (
            terminal is None
            or terminal.phase != "seed_complete"
            or len(terminal.unscored_seeds) != len(seeds)
        ):
            raise ValueError("v7 candidate ended without a complete unscored transaction")
        candidate_trials = _candidate_trials(candidate.candidate_id, tuple(rows))
        record = CompletedSelectionCandidate(
            candidate_id=candidate.candidate_id,
            unscored_seeds=deepcopy(terminal.unscored_seeds),
            trials=candidate_trials,
            trial_digest=selection._trial_digest(candidate_trials),
            sleep_history_digest=selection._sleep_history_digest(candidate_trials),
        )
        for row in rows:
            pending[candidate.candidate_id, row.seed] = row
        trials.extend(candidate_trials)
        checkpoint = replace(
            adapter.checkpoint,
            candidate_index=index + 1,
            completed_candidates=adapter.checkpoint.completed_candidates + (record,),
            active_v6=None,
        )
        store.save(checkpoint)
    return checkpoint


def _candidate_trials(
    candidate_id: str, rows: tuple[arrived._PendingSeed, ...]
) -> tuple[selection.ArrivedOuterTrial, ...]:
    return tuple(
        selection._score_outer_trial(candidate_id, row, method)
        for row in rows
        for method in base.CONTINUAL_MODEL_ORDER
    )
