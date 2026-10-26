"""Read old TRAIN fit on the disposable probe learner without teaching it.

Admit at most two detached TRAIN batches alongside the original probe envelope.
Score returned predictions at the caller's existing checkpoints, with separate
partial-work accounting. This module supplies no updates, opaque-state checks,
task selection, budget renewal, persistence, or final-data authority.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _probability_scores, _validate_batch
from src.core.categorical_probability_scores import probability_scores
from src.core.probe_decision_diagnostics import decision_class_diagnostics


def validate_training_readout_envelope(training, evaluation, readouts) -> None:
    """Validate the full additional envelope before any owned copies or calls."""
    if type(readouts) is not tuple or len(readouts) > 2:
        raise ValueError("training_readouts must be an exact tuple of zero to two TRAIN batches")
    if not readouts:
        return
    original = (*training, *evaluation)
    arrays = [array for batch in original for array in (batch.features, batch.targets)]
    entries = sum(array.size for array in arrays)
    for batch in readouts:
        if (
            type(batch) is not AttributionBatch
            or type(batch.role) is not str
            or batch.role != "train"
            or type(batch.features) is not np.ndarray
            or batch.features.dtype != np.float64
            or batch.features.ndim != 2
            or not 1 <= batch.features.shape[0] <= 128
            or batch.features.shape[1] != training[0].features.shape[1]
            or type(batch.targets) is not np.ndarray
            or batch.targets.dtype != np.float64
            or batch.targets.shape != (batch.features.shape[0], 1)
            or batch.num_classes != training[0].num_classes
            or (batch.episode is None) != (training[0].episode is None)
        ):
            raise ValueError(
                "training_readouts require matching complete or metadata-free TRAIN batches"
            )
        entries += batch.features.size + batch.targets.size
        arrays.extend((batch.features, batch.targets))
    if entries > 65536:
        raise ValueError("probe and training_readouts exceed 65536 aggregate entries")
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    ):
        raise ValueError("all probe and training_readouts arrays require disjoint storage")


def validate_training_readouts(training, evaluation, readouts) -> None:
    """Validate values/episode binding after the entire array envelope is bounded."""
    seen = {row_id for batch in (*training, *evaluation) for row_id in batch.row_ids}
    for batch in readouts:
        _validate_batch(batch, "train", seen)


@dataclass
class TrainingReadoutProgress:
    score_forward_calls_attempted: int = 0
    score_forward_calls_completed: int = 0
    attempted_rows: int = 0
    completed_rows: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float | None = None
    failure_work_may_be_unobserved: bool = False


def score_training_readouts(
    learner,
    batches,
    updates,
    budget,
    elapsed,
    progress,
    *,
    decision_diagnostics=False,
    check_boundary=None,
) -> None:
    """Publish partial monitor work after the original F checkpoint is paid."""
    point: dict[str, Any] = dict(updates=updates, status="running", scores={})
    if decision_diagnostics:
        point["decision_diagnostics"] = {}
    progress.checkpoints.append(point)
    try:
        for index, batch in enumerate(batches):
            budget.before_final()
            started = elapsed.read()
            if progress.elapsed_seconds is None:
                progress.elapsed_seconds = 0.0
            if check_boundary is not None:
                check_boundary()
            progress.score_forward_calls_attempted += 1
            progress.attempted_rows += len(batch.features)
            progress.failure_work_may_be_unobserved = True
            prediction = learner.predict(batch.features.copy())
            # Native returns are paid even when the subsequent guard fails.
            progress.score_forward_calls_completed += 1
            progress.completed_rows += len(batch.features)
            if check_boundary is not None:
                check_boundary()
            key = f"batch-{index}" if batch.episode is None else batch.episode.episode_id
            point["scores"][key] = (
                _probability_scores(prediction, batch.targets)
                if batch.num_classes == 2
                else probability_scores(prediction, batch.targets, batch.num_classes)
            )
            if decision_diagnostics:
                detail = decision_class_diagnostics(prediction, batch.targets, batch.num_classes)
                detail.update(
                    row_ids=list(batch.row_ids),
                    episode_id=None if batch.episode is None else batch.episode.episode_id,
                    context=batch.context,
                    role=batch.role,
                    sequence_kind="batch_row_order"
                    if batch.episode is None
                    else "complete_episode",
                )
                point["decision_diagnostics"][key] = detail
            budget.before_final()
            progress.elapsed_seconds += elapsed.read() - started
            progress.failure_work_may_be_unobserved = False
        if check_boundary is not None:
            check_boundary()
        point["status"] = "complete"
    except Exception:
        point["status"] = "failed"
        raise
