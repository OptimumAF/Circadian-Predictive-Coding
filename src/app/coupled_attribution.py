"""Train a bounded delayed stream, then score its detached model checkpoints.

Inputs are declared train/final-test batches, an owned-fork learner, fixed
checkpoint counts, and an execution budget. Labels from heldout batches are used
only by application metrics after all training completes. This module does not
promote candidates, tune policies, establish provenance, or release scientific
claims. Exceptions leave the invocation incomplete and propagate to the caller.
"""

from dataclasses import asdict, dataclass
from copy import deepcopy
from typing import Any, Callable

import numpy as np

from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.actor_ports import ForkableLearner
from src.core.coupled_energy import Array
from src.core.experience import (
    Experience,
    ExperiencePermissions,
    LabelArrival,
    LogicalClock,
    require_identifier,
)


@dataclass(frozen=True)
class AttributionBatch:
    role: str
    context: str
    row_ids: tuple[str, ...]
    features: Array
    targets: Array


def _identifier(value: str, name: str) -> None:
    require_identifier(value, name)


def _matrix(value: Array, name: str) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.ndim != 2
        or min(value.shape) < 1
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite float64 matrix")


def _validate_batch(batch: AttributionBatch, role: str, seen_rows: set[str]) -> None:
    if type(batch) is not AttributionBatch:
        raise ValueError("batches must contain exact AttributionBatch records")
    if type(batch.role) is not str or batch.role != role:
        raise ValueError(f"batch role must be {role}")
    _identifier(batch.context, "batch context")
    _matrix(batch.features, "batch features")
    _matrix(batch.targets, "batch targets")
    rows = len(batch.features)
    if role == "train" and rows > 1024:
        raise ValueError("training batches must contain at most 1024 rows")
    if batch.targets.shape != (rows, 1) or np.any((batch.targets < 0) | (batch.targets > 1)):
        raise ValueError("batch targets must have shape (batch, 1) and values in [0, 1]")
    if type(batch.row_ids) is not tuple or len(batch.row_ids) != rows:
        raise ValueError("row_ids must be a tuple matching the batch row count")
    for row_id in batch.row_ids:
        _identifier(row_id, "row ID")
        if row_id in seen_rows:
            raise ValueError("row IDs must be globally unique across training and heldout data")
        seen_rows.add(row_id)


def _copy_batch(batch: AttributionBatch) -> AttributionBatch:
    return AttributionBatch(
        batch.role, batch.context, batch.row_ids, batch.features.copy(), batch.targets.copy()
    )


def _validated_batches(
    training: tuple[AttributionBatch, ...], heldout: tuple[AttributionBatch, ...]
) -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    if type(training) is not tuple or not 1 <= len(training) <= 512:
        raise ValueError("training must be a tuple containing 1..512 batches")
    if type(heldout) is not tuple or not heldout:
        raise ValueError("heldout must be a nonempty tuple")
    seen_rows: set[str] = set()
    for batch in training:
        _validate_batch(batch, "train", seen_rows)
    for batch in heldout:
        _validate_batch(batch, "final_test", seen_rows)
    contexts = tuple(batch.context for batch in heldout)
    if len(contexts) != len(set(contexts)):
        raise ValueError("heldout contexts must be unique")
    if any(batch.context not in contexts for batch in training):
        raise ValueError("every training context must have a heldout context")
    width = training[0].features.shape[1]
    if any(batch.features.shape[1] != width for batch in (*training, *heldout)):
        raise ValueError("all training and heldout feature widths must match")
    for train_batch in training:
        for test_batch in heldout:
            if any(
                np.shares_memory(train_array, test_array)
                for train_array in (train_batch.features, train_batch.targets)
                for test_array in (test_batch.features, test_batch.targets)
            ):
                raise ValueError("training and heldout arrays must not share memory")
    # Copy before any external learner/budget call can mutate caller-owned data.
    return tuple(map(_copy_batch, training)), tuple(map(_copy_batch, heldout))


def _validate_controls(
    checkpoint_updates: tuple[int, ...], label_delay: int, budget: ToyBudgetSession, updates: int
) -> None:
    if (
        type(checkpoint_updates) is not tuple
        or not 2 <= len(checkpoint_updates) <= 32
        or any(type(point) is not int for point in checkpoint_updates)
        or checkpoint_updates[0] != 0
        or checkpoint_updates[-1] != updates
        or any(left >= right for left, right in zip(checkpoint_updates, checkpoint_updates[1:]))
    ):
        raise ValueError("checkpoints must be 2..32 strictly increasing native counts from 0 to n")
    if type(label_delay) is not int or not 0 <= label_delay <= 512:
        raise ValueError("label_delay must be a native integer in 0..512")
    if type(budget) is not ToyBudgetSession:
        raise ValueError("attribution stream requires an exact ToyBudgetSession")


def _record_tick(
    runtime: ActorShadowRuntime[Array, Array, Array, Any],
    training: tuple[AttributionBatch, ...],
    tick: int,
    delay: int,
) -> None:
    if tick < len(training):
        runtime.record_experience(
            Experience(
                f"batch-{tick}",
                "attribution",
                tick,
                "a0",
                training[tick].features,
                "train",
                ExperiencePermissions(training=True),
            )
        )
    label_index = tick - delay
    if 0 <= label_index < len(training):
        runtime.record_label(
            LabelArrival(
                f"label-{label_index}",
                f"batch-{label_index}",
                "attribution",
                tick,
                "a0",
                training[label_index].targets,
            )
        )


def _probability_scores(predictions: Array, targets: Array) -> dict[str, float]:
    _matrix(predictions, "heldout predictions")
    if predictions.shape != targets.shape or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("heldout predictions must have shape (batch, 1) and values in [0, 1]")
    clipped = np.clip(predictions, 1e-12, 1 - 1e-12)
    bce = -float(np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)))
    accuracy = float(np.mean((predictions >= 0.5) == (targets >= 0.5)))
    return {"bce": bce, "accuracy": accuracy}


def _score_checkpoints(
    learner: ForkableLearner[Array, Array, Array, Any],
    states: list[tuple[int, Any]],
    training: tuple[AttributionBatch, ...],
    heldout: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
) -> list[dict[str, Any]]:
    budget.before_final()
    scorer = learner.fork()
    if scorer is learner:
        raise ValueError("scorer fork must be distinct from the source learner")
    budget.before_final()
    checkpoints = []
    for updates, state in states:
        budget.before_final()
        scorer.restore_state(state)
        budget.before_final()
        scores = {}
        for batch in heldout:
            budget.before_final()
            predictions = scorer.predict(batch.features.copy())
            budget.before_final()
            scores[batch.context] = _probability_scores(predictions, batch.targets)
            budget.before_final()
        checkpoints.append(
            {
                "updates": updates,
                "training_context": None if updates == 0 else training[updates - 1].context,
                "scores": scores,
            }
        )
    return checkpoints


def run_attribution_stream(
    learner: ForkableLearner[Array, Array, Array, Any],
    training: tuple[AttributionBatch, ...],
    heldout: tuple[AttributionBatch, ...],
    *,
    checkpoint_updates: tuple[int, ...],
    label_delay: int,
    budget: ToyBudgetSession,
    observe_checkpoint: Callable[[int, Any, AttributionBatch | None], None] | None = None,
) -> dict[str, Any]:
    """Train all delayed batches, then score fixed detached checkpoint states.

    Checkpoint counts must include zero and the complete stream. Accuracy uses
    threshold 0.5 for predictions and targets; BCE always clips at 1e-12.
    Declared roles/row IDs and disjoint storage are checked without claiming
    physical provenance. Opaque model state never appears in the returned report.
    Optional checkpoint observers receive separate state copies and only the
    last arrived training batch; initial observation has no labels or batch.
    """
    training, heldout = _validated_batches(training, heldout)
    _validate_controls(checkpoint_updates, label_delay, budget, len(training))
    if observe_checkpoint is not None and not callable(observe_checkpoint):
        raise ValueError("observe_checkpoint must be callable or None")
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        learner,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=len(training),
        max_consolidations=0,
    )
    states = [(0, runtime.candidate_snapshot().state)]
    if observe_checkpoint is not None:
        budget.before_final()
        observe_checkpoint(0, deepcopy(states[0][1]), None)
        budget.before_final()
    completed = 0
    for tick in range(len(training) + label_delay):
        clock.advance_to(tick)
        _record_tick(runtime, training, tick, label_delay)
        completed += len(runtime.train_ready(max_updates=1))
        if completed in checkpoint_updates and completed != states[-1][0]:
            states.append((completed, runtime.candidate_snapshot().state))
            if observe_checkpoint is not None:
                budget.before_final()
                observe_checkpoint(
                    completed, deepcopy(states[-1][1]), _copy_batch(training[completed - 1])
                )
                budget.before_final()
    receipts = runtime.applied_updates
    if completed != len(training) or len(receipts) != len(training):
        raise RuntimeError("attribution stream did not commit exactly every training batch")
    if tuple(point for point, _ in states) != checkpoint_updates:
        raise RuntimeError("attribution stream did not capture every requested checkpoint")
    # Why defer every score: even intermediate checkpoints cannot expose heldout
    # outcomes to the subsequent training trajectory or its fixed controls.
    checkpoints = _score_checkpoints(learner, states, training, heldout, budget)
    budget.before_final()
    return {
        "completed_updates": completed,
        "training_examples": sum(len(batch.features) for batch in training),
        "score_forward_calls": len(states) * len(heldout),
        "checkpoints": checkpoints,
        "applied_updates": [asdict(receipt) for receipt in receipts],
    }
