"""Owned affine wake updates on arrived public296 TRAIN batches.

The app owns roles, delayed labels and lifetime budgets. This port owns two
optional fitted heads and native fit/row counters. One homogeneous batch replaces
one head; no training rows, optimizer, RNG, replay or success chain is retained.
An unfitted head predicts uniform probabilities. This is squared-error fitting,
not gradient descent or cross-entropy optimization.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
from src.core.categorical_linear_fit import (
    CategoricalLinearFit,
    fit_categorical_linear,
    predict_categorical_linear,
)
from src.core.categorical_probability_scores import (
    validate_class_probabilities,
    validate_class_targets,
)
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.minigrid_context_encoding import context_public_history

DIAGNOSTIC = "arrived_public_cue_one_hot_least_squares_squared_error_v1"
FailureWork = tuple[int, int, int, int, bool]


@dataclass(frozen=True, slots=True)
class ContextLinearTrainingSnapshot:
    policy: ContextLinearPolicy
    heads: tuple[CategoricalLinearFit | None, CategoricalLinearFit | None] = (None, None)
    completed_updates: int = 0
    training_examples: int = 0
    failed_work: FailureWork | None = None


def copy_context_linear_training_snapshot(state):
    if (
        type(state) is not ContextLinearTrainingSnapshot
        or type(state.policy) is not ContextLinearPolicy
        or type(state.heads) is not tuple
        or len(state.heads) != 2
    ):
        raise ValueError("complete exact affine TRAIN snapshot required")
    policy = replace(state.policy)
    heads = []
    for head in state.heads:
        if head is not None and type(head) is not CategoricalLinearFit:
            raise ValueError("affine TRAIN heads must be exact fits or unfitted None")
        owned = None if head is None else replace(head)
        if owned is not None and (owned.feature_dim, owned.num_classes) != (296, 7):
            raise ValueError("affine TRAIN heads require296/K7 geometry")
        heads.append(owned)
    updates, rows = state.completed_updates, state.training_examples
    fitted = sum(head is not None for head in heads)
    if (
        type(updates) is not int
        or type(rows) is not int
        or not 0 <= updates <= rows <= min(128 * updates, 2**53 - 129)
        or (updates == 0) != (fitted == 0)
        or fitted > updates
        or sum(h.fit_rows for h in heads if h is not None) > rows
    ):
        raise ValueError("affine TRAIN fit/row counters disagree with fitted heads")
    work = state.failed_work
    if work is not None and (
        type(work) is not tuple
        or len(work) != 5
        or any(type(v) is not int for v in work[:4])
        or work[0] != 1
        or not 1 <= work[1] <= 128
        or work[2] not in (0, 1)
        or work[3] != work[1] * work[2]
        or work[4] is not True
    ):
        raise ValueError("affine TRAIN failed-work evidence is malformed")
    return ContextLinearTrainingSnapshot(policy, (heads[0], heads[1]), updates, rows, work)


def _project(features):
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype(np.float64)
        or features.ndim != 2
        or features.shape[1] != 296
        or not 1 <= len(features) <= 128
    ):
        raise ValueError("affine TRAIN port requires exact float64(B1..128,296)")
    inputs = features.copy()
    projected = np.array(
        [context_public_history(tuple(float(v) for v in row)) for row in inputs],
        dtype=np.float64,
    )
    return inputs, projected


def _unchanged(actual, expected):
    return (
        type(actual) is np.ndarray
        and actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and np.array_equal(actual, expected)
    )


def _failure(work: FailureWork):
    return dict(
        fit_calls_attempted=work[0],
        fit_rows_attempted=work[1],
        fit_calls_completed=work[2],
        fit_rows_completed=work[3],
        failure_work_may_be_unobserved=work[4],
    )


class ContextLinearLearner:
    def __init__(self):
        self._state = ContextLinearTrainingSnapshot(ContextLinearPolicy())

    def snapshot_state(self):
        return copy_context_linear_training_snapshot(self._state)

    @property
    def completed_updates(self):
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self):
        return self.snapshot_state().training_examples

    def restore_state(self, state):
        if self.snapshot_state().failed_work is not None:
            raise ValueError("failed affine owner cannot erase paid work through restore")
        self._state = copy_context_linear_training_snapshot(state)

    def fork(self):
        child = ContextLinearLearner()
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features, targets):
        before = self.snapshot_state()
        if before.failed_work is not None:
            raise TrainingUpdateFailed(
                "affine owner already failed", DIAGNOSTIC, _failure(before.failed_work)
            )
        inputs, projected = _project(features)
        validate_class_targets(targets, len(inputs), 7)
        labels = targets.copy()
        index = int(projected[0, 297])
        if not np.all(projected[:, 296 + index] == 1):
            raise ValueError("one arrived affine update requires a homogeneous initial public cue")
        design, native_labels = projected[:, :296].copy(), labels.copy()
        expected_design = design.copy()
        completed = 0
        try:
            model = fit_categorical_linear(design, native_labels, 7)
            completed = 1
            model = replace(model)
            if (model.feature_dim, model.num_classes, model.fit_rows) != (296, 7, len(inputs)):
                raise ValueError("affine solver returned incompatible fit geometry")
            if (
                not _unchanged(design, expected_design)
                or not _unchanged(native_labels, labels)
                or not _unchanged(features, inputs)
                or not _unchanged(targets, labels)
                or self.snapshot_state() != before
            ):
                raise ValueError("affine fit inputs or preceding state changed")
            heads = list(before.heads)
            heads[index] = model
            after = copy_context_linear_training_snapshot(
                replace(
                    before,
                    heads=(heads[0], heads[1]),
                    completed_updates=before.completed_updates + 1,
                    training_examples=before.training_examples + len(inputs),
                )
            )
            diagnostic = TrainingDiagnostic(DIAGNOSTIC, model.squared_error)
        except Exception as error:
            # Why: a failed/unknown physical solve is paid evidence, not a
            # successful app update. Preserve the preceding model and latch it.
            work = (1, len(inputs), completed, completed * len(inputs), True)
            self._state = replace(before, failed_work=work)
            raise TrainingUpdateFailed(
                f"{type(error).__name__}: {error}"[:512], DIAGNOSTIC, _failure(work)
            ) from error
        self._state = after
        return diagnostic

    def predict(self, features):
        before = self.snapshot_state()
        inputs, projected = _project(features)
        result = np.full((len(inputs), 7), 1 / 7, dtype=np.float64)
        for index, head in enumerate(before.heads):
            selected = projected[:, 296 + index] == 1
            if head is not None and selected.any():
                values = predict_categorical_linear(head, projected[selected, :296])
                validate_class_probabilities(values, 7)
                if values.shape != (int(selected.sum()), 7):
                    raise ValueError("affine prediction returned incompatible rows")
                result[selected] = values
            if not _unchanged(features, inputs) or self.snapshot_state() != before:
                raise ValueError("affine prediction inputs or preceding state changed")
        return result
