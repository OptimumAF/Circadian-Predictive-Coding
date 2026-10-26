"""Readonly ownership and public-cue inference for two already fitted heads.

Inputs are complete CategoricalLinearFit models and raw public296 histories.
Snapshots have no native update/optimizer/clock authority. Immutable coefficient
tuples may be shared; model wrappers and owner bindings are detached. No fitting,
environment, IO, selection, promotion, labels or training belongs here.
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.core.categorical_linear_fit import CategoricalLinearFit, predict_categorical_linear
from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID, context_public_history


@dataclass(frozen=True, slots=True)
class ContextLinearPolicy:
    feature_dim: int = 296
    num_classes: int = 7
    encoding_id: str = CONTEXT_HISTORY_ENCODING_ID

    def __post_init__(self):
        if (
            type(self) is not ContextLinearPolicy
            or type(self.feature_dim) is not int
            or self.feature_dim != 296
            or type(self.num_classes) is not int
            or self.num_classes != 7
            or type(self.encoding_id) is not str
            or self.encoding_id != CONTEXT_HISTORY_ENCODING_ID
        ):
            raise ValueError("fixed public-cue296/K7 affine policy required")


@dataclass(frozen=True, slots=True)
class ContextLinearSnapshot:
    policy: ContextLinearPolicy
    heads: tuple[CategoricalLinearFit, CategoricalLinearFit]


def copy_context_linear_snapshot(state):
    if (
        type(state) is not ContextLinearSnapshot
        or type(state.policy) is not ContextLinearPolicy
        or type(state.heads) is not tuple
        or len(state.heads) != 2
        or any(type(head) is not CategoricalLinearFit for head in state.heads)
    ):
        raise ValueError("complete immutable two-head affine snapshot required")
    policy = replace(state.policy)
    # Why: independent wrappers detect field replacement across owners; immutable
    # tuple/float leaves need no new numeric allocation or optimizer history.
    heads = tuple(replace(head) for head in state.heads)
    if any((h.feature_dim, h.num_classes, h.fit_rows) != (296, 7, 27) for h in heads):
        raise ValueError("affine game heads require the qualified27-row296/K7 geometry")
    return ContextLinearSnapshot(policy, (heads[0], heads[1]))


def snapshot_from_context_fit_report(report: dict[str, Any]):
    """Convert only the complete full-fit report at the trusted CLI boundary."""
    if (
        type(report) is not dict
        or report.get("definition") != "minigrid_public_cue_linear_train_fit_v1"
        or report.get("status") != "complete"
        or report.get("role") != "train"
        or report.get("fit_witness") is not True
        or report.get("source_unchanged") is not True
        or set(report.get("heads", {})) != {"blue", "red"}
        or report["work"]["fit_calls_completed"] != 2
        or report["work"]["fit_rows_completed"] != 54
        or report["work"]["failure_work_may_be_unobserved"] is not False
    ):
        raise ValueError("complete qualified TRAIN affine fit required before games")
    heads = []
    for cue in ("blue", "red"):
        head = report["heads"][cue]
        if head["status"] != "complete" or head["source_unchanged"] is not True:
            raise ValueError("both affine heads must have completed")
        fields = dict(head["fit"])
        fields["coefficients"] = tuple(tuple(row) for row in fields["coefficients"])
        fields["singular_values"] = tuple(fields["singular_values"])
        heads.append(CategoricalLinearFit(**fields))
    return copy_context_linear_snapshot(
        ContextLinearSnapshot(ContextLinearPolicy(), (heads[0], heads[1]))
    )


class ContextLinearReader:
    """Frozen fitted binding with ordinary snapshot/fork/predict ports only."""

    def __init__(self, state):
        self._state = copy_context_linear_snapshot(state)
        self._binding = copy_context_linear_snapshot(self._state)

    def snapshot_state(self):
        owned = copy_context_linear_snapshot(self._state)
        if owned != self._binding:
            raise ValueError("readonly affine fitted binding changed")
        return owned

    def restore_state(self, state):
        owned = copy_context_linear_snapshot(state)
        if owned != self.snapshot_state():
            raise ValueError("readonly affine reader refuses a different fitted binding")
        self._state = owned

    def fork(self):
        return ContextLinearReader(self.snapshot_state())

    def predict(self, features):
        before = self.snapshot_state()
        if (
            type(features) is not np.ndarray
            or features.dtype != np.dtype(np.float64)
            or features.ndim != 2
            or features.shape[1] != 296
            or not 1 <= len(features) <= 128
        ):
            raise ValueError("affine reader requires exact float64(B1..128,296)")
        inputs = features.copy()
        projected = np.array(
            [context_public_history(tuple(float(v) for v in row)) for row in inputs],
            dtype=np.float64,
        )
        result = np.empty((len(inputs), 7), dtype=np.float64)
        for index, head in enumerate(before.heads):
            selected = projected[:, 296 + index] == 1
            if selected.any():
                values = predict_categorical_linear(head, projected[selected, :296])
                validate_class_probabilities(values, 7)
                if values.shape != (int(selected.sum()), 7):
                    raise ValueError("affine head returned a different row count")
                result[selected] = values
                if (
                    type(features) is not np.ndarray
                    or features.dtype != inputs.dtype
                    or features.shape != inputs.shape
                    or not np.array_equal(features, inputs)
                    or self.snapshot_state() != before
                ):
                    raise ValueError("affine prediction input or fitted binding changed")
        return result
