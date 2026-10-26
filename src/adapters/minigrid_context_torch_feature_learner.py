"""Route raw296 initial/current public views into two existing owned neural heads.

Inputs: bounded public histories, hard K7 labels and complete owned snapshots.
Outputs: unchanged native diagnostics, probabilities/logits and full continuation.
No roles, budgets, task-name selection, clock, replay, IO or new neural objective.
"""

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from src.adapters.torch_feature_learner import (
    TorchFeatureLearner,
    TorchFeaturePolicy,
    TorchFeatureSnapshot,
    _policy,
    _snapshot,
)
from src.app.coupled_attribution import _same_value
from src.core.action_memory import _feature_keys
from src.core.categorical_probability_scores import (
    validate_class_probabilities,
    validate_class_targets,
)
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID, context_public_history

FAILURE_DEFINITION = "public_cue_native_feature_failed_work_v1"
# Operation0=train/1=predict, last head, stage entries/returns, native call/row
# entries/returns, unknown-work flag. One scalar failure, never a success chain.
FailureWork = tuple[int, int, int, int, int, int, int, int, bool]


def _record(value, kind):
    if type(value) is not kind or set(vars(value)) != {f.name for f in fields(kind)}:
        raise ValueError(f"an exact complete {kind.__name__} is required")


@dataclass(frozen=True)
class ContextTorchFeaturePolicy:
    native_policy: TorchFeaturePolicy
    encoding_id: str = CONTEXT_HISTORY_ENCODING_ID

    def __post_init__(self):
        if type(self.encoding_id) is not str or self.encoding_id != CONTEXT_HISTORY_ENCODING_ID:
            raise ValueError("context neural policy requires its public initial-cue encoding")
        native = _policy(self.native_policy)
        if native.feature_dim != 296 or native.num_classes != 7:
            raise ValueError("context neural heads require canonical296/K7 native geometry")
        object.__setattr__(self, "native_policy", native)

    @property
    def method(self):
        return "public_context_" + self.native_policy.method

    @property
    def feature_dim(self):
        return 296

    @property
    def num_classes(self):
        return 7

    @property
    def hidden_dim(self):
        return self.native_policy.hidden_dim


def copy_context_torch_feature_policy(policy):
    _record(policy, ContextTorchFeaturePolicy)
    return ContextTorchFeaturePolicy(policy.native_policy, policy.encoding_id)


@dataclass(frozen=True)
class ContextTorchFeatureSnapshot:
    policy: ContextTorchFeaturePolicy
    heads: tuple[TorchFeatureSnapshot, TorchFeatureSnapshot]
    failed_work: FailureWork | None = None

    @property
    def completed_updates(self):
        return sum(h.completed_updates for h in self.heads)

    @property
    def training_examples(self):
        return sum(h.training_examples for h in self.heads)


def _validate_failure(work):
    if work is None:
        return
    if (
        type(work) is not tuple
        or len(work) != 9
        or any(type(v) is not int for v in work[:8])
        or work[8] is not True
    ):
        raise ValueError("context neural failed-work record is malformed")
    operation, index, stages, staged, calls, rows, returned, returned_rows, _ = work
    maximum = 1 if operation == 0 else 2
    if (
        operation not in (0, 1)
        or index not in (0, 1)
        or not 0 <= staged <= stages <= maximum
        or not 0 <= returned <= calls <= staged
        or not 0 <= returned_rows <= rows <= 128
        or (calls == 0) != (rows == 0)
        or calls > rows
        or returned > returned_rows
        or (returned == 0 and returned_rows != 0)
        or (returned == calls and returned_rows != rows)
    ):
        raise ValueError("context neural failed-work counters disagree")


def copy_context_torch_feature_snapshot(state):
    _record(state, ContextTorchFeatureSnapshot)
    policy = copy_context_torch_feature_policy(state.policy)
    if type(state.heads) is not tuple or len(state.heads) != 2:
        raise ValueError("context neural snapshot requires two complete native heads")
    heads = tuple(_snapshot(h) for h in state.heads)
    if any(h.policy != policy.native_policy for h in heads):
        raise ValueError("context neural head policy differs from its declared common policy")
    if (
        max(sum(h.completed_updates for h in heads), sum(h.training_examples for h in heads))
        > 2**53 - 1
    ):
        raise ValueError("context neural cumulative counters exceed exact scalar bound")
    _validate_failure(state.failed_work)
    return ContextTorchFeatureSnapshot(policy, (heads[0], heads[1]), state.failed_work)


def _failure(work):
    return dict(
        prediction_operation=bool(work[0]),
        head_index=work[1],
        native_stages_attempted=work[2],
        native_stages_returned=work[3],
        native_calls_attempted=work[4],
        native_rows_attempted=work[5],
        native_calls_returned=work[6],
        native_rows_returned=work[7],
        failure_work_may_be_unobserved=work[8],
    )


def _project(features):
    rows = _feature_keys(features, 296)
    owned = features.copy()
    projected = np.array([context_public_history(row) for row in rows], dtype=np.float64)
    return owned, projected[:, :296].copy(), projected[:, 297].astype(np.int64)


def _stage(head, work):
    work[2] += 1
    native = TorchFeatureLearner(head.policy)
    native.restore_state(head)
    work[3] += 1
    return native


class ContextTorchFeatureLearner:
    """Full snapshot ownership; native staging costs remain explicit paid work."""

    def __init__(self, policy: ContextTorchFeaturePolicy):
        owned = copy_context_torch_feature_policy(policy)
        heads = tuple(TorchFeatureLearner(owned.native_policy).snapshot_state() for _ in range(2))
        self._state = copy_context_torch_feature_snapshot(
            ContextTorchFeatureSnapshot(owned, (heads[0], heads[1]))
        )

    @property
    def policy(self):
        return copy_context_torch_feature_policy(self._state.policy)

    @property
    def completed_updates(self):
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self):
        return self.snapshot_state().training_examples

    def snapshot_state(self):
        return copy_context_torch_feature_snapshot(self._state)

    def restore_state(self, state):
        if self.snapshot_state().failed_work is not None:
            raise ValueError("failed context neural owner cannot erase paid work through restore")
        owned = copy_context_torch_feature_snapshot(state)
        if owned.policy != self.policy:
            raise ValueError("context neural snapshot policy/encoding differs from owner")
        self._state = owned

    def fork(self):
        # Why: native snapshots already own every tensor/RNG/optimizer field.
        # Fork/restore copy those data; only actual native operations build heads.
        child = object.__new__(ContextTorchFeatureLearner)
        child._state = self.snapshot_state()
        return child

    def _ready(self):
        before = self.snapshot_state()
        if before.failed_work is not None:
            raise TrainingUpdateFailed(
                "context neural owner already failed",
                FAILURE_DEFINITION,
                _failure(before.failed_work),
            )
        return self._state, before

    def _guard(self, owner, before, features, inputs):
        if (
            self._state is not owner
            or not _same_value(self.snapshot_state(), before)
            or not _same_value(features, inputs)
        ):
            raise ValueError("context neural source record or original input changed")

    def _failed(self, before, work, error):
        # Preserve the detached known prefix even when an injected callback
        # corrupts this owner. Paid/unknown staged work cannot become a success.
        self._state = replace(before, failed_work=tuple(work))
        raise TrainingUpdateFailed(
            f"{type(error).__name__}: {error}"[:512],
            FAILURE_DEFINITION,
            _failure(self._state.failed_work),
        ) from error

    def train_batch(self, features, targets):
        owner, before = self._ready()
        inputs, design, cues = _project(features)
        if not np.all(cues == cues[0]):
            raise ValueError("one context neural update requires a homogeneous initial public cue")
        validate_class_targets(targets, len(inputs), 7)
        labels, expected_design = targets.copy(), design.copy()
        expected_labels = labels.copy()
        index = int(cues[0])
        work: list[Any] = [0, index, 0, 0, 0, 0, 0, 0, True]
        try:
            native = _stage(before.heads[index], work)
            self._guard(owner, before, features, inputs)
            work[4], work[5] = 1, len(inputs)
            result = native.train_batch(design, labels)
            work[6], work[7] = 1, len(inputs)
            self._guard(owner, before, features, inputs)
            after = native.snapshot_state()
            if (
                after.completed_updates != before.heads[index].completed_updates + 1
                or after.training_examples != before.heads[index].training_examples + len(inputs)
                or after.policy != before.policy.native_policy
                or type(result) is not TrainingDiagnostic
                or not _same_value(design, expected_design)
                or not _same_value(labels, expected_labels)
                or not _same_value(targets, expected_labels)
            ):
                raise ValueError(
                    "context neural native return/input/clock differs from one staged update"
                )
            heads = list(before.heads)
            heads[index] = after
            candidate = copy_context_torch_feature_snapshot(
                ContextTorchFeatureSnapshot(before.policy, (heads[0], heads[1]))
            )
            self._guard(owner, before, features, inputs)
            self._state = candidate
            return result
        except Exception as error:
            self._failed(before, work, error)

    def _predict(self, features, logits):
        owner, before = self._ready()
        inputs, design, cues = _project(features)
        result = np.empty((len(inputs), 7), dtype=np.float64)
        work: list[Any] = [1, 0, 0, 0, 0, 0, 0, 0, True]
        try:
            for index in (0, 1):
                selected = cues == index
                if not selected.any():
                    continue
                work[1] = index
                native = _stage(before.heads[index], work)
                rows = design[selected].copy()
                expected_rows = rows.copy()
                self._guard(owner, before, features, inputs)
                work[4] += 1
                work[5] += len(rows)
                values = native.predict_logits(rows) if logits else native.predict(rows)
                work[6] += 1
                work[7] += len(rows)
                self._guard(owner, before, features, inputs)
                if (
                    not _same_value(native.snapshot_state(), before.heads[index])
                    or not _same_value(rows, expected_rows)
                    or type(values) is not np.ndarray
                    or values.dtype != np.float64
                    or values.shape != (len(rows), 7)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        "context neural read changed native state/input or returned invalid values"
                    )
                if not logits:
                    validate_class_probabilities(values, 7)
                result[selected] = values
            self._guard(owner, before, features, inputs)
            return result
        except Exception as error:
            self._failed(before, work, error)

    def predict(self, features):
        return self._predict(features, False)

    def predict_logits(self, features):
        return self._predict(features, True)
