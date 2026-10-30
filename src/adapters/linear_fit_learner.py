"""Atomic current-packet fixed LS through the existing ordinary learner port.

Caller owns admitted TRAIN roles, label release, budgets and permanent attempts.
Default fits replace the whole model from this packet. Explicit minimum-change
fits also use prior coefficients; no stored rows/replay,
optimizer, controller, retained rows, fit selection or stream restoration authority.
Use under the existing exclusive runtime lease; this is not a lock manager.
"""

import random

import numpy as np

from src.core.categorical_linear_fit import (
    _features,
    fit_categorical_linear,
    update_categorical_linear,
)
from src.core.categorical_linear_state import (
    MAX_COUNT,
    LinearFitPolicy,
    LinearFitSnapshot,
    copy_linear_policy,
    copy_linear_snapshot,
    linear_state_logits,
    predict_linear_state,
)
from src.core.categorical_probability_scores import validate_class_targets
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed

LINEAR_RESIDUAL_ID = "current_packet_one_hot_least_squares_sum_squared_residual_v1"
MINIMUM_CHANGE_RESIDUAL_ID = "minimum_change_one_hot_least_squares_sum_squared_residual_v1"


def _rng_signature():
    state = np.random.get_state()
    assert isinstance(state, tuple)
    return (
        (state[0], state[1].dtype.str, state[1].shape, state[1].tobytes(), *state[2:]),
        random.getstate(),
    )


def _same_array(left: Array, right: Array) -> bool:
    return (
        type(left) is np.ndarray
        and left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left, right)
    )


class LinearFitLearner:
    def __init__(self, policy: LinearFitPolicy):
        self._policy = copy_linear_policy(policy)
        self._state = LinearFitSnapshot(self._policy, None, 0, 0)

    @property
    def policy(self) -> LinearFitPolicy:
        return copy_linear_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> LinearFitSnapshot:
        state = copy_linear_snapshot(self._state)
        if state.policy != self.policy:
            raise ValueError("linear source state differs from bound policy")
        return state

    def restore_state(self, snapshot: LinearFitSnapshot) -> None:
        state = copy_linear_snapshot(snapshot)
        if state.policy != self.policy:
            raise ValueError("linear snapshot differs from bound policy")
        self._state = state

    def fork(self) -> "LinearFitLearner":
        child = LinearFitLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        return self._train_batch(features, targets, minimum_change=False)

    def train_batch_minimum_change(self, features: Array, targets: Array) -> TrainingDiagnostic:
        """Explicit nonsticky operation; caller still owns roles/budgets/lease.

        An unfitted owner uses the same cold solve. Subsequent calls preserve
        only prior components invisible to the new rows, not all old predictions.
        """
        return self._train_batch(features, targets, minimum_change=True)

    def _train_batch(
        self, features: Array, targets: Array, *, minimum_change: bool
    ) -> TrainingDiagnostic:
        before = self.snapshot_state()
        # Validate shape/targets before copying or paying the native solver.
        _features(features, self.policy.feature_dim)
        validate_class_targets(targets, len(features), self.policy.num_classes)
        if (
            before.completed_updates == MAX_COUNT
            or before.training_examples + len(features) > MAX_COUNT
        ):
            raise ValueError("linear committed counters would overflow")
        owners = (id(self._policy), id(self._state))
        rng = _rng_signature()
        originals = (features.copy(), targets.copy())
        staged = (features.copy(), targets.copy())
        rows = len(originals[0])
        # Why: never expose the comparison pin to a solver callback. A separate
        # full staged prior is checked after even a successful warm solve.
        prior = (
            copy_linear_snapshot(before) if minimum_change and before.model is not None else None
        )
        definition = MINIMUM_CHANGE_RESIDUAL_ID if minimum_change else LINEAR_RESIDUAL_ID
        returned = 0
        try:
            if prior is None:
                model = fit_categorical_linear(*staged, self.policy.num_classes)
            else:
                assert prior.model is not None
                model = update_categorical_linear(prior.model, *staged)
            returned = 1
            if model.fit_rows != rows:
                raise ValueError("linear fit returned wrong packet row count")
            after = LinearFitSnapshot(
                before.policy,
                model,
                before.completed_updates + 1,
                before.training_examples + rows,
            )
            after = copy_linear_snapshot(after)
            diagnostic = TrainingDiagnostic(definition, model.squared_error)
            unchanged = (
                owners == (id(self._policy), id(self._state))
                and self.snapshot_state() == before
                and _rng_signature() == rng
                and (prior is None or copy_linear_snapshot(prior) == before)
                and all(
                    _same_array(a, b)
                    for pair in ((features, targets), staged)
                    for a, b in zip(pair, originals, strict=True)
                )
            )
            if not unchanged:
                raise ValueError("linear donor, inputs or RNG changed")
        except Exception as error:
            # The ordinary runner can preserve paid but unpublished work without
            # adapter imports. This receipt owns scalars, never a model/history.
            raise TrainingUpdateFailed(
                f"{type(error).__name__}: {error}"[:512],
                definition,
                dict(
                    fit_attempts=1,
                    fit_rows_attempted=rows,
                    fit_returns=returned,
                    fit_rows_returned=returned * rows,
                    candidate_commits=0,
                    unreturned_work_unknown=returned == 0,
                    unpublished_work=True,
                ),
            ) from error
        # One validated owned replacement, including counters. Foreign mutations
        # are refused, never hidden by restoring an old source/input/RNG snapshot.
        self._state = after
        return diagnostic

    def predict(self, features: Array) -> Array:
        return predict_linear_state(self.snapshot_state(), features)

    def predict_logits(self, features: Array) -> Array:
        return linear_state_logits(self.snapshot_state(), features)
