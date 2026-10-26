"""Frozen per-episode prediction from public initial/current local frames.

The session owns a native learner fork and one initial eight-coordinate frame.
Inputs are individual public float64 rows; outputs are owned five-action
probabilities. This adapter does not train, select actions, repair probabilities,
read labels/context metadata, run physics, or retain a sequence of past frames.
"""

from copy import deepcopy
from typing import Any, Callable

import numpy as np

from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.coupled_energy import Array
from src.core.key_door_public_history import PublicInitialFrameHistory


def _checked_policy(policy: Any) -> Any:
    if (
        type(getattr(policy, "feature_dim", None)) is not int
        or policy.feature_dim != 16
        or type(getattr(policy, "num_classes", None)) is not int
        or policy.num_classes != 5
        or type(getattr(policy, "method", None)) is not str
        or policy.method not in ("bp", "pc", "neutral_cpc", "gated_cpc", "observation_memory")
    ):
        raise ValueError("public history requires a matched 16-input five-action feature policy")
    return deepcopy(policy)


def _owned_predictor(
    predictor: Any, same_state: Callable[[Any, Any], bool]
) -> tuple[Any, Any, Any]:
    policy = _checked_policy(predictor.policy)
    frozen = deepcopy(predictor.snapshot_state())
    learner = predictor.fork()
    if learner is predictor:
        raise ValueError("public history session requires an independently owned native fork")

    def verify_source() -> None:
        if not same_state(predictor.snapshot_state(), frozen) or not same_state(
            _checked_policy(predictor.policy), policy
        ):
            raise ValueError("public history source state changed while creating its session")

    verify_source()
    learner.restore_state(deepcopy(frozen))
    verify_source()
    if not same_state(learner.snapshot_state(), frozen) or not same_state(
        _checked_policy(learner.policy), policy
    ):
        raise ValueError("public history fork did not restore the complete frozen native state")
    # A child readback is an external call too: close its source-mutation window
    # before the caller can admit the first public observation or prediction.
    verify_source()
    return learner, frozen, policy


def _public_row(features: Array) -> tuple[float, ...]:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.shape != (1, 8)
        or not np.isfinite(features).all()
        or not ((features >= -1) & (features <= 1)).all()
    ):
        raise ValueError("public history input must be exact finite float64 (1, 8) in [-1, 1]")
    return tuple(float(value) for value in features[0])


class PublicHistoryPredictionSession:
    """One frozen native owner, one initial frame, and at most twelve calls."""

    def __init__(
        self, predictor: Any, *, same_state: Callable[[Any, Any], bool] | None = None
    ) -> None:
        if same_state is not None and not callable(same_state):
            raise ValueError("same_state must be callable")
        self._same_state = _same_state if same_state is None else same_state
        self._learner, self._frozen, self._policy = _owned_predictor(predictor, self._same_state)
        self._history = PublicInitialFrameHistory()
        self._completed_steps = 0
        self._attempted_steps = 0
        self._encoded_frames = 0
        self._native_prediction_returns = 0
        self._failed = False

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def attempted_steps(self) -> int:
        return self._attempted_steps

    @property
    def encoded_frames(self) -> int:
        return self._encoded_frames

    @property
    def native_prediction_returns(self) -> int:
        return self._native_prediction_returns

    def snapshot_state(self) -> Any:
        """Read the actual full native state, including after a failed call."""
        try:
            current = self._learner.snapshot_state()
            if not self._same_state(current, self._frozen) or not self._same_state(
                _checked_policy(self._learner.policy), self._policy
            ):
                raise ValueError("public history session frozen native state changed")
            return deepcopy(current)
        except Exception:
            self._failed = True
            raise

    def _admit_call(self) -> None:
        counts = (
            self._completed_steps,
            self._native_prediction_returns,
            self._encoded_frames,
            self._attempted_steps,
        )
        if (
            self._failed
            or any(type(value) is not int for value in counts)
            or not 0 <= counts[0] <= counts[1] <= counts[2] <= counts[3] < 12
        ):
            self._failed = True
            raise ValueError("public history session stopped, exhausted, or counters invalid")
        self._attempted_steps += 1

    def predict(self, features: Array) -> Array:
        self._admit_call()
        try:
            frame = _public_row(features)
            self.snapshot_state()
            encoded = (
                self._history.reset(frame)
                if self._encoded_frames == 0
                else self._history.encode(frame)
            )
            values = np.array((encoded,), dtype=np.float64)
            self._encoded_frames += 1
            result = self._learner.predict(values)
            # Count an observed native return before validation or the postguard
            # can fail. Failed calls never erase already observed native work.
            self._native_prediction_returns += 1
            self.snapshot_state()
            validate_class_probabilities(result, 5)
            if result.shape != (1, 5):
                raise ValueError("public history native prediction must have exactly one row")
            owned: Array = result.copy()
            self._completed_steps += 1
            return owned
        except Exception:
            self._failed = True
            raise
