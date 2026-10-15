"""Binary backpropagation matched to the coupled learner's feedforward network.

Finite float64 parameters/batches use the same tanh, logit BCE and sigmoid as
coupled_energy. This baseline differentiates through every feedforward hidden
state. It has no latent inference, precisions, chemistry, RNG, or runtime policy.
Snapshots own their arrays; publication is failure atomic for a single owner.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.core.coupled_energy import (
    Array,
    CoupledParameters,
    _finite,
    _matrix,
    _sigmoid,
    _validate_features,
    _validate_parameters,
    feedforward_latents,
    predict_coupled,
)

MATCHED_BACKPROP_LOSS_ID = "numpy_matched_binary_bce_preupdate_v1"


@dataclass(frozen=True)
class MatchedBackpropEvaluation:
    loss: float
    parameter_gradients: CoupledParameters
    loss_definition: str = MATCHED_BACKPROP_LOSS_ID


@dataclass(frozen=True)
class MatchedBackpropSnapshot:
    """Complete trusted local model state, with no training or serving authority."""

    format_version: int
    parameters: CoupledParameters


def _copy_parameters(parameters: CoupledParameters) -> CoupledParameters:
    _validate_parameters(parameters)
    return CoupledParameters(
        tuple(weight.copy() for weight in parameters.hidden_weights),
        tuple(bias.copy() for bias in parameters.hidden_biases),
        parameters.output_weight.copy(),
        parameters.output_bias.copy(),
    )


def _evaluate_feedforward(
    parameters: CoupledParameters, features: Array, targets: Array
) -> MatchedBackpropEvaluation:
    latents = feedforward_latents(parameters, features)
    logits = latents[-1] @ parameters.output_weight + parameters.output_bias
    _finite((logits,))
    loss = float(np.mean(np.logaddexp(0, logits) - targets * logits))
    if not isfinite(loss):
        raise FloatingPointError("nonfinite matched backpropagation loss")
    output_delta = (_sigmoid(logits) - targets) / len(features)
    delta = output_delta @ parameters.output_weight.T
    hidden_deltas = []
    for index in range(len(latents) - 1, -1, -1):
        delta = delta * (1 - latents[index] ** 2)
        _finite((delta,))
        hidden_deltas.append(delta)
        if index:
            delta = delta @ parameters.hidden_weights[index].T
    hidden_deltas.reverse()
    gradients = CoupledParameters(
        tuple(
            lower.T @ hidden_delta
            for lower, hidden_delta in zip((features, *latents[:-1]), hidden_deltas)
        ),
        tuple(np.sum(hidden_delta, axis=0, keepdims=True) for hidden_delta in hidden_deltas),
        latents[-1].T @ output_delta,
        np.sum(output_delta, axis=0, keepdims=True),
    )
    _finite(
        (
            *gradients.hidden_weights,
            *gradients.hidden_biases,
            gradients.output_weight,
            gradients.output_bias,
        )
    )
    return MatchedBackpropEvaluation(loss, gradients)


def evaluate_matched_backprop(
    parameters: CoupledParameters, features: Array, targets: Array
) -> MatchedBackpropEvaluation:
    """Evaluate mean logit BCE and its complete feedforward parameter gradient.

    Inputs are borrowed and quiescent. Targets have shape (batch, 1) and lie in
    [0, 1]. All returned gradient arrays are detached and include the batch mean.
    """
    _validate_features(parameters, features)
    _matrix(targets, "targets")
    if targets.shape != (len(features), 1) or np.any((targets < 0) | (targets > 1)):
        raise ValueError("targets must have shape (batch, 1) and values in [0, 1]")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return _evaluate_feedforward(parameters, features, targets)


def _learning_rate(value: float) -> float:
    if type(value) not in (float, int):
        raise ValueError("learning_rate must be a native positive finite number")
    try:
        rate = float(value)
    except OverflowError:
        raise ValueError("learning_rate must be finite") from None
    if not isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    return rate


def _updated_parameters(
    parameters: CoupledParameters, gradients: CoupledParameters, rate: float
) -> CoupledParameters:
    return CoupledParameters(
        tuple(w - rate * g for w, g in zip(parameters.hidden_weights, gradients.hidden_weights)),
        tuple(b - rate * g for b, g in zip(parameters.hidden_biases, gradients.hidden_biases)),
        parameters.output_weight - rate * gradients.output_weight,
        parameters.output_bias - rate * gradients.output_bias,
    )


class MatchedBackpropNetwork:
    """Own a fixed topology and perform simultaneous full-batch gradient descent.

    Initialization consumes explicit coupled parameters without making random
    draws. Full snapshots restore model tensors only; rates are supplied per call.
    """

    def __init__(self, parameters: CoupledParameters) -> None:
        self._parameters = _copy_parameters(parameters)
        self._input_dim = self._parameters.hidden_weights[0].shape[0]
        self._hidden_dims = tuple(weight.shape[1] for weight in self._parameters.hidden_weights)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self._hidden_dims

    def snapshot_parameters(self) -> CoupledParameters:
        return _copy_parameters(self._parameters)

    def _compatible_parameters(self, parameters: CoupledParameters) -> CoupledParameters:
        candidate = _copy_parameters(parameters)
        if (
            candidate.hidden_weights[0].shape[0] != self.input_dim
            or tuple(weight.shape[1] for weight in candidate.hidden_weights) != self.hidden_dims
        ):
            raise ValueError("parameter topology differs from this model")
        return candidate

    def restore_parameters(self, parameters: CoupledParameters) -> None:
        """Validate exact topology and detach every array before publication."""
        candidate = self._compatible_parameters(parameters)
        self._parameters = candidate

    def snapshot_state(self) -> MatchedBackpropSnapshot:
        return MatchedBackpropSnapshot(1, self.snapshot_parameters())

    def restore_state(self, snapshot: MatchedBackpropSnapshot) -> None:
        if (
            type(snapshot) is not MatchedBackpropSnapshot
            or type(snapshot.format_version) is not int
            or snapshot.format_version != 1
        ):
            raise ValueError("unsupported matched backpropagation snapshot format")
        candidate = self._compatible_parameters(snapshot.parameters)
        self._parameters = candidate

    def predict_proba(self, X: Array) -> Array:
        """Return the same target-free feedforward probabilities as coupled PC."""
        return predict_coupled(self._parameters, X)

    def train_epoch(self, X: Array, y: Array, learning_rate: float) -> MatchedBackpropEvaluation:
        """Return preupdate loss/gradients after validating the complete update."""
        rate = _learning_rate(learning_rate)
        evaluation = evaluate_matched_backprop(self._parameters, X, y)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            candidate = _updated_parameters(self._parameters, evaluation.parameter_gradients, rate)
        _validate_parameters(candidate)
        # Why evaluate again: finite tensors alone do not guarantee finite logits,
        # loss, or gradients. No model state is published until all are usable.
        evaluate_matched_backprop(candidate, X, y)
        self._parameters = candidate
        return evaluation
