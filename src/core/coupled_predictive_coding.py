"""Opt-in gradient descent on the versioned coupled binary PC energy.

Training borrows finite float64 batches and returns a pre-parameter-update energy.
Snapshots own their arrays. Initialization uses N(0, 1/sqrt(fan-in)) and zero
biases; paper scaling reproduction is outside this learner. No replay, sleep,
chemistry, runtime adapter or neutral CPC control is implemented here.

Example: model = CoupledPredictiveCodingNetwork(2, (4, 3), seed=0)
Then model.train_epoch(X, y, 0.01, 5, 0.02) performs one full-batch update.
"""

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from src.core.coupled_energy import (
    COUPLED_ENERGY_ID,
    Array,
    CoupledEvaluation,
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
    predict_coupled,
)
from src.core.coupled_inference import (
    CoupledInferenceDiagnostics,
    CoupledInferencePolicy,
    infer_coupled,
)


@dataclass(frozen=True)
class CoupledTrainResult:
    """Energy after latent inference, before the atomic parameter update."""

    energy: float
    energy_definition: str = COUPLED_ENERGY_ID
    inference: CoupledInferenceDiagnostics | None = None


@dataclass(frozen=True)
class CoupledWakeState:
    """Complete adaptive wake state and its fixed chemistry-policy binding."""

    chemical_decay: float
    chemical_buildup_rate: float
    plasticity_sensitivity: float
    min_plasticity: float
    chemical_state: Array
    plasticity_state: Array
    wake_updates: int


@dataclass(frozen=True)
class CoupledSnapshot:
    """Trusted local model state; no runtime budget, data role or authority."""

    format_version: int
    parameters: CoupledParameters
    precisions: tuple[float, ...]
    wake_state: CoupledWakeState | None = None


def _positive_number(value: float | int, name: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be a native positive finite number")
    try:
        resolved = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


def _positive_dimension(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a native positive integer")
    return value


def _copy_matrix(value: Array, shape: tuple[int, int], name: str) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite float64 matrix of shape {shape}")
    return value.copy()


def _parameter_arrays(parameters: CoupledParameters) -> tuple[Array, ...]:
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def _updated_parameters(
    parameters: CoupledParameters, gradients: CoupledParameters, rate: float
) -> CoupledParameters:
    return CoupledParameters(
        tuple(w - rate * g for w, g in zip(parameters.hidden_weights, gradients.hidden_weights)),
        tuple(b - rate * g for b, g in zip(parameters.hidden_biases, gradients.hidden_biases)),
        parameters.output_weight - rate * gradients.output_weight,
        parameters.output_bias - rate * gradients.output_bias,
    )


class CoupledPredictiveCodingNetwork:
    """One or more free hidden layers with fixed-step simultaneous relaxation.

    Dimensions/seed/step count require native ints (bools are refused); dimensions
    are positive, seed nonnegative, and inference steps lie in 1..10000. Rates
    require native int/float positive finite values; precisions additionally lie
    in [1e-6, 1e6]. Batches and parameters require finite float64 numpy matrices.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        seed: int,
        precisions: tuple[float, ...] | None = None,
    ) -> None:
        self._input_dim = _positive_dimension(input_dim, "input_dim")
        if type(hidden_dims) is not tuple or not hidden_dims:
            raise ValueError("hidden_dims must be a nonempty tuple")
        self._hidden_dims = tuple(
            _positive_dimension(width, "hidden width") for width in hidden_dims
        )
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a native nonnegative integer")
        resolved = (1.0,) * len(hidden_dims) if precisions is None else precisions
        if type(resolved) is not tuple or len(resolved) != len(hidden_dims):
            raise ValueError("precisions must be a tuple matching hidden depth")
        self._precisions = tuple(_positive_number(beta, "precision") for beta in resolved)
        if any(not 1e-6 <= beta <= 1e6 for beta in self._precisions):
            raise ValueError("precisions must be in [1e-6, 1e6]")
        rng = np.random.default_rng(seed)
        widths = (input_dim, *hidden_dims)
        self._parameters = CoupledParameters(
            tuple(
                rng.normal(0.0, 1 / sqrt(lower), size=(lower, upper))
                for lower, upper in zip(widths, widths[1:])
            ),
            tuple(np.zeros((1, width), dtype=np.float64) for width in hidden_dims),
            rng.normal(0.0, 1 / sqrt(hidden_dims[-1]), size=(hidden_dims[-1], 1)),
            np.zeros((1, 1), dtype=np.float64),
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self._hidden_dims

    @property
    def precisions(self) -> tuple[float, ...]:
        return self._precisions

    def snapshot_parameters(self) -> CoupledParameters:
        """Return detached arrays; editing a snapshot cannot alter this model."""
        return self._copy_parameters(self._parameters)

    def snapshot_state(self) -> CoupledSnapshot:
        """Detach the complete PC model state, including its precision binding."""
        return CoupledSnapshot(1, self.snapshot_parameters(), self.precisions)

    def _validated_snapshot_parameters(self, snapshot: CoupledSnapshot) -> CoupledParameters:
        if (
            type(snapshot) is not CoupledSnapshot
            or type(snapshot.format_version) is not int
            or snapshot.format_version != 1
        ):
            raise ValueError("unsupported coupled snapshot format")
        if type(snapshot.precisions) is not tuple or len(snapshot.precisions) != len(
            self.precisions
        ):
            raise ValueError("snapshot precision depth differs from this model")
        precisions = tuple(
            _positive_number(value, "snapshot precision") for value in snapshot.precisions
        )
        if precisions != self.precisions:
            raise ValueError("snapshot precisions differ from this model")
        return self._copy_parameters(snapshot.parameters)

    def restore_state(self, snapshot: CoupledSnapshot) -> None:
        """Restore compatible complete PC state after validating every field."""
        candidate = self._validated_snapshot_parameters(snapshot)
        if snapshot.wake_state is not None:
            raise ValueError("PC snapshot must not contain circadian wake state")
        self._parameters = candidate

    def _copy_parameters(self, parameters: CoupledParameters) -> CoupledParameters:
        if type(parameters) is not CoupledParameters:
            raise ValueError("parameters must be CoupledParameters")
        weights, biases = parameters.hidden_weights, parameters.hidden_biases
        if (
            type(weights) is not tuple
            or type(biases) is not tuple
            or len(weights) != len(self.hidden_dims)
            or len(biases) != len(self.hidden_dims)
        ):
            raise ValueError("parameter depth differs from this model")
        widths = (self.input_dim, *self.hidden_dims)
        return CoupledParameters(
            tuple(
                _copy_matrix(weight, (lower, upper), "hidden weight")
                for weight, lower, upper in zip(weights, widths, widths[1:])
            ),
            tuple(
                _copy_matrix(bias, (1, width), "hidden bias")
                for bias, width in zip(biases, self.hidden_dims)
            ),
            _copy_matrix(parameters.output_weight, (self.hidden_dims[-1], 1), "output weight"),
            _copy_matrix(parameters.output_bias, (1, 1), "output bias"),
        )

    def restore_parameters(self, parameters: CoupledParameters) -> None:
        """Validate this exact topology and copy everything before publication."""
        candidate = self._copy_parameters(parameters)
        self._parameters = candidate

    def predict_proba(self, X: Array) -> Array:
        """Return target-free feedforward binary probabilities."""
        return predict_coupled(self._parameters, X)

    def train_epoch(
        self,
        X: Array,
        y: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
        *,
        inference_policy: CoupledInferencePolicy | None = None,
    ) -> CoupledTrainResult:
        rate, latents, evaluation, diagnostics = self._training_with_policy(
            X, y, learning_rate, inference_steps, inference_learning_rate, inference_policy
        )
        candidate = self._prepare_parameters(evaluation.parameter_gradients, rate, X, y, latents)
        self._parameters = candidate
        return CoupledTrainResult(evaluation.energy, inference=diagnostics)

    def _training_with_policy(
        self,
        X: Array,
        y: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
        inference_policy: CoupledInferencePolicy | None,
    ) -> tuple[float, tuple[Array, ...], CoupledEvaluation, CoupledInferenceDiagnostics | None]:
        """Select per-call inference without altering the accepted fixed loop."""
        if inference_policy is None:
            rate, latents, evaluation = self._training_evaluation(
                X, y, learning_rate, inference_steps, inference_learning_rate
            )
            return rate, latents, evaluation, None
        rate = _positive_number(learning_rate, "learning_rate")
        inferred = infer_coupled(
            self._parameters,
            X,
            y,
            self.precisions,
            max_steps=inference_steps,
            learning_rate=inference_learning_rate,
            policy=inference_policy,
        )
        return rate, inferred.latents, inferred.evaluation, inferred.diagnostics

    def _training_evaluation(
        self,
        X: Array,
        y: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> tuple[float, tuple[Array, ...], CoupledEvaluation]:
        """Shared inference before any PC or CPC state publication."""
        rate = _positive_number(learning_rate, "learning_rate")
        inference_rate = _positive_number(inference_learning_rate, "inference_learning_rate")
        if type(inference_steps) is not int or not 1 <= inference_steps <= 10000:
            raise ValueError("inference_steps must be a native integer in 1..10000")
        parameters = self._parameters
        latents = feedforward_latents(parameters, X)
        for _ in range(inference_steps):
            evaluation = evaluate_coupled(parameters, X, y, latents, self.precisions)
            # All partials use the same old latent tuple; predictions are rebuilt
            # by the kernel on the following evaluation of the coupled energy.
            with np.errstate(over="ignore", invalid="ignore"):
                latents = tuple(
                    z - inference_rate * gradient
                    for z, gradient in zip(latents, evaluation.latent_gradients)
                )
            if any(not np.isfinite(z).all() for z in latents):
                raise FloatingPointError("nonfinite inferred latents")
        relaxed = evaluate_coupled(parameters, X, y, latents, self.precisions)
        return rate, latents, relaxed

    def _prepare_parameters(
        self,
        gradients: CoupledParameters,
        rate: float,
        X: Array,
        y: Array,
        latents: tuple[Array, ...],
    ) -> CoupledParameters:
        """Validate a complete prospective bundle, including its coupled energy."""
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = _updated_parameters(self._parameters, gradients, rate)
        if any(not np.isfinite(array).all() for array in _parameter_arrays(candidate)):
            raise FloatingPointError("nonfinite candidate parameters")
        checked = evaluate_coupled(candidate, X, y, latents, self.precisions)
        if not isfinite(checked.energy):
            raise FloatingPointError("nonfinite candidate energy")
        # Why prepare without mutation: CPC must validate its chemistry before
        # publishing parameters, just as PC must validate every parameter first.
        return candidate
