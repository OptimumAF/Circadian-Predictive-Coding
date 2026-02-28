"""A configurable multi-hidden-layer predictive-coding style network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.activations import sigmoid, tanh, tanh_derivative_from_linear
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PredictiveCodingTrainResult:
    """Metrics from one predictive-coding optimization epoch."""

    energy: float


class PredictiveCodingNetwork:
    """Predictive-coding inspired model with iterative latent-state inference."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seed: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        resolved_hidden_dims = self._resolve_hidden_dims(
            hidden_dim=hidden_dim,
            hidden_dims=hidden_dims,
        )

        rng = np.random.default_rng(seed)
        self._hidden_weights: list[Array] = []
        self._hidden_biases: list[Array] = []

        previous_dim = input_dim
        for layer_dim in resolved_hidden_dims:
            self._hidden_weights.append(
                rng.normal(0.0, 0.5, size=(previous_dim, layer_dim))
            )
            self._hidden_biases.append(np.zeros((1, layer_dim), dtype=np.float64))
            previous_dim = layer_dim

        self.weight_hidden_output = rng.normal(0.0, 0.5, size=(previous_dim, 1))
        self.bias_output = np.zeros((1, 1), dtype=np.float64)

        # Backward-compatible aliases for single-hidden-layer callers.
        self.weight_input_hidden = self._hidden_weights[0]
        self.bias_hidden = self._hidden_biases[0]
        self.hidden_dims = tuple(resolved_hidden_dims)

        self._traffic_sums = [
            np.zeros(layer_dim, dtype=np.float64) for layer_dim in resolved_hidden_dims
        ]
        self._traffic_steps = 0

    def train_epoch(
        self,
        input_batch: Array,
        target_batch: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> PredictiveCodingTrainResult:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if inference_learning_rate <= 0.0:
            raise ValueError("inference_learning_rate must be positive")

        hidden_linear_priors: list[Array] = []
        hidden_priors: list[Array] = []
        prior_activation = input_batch
        for layer_weight, layer_bias in zip(self._hidden_weights, self._hidden_biases):
            hidden_linear = prior_activation @ layer_weight + layer_bias
            hidden_prior = tanh(hidden_linear)
            hidden_linear_priors.append(hidden_linear)
            hidden_priors.append(hidden_prior)
            prior_activation = hidden_prior

        hidden_states = [hidden_prior.copy() for hidden_prior in hidden_priors]
        for _ in range(inference_steps):
            output_linear = hidden_states[-1] @ self.weight_hidden_output + self.bias_output
            output_prediction = sigmoid(output_linear)
            output_error = output_prediction - target_batch

            hidden_errors = [
                hidden_states[layer_index] - hidden_priors[layer_index]
                for layer_index in range(len(hidden_states))
            ]
            hidden_gradients = [np.zeros_like(hidden_state) for hidden_state in hidden_states]
            hidden_gradients[-1] = hidden_errors[-1] + (output_error @ self.weight_hidden_output.T)
            for layer_index in range(len(hidden_states) - 2, -1, -1):
                topdown = hidden_errors[layer_index + 1] @ self._hidden_weights[layer_index + 1].T
                hidden_gradients[layer_index] = hidden_errors[layer_index] + topdown
            for layer_index, hidden_gradient in enumerate(hidden_gradients):
                hidden_states[layer_index] -= inference_learning_rate * hidden_gradient

        output_linear = hidden_states[-1] @ self.weight_hidden_output + self.bias_output
        output_prediction = sigmoid(output_linear)
        output_error = output_prediction - target_batch
        hidden_errors = [
            hidden_states[layer_index] - hidden_priors[layer_index]
            for layer_index in range(len(hidden_states))
        ]

        sample_count = float(input_batch.shape[0])
        grad_hidden_output = (hidden_states[-1].T @ output_error) / sample_count
        grad_output_bias = np.sum(output_error, axis=0, keepdims=True) / sample_count

        grad_hidden_weights: list[Array] = []
        grad_hidden_biases: list[Array] = []
        previous_prior_activation = input_batch
        for layer_index in range(len(hidden_states)):
            hidden_prior_gradient = (
                -hidden_errors[layer_index]
            ) * tanh_derivative_from_linear(hidden_linear_priors[layer_index])
            grad_hidden_weights.append((previous_prior_activation.T @ hidden_prior_gradient) / sample_count)
            grad_hidden_biases.append(np.sum(hidden_prior_gradient, axis=0, keepdims=True) / sample_count)
            previous_prior_activation = hidden_priors[layer_index]

        self.weight_hidden_output -= learning_rate * grad_hidden_output
        self.bias_output -= learning_rate * grad_output_bias
        for layer_index in range(len(self._hidden_weights)):
            self._hidden_weights[layer_index] -= learning_rate * grad_hidden_weights[layer_index]
            self._hidden_biases[layer_index] -= learning_rate * grad_hidden_biases[layer_index]

        self._record_hidden_traffic(hidden_states)
        energy = self._compute_energy(
            output_prediction=output_prediction,
            target_batch=target_batch,
            hidden_errors=hidden_errors,
        )
        return PredictiveCodingTrainResult(energy=energy)

    def predict_proba(self, input_batch: Array) -> Array:
        activation = input_batch
        for layer_weight, layer_bias in zip(self._hidden_weights, self._hidden_biases):
            hidden_linear = activation @ layer_weight + layer_bias
            activation = tanh(hidden_linear)
        output_linear = activation @ self.weight_hidden_output + self.bias_output
        return sigmoid(output_linear)

    def predict_label(self, input_batch: Array) -> Array:
        probabilities = self.predict_proba(input_batch)
        return (probabilities >= 0.5).astype(np.float64)

    def compute_accuracy(self, input_batch: Array, target_batch: Array) -> float:
        prediction = self.predict_label(input_batch)
        return float(np.mean(prediction == target_batch))

    def get_layer_traffic(self) -> list[LayerTraffic]:
        traffic_layers: list[LayerTraffic] = []
        for layer_index, traffic_sum in enumerate(self._traffic_sums):
            if self._traffic_steps == 0:
                mean_traffic = traffic_sum.copy()
            else:
                mean_traffic = traffic_sum / float(self._traffic_steps)
            traffic_layers.append(
                LayerTraffic(
                    layer_name=f"hidden_{layer_index}",
                    mean_abs_activation=mean_traffic,
                )
            )
        return traffic_layers

    def apply_neuron_proposals(self, proposals: list[NeuronChangeProposal]) -> None:
        for proposal in proposals:
            if proposal.add_count > 0 or proposal.remove_indices:
                raise NotImplementedError(
                    "Dynamic neuron changes are not implemented yet for PredictiveCodingNetwork."
                )

    def _compute_energy(self, output_prediction: Array, target_batch: Array, hidden_errors: list[Array]) -> float:
        bce = self._binary_cross_entropy(output_prediction, target_batch)
        penalty = float(
            np.mean(
                np.concatenate(
                    [np.square(hidden_error).reshape(-1) for hidden_error in hidden_errors]
                )
            )
        )
        return bce + 0.5 * penalty

    def _binary_cross_entropy(self, output_prediction: Array, target_batch: Array) -> float:
        epsilon = 1e-8
        clipped_prediction = np.clip(output_prediction, epsilon, 1.0 - epsilon)
        return float(
            np.mean(
                -(
                    target_batch * np.log(clipped_prediction)
                    + (1.0 - target_batch) * np.log(1.0 - clipped_prediction)
                )
            )
        )

    def _record_hidden_traffic(self, hidden_states: list[Array]) -> None:
        for layer_index, hidden_state in enumerate(hidden_states):
            self._traffic_sums[layer_index] += np.mean(np.abs(hidden_state), axis=0)
        self._traffic_steps += 1

    def _resolve_hidden_dims(
        self,
        hidden_dim: int,
        hidden_dims: list[int] | tuple[int, ...] | None,
    ) -> list[int]:
        if hidden_dims is None:
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            return [hidden_dim]

        resolved = [int(value) for value in hidden_dims]
        if not resolved:
            raise ValueError("hidden_dims cannot be empty")
        if any(value <= 0 for value in resolved):
            raise ValueError("all hidden_dims values must be positive")
        return resolved
