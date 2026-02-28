"""A configurable multi-hidden-layer MLP trained with backpropagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.activations import sigmoid, tanh, tanh_derivative_from_linear
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BackpropTrainResult:
    """Metrics produced by one optimization step."""

    loss: float


class BackpropMLP:
    """Binary classification MLP optimized with gradient descent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seed: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        resolved_hidden_dims = self._resolve_hidden_dims(
            hidden_dim=hidden_dim,
            hidden_dims=hidden_dims,
        )
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

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

        # Backward-compatible aliases for single-hidden-layer consumers.
        self.weight_input_hidden = self._hidden_weights[0]
        self.bias_hidden = self._hidden_biases[0]
        self.hidden_dims = tuple(resolved_hidden_dims)

        self._traffic_sums = [
            np.zeros(layer_dim, dtype=np.float64) for layer_dim in resolved_hidden_dims
        ]
        self._traffic_steps = 0

    def train_epoch(self, input_batch: Array, target_batch: Array, learning_rate: float) -> BackpropTrainResult:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        hidden_linears, hidden_activations, output_activation = self._forward(input_batch)
        sample_count = float(input_batch.shape[0])

        output_grad = (output_activation - target_batch) / sample_count
        grad_hidden_output = hidden_activations[-1].T @ output_grad
        grad_output_bias = np.sum(output_grad, axis=0, keepdims=True)

        hidden_deltas: list[Array] = [np.zeros_like(activation) for activation in hidden_activations]
        hidden_deltas[-1] = (output_grad @ self.weight_hidden_output.T) * tanh_derivative_from_linear(
            hidden_linears[-1]
        )
        for layer_index in range(len(hidden_deltas) - 2, -1, -1):
            next_weight = self._hidden_weights[layer_index + 1]
            hidden_deltas[layer_index] = (
                hidden_deltas[layer_index + 1] @ next_weight.T
            ) * tanh_derivative_from_linear(hidden_linears[layer_index])

        grad_hidden_weights: list[Array] = []
        grad_hidden_biases: list[Array] = []
        previous_activation = input_batch
        for layer_index, layer_delta in enumerate(hidden_deltas):
            grad_hidden_weights.append(previous_activation.T @ layer_delta)
            grad_hidden_biases.append(np.sum(layer_delta, axis=0, keepdims=True))
            previous_activation = hidden_activations[layer_index]

        self.weight_hidden_output -= learning_rate * grad_hidden_output
        self.bias_output -= learning_rate * grad_output_bias
        for layer_index in range(len(self._hidden_weights)):
            self._hidden_weights[layer_index] -= learning_rate * grad_hidden_weights[layer_index]
            self._hidden_biases[layer_index] -= learning_rate * grad_hidden_biases[layer_index]

        self._record_hidden_traffic(hidden_activations)
        loss = self._binary_cross_entropy(output_activation, target_batch)
        return BackpropTrainResult(loss=loss)

    def predict_proba(self, input_batch: Array) -> Array:
        _, _, output_activation = self._forward(input_batch)
        return output_activation

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
                    "Dynamic neuron changes are not implemented yet for BackpropMLP."
                )

    def _forward(self, input_batch: Array) -> tuple[list[Array], list[Array], Array]:
        hidden_linears: list[Array] = []
        hidden_activations: list[Array] = []
        activation = input_batch
        for layer_weight, layer_bias in zip(self._hidden_weights, self._hidden_biases):
            hidden_linear = activation @ layer_weight + layer_bias
            hidden_activation = tanh(hidden_linear)
            hidden_linears.append(hidden_linear)
            hidden_activations.append(hidden_activation)
            activation = hidden_activation
        output_linear = activation @ self.weight_hidden_output + self.bias_output
        output_activation = sigmoid(output_linear)
        return hidden_linears, hidden_activations, output_activation

    def _binary_cross_entropy(self, predictions: Array, targets: Array) -> float:
        epsilon = 1e-8
        clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
        loss = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
        return float(np.mean(loss))

    def _record_hidden_traffic(self, hidden_activations: list[Array]) -> None:
        for layer_index, activation in enumerate(hidden_activations):
            self._traffic_sums[layer_index] += np.mean(np.abs(activation), axis=0)
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
