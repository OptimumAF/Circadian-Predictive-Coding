"""A simple one-hidden-layer MLP trained with backpropagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.activations import (
    sigmoid,
    tanh,
    tanh_derivative_from_linear,
)
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BackpropTrainResult:
    """Metrics produced by one optimization step."""

    loss: float


class BackpropMLP:
    """Binary classification MLP optimized with gradient descent."""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int) -> None:
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")

        rng = np.random.default_rng(seed)
        self.weight_input_hidden = rng.normal(0.0, 0.5, size=(input_dim, hidden_dim))
        self.bias_hidden = np.zeros((1, hidden_dim), dtype=np.float64)
        self.weight_hidden_output = rng.normal(0.0, 0.5, size=(hidden_dim, 1))
        self.bias_output = np.zeros((1, 1), dtype=np.float64)

        self._traffic_sum = np.zeros(hidden_dim, dtype=np.float64)
        self._traffic_steps = 0

    def train_epoch(self, input_batch: Array, target_batch: Array, learning_rate: float) -> BackpropTrainResult:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        hidden_linear, hidden_activation, output_activation = self._forward(input_batch)
        sample_count = float(input_batch.shape[0])

        output_grad = (output_activation - target_batch) / sample_count
        grad_hidden_output = hidden_activation.T @ output_grad
        grad_output_bias = np.sum(output_grad, axis=0, keepdims=True)

        hidden_grad = (output_grad @ self.weight_hidden_output.T) * tanh_derivative_from_linear(hidden_linear)
        grad_input_hidden = input_batch.T @ hidden_grad
        grad_hidden_bias = np.sum(hidden_grad, axis=0, keepdims=True)

        self.weight_hidden_output -= learning_rate * grad_hidden_output
        self.bias_output -= learning_rate * grad_output_bias
        self.weight_input_hidden -= learning_rate * grad_input_hidden
        self.bias_hidden -= learning_rate * grad_hidden_bias

        self._record_hidden_traffic(hidden_activation)
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
        if self._traffic_steps == 0:
            mean_traffic = self._traffic_sum.copy()
        else:
            mean_traffic = self._traffic_sum / float(self._traffic_steps)
        return [LayerTraffic(layer_name="hidden", mean_abs_activation=mean_traffic)]

    def apply_neuron_proposals(self, proposals: list[NeuronChangeProposal]) -> None:
        for proposal in proposals:
            if proposal.add_count > 0 or proposal.remove_indices:
                raise NotImplementedError(
                    "Dynamic neuron changes are not implemented yet for BackpropMLP."
                )

    def _forward(self, input_batch: Array) -> tuple[Array, Array, Array]:
        hidden_linear = input_batch @ self.weight_input_hidden + self.bias_hidden
        hidden_activation = tanh(hidden_linear)
        output_linear = hidden_activation @ self.weight_hidden_output + self.bias_output
        output_activation = sigmoid(output_linear)
        return hidden_linear, hidden_activation, output_activation

    def _binary_cross_entropy(self, predictions: Array, targets: Array) -> float:
        epsilon = 1e-8
        clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
        loss = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
        return float(np.mean(loss))

    def _record_hidden_traffic(self, hidden_activation: Array) -> None:
        self._traffic_sum += np.mean(np.abs(hidden_activation), axis=0)
        self._traffic_steps += 1

