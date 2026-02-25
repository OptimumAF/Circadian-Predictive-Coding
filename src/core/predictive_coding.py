"""A lightweight predictive-coding style network for binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.activations import (
    sigmoid,
    sigmoid_derivative_from_linear,
    tanh,
    tanh_derivative_from_linear,
)
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PredictiveCodingTrainResult:
    """Metrics from one predictive-coding optimization epoch."""

    energy: float


class PredictiveCodingNetwork:
    """Predictive-coding inspired model with iterative latent-state inference.

    Why this: hidden states are inferred by minimizing local prediction errors
    before weight updates, which makes the contrast with backprop explicit.
    """

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

        hidden_linear_prior = input_batch @ self.weight_input_hidden + self.bias_hidden
        hidden_prior = tanh(hidden_linear_prior)
        hidden_state = hidden_prior.copy()

        for _ in range(inference_steps):
            output_linear = hidden_state @ self.weight_hidden_output + self.bias_output
            output_prediction = sigmoid(output_linear)
            output_error = output_prediction - target_batch

            hidden_error = hidden_state - hidden_prior
            output_to_hidden = (output_error * sigmoid_derivative_from_linear(output_linear)) @ (
                self.weight_hidden_output.T
            )
            hidden_gradient = hidden_error + output_to_hidden
            hidden_state -= inference_learning_rate * hidden_gradient

        output_linear = hidden_state @ self.weight_hidden_output + self.bias_output
        output_prediction = sigmoid(output_linear)
        output_error = output_prediction - target_batch
        hidden_error = hidden_state - hidden_prior

        sample_count = float(input_batch.shape[0])
        grad_hidden_output = hidden_state.T @ (
            output_error * sigmoid_derivative_from_linear(output_linear)
        )
        grad_hidden_output /= sample_count
        grad_output_bias = np.sum(
            output_error * sigmoid_derivative_from_linear(output_linear), axis=0, keepdims=True
        ) / sample_count

        hidden_prior_gradient = (-hidden_error) * tanh_derivative_from_linear(hidden_linear_prior)
        grad_input_hidden = (input_batch.T @ hidden_prior_gradient) / sample_count
        grad_hidden_bias = np.sum(hidden_prior_gradient, axis=0, keepdims=True) / sample_count

        self.weight_hidden_output -= learning_rate * grad_hidden_output
        self.bias_output -= learning_rate * grad_output_bias
        self.weight_input_hidden -= learning_rate * grad_input_hidden
        self.bias_hidden -= learning_rate * grad_hidden_bias

        self._record_hidden_traffic(hidden_state)
        energy = self._compute_energy(output_error, hidden_error)
        return PredictiveCodingTrainResult(energy=energy)

    def predict_proba(self, input_batch: Array) -> Array:
        hidden_linear = input_batch @ self.weight_input_hidden + self.bias_hidden
        hidden_activation = tanh(hidden_linear)
        output_linear = hidden_activation @ self.weight_hidden_output + self.bias_output
        return sigmoid(output_linear)

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
                    "Dynamic neuron changes are not implemented yet for PredictiveCodingNetwork."
                )

    def _compute_energy(self, output_error: Array, hidden_error: Array) -> float:
        return float(0.5 * (np.mean(np.square(output_error)) + np.mean(np.square(hidden_error))))

    def _record_hidden_traffic(self, hidden_state: Array) -> None:
        self._traffic_sum += np.mean(np.abs(hidden_state), axis=0)
        self._traffic_steps += 1

