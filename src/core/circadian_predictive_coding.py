"""Circadian predictive-coding network with chemical homeostasis and sleep events."""

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
from src.core.predictive_coding import PredictiveCodingTrainResult

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CircadianConfig:
    """Hyperparameters for circadian-style plasticity and sleep behavior."""

    chemical_decay: float = 0.995
    chemical_buildup_rate: float = 0.02
    plasticity_sensitivity: float = 0.7
    min_plasticity: float = 0.20
    split_threshold: float = 0.80
    prune_threshold: float = 0.08
    max_split_per_sleep: int = 2
    max_prune_per_sleep: int = 2
    split_noise_scale: float = 0.02
    sleep_reset_factor: float = 0.45


@dataclass(frozen=True)
class SleepEventResult:
    """Result summary for one sleep consolidation event."""

    old_hidden_dim: int
    new_hidden_dim: int
    split_indices: tuple[int, ...]
    pruned_indices: tuple[int, ...]


class CircadianPredictiveCodingNetwork:
    """Predictive coding with circadian chemical state and structural sleep updates.

    Why this: a separate chemical layer makes neuron usage history explicit and
    allows sleep-time structural decisions without mixing them into each step.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seed: int,
        circadian_config: CircadianConfig | None = None,
        min_hidden_dim: int = 4,
        max_hidden_dim: int | None = None,
    ) -> None:
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if min_hidden_dim <= 0:
            raise ValueError("min_hidden_dim must be positive")
        if min_hidden_dim > hidden_dim:
            raise ValueError("min_hidden_dim cannot exceed initial hidden_dim")

        config = circadian_config or CircadianConfig()
        self._validate_config(config)
        self.config = config

        self.max_hidden_dim = max_hidden_dim if max_hidden_dim is not None else max(hidden_dim * 4, 16)
        if self.max_hidden_dim < hidden_dim:
            raise ValueError("max_hidden_dim cannot be smaller than initial hidden_dim")

        rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(seed + 10_001)
        self.weight_input_hidden = rng.normal(0.0, 0.5, size=(input_dim, hidden_dim))
        self.bias_hidden = np.zeros((1, hidden_dim), dtype=np.float64)
        self.weight_hidden_output = rng.normal(0.0, 0.5, size=(hidden_dim, 1))
        self.bias_output = np.zeros((1, 1), dtype=np.float64)

        self._hidden_chemical = np.zeros(hidden_dim, dtype=np.float64)
        self._traffic_sum = np.zeros(hidden_dim, dtype=np.float64)
        self._traffic_steps = 0
        self._min_hidden_dim = min_hidden_dim

    @property
    def hidden_dim(self) -> int:
        """Current hidden-layer width."""
        return int(self.weight_input_hidden.shape[1])

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
        output_term = output_error * sigmoid_derivative_from_linear(output_linear)
        grad_hidden_output = (hidden_state.T @ output_term) / sample_count
        grad_output_bias = np.sum(output_term, axis=0, keepdims=True) / sample_count

        hidden_prior_gradient = (-hidden_error) * tanh_derivative_from_linear(hidden_linear_prior)
        grad_input_hidden = (input_batch.T @ hidden_prior_gradient) / sample_count
        grad_hidden_bias = np.sum(hidden_prior_gradient, axis=0, keepdims=True) / sample_count

        self._update_chemical_layer(hidden_state)
        plasticity = self.get_plasticity_state()

        gated_input_hidden = grad_input_hidden * plasticity[np.newaxis, :]
        gated_hidden_output = grad_hidden_output * plasticity[:, np.newaxis]
        gated_hidden_bias = grad_hidden_bias * plasticity[np.newaxis, :]

        self.weight_hidden_output -= learning_rate * gated_hidden_output
        self.bias_output -= learning_rate * grad_output_bias
        self.weight_input_hidden -= learning_rate * gated_input_hidden
        self.bias_hidden -= learning_rate * gated_hidden_bias

        self._record_hidden_traffic(hidden_state)
        energy = self._compute_energy(output_error, hidden_error)
        return PredictiveCodingTrainResult(energy=energy)

    def sleep_event(self) -> SleepEventResult:
        """Consolidate network structure using chemical accumulation."""
        old_hidden_dim = self.hidden_dim
        split_indices = self._select_split_indices()
        self._split_neurons(split_indices)

        pruned_indices = self._select_prune_indices()
        self._prune_neurons(pruned_indices)

        # Sleep partially clears chemistry after consolidation.
        self._hidden_chemical *= self.config.sleep_reset_factor

        return SleepEventResult(
            old_hidden_dim=old_hidden_dim,
            new_hidden_dim=self.hidden_dim,
            split_indices=split_indices,
            pruned_indices=pruned_indices,
        )

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
        return [
            LayerTraffic(layer_name="hidden", mean_abs_activation=mean_traffic),
            LayerTraffic(layer_name="chemical", mean_abs_activation=self._hidden_chemical.copy()),
        ]

    def get_chemical_state(self) -> Array:
        """Return the current chemical layer state."""
        return self._hidden_chemical.copy()

    def set_chemical_state(self, chemical_state: Array) -> None:
        """Set chemical layer state for controlled experiments."""
        if chemical_state.shape != self._hidden_chemical.shape:
            raise ValueError("chemical_state shape must match current hidden_dim")
        if np.any(chemical_state < 0.0):
            raise ValueError("chemical_state cannot contain negative values")
        self._hidden_chemical = chemical_state.copy()

    def get_plasticity_state(self) -> Array:
        """Map chemical buildup to plasticity factors."""
        plasticity = np.exp(-self.config.plasticity_sensitivity * self._hidden_chemical)
        return np.clip(plasticity, self.config.min_plasticity, 1.0)

    def apply_neuron_proposals(self, proposals: list[NeuronChangeProposal]) -> None:
        for proposal in proposals:
            if proposal.add_count > 0 or proposal.remove_indices:
                raise NotImplementedError(
                    "CircadianPredictiveCodingNetwork uses sleep_event() for structural updates."
                )

    def _compute_energy(self, output_error: Array, hidden_error: Array) -> float:
        return float(0.5 * (np.mean(np.square(output_error)) + np.mean(np.square(hidden_error))))

    def _record_hidden_traffic(self, hidden_state: Array) -> None:
        self._traffic_sum += np.mean(np.abs(hidden_state), axis=0)
        self._traffic_steps += 1

    def _update_chemical_layer(self, hidden_state: Array) -> None:
        activity = np.mean(np.abs(hidden_state), axis=0)
        self._hidden_chemical = (
            self.config.chemical_decay * self._hidden_chemical
            + self.config.chemical_buildup_rate * activity
        )

    def _select_split_indices(self) -> tuple[int, ...]:
        current_dim = self.hidden_dim
        remaining_capacity = self.max_hidden_dim - current_dim
        if remaining_capacity <= 0 or self.config.max_split_per_sleep <= 0:
            return ()

        candidates = np.where(self._hidden_chemical >= self.config.split_threshold)[0]
        if candidates.size == 0:
            return ()

        sorted_candidates = candidates[np.argsort(self._hidden_chemical[candidates])[::-1]]
        split_count = min(
            int(sorted_candidates.size),
            int(self.config.max_split_per_sleep),
            int(remaining_capacity),
        )
        chosen = sorted_candidates[:split_count]
        return tuple(int(index) for index in chosen.tolist())

    def _select_prune_indices(self) -> tuple[int, ...]:
        removable = self.hidden_dim - self._min_hidden_dim
        if removable <= 0 or self.config.max_prune_per_sleep <= 0:
            return ()

        candidates = np.where(self._hidden_chemical <= self.config.prune_threshold)[0]
        if candidates.size == 0:
            return ()

        sorted_candidates = candidates[np.argsort(self._hidden_chemical[candidates])]
        prune_count = min(
            int(sorted_candidates.size),
            int(self.config.max_prune_per_sleep),
            int(removable),
        )
        chosen = sorted_candidates[:prune_count]
        return tuple(sorted(int(index) for index in chosen.tolist()))

    def _split_neurons(self, split_indices: tuple[int, ...]) -> None:
        if not split_indices:
            return

        new_in_columns: list[Array] = []
        new_hidden_bias_values: list[float] = []
        new_out_rows: list[Array] = []
        new_chemical_values: list[float] = []
        new_traffic_values: list[float] = []

        for index in split_indices:
            input_column = self.weight_input_hidden[:, index]
            output_row = self.weight_hidden_output[index, :]
            bias_value = float(self.bias_hidden[0, index])
            chemical_value = float(self._hidden_chemical[index])

            input_noise = self._rng.normal(
                loc=0.0, scale=self.config.split_noise_scale, size=input_column.shape
            )
            output_noise = self._rng.normal(
                loc=0.0, scale=self.config.split_noise_scale, size=output_row.shape
            )
            bias_noise = float(self._rng.normal(loc=0.0, scale=self.config.split_noise_scale))

            new_in_columns.append((input_column + input_noise).astype(np.float64))
            new_out_rows.append((output_row + output_noise).astype(np.float64))
            new_hidden_bias_values.append(bias_value + bias_noise)
            new_chemical_values.append(chemical_value * 0.5)
            new_traffic_values.append(0.0)

            self._hidden_chemical[index] = chemical_value * 0.5

        appended_input_columns = np.column_stack(new_in_columns)
        self.weight_input_hidden = np.hstack([self.weight_input_hidden, appended_input_columns])

        appended_output_rows = np.vstack(new_out_rows)
        self.weight_hidden_output = np.vstack([self.weight_hidden_output, appended_output_rows])

        appended_bias = np.array(new_hidden_bias_values, dtype=np.float64).reshape(1, -1)
        self.bias_hidden = np.hstack([self.bias_hidden, appended_bias])

        self._hidden_chemical = np.concatenate(
            [self._hidden_chemical, np.array(new_chemical_values, dtype=np.float64)]
        )
        self._traffic_sum = np.concatenate(
            [self._traffic_sum, np.array(new_traffic_values, dtype=np.float64)]
        )

    def _prune_neurons(self, prune_indices: tuple[int, ...]) -> None:
        if not prune_indices:
            return
        mask = np.ones(self.hidden_dim, dtype=bool)
        mask[list(prune_indices)] = False

        self.weight_input_hidden = self.weight_input_hidden[:, mask]
        self.weight_hidden_output = self.weight_hidden_output[mask, :]
        self.bias_hidden = self.bias_hidden[:, mask]
        self._hidden_chemical = self._hidden_chemical[mask]
        self._traffic_sum = self._traffic_sum[mask]

    def _validate_config(self, config: CircadianConfig) -> None:
        if not (0.0 <= config.chemical_decay <= 1.0):
            raise ValueError("chemical_decay must be between 0 and 1")
        if config.chemical_buildup_rate <= 0.0:
            raise ValueError("chemical_buildup_rate must be positive")
        if config.plasticity_sensitivity <= 0.0:
            raise ValueError("plasticity_sensitivity must be positive")
        if not (0.0 < config.min_plasticity <= 1.0):
            raise ValueError("min_plasticity must be in (0, 1]")
        if config.max_split_per_sleep < 0 or config.max_prune_per_sleep < 0:
            raise ValueError("max split/prune per sleep must be non-negative")
        if config.split_noise_scale < 0.0:
            raise ValueError("split_noise_scale must be non-negative")
        if not (0.0 <= config.sleep_reset_factor <= 1.0):
            raise ValueError("sleep_reset_factor must be between 0 and 1")
