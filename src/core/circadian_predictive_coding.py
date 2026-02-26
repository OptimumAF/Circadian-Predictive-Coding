"""Circadian predictive-coding network with chemical homeostasis and sleep events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.activations import (
    sigmoid,
    tanh,
    tanh_derivative_from_linear,
)
from src.core.neuron_adaptation import (
    LayerTraffic,
    NeuronAdaptationPolicy,
    NeuronChangeProposal,
)
from src.core.predictive_coding import PredictiveCodingTrainResult

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ReplaySnapshot:
    """Stored wake snapshot for optional sleep replay consolidation."""

    input_batch: Array
    target_batch: Array
    priority: float
    positive_fraction: float


@dataclass(frozen=True)
class CircadianConfig:
    """Hyperparameters for circadian-style plasticity and sleep behavior."""

    chemical_decay: float = 0.995
    chemical_buildup_rate: float = 0.02
    use_saturating_chemical: bool = False
    chemical_max_value: float = 2.5
    chemical_saturation_gain: float = 1.0
    use_dual_chemical: bool = False
    dual_fast_mix: float = 0.70
    slow_chemical_decay: float = 0.999
    slow_buildup_scale: float = 0.25
    plasticity_sensitivity: float = 0.7
    use_adaptive_plasticity_sensitivity: bool = False
    plasticity_sensitivity_min: float = 0.35
    plasticity_sensitivity_max: float = 1.20
    plasticity_importance_mix: float = 0.50
    min_plasticity: float = 0.20

    # Static thresholds are still available, but adaptive percentile thresholds
    # can be enabled to react to changing chemical distributions over training.
    use_adaptive_thresholds: bool = False
    adaptive_split_percentile: float = 85.0
    adaptive_prune_percentile: float = 20.0
    split_threshold: float = 0.80
    prune_threshold: float = 0.08
    split_hysteresis_margin: float = 0.0
    prune_hysteresis_margin: float = 0.0
    split_cooldown_epochs: int = 0
    prune_cooldown_epochs: int = 0

    # Blend chemical accumulation and output weight-norm for split/prune ranking.
    split_weight_norm_mix: float = 0.30
    prune_weight_norm_mix: float = 0.30
    split_importance_mix: float = 0.20
    prune_importance_mix: float = 0.35
    importance_ema_decay: float = 0.95

    max_split_per_sleep: int = 2
    max_prune_per_sleep: int = 2
    split_noise_scale: float = 0.02
    sleep_reset_factor: float = 0.45

    # Adaptive sleep trigger (optional) based on energy plateau and chemistry variance.
    use_adaptive_sleep_trigger: bool = False
    min_epochs_between_sleep: int = 10
    sleep_energy_window: int = 8
    sleep_plateau_delta: float = 1e-3
    sleep_chemical_variance_threshold: float = 0.02

    # Gradual pruning: marked neurons decay for a few epochs before removal.
    prune_decay_steps: int = 1
    prune_decay_factor: float = 0.60

    # Optional post-sleep global downscaling for homeostasis.
    homeostatic_downscale_factor: float = 1.0
    homeostasis_target_input_norm: float = 0.0
    homeostasis_target_output_norm: float = 0.0
    homeostasis_strength: float = 0.50

    # Optional replay-style consolidation from a small recent-memory buffer.
    replay_steps: int = 0
    replay_memory_size: int = 8
    replay_learning_rate: float = 0.01
    replay_inference_steps: int = 12
    replay_inference_learning_rate: float = 0.15
    replay_prioritized: bool = True
    replay_class_balanced: bool = True


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
        self._hidden_chemical_fast = np.zeros(hidden_dim, dtype=np.float64)
        self._hidden_chemical_slow = np.zeros(hidden_dim, dtype=np.float64)
        self._neuron_age = np.zeros(hidden_dim, dtype=np.float64)
        self._traffic_sum = np.zeros(hidden_dim, dtype=np.float64)
        self._importance_ema = np.zeros(hidden_dim, dtype=np.float64)
        self._traffic_steps = 0
        self._min_hidden_dim = min_hidden_dim
        self._prune_ttl = np.zeros(hidden_dim, dtype=np.int32)
        self._prune_marked = np.zeros(hidden_dim, dtype=bool)
        self._split_cooldown = np.zeros(hidden_dim, dtype=np.int32)
        self._prune_cooldown = np.zeros(hidden_dim, dtype=np.int32)

        self._epoch_count = 0
        self._epochs_since_sleep = 0
        self._energy_history: list[float] = []
        self._replay_memory: deque[ReplaySnapshot] = deque(
            maxlen=self.config.replay_memory_size
        )

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
        energy = self._run_training_step(
            input_batch=input_batch,
            target_batch=target_batch,
            learning_rate=learning_rate,
            inference_steps=inference_steps,
            inference_learning_rate=inference_learning_rate,
            update_epoch_state=True,
            store_replay_snapshot=True,
        )
        return PredictiveCodingTrainResult(energy=energy)

    def _run_training_step(
        self,
        input_batch: Array,
        target_batch: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
        update_epoch_state: bool,
        store_replay_snapshot: bool,
    ) -> float:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if inference_learning_rate <= 0.0:
            raise ValueError("inference_learning_rate must be positive")

        self._decay_cooldowns()
        self._apply_prune_decay_step()

        hidden_linear_prior = input_batch @ self.weight_input_hidden + self.bias_hidden
        hidden_prior = tanh(hidden_linear_prior)
        hidden_state = hidden_prior.copy()

        for _ in range(inference_steps):
            output_linear = hidden_state @ self.weight_hidden_output + self.bias_output
            output_prediction = sigmoid(output_linear)
            output_error = output_prediction - target_batch

            hidden_error = hidden_state - hidden_prior
            output_to_hidden = output_error @ self.weight_hidden_output.T
            hidden_gradient = hidden_error + output_to_hidden
            hidden_state -= inference_learning_rate * hidden_gradient

        output_linear = hidden_state @ self.weight_hidden_output + self.bias_output
        output_prediction = sigmoid(output_linear)
        output_error = output_prediction - target_batch
        hidden_error = hidden_state - hidden_prior

        sample_count = float(input_batch.shape[0])
        output_term = output_error
        grad_hidden_output = (hidden_state.T @ output_term) / sample_count
        grad_output_bias = np.sum(output_term, axis=0, keepdims=True) / sample_count

        hidden_prior_gradient = (-hidden_error) * tanh_derivative_from_linear(hidden_linear_prior)
        grad_input_hidden = (input_batch.T @ hidden_prior_gradient) / sample_count
        grad_hidden_bias = np.sum(hidden_prior_gradient, axis=0, keepdims=True) / sample_count

        self._update_chemical_layer(hidden_state)
        self._update_importance_ema(grad_hidden_output)
        plasticity = self.get_plasticity_state()

        gated_input_hidden = grad_input_hidden * plasticity[np.newaxis, :]
        gated_hidden_output = grad_hidden_output * plasticity[:, np.newaxis]
        gated_hidden_bias = grad_hidden_bias * plasticity[np.newaxis, :]

        self.weight_hidden_output -= learning_rate * gated_hidden_output
        self.bias_output -= learning_rate * grad_output_bias
        self.weight_input_hidden -= learning_rate * gated_input_hidden
        self.bias_hidden -= learning_rate * gated_hidden_bias

        self._record_hidden_traffic(hidden_state)
        energy = self._compute_energy(
            output_prediction=output_prediction,
            target_batch=target_batch,
            hidden_error=hidden_error,
        )
        if update_epoch_state:
            self._epoch_count += 1
            self._neuron_age += 1.0
            self._epochs_since_sleep += 1
            self._energy_history.append(energy)
            max_history = max(self.config.sleep_energy_window * 4, 16)
            if len(self._energy_history) > max_history:
                self._energy_history = self._energy_history[-max_history:]
        if store_replay_snapshot:
            self._store_replay_snapshot(input_batch, target_batch)
        return energy

    def should_trigger_sleep(self) -> bool:
        """Return whether adaptive sleep criteria indicate consolidation is needed."""
        if not self.config.use_adaptive_sleep_trigger:
            return False
        if self._epochs_since_sleep < self.config.min_epochs_between_sleep:
            return False
        if len(self._energy_history) < self.config.sleep_energy_window:
            return False

        recent = self._energy_history[-self.config.sleep_energy_window :]
        energy_improvement = recent[0] - recent[-1]
        plateau = energy_improvement <= self.config.sleep_plateau_delta
        chemical_variance = float(np.var(self._hidden_chemical))
        high_chemical_variance = (
            chemical_variance >= self.config.sleep_chemical_variance_threshold
        )
        return plateau and high_chemical_variance

    def sleep_event(
        self,
        adaptation_policy: NeuronAdaptationPolicy | None = None,
        force_sleep: bool = True,
    ) -> SleepEventResult:
        """Consolidate structure; optionally trigger only when adaptive criteria fire."""
        if not force_sleep and not self.should_trigger_sleep():
            return SleepEventResult(
                old_hidden_dim=self.hidden_dim,
                new_hidden_dim=self.hidden_dim,
                split_indices=(),
                pruned_indices=(),
            )

        old_hidden_dim = self.hidden_dim
        split_indices: tuple[int, ...]
        pruned_indices: tuple[int, ...]
        if adaptation_policy is None:
            split_indices = self._select_split_indices()
            pruned_indices = self._select_prune_indices()
        else:
            split_indices, pruned_indices = self._derive_indices_from_policy(adaptation_policy)

        self._split_neurons(split_indices)
        self._schedule_or_prune(pruned_indices)
        self._apply_homeostatic_downscaling()
        self._run_replay_consolidation()

        # Sleep partially clears chemistry after consolidation to reset plasticity gate.
        self._hidden_chemical *= self.config.sleep_reset_factor
        if self.config.use_dual_chemical:
            self._hidden_chemical_fast *= self.config.sleep_reset_factor
            self._hidden_chemical_slow *= self.config.sleep_reset_factor
        self._epochs_since_sleep = 0

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
        self._hidden_chemical_fast = chemical_state.copy()
        self._hidden_chemical_slow = chemical_state.copy()

    def get_plasticity_state(self) -> Array:
        """Map chemical buildup to plasticity factors."""
        sensitivity = self._compute_plasticity_sensitivity()
        plasticity = np.exp(-sensitivity * self._hidden_chemical)
        return np.clip(plasticity, self.config.min_plasticity, 1.0)

    def _compute_plasticity_sensitivity(self) -> Array:
        if not self.config.use_adaptive_plasticity_sensitivity:
            return np.full_like(self._hidden_chemical, self.config.plasticity_sensitivity)

        age_component = self._normalize_vector_zero_base(self._neuron_age)
        importance_component = self._normalize_vector_zero_base(self._importance_ema)
        importance_mix = np.clip(self.config.plasticity_importance_mix, 0.0, 1.0)
        stability = (
            importance_mix * importance_component + (1.0 - importance_mix) * age_component
        )
        span = self.config.plasticity_sensitivity_max - self.config.plasticity_sensitivity_min
        return self.config.plasticity_sensitivity_min + span * stability

    def apply_neuron_proposals(self, proposals: list[NeuronChangeProposal]) -> None:
        split_indices, prune_indices = self._indices_from_proposals(proposals)
        self._split_neurons(split_indices)
        self._schedule_or_prune(prune_indices)

    def _compute_energy(self, output_prediction: Array, target_batch: Array, hidden_error: Array) -> float:
        bce = self._binary_cross_entropy(output_prediction, target_batch)
        hidden_penalty = 0.5 * float(np.mean(np.square(hidden_error)))
        return bce + hidden_penalty

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

    def _record_hidden_traffic(self, hidden_state: Array) -> None:
        self._traffic_sum += np.mean(np.abs(hidden_state), axis=0)
        self._traffic_steps += 1

    def _update_importance_ema(self, grad_hidden_output: Array) -> None:
        importance = np.mean(np.abs(grad_hidden_output), axis=1)
        decay = self.config.importance_ema_decay
        self._importance_ema = decay * self._importance_ema + (1.0 - decay) * importance

    def _update_chemical_layer(self, hidden_state: Array) -> None:
        activity = np.mean(np.abs(hidden_state), axis=0)
        if not self.config.use_dual_chemical:
            self._hidden_chemical = self._accumulate_chemical(
                current=self._hidden_chemical,
                decay=self.config.chemical_decay,
                buildup_rate=self.config.chemical_buildup_rate,
                activity=activity,
            )
            self._hidden_chemical_fast = self._hidden_chemical.copy()
            self._hidden_chemical_slow = self._hidden_chemical.copy()
            return

        self._hidden_chemical_fast = self._accumulate_chemical(
            current=self._hidden_chemical_fast,
            decay=self.config.chemical_decay,
            buildup_rate=self.config.chemical_buildup_rate,
            activity=activity,
        )
        slow_rate = self.config.chemical_buildup_rate * self.config.slow_buildup_scale
        self._hidden_chemical_slow = self._accumulate_chemical(
            current=self._hidden_chemical_slow,
            decay=self.config.slow_chemical_decay,
            buildup_rate=slow_rate,
            activity=activity,
        )
        fast_mix = np.clip(self.config.dual_fast_mix, 0.0, 1.0)
        self._hidden_chemical = (
            fast_mix * self._hidden_chemical_fast + (1.0 - fast_mix) * self._hidden_chemical_slow
        )

    def _accumulate_chemical(
        self, current: Array, decay: float, buildup_rate: float, activity: Array
    ) -> Array:
        decayed = decay * current
        increment = buildup_rate * activity
        if not self.config.use_saturating_chemical:
            return decayed + increment

        max_value = self.config.chemical_max_value
        headroom = np.maximum(max_value - decayed, 0.0)
        scaled_increment = (self.config.chemical_saturation_gain * increment) / max(max_value, 1e-8)
        saturating_delta = headroom * (1.0 - np.exp(-scaled_increment))
        updated = decayed + saturating_delta
        return np.minimum(updated, max_value)

    def _decay_cooldowns(self) -> None:
        self._split_cooldown = np.maximum(0, self._split_cooldown - 1)
        self._prune_cooldown = np.maximum(0, self._prune_cooldown - 1)

    def _select_split_indices(self) -> tuple[int, ...]:
        split_threshold, _ = self._resolve_split_prune_thresholds()
        split_threshold += self.config.split_hysteresis_margin
        current_dim = self.hidden_dim
        remaining_capacity = self.max_hidden_dim - current_dim
        if remaining_capacity <= 0 or self.config.max_split_per_sleep <= 0:
            return ()

        candidates = np.where(
            np.logical_and.reduce(
                (
                    self._hidden_chemical >= split_threshold,
                    self._split_cooldown <= 0,
                    ~self._prune_marked,
                )
            )
        )[0]
        if candidates.size == 0:
            return ()

        split_scores = self._compute_split_scores()
        sorted_candidates = candidates[np.argsort(split_scores[candidates])[::-1]]
        split_count = min(
            int(sorted_candidates.size),
            int(self.config.max_split_per_sleep),
            int(remaining_capacity),
        )
        chosen = sorted_candidates[:split_count]
        return tuple(int(index) for index in chosen.tolist())

    def _select_prune_indices(self) -> tuple[int, ...]:
        _, prune_threshold = self._resolve_split_prune_thresholds()
        prune_threshold -= self.config.prune_hysteresis_margin
        removable = self.hidden_dim - self._min_hidden_dim
        if removable <= 0 or self.config.max_prune_per_sleep <= 0:
            return ()

        candidates = np.where(
            np.logical_and.reduce(
                (
                    self._hidden_chemical <= prune_threshold,
                    ~self._prune_marked,
                    self._prune_cooldown <= 0,
                )
            )
        )[0]
        if candidates.size == 0:
            return ()

        prune_scores = self._compute_prune_scores()
        sorted_candidates = candidates[np.argsort(prune_scores[candidates])[::-1]]
        prune_count = min(
            int(sorted_candidates.size),
            int(self.config.max_prune_per_sleep),
            int(removable),
        )
        chosen = sorted_candidates[:prune_count]
        return tuple(sorted(int(index) for index in chosen.tolist()))

    def _resolve_split_prune_thresholds(self) -> tuple[float, float]:
        if not self.config.use_adaptive_thresholds:
            return self.config.split_threshold, self.config.prune_threshold

        split_threshold = float(
            np.percentile(self._hidden_chemical, self.config.adaptive_split_percentile)
        )
        prune_threshold = float(
            np.percentile(self._hidden_chemical, self.config.adaptive_prune_percentile)
        )
        return split_threshold, prune_threshold

    def _compute_split_scores(self) -> Array:
        chemical_component = self._normalize_vector(self._hidden_chemical)
        output_weight_norm = np.linalg.norm(self.weight_hidden_output, axis=1)
        norm_component = self._normalize_vector(output_weight_norm)
        importance_component = self._normalize_vector(self._importance_ema)
        norm_mix = np.clip(self.config.split_weight_norm_mix, 0.0, 1.0)
        importance_mix = np.clip(self.config.split_importance_mix, 0.0, 1.0)
        chemical_mix = max(0.0, 1.0 - norm_mix - importance_mix)
        return (
            chemical_mix * chemical_component
            + norm_mix * norm_component
            + importance_mix * importance_component
        )

    def _compute_prune_scores(self) -> Array:
        chemical_component = 1.0 - self._normalize_vector(self._hidden_chemical)
        output_weight_norm = np.linalg.norm(self.weight_hidden_output, axis=1)
        norm_component = 1.0 - self._normalize_vector(output_weight_norm)
        importance_component = 1.0 - self._normalize_vector(self._importance_ema)
        norm_mix = np.clip(self.config.prune_weight_norm_mix, 0.0, 1.0)
        importance_mix = np.clip(self.config.prune_importance_mix, 0.0, 1.0)
        chemical_mix = max(0.0, 1.0 - norm_mix - importance_mix)
        return (
            chemical_mix * chemical_component
            + norm_mix * norm_component
            + importance_mix * importance_component
        )

    def _normalize_vector(self, values: Array) -> Array:
        max_value = float(np.max(values))
        min_value = float(np.min(values))
        span = max_value - min_value
        if span <= 1e-12:
            return np.ones_like(values)
        return (values - min_value) / span

    def _normalize_vector_zero_base(self, values: Array) -> Array:
        max_value = float(np.max(values))
        min_value = float(np.min(values))
        span = max_value - min_value
        if span <= 1e-12:
            return np.zeros_like(values)
        return (values - min_value) / span

    def _derive_indices_from_policy(
        self, adaptation_policy: NeuronAdaptationPolicy
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        traffic = self.get_layer_traffic()
        proposals = adaptation_policy.propose(traffic)
        return self._indices_from_proposals(proposals)

    def _indices_from_proposals(
        self, proposals: list[NeuronChangeProposal]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        add_count = 0
        remove_indices: set[int] = set()
        for proposal in proposals:
            if proposal.layer_name != "hidden":
                continue
            add_count += max(0, int(proposal.add_count))
            for index in proposal.remove_indices:
                if 0 <= int(index) < self.hidden_dim:
                    remove_indices.add(int(index))

        split_candidates = list(self._select_split_indices())
        if add_count > len(split_candidates):
            split_rank = np.argsort(self._compute_split_scores())[::-1]
            for index in split_rank.tolist():
                if index not in split_candidates:
                    split_candidates.append(int(index))
                if len(split_candidates) >= add_count:
                    break
        split_indices = tuple(split_candidates[:add_count]) if add_count > 0 else ()
        prune_indices = tuple(sorted(remove_indices))
        return split_indices, prune_indices

    def _split_neurons(self, split_indices: tuple[int, ...]) -> None:
        if not split_indices:
            return

        new_in_columns: list[Array] = []
        new_hidden_bias_values: list[float] = []
        new_out_rows: list[Array] = []
        new_chemical_values: list[float] = []
        new_chemical_fast_values: list[float] = []
        new_chemical_slow_values: list[float] = []
        new_age_values: list[float] = []
        new_traffic_values: list[float] = []
        new_importance_values: list[float] = []
        new_split_cooldowns: list[int] = []
        new_prune_cooldowns: list[int] = []

        for index in split_indices:
            input_column = self.weight_input_hidden[:, index]
            output_row = self.weight_hidden_output[index, :].copy()
            bias_value = float(self.bias_hidden[0, index])
            chemical_value = float(self._hidden_chemical[index])
            chemical_fast_value = float(self._hidden_chemical_fast[index])
            chemical_slow_value = float(self._hidden_chemical_slow[index])
            importance_value = float(self._importance_ema[index])
            parent_norm = float(np.linalg.norm(output_row))
            noise_scale = self.config.split_noise_scale * max(parent_norm, 1e-3)
            output_noise = self._rng.normal(loc=0.0, scale=noise_scale, size=output_row.shape)
            self.weight_hidden_output[index, :] = 0.5 * output_row + output_noise

            # Keep split function-preserving by keeping duplicate activations and
            # making output rows sum to the original row.
            new_in_columns.append(input_column.astype(np.float64))
            new_out_rows.append((0.5 * output_row - output_noise).astype(np.float64))
            new_hidden_bias_values.append(bias_value)
            new_chemical_values.append(chemical_value * 0.5)
            new_chemical_fast_values.append(chemical_fast_value * 0.5)
            new_chemical_slow_values.append(chemical_slow_value * 0.5)
            new_age_values.append(0.0)
            new_traffic_values.append(0.0)
            new_importance_values.append(importance_value)
            new_split_cooldowns.append(self.config.split_cooldown_epochs)
            new_prune_cooldowns.append(self.config.prune_cooldown_epochs)

            self._hidden_chemical[index] = chemical_value * 0.5
            self._hidden_chemical_fast[index] = chemical_fast_value * 0.5
            self._hidden_chemical_slow[index] = chemical_slow_value * 0.5
            self._split_cooldown[index] = self.config.split_cooldown_epochs
            self._prune_cooldown[index] = self.config.prune_cooldown_epochs

        appended_input_columns = np.column_stack(new_in_columns)
        self.weight_input_hidden = np.hstack([self.weight_input_hidden, appended_input_columns])

        appended_output_rows = np.vstack(new_out_rows)
        self.weight_hidden_output = np.vstack([self.weight_hidden_output, appended_output_rows])

        appended_bias = np.array(new_hidden_bias_values, dtype=np.float64).reshape(1, -1)
        self.bias_hidden = np.hstack([self.bias_hidden, appended_bias])

        self._hidden_chemical = np.concatenate(
            [self._hidden_chemical, np.array(new_chemical_values, dtype=np.float64)]
        )
        self._hidden_chemical_fast = np.concatenate(
            [self._hidden_chemical_fast, np.array(new_chemical_fast_values, dtype=np.float64)]
        )
        self._hidden_chemical_slow = np.concatenate(
            [self._hidden_chemical_slow, np.array(new_chemical_slow_values, dtype=np.float64)]
        )
        self._neuron_age = np.concatenate(
            [self._neuron_age, np.array(new_age_values, dtype=np.float64)]
        )
        self._traffic_sum = np.concatenate(
            [self._traffic_sum, np.array(new_traffic_values, dtype=np.float64)]
        )
        self._importance_ema = np.concatenate(
            [self._importance_ema, np.array(new_importance_values, dtype=np.float64)]
        )
        self._prune_ttl = np.concatenate(
            [self._prune_ttl, np.zeros(len(split_indices), dtype=np.int32)]
        )
        self._prune_marked = np.concatenate(
            [self._prune_marked, np.zeros(len(split_indices), dtype=bool)]
        )
        self._split_cooldown = np.concatenate(
            [self._split_cooldown, np.array(new_split_cooldowns, dtype=np.int32)]
        )
        self._prune_cooldown = np.concatenate(
            [self._prune_cooldown, np.array(new_prune_cooldowns, dtype=np.int32)]
        )

    def _schedule_or_prune(self, prune_indices: tuple[int, ...]) -> None:
        if not prune_indices:
            return
        if self.config.prune_decay_steps <= 1:
            self._remove_neurons(prune_indices)
            return

        for index in prune_indices:
            if 0 <= index < self.hidden_dim:
                self._prune_marked[index] = True
                self._prune_ttl[index] = self.config.prune_decay_steps
                self._prune_cooldown[index] = self.config.prune_cooldown_epochs

    def _apply_prune_decay_step(self) -> None:
        if self.config.prune_decay_steps <= 1:
            return
        active_indices = np.where(self._prune_marked)[0]
        if active_indices.size == 0:
            return

        decay_factor = self.config.prune_decay_factor
        self.weight_input_hidden[:, active_indices] *= decay_factor
        self.weight_hidden_output[active_indices, :] *= decay_factor
        self.bias_hidden[:, active_indices] *= decay_factor
        self._hidden_chemical[active_indices] *= decay_factor
        self._hidden_chemical_fast[active_indices] *= decay_factor
        self._hidden_chemical_slow[active_indices] *= decay_factor
        self._traffic_sum[active_indices] *= decay_factor
        self._importance_ema[active_indices] *= decay_factor

        self._prune_ttl[active_indices] -= 1
        finalize_indices = np.where(np.logical_and(self._prune_marked, self._prune_ttl <= 0))[0]
        if finalize_indices.size > 0:
            self._remove_neurons(tuple(int(index) for index in finalize_indices.tolist()))

    def _remove_neurons(self, prune_indices: tuple[int, ...]) -> None:
        unique_indices = sorted({index for index in prune_indices if 0 <= index < self.hidden_dim})
        if not unique_indices:
            return
        if self.hidden_dim - len(unique_indices) < self._min_hidden_dim:
            allowed = self.hidden_dim - self._min_hidden_dim
            unique_indices = unique_indices[:allowed]
        if not unique_indices:
            return

        mask = np.ones(self.hidden_dim, dtype=bool)
        mask[unique_indices] = False

        self.weight_input_hidden = self.weight_input_hidden[:, mask]
        self.weight_hidden_output = self.weight_hidden_output[mask, :]
        self.bias_hidden = self.bias_hidden[:, mask]
        self._hidden_chemical = self._hidden_chemical[mask]
        self._hidden_chemical_fast = self._hidden_chemical_fast[mask]
        self._hidden_chemical_slow = self._hidden_chemical_slow[mask]
        self._neuron_age = self._neuron_age[mask]
        self._traffic_sum = self._traffic_sum[mask]
        self._importance_ema = self._importance_ema[mask]
        self._prune_ttl = self._prune_ttl[mask]
        self._prune_marked = self._prune_marked[mask]
        self._split_cooldown = self._split_cooldown[mask]
        self._prune_cooldown = self._prune_cooldown[mask]

    def _apply_homeostatic_downscaling(self) -> None:
        scale_factor = self.config.homeostatic_downscale_factor
        if abs(scale_factor - 1.0) > 1e-12:
            self.weight_input_hidden *= scale_factor
            self.weight_hidden_output *= scale_factor
            self.bias_hidden *= scale_factor
            self.bias_output *= scale_factor

        self._match_target_norms()

    def _match_target_norms(self) -> None:
        if self.config.homeostasis_target_input_norm > 0.0:
            self.weight_input_hidden = self._scale_columns_to_target(
                matrix=self.weight_input_hidden,
                target_norm=self.config.homeostasis_target_input_norm,
                strength=self.config.homeostasis_strength,
            )
        if self.config.homeostasis_target_output_norm > 0.0:
            self.weight_hidden_output = self._scale_rows_to_target(
                matrix=self.weight_hidden_output,
                target_norm=self.config.homeostasis_target_output_norm,
                strength=self.config.homeostasis_strength,
            )

    def _scale_columns_to_target(self, matrix: Array, target_norm: float, strength: float) -> Array:
        norms = np.linalg.norm(matrix, axis=0)
        safe_norms = np.maximum(norms, 1e-8)
        raw_scale = np.power(target_norm / safe_norms, strength)
        scale = np.clip(raw_scale, 0.5, 2.0)
        return matrix * scale[np.newaxis, :]

    def _scale_rows_to_target(self, matrix: Array, target_norm: float, strength: float) -> Array:
        norms = np.linalg.norm(matrix, axis=1)
        safe_norms = np.maximum(norms, 1e-8)
        raw_scale = np.power(target_norm / safe_norms, strength)
        scale = np.clip(raw_scale, 0.5, 2.0)
        return matrix * scale[:, np.newaxis]

    def _store_replay_snapshot(self, input_batch: Array, target_batch: Array) -> None:
        if self.config.replay_memory_size <= 0:
            return
        output_prediction = self.predict_proba(input_batch)
        priority = float(np.mean(np.abs(output_prediction - target_batch)))
        positive_fraction = float(np.mean(target_batch))
        self._replay_memory.append(
            ReplaySnapshot(
                input_batch=input_batch.copy(),
                target_batch=target_batch.copy(),
                priority=priority,
                positive_fraction=positive_fraction,
            )
        )

    def _run_replay_consolidation(self) -> None:
        if self.config.replay_steps <= 0 or len(self._replay_memory) == 0:
            return

        replay_count = min(self.config.replay_steps, len(self._replay_memory))
        replay_snapshots = self._select_replay_snapshots(replay_count)
        for snapshot in replay_snapshots:
            self._run_training_step(
                input_batch=snapshot.input_batch,
                target_batch=snapshot.target_batch,
                learning_rate=self.config.replay_learning_rate,
                inference_steps=self.config.replay_inference_steps,
                inference_learning_rate=self.config.replay_inference_learning_rate,
                update_epoch_state=False,
                store_replay_snapshot=False,
            )

    def _select_replay_snapshots(self, replay_count: int) -> list[ReplaySnapshot]:
        if replay_count <= 0:
            return []
        snapshots = list(self._replay_memory)
        if not self.config.replay_prioritized:
            return snapshots[-replay_count:]

        sorted_snapshots = sorted(snapshots, key=lambda snapshot: snapshot.priority, reverse=True)
        if not self.config.replay_class_balanced:
            return sorted_snapshots[:replay_count]

        positive = [snapshot for snapshot in sorted_snapshots if snapshot.positive_fraction >= 0.5]
        negative = [snapshot for snapshot in sorted_snapshots if snapshot.positive_fraction < 0.5]
        if not positive or not negative:
            return sorted_snapshots[:replay_count]

        selected: list[ReplaySnapshot] = []
        positive_index = 0
        negative_index = 0
        while len(selected) < replay_count:
            if positive_index < len(positive):
                selected.append(positive[positive_index])
                positive_index += 1
                if len(selected) >= replay_count:
                    break
            if negative_index < len(negative):
                selected.append(negative[negative_index])
                negative_index += 1
                if len(selected) >= replay_count:
                    break
            if positive_index >= len(positive) and negative_index >= len(negative):
                break
        return selected

    def _validate_config(self, config: CircadianConfig) -> None:
        if not (0.0 <= config.chemical_decay <= 1.0):
            raise ValueError("chemical_decay must be between 0 and 1")
        if config.chemical_max_value <= 0.0:
            raise ValueError("chemical_max_value must be positive")
        if config.chemical_saturation_gain <= 0.0:
            raise ValueError("chemical_saturation_gain must be positive")
        if not (0.0 <= config.slow_chemical_decay <= 1.0):
            raise ValueError("slow_chemical_decay must be between 0 and 1")
        if config.chemical_buildup_rate <= 0.0:
            raise ValueError("chemical_buildup_rate must be positive")
        if config.slow_buildup_scale <= 0.0:
            raise ValueError("slow_buildup_scale must be positive")
        if not (0.0 <= config.dual_fast_mix <= 1.0):
            raise ValueError("dual_fast_mix must be between 0 and 1")
        if config.plasticity_sensitivity <= 0.0:
            raise ValueError("plasticity_sensitivity must be positive")
        if config.plasticity_sensitivity_min <= 0.0:
            raise ValueError("plasticity_sensitivity_min must be positive")
        if config.plasticity_sensitivity_max < config.plasticity_sensitivity_min:
            raise ValueError(
                "plasticity_sensitivity_max must be greater than or equal to plasticity_sensitivity_min"
            )
        if not (0.0 <= config.plasticity_importance_mix <= 1.0):
            raise ValueError("plasticity_importance_mix must be between 0 and 1")
        if not (0.0 < config.min_plasticity <= 1.0):
            raise ValueError("min_plasticity must be in (0, 1]")
        if not (0.0 <= config.split_weight_norm_mix <= 1.0):
            raise ValueError("split_weight_norm_mix must be between 0 and 1")
        if not (0.0 <= config.prune_weight_norm_mix <= 1.0):
            raise ValueError("prune_weight_norm_mix must be between 0 and 1")
        if not (0.0 <= config.adaptive_split_percentile <= 100.0):
            raise ValueError("adaptive_split_percentile must be between 0 and 100")
        if not (0.0 <= config.adaptive_prune_percentile <= 100.0):
            raise ValueError("adaptive_prune_percentile must be between 0 and 100")
        if config.split_hysteresis_margin < 0.0 or config.prune_hysteresis_margin < 0.0:
            raise ValueError("split/prune hysteresis margins must be non-negative")
        if config.split_cooldown_epochs < 0 or config.prune_cooldown_epochs < 0:
            raise ValueError("split/prune cooldown epochs must be non-negative")
        if not (0.0 <= config.split_importance_mix <= 1.0):
            raise ValueError("split_importance_mix must be between 0 and 1")
        if not (0.0 <= config.prune_importance_mix <= 1.0):
            raise ValueError("prune_importance_mix must be between 0 and 1")
        if not (0.0 <= config.importance_ema_decay < 1.0):
            raise ValueError("importance_ema_decay must be in [0, 1)")
        if config.min_epochs_between_sleep < 0:
            raise ValueError("min_epochs_between_sleep must be non-negative")
        if config.sleep_energy_window <= 1:
            raise ValueError("sleep_energy_window must be greater than 1")
        if config.sleep_plateau_delta < 0.0:
            raise ValueError("sleep_plateau_delta must be non-negative")
        if config.sleep_chemical_variance_threshold < 0.0:
            raise ValueError("sleep_chemical_variance_threshold must be non-negative")
        if config.max_split_per_sleep < 0 or config.max_prune_per_sleep < 0:
            raise ValueError("max split/prune per sleep must be non-negative")
        if config.split_noise_scale < 0.0:
            raise ValueError("split_noise_scale must be non-negative")
        if not (0.0 <= config.sleep_reset_factor <= 1.0):
            raise ValueError("sleep_reset_factor must be between 0 and 1")
        if config.prune_decay_steps < 1:
            raise ValueError("prune_decay_steps must be at least 1")
        if not (0.0 < config.prune_decay_factor <= 1.0):
            raise ValueError("prune_decay_factor must be in (0, 1]")
        if not (0.0 < config.homeostatic_downscale_factor <= 1.0):
            raise ValueError("homeostatic_downscale_factor must be in (0, 1]")
        if config.homeostasis_target_input_norm < 0.0:
            raise ValueError("homeostasis_target_input_norm must be non-negative")
        if config.homeostasis_target_output_norm < 0.0:
            raise ValueError("homeostasis_target_output_norm must be non-negative")
        if not (0.0 < config.homeostasis_strength <= 1.0):
            raise ValueError("homeostasis_strength must be in (0, 1]")
        if config.replay_steps < 0:
            raise ValueError("replay_steps must be non-negative")
        if config.replay_memory_size < 0:
            raise ValueError("replay_memory_size must be non-negative")
        if config.replay_steps > 0:
            if config.replay_learning_rate <= 0.0:
                raise ValueError("replay_learning_rate must be positive")
            if config.replay_inference_steps <= 0:
                raise ValueError("replay_inference_steps must be positive")
            if config.replay_inference_learning_rate <= 0.0:
                raise ValueError("replay_inference_learning_rate must be positive")
