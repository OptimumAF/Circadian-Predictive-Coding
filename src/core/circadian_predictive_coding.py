"""Circadian predictive-coding network with chemical homeostasis and sleep events."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field, fields
from hashlib import sha256
from math import isfinite
from numbers import Integral
from time import perf_counter
from typing import Any, Self

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
    NeuronLineageSnapshot,
    PruneOutcome,
)
from src.core.predictive_coding import PredictiveCodingTrainResult
from src.core.replay_retention import (
    DEFAULT_REPLAY_RETENTION_POLICY,
    ReplayExposureSnapshot,
    ReplayRetentionPolicy,
)
from src.core.sleep_clocks import SleepClockSnapshot, SleepEpochProgress
from src.core.sleep_telemetry import (
    ChemicalSummaries,
    ChemicalSummary,
    SleepBudgets,
    SleepDurations,
    SleepEventTelemetry,
    SleepReplayUsage,
    SleepStructuralChanges,
)
from src.core.training_validation import (
    require_finite_training_arrays,
    validate_binary_training_batch,
    validate_positive_finite_learning_rate,
)
from src.core.dimension_validation import require_positive_integer_dimension

Array = NDArray[np.float64]
NUMPY_CIRCADIAN_ENERGY_ID = "numpy_circadian_bce_plus_half_mean_final_hidden_error_sq_v1"


@dataclass(frozen=True)
class ReplaySnapshot:
    """Stored wake snapshot for optional sleep replay consolidation."""

    input_batch: Array
    target_batch: Array
    priority: float
    positive_fraction: float


def _summarize_chemical(values: Array) -> ChemicalSummary:
    """Copy only scalar statistics from an aligned NumPy chemical vector."""
    return ChemicalSummary(
        count=int(values.size),
        minimum=float(np.min(values)),
        mean=float(np.mean(values)),
        maximum=float(np.max(values)),
    )


@dataclass(frozen=True)
class ReplayRetentionBudget:
    """Hard limits on unique labeled examples held for optional sleep replay."""

    max_examples: int
    max_bytes: int

    def __post_init__(self) -> None:
        if type(self.max_examples) is not int or self.max_examples <= 0:
            raise ValueError("replay example budget must be a positive integer")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("replay byte budget must be a positive integer")


WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY = "wake_only_adaptive_v1"


@dataclass(frozen=True)
class ReplayRetentionSnapshot:
    """Content IDs and array storage actually retained at a phase boundary."""

    sample_ids: tuple[str, ...]
    example_count: int
    retained_bytes: int


def replay_sample_id(input_row: Array, target_row: Array) -> str:
    """Identify one observed labeled row without exposing its raw values."""
    if input_row.ndim != 2 or input_row.shape[0] != 1 or target_row.shape != (1, 1):
        raise ValueError("replay sample ID requires one input and one target row")
    digest = sha256(b"numpy_labeled_replay_row_v1")
    for values in (input_row, target_row):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


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

    # Optional reward modulation: scale wake updates toward harder batches.
    use_reward_modulated_learning: bool = False
    reward_baseline_decay: float = 0.95
    reward_difficulty_exponent: float = 1.0
    reward_scale_min: float = 0.75
    reward_scale_max: float = 1.5

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
    sleep_warmup_steps: int = 0
    sleep_split_only_until_fraction: float = 0.50
    sleep_prune_only_after_fraction: float = 0.85
    sleep_max_change_fraction: float = 1.0
    sleep_min_change_count: int = 1
    prune_min_age_steps: int = 0

    # Adaptive sleep trigger (optional) based on energy plateau and chemistry variance.
    use_adaptive_sleep_trigger: bool = False
    min_epochs_between_sleep: int = 10
    sleep_energy_window: int = 8
    sleep_plateau_delta: float = 1e-3
    sleep_chemical_variance_threshold: float = 0.02

    # Optional adaptive budget scaling for split/prune counts during sleep.
    use_adaptive_sleep_budget: bool = False
    adaptive_sleep_budget_min_scale: float = 0.25
    adaptive_sleep_budget_max_scale: float = 1.0
    adaptive_sleep_budget_plateau_weight: float = 0.6
    adaptive_sleep_budget_variance_weight: float = 0.4

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

    # Legacy preserves budget-gated historical sleep. Components separate
    # consolidation from structure; disabled rejects even forced events.
    sleep_mode: str = "legacy"
    sleep_enable_chemical_reset: bool = True
    sleep_enable_replay: bool = True
    sleep_enable_homeostasis: bool = True
    sleep_enable_split: bool = True
    sleep_enable_prune: bool = True

    @classmethod
    def matched_pc_control(cls) -> Self:
        """Return the shallow PC parity control with circadian effects disabled.

        Chemistry can still be observed, but clipping plasticity to one
        prevents it from changing wake updates. Sleep has zero budgets and
        replay has no memory, so even a forced sleep event is a no-op.
        """
        return cls(
            use_saturating_chemical=False,
            use_dual_chemical=False,
            use_adaptive_plasticity_sensitivity=False,
            min_plasticity=1.0,
            use_reward_modulated_learning=False,
            use_adaptive_thresholds=False,
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            split_noise_scale=0.0,
            sleep_reset_factor=1.0,
            use_adaptive_sleep_trigger=False,
            use_adaptive_sleep_budget=False,
            homeostatic_downscale_factor=1.0,
            homeostasis_target_input_norm=0.0,
            homeostasis_target_output_norm=0.0,
            replay_steps=0,
            replay_memory_size=0,
        )


@dataclass(frozen=True)
class SleepEventResult:
    """Result summary for one sleep consolidation event."""

    old_hidden_dim: int
    new_hidden_dim: int
    split_indices: tuple[int, ...]
    pruned_indices: tuple[int, ...]
    performed: bool = False
    lineage_before: NeuronLineageSnapshot | None = None
    lineage_after: NeuronLineageSnapshot | None = None
    prune_outcome: PruneOutcome | None = None
    telemetry: SleepEventTelemetry | None = dataclass_field(default=None, compare=False)


@dataclass(frozen=True)
class CircadianTrainResult(PredictiveCodingTrainResult):
    """Training diagnostic with removals finalized during this wake update."""

    energy_definition: str = NUMPY_CIRCADIAN_ENERGY_ID
    prune_outcome: PruneOutcome = dataclass_field(default_factory=PruneOutcome)


@dataclass(frozen=True)
class CircadianNetworkSnapshot:
    """Detached in-memory state for one compatible NumPy circadian model."""

    format_version: int
    input_dim: int
    initial_hidden_dims: tuple[int, ...]
    min_hidden_dim: int
    max_hidden_dim: int
    config: CircadianConfig
    state: dict[str, Any]


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
        hidden_dims: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        input_dim = require_positive_integer_dimension(input_dim, "input_dim")
        min_hidden_dim = require_positive_integer_dimension(min_hidden_dim, "min_hidden_dim")
        resolved_hidden_dims = self._resolve_hidden_dims(
            hidden_dim=hidden_dim,
            hidden_dims=hidden_dims,
        )
        adaptive_hidden_dim = resolved_hidden_dims[-1]
        pre_hidden_dims = resolved_hidden_dims[:-1]
        self.input_dim = input_dim
        if min_hidden_dim > adaptive_hidden_dim:
            raise ValueError("min_hidden_dim cannot exceed initial hidden_dim")

        config = circadian_config or CircadianConfig()
        self._validate_config(config)
        self.config = config

        self.max_hidden_dim = (
            max_hidden_dim if max_hidden_dim is not None else max(adaptive_hidden_dim * 4, 16)
        )
        self.max_hidden_dim = require_positive_integer_dimension(
            self.max_hidden_dim, "max_hidden_dim"
        )
        if self.max_hidden_dim < adaptive_hidden_dim:
            raise ValueError("max_hidden_dim cannot be smaller than initial hidden_dim")

        rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(seed + 10_001)
        self._pre_hidden_weights: list[Array] = []
        self._pre_hidden_biases: list[Array] = []
        previous_dim = input_dim
        for layer_dim in pre_hidden_dims:
            self._pre_hidden_weights.append(rng.normal(0.0, 0.5, size=(previous_dim, layer_dim)))
            self._pre_hidden_biases.append(np.zeros((1, layer_dim), dtype=np.float64))
            previous_dim = layer_dim

        self.weight_input_hidden = rng.normal(0.0, 0.5, size=(previous_dim, adaptive_hidden_dim))
        self.bias_hidden = np.zeros((1, adaptive_hidden_dim), dtype=np.float64)
        self.weight_hidden_output = rng.normal(0.0, 0.5, size=(adaptive_hidden_dim, 1))
        self.bias_output = np.zeros((1, 1), dtype=np.float64)
        self.hidden_dims = tuple(resolved_hidden_dims)
        self.pre_hidden_dims = tuple(pre_hidden_dims)

        self._hidden_chemical = np.zeros(adaptive_hidden_dim, dtype=np.float64)
        self._hidden_chemical_fast = np.zeros(adaptive_hidden_dim, dtype=np.float64)
        self._hidden_chemical_slow = np.zeros(adaptive_hidden_dim, dtype=np.float64)
        self._neuron_age = np.zeros(adaptive_hidden_dim, dtype=np.float64)
        self._traffic_sum = np.zeros(adaptive_hidden_dim, dtype=np.float64)
        self._importance_ema = np.zeros(adaptive_hidden_dim, dtype=np.float64)
        self._neuron_ids = np.arange(adaptive_hidden_dim, dtype=np.int64)
        self._parent_ids = np.full(adaptive_hidden_dim, -1, dtype=np.int64)
        self._next_neuron_id = adaptive_hidden_dim
        self._traffic_steps = 0
        self._min_hidden_dim = min_hidden_dim
        self._prune_ttl = np.zeros(adaptive_hidden_dim, dtype=np.int32)
        self._prune_marked = np.zeros(adaptive_hidden_dim, dtype=bool)
        self._split_cooldown = np.zeros(adaptive_hidden_dim, dtype=np.int32)
        self._prune_cooldown = np.zeros(adaptive_hidden_dim, dtype=np.int32)

        self._epoch_count = 0
        self._epochs_since_sleep = 0
        self._wake_examples = 0
        self._replay_updates = 0
        self._sleep_events = 0
        self._energy_history: list[float] = []
        self._reward_error_ema: float | None = None
        self._last_reward_scale = 1.0
        self._replay_memory: deque[ReplaySnapshot] = deque(maxlen=self.config.replay_memory_size)

    @property
    def hidden_dim(self) -> int:
        """Current hidden-layer width."""
        return int(self.weight_input_hidden.shape[1])

    def get_sleep_clocks(self) -> SleepClockSnapshot:
        """Return successful wake, replay, and sleep counts without runner epochs."""
        return SleepClockSnapshot(
            wake_batches=self._epoch_count,
            wake_examples=self._wake_examples,
            wake_batches_since_sleep=self._epochs_since_sleep,
            replay_updates=self._replay_updates,
            sleep_events=self._sleep_events,
        )

    def configure_replay_retention(
        self, budget: ReplayRetentionBudget, *, policy: ReplayRetentionPolicy | None = None
    ) -> None:
        """Enable per-example retention before the first wake update."""
        if not isinstance(budget, ReplayRetentionBudget):
            raise TypeError("replay retention requires a ReplayRetentionBudget")
        if policy is not None and type(policy) is not ReplayRetentionPolicy:
            raise TypeError("replay retention policy must be a ReplayRetentionPolicy")
        if (
            self._epoch_count != 0
            or self._replay_memory
            or hasattr(self, "_replay_retention_budget")
        ):
            raise ValueError("replay retention must be configured once before training")
        # Why this: keep historical batch-snapshot state/checkpoints unchanged
        # unless a versioned caller explicitly selects bounded retention.
        self._replay_retention_budget = budget
        if policy is not None and policy != DEFAULT_REPLAY_RETENTION_POLICY:
            self._replay_retention_policy = policy
            # Why this: exposure reporting is an explicit v8 policy cost. Old
            # bounded and batch-snapshot checkpoint field sets stay identical.
            self._replay_observed_ids: set[str] = set()
            self._replay_duplicate_ids: set[str] = set()
            self._replay_duplicate_occurrences = 0
            self._replay_exposed_ids: set[str] = set()
            self._replay_exposure_updates = 0
        self._replay_memory = deque()

    def configure_replay_side_effect_policy(self, policy: str) -> None:
        """Opt into wake-only adaptive state for replay before any wake update."""
        if policy != WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY or type(policy) is not str:
            raise ValueError("unsupported replay side-effect policy")
        if (
            self._epoch_count != 0
            or self._replay_memory
            or hasattr(self, "_replay_side_effect_policy")
        ):
            raise ValueError("replay side-effect policy must be configured once before training")
        # Why this: historical models retain their original snapshot field
        # set and config digest; only versioned callers add the policy field.
        self._replay_side_effect_policy = policy

    def get_replay_side_effect_policy(self) -> str:
        """Name the active replay side-effect contract."""
        return getattr(self, "_replay_side_effect_policy", "historical")

    def get_replay_retention(self) -> ReplayRetentionSnapshot:
        """Describe the bounded labeled examples currently held for replay."""
        if not hasattr(self, "_replay_retention_budget"):
            raise ValueError("replay retention budget is not configured")
        ids = tuple(
            sorted(
                replay_sample_id(item.input_batch, item.target_batch)
                for item in self._replay_memory
            )
        )
        return ReplayRetentionSnapshot(
            sample_ids=ids,
            example_count=len(ids),
            retained_bytes=sum(
                item.input_batch.nbytes + item.target_batch.nbytes for item in self._replay_memory
            ),
        )

    def preview_unprioritized_replay_ids(self, replay_count: int) -> tuple[str, ...]:
        """Expose the exact retained-order selection before a matched sleep."""
        if not hasattr(self, "_replay_retention_policy") or self.config.replay_prioritized:
            raise ValueError("replay ID preview requires an explicit unprioritized policy")
        if type(replay_count) is not int or replay_count <= 0:
            raise ValueError("replay ID preview count must be positive")
        return tuple(
            replay_sample_id(item.input_batch, item.target_batch)
            for item in self._select_replay_snapshots(replay_count)
        )

    def get_replay_retained_order_ids(self) -> tuple[str, ...]:
        """Expose bounded buffer order for a prediction-free replay preflight."""
        if not hasattr(self, "_replay_retention_policy"):
            raise ValueError("replay order requires an explicit retention policy")
        return tuple(
            replay_sample_id(item.input_batch, item.target_batch) for item in self._replay_memory
        )

    def get_replay_exposure(self) -> ReplayExposureSnapshot:
        """Describe distinct observed and applied replay IDs for a selected policy."""
        if not hasattr(self, "_replay_retention_policy"):
            raise ValueError("replay exposure requires an explicit retention policy")
        return ReplayExposureSnapshot(
            observed_ids=tuple(sorted(self._replay_observed_ids)),
            duplicate_ids=tuple(sorted(self._replay_duplicate_ids)),
            duplicate_occurrences=self._replay_duplicate_occurrences,
            exposed_ids=tuple(sorted(self._replay_exposed_ids)),
            replay_updates=self._replay_exposure_updates,
        )

    def get_neuron_lineage(self) -> NeuronLineageSnapshot:
        """Return active IDs; a parent ID remains meaningful after its prune."""
        return NeuronLineageSnapshot(
            neuron_ids=tuple(int(value) for value in self._neuron_ids),
            parent_ids=tuple(None if value < 0 else int(value) for value in self._parent_ids),
            next_neuron_id=self._next_neuron_id,
        )

    def get_pending_prune_ids(self) -> tuple[int, ...]:
        """Return active IDs marked for delayed removal."""
        if (
            self._prune_marked.shape != self._neuron_ids.shape
            or self._prune_marked.dtype != np.bool_
        ):
            raise ValueError("pending prune mask has incompatible shape or dtype")
        return tuple(int(value) for value in self._neuron_ids[self._prune_marked])

    def snapshot_state(self) -> CircadianNetworkSnapshot:
        """Copy every model-owned value, including replay arrays and RNG state."""
        # Why this: topology and replay share many mutable arrays; a complete
        # in-memory dictionary copy avoids omitting a newly added state field.
        return CircadianNetworkSnapshot(
            format_version=2,
            input_dim=self.input_dim,
            initial_hidden_dims=self.hidden_dims,
            min_hidden_dim=self._min_hidden_dim,
            max_hidden_dim=self.max_hidden_dim,
            config=self.config,
            state=deepcopy(self.__dict__),
        )

    def restore_state(self, snapshot: CircadianNetworkSnapshot) -> None:
        """Restore a detached compatible state after validating its topology."""
        if not isinstance(snapshot, CircadianNetworkSnapshot) or snapshot.format_version != 2:
            raise ValueError("Unsupported circadian snapshot version")
        if (
            snapshot.input_dim != self.input_dim
            or snapshot.initial_hidden_dims != self.hidden_dims
            or snapshot.min_hidden_dim != self._min_hidden_dim
            or snapshot.max_hidden_dim != self.max_hidden_dim
            or snapshot.config != self.config
        ):
            raise ValueError("Circadian snapshot config or model dimensions are incompatible")
        restored = deepcopy(snapshot.state)
        if restored.keys() != self.__dict__.keys():
            raise ValueError("Circadian snapshot fields are incompatible")
        if restored.get("_replay_side_effect_policy") != getattr(
            self, "_replay_side_effect_policy", None
        ):
            raise ValueError("Circadian snapshot replay side-effect policy is incompatible")
        if (
            restored["input_dim"] != snapshot.input_dim
            or restored["hidden_dims"] != snapshot.initial_hidden_dims
            or restored["_min_hidden_dim"] != snapshot.min_hidden_dim
            or restored["max_hidden_dim"] != snapshot.max_hidden_dim
            or restored["config"] != snapshot.config
        ):
            raise ValueError("Circadian snapshot metadata are inconsistent")
        candidate = object.__new__(type(self))
        candidate.__dict__.update(restored)
        candidate._validate_training_topology()
        if (
            not isinstance(candidate._rng, np.random.Generator)
            or not isinstance(candidate._replay_memory, deque)
            or getattr(candidate, "_replay_retention_budget", None)
            != getattr(self, "_replay_retention_budget", None)
            or getattr(candidate, "_replay_retention_policy", None)
            != getattr(self, "_replay_retention_policy", None)
            or candidate._replay_memory.maxlen
            != (
                None
                if hasattr(self, "_replay_retention_budget")
                else self.config.replay_memory_size
            )
        ):
            raise ValueError("Circadian snapshot replay or RNG state is incompatible")
        if hasattr(self, "_replay_retention_budget"):
            retained = candidate.get_replay_retention()
            budget = self._replay_retention_budget
            if (
                retained.example_count > budget.max_examples
                or retained.retained_bytes > budget.max_bytes
                or len(set(retained.sample_ids)) != retained.example_count
            ):
                raise ValueError("Circadian snapshot replay exceeds its observed budget")
        if hasattr(self, "_replay_retention_policy"):
            try:
                exposure = candidate.get_replay_exposure()
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError("Circadian snapshot replay exposure is incompatible") from error
            if exposure.replay_updates > candidate._replay_updates or not set(
                retained.sample_ids
            ).issubset(exposure.observed_ids):
                raise ValueError("Circadian snapshot replay exposure is inconsistent")
        self.__dict__.clear()
        self.__dict__.update(restored)

    def train_epoch(
        self,
        input_batch: Array,
        target_batch: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> CircadianTrainResult:
        pending_before = self.get_pending_prune_ids()
        energy = self._run_training_step(
            input_batch=input_batch,
            target_batch=target_batch,
            learning_rate=learning_rate,
            inference_steps=inference_steps,
            inference_learning_rate=inference_learning_rate,
            update_epoch_state=True,
            store_replay_snapshot=True,
        )
        removed: tuple[int, ...] = ()
        if pending_before:
            active_ids = set(int(value) for value in self._neuron_ids)
            removed = tuple(value for value in pending_before if value not in active_ids)
        return CircadianTrainResult(
            energy=energy,
            energy_definition=NUMPY_CIRCADIAN_ENERGY_ID,
            prune_outcome=PruneOutcome(removed_neuron_ids=removed),
        )

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
        validate_positive_finite_learning_rate(learning_rate)
        if not isinstance(inference_steps, Integral) or inference_steps <= 0:
            raise ValueError("inference_steps must be a positive integer")
        if not isfinite(inference_learning_rate) or inference_learning_rate <= 0.0:
            raise ValueError("inference_learning_rate must be positive and finite")
        validate_binary_training_batch(input_batch, target_batch, self.input_dim)
        self._validate_training_topology()
        require_finite_training_arrays(
            [
                *self._pre_hidden_weights,
                *self._pre_hidden_biases,
                self.weight_input_hidden,
                self.bias_hidden,
                self.weight_hidden_output,
                self.bias_output,
                self._hidden_chemical,
                self._hidden_chemical_fast,
                self._hidden_chemical_slow,
                self._importance_ema,
                self._traffic_sum,
                self._neuron_age,
            ],
            "model state before training",
        )
        prune_state = self._snapshot_active_prune_state()
        try:
            return self._run_validated_training_step(
                input_batch,
                target_batch,
                learning_rate,
                inference_steps,
                inference_learning_rate,
                update_epoch_state,
                store_replay_snapshot,
            )
        except FloatingPointError:
            if prune_state is not None:
                for name, original in prune_state.items():
                    setattr(self, name, original)
            raise

    def _validate_training_topology(self) -> None:
        """Reject misaligned parameter and adaptive widths before wake mutation."""
        if len(self._pre_hidden_weights) != len(self._pre_hidden_biases):
            raise ValueError("model topology has mismatched pre-hidden layers")
        previous_width = self.input_dim
        for weight, bias in zip(self._pre_hidden_weights, self._pre_hidden_biases):
            if weight.ndim != 2 or weight.shape[0] != previous_width or weight.shape[1] <= 0:
                raise ValueError("model topology has incompatible pre-hidden weight width")
            previous_width = weight.shape[1]
            if bias.shape != (1, previous_width):
                raise ValueError("model topology has incompatible pre-hidden bias width")
        input_weight = self.weight_input_hidden
        if (
            input_weight.ndim != 2
            or input_weight.shape[0] != previous_width
            or input_weight.shape[1] <= 0
        ):
            raise ValueError("model topology has incompatible adaptive input width")
        width = input_weight.shape[1]
        if (
            self.bias_hidden.shape != (1, width)
            or self.weight_hidden_output.shape != (width, 1)
            or self.bias_output.shape != (1, 1)
        ):
            raise ValueError("model topology has incompatible output or bias width")
        for name in (
            "_hidden_chemical",
            "_hidden_chemical_fast",
            "_hidden_chemical_slow",
            "_importance_ema",
            "_traffic_sum",
            "_neuron_age",
            "_split_cooldown",
            "_prune_cooldown",
            "_prune_ttl",
            "_prune_marked",
            "_neuron_ids",
            "_parent_ids",
        ):
            if getattr(self, name).shape != (width,):
                raise ValueError(f"model topology has incompatible {name} width")
        if (
            self._prune_marked.dtype != np.bool_
            or self._prune_ttl.dtype != np.int32
            or np.any(self._prune_ttl < 0)
            or np.any(self._prune_ttl[self._prune_marked] <= 0)
            or np.any(self._prune_ttl[~self._prune_marked] != 0)
            or width - int(np.count_nonzero(self._prune_marked)) < self._min_hidden_dim
        ):
            raise ValueError("model topology has invalid pending prune mask or ttl")
        if (
            self._neuron_ids.dtype != np.int64
            or self._parent_ids.dtype != np.int64
            or type(self._next_neuron_id) is not int
            or self._next_neuron_id <= 0
            or np.any(self._neuron_ids < 0)
            or np.any(self._neuron_ids >= self._next_neuron_id)
            or np.unique(self._neuron_ids).size != width
            or np.any(self._parent_ids < -1)
            or np.any(self._parent_ids >= self._neuron_ids)
        ):
            raise ValueError("model topology has invalid neuron lineage")

    def _snapshot_active_prune_state(self) -> dict[str, Array] | None:
        if not np.any(self._prune_marked):
            return None
        names = (
            "weight_input_hidden",
            "bias_hidden",
            "weight_hidden_output",
            "_hidden_chemical",
            "_hidden_chemical_fast",
            "_hidden_chemical_slow",
            "_importance_ema",
            "_traffic_sum",
            "_neuron_age",
            "_split_cooldown",
            "_prune_cooldown",
            "_prune_ttl",
            "_prune_marked",
            "_neuron_ids",
            "_parent_ids",
        )
        return {name: getattr(self, name).copy() for name in names}

    def _run_validated_training_step(
        self,
        input_batch: Array,
        target_batch: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
        update_epoch_state: bool,
        store_replay_snapshot: bool,
    ) -> float:

        initial_hidden_dim = self.hidden_dim
        wake_only_replay = (
            not update_epoch_state
            and not store_replay_snapshot
            and self.get_replay_side_effect_policy() == WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY
        )
        if not wake_only_replay:
            self._apply_prune_decay_step()

        pre_hidden_linears, pre_hidden_activations, adaptive_input = self._forward_pre_hidden(
            input_batch
        )
        if any(not np.all(np.isfinite(linear)) for linear in pre_hidden_linears):
            raise FloatingPointError("nonfinite latent prior during relaxation")
        hidden_linear_prior = adaptive_input @ self.weight_input_hidden + self.bias_hidden
        if not np.all(np.isfinite(hidden_linear_prior)):
            raise FloatingPointError("nonfinite latent prior during relaxation")
        hidden_prior = tanh(hidden_linear_prior)
        hidden_state = hidden_prior.copy()

        for _ in range(inference_steps):
            output_linear = hidden_state @ self.weight_hidden_output + self.bias_output
            if not np.all(np.isfinite(output_linear)):
                raise FloatingPointError("nonfinite latent relaxation logits")
            output_prediction = sigmoid(output_linear)
            output_error = output_prediction - target_batch

            hidden_error = hidden_state - hidden_prior
            output_to_hidden = output_error @ self.weight_hidden_output.T
            hidden_gradient = hidden_error + output_to_hidden
            hidden_state -= inference_learning_rate * hidden_gradient
            if not np.all(np.isfinite(hidden_state)):
                raise FloatingPointError("nonfinite latent state during relaxation")

        output_linear = hidden_state @ self.weight_hidden_output + self.bias_output
        if not np.all(np.isfinite(output_linear)):
            raise FloatingPointError("nonfinite latent relaxation logits")
        output_prediction = sigmoid(output_linear)
        output_error = output_prediction - target_batch
        hidden_error = hidden_state - hidden_prior

        sample_count = float(input_batch.shape[0])
        output_term = output_error
        grad_hidden_output = (hidden_state.T @ output_term) / sample_count
        grad_output_bias = np.sum(output_term, axis=0, keepdims=True) / sample_count

        hidden_prior_gradient = (-hidden_error) * tanh_derivative_from_linear(hidden_linear_prior)
        grad_input_hidden = (adaptive_input.T @ hidden_prior_gradient) / sample_count
        grad_hidden_bias = np.sum(hidden_prior_gradient, axis=0, keepdims=True) / sample_count
        pre_hidden_delta = hidden_prior_gradient @ self.weight_input_hidden.T

        energy = self._compute_energy(
            output_prediction=output_prediction,
            target_batch=target_batch,
            hidden_error=hidden_error,
        )
        if not np.isfinite(energy):
            raise FloatingPointError("nonfinite training diagnostic")
        require_finite_training_arrays(
            [
                grad_hidden_output,
                grad_output_bias,
                grad_input_hidden,
                grad_hidden_bias,
                pre_hidden_delta,
            ],
            "training gradient",
        )

        adaptive_before = (
            self._hidden_chemical,
            self._hidden_chemical_fast,
            self._hidden_chemical_slow,
            self._importance_ema,
            self._reward_error_ema,
            self._last_reward_scale,
        )
        try:
            if wake_only_replay:
                # Why this: mutating then restoring adaptive state would still
                # train against a replay-modified plasticity gate.
                reward_scale = self._compute_reward_scale(output_error, update_baseline=False)
            else:
                self._update_chemical_layer(hidden_state)
                reward_scale = self._compute_reward_scale(output_error)
                self._last_reward_scale = reward_scale
                self._update_importance_ema(grad_hidden_output, reward_scale=reward_scale)
            plasticity = self.get_plasticity_state()

            gated_input_hidden = grad_input_hidden * plasticity[np.newaxis, :]
            gated_hidden_output = grad_hidden_output * plasticity[:, np.newaxis]
            gated_hidden_bias = grad_hidden_bias * plasticity[np.newaxis, :]
            effective_learning_rate = learning_rate * reward_scale
            with np.errstate(over="ignore", invalid="ignore"):
                new_hidden_output = (
                    self.weight_hidden_output - effective_learning_rate * gated_hidden_output
                )
                new_output_bias = self.bias_output - effective_learning_rate * grad_output_bias
                new_input_hidden = (
                    self.weight_input_hidden - effective_learning_rate * gated_input_hidden
                )
                new_hidden_bias = self.bias_hidden - effective_learning_rate * gated_hidden_bias
                new_pre_weights = list(self._pre_hidden_weights)
                new_pre_biases = list(self._pre_hidden_biases)
                running_delta = pre_hidden_delta
                for layer_index in range(len(self._pre_hidden_weights) - 1, -1, -1):
                    local_delta = running_delta * tanh_derivative_from_linear(
                        pre_hidden_linears[layer_index]
                    )
                    previous_activation = (
                        input_batch if layer_index == 0 else pre_hidden_activations[layer_index - 1]
                    )
                    grad_pre_weight = (previous_activation.T @ local_delta) / sample_count
                    grad_pre_bias = np.sum(local_delta, axis=0, keepdims=True) / sample_count
                    if layer_index > 0:
                        running_delta = local_delta @ self._pre_hidden_weights[layer_index].T
                    new_pre_weights[layer_index] = (
                        self._pre_hidden_weights[layer_index]
                        - effective_learning_rate * grad_pre_weight
                    )
                    new_pre_biases[layer_index] = (
                        self._pre_hidden_biases[layer_index]
                        - effective_learning_rate * grad_pre_bias
                    )
                new_traffic = self._traffic_sum + np.mean(np.abs(hidden_state), axis=0)
                new_age = self._neuron_age + 1.0

            require_finite_training_arrays(
                [
                    new_hidden_output,
                    new_output_bias,
                    new_input_hidden,
                    new_hidden_bias,
                    *new_pre_weights,
                    *new_pre_biases,
                    new_traffic,
                    new_age,
                    self._hidden_chemical,
                    self._hidden_chemical_fast,
                    self._hidden_chemical_slow,
                    self._importance_ema,
                ],
                "parameter or adaptive update",
            )
            if not isfinite(reward_scale) or (
                self._reward_error_ema is not None and not isfinite(self._reward_error_ema)
            ):
                raise FloatingPointError("nonfinite adaptive update")
        except Exception:
            (
                self._hidden_chemical,
                self._hidden_chemical_fast,
                self._hidden_chemical_slow,
                self._importance_ema,
                self._reward_error_ema,
                self._last_reward_scale,
            ) = adaptive_before
            raise

        if not wake_only_replay:
            self._decay_cooldowns()
        np.copyto(self.weight_hidden_output, new_hidden_output)
        np.copyto(self.bias_output, new_output_bias)
        np.copyto(self.weight_input_hidden, new_input_hidden)
        np.copyto(self.bias_hidden, new_hidden_bias)
        for old, new in zip(self._pre_hidden_weights, new_pre_weights):
            np.copyto(old, new)
        for old, new in zip(self._pre_hidden_biases, new_pre_biases):
            np.copyto(old, new)
        if not wake_only_replay:
            self._record_hidden_traffic(hidden_state)
        if update_epoch_state:
            self._epoch_count += 1
            self._wake_examples += int(input_batch.shape[0])
            np.copyto(self._neuron_age, new_age)
            self._epochs_since_sleep += 1
            self._reset_history_if_width_changed(initial_hidden_dim)
            self._energy_history.append(energy)
            max_history = max(self.config.sleep_energy_window * 4, 16)
            if len(self._energy_history) > max_history:
                self._energy_history = self._energy_history[-max_history:]
        if store_replay_snapshot:
            self._store_replay_snapshot(input_batch, target_batch)
        return energy

    def should_trigger_sleep(self) -> bool:
        """Return whether adaptive sleep criteria indicate consolidation is needed."""
        if self.config.sleep_mode == "disabled":
            return False
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
        high_chemical_variance = chemical_variance >= self.config.sleep_chemical_variance_threshold
        return plateau and high_chemical_variance

    def sleep_event(
        self,
        adaptation_policy: NeuronAdaptationPolicy | None = None,
        force_sleep: bool = True,
        current_step: int | None = None,
        total_steps: int | None = None,
        epoch_progress: SleepEpochProgress | None = None,
    ) -> SleepEventResult:
        """Consolidate structure; optionally trigger only when adaptive criteria fire."""
        if epoch_progress is not None:
            if current_step is not None or total_steps is not None:
                raise ValueError("epoch_progress cannot be combined with current_step/total_steps")
            current_step = epoch_progress.completed_epochs
            total_steps = epoch_progress.total_epochs
        started_at = perf_counter()
        if self.config.sleep_mode == "disabled":
            return self._skipped_sleep_result(
                trigger_reason="disabled",
                reason="sleep_disabled",
                started_at=started_at,
                completed_epoch=current_step,
            )
        if not force_sleep and not self.should_trigger_sleep():
            return self._skipped_sleep_result(
                trigger_reason="not_due",
                reason="adaptive_not_due",
                started_at=started_at,
                completed_epoch=current_step,
            )
        split_budget, prune_budget, should_skip = self._resolve_sleep_budgets(
            current_step=current_step,
            total_steps=total_steps,
        )
        if should_skip or (
            self.config.sleep_mode == "legacy" and split_budget <= 0 and prune_budget <= 0
        ):
            return self._skipped_sleep_result(
                trigger_reason="budget_skipped",
                reason="warmup" if should_skip else "zero_structural_budget",
                started_at=started_at,
                completed_epoch=current_step,
                split_budget=split_budget,
                prune_budget=prune_budget,
            )

        if self.config.sleep_mode == "components":
            split_budget = split_budget if self.config.sleep_enable_split else 0
            prune_budget = prune_budget if self.config.sleep_enable_prune else 0

        before_chemistry = self._chemical_summaries()
        snapshot = self.snapshot_state()
        try:
            return self._execute_sleep_event(
                adaptation_policy=adaptation_policy,
                split_budget=split_budget,
                prune_budget=prune_budget,
                before_chemistry=before_chemistry,
                completed_epoch=current_step,
                trigger_reason="forced" if force_sleep else "adaptive",
                started_at=started_at,
            )
        except Exception:
            self.restore_state(snapshot)
            raise

    def _execute_sleep_event(
        self,
        *,
        adaptation_policy: NeuronAdaptationPolicy | None,
        split_budget: int,
        prune_budget: int,
        before_chemistry: ChemicalSummaries,
        completed_epoch: int | None,
        trigger_reason: str,
        started_at: float,
    ) -> SleepEventResult:
        old_hidden_dim = self.hidden_dim
        split_indices: tuple[int, ...]
        pruned_indices: tuple[int, ...]
        if split_budget <= 0 and prune_budget <= 0:
            split_indices = ()
            pruned_indices = ()
        elif adaptation_policy is None:
            pruned_indices = self._select_prune_indices(max_prune_limit=prune_budget)
            split_indices = self._select_split_indices(
                max_split_limit=split_budget, excluded_indices=pruned_indices
            )
            self._validate_builtin_sleep_indices(
                split_indices, pruned_indices, split_budget=split_budget, prune_budget=prune_budget
            )
        else:
            split_indices, pruned_indices = self._derive_indices_from_policy(
                adaptation_policy, split_budget=split_budget, prune_budget=prune_budget
            )

        lineage_before = self.get_neuron_lineage()
        proposed_ids = tuple(lineage_before.neuron_ids[index] for index in pruned_indices)
        self._split_neurons(split_indices)
        split_lineage = self.get_neuron_lineage()
        split_pairs = tuple(
            (lineage_before.neuron_ids[index], child_id)
            for index, child_id in zip(
                split_indices, split_lineage.neuron_ids[old_hidden_dim:], strict=True
            )
        )
        self._schedule_or_prune(pruned_indices)
        pending_ids = set(self.get_pending_prune_ids())
        scheduled_ids = tuple(value for value in proposed_ids if value in pending_ids)
        if self.config.sleep_mode == "legacy" or self.config.sleep_enable_homeostasis:
            self._apply_homeostatic_downscaling()
        replay_examples = replay_updates = 0
        if self.config.sleep_mode == "legacy" or self.config.sleep_enable_replay:
            replay_examples, replay_updates = self._run_replay_consolidation()

        # Sleep partially clears chemistry after consolidation to reset plasticity gate.
        if self.config.sleep_mode == "legacy" or self.config.sleep_enable_chemical_reset:
            self._hidden_chemical *= self.config.sleep_reset_factor
            if self.config.use_dual_chemical:
                self._hidden_chemical_fast *= self.config.sleep_reset_factor
                self._hidden_chemical_slow *= self.config.sleep_reset_factor
        self._reset_history_if_width_changed(old_hidden_dim)
        self._epochs_since_sleep = 0
        self._sleep_events += 1

        lineage_after = self.get_neuron_lineage()
        active_ids = set(lineage_after.neuron_ids)
        removed_ids = tuple(value for value in lineage_before.neuron_ids if value not in active_ids)
        self._validate_sleep_post_state()
        telemetry = self._executed_sleep_telemetry(
            before_chemistry=before_chemistry,
            before_width=old_hidden_dim,
            split_pairs=split_pairs,
            proposed_prune_ids=proposed_ids,
            scheduled_prune_ids=scheduled_ids,
            removed_prune_ids=removed_ids,
            split_budget=split_budget,
            prune_budget=prune_budget,
            replay_examples=replay_examples,
            replay_updates=replay_updates,
            completed_epoch=completed_epoch,
            trigger_reason=trigger_reason,
            started_at=started_at,
        )
        return SleepEventResult(
            old_hidden_dim=old_hidden_dim,
            new_hidden_dim=self.hidden_dim,
            split_indices=split_indices,
            pruned_indices=pruned_indices,
            performed=True,
            lineage_before=lineage_before,
            lineage_after=lineage_after,
            prune_outcome=PruneOutcome(proposed_ids, scheduled_ids, removed_ids),
            telemetry=telemetry,
        )

    def _chemical_summaries(self) -> ChemicalSummaries:
        return ChemicalSummaries(
            primary=_summarize_chemical(self._hidden_chemical),
            fast=_summarize_chemical(self._hidden_chemical_fast),
            slow=_summarize_chemical(self._hidden_chemical_slow),
        )

    def get_sleep_chemical_summaries(self) -> ChemicalSummaries:
        """Return read-only chemistry facts for a runner's unscheduled epoch."""
        return self._chemical_summaries()

    def _skipped_sleep_result(
        self,
        *,
        trigger_reason: str,
        reason: str,
        started_at: float,
        completed_epoch: int | None,
        split_budget: int = 0,
        prune_budget: int = 0,
    ) -> SleepEventResult:
        chemistry = self._chemical_summaries()
        duration = max(0.0, perf_counter() - started_at)
        telemetry = SleepEventTelemetry(
            format_version=1,
            trigger_reason=trigger_reason,
            outcome="skipped",
            reason=reason,
            completed_epoch=completed_epoch,
            wake_batches=self._epoch_count,
            budgets=SleepBudgets(split_budget, prune_budget, 0, None),
            changes=SleepStructuralChanges((), (), (), (), (), (), ()),
            before_width=self.hidden_dim,
            proposed_width=self.hidden_dim,
            final_width=self.hidden_dim,
            guard=None,
            replay=SleepReplayUsage(0, 0, 0, 0),
            chemistry_before=chemistry,
            chemistry_proposed=chemistry,
            chemistry_final=chemistry,
            durations=SleepDurations(duration, duration),
        )
        return SleepEventResult(
            old_hidden_dim=self.hidden_dim,
            new_hidden_dim=self.hidden_dim,
            split_indices=(),
            pruned_indices=(),
            telemetry=telemetry,
        )

    def _executed_sleep_telemetry(
        self,
        *,
        before_chemistry: ChemicalSummaries,
        before_width: int,
        split_pairs: tuple[tuple[int, int], ...],
        proposed_prune_ids: tuple[int, ...],
        scheduled_prune_ids: tuple[int, ...],
        removed_prune_ids: tuple[int, ...],
        split_budget: int,
        prune_budget: int,
        replay_examples: int,
        replay_updates: int,
        completed_epoch: int | None,
        trigger_reason: str,
        started_at: float,
    ) -> SleepEventTelemetry:
        chemistry_after = self._chemical_summaries()
        duration = max(0.0, perf_counter() - started_at)
        replay_limit = (
            min(self.config.replay_steps, len(self._replay_memory))
            if self.config.sleep_mode == "legacy" or self.config.sleep_enable_replay
            else 0
        )
        return SleepEventTelemetry(
            format_version=1,
            trigger_reason=trigger_reason,
            outcome="applied",
            reason="core_executed",
            completed_epoch=completed_epoch,
            wake_batches=self._epoch_count,
            budgets=SleepBudgets(split_budget, prune_budget, replay_limit, None),
            changes=SleepStructuralChanges(
                split_pairs,
                split_pairs,
                proposed_prune_ids,
                scheduled_prune_ids,
                removed_prune_ids,
                scheduled_prune_ids,
                removed_prune_ids,
            ),
            before_width=before_width,
            proposed_width=self.hidden_dim,
            final_width=self.hidden_dim,
            guard=None,
            replay=SleepReplayUsage(
                replay_examples, replay_updates, replay_examples, replay_updates
            ),
            chemistry_before=before_chemistry,
            chemistry_proposed=chemistry_after,
            chemistry_final=chemistry_after,
            durations=SleepDurations(duration, duration),
        )

    def _validate_sleep_post_state(self) -> None:
        self._validate_training_topology()
        arrays = [
            *self._pre_hidden_weights,
            *self._pre_hidden_biases,
            self.weight_input_hidden,
            self.bias_hidden,
            self.weight_hidden_output,
            self.bias_output,
            self._hidden_chemical,
            self._hidden_chemical_fast,
            self._hidden_chemical_slow,
            self._importance_ema,
            self._traffic_sum,
            self._neuron_age,
        ]
        for replay in self._replay_memory:
            arrays.extend((replay.input_batch, replay.target_batch))
            if not isfinite(replay.priority) or not isfinite(replay.positive_fraction):
                raise FloatingPointError("nonfinite sleep replay state")
        require_finite_training_arrays(arrays, "sleep state")
        if (
            any(not isfinite(value) for value in self._energy_history)
            or (self._reward_error_ema is not None and not isfinite(self._reward_error_ema))
            or not isfinite(self._last_reward_scale)
        ):
            raise FloatingPointError("nonfinite sleep scalar state")

    def predict_proba(self, input_batch: Array) -> Array:
        _, _, adaptive_input = self._forward_pre_hidden(input_batch)
        hidden_linear = adaptive_input @ self.weight_input_hidden + self.bias_hidden
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

    def get_last_reward_scale(self) -> float:
        """Return most recent reward modulation scale from wake training."""
        return float(self._last_reward_scale)

    def _compute_plasticity_sensitivity(self) -> Array:
        if not self.config.use_adaptive_plasticity_sensitivity:
            return np.full_like(self._hidden_chemical, self.config.plasticity_sensitivity)

        age_component = self._normalize_vector_zero_base(self._neuron_age)
        importance_component = self._normalize_vector_zero_base(self._importance_ema)
        importance_mix = np.clip(self.config.plasticity_importance_mix, 0.0, 1.0)
        stability = importance_mix * importance_component + (1.0 - importance_mix) * age_component
        span = self.config.plasticity_sensitivity_max - self.config.plasticity_sensitivity_min
        return self.config.plasticity_sensitivity_min + span * stability

    def apply_neuron_proposals(self, proposals: list[NeuronChangeProposal]) -> PruneOutcome:
        old_hidden_dim = self.hidden_dim
        split_indices, prune_indices = self._indices_from_proposals(proposals)
        lineage_before = self.get_neuron_lineage()
        proposed_ids = tuple(lineage_before.neuron_ids[index] for index in prune_indices)
        self._split_neurons(split_indices)
        self._schedule_or_prune(prune_indices)
        self._reset_history_if_width_changed(old_hidden_dim)
        pending_ids = set(self.get_pending_prune_ids())
        active_ids = set(self.get_neuron_lineage().neuron_ids)
        return PruneOutcome(
            proposed_neuron_ids=proposed_ids,
            scheduled_neuron_ids=tuple(value for value in proposed_ids if value in pending_ids),
            removed_neuron_ids=tuple(
                value for value in lineage_before.neuron_ids if value not in active_ids
            ),
        )

    def _reset_history_if_width_changed(self, old_hidden_dim: int) -> None:
        # Why this: the diagnostic averages hidden residuals over current width;
        # a mixed-width plateau window cannot support a coherent trend.
        if self.config.sleep_mode == "components" and self.hidden_dim != old_hidden_dim:
            self._energy_history.clear()

    def _forward_pre_hidden(self, input_batch: Array) -> tuple[list[Array], list[Array], Array]:
        pre_hidden_linears: list[Array] = []
        pre_hidden_activations: list[Array] = []
        activation = input_batch
        for layer_weight, layer_bias in zip(self._pre_hidden_weights, self._pre_hidden_biases):
            hidden_linear = activation @ layer_weight + layer_bias
            activation = tanh(hidden_linear)
            pre_hidden_linears.append(hidden_linear)
            pre_hidden_activations.append(activation)
        return pre_hidden_linears, pre_hidden_activations, activation

    def _compute_energy(
        self, output_prediction: Array, target_batch: Array, hidden_error: Array
    ) -> float:
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

    def _update_importance_ema(self, grad_hidden_output: Array, reward_scale: float) -> None:
        importance = np.mean(np.abs(grad_hidden_output), axis=1) * float(reward_scale)
        decay = self.config.importance_ema_decay
        self._importance_ema = decay * self._importance_ema + (1.0 - decay) * importance

    def _compute_reward_scale(self, output_error: Array, *, update_baseline: bool = True) -> float:
        if not self.config.use_reward_modulated_learning:
            return 1.0

        batch_error = float(np.mean(np.abs(output_error)))
        baseline = batch_error if self._reward_error_ema is None else self._reward_error_ema
        difficulty_ratio = batch_error / max(float(baseline), 1e-8)
        raw_scale = difficulty_ratio**self.config.reward_difficulty_exponent
        reward_scale = float(
            np.clip(raw_scale, self.config.reward_scale_min, self.config.reward_scale_max)
        )

        # Why this: update baseline after computing ratio so scale reflects
        # current surprise against past performance, not a blended present.
        if update_baseline:
            decay = self.config.reward_baseline_decay
            self._reward_error_ema = decay * float(baseline) + (1.0 - decay) * batch_error
        return reward_scale

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

    def _select_split_indices(
        self,
        max_split_limit: int | None = None,
        excluded_indices: tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        split_threshold, _ = self._resolve_split_prune_thresholds()
        split_threshold += self.config.split_hysteresis_margin
        current_dim = self.hidden_dim
        remaining_capacity = self.max_hidden_dim - current_dim
        if remaining_capacity <= 0 or self.config.max_split_per_sleep <= 0:
            return ()
        limit = self.config.max_split_per_sleep if max_split_limit is None else max_split_limit
        if limit <= 0:
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
        if excluded_indices:
            candidates = candidates[~np.isin(candidates, excluded_indices)]
        if candidates.size == 0:
            return ()

        split_scores = self._compute_split_scores()
        sorted_candidates = candidates[np.argsort(split_scores[candidates])[::-1]]
        split_count = min(
            int(sorted_candidates.size),
            int(limit),
            int(remaining_capacity),
        )
        chosen = sorted_candidates[:split_count]
        return tuple(int(index) for index in chosen.tolist())

    def _select_prune_indices(self, max_prune_limit: int | None = None) -> tuple[int, ...]:
        _, prune_threshold = self._resolve_split_prune_thresholds()
        prune_threshold -= self.config.prune_hysteresis_margin
        # Pending gradual prunes already reserve slots above the minimum width.
        removable = (
            self.hidden_dim - self._min_hidden_dim - int(np.count_nonzero(self._prune_marked))
        )
        if removable <= 0 or self.config.max_prune_per_sleep <= 0:
            return ()
        limit = self.config.max_prune_per_sleep if max_prune_limit is None else max_prune_limit
        if limit <= 0:
            return ()

        candidates = np.where(
            np.logical_and.reduce(
                (
                    self._hidden_chemical <= prune_threshold,
                    ~self._prune_marked,
                    self._prune_cooldown <= 0,
                    self._neuron_age >= float(self.config.prune_min_age_steps),
                )
            )
        )[0]
        if candidates.size == 0:
            return ()

        prune_scores = self._compute_prune_scores()
        sorted_candidates = candidates[np.argsort(prune_scores[candidates])[::-1]]
        prune_count = min(
            int(sorted_candidates.size),
            int(limit),
            int(removable),
        )
        chosen = sorted_candidates[:prune_count]
        return tuple(sorted(int(index) for index in chosen.tolist()))

    def _validate_builtin_sleep_indices(
        self,
        split_indices: tuple[int, ...],
        prune_indices: tuple[int, ...],
        *,
        split_budget: int,
        prune_budget: int,
    ) -> None:
        """Reject a malformed combined proposal before split noise is drawn."""
        split_cap = min(
            split_budget,
            self._resolve_structural_budget(self.config.max_split_per_sleep, 1.0),
        )
        prune_cap = min(
            prune_budget,
            self._resolve_structural_budget(self.config.max_prune_per_sleep, 1.0),
        )
        if not isinstance(split_indices, tuple) or len(split_indices) > split_cap:
            raise ValueError("built-in split budget exceeded")
        if not isinstance(prune_indices, tuple) or len(prune_indices) > prune_cap:
            raise ValueError("built-in prune budget exceeded")
        if self.hidden_dim + len(split_indices) > self.max_hidden_dim:
            raise ValueError("built-in split exceeds maximum hidden width")
        pending = int(np.count_nonzero(self._prune_marked))
        if self.hidden_dim - pending - len(prune_indices) < self._min_hidden_dim:
            raise ValueError("built-in prune exceeds minimum hidden width")
        for label, indices in (("split", split_indices), ("prune", prune_indices)):
            if any(type(index) is not int or not 0 <= index < self.hidden_dim for index in indices):
                raise ValueError(f"built-in {label} index is out of range")
        if len(set(split_indices)) != len(split_indices) or len(set(prune_indices)) != len(
            prune_indices
        ):
            raise ValueError("built-in split or prune indices must be unique")
        if set(split_indices) & set(prune_indices):
            raise ValueError("built-in split and prune indices must not overlap")
        split_threshold, prune_threshold = self._resolve_split_prune_thresholds()
        split_threshold += self.config.split_hysteresis_margin
        prune_threshold -= self.config.prune_hysteresis_margin
        for index in split_indices:
            if (
                self._hidden_chemical[index] < split_threshold
                or self._split_cooldown[index] > 0
                or self._prune_marked[index]
            ):
                raise ValueError("built-in split index is ineligible")
        for index in prune_indices:
            if (
                self._hidden_chemical[index] > prune_threshold
                or self._prune_cooldown[index] > 0
                or self._prune_marked[index]
                or self._neuron_age[index] < float(self.config.prune_min_age_steps)
            ):
                raise ValueError("built-in prune index is ineligible")

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

    def _resolve_sleep_budgets(
        self, current_step: int | None, total_steps: int | None
    ) -> tuple[int, int, bool]:
        budget_scale = self._compute_adaptive_sleep_budget_scale()
        split_budget = self._resolve_structural_budget(
            self.config.max_split_per_sleep, budget_scale=budget_scale
        )
        prune_budget = self._resolve_structural_budget(
            self.config.max_prune_per_sleep, budget_scale=budget_scale
        )
        if current_step is None or total_steps is None or total_steps <= 0:
            return split_budget, prune_budget, False

        if current_step <= self.config.sleep_warmup_steps:
            return 0, 0, True

        progress = float(current_step) / float(total_steps)
        if progress < self.config.sleep_split_only_until_fraction:
            prune_budget = 0
        elif progress >= self.config.sleep_prune_only_after_fraction:
            split_budget = 0
        return split_budget, prune_budget, False

    def _compute_adaptive_sleep_budget_scale(self) -> float:
        if not self.config.use_adaptive_sleep_budget:
            return 1.0
        if len(self._energy_history) < self.config.sleep_energy_window:
            return (
                self.config.adaptive_sleep_budget_min_scale
                if self.config.sleep_mode == "components"
                else 1.0
            )

        recent = self._energy_history[-self.config.sleep_energy_window :]
        energy_improvement = max(0.0, float(recent[0] - recent[-1]))

        plateau_delta = self.config.sleep_plateau_delta
        if plateau_delta <= 1e-12:
            plateau_score = 1.0 if energy_improvement <= 0.0 else 0.0
        else:
            plateau_score = float(
                np.clip((plateau_delta - energy_improvement) / plateau_delta, 0.0, 1.0)
            )

        variance_threshold = self.config.sleep_chemical_variance_threshold
        if variance_threshold <= 1e-12:
            variance_score = 1.0
        else:
            chemical_variance = float(np.var(self._hidden_chemical))
            variance_score = float(np.clip(chemical_variance / variance_threshold, 0.0, 1.0))

        plateau_weight = max(0.0, self.config.adaptive_sleep_budget_plateau_weight)
        variance_weight = max(0.0, self.config.adaptive_sleep_budget_variance_weight)
        total_weight = plateau_weight + variance_weight
        if total_weight <= 1e-12:
            combined_signal = 1.0
        else:
            combined_signal = (
                plateau_weight * plateau_score + variance_weight * variance_score
            ) / total_weight

        min_scale = self.config.adaptive_sleep_budget_min_scale
        max_scale = self.config.adaptive_sleep_budget_max_scale
        return float(min_scale + (max_scale - min_scale) * combined_signal)

    def _resolve_structural_budget(self, configured_limit: int, budget_scale: float) -> int:
        if configured_limit <= 0:
            return 0
        if budget_scale <= 0.0:
            return 0
        scaled_limit = int(np.floor(float(configured_limit) * budget_scale))
        if scaled_limit <= 0:
            scaled_limit = 1
        fraction = self.config.sleep_max_change_fraction
        if fraction <= 0.0:
            return 0
        by_fraction = int(np.floor(float(self.hidden_dim) * fraction))
        by_fraction = max(by_fraction, int(self.config.sleep_min_change_count))
        return min(int(scaled_limit), int(by_fraction))

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
        self,
        adaptation_policy: NeuronAdaptationPolicy,
        *,
        split_budget: int,
        prune_budget: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        traffic = self.get_layer_traffic()
        proposals = adaptation_policy.propose(traffic)
        return self._indices_from_proposals(
            proposals, split_budget=split_budget, prune_budget=prune_budget
        )

    def _indices_from_proposals(
        self,
        proposals: list[NeuronChangeProposal],
        *,
        split_budget: int | None = None,
        prune_budget: int | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        add_count, remove_indices = self._parse_proposal_requests(proposals)
        eligible = self._eligible_proposal_split_sources(
            add_count, remove_indices, split_budget=split_budget, prune_budget=prune_budget
        )
        split_indices = self._rank_proposal_split_sources(eligible, add_count)
        return split_indices, tuple(sorted(remove_indices))

    def _parse_proposal_requests(
        self, proposals: list[NeuronChangeProposal]
    ) -> tuple[int, set[int]]:
        add_count = 0
        remove_indices: set[int] = set()
        for proposal in proposals:
            if not isinstance(proposal, NeuronChangeProposal):
                raise ValueError("proposal must be a NeuronChangeProposal")
            if proposal.layer_name != "hidden":
                raise ValueError("proposal layer_name must be 'hidden'")
            if type(proposal.add_count) is not int or proposal.add_count < 0:
                raise ValueError("proposal add_count must be a nonnegative integer")
            add_count += proposal.add_count
            for index in proposal.remove_indices:
                if type(index) is not int or not 0 <= index < self.hidden_dim:
                    raise ValueError("proposal remove index is out of range")
                if index in remove_indices:
                    raise ValueError("proposal remove indices must be unique")
                remove_indices.add(index)
        return add_count, remove_indices

    def _eligible_proposal_split_sources(
        self,
        add_count: int,
        remove_indices: set[int],
        *,
        split_budget: int | None,
        prune_budget: int | None,
    ) -> NDArray[np.int64]:
        split_cap = self._resolve_structural_budget(self.config.max_split_per_sleep, 1.0)
        prune_cap = self._resolve_structural_budget(self.config.max_prune_per_sleep, 1.0)
        if split_budget is not None:
            split_cap = min(split_cap, split_budget)
        if prune_budget is not None:
            prune_cap = min(prune_cap, prune_budget)
        if add_count > split_cap or len(remove_indices) > prune_cap:
            raise ValueError("proposal exceeds split, prune, or change-fraction budget")
        if self.hidden_dim + add_count > self.max_hidden_dim:
            raise ValueError("proposal exceeds maximum hidden width")
        pending_prunes = int(np.count_nonzero(self._prune_marked))
        if self.hidden_dim - pending_prunes - len(remove_indices) < self._min_hidden_dim:
            raise ValueError("proposal exceeds minimum hidden width")
        for index in remove_indices:
            if (
                self._prune_marked[index]
                or self._prune_cooldown[index] > 0
                or self._neuron_age[index] < float(self.config.prune_min_age_steps)
            ):
                raise ValueError("proposal prune index violates age, cooldown, or pending state")

        # Why this: an explicit prune owns its original neuron; a split must
        # choose another eligible parent before either tensor shape changes.
        eligible = np.array(
            [
                index
                for index in range(self.hidden_dim)
                if index not in remove_indices
                and self._split_cooldown[index] <= 0
                and not self._prune_marked[index]
            ],
            dtype=np.int64,
        )
        if add_count > len(eligible):
            raise ValueError("proposal has too few eligible split sources")
        return eligible

    def _rank_proposal_split_sources(
        self, eligible: NDArray[np.int64], add_count: int
    ) -> tuple[int, ...]:
        if add_count == 0:
            return ()
        split_threshold, _ = self._resolve_split_prune_thresholds()
        split_threshold += self.config.split_hysteresis_margin
        scores = self._compute_split_scores()
        ranked = eligible[np.argsort(scores[eligible])[::-1]]
        preferred = ranked[self._hidden_chemical[ranked] >= split_threshold]
        fallback = ranked[self._hidden_chemical[ranked] < split_threshold]
        return tuple(int(index) for index in np.concatenate((preferred, fallback))[:add_count])

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
        new_neuron_ids: list[int] = []
        new_parent_ids: list[int] = []
        new_split_cooldowns: list[int] = []
        new_prune_cooldowns: list[int] = []

        for index in split_indices:
            new_neuron_ids.append(self._next_neuron_id)
            new_parent_ids.append(int(self._neuron_ids[index]))
            self._next_neuron_id += 1
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
        self._neuron_ids = np.concatenate(
            [self._neuron_ids, np.array(new_neuron_ids, dtype=np.int64)]
        )
        self._parent_ids = np.concatenate(
            [self._parent_ids, np.array(new_parent_ids, dtype=np.int64)]
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
        self._neuron_ids = self._neuron_ids[mask]
        self._parent_ids = self._parent_ids[mask]
        self._prune_ttl = self._prune_ttl[mask]
        self._prune_marked = self._prune_marked[mask]
        self._split_cooldown = self._split_cooldown[mask]
        self._prune_cooldown = self._prune_cooldown[mask]

    def _apply_homeostatic_downscaling(self) -> None:
        scale_factor = self.config.homeostatic_downscale_factor
        if abs(scale_factor - 1.0) > 1e-12:
            for layer_index in range(len(self._pre_hidden_weights)):
                self._pre_hidden_weights[layer_index] *= scale_factor
                self._pre_hidden_biases[layer_index] *= scale_factor
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
        budget = getattr(self, "_replay_retention_budget", None)
        if budget is not None:
            policy = getattr(self, "_replay_retention_policy", DEFAULT_REPLAY_RETENTION_POLICY)
            predictions = self.predict_proba(input_batch)
            retained = {
                replay_sample_id(item.input_batch, item.target_batch): item
                for item in self._replay_memory
            }
            for index in range(input_batch.shape[0]):
                input_row = input_batch[index : index + 1].copy()
                target_row = target_batch[index : index + 1].copy()
                sample_id = replay_sample_id(input_row, target_row)
                if hasattr(self, "_replay_retention_policy"):
                    if sample_id in self._replay_observed_ids:
                        self._replay_duplicate_ids.add(sample_id)
                        self._replay_duplicate_occurrences += 1
                    self._replay_observed_ids.add(sample_id)
                sample_bytes = input_row.nbytes + target_row.nbytes
                if sample_bytes > budget.max_bytes:
                    continue
                if policy.name == "recent_fifo":
                    retained.pop(sample_id, None)
                retained[sample_id] = ReplaySnapshot(
                    input_batch=input_row,
                    target_batch=target_row,
                    priority=float(np.abs(predictions[index, 0] - target_row[0, 0])),
                    positive_fraction=float(target_row[0, 0]),
                )
                while (
                    len(retained) > budget.max_examples
                    or sum(
                        item.input_batch.nbytes + item.target_batch.nbytes
                        for item in retained.values()
                    )
                    > budget.max_bytes
                ):
                    # Why this: both limits apply to the same retained rows;
                    # only eviction order differs between declared policies.
                    del retained[policy.eviction_id(retained)]
            self._replay_memory = deque(retained.values())
            return
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

    def _run_replay_consolidation(self) -> tuple[int, int]:
        if self.config.replay_steps <= 0 or len(self._replay_memory) == 0:
            return 0, 0

        replay_count = min(self.config.replay_steps, len(self._replay_memory))
        replay_snapshots = self._select_replay_snapshots(replay_count)
        examples = 0
        updates = 0
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
            self._replay_updates += 1
            if hasattr(self, "_replay_retention_policy"):
                for index in range(snapshot.input_batch.shape[0]):
                    self._replay_exposed_ids.add(
                        replay_sample_id(
                            snapshot.input_batch[index : index + 1],
                            snapshot.target_batch[index : index + 1],
                        )
                    )
                self._replay_exposure_updates += 1
            examples += int(snapshot.input_batch.shape[0])
            updates += 1
        return examples, updates

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

    def _resolve_hidden_dims(
        self,
        hidden_dim: int,
        hidden_dims: list[int] | tuple[int, ...] | None,
    ) -> list[int]:
        require_positive_integer_dimension(hidden_dim, "hidden_dim")
        if hidden_dims is None:
            return [hidden_dim]

        resolved = [
            require_positive_integer_dimension(value, f"hidden_dims[{index}]")
            for index, value in enumerate(hidden_dims)
        ]
        if not resolved:
            raise ValueError("hidden_dims cannot be empty")
        return resolved

    def _validate_config(self, config: CircadianConfig) -> None:
        for field in fields(config):
            if isinstance(field.default, (int, float)):
                value = getattr(config, field.name)
                try:
                    finite = isfinite(value)
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError(f"{field.name} must be finite")
        if config.sleep_mode not in {"legacy", "components", "disabled"}:
            raise ValueError("sleep_mode must be one of: legacy, components, disabled")
        component_names = (
            "sleep_enable_chemical_reset",
            "sleep_enable_replay",
            "sleep_enable_homeostasis",
            "sleep_enable_split",
            "sleep_enable_prune",
        )
        if any(not isinstance(getattr(config, name), bool) for name in component_names):
            raise ValueError("sleep component switches must be booleans")
        if config.sleep_mode == "legacy" and any(
            not getattr(config, name) for name in component_names
        ):
            raise ValueError("sleep_mode='components' is required for component switches")
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
        if not (0.0 <= config.reward_baseline_decay < 1.0):
            raise ValueError("reward_baseline_decay must be in [0, 1)")
        if config.reward_difficulty_exponent <= 0.0:
            raise ValueError("reward_difficulty_exponent must be positive")
        if config.reward_scale_min <= 0.0:
            raise ValueError("reward_scale_min must be positive")
        if config.reward_scale_max < config.reward_scale_min:
            raise ValueError("reward_scale_max must be >= reward_scale_min")
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
        if config.adaptive_sleep_budget_min_scale <= 0.0:
            raise ValueError("adaptive_sleep_budget_min_scale must be positive")
        if config.adaptive_sleep_budget_max_scale < config.adaptive_sleep_budget_min_scale:
            raise ValueError("adaptive_sleep_budget_max_scale must be >= min scale")
        if config.adaptive_sleep_budget_max_scale > 1.0:
            raise ValueError("adaptive_sleep_budget_max_scale must be <= 1.0")
        if config.adaptive_sleep_budget_plateau_weight < 0.0:
            raise ValueError("adaptive_sleep_budget_plateau_weight must be non-negative")
        if config.adaptive_sleep_budget_variance_weight < 0.0:
            raise ValueError("adaptive_sleep_budget_variance_weight must be non-negative")
        if config.max_split_per_sleep < 0 or config.max_prune_per_sleep < 0:
            raise ValueError("max split/prune per sleep must be non-negative")
        if config.split_noise_scale < 0.0:
            raise ValueError("split_noise_scale must be non-negative")
        if not (0.0 <= config.sleep_reset_factor <= 1.0):
            raise ValueError("sleep_reset_factor must be between 0 and 1")
        if config.sleep_warmup_steps < 0:
            raise ValueError("sleep_warmup_steps must be non-negative")
        if not (0.0 <= config.sleep_split_only_until_fraction <= 1.0):
            raise ValueError("sleep_split_only_until_fraction must be between 0 and 1")
        if not (0.0 <= config.sleep_prune_only_after_fraction <= 1.0):
            raise ValueError("sleep_prune_only_after_fraction must be between 0 and 1")
        if config.sleep_split_only_until_fraction > config.sleep_prune_only_after_fraction:
            raise ValueError(
                "sleep_split_only_until_fraction cannot exceed sleep_prune_only_after_fraction"
            )
        if not (0.0 <= config.sleep_max_change_fraction <= 1.0):
            raise ValueError("sleep_max_change_fraction must be between 0 and 1")
        if config.sleep_min_change_count < 0:
            raise ValueError("sleep_min_change_count must be non-negative")
        if config.prune_min_age_steps < 0:
            raise ValueError("prune_min_age_steps must be non-negative")
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
