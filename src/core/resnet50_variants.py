"""ResNet-50 variants for backprop, predictive coding, and circadian predictive coding."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass, fields
from math import isfinite
from numbers import Integral
from typing import Self
from typing import Any

from src.core.dimension_validation import require_positive_integer_dimension
from src.core.neuron_adaptation import NeuronLineageSnapshot, PruneOutcome
from src.core.sleep_clocks import SleepClockSnapshot, SleepEpochProgress
from src.shared.torch_runtime import require_torch, require_torchvision_models

TORCH_PC_ENERGY_ID = "torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1"


@dataclass(frozen=True)
class CircadianHeadConfig:
    """Circadian head dynamics for split/prune and plasticity."""

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
    use_reward_modulated_learning: bool = False
    reward_baseline_decay: float = 0.95
    reward_difficulty_exponent: float = 1.0
    reward_scale_min: float = 0.75
    reward_scale_max: float = 1.5
    use_adaptive_thresholds: bool = False
    adaptive_split_percentile: float = 85.0
    adaptive_prune_percentile: float = 20.0
    split_threshold: float = 0.80
    prune_threshold: float = 0.08
    split_hysteresis_margin: float = 0.0
    prune_hysteresis_margin: float = 0.0
    split_cooldown_steps: int = 0
    prune_cooldown_steps: int = 0
    split_weight_norm_mix: float = 0.30
    prune_weight_norm_mix: float = 0.30
    split_importance_mix: float = 0.20
    prune_importance_mix: float = 0.35
    importance_ema_decay: float = 0.95
    max_split_per_sleep: int = 2
    max_prune_per_sleep: int = 2
    split_noise_scale: float = 0.01
    sleep_reset_factor: float = 0.45
    sleep_warmup_steps: int = 0
    sleep_split_only_until_fraction: float = 0.50
    sleep_prune_only_after_fraction: float = 0.85
    sleep_max_change_fraction: float = 1.0
    sleep_min_change_count: int = 1
    prune_min_age_steps: int = 0
    homeostatic_downscale_factor: float = 1.0
    homeostasis_target_input_norm: float = 0.0
    homeostasis_target_output_norm: float = 0.0
    homeostasis_strength: float = 0.50
    use_adaptive_sleep_trigger: bool = False
    min_sleep_steps: int = 40
    sleep_energy_window: int = 32
    sleep_plateau_delta: float = 1e-4
    sleep_chemical_variance_threshold: float = 0.02
    use_adaptive_sleep_budget: bool = False
    adaptive_sleep_budget_min_scale: float = 0.25
    adaptive_sleep_budget_max_scale: float = 1.0
    adaptive_sleep_budget_plateau_weight: float = 0.6
    adaptive_sleep_budget_variance_weight: float = 0.4

    # Torch has no sleep replay. Legacy preserves budget-gated sleep;
    # components separates the available actions from structure.
    sleep_mode: str = "legacy"
    sleep_enable_chemical_reset: bool = True
    sleep_enable_homeostasis: bool = True
    sleep_enable_split: bool = True
    sleep_enable_prune: bool = True

    @classmethod
    def matched_pc_control(cls) -> Self:
        """Return the shallow PC parity control with circadian effects disabled.

        Chemistry remains observable, while unit plasticity, unit reward,
        and zero sleep budgets make training and forced sleep match PC.
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
        )


@dataclass(frozen=True)
class SleepEventResult:
    """Sleep consolidation result."""

    old_hidden_dim: int
    new_hidden_dim: int
    split_indices: tuple[int, ...]
    pruned_indices: tuple[int, ...]
    performed: bool = False
    lineage_before: NeuronLineageSnapshot | None = None
    lineage_after: NeuronLineageSnapshot | None = None
    prune_outcome: PruneOutcome | None = None


@dataclass(frozen=True)
class CircadianClassifierSnapshot:
    """Detached in-memory state of a circadian head and its feature extractor."""

    format_version: int
    device: str
    freeze_backbone: bool
    backbone_module_types: tuple[tuple[str, str], ...]
    backbone_state: dict[str, Any]
    backbone_training: dict[str, bool]
    backbone_requires_grad: dict[str, bool]
    head_state: dict[str, Any]


def _count_parameters(module_or_tensor: Any) -> int:
    torch = require_torch()
    if hasattr(module_or_tensor, "parameters"):
        return int(sum(parameter.numel() for parameter in module_or_tensor.parameters()))
    if isinstance(module_or_tensor, torch.Tensor):
        return int(module_or_tensor.numel())
    raise ValueError("Unsupported parameter container.")


def _build_resnet50_backbone(
    device: Any, freeze_backbone: bool, backbone_weights: str
) -> tuple[Any, int]:
    torch = require_torch()
    models = require_torchvision_models()

    if backbone_weights == "none":
        weights = None
    elif backbone_weights == "imagenet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        raise ValueError("backbone_weights must be one of: none, imagenet.")

    try:
        backbone = models.resnet50(weights=weights)
    except Exception as exc:  # pragma: no cover - environment/network dependent
        raise RuntimeError(
            f"Failed to initialize ResNet-50 with backbone_weights='{backbone_weights}'."
        ) from exc

    feature_dim = int(backbone.fc.in_features)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device)
    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        backbone.eval()
    return backbone, feature_dim


class BackpropResNet50Classifier:
    """Standard ResNet-50 classifier trained with backpropagation."""

    def __init__(
        self,
        num_classes: int,
        device: Any,
        freeze_backbone: bool = False,
        backbone_weights: str = "none",
    ) -> None:
        torch = require_torch()
        self._torch = torch
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.backbone, feature_dim = _build_resnet50_backbone(
            device=device,
            freeze_backbone=freeze_backbone,
            backbone_weights=backbone_weights,
        )
        self.classifier = torch.nn.Linear(feature_dim, num_classes).to(device)

    def forward_logits(self, images: Any) -> Any:
        features = self.backbone(images)
        return self.classifier(features)

    def trainable_parameters(self) -> list[Any]:
        parameters = list(self.classifier.parameters())
        if not self.freeze_backbone:
            parameters = list(self.backbone.parameters()) + parameters
        return parameters

    def parameter_count(self) -> int:
        return _count_parameters(self.backbone) + _count_parameters(self.classifier)

    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.trainable_parameters()))


class PredictiveCodingHead:
    """Predictive-coding head updated with local errors and iterative inference."""

    energy_definition = TORCH_PC_ENERGY_ID

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        device: Any,
        seed: int,
    ) -> None:
        feature_dim = require_positive_integer_dimension(feature_dim, "feature_dim")
        hidden_dim = require_positive_integer_dimension(hidden_dim, "hidden_dim")
        num_classes = require_positive_integer_dimension(num_classes, "num_classes")

        torch = require_torch()
        self._torch = torch
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        generator = torch.Generator()
        generator.manual_seed(seed)

        self.weight_feature_hidden = (
            0.05 * torch.randn((feature_dim, hidden_dim), generator=generator, dtype=torch.float32)
        ).to(device)
        self.bias_hidden = torch.zeros((1, hidden_dim), dtype=torch.float32, device=device)
        self.weight_hidden_output = (
            0.05 * torch.randn((hidden_dim, num_classes), generator=generator, dtype=torch.float32)
        ).to(device)
        self.bias_output = torch.zeros((1, num_classes), dtype=torch.float32, device=device)

        self._traffic_sum = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        self._traffic_steps = 0

    @property
    def hidden_dim(self) -> int:
        return int(self.weight_feature_hidden.shape[1])

    def _validate_training_inputs(
        self,
        features: Any,
        targets: Any,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> None:
        if not isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite.")
        if not isinstance(inference_steps, Integral) or inference_steps <= 0:
            raise ValueError("inference_steps must be a positive integer.")
        if not isfinite(inference_learning_rate) or inference_learning_rate <= 0.0:
            raise ValueError("inference_learning_rate must be positive and finite.")
        self._validate_training_topology()
        torch = self._torch
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be 2D Torch tensors.")
        if features.shape[0] == 0:
            raise ValueError("training batch must be nonempty.")
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"feature width must be {self.feature_dim}; got {features.shape[1]}.")
        if not torch.is_floating_point(features):
            raise ValueError("features must have a floating dtype.")
        if features.dtype != self.weight_feature_hidden.dtype:
            raise ValueError("features dtype must match head weights dtype.")
        if features.device != self.weight_feature_hidden.device:
            raise ValueError("features must be on the head device.")
        if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
            raise ValueError("targets must be 1D Torch tensors.")
        if targets.shape[0] != features.shape[0]:
            raise ValueError("features and targets must have the same batch size.")
        if targets.dtype != torch.long:
            raise ValueError("targets must have int64 class-index dtype.")
        if targets.device != features.device:
            raise ValueError("targets must be on the feature device.")

        finite_features = torch.isfinite(features).all()
        labels_in_range = ((targets >= 0) & (targets < self.num_classes)).all()
        # One synchronization on valid batches; inspect the failed component
        # only when the combined validation rejects the batch.
        if not bool((finite_features & labels_in_range).item()):
            if not bool(finite_features.item()):
                raise ValueError("features must be finite.")
            raise ValueError(f"targets must be in class range [0, {self.num_classes}).")

    def _validate_training_topology(self) -> None:
        """Require aligned model tensors before a training update."""
        weight = self.weight_feature_hidden
        if weight.ndim != 2 or weight.shape[0] != self.feature_dim or weight.shape[1] <= 0:
            raise ValueError("model topology has incompatible feature-hidden width")
        width = weight.shape[1]
        if (
            self.bias_hidden.shape != (1, width)
            or self.weight_hidden_output.shape != (width, self.num_classes)
            or self.bias_output.shape != (1, self.num_classes)
            or self._traffic_sum.shape != (width,)
        ):
            raise ValueError("model topology has incompatible output or bias width")

    def _require_finite_relaxed_state(
        self, hidden_linear_prior: Any, hidden_state: Any, output_logits: Any
    ) -> None:
        # Check once before parameter updates; per-step GPU synchronizations would
        # change the benchmark's training-time budget substantially.
        finite = (
            self._torch.isfinite(hidden_linear_prior).all()
            & self._torch.isfinite(hidden_state).all()
            & self._torch.isfinite(output_logits).all()
        )
        if not bool(finite.item()):
            raise FloatingPointError("nonfinite latent state or logits during relaxation")

    def _require_finite_training_update(self, candidates: list[Any], energy: Any) -> float:
        """Check candidate tensors using the diagnostic's existing host read."""
        torch = self._torch
        finite = torch.isfinite(energy)
        for candidate in candidates:
            finite = finite & torch.isfinite(candidate).all()
        checked_energy = torch.where(finite, energy, torch.full_like(energy, float("nan")))
        energy_value = float(checked_energy.item())
        if not isfinite(energy_value):
            raise FloatingPointError("nonfinite training update or diagnostic")
        return energy_value

    def train_step(
        self,
        features: Any,
        targets: Any,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> float:
        torch = self._torch
        functional = torch.nn.functional
        self._validate_training_inputs(
            features, targets, learning_rate, inference_steps, inference_learning_rate
        )
        sample_count = float(features.shape[0])

        hidden_linear_prior = features @ self.weight_feature_hidden + self.bias_hidden
        hidden_prior = torch.tanh(hidden_linear_prior)
        hidden_state = hidden_prior.clone()
        target_one_hot = functional.one_hot(targets, num_classes=self.num_classes).to(features.dtype)

        for _ in range(inference_steps):
            output_logits = hidden_state @ self.weight_hidden_output + self.bias_output
            output_probabilities = functional.softmax(output_logits, dim=1)
            output_error = output_probabilities - target_one_hot
            hidden_error = hidden_state - hidden_prior
            output_to_hidden = output_error @ self.weight_hidden_output.T
            hidden_gradient = hidden_error + output_to_hidden
            hidden_state = hidden_state - inference_learning_rate * hidden_gradient

        output_logits = hidden_state @ self.weight_hidden_output + self.bias_output
        self._require_finite_relaxed_state(hidden_linear_prior, hidden_state, output_logits)
        output_probabilities = functional.softmax(output_logits, dim=1)
        output_error = output_probabilities - target_one_hot
        hidden_error = hidden_state - hidden_prior

        grad_hidden_output = (hidden_state.T @ output_error) / sample_count
        grad_output_bias = torch.mean(output_error, dim=0, keepdim=True)
        hidden_prior_grad = (-hidden_error) * (1.0 - hidden_prior * hidden_prior)
        grad_feature_hidden = (features.T @ hidden_prior_grad) / sample_count
        grad_hidden_bias = torch.mean(hidden_prior_grad, dim=0, keepdim=True)

        new_hidden_output = self.weight_hidden_output - learning_rate * grad_hidden_output
        new_output_bias = self.bias_output - learning_rate * grad_output_bias
        new_feature_hidden = self.weight_feature_hidden - learning_rate * grad_feature_hidden
        new_hidden_bias = self.bias_hidden - learning_rate * grad_hidden_bias
        new_traffic = self._traffic_sum + torch.mean(torch.abs(hidden_state), dim=0)
        energy = 0.5 * (torch.mean(output_error * output_error) + torch.mean(hidden_error * hidden_error))
        energy_value = self._require_finite_training_update(
            [new_hidden_output, new_output_bias, new_feature_hidden, new_hidden_bias, new_traffic],
            energy,
        )
        self.weight_hidden_output = new_hidden_output
        self.bias_output = new_output_bias
        self.weight_feature_hidden = new_feature_hidden
        self.bias_hidden = new_hidden_bias
        self._traffic_sum = new_traffic
        self._traffic_steps += 1
        return energy_value

    def predict_logits(self, features: Any) -> Any:
        torch = self._torch
        hidden_state = torch.tanh(features @ self.weight_feature_hidden + self.bias_hidden)
        return hidden_state @ self.weight_hidden_output + self.bias_output

    def parameter_count(self) -> int:
        return (
            int(self.weight_feature_hidden.numel())
            + int(self.bias_hidden.numel())
            + int(self.weight_hidden_output.numel())
            + int(self.bias_output.numel())
        )

    def mean_hidden_traffic(self) -> Any:
        if self._traffic_steps == 0:
            return self._traffic_sum.clone()
        return self._traffic_sum / float(self._traffic_steps)


class BackpropMLPHead:
    """Trainable tanh MLP with the same initial tensors as a predictive-coding head."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        device: Any,
        seed: int,
    ) -> None:
        torch = require_torch()
        self._torch = torch
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Clone the PC initializer so a shared seed gives bitwise-equal, independent tensors.
        initial_head = PredictiveCodingHead(feature_dim, hidden_dim, num_classes, device, seed)
        self.weight_feature_hidden = torch.nn.Parameter(initial_head.weight_feature_hidden.clone())
        self.bias_hidden = torch.nn.Parameter(initial_head.bias_hidden.clone())
        self.weight_hidden_output = torch.nn.Parameter(initial_head.weight_hidden_output.clone())
        self.bias_output = torch.nn.Parameter(initial_head.bias_output.clone())

    @property
    def hidden_dim(self) -> int:
        return int(self.weight_feature_hidden.shape[1])

    def forward_logits(self, features: Any) -> Any:
        hidden_state = self._torch.tanh(features @ self.weight_feature_hidden + self.bias_hidden)
        return hidden_state @ self.weight_hidden_output + self.bias_output

    def trainable_parameters(self) -> list[Any]:
        return [
            self.weight_feature_hidden,
            self.bias_hidden,
            self.weight_hidden_output,
            self.bias_output,
        ]

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.trainable_parameters()))


class BackpropMLPResNet50Classifier:
    """ResNet-50 features and a backprop MLP head for matched PC comparisons."""

    def __init__(
        self,
        num_classes: int,
        device: Any,
        head_hidden_dim: int = 256,
        seed: int = 7,
        freeze_backbone: bool = True,
        backbone_weights: str = "none",
    ) -> None:
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.backbone, feature_dim = _build_resnet50_backbone(
            device=device,
            freeze_backbone=freeze_backbone,
            backbone_weights=backbone_weights,
        )
        self.head = BackpropMLPHead(
            feature_dim=feature_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )

    def extract_features(self, images: Any) -> Any:
        torch = require_torch()
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)

    def forward_logits(self, images: Any) -> Any:
        return self.head.forward_logits(self.extract_features(images))

    def trainable_parameters(self) -> list[Any]:
        parameters = self.head.trainable_parameters()
        if not self.freeze_backbone:
            parameters = list(self.backbone.parameters()) + parameters
        return parameters

    def parameter_count(self) -> int:
        return _count_parameters(self.backbone) + self.head.parameter_count()

    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.trainable_parameters()))


class CircadianPredictiveCodingHead(PredictiveCodingHead):
    """Predictive-coding head with chemical-state plasticity and sleep events."""

    _SNAPSHOT_TENSORS = {
        "weight_feature_hidden": "weight_feature_hidden",
        "bias_hidden": "bias_hidden",
        "weight_hidden_output": "weight_hidden_output",
        "bias_output": "bias_output",
        "chemical": "_chemical",
        "chemical_fast": "_chemical_fast",
        "chemical_slow": "_chemical_slow",
        "neuron_age": "_neuron_age",
        "importance_ema": "_importance_ema",
        "traffic_sum": "_traffic_sum",
        "split_cooldown": "_split_cooldown",
        "prune_cooldown": "_prune_cooldown",
        "neuron_ids": "_neuron_ids",
        "parent_ids": "_parent_ids",
    }
    _SNAPSHOT_COUNTERS = {
        "traffic_steps": "_traffic_steps",
        "wake_examples": "_wake_examples",
        "sleep_events": "_sleep_events",
        "steps_since_sleep": "_steps_since_sleep",
        "next_neuron_id": "_next_neuron_id",
    }

    def _validate_training_topology(self) -> None:
        super()._validate_training_topology()
        width = self.hidden_dim
        for name in (
            "_chemical",
            "_chemical_fast",
            "_chemical_slow",
            "_importance_ema",
            "_neuron_age",
            "_split_cooldown",
            "_prune_cooldown",
            "_neuron_ids",
            "_parent_ids",
        ):
            if getattr(self, name).shape != (width,):
                raise ValueError(f"model topology has incompatible {name} width")
        if (
            self._neuron_ids.dtype != self._torch.int64
            or self._parent_ids.dtype != self._torch.int64
            or self._neuron_ids.device != self.weight_feature_hidden.device
            or self._parent_ids.device != self.weight_feature_hidden.device
            or type(self._next_neuron_id) is not int
            or self._next_neuron_id < width
        ):
            raise ValueError("model topology has incompatible neuron lineage metadata")
        ids = self._neuron_ids
        valid_lineage = (
            (ids >= 0).all()
            & (ids < self._next_neuron_id).all()
            & (self._parent_ids >= -1).all()
            & (self._parent_ids < ids).all()
            & (ids[1:] > ids[:-1]).all()
        )
        if not bool(valid_lineage.item()):
            raise ValueError("model topology has invalid neuron lineage")

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        device: Any,
        seed: int,
        config: CircadianHeadConfig | None = None,
        min_hidden_dim: int = 16,
        max_hidden_dim: int = 4096,
    ) -> None:
        min_hidden_dim = require_positive_integer_dimension(min_hidden_dim, "min_hidden_dim")
        max_hidden_dim = require_positive_integer_dimension(max_hidden_dim, "max_hidden_dim")
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )
        torch = require_torch()
        self._torch = torch
        self.config = config or CircadianHeadConfig()
        self._validate_config(self.config)
        self._min_hidden_dim = min_hidden_dim
        self._max_hidden_dim = max_hidden_dim
        if self._min_hidden_dim > hidden_dim:
            raise ValueError("min_hidden_dim cannot exceed initial hidden_dim.")
        if self._max_hidden_dim < hidden_dim:
            raise ValueError("max_hidden_dim cannot be less than initial hidden_dim.")

        self._chemical = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        self._chemical_fast = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        self._chemical_slow = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        self._neuron_age = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        self._importance_ema = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        self._split_cooldown = torch.zeros(hidden_dim, dtype=torch.int32, device=device)
        self._prune_cooldown = torch.zeros(hidden_dim, dtype=torch.int32, device=device)
        self._neuron_ids = torch.arange(hidden_dim, dtype=torch.int64, device=device)
        self._parent_ids = torch.full((hidden_dim,), -1, dtype=torch.int64, device=device)
        self._next_neuron_id = hidden_dim
        self._steps_since_sleep = 0
        self._wake_examples = 0
        self._sleep_events = 0
        self._energy_history: list[float] = []
        self._reward_error_ema: float | None = None
        self._last_reward_scale = 1.0
        generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed + 9_999)
        self._split_generator = generator

    def get_neuron_lineage(self) -> NeuronLineageSnapshot:
        """Return active stable IDs and birth-parent IDs independent of tensor indices."""
        return NeuronLineageSnapshot(
            neuron_ids=tuple(int(value) for value in self._neuron_ids.tolist()),
            parent_ids=tuple(
                None if value < 0 else int(value) for value in self._parent_ids.tolist()
            ),
            next_neuron_id=self._next_neuron_id,
        )

    def get_pending_prune_ids(self) -> tuple[int, ...]:
        """Torch removes selected neurons during sleep and has no pending marks."""
        return ()

    def get_sleep_clocks(self) -> SleepClockSnapshot:
        """Return successful wake/sleep counts; replay is unavailable for this head."""
        return SleepClockSnapshot(
            wake_batches=self._traffic_steps,
            wake_examples=self._wake_examples,
            wake_batches_since_sleep=self._steps_since_sleep,
            replay_updates=0,
            sleep_events=self._sleep_events,
        )

    def train_step(
        self,
        features: Any,
        targets: Any,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> float:
        torch = self._torch
        functional = torch.nn.functional
        self._validate_training_inputs(
            features, targets, learning_rate, inference_steps, inference_learning_rate
        )
        sample_count = float(features.shape[0])

        hidden_linear_prior = features @ self.weight_feature_hidden + self.bias_hidden
        hidden_prior = torch.tanh(hidden_linear_prior)
        hidden_state = hidden_prior.clone()
        target_one_hot = functional.one_hot(targets, num_classes=self.num_classes).to(features.dtype)

        for _ in range(inference_steps):
            output_logits = hidden_state @ self.weight_hidden_output + self.bias_output
            output_probabilities = functional.softmax(output_logits, dim=1)
            output_error = output_probabilities - target_one_hot
            hidden_error = hidden_state - hidden_prior
            output_to_hidden = output_error @ self.weight_hidden_output.T
            hidden_gradient = hidden_error + output_to_hidden
            hidden_state = hidden_state - inference_learning_rate * hidden_gradient

        output_logits = hidden_state @ self.weight_hidden_output + self.bias_output
        self._require_finite_relaxed_state(hidden_linear_prior, hidden_state, output_logits)
        output_probabilities = functional.softmax(output_logits, dim=1)
        output_error = output_probabilities - target_one_hot
        hidden_error = hidden_state - hidden_prior

        grad_hidden_output = (hidden_state.T @ output_error) / sample_count
        grad_output_bias = torch.mean(output_error, dim=0, keepdim=True)
        hidden_prior_grad = (-hidden_error) * (1.0 - hidden_prior * hidden_prior)
        grad_feature_hidden = (features.T @ hidden_prior_grad) / sample_count
        grad_hidden_bias = torch.mean(hidden_prior_grad, dim=0, keepdim=True)

        adaptive_before = (
            self._chemical,
            self._chemical_fast,
            self._chemical_slow,
            self._importance_ema,
            self._reward_error_ema,
            self._last_reward_scale,
        )
        try:
            self._update_chemical(hidden_state)
            reward_scale = self._compute_reward_scale(output_error)
            self._last_reward_scale = reward_scale
            self._update_importance_ema(grad_hidden_output, reward_scale=reward_scale)
            plasticity = self._plasticity()

            grad_hidden_output = grad_hidden_output * plasticity[:, None]
            grad_feature_hidden = grad_feature_hidden * plasticity[None, :]
            grad_hidden_bias = grad_hidden_bias * plasticity[None, :]
            effective_learning_rate = learning_rate * reward_scale

            new_hidden_output = self.weight_hidden_output - effective_learning_rate * grad_hidden_output
            new_output_bias = self.bias_output - effective_learning_rate * grad_output_bias
            new_feature_hidden = self.weight_feature_hidden - effective_learning_rate * grad_feature_hidden
            new_hidden_bias = self.bias_hidden - effective_learning_rate * grad_hidden_bias
            new_traffic = self._traffic_sum + torch.mean(torch.abs(hidden_state), dim=0)
            new_age = self._neuron_age + 1.0
            energy = 0.5 * (
                torch.mean(output_error * output_error) + torch.mean(hidden_error * hidden_error)
            )
            energy_value = self._require_finite_training_update(
                [
                    new_hidden_output,
                    new_output_bias,
                    new_feature_hidden,
                    new_hidden_bias,
                    new_traffic,
                    new_age,
                    self._chemical,
                    self._chemical_fast,
                    self._chemical_slow,
                    self._importance_ema,
                ],
                energy,
            )
            if not isfinite(reward_scale) or (
                self._reward_error_ema is not None and not isfinite(self._reward_error_ema)
            ):
                raise FloatingPointError("nonfinite training update or diagnostic")
        except Exception:
            (
                self._chemical,
                self._chemical_fast,
                self._chemical_slow,
                self._importance_ema,
                self._reward_error_ema,
                self._last_reward_scale,
            ) = adaptive_before
            raise

        self._decay_cooldowns()
        self.weight_hidden_output = new_hidden_output
        self.bias_output = new_output_bias
        self.weight_feature_hidden = new_feature_hidden
        self.bias_hidden = new_hidden_bias
        self._traffic_sum = new_traffic
        self._traffic_steps += 1
        self._neuron_age = new_age
        self._steps_since_sleep += 1
        self._wake_examples += int(features.shape[0])
        self._energy_history.append(energy_value)
        max_history = max(self.config.sleep_energy_window * 4, 16)
        if len(self._energy_history) > max_history:
            self._energy_history = self._energy_history[-max_history:]
        return energy_value

    def sleep_event(
        self,
        current_step: int | None = None,
        total_steps: int | None = None,
        force_sleep: bool = True,
        epoch_progress: SleepEpochProgress | None = None,
    ) -> SleepEventResult:
        if epoch_progress is not None:
            if current_step is not None or total_steps is not None:
                raise ValueError("epoch_progress cannot be combined with current_step/total_steps")
            current_step = epoch_progress.completed_epochs
            total_steps = epoch_progress.total_epochs
        if self.config.sleep_mode == "disabled":
            return SleepEventResult(
                old_hidden_dim=self.hidden_dim,
                new_hidden_dim=self.hidden_dim,
                split_indices=(),
                pruned_indices=(),
            )
        if not force_sleep and not self.should_trigger_sleep():
            return SleepEventResult(
                old_hidden_dim=self.hidden_dim,
                new_hidden_dim=self.hidden_dim,
                split_indices=(),
                pruned_indices=(),
            )
        split_budget, prune_budget, should_skip = self._resolve_sleep_budgets(
            current_step=current_step,
            total_steps=total_steps,
        )
        if should_skip or (
            self.config.sleep_mode == "legacy" and split_budget <= 0 and prune_budget <= 0
        ):
            return SleepEventResult(
                old_hidden_dim=self.hidden_dim,
                new_hidden_dim=self.hidden_dim,
                split_indices=(),
                pruned_indices=(),
            )
        if self.config.sleep_mode == "components":
            split_budget = split_budget if self.config.sleep_enable_split else 0
            prune_budget = prune_budget if self.config.sleep_enable_prune else 0

        snapshot = self.snapshot_state()
        try:
            return self._execute_sleep_event(split_budget=split_budget, prune_budget=prune_budget)
        except Exception:
            self.restore_state(snapshot)
            raise

    def _execute_sleep_event(self, *, split_budget: int, prune_budget: int) -> SleepEventResult:
        old_hidden_dim = self.hidden_dim
        split_indices, prune_indices = self._plan_structural_sleep(
            split_budget=split_budget, prune_budget=prune_budget
        )
        lineage_before = self.get_neuron_lineage()
        self._apply_split(split_indices)
        proposed_ids = tuple(int(value) for value in self._neuron_ids[list(prune_indices)].tolist())
        self._apply_prune(prune_indices)
        if self.config.sleep_mode == "legacy" or self.config.sleep_enable_homeostasis:
            self._apply_homeostatic_downscaling()

        if self.config.sleep_mode == "legacy" or self.config.sleep_enable_chemical_reset:
            self._chemical = self._chemical * self.config.sleep_reset_factor
            if self.config.use_dual_chemical:
                self._chemical_fast = self._chemical_fast * self.config.sleep_reset_factor
                self._chemical_slow = self._chemical_slow * self.config.sleep_reset_factor
        if self.config.sleep_mode == "components" and self.hidden_dim != old_hidden_dim:
            self._energy_history.clear()
        self._steps_since_sleep = 0
        self._sleep_events += 1
        self._validate_sleep_post_state()
        return SleepEventResult(
            old_hidden_dim=old_hidden_dim,
            new_hidden_dim=self.hidden_dim,
            split_indices=split_indices,
            pruned_indices=prune_indices,
            performed=True,
            lineage_before=lineage_before,
            lineage_after=self.get_neuron_lineage(),
            prune_outcome=PruneOutcome(
                proposed_neuron_ids=proposed_ids,
                removed_neuron_ids=proposed_ids,
            ),
        )

    def _validate_sleep_post_state(self) -> None:
        self._validate_training_topology()
        torch = self._torch
        tensors = [getattr(self, attr) for attr in self._SNAPSHOT_TENSORS.values()]
        if not bool(torch.stack([torch.isfinite(value).all() for value in tensors]).all().item()):
            raise FloatingPointError("nonfinite sleep head state")
        if (
            any(not isfinite(value) for value in self._energy_history)
            or (self._reward_error_ema is not None and not isfinite(self._reward_error_ema))
            or not isfinite(self._last_reward_scale)
        ):
            raise FloatingPointError("nonfinite sleep scalar state")

    def _plan_structural_sleep(
        self, *, split_budget: int, prune_budget: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Validate the post-split proposal on detached tensors and RNG."""
        if split_budget <= 0 and prune_budget <= 0:
            return (), ()
        self._validate_training_topology()
        split_indices = self._select_split_indices(max_split_limit=split_budget)
        self._validate_split_proposal(split_indices, split_budget)
        if prune_budget <= 0:
            return split_indices, ()
        if not split_indices:
            prune_indices = self._select_prune_indices(max_prune_limit=prune_budget)
            self._validate_prune_proposal(prune_indices, prune_budget)
            return (), prune_indices

        # Why this: Torch historically selects prune after a noisy split.
        # Simulating on detached head tensors preserves that rule without
        # changing live weights or advancing the live split generator.
        candidate = copy(self)
        candidate.__dict__ = dict(self.__dict__)
        for attr in self._SNAPSHOT_TENSORS.values():
            candidate.__dict__[attr] = getattr(self, attr).detach().clone()
        generator = self._torch.Generator(device=self._split_generator.device)
        generator.set_state(self._split_generator.get_state())
        candidate._split_generator = generator
        candidate._apply_split(split_indices)
        prune_indices = candidate._select_prune_indices(max_prune_limit=prune_budget)
        candidate._validate_prune_proposal(prune_indices, prune_budget)
        candidate._apply_prune(prune_indices)
        candidate._validate_training_topology()
        return split_indices, prune_indices

    def _validate_split_proposal(self, indices: tuple[int, ...], budget: int) -> None:
        cap = min(budget, self._resolve_structural_budget(self.config.max_split_per_sleep, 1.0))
        if not isinstance(indices, tuple) or len(indices) > cap:
            raise ValueError("built-in split budget exceeded")
        if self.hidden_dim + len(indices) > self._max_hidden_dim:
            raise ValueError("built-in split exceeds maximum hidden width")
        if any(type(index) is not int or not 0 <= index < self.hidden_dim for index in indices):
            raise ValueError("built-in split index is out of range")
        if len(set(indices)) != len(indices):
            raise ValueError("built-in split indices must be unique")
        split_threshold, _ = self._resolve_split_prune_thresholds()
        split_threshold += self.config.split_hysteresis_margin
        for index in indices:
            if self._chemical[index] < split_threshold or self._split_cooldown[index] > 0:
                raise ValueError("built-in split index is ineligible")

    def _validate_prune_proposal(self, indices: tuple[int, ...], budget: int) -> None:
        cap = min(budget, self.config.max_prune_per_sleep)
        if not isinstance(indices, tuple) or len(indices) > cap:
            raise ValueError("built-in prune budget exceeded")
        if self.hidden_dim - len(indices) < self._min_hidden_dim:
            raise ValueError("built-in prune exceeds minimum hidden width")
        if any(type(index) is not int or not 0 <= index < self.hidden_dim for index in indices):
            raise ValueError("built-in prune index is out of range")
        if len(set(indices)) != len(indices):
            raise ValueError("built-in prune indices must be unique")
        _, prune_threshold = self._resolve_split_prune_thresholds()
        prune_threshold -= self.config.prune_hysteresis_margin
        for index in indices:
            if (
                self._chemical[index] > prune_threshold
                or self._prune_cooldown[index] > 0
                or self._neuron_age[index] < float(self.config.prune_min_age_steps)
            ):
                raise ValueError("built-in prune index is ineligible")

    def should_trigger_sleep(self) -> bool:
        if self.config.sleep_mode == "disabled":
            return False
        if not self.config.use_adaptive_sleep_trigger:
            return False
        if self._steps_since_sleep < self.config.min_sleep_steps:
            return False
        if len(self._energy_history) < self.config.sleep_energy_window:
            return False

        recent = self._energy_history[-self.config.sleep_energy_window :]
        energy_improvement = recent[0] - recent[-1]
        plateau = energy_improvement <= self.config.sleep_plateau_delta
        chemical_variance = float(self._torch.var(self._chemical).item())
        high_chemical_variance = (
            chemical_variance >= self.config.sleep_chemical_variance_threshold
        )
        return plateau and high_chemical_variance

    def snapshot_state(self) -> dict[str, Any]:
        state = {
            name: getattr(self, attr).detach().clone()
            for name, attr in self._SNAPSHOT_TENSORS.items()
        }
        state.update(
            {name: getattr(self, attr) for name, attr in self._SNAPSHOT_COUNTERS.items()}
        )
        state.update(
            format_version=2,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            min_hidden_dim=self._min_hidden_dim,
            max_hidden_dim=self._max_hidden_dim,
            device=str(self.device),
            generator_device=str(self._split_generator.device),
            config=asdict(self.config),
            split_generator_state=self._split_generator.get_state().clone(),
            energy_history=list(self._energy_history),
            reward_error_ema=self._reward_error_ema,
            last_reward_scale=self._last_reward_scale,
        )
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        staged, generator = self._prepare_snapshot_restore(state)
        self._commit_snapshot_restore(staged, generator)

    def _prepare_snapshot_restore(self, state: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        expected = self.snapshot_state()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("snapshot fields are incompatible with this Torch head")
        for name in (
            "format_version", "feature_dim", "num_classes", "min_hidden_dim",
            "max_hidden_dim", "device", "generator_device", "config",
        ):
            if state[name] != expected[name]:
                raise ValueError(f"snapshot {name} is incompatible with this Torch head")

        torch = self._torch
        staged: dict[str, Any] = {}
        for name, attr in self._SNAPSHOT_TENSORS.items():
            value = state[name]
            reference = expected[name]
            if not isinstance(value, torch.Tensor) or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"snapshot {name} has incompatible dtype or device")
            staged[attr] = value.detach().clone()
        for name, attr in self._SNAPSHOT_COUNTERS.items():
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"snapshot {name} must be a nonnegative integer")
            staged[attr] = value

        history = state["energy_history"]
        if not isinstance(history, list) or any(
            type(value) is not float or not isfinite(value) for value in history
        ):
            raise ValueError("snapshot energy_history must contain finite floats")
        staged["_energy_history"] = list(history)
        reward = state["reward_error_ema"]
        scale = state["last_reward_scale"]
        if reward is not None and (type(reward) is not float or not isfinite(reward)):
            raise ValueError("snapshot reward_error_ema must be finite or None")
        if type(scale) is not float or not isfinite(scale):
            raise ValueError("snapshot last_reward_scale must be finite")
        staged["_reward_error_ema"] = reward
        staged["_last_reward_scale"] = scale

        candidate = object.__new__(type(self))
        candidate.__dict__ = {**self.__dict__, **staged}
        try:
            candidate._validate_training_topology()
        except ValueError as exc:
            raise ValueError(f"snapshot topology is incompatible: {exc}") from exc
        if not self._min_hidden_dim <= candidate.hidden_dim <= self._max_hidden_dim:
            raise ValueError("snapshot hidden width is outside configured bounds")
        generator_state = state["split_generator_state"]
        if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8:
            raise ValueError("snapshot split generator state must be a byte tensor")
        generator = torch.Generator(device=self._split_generator.device)
        try:
            generator.set_state(generator_state.detach().clone())
        except (RuntimeError, ValueError) as exc:
            raise ValueError("snapshot split generator state is invalid") from exc
        return staged, generator

    def _commit_snapshot_restore(self, staged: dict[str, Any], generator: Any) -> None:
        # Why this: stage every value, including RNG, before the single live-state update.
        self.__dict__.update(staged)
        self._split_generator = generator

    def mean_chemical(self) -> Any:
        return self._chemical.clone()

    def last_reward_scale(self) -> float:
        return float(self._last_reward_scale)

    def _plasticity(self) -> Any:
        torch = self._torch
        sensitivity = self._plasticity_sensitivity_vector()
        values = torch.exp(-sensitivity * self._chemical)
        return torch.clamp(values, min=self.config.min_plasticity, max=1.0)

    def _plasticity_sensitivity_vector(self) -> Any:
        torch = self._torch
        if not self.config.use_adaptive_plasticity_sensitivity:
            return torch.full_like(self._chemical, self.config.plasticity_sensitivity)

        age_component = self._normalize_tensor_zero_base(self._neuron_age)
        importance_component = self._normalize_tensor_zero_base(self._importance_ema)
        importance_mix = float(max(0.0, min(1.0, self.config.plasticity_importance_mix)))
        stability = importance_mix * importance_component + (1.0 - importance_mix) * age_component
        span = self.config.plasticity_sensitivity_max - self.config.plasticity_sensitivity_min
        base = torch.full_like(self._chemical, self.config.plasticity_sensitivity_min)
        return base + span * stability

    def _update_importance_ema(self, grad_hidden_output: Any, reward_scale: float) -> None:
        torch = self._torch
        importance = torch.mean(torch.abs(grad_hidden_output), dim=1) * reward_scale
        decay = self.config.importance_ema_decay
        self._importance_ema = decay * self._importance_ema + (1.0 - decay) * importance

    def _compute_reward_scale(self, output_error: Any) -> float:
        torch = self._torch
        if not self.config.use_reward_modulated_learning:
            return 1.0

        batch_error = float(torch.mean(torch.abs(output_error)).item())
        baseline = batch_error if self._reward_error_ema is None else self._reward_error_ema
        difficulty_ratio = batch_error / max(float(baseline), 1e-8)
        raw_scale = difficulty_ratio ** self.config.reward_difficulty_exponent
        reward_scale = float(
            max(self.config.reward_scale_min, min(self.config.reward_scale_max, raw_scale))
        )
        decay = self.config.reward_baseline_decay
        self._reward_error_ema = decay * float(baseline) + (1.0 - decay) * batch_error
        return reward_scale

    def _update_chemical(self, hidden_state: Any) -> None:
        torch = self._torch
        activity = torch.mean(torch.abs(hidden_state), dim=0)
        if not self.config.use_dual_chemical:
            self._chemical = self._accumulate_chemical(
                current=self._chemical,
                decay=self.config.chemical_decay,
                buildup_rate=self.config.chemical_buildup_rate,
                activity=activity,
            )
            self._chemical_fast = self._chemical.clone()
            self._chemical_slow = self._chemical.clone()
            return

        self._chemical_fast = self._accumulate_chemical(
            current=self._chemical_fast,
            decay=self.config.chemical_decay,
            buildup_rate=self.config.chemical_buildup_rate,
            activity=activity,
        )
        slow_rate = self.config.chemical_buildup_rate * self.config.slow_buildup_scale
        self._chemical_slow = self._accumulate_chemical(
            current=self._chemical_slow,
            decay=self.config.slow_chemical_decay,
            buildup_rate=slow_rate,
            activity=activity,
        )
        fast_mix = float(max(0.0, min(1.0, self.config.dual_fast_mix)))
        self._chemical = fast_mix * self._chemical_fast + (1.0 - fast_mix) * self._chemical_slow

    def _accumulate_chemical(
        self, current: Any, decay: float, buildup_rate: float, activity: Any
    ) -> Any:
        torch = self._torch
        decayed = decay * current
        increment = buildup_rate * activity
        if not self.config.use_saturating_chemical:
            return decayed + increment

        max_value = self.config.chemical_max_value
        headroom = torch.clamp(max_value - decayed, min=0.0)
        scaled_increment = (self.config.chemical_saturation_gain * increment) / max(max_value, 1e-8)
        saturating_delta = headroom * (1.0 - torch.exp(-scaled_increment))
        return torch.minimum(decayed + saturating_delta, torch.full_like(decayed, max_value))

    def _decay_cooldowns(self) -> None:
        torch = self._torch
        self._split_cooldown = torch.clamp(self._split_cooldown - 1, min=0)
        self._prune_cooldown = torch.clamp(self._prune_cooldown - 1, min=0)

    def _select_split_indices(self, max_split_limit: int | None = None) -> tuple[int, ...]:
        torch = self._torch
        split_threshold, _ = self._resolve_split_prune_thresholds()
        split_threshold += self.config.split_hysteresis_margin
        remaining_capacity = self._max_hidden_dim - self.hidden_dim
        if remaining_capacity <= 0 or self.config.max_split_per_sleep <= 0:
            return ()
        limit = self.config.max_split_per_sleep if max_split_limit is None else max_split_limit
        if limit <= 0:
            return ()
        candidates = torch.where(
            (self._chemical >= split_threshold) & (self._split_cooldown <= 0)
        )[0]
        if int(candidates.numel()) == 0:
            return ()
        split_scores = self._compute_split_scores()
        candidate_values = split_scores[candidates]
        sorted_order = torch.argsort(candidate_values, descending=True)
        sorted_candidates = candidates[sorted_order]
        split_count = min(int(sorted_candidates.numel()), int(limit), remaining_capacity)
        chosen = sorted_candidates[:split_count].tolist()
        return tuple(int(index) for index in chosen)

    def _select_prune_indices(self, max_prune_limit: int | None = None) -> tuple[int, ...]:
        torch = self._torch
        _, prune_threshold = self._resolve_split_prune_thresholds()
        prune_threshold -= self.config.prune_hysteresis_margin
        removable = self.hidden_dim - self._min_hidden_dim
        if removable <= 0 or self.config.max_prune_per_sleep <= 0:
            return ()
        limit = self.config.max_prune_per_sleep if max_prune_limit is None else max_prune_limit
        if limit <= 0:
            return ()
        candidates = torch.where(
            (self._chemical <= prune_threshold)
            & (self._prune_cooldown <= 0)
            & (self._neuron_age >= float(self.config.prune_min_age_steps))
        )[0]
        if int(candidates.numel()) == 0:
            return ()
        prune_scores = self._compute_prune_scores()
        candidate_values = prune_scores[candidates]
        sorted_order = torch.argsort(candidate_values, descending=True)
        sorted_candidates = candidates[sorted_order]
        prune_count = min(int(sorted_candidates.numel()), int(limit), removable)
        chosen = sorted_candidates[:prune_count].tolist()
        return tuple(sorted(int(index) for index in chosen))

    def _resolve_split_prune_thresholds(self) -> tuple[float, float]:
        torch = self._torch
        if not self.config.use_adaptive_thresholds:
            return self.config.split_threshold, self.config.prune_threshold

        split_threshold = float(torch.quantile(self._chemical, self.config.adaptive_split_percentile / 100.0))
        prune_threshold = float(torch.quantile(self._chemical, self.config.adaptive_prune_percentile / 100.0))
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
        torch = self._torch
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
            plateau_score = max(0.0, min(1.0, (plateau_delta - energy_improvement) / plateau_delta))

        variance_threshold = self.config.sleep_chemical_variance_threshold
        if variance_threshold <= 1e-12:
            variance_score = 1.0
        else:
            chemical_variance = float(torch.var(self._chemical).item())
            variance_score = max(0.0, min(1.0, chemical_variance / variance_threshold))

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
        scaled_limit = int(float(configured_limit) * budget_scale)
        if scaled_limit <= 0:
            scaled_limit = 1
        fraction = self.config.sleep_max_change_fraction
        if fraction <= 0.0:
            return 0
        by_fraction = int(float(self.hidden_dim) * fraction)
        by_fraction = max(by_fraction, int(self.config.sleep_min_change_count))
        return min(int(scaled_limit), int(by_fraction))

    def _compute_split_scores(self) -> Any:
        norm_scores = self._normalize_tensor(self._row_norm(self.weight_hidden_output))
        chemical_scores = self._normalize_tensor(self._chemical)
        importance_scores = self._normalize_tensor(self._importance_ema)
        norm_mix = max(0.0, min(1.0, self.config.split_weight_norm_mix))
        importance_mix = max(0.0, min(1.0, self.config.split_importance_mix))
        chemical_mix = max(0.0, 1.0 - norm_mix - importance_mix)
        return chemical_mix * chemical_scores + norm_mix * norm_scores + importance_mix * importance_scores

    def _compute_prune_scores(self) -> Any:
        norm_scores = 1.0 - self._normalize_tensor(self._row_norm(self.weight_hidden_output))
        chemical_scores = 1.0 - self._normalize_tensor(self._chemical)
        importance_scores = 1.0 - self._normalize_tensor(self._importance_ema)
        norm_mix = max(0.0, min(1.0, self.config.prune_weight_norm_mix))
        importance_mix = max(0.0, min(1.0, self.config.prune_importance_mix))
        chemical_mix = max(0.0, 1.0 - norm_mix - importance_mix)
        return chemical_mix * chemical_scores + norm_mix * norm_scores + importance_mix * importance_scores

    def _row_norm(self, matrix: Any) -> Any:
        torch = self._torch
        return torch.linalg.norm(matrix, dim=1)

    def _normalize_tensor(self, values: Any) -> Any:
        torch = self._torch
        max_value = torch.max(values)
        min_value = torch.min(values)
        span = max_value - min_value
        if float(span.item()) <= 1e-12:
            return torch.ones_like(values)
        return (values - min_value) / span

    def _normalize_tensor_zero_base(self, values: Any) -> Any:
        torch = self._torch
        max_value = torch.max(values)
        min_value = torch.min(values)
        span = max_value - min_value
        if float(span.item()) <= 1e-12:
            return torch.zeros_like(values)
        return (values - min_value) / span

    def _apply_split(self, split_indices: tuple[int, ...]) -> None:
        if not split_indices:
            return
        torch = self._torch
        child_ids = torch.arange(
            self._next_neuron_id,
            self._next_neuron_id + len(split_indices),
            dtype=torch.int64,
            device=self.device,
        )
        child_parents = self._neuron_ids[list(split_indices)].clone()

        input_columns = []
        output_rows = []
        bias_values = []
        chemical_values = []
        chemical_fast_values = []
        chemical_slow_values = []
        age_values = []
        traffic_values = []
        importance_values = []
        split_cooldown_values = []
        prune_cooldown_values = []

        for index in split_indices:
            input_column = self.weight_feature_hidden[:, index]
            output_row = self.weight_hidden_output[index, :].clone()
            bias_value = self.bias_hidden[0, index]
            chemical_value = self._chemical[index]
            chemical_fast_value = self._chemical_fast[index]
            chemical_slow_value = self._chemical_slow[index]
            age_value = self._neuron_age[index]
            importance_value = self._importance_ema[index]
            parent_norm = float(torch.linalg.norm(output_row).item())
            noise_scale = self.config.split_noise_scale * max(parent_norm, 1e-3)
            output_noise = noise_scale * torch.randn(
                output_row.shape,
                generator=self._split_generator,
                dtype=torch.float32,
                device=self.device,
            )

            self.weight_hidden_output[index, :] = 0.5 * output_row + output_noise

            # Keep split function-preserving by duplicating incoming pathway while
            # ensuring the parent and child outgoing rows sum to the original row.
            input_columns.append(input_column.unsqueeze(1))
            output_rows.append((0.5 * output_row - output_noise).unsqueeze(0))
            bias_values.append(bias_value.reshape(1, 1))
            chemical_values.append((chemical_value * 0.5).reshape(1))
            chemical_fast_values.append((chemical_fast_value * 0.5).reshape(1))
            chemical_slow_values.append((chemical_slow_value * 0.5).reshape(1))
            age_values.append(torch.tensor([0.0], dtype=torch.float32, device=self.device))
            traffic_values.append(torch.zeros((1,), dtype=torch.float32, device=self.device))
            importance_values.append(importance_value.reshape(1))
            split_cooldown_values.append(
                torch.tensor([self.config.split_cooldown_steps], dtype=torch.int32, device=self.device)
            )
            prune_cooldown_values.append(
                torch.tensor([self.config.prune_cooldown_steps], dtype=torch.int32, device=self.device)
            )

            self._chemical[index] = chemical_value * 0.5
            self._chemical_fast[index] = chemical_fast_value * 0.5
            self._chemical_slow[index] = chemical_slow_value * 0.5
            self._neuron_age[index] = age_value
            self._split_cooldown[index] = self.config.split_cooldown_steps
            self._prune_cooldown[index] = self.config.prune_cooldown_steps

        self.weight_feature_hidden = torch.cat([self.weight_feature_hidden] + input_columns, dim=1)
        self.weight_hidden_output = torch.cat([self.weight_hidden_output] + output_rows, dim=0)
        self.bias_hidden = torch.cat([self.bias_hidden, torch.cat(bias_values, dim=1)], dim=1)
        self._chemical = torch.cat([self._chemical, torch.cat(chemical_values, dim=0)], dim=0)
        self._chemical_fast = torch.cat([self._chemical_fast, torch.cat(chemical_fast_values, dim=0)], dim=0)
        self._chemical_slow = torch.cat([self._chemical_slow, torch.cat(chemical_slow_values, dim=0)], dim=0)
        self._neuron_age = torch.cat([self._neuron_age, torch.cat(age_values, dim=0)], dim=0)
        self._traffic_sum = torch.cat([self._traffic_sum, torch.cat(traffic_values, dim=0)], dim=0)
        self._importance_ema = torch.cat([self._importance_ema, torch.cat(importance_values, dim=0)], dim=0)
        self._split_cooldown = torch.cat(
            [self._split_cooldown, torch.cat(split_cooldown_values, dim=0)], dim=0
        )
        self._prune_cooldown = torch.cat(
            [self._prune_cooldown, torch.cat(prune_cooldown_values, dim=0)], dim=0
        )
        self._neuron_ids = torch.cat([self._neuron_ids, child_ids], dim=0)
        self._parent_ids = torch.cat([self._parent_ids, child_parents], dim=0)
        self._next_neuron_id += len(split_indices)

    def _apply_prune(self, prune_indices: tuple[int, ...]) -> None:
        if not prune_indices:
            return
        torch = self._torch

        mask = torch.ones(self.hidden_dim, dtype=torch.bool, device=self.device)
        mask[list(prune_indices)] = False
        self.weight_feature_hidden = self.weight_feature_hidden[:, mask]
        self.weight_hidden_output = self.weight_hidden_output[mask, :]
        self.bias_hidden = self.bias_hidden[:, mask]
        self._chemical = self._chemical[mask]
        self._chemical_fast = self._chemical_fast[mask]
        self._chemical_slow = self._chemical_slow[mask]
        self._neuron_age = self._neuron_age[mask]
        self._traffic_sum = self._traffic_sum[mask]
        self._importance_ema = self._importance_ema[mask]
        self._split_cooldown = self._split_cooldown[mask]
        self._prune_cooldown = self._prune_cooldown[mask]
        self._neuron_ids = self._neuron_ids[mask]
        self._parent_ids = self._parent_ids[mask]

    def _apply_homeostatic_downscaling(self) -> None:
        scale_factor = self.config.homeostatic_downscale_factor
        if abs(scale_factor - 1.0) > 1e-12:
            self.weight_feature_hidden = self.weight_feature_hidden * scale_factor
            self.weight_hidden_output = self.weight_hidden_output * scale_factor
            self.bias_hidden = self.bias_hidden * scale_factor
            self.bias_output = self.bias_output * scale_factor
        self._match_target_norms()

    def _match_target_norms(self) -> None:
        if self.config.homeostasis_target_input_norm > 0.0:
            self.weight_feature_hidden = self._scale_columns_to_target(
                matrix=self.weight_feature_hidden,
                target_norm=self.config.homeostasis_target_input_norm,
                strength=self.config.homeostasis_strength,
            )
        if self.config.homeostasis_target_output_norm > 0.0:
            self.weight_hidden_output = self._scale_rows_to_target(
                matrix=self.weight_hidden_output,
                target_norm=self.config.homeostasis_target_output_norm,
                strength=self.config.homeostasis_strength,
            )

    def _scale_columns_to_target(self, matrix: Any, target_norm: float, strength: float) -> Any:
        torch = self._torch
        norms = torch.linalg.norm(matrix, dim=0)
        safe_norms = torch.clamp(norms, min=1e-8)
        raw_scale = torch.pow(target_norm / safe_norms, strength)
        scale = torch.clamp(raw_scale, min=0.5, max=2.0)
        return matrix * scale[None, :]

    def _scale_rows_to_target(self, matrix: Any, target_norm: float, strength: float) -> Any:
        torch = self._torch
        norms = torch.linalg.norm(matrix, dim=1)
        safe_norms = torch.clamp(norms, min=1e-8)
        raw_scale = torch.pow(target_norm / safe_norms, strength)
        scale = torch.clamp(raw_scale, min=0.5, max=2.0)
        return matrix * scale[:, None]

    def _validate_config(self, config: CircadianHeadConfig) -> None:
        for field in fields(config):
            if isinstance(field.default, (int, float)):
                value = getattr(config, field.name)
                try:
                    finite = isfinite(value)
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError(f"{field.name} must be finite.")
        if config.sleep_mode not in {"legacy", "components", "disabled"}:
            raise ValueError("sleep_mode must be one of: legacy, components, disabled.")
        component_names = (
            "sleep_enable_chemical_reset",
            "sleep_enable_homeostasis",
            "sleep_enable_split",
            "sleep_enable_prune",
        )
        if any(not isinstance(getattr(config, name), bool) for name in component_names):
            raise ValueError("sleep component switches must be booleans.")
        if config.sleep_mode == "legacy" and any(
            not getattr(config, name) for name in component_names
        ):
            raise ValueError("sleep_mode='components' is required for component switches.")
        if not (0.0 <= config.chemical_decay <= 1.0):
            raise ValueError("chemical_decay must be between 0 and 1.")
        if config.chemical_max_value <= 0.0:
            raise ValueError("chemical_max_value must be positive.")
        if config.chemical_saturation_gain <= 0.0:
            raise ValueError("chemical_saturation_gain must be positive.")
        if not (0.0 <= config.slow_chemical_decay <= 1.0):
            raise ValueError("slow_chemical_decay must be between 0 and 1.")
        if not (0.0 <= config.dual_fast_mix <= 1.0):
            raise ValueError("dual_fast_mix must be between 0 and 1.")
        if config.chemical_buildup_rate <= 0.0:
            raise ValueError("chemical_buildup_rate must be positive.")
        if config.slow_buildup_scale <= 0.0:
            raise ValueError("slow_buildup_scale must be positive.")
        if config.plasticity_sensitivity <= 0.0:
            raise ValueError("plasticity_sensitivity must be positive.")
        if config.plasticity_sensitivity_min <= 0.0:
            raise ValueError("plasticity_sensitivity_min must be positive.")
        if config.plasticity_sensitivity_max < config.plasticity_sensitivity_min:
            raise ValueError(
                "plasticity_sensitivity_max must be greater than or equal to plasticity_sensitivity_min."
            )
        if not (0.0 <= config.plasticity_importance_mix <= 1.0):
            raise ValueError("plasticity_importance_mix must be between 0 and 1.")
        if not (0.0 < config.min_plasticity <= 1.0):
            raise ValueError("min_plasticity must be in (0, 1].")
        if not (0.0 <= config.reward_baseline_decay < 1.0):
            raise ValueError("reward_baseline_decay must be in [0, 1).")
        if config.reward_difficulty_exponent <= 0.0:
            raise ValueError("reward_difficulty_exponent must be positive.")
        if config.reward_scale_min <= 0.0:
            raise ValueError("reward_scale_min must be positive.")
        if config.reward_scale_max < config.reward_scale_min:
            raise ValueError("reward_scale_max must be >= reward_scale_min.")
        if not (0.0 <= config.adaptive_split_percentile <= 100.0):
            raise ValueError("adaptive_split_percentile must be between 0 and 100.")
        if not (0.0 <= config.adaptive_prune_percentile <= 100.0):
            raise ValueError("adaptive_prune_percentile must be between 0 and 100.")
        if config.split_hysteresis_margin < 0.0 or config.prune_hysteresis_margin < 0.0:
            raise ValueError("split/prune hysteresis margins must be non-negative.")
        if config.split_cooldown_steps < 0 or config.prune_cooldown_steps < 0:
            raise ValueError("split/prune cooldown steps must be non-negative.")
        if not (0.0 <= config.split_weight_norm_mix <= 1.0):
            raise ValueError("split_weight_norm_mix must be between 0 and 1.")
        if not (0.0 <= config.prune_weight_norm_mix <= 1.0):
            raise ValueError("prune_weight_norm_mix must be between 0 and 1.")
        if not (0.0 <= config.split_importance_mix <= 1.0):
            raise ValueError("split_importance_mix must be between 0 and 1.")
        if not (0.0 <= config.prune_importance_mix <= 1.0):
            raise ValueError("prune_importance_mix must be between 0 and 1.")
        if not (0.0 <= config.importance_ema_decay < 1.0):
            raise ValueError("importance_ema_decay must be in [0, 1).")
        if config.max_split_per_sleep < 0 or config.max_prune_per_sleep < 0:
            raise ValueError("max split/prune per sleep must be non-negative.")
        if config.split_noise_scale < 0.0:
            raise ValueError("split_noise_scale must be non-negative.")
        if not (0.0 <= config.sleep_reset_factor <= 1.0):
            raise ValueError("sleep_reset_factor must be between 0 and 1.")
        if config.sleep_warmup_steps < 0:
            raise ValueError("sleep_warmup_steps must be non-negative.")
        if not (0.0 <= config.sleep_split_only_until_fraction <= 1.0):
            raise ValueError("sleep_split_only_until_fraction must be between 0 and 1.")
        if not (0.0 <= config.sleep_prune_only_after_fraction <= 1.0):
            raise ValueError("sleep_prune_only_after_fraction must be between 0 and 1.")
        if config.sleep_split_only_until_fraction > config.sleep_prune_only_after_fraction:
            raise ValueError(
                "sleep_split_only_until_fraction cannot exceed sleep_prune_only_after_fraction."
            )
        if not (0.0 <= config.sleep_max_change_fraction <= 1.0):
            raise ValueError("sleep_max_change_fraction must be between 0 and 1.")
        if config.sleep_min_change_count < 0:
            raise ValueError("sleep_min_change_count must be non-negative.")
        if config.prune_min_age_steps < 0:
            raise ValueError("prune_min_age_steps must be non-negative.")
        if not (0.0 < config.homeostatic_downscale_factor <= 1.0):
            raise ValueError("homeostatic_downscale_factor must be in (0, 1].")
        if config.homeostasis_target_input_norm < 0.0 or config.homeostasis_target_output_norm < 0.0:
            raise ValueError("homeostasis target norms must be non-negative.")
        if not (0.0 < config.homeostasis_strength <= 1.0):
            raise ValueError("homeostasis_strength must be in (0, 1].")
        if config.min_sleep_steps < 0:
            raise ValueError("min_sleep_steps must be non-negative.")
        if config.sleep_energy_window < 2:
            raise ValueError("sleep_energy_window must be at least 2.")
        if config.sleep_plateau_delta < 0.0:
            raise ValueError("sleep_plateau_delta must be non-negative.")
        if config.sleep_chemical_variance_threshold < 0.0:
            raise ValueError("sleep_chemical_variance_threshold must be non-negative.")
        if config.adaptive_sleep_budget_min_scale <= 0.0:
            raise ValueError("adaptive_sleep_budget_min_scale must be positive.")
        if config.adaptive_sleep_budget_max_scale < config.adaptive_sleep_budget_min_scale:
            raise ValueError("adaptive_sleep_budget_max_scale must be >= min scale.")
        if config.adaptive_sleep_budget_max_scale > 1.0:
            raise ValueError("adaptive_sleep_budget_max_scale must be <= 1.0.")
        if config.adaptive_sleep_budget_plateau_weight < 0.0:
            raise ValueError("adaptive_sleep_budget_plateau_weight must be non-negative.")
        if config.adaptive_sleep_budget_variance_weight < 0.0:
            raise ValueError("adaptive_sleep_budget_variance_weight must be non-negative.")


class PredictiveCodingResNet50Classifier:
    """ResNet-50 feature extractor with predictive-coding classifier head."""

    def __init__(
        self,
        num_classes: int,
        device: Any,
        head_hidden_dim: int = 256,
        seed: int = 7,
        freeze_backbone: bool = True,
        backbone_weights: str = "none",
    ) -> None:
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.backbone, feature_dim = _build_resnet50_backbone(
            device=device,
            freeze_backbone=freeze_backbone,
            backbone_weights=backbone_weights,
        )
        self.head = PredictiveCodingHead(
            feature_dim=feature_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )

    def extract_features(self, images: Any) -> Any:
        torch = require_torch()
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)

    def train_step(
        self,
        images: Any,
        targets: Any,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> float:
        features = self.extract_features(images)
        return self.head.train_step(
            features=features,
            targets=targets,
            learning_rate=learning_rate,
            inference_steps=inference_steps,
            inference_learning_rate=inference_learning_rate,
        )

    def predict_logits(self, images: Any) -> Any:
        features = self.extract_features(images)
        return self.head.predict_logits(features)

    def parameter_count(self) -> int:
        return _count_parameters(self.backbone) + self.head.parameter_count()

    def trainable_parameter_count(self) -> int:
        if self.freeze_backbone:
            return self.head.parameter_count()
        return self.parameter_count()


class CircadianPredictiveCodingResNet50Classifier:
    """ResNet-50 feature extractor with circadian predictive-coding head."""

    def __init__(
        self,
        num_classes: int,
        device: Any,
        head_hidden_dim: int = 256,
        seed: int = 7,
        freeze_backbone: bool = True,
        backbone_weights: str = "none",
        circadian_config: CircadianHeadConfig | None = None,
        min_hidden_dim: int = 64,
        max_hidden_dim: int = 2048,
    ) -> None:
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.backbone, feature_dim = _build_resnet50_backbone(
            device=device,
            freeze_backbone=freeze_backbone,
            backbone_weights=backbone_weights,
        )
        self.head = CircadianPredictiveCodingHead(
            feature_dim=feature_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            device=device,
            seed=seed,
            config=circadian_config,
            min_hidden_dim=min_hidden_dim,
            max_hidden_dim=max_hidden_dim,
        )

    def extract_features(self, images: Any) -> Any:
        torch = require_torch()
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)

    def train_step(
        self,
        images: Any,
        targets: Any,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> float:
        features = self.extract_features(images)
        return self.head.train_step(
            features=features,
            targets=targets,
            learning_rate=learning_rate,
            inference_steps=inference_steps,
            inference_learning_rate=inference_learning_rate,
        )

    def predict_logits(self, images: Any) -> Any:
        features = self.extract_features(images)
        return self.head.predict_logits(features)

    def sleep_event(
        self,
        current_step: int | None = None,
        total_steps: int | None = None,
        force_sleep: bool = True,
        epoch_progress: SleepEpochProgress | None = None,
    ) -> SleepEventResult:
        return self.head.sleep_event(
            current_step=current_step,
            total_steps=total_steps,
            force_sleep=force_sleep,
            epoch_progress=epoch_progress,
        )

    def get_sleep_clocks(self) -> SleepClockSnapshot:
        return self.head.get_sleep_clocks()

    def should_trigger_sleep(self) -> bool:
        return self.head.should_trigger_sleep()

    def snapshot_state(self) -> dict[str, Any]:
        return self.head.snapshot_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        self.head.restore_state(state)

    def snapshot_full_state(self) -> CircadianClassifierSnapshot:
        """Copy the backbone and head without changing the head-only sleep guard."""
        return CircadianClassifierSnapshot(
            format_version=1,
            device=str(self.device),
            freeze_backbone=self.freeze_backbone,
            backbone_module_types=self._backbone_module_types(),
            backbone_state=deepcopy(self.backbone.state_dict()),
            backbone_training={
                name: module.training for name, module in self.backbone.named_modules()
            },
            backbone_requires_grad={
                name: parameter.requires_grad
                for name, parameter in self.backbone.named_parameters()
            },
            head_state=self.head.snapshot_state(),
        )

    def restore_full_state(self, snapshot: CircadianClassifierSnapshot) -> None:
        """Restore a compatible, detached backbone and head state in memory."""
        if not isinstance(snapshot, CircadianClassifierSnapshot) or snapshot.format_version != 1:
            raise ValueError("snapshot format is incompatible with this classifier")
        if snapshot.device != str(self.device) or snapshot.freeze_backbone != self.freeze_backbone:
            raise ValueError("snapshot device or freeze_backbone is incompatible")
        if snapshot.backbone_module_types != self._backbone_module_types():
            raise ValueError("snapshot backbone module structure is incompatible")
        current_state = self.backbone.state_dict()
        if snapshot.backbone_state.keys() != current_state.keys():
            raise ValueError("snapshot backbone state fields are incompatible")
        torch = require_torch()
        for name, current in current_state.items():
            value = snapshot.backbone_state[name]
            if torch.is_tensor(current):
                if (
                    not torch.is_tensor(value)
                    or value.shape != current.shape
                    or value.dtype != current.dtype
                    or value.device != current.device
                ):
                    raise ValueError(f"snapshot backbone tensor {name} is incompatible")
            elif type(value) is not type(current):
                raise ValueError(f"snapshot backbone field {name} is incompatible")
        modules = dict(self.backbone.named_modules())
        parameters = dict(self.backbone.named_parameters())
        if snapshot.backbone_training.keys() != modules.keys() or any(
            type(value) is not bool for value in snapshot.backbone_training.values()
        ):
            raise ValueError("snapshot backbone training modes are incompatible")
        if snapshot.backbone_requires_grad.keys() != parameters.keys() or any(
            type(value) is not bool for value in snapshot.backbone_requires_grad.values()
        ):
            raise ValueError("snapshot backbone gradient flags are incompatible")
        staged_head, generator = self.head._prepare_snapshot_restore(snapshot.head_state)
        staged_backbone = deepcopy(snapshot.backbone_state)

        # Why this: the guard keeps copying only the head; full restore is an explicit operation.
        self.backbone.load_state_dict(staged_backbone, strict=True)
        for name, module in modules.items():
            module.training = snapshot.backbone_training[name]
        for name, parameter in parameters.items():
            parameter.requires_grad_(snapshot.backbone_requires_grad[name])
        self.head._commit_snapshot_restore(staged_head, generator)

    def _backbone_module_types(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, f"{type(module).__module__}.{type(module).__qualname__}")
            for name, module in self.backbone.named_modules()
        )

    def parameter_count(self) -> int:
        return _count_parameters(self.backbone) + self.head.parameter_count()

    def trainable_parameter_count(self) -> int:
        if self.freeze_backbone:
            return self.head.parameter_count()
        return self.parameter_count()
