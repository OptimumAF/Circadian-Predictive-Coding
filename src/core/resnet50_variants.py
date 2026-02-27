"""ResNet-50 variants for backprop, predictive coding, and circadian predictive coding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from typing import Any

from src.shared.torch_runtime import require_torch, require_torchvision_models


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


@dataclass(frozen=True)
class SleepEventResult:
    """Sleep consolidation result."""

    old_hidden_dim: int
    new_hidden_dim: int
    split_indices: tuple[int, ...]
    pruned_indices: tuple[int, ...]


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

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        device: Any,
        seed: int,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

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
        output_probabilities = functional.softmax(output_logits, dim=1)
        output_error = output_probabilities - target_one_hot
        hidden_error = hidden_state - hidden_prior

        grad_hidden_output = (hidden_state.T @ output_error) / sample_count
        grad_output_bias = torch.mean(output_error, dim=0, keepdim=True)
        hidden_prior_grad = (-hidden_error) * (1.0 - hidden_prior * hidden_prior)
        grad_feature_hidden = (features.T @ hidden_prior_grad) / sample_count
        grad_hidden_bias = torch.mean(hidden_prior_grad, dim=0, keepdim=True)

        self.weight_hidden_output = self.weight_hidden_output - learning_rate * grad_hidden_output
        self.bias_output = self.bias_output - learning_rate * grad_output_bias
        self.weight_feature_hidden = self.weight_feature_hidden - learning_rate * grad_feature_hidden
        self.bias_hidden = self.bias_hidden - learning_rate * grad_hidden_bias

        self._traffic_sum = self._traffic_sum + torch.mean(torch.abs(hidden_state), dim=0)
        self._traffic_steps += 1

        energy = 0.5 * (torch.mean(output_error * output_error) + torch.mean(hidden_error * hidden_error))
        return float(energy.item())

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


class CircadianPredictiveCodingHead(PredictiveCodingHead):
    """Predictive-coding head with chemical-state plasticity and sleep events."""

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
        self._steps_since_sleep = 0
        self._energy_history: list[float] = []
        generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed + 9_999)
        self._split_generator = generator

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
        sample_count = float(features.shape[0])
        self._decay_cooldowns()

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
        output_probabilities = functional.softmax(output_logits, dim=1)
        output_error = output_probabilities - target_one_hot
        hidden_error = hidden_state - hidden_prior

        grad_hidden_output = (hidden_state.T @ output_error) / sample_count
        grad_output_bias = torch.mean(output_error, dim=0, keepdim=True)
        hidden_prior_grad = (-hidden_error) * (1.0 - hidden_prior * hidden_prior)
        grad_feature_hidden = (features.T @ hidden_prior_grad) / sample_count
        grad_hidden_bias = torch.mean(hidden_prior_grad, dim=0, keepdim=True)

        self._update_chemical(hidden_state)
        self._update_importance_ema(grad_hidden_output)
        plasticity = self._plasticity()

        grad_hidden_output = grad_hidden_output * plasticity[:, None]
        grad_feature_hidden = grad_feature_hidden * plasticity[None, :]
        grad_hidden_bias = grad_hidden_bias * plasticity[None, :]

        self.weight_hidden_output = self.weight_hidden_output - learning_rate * grad_hidden_output
        self.bias_output = self.bias_output - learning_rate * grad_output_bias
        self.weight_feature_hidden = self.weight_feature_hidden - learning_rate * grad_feature_hidden
        self.bias_hidden = self.bias_hidden - learning_rate * grad_hidden_bias

        self._traffic_sum = self._traffic_sum + torch.mean(torch.abs(hidden_state), dim=0)
        self._traffic_steps += 1
        self._neuron_age = self._neuron_age + 1.0

        energy = 0.5 * (torch.mean(output_error * output_error) + torch.mean(hidden_error * hidden_error))
        energy_value = float(energy.item())
        self._steps_since_sleep += 1
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
    ) -> SleepEventResult:
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
        if should_skip or (split_budget <= 0 and prune_budget <= 0):
            return SleepEventResult(
                old_hidden_dim=self.hidden_dim,
                new_hidden_dim=self.hidden_dim,
                split_indices=(),
                pruned_indices=(),
            )
        old_hidden_dim = self.hidden_dim
        split_indices = self._select_split_indices(max_split_limit=split_budget)
        self._apply_split(split_indices)

        prune_indices = self._select_prune_indices(max_prune_limit=prune_budget)
        self._apply_prune(prune_indices)
        self._apply_homeostatic_downscaling()

        self._chemical = self._chemical * self.config.sleep_reset_factor
        if self.config.use_dual_chemical:
            self._chemical_fast = self._chemical_fast * self.config.sleep_reset_factor
            self._chemical_slow = self._chemical_slow * self.config.sleep_reset_factor
        self._steps_since_sleep = 0
        return SleepEventResult(
            old_hidden_dim=old_hidden_dim,
            new_hidden_dim=self.hidden_dim,
            split_indices=split_indices,
            pruned_indices=prune_indices,
        )

    def should_trigger_sleep(self) -> bool:
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
        return {
            "weight_feature_hidden": self.weight_feature_hidden.clone(),
            "bias_hidden": self.bias_hidden.clone(),
            "weight_hidden_output": self.weight_hidden_output.clone(),
            "bias_output": self.bias_output.clone(),
            "chemical": self._chemical.clone(),
            "chemical_fast": self._chemical_fast.clone(),
            "chemical_slow": self._chemical_slow.clone(),
            "neuron_age": self._neuron_age.clone(),
            "importance_ema": self._importance_ema.clone(),
            "traffic_sum": self._traffic_sum.clone(),
            "traffic_steps": self._traffic_steps,
            "split_cooldown": self._split_cooldown.clone(),
            "prune_cooldown": self._prune_cooldown.clone(),
            "steps_since_sleep": self._steps_since_sleep,
            "energy_history": list(self._energy_history),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.weight_feature_hidden = cast(Any, state["weight_feature_hidden"]).clone()
        self.bias_hidden = cast(Any, state["bias_hidden"]).clone()
        self.weight_hidden_output = cast(Any, state["weight_hidden_output"]).clone()
        self.bias_output = cast(Any, state["bias_output"]).clone()
        self._chemical = cast(Any, state["chemical"]).clone()
        self._chemical_fast = cast(Any, state["chemical_fast"]).clone()
        self._chemical_slow = cast(Any, state["chemical_slow"]).clone()
        self._neuron_age = cast(Any, state["neuron_age"]).clone()
        self._importance_ema = cast(Any, state["importance_ema"]).clone()
        self._traffic_sum = cast(Any, state["traffic_sum"]).clone()
        self._traffic_steps = int(state["traffic_steps"])
        self._split_cooldown = cast(Any, state["split_cooldown"]).clone()
        self._prune_cooldown = cast(Any, state["prune_cooldown"]).clone()
        self._steps_since_sleep = int(state["steps_since_sleep"])
        self._energy_history = list(cast(list[float], state["energy_history"]))

    def mean_chemical(self) -> Any:
        return self._chemical.clone()

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

    def _update_importance_ema(self, grad_hidden_output: Any) -> None:
        torch = self._torch
        importance = torch.mean(torch.abs(grad_hidden_output), dim=1)
        decay = self.config.importance_ema_decay
        self._importance_ema = decay * self._importance_ema + (1.0 - decay) * importance

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
        split_budget = self._resolve_structural_budget(self.config.max_split_per_sleep)
        prune_budget = self._resolve_structural_budget(self.config.max_prune_per_sleep)
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

    def _resolve_structural_budget(self, configured_limit: int) -> int:
        if configured_limit <= 0:
            return 0
        fraction = self.config.sleep_max_change_fraction
        if fraction <= 0.0:
            return 0
        by_fraction = int(float(self.hidden_dim) * fraction)
        by_fraction = max(by_fraction, int(self.config.sleep_min_change_count))
        return min(int(configured_limit), int(by_fraction))

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
    ) -> SleepEventResult:
        return self.head.sleep_event(
            current_step=current_step,
            total_steps=total_steps,
            force_sleep=force_sleep,
        )

    def should_trigger_sleep(self) -> bool:
        return self.head.should_trigger_sleep()

    def snapshot_state(self) -> dict[str, Any]:
        return self.head.snapshot_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        self.head.restore_state(state)

    def parameter_count(self) -> int:
        return _count_parameters(self.backbone) + self.head.parameter_count()

    def trainable_parameter_count(self) -> int:
        if self.freeze_backbone:
            return self.head.parameter_count()
        return self.parameter_count()
