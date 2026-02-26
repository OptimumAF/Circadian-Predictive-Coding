"""ResNet-50 variants for backprop, predictive coding, and circadian predictive coding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.shared.torch_runtime import require_torch, require_torchvision_models


@dataclass(frozen=True)
class CircadianHeadConfig:
    """Circadian head dynamics for split/prune and plasticity."""

    chemical_decay: float = 0.995
    chemical_buildup_rate: float = 0.02
    plasticity_sensitivity: float = 0.7
    min_plasticity: float = 0.20
    split_threshold: float = 0.80
    prune_threshold: float = 0.08
    max_split_per_sleep: int = 2
    max_prune_per_sleep: int = 2
    split_noise_scale: float = 0.01
    sleep_reset_factor: float = 0.45


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

        energy = 0.5 * (torch.mean(output_error * output_error) + torch.mean(hidden_error * hidden_error))
        return float(energy.item())

    def sleep_event(self) -> SleepEventResult:
        old_hidden_dim = self.hidden_dim
        split_indices = self._select_split_indices()
        self._apply_split(split_indices)

        prune_indices = self._select_prune_indices()
        self._apply_prune(prune_indices)

        self._chemical = self._chemical * self.config.sleep_reset_factor
        return SleepEventResult(
            old_hidden_dim=old_hidden_dim,
            new_hidden_dim=self.hidden_dim,
            split_indices=split_indices,
            pruned_indices=prune_indices,
        )

    def mean_chemical(self) -> Any:
        return self._chemical.clone()

    def _plasticity(self) -> Any:
        torch = self._torch
        values = torch.exp(-self.config.plasticity_sensitivity * self._chemical)
        return torch.clamp(values, min=self.config.min_plasticity, max=1.0)

    def _update_chemical(self, hidden_state: Any) -> None:
        torch = self._torch
        activity = torch.mean(torch.abs(hidden_state), dim=0)
        self._chemical = self.config.chemical_decay * self._chemical + self.config.chemical_buildup_rate * activity

    def _select_split_indices(self) -> tuple[int, ...]:
        torch = self._torch
        remaining_capacity = self._max_hidden_dim - self.hidden_dim
        if remaining_capacity <= 0 or self.config.max_split_per_sleep <= 0:
            return ()
        candidates = torch.where(self._chemical >= self.config.split_threshold)[0]
        if int(candidates.numel()) == 0:
            return ()
        candidate_values = self._chemical[candidates]
        sorted_order = torch.argsort(candidate_values, descending=True)
        sorted_candidates = candidates[sorted_order]
        split_count = min(int(sorted_candidates.numel()), self.config.max_split_per_sleep, remaining_capacity)
        chosen = sorted_candidates[:split_count].tolist()
        return tuple(int(index) for index in chosen)

    def _select_prune_indices(self) -> tuple[int, ...]:
        torch = self._torch
        removable = self.hidden_dim - self._min_hidden_dim
        if removable <= 0 or self.config.max_prune_per_sleep <= 0:
            return ()
        candidates = torch.where(self._chemical <= self.config.prune_threshold)[0]
        if int(candidates.numel()) == 0:
            return ()
        candidate_values = self._chemical[candidates]
        sorted_order = torch.argsort(candidate_values, descending=False)
        sorted_candidates = candidates[sorted_order]
        prune_count = min(int(sorted_candidates.numel()), self.config.max_prune_per_sleep, removable)
        chosen = sorted_candidates[:prune_count].tolist()
        return tuple(sorted(int(index) for index in chosen))

    def _apply_split(self, split_indices: tuple[int, ...]) -> None:
        if not split_indices:
            return
        torch = self._torch

        input_columns = []
        output_rows = []
        bias_values = []
        chemical_values = []
        traffic_values = []

        for index in split_indices:
            input_column = self.weight_feature_hidden[:, index]
            output_row = self.weight_hidden_output[index, :].clone()
            bias_value = self.bias_hidden[0, index]
            chemical_value = self._chemical[index]
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
            traffic_values.append(torch.zeros((1,), dtype=torch.float32, device=self.device))

            self._chemical[index] = chemical_value * 0.5

        self.weight_feature_hidden = torch.cat([self.weight_feature_hidden] + input_columns, dim=1)
        self.weight_hidden_output = torch.cat([self.weight_hidden_output] + output_rows, dim=0)
        self.bias_hidden = torch.cat([self.bias_hidden, torch.cat(bias_values, dim=1)], dim=1)
        self._chemical = torch.cat([self._chemical, torch.cat(chemical_values, dim=0)], dim=0)
        self._traffic_sum = torch.cat([self._traffic_sum, torch.cat(traffic_values, dim=0)], dim=0)

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
        self._traffic_sum = self._traffic_sum[mask]

    def _validate_config(self, config: CircadianHeadConfig) -> None:
        if not (0.0 <= config.chemical_decay <= 1.0):
            raise ValueError("chemical_decay must be between 0 and 1.")
        if config.chemical_buildup_rate <= 0.0:
            raise ValueError("chemical_buildup_rate must be positive.")
        if not (0.0 < config.min_plasticity <= 1.0):
            raise ValueError("min_plasticity must be in (0, 1].")
        if config.max_split_per_sleep < 0 or config.max_prune_per_sleep < 0:
            raise ValueError("max split/prune per sleep must be non-negative.")
        if config.split_noise_scale < 0.0:
            raise ValueError("split_noise_scale must be non-negative.")
        if not (0.0 <= config.sleep_reset_factor <= 1.0):
            raise ValueError("sleep_reset_factor must be between 0 and 1.")


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

    def sleep_event(self) -> SleepEventResult:
        return self.head.sleep_event()

    def parameter_count(self) -> int:
        return _count_parameters(self.backbone) + self.head.parameter_count()

    def trainable_parameter_count(self) -> int:
        if self.freeze_backbone:
            return self.head.parameter_count()
        return self.parameter_count()
