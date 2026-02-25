"""Synthetic image datasets for fast and reproducible ResNet benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.shared.torch_runtime import require_torch


@dataclass(frozen=True)
class SyntheticVisionDatasetConfig:
    """Config for synthetic classification datasets."""

    train_samples: int = 2000
    test_samples: int = 500
    num_classes: int = 10
    image_size: int = 96
    batch_size: int = 32
    noise_std: float = 0.04
    seed: int = 7
    num_workers: int = 0


@dataclass(frozen=True)
class VisionDataLoaders:
    """Container for benchmark dataloaders."""

    train_loader: Any
    test_loader: Any
    num_classes: int


class SyntheticPatternDataset:
    """Easy synthetic task where classes map to fixed spatial/channel patterns."""

    def __init__(
        self,
        sample_count: int,
        num_classes: int,
        image_size: int,
        noise_std: float,
        seed: int,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        if image_size < 32:
            raise ValueError("image_size must be at least 32 for ResNet-50.")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")

        torch = require_torch()
        self._torch = torch
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.labels = torch.randint(
            low=0,
            high=num_classes,
            size=(sample_count,),
            generator=generator,
            dtype=torch.long,
        )
        self.images = torch.zeros((sample_count, 3, image_size, image_size), dtype=torch.float32)

        patch_size = max(image_size // 6, 6)
        grid_side = int(num_classes**0.5)
        if grid_side * grid_side < num_classes:
            grid_side += 1
        stride = max((image_size - patch_size - 2) // max(grid_side, 1), 1)

        for sample_index in range(sample_count):
            class_index = int(self.labels[sample_index].item())
            row = class_index // grid_side
            col = class_index % grid_side
            start_y = min(1 + row * stride, image_size - patch_size - 1)
            start_x = min(1 + col * stride, image_size - patch_size - 1)

            channel_index = class_index % 3
            secondary_channel = (class_index + 1) % 3
            intensity = 0.6 + 0.35 * (class_index / max(num_classes - 1, 1))

            self.images[sample_index, channel_index, start_y : start_y + patch_size, start_x : start_x + patch_size] = intensity
            self.images[sample_index, secondary_channel, start_y : start_y + (patch_size // 2), start_x : start_x + (patch_size // 2)] = 0.35

            diagonal_start = image_size - start_y - patch_size
            diagonal_end = image_size - start_x - patch_size
            diagonal_y = max(min(diagonal_start, image_size - patch_size - 1), 0)
            diagonal_x = max(min(diagonal_end, image_size - patch_size - 1), 0)
            self.images[sample_index, channel_index, diagonal_y : diagonal_y + (patch_size // 3), diagonal_x : diagonal_x + patch_size] = 0.2

        noise = noise_std * torch.randn(self.images.shape, generator=generator, dtype=torch.float32)
        self.images = torch.clamp(self.images + noise, min=0.0, max=1.0)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.images[index], self.labels[index]


def build_synthetic_vision_dataloaders(config: SyntheticVisionDatasetConfig) -> VisionDataLoaders:
    """Create deterministic train/test dataloaders for speed benchmarking."""
    torch = require_torch()

    train_dataset = SyntheticPatternDataset(
        sample_count=config.train_samples,
        num_classes=config.num_classes,
        image_size=config.image_size,
        noise_std=config.noise_std,
        seed=config.seed,
    )
    test_dataset = SyntheticPatternDataset(
        sample_count=config.test_samples,
        num_classes=config.num_classes,
        image_size=config.image_size,
        noise_std=config.noise_std,
        seed=config.seed + 1,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed + 2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        generator=loader_generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    return VisionDataLoaders(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=config.num_classes,
    )

