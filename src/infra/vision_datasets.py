"""Synthetic image datasets for fast and reproducible ResNet benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping

from src.shared.torch_runtime import (
    require_torch,
    require_torchvision_datasets,
    require_torchvision_transforms,
)


@dataclass(frozen=True)
class SyntheticVisionDatasetConfig:
    """Config for synthetic classification datasets."""

    train_samples: int = 2000
    validation_samples: int = 64
    guard_samples: int = 0
    test_samples: int = 500
    num_classes: int = 10
    image_size: int = 96
    batch_size: int = 32
    noise_std: float = 0.04
    difficulty: str = "easy"
    seed: int = 7
    num_workers: int = 0


@dataclass(frozen=True)
class TorchVisionDatasetConfig:
    """Config for torchvision-backed benchmark datasets."""

    dataset_name: str = "cifar100"
    data_root: str = "data"
    batch_size: int = 64
    image_size: int = 96
    seed: int = 7
    num_workers: int = 0
    download: bool = True
    train_subset_size: int = 0
    validation_subset_size: int = 1000
    guard_subset_size: int = 0
    test_subset_size: int = 0
    use_augmentation: bool = True


@dataclass(frozen=True)
class VisionDataLoaders:
    """Role-separated loaders and immutable identities for their samples."""

    train_loader: Any
    validation_loader: Any
    guard_loader: Any
    test_loader: Any
    num_classes: int
    sample_ids: Mapping[str, tuple[str, ...]]
    split_hashes: Mapping[str, str]


class _UnavailableFinalTestLoader:
    """Fail if a development-only loader is accidentally used for scoring."""

    def __iter__(self) -> Any:
        raise RuntimeError("final test is unavailable in this development-only loader")

    def __len__(self) -> int:
        raise RuntimeError("final test is unavailable in this development-only loader")


class SyntheticPatternDataset:
    """Synthetic task with adjustable difficulty and class overlap."""

    def __init__(
        self,
        sample_count: int,
        num_classes: int,
        image_size: int,
        noise_std: float,
        difficulty: str,
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
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be one of: easy, medium, hard.")

        torch = require_torch()
        self._torch = torch
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._difficulty = difficulty

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
        overlap_group_size = self._resolve_overlap_group_size(num_classes)

        for sample_index in range(sample_count):
            class_index = int(self.labels[sample_index].item())
            canonical_class = class_index % overlap_group_size
            row = canonical_class // grid_side
            col = canonical_class % grid_side
            start_y = min(1 + row * stride, image_size - patch_size - 1)
            start_x = min(1 + col * stride, image_size - patch_size - 1)

            jitter_limit = self._resolve_jitter_limit(patch_size)
            if jitter_limit > 0:
                offset_y = int(
                    torch.randint(-jitter_limit, jitter_limit + 1, (1,), generator=generator).item()
                )
                offset_x = int(
                    torch.randint(-jitter_limit, jitter_limit + 1, (1,), generator=generator).item()
                )
                start_y = max(1, min(start_y + offset_y, image_size - patch_size - 1))
                start_x = max(1, min(start_x + offset_x, image_size - patch_size - 1))

            channel_index = self._resolve_channel_index(class_index, overlap_group_size)
            secondary_channel = (channel_index + 1) % 3
            intensity = self._resolve_primary_intensity(class_index, num_classes)
            secondary_intensity = self._resolve_secondary_intensity()

            self.images[
                sample_index,
                channel_index,
                start_y : start_y + patch_size,
                start_x : start_x + patch_size,
            ] = intensity
            self.images[
                sample_index,
                secondary_channel,
                start_y : start_y + (patch_size // 2),
                start_x : start_x + (patch_size // 2),
            ] = secondary_intensity

            diagonal_start = image_size - start_y - patch_size
            diagonal_end = image_size - start_x - patch_size
            diagonal_y = max(min(diagonal_start, image_size - patch_size - 1), 0)
            diagonal_x = max(min(diagonal_end, image_size - patch_size - 1), 0)
            diagonal_height = max(patch_size // 3, 2)
            self.images[
                sample_index,
                channel_index,
                diagonal_y : diagonal_y + diagonal_height,
                diagonal_x : diagonal_x + patch_size,
            ] = self._resolve_diagonal_intensity()

            self._add_distractor_patch(
                sample_index=sample_index,
                patch_size=patch_size,
                image_size=image_size,
                num_classes=num_classes,
                generator=generator,
            )
            self._add_occlusion_blob(
                sample_index=sample_index,
                patch_size=patch_size,
                image_size=image_size,
                generator=generator,
            )

        effective_noise = noise_std * self._resolve_noise_multiplier()
        noise = effective_noise * torch.randn(
            self.images.shape, generator=generator, dtype=torch.float32
        )
        self.images = torch.clamp(self.images + noise, min=0.0, max=1.0)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.images[index], self.labels[index]

    def _resolve_overlap_group_size(self, num_classes: int) -> int:
        if self._difficulty == "easy":
            return num_classes
        if self._difficulty == "medium":
            return max(num_classes // 2, 2)
        return max(num_classes // 3, 2)

    def _resolve_jitter_limit(self, patch_size: int) -> int:
        if self._difficulty == "easy":
            return 0
        if self._difficulty == "medium":
            return max(patch_size // 4, 1)
        return max(patch_size // 2, 1)

    def _resolve_channel_index(self, class_index: int, overlap_group_size: int) -> int:
        if self._difficulty == "easy":
            return class_index % 3
        if self._difficulty == "medium":
            return (class_index + overlap_group_size) % 3
        return (class_index // max(overlap_group_size, 1)) % 3

    def _resolve_primary_intensity(self, class_index: int, num_classes: int) -> float:
        normalized = class_index / max(num_classes - 1, 1)
        if self._difficulty == "easy":
            return float(0.60 + 0.35 * normalized)
        if self._difficulty == "medium":
            return float(0.50 + 0.28 * normalized)
        return float(0.42 + 0.22 * normalized)

    def _resolve_secondary_intensity(self) -> float:
        if self._difficulty == "easy":
            return 0.35
        if self._difficulty == "medium":
            return 0.30
        return 0.24

    def _resolve_diagonal_intensity(self) -> float:
        if self._difficulty == "easy":
            return 0.20
        if self._difficulty == "medium":
            return 0.18
        return 0.15

    def _resolve_noise_multiplier(self) -> float:
        if self._difficulty == "easy":
            return 1.0
        if self._difficulty == "medium":
            return 1.6
        return 2.3

    def _add_distractor_patch(
        self,
        sample_index: int,
        patch_size: int,
        image_size: int,
        num_classes: int,
        generator: Any,
    ) -> None:
        torch = self._torch
        if self._difficulty == "easy":
            return

        distractor_probability = 0.55 if self._difficulty == "medium" else 0.90
        probability_sample = float(torch.rand((1,), generator=generator).item())
        if probability_sample > distractor_probability:
            return

        distractor_class = int(torch.randint(0, num_classes, (1,), generator=generator).item())
        distractor_channel = distractor_class % 3
        y = int(torch.randint(0, image_size - patch_size, (1,), generator=generator).item())
        x = int(torch.randint(0, image_size - patch_size, (1,), generator=generator).item())
        intensity = 0.14 if self._difficulty == "medium" else 0.20
        self.images[
            sample_index,
            distractor_channel,
            y : y + max(patch_size // 2, 2),
            x : x + max(patch_size // 2, 2),
        ] = intensity

    def _add_occlusion_blob(
        self,
        sample_index: int,
        patch_size: int,
        image_size: int,
        generator: Any,
    ) -> None:
        torch = self._torch
        if self._difficulty != "hard":
            return

        channel = int(torch.randint(0, 3, (1,), generator=generator).item())
        blob_h = max(patch_size // 2, 2)
        blob_w = max(patch_size // 2, 2)
        y = int(torch.randint(0, image_size - blob_h, (1,), generator=generator).item())
        x = int(torch.randint(0, image_size - blob_w, (1,), generator=generator).item())
        blob_value = float(torch.rand((1,), generator=generator).item() * 0.4)
        self.images[sample_index, channel, y : y + blob_h, x : x + blob_w] = blob_value


def build_synthetic_vision_dataloaders(config: SyntheticVisionDatasetConfig) -> VisionDataLoaders:
    """Create independent deterministic train/validation/test datasets."""
    if config.validation_samples <= 0:
        raise ValueError("validation_samples must be positive.")
    if config.guard_samples < 0:
        raise ValueError("guard_samples must be non-negative.")
    torch = require_torch()

    train_dataset = SyntheticPatternDataset(
        sample_count=config.train_samples,
        num_classes=config.num_classes,
        image_size=config.image_size,
        noise_std=config.noise_std,
        difficulty=config.difficulty,
        seed=config.seed,
    )
    validation_dataset = SyntheticPatternDataset(
        sample_count=config.validation_samples,
        num_classes=config.num_classes,
        image_size=config.image_size,
        noise_std=config.noise_std,
        difficulty=config.difficulty,
        seed=config.seed + 3,
    )
    guard_dataset = (
        SyntheticPatternDataset(
            sample_count=config.guard_samples,
            num_classes=config.num_classes,
            image_size=config.image_size,
            noise_std=config.noise_std,
            difficulty=config.difficulty,
            seed=config.seed + 4,
        )
        if config.guard_samples > 0 else None
    )
    test_dataset = SyntheticPatternDataset(
        sample_count=config.test_samples,
        num_classes=config.num_classes,
        image_size=config.image_size,
        noise_std=config.noise_std,
        difficulty=config.difficulty,
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
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    guard_loader = (
        torch.utils.data.DataLoader(
            guard_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=False,
        )
        if guard_dataset is not None else validation_loader
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    dataset_tag = (
        f"synthetic/classes={config.num_classes}/size={config.image_size}"
        f"/noise={config.noise_std!r}/difficulty={config.difficulty}"
    )
    sample_ids = {
        "train": tuple(
            f"{dataset_tag}/train/seed={config.seed}/{index}" for index in range(len(train_dataset))
        ),
        "validation": tuple(
            f"{dataset_tag}/validation/seed={config.seed + 3}/{index}"
            for index in range(len(validation_dataset))
        ),
        "test": tuple(
            f"{dataset_tag}/test/seed={config.seed + 1}/{index}"
            for index in range(len(test_dataset))
        ),
    }
    if guard_dataset is not None:
        sample_ids["guard"] = tuple(
            f"{dataset_tag}/guard/seed={config.seed + 4}/{index}"
            for index in range(len(guard_dataset))
        )
    return VisionDataLoaders(
        train_loader=train_loader,
        validation_loader=validation_loader,
        guard_loader=guard_loader,
        test_loader=test_loader,
        num_classes=config.num_classes,
        sample_ids=MappingProxyType(sample_ids),
        split_hashes=_hash_splits(sample_ids),
    )


def build_torchvision_vision_dataloaders(
    config: TorchVisionDatasetConfig, *, include_final_test: bool = True
) -> VisionDataLoaders:
    """Create role-separated dataloaders backed by torchvision datasets."""
    if type(include_final_test) is not bool:
        raise ValueError("include_final_test must be a boolean")
    if config.dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("dataset_name must be one of: cifar10, cifar100.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.image_size < 32:
        raise ValueError("image_size must be at least 32 for ResNet-50.")
    if config.train_subset_size < 0 or config.test_subset_size < 0:
        raise ValueError("train/test subset sizes must be non-negative.")
    if config.validation_subset_size <= 0:
        raise ValueError("validation_subset_size must be positive.")
    if config.guard_subset_size < 0:
        raise ValueError("guard_subset_size must be non-negative.")

    torch = require_torch()
    datasets = require_torchvision_datasets()
    transforms = require_torchvision_transforms()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms: list[Any] = [transforms.Resize((config.image_size, config.image_size))]
    if config.use_augmentation:
        train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    train_transform = transforms.Compose(train_transforms)

    dataset_class = datasets.CIFAR100 if config.dataset_name == "cifar100" else datasets.CIFAR10
    train_dataset = dataset_class(
        root=config.data_root,
        train=True,
        download=config.download,
        transform=train_transform,
    )
    validation_dataset = dataset_class(
        root=config.data_root,
        train=True,
        download=False,
        transform=test_transform,
    )
    # Both held-out roles use the same deterministic source view, avoiding
    # another in-memory CIFAR copy while keeping their indices disjoint.
    guard_dataset = validation_dataset if config.guard_subset_size > 0 else None
    test_dataset = (
        dataset_class(
            root=config.data_root,
            train=False,
            download=config.download,
            transform=test_transform,
        )
        if include_final_test
        else None
    )

    total_train_size = len(train_dataset)
    if len(validation_dataset) != total_train_size:
        raise ValueError("Training and validation views must have the same source size.")
    required_roles = 3 if guard_dataset is not None else 2
    if total_train_size < required_roles:
        raise ValueError(f"At least {required_roles} source training examples are required.")

    # Why this: a requested training subset keeps its original size when the
    # source has room, while validation always uses different source examples.
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    ordered_indices = torch.randperm(total_train_size, generator=generator).tolist()
    if config.train_subset_size > 0:
        train_count = min(config.train_subset_size, total_train_size - required_roles + 1)
        remaining = total_train_size - train_count
        validation_count = min(
            config.validation_subset_size,
            remaining - (1 if guard_dataset is not None else 0),
        )
    else:
        validation_count = min(config.validation_subset_size, total_train_size - required_roles + 1)
        if guard_dataset is not None:
            guard_count = min(config.guard_subset_size, total_train_size - validation_count - 1)
            train_count = total_train_size - validation_count - guard_count
        else:
            train_count = total_train_size - validation_count
    if guard_dataset is not None and config.train_subset_size > 0:
        guard_count = min(config.guard_subset_size, total_train_size - train_count - validation_count)
    train_indices = ordered_indices[:train_count]
    validation_indices = ordered_indices[train_count : train_count + validation_count]
    guard_indices = ordered_indices[
        train_count + validation_count : train_count + validation_count + guard_count
    ] if guard_dataset is not None else []
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(validation_dataset, validation_indices)
    if guard_dataset is not None:
        guard_dataset = torch.utils.data.Subset(guard_dataset, guard_indices)

    test_indices: Any = ()
    if test_dataset is not None:
        if config.test_subset_size > 0:
            test_dataset = _select_subset(
                dataset=test_dataset,
                subset_size=config.test_subset_size,
                seed=config.seed + 1,
                torch_module=torch,
            )
            test_indices = test_dataset.indices
        else:
            test_indices = range(len(test_dataset))

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
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    guard_loader = (
        torch.utils.data.DataLoader(
            guard_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=False,
        )
        if guard_dataset is not None else validation_loader
    )
    test_loader = (
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False,
        )
        if test_dataset is not None
        else _UnavailableFinalTestLoader()
    )

    num_classes = 100 if config.dataset_name == "cifar100" else 10
    sample_ids = {
        "train": tuple(f"{config.dataset_name}/train/{index}" for index in train_indices),
        "validation": tuple(f"{config.dataset_name}/train/{index}" for index in validation_indices),
    }
    if test_dataset is not None:
        sample_ids["test"] = tuple(
            f"{config.dataset_name}/test/{index}" for index in test_indices
        )
    if guard_dataset is not None:
        sample_ids["guard"] = tuple(
            f"{config.dataset_name}/train/{index}" for index in guard_indices
        )
    return VisionDataLoaders(
        train_loader=train_loader,
        validation_loader=validation_loader,
        guard_loader=guard_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        sample_ids=MappingProxyType(sample_ids),
        split_hashes=_hash_splits(sample_ids),
    )


def _hash_splits(sample_ids: Mapping[str, tuple[str, ...]]) -> Mapping[str, str]:
    """Hash ordered, namespaced identities without reading labels or images."""
    return MappingProxyType(
        {
            role: sha256("\n".join(ids).encode("utf-8")).hexdigest()
            for role, ids in sample_ids.items()
        }
    )


def _select_subset(dataset: Any, subset_size: int, seed: int, torch_module: Any) -> Any:
    if subset_size <= 0:
        return dataset
    total_size = len(dataset)
    keep_size = min(subset_size, total_size)
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    indices = torch_module.randperm(total_size, generator=generator)[:keep_size].tolist()
    return torch_module.utils.data.Subset(dataset, indices)
