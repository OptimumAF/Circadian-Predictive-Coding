from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transforms = pytest.importorskip("torchvision.transforms")

from src.infra.vision_datasets import (  # noqa: E402
    SyntheticVisionDatasetConfig,
    TorchVisionDatasetConfig,
    build_synthetic_vision_dataloaders,
    build_torchvision_vision_dataloaders,
)


def test_should_build_repeatable_disjoint_synthetic_splits() -> None:
    config = SyntheticVisionDatasetConfig(
        train_samples=12,
        validation_samples=7,
        test_samples=5,
        num_classes=3,
        image_size=32,
        batch_size=4,
        seed=19,
    )

    first = build_synthetic_vision_dataloaders(config)
    second = build_synthetic_vision_dataloaders(config)

    assert len(first.train_loader.dataset) == 12
    assert len(first.validation_loader.dataset) == 7
    assert len(first.test_loader.dataset) == 5
    assert first.sample_ids == second.sample_ids
    assert first.split_hashes == second.split_hashes
    assert len(set(first.split_hashes.values())) == 3
    _assert_disjoint_ids(first.sample_ids)
    for role in ("train", "validation", "test"):
        first_dataset = getattr(first, f"{role}_loader").dataset
        second_dataset = getattr(second, f"{role}_loader").dataset
        assert torch.equal(first_dataset.images, second_dataset.images)
        assert torch.equal(first_dataset.labels, second_dataset.labels)
    with pytest.raises(TypeError):
        first.split_hashes["train"] = "changed"  # type: ignore[index]


def test_should_keep_synthetic_guard_disjoint_from_selection_and_final_test() -> None:
    config = SyntheticVisionDatasetConfig(
        train_samples=12, validation_samples=7, guard_samples=6,
        test_samples=5, num_classes=3, image_size=32, batch_size=4, seed=19,
    )
    first = build_synthetic_vision_dataloaders(config)
    second = build_synthetic_vision_dataloaders(config)

    assert len(first.guard_loader.dataset) == 6
    assert first.guard_loader is not first.validation_loader
    assert set(first.split_hashes) == {"train", "validation", "guard", "test"}
    assert first.split_hashes == second.split_hashes
    _assert_disjoint_ids(first.sample_ids)


def test_should_split_cifar_source_with_deterministic_validation_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image
    from src.infra import vision_datasets

    class FakeCifar:
        def __init__(self, root: str, train: bool, download: bool, transform: Any) -> None:
            del root, download
            self.train = train
            self.transform = transform

        def __len__(self) -> int:
            return 20 if self.train else 8

        def __getitem__(self, index: int) -> tuple[Any, int]:
            image = Image.new("RGB", (32, 32), color=(index, 0, 0))
            return self.transform(image), index % 10

    monkeypatch.setattr(
        vision_datasets,
        "require_torchvision_datasets",
        lambda: SimpleNamespace(CIFAR10=FakeCifar, CIFAR100=FakeCifar),
    )
    config = TorchVisionDatasetConfig(
        dataset_name="cifar10",
        train_subset_size=7,
        validation_subset_size=5,
        guard_subset_size=4,
        test_subset_size=4,
        image_size=32,
        batch_size=4,
        seed=23,
    )

    first = build_torchvision_vision_dataloaders(config)
    second = build_torchvision_vision_dataloaders(config)

    assert len(first.train_loader.dataset) == 7
    assert len(first.validation_loader.dataset) == 5
    assert len(first.guard_loader.dataset) == 4
    assert first.guard_loader.dataset.dataset is first.validation_loader.dataset.dataset
    assert len(first.test_loader.dataset) == 4
    assert first.sample_ids == second.sample_ids
    assert first.split_hashes == second.split_hashes
    _assert_disjoint_ids(first.sample_ids)

    train_transform = first.train_loader.dataset.dataset.transform
    validation_transform = first.validation_loader.dataset.dataset.transform
    guard_transform = first.guard_loader.dataset.dataset.transform
    assert any(
        isinstance(item, transforms.RandomHorizontalFlip) for item in train_transform.transforms
    )
    assert not any(
        isinstance(item, transforms.RandomHorizontalFlip)
        for item in validation_transform.transforms
    )
    assert not any(
        isinstance(item, transforms.RandomHorizontalFlip)
        for item in guard_transform.transforms
    )
    first_view, _ = first.validation_loader.dataset[0]
    second_view, _ = first.validation_loader.dataset[0]
    assert torch.equal(first_view, second_view)


def test_should_reserve_validation_from_full_cifar_train_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.infra import vision_datasets

    class FakeCifar:
        def __init__(self, root: str, train: bool, download: bool, transform: Any) -> None:
            del root, download
            self.train = train
            self.transform = transform

        def __len__(self) -> int:
            return 20 if self.train else 8

    monkeypatch.setattr(
        vision_datasets,
        "require_torchvision_datasets",
        lambda: SimpleNamespace(CIFAR10=FakeCifar, CIFAR100=FakeCifar),
    )

    loaders = build_torchvision_vision_dataloaders(
        TorchVisionDatasetConfig(
            dataset_name="cifar10",
            train_subset_size=0,
            validation_subset_size=4,
            test_subset_size=0,
            image_size=32,
        )
    )

    assert len(loaders.train_loader.dataset) == 16
    assert len(loaders.validation_loader.dataset) == 4
    assert len(loaders.test_loader.dataset) == 8
    _assert_disjoint_ids(loaders.sample_ids)


def test_should_build_development_roles_without_opening_cifar_final_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.infra import vision_datasets

    final_constructions = 0

    class FakeCifar:
        def __init__(self, root: str, train: bool, download: bool, transform: Any) -> None:
            nonlocal final_constructions
            del root, download, transform
            if not train:
                final_constructions += 1
                raise AssertionError("final CIFAR source constructed during development probe")

        def __len__(self) -> int:
            return 20

    monkeypatch.setattr(
        vision_datasets,
        "require_torchvision_datasets",
        lambda: SimpleNamespace(CIFAR10=FakeCifar, CIFAR100=FakeCifar),
    )
    loaders = build_torchvision_vision_dataloaders(
        TorchVisionDatasetConfig(
            dataset_name="cifar10",
            train_subset_size=7,
            validation_subset_size=5,
            guard_subset_size=4,
            test_subset_size=4,
            image_size=32,
            batch_size=4,
            seed=23,
        ),
        include_final_test=False,
    )

    assert final_constructions == 0
    assert set(loaders.sample_ids) == set(loaders.split_hashes) == {
        "train",
        "guard",
        "validation",
    }
    _assert_disjoint_ids(loaders.sample_ids)
    with pytest.raises(RuntimeError, match="final test is unavailable"):
        iter(loaders.test_loader)


def _assert_disjoint_ids(sample_ids: Any) -> None:
    roles = tuple(sample_ids)
    for first_index, first_role in enumerate(roles):
        for second_role in roles[first_index + 1 :]:
            assert set(sample_ids[first_role]).isdisjoint(sample_ids[second_role])
