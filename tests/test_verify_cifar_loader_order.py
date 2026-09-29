"""Exercise the real-cache verifier with a bounded stochastic source stub."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from scripts import verify_cifar_loader_order as verifier  # noqa: E402
from src.infra.vision_datasets import VisionDataLoaders  # noqa: E402


class SealedLoader:
    def __iter__(self) -> Any:
        raise AssertionError("verifier opened a held-out role")


def test_verifier_replays_stochastic_views_without_opening_held_out_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cifar-10-batches-py"
    cache.mkdir()
    (cache / "data_batch_1").touch()
    (cache / "test_batch").touch()

    def build(config: Any) -> VisionDataLoaders:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()]
        )
        source = torchvision.datasets.FakeData(
            size=16, image_size=(3, 32, 32), num_classes=10, transform=transform
        )
        train = torch.utils.data.Subset(source, list(range(8)))
        sample_ids = {
            "train": tuple(f"cifar10/train/{index}" for index in range(8)),
            "guard": tuple(f"cifar10/train/{index}" for index in range(8, 12)),
            "validation": tuple(f"cifar10/train/{index}" for index in range(12, 16)),
            "test": tuple(f"cifar10/test/{index}" for index in range(4)),
        }
        return VisionDataLoaders(
            train_loader=torch.utils.data.DataLoader(
                train,
                batch_size=4,
                shuffle=True,
                num_workers=config.num_workers,
                generator=torch.Generator().manual_seed(config.seed + 2),
            ),
            guard_loader=SealedLoader(),
            validation_loader=SealedLoader(),
            test_loader=SealedLoader(),
            num_classes=10,
            sample_ids=sample_ids,
            split_hashes={role: role for role in sample_ids},
        )

    monkeypatch.setattr(verifier, "build_torchvision_vision_dataloaders", build)
    result = verifier.verify_cifar_loader(tmp_path, seed=73)

    assert result["dataset"] == "cifar10"
    assert result["download"] is False
    assert result["final_test_iterated"] is False
    for worker_count in ("0", "2"):
        report = result["workers"][worker_count]
        assert report["role_counts"] == {"train": 8, "guard": 4, "validation": 4, "test": 4}
        assert len(report["train_batches"]) == 2
        assert len({batch["view_hash"] for batch in report["train_batches"]}) == 2


def test_training_seal_rejects_early_test_iteration_and_restores_runner_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = verifier.vision_benchmark
    built: list[Any] = []

    def build(config: Any) -> Any:
        del config
        loaders = VisionDataLoaders(
            train_loader=None,
            validation_loader=None,
            guard_loader=None,
            test_loader=["held-out"],
            num_classes=10,
            sample_ids={},
            split_hashes={},
        )
        built.append(loaders)
        return loaders

    def train(variant: str, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(model_name=variant)

    def run(config: Any) -> Any:
        loaders = benchmark._build_benchmark_loaders(config)
        with pytest.raises(AssertionError, match="before all models trained"):
            list(loaders.test_loader)
        train_variant: Any = benchmark._train_seeded_variant
        for variant in verifier.MODEL_ORDER:
            train_variant(variant, None, None, None, None)
        assert list(loaders.test_loader) == ["held-out"]
        return SimpleNamespace(trained_model_hashes={"backprop": "trained"})

    monkeypatch.setattr(benchmark, "_build_benchmark_loaders", build)
    monkeypatch.setattr(benchmark, "_train_seeded_variant", train)
    monkeypatch.setattr(benchmark, "run_resnet50_benchmark", run)
    report = verifier.verify_cifar_training_seal(tmp_path)

    assert report == {
        "trained_order": list(verifier.MODEL_ORDER),
        "final_test_iterations_after_training": 1,
        "trained_model_hashes": {"backprop": "trained"},
    }
    assert len(built) == 1
    assert benchmark._build_benchmark_loaders is build
    assert benchmark._train_seeded_variant is train
