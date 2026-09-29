"""Check seeded CIFAR-10 loader order, views, and optional training seal.

Inputs are a complete local torchvision CIFAR-10 cache. Output is one JSON
summary of role identities and exact CPU train-batch/view hashes. By default
it does not train; the optional tiny CPU training pass tests final-test timing.
Neither mode makes a model-ranking claim.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from typing import Any
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app import resnet50_benchmark as vision_benchmark  # noqa: E402
from src.app.resnet50_benchmark import (  # noqa: E402
    ResNet50BenchmarkConfig,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
)
from src.app.seeded_vision_loader import SeededEpochTrainLoader  # noqa: E402
from src.infra.vision_datasets import (  # noqa: E402
    TorchVisionDatasetConfig,
    build_torchvision_vision_dataloaders,
)
from src.shared.torch_runtime import require_torch  # noqa: E402


MODEL_ORDER = ("backprop", "predictive", "circadian")


class _IndexedTrainDataset:
    """Return the source sample ID alongside each real transformed image."""

    def __init__(self, source: Any, sample_ids: tuple[str, ...]) -> None:
        self.source = source
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        image, label = self.source[index]
        return image, label, self.sample_ids[index]


def _digest_batch(images: Any, labels: Any, sample_ids: tuple[str, ...]) -> dict[str, Any]:
    digest = sha256()
    digest.update(images.detach().cpu().contiguous().numpy().tobytes())
    digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
    return {"ids": sample_ids, "view_hash": digest.hexdigest()}


def _collect_order(torch: Any, loader: Any, seed: int, order: tuple[str, ...]) -> dict[str, Any]:
    batches_by_model: dict[str, Any] = {}
    for model in order:
        seeded = SeededEpochTrainLoader(torch, loader, seed + 3_001)
        batches_by_model[model] = tuple(
            _digest_batch(images, labels, tuple(sample_ids))
            for images, labels, sample_ids in seeded
        )
        # Unrelated model work must not change the next model's train view.
        torch.rand(3)
        np.random.random()
        random.random()
    return batches_by_model


def verify_cifar_loader(data_root: Path, seed: int = 73) -> dict[str, Any]:
    """Verify exact seeded train views on one tiny real-CIFAR role split."""
    cache = data_root / "cifar-10-batches-py"
    if not (cache / "data_batch_1").is_file() or not (cache / "test_batch").is_file():
        raise FileNotFoundError(
            f"Complete CIFAR-10 cache required at {cache}; no download attempted"
        )
    torch = require_torch()
    workers: dict[str, Any] = {}
    for worker_count in (0, 2):
        loaders = build_torchvision_vision_dataloaders(
            TorchVisionDatasetConfig(
                dataset_name="cifar10",
                data_root=str(data_root),
                batch_size=4,
                image_size=32,
                seed=seed,
                num_workers=worker_count,
                download=False,
                train_subset_size=8,
                validation_subset_size=4,
                guard_subset_size=4,
                test_subset_size=4,
                use_augmentation=True,
            )
        )
        role_ids = {role: set(ids) for role, ids in loaders.sample_ids.items()}
        if any(
            role_ids[left] & role_ids[right]
            for left in role_ids
            for right in role_ids
            if left < right
        ):
            raise AssertionError("CIFAR role sample IDs overlap")
        source_loader = loaders.train_loader
        train_loader = torch.utils.data.DataLoader(
            _IndexedTrainDataset(source_loader.dataset, loaders.sample_ids["train"]),
            batch_size=source_loader.batch_size,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=False,
            generator=torch.Generator().manual_seed(seed + 2),
        )
        forward = _collect_order(torch, train_loader, seed, MODEL_ORDER)
        reverse = _collect_order(torch, train_loader, seed, tuple(reversed(MODEL_ORDER)))
        expected = forward[MODEL_ORDER[0]]
        if any(batches != expected for batches in (*forward.values(), *reverse.values())):
            raise AssertionError(
                f"CIFAR seeded views changed with model order ({worker_count} workers)"
            )
        observed_ids = [sample_id for batch in expected for sample_id in batch["ids"]]
        if len(observed_ids) != 8 or set(observed_ids) != role_ids["train"]:
            raise AssertionError("CIFAR shuffled train IDs do not cover the fixed subset")
        workers[str(worker_count)] = {
            "train_batches": expected,
            "split_hashes": dict(loaders.split_hashes),
            "role_counts": {role: len(ids) for role, ids in role_ids.items()},
        }
    if workers["0"]["split_hashes"] != workers["2"]["split_hashes"]:
        raise AssertionError("CIFAR role identity changed with worker count")
    return {
        "protocol_id": VISION_SEEDED_UNMATCHED_PROTOCOL,
        "dataset": "cifar10",
        "seed": seed,
        "data_root": str(data_root),
        "download": False,
        "model_orders": [MODEL_ORDER, tuple(reversed(MODEL_ORDER))],
        "workers": workers,
        "final_test_iterated": False,
    }


def verify_cifar_training_seal(data_root: Path, seed: int = 73) -> dict[str, Any]:
    """Run one tiny seeded CPU pass with final test sealed during training."""
    config = ResNet50BenchmarkConfig(
        protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
        dataset_name="cifar10",
        dataset_data_root=str(data_root),
        dataset_download=False,
        dataset_train_subset_size=8,
        dataset_guard_subset_size=4,
        dataset_validation_subset_size=4,
        dataset_test_subset_size=4,
        dataset_num_workers=0,
        dataset_use_augmentation=True,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=seed,
        device="cpu",
        target_accuracy=None,
        evaluation_batches=1,
        inference_batches=1,
        warmup_batches=0,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=32,
        circadian_inference_steps=2,
        circadian_sleep_interval=0,
    )
    trained: list[str] = []
    final_test_iterations = 0
    original_build = vision_benchmark._build_benchmark_loaders
    original_train = vision_benchmark._train_seeded_variant

    class SealedTestLoader:
        def __init__(self, source: Any) -> None:
            self.source = source

        def __iter__(self) -> Any:
            nonlocal final_test_iterations
            if len(trained) != len(MODEL_ORDER):
                raise AssertionError("CIFAR final test opened before all models trained")
            final_test_iterations += 1
            return iter(self.source)

    def build_sealed_loaders(candidate: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(candidate)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    def record_completed_train(variant: str, *args: Any, **kwargs: Any) -> Any:
        outcome = original_train(variant, *args, **kwargs)
        trained.append(variant)
        return outcome

    with (
        patch.object(vision_benchmark, "_build_benchmark_loaders", build_sealed_loaders),
        patch.object(vision_benchmark, "_train_seeded_variant", record_completed_train),
    ):
        result = vision_benchmark.run_resnet50_benchmark(config)
    if tuple(trained) != MODEL_ORDER or final_test_iterations == 0:
        raise AssertionError("CIFAR final-test seal check did not finish all stages")
    return {
        "trained_order": trained,
        "final_test_iterations_after_training": final_test_iterations,
        "trained_model_hashes": result.trained_model_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--check-training-seal",
        action="store_true",
        help="also run one tiny real-CIFAR CPU training pass with final test sealed",
    )
    args = parser.parse_args()
    result = verify_cifar_loader(args.data_root, args.seed)
    if args.check_training_seal:
        result["training_seal"] = verify_cifar_training_seal(args.data_root, args.seed)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
