"""Measure one pretrained CIFAR development feature setup without training."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torchvision.models import ResNet50_Weights  # noqa: E402

from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.matched_head_tuning import _build_seed_bank  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402

REQUEST_PATH = REPO_ROOT / "data" / "cifar-feature-profile-seed101-request.json"
RESULT_PATH = REPO_ROOT / "data" / "cifar-feature-profile-seed101-result.json"
FAILURE_PATH = REPO_ROOT / "data" / "cifar-feature-profile-seed101-failure.json"
WALL_BUDGET_SECONDS = 60


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _weight_source() -> tuple[Path, str]:
    url = ResNet50_Weights.IMAGENET1K_V2.url
    cached = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    if not cached.is_file():
        raise FileNotFoundError(f"Pretrained ResNet-50 V2 cache missing: {cached}")
    return cached, url


def _base_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        dataset_name="cifar10",
        dataset_data_root=str(REPO_ROOT / "data"),
        dataset_download=False,
        dataset_train_subset_size=128,
        dataset_guard_subset_size=64,
        dataset_validation_subset_size=64,
        dataset_test_subset_size=16,
        dataset_num_workers=0,
        dataset_use_augmentation=True,
        image_size=32,
        batch_size=16,
        epochs=1,
        seed=101,
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="imagenet",
        predictive_head_hidden_dim=16,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=16,
    )


def main() -> None:
    if any(path.exists() for path in (REQUEST_PATH, RESULT_PATH, FAILURE_PATH)):
        raise FileExistsError("CIFAR feature profile already has an artifact")
    weight_file, weight_url = _weight_source()
    digest = sha256()
    with weight_file.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    config = _base_config()
    _save_json(
        REQUEST_PATH,
        {
            "config": asdict(config),
            "weight_url": weight_url,
            "weight_file": str(weight_file),
            "weight_bytes": weight_file.stat().st_size,
            "weight_sha256": digest.hexdigest(),
            "wall_budget_seconds": WALL_BUDGET_SECONDS,
            "final_test_iterations_allowed": 0,
        },
    )

    original_build = matched._build_benchmark_loaders

    class SealedTestLoader:
        def __iter__(self) -> Any:
            raise AssertionError("Pretrained feature profile opened final test")

    def build_sealed_loaders(candidate: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(candidate)
        return replace(loaders, test_loader=SealedTestLoader())

    started = monotonic()
    try:
        with patch.object(matched, "_build_benchmark_loaders", build_sealed_loaders):
            bank = _build_seed_bank(torch, torch.device("cpu"), config)
        elapsed = monotonic() - started
        if elapsed > WALL_BUDGET_SECONDS:
            raise TimeoutError(f"Feature setup exceeded {WALL_BUDGET_SECONDS} s")
        batches = {role: getattr(bank, role) for role in ("train", "guard", "validation")}
        _save_json(
            RESULT_PATH,
            {
                "weight_url": weight_url,
                "weight_bytes": weight_file.stat().st_size,
                "weight_sha256": digest.hexdigest(),
                "seed": config.seed,
                "elapsed_seconds": round(elapsed, 3),
                "backbone_hash": bank.backbone_hash,
                "split_hashes": bank.split_hashes,
                "feature_hashes": bank.feature_hashes,
                "feature_bytes": {
                    role: sum(
                        tensor.numel() * tensor.element_size() for pair in values for tensor in pair
                    )
                    for role, values in batches.items()
                },
                "role_counts": {role: len(bank.loaders.sample_ids[role]) for role in batches},
                "final_test_iterations": 0,
            },
        )
    except Exception as error:
        _save_json(
            FAILURE_PATH,
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(monotonic() - started, 3),
            },
        )
        raise
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 3),
                "role_counts": {role: len(bank.loaders.sample_ids[role]) for role in batches},
                "result_path": str(RESULT_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
