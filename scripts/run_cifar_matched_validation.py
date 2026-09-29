"""Freeze one bounded real-CIFAR matched-head selection without test access.

Inputs are the verified local CIFAR-10 cache and fixed settings below. Outputs
are request, validation-selection, and confirmation-manifest JSON artifacts.
This script does not score final test or compare final model performance.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import md5
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.matched_head_tuning import (  # noqa: E402
    HeadTuningCandidate,
    run_matched_head_tuning,
)
from src.app.repeated_head_confirmation import create_confirmation_manifest  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402

ARCHIVE = REPO_ROOT / "data" / "cifar-10-python.tar.gz"
ARCHIVE_BYTES = 170_498_071
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
OUTPUT_DIR = REPO_ROOT / "artifacts"
SELECTION_SEEDS = (73,)
CONFIRMATION_SEEDS = (83, 89, 97)
WALL_TIME_BUDGET_SECONDS = 0.05
WALL_TIME_EPOCH_CAP = 1000
WALL_BUDGET_SECONDS = 180


def _verify_archive() -> None:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != ARCHIVE_BYTES:
        raise FileNotFoundError("Verified CIFAR-10 archive is required under data/")
    digest = md5()
    with ARCHIVE.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("CIFAR-10 archive MD5 differs from torchvision")


def _base_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        dataset_name="cifar10",
        dataset_data_root=str(ARCHIVE.parent),
        dataset_download=False,
        dataset_train_subset_size=32,
        dataset_guard_subset_size=16,
        dataset_validation_subset_size=16,
        dataset_test_subset_size=16,
        dataset_num_workers=0,
        dataset_use_augmentation=True,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=SELECTION_SEEDS[0],
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
    )


def _candidates(
    base: ResNet50BenchmarkConfig,
) -> dict[str, tuple[HeadTuningCandidate, ...]]:
    # Why this: reuse the prior smoke's equal two-trial grid, fixed before
    # seeing real-CIFAR validation or final-test outcomes.
    fields = (
        ("backprop_mlp", "backprop_learning_rate"),
        ("predictive_coding", "predictive_learning_rate"),
        ("circadian_predictive_coding", "circadian_learning_rate"),
    )
    return {
        head: (
            HeadTuningCandidate("a", base),
            HeadTuningCandidate("b", replace(base, **{field: getattr(base, field) * 0.8})),
        )
        for head, field in fields
    }


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    paths = {
        name: OUTPUT_DIR / f"benchmark_cifar_v2_{name}_smoke.json"
        for name in ("request", "selection", "manifest")
    }
    if (
        len(CONFIRMATION_SEEDS) not in (3, 4)
        or len(set(CONFIRMATION_SEEDS)) != len(CONFIRMATION_SEEDS)
        or set(CONFIRMATION_SEEDS) & set(SELECTION_SEEDS)
    ):
        raise ValueError("Confirmation requires 3–4 distinct seeds disjoint from selection")
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("CIFAR matched selection artifacts already exist")
    _verify_archive()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _base_config()
    candidates = _candidates(base)
    _save_json(
        paths["request"],
        {
            "archive": str(ARCHIVE),
            "archive_bytes": ARCHIVE_BYTES,
            "archive_md5": ARCHIVE_MD5,
            "source": "https://zenodo.org/records/10089977",
            "selection_seeds": SELECTION_SEEDS,
            "confirmation_seeds": CONFIRMATION_SEEDS,
            "candidates_per_head": 2,
            "wall_time_budget_seconds": WALL_TIME_BUDGET_SECONDS,
            "wall_time_epoch_cap": WALL_TIME_EPOCH_CAP,
            "selection_wall_budget_seconds": WALL_BUDGET_SECONDS,
            "base_config": asdict(base),
            "candidates": {
                name: [asdict(candidate) for candidate in options]
                for name, options in candidates.items()
            },
        },
    )

    original_build = matched._build_benchmark_loaders

    class SealedTestLoader:
        def __iter__(self) -> Any:
            raise AssertionError("Final CIFAR test opened during validation selection")

    def build_sealed_loaders(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader())

    started = monotonic()
    with patch.object(matched, "_build_benchmark_loaders", build_sealed_loaders):
        selection = run_matched_head_tuning(
            base,
            candidates,
            seeds=SELECTION_SEEDS,
            candidates_per_head=2,
            confirm_test=False,
        )
    elapsed = monotonic() - started
    if elapsed > WALL_BUDGET_SECONDS:
        raise TimeoutError(f"CIFAR selection exceeded {WALL_BUDGET_SECONDS} s: {elapsed:.1f} s")
    if len(selection.attempts) != 6 or len(selection.selections) != 3:
        raise AssertionError("CIFAR validation selection omitted an equal trial")
    _save_json(paths["selection"], asdict(selection))
    manifest = create_confirmation_manifest(
        selection,
        confirmation_seeds=CONFIRMATION_SEEDS,
        wall_time_budget_seconds=WALL_TIME_BUDGET_SECONDS,
        wall_time_epoch_cap=WALL_TIME_EPOCH_CAP,
    )
    _save_json(paths["manifest"], asdict(manifest))
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "selection_seeds": SELECTION_SEEDS,
                "confirmation_seeds": CONFIRMATION_SEEDS,
                "selections": {item.head_name: item.candidate_id for item in selection.selections},
                "attempts": len(selection.attempts),
                "selection_seconds": round(elapsed, 2),
                "final_test_iterations": 0,
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
