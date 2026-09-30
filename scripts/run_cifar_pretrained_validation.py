"""Freeze the predeclared pretrained-CIFAR matched-head validation selection.

This entry point writes a request before training, seals final test, and saves
the complete selection and confirmation manifest under distinct artifact names.
It never scores the final test. See P1.8k in DEVELOPMENT_PLAN.md.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import md5, sha256
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
from src.app.matched_head_tuning import (  # noqa: E402
    HeadTuningCandidate,
    MatchedHeadTuningError,
    MatchedHeadTuningResult,
    run_matched_head_tuning,
)
from src.app.repeated_head_confirmation import (  # noqa: E402
    create_confirmation_manifest,
    restore_confirmation_manifest,
)
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402

ARCHIVE = REPO_ROOT / "data" / "cifar-10-python.tar.gz"
ARCHIVE_BYTES = 170_498_071
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
WEIGHT_BYTES = 102_540_417
WEIGHT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
WEIGHT_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREFIX = "benchmark_cifar_pretrained_v1"
SELECTION_SEEDS = (113,)
CONFIRMATION_SEEDS = (127, 131, 137)
WALL_TIME_BUDGET_SECONDS = 0.5
WALL_TIME_EPOCH_CAP = 1000
SELECTION_LIMIT_SECONDS = 120


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _verify_local_inputs() -> dict[str, Any]:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != ARCHIVE_BYTES:
        raise FileNotFoundError("Predeclared CIFAR-10 archive is missing or changed size")
    archive_digest = md5()
    with ARCHIVE.open("rb") as stream:
        while block := stream.read(1 << 20):
            archive_digest.update(block)
    if archive_digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("Predeclared CIFAR-10 archive MD5 changed")

    if ResNet50_Weights.IMAGENET1K_V2.url != WEIGHT_URL:
        raise ValueError("Predeclared ImageNet V2 weight URL changed")
    weight_file = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_URL.rsplit("/", 1)[-1]
    if not weight_file.is_file() or weight_file.stat().st_size != WEIGHT_BYTES:
        raise FileNotFoundError("Predeclared ImageNet V2 weight cache is missing or changed size")
    weight_digest = sha256()
    with weight_file.open("rb") as stream:
        while block := stream.read(1 << 20):
            weight_digest.update(block)
    if weight_digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Predeclared ImageNet V2 weight SHA-256 changed")
    return {
        "archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE_BYTES,
        "archive_md5": ARCHIVE_MD5,
        "archive_source": "https://zenodo.org/records/10089977",
        "weight_file": str(weight_file),
        "weight_bytes": WEIGHT_BYTES,
        "weight_sha256": WEIGHT_SHA256,
        "weight_url": WEIGHT_URL,
    }


def _base_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        dataset_name="cifar10",
        dataset_data_root=str(ARCHIVE.parent),
        dataset_download=False,
        dataset_train_subset_size=1024,
        dataset_guard_subset_size=256,
        dataset_validation_subset_size=256,
        dataset_test_subset_size=512,
        dataset_num_workers=0,
        dataset_use_augmentation=True,
        image_size=32,
        batch_size=32,
        epochs=1,
        seed=SELECTION_SEEDS[0],
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="imagenet",
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
    # Why this: preserve the earlier equal two-trial grid before seeing scores.
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


def _verify_equal_inputs(selection: MatchedHeadTuningResult) -> None:
    if len(selection.attempts) != 6 or len(selection.trials) != 6:
        raise AssertionError("Pretrained validation selection omitted an equal trial")
    if any(attempt.status != "complete" for attempt in selection.attempts):
        raise AssertionError("Pretrained validation selection has an incomplete trial")
    if len(selection.selections) != 3 or selection.confirmations:
        raise AssertionError("Pretrained validation selection or final-test state is incomplete")
    for field in ("split_hashes", "feature_hashes", "backbone_hash", "initial_head_hash"):
        values = {json.dumps(getattr(trial, field), sort_keys=True) for trial in selection.trials}
        if len(values) != 1:
            raise AssertionError(f"Matched validation trials differ in {field}")


def main() -> None:
    paths = {
        name: ARTIFACT_DIR / f"{PREFIX}_{name}_smoke.json"
        for name in ("request", "selection", "manifest", "failure")
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Pretrained-CIFAR validation artifact already exists")
    if set(SELECTION_SEEDS) & set(CONFIRMATION_SEEDS):
        raise ValueError("Selection and confirmation seeds must be disjoint")
    inputs = _verify_local_inputs()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    base = _base_config()
    candidates = _candidates(base)
    _save_json(
        paths["request"],
        {
            **inputs,
            "selection_seeds": SELECTION_SEEDS,
            "confirmation_seeds": CONFIRMATION_SEEDS,
            "candidates_per_head": 2,
            "wall_time_budget_seconds": WALL_TIME_BUDGET_SECONDS,
            "wall_time_epoch_cap": WALL_TIME_EPOCH_CAP,
            "selection_limit_seconds": SELECTION_LIMIT_SECONDS,
            "confirmation_limit_seconds": 480,
            "base_config": asdict(base),
            "candidates": {
                name: [asdict(candidate) for candidate in options]
                for name, options in candidates.items()
            },
        },
    )

    original_build = matched._build_benchmark_loaders
    test_iterations = [0]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            test_iterations[0] += 1
            raise AssertionError("Final CIFAR test opened during pretrained validation selection")

    def build_sealed_loaders(
        config: ResNet50BenchmarkConfig, *, include_final_test: bool = True
    ) -> Any:
        # Why this: a sealed iterator still allows early final-source construction.
        if include_final_test:
            raise AssertionError("Pretrained validation selection requested the final source")
        loaders = original_build(config, include_final_test=False)
        return replace(loaders, test_loader=SealedTestLoader())

    started = monotonic()
    try:
        with patch.object(matched, "_build_benchmark_loaders", build_sealed_loaders):
            selection = run_matched_head_tuning(
                base,
                candidates,
                seeds=SELECTION_SEEDS,
                candidates_per_head=2,
                confirm_test=False,
                development_only_source=True,
            )
        elapsed = monotonic() - started
        if elapsed > SELECTION_LIMIT_SECONDS:
            raise TimeoutError(f"Selection exceeded {SELECTION_LIMIT_SECONDS} s: {elapsed:.2f} s")
        if test_iterations[0] != 0:
            raise AssertionError("Final CIFAR test was iterated during selection")
        _verify_equal_inputs(selection)
        _save_json(paths["selection"], asdict(selection))
        manifest = create_confirmation_manifest(
            selection,
            confirmation_seeds=CONFIRMATION_SEEDS,
            wall_time_budget_seconds=WALL_TIME_BUDGET_SECONDS,
            wall_time_epoch_cap=WALL_TIME_EPOCH_CAP,
        )
        _save_json(paths["manifest"], asdict(manifest))
        restored = restore_confirmation_manifest(json.loads(paths["manifest"].read_text()))
        if restored != manifest:
            raise AssertionError("Saved confirmation manifest did not round-trip")
    except Exception as error:
        failure: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": round(monotonic() - started, 2),
            "final_test_iterations": test_iterations[0],
        }
        if isinstance(error, MatchedHeadTuningError):
            failure["attempts"] = [asdict(item) for item in error.attempts]
            failure["trials"] = [asdict(item) for item in error.trials]
        _save_json(paths["failure"], failure)
        raise
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "selection_seeds": SELECTION_SEEDS,
                "confirmation_seeds": CONFIRMATION_SEEDS,
                "selections": {item.head_name: item.candidate_id for item in selection.selections},
                "attempts": len(selection.attempts),
                "selection_seconds": round(elapsed, 2),
                "final_test_iterations": test_iterations[0],
                "artifacts": {name: str(path) for name, path in paths.items() if path.exists()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
