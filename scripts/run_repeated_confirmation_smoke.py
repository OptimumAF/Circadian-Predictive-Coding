"""Run one predeclared, tiny CPU matched-head confirmation with saved evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.matched_head_tuning import (  # noqa: E402
    HeadTuningCandidate,
    MatchedHeadTuningError,
    run_matched_head_tuning,
)
from src.app.repeated_head_confirmation import (  # noqa: E402
    create_confirmation_manifest,
    run_repeated_confirmation,
)
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402

SELECTION_SEEDS = (47,)
CONFIRMATION_SEEDS = (53, 59, 61)
WALL_TIME_BUDGET_SECONDS = 0.05
WALL_TIME_EPOCH_CAP = 1000
OUTPUT_DIR = REPO_ROOT / "artifacts"
FAILURE_SCHEMA = "vision_matched_head_repeated_failure_v1"


def _base_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
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
        json.dump(asdict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"benchmark_repeated_{name}_smoke.json"
        for name in ("selection", "manifest", "result", "failure")
    }


def _save_failure(paths: dict[str, Path], stage: str, error: Exception) -> None:
    # Why this: a failed confirmation must retain the frozen selection and
    # manifest without presenting a missing result as a completed study.
    record: dict[str, Any] = {
        "schema_id": FAILURE_SCHEMA,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "existing_artifacts": {
            name: {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()
            if name != "failure" and path.is_file()
        },
    }
    if isinstance(error, MatchedHeadTuningError):
        record["attempts"] = [asdict(attempt) for attempt in error.attempts]
        record["trials"] = [asdict(trial) for trial in error.trials]
        record["selections"] = [asdict(choice) for choice in error.selections]
    with paths["failure"].open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(*, output_dir: Path = OUTPUT_DIR) -> None:
    paths = _artifact_paths(output_dir)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Repeated-confirmation smoke artifacts already exist.")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _base_config()
    stage = "selection"
    try:
        selection = run_matched_head_tuning(
            base,
            _candidates(base),
            seeds=SELECTION_SEEDS,
            candidates_per_head=2,
            confirm_test=False,
        )
        stage = "manifest"
        manifest = create_confirmation_manifest(
            selection,
            confirmation_seeds=CONFIRMATION_SEEDS,
            wall_time_budget_seconds=WALL_TIME_BUDGET_SECONDS,
            wall_time_epoch_cap=WALL_TIME_EPOCH_CAP,
        )
        # Keep both records on disk before the confirmation route can read test labels.
        stage = "selection_artifact"
        _save_json(paths["selection"], selection)
        stage = "manifest_artifact"
        _save_json(paths["manifest"], manifest)
        stage = "confirmation"
        result = run_repeated_confirmation(manifest)
        stage = "result_artifact"
        _save_json(paths["result"], result)
    except Exception as error:
        _save_failure(paths, stage, error)
        raise
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "selection_seeds": SELECTION_SEEDS,
                "confirmation_seeds": CONFIRMATION_SEEDS,
                "artifacts": {name: str(path) for name, path in paths.items() if name != "failure"},
                "fixed_data_accuracy": {
                    name: asdict(summary) for name, summary in result.fixed_data_accuracy.items()
                },
                "wall_time_accuracy": {
                    name: asdict(summary) for name, summary in result.wall_time_accuracy.items()
                },
                "observed_train_rss": {
                    name: asdict(summary) for name, summary in result.observed_train_rss.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="new local directory for the fixed selection/confirmation artifacts",
    )
    main(output_dir=parser.parse_args().output_dir)
