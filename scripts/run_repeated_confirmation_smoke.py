"""Run one predeclared, tiny CPU matched-head confirmation with saved evidence."""

from __future__ import annotations

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


def main() -> None:
    paths = {
        name: OUTPUT_DIR / f"benchmark_repeated_{name}_smoke.json"
        for name in ("selection", "manifest", "result")
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Repeated-confirmation smoke artifacts already exist.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = _base_config()
    selection = run_matched_head_tuning(
        base,
        _candidates(base),
        seeds=SELECTION_SEEDS,
        candidates_per_head=2,
        confirm_test=False,
    )
    manifest = create_confirmation_manifest(
        selection,
        confirmation_seeds=CONFIRMATION_SEEDS,
        wall_time_budget_seconds=WALL_TIME_BUDGET_SECONDS,
        wall_time_epoch_cap=WALL_TIME_EPOCH_CAP,
    )
    # Keep both records on disk before the confirmation route can read test labels.
    _save_json(paths["selection"], selection)
    _save_json(paths["manifest"], manifest)
    result = run_repeated_confirmation(manifest)
    _save_json(paths["result"], result)
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "selection_seeds": SELECTION_SEEDS,
                "confirmation_seeds": CONFIRMATION_SEEDS,
                "artifacts": {name: str(path) for name, path in paths.items()},
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
    main()
