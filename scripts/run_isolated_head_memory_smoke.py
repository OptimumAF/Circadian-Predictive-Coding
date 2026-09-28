"""Run one tiny synthetic, process-isolated matched-head memory check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.isolated_head_memory import run_process_isolated_fixed_width_memory  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402


def main() -> None:
    config = ResNet50BenchmarkConfig(
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=47,
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=16,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
    )
    result = run_process_isolated_fixed_width_memory(config, timeout_seconds=60)
    summary = {
        "protocol_id": result.protocol_id,
        "source_protocol_id": result.source_protocol_id,
        "memory_scope": result.memory_scope,
        "seed": config.seed,
        "head_reports": {
            name: {
                "pid": report.pid,
                "head_parameters": report.head_parameters,
                "feature_bytes": report.feature_bytes,
                "setup_rss_start_bytes": report.setup_rss_start_bytes,
                "setup_rss_peak_observed_bytes": report.setup_rss_peak_observed_bytes,
                "pretrain_rss_bytes": report.pretrain_rss_bytes,
                "train_rss_start_bytes": report.train_rss_start_bytes,
                "train_rss_peak_observed_bytes": report.train_rss_peak_observed_bytes,
                "guard_examples_scored": report.guard_examples_scored,
                "sleep_attempts": report.sleep_attempts,
                "backbone_hash": report.backbone_hash,
                "initial_head_hash": report.initial_head_hash,
                "feature_hashes": report.feature_hashes,
            }
            for name, report in result.reports.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
