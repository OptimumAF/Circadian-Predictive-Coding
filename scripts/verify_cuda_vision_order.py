"""Check P1.7g's predeclared real-CIFAR seeded vision reversal on CUDA.

Each order runs in a separate bounded process. Raw reports are saved before
comparison, including when exact state hashes or declared metrics disagree.
This is an unmatched-reference reproducibility check, not a head ranking.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import md5
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app import resnet50_benchmark as vision  # noqa: E402
from src.app.resnet50_benchmark import (  # noqa: E402
    ResNet50BenchmarkConfig,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
)

ARCHIVE = REPO_ROOT / "data" / "cifar-10-python.tar.gz"
ARCHIVE_BYTES = 170_498_071
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
OUTPUT_DIR = REPO_ROOT / "data"
PREFIX = "cuda-vision-order-seed149"
FORWARD_ORDER = ("backprop", "predictive", "circadian")
REVERSE_ORDER = tuple(reversed(FORWARD_ORDER))
SEED = 149
WORKER_LIMIT_SECONDS = 120
METRIC_TOLERANCE = 1e-6
CUBLAS_CONFIG = ":4096:8"


def _path(name: str) -> Path:
    return OUTPUT_DIR / f"{PREFIX}-{name}.json"


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
        dataset_name="cifar10",
        dataset_data_root=str(ARCHIVE.parent),
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
        seed=SEED,
        device="cuda",
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
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
    )


def _verify_cache() -> None:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != ARCHIVE_BYTES:
        raise FileNotFoundError("Verified local CIFAR-10 archive is required")
    digest = md5()
    with ARCHIVE.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("Local CIFAR-10 archive MD5 changed")
    cache = ARCHIVE.parent / "cifar-10-batches-py"
    if not (cache / "data_batch_1").is_file() or not (cache / "test_batch").is_file():
        raise FileNotFoundError("Extracted CIFAR-10 cache is incomplete")


def _run_order(order_name: str) -> None:
    import torch
    import torchvision

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
        raise RuntimeError("CUBLAS workspace does not match the predeclared setting")
    if torch.__version__ != "2.14.0+cu130" or torchvision.__version__ != "0.29.0+cu130":
        raise RuntimeError("CUDA package versions changed after P1.7f")
    if not torch.cuda.is_available():
        raise RuntimeError("P1.7g requires actual CUDA execution")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    order = FORWARD_ORDER if order_name == "forward" else REVERSE_ORDER
    completed: list[str] = []
    test_iterations = [0]
    original_build = vision._build_benchmark_loaders
    original_train = vision._train_seeded_variant

    class SealedTestLoader:
        def __init__(self, source: Any) -> None:
            self.source = source

        def __iter__(self) -> Any:
            if len(completed) != len(order):
                raise AssertionError("CUDA final test opened before every model finished")
            test_iterations[0] += 1
            return iter(self.source)

    def build_sealed_loaders(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    def record_training(variant: str, *args: Any, **kwargs: Any) -> Any:
        outcome = original_train(variant, *args, **kwargs)
        completed.append(variant)
        return outcome

    started = monotonic()
    with (
        patch.object(vision, "_build_benchmark_loaders", build_sealed_loaders),
        patch.object(vision, "_train_seeded_variant", record_training),
    ):
        result = vision.run_resnet50_benchmark(_config(), model_order=order)
    torch.cuda.synchronize()
    if tuple(completed) != order or result.training_order != order or test_iterations[0] == 0:
        raise AssertionError("CUDA order run did not complete its sealed training/test stages")
    if result.trained_model_hashes is None:
        raise AssertionError("Seeded CUDA reference omitted trained-model hashes")
    _save_json(
        _path(order_name),
        {
            "order": order,
            "completed_training": completed,
            "final_test_iterations_after_training": test_iterations[0],
            "elapsed_seconds": round(monotonic() - started, 3),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": CUBLAS_CONFIG,
            "result": asdict(result),
        },
    )
    print(json.dumps({"completed_order": order_name, "elapsed_seconds": monotonic() - started}))


def _compare(forward: dict[str, Any], reverse: dict[str, Any]) -> dict[str, Any]:
    left = forward["result"]
    right = reverse["result"]
    left_reports = {item["model_name"]: item for item in left["reports"]}
    right_reports = {item["model_name"]: item for item in right["reports"]}
    metrics = ("validation_accuracy", "test_accuracy", "final_cross_entropy")
    deltas = {
        name: {
            metric: abs(left_reports[name][metric] - right_reports[name][metric])
            for metric in metrics
        }
        for name in left_reports
    }
    circadian = "CircadianPredictiveCodingResNet50"
    checks = {
        "config_equal": left["config"] == right["config"],
        "role_hashes_equal": left["split_hashes"] == right["split_hashes"],
        "trained_model_hashes_equal": left["trained_model_hashes"] == right["trained_model_hashes"],
        "model_sets_equal": set(left_reports) == set(right_reports),
        "test_sealed_in_both": all(
            item["final_test_iterations_after_training"] > 0 for item in (forward, reverse)
        ),
        "forced_sleep_attempted_in_both": all(
            reports[circadian]["circadian_sleep_attempts"] >= 1
            for reports in (left_reports, right_reports)
        ),
        "metric_tolerance_met": all(
            isfinite(delta) and delta <= METRIC_TOLERANCE
            for by_model in deltas.values()
            for delta in by_model.values()
        ),
    }
    return {
        "protocol_id": VISION_SEEDED_UNMATCHED_PROTOCOL,
        "seed": SEED,
        "device": "cuda",
        "metric_absolute_tolerance": METRIC_TOLERANCE,
        "checks": checks,
        "metric_absolute_deltas": deltas,
        "forward_order": forward["order"],
        "reverse_order": reverse["order"],
        "forward_sleep_attempts": left_reports[circadian]["circadian_sleep_attempts"],
        "reverse_sleep_attempts": right_reports[circadian]["circadian_sleep_attempts"],
        "passed": all(checks.values()),
    }


def _run_parent() -> None:
    names = ("request", "forward", "reverse", "result", "failure")
    if any(_path(name).exists() for name in names):
        raise FileExistsError("CUDA vision order artifacts already exist")
    _verify_cache()
    _save_json(
        _path("request"),
        {
            "config": asdict(_config()),
            "archive": str(ARCHIVE),
            "archive_bytes": ARCHIVE_BYTES,
            "archive_md5": ARCHIVE_MD5,
            "orders": (FORWARD_ORDER, REVERSE_ORDER),
            "worker_limit_seconds": WORKER_LIMIT_SECONDS,
            "metric_absolute_tolerance": METRIC_TOLERANCE,
            "require_exact_trained_model_hashes": True,
            "require_final_test_seal": True,
            "cublas_workspace_config": CUBLAS_CONFIG,
        },
    )
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=CUBLAS_CONFIG)
    started = monotonic()
    try:
        for order_name in ("forward", "reverse"):
            subprocess.run(
                [sys.executable, __file__, "--worker", order_name],
                check=True,
                timeout=WORKER_LIMIT_SECONDS,
                env=environment,
                cwd=REPO_ROOT,
            )
        comparison = _compare(
            json.loads(_path("forward").read_text(encoding="utf-8")),
            json.loads(_path("reverse").read_text(encoding="utf-8")),
        )
        _save_json(_path("result"), comparison)
        if not comparison["passed"]:
            raise AssertionError("CUDA model-order reversal failed the predeclared checks")
    except Exception as error:
        _save_json(
            _path("failure"),
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(monotonic() - started, 3),
                "saved_artifacts": [name for name in names if _path(name).exists()],
            },
        )
        raise
    print(json.dumps(comparison, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("forward", "reverse"))
    args = parser.parse_args()
    if args.worker:
        _run_order(args.worker)
    else:
        _run_parent()


if __name__ == "__main__":
    main()
