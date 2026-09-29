"""One predeclared CUDA development-feature cost probe without a test source.

``prepare`` writes the fixed request. ``run`` checks source integrity and a
quiet GPU, then bounds a fresh worker to 120 seconds. The worker constructs
only train, inner guard, and outer validation roles. No head is trained.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch  # noqa: E402
import torchvision  # noqa: E402

from scripts import run_cifar_pretrained_validation as prior_study  # noqa: E402
from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.matched_head_tuning import _build_seed_bank  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.shared.process_memory import ProcessRssSampler  # noqa: E402


REQUEST_PATH = REPO_ROOT / "data" / "cifar-representative-feasibility-v1-request.json"
RESULT_PATH = REPO_ROOT / "data" / "cifar-representative-feasibility-v1-result.json"
FAILURE_PATH = REPO_ROOT / "data" / "cifar-representative-feasibility-v1-failure.json"
ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
EXPECTED_TORCH = "2.14.0+cu130"
EXPECTED_TORCHVISION = "0.29.0+cu130"
PROBE_TIMEOUT_SECONDS = 120
QUIET_READINGS = 3
QUIET_SPACING_SECONDS = 5
MAX_UTILIZATION_PERCENT = 10
MIN_FREE_MIB = 5120
ROLE_ORDER = ("train", "guard", "validation")


def _config() -> ResNet50BenchmarkConfig:
    # Why this: change only scale, seed, and device from the matched P1.8n controls.
    return replace(
        prior_study._base_config(),
        seed=173,
        device="cuda",
        image_size=224,
        dataset_train_subset_size=4096,
        dataset_guard_subset_size=512,
        dataset_validation_subset_size=512,
        batch_size=32,
    )


def _request() -> dict[str, Any]:
    return {
        "schema": "cifar_representative_feature_feasibility_v1",
        "purpose": "development_only_feature_cost_no_head_training_or_final_test",
        "config": asdict(_config()),
        "sources": {
            "archive": str(prior_study.ARCHIVE),
            "archive_bytes": prior_study.ARCHIVE_BYTES,
            "archive_md5": prior_study.ARCHIVE_MD5,
            "archive_sha256": ARCHIVE_SHA256,
            "weight_bytes": prior_study.WEIGHT_BYTES,
            "weight_sha256": prior_study.WEIGHT_SHA256,
            "weight_url": prior_study.WEIGHT_URL,
        },
        "torch": EXPECTED_TORCH,
        "torchvision": EXPECTED_TORCHVISION,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "roles": {"train": 4096, "guard": 512, "validation": 512},
        "final_test_policy": "do_not_construct_or_iterate",
        "probe_timeout_seconds": PROBE_TIMEOUT_SECONDS,
        "quiet_window": {
            "readings": QUIET_READINGS,
            "spacing_seconds": QUIET_SPACING_SECONDS,
            "max_utilization_percent": MAX_UTILIZATION_PERCENT,
            "min_free_mib": MIN_FREE_MIB,
        },
        "memory_scopes": ["worker_process_rss", "worker_cuda_allocator"],
    }


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _save_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value))


def prepare_request(path: Path = REQUEST_PATH) -> None:
    """Freeze all source, scale, role, device, and time settings before loading data."""
    _save_exclusive(path, _request())


def _read_request(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError("predeclared representative feasibility request is missing")
    raw = path.read_bytes()
    if raw != _canonical_bytes(_request()):
        raise ValueError("predeclared representative feasibility request changed")
    return json.loads(raw), sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _verify_sources(request: dict[str, Any]) -> dict[str, Any]:
    found = prior_study._verify_local_inputs()
    source = request["sources"]
    if any(found[key] != source[key] for key in found if key in source):
        raise ValueError("local CIFAR archive or pretrained weight provenance changed")
    if _hash_file(prior_study.ARCHIVE) != source["archive_sha256"]:
        raise ValueError("local CIFAR archive SHA-256 changed")
    return {**found, "archive_sha256": source["archive_sha256"]}


def _gpu_sample() -> dict[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    )
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("feasibility gate requires exactly one NVIDIA GPU")
    utilization, free_mib, used_mib = (int(part.strip()) for part in rows[0].split(","))
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "utilization_percent": utilization,
        "free_mib": free_mib,
        "used_mib": used_mib,
    }


def _quiet_window() -> list[dict[str, Any]]:
    readings = []
    for index in range(QUIET_READINGS):
        if index:
            sleep(QUIET_SPACING_SECONDS)
        readings.append(_gpu_sample())
    return readings


def _verify_cuda_runtime() -> None:
    if torch.__version__ != EXPECTED_TORCH or torchvision.__version__ != EXPECTED_TORCHVISION:
        raise RuntimeError("representative feasibility requires the verified CUDA package pair")
    if not torch.cuda.is_available() or os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("representative feasibility requires deterministic CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _measure_worker(request: dict[str, Any]) -> dict[str, Any]:
    _verify_cuda_runtime()
    config = _config()
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_start = torch.cuda.memory_allocated(device)
    reserved_start = torch.cuda.memory_reserved(device)
    original_materialize = matched._materialize_role_features
    role_measurements: dict[str, dict[str, Any]] = {}

    def timed_materialize(*args: Any, **kwargs: Any) -> Any:
        role = ROLE_ORDER[len(role_measurements)]
        torch.cuda.synchronize(device)
        started = monotonic()
        batches = original_materialize(*args, **kwargs)
        torch.cuda.synchronize(device)
        role_measurements[role] = {
            "seconds": monotonic() - started,
            "batches": len(batches),
            "examples": sum(int(labels.shape[0]) for _, labels in batches),
            "feature_bytes": sum(
                tensor.numel() * tensor.element_size() for pair in batches for tensor in pair
            ),
        }
        return batches

    started = monotonic()
    with (
        patch.object(matched, "_materialize_role_features", timed_materialize),
        ProcessRssSampler(interval_seconds=0.01) as rss,
    ):
        bank = _build_seed_bank(torch, device, config, include_final_test=False)
    torch.cuda.synchronize(device)
    elapsed = monotonic() - started
    if (
        set(bank.loaders.sample_ids) != set(ROLE_ORDER)
        or set(bank.loaders.split_hashes) != set(ROLE_ORDER)
        or set(role_measurements) != set(ROLE_ORDER)
        or any(
            role_measurements[role]["examples"] != count for role, count in request["roles"].items()
        )
    ):
        raise AssertionError("development-only feature bank crossed a role or count boundary")
    if elapsed > PROBE_TIMEOUT_SECONDS:
        raise TimeoutError("predeclared feature probe exceeded its 120 s budget")
    return {
        "seed": config.seed,
        "image_size": config.image_size,
        "elapsed_seconds": elapsed,
        "setup_seconds": elapsed - sum(item["seconds"] for item in role_measurements.values()),
        "roles": role_measurements,
        "split_hashes": bank.split_hashes,
        "feature_hashes": bank.feature_hashes,
        "backbone_hash": bank.backbone_hash,
        "final_test_constructions": 0,
        "final_test_iterations": 0,
        "rss_scope": "worker_feature_setup_absolute_observed_rss",
        "rss": asdict(rss.snapshot()),
        "cuda_scope": "worker_torch_allocator_feature_setup",
        "cuda_allocated_start_bytes": allocated_start,
        "cuda_reserved_start_bytes": reserved_start,
        "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
        "cuda_allocated_end_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_end_bytes": torch.cuda.memory_reserved(device),
    }


def run_probe(
    request_path: Path = REQUEST_PATH,
    result_path: Path = RESULT_PATH,
    failure_path: Path = FAILURE_PATH,
) -> dict[str, Any]:
    """Check the unchanged request, quiet window, and bounded worker."""
    request, request_digest = _read_request(request_path)
    if result_path.exists() or failure_path.exists():
        raise FileExistsError("representative feasibility already has a result or failure")
    _verify_cuda_runtime()
    sources = _verify_sources(request)
    readings = _quiet_window()
    quiet = all(
        row["utilization_percent"] <= MAX_UTILIZATION_PERCENT and row["free_mib"] >= MIN_FREE_MIB
        for row in readings
    )
    if not quiet:
        _save_exclusive(
            failure_path,
            {
                "status": "deferred_busy_gpu",
                "request_sha256": request_digest,
                "quiet_window": readings,
                "final_test_constructions": 0,
            },
        )
        raise RuntimeError("predeclared quiet GPU gate was not met")
    command = (
        sys.executable,
        "-m",
        "scripts.profile_cifar_representative_feasibility",
        "worker",
        "--request",
        str(request_path),
    )
    started = monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        measurement = json.loads(completed.stdout)
        if measurement["final_test_constructions"] or measurement["final_test_iterations"]:
            raise AssertionError("development-only worker opened a final test")
        if sha256(request_path.read_bytes()).hexdigest() != request_digest:
            raise ValueError("predeclared request changed during the feature probe")
    except Exception as error:
        _save_exclusive(
            failure_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "request_sha256": request_digest,
                "source_hashes": sources,
                "quiet_window": readings,
                "worker_elapsed_seconds": monotonic() - started,
                "final_test_constructions": 0,
            },
        )
        raise
    report = {
        "schema": "cifar_representative_feature_feasibility_result_v1",
        "request_sha256": request_digest,
        "source_hashes": sources,
        "quiet_window": readings,
        "worker_elapsed_seconds": monotonic() - started,
        "measurement": measurement,
        "gpu_after": _gpu_sample(),
    }
    _save_exclusive(result_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--request", type=Path, default=REQUEST_PATH)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, default=REQUEST_PATH)
    run.add_argument("--result", type=Path, default=RESULT_PATH)
    run.add_argument("--failure", type=Path, default=FAILURE_PATH)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_request(args.request)
        print(args.request)
    elif args.command == "worker":
        request, _ = _read_request(args.request)
        print(json.dumps(_measure_worker(request), sort_keys=True, allow_nan=False))
    else:
        report = run_probe(args.request, args.result, args.failure)
        print(
            json.dumps(
                {
                    "result": str(args.result),
                    "elapsed_seconds": report["measurement"]["elapsed_seconds"],
                    "roles": report["measurement"]["roles"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
