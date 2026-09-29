"""Confirm P1.8m's frozen CUDA manifest after a measured quiet GPU window.

The preflight checks source and manifest integrity without final-test access.
The full route defers itself when the predeclared GPU load gate is unmet.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch  # noqa: E402
import torchvision  # noqa: E402

from scripts import run_cifar_pretrained_validation as cpu_study  # noqa: E402
from src.app.repeated_head_confirmation import (  # noqa: E402
    RepeatedConfirmationManifest,
    RepeatedConfirmationResult,
    restore_confirmation_manifest,
    run_repeated_confirmation,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREFIX = "benchmark_cifar_pretrained_cuda_v1"
RESULT_PATH = ARTIFACT_DIR / f"{PREFIX}_result_smoke.json"
FAILURE_PATH = ARTIFACT_DIR / f"{PREFIX}_failure_smoke.json"
LIMIT_SECONDS = 480
MIN_FREE_MIB = 5120
MAX_UTILIZATION_PERCENT = 10
QUIET_READINGS = 3
QUIET_SPACING_SECONDS = 5


def _read_json(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACT_DIR / f"{PREFIX}_{name}_smoke.json").read_text(encoding="utf-8"))


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _verify_saved_request() -> RepeatedConfirmationManifest:
    request = _read_json("request")
    selection = _read_json("selection")
    manifest = restore_confirmation_manifest(_read_json("manifest"))
    if torch.__version__ != "2.14.0+cu130" or torchvision.__version__ != "0.29.0+cu130":
        raise RuntimeError("CUDA package versions changed after validation selection")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA confirmation requires a CUDA device")
    inputs = cpu_study._verify_local_inputs()
    if any(request[key] != value for key, value in inputs.items()):
        raise ValueError("Saved CUDA archive or checkpoint provenance changed")
    source_digest = sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        source_digest != manifest.source_selection_digest
        or request["base_config"] != asdict(manifest.base_config)
        or selection["base_config"] != request["base_config"]
        or request["selection_seeds"] != list(manifest.selection_seeds)
        or request["confirmation_seeds"] != list(manifest.confirmation_seeds)
        or manifest.selection_seeds != (151,)
        or manifest.confirmation_seeds != (157, 163, 167)
        or selection["seeds"] != request["selection_seeds"]
        or selection["candidates_per_head"] != request["candidates_per_head"]
        or selection["confirmations"]
        or request["wall_time_budget_seconds"] != manifest.wall_time_budget_seconds
        or request["wall_time_epoch_cap"] != manifest.wall_time_epoch_cap
        or request["confirmation_limit_seconds"] != LIMIT_SECONDS
        or request["deterministic_algorithms"] is not True
        or request["cublas_workspace_config"] != os.environ["CUBLAS_WORKSPACE_CONFIG"]
        or request["torch"] != torch.__version__
        or request["torchvision"] != torchvision.__version__
    ):
        raise ValueError("Saved CUDA request, selection, and manifest disagree")
    if len(selection["attempts"]) != 6 or any(
        attempt["status"] != "complete" for attempt in selection["attempts"]
    ):
        raise ValueError("Saved CUDA validation trial ledger is incomplete")
    selected = {item["head_name"]: item["candidate_id"] for item in selection["selections"]}
    if len(selected) != 3 or len(selection["selections"]) != 3:
        raise ValueError("Saved CUDA validation selections are incomplete")
    for item in manifest.selected_heads:
        options = request["candidates"][item.head_name]
        chosen = [
            candidate for candidate in options if candidate["candidate_id"] == item.candidate_id
        ]
        if (
            selected[item.head_name] != item.candidate_id
            or len(options) != 2
            or len(chosen) != 1
            or chosen[0]["config"] != asdict(item.config)
        ):
            raise ValueError(f"Saved CUDA candidate changed for {item.head_name}")
    return manifest


def _gpu_sample() -> dict[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    )
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("Quiet-window gate requires exactly one NVIDIA GPU")
    utilization, free_mib, used_mib = (int(part.strip()) for part in rows[0].split(","))
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "utilization_percent": utilization,
        "free_mib": free_mib,
        "used_mib": used_mib,
    }


def _quiet_window() -> tuple[dict[str, Any], ...]:
    samples = []
    for index in range(QUIET_READINGS):
        if index:
            sleep(QUIET_SPACING_SECONDS)
        samples.append(_gpu_sample())
    return tuple(samples)


def _verify_result(
    result: RepeatedConfirmationResult,
    manifest: RepeatedConfirmationManifest,
) -> None:
    if result.manifest != manifest or len(result.fixed_data.confirmations) != 9:
        raise AssertionError("CUDA fixed-data confirmation is incomplete")
    if len(result.wall_time) != 3 or len(result.capacity_memory) != 3:
        raise AssertionError("CUDA confirmation omitted a declared seed/scope")
    if any(len(report.reports) != 3 for report in result.capacity_memory):
        raise AssertionError("CUDA process memory omitted a matched head")
    for summaries in (
        result.fixed_data_accuracy,
        result.wall_time_accuracy,
        result.observed_train_rss,
    ):
        if set(summaries) != {"backprop_mlp", "predictive_coding", "circadian_predictive_coding"}:
            raise AssertionError("CUDA confirmation summary omitted a matched head")
        if any(summary.seeds != manifest.confirmation_seeds for summary in summaries.values()):
            raise AssertionError("CUDA confirmation summary changed the declared seed set")


def _run_confirmation() -> None:
    if RESULT_PATH.exists() or FAILURE_PATH.exists():
        raise FileExistsError("CUDA confirmation already has a result or failure")
    manifest = _verify_saved_request()
    readings = _quiet_window()
    quiet = all(
        item["utilization_percent"] <= MAX_UTILIZATION_PERCENT and item["free_mib"] >= MIN_FREE_MIB
        for item in readings
    )
    if not quiet:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        deferred = ARTIFACT_DIR / f"{PREFIX}_deferred_{timestamp}_smoke.json"
        _save_json(
            deferred,
            {
                "manifest_digest": manifest.manifest_digest,
                "readings": readings,
                "max_utilization_percent": MAX_UTILIZATION_PERCENT,
                "min_free_mib": MIN_FREE_MIB,
                "final_test_iterations": 0,
            },
        )
        print(json.dumps({"deferred": str(deferred), "readings": readings}, sort_keys=True))
        raise SystemExit(2)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    started = monotonic()
    try:
        result = run_repeated_confirmation(manifest)
        elapsed = monotonic() - started
        if elapsed > LIMIT_SECONDS:
            raise TimeoutError(f"CUDA confirmation exceeded {LIMIT_SECONDS} s")
        _verify_result(result, manifest)
        after = _gpu_sample()
        _save_json(
            RESULT_PATH,
            {
                "result": asdict(result),
                "quiet_window": readings,
                "gpu_after": after,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
    except Exception as error:
        _save_json(
            FAILURE_PATH,
            {
                "manifest_digest": manifest.manifest_digest,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(monotonic() - started, 3),
                "quiet_window": readings,
            },
        )
        raise
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "elapsed_seconds": round(elapsed, 3),
                "result_path": str(RESULT_PATH),
                "fixed_data_accuracy": {
                    name: asdict(summary) for name, summary in result.fixed_data_accuracy.items()
                },
                "wall_time_accuracy": {
                    name: asdict(summary) for name, summary in result.wall_time_accuracy.items()
                },
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        manifest = _verify_saved_request()
        print(json.dumps({"manifest_digest": manifest.manifest_digest, "final_test_iterations": 0}))
    else:
        _run_confirmation()


if __name__ == "__main__":
    main()
