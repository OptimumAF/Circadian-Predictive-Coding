"""Run the frozen 224-pixel CUDA validation selection without final source.

The parent verifies the immutable request, source hashes, and quiet GPU.
A child has a hard 180-second limit and returns all six development trials.
Only after a complete selection does the parent save the confirmation freeze.
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
from time import monotonic
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch  # noqa: E402
import torchvision  # noqa: E402

from scripts import prepare_cifar_representative_study as study  # noqa: E402
from scripts import profile_cifar_representative_feasibility as probe  # noqa: E402
from scripts import run_cifar_pretrained_validation as prior_study  # noqa: E402
from src.app.matched_head_tuning import (  # noqa: E402
    MatchedHeadTuningError,
    run_matched_head_tuning,
)
from src.app.repeated_head_confirmation import (  # noqa: E402
    create_confirmation_manifest,
    restore_confirmation_manifest,
)
from src.infra import vision_datasets  # noqa: E402


REQUEST_PATH = study.REQUEST_PATH
REQUEST_SHA256 = "baa4bcaf5138946d01ad0d7c1b8babc21350ae18e53e24bc4e37a9cedf3dc773"
RESULT_PATH = REPO_ROOT / "data" / "cifar-representative-selection-v1-result.json"
MANIFEST_PATH = REPO_ROOT / "data" / "cifar-representative-selection-v1-manifest.json"
FAILURE_PATH = REPO_ROOT / "data" / "cifar-representative-selection-v1-failure.json"
JOURNAL_PATH = REPO_ROOT / "data" / "cifar-representative-selection-v1-attempts.jsonl"
HEAD_NAMES = ("backprop_mlp", "predictive_coding", "circadian_predictive_coding")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _save_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value))


def _verify_saved_request(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise ValueError("frozen representative study request is missing")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != REQUEST_SHA256:
        raise ValueError("frozen representative study request digest changed")
    request: dict[str, Any] = json.loads(raw)
    probe_result, probe_digest = study._read_verified_probe()
    if request != study._study_request(probe_result, probe_digest):
        raise ValueError("frozen representative study request content changed")
    verified_sources = probe._verify_sources(probe._request())
    if request["source"] != verified_sources:
        raise ValueError("frozen representative source hashes changed")
    return request, verified_sources


def _verify_cuda_runtime() -> None:
    if (
        torch.__version__ != probe.EXPECTED_TORCH
        or torchvision.__version__ != probe.EXPECTED_TORCHVISION
    ):
        raise RuntimeError("frozen representative study requires the verified CUDA pair")
    if not torch.cuda.is_available() or os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("frozen representative selection requires deterministic CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _append_journal(path: Path, attempt: Any, trial: Any) -> None:
    # Why this: a killed worker still leaves each completed trial and the
    # identity of the candidate that was in flight at the deadline.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"attempt": asdict(attempt), "trial": asdict(trial) if trial else None},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _read_journal(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )


def _worker_selection(request: dict[str, Any], journal_path: Path) -> dict[str, Any]:
    _verify_cuda_runtime()
    base = study._base_config()
    candidates = prior_study._candidates(base)
    if (
        asdict(base) != request["base_config"]
        or {
            name: [asdict(candidate) for candidate in options]
            for name, options in candidates.items()
        }
        != request["candidate_grid"]
    ):
        raise ValueError("worker configuration differs from frozen candidate grid")
    final_constructions = 0
    original_datasets = vision_datasets.require_torchvision_datasets

    def sealed_datasets() -> Any:
        module = original_datasets()

        def seal(factory: Any) -> Any:
            def construct(*args: Any, **kwargs: Any) -> Any:
                nonlocal final_constructions
                if kwargs.get("train") is False:
                    final_constructions += 1
                    raise AssertionError("selection constructed the CIFAR final-test source")
                return factory(*args, **kwargs)

            return construct

        return SimpleNamespace(CIFAR10=seal(module.CIFAR10), CIFAR100=seal(module.CIFAR100))

    with patch.object(vision_datasets, "require_torchvision_datasets", sealed_datasets):
        selection = run_matched_head_tuning(
            base,
            candidates,
            seeds=tuple(request["selection_seeds"]),
            candidates_per_head=request["candidates_per_head"],
            confirm_test=False,
            development_only_source=True,
            attempt_observer=lambda attempt, trial: _append_journal(journal_path, attempt, trial),
        )
    if final_constructions or selection.confirmations:
        raise AssertionError("validation selection opened final test")
    prior_study._verify_equal_inputs(selection)
    manifest = create_confirmation_manifest(
        selection,
        confirmation_seeds=tuple(request["confirmation_seeds"]),
        wall_time_budget_seconds=request["scopes"]["wall_time"]["per_head_seconds"],
        wall_time_epoch_cap=request["scopes"]["wall_time"]["epoch_cap"],
    )
    serialized_manifest = asdict(manifest)
    if restore_confirmation_manifest(serialized_manifest) != manifest:
        raise AssertionError("representative confirmation manifest did not round-trip")
    return {
        "status": "complete",
        "selection": asdict(selection),
        "confirmation_manifest": serialized_manifest,
        "final_test_source_constructions": final_constructions,
        "final_test_iterations": 0,
    }


def _validate_worker_payload(payload: dict[str, Any], request: dict[str, Any]) -> None:
    if payload.get("status") != "complete" or payload["final_test_source_constructions"] != 0:
        raise ValueError("selection worker did not complete under the final-source seal")
    selection = payload["selection"]
    attempts, trials, choices = (
        selection["attempts"],
        selection["trials"],
        selection["selections"],
    )
    expected = {
        (name, candidate, request["selection_seeds"][0])
        for name in HEAD_NAMES
        for candidate in ("a", "b")
    }
    if (
        len(attempts) != 6
        or len(trials) != 6
        or len(choices) != 3
        or selection["confirmations"]
        or {(row["head_name"], row["candidate_id"], row["seed"]) for row in trials} != expected
        or {(row["head_name"], row["candidate_id"], row["seed"]) for row in attempts} != expected
        or any(row["status"] != "complete" for row in attempts)
        or any(set(row["split_hashes"]) != {"train", "guard", "validation"} for row in trials)
    ):
        raise ValueError("representative validation selection trial ledger is incomplete")
    for field in ("split_hashes", "feature_hashes", "backbone_hash", "initial_head_hash"):
        if len({_digest(row[field]) for row in trials}) != 1:
            raise ValueError(f"representative trials received different {field}")
    for attempt, trial in zip(attempts, trials):
        if any(
            attempt[key] != trial[key] for key in ("head_name", "candidate_id", "seed", "config")
        ):
            raise ValueError("representative attempt and trial identities differ")
        grid = request["candidate_grid"][trial["head_name"]]
        declared = next(
            (row["config"] for row in grid if row["candidate_id"] == trial["candidate_id"]), None
        )
        if trial["config"] != declared:
            raise ValueError("representative trial changed its frozen candidate config")
    selected = {row["head_name"]: row for row in choices}
    if set(selected) != set(HEAD_NAMES):
        raise ValueError("representative selection omitted a head")
    for name in HEAD_NAMES:
        rows = [row for row in trials if row["head_name"] == name]
        best = max(rows, key=lambda row: row["validation_accuracy"])
        if selected[name]["candidate_id"] != best["candidate_id"]:
            raise ValueError("representative selection disagrees with frozen outer scores")
    manifest = restore_confirmation_manifest(payload["confirmation_manifest"])
    if (
        manifest.selection_seeds != tuple(request["selection_seeds"])
        or manifest.confirmation_seeds != tuple(request["confirmation_seeds"])
        or manifest.wall_time_budget_seconds != request["scopes"]["wall_time"]["per_head_seconds"]
        or manifest.wall_time_epoch_cap != request["scopes"]["wall_time"]["epoch_cap"]
        or manifest.source_selection_digest != _digest(selection)
    ):
        raise ValueError("representative confirmation manifest changed seeds or wall budget")


def _freeze_manifest(
    request: dict[str, Any], selection_result: dict[str, Any], typed_manifest: dict[str, Any]
) -> dict[str, Any]:
    frozen = {
        "schema": "cifar_representative_selection_freeze_v1",
        "request_sha256": REQUEST_SHA256,
        "selection_sha256": _digest(selection_result["selection"]),
        "source_hashes": request["source"],
        "selection_seeds": request["selection_seeds"],
        "confirmation_seeds": request["confirmation_seeds"],
        "scope_limits": request["scopes"],
        "confirmation_total_limit_seconds": request["confirmation_total_limit_seconds"],
        "typed_manifest": typed_manifest,
    }
    return {**frozen, "freeze_digest": _digest(frozen)}


def run_selection(
    request_path: Path = REQUEST_PATH,
    result_path: Path = RESULT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    failure_path: Path = FAILURE_PATH,
    journal_path: Path = JOURNAL_PATH,
) -> dict[str, Any]:
    """Run one bounded validation grid and save its frozen confirmation scope."""
    if any(path.exists() for path in (result_path, manifest_path, failure_path, journal_path)):
        raise FileExistsError("representative selection already has an artifact")
    request, sources = _verify_saved_request(request_path)
    _verify_cuda_runtime()
    readings = probe._quiet_window()
    quiet = all(
        row["utilization_percent"] <= request["quiet_window"]["max_utilization_percent"]
        and row["free_mib"] >= request["quiet_window"]["min_free_mib"]
        for row in readings
    )
    if not quiet:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        deferred = result_path.with_name(
            f"cifar-representative-selection-v1-deferred-{timestamp}.json"
        )
        _save_exclusive(
            deferred,
            {
                "status": "deferred_busy_gpu",
                "request_sha256": REQUEST_SHA256,
                "quiet_window": readings,
                "final_test_source_constructions": 0,
            },
        )
        raise RuntimeError("representative selection quiet GPU gate was not met")
    command = (
        sys.executable,
        "-m",
        "scripts.run_cifar_representative_selection",
        "worker",
        "--request",
        str(request_path),
        "--journal",
        str(journal_path),
    )
    worker_start = monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=request["selection_limit_seconds"],
            check=False,
        )
        payload = json.loads(completed.stdout)
        if completed.returncode != 0 or payload.get("status") != "complete":
            raise RuntimeError(payload.get("error", "validation selection worker failed"))
        _validate_worker_payload(payload, request)
        journal = _read_journal(journal_path)
        completed_rows = [row for row in journal if row["attempt"]["status"] == "complete"]
        if (
            len(journal) != 12
            or [row["attempt"] for row in completed_rows] != payload["selection"]["attempts"]
            or [row["trial"] for row in completed_rows] != payload["selection"]["trials"]
        ):
            raise ValueError("durable attempt journal disagrees with selection result")
        if sha256(request_path.read_bytes()).hexdigest() != REQUEST_SHA256:
            raise ValueError("frozen request changed during validation selection")
        gpu_after = probe._gpu_sample()
    except Exception as error:
        failure: dict[str, Any] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "request_sha256": REQUEST_SHA256,
            "source_hashes": sources,
            "quiet_window": readings,
            "worker_elapsed_seconds": monotonic() - worker_start,
            "attempt_journal": _read_journal(journal_path),
            "final_test_source_constructions": None,
        }
        if "payload" in locals():
            failure["worker"] = payload
        if "completed" in locals():
            failure["worker_stderr"] = completed.stderr[-4000:]
        _save_exclusive(failure_path, failure)
        raise
    report = {
        "schema": "cifar_representative_validation_selection_result_v1",
        "request_sha256": REQUEST_SHA256,
        "source_hashes": sources,
        "quiet_window": readings,
        "worker_elapsed_seconds": monotonic() - worker_start,
        "gpu_after": gpu_after,
        "attempt_journal_sha256": sha256(journal_path.read_bytes()).hexdigest(),
        "selection": payload["selection"],
        "final_test_source_constructions": payload["final_test_source_constructions"],
        "final_test_iterations": payload["final_test_iterations"],
    }
    frozen = _freeze_manifest(request, report, payload["confirmation_manifest"])
    _save_exclusive(result_path, report)
    _save_exclusive(manifest_path, frozen)
    return {"result": report, "manifest": frozen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, default=REQUEST_PATH)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        try:
            request, _ = _verify_saved_request(args.request)
            print(
                json.dumps(
                    _worker_selection(request, args.journal), sort_keys=True, allow_nan=False
                )
            )
        except Exception as error:
            failure: dict[str, Any] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            if isinstance(error, MatchedHeadTuningError):
                failure["attempts"] = [asdict(row) for row in error.attempts]
                failure["trials"] = [asdict(row) for row in error.trials]
            print(json.dumps(failure, sort_keys=True, allow_nan=False))
            raise SystemExit(1) from error
    else:
        outcome = run_selection(args.request)
        print(
            json.dumps(
                {
                    "result": str(RESULT_PATH),
                    "manifest": str(MANIFEST_PATH),
                    "freeze_digest": outcome["manifest"]["freeze_digest"],
                    "attempts": len(outcome["result"]["selection"]["attempts"]),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
