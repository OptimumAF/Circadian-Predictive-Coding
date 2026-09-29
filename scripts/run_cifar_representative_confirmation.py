"""Run the frozen CIFAR matched confirmation in separately capped scopes.

The parent restores exact selection evidence, measures a quiet CUDA window,
then launches one bounded child per declared scope. Each child restores the
same freeze before any dataset construction and saves partial seed results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import profile_cifar_representative_feasibility as probe  # noqa: E402
from scripts import run_cifar_representative_selection as selection  # noqa: E402
from scripts.restore_cifar_representative_selection import (  # noqa: E402
    RestoredRepresentativeSelection,
    read_saved_selection,
)
from src.app import isolated_head_memory as isolated  # noqa: E402
from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app import matched_head_tuning as tuning  # noqa: E402
from src.app import repeated_head_confirmation as repeated  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
PREFIX = "cifar-representative-confirmation-v1"
GATE_NAME = f"{PREFIX}-gate.json"
RESULT_NAME = f"{PREFIX}-result.json"
FAILURE_NAME = f"{PREFIX}-failure.json"
FIXED_NAME = f"{PREFIX}-fixed-data.json"
ATTEMPTS_NAME = f"{PREFIX}-fixed-data-attempts.jsonl"
SCOPES = ("fixed_data", "wall_time", "capacity_memory")
MEMORY_CHILD_LIMIT_SECONDS = 60.0


def _save_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _seed_path(data_dir: Path, scope: str, seed: int) -> Path:
    return data_dir / f"{PREFIX}-{scope}-seed{seed}.json"


def _artifact_paths(data_dir: Path, seeds: tuple[int, ...]) -> tuple[Path, ...]:
    return (
        data_dir / GATE_NAME,
        data_dir / RESULT_NAME,
        data_dir / FAILURE_NAME,
        data_dir / FIXED_NAME,
        data_dir / ATTEMPTS_NAME,
        *(_seed_path(data_dir, "wall-time", seed) for seed in seeds),
        *(_seed_path(data_dir, "memory", seed) for seed in seeds),
    )


def _quiet_enough(readings: list[dict[str, Any]], request: dict[str, Any]) -> bool:
    gate = request["quiet_window"]
    return len(readings) == gate["readings"] and all(
        row["utilization_percent"] <= gate["max_utilization_percent"]
        and row["free_mib"] >= gate["min_free_mib"]
        for row in readings
    )


def _scope_timeout(request: dict[str, Any], scope: str, started: float, now: float) -> float:
    remaining = request["confirmation_total_limit_seconds"] - (now - started)
    if remaining <= 0:
        raise TimeoutError("total confirmation budget expired before next scope")
    return min(float(request["scopes"][scope]["limit_seconds"]), remaining)


def _verify_gate(data_dir: Path, restored: RestoredRepresentativeSelection) -> dict[str, Any]:
    path = data_dir / GATE_NAME
    if not path.is_file():
        raise FileNotFoundError("representative confirmation quiet gate is missing")
    gate: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "quiet"
        or gate.get("request_sha256") != selection.REQUEST_SHA256
        or gate.get("freeze_digest") != restored.freeze["freeze_digest"]
        or gate.get("runner_sha256") != sha256(Path(__file__).read_bytes()).hexdigest()
        or not _quiet_enough(gate.get("readings", []), restored.request)
    ):
        raise ValueError("representative confirmation quiet gate changed")
    return gate


def _scope_artifact(
    restored: RestoredRepresentativeSelection,
    scope: str,
    seed: int | None,
    report: Any,
) -> dict[str, Any]:
    return {
        "schema": "cifar_representative_confirmation_scope_v1",
        "scope": scope,
        "seed": seed,
        "request_sha256": selection.REQUEST_SHA256,
        "freeze_digest": restored.freeze["freeze_digest"],
        "manifest_digest": restored.confirmation_manifest.manifest_digest,
        "report": asdict(report),
    }


def _run_fixed_data(data_dir: Path, restored: RestoredRepresentativeSelection) -> tuple[Path, ...]:
    manifest = restored.confirmation_manifest
    journal = data_dir / ATTEMPTS_NAME
    result = tuning.run_matched_head_tuning(
        manifest.base_config,
        repeated._candidate_map(manifest.selected_heads),
        seeds=manifest.confirmation_seeds,
        candidates_per_head=1,
        attempt_observer=lambda attempt, trial: selection._append_journal(journal, attempt, trial),
    )
    repeated._verify_fixed_data(manifest, result)
    path = data_dir / FIXED_NAME
    _save_exclusive(path, _scope_artifact(restored, "fixed_data", None, result))
    return (path,)


def _run_wall_time(data_dir: Path, restored: RestoredRepresentativeSelection) -> tuple[Path, ...]:
    manifest = restored.confirmation_manifest
    combined = repeated._combine_selected_config(manifest)
    saved = []
    for seed in manifest.confirmation_seeds:
        config = replace(combined, seed=seed, epochs=manifest.wall_time_epoch_cap)
        report = matched.run_three_head_fixed_feature_wall_time_benchmark(
            config, wall_time_budget_seconds=manifest.wall_time_budget_seconds
        )
        path = _seed_path(data_dir, "wall-time", seed)
        _save_exclusive(path, _scope_artifact(restored, "wall_time", seed, report))
        saved.append(path)
    return tuple(saved)


def _run_capacity_memory(
    data_dir: Path, restored: RestoredRepresentativeSelection
) -> tuple[Path, ...]:
    manifest = restored.confirmation_manifest
    combined = repeated._combine_selected_config(manifest)
    saved = []
    for seed in manifest.confirmation_seeds:
        report = isolated.run_process_isolated_fixed_width_memory(
            replace(combined, seed=seed), timeout_seconds=MEMORY_CHILD_LIMIT_SECONDS
        )
        path = _seed_path(data_dir, "memory", seed)
        _save_exclusive(path, _scope_artifact(restored, "capacity_memory", seed, report))
        saved.append(path)
    return tuple(saved)


def _run_worker(scope: str, data_dir: Path) -> None:
    restored = read_saved_selection()
    selection._verify_cuda_runtime()
    _verify_gate(data_dir, restored)
    if scope == "fixed_data":
        saved = _run_fixed_data(data_dir, restored)
    elif scope == "wall_time":
        saved = _run_wall_time(data_dir, restored)
    elif scope == "capacity_memory":
        saved = _run_capacity_memory(data_dir, restored)
    else:
        raise ValueError(f"Unknown representative confirmation scope: {scope}")
    print(json.dumps({"scope": scope, "saved": [str(path) for path in saved]}, sort_keys=True))


def _launch_scope(scope: str, data_dir: Path, timeout: float) -> str:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "scripts.run_cifar_representative_confirmation",
            "worker",
            "--scope",
            scope,
            "--output-dir",
            str(data_dir),
        ),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{scope} worker failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    return completed.stdout.strip()


def _available_artifacts(data_dir: Path, seeds: tuple[int, ...]) -> dict[str, str]:
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in _artifact_paths(data_dir, seeds)
        if path.is_file() and path.name not in {RESULT_NAME, FAILURE_NAME}
    }


def run_confirmation(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Run only the three predeclared scopes after exact and quiet gates."""
    restored = read_saved_selection()
    seeds = restored.confirmation_manifest.confirmation_seeds
    if any(path.exists() for path in _artifact_paths(data_dir, seeds)):
        raise FileExistsError("representative confirmation already has an artifact")
    selection._verify_cuda_runtime()
    readings = probe._quiet_window()
    if not _quiet_enough(readings, restored.request):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        deferred = data_dir / f"{PREFIX}-deferred-{timestamp}.json"
        _save_exclusive(
            deferred,
            {
                "status": "deferred_busy_gpu",
                "request_sha256": selection.REQUEST_SHA256,
                "freeze_digest": restored.freeze["freeze_digest"],
                "readings": readings,
                "final_test_iterations": 0,
            },
        )
        raise RuntimeError("representative confirmation quiet CUDA gate was not met")
    gate = {
        "status": "quiet",
        "request_sha256": selection.REQUEST_SHA256,
        "freeze_digest": restored.freeze["freeze_digest"],
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "readings": readings,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    _save_exclusive(data_dir / GATE_NAME, gate)
    started = monotonic()
    elapsed_by_scope: dict[str, float] = {}
    completed_output: dict[str, str] = {}
    active_scope = "preflight"
    try:
        from scripts import audit_cifar_representative_confirmation as audit

        fixed_result: tuning.MatchedHeadTuningResult | None = None
        for active_scope in SCOPES:
            timeout = _scope_timeout(restored.request, active_scope, started, monotonic())
            scope_start = monotonic()
            completed_output[active_scope] = _launch_scope(active_scope, data_dir, timeout)
            elapsed_by_scope[active_scope] = monotonic() - scope_start
            if (
                elapsed_by_scope[active_scope]
                > restored.request["scopes"][active_scope]["limit_seconds"]
            ):
                raise TimeoutError(f"{active_scope} exceeded its frozen scope budget")
            if active_scope == "fixed_data":
                fixed_result = audit.audit_fixed_scope(data_dir, restored)
            elif active_scope == "wall_time":
                if fixed_result is None:
                    raise AssertionError("fixed-data audit did not complete")
                audit.audit_wall_scope(data_dir, restored, fixed_result)
            else:
                if fixed_result is None:
                    raise AssertionError("fixed-data audit did not complete")
                audit.audit_memory_scope(data_dir, restored, fixed_result)
        if monotonic() - started > restored.request["confirmation_total_limit_seconds"]:
            raise TimeoutError("total confirmation budget exceeded")
        report = audit.audit_saved_scopes(data_dir, restored)
        result = {
            "schema": "cifar_representative_confirmation_result_v1",
            "request_sha256": selection.REQUEST_SHA256,
            "freeze_digest": restored.freeze["freeze_digest"],
            "manifest_digest": restored.confirmation_manifest.manifest_digest,
            "runner_sha256": gate["runner_sha256"],
            "quiet_window": readings,
            "gpu_after": probe._gpu_sample(),
            "scope_elapsed_seconds": elapsed_by_scope,
            "total_elapsed_seconds": monotonic() - started,
            "scope_artifact_sha256": _available_artifacts(data_dir, seeds),
            "result": asdict(report),
        }
        if result["total_elapsed_seconds"] > restored.request["confirmation_total_limit_seconds"]:
            raise TimeoutError("total confirmation budget exceeded during audit")
        _save_exclusive(data_dir / RESULT_NAME, result)
        return result
    except Exception as error:
        _save_exclusive(
            data_dir / FAILURE_NAME,
            {
                "status": "failed",
                "scope": active_scope,
                "error_type": type(error).__name__,
                "error": str(error),
                "request_sha256": selection.REQUEST_SHA256,
                "freeze_digest": restored.freeze["freeze_digest"],
                "quiet_window": readings,
                "scope_elapsed_seconds": elapsed_by_scope,
                "total_elapsed_seconds": monotonic() - started,
                "completed_worker_output": completed_output,
                "available_artifact_sha256": _available_artifacts(data_dir, seeds),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--scope", choices=SCOPES, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        _run_worker(args.scope, args.output_dir)
    else:
        result = run_confirmation()
        print(
            json.dumps(
                {
                    "result": str(DATA_DIR / RESULT_NAME),
                    "total_elapsed_seconds": result["total_elapsed_seconds"],
                    "manifest_digest": result["manifest_digest"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
