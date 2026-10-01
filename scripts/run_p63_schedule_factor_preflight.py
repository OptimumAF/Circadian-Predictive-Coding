"""Launch and audit the bounded P6.3 matched schedule train-only gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic
from typing import Any

import numpy as np

from scripts import run_p63_sleep_factor_preflight as artifacts
from src.app.continual_schedule_factor_preflight import (
    PROTOCOL_ID,
    fixed_schedule_factor_manifest,
    run_schedule_factor_preflight,
    validate_schedule_factor_manifest,
)
from src.app.continual_schedule_factor_validation import verify_schedule_preflight_payload
from src.shared.process_memory import ProcessRssSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
WALL_LIMIT_SECONDS = 120
ADDITIONAL_SOURCE_SHA256 = {
    "src/app/continual_schedule_factor_preflight.py": "2b7f80dd603ea72d564feae445c8ad3168b258685e2f7a05cda046bac4397bcd",
    "src/app/continual_schedule_factor_validation.py": "739b9a23d268de79f30cec72b395f045ed75a3b745f39af8b762a469caa9374a",
    "src/app/continual_replay_factor_pilot.py": "f472668e0bf49b31465609880f6e2c4a7a104a6b558de99a5ba76886469a8894",
    "src/core/shared_replay_schedule.py": "1961f66262e33a0b73bdaf690336f8e79f6ba5047a8df53cd61c8005f10975d0",
    "src/core/replay_retention.py": "f5c5e9a793ad771c1a7ba0dc29411fb36ceafc34c37ca786cbae2efeeaa1108b",
    "src/core/continual_metrics.py": "8eb491da4221a232a32a7cc1eb7b08a3c36a96fb748543450ac068dac3ae7f78",
    "scripts/run_p63_sleep_factor_preflight.py": "dae68cbdb8ef0464ff9a4971bfaedc2e50dc4d75980693a236a5a92eb38d0d3c",
}


def check_source_hashes() -> dict[str, str]:
    sources = artifacts.check_source_hashes()
    for name, expected in ADDITIONAL_SOURCE_SHA256.items():
        actual = sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"P6.3 schedule frozen source changed: {name}")
        sources[name] = actual
    return sources


def manifest_payload() -> dict[str, Any]:
    return json.loads(json.dumps(asdict(fixed_schedule_factor_manifest()), allow_nan=False))


def verify_result(payload: dict[str, Any]) -> None:
    verify_schedule_preflight_payload(payload, manifest_payload())


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"schedule-factor-preflight.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }


def _work_summary(result: dict[str, Any]) -> dict[str, Any]:
    decisions = [
        decision
        for seed in result["seed_results"]
        for row in seed["opportunities"]
        for decision in row["decisions"]
    ]
    return {
        "executed_optimizer_updates": sum(
            seed["executed_optimizer_updates"] for seed in result["seed_results"]
        ),
        "guarded_attempts": sum(row["attempted"] for row in decisions),
        "accepted_sleep_events": sum(row["outcome"] == "accepted" for row in decisions),
        "rolled_back_events": sum(row["outcome"] == "rolled_back" for row in decisions),
        "applied_replay_updates": sum(
            method["applied_replay_updates"]
            for seed in result["seed_results"]
            for method in seed["methods"]
        ),
        "rejected_executed_replay_updates": sum(
            method["rejected_executed_replay_updates"]
            for seed in result["seed_results"]
            for method in seed["methods"]
        ),
    }


def run_bounded_preflight(output_dir: Path) -> dict[str, Any]:
    sources = check_source_hashes()
    manifest = fixed_schedule_factor_manifest()
    maximum = validate_schedule_factor_manifest(manifest)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.3 schedule output already exists: {occupied}")
    command = [sys.executable, "-m", "scripts.run_p63_schedule_factor_preflight", "--worker"]
    resolved = manifest_payload()
    request = {
        "schema_id": "p63_schedule_factor_preflight_request_v1",
        "manifest": resolved,
        "manifest_sha256": artifacts.digest_json(resolved),
        "source_sha256": sources,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_wake_optimizer_updates": 792,
        "maximum_executed_optimizer_updates": maximum,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "max_process_rss_bytes": manifest.max_process_rss_bytes,
        "command": command,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifacts.write_exclusive(paths["request"], request)
    started = monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=WALL_LIMIT_SECONDS,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"worker exited {process.returncode}: {process.stderr}")
        payload = artifacts.parse_finite_json(process.stdout)
        verify_result(payload["result"])
        artifacts.verify_memory(payload["process_rss"], manifest.max_process_rss_bytes)
        artifacts.write_exclusive(paths["result"], payload["result"])
    except (Exception, subprocess.TimeoutExpired) as exc:
        artifacts.write_exclusive(
            paths["failure"],
            {
                "schema_id": "p63_schedule_factor_preflight_failure_v1",
                "reason": "wall_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "worker_or_audit",
                "error": str(exc),
                "elapsed_seconds": monotonic() - started,
            },
        )
        raise
    audit = {
        "schema_id": "p63_schedule_factor_preflight_audit_v1",
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "seed_count": len(manifest.seeds),
        "cell_count": len(manifest.seeds) * len(manifest.arms),
        "decision_count": 3 * 24 * 3,
        "work": _work_summary(payload["result"]),
        "process_rss": payload["process_rss"],
        "request_sha256": sha256(paths["request"].read_bytes()).hexdigest(),
        "result_sha256": sha256(paths["result"].read_bytes()).hexdigest(),
        "elapsed_seconds": monotonic() - started,
    }
    artifacts.write_exclusive(paths["audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p63-schedule-factor-preflight")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        check_source_hashes()
        manifest = fixed_schedule_factor_manifest()
        with ProcessRssSampler(interval_seconds=0.005) as sampler:
            result = run_schedule_factor_preflight(manifest)
        memory = sampler.snapshot()
        if memory.peak_bytes > manifest.max_process_rss_bytes:
            raise ValueError("P6.3 schedule observed worker RSS exceeded cap")
        print(
            json.dumps(
                {"result": asdict(result), "process_rss": asdict(memory)},
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    print(json.dumps(run_bounded_preflight(options.output_dir), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
