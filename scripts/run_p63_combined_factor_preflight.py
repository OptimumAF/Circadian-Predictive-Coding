"""Launch and audit bounded combined/full-minus-one train-only facts."""

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

from scripts import run_p63_schedule_factor_preflight as c5_adapter
from scripts import run_p63_sleep_factor_preflight as artifacts
from src.app.continual_combined_factor_manifest import (
    PROTOCOL_ID,
    fixed_combined_manifest,
    validate_combined_manifest,
)
from src.app.continual_combined_factor_preflight import json_value, run_combined_preflight
from src.app.continual_combined_factor_validation import verify_combined_payload
from src.shared.process_memory import ProcessRssSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
WALL_LIMIT_SECONDS = 120
ADDITIONAL_SOURCE_SHA256 = {
    "src/app/continual_combined_factor_manifest.py": "213d239743617c7aefe6c1848049c9ac8160c26e3c741b91807524b292a75000",
    "src/app/continual_combined_factor_preflight.py": "f19a3e33df1bdcf25c429d10f3da0e0c4edab361fe9efe513a150750363e3a78",
    "src/app/continual_combined_factor_validation.py": "3bc221199a9036ceb234621d16166a8682f209fc50caf033bcc3fad1b2004d10",
    "scripts/run_p63_schedule_factor_preflight.py": "4490f9201b565d3283434da786799cc3d9ac8242218cfc8f9b881de9011b7fbe",
    "src/core/sleep_clocks.py": "c08c3e8bc662f4136a853f9a19e8f0602f1433d119ff58321223a9a47f87a598",
    "src/core/neuron_adaptation.py": "a119a2dcf27fe3682fcef12b387db5dd0f1bd831eea11d0583cf823a89602493",
}


def check_source_hashes() -> dict[str, str]:
    sources = c5_adapter.check_source_hashes()
    for name, expected in ADDITIONAL_SOURCE_SHA256.items():
        actual = sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"P6.3 combined frozen source changed: {name}")
        sources[name] = actual
    return sources


def manifest_payload() -> dict[str, Any]:
    return json_value(fixed_combined_manifest())


def verify_result(payload: dict[str, Any]) -> None:
    verify_combined_payload(payload, manifest_payload())


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"combined-factor-preflight.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }


def _work_summary(result: dict[str, Any]) -> dict[str, int]:
    decisions = [
        row["event"]
        for seed in result["seed_results"]
        for opportunity in seed["opportunities"]
        for row in opportunity["decisions"]
    ]
    methods = [row for seed in result["seed_results"] for row in seed["methods"]]
    return {
        "executed_optimizer_updates": sum(
            seed["executed_optimizer_updates"] for seed in result["seed_results"]
        ),
        "wake_optimizer_updates": sum(row["wake_updates"] for row in methods),
        "guarded_attempts": sum(row["guard"] is not None for row in decisions),
        "guard_evaluations": sum(row["guard_evaluations"] for row in methods),
        "guard_examples": sum(row["guard_examples"] for row in methods),
        "accepted_own_sleep_events": sum(row["outcome"] == "accepted" for row in decisions),
        "rolled_back_events": sum(row["outcome"] == "rolled_back" for row in decisions),
        "applied_replay_updates": sum(row["applied_replay_updates"] for row in methods),
        "rejected_executed_replay_updates": sum(
            row["rejected_executed_replay_updates"] for row in methods
        ),
        "proposed_splits": sum(len(row["changes"]["proposed_split_pairs"]) for row in decisions),
        "applied_splits": sum(len(row["changes"]["applied_split_pairs"]) for row in decisions),
        "proposed_removed_prunes": sum(
            len(row["changes"]["proposed_removed_prune_ids"]) for row in decisions
        ),
        "applied_removed_prunes": sum(
            len(row["changes"]["applied_removed_prune_ids"]) for row in decisions
        ),
    }


def run_bounded_preflight(output_dir: Path) -> dict[str, Any]:
    sources = check_source_hashes()
    manifest = fixed_combined_manifest()
    maximum = validate_combined_manifest(manifest)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.3 combined output already exists: {occupied}")
    command = [sys.executable, "-m", "scripts.run_p63_combined_factor_preflight", "--worker"]
    resolved = manifest_payload()
    request = {
        "schema_id": "p63_combined_factor_preflight_request_v1",
        "manifest": resolved,
        "manifest_sha256": artifacts.digest_json(resolved),
        "source_sha256": sources,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_wake_optimizer_updates": 1224,
        "maximum_executed_optimizer_updates": maximum,
        "planned_guarded_attempts": 144,
        "planned_guard_evaluations": 288,
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
    except Exception as exc:
        artifacts.write_exclusive(
            paths["failure"],
            {
                "schema_id": "p63_combined_factor_preflight_failure_v1",
                "reason": "wall_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "worker_or_audit",
                "error": str(exc),
                "elapsed_seconds": monotonic() - started,
            },
        )
        raise
    audit = {
        "schema_id": "p63_combined_factor_preflight_audit_v1",
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "seed_count": len(manifest.seeds),
        "cell_count": len(manifest.seeds) * len(manifest.arms),
        "decision_count": 648,
        "work": _work_summary(payload["result"]),
        "process_rss": payload["process_rss"],
        "request_sha256": sha256(paths["request"].read_bytes()).hexdigest(),
        "result_sha256": sha256(paths["result"].read_bytes()).hexdigest(),
        "elapsed_seconds": monotonic() - started,
    }
    artifacts.write_exclusive(paths["audit"], audit)
    return audit


def _run_worker() -> None:
    check_source_hashes()
    manifest = fixed_combined_manifest()
    with ProcessRssSampler(interval_seconds=0.005) as sampler:
        result = run_combined_preflight(manifest)
        # Why this: include result serialization allocations in sampled RSS.
        result_json = json.dumps(asdict(result), sort_keys=True, allow_nan=False)
    memory = sampler.snapshot()
    if memory.peak_bytes > manifest.max_process_rss_bytes:
        raise ValueError("P6.3 combined observed worker RSS exceeded cap")
    sys.stdout.write('{"result":')
    sys.stdout.write(result_json)
    sys.stdout.write(',"process_rss":')
    sys.stdout.write(json.dumps(asdict(memory), sort_keys=True, allow_nan=False))
    sys.stdout.write("}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p63-combined-factor-preflight")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        _run_worker()
        return
    print(json.dumps(run_bounded_preflight(options.output_dir), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
