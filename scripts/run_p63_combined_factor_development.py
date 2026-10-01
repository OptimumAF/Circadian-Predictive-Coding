"""Launch and audit bounded combined/full-minus-one development scoring."""

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

from scripts import run_p63_combined_factor_preflight as c7_adapter
from scripts import run_p63_sleep_factor_preflight as artifacts
from src.app.continual_combined_factor_development import (
    ARMS,
    CONTRAST_FIELDS,
    CONTRASTS,
    PROTOCOL_ID,
    REFERENCE_SHA256,
    SCORE_FIELDS,
    run_combined_factor_development,
)
from src.app.continual_combined_factor_manifest import (
    DEVELOPMENT_SEEDS,
    PARITY_PAIRS,
    fixed_combined_manifest,
    validate_combined_manifest,
)
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID, TwoTaskAccuracy
from src.shared.process_memory import ProcessRssSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "artifacts/runs/p63-combined-factor-preflight"
WALL_LIMIT_SECONDS = 120
OUTER_EVALUATIONS = 153
OUTER_EXAMPLES = 3060
ADDITIONAL_SOURCE_SHA256 = {
    "src/app/continual_combined_factor_development.py": "be4c0156c8921a9f0d4580e93695bd4464c1d6241a9bfa7c3fd531eb199994fa",
    "src/app/continual_sleep_factor_development.py": "e6a498c107d0071b822309907691e225b1a1ae828c93f8435c94ba0cfa94bf88",
    "scripts/run_p63_combined_factor_preflight.py": "9c3d640638b48d7a2edc9cc4585e205cdb3f2d86bd1c72a6e7a8438bd0f3d404",
}


def check_source_hashes() -> dict[str, str]:
    sources = c7_adapter.check_source_hashes()
    for name, expected in ADDITIONAL_SOURCE_SHA256.items():
        actual = sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"P6.3 combined development frozen source changed: {name}")
        sources[name] = actual
    return sources


def read_reference() -> dict[str, Any]:
    paths = c7_adapter.artifact_paths(REFERENCE_DIR)
    if paths["failure"].exists() or any(
        not paths[name].is_file() for name in ("request", "result", "audit")
    ):
        raise ValueError("P6.3 combined development lacks a complete c7 reference")
    if sha256(paths["result"].read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("P6.3 combined development c7 result bytes differ")
    request = artifacts.parse_finite_json(paths["request"].read_text(encoding="utf-8"))
    result = artifacts.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = artifacts.parse_finite_json(paths["audit"].read_text(encoding="utf-8"))
    c7_adapter.verify_result(result)
    manifest = c7_adapter.manifest_payload()
    if (
        request["manifest"] != manifest
        or request["manifest_sha256"] != artifacts.digest_json(manifest)
        or request["source_sha256"] != c7_adapter.check_source_hashes()
        or request["adapter_sha256"] != sha256(Path(c7_adapter.__file__).read_bytes()).hexdigest()
        or request["planned_wake_optimizer_updates"] != 1224
        or request["maximum_executed_optimizer_updates"] != 1548
        or request["planned_guarded_attempts"] != 144
        or request["planned_guard_evaluations"] != 288
        or request["wall_limit_seconds"] != WALL_LIMIT_SECONDS
        or request["max_process_rss_bytes"] != manifest["max_process_rss_bytes"]
        or audit["status"] != "completed"
        or audit["protocol_id"] != manifest["protocol_id"]
        or audit["seed_count"] != 3
        or audit["cell_count"] != 51
        or audit["decision_count"] != 648
        or audit["work"] != c7_adapter._work_summary(result)
        or not 0 <= audit["elapsed_seconds"] <= WALL_LIMIT_SECONDS
        or audit["request_sha256"] != sha256(paths["request"].read_bytes()).hexdigest()
        or audit["result_sha256"] != REFERENCE_SHA256
    ):
        raise ValueError("P6.3 combined development c7 request or audit differs")
    artifacts.verify_memory(audit["process_rss"], manifest["max_process_rss_bytes"])
    return result


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"combined-factor-development.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }


def verify_result(result: dict[str, Any], reference: dict[str, Any]) -> None:
    c7_adapter.verify_result(reference)
    if (
        result["protocol_id"] != PROTOCOL_ID
        or result["metric_contract_id"] != TWO_TASK_METRIC_CONTRACT_ID
        or result["reference_sha256"] != REFERENCE_SHA256
        or result["train_facts"] != reference
        or result["outer_selection_scored"] is not True
        or result["final_released"] is not False
        or [seed["seed"] for seed in result["scored_seeds"]] != list(DEVELOPMENT_SEEDS)
    ):
        raise ValueError("P6.3 combined development protocol or global train gate differs")
    for seed in result["scored_seeds"]:
        arms = seed["arms"]
        if [arm["name"] for arm in arms] != list(ARMS):
            raise ValueError("P6.3 combined development model cells differ")
        by_name = {arm["name"]: arm for arm in arms}
        for arm in arms:
            values = TwoTaskAccuracy(arm["a_after_a"], arm["a_after_b"], arm["b_after_b"])
            if (
                arm["final_mean_task_accuracy"] != values.final_mean_task_accuracy
                or arm["signed_forgetting_a"] != values.signed_forgetting_a
                or arm["retention_ratio_a"] != values.retention_ratio_a
            ):
                raise ValueError("P6.3 combined development derived metrics differ")
        for left_name, right_name in PARITY_PAIRS:
            if any(
                by_name[left_name][field] != by_name[right_name][field] for field in SCORE_FIELDS
            ):
                raise ValueError("P6.3 combined development neutral PC outcome parity differs")
        if [(row["left"], row["right"]) for row in seed["contrasts"]] != list(CONTRASTS):
            raise ValueError("P6.3 combined development paired contrasts differ")
        for row in seed["contrasts"]:
            left, right = by_name[row["left"]], by_name[row["right"]]
            for field in CONTRAST_FIELDS:
                if row[field] != left[field] - right[field]:
                    raise ValueError(f"P6.3 combined development contrast differs: {field}")


def run_bounded_development(output_dir: Path) -> dict[str, Any]:
    sources = check_source_hashes()
    reference = read_reference()
    manifest = fixed_combined_manifest()
    maximum = validate_combined_manifest(manifest)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.3 combined development output already exists: {occupied}")
    command = [sys.executable, "-m", "scripts.run_p63_combined_factor_development", "--worker"]
    resolved = c7_adapter.manifest_payload()
    request = {
        "schema_id": "p63_combined_factor_development_request_v1",
        "protocol_id": PROTOCOL_ID,
        "metric_contract_id": TWO_TASK_METRIC_CONTRACT_ID,
        "manifest": resolved,
        "manifest_sha256": artifacts.digest_json(resolved),
        "source_sha256": sources,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": REFERENCE_SHA256,
        "paired_contrasts": CONTRASTS,
        "planned_wake_optimizer_updates": 1224,
        "maximum_executed_optimizer_updates": maximum,
        "planned_guarded_attempts": 144,
        "planned_guard_evaluations": 288,
        "planned_outer_evaluations": OUTER_EVALUATIONS,
        "planned_outer_examples": OUTER_EXAMPLES,
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
        verify_result(payload["result"], reference)
        artifacts.verify_memory(payload["process_rss"], manifest.max_process_rss_bytes)
        artifacts.write_exclusive(paths["result"], payload["result"])
    except Exception as exc:
        artifacts.write_exclusive(
            paths["failure"],
            {
                "schema_id": "p63_combined_factor_development_failure_v1",
                "reason": "wall_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "worker_or_audit",
                "error": str(exc),
                "elapsed_seconds": monotonic() - started,
            },
        )
        raise
    audit = {
        "schema_id": "p63_combined_factor_development_audit_v1",
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "seed_count": len(DEVELOPMENT_SEEDS),
        "cell_count": len(DEVELOPMENT_SEEDS) * len(ARMS),
        "decision_count": 648,
        "outer_evaluations": OUTER_EVALUATIONS,
        "outer_examples": OUTER_EXAMPLES,
        "contrast_count": len(DEVELOPMENT_SEEDS) * len(CONTRASTS),
        "reference_sha256": REFERENCE_SHA256,
        "work": c7_adapter._work_summary(payload["result"]["train_facts"]),
        "process_rss": payload["process_rss"],
        "request_sha256": sha256(paths["request"].read_bytes()).hexdigest(),
        "result_sha256": sha256(paths["result"].read_bytes()).hexdigest(),
        "elapsed_seconds": monotonic() - started,
    }
    artifacts.write_exclusive(paths["audit"], audit)
    return audit


def _run_worker() -> None:
    check_source_hashes()
    reference = read_reference()
    manifest = fixed_combined_manifest()
    with ProcessRssSampler(interval_seconds=0.005) as sampler:
        result = run_combined_factor_development(manifest, reference)
        # Why this: include retained copies, serialization and output validation in RSS.
        result_json = json.dumps(asdict(result), sort_keys=True, allow_nan=False)
        verify_result(artifacts.parse_finite_json(result_json), reference)
    memory = sampler.snapshot()
    if memory.peak_bytes > manifest.max_process_rss_bytes:
        raise ValueError("P6.3 combined development observed worker RSS exceeded cap")
    sys.stdout.write('{"result":')
    sys.stdout.write(result_json)
    sys.stdout.write(',"process_rss":')
    sys.stdout.write(json.dumps(asdict(memory), sort_keys=True, allow_nan=False))
    sys.stdout.write("}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p63-combined-factor-development")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        _run_worker()
        return
    print(json.dumps(run_bounded_development(options.output_dir), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
