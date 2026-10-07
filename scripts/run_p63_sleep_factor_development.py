"""Launch and audit the bounded P6.3 sleep-factor outer-development score."""

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

from scripts import run_p63_sleep_factor_preflight as c3_adapter
from src.app.continual_sleep_factor_development import (
    CONTRASTS,
    PROTOCOL_ID,
    REFERENCE_SHA256,
    run_sleep_factor_development,
)
from src.app.continual_sleep_factor_preflight import (
    ARMS,
    DEVELOPMENT_SEEDS,
    fixed_sleep_factor_manifest,
    validate_sleep_factor_manifest,
)
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID, TwoTaskAccuracy
from src.shared.process_memory import ProcessRssSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "artifacts/runs/p63-sleep-factor-preflight"
WALL_LIMIT_SECONDS = 120
# Freeze these before the first public scored run; c3_adapter pins its 13 sources.
ADDITIONAL_SOURCE_SHA256 = {
    "src/app/continual_sleep_factor_development.py": "e6a498c107d0071b822309907691e225b1a1ae828c93f8435c94ba0cfa94bf88",
    "src/core/continual_metrics.py": "8eb491da4221a232a32a7cc1eb7b08a3c36a96fb748543450ac068dac3ae7f78",
    "scripts/run_p63_sleep_factor_preflight.py": "dae68cbdb8ef0464ff9a4971bfaedc2e50dc4d75980693a236a5a92eb38d0d3c",
}


def check_source_hashes() -> dict[str, str]:
    sources = c3_adapter.check_source_hashes()
    for name, expected in ADDITIONAL_SOURCE_SHA256.items():
        observed = sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"P6.3 scored development frozen source changed: {name}")
        sources[name] = observed
    return sources


def read_reference() -> dict[str, Any]:
    paths = c3_adapter.artifact_paths(REFERENCE_DIR)
    if paths["failure"].exists() or any(
        not paths[name].is_file() for name in ("request", "result", "audit")
    ):
        raise ValueError("P6.3 scored development lacks a complete c3 reference")
    if sha256(paths["result"].read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("P6.3 scored development c3 result bytes differ")
    request = c3_adapter.parse_finite_json(paths["request"].read_text(encoding="utf-8"))
    result = c3_adapter.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = c3_adapter.parse_finite_json(paths["audit"].read_text(encoding="utf-8"))
    c3_adapter.verify_result(result)
    if (
        request["manifest"] != c3_adapter.manifest_payload()
        or request["manifest_sha256"] != c3_adapter.digest_json(request["manifest"])
        or request["source_sha256"] != c3_adapter.check_source_hashes()
        or request["adapter_sha256"]
        != sha256(
            (REPO_ROOT / "scripts/run_p63_sleep_factor_preflight.py").read_bytes()
        ).hexdigest()
        or audit["status"] != "completed"
        or audit["request_sha256"] != sha256(paths["request"].read_bytes()).hexdigest()
        or audit["result_sha256"] != REFERENCE_SHA256
    ):
        raise ValueError("P6.3 scored development c3 request or audit differs")
    return result


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"sleep-factor-development.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }


def verify_result(result: dict[str, Any], reference: dict[str, Any]) -> None:
    if (
        result["protocol_id"] != PROTOCOL_ID
        or result["metric_contract_id"] != TWO_TASK_METRIC_CONTRACT_ID
        or result["reference_sha256"] != REFERENCE_SHA256
        or result["train_facts"] != reference
        or result["outer_selection_scored"] is not True
        or result["final_released"] is not False
    ):
        raise ValueError("P6.3 scored development protocol or global train gate differs")
    scored = result["scored_seeds"]
    if [seed["seed"] for seed in scored] != list(DEVELOPMENT_SEEDS):
        raise ValueError("P6.3 scored development seed cells differ")
    for seed in scored:
        arms = seed["arms"]
        if [arm["name"] for arm in arms] != list(ARMS):
            raise ValueError("P6.3 scored development arm cells differ")
        by_name = {arm["name"]: arm for arm in arms}
        for arm in arms:
            _verify_metrics(arm)
        if any(
            by_name["pc_8"][field] != by_name["neutral_sham"][field]
            for field in (
                "a_after_a",
                "a_after_b",
                "b_after_b",
                "final_mean_task_accuracy",
                "signed_forgetting_a",
                "retention_ratio_a",
            )
        ):
            raise ValueError("P6.3 scored development ordinary PC/sham parity differs")
        contrasts = seed["contrasts"]
        if [(row["left"], row["right"]) for row in contrasts] != list(CONTRASTS):
            raise ValueError("P6.3 scored development paired contrasts differ")
        for row in contrasts:
            left, right = by_name[row["left"]], by_name[row["right"]]
            for field in (
                "a_after_a",
                "a_after_b",
                "b_after_b",
                "final_mean_task_accuracy",
                "signed_forgetting_a",
            ):
                if row[field] != left[field] - right[field]:
                    raise ValueError(f"P6.3 scored development contrast differs: {field}")


def _verify_metrics(arm: dict[str, Any]) -> None:
    metrics = TwoTaskAccuracy(arm["a_after_a"], arm["a_after_b"], arm["b_after_b"])
    if (
        arm["final_mean_task_accuracy"] != metrics.final_mean_task_accuracy
        or arm["signed_forgetting_a"] != metrics.signed_forgetting_a
        or arm["retention_ratio_a"] != metrics.retention_ratio_a
    ):
        raise ValueError(f"P6.3 scored development derived metrics differ: {arm['name']}")


def run_bounded_development(output_dir: Path) -> dict[str, Any]:
    sources = check_source_hashes()
    reference = read_reference()
    manifest = fixed_sleep_factor_manifest()
    planned = validate_sleep_factor_manifest(manifest)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.3 scored development output already exists: {occupied}")
    command = [sys.executable, "-m", "scripts.run_p63_sleep_factor_development", "--worker"]
    request = {
        "schema_id": "p63_sleep_factor_development_request_v1",
        "protocol_id": PROTOCOL_ID,
        "metric_contract_id": TWO_TASK_METRIC_CONTRACT_ID,
        "manifest": c3_adapter.manifest_payload(),
        "manifest_sha256": c3_adapter.digest_json(c3_adapter.manifest_payload()),
        "source_sha256": sources,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": REFERENCE_SHA256,
        "planned_optimizer_updates": planned,
        "planned_outer_evaluations": len(DEVELOPMENT_SEEDS) * len(ARMS) * 3,
        "planned_outer_examples": len(DEVELOPMENT_SEEDS) * len(ARMS) * (24 + 24 + 12),
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "max_process_rss_bytes": manifest.max_process_rss_bytes,
        "command": command,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    c3_adapter.write_exclusive(paths["request"], request)
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
        payload = c3_adapter.parse_finite_json(process.stdout)
        verify_result(payload["result"], reference)
        c3_adapter.verify_memory(payload["process_rss"], manifest.max_process_rss_bytes)
        c3_adapter.write_exclusive(paths["result"], payload["result"])
    except (Exception, subprocess.TimeoutExpired) as exc:
        c3_adapter.write_exclusive(
            paths["failure"],
            {
                "schema_id": "p63_sleep_factor_development_failure_v1",
                "reason": "wall_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "worker_or_audit",
                "error": str(exc),
                "elapsed_seconds": monotonic() - started,
            },
        )
        raise
    audit = {
        "schema_id": "p63_sleep_factor_development_audit_v1",
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "seed_count": len(DEVELOPMENT_SEEDS),
        "cell_count": len(DEVELOPMENT_SEEDS) * len(ARMS),
        "outer_evaluations": len(DEVELOPMENT_SEEDS) * len(ARMS) * 3,
        "outer_examples": len(DEVELOPMENT_SEEDS) * len(ARMS) * (24 + 24 + 12),
        "planned_optimizer_updates": planned,
        "reference_sha256": REFERENCE_SHA256,
        "process_rss": payload["process_rss"],
        "request_sha256": sha256(paths["request"].read_bytes()).hexdigest(),
        "result_sha256": sha256(paths["result"].read_bytes()).hexdigest(),
        "elapsed_seconds": monotonic() - started,
    }
    c3_adapter.write_exclusive(paths["audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p63-sleep-factor-development")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        check_source_hashes()
        reference = read_reference()
        manifest = fixed_sleep_factor_manifest()
        with ProcessRssSampler(interval_seconds=0.005) as sampler:
            result = run_sleep_factor_development(manifest, reference)
        memory = sampler.snapshot()
        if memory.peak_bytes > manifest.max_process_rss_bytes:
            raise ValueError("P6.3 scored development observed process RSS exceeded cap")
        print(
            json.dumps(
                {"result": asdict(result), "process_rss": asdict(memory)},
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    audit = run_bounded_development(options.output_dir)
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
