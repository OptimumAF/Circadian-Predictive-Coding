"""Launch and audit the bounded P6.3 development-only gating pilot.

Inputs: the frozen pilot manifest and a fresh local output directory.
Outputs: exclusive request/result/audit or failure files. The worker never
opens final roles; this adapter does not select a treatment from scores.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import monotonic
from typing import Any

import numpy as np

from src.app.continual_gating_pilot import (
    ARMS,
    DEVELOPMENT_SEEDS,
    PROTOCOL_ID,
    fixed_gating_pilot_manifest,
    run_gating_pilot,
    validate_gating_pilot_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WALL_LIMIT_SECONDS = 120
SOURCE_SHA256 = {
    "src/app/continual_gating_pilot.py": "9a917fe219907a7aa9470f82a1c510863959f5bea8ff9f43db843d9c44ea8c09",
    "src/app/continual_trigger_replay_schedule.py": "e4e1316fb94beff2c005dfca0acaad3e729cc9174be0fac9a9a0ede0219ab042",
    "src/app/continual_arrived_benchmark.py": "21b8d25f7174cf16187054bca6bd41f6ed17dbede2284eb61efd69f16b275ae7",
    "src/app/continual_shift_benchmark.py": "53037d6cfbf4a24a807eb5a4b67d5525cc422e1d209e39ab5c54c1ec0018f5fa",
    "src/core/circadian_predictive_coding.py": "08f36db0a5c1f71d5de198fe0cf1c1be0e6a37eaf5855dec53bbb4a299ec27aa",
    "src/core/predictive_coding.py": "681eae4c48c52b4b0731d859f9947c224ff2c48f4a7ac0d593b1d4b758eeed46",
    "src/infra/continual_roles.py": "b3ea9afd00586a7e5aa2343a73f8f3ee245fa72d4162bb806f5439c86c904dd1",
    "src/infra/datasets.py": "b3ca4e1939afdcced4b22404daaec41303f71f4d3461c797b75f63cbfece747e",
}
ROLE_COUNTS = {
    "a_train": 72,
    "a_inner_guard": 24,
    "a_outer_selection": 24,
    "b_train": 36,
    "b_inner_guard": 12,
    "b_outer_selection": 12,
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def digest_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def manifest_payload() -> dict[str, Any]:
    return json.loads(json.dumps(asdict(fixed_gating_pilot_manifest()), allow_nan=False))


def check_source_hashes() -> dict[str, str]:
    observed = {name: sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_SHA256}
    for name, expected in SOURCE_SHA256.items():
        if observed[name] != expected:
            raise ValueError(f"P6.3 frozen source changed: {name}")
    return observed


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"gating-pilot.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }


def write_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reject_nonfinite(token: str) -> object:
    raise ValueError(f"P6.3 JSON contains nonfinite token {token}")


def parse_finite_json(payload: str) -> dict[str, Any]:
    value = json.loads(payload, parse_constant=reject_nonfinite)

    def inspect(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("P6.3 JSON contains a nonfinite number")
        if isinstance(item, dict):
            for child in item.values():
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)

    inspect(value)
    if not isinstance(value, dict):
        raise ValueError("P6.3 result must be a JSON object")
    return value


def verify_result(payload: dict[str, Any]) -> None:
    manifest = manifest_payload()
    if (
        payload["protocol_id"] != PROTOCOL_ID
        or payload["manifest"] != manifest
        or payload["final_released"] is not False
    ):
        raise ValueError("P6.3 result protocol, manifest, or final seal differs")
    rows = payload["seed_results"]
    if [row["seed"] for row in rows] != list(DEVELOPMENT_SEEDS):
        raise ValueError("P6.3 result has missing or reordered development seeds")
    for row in rows:
        if row["source_role_counts"] != ROLE_COUNTS:
            raise ValueError("P6.3 source role counts differ")
        hashes = row["source_role_hashes"]
        if set(hashes) != set(ROLE_COUNTS) or any(
            not HEX_SHA256.fullmatch(value) for value in hashes.values()
        ):
            raise ValueError("P6.3 development role hashes are incomplete")
        if [method["method"] for method in row["methods"]] != list(ARMS):
            raise ValueError("P6.3 method rows are incomplete")
        ordinary, neutral, gating = row["methods"]
        if (
            len({method["initial_parameter_sha256"] for method in row["methods"]}) != 1
            or ordinary["final_parameter_sha256"] != neutral["final_parameter_sha256"]
            or ordinary["development"] != neutral["development"]
        ):
            raise ValueError("P6.3 ordinary/neutral PC parity differs")
        if ordinary["minimum_plasticity"] is not None or neutral["minimum_plasticity"] != 1.0:
            raise ValueError("P6.3 neutral plasticity fact differs")
        minimum = gating["minimum_plasticity"]
        if type(minimum) not in (int, float) or not 0.2 <= minimum < 1.0:
            raise ValueError("P6.3 chemical gating did not become active")
        for method in row["methods"]:
            _verify_method_work(method)


def _verify_method_work(method: dict[str, Any]) -> None:
    expected = {
        "wake_updates": 24,
        "train_presentations": 1296,
        "latent_iterations": 48,
        "example_inference_iterations": 2592,
        "sleep_attempts": 0,
        "replay_updates": 0,
        "hidden_width_start": 8,
        "hidden_width_end": 8,
        "parameter_count_start": 33,
        "parameter_count_end": 33,
    }
    if any(method[key] != value for key, value in expected.items()):
        raise ValueError(f"P6.3 work or capacity differs for {method['method']}")
    if any(
        not HEX_SHA256.fullmatch(method[key])
        for key in ("initial_parameter_sha256", "final_parameter_sha256")
    ):
        raise ValueError("P6.3 model parameter digest missing")
    metrics = method["development"]
    for key in ("a_after_a", "a_after_b", "b_after_b", "final_mean_task_accuracy"):
        if type(metrics[key]) not in (float, int) or not 0.0 <= metrics[key] <= 1.0:
            raise ValueError(f"P6.3 invalid development metric {key}")
    if not math.isclose(
        metrics["signed_forgetting"], metrics["a_after_a"] - metrics["a_after_b"], abs_tol=1e-12
    ) or not math.isclose(
        metrics["final_mean_task_accuracy"],
        (metrics["a_after_b"] + metrics["b_after_b"]) / 2.0,
        abs_tol=1e-12,
    ):
        raise ValueError("P6.3 derived development metric differs")


def run_bounded_pilot(output_dir: Path) -> dict[str, Any]:
    sources = check_source_hashes()
    manifest = fixed_gating_pilot_manifest()
    planned = validate_gating_pilot_manifest(manifest)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.3 pilot output already exists: {occupied}")
    command = [sys.executable, "-m", "scripts.run_p63_gating_pilot", "--worker"]
    resolved_manifest = manifest_payload()
    request = {
        "schema_id": "p63_gating_request_v1",
        "manifest": resolved_manifest,
        "manifest_sha256": digest_json(resolved_manifest),
        "source_sha256": sources,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_wake_updates": planned,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "command": command,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_exclusive(paths["request"], request)
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
        result = parse_finite_json(process.stdout)
        verify_result(result)
        write_exclusive(paths["result"], result)
    except (Exception, subprocess.TimeoutExpired) as exc:
        write_exclusive(
            paths["failure"],
            {
                "schema_id": "p63_gating_failure_v1",
                "reason": "wall_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "worker_or_audit",
                "error": str(exc),
                "elapsed_seconds": monotonic() - started,
            },
        )
        raise
    audit = {
        "schema_id": "p63_gating_audit_v1",
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "seed_count": len(DEVELOPMENT_SEEDS),
        "cell_count": len(DEVELOPMENT_SEEDS) * len(ARMS),
        "planned_wake_updates": planned,
        "request_sha256": sha256(paths["request"].read_bytes()).hexdigest(),
        "result_sha256": sha256(paths["result"].read_bytes()).hexdigest(),
        "elapsed_seconds": monotonic() - started,
    }
    write_exclusive(paths["audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/p63-gating-pilot"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        check_source_hashes()
        result = run_gating_pilot(fixed_gating_pilot_manifest())
        print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
        return
    print(json.dumps(run_bounded_pilot(options.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
