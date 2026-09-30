"""Run and audit the bounded development-only P6.3 replay factor pilot."""

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

from src.app.continual_replay_factor_pilot import (
    ARMS,
    DEVELOPMENT_SEEDS,
    ON_ARMS,
    PROTOCOL_ID,
    fixed_replay_pilot_manifest,
    run_replay_pilot,
    validate_replay_pilot_manifest,
)
from src.core.continual_metrics import TwoTaskAccuracy


REPO_ROOT = Path(__file__).resolve().parents[1]
WALL_LIMIT_SECONDS = 120
SOURCE_SHA256 = {
    "src/app/continual_replay_factor_pilot.py": "f472668e0bf49b31465609880f6e2c4a7a104a6b558de99a5ba76886469a8894",
    "src/app/continual_trigger_replay_schedule.py": "e4e1316fb94beff2c005dfca0acaad3e729cc9174be0fac9a9a0ede0219ab042",
    "src/app/continual_arrived_benchmark.py": "21b8d25f7174cf16187054bca6bd41f6ed17dbede2284eb61efd69f16b275ae7",
    "src/app/continual_shift_benchmark.py": "53037d6cfbf4a24a807eb5a4b67d5525cc422e1d209e39ab5c54c1ec0018f5fa",
    "src/core/backprop_mlp.py": "bb961b86a5034869356ed1f4a2e609a8b6c1d087e80c8e87b1ec26b07f17b85b",
    "src/core/predictive_coding.py": "681eae4c48c52b4b0731d859f9947c224ff2c48f4a7ac0d593b1d4b758eeed46",
    "src/core/circadian_predictive_coding.py": "08f36db0a5c1f71d5de198fe0cf1c1be0e6a37eaf5855dec53bbb4a299ec27aa",
    "src/core/continual_metrics.py": "8eb491da4221a232a32a7cc1eb7b08a3c36a96fb748543450ac068dac3ae7f78",
    "src/core/shared_replay_schedule.py": "1961f66262e33a0b73bdaf690336f8e79f6ba5047a8df53cd61c8005f10975d0",
    "src/core/replay_retention.py": "f5c5e9a793ad771c1a7ba0dc29411fb36ceafc34c37ca786cbae2efeeaa1108b",
    "src/infra/continual_roles.py": "b3ea9afd00586a7e5aa2343a73f8f3ee245fa72d4162bb806f5439c86c904dd1",
    "src/infra/datasets.py": "b3ca4e1939afdcced4b22404daaec41303f71f4d3461c797b75f63cbfece747e",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROLE_COUNTS = {
    "a_train": 72,
    "a_inner_guard": 24,
    "a_outer_selection": 24,
    "b_train": 36,
    "b_inner_guard": 12,
    "b_outer_selection": 12,
}


def digest_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def manifest_payload() -> dict[str, Any]:
    return json.loads(json.dumps(asdict(fixed_replay_pilot_manifest()), allow_nan=False))


def check_source_hashes() -> dict[str, str]:
    observed = {name: sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_SHA256}
    for name, expected in SOURCE_SHA256.items():
        if observed[name] != expected:
            raise ValueError(f"P6.3 replay factor frozen source changed: {name}")
    return observed


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"replay-factor-pilot.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }


def write_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reject_nonfinite(token: str) -> object:
    raise ValueError(f"P6.3 replay factor JSON contains nonfinite token {token}")


def parse_finite_json(payload: str) -> dict[str, Any]:
    value = json.loads(payload, parse_constant=reject_nonfinite)

    def inspect(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("P6.3 replay factor JSON contains a nonfinite number")
        if isinstance(item, dict):
            for child in item.values():
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)

    inspect(value)
    if not isinstance(value, dict):
        raise ValueError("P6.3 replay factor result must be a JSON object")
    return value


def verify_result(payload: dict[str, Any]) -> None:
    if (
        payload["protocol_id"] != PROTOCOL_ID
        or payload["manifest"] != manifest_payload()
        or payload["final_released"] is not False
    ):
        raise ValueError("P6.3 replay factor protocol, manifest, or final seal differs")
    rows = payload["seed_results"]
    if [row["seed"] for row in rows] != list(DEVELOPMENT_SEEDS):
        raise ValueError("P6.3 replay factor development seeds differ")
    for row in rows:
        _verify_seed(row)


def _verify_seed(row: dict[str, Any]) -> None:
    if row["source_role_counts"] != ROLE_COUNTS:
        raise ValueError("P6.3 replay factor source role counts differ")
    hashes = row["source_role_hashes"]
    if set(hashes) != set(ROLE_COUNTS) or any(
        not HEX_SHA256.fullmatch(value) for value in hashes.values()
    ):
        raise ValueError("P6.3 replay factor development role hashes are incomplete")
    boundaries = row["boundaries"]
    if [(item["phase"], item["epoch"]) for item in boundaries] != [
        (phase, epoch) for phase in ("a", "b") for epoch in (4, 8, 12)
    ]:
        raise ValueError("P6.3 replay factor boundaries differ")
    for boundary in boundaries:
        selected = boundary["selected_ids"]
        if (
            len(selected) != 2
            or len(set(selected)) != 2
            or boundary["retained_examples"] != 8
            or boundary["retained_bytes"] != 192
            or boundary["retained_order_ids"][-2:] != selected
            or len(boundary["retained_order_ids"]) != 8
            or set(boundary["applied_ids_by_method"]) != set(ARMS)
            or any(
                boundary["applied_ids_by_method"][name] != (selected if name in ON_ARMS else [])
                for name in ARMS
            )
        ):
            raise ValueError("P6.3 replay factor retained, selected, or applied IDs differ")
    methods = row["methods"]
    if [method["method"] for method in methods] != list(ARMS):
        raise ValueError("P6.3 replay factor method rows are incomplete")
    if len({method["initial_parameter_sha256"] for method in methods[:6]}) != 1:
        raise ValueError("P6.3 replay factor width-eight initialization differs")
    if methods[6]["initial_parameter_sha256"] != methods[7]["initial_parameter_sha256"]:
        raise ValueError("P6.3 replay factor planned-width initialization differs")
    if (
        methods[2]["final_parameter_sha256"] != methods[4]["final_parameter_sha256"]
        or methods[3]["final_parameter_sha256"] != methods[5]["final_parameter_sha256"]
        or methods[2]["development"] != methods[4]["development"]
        or methods[3]["development"] != methods[5]["development"]
    ):
        raise ValueError("P6.3 replay factor neutral circadian parity differs")
    for method in methods:
        _verify_method(method)


def _verify_method(method: dict[str, Any]) -> None:
    name = method["method"]
    is_pc = not name.startswith("backprop")
    on = name in ON_ARMS
    width, parameters = (12, 49) if "width12" in name else (8, 33)
    expected = {
        "wake_updates": 24,
        "wake_presentations": 1296,
        "wake_inference_loops": 48 if is_pc else 0,
        "wake_example_inference_iterations": 2592 if is_pc else 0,
        "replay_updates": 12 if on else 0,
        "replay_presentations": 12 if on else 0,
        "replay_inference_loops": 24 if on and is_pc else 0,
        "sleep_attempts": 6 if name.startswith("circadian") else 0,
        "hidden_width_start": width,
        "hidden_width_end": width,
        "parameter_count_start": parameters,
        "parameter_count_end": parameters,
    }
    if any(method[key] != value for key, value in expected.items()):
        raise ValueError(f"P6.3 replay factor work/capacity differs: {name}")
    if any(
        not HEX_SHA256.fullmatch(method[key])
        for key in ("initial_parameter_sha256", "final_parameter_sha256")
    ):
        raise ValueError(f"P6.3 replay factor parameter hash missing: {name}")
    metrics = method["development"]
    observed = TwoTaskAccuracy(metrics["a_after_a"], metrics["a_after_b"], metrics["b_after_b"])
    if (
        not math.isclose(
            metrics["final_mean_task_accuracy"], observed.final_mean_task_accuracy, abs_tol=1e-12
        )
        or not math.isclose(
            metrics["signed_forgetting_a"], observed.signed_forgetting_a, abs_tol=1e-12
        )
        or metrics["retention_ratio_a"] != observed.retention_ratio_a
    ):
        raise ValueError(f"P6.3 replay factor derived metrics differ: {name}")


def run_bounded_pilot(output_dir: Path) -> dict[str, Any]:
    sources = check_source_hashes()
    manifest = fixed_replay_pilot_manifest()
    planned = validate_replay_pilot_manifest(manifest)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.3 replay factor output already exists: {occupied}")
    command = [sys.executable, "-m", "scripts.run_p63_replay_factor_pilot", "--worker"]
    resolved_manifest = manifest_payload()
    request = {
        "schema_id": "p63_replay_factor_request_v1",
        "manifest": resolved_manifest,
        "manifest_sha256": digest_json(resolved_manifest),
        "source_sha256": sources,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_optimizer_updates": planned,
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
                "schema_id": "p63_replay_factor_failure_v1",
                "reason": "wall_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "worker_or_audit",
                "error": str(exc),
                "elapsed_seconds": monotonic() - started,
            },
        )
        raise
    audit = {
        "schema_id": "p63_replay_factor_audit_v1",
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "seed_count": len(DEVELOPMENT_SEEDS),
        "cell_count": len(DEVELOPMENT_SEEDS) * len(ARMS),
        "planned_optimizer_updates": planned,
        "request_sha256": sha256(paths["request"].read_bytes()).hexdigest(),
        "result_sha256": sha256(paths["result"].read_bytes()).hexdigest(),
        "elapsed_seconds": monotonic() - started,
    }
    write_exclusive(paths["audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p63-replay-factor-pilot")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        check_source_hashes()
        result = run_replay_pilot(fixed_replay_pilot_manifest())
        print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
        return
    audit = run_bounded_pilot(options.output_dir)
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
