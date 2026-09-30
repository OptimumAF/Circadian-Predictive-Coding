"""Launch and audit the bounded P6.3 train-only sleep-factor gate."""

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

from src.app.continual_sleep_factor_preflight import (
    ARMS,
    DEVELOPMENT_SEEDS,
    SLEEP_ARMS,
    PROTOCOL_ID,
    fixed_sleep_factor_manifest,
    run_sleep_factor_preflight,
    validate_sleep_factor_manifest,
)
from src.shared.process_memory import ProcessRssSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
WALL_LIMIT_SECONDS = 120
SOURCE_SHA256 = {
    "src/app/continual_sleep_factor_preflight.py": "f5b7d8d8f071ecd7c429f87581e52033b3a99e754721b7dd01635241f75bb6ff",
    "src/app/continual_trigger_replay_schedule.py": "e4e1316fb94beff2c005dfca0acaad3e729cc9174be0fac9a9a0ede0219ab042",
    "src/app/continual_arrived_benchmark.py": "21b8d25f7174cf16187054bca6bd41f6ed17dbede2284eb61efd69f16b275ae7",
    "src/app/continual_shift_benchmark.py": "53037d6cfbf4a24a807eb5a4b67d5525cc422e1d209e39ab5c54c1ec0018f5fa",
    "src/app/numpy_sleep_decisions.py": "085aa77b010dee217bea6467119addc3c3cca14437b669d350d404f00e5950b0",
    "src/app/sleep_schedule.py": "23cbde1e0f3c60bf35fd143e3e7ee63955862818c0b686c4ecb5eadb5a46b5c9",
    "src/core/backprop_mlp.py": "bb961b86a5034869356ed1f4a2e609a8b6c1d087e80c8e87b1ec26b07f17b85b",
    "src/core/predictive_coding.py": "681eae4c48c52b4b0731d859f9947c224ff2c48f4a7ac0d593b1d4b758eeed46",
    "src/core/circadian_predictive_coding.py": "08f36db0a5c1f71d5de198fe0cf1c1be0e6a37eaf5855dec53bbb4a299ec27aa",
    "src/core/sleep_telemetry.py": "b63c33b379254a5423831de68b7ec40cbc46d7ad8c96ef2ac6bc9fd1ea058a7d",
    "src/infra/continual_roles.py": "b3ea9afd00586a7e5aa2343a73f8f3ee245fa72d4162bb806f5439c86c904dd1",
    "src/infra/datasets.py": "b3ca4e1939afdcced4b22404daaec41303f71f4d3461c797b75f63cbfece747e",
    "src/shared/process_memory.py": "78662728047b0a6fdd8b29fd0e54a1c09dbbac9d90e6d8c9dfbb5225bc6e8f10",
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
    return json.loads(json.dumps(asdict(fixed_sleep_factor_manifest()), allow_nan=False))


def check_source_hashes() -> dict[str, str]:
    observed = {name: sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_SHA256}
    for name, expected in SOURCE_SHA256.items():
        if observed[name] != expected:
            raise ValueError(f"P6.3 sleep factor frozen source changed: {name}")
    return observed


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / f"sleep-factor-preflight.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }


def write_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reject_nonfinite(token: str) -> object:
    raise ValueError(f"P6.3 sleep factor JSON contains nonfinite token {token}")


def parse_finite_json(payload: str) -> dict[str, Any]:
    value = json.loads(payload, parse_constant=reject_nonfinite)

    def inspect(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("P6.3 sleep factor JSON contains a nonfinite number")
        if isinstance(item, dict):
            for child in item.values():
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)

    inspect(value)
    if not isinstance(value, dict):
        raise ValueError("P6.3 sleep factor worker output must be a JSON object")
    return value


def verify_result(payload: dict[str, Any]) -> None:
    if (
        payload["protocol_id"] != PROTOCOL_ID
        or payload["manifest"] != manifest_payload()
        or payload["final_released"] is not False
        or payload["outer_selection_scored"] is not False
    ):
        raise ValueError("P6.3 sleep factor protocol, manifest, or evaluation seal differs")
    rows = payload["seed_results"]
    if [row["seed"] for row in rows] != list(DEVELOPMENT_SEEDS):
        raise ValueError("P6.3 sleep factor development seeds differ")
    for row in rows:
        _verify_seed(row)


def _verify_seed(row: dict[str, Any]) -> None:
    if row["final_released"] is not False or row["role_counts"] != ROLE_COUNTS:
        raise ValueError("P6.3 sleep factor seed final seal or role counts differ")
    hashes = row["role_hashes"]
    if set(hashes) != set(ROLE_COUNTS) or any(
        not HEX_SHA256.fullmatch(value) for value in hashes.values()
    ):
        raise ValueError("P6.3 sleep factor development role hashes are incomplete")
    arms = row["arms"]
    if [arm["name"] for arm in arms] != list(ARMS):
        raise ValueError("P6.3 sleep factor arm rows are incomplete")
    if len({arm["initial_parameter_sha256"] for arm in arms[:7]}) != 1:
        raise ValueError("P6.3 sleep factor width-eight initialization differs")
    if arms[7]["initial_parameter_sha256"] != arms[8]["initial_parameter_sha256"]:
        raise ValueError("P6.3 sleep factor planned-width initialization differs")
    if arms[1]["final_parameter_sha256"] != arms[2]["final_parameter_sha256"]:
        raise ValueError("P6.3 sleep factor neutral PC parity differs")
    if (
        arms[5]["pre_sleep_parameter_sha256"] != arms[6]["pre_sleep_parameter_sha256"]
        or arms[5]["post_sleep_parameter_sha256"] != arms[6]["post_sleep_parameter_sha256"]
    ):
        raise ValueError("P6.3 sleep factor gated pair boundary parity differs")
    for arm in arms:
        _verify_arm(arm, hashes["a_inner_guard"])


def _verify_arm(arm: dict[str, Any], guard_hash: str) -> None:
    name = arm["name"]
    width, parameters = (12, 49) if name.endswith("_12") else (8, 33)
    expected = {
        "wake_updates": 24,
        "wake_presentations": 1296,
        "latent_inference_loops": 0 if name.startswith("backprop") else 48,
        "example_inference_iterations": 0 if name.startswith("backprop") else 2592,
        "replay_updates": 0,
        "width_initial": width,
        "width_final": width,
        "parameters_initial": parameters,
        "parameters_final": parameters,
    }
    if any(arm[key] != value for key, value in expected.items()):
        raise ValueError(f"P6.3 sleep factor work/capacity differs: {name}")
    if any(
        not HEX_SHA256.fullmatch(arm[key])
        for key in (
            "initial_parameter_sha256",
            "pre_sleep_parameter_sha256",
            "post_sleep_parameter_sha256",
            "final_parameter_sha256",
        )
    ):
        raise ValueError(f"P6.3 sleep factor parameter hash missing: {name}")
    if name in {"gating_sham", "gating_reset"}:
        minimum = arm["minimum_a_plasticity"]
        if type(minimum) not in (float, int) or not 0.2 <= minimum < 1.0:
            raise ValueError(f"P6.3 sleep factor chemical gating inactive: {name}")
    elif arm["minimum_a_plasticity"] is not None:
        raise ValueError(f"P6.3 sleep factor unexpected chemical diagnostic: {name}")
    sleep = arm["sleep"]
    if name not in SLEEP_ARMS:
        if sleep is not None or arm["width_peak"] != width or arm["parameters_peak"] != parameters:
            raise ValueError(f"P6.3 sleep factor no-sleep work differs: {name}")
        return
    if sleep is None:
        raise ValueError(f"P6.3 sleep factor missing guarded event: {name}")
    _verify_sleep(name, arm, sleep, guard_hash)


def _verify_sleep(name: str, arm: dict[str, Any], sleep: dict[str, Any], guard_hash: str) -> None:
    before = sleep["guard_pre_accuracy"]
    after = sleep["guard_post_accuracy"]
    accepted = sleep["outcome"] == "accepted"
    if (
        sleep["outcome"] not in {"accepted", "rolled_back"}
        or sleep["guard_role_hash"] != guard_hash
        or sleep["guard_evaluations"] != 2
        or type(before) not in (float, int)
        or type(after) not in (float, int)
        or not 0 <= before <= 1
        or not 0 <= after <= 1
        or accepted != (after >= before)
        or sleep["replay_updates"] != 0
        or sleep["width_before"] != 8
        or sleep["width_after"] != 8
        or arm["width_peak"] != sleep["transient_peak_width"]
        or arm["parameters_peak"] != 4 * arm["width_peak"] + 1
    ):
        raise ValueError(f"P6.3 sleep factor guard/capacity differs: {name}")
    proposed = sleep["proposed_split_pairs"]
    removed = sleep["proposed_removed_prune_ids"]
    if name == "structure_only":
        if (
            len(proposed) != 1
            or len(removed) != 1
            or sleep["transient_peak_width"] != 9
            or len(sleep["applied_split_pairs"]) != int(accepted)
            or len(sleep["applied_removed_prune_ids"]) != int(accepted)
        ):
            raise ValueError("P6.3 structural proposal or rollback differs")
    elif proposed or removed or sleep["transient_peak_width"] != 8:
        raise ValueError(f"P6.3 nonstructural arm changed topology: {name}")
    if name in {"neutral_sham", "gating_sham", "gating_reset"} and (
        arm["pre_sleep_parameter_sha256"] != arm["post_sleep_parameter_sha256"]
    ):
        raise ValueError(f"P6.3 no-weight sleep changed parameters: {name}")
    if (
        name == "homeostasis_only"
        and accepted
        and (arm["pre_sleep_parameter_sha256"] == arm["post_sleep_parameter_sha256"])
    ):
        raise ValueError("P6.3 accepted homeostasis was inactive")
    if (
        name == "gating_reset"
        and accepted
        and not (0.0 < sleep["chemical_proposed_mean"] < sleep["chemical_before_mean"])
    ):
        raise ValueError("P6.3 accepted chemical reset was inactive")


def verify_memory(memory: dict[str, Any], limit: int) -> None:
    if (
        type(memory["pid"]) is not int
        or memory["pid"] <= 0
        or type(memory["start_bytes"]) is not int
        or type(memory["peak_bytes"]) is not int
        or type(memory["sample_count"]) is not int
        or memory["sample_count"] < 2
        or not 0 < memory["start_bytes"] <= memory["peak_bytes"] <= limit
        or memory["interval_seconds"] != 0.005
    ):
        raise ValueError("P6.3 sleep factor process RSS differs or exceeds cap")


def run_bounded_preflight(output_dir: Path) -> dict[str, Any]:
    sources = check_source_hashes()
    manifest = fixed_sleep_factor_manifest()
    planned = validate_sleep_factor_manifest(manifest)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.3 sleep factor output already exists: {occupied}")
    command = [sys.executable, "-m", "scripts.run_p63_sleep_factor_preflight", "--worker"]
    resolved_manifest = manifest_payload()
    request = {
        "schema_id": "p63_sleep_factor_preflight_request_v1",
        "manifest": resolved_manifest,
        "manifest_sha256": digest_json(resolved_manifest),
        "source_sha256": sources,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_optimizer_updates": planned,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "max_process_rss_bytes": manifest.max_process_rss_bytes,
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
        payload = parse_finite_json(process.stdout)
        verify_result(payload["result"])
        verify_memory(payload["process_rss"], manifest.max_process_rss_bytes)
        write_exclusive(paths["result"], payload["result"])
    except (Exception, subprocess.TimeoutExpired) as exc:
        write_exclusive(
            paths["failure"],
            {
                "schema_id": "p63_sleep_factor_preflight_failure_v1",
                "reason": "wall_limit"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "worker_or_audit",
                "error": str(exc),
                "elapsed_seconds": monotonic() - started,
            },
        )
        raise
    audit = {
        "schema_id": "p63_sleep_factor_preflight_audit_v1",
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "seed_count": len(DEVELOPMENT_SEEDS),
        "cell_count": len(DEVELOPMENT_SEEDS) * len(ARMS),
        "guarded_sleep_attempts": len(DEVELOPMENT_SEEDS) * len(SLEEP_ARMS),
        "planned_optimizer_updates": planned,
        "process_rss": payload["process_rss"],
        "request_sha256": sha256(paths["request"].read_bytes()).hexdigest(),
        "result_sha256": sha256(paths["result"].read_bytes()).hexdigest(),
        "elapsed_seconds": monotonic() - started,
    }
    write_exclusive(paths["audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p63-sleep-factor-preflight")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        check_source_hashes()
        manifest = fixed_sleep_factor_manifest()
        with ProcessRssSampler(interval_seconds=0.005) as sampler:
            result = run_sleep_factor_preflight(manifest)
        memory = sampler.snapshot()
        if memory.peak_bytes > manifest.max_process_rss_bytes:
            raise ValueError("P6.3 sleep factor observed process RSS exceeded cap")
        print(
            json.dumps(
                {"result": asdict(result), "process_rss": asdict(memory)},
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    audit = run_bounded_preflight(options.output_dir)
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
