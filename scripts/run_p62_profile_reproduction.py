"""Run one frozen P6.2 continual profile with local time and artifact gates.

Inputs: a named historical profile and a fresh local output directory.
Outputs: an exclusive prelaunch request, CLI text/result/config, and audit or
failure JSON. This adapter neither changes model settings nor selects scores.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import fmean, pstdev
import subprocess
import sys
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_continual_shift_benchmark import (  # noqa: E402
    _build_baseline_circadian_config,
    _build_hardest_case_circadian_config,
    _build_profile_defaults,
    _build_strength_case_circadian_config,
)
from src.app.continual_shift_benchmark import (  # noqa: E402
    CONTINUAL_MODEL_ORDER,
    CONTINUAL_VALIDATION_PROTOCOL,
)


SEEDS = (3, 7, 11, 19, 23, 31, 37)
PROFILE_BUDGET_SECONDS = {"baseline": 240, "strength-case": 240, "hardest-case": 600}
POLICY_SHA256 = {
    "baseline": "f3f6801e915a409cb2d2ae50876bcb0e27cc9f1f1c044cba0b16e783499b645c",
    "strength-case": "7c51a9c764e6fc95ecb2f2bb94855e6a94bd11c6baf6acf86b743e6540c4732a",
    "hardest-case": "91974bbe5f4aadf75a1bc650239284360ecdffae05d219186d94b98f5f03e0e3",
}
SOURCE_SHA256 = {
    "scripts/run_continual_shift_benchmark.py": (
        "a605b8ff4d725299b90fb3365de377a40b279974bb3193d427878ebe7b032044"
    ),
    "src/app/continual_shift_benchmark.py": (
        "53037d6cfbf4a24a807eb5a4b67d5525cc422e1d209e39ab5c54c1ec0018f5fa"
    ),
    "src/core/circadian_predictive_coding.py": (
        "08f36db0a5c1f71d5de198fe0cf1c1be0e6a37eaf5855dec53bbb4a299ec27aa"
    ),
}
ROLE_NAMES = {
    "phase_a_train",
    "phase_a_validation",
    "phase_a_test",
    "phase_b_train",
    "phase_b_validation",
    "phase_b_test",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHIFT_METRICS = (
    "phase_a_pre_accuracy",
    "phase_a_post_accuracy",
    "phase_b_post_accuracy",
    "retention_ratio",
    "balanced_score",
)
REPORT_METRICS = (
    ("A_pre", "phase_a_pre_accuracy"),
    ("A_post", "phase_a_post_accuracy"),
    ("B_post", "phase_b_post_accuracy"),
    ("retention", "retention_ratio"),
    ("balanced", "balanced_score"),
)


def digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def require_frozen_sources() -> dict[str, str]:
    observed = {name: sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_SHA256}
    for name, expected in SOURCE_SHA256.items():
        if observed[name] != expected:
            raise ValueError(f"P6.2 frozen source changed: {name}")
    return observed


def profile_policy(profile: str) -> dict[str, Any]:
    builders = {
        "baseline": _build_baseline_circadian_config,
        "strength-case": _build_strength_case_circadian_config,
        "hardest-case": _build_hardest_case_circadian_config,
    }
    policy = asdict(builders[profile]())
    if digest_json(policy) != POLICY_SHA256[profile]:
        raise ValueError(f"P6.2 frozen circadian policy changed: {profile}")
    return policy


def artifact_paths(output_dir: Path, profile: str) -> dict[str, Path]:
    return {
        "request": output_dir / f"{profile}.request.json",
        "text": output_dir / f"{profile}.txt",
        "result": output_dir / f"{profile}.json",
        "config": output_dir / f"{profile}-config.json",
        "audit": output_dir / f"{profile}.audit.json",
        "failure": output_dir / f"{profile}.failure.json",
    }


def build_command(profile: str, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.run_continual_shift_benchmark",
        "--protocol-id",
        CONTINUAL_VALIDATION_PROTOCOL,
        "--profile",
        profile,
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--output-file",
        str(paths["text"]),
        "--json-result",
        str(paths["result"]),
        "--resolved-config",
        str(paths["config"]),
    ]


def write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reject_nonfinite(token: str) -> object:
    raise ValueError(f"nonfinite P6.2 JSON: {token}")


def decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def read_finite_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)

    def inspect(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"nonfinite P6.2 number in {path}")
        if isinstance(item, dict):
            for child in item.values():
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)

    inspect(value)
    if not isinstance(value, dict):
        raise ValueError(f"P6.2 JSON must be an object: {path}")
    return value


def verify_text_summary(report: str, result: dict[str, Any]) -> None:
    """Cross-check the human report against the saved numeric and role rows."""
    config = result["config"]
    aggregate = result["aggregate"]
    rows = result["seed_results"]
    report_lines = set(report.splitlines())
    expected_lines = {
        f"Protocol: {config['protocol_id']}",
        f"Circadian sleep mode: {config['circadian_config']['sleep_mode']}",
        f"Seeds: {result['seeds']}",
        f"Split hashes by seed: { {row['seed']: row['split_hashes'] for row in rows} }",
        f"Phase B train fraction: {config['phase_b_train_fraction']:.2f}",
    }
    labels = {
        "backprop": "Backprop",
        "predictive_coding": "Predictive coding",
        "circadian_predictive_coding": "Circadian predictive coding",
    }
    for name, label in labels.items():
        fields = ", ".join(
            f"{title}={aggregate[name][f'mean_{metric}']:.3f}"
            f"+/-{aggregate[name][f'std_{metric}']:.3f}"
            for title, metric in REPORT_METRICS
        )
        if name == "circadian_predictive_coding":
            fields += ", " + ", ".join(
                f"{title}={aggregate[name][f'mean_{field}']:.2f}"
                for title, field in (
                    ("sleep_events", "sleep_event_count"),
                    ("splits", "total_splits"),
                    ("prunes", "total_prunes"),
                    ("hidden_end", "hidden_dim_end"),
                )
            )
        expected_lines.add(f"{label}: {fields}")
    if missing := expected_lines - report_lines:
        raise ValueError(f"P6.2 text/JSON summary differs: {sorted(missing)}")


def verify_artifacts(
    profile: str, paths: dict[str, Path], stdout: str, policy: dict[str, Any]
) -> dict[str, Any]:
    result = read_finite_json(paths["result"])
    resolved = read_finite_json(paths["config"])
    config = result["config"]
    defaults = asdict(_build_profile_defaults(profile))
    if resolved["schema_id"] != "continual_resolved_config_v1":
        raise ValueError("P6.2 resolved config schema differs")
    if resolved["preset"] != profile or resolved["config"] != config:
        raise ValueError("P6.2 resolved config differs from result/profile")
    if resolved["overrides"] != {} or resolved["seeds"] != list(SEEDS):
        raise ValueError("P6.2 resolved seeds or overrides differ")
    if result["seeds"] != list(SEEDS) or config["protocol_id"] != CONTINUAL_VALIDATION_PROTOCOL:
        raise ValueError("P6.2 protocol or result seeds differ")
    if config["validation_fraction"] != 0.2 or config["model_order"] != list(CONTINUAL_MODEL_ORDER):
        raise ValueError("P6.2 validation or model order differs")
    if config["circadian_config"] != policy:
        raise ValueError("P6.2 circadian policy differs")
    for field, value in defaults.items():
        key = field.replace("sleep_interval", "circadian_sleep_interval")
        if config[key] != (list(value) if isinstance(value, tuple) else value):
            raise ValueError(f"P6.2 preset field differs: {key}")
    rows = result["seed_results"]
    if [row["seed"] for row in rows] != list(SEEDS):
        raise ValueError("P6.2 per-seed rows differ")
    for row in rows:
        hashes = row["split_hashes"]
        if set(hashes) != ROLE_NAMES or any(not HEX_SHA256.fullmatch(d) for d in hashes.values()):
            raise ValueError(f"P6.2 role hashes missing for seed {row['seed']}")
        if len(set(hashes.values())) != len(ROLE_NAMES):
            raise ValueError(f"P6.2 role identities collide for seed {row['seed']}")
        for name in CONTINUAL_MODEL_ORDER:
            if name not in row:
                raise ValueError(f"P6.2 method missing for seed {row['seed']}: {name}")
    aggregate = result["aggregate"]
    if aggregate["run_count"] != len(SEEDS):
        raise ValueError("P6.2 aggregate seed count differs")
    for name in CONTINUAL_MODEL_ORDER:
        for metric in SHIFT_METRICS:
            values = [row[name][metric] for row in rows]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError(f"P6.2 invalid per-seed {name}.{metric}")
            mean = aggregate[name][f"mean_{metric}"]
            spread = aggregate[name][f"std_{metric}"]
            if not math.isclose(mean, fmean(values), abs_tol=1e-12) or not math.isclose(
                spread, pstdev(values), abs_tol=1e-12
            ):
                raise ValueError(f"P6.2 aggregate differs for {name}.{metric}")
    for field in ("sleep_event_count", "total_splits", "total_prunes", "hidden_dim_end"):
        values = [row["circadian_predictive_coding"][field] for row in rows]
        if not math.isclose(
            aggregate["circadian_predictive_coding"][f"mean_{field}"],
            fmean(values),
            abs_tol=1e-12,
        ):
            raise ValueError(f"P6.2 circadian aggregate differs for {field}")
    report = paths["text"].read_text(encoding="utf-8")
    if report != stdout or f"Protocol: {CONTINUAL_VALIDATION_PROTOCOL}" not in report:
        raise ValueError("P6.2 text/stdout protocol differs")
    verify_text_summary(report, result)
    return {
        "schema_id": "p62_profile_audit_v1",
        "profile": profile,
        "protocol_id": CONTINUAL_VALIDATION_PROTOCOL,
        "seeds": list(SEEDS),
        "result_count": len(rows),
        "policy_sha256": digest_json(policy),
        "artifact_sha256": {
            name: sha256(paths[name].read_bytes()).hexdigest()
            for name in ("request", "text", "result", "config")
        },
    }


def run_profile(profile: str, output_dir: Path) -> dict[str, Any]:
    if profile not in PROFILE_BUDGET_SECONDS:
        raise ValueError(f"unknown P6.2 profile: {profile}")
    sources = require_frozen_sources()
    policy = profile_policy(profile)
    output_dir = output_dir.resolve()
    paths = artifact_paths(output_dir, profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"P6.2 profile output already exists: {occupied}")
    command = build_command(profile, paths)
    request = {
        "schema_id": "p62_profile_request_v1",
        "profile": profile,
        "protocol_id": CONTINUAL_VALIDATION_PROTOCOL,
        "seeds": list(SEEDS),
        "profile_defaults": asdict(_build_profile_defaults(profile)),
        "circadian_policy_sha256": digest_json(policy),
        "source_sha256": sources,
        "wall_limit_seconds": PROFILE_BUDGET_SECONDS[profile],
        "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_exclusive(paths["request"], request)
    started = monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=PROFILE_BUDGET_SECONDS[profile],
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        failure = {
            "schema_id": "p62_profile_failure_v1",
            "profile": profile,
            "reason": "wall_limit",
            "elapsed_seconds": monotonic() - started,
            "stdout": decode_process_output(exc.stdout),
            "stderr": decode_process_output(exc.stderr),
        }
        write_json_exclusive(paths["failure"], failure)
        raise RuntimeError(f"P6.2 {profile} exceeded its local wall limit") from exc
    if process.returncode != 0:
        write_json_exclusive(
            paths["failure"],
            {
                "schema_id": "p62_profile_failure_v1",
                "profile": profile,
                "reason": "cli_exit",
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            },
        )
        raise RuntimeError(f"P6.2 {profile} CLI exited {process.returncode}: {process.stderr}")
    try:
        audit = verify_artifacts(profile, paths, process.stdout, policy)
    except Exception as exc:
        write_json_exclusive(
            paths["failure"],
            {
                "schema_id": "p62_profile_failure_v1",
                "profile": profile,
                "reason": "artifact_validation",
                "error": str(exc),
            },
        )
        raise
    audit["elapsed_seconds"] = monotonic() - started
    write_json_exclusive(paths["audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_BUDGET_SECONDS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/p62-profiles"))
    options = parser.parse_args()
    print(json.dumps(run_profile(options.profile, options.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
