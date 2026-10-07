"""Compare two P6.2 profile runs, excluding measured sleep durations only."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scripts.run_p62_profile_reproduction import (
    PROFILE_BUDGET_SECONDS,
    artifact_paths,
    read_finite_json,
)


def differences(left: Any, right: Any, path: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [path]
        return [item for key in left for item in differences(left[key], right[key], (*path, key))]
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        return [
            item
            for index, (a, b) in enumerate(zip(left, right))
            for item in differences(a, b, (*path, index))
        ]
    return [] if left == right else [path]


def is_measured_duration(path: tuple[object, ...]) -> bool:
    return (
        len(path) == 7
        and path[0] == "seed_results"
        and type(path[1]) is int
        and path[2] == "circadian_predictive_coding"
        and path[3] == "sleep_events"
        and type(path[4]) is int
        and path[5] == "durations"
        and path[6] in {"attempt_seconds", "core_seconds"}
    )


def deterministic_digest(result: dict[str, Any]) -> str:
    copy = deepcopy(result)
    for row in copy["seed_results"]:
        for event in row["circadian_predictive_coding"]["sleep_events"]:
            durations = event.get("durations")
            if isinstance(durations, dict):
                durations.pop("attempt_seconds", None)
                durations.pop("core_seconds", None)
                if not durations:
                    event.pop("durations")
    encoded = json.dumps(copy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def verify_saved_hashes(paths: dict[str, Path], profile: str) -> dict[str, Any]:
    audit = read_finite_json(paths["audit"])
    if audit["profile"] != profile or audit["result_count"] != 7:
        raise ValueError(f"P6.2 repeat audit identity differs for {profile}")
    for name in ("request", "text", "result", "config"):
        if sha256(paths[name].read_bytes()).hexdigest() != audit["artifact_sha256"][name]:
            raise ValueError(f"P6.2 repeat artifact digest differs: {profile}/{name}")
    return audit


def compare_profile(profile: str, first_dir: Path, second_dir: Path) -> dict[str, Any]:
    first = artifact_paths(first_dir, profile)
    second = artifact_paths(second_dir, profile)
    first_audit = verify_saved_hashes(first, profile)
    second_audit = verify_saved_hashes(second, profile)
    for field in ("protocol_id", "seeds", "policy_sha256", "result_count"):
        if first_audit[field] != second_audit[field]:
            raise ValueError(f"P6.2 repeat audit {field} differs for {profile}")
    first_request = read_finite_json(first["request"])
    second_request = read_finite_json(second["request"])
    for field in (
        "profile",
        "protocol_id",
        "seeds",
        "profile_defaults",
        "circadian_policy_sha256",
        "source_sha256",
        "wall_limit_seconds",
    ):
        if first_request[field] != second_request[field]:
            raise ValueError(f"P6.2 repeat request {field} differs for {profile}")
    for name in ("text", "config"):
        if first[name].read_bytes() != second[name].read_bytes():
            raise ValueError(f"P6.2 repeat {name} bytes differ for {profile}")
    left = read_finite_json(first["result"])
    right = read_finite_json(second["result"])
    changed = differences(left, right)
    if any(not is_measured_duration(path) for path in changed):
        raise ValueError(f"P6.2 repeat deterministic result differs for {profile}")
    left_digest = deterministic_digest(left)
    if left_digest != deterministic_digest(right):
        raise ValueError(f"P6.2 repeat deterministic digest differs for {profile}")
    return {
        "profile": profile,
        "deterministic_result_sha256": left_digest,
        "measured_duration_fields_differed": len(changed),
        "text_sha256": first_audit["artifact_sha256"]["text"],
        "config_sha256": first_audit["artifact_sha256"]["config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    options = parser.parse_args()
    reports = [
        compare_profile(profile, options.first_dir, options.second_dir)
        for profile in PROFILE_BUDGET_SECONDS
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
