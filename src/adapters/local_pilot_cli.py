"""CLI resource planning only; no model, dataset or execution dispatch.

Inputs are resource/context flags. Output is JSON and a success/refusal exit code.
Actual work accounting, experiment scoring and permissions belong to other ports.
"""

import argparse
from dataclasses import asdict
import json
from typing import Sequence

from src.core.local_pilot_budget import (
    FIRST_LOCAL_PILOT_LIMITS,
    LOCAL_PILOT_BUDGET_ID,
    PilotRequest,
    PilotResources,
    validate_local_pilot_request,
)

_RESOURCE_FLAGS = {
    "wall_seconds": float,
    "training_updates": int,
    "environment_steps": int,
    "cpu_threads": int,
    "process_memory_bytes": int,
    "replay_bytes": int,
    "model_download_bytes": int,
    "storage_bytes": int,
    "gpu_memory_bytes": int,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a local simulated pilot's planned resources."
    )
    for name, converter in _RESOURCE_FLAGS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=converter,
            default=getattr(FIRST_LOCAL_PILOT_LIMITS, name),
        )
    parser.add_argument("--execution-target", default="local")
    parser.add_argument("--environment-kind", default="simulation")
    parser.add_argument("--device", default="cpu")
    return parser


def run_preflight(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    try:
        resources = PilotResources(**{name: getattr(parsed, name) for name in _RESOURCE_FLAGS})
        request = validate_local_pilot_request(
            PilotRequest(resources, parsed.execution_target, parsed.environment_kind, parsed.device)
        )
    except ValueError as error:
        print(
            json.dumps(
                {"status": "rejected", "budget_id": LOCAL_PILOT_BUDGET_ID, "error": str(error)}
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "accepted",
                "scope": "resource_preflight_only",
                "budget_id": LOCAL_PILOT_BUDGET_ID,
                "request": asdict(request),
                "limits": asdict(FIRST_LOCAL_PILOT_LIMITS),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    raise SystemExit(run_preflight())
