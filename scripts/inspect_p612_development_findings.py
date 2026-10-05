"""Retain complete fixed development evidence through unchanged validator ports.

Only the output path is configurable. This adapter performs stored-input
validation and projection, with no training, scoring or final-source dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts import inspect_p67_confirmation_scope as original
from scripts import run_p63_combined_factor_preflight as combined
from scripts import run_p63_parent_factor_preflight as parent
from scripts import run_p63_schedule_factor_preflight as schedule
from scripts import run_p63_sleep_factor_preflight as sleep
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.app.continual_findings_development import build_development_ledger
from src.infra.continual_confirmation_io import write_exclusive
from src.infra.continual_confirmation_training_references import stream_file_identity
from src.infra.continual_findings_development_bindings import (
    VALIDATION_SECONDS,
    read_development_inputs,
    verify_development_input_bindings,
)


def _preflight(family: str, result: dict[str, Any]) -> None:
    {"sleep": sleep, "schedule": schedule, "combined": combined, "parent": parent}[
        family
    ].verify_result(result)


def _verifiers() -> dict[str, Any]:
    # Why this: retain the original complete bundle/reference gates unchanged;
    # only this outer composition knows their adapter modules.
    return {
        name: (
            lambda directory, family, adapter=adapter: original._verify_bundle(
                adapter, directory, family
            )
        )
        for name, (adapter, _) in original.ADAPTERS.items()
    }


def inspect_development_findings() -> dict[str, Any]:
    inputs = read_development_inputs(original.REPO_ROOT, _verifiers(), _preflight)
    result = build_development_ledger(inputs)
    verify_development_input_bindings(original.REPO_ROOT, inputs)
    return result


def publish_development_findings(output_file: Path) -> dict[str, Any]:
    started = monotonic()
    if output_file.exists():
        raise FileExistsError(f"development findings output already exists: {output_file}")
    body = inspect_development_findings()
    elapsed_seconds(
        monotonic() - started, "development findings prepublication", VALIDATION_SECONDS
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive(output_file, body)
    verify_development_input_bindings(original.REPO_ROOT, body["original_inputs"])
    elapsed_seconds(
        monotonic() - started, "complete development findings publication", VALIDATION_SECONDS
    )
    return {
        "output_file": str(output_file.resolve()),
        "identity": stream_file_identity(output_file),
        "coverage": body["coverage"],
        "validation_scope": "complete_current_development_inputs_and_pure_projection_no_confirmation_reader",
        "new_training_scoring_or_final_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True)
    options = parser.parse_args()
    print(
        json.dumps(
            publish_development_findings(options.output_file), sort_keys=True, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
