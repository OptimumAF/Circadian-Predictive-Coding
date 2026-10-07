"""Write a complete toy comparison result as a local, JSON-safe artifact.

Inputs are a finished report and destination path. This adapter does not run
training or choose a comparison metric.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from src.app.experiment_runner import ExperimentResult
from src.infra.local_result_json import write_local_json_payload, write_local_result_json


def write_toy_result_json(
    result: ExperimentResult,
    path: str | Path,
    resolved_config: Mapping[str, object] | None = None,
) -> None:
    """Write once after training and final scoring, refusing an existing path."""
    if resolved_config is None:
        write_local_result_json(result, path)
        return
    # Why this: keep old top-level report keys and direct writer calls intact.
    payload = asdict(result)
    payload["resolved_config"] = dict(resolved_config)
    write_local_json_payload(payload, path)
