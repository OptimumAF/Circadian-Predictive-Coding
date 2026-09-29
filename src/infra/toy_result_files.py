"""Write a complete toy comparison result as a local, JSON-safe artifact.

Inputs are a finished report and destination path. This adapter does not run
training or choose a comparison metric.
"""

from __future__ import annotations

from pathlib import Path

from src.app.experiment_runner import ExperimentResult
from src.infra.local_result_json import write_local_result_json


def write_toy_result_json(result: ExperimentResult, path: str | Path) -> None:
    """Write once after training and final scoring, refusing an existing path."""
    write_local_result_json(result, path)
