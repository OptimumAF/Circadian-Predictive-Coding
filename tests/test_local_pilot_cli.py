"""Public preflight returns useful refusal codes and creates no artifacts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.adapters.local_pilot_cli import run_preflight


def test_should_report_default_local_simulated_cpu_limits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_preflight([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "accepted"
    assert result["scope"] == "resource_preflight_only"
    assert result["request"]["execution_target"] == "local"
    assert result["request"]["environment_kind"] == "simulation"
    assert result["request"]["device"] == "cpu"
    assert result["limits"] == {
        "wall_seconds": 60.0,
        "training_updates": 512,
        "environment_steps": 1024,
        "cpu_threads": 1,
        "process_memory_bytes": 268435456,
        "replay_bytes": 1048576,
        "model_download_bytes": 0,
        "storage_bytes": 33554432,
        "gpu_memory_bytes": 0,
    }


@pytest.mark.parametrize(
    "arguments,error",
    [
        ("--wall-seconds nan", "wall_seconds"),
        ("--training-updates 513", "training_updates"),
        ("--model-download-bytes 1", "model_download_bytes"),
        ("--execution-target cloud", "local execution"),
    ],
)
def test_should_emit_useful_refusal_json(
    arguments: str, error: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_preflight(arguments.split()) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "rejected" and error in result["error"]


def test_should_run_public_entrypoint_without_creating_files(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(repository))
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_local_pilot_preflight", "--storage-bytes", "33554433"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 2 and completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "rejected"
    assert list(tmp_path.iterdir()) == []
