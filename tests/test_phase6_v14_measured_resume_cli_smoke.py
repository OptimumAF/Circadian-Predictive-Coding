"""Exercise a checked interrupted v14 CLI resume and measured outputs."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_versioned_v14_bundle as bundle_cli
from src.infra.measured_observation_files import verify_wake_diagnostic_sidecar
from src.infra.v14_resume_files import V14ResumeFiles
from src.infra.versioned_run_files import verify_run_bundle


REPOSITORY = Path(__file__).resolve().parents[1]
TRAINING_SHA = "174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324"
OUTCOMES_SHA = "ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f"


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite measured v14 value: {value}")


def _run(module: str, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert isinstance(result, dict)
    return result


def _start_interrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, V14ResumeFiles]:
    run_id = "p61-v14-measured-resume"
    run = tmp_path / run_id
    original_train = bundle_cli.run_trigger_replay_training
    trained = 0

    def interrupt_after_one(*args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        if trained == 1:
            raise RuntimeError("injected trial interruption")
        trained += 1
        return original_train(*args, **kwargs)

    monkeypatch.setattr(bundle_cli, "run_trigger_replay_training", interrupt_after_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v14",
            "--run-id",
            run_id,
            "--output-root",
            str(tmp_path),
            "--resumable",
            "--capture-wake-diagnostics",
        ],
    )
    with pytest.raises(RuntimeError, match="injected trial interruption"):
        bundle_cli.main()
    files = V14ResumeFiles(tmp_path, run_id)
    interrupted = files.load()
    assert interrupted.status == "failed" and interrupted.next_trial_index == 1
    assert interrupted.next_cell == (47, "adaptive")
    assert interrupted.checkpoint_file is not None
    assert (files.directory / interrupted.checkpoint_file).is_file()
    assert not run.exists()
    return run, files


def _resume_and_assert_bundle(run: Path, files: V14ResumeFiles) -> dict[str, Any]:
    resumed = _run(
        "scripts.run_versioned_v14_bundle",
        "--run-id",
        run.name,
        "--output-root",
        str(run.parent),
        "--resume",
        "--capture-wake-diagnostics",
    )
    manifest_bytes = (run / "manifest.json").read_bytes()
    assert resumed == {"run": str(run), "manifest_sha256": sha256(manifest_bytes).hexdigest()}
    assert files.load().status == "completed" and files.load().next_trial_index == 6
    assert _run("scripts.run_versioned_v14_bundle", "--verify-run", str(run)) == {
        "run": str(run),
        "status": "completed",
    }
    assert verify_run_bundle(run)["status"] == "completed"
    assert sha256((run / "training.json").read_bytes()).hexdigest() == TRAINING_SHA
    assert sha256((run / "outcomes.json").read_bytes()).hexdigest() == OUTCOMES_SHA
    return resumed


def _assert_sidecar(run: Path) -> None:
    sidecar = verify_wake_diagnostic_sidecar(run)
    assert sidecar["file"]["record_count"] == 432
    measurements = run / "measurements-v1"
    assert {entry.name for entry in measurements.iterdir()} == {
        "measurement-manifest.json",
        "wake-diagnostics.jsonl",
    }
    wake = [
        json.loads(line, parse_constant=_reject_nonfinite)
        for line in (measurements / "wake-diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(wake) == 432
    assert all(isfinite(row["metric_value"]) for row in wake)


def _assert_measured_projection(run: Path) -> None:
    projected = _run("scripts.project_v14_observations", "--run-measured", str(run))
    projection = run / "observations-measured-v1"
    assert projected["projection"] == str(projection)
    assert _run("scripts.project_v14_observations", "--verify-measured-run", str(run)) == {
        "run": str(run),
        "projection_id": "v14_measured_observation_projection_v1",
    }
    metadata = json.loads((projection / "projection-manifest.json").read_text(encoding="utf-8"))
    expected = {
        "wake-epochs.jsonl": 144,
        "sleep-events.jsonl": 144,
        "topology.jsonl": 144,
        "replay.jsonl": 144,
        "validation.jsonl": 12,
        "role-access.jsonl": 516,
        "final-results.jsonl": 18,
        "summary.csv": 18,
        "wake-metrics.jsonl": 432,
        "wake-metrics.csv": 432,
    }
    assert {name: facts["record_count"] for name, facts in metadata["files"].items()} == expected
    assert {entry.name for entry in projection.iterdir()} == set(expected) | {
        "projection-manifest.json"
    }
    for name, facts in metadata["files"].items():
        assert sha256((projection / name).read_bytes()).hexdigest() == facts["sha256"]
    with (projection / "wake-metrics.csv").open(encoding="utf-8", newline="") as source:
        assert len(list(csv.DictReader(source))) == 432
    assert sha256((run / "training.json").read_bytes()).hexdigest() == TRAINING_SHA
    assert sha256((run / "outcomes.json").read_bytes()).hexdigest() == OUTCOMES_SHA


def test_should_resume_interrupted_v14_cli_and_publish_measured_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, files = _start_interrupted_run(tmp_path, monkeypatch)
    resumed = _resume_and_assert_bundle(run, files)
    _assert_sidecar(run)
    _assert_measured_projection(run)
    completed_resume = _run(
        "scripts.run_versioned_v14_bundle",
        "--run-id",
        run.name,
        "--output-root",
        str(tmp_path),
        "--resume",
        "--capture-wake-diagnostics",
    )
    assert completed_resume == resumed
    assert files.load().status == "completed"
