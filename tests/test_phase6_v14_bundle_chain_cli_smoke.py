"""Read one fixed v14 bundle through every public derived-artifact CLI."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
TRAINING_SHA = "174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324"
OUTCOMES_SHA = "ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f"
OBSERVATION_COUNTS = {
    "wake-epochs.jsonl": 144,
    "sleep-events.jsonl": 144,
    "topology.jsonl": 144,
    "replay.jsonl": 144,
    "validation.jsonl": 12,
    "role-access.jsonl": 516,
    "final-results.jsonl": 18,
    "summary.csv": 18,
}


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite v14 artifact value: {value}")


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


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(result, dict)
    return result


def _assert_files(directory: Path, expected: set[str]) -> None:
    assert {entry.name for entry in directory.iterdir()} == expected
    assert all((directory / name).is_file() for name in expected)


def _create_bundle(tmp_path: Path) -> Path:
    run = tmp_path / "p61-v14-chain"
    created = _run(
        "scripts.run_versioned_v14_bundle",
        "--run-id",
        run.name,
        "--output-root",
        str(tmp_path),
        "--preset",
        "fixed-v14",
    )
    assert created == {
        "run": str(run),
        "manifest_sha256": sha256((run / "manifest.json").read_bytes()).hexdigest(),
    }
    assert _run("scripts.run_versioned_v14_bundle", "--verify-run", str(run)) == {
        "run": str(run),
        "status": "completed",
    }
    _assert_files(run, {"manifest.json", "training.json", "outcomes.json"})
    manifest = _read_json(run / "manifest.json")
    assert manifest["run_id"] == run.name and manifest["status"] == "completed"
    assert set(manifest["files"]) == {"training", "outcomes"}
    assert manifest["files"]["training"]["sha256"] == TRAINING_SHA
    assert manifest["files"]["outcomes"]["sha256"] == OUTCOMES_SHA
    assert sha256((run / "training.json").read_bytes()).hexdigest() == TRAINING_SHA
    assert sha256((run / "outcomes.json").read_bytes()).hexdigest() == OUTCOMES_SHA
    assert len(_read_json(run / "training.json")["rows"]) == 6
    assert len(_read_json(run / "outcomes.json")["outcomes"]) == 6
    return run


def _assert_observations(run: Path) -> None:
    projected = _run("scripts.project_v14_observations", "--run", str(run))
    observations = run / "observations-v1"
    assert projected["projection"] == str(observations)
    assert _run("scripts.project_v14_observations", "--verify-run", str(run)) == {
        "run": str(run),
        "projection_id": "v14_observation_projection_v1",
    }
    _assert_files(observations, set(OBSERVATION_COUNTS) | {"projection-manifest.json"})
    observation_manifest = _read_json(observations / "projection-manifest.json")
    assert {
        name: facts["record_count"] for name, facts in observation_manifest["files"].items()
    } == (OBSERVATION_COUNTS)
    for name, count in OBSERVATION_COUNTS.items():
        path = observations / name
        assert (
            sha256(path.read_bytes()).hexdigest() == observation_manifest["files"][name]["sha256"]
        )
        if name.endswith(".jsonl"):
            assert (
                len(
                    [
                        json.loads(line, parse_constant=_reject_nonfinite)
                        for line in path.read_text(encoding="utf-8").splitlines()
                    ]
                )
                == count
            )
        else:
            with path.open(encoding="utf-8", newline="") as source:
                assert len(list(csv.DictReader(source))) == count


def _assert_report(run: Path) -> None:
    reported = _run("scripts.build_v14_artifact_report", "--run", str(run))
    report = run / "summary-report-v1"
    assert reported == {"report": str(report), "report_id": "v14_artifact_report_v1"}
    assert _run("scripts.build_v14_artifact_report", "--verify-run", str(run)) == {
        "run": str(run),
        "report_id": "v14_artifact_report_v1",
    }
    _assert_files(report, {"report-manifest.json", "summary.json", "summary.csv"})
    report_manifest = _read_json(report / "report-manifest.json")
    assert set(report_manifest["files"]) == {"summary.json", "summary.csv"}
    for name in report_manifest["files"]:
        assert (
            sha256((report / name).read_bytes()).hexdigest()
            == report_manifest["files"][name]["sha256"]
        )
    assert len(_read_json(report / "summary.json")["rows"]) == 9


def _assert_dashboard(run: Path) -> None:
    rendered = _run("scripts.build_v14_dashboard", "--run", str(run))
    dashboard = run / "dashboard-v1"
    assert rendered == {"dashboard": str(dashboard), "dashboard_id": "v14_verified_dashboard_v1"}
    assert _run("scripts.build_v14_dashboard", "--verify-run", str(run)) == {
        "run": str(run),
        "dashboard_id": "v14_verified_dashboard_v1",
    }
    figures = {
        "a-after-b-accuracy.png",
        "b-after-b-accuracy.png",
        "balanced-score.png",
        "signed-forgetting.png",
    }
    _assert_files(dashboard, figures | {"dashboard-manifest.json", "dashboard.html"})
    dashboard_manifest = _read_json(dashboard / "dashboard-manifest.json")
    assert set(dashboard_manifest["files"]) == figures | {"dashboard.html"}
    for name in figures:
        assert (dashboard / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for name in dashboard_manifest["files"]:
        assert (
            sha256((dashboard / name).read_bytes()).hexdigest()
            == dashboard_manifest["files"][name]["sha256"]
        )
    assert "<html" in (dashboard / "dashboard.html").read_text(encoding="utf-8").lower()
    assert sha256((run / "training.json").read_bytes()).hexdigest() == TRAINING_SHA
    assert sha256((run / "outcomes.json").read_bytes()).hexdigest() == OUTCOMES_SHA


def test_should_verify_complete_v14_bundle_to_dashboard_chain(tmp_path: Path) -> None:
    run = _create_bundle(tmp_path)
    _assert_observations(run)
    _assert_report(run)
    _assert_dashboard(run)
