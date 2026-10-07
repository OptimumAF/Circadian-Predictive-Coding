"""P5.2b2 binds measured wake diagnostics to globally scored v14 runs."""

from __future__ import annotations

import csv
from hashlib import sha256
from io import StringIO
import json
from math import isfinite
from pathlib import Path
from shutil import copyfile
from typing import Any

import pytest

import scripts.run_versioned_v14_bundle as bundle_adapter
from scripts.run_versioned_v14_bundle import run_versioned_v14_bundle
from src.infra.measured_observation_files import (
    verify_measured_observation_projection,
    verify_wake_diagnostic_sidecar,
    write_measured_observation_projection,
)
from src.infra.versioned_run_files import verify_run_bundle

from v14_portable_value_fixtures import fixed_v14_payloads, recorded_v14_environment


@pytest.fixture(scope="module")
def measured_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("v14-measured")
    return (
        run_versioned_v14_bundle(root, "measured-v14-a", Path.cwd(), capture_wake_diagnostics=True),
        run_versioned_v14_bundle(root, "measured-v14-b", Path.cwd(), capture_wake_diagnostics=True),
    )


def _records(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.splitlines()]


def test_should_bind_complete_measured_grid_to_unchanged_v14_outputs(
    measured_runs: tuple[Path, Path],
) -> None:
    first, second = measured_runs
    for run in measured_runs:
        assert verify_run_bundle(run)["status"] == "completed"
        metadata = verify_wake_diagnostic_sidecar(run)
        assert metadata["observation_id"] == "v14_wake_diagnostics_v1"
        rows = _records((run / "measurements-v1/wake-diagnostics.jsonl").read_bytes())
        assert len(rows) == 432
        assert {
            method: sum(row["method"] == method for row in rows)
            for method in ("backprop", "predictive_coding", "circadian_predictive_coding")
        } == {
            "backprop": 144,
            "predictive_coding": 144,
            "circadian_predictive_coding": 144,
        }
        assert all(isfinite(row["metric_value"]) for row in rows)
        assert rows[0]["seed"] == 47 and rows[0]["arm"] == "periodic"
        training, outcomes, diagnostics = fixed_v14_payloads()
        assert (run / "training.json").read_bytes() == training
        assert (run / "outcomes.json").read_bytes() == outcomes
        assert (run / "measurements-v1/wake-diagnostics.jsonl").read_bytes() == diagnostics
    assert (first / "measurements-v1/wake-diagnostics.jsonl").read_bytes() == (
        second / "measurements-v1/wake-diagnostics.jsonl"
    ).read_bytes()


@recorded_v14_environment
def test_should_reproduce_recorded_measured_v14_outputs(measured_runs: tuple[Path, Path]) -> None:
    for run in measured_runs:
        assert sha256((run / "training.json").read_bytes()).hexdigest() == (
            "174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324"
        )
        assert sha256((run / "outcomes.json").read_bytes()).hexdigest() == (
            "ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f"
        )


def test_should_derive_measured_jsonl_and_csv_only_after_complete_score(
    measured_runs: tuple[Path, Path],
) -> None:
    projections = [write_measured_observation_projection(run) for run in measured_runs]
    metadata = [verify_measured_observation_projection(run) for run in measured_runs]
    assert metadata[0]["files"] == metadata[1]["files"]
    assert metadata[0]["measurement_manifest_sha256"] != metadata[1]["measurement_manifest_sha256"]
    first = projections[0]
    assert {name: info["record_count"] for name, info in metadata[0]["files"].items()} == {
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
    jsonl = _records((first / "wake-metrics.jsonl").read_bytes())
    csv_rows = list(csv.DictReader(StringIO((first / "wake-metrics.csv").read_text())))
    assert len(jsonl) == len(csv_rows) == 432
    assert _records((first / "replay.jsonl").read_bytes())[0]["retention"]["example_count"] == 8
    assert "circadian_replay_exposure" in _records((first / "final-results.jsonl").read_bytes())[0]
    for source, summary in zip(jsonl, csv_rows, strict=True):
        assert int(summary["seed"]) == source["seed"]
        assert summary["method"] == source["method"]
        assert float(summary["metric_value"]) == source["metric_value"]
    with pytest.raises(FileExistsError):
        write_measured_observation_projection(measured_runs[0])


def test_should_reject_changed_sidecar_and_rehashed_projection_file(
    measured_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    # Why this: tamper checks must be runnable alone and must not mutate the
    # module-scoped source shared with the projection and repeat tests.
    source = measured_runs[0]
    run = tmp_path / source.name
    for name in (
        "manifest.json",
        "training.json",
        "outcomes.json",
        "measurements-v1/measurement-manifest.json",
        "measurements-v1/wake-diagnostics.jsonl",
    ):
        destination = run / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source / name, destination)
    directory = write_measured_observation_projection(run)
    manifest_path = run / "measurements-v1/measurement-manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest_path.unlink()
    with pytest.raises(ValueError, match="wake diagnostic"):
        verify_wake_diagnostic_sidecar(run)
    manifest_path.write_bytes(original_manifest)

    data_path = run / "measurements-v1/wake-diagnostics.jsonl"
    original = data_path.read_bytes()
    data_path.write_bytes(original.replace(b'"metric_name":"loss"', b'"metric_name":"fake"', 1))
    with pytest.raises(ValueError, match="wake diagnostic"):
        verify_wake_diagnostic_sidecar(run)
    data_path.write_bytes(original)

    csv_path = directory / "wake-metrics.csv"
    changed = csv_path.read_bytes().replace(b"loss", b"fake", 1)
    csv_path.write_bytes(changed)
    metadata = json.loads((directory / "projection-manifest.json").read_bytes())
    metadata["files"]["wake-metrics.csv"]["sha256"] = sha256(changed).hexdigest()
    (directory / "projection-manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="projection"):
        verify_measured_observation_projection(run)


def test_should_not_publish_measurements_when_global_scoring_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_scoring(_study: object) -> None:
        raise RuntimeError("final gate failed")

    monkeypatch.setattr(bundle_adapter, "score_trigger_replay_training_study", fail_scoring)
    with pytest.raises(RuntimeError, match="final gate failed"):
        bundle_adapter.run_versioned_v14_bundle(
            tmp_path, "failed-measured-v14", Path.cwd(), capture_wake_diagnostics=True
        )
    assert not (tmp_path / "failed-measured-v14").exists()
