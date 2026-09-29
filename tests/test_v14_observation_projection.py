"""P5.2a derives only observations actually present in verified v14 output."""

from __future__ import annotations

import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_versioned_v14_bundle import run_versioned_v14_bundle
from src.app.v14_observation_projection import build_v14_observation_projection
from src.infra.observation_projection_files import (
    verify_observation_projection,
    write_observation_projection,
)
from src.infra.versioned_run_files import verify_run_bundle


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_root = tmp_path_factory.mktemp("v14-observations")
    return run_versioned_v14_bundle(output_root, "p52-v14-fixture", Path.cwd())


def _source_records(run_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_run_bundle(run_path)
    return (
        json.loads((run_path / "training.json").read_bytes()),
        json.loads((run_path / "outcomes.json").read_bytes()),
    )


def _rows(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.splitlines()]


def test_should_project_every_observed_cell_without_inventing_wake_metrics(
    completed_run: Path,
) -> None:
    training, outcomes = _source_records(completed_run)
    projection = build_v14_observation_projection(training, outcomes)
    assert projection.counts == {
        "wake-epochs.jsonl": 144,
        "sleep-events.jsonl": 144,
        "topology.jsonl": 144,
        "replay.jsonl": 144,
        "validation.jsonl": 12,
        "role-access.jsonl": 516,
        "final-results.jsonl": 18,
        "summary.csv": 18,
    }
    wake = _rows(projection.files["wake-epochs.jsonl"])
    assert all(row["wake_metrics_status"] == "unavailable_not_recorded" for row in wake)
    assert all("loss" not in row and "energy" not in row for row in wake)
    assert [(row["seed"], row["arm"], row["phase"], row["epoch"]) for row in wake[:2]] == [
        (47, "periodic", "a", 1),
        (47, "periodic", "a", 2),
    ]
    sleep = _rows(projection.files["sleep-events.jsonl"])
    assert sum(row["event"]["outcome"] == "accepted" for row in sleep) == 12
    assert sum(row["event"]["outcome"] == "skipped" for row in sleep) == 132
    replay = _rows(projection.files["replay.jsonl"])
    assert (
        sum(sum(item["optimizer_updates"] for item in row["applied_by_method"]) for row in replay)
        == 72
    )
    validation = _rows(projection.files["validation.jsonl"])
    assert len(validation) == 12
    assert all(row["guard_decision"]["performed"] for row in validation)
    assert all(row["guard_decision"]["role_hash"] for row in validation)
    accesses = _rows(projection.files["role-access.jsonl"])
    assert len(accesses) == 516
    assert all(row["access"]["role"] != "final_test" for row in accesses)
    assert len(_rows(projection.files["final-results.jsonl"])) == 18
    assert projection.files == build_v14_observation_projection(training, outcomes).files


def test_should_derive_csv_only_from_raw_final_rows(completed_run: Path) -> None:
    training, outcomes = _source_records(completed_run)
    projection = build_v14_observation_projection(training, outcomes)
    final_rows = _rows(projection.files["final-results.jsonl"])
    csv_rows = list(csv.DictReader(StringIO(projection.files["summary.csv"].decode())))
    assert len(csv_rows) == len(final_rows) == 18
    for source, summary in zip(final_rows, csv_rows, strict=True):
        assert int(summary["seed"]) == source["seed"]
        assert summary["arm"] == source["arm"]
        assert summary["method"] == source["method"]["method"]
        assert float(summary["balanced_score"]) == source["method"]["balanced_score"]
    assert sha256((completed_run / "outcomes.json").read_bytes()).hexdigest() == (
        "ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f"
    )


def test_should_write_once_verify_and_reject_rehashed_forgery(completed_run: Path) -> None:
    directory = write_observation_projection(completed_run)
    metadata = verify_observation_projection(completed_run)
    assert metadata["source_run_id"] == "p52-v14-fixture"
    assert (completed_run / "training.json").exists()
    assert (completed_run / "outcomes.json").exists()
    with pytest.raises(FileExistsError):
        write_observation_projection(completed_run)

    wake_path = directory / "wake-epochs.jsonl"
    original = wake_path.read_bytes()
    wake_path.unlink()
    with pytest.raises(ValueError, match="projection"):
        verify_observation_projection(completed_run)
    wake_path.write_bytes(original)
    changed = original.replace(b"unavailable_not_recorded", b"fabricated_metric_recorded", 1)
    wake_path.write_bytes(changed)
    metadata["files"]["wake-epochs.jsonl"]["sha256"] = sha256(changed).hexdigest()
    (directory / "projection-manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="projection"):
        verify_observation_projection(completed_run)


def test_should_reject_changed_source_bundle_before_projection(
    tmp_path: Path, completed_run: Path
) -> None:
    # A copied pair with an incompatible directory ID must fail the P5.1
    # verifier before the P5.2 projection can consume any result record.
    copied = tmp_path / "wrong-run-id"
    copied.mkdir()
    for name in ("manifest.json", "training.json", "outcomes.json"):
        (copied / name).write_bytes((completed_run / name).read_bytes())
    with pytest.raises(ValueError, match="directory"):
        write_observation_projection(copied)


def test_should_reject_reordered_epoch_even_with_updated_source_hash(
    tmp_path: Path, completed_run: Path
) -> None:
    copied = tmp_path / completed_run.name
    copied.mkdir()
    for name in ("manifest.json", "training.json", "outcomes.json"):
        (copied / name).write_bytes((completed_run / name).read_bytes())
    training = json.loads((copied / "training.json").read_bytes())
    items = training["rows"][0]["opportunities"]
    items[0], items[1] = items[1], items[0]
    changed = (json.dumps(training, indent=2, sort_keys=True) + "\n").encode()
    (copied / "training.json").write_bytes(changed)
    manifest = json.loads((copied / "manifest.json").read_bytes())
    manifest["files"]["training"]["sha256"] = sha256(changed).hexdigest()
    (copied / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="reordered"):
        write_observation_projection(copied)
    assert not (copied / "observations-v1").exists()
