"""P5.6 tables derive every cell from a completed, verified v14 bundle."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from src.app.v14_artifact_report import build_v14_artifact_report
from src.infra import v14_artifact_report_files


def _records() -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = [47, 53]
    arms = ["periodic", "adaptive", "no_sleep"]
    methods = ["backprop", "predictive_coding", "circadian_predictive_coding"]
    manifest = {
        "run_id": "report-fixture",
        "status": "completed",
        "protocol_versions": {"outcomes": "continual_trigger_replay_outcomes_v14"},
        "resolved_config": {
            "seeds": seeds,
            "arms": arms,
            "arrived": {"training": {"model_order": methods}},
        },
        "source": {
            "commit_sha": "a" * 40,
            "dirty": True,
            "workspace_sha256": "b" * 64,
            "unavailable_reason": None,
        },
        "files": {
            "training": {"path": "training.json", "sha256": "c" * 64},
            "outcomes": {"path": "outcomes.json", "sha256": "d" * 64},
        },
    }
    outcomes = {
        "protocol_id": "continual_trigger_replay_outcomes_v14",
        "outcomes": [
            {
                "seed": seed,
                "arm": arm,
                "methods": [
                    {
                        "method": method,
                        "balanced_score": 0.5 + seed_index * 0.1 + arm_index * 0.01,
                        "signed_forgetting": -0.1 + seed_index * 0.02,
                        "a_after_b_accuracy": 0.6 + seed_index * 0.1,
                        "b_after_b_accuracy": 0.7 + arm_index * 0.01,
                    }
                    for method in methods
                ],
            }
            for seed_index, seed in enumerate(seeds)
            for arm_index, arm in enumerate(arms)
        ],
    }
    return manifest, outcomes


def _source_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    manifest, outcomes = _records()
    run_path = tmp_path / manifest["run_id"]
    run_path.mkdir()
    outcome_bytes = (json.dumps(outcomes, sort_keys=True) + "\n").encode()
    manifest["files"]["outcomes"]["sha256"] = sha256(outcome_bytes).hexdigest()
    (run_path / "outcomes.json").write_bytes(outcome_bytes)
    (run_path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(v14_artifact_report_files, "verify_run_bundle", lambda _path: manifest)
    return run_path


def test_should_preserve_all_arms_methods_and_seed_spread_without_winner() -> None:
    manifest, outcomes = _records()

    report = build_v14_artifact_report(manifest, outcomes)

    summary = report.summary
    assert summary["benchmark_track"] == "numpy_synthetic_continual_v14"
    assert summary["protocol_id"] == "continual_trigger_replay_outcomes_v14"
    assert summary["source"]["commit_sha"] == "a" * 40
    assert summary["source"]["dirty"] is True
    assert summary["seeds"] == [47, 53]
    assert summary["seed_count"] == 2
    assert summary["expected_cells"] == summary["completed_cells"] == 18
    assert summary["failed_cells_in_bundle"] == 0
    assert summary["external_attempt_failures"] == "not_recorded_in_bundle"
    assert summary["failure_scope"] == "published_completed_bundle_only"
    assert summary["interpretation_scope"] == "descriptive_only_no_causal_attribution"
    assert [(row["arm"], row["method"]) for row in summary["rows"]] == [
        (arm, method)
        for arm in manifest["resolved_config"]["arms"]
        for method in manifest["resolved_config"]["arrived"]["training"]["model_order"]
    ]
    first = summary["rows"][0]
    assert first["seed_count"] == 2
    assert first["balanced_score"]["mean"] == pytest.approx(0.55)
    assert first["balanced_score"]["min"] == pytest.approx(0.5)
    assert first["balanced_score"]["max"] == pytest.approx(0.6)
    assert first["balanced_score"]["range"] == pytest.approx(0.1)
    assert b"balanced_score_range" in report.csv_bytes
    assert report.json_bytes == build_v14_artifact_report(manifest, outcomes).json_bytes


@pytest.mark.parametrize("change", ["missing", "reordered_methods", "nonfinite", "incomplete"])
def test_should_reject_malformed_or_incomplete_source_records(change: str) -> None:
    manifest, outcomes = _records()
    if change == "missing":
        outcomes["outcomes"].pop()
    elif change == "reordered_methods":
        outcomes["outcomes"][0]["methods"].reverse()
    elif change == "nonfinite":
        outcomes["outcomes"][0]["methods"][0]["balanced_score"] = float("nan")
    else:
        manifest["status"] = "incomplete"

    with pytest.raises(ValueError):
        build_v14_artifact_report(manifest, outcomes)


def test_should_verify_source_before_writing_and_reject_missing_bundle(tmp_path: Path) -> None:
    run_path = tmp_path / "missing-run"
    run_path.mkdir()

    with pytest.raises(ValueError, match="manifest"):
        v14_artifact_report_files.write_v14_artifact_report(run_path)

    assert not (run_path / "summary-report-v1").exists()


def test_should_write_once_and_verify_exact_derived_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = _source_directory(tmp_path, monkeypatch)

    destination = v14_artifact_report_files.write_v14_artifact_report(run_path)
    metadata = v14_artifact_report_files.verify_v14_artifact_report(run_path)

    assert metadata["source_run_id"] == "report-fixture"
    assert set(destination.iterdir()) == {
        destination / "summary.json",
        destination / "summary.csv",
        destination / "report-manifest.json",
    }
    with pytest.raises(FileExistsError):
        v14_artifact_report_files.write_v14_artifact_report(run_path)
    (destination / "summary.csv").write_bytes(b"hand-edited chart input\n")
    with pytest.raises(ValueError, match="report"):
        v14_artifact_report_files.verify_v14_artifact_report(run_path)


def test_should_reject_changed_source_even_when_verifier_was_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = _source_directory(tmp_path, monkeypatch)
    outcomes_path = run_path / "outcomes.json"
    changed = json.loads(outcomes_path.read_text(encoding="utf-8"))
    changed["outcomes"][0]["methods"][0]["balanced_score"] = 0.99
    outcomes_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        v14_artifact_report_files.write_v14_artifact_report(run_path)

    assert not (run_path / "summary-report-v1").exists()


def test_should_reject_external_attempt_claim_when_manifest_has_no_git_source() -> None:
    manifest, outcomes = _records()
    missing_git = deepcopy(manifest)
    missing_git["source"] = {
        "commit_sha": None,
        "dirty": None,
        "workspace_sha256": None,
        "unavailable_reason": "Git metadata unavailable in fixture",
    }

    summary = build_v14_artifact_report(missing_git, outcomes).summary

    assert summary["source"]["commit_sha"] is None
    assert summary["source"]["unavailable_reason"] == "Git metadata unavailable in fixture"
    assert summary["external_attempt_failures"] == "not_recorded_in_bundle"
