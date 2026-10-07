"""P5.6b renders only a verified artifact table, never selected scores."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from src.app.v14_dashboard_projection import build_v14_dashboard
from src.infra import v14_dashboard_files


def _summary() -> dict[str, Any]:
    seeds = [47, 53]
    arms = ("periodic", "adaptive", "no_sleep")
    methods = ("backprop", "predictive_coding", "circadian_predictive_coding")
    rows = []
    for arm_index, arm in enumerate(arms):
        for method_index, method in enumerate(methods):
            center = 0.4 + arm_index * 0.05 + method_index * 0.02
            rows.append(
                {
                    "arm": arm,
                    "method": method,
                    "seed_count": 2,
                    "balanced_score": {
                        "mean": center,
                        "min": center - 0.1,
                        "max": center + 0.1,
                        "range": 0.2,
                    },
                    "signed_forgetting": {"mean": -0.1, "min": -0.2, "max": 0.0, "range": 0.2},
                    "a_after_b_accuracy": {
                        "mean": center,
                        "min": center - 0.1,
                        "max": center + 0.1,
                        "range": 0.2,
                    },
                    "b_after_b_accuracy": {
                        "mean": center,
                        "min": center - 0.1,
                        "max": center + 0.1,
                        "range": 0.2,
                    },
                }
            )
    return {
        "schema_id": "v14_artifact_report_v1",
        "benchmark_track": "numpy_synthetic_continual_v14",
        "protocol_id": "continual_trigger_replay_outcomes_v14",
        "source_run_id": "dashboard-fixture",
        "interpretation_scope": "descriptive_only_no_causal_attribution",
        "source": {
            "commit_sha": "a" * 40,
            "dirty": True,
            "workspace_sha256": "b" * 64,
            "unavailable_reason": None,
        },
        "seeds": seeds,
        "seed_count": 2,
        "expected_cells": 18,
        "completed_cells": 18,
        "failed_cells_in_bundle": 0,
        "external_attempt_failures": "not_recorded_in_bundle",
        "failure_scope": "published_completed_bundle_only",
        "rows": rows,
    }


def _verified_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run_path = tmp_path / "dashboard-fixture"
    report_path = run_path / "summary-report-v1"
    report_path.mkdir(parents=True)
    summary_bytes = (json.dumps(_summary(), sort_keys=True) + "\n").encode()
    csv_bytes = b"source-verified fixture CSV\n"
    (report_path / "summary.json").write_bytes(summary_bytes)
    (report_path / "summary.csv").write_bytes(csv_bytes)
    metadata = {
        "report_id": "v14_artifact_report_v1",
        "source_run_id": "dashboard-fixture",
        "files": {
            "summary.json": {"sha256": sha256(summary_bytes).hexdigest()},
            "summary.csv": {"sha256": sha256(csv_bytes).hexdigest()},
        },
    }
    (report_path / "report-manifest.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(v14_dashboard_files, "verify_v14_artifact_report", lambda _: metadata)
    return run_path


def test_should_render_all_fixed_rows_plots_and_provenance_without_ranking() -> None:
    summary = _summary()

    projection = build_v14_dashboard(summary)

    assert set(projection.files) == {
        "dashboard.html",
        "balanced-score.png",
        "signed-forgetting.png",
        "a-after-b-accuracy.png",
        "b-after-b-accuracy.png",
    }
    page = projection.files["dashboard.html"].decode("utf-8")
    assert page.count('class="data-row"') == 9
    assert "47, 53" in page
    assert "18 of 18" in page
    assert "published completed bundle only" in page
    assert "not recorded in this bundle" in page
    assert "continual_trigger_replay_outcomes_v14" in page
    assert "numpy_synthetic_continual_v14" in page
    assert "a" * 40 in page
    assert "descriptive" in page.lower()
    assert "historical dashboard" in page.lower()
    assert "winner" not in page.lower()
    assert page.index("periodic") < page.index("adaptive") < page.index("no_sleep")
    for name, payload in projection.files.items():
        if name.endswith(".png"):
            assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert projection.files == build_v14_dashboard(summary).files


def test_should_keep_historical_dashboard_provenance_warning_visible() -> None:
    historical_page = (Path(__file__).resolve().parents[1] / "docs" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "Historical snapshot:" in historical_page
    assert 'href="historical-benchmark-provenance.md"' in historical_page
    assert "not corrected-protocol results" in historical_page


@pytest.mark.parametrize(
    "change", ["row", "seed_count", "range", "nonfinite", "failure_scope", "track"]
)
def test_should_reject_incomplete_or_untruthful_report_summary(change: str) -> None:
    summary = deepcopy(_summary())
    if change == "row":
        summary["rows"].pop()
    elif change == "seed_count":
        summary["seed_count"] = 3
    elif change == "range":
        summary["rows"][0]["balanced_score"]["range"] = 0.9
    elif change == "nonfinite":
        summary["rows"][0]["balanced_score"]["mean"] = float("nan")
    elif change == "failure_scope":
        summary["external_attempt_failures"] = 0
    else:
        summary["benchmark_track"] = "unmatched_vision"

    with pytest.raises(ValueError):
        build_v14_dashboard(summary)


def test_should_verify_report_before_writing_and_reject_missing_source(tmp_path: Path) -> None:
    run_path = tmp_path / "missing-run"
    run_path.mkdir()

    with pytest.raises(ValueError, match="manifest"):
        v14_dashboard_files.write_v14_dashboard(run_path)

    assert not (run_path / "dashboard-v1").exists()


@pytest.mark.parametrize("edited_file", ["balanced-score.png", "dashboard.html"])
def test_should_publish_once_and_rederive_html_and_plot_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edited_file: str
) -> None:
    run_path = _verified_report(tmp_path, monkeypatch)

    directory = v14_dashboard_files.write_v14_dashboard(run_path)
    metadata = v14_dashboard_files.verify_v14_dashboard(run_path)

    assert metadata["source_run_id"] == "dashboard-fixture"
    assert metadata["source_report_id"] == "v14_artifact_report_v1"
    assert len(metadata["files"]) == 5
    with pytest.raises(FileExistsError):
        v14_dashboard_files.write_v14_dashboard(run_path)
    (directory / edited_file).write_bytes(b"hand-edited dashboard output")
    with pytest.raises(ValueError, match="dashboard"):
        v14_dashboard_files.verify_v14_dashboard(run_path)


def test_should_refuse_changed_report_after_verifier_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = _verified_report(tmp_path, monkeypatch)
    summary_path = run_path / "summary-report-v1" / "summary.json"
    changed = json.loads(summary_path.read_text(encoding="utf-8"))
    changed["rows"][0]["balanced_score"]["mean"] = 0.99
    summary_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        v14_dashboard_files.write_v14_dashboard(run_path)

    assert not (run_path / "dashboard-v1").exists()
