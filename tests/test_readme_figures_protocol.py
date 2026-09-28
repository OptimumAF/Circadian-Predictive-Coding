"""Protocol and overwrite gates for README figure regeneration."""

from __future__ import annotations

import csv
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

figures = import_module("scripts.generate_readme_figures")


def write_paired_inputs(tmp_path: Path) -> tuple[Path, Path]:
    summary_path = tmp_path / "tiny_summary.csv"
    result_path = tmp_path / "tiny.json"
    rows = [
        {
            "model_name": name,
            "test_accuracy_mean": 0.4 + index * 0.1,
            "train_samples_per_second_mean": 100.0 + index * 10,
            "inference_latency_p95_ms_mean": 2.0 + index,
            "balanced_score": 0.3 + index * 0.1,
        }
        for index, name in enumerate(figures.MODEL_ORDER)
    ]
    with summary_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result_path.write_text(
        json.dumps({
            "protocol_id": "vision_guard_separated_unmatched_v2",
            "dataset": {"name": "cifar100"},
            "summary": rows,
        }),
        encoding="utf-8",
    )
    return summary_path, result_path


def test_paired_result_manifest_must_match_summary_rows(tmp_path: Path) -> None:
    summary_path, result_path = write_paired_inputs(tmp_path)

    rows, protocol_id, paired_path = figures.load_figure_source(summary_path, None, False)
    assert len(rows) == 3
    assert protocol_id == "vision_guard_separated_unmatched_v2"
    assert paired_path == result_path

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"][0]["test_accuracy_mean"] = 0.99
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        figures.load_figure_source(summary_path, None, False)


def test_seeded_vision_protocol_keeps_a_distinct_figure_source(tmp_path: Path) -> None:
    summary_path, result_path = write_paired_inputs(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["protocol_id"] = "vision_guard_separated_seeded_unmatched_v3"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    _, protocol_id, paired_path = figures.load_figure_source(summary_path, None, False)

    assert protocol_id == "vision_guard_separated_seeded_unmatched_v3"
    assert paired_path == result_path


def test_unversioned_csv_requires_explicit_legacy_choice(tmp_path: Path) -> None:
    summary_path, result_path = write_paired_inputs(tmp_path)
    result_path.unlink()

    with pytest.raises(FileNotFoundError, match="Paired result JSON"):
        figures.load_figure_source(summary_path, None, False)
    _, protocol_id, paired_path = figures.load_figure_source(summary_path, None, True)
    assert protocol_id == "historical_unknown_v0"
    assert paired_path is None


def test_duplicate_csv_model_cannot_be_silently_mixed(tmp_path: Path) -> None:
    summary_path, _ = write_paired_inputs(tmp_path)
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    summary_path.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate model"):
        figures.load_figure_source(summary_path, None, False)


def test_existing_figure_stops_before_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path, _ = write_paired_inputs(tmp_path)
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    historical_path = output_dir / "benchmark_accuracy.png"
    historical_path.write_bytes(b"historical")
    monkeypatch.setattr(figures, "parse_args", lambda: make_args(summary_path, output_dir))
    monkeypatch.setattr(
        figures, "draw_compact_overview_chart",
        lambda **kwargs: pytest.fail("Rendering began before output preflight"),
    )

    with pytest.raises(FileExistsError, match="Figure output already exists"):
        figures.main()
    assert historical_path.read_bytes() == b"historical"


def test_tiny_corrected_render_writes_versioned_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path, _ = write_paired_inputs(tmp_path)
    output_dir = tmp_path / "new-figures"
    monkeypatch.setattr(figures, "parse_args", lambda: make_args(summary_path, output_dir))

    figures.main()

    assert all((output_dir / name).exists() for name in figures.FIGURE_FILENAMES)
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["protocol_id"] == "vision_guard_separated_unmatched_v2"
    assert len(provenance["summary_sha256"]) == 64
    assert provenance["sleep_gif_status"].startswith("illustrative")


def make_args(summary_path: Path, output_dir: Path) -> Any:
    return SimpleNamespace(
        summary_csv=str(summary_path), result_json=None, legacy_unversioned=False,
        output_dir=str(output_dir), sleep_cycles=1, start_hidden=8, splits=0, prunes=0,
    )
