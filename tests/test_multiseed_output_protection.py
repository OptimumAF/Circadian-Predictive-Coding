"""Existing multi-seed result artifacts are never replaced by a new run."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.app.resnet50_benchmark import ModelSpeedReport

script = import_module("scripts.run_multiseed_resnet_benchmark")


def test_existing_output_stops_multiseed_run_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    prefix = tmp_path / "historical"
    existing = prefix.with_name("historical_summary.csv")
    existing.write_text("historical", encoding="utf-8")
    args = SimpleNamespace(output_prefix=str(prefix), seeds="7")
    monkeypatch.setattr(script, "build_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(
        script, "build_base_config",
        lambda args: pytest.fail("Training setup must not start before output preflight"),
    )

    with pytest.raises(FileExistsError, match="Multi-seed output already exists"):
        script.main()
    assert existing.read_text(encoding="utf-8") == "historical"


def test_winner_selection_receives_no_final_test_metrics() -> None:
    summaries = [
        {
            "model_name": "validation_winner", "benchmark_track": "unmatched_reference",
            "validation_accuracy_mean": 0.9,
            "train_samples_per_second_mean": 100.0,
            "inference_samples_per_second_mean": 100.0,
            "test_accuracy_mean": 0.1,
        },
        {
            "model_name": "test_winner", "benchmark_track": "unmatched_reference",
            "validation_accuracy_mean": 0.2,
            "train_samples_per_second_mean": 100.0,
            "inference_samples_per_second_mean": 100.0,
            "test_accuracy_mean": 0.99,
        },
    ]

    selection_rows = script.validation_selection_rows(summaries)

    assert all("test_accuracy_mean" not in row for row in selection_rows)
    assert script.best_by_metric(selection_rows, "validation_accuracy_mean") == (
        "validation_winner"
    )
    assert script.compute_efficiency_winner(selection_rows) == "validation_winner"
    assert summaries[0]["test_accuracy_mean"] == 0.1


def test_winner_selection_rejects_mixed_benchmark_tracks() -> None:
    rows = [
        {
            "model_name": "matched", "benchmark_track": "frozen_shared_representation",
            "validation_accuracy_mean": 0.8,
            "train_samples_per_second_mean": 100.0,
            "inference_samples_per_second_mean": 100.0,
        },
        {
            "model_name": "practical", "benchmark_track": "end_to_end_backprop",
            "validation_accuracy_mean": 0.7,
            "train_samples_per_second_mean": 100.0,
            "inference_samples_per_second_mean": 100.0,
        },
    ]
    with pytest.raises(ValueError, match="different benchmark tracks"):
        script.compute_efficiency_winner(rows)
    with pytest.raises(ValueError, match="different benchmark tracks"):
        script.best_by_metric(rows, "validation_accuracy_mean")


def test_winner_selection_requires_track_metadata() -> None:
    row = {
        "model_name": "unknown", "validation_accuracy_mean": 0.5,
        "train_samples_per_second_mean": 100.0,
        "inference_samples_per_second_mean": 100.0,
    }
    with pytest.raises(ValueError, match="explicit benchmark_track"):
        script.compute_efficiency_winner([row])


def test_multiseed_rows_retain_track_and_head_metadata() -> None:
    report = ModelSpeedReport(
        model_name="BackpropResNet50", epochs_ran=1,
        final_metric_name="cross_entropy", final_metric_value=0.5,
        validation_accuracy=0.6, test_accuracy=0.5,
        train_seconds=1.0, train_samples_per_second=8.0,
        mean_train_step_ms=1.0, inference_latency_mean_ms=1.0,
        inference_latency_p95_ms=1.0, inference_samples_per_second=8.0,
        total_parameters=100, trainable_parameters=100,
        benchmark_track="end_to_end_backprop", backbone_trainable=True,
        backbone_pretraining="none", head_type="linear",
        final_cross_entropy=0.5,
    )

    row = script.report_to_row(7, report)
    summary = script.aggregate_rows([row])[0]

    for key in ("benchmark_track", "backbone_trainable", "backbone_pretraining", "head_type"):
        assert row[key] == summary[key]
    assert summary["benchmark_track"] == "end_to_end_backprop"

    pc_report = replace(
        report,
        model_name="PredictiveCodingResNet50",
        head_type="predictive_coding",
        final_energy=0.17,
        training_energy_id="torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1",
    )
    pc_row = script.report_to_row(7, pc_report)
    pc_summary = script.aggregate_rows([pc_row])[0]
    assert pc_row["final_energy"] == 0.17
    assert pc_row["training_energy_id"] == pc_summary["training_energy_id"]
    with pytest.raises(ValueError, match="training_energy_id"):
        script.aggregate_rows([pc_row, {**pc_row, "training_energy_id": "different_formula"}])
