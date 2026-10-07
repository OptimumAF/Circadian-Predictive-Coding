"""Regression checks for validation-only decisions in legacy tuning scripts."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from src.app.resnet50_benchmark import ModelDevelopmentReport, ResNet50BenchmarkConfig

# Scripts are run as top-level files by mypy; dynamic imports avoid loading them a second
# time under the scripts namespace during the repository-wide type check.
run_circadian_policy_sweep = import_module("scripts.run_circadian_policy_sweep")
run_pareto_hard_tuning = import_module("scripts.run_pareto_hard_tuning")


def make_trial(name: str, validation_accuracy: float, test_accuracy: float) -> dict[str, Any]:
    return {
        "trial": name,
        "params": {},
        "report": {
            "model_name": "circadian",
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "train_samples_per_second": 100.0,
            "inference_samples_per_second": 200.0,
        },
    }


def sample_development_report() -> ModelDevelopmentReport:
    return ModelDevelopmentReport(
        model_name="circadian", epochs_ran=1, validation_accuracy=0.8,
        validation_cross_entropy=0.5, train_seconds=2.0,
        train_samples_per_second=100.0, mean_train_step_ms=10.0,
        inference_latency_mean_ms=1.0, inference_latency_p95_ms=2.0,
        inference_samples_per_second=200.0, total_parameters=2_000_000,
        trainable_parameters=1_000_000,
        benchmark_track="unmatched_reference", backbone_trainable=False,
        backbone_pretraining="imagenet", head_type="circadian_predictive_coding",
        final_energy=0.17,
        training_energy_id="torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1",
    )


def test_policy_sweep_selection_ignores_final_test_accuracy() -> None:
    trials = [make_trial("validation_winner", 0.9, 0.1), make_trial("test_winner", 0.2, 0.99)]

    summary = run_circadian_policy_sweep.summarize_trials(trials)

    assert summary["best_validation_accuracy"]["trial"] == "validation_winner"
    assert summary["top10_by_validation_accuracy"][0]["trial"] == "validation_winner"
    assert summary["best_balanced"]["trial"] == "validation_winner"
    assert trials[0]["report"]["balanced_score"] > trials[1]["report"]["balanced_score"]
    assert trials[1]["report"]["test_accuracy"] == 0.99


def test_pareto_selection_ignores_final_test_accuracy() -> None:
    trials = [make_trial("validation_winner", 0.9, 0.1), make_trial("test_winner", 0.2, 0.99)]

    summary = run_pareto_hard_tuning.summarize_model_trials(trials)
    global_best = run_pareto_hard_tuning.best_from_all_trials(
        [trial["report"] for trial in trials], "validation_accuracy"
    )

    assert summary["top10_by_validation_accuracy"][0]["trial"] == "validation_winner"
    assert summary["best_balanced"]["trial"] == "validation_winner"
    assert [trial["trial"] for trial in summary["pareto_front"]] == ["validation_winner"]
    assert global_best is trials[0]["report"]
    assert trials[1]["report"]["test_accuracy"] == 0.99


def test_pareto_efficiency_uses_validation_accuracy() -> None:
    report = sample_development_report()

    row = run_pareto_hard_tuning.report_to_dict(report)
    aggregate = run_pareto_hard_tuning._aggregate_reports([row])

    assert "test_accuracy" not in row
    assert row["validation_cross_entropy"] == 0.5
    assert row["validation_accuracy_per_train_second"] == pytest.approx(0.4)
    assert row["validation_accuracy_per_million_trainable_params"] == pytest.approx(0.8)
    assert aggregate["validation_accuracy"] == pytest.approx(0.8)
    assert "test_accuracy" not in aggregate
    assert aggregate["validation_cross_entropy"] == pytest.approx(0.5)
    assert row["backbone_pretraining"] == "imagenet"
    assert aggregate["head_type"] == "circadian_predictive_coding"
    assert row["training_energy_id"] == aggregate["training_energy_id"]
    with pytest.raises(ValueError, match="training_energy_id"):
        run_pareto_hard_tuning._aggregate_reports(
            [row, {**row, "training_energy_id": "different_formula"}]
        )


def test_policy_trial_report_has_no_final_test_fields() -> None:
    row = run_circadian_policy_sweep.report_to_dict(sample_development_report())
    assert row["validation_cross_entropy"] == 0.5
    assert "test_accuracy" not in row
    assert "final_cross_entropy" not in row
    assert row["benchmark_track"] == "unmatched_reference"
    assert row["training_energy_id"] == "torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1"


class SealedTestLoaders:
    train_loader = object()
    validation_loader = object()
    guard_loader = object()
    num_classes = 3
    split_hashes = {"train": "a", "validation": "b", "guard": "d", "test": "c"}

    @property
    def test_loader(self) -> Any:
        pytest.fail("A tuning trial opened the final test loader")


def test_policy_sweep_passes_training_guard_and_outer_validation_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output_path = tmp_path / "policy.json"
    monkeypatch.setattr(run_circadian_policy_sweep, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(run_circadian_policy_sweep, "require_torch", object)
    monkeypatch.setattr(run_circadian_policy_sweep, "_set_seed", lambda *args: None)
    monkeypatch.setattr(run_circadian_policy_sweep, "_resolve_device", lambda *args: "cpu")
    monkeypatch.setattr(
        run_circadian_policy_sweep, "build_synthetic_vision_dataloaders",
        lambda *args: SealedTestLoaders(),
    )
    monkeypatch.setattr(run_circadian_policy_sweep, "build_candidates", lambda *args: [{}])

    def candidate(**kwargs: Any) -> ModelDevelopmentReport:
        assert kwargs["variant"] == "circadian"
        assert not hasattr(kwargs["loaders"], "test_loader")
        assert kwargs["loaders"].guard_loader is not kwargs["loaders"].validation_loader
        return sample_development_report()

    monkeypatch.setattr(run_circadian_policy_sweep, "benchmark_validation_candidate", candidate)
    run_circadian_policy_sweep.main()
    output = output_path.read_text(encoding="utf-8")
    assert '"final_test_usage": "none"' in output
    assert '"final_test_confirmation": "pending"' in output
    assert '"split_hashes"' in output
    assert '"planned_max_training_updates": 800' in output
    assert '"max_planned_training_updates": 1000' in output
    assert '"test_accuracy"' not in output


def test_pareto_trial_passes_training_guard_and_outer_validation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pareto_hard_tuning, "_set_seed", lambda *args: None)
    monkeypatch.setattr(
        run_pareto_hard_tuning, "_build_loaders_for_config",
        lambda *args: SealedTestLoaders(),
    )

    def candidate(**kwargs: Any) -> ModelDevelopmentReport:
        assert kwargs["variant"] == "backprop"
        assert not hasattr(kwargs["loaders"], "test_loader")
        assert kwargs["loaders"].guard_loader is not kwargs["loaders"].validation_loader
        return sample_development_report()

    monkeypatch.setattr(run_pareto_hard_tuning, "benchmark_validation_candidate", candidate)
    trial = run_pareto_hard_tuning._run_multiseed_trial(
        base=ResNet50BenchmarkConfig(), override={}, seeds=(7,),
        torch_module=object(), device="cpu", variant="backprop",
    )
    assert trial["report"]["validation_accuracy"] == pytest.approx(0.8)
    assert "test_accuracy" not in trial["report"]
    assert trial["seed_reports"][0]["split_hashes"] == SealedTestLoaders.split_hashes


@pytest.mark.parametrize("module", [run_circadian_policy_sweep, run_pareto_hard_tuning])
def test_sweep_refuses_to_overwrite_existing_output(
    module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "historical.json"
    output_path.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(module, "require_torch", lambda: pytest.fail("Sweep should not start."))

    with pytest.raises(FileExistsError, match="Sweep output already exists"):
        module.main()

    assert output_path.read_text(encoding="utf-8") == "historical"
