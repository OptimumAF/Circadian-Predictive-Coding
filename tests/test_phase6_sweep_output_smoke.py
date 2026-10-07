"""Check public sweep preflights and complete fixture-only output writers."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from importlib import import_module
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from src.app.resnet50_benchmark import ModelDevelopmentReport


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = import_module("scripts.run_circadian_policy_sweep")
PARETO = import_module("scripts.run_pareto_hard_tuning")


def fixture_report(model_name: str, head_type: str) -> ModelDevelopmentReport:
    # Equal fixture metrics exercise serialization without choosing an algorithm.
    return ModelDevelopmentReport(
        model_name=model_name,
        epochs_ran=1,
        validation_accuracy=0.5,
        validation_cross_entropy=0.5,
        train_seconds=1.0,
        train_samples_per_second=8.0,
        mean_train_step_ms=1.0,
        inference_latency_mean_ms=1.0,
        inference_latency_p95_ms=1.0,
        inference_samples_per_second=8.0,
        total_parameters=100,
        trainable_parameters=100,
        benchmark_track="unmatched_reference",
        backbone_trainable=False,
        backbone_pretraining="imagenet",
        head_type=head_type,
        final_energy=0.5 if model_name != "backprop" else None,
        training_energy_id=(
            "torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1"
            if model_name != "backprop"
            else None
        ),
    )


def reject_nonfinite(token: str) -> Any:
    raise AssertionError(f"Nonfinite JSON output: {token}")


@pytest.mark.parametrize(
    ("filename", "sweep_name", "candidates", "trials", "updates", "examples"),
    [
        ("run_circadian_policy_sweep.py", "circadian_policy", 18, 18, 14_400, 900_000),
        ("run_pareto_hard_tuning.py", "pareto_hard", 34, 102, 81_600, 5_100_000),
    ],
)
def test_public_sweep_cli_estimates_and_refuses_default_launch_before_work(
    tmp_path: Path,
    filename: str,
    sweep_name: str,
    candidates: int,
    trials: int,
    updates: int,
    examples: int,
) -> None:
    script = REPO_ROOT / "scripts" / filename
    estimate = subprocess.run(
        [sys.executable, str(script), "--estimate-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert estimate.returncode == 0, estimate.stderr
    assert estimate.stderr == ""
    record = json.loads(estimate.stdout, parse_constant=reject_nonfinite)
    assert record["sweep"] == sweep_name
    assert record["candidate_count"] == candidates
    assert record["trial_count"] == trials
    assert record["planned_max_training_updates"] == updates
    assert record["planned_max_training_examples"] == examples
    if sweep_name == "pareto_hard":
        assert record["candidate_counts"] == {
            "backprop": 10,
            "predictive": 12,
            "circadian": 12,
        }
    assert list(tmp_path.iterdir()) == []

    refused = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert refused.returncode != 0
    assert f"{updates} training updates" in refused.stderr
    assert "1000" in refused.stderr
    assert "Wrote " not in refused.stdout
    assert list(tmp_path.iterdir()) == []


def test_policy_stub_writer_preserves_full_grid_and_validation_only_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "policy-fixture.json"
    monkeypatch.setattr(POLICY, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(POLICY, "require_torch", object)
    monkeypatch.setattr(POLICY, "_set_seed", lambda *_args: None)
    monkeypatch.setattr(POLICY, "_resolve_device", lambda *_args: "cpu")

    class DevelopmentOnlyLoaders:
        train_loader = object()
        guard_loader = object()
        validation_loader = object()
        num_classes = 10
        split_hashes = {role: role * 64 for role in ("train", "guard", "validation", "test")}

        @property
        def test_loader(self) -> Any:
            pytest.fail("Policy fixture opened the final-test loader")

    monkeypatch.setattr(
        POLICY, "build_synthetic_vision_dataloaders", lambda *_args: DevelopmentOnlyLoaders()
    )
    calls = 0

    def stub_candidate(**kwargs: Any) -> ModelDevelopmentReport:
        nonlocal calls
        calls += 1
        assert kwargs["variant"] == "circadian"
        assert not hasattr(kwargs["loaders"], "test_loader")
        return fixture_report("circadian", "circadian_predictive_coding")

    monkeypatch.setattr(POLICY, "benchmark_validation_candidate", stub_candidate)
    original_candidates = POLICY.build_candidates(None)
    assert len(original_candidates) == 18
    digest = sha256(json.dumps(original_candidates, sort_keys=True, separators=(",", ":")).encode())
    assert digest.hexdigest() == "334a1b15c0c0116d8da596b7d936a06411bfe355ef28be10288bc9a002ff1ec7"

    # This test-only cap reaches a stub writer. No Torch model, CUDA, or data runs.
    POLICY.main(max_planned_training_updates=14_400)

    stdout = capsys.readouterr().out
    assert f"Wrote {output_path}" in stdout
    assert calls == 18
    assert list(tmp_path.iterdir()) == [output_path]
    result = json.loads(output_path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    assert result["protocol_id"] == "vision_guard_separated_unmatched_v2"
    assert result["selection_metric"] == "validation_accuracy"
    assert result["final_test_usage"] == "none"
    assert result["final_test_confirmation"] == "pending"
    assert result["inference_split"] == "validation"
    assert result["prelaunch_estimate"]["planned_max_training_updates"] == 14_400
    assert result["max_planned_training_updates"] == 14_400
    assert result["dataset"]["split_hashes"] == DevelopmentOnlyLoaders.split_hashes
    assert len(result["trials"]) == 18
    assert [row["trial"] for row in result["trials"]] == list(range(1, 19))
    assert [row["params"] for row in result["trials"]] == original_candidates
    assert all(row["report"]["validation_accuracy"] == 0.5 for row in result["trials"])
    assert all("test_accuracy" not in row["report"] for row in result["trials"])
    assert len(result["top10_by_validation_accuracy"]) == 10
    assert len(result["top10_by_balanced_score"]) == 10
    assert math.isfinite(result["best_balanced"]["report"]["balanced_score"])

    before = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="Sweep output already exists"):
        POLICY.main(max_planned_training_updates=14_400)
    assert output_path.read_bytes() == before


def test_pareto_stub_writer_preserves_all_family_rows_and_seed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "pareto-fixture.json"
    monkeypatch.setattr(PARETO, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(PARETO, "require_torch", object)
    monkeypatch.setattr(PARETO, "_set_seed", lambda *_args: None)
    monkeypatch.setattr(PARETO, "_resolve_device", lambda *_args: "cpu")
    monkeypatch.setattr(
        PARETO,
        "build_synthetic_vision_dataloaders",
        lambda *_args: pytest.fail("Pareto fixture constructed real data"),
    )
    expected_candidates = {
        family: getattr(PARETO, f"build_{family}_candidates")()
        for family in ("backprop", "predictive", "circadian")
    }
    head_types = {
        "backprop": "linear",
        "predictive": "predictive_coding",
        "circadian": "circadian_predictive_coding",
    }
    seen: list[str] = []

    def make_stub_runner(family: str):
        def run(_base: Any, _torch: Any, _device: Any, seeds: Any, *, candidate_params: Any):
            assert seeds == (7, 13, 29)
            assert candidate_params == expected_candidates[family]
            seen.append(family)
            trials = []
            for index, override in enumerate(candidate_params, start=1):
                seed_rows = []
                for seed in seeds:
                    report = replace(
                        fixture_report(family, head_types[family]),
                        model_name=family,
                    )
                    row = PARETO.report_to_dict(report)
                    row["seed"] = seed
                    row["split_hashes"] = {
                        role: role * 64 for role in ("train", "guard", "validation")
                    }
                    seed_rows.append(row)
                trials.append(
                    {
                        "trial": index,
                        "params": override,
                        "report": PARETO._aggregate_reports(seed_rows),
                        "seed_reports": seed_rows,
                    }
                )
            return trials

        return run

    for family in expected_candidates:
        monkeypatch.setattr(PARETO, f"run_{family}_sweep", make_stub_runner(family))

    # This test-only cap reaches stubbed rows; the real 102-trial run stays gated.
    PARETO.main(max_planned_training_updates=81_600)

    stdout = capsys.readouterr().out
    assert f"Wrote {output_path}" in stdout
    assert seen == ["backprop", "predictive", "circadian"]
    assert list(tmp_path.iterdir()) == [output_path]
    result = json.loads(output_path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    assert result["protocol_id"] == "vision_guard_separated_unmatched_v2"
    assert result["selection_metric"] == "validation_accuracy"
    assert result["final_test_usage"] == "none"
    assert result["final_test_confirmation"] == "pending"
    assert result["inference_split"] == "validation"
    assert result["dataset"]["seeds"] == [7, 13, 29]
    assert result["prelaunch_estimate"]["candidate_counts"] == {
        "backprop": 10,
        "predictive": 12,
        "circadian": 12,
    }
    assert result["prelaunch_estimate"]["planned_max_training_updates"] == 81_600
    assert result["max_planned_training_updates"] == 81_600
    for family, candidates in expected_candidates.items():
        group = result[family]
        assert len(group["trials"]) == len(candidates)
        assert [trial["trial"] for trial in group["trials"]] == list(range(1, len(candidates) + 1))
        assert [trial["params"] for trial in group["trials"]] == candidates
        assert len(group["top10_by_validation_accuracy"]) == 10
        assert len(group["top10_by_balanced_score"]) == 10
        for trial in group["trials"]:
            assert len(trial["seed_reports"]) == 3
            assert [row["seed"] for row in trial["seed_reports"]] == [7, 13, 29]
            assert trial["report"]["validation_accuracy"] == 0.5
            assert math.isfinite(trial["report"]["balanced_score"])
            assert "test_accuracy" not in trial["report"]
            assert all("test_accuracy" not in row for row in trial["seed_reports"])
    assert all("test_accuracy" not in result[key] for key in result if key.startswith("global_"))

    before = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="Sweep output already exists"):
        PARETO.main(max_planned_training_updates=81_600)
    assert output_path.read_bytes() == before
