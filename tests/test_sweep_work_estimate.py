"""A sweep must reveal its planned work before Torch or data construction."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path

import pytest

from src.app.resnet50_benchmark import ResNet50BenchmarkConfig
from src.app.sweep_work_estimate import (
    estimate_vision_candidate_work,
    require_planned_training_limit,
)


policy_sweep = import_module("scripts.run_circadian_policy_sweep")
pareto_sweep = import_module("scripts.run_pareto_hard_tuning")


def test_should_count_actual_batches_epochs_candidates_and_seeds() -> None:
    first = replace(ResNet50BenchmarkConfig(), train_samples=5, batch_size=2, epochs=3)
    second = replace(first, train_samples=8, batch_size=4, epochs=2)

    estimate = estimate_vision_candidate_work((first, second), seed_count=2)

    assert estimate.candidate_count == 2
    assert estimate.seed_count == 2
    assert estimate.trial_count == 4
    assert estimate.planned_max_training_updates == 26
    assert estimate.planned_max_training_examples == 62
    assert (
        estimate.counting_rule
        == "epochs * ceil(train_examples / batch_size) per candidate and seed"
    )


@pytest.mark.parametrize(
    ("configs", "seeds", "message"),
    [
        ((), 1, "candidate"),
        ((ResNet50BenchmarkConfig(),), 0, "seed"),
        ((ResNet50BenchmarkConfig(),), True, "seed"),
        ((replace(ResNet50BenchmarkConfig(), train_samples=0),), 1, "train_samples"),
        ((replace(ResNet50BenchmarkConfig(), batch_size=0),), 1, "batch_size"),
        ((replace(ResNet50BenchmarkConfig(), epochs=0),), 1, "epochs"),
        ((replace(ResNet50BenchmarkConfig(), dataset_name="cifar10"),), 1, "synthetic"),
    ],
)
def test_should_reject_unknown_or_invalid_estimate_inputs(
    configs: tuple[ResNet50BenchmarkConfig, ...], seeds: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_vision_candidate_work(configs, seed_count=seeds)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_should_reject_invalid_launch_limit(limit: object) -> None:
    estimate = estimate_vision_candidate_work((ResNet50BenchmarkConfig(),), seed_count=1)

    with pytest.raises(ValueError, match="launch limit"):
        require_planned_training_limit(estimate, limit)


def test_should_refuse_policy_default_before_torch_or_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(policy_sweep, "OUTPUT_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(policy_sweep, "require_torch", lambda: pytest.fail("Torch initialized"))
    monkeypatch.setattr(
        policy_sweep, "build_synthetic_vision_dataloaders", lambda *args: pytest.fail("data built")
    )

    with pytest.raises(ValueError, match="14400 training updates.*1000"):
        policy_sweep.main()

    assert '"candidate_count": 18' in capsys.readouterr().out
    assert not policy_sweep.OUTPUT_PATH.exists()


def test_should_print_policy_estimate_without_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(policy_sweep, "OUTPUT_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(policy_sweep, "require_torch", lambda: pytest.fail("Torch initialized"))

    policy_sweep.main(estimate_only=True)

    output = capsys.readouterr().out
    assert '"planned_max_training_updates": 14400' in output
    assert '"planned_max_training_examples": 900000' in output
    assert not policy_sweep.OUTPUT_PATH.exists()


def test_should_require_explicit_limit_for_large_policy_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(policy_sweep, "OUTPUT_PATH", tmp_path / "policy.json")

    class LaunchReached(Exception):
        pass

    def mark_launch() -> None:
        raise LaunchReached

    monkeypatch.setattr(policy_sweep, "require_torch", mark_launch)

    with pytest.raises(LaunchReached):
        policy_sweep.main(max_planned_training_updates=14_400)


def test_should_reject_candidate_that_changes_shared_loader_before_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(policy_sweep, "OUTPUT_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(policy_sweep, "build_candidates", lambda _base: [{"train_samples": 1}])
    monkeypatch.setattr(policy_sweep, "require_torch", lambda: pytest.fail("Torch initialized"))

    with pytest.raises(ValueError, match="circadian fields only"):
        policy_sweep.main()


@pytest.mark.parametrize(
    ("builder_name", "count", "expected_sha256"),
    [
        (
            "build_backprop_candidates",
            10,
            "425e85d723384f3c1fbfbd43f7a1364cba1d29252c75f9741eb4651c24f62588",
        ),
        (
            "build_predictive_candidates",
            12,
            "50858f59efa6b96077a589230ffd0f0b5ca4893a2cbdf14fc439fa1fa7d29356",
        ),
        (
            "build_circadian_candidates",
            12,
            "701e8f5874b7ef9b4921c36bf4f8208feb47bab321ecabf199d9edec21f6633a",
        ),
    ],
)
def test_should_preserve_pareto_candidate_values_and_order(
    builder_name: str, count: int, expected_sha256: str
) -> None:
    candidates = getattr(pareto_sweep, builder_name)()

    assert len(candidates) == count
    actual = sha256(json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode())
    assert actual.hexdigest() == expected_sha256


def test_should_refuse_pareto_default_before_torch_or_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pareto_sweep, "OUTPUT_PATH", tmp_path / "pareto.json")
    monkeypatch.setattr(pareto_sweep, "require_torch", lambda: pytest.fail("Torch initialized"))
    monkeypatch.setattr(
        pareto_sweep, "build_synthetic_vision_dataloaders", lambda *args: pytest.fail("data built")
    )

    with pytest.raises(ValueError, match="81600 training updates.*1000"):
        pareto_sweep.main()

    output = capsys.readouterr().out
    assert '"candidate_count": 34' in output
    assert '"trial_count": 102' in output
    assert not pareto_sweep.OUTPUT_PATH.exists()


def test_should_print_pareto_estimate_without_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pareto_sweep, "OUTPUT_PATH", tmp_path / "pareto.json")
    monkeypatch.setattr(pareto_sweep, "require_torch", lambda: pytest.fail("Torch initialized"))

    pareto_sweep.main(estimate_only=True)

    output = capsys.readouterr().out
    assert '"planned_max_training_updates": 81600' in output
    assert '"planned_max_training_examples": 5100000' in output
    assert '"backprop": 10' in output
    assert '"predictive": 12' in output
    assert '"circadian": 12' in output
    assert not pareto_sweep.OUTPUT_PATH.exists()


def test_should_require_explicit_limit_for_large_pareto_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pareto_sweep, "OUTPUT_PATH", tmp_path / "pareto.json")

    class LaunchReached(Exception):
        pass

    def mark_launch() -> None:
        raise LaunchReached

    monkeypatch.setattr(pareto_sweep, "require_torch", mark_launch)

    with pytest.raises(LaunchReached):
        pareto_sweep.main(max_planned_training_updates=81_600)


def test_should_reject_pareto_candidate_that_changes_wrong_family_before_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pareto_sweep, "OUTPUT_PATH", tmp_path / "pareto.json")
    monkeypatch.setattr(pareto_sweep, "build_predictive_candidates", lambda: [{"train_samples": 1}])
    monkeypatch.setattr(pareto_sweep, "require_torch", lambda: pytest.fail("Torch initialized"))

    with pytest.raises(ValueError, match="predictive fields only"):
        pareto_sweep.main()


def test_should_pass_preflighted_pareto_candidates_and_save_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "pareto.json"
    monkeypatch.setattr(pareto_sweep, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(pareto_sweep, "require_torch", object)
    monkeypatch.setattr(pareto_sweep, "_set_seed", lambda *_args: None)
    monkeypatch.setattr(pareto_sweep, "_resolve_device", lambda *_args: "cpu")
    received: list[tuple[str, tuple[int, ...], list[dict[str, object]]]] = []

    def make_runner(family: str):
        def run(_base, _torch, _device, seeds, *, candidate_params):
            received.append((family, seeds, candidate_params))
            return []

        return run

    for family in ("backprop", "predictive", "circadian"):
        monkeypatch.setattr(pareto_sweep, f"run_{family}_sweep", make_runner(family))
    monkeypatch.setattr(pareto_sweep, "summarize_model_trials", lambda _trials: {"trials": []})
    monkeypatch.setattr(
        pareto_sweep, "collect_all_trial_reports", lambda _output: [{"model_name": "backprop"}]
    )
    monkeypatch.setattr(pareto_sweep, "best_from_all_trials", lambda rows, key: rows[0])

    pareto_sweep.main(max_planned_training_updates=81_600)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["prelaunch_estimate"]["planned_max_training_updates"] == 81_600
    assert saved["prelaunch_estimate"]["candidate_counts"] == {
        "backprop": 10,
        "predictive": 12,
        "circadian": 12,
    }
    assert saved["max_planned_training_updates"] == 81_600
    assert saved["selection_metric"] == "validation_accuracy"
    assert saved["final_test_usage"] == "none"
    assert [(family, len(candidates)) for family, _, candidates in received] == [
        ("backprop", 10),
        ("predictive", 12),
        ("circadian", 12),
    ]
    assert all(seeds == (7, 13, 29) for _, seeds, _ in received)
