"""Corrected sleep counts consolidation without changing legacy counts."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.app.continual_shift_benchmark import _apply_scheduled_sleep
from src.app.experiment_runner import ExperimentConfig, format_experiment_result, run_experiment
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork


def _numpy_config(mode: str) -> CircadianConfig:
    return CircadianConfig(
        sleep_mode=mode,
        sleep_enable_chemical_reset=True,
        sleep_enable_replay=False if mode == "components" else True,
        sleep_enable_homeostasis=False if mode == "components" else True,
        sleep_enable_split=False if mode == "components" else True,
        sleep_enable_prune=False if mode == "components" else True,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        replay_steps=0,
        sleep_reset_factor=0.5,
    )


def _adaptive_config(mode: str) -> CircadianConfig:
    return replace(
        _numpy_config(mode),
        use_adaptive_sleep_trigger=True,
        min_epochs_between_sleep=2,
        sleep_energy_window=2,
        sleep_plateau_delta=1e9,
        sleep_chemical_variance_threshold=0.0,
    )


@pytest.mark.parametrize("mode,expected_count", [("components", 2), ("legacy", 0), ("disabled", 0)])
def test_toy_report_counts_corrected_no_topology_sleep_only(mode: str, expected_count: int) -> None:
    config = ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_config=_numpy_config(mode),
        random_seed=97,
    )
    result = run_experiment(config)
    assert result.circadian_sleep.event_count == expected_count
    assert result.circadian_sleep.sleep_mode == mode
    assert f"Circadian sleep: mode={mode}" in format_experiment_result(result)
    assert result.circadian_sleep.total_splits == 0
    assert result.circadian_sleep.total_prunes == 0
    assert result.circadian_sleep.hidden_dim_end == 4


@pytest.mark.parametrize("mode,expected_count", [("components", 1), ("legacy", 0), ("disabled", 0)])
def test_continual_sleep_count_uses_corrected_event_signal(mode: str, expected_count: int) -> None:
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=101, min_hidden_dim=4, circadian_config=_numpy_config(mode)
    )
    model.set_chemical_state(np.array([0.1, 0.2, 0.3, 0.4]))
    count, splits, prunes = _apply_scheduled_sleep(
        model=model,
        sleep_interval=1,
        epoch_index=1,
        global_epoch=1,
        total_epochs=2,
        force_sleep=True,
        sleep_event_count=0,
        total_splits=0,
        total_prunes=0,
    )
    assert count == expected_count
    assert splits == prunes == 0
    if mode == "components":
        np.testing.assert_allclose(model.get_chemical_state(), [0.05, 0.1, 0.15, 0.2])


@pytest.mark.parametrize("mode,expected_count", [("components", 1), ("legacy", 0), ("disabled", 0)])
def test_toy_adaptive_trigger_runs_without_periodic_interval_in_component_mode(
    mode: str, expected_count: int
) -> None:
    result = run_experiment(
        ExperimentConfig(
            sample_count=80,
            hidden_dim=4,
            epoch_count=3,
            pc_inference_steps=2,
            circadian_inference_steps=2,
            circadian_sleep_interval=0,
            circadian_force_sleep=False,
            circadian_config=_adaptive_config(mode),
            random_seed=107,
        )
    )
    assert result.circadian_sleep.event_count == expected_count


@pytest.mark.parametrize("mode,expected_count", [("components", 1), ("legacy", 0), ("disabled", 0)])
def test_continual_adaptive_trigger_runs_without_periodic_interval_in_component_mode(
    mode: str, expected_count: int
) -> None:
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=109, min_hidden_dim=4, circadian_config=_adaptive_config(mode)
    )
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    for _ in range(2):
        model.train_epoch(features, targets, 0.03, 2, 0.2)
    count, splits, prunes = _apply_scheduled_sleep(
        model=model,
        sleep_interval=0,
        epoch_index=2,
        global_epoch=2,
        total_epochs=3,
        force_sleep=False,
        sleep_event_count=0,
        total_splits=0,
        total_prunes=0,
    )
    assert count == expected_count
    assert splits == prunes == 0


def test_vision_builder_exposes_component_mode_and_guard_checks_no_topology_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from src.app import matched_head_benchmark
    from src.app.matched_head_benchmark import _guarded_sleep_event
    from src.app.resnet50_benchmark import ResNet50BenchmarkConfig, _build_circadian_head_config
    from src.core.resnet50_variants import CircadianPredictiveCodingHead

    config = ResNet50BenchmarkConfig(
        epochs=1,
        circadian_force_sleep=True,
        circadian_enable_sleep_rollback=True,
        circadian_sleep_rollback_eval_batches=1,
        circadian_sleep_mode="components",
        circadian_sleep_enable_chemical_reset=True,
        circadian_sleep_enable_homeostasis=False,
        circadian_sleep_enable_split=False,
        circadian_sleep_enable_prune=False,
        circadian_max_split_per_sleep=0,
        circadian_max_prune_per_sleep=0,
        circadian_sleep_warmup_steps=0,
        circadian_sleep_reset_factor=0.5,
    )
    head_config = _build_circadian_head_config(config)
    assert head_config.sleep_mode == "components"
    head = CircadianPredictiveCodingHead(
        2, 3, 3, torch.device("cpu"), seed=103, config=head_config, min_hidden_dim=3
    )
    head._chemical = torch.tensor([0.2, 0.4, 0.6])
    guard_calls: list[int] = []

    def evaluate(*args: object, **kwargs: object) -> tuple[float, float]:
        guard_calls.append(1)
        return 1.0, 0.0

    monkeypatch.setattr(matched_head_benchmark, "_evaluate_head", evaluate)
    event, rolled_back = _guarded_sleep_event(
        torch,
        torch.device("cpu"),
        head,
        ((torch.zeros((1, 2)), torch.tensor([0])),),
        config,
        epoch=1,
        force_sleep=True,
    )
    assert event.performed is True
    assert event.split_indices == event.pruned_indices == ()
    assert rolled_back is False
    assert len(guard_calls) == 2
    torch.testing.assert_close(head._chemical, torch.tensor([0.1, 0.2, 0.3]))


def test_vision_cli_passes_component_switches_to_benchmark(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.adapters import resnet_benchmark_cli

    captured: list[Any] = []

    def capture_benchmark(config: object) -> object:
        captured.append(config)
        return object()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resnet-benchmark",
            "--circ-sleep-mode",
            "components",
            "--circ-disable-homeostasis",
            "--circ-disable-split",
            "--circ-disable-prune",
        ],
    )
    monkeypatch.setattr(
        resnet_benchmark_cli,
        "run_resnet50_benchmark",
        capture_benchmark,
    )
    monkeypatch.setattr(resnet_benchmark_cli, "format_resnet50_benchmark_result", lambda _: "ok")
    resnet_benchmark_cli.main()
    assert capsys.readouterr().out.strip() == "ok"
    assert len(captured) == 1
    config = captured[0]
    assert config.circadian_sleep_mode == "components"
    assert config.circadian_sleep_enable_chemical_reset is True
    assert config.circadian_sleep_enable_homeostasis is False
    assert config.circadian_sleep_enable_split is False
    assert config.circadian_sleep_enable_prune is False


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"circadian_sleep_mode": "unknown"}, "circadian_sleep_mode"),
        ({"circadian_sleep_enable_split": False}, "circadian_sleep_mode"),
        (
            {"circadian_sleep_mode": "components", "circadian_sleep_enable_prune": 0},
            "circadian_sleep_enable_prune",
        ),
    ],
)
def test_vision_benchmark_rejects_invalid_component_configuration_early(
    changes: dict[str, Any], field: str
) -> None:
    from src.app.resnet50_benchmark import ResNet50BenchmarkConfig, _validate_benchmark_config

    with pytest.raises(ValueError, match=field):
        _validate_benchmark_config(replace(ResNet50BenchmarkConfig(), **changes))
