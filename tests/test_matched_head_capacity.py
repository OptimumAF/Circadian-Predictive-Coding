"""Fixed-width capacity-control gates for the shared-feature head track."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import matched_head_benchmark as benchmark  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.core.resnet50_variants import CircadianPredictiveCodingHead  # noqa: E402
from src.shared.process_memory import read_process_rss_bytes  # noqa: E402


def _fixed_width_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=47,
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=16,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
        circadian_enable_sleep_rollback=True,
    )


def test_fixed_width_control_preserves_capacity_during_guarded_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_observations: list[tuple[int, int, bool, tuple[int, ...], tuple[int, ...]]] = []
    original_sleep = CircadianPredictiveCodingHead.sleep_event

    def observe_sleep(self: CircadianPredictiveCodingHead, *args: Any, **kwargs: Any) -> Any:
        before_count = self.parameter_count()
        before_chemical = self.mean_chemical()
        event = original_sleep(self, *args, **kwargs)
        sleep_observations.append(
            (
                before_count,
                self.parameter_count(),
                not bool(torch.equal(before_chemical, self.mean_chemical())),
                event.split_indices,
                event.pruned_indices,
            )
        )
        return event

    monkeypatch.setattr(CircadianPredictiveCodingHead, "sleep_event", observe_sleep)
    config = _fixed_width_config()
    result = benchmark.run_three_head_fixed_width_capacity_benchmark(config)

    assert result.protocol_id == "vision_three_head_fixed_width_capacity_memory_v1"
    assert result.memory_telemetry_enabled is True
    assert result.capacity_control is not None
    assert result.capacity_control.mode == "fixed_width_parameter_matched"
    assert result.capacity_control.hidden_dim == 16
    assert set(result.capacity_control.initial_head_parameters.values()) == {
        result.capacity_control.head_parameters
    }
    assert set(result.capacity_control.final_head_parameters.values()) == {
        result.capacity_control.head_parameters
    }
    assert len(set(result.initial_head_hashes.values())) == 1
    assert sleep_observations == [
        (
            result.capacity_control.head_parameters,
            result.capacity_control.head_parameters,
            True,
            (),
            (),
        )
    ]
    assert result.circadian.sleep_attempts == 1
    assert result.circadian.guard_examples_scored > result.predictive_coding.guard_examples_scored
    assert result.circadian.hidden_dim_start == result.circadian.hidden_dim_end == 16
    assert result.circadian.total_splits == result.circadian.total_prunes == 0
    for report in (result.backprop, result.predictive_coding, result.circadian):
        assert report.trainable_parameters == result.capacity_control.head_parameters
        assert report.seen_samples == 8
        assert report.wake_batches == 2
        if read_process_rss_bytes() is not None:
            assert report.process_rss_start_bytes is not None
            assert report.process_rss_peak_observed_bytes is not None
            assert report.process_rss_peak_observed_bytes >= report.process_rss_start_bytes
    assert result.feature_bytes > 0  # Static cache storage, separate from sampled RSS.

    # The adaptive-width route keeps its existing protocol and the same inputs.
    adaptive = benchmark.run_three_head_fixed_feature_benchmark(
        replace(config, circadian_max_hidden_dim=32)
    )
    assert adaptive.protocol_id == "vision_three_head_fixed_feature_v1"
    assert adaptive.capacity_control is None
    assert result.backbone_hash == adaptive.backbone_hash
    assert result.feature_hashes == adaptive.feature_hashes
    assert result.initial_head_hashes == adaptive.initial_head_hashes


@pytest.mark.parametrize(
    "change",
    [
        {"circadian_max_hidden_dim": 32},
        {"circadian_min_hidden_dim": 15},
        {"target_accuracy": 0.9},
        {"circadian_force_sleep": False},
        {"circadian_sleep_mode": "disabled"},
        {
            "circadian_sleep_mode": "components",
            "circadian_sleep_enable_split": False,
            "circadian_sleep_enable_prune": False,
        },
        {"circadian_sleep_interval": 0},
        {"circadian_sleep_warmup_steps": 1},
        {"circadian_enable_sleep_rollback": False},
        {"circadian_max_split_per_sleep": 0},
        {"circadian_max_prune_per_sleep": 0},
        {"circadian_sleep_max_change_fraction": 0.0},
        {"backprop_freeze_backbone": False},
    ],
)
def test_capacity_route_rejects_invalid_control_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_build_benchmark_loaders",
        lambda config: pytest.fail("Invalid capacity config reached data loading"),
    )
    with pytest.raises(ValueError):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            replace(_fixed_width_config(), **change)
        )


def test_capacity_mismatch_fails_before_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = torch.ones((2, 5), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    role_loader = [(features, labels)]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            raise AssertionError("Final test opened before capacity check")

    monkeypatch.setattr(
        benchmark,
        "_build_benchmark_loaders",
        lambda config: SimpleNamespace(
            train_loader=role_loader,
            guard_loader=list(role_loader),
            validation_loader=list(role_loader),
            test_loader=SealedTestLoader(),
            num_classes=3,
            split_hashes={role: role for role in ("train", "guard", "validation", "test")},
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    original_train = benchmark._train_circadian_head

    def report_bad_capacity(*args: Any) -> Any:
        outcome = original_train(*args)
        return replace(outcome, parameter_count=outcome.parameter_count + 1)

    monkeypatch.setattr(benchmark, "_train_circadian_head", report_bad_capacity)
    with pytest.raises(AssertionError, match="capacity"):
        benchmark.run_three_head_fixed_width_capacity_benchmark(_fixed_width_config())
