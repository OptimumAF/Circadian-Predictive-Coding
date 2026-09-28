"""CPU gates for the staged fixed-feature backprop/PC comparison."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import matched_head_benchmark  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.core.resnet50_variants import (  # noqa: E402
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)
from src.shared.process_memory import read_process_rss_bytes  # noqa: E402


def tiny_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=8, validation_samples=8, guard_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1,
        seed=47, device="cpu", target_accuracy=None,
        backprop_freeze_backbone=True, backbone_weights="none",
        predictive_head_hidden_dim=16, predictive_inference_steps=2,
    )


def tiny_three_head_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=8, validation_samples=8, guard_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1,
        seed=47, device="cpu", target_accuracy=None,
        backprop_freeze_backbone=True, backbone_weights="none",
        predictive_head_hidden_dim=16, predictive_inference_steps=2,
        circadian_head_hidden_dim=16, circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=32, circadian_inference_steps=2,
        circadian_sleep_interval=0, circadian_use_adaptive_sleep_trigger=False,
    )


def test_disabled_sleep_does_not_attempt_matched_head_guard_event() -> None:
    config = replace(
        tiny_three_head_config(),
        circadian_sleep_interval=1,
        circadian_sleep_mode="disabled",
        circadian_sleep_warmup_steps=0,
    )
    result = matched_head_benchmark.run_three_head_fixed_feature_benchmark(config)
    assert result.circadian.sleep_attempts == 0
    assert result.circadian.total_splits == 0
    assert result.circadian.total_prunes == 0


def test_real_cpu_backbone_produces_identical_initial_heads_and_feature_bank() -> None:
    result = matched_head_benchmark.run_two_head_fixed_feature_benchmark(tiny_config())
    repeated = matched_head_benchmark.run_two_head_fixed_feature_benchmark(tiny_config())

    assert result.protocol_id == "vision_two_head_fixed_feature_v1"
    assert result.source_protocol_id == "vision_guard_separated_unmatched_v2"
    assert result.backbone_weights == "none"
    assert len(result.backbone_hash) == 64
    assert result.initial_head_hashes["backprop_mlp"] == (
        result.initial_head_hashes["predictive_coding"]
    )
    assert set(result.feature_hashes) == {"train", "guard", "validation", "test"}
    assert set(result.split_hashes) == {"train", "guard", "validation", "test"}
    assert result.feature_bytes > 0
    assert result.benchmark_track == "frozen_shared_representation"
    assert result.backbone_trainable is False
    assert result.backbone_pretraining == "none"
    assert result.backbone_parameters > 0
    for report, head_type in (
        (result.backprop, "backprop_mlp"),
        (result.predictive_coding, "predictive_coding"),
    ):
        assert report.benchmark_track == result.benchmark_track
        assert report.backbone_trainable is False
        assert report.backbone_pretraining == "none"
        assert report.head_type == head_type
        assert report.total_parameters == result.backbone_parameters + report.trainable_parameters
    assert result.backprop.trainable_parameters == result.predictive_coding.trainable_parameters
    assert result.backprop.seen_samples == result.predictive_coding.seen_samples == 8
    assert 0.0 <= result.backprop.test_accuracy <= 1.0
    assert 0.0 <= result.predictive_coding.test_accuracy <= 1.0
    assert result.backbone_hash == repeated.backbone_hash
    assert result.initial_head_hashes == repeated.initial_head_hashes
    assert result.feature_hashes == repeated.feature_hashes
    assert result.backprop.validation_accuracy == repeated.backprop.validation_accuracy
    assert result.predictive_coding.test_accuracy == repeated.predictive_coding.test_accuracy


@pytest.mark.parametrize("include_circadian", [False, True])
def test_final_test_loader_stays_sealed_until_all_included_heads_train(
    monkeypatch: pytest.MonkeyPatch, include_circadian: bool,
) -> None:
    original_backprop = matched_head_benchmark._train_backprop_head
    original_predictive = matched_head_benchmark._train_predictive_head
    original_circadian = matched_head_benchmark._train_circadian_head
    trained: list[str] = []
    seen_batches: list[Any] = []
    expected = ["backprop", "predictive"]
    result: matched_head_benchmark.TwoHeadFixedFeatureResult
    if include_circadian:
        expected.append("circadian")
    feature_dim = 5
    images = torch.tensor(
        [[0.2, 0.1, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    role_loader = [(images, labels)]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            if trained != expected:
                raise AssertionError("Final test opened before all heads trained")
            return iter(role_loader)

    loaders = SimpleNamespace(
        train_loader=role_loader, guard_loader=list(role_loader),
        validation_loader=list(role_loader), test_loader=SealedTestLoader(),
        num_classes=3,
        split_hashes={role: role for role in ("train", "guard", "validation", "test")},
    )
    monkeypatch.setattr(matched_head_benchmark, "_build_benchmark_loaders", lambda config: loaders)
    monkeypatch.setattr(
        matched_head_benchmark, "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), feature_dim),
    )

    def track_backprop(*args: Any) -> Any:
        seen_batches.append(args[3:6])
        outcome = original_backprop(*args)
        trained.append("backprop")
        return outcome

    def track_predictive(*args: Any) -> Any:
        seen_batches.append(args[3:6])
        outcome = original_predictive(*args)
        trained.append("predictive")
        return outcome

    def track_circadian(*args: Any) -> Any:
        seen_batches.append(args[3:6])
        outcome = original_circadian(*args)
        trained.append("circadian")
        return outcome

    monkeypatch.setattr(matched_head_benchmark, "_train_backprop_head", track_backprop)
    monkeypatch.setattr(matched_head_benchmark, "_train_predictive_head", track_predictive)
    monkeypatch.setattr(matched_head_benchmark, "_train_circadian_head", track_circadian)

    if include_circadian:
        result = matched_head_benchmark.run_three_head_fixed_feature_benchmark(
            tiny_three_head_config()
        )
    else:
        result = matched_head_benchmark.run_two_head_fixed_feature_benchmark(tiny_config())

    assert trained == expected
    for trainer_batches in seen_batches[1:]:
        assert all(left is right for left, right in zip(seen_batches[0], trainer_batches))
    assert result.backprop.seen_samples == result.predictive_coding.seen_samples == 2
    if include_circadian:
        assert isinstance(result, matched_head_benchmark.ThreeHeadFixedFeatureResult)
        assert result.circadian.seen_samples == 2


def test_two_head_route_requires_frozen_guarded_backbone() -> None:
    with pytest.raises(ValueError, match="frozen"):
        matched_head_benchmark.run_two_head_fixed_feature_benchmark(
            ResNet50BenchmarkConfig(backprop_freeze_backbone=False)
        )


def test_three_head_cpu_gate_uses_one_feature_bank_and_equal_initial_tensors() -> None:
    result = matched_head_benchmark.run_three_head_fixed_feature_benchmark(
        tiny_three_head_config()
    )
    repeated = matched_head_benchmark.run_three_head_fixed_feature_benchmark(
        tiny_three_head_config()
    )

    assert result.protocol_id == "vision_three_head_fixed_feature_v1"
    assert len(set(result.initial_head_hashes.values())) == 1
    assert set(result.initial_head_hashes) == {
        "backprop_mlp", "predictive_coding", "circadian_predictive_coding",
    }
    assert result.backbone_hash == repeated.backbone_hash
    assert result.feature_hashes == repeated.feature_hashes
    assert result.circadian.test_accuracy == repeated.circadian.test_accuracy
    assert result.circadian.seen_samples == 8
    assert result.circadian.hidden_dim_start == result.circadian.hidden_dim_end == 16
    assert result.circadian.total_splits == result.circadian.total_prunes == 0
    assert result.circadian.head_type == "circadian_predictive_coding"
    assert result.circadian.total_parameters == (
        result.backbone_parameters + result.circadian.trainable_parameters
    )


def test_three_head_rejects_unmatched_width_before_loading_data() -> None:
    with pytest.raises(ValueError, match="same initial hidden width"):
        matched_head_benchmark.run_three_head_fixed_feature_benchmark(tiny_config())


def test_three_head_training_state_ignores_outer_validation_and_final_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = torch.tensor(
        [[0.2, 0.1, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
        dtype=torch.float32,
    )
    original_labels = torch.tensor([0, 1], dtype=torch.long)
    changed_labels = torch.tensor([2, 2], dtype=torch.long)
    outer_labels = original_labels

    def build_loaders(config: ResNet50BenchmarkConfig) -> SimpleNamespace:
        return SimpleNamespace(
            train_loader=[(features, original_labels)],
            guard_loader=[(features, original_labels)],
            validation_loader=[(features, outer_labels)],
            test_loader=[(features, outer_labels)],
            num_classes=3,
            split_hashes={role: role for role in ("train", "guard", "validation", "test")},
        )

    monkeypatch.setattr(matched_head_benchmark, "_build_benchmark_loaders", build_loaders)
    monkeypatch.setattr(
        matched_head_benchmark, "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    config = replace(
        tiny_three_head_config(), circadian_sleep_interval=1,
        circadian_force_sleep=True, circadian_sleep_warmup_steps=0,
    )
    original = matched_head_benchmark.run_three_head_fixed_feature_benchmark(config)
    outer_labels = changed_labels
    changed = matched_head_benchmark.run_three_head_fixed_feature_benchmark(config)

    assert original.trained_head_hashes == changed.trained_head_hashes
    assert original.feature_hashes["guard"] == changed.feature_hashes["guard"]
    assert original.feature_hashes["validation"] != changed.feature_hashes["validation"]
    assert original.feature_hashes["test"] != changed.feature_hashes["test"]


def test_three_head_result_is_invariant_to_model_execution_order() -> None:
    config = replace(
        tiny_three_head_config(), circadian_sleep_interval=1,
        circadian_force_sleep=True, circadian_sleep_warmup_steps=0,
        evaluation_batches=1,
    )
    forward = matched_head_benchmark.run_three_head_fixed_feature_benchmark(config)
    reverse = matched_head_benchmark.run_three_head_fixed_feature_benchmark(
        config,
        model_order=(
            "circadian_predictive_coding", "predictive_coding", "backprop_mlp",
        ),
    )

    assert forward.training_order == (
        "backprop_mlp", "predictive_coding", "circadian_predictive_coding",
    )
    assert reverse.training_order == tuple(reversed(forward.training_order))
    assert forward.backbone_hash == reverse.backbone_hash
    assert forward.split_hashes == reverse.split_hashes
    assert forward.feature_hashes == reverse.feature_hashes
    assert forward.initial_head_hashes == reverse.initial_head_hashes
    assert forward.trained_head_hashes == reverse.trained_head_hashes
    for report in (forward.backprop, forward.predictive_coding, forward.circadian):
        assert report.seen_samples == 8
        assert report.wake_batches == 2
        assert report.replay_examples == 0
    assert forward.backprop.latent_relaxation_steps == 0
    assert forward.predictive_coding.latent_relaxation_steps == 4
    assert forward.circadian.latent_relaxation_steps == 4
    assert forward.backprop.guard_examples_scored == 4
    assert forward.predictive_coding.guard_examples_scored == 4
    assert forward.circadian.guard_examples_scored == 20
    assert forward.circadian.sleep_attempts == 1
    for name in ("backprop", "predictive_coding", "circadian"):
        left = getattr(forward, name)
        right = getattr(reverse, name)
        assert left.validation_accuracy == pytest.approx(right.validation_accuracy, abs=1e-7)
        assert left.test_accuracy == pytest.approx(right.test_accuracy, abs=1e-7)
        assert left.test_cross_entropy == pytest.approx(right.test_cross_entropy, abs=1e-7)
        assert left.total_splits == right.total_splits
        assert left.total_prunes == right.total_prunes
        assert left.total_rollbacks == right.total_rollbacks


def test_three_head_rejects_duplicate_or_missing_model_order() -> None:
    with pytest.raises(ValueError, match="permutation"):
        matched_head_benchmark.run_three_head_fixed_feature_benchmark(
            tiny_three_head_config(),
            model_order=("backprop_mlp", "backprop_mlp", "predictive_coding"),
        )


def test_matched_trained_hash_covers_adaptive_state_beyond_initial_parameters() -> None:
    circadian = CircadianPredictiveCodingHead(
        feature_dim=4, hidden_dim=16, num_classes=3,
        device=torch.device("cpu"), seed=47,
    )
    parameter_hash = matched_head_benchmark._hash_head(circadian)
    initial_state_hash = matched_head_benchmark._hash_trained_head(circadian)
    circadian._chemical[0] = 0.25
    chemical_hash = matched_head_benchmark._hash_trained_head(circadian)
    assert matched_head_benchmark._hash_head(circadian) == parameter_hash
    assert chemical_hash != initial_state_hash
    torch.rand((1,), generator=circadian._split_generator)
    assert matched_head_benchmark._hash_trained_head(circadian) != chemical_hash

    predictive = PredictiveCodingHead(
        feature_dim=4, hidden_dim=16, num_classes=3,
        device=torch.device("cpu"), seed=47,
    )
    predictive_hash = matched_head_benchmark._hash_trained_head(predictive)
    predictive._traffic_sum[0] = 0.25
    assert matched_head_benchmark._hash_trained_head(predictive) != predictive_hash


def test_wall_time_runner_rejects_invalid_budget_and_target_stopping() -> None:
    for invalid_budget in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            matched_head_benchmark.run_three_head_fixed_feature_wall_time_benchmark(
                tiny_three_head_config(), wall_time_budget_seconds=invalid_budget,
            )
    with pytest.raises(ValueError, match="target_accuracy"):
        matched_head_benchmark.run_three_head_fixed_feature_wall_time_benchmark(
            replace(tiny_three_head_config(), target_accuracy=0.8),
            wall_time_budget_seconds=1.0,
        )


def test_cuda_memory_peak_hooks_reset_and_read_allocator_state() -> None:
    calls: list[str] = []

    class FakeCuda:
        def synchronize(self, device: str) -> None:
            calls.append(f"sync:{device}")

        def memory_allocated(self, device: str) -> int:
            calls.append(f"allocated:{device}")
            return 1024

        def reset_peak_memory_stats(self, device: str) -> None:
            calls.append(f"reset:{device}")

        def max_memory_allocated(self, device: str) -> int:
            calls.append(f"peak_allocated:{device}")
            return 2048

        def max_memory_reserved(self, device: str) -> int:
            calls.append(f"peak_reserved:{device}")
            return 4096

    fake_torch = SimpleNamespace(cuda=FakeCuda())
    start = matched_head_benchmark._reset_cuda_memory_peak(fake_torch, "cuda:0")
    peak = matched_head_benchmark._read_cuda_memory_peak(fake_torch, "cuda:0")

    assert start == 1024
    assert peak == (2048, 4096)
    assert calls == [
        "sync:cuda:0", "allocated:cuda:0", "reset:cuda:0",
        "sync:cuda:0", "peak_allocated:cuda:0", "peak_reserved:cuda:0",
    ]


def test_wall_time_deadline_stops_before_next_wake_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClock:
        time = 0.0

        def read(self) -> float:
            return self.time

        def advance(self) -> float:
            self.time += 1.0
            return 0.0

    clock = FakeClock()
    head = PredictiveCodingHead(
        feature_dim=4, hidden_dim=16, num_classes=3,
        device=torch.device("cpu"), seed=47,
    )
    monkeypatch.setattr(head, "train_step", lambda **kwargs: clock.advance())
    batch = (torch.ones((2, 4)), torch.tensor([0, 1]))
    outcome = matched_head_benchmark._train_predictive_head(
        torch, torch.device("cpu"), head,
        (batch, batch, batch), (batch,), (batch,),
        replace(tiny_three_head_config(), epochs=3),
        time_budget_seconds=2.0, clock=clock.read,
    )

    assert outcome.stop_reason == "deadline"
    assert outcome.epochs_ran == 0
    assert outcome.wake_batches == 2
    assert outcome.seen_samples == 4
    assert outcome.latent_relaxation_steps == 4
    assert outcome.guard_examples_scored == 0
    assert outcome.train_seconds == 2.0


def test_wall_time_route_seals_final_test_and_reports_each_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished: list[str] = []
    test_openings: list[bool] = []
    features = torch.ones((2, 5), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    role_loader = [(features, labels)]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            if len(finished) != 3:
                raise AssertionError("Final test opened before all deadline trainers returned")
            test_openings.append(True)
            return iter(role_loader)

    monkeypatch.setattr(
        matched_head_benchmark, "_build_benchmark_loaders",
        lambda config: SimpleNamespace(
            train_loader=role_loader, guard_loader=list(role_loader),
            validation_loader=list(role_loader), test_loader=SealedTestLoader(),
            num_classes=3,
            split_hashes={role: role for role in ("train", "guard", "validation", "test")},
        ),
    )
    monkeypatch.setattr(
        matched_head_benchmark, "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    for name in ("backprop", "predictive", "circadian"):
        method_name = f"_train_{name}_head"
        original = getattr(matched_head_benchmark, method_name)

        def tracked(*args: Any, _name: str = name, _original: Any = original) -> Any:
            outcome = _original(*args)
            finished.append(_name)
            return outcome

        monkeypatch.setattr(matched_head_benchmark, method_name, tracked)

    budget_seconds = 0.02
    result = matched_head_benchmark.run_three_head_fixed_feature_wall_time_benchmark(
        replace(tiny_three_head_config(), epochs=1000),
        wall_time_budget_seconds=budget_seconds,
    )

    assert finished == ["backprop", "predictive", "circadian"]
    assert result.protocol_id == "vision_three_head_fixed_feature_wall_time_v1"
    assert result.wall_time_budget_seconds == budget_seconds
    assert result.memory_telemetry_enabled is False
    assert result.process_rss_sample_interval_seconds is None
    assert len(test_openings) == 1
    assert len(set(result.initial_head_hashes.values())) == 1
    for report in (result.backprop, result.predictive_coding, result.circadian):
        assert report.stop_reason == "deadline"
        assert report.train_seconds >= budget_seconds
        assert report.deadline_overshoot_seconds == pytest.approx(
            report.train_seconds - budget_seconds,
        )
        assert report.seen_samples >= report.epochs_ran * 2
        assert report.process_rss_start_bytes is None
        assert report.process_rss_peak_observed_bytes is None
        assert report.process_rss_samples == 0
        assert report.cuda_allocated_start_bytes is None
        assert report.cuda_allocated_peak_bytes is None
        assert report.cuda_reserved_peak_bytes is None

    finished.clear()
    with pytest.raises(ValueError, match="increase epochs"):
        matched_head_benchmark.run_three_head_fixed_feature_wall_time_benchmark(
            tiny_three_head_config(), wall_time_budget_seconds=60.0,
        )
    assert len(test_openings) == 1


@pytest.mark.parametrize("wall_time", [False, True])
def test_memory_profile_is_versioned_and_reports_observed_peak(wall_time: bool) -> None:
    config = tiny_three_head_config()
    if wall_time:
        result = matched_head_benchmark.run_three_head_fixed_feature_wall_time_benchmark(
            replace(config, epochs=1000), wall_time_budget_seconds=0.02,
            measure_memory=True,
        )
        expected_protocol = "vision_three_head_fixed_feature_wall_time_memory_v2"
    else:
        result = matched_head_benchmark.run_three_head_fixed_feature_benchmark(
            config, measure_memory=True,
        )
        expected_protocol = "vision_three_head_fixed_feature_memory_v1"

    assert result.protocol_id == expected_protocol
    assert result.memory_telemetry_enabled is True
    assert result.process_rss_sample_interval_seconds == 0.005
    for report in (result.backprop, result.predictive_coding, result.circadian):
        if read_process_rss_bytes() is not None:
            assert report.process_rss_start_bytes is not None
            assert report.process_rss_peak_observed_bytes is not None
            assert report.process_rss_peak_observed_bytes >= report.process_rss_start_bytes
            assert report.process_rss_peak_observed_bytes > result.feature_bytes
            assert report.process_rss_samples >= 2
        assert report.cuda_allocated_start_bytes is None
        assert report.cuda_allocated_peak_bytes is None
        assert report.cuda_reserved_peak_bytes is None


def test_feature_views_ignore_unrelated_backbone_rng_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = torch.tensor([0, 1], dtype=torch.long)

    class RandomViewLoader:
        def __iter__(self) -> Any:
            return iter([(torch.rand((2, 5)), labels)])

    monkeypatch.setattr(
        matched_head_benchmark, "_build_benchmark_loaders",
        lambda config: SimpleNamespace(
            train_loader=RandomViewLoader(), guard_loader=RandomViewLoader(),
            validation_loader=RandomViewLoader(), test_loader=RandomViewLoader(),
            num_classes=3,
            split_hashes={role: role for role in ("train", "guard", "validation", "test")},
        ),
    )
    draw_count = 0

    def build_backbone(**kwargs: Any) -> tuple[Any, int]:
        _ = torch.rand((draw_count,))
        return torch.nn.Identity(), 5

    monkeypatch.setattr(matched_head_benchmark, "_build_resnet50_backbone", build_backbone)
    first = matched_head_benchmark.run_two_head_fixed_feature_benchmark(tiny_config())
    draw_count = 101
    second = matched_head_benchmark.run_two_head_fixed_feature_benchmark(tiny_config())

    assert first.feature_hashes == second.feature_hashes
    assert first.trained_head_hashes == second.trained_head_hashes
