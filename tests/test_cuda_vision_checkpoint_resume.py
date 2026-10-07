"""Actual CUDA continuation for the unmatched vision checkpoint routes."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
import random
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")
from torch import nn  # noqa: E402

from src.app import resnet50_benchmark as vision  # noqa: E402
from src.app.resnet50_benchmark import (  # noqa: E402
    ResNet50BenchmarkConfig,
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
    VISION_VALIDATION_UNMATCHED_PROTOCOL,
)
from src.core import resnet50_variants  # noqa: E402
from src.infra.circadian_checkpoint_files import (  # noqa: E402
    TrustedLocalVisionCheckpointStore,
)
from src.infra.local_result_json import write_local_result_json  # noqa: E402


class InterruptedAfterSave(Exception):
    pass


class InterruptAtStage:
    def __init__(self, store: TrustedLocalVisionCheckpointStore, stage: str) -> None:
        self.store = store
        self.stage = stage

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        progress = checkpoint.active_circadian
        if progress is not None and progress.stage == self.stage:
            raise InterruptedAfterSave()


class CaptureBeforeSleepStore:
    def __init__(self, store: TrustedLocalVisionCheckpointStore) -> None:
        self.store = store
        self.entry: Any = None

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        progress = checkpoint.active_circadian
        if (
            self.entry is None
            and progress is not None
            and progress.stage == "before_sleep"
            and not progress.sleep_events
        ):
            self.entry = checkpoint
        self.store.save(checkpoint)


def _config(protocol_id: str, *, reject: bool) -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        protocol_id=protocol_id,
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=2,
        seed=73,
        device="cuda:0",
        target_accuracy=None,
        inference_batches=1,
        warmup_batches=0,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=4,
        circadian_min_hidden_dim=4,
        circadian_max_hidden_dim=6,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_sleep_mode="components",
        circadian_use_adaptive_sleep_trigger=False,
        circadian_use_adaptive_sleep_budget=False,
        circadian_use_adaptive_thresholds=False,
        circadian_split_threshold=0.0,
        circadian_split_hysteresis_margin=0.0,
        circadian_split_cooldown_steps=0,
        circadian_sleep_max_change_fraction=1.0,
        circadian_max_prune_per_sleep=0,
        circadian_sleep_enable_prune=False,
        circadian_sleep_enable_homeostasis=False,
        circadian_sleep_enable_chemical_reset=False,
        circadian_sleep_split_only_until_fraction=1.0,
        circadian_sleep_prune_only_after_fraction=1.0,
        circadian_sleep_rollback_tolerance=0.0 if reject else 100.0,
    )


def _install_cuda_random_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.output = torch.nn.Linear(3, 16)

        def forward(self, images: Any) -> Any:
            # Why this: an active CUDA process draw must affect continuation.
            noisy = images + 0.01 * torch.rand_like(images)
            return self.output(self.pool(noisy).flatten(1))

    def build(device: Any, freeze_backbone: bool, backbone_weights: str) -> Any:
        assert backbone_weights == "none"
        model = TinyBackbone().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(not freeze_backbone)
        return model, 16

    monkeypatch.setattr(resnet50_variants, "_build_resnet50_backbone", build)


@pytest.fixture
def cuda_random_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_cuda_random_backbone(monkeypatch)


def _seed_all() -> None:
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    torch.cuda.manual_seed_all(814)


def _next_draws() -> tuple[float, float, float, float]:
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), device="cuda:0")),
    )


def _assert_same_non_timing(actual: Any, expected: Any) -> None:
    excluded = {
        "train_seconds",
        "train_samples_per_second",
        "mean_train_step_ms",
        "inference_latency_mean_ms",
        "inference_latency_p95_ms",
        "inference_samples_per_second",
    }
    for left, right in zip(actual.reports, expected.reports, strict=True):
        for field in fields(left):
            if field.name == "sleep_events":
                assert len(left.sleep_events) == len(right.sleep_events)
                for actual_event, expected_event in zip(
                    left.sleep_events, right.sleep_events, strict=True
                ):
                    actual_facts, expected_facts = asdict(actual_event), asdict(expected_event)
                    actual_facts.pop("durations")
                    expected_facts.pop("durations")
                    assert actual_facts == expected_facts
                continue
            if field.name not in excluded:
                assert getattr(left, field.name) == pytest.approx(getattr(right, field.name))


@pytest.mark.parametrize(
    ("protocol_id", "stage", "reject"),
    [
        (VISION_SEEDED_UNMATCHED_PROTOCOL, "wake", False),
        (VISION_SEEDED_UNMATCHED_PROTOCOL, "after_sleep", False),
        (VISION_SEEDED_UNMATCHED_PROTOCOL, "after_sleep", True),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "after_sleep", False),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "after_sleep", True),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "wake", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "after_sleep", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "after_sleep", True),
    ],
)
def test_cuda_vision_resume_matches_uninterrupted_learning_and_rng(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    cuda_random_backbone: None,
    protocol_id: str,
    stage: str,
    reject: bool,
) -> None:
    config = _config(protocol_id, reject=reject)
    order = (
        ("circadian", "predictive", "backprop")
        if protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL
        else None
    )

    def score(
        _torch: Any,
        model: Any,
        loader: Any,
        _device: Any,
        max_batches: int | None = None,
        *,
        on_examples_scored: Any = None,
    ) -> tuple[float, float]:
        if on_examples_scored is not None:
            batch_count = len(loader) if max_batches is None else min(len(loader), max_batches)
            on_examples_scored(min(len(loader.dataset), batch_count * loader.batch_size))
        return 0.8, 2.0 if reject and model.head.hidden_dim > 4 else 0.5

    monkeypatch.setattr(vision, "_compute_pc_metrics", score)

    _seed_all()
    control = vision.run_resnet50_benchmark(
        config,
        model_order=order,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "control-cuda.ckpt"),
    )
    expected_draws = _next_draws()

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision-cuda.ckpt")
    _seed_all()
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=InterruptAtStage(store, stage),
        )
    saved = store.load()
    assert saved.active_circadian is not None
    assert saved.active_circadian.stage == stage
    assert saved.torch_cuda_device == "cuda:0"
    assert saved.torch_cuda_random_state is not None
    assert saved.torch_cuda_random_state.dtype == torch.uint8
    assert saved.active_circadian.outer_entry_torch_cuda_state is not None
    assert saved.active_circadian.outer_entry_torch_cuda_state.dtype == torch.uint8
    assert saved.active_circadian.classifier_state.head_state["generator_device"] == "cuda"
    assert saved.active_circadian.loader_state.next_batch_index == (1 if stage == "wake" else 0)
    torch.rand((3,), device="cuda:0")  # A fresh process need not share this stream.
    random.random()
    np.random.random()

    resumed = vision.run_resnet50_benchmark(
        config, model_order=order, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed.training_order == control.training_order
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _assert_same_non_timing(resumed, control)
    circadian = next(
        report
        for report in resumed.reports
        if report.model_name == "CircadianPredictiveCodingResNet50"
    )
    assert circadian.circadian_sleep_attempts > 0
    assert len(circadian.sleep_events) == config.epochs
    assert circadian.sleep_events[0].outcome == ("rolled_back" if reject else "accepted")
    assert circadian.sleep_events[1].outcome == ("skipped" if reject else "accepted")
    expected_role = (
        "validation" if protocol_id == VISION_VALIDATION_UNMATCHED_PROTOCOL else "inner_guard"
    )
    expected_hash = control.split_hashes["validation" if expected_role == "validation" else "guard"]
    for event in circadian.sleep_events[:1] if reject else circadian.sleep_events:
        assert event.guard is not None
        assert event.guard.role == expected_role
        assert event.guard.role_hash == expected_hash
        assert event.guard.metric_name == config.circadian_sleep_rollback_metric
        assert event.guard.examples_scored == 2 * config.guard_samples
        assert event.durations.attempt_seconds >= event.durations.core_seconds
        if reject:
            assert event.changes.applied_split_pairs == ()
            assert event.changes.proposed_split_pairs
    if reject:
        assert circadian.sleep_events[1].reason == "rollback_cooldown"
        assert circadian.sleep_events[1].guard is None
    result_path = tmp_path / "cuda-vision-result.json"
    write_local_result_json(resumed, result_path)
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    saved_circadian = next(
        report
        for report in saved_result["reports"]
        if report["head_type"] == "circadian_predictive_coding"
    )
    assert saved_circadian["sleep_events"][0]["outcome"] == circadian.sleep_events[0].outcome
    if stage == "after_sleep":
        assert (
            circadian.circadian_total_rollbacks > 0
            if reject
            else circadian.circadian_total_splits > 0
        )
    assert _next_draws() == expected_draws


@pytest.mark.parametrize(
    "protocol_id",
    [
        VISION_VALIDATION_UNMATCHED_PROTOCOL,
        VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
        VISION_SEEDED_UNMATCHED_PROTOCOL,
    ],
)
@pytest.mark.parametrize("failure_stage", ["pre", "core", "post"])
def test_cuda_vision_failed_sleep_resumes_with_error_history_and_rng(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    cuda_random_backbone: None,
    protocol_id: str,
    failure_stage: str,
) -> None:
    config = _config(protocol_id, reject=False)
    order = (
        ("circadian", "predictive", "backprop")
        if protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL
        else None
    )
    failure_enabled = False
    guard_calls = 0

    def disturb_random_and_head(model: Any) -> None:
        model.head._chemical[0] += 0.5
        random.random()
        np.random.random()
        torch.rand(())
        torch.rand((), device="cuda:0")
        torch.rand((), device="cuda:0", generator=model.head._split_generator)

    def score(
        _torch: Any,
        model: Any,
        loader: Any,
        _device: Any,
        max_batches: int | None = None,
        *,
        on_examples_scored: Any = None,
    ) -> tuple[float, float]:
        nonlocal failure_enabled, guard_calls
        if on_examples_scored is not None:
            batch_count = len(loader) if max_batches is None else min(len(loader), max_batches)
            on_examples_scored(min(len(loader.dataset), batch_count * loader.batch_size))
            guard_calls += 1
            if failure_enabled and (
                (failure_stage == "pre" and guard_calls == 1)
                or (failure_stage == "post" and guard_calls == 2)
            ):
                disturb_random_and_head(model)
                failure_enabled = False
                raise RuntimeError(f"CUDA {failure_stage} guard failed")
        return 0.8, 0.5

    monkeypatch.setattr(vision, "_compute_pc_metrics", score)
    original_sleep = resnet50_variants.CircadianPredictiveCodingResNet50Classifier.sleep_event

    def sleep_or_fail(self: Any, **kwargs: Any) -> Any:
        nonlocal failure_enabled
        if failure_enabled and failure_stage == "core":
            disturb_random_and_head(self)
            failure_enabled = False
            raise RuntimeError("CUDA core failed")
        return original_sleep(self, **kwargs)

    monkeypatch.setattr(
        resnet50_variants.CircadianPredictiveCodingResNet50Classifier,
        "sleep_event",
        sleep_or_fail,
    )
    original_build = vision._build_benchmark_loaders
    final_allowed = True
    final_reads = 0

    class SealedTestLoader:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __iter__(self) -> Any:
            nonlocal final_reads
            if not final_allowed:
                raise AssertionError("CUDA error opened final test before resume")
            final_reads += 1
            return iter(self.wrapped)

    def build(run_config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(run_config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    control_recording = CaptureBeforeSleepStore(
        TrustedLocalVisionCheckpointStore(tmp_path / "control-cuda-error.ckpt")
    )
    _seed_all()
    control = vision.run_resnet50_benchmark(
        config, model_order=order, checkpoint_store=control_recording
    )
    expected_draws = _next_draws()
    assert control_recording.entry is not None

    store = TrustedLocalVisionCheckpointStore(tmp_path / "cuda-error.ckpt")
    recording = CaptureBeforeSleepStore(store)
    _seed_all()
    final_allowed = False
    final_reads = 0
    failure_enabled = True
    guard_calls = 0
    with pytest.raises(RuntimeError, match="CUDA .* failed"):
        vision.run_resnet50_benchmark(config, model_order=order, checkpoint_store=recording)
    assert final_reads == 0
    saved = store.load()
    entry = recording.entry
    assert entry is not None
    assert saved.active_circadian is not None
    assert saved.active_circadian.stage == "before_sleep"
    assert saved.active_circadian.sleep_attempts == 1
    assert saved.python_random_state == entry.python_random_state
    np.testing.assert_equal(saved.numpy_random_state, entry.numpy_random_state)
    assert torch.equal(saved.torch_cpu_random_state, entry.torch_cpu_random_state)
    assert torch.equal(saved.torch_cuda_random_state, entry.torch_cuda_random_state)
    assert torch.equal(
        saved.active_circadian.classifier_state.head_state["split_generator_state"],
        entry.active_circadian.classifier_state.head_state["split_generator_state"],
    )
    error = saved.active_circadian.sleep_events[0]
    assert error.outcome == "error"
    assert error.reason == (
        "sleep_core_exception"
        if failure_stage == "core"
        else f"inner_guard_{failure_stage}_exception"
    )
    assert error.guard is not None
    expected_role = (
        "validation" if protocol_id == VISION_VALIDATION_UNMATCHED_PROTOCOL else "inner_guard"
    )
    assert error.guard.role == expected_role
    assert (
        error.guard.role_hash
        == control.split_hashes["validation" if expected_role == "validation" else "guard"]
    )
    assert error.guard.metric_name == config.circadian_sleep_rollback_metric
    assert error.guard.examples_scored == (16 if failure_stage == "post" else 8)
    assert (error.guard.pre_accuracy is not None) == (failure_stage != "pre")
    assert bool(error.changes.proposed_split_pairs) == (failure_stage == "post")
    assert error.changes.applied_split_pairs == ()

    final_allowed = True
    resumed = vision.run_resnet50_benchmark(
        config, model_order=order, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert final_reads > 0
    assert resumed.trained_model_hashes == control.trained_model_hashes
    excluded = {
        "sleep_events",
        "circadian_sleep_attempts",
        "train_seconds",
        "train_samples_per_second",
        "mean_train_step_ms",
        "inference_latency_mean_ms",
        "inference_latency_p95_ms",
        "inference_samples_per_second",
    }
    for actual_report, expected_report in zip(resumed.reports, control.reports, strict=True):
        for field in fields(actual_report):
            if field.name not in excluded:
                assert getattr(actual_report, field.name) == pytest.approx(
                    getattr(expected_report, field.name)
                )
    actual_circadian = next(
        report for report in resumed.reports if report.head_type == "circadian_predictive_coding"
    )
    expected_circadian = next(
        report for report in control.reports if report.head_type == "circadian_predictive_coding"
    )
    assert (
        actual_circadian.circadian_sleep_attempts == expected_circadian.circadian_sleep_attempts + 1
    )
    assert [event.outcome for event in actual_circadian.sleep_events[:2]] == ["error", "accepted"]
    for actual_event, expected_event in zip(
        actual_circadian.sleep_events[1:], expected_circadian.sleep_events, strict=True
    ):
        actual_facts, expected_facts = asdict(actual_event), asdict(expected_event)
        actual_facts.pop("durations")
        expected_facts.pop("durations")
        assert actual_facts == expected_facts
    assert _next_draws() == expected_draws
    result_path = tmp_path / "cuda-error-result.json"
    write_local_result_json(resumed, result_path)
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    saved_circadian = next(
        report
        for report in saved_result["reports"]
        if report["head_type"] == "circadian_predictive_coding"
    )
    assert [event["outcome"] for event in saved_circadian["sleep_events"][:2]] == [
        "error",
        "accepted",
    ]


def test_cuda_vision_resume_rejects_bad_device_state_before_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, cuda_random_backbone: None
) -> None:
    config = _config(VISION_SEEDED_UNMATCHED_PROTOCOL, reject=False)
    order = ("circadian", "predictive", "backprop")
    store = TrustedLocalVisionCheckpointStore(tmp_path / "bad-cuda-state.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=InterruptAtStage(store, "wake"),
        )
    saved = store.load()
    progress = saved.active_circadian
    assert progress is not None
    classifier = progress.classifier_state
    bad_outer = replace(
        saved,
        active_circadian=replace(
            progress, outer_entry_torch_cuda_state=torch.empty(0, dtype=torch.uint8)
        ),
    )
    bad_local = replace(
        saved,
        active_circadian=replace(
            progress,
            classifier_state=replace(
                classifier, head_state={**classifier.head_state, "generator_device": "cuda:9"}
            ),
        ),
    )
    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args, **kwargs: pytest.fail("training started before CUDA preflight"),
    )
    previous_python = random.getstate()
    previous_numpy = np.random.get_state()
    previous_cpu = torch.get_rng_state().clone()
    previous_cuda = torch.cuda.get_rng_state("cuda:0").clone()
    for bad in (
        replace(saved, torch_cuda_device="cuda:9"),
        replace(saved, torch_cuda_random_state=torch.empty(0, dtype=torch.uint8)),
        replace(saved, torch_cuda_random_state=None),
        bad_outer,
        bad_local,
    ):
        store.save(bad)
        with pytest.raises(ValueError, match="checkpoint"):
            vision.run_resnet50_benchmark(
                config,
                model_order=order,
                checkpoint_store=store,
                resume_from_checkpoint=True,
            )
        assert torch.equal(torch.get_rng_state(), previous_cpu)
        assert torch.equal(torch.cuda.get_rng_state("cuda:0"), previous_cuda)
        assert random.getstate() == previous_python
        np.testing.assert_equal(np.random.get_state(), previous_numpy)
