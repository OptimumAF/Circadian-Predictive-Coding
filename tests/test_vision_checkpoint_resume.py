"""Trusted local checkpoints preserve completed unmatched vision models."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
import random
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402

from src.app import resnet50_benchmark as vision  # noqa: E402
from src.app.sleep_schedule import decide_sleep_attempt  # noqa: E402
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


class InterruptedAfterSave(Exception):
    pass


class InterruptingStore:
    def __init__(
        self,
        store: TrustedLocalVisionCheckpointStore,
        after_variants: int = 1,
    ) -> None:
        self.store = store
        self.after_variants = after_variants

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if checkpoint.next_variant_index == self.after_variants:
            raise InterruptedAfterSave()


class InterruptAtActiveStage:
    def __init__(
        self,
        store: TrustedLocalVisionCheckpointStore,
        stage: str,
        completed_epoch: int,
    ) -> None:
        self.store = store
        self.stage = stage
        self.completed_epoch = completed_epoch

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        active = getattr(checkpoint, "active_circadian", None)
        if (
            active is not None
            and active.stage == self.stage
            and active.completed_epoch == self.completed_epoch
        ):
            raise InterruptedAfterSave()


class StochasticVisionDataset:
    """Raw examples with process-local augmentation for spawned workers."""

    def __init__(self, dataset: Any) -> None:
        self.images = dataset.images
        self.labels = dataset.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        jitter = 0.01 * (float(torch.rand(())) + float(np.random.random()) + random.random())
        return self.images[index] + jitter, self.labels[index]


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=73,
        device="cpu",
        target_accuracy=None,
        inference_batches=1,
        warmup_batches=0,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=32,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
    )


def _sleep_config(reject: bool = False) -> ResNet50BenchmarkConfig:
    return replace(
        _config(),
        epochs=2,
        circadian_sleep_mode="components",
        circadian_use_adaptive_sleep_trigger=False,
        circadian_use_adaptive_sleep_budget=False,
        circadian_use_adaptive_thresholds=False,
        circadian_head_hidden_dim=4,
        circadian_min_hidden_dim=4,
        circadian_max_hidden_dim=6,
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


def _same_learning_reports(actual: Any, expected: Any) -> None:
    timed = {
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
            if field.name not in timed:
                assert getattr(left, field.name) == pytest.approx(getattr(right, field.name))


def _controlled_pc_score(reject: bool) -> Any:
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

    return score


@pytest.mark.parametrize(
    "protocol",
    [
        VISION_VALIDATION_UNMATCHED_PROTOCOL,
        VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
        VISION_SEEDED_UNMATCHED_PROTOCOL,
    ],
)
@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("metric_name", ["cross_entropy", "accuracy"])
def test_unmatched_vision_reports_guarded_sleep_proposal_and_role(
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    reject: bool,
    metric_name: str,
) -> None:
    config = replace(
        _sleep_config(reject=reject),
        protocol_id=protocol,
        circadian_sleep_rollback_metric=metric_name,
    )
    original_score = vision._compute_pc_metrics

    def score(*args: Any, **kwargs: Any) -> tuple[float, float]:
        accuracy, cross_entropy = original_score(*args, **kwargs)
        model = args[1]
        if isinstance(model, resnet50_variants.CircadianPredictiveCodingResNet50Classifier):
            if model.head.hidden_dim > 4:
                return 0.7, 2.0
            return 0.8, 0.5
        return accuracy, cross_entropy

    monkeypatch.setattr(vision, "_compute_pc_metrics", score)

    result = vision.run_resnet50_benchmark(config)
    reports = {report.model_name: report for report in result.reports}
    assert reports["BackpropResNet50"].sleep_events == ()
    assert reports["PredictiveCodingResNet50"].sleep_events == ()
    circadian = reports["CircadianPredictiveCodingResNet50"]
    assert len(circadian.sleep_events) == config.epochs
    outcome = "rolled_back" if reject else "accepted"
    if reject:
        assert circadian.circadian_total_rollbacks > 0
    else:
        assert circadian.circadian_total_splits > 0
    event = next(event for event in circadian.sleep_events if event.outcome == outcome)
    assert event.trigger_reason == "periodic"
    assert event.reason == ("guard_rejected" if reject else "guard_accepted")
    assert event.guard is not None
    assert event.guard.role == (
        "validation" if protocol == VISION_VALIDATION_UNMATCHED_PROTOCOL else "inner_guard"
    )
    assert event.guard.metric_name == config.circadian_sleep_rollback_metric
    assert event.guard.delta is not None
    assert event.guard.tolerance == config.circadian_sleep_rollback_tolerance
    assert event.guard.examples_scored == 2 * config.guard_samples
    assert event.changes.proposed_split_pairs
    assert event.changes.applied_split_pairs == (
        () if reject else event.changes.proposed_split_pairs
    )
    assert event.final_width == (event.before_width if reject else event.proposed_width)
    assert event.durations.attempt_seconds >= event.durations.core_seconds
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"circadian_sleep_mode": "disabled"}, "sleep_disabled"),
        (
            {"circadian_sleep_interval": 3, "circadian_use_adaptive_sleep_trigger": False},
            "schedule_not_due",
        ),
    ],
)
def test_unmatched_vision_reports_unattempted_sleep_without_guard(
    tiny_vision_backbone: None, changes: dict[str, Any], reason: str
) -> None:
    config = replace(_config(), epochs=2, **changes)
    result = vision.run_resnet50_benchmark(config)
    circadian = next(
        report for report in result.reports if report.head_type == "circadian_predictive_coding"
    )
    assert circadian.circadian_sleep_attempts == 0
    assert [event.reason for event in circadian.sleep_events] == [reason, reason]
    assert all(
        event.outcome == "skipped" and event.guard is None for event in circadian.sleep_events
    )
    json.dumps(asdict(result), allow_nan=False)


def test_unmatched_vision_reports_unguarded_core_decision(
    tiny_vision_backbone: None,
) -> None:
    config = replace(_sleep_config(), epochs=1, circadian_enable_sleep_rollback=False)
    result = vision.run_resnet50_benchmark(config)
    circadian = next(
        report for report in result.reports if report.head_type == "circadian_predictive_coding"
    )
    assert circadian.circadian_sleep_attempts == 1
    assert len(circadian.sleep_events) == 1
    event = circadian.sleep_events[0]
    assert event.outcome == "applied"
    assert event.reason == "core_executed"
    assert event.guard is None
    assert event.changes.proposed_split_pairs == event.changes.applied_split_pairs
    json.dumps(asdict(result), allow_nan=False)


def test_unmatched_vision_records_guarded_core_warmup_skip(
    tiny_vision_backbone: None,
) -> None:
    config = replace(_sleep_config(), epochs=1, circadian_sleep_warmup_steps=99)
    result = vision.run_resnet50_benchmark(config)
    circadian = next(
        report for report in result.reports if report.head_type == "circadian_predictive_coding"
    )
    assert circadian.circadian_sleep_attempts == 1
    assert len(circadian.sleep_events) == 1
    event = circadian.sleep_events[0]
    assert event.outcome == "skipped"
    assert event.reason == "warmup"
    assert event.guard is not None
    assert event.guard.examples_scored == 2 * config.guard_samples
    assert event.changes.proposed_split_pairs == ()
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("change", ["missing", "role", "exposure", "reason", "clock", "counter"])
def test_active_vision_sleep_history_tamper_rejects_before_restore(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, tiny_vision_backbone: None, change: str
) -> None:
    config = _sleep_config(reject=True)
    monkeypatch.setattr(vision, "_compute_pc_metrics", _controlled_pc_score(True))
    store = TrustedLocalVisionCheckpointStore(tmp_path / "active-history.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(store, "after_sleep", 1),
        )
    saved = store.load()
    active = saved.active_circadian
    assert active is not None
    assert len(active.sleep_events) == 1
    event = active.sleep_events[0]
    assert event.guard is not None
    if change == "missing":
        active = replace(active, sleep_events=())
    elif change == "counter":
        active = replace(active, sleep_attempts=active.sleep_attempts + 1)
    else:
        changed_event = {
            "role": replace(event, guard=replace(event.guard, role="validation")),
            "exposure": replace(event, guard=replace(event.guard, examples_scored=1)),
            "reason": replace(event, reason="guard_accepted"),
            "clock": replace(event, wake_batches=event.wake_batches + 1),
        }[change]
        active = replace(active, sleep_events=(changed_event,))
    store.save(replace(saved, active_circadian=active))
    monkeypatch.setattr(
        vision, "_train_circadian", lambda *args, **kwargs: pytest.fail("tamper reached training")
    )
    with pytest.raises(ValueError, match="sleep history"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store, resume_from_checkpoint=True)


def test_completed_vision_sleep_history_tamper_rejects_before_final_test(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, tiny_vision_backbone: None
) -> None:
    config = _sleep_config(reject=True)
    monkeypatch.setattr(vision, "_compute_pc_metrics", _controlled_pc_score(True))
    store = TrustedLocalVisionCheckpointStore(tmp_path / "completed-history.ckpt")
    vision.run_resnet50_benchmark(config, checkpoint_store=store)
    saved = store.load()
    circadian = saved.completed_outcomes[-1]
    assert circadian.sleep_events
    changed = replace(circadian, sleep_events=())
    store.save(replace(saved, completed_outcomes=(*saved.completed_outcomes[:-1], changed)))
    monkeypatch.setattr(
        vision, "_finalize_test_report", lambda *args, **kwargs: pytest.fail("tamper reached test")
    )
    with pytest.raises(ValueError, match="sleep history"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store, resume_from_checkpoint=True)


def test_old_vision_checkpoint_format_rejects_before_model_restore(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, tiny_vision_backbone: None
) -> None:
    config = _sleep_config()
    store = TrustedLocalVisionCheckpointStore(tmp_path / "old-format.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(store, "after_sleep", 1),
        )
    saved = store.load()
    store.save(replace(saved, format_version=1))
    monkeypatch.setattr(
        vision, "_train_circadian", lambda *args, **kwargs: pytest.fail("old file reached training")
    )
    with pytest.raises(ValueError, match="checkpoint format"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store, resume_from_checkpoint=True)


@pytest.mark.parametrize(
    "protocol",
    [
        VISION_VALIDATION_UNMATCHED_PROTOCOL,
        VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
        VISION_SEEDED_UNMATCHED_PROTOCOL,
    ],
)
@pytest.mark.parametrize("failure_stage", ["pre", "core", "post"])
@pytest.mark.parametrize("metric_name", ["cross_entropy", "accuracy"])
def test_failed_vision_sleep_is_saved_and_resumed_in_same_epoch(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    failure_stage: str,
    metric_name: str,
) -> None:
    config = replace(
        _sleep_config(), protocol_id=protocol, circadian_sleep_rollback_metric=metric_name
    )
    controlled_score = _controlled_pc_score(False)
    failure = {"enabled": False, "guard_calls": 0}

    def score(*args: Any, **kwargs: Any) -> tuple[float, float]:
        result = controlled_score(*args, **kwargs)
        if (
            isinstance(args[1], resnet50_variants.CircadianPredictiveCodingResNet50Classifier)
            and kwargs.get("on_examples_scored") is not None
        ):
            failure["guard_calls"] += 1
            fail_on = 1 if failure_stage == "pre" else 2 if failure_stage == "post" else None
            if failure["enabled"] and failure["guard_calls"] == fail_on:
                failure["enabled"] = False
                args[1].head._chemical[0] += 0.5
                random.random()
                np.random.random()
                torch.rand(())
                raise RuntimeError(f"{failure_stage} guard failed")
        return result

    monkeypatch.setattr(vision, "_compute_pc_metrics", score)
    original_sleep = resnet50_variants.CircadianPredictiveCodingResNet50Classifier.sleep_event

    def sleep_or_fail(self: Any, **kwargs: Any) -> Any:
        if failure_stage == "core" and failure["enabled"]:
            failure["enabled"] = False
            self.head._chemical[0] += 0.5
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError("core failed")
        return original_sleep(self, **kwargs)

    class RecordingStore:
        def __init__(self, wrapped: TrustedLocalVisionCheckpointStore) -> None:
            self.wrapped = wrapped
            self.before: Any = None

        def load(self) -> Any:
            return self.wrapped.load()

        def save(self, checkpoint: Any) -> None:
            active = checkpoint.active_circadian
            if (
                active is not None
                and active.stage == "before_sleep"
                and not active.sleep_events
                and self.before is None
            ):
                self.before = checkpoint
            self.wrapped.save(checkpoint)

    control_store = TrustedLocalVisionCheckpointStore(tmp_path / "control-post.ckpt")
    control_recording = RecordingStore(control_store)
    random.seed(997)
    np.random.seed(998)
    torch.manual_seed(999)
    control = vision.run_resnet50_benchmark(config, checkpoint_store=control_recording)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    original_build = vision._build_benchmark_loaders
    final_allowed = False
    final_reads = 0

    class SealedTestLoader:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __iter__(self) -> Any:
            nonlocal final_reads
            if not final_allowed:
                raise AssertionError("vision error opened final test before resume")
            final_reads += 1
            return iter(self.wrapped)

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    monkeypatch.setattr(
        resnet50_variants.CircadianPredictiveCodingResNet50Classifier,
        "sleep_event",
        sleep_or_fail,
    )
    store = TrustedLocalVisionCheckpointStore(tmp_path / "post-error.ckpt")

    recording = RecordingStore(store)
    random.seed(997)
    np.random.seed(998)
    torch.manual_seed(999)
    failure["enabled"] = True
    failure["guard_calls"] = 0
    with pytest.raises(RuntimeError, match="guard failed|core failed"):
        vision.run_resnet50_benchmark(config, checkpoint_store=recording)
    assert final_reads == 0
    saved = store.load()
    assert recording.before is not None
    assert control_recording.before is not None
    assert recording.before.python_random_state == control_recording.before.python_random_state
    assert saved.python_random_state == recording.before.python_random_state
    np.testing.assert_equal(saved.numpy_random_state, recording.before.numpy_random_state)
    assert torch.equal(saved.torch_cpu_random_state, recording.before.torch_cpu_random_state)
    active = saved.active_circadian
    assert active is not None
    assert active.stage == "before_sleep"
    assert active.sleep_attempts == 1
    assert active.wake_batches == 2
    assert len(active.sleep_events) == 1
    error = active.sleep_events[0]
    assert error.outcome == "error"
    assert error.reason == (
        "sleep_core_exception"
        if failure_stage == "core"
        else f"inner_guard_{failure_stage}_exception"
    )
    assert error.guard is not None
    assert error.guard.role == (
        "validation" if protocol == VISION_VALIDATION_UNMATCHED_PROTOCOL else "inner_guard"
    )
    assert (
        error.guard.role_hash
        == control.split_hashes[
            "validation" if protocol == VISION_VALIDATION_UNMATCHED_PROTOCOL else "guard"
        ]
    )
    assert error.guard.metric_name == metric_name
    assert error.guard.examples_scored == (
        2 * config.guard_samples if failure_stage == "post" else config.guard_samples
    )
    assert (error.guard.pre_cross_entropy is not None) == (failure_stage != "pre")
    assert error.guard.post_cross_entropy is None
    assert bool(error.changes.proposed_split_pairs) == (failure_stage == "post")
    assert error.changes.applied_split_pairs == ()

    final_allowed = True
    resumed = vision.run_resnet50_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert final_reads > 0
    assert resumed.trained_model_hashes == control.trained_model_hashes
    for actual, expected in zip(resumed.reports, control.reports, strict=True):
        if actual.head_type != "circadian_predictive_coding":
            _same_learning_reports(
                replace(resumed, reports=[actual]), replace(control, reports=[expected])
            )
    circadian = next(
        report for report in resumed.reports if report.head_type == "circadian_predictive_coding"
    )
    assert [event.outcome for event in circadian.sleep_events[:2]] == ["error", "accepted"]
    control_circadian = next(
        report for report in control.reports if report.head_type == "circadian_predictive_coding"
    )
    assert circadian.circadian_sleep_attempts == control_circadian.circadian_sleep_attempts + 1
    for field in fields(circadian):
        if field.name not in {
            "sleep_events",
            "circadian_sleep_attempts",
            "train_seconds",
            "train_samples_per_second",
            "mean_train_step_ms",
            "inference_latency_mean_ms",
            "inference_latency_p95_ms",
            "inference_samples_per_second",
        }:
            assert getattr(circadian, field.name) == pytest.approx(
                getattr(control_circadian, field.name)
            )
    for actual_event, expected_event in zip(
        circadian.sleep_events[1:], control_circadian.sleep_events, strict=True
    ):
        actual_facts, expected_facts = asdict(actual_event), asdict(expected_event)
        actual_facts.pop("durations")
        expected_facts.pop("durations")
        assert actual_facts == expected_facts
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw
    json.dumps(asdict(resumed), allow_nan=False)


@pytest.mark.parametrize(
    ("failure_stage", "expected_scored", "reason"),
    [
        ("pre_exception", 1, "inner_guard_pre_exception"),
        ("post_exception", 4, "inner_guard_post_exception"),
        ("pre_nonfinite", 3, "inner_guard_pre_nonfinite"),
        ("post_nonfinite", 6, "inner_guard_post_nonfinite"),
    ],
)
def test_vision_guard_error_counts_completed_batches_in_partial_pass(
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    failure_stage: str,
    expected_scored: int,
    reason: str,
) -> None:
    config = replace(_sleep_config(), circadian_sleep_rollback_eval_batches=2)
    model = vision._new_circadian_classifier(torch.device("cpu"), 3, config)
    guard = [
        (torch.full((1, 3, 32, 32), 0.2), torch.tensor([1])),
        (torch.full((2, 3, 32, 32), 0.4), torch.tensor([0, 2])),
    ]
    original_predict = model.predict_logits
    calls = 0

    def predict(images: Any) -> Any:
        nonlocal calls
        calls += 1
        fail_on = 2 if failure_stage.startswith("pre") else 4
        if calls == fail_on:
            if failure_stage.endswith("exception"):
                raise RuntimeError("guard batch failed")
            return torch.full((len(images), 3), float("nan"))
        return original_predict(images)

    monkeypatch.setattr(model, "predict_logits", predict)
    decision = decide_sleep_attempt(
        sleep_mode="components",
        completed_epochs=1,
        interval_epochs=1,
        adaptive_due=False,
        force_periodic=True,
    )
    random.seed(1011)
    np.random.seed(1012)
    torch.manual_seed(1013)
    before = model.snapshot_state()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    errors: list[Any] = []

    with pytest.raises((RuntimeError, FloatingPointError), match="guard batch failed|nonfinite"):
        vision._guarded_circadian_sleep_event(
            torch,
            torch.device("cpu"),
            model,
            guard,
            config,
            1,
            True,
            2,
            decision=decision,
            guard_role_hash="a" * 64,
            on_error=errors.append,
        )
    assert len(errors) == 1
    assert errors[0].reason == reason
    assert errors[0].guard is not None
    assert errors[0].guard.examples_scored == expected_scored
    assert (errors[0].guard.pre_accuracy is not None) == failure_stage.startswith("post")
    for name, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, model.snapshot_state()[name]), name
        else:
            assert value == model.snapshot_state()[name], name
    assert random.getstate() == python_before
    np.testing.assert_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_partial_vision_guard_error_checkpoint_rejects_impossible_batch_exposure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, tiny_vision_backbone: None
) -> None:
    config = _sleep_config()
    original_score = vision._compute_pc_metrics
    fail_once = True

    def score(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal fail_once
        model = args[1]
        if (
            fail_once
            and isinstance(model, resnet50_variants.CircadianPredictiveCodingResNet50Classifier)
            and kwargs.get("on_examples_scored") is not None
        ):
            fail_once = False
            original_predict = model.predict_logits
            calls = 0

            def predict(images: Any) -> Any:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("second guard batch failed")
                return original_predict(images)

            setattr(model, "predict_logits", predict)
            try:
                return original_score(*args, **kwargs)
            finally:
                setattr(model, "predict_logits", original_predict)
        return original_score(*args, **kwargs)

    monkeypatch.setattr(vision, "_compute_pc_metrics", score)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "partial-error.ckpt")
    with pytest.raises(RuntimeError, match="second guard batch failed"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store)
    saved = store.load()
    active = saved.active_circadian
    assert active is not None
    assert active.stage == "before_sleep"
    assert len(active.sleep_events) == 1
    error = active.sleep_events[0]
    assert error.reason == "inner_guard_pre_exception"
    assert error.guard is not None
    assert error.guard.examples_scored == config.batch_size

    impossible = replace(error, guard=replace(error.guard, examples_scored=5))
    store.save(replace(saved, active_circadian=replace(active, sleep_events=(impossible,))))
    with monkeypatch.context() as temporary:
        temporary.setattr(
            vision,
            "_train_circadian",
            lambda *args, **kwargs: pytest.fail("tamper reached training"),
        )
        with pytest.raises(ValueError, match="sleep history error exposure"):
            vision.run_resnet50_benchmark(
                config, checkpoint_store=store, resume_from_checkpoint=True
            )

    store.save(saved)
    resumed = vision.run_resnet50_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    circadian = next(
        report for report in resumed.reports if report.head_type == "circadian_predictive_coding"
    )
    assert [event.outcome for event in circadian.sleep_events[:2]] == ["error", "accepted"]
    json.dumps(asdict(resumed), allow_nan=False)


@pytest.mark.parametrize(
    "protocol",
    [
        VISION_VALIDATION_UNMATCHED_PROTOCOL,
        VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
        VISION_SEEDED_UNMATCHED_PROTOCOL,
    ],
)
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "role",
        "role_hash",
        "metric",
        "exposure",
        "reason",
        "clock",
        "counter",
        "proposal",
    ],
)
def test_failed_vision_sleep_history_tamper_rejects_before_restore(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    change: str,
) -> None:
    config = replace(_sleep_config(), protocol_id=protocol)
    original_score = vision._compute_pc_metrics

    def fail_pre_guard(*args: Any, **kwargs: Any) -> tuple[float, float]:
        if (
            isinstance(args[1], resnet50_variants.CircadianPredictiveCodingResNet50Classifier)
            and kwargs.get("on_examples_scored") is not None
        ):
            raise RuntimeError("pre guard failed")
        return original_score(*args, **kwargs)

    monkeypatch.setattr(vision, "_compute_pc_metrics", fail_pre_guard)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "failed-history.ckpt")
    with pytest.raises(RuntimeError, match="pre guard failed"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store)
    saved = store.load()
    active = saved.active_circadian
    assert active is not None and active.stage == "before_sleep"
    assert len(active.sleep_events) == 1
    failed = active.sleep_events[0]
    assert failed.outcome == "error" and failed.guard is not None
    if change == "missing":
        active = replace(active, sleep_events=())
    elif change == "counter":
        active = replace(active, sleep_attempts=active.sleep_attempts + 1)
    else:
        guard = failed.guard
        changed = {
            "role": replace(
                failed,
                guard=replace(
                    guard, role="inner_guard" if guard.role == "validation" else "validation"
                ),
            ),
            "role_hash": replace(failed, guard=replace(guard, role_hash="0" * 64)),
            "metric": replace(
                failed,
                guard=replace(
                    guard,
                    metric_name="accuracy"
                    if guard.metric_name == "cross_entropy"
                    else "cross_entropy",
                ),
            ),
            "exposure": replace(failed, guard=replace(guard, examples_scored=1)),
            "reason": replace(failed, reason="inner_guard_post_exception"),
            "clock": replace(failed, wake_batches=failed.wake_batches + 1),
            "proposal": replace(failed, budgets=replace(failed.budgets, split_limit=1)),
        }[change]
        active = replace(active, sleep_events=(changed,))
    store.save(replace(saved, active_circadian=active))
    monkeypatch.setattr(
        vision, "_train_circadian", lambda *args, **kwargs: pytest.fail("tamper reached training")
    )
    with pytest.raises(ValueError, match="sleep history"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store, resume_from_checkpoint=True)


def test_two_vision_sleep_errors_keep_same_epoch_order_before_resolution(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, tiny_vision_backbone: None
) -> None:
    config = _sleep_config()
    controlled_score = _controlled_pc_score(False)
    stage = "control"
    guard_calls = 0

    def score(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal stage, guard_calls
        result = controlled_score(*args, **kwargs)
        if (
            isinstance(args[1], resnet50_variants.CircadianPredictiveCodingResNet50Classifier)
            and kwargs.get("on_examples_scored") is not None
        ):
            guard_calls += 1
            if (stage == "pre" and guard_calls == 1) or (stage == "post" and guard_calls == 2):
                failed_stage = stage
                stage = "post" if stage == "pre" else "resolved"
                raise RuntimeError(f"{failed_stage} guard failed")
        return result

    monkeypatch.setattr(vision, "_compute_pc_metrics", score)
    random.seed(1101)
    np.random.seed(1102)
    torch.manual_seed(1103)
    control = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "control-two.ckpt"),
    )

    random.seed(1101)
    np.random.seed(1102)
    torch.manual_seed(1103)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "two-errors.ckpt")
    stage = "pre"
    guard_calls = 0
    with pytest.raises(RuntimeError, match="pre guard failed"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store)
    first_active = store.load().active_circadian
    assert first_active is not None
    assert [event.reason for event in first_active.sleep_events] == ["inner_guard_pre_exception"]

    guard_calls = 0
    with pytest.raises(RuntimeError, match="post guard failed"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store, resume_from_checkpoint=True)
    saved = store.load()
    assert saved.active_circadian is not None
    assert saved.active_circadian.stage == "before_sleep"
    assert saved.active_circadian.sleep_attempts == 2
    assert [event.reason for event in saved.active_circadian.sleep_events] == [
        "inner_guard_pre_exception",
        "inner_guard_post_exception",
    ]

    guard_calls = 0
    resumed = vision.run_resnet50_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    circadian = next(
        report for report in resumed.reports if report.head_type == "circadian_predictive_coding"
    )
    control_circadian = next(
        report for report in control.reports if report.head_type == "circadian_predictive_coding"
    )
    assert circadian.circadian_sleep_attempts == control_circadian.circadian_sleep_attempts + 2
    assert [event.outcome for event in circadian.sleep_events[:3]] == ["error", "error", "accepted"]
    assert [event.completed_epoch for event in circadian.sleep_events[:3]] == [1, 1, 1]
    json.dumps(asdict(resumed), allow_nan=False)


def test_unguarded_vision_core_error_is_restored_on_explicit_resume(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, tiny_vision_backbone: None
) -> None:
    config = replace(_sleep_config(), circadian_enable_sleep_rollback=False)
    random.seed(1121)
    np.random.seed(1122)
    torch.manual_seed(1123)
    control = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "control-unguarded.ckpt"),
    )
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    original_sleep = resnet50_variants.CircadianPredictiveCodingResNet50Classifier.sleep_event
    fail_once = True

    def sleep_or_fail(self: Any, **kwargs: Any) -> Any:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            self.head._chemical[0] += 0.5
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError("unguarded vision core failed")
        return original_sleep(self, **kwargs)

    monkeypatch.setattr(
        resnet50_variants.CircadianPredictiveCodingResNet50Classifier,
        "sleep_event",
        sleep_or_fail,
    )
    random.seed(1121)
    np.random.seed(1122)
    torch.manual_seed(1123)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "unguarded-error.ckpt")
    with pytest.raises(RuntimeError, match="unguarded vision core failed"):
        vision.run_resnet50_benchmark(config, checkpoint_store=store)
    saved = store.load()
    assert saved.active_circadian is not None
    assert saved.active_circadian.stage == "before_sleep"
    assert [event.reason for event in saved.active_circadian.sleep_events] == [
        "sleep_core_exception"
    ]
    assert saved.active_circadian.sleep_events[0].guard is None

    resumed = vision.run_resnet50_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    circadian = next(
        report for report in resumed.reports if report.head_type == "circadian_predictive_coding"
    )
    assert [event.outcome for event in circadian.sleep_events[:2]] == ["error", "applied"]
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.fixture
def tiny_vision_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep repeated sleep-boundary files small while training real heads."""

    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.output = torch.nn.Linear(3, 16)

        def forward(self, images: Any) -> Any:
            return self.output(self.pool(images).flatten(1))

    def build(device: Any, freeze_backbone: bool, backbone_weights: str) -> Any:
        assert backbone_weights == "none"
        model = TinyBackbone().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(not freeze_backbone)
        return model, 16

    monkeypatch.setattr(resnet50_variants, "_build_resnet50_backbone", build)


def test_seeded_vision_file_resume_keeps_completed_model_and_final_test_sealed(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    control = vision.run_resnet50_benchmark(config)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    original_build = vision._build_benchmark_loaders
    original_train = vision._train_seeded_variant
    trained: list[str] = []

    class SealedTestLoader:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __iter__(self) -> Any:
            if len(trained) != 3:
                raise AssertionError("final test opened before all three models trained")
            return iter(self.wrapped)

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    def train(name: str, *args: Any, **kwargs: Any) -> Any:
        result = original_train(name, *args, **kwargs)
        trained.append(name)
        return result

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    monkeypatch.setattr(vision, "_train_seeded_variant", train)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))
    assert trained == ["backprop"]

    _ = (random.random(), np.random.random(), torch.rand(()))
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert trained == ["backprop", "predictive", "circadian"]
    assert resumed.training_order == control.training_order
    assert resumed.split_hashes == control.split_hashes
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


def test_vision_file_resume_from_all_trained_models_scores_final_test_once(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
) -> None:
    config = _config()
    random.seed(731)
    np.random.seed(732)
    torch.manual_seed(733)
    control = vision.run_resnet50_benchmark(config)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(731)
    np.random.seed(732)
    torch.manual_seed(733)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptingStore(store, after_variants=3),
        )
    assert store.load().next_variant_index == 3
    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args, **kwargs: pytest.fail("terminal checkpoint retrained a model"),
    )
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    "protocol",
    [VISION_VALIDATION_UNMATCHED_PROTOCOL, VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL],
)
def test_earlier_vision_file_resume_preserves_shared_loader_across_models(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
) -> None:
    config = replace(_config(), protocol_id=protocol)
    trained: list[tuple[str, str]] = []
    for name in ("backprop", "predictive", "circadian"):
        method_name = f"_train_{name}"
        original_train = getattr(vision, method_name)

        def train(
            *args: Any, _name: str = name, _original: Any = original_train, **kwargs: Any
        ) -> Any:
            outcome = _original(*args, **kwargs)
            trained.append((_name, vision._hash_trained_model(outcome.model)))
            return outcome

        monkeypatch.setattr(vision, method_name, train)

    random.seed(741)
    np.random.seed(742)
    torch.manual_seed(743)
    control = vision.run_resnet50_benchmark(config)
    model_names = {
        "backprop": "BackpropResNet50",
        "predictive": "PredictiveCodingResNet50",
        "circadian": "CircadianPredictiveCodingResNet50",
    }
    control_hashes = {model_names[name]: digest for name, digest in trained}
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    original_build = vision._build_benchmark_loaders
    trained.clear()

    class SealedTestLoader:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __iter__(self) -> Any:
            if len(trained) != 3:
                raise AssertionError("earlier protocol final test opened before training")
            return iter(self.wrapped)

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(741)
    np.random.seed(742)
    torch.manual_seed(743)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))
    assert [name for name, _ in trained] == ["backprop"]
    _ = (random.random(), np.random.random(), torch.rand(()))
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert [name for name, _ in trained] == ["backprop", "predictive", "circadian"]
    assert resumed.split_hashes == control.split_hashes
    assert resumed.trained_model_hashes == control_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    ("protocol", "stage", "reject"),
    [
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "wake", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "before_sleep", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "after_sleep", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "after_sleep", True),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "wake", False),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "before_sleep", False),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "after_sleep", False),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "after_sleep", True),
    ],
)
def test_earlier_vision_file_resume_across_guarded_sleep_matches_uninterrupted(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    stage: str,
    reject: bool,
) -> None:
    config = replace(_sleep_config(reject), protocol_id=protocol)
    monkeypatch.setattr(vision, "_compute_pc_metrics", _controlled_pc_score(reject))
    random.seed(821)
    np.random.seed(822)
    torch.manual_seed(823)
    control = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "control.ckpt"),
    )
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(821)
    np.random.seed(822)
    torch.manual_seed(823)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(
                store,
                stage,
                0 if stage == "wake" else 1,
            ),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.split_hashes == control.split_hashes
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    circadian = next(
        report
        for report in resumed.reports
        if report.model_name == "CircadianPredictiveCodingResNet50"
    )
    assert circadian.circadian_sleep_attempts > 0
    assert (
        circadian.circadian_total_rollbacks > 0 if reject else circadian.circadian_total_splits > 0
    )
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    ("protocol", "workers"),
    [
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, 0),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, 2),
    ],
)
def test_earlier_vision_file_resume_replays_shared_augmented_loader(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    workers: int,
) -> None:
    config = replace(_config(), protocol_id=protocol)
    original_build = vision._build_benchmark_loaders

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        train_loader = torch.utils.data.DataLoader(
            StochasticVisionDataset(loaders.train_loader.dataset),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=workers,
            generator=torch.Generator().manual_seed(99),
        )
        return replace(loaders, train_loader=train_loader)

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    random.seed(941)
    np.random.seed(942)
    torch.manual_seed(943)
    control = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "control.ckpt"),
    )
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(941)
    np.random.seed(942)
    torch.manual_seed(943)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    ("protocol", "damage"),
    [
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "shared"),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "cursor"),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "replay"),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "data"),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "data"),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "checksum"),
    ],
)
def test_earlier_vision_file_resume_rejects_incompatible_state_before_training(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    damage: str,
) -> None:
    config = replace(_sleep_config(), protocol_id=protocol)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )
    saved = store.load()
    active = saved.active_circadian
    assert active is not None
    if damage == "shared":
        store.save(
            replace(
                saved,
                shared_train_generator_state=torch.Generator().manual_seed(99).get_state(),
            )
        )
    elif damage == "cursor":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    loader_state=replace(active.loader_state, next_batch_index=0),
                ),
            )
        )
    elif damage == "replay":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    loader_state=replace(
                        active.loader_state,
                        epoch_entry_generator_state=(
                            torch.Generator().manual_seed(999).get_state()
                        ),
                    ),
                ),
            )
        )
    elif damage == "data":
        original_build = vision._build_benchmark_loaders

        def changed_data(config: ResNet50BenchmarkConfig) -> Any:
            loaders = original_build(config)
            labels = loaders.guard_loader.dataset.labels
            labels[0] = (labels[0] + 1) % config.num_classes
            return loaders

        monkeypatch.setattr(vision, "_build_benchmark_loaders", changed_data)
    else:
        path = tmp_path / "vision.ckpt"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))

    if damage != "replay":
        monkeypatch.setattr(
            vision,
            "_train_circadian",
            lambda *args, **kwargs: pytest.fail("training began before shared-stream preflight"),
        )
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="checkpoint"):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )
    assert random.getstate() == before_python
    np.testing.assert_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_vision_file_resume_rejects_changed_development_data_before_training(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
) -> None:
    config = _config()
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))

    original_build = vision._build_benchmark_loaders

    def changed_data(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        labels = loaders.train_loader.dataset.labels
        labels[0] = (labels[0] + 1) % config.num_classes
        return loaders

    monkeypatch.setattr(vision, "_build_benchmark_loaders", changed_data)
    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args: pytest.fail("training started before data validation"),
    )
    with pytest.raises(ValueError, match="data"):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )


@pytest.mark.parametrize("damage", ["samples", "steps", "metric", "hash", "model"])
def test_vision_file_resume_rejects_bad_completed_baseline_before_training(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    damage: str,
) -> None:
    config = _config()
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))
    saved = store.load()
    baseline = saved.completed_outcomes[0]
    if damage == "samples":
        changed = replace(baseline, seen_samples=baseline.seen_samples - 1)
    elif damage == "steps":
        changed = replace(baseline, step_times_ms=baseline.step_times_ms[:-1])
    elif damage == "metric":
        changed = replace(baseline, validation_accuracy=float("nan"))
    elif damage == "model":
        model = baseline.model
        state = model.state
        key = next(iter(state.output_state))
        changed = replace(
            baseline,
            model=replace(
                model,
                state=replace(
                    state,
                    output_state={**state.output_state, key: torch.zeros(1)},
                ),
            ),
        )
    else:
        store.save(replace(saved, completed_hashes=("wrong",)))
        changed = baseline
    if damage != "hash":
        store.save(replace(saved, completed_outcomes=(changed,)))

    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args, **kwargs: pytest.fail("training started before baseline preflight"),
    )
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="checkpoint"):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )
    assert random.getstate() == before_python
    np.testing.assert_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.get_rng_state(), before_torch)


@pytest.mark.parametrize(
    ("stage", "reject", "model_order"),
    [
        ("wake", False, None),
        ("before_sleep", False, None),
        ("after_sleep", False, None),
        ("after_sleep", True, None),
        ("wake", False, ("circadian", "predictive", "backprop")),
    ],
)
def test_vision_file_resume_across_guarded_sleep_matches_uninterrupted_state(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    stage: str,
    reject: bool,
    model_order: tuple[str, ...] | None,
) -> None:
    config = _sleep_config(reject)
    monkeypatch.setattr(vision, "_compute_pc_metrics", _controlled_pc_score(reject))
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    control = vision.run_resnet50_benchmark(config, model_order=model_order)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=model_order,
            checkpoint_store=InterruptAtActiveStage(store, stage, 0 if stage == "wake" else 1),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        model_order=model_order,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.training_order == control.training_order
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    circadian = next(
        report
        for report in resumed.reports
        if report.model_name == "CircadianPredictiveCodingResNet50"
    )
    assert circadian.circadian_sleep_attempts > 0
    assert (
        circadian.circadian_total_rollbacks > 0 if reject else circadian.circadian_total_splits > 0
    )
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize("workers", [0, 2])
def test_vision_file_resume_replays_augmented_train_loader(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    workers: int,
) -> None:
    config = _config()
    order = ("circadian", "predictive", "backprop")
    original_build = vision._build_benchmark_loaders

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        train_loader = torch.utils.data.DataLoader(
            StochasticVisionDataset(loaders.train_loader.dataset),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=workers,
            generator=torch.Generator().manual_seed(99),
        )
        return replace(loaders, train_loader=train_loader)

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    random.seed(921)
    np.random.seed(922)
    torch.manual_seed(923)
    control = vision.run_resnet50_benchmark(config, model_order=order)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(921)
    np.random.seed(922)
    torch.manual_seed(923)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        model_order=order,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    "damage",
    ["cursor", "counter", "classifier", "rng", "checksum", "order", "config", "data"],
)
def test_vision_file_resume_rejects_bad_active_progress_before_training_or_rng_change(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    damage: str,
) -> None:
    config = _sleep_config()
    order = ("circadian", "predictive", "backprop")
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )

    saved = store.load()
    active = saved.active_circadian
    assert active is not None
    if damage == "cursor":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    loader_state=replace(active.loader_state, next_batch_index=0),
                ),
            )
        )
    elif damage == "counter":
        store.save(
            replace(
                saved,
                active_circadian=replace(active, wake_batches=active.wake_batches + 1),
            )
        )
    elif damage == "classifier":
        classifier = active.classifier_state
        key = next(iter(classifier.backbone_state))
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    classifier_state=replace(
                        classifier,
                        backbone_state={**classifier.backbone_state, key: torch.zeros(1)},
                    ),
                ),
            )
        )
    elif damage == "rng":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    outer_entry_torch_state=torch.empty(0, dtype=torch.uint8),
                ),
            )
        )
    elif damage == "checksum":
        path = tmp_path / "vision.ckpt"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    elif damage == "order":
        order = ("backprop", "predictive", "circadian")
    elif damage == "config":
        config = replace(config, circadian_learning_rate=config.circadian_learning_rate / 2)
    else:
        original_build = vision._build_benchmark_loaders

        def changed_data(config: ResNet50BenchmarkConfig) -> Any:
            loaders = original_build(config)
            labels = loaders.guard_loader.dataset.labels
            labels[0] = (labels[0] + 1) % config.num_classes
            return loaders

        monkeypatch.setattr(vision, "_build_benchmark_loaders", changed_data)

    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args, **kwargs: pytest.fail("training started before checkpoint preflight"),
    )
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="checkpoint"):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )
    assert random.getstate() == before_python
    np.testing.assert_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.get_rng_state(), before_torch)
