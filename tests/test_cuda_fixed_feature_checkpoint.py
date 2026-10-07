"""Actual-device fixed-feature checkpoint continuation without memory claims."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
import random
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")

from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.core.resnet50_variants import CircadianPredictiveCodingHead  # noqa: E402
from src.infra.circadian_checkpoint_files import (  # noqa: E402
    TrustedLocalCircadianCheckpointStore,
)
from src.infra.local_result_json import write_local_result_json  # noqa: E402


class InterruptedCudaRun(Exception):
    pass


class InterruptingStore:
    def __init__(self, store: TrustedLocalCircadianCheckpointStore, stage: str) -> None:
        self.store = store
        self.stage = stage

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if checkpoint.combined.position.stage == self.stage:
            raise InterruptedCudaRun()


class CaptureBeforeSleepStore:
    def __init__(self, store: TrustedLocalCircadianCheckpointStore) -> None:
        self.store = store
        self.entry: Any = None

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        if (
            self.entry is None
            and checkpoint.combined.position.stage == "before_sleep"
            and not checkpoint.sleep_events
        ):
            self.entry = checkpoint
        self.store.save(checkpoint)


@pytest.fixture(autouse=True)
def preserve_random_streams() -> Any:
    if not torch.cuda.is_available():
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        epochs=2,
        target_accuracy=None,
        device="cuda:0",
        predictive_head_hidden_dim=4,
        circadian_head_hidden_dim=4,
        circadian_min_hidden_dim=3,
        circadian_max_hidden_dim=6,
        circadian_inference_steps=2,
        circadian_sleep_mode="components",
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_use_adaptive_sleep_trigger=False,
        circadian_use_adaptive_sleep_budget=False,
        circadian_use_adaptive_thresholds=False,
        circadian_sleep_warmup_steps=0,
        circadian_split_threshold=0.8,
        circadian_split_weight_norm_mix=0.0,
        circadian_split_importance_mix=0.0,
        circadian_split_hysteresis_margin=0.0,
        circadian_split_cooldown_steps=0,
        circadian_max_split_per_sleep=1,
        circadian_max_prune_per_sleep=0,
        circadian_sleep_enable_prune=False,
        circadian_sleep_enable_homeostasis=False,
        circadian_sleep_enable_chemical_reset=False,
        circadian_use_dual_chemical=False,
        circadian_sleep_rollback_eval_batches=1,
        evaluation_batches=1,
        train_samples=4,
        guard_samples=2,
        validation_samples=2,
        test_samples=2,
        batch_size=2,
        backprop_freeze_backbone=True,
        backbone_weights="none",
    )


def _head(config: ResNet50BenchmarkConfig) -> CircadianPredictiveCodingHead:
    head = CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cuda:0"),
        seed=241,
        config=matched._build_circadian_head_config(config),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )
    head._chemical = torch.tensor([0.95, 0.0, 0.0, 0.0], device="cuda:0")
    return head


def _batches() -> tuple[Any, Any, Any]:
    train = (
        (torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]]), torch.tensor([1, 0])),
        (torch.tensor([[0.1, 0.2, -0.4], [0.3, -0.2, 0.1]]), torch.tensor([0, 1])),
    )
    guard = ((torch.tensor([[0.2, 0.1, -0.1]]), torch.tensor([1])),)
    validation = ((torch.tensor([[0.1, -0.2, 0.3]]), torch.tensor([0])),)
    return train, guard, validation


def _seed_process() -> None:
    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    torch.cuda.manual_seed_all(703)


def _run(
    head: CircadianPredictiveCodingHead,
    config: ResNet50BenchmarkConfig,
    *,
    store: Any = None,
    resume: bool = False,
) -> Any:
    return matched._train_circadian_head(
        torch,
        torch.device("cuda:0"),
        head,
        *_batches(),
        config,
        checkpoint_store=store,
        resume_from_checkpoint=resume,
    )


def _assert_head_equal(actual: Any, expected: Any) -> None:
    left, right = actual.snapshot_state(), expected.snapshot_state()
    assert left.keys() == right.keys()
    for name, value in left.items():
        if torch.is_tensor(value):
            assert torch.equal(value, right[name]), name
        else:
            assert value == right[name], name


def _draw_process() -> tuple[Any, ...]:
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), device="cuda:0")),
    )


@pytest.mark.parametrize(
    ("stage", "reject"),
    [("wake", False), ("after_sleep", False), ("after_sleep", True)],
)
def test_cuda_file_resume_matches_learning_and_all_random_streams(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, stage: str, reject: bool
) -> None:
    config = _config()
    original_evaluate = matched._evaluate_head
    guard_calls = 0

    def evaluate(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal guard_calls
        result = original_evaluate(*args, **kwargs)
        if kwargs.get("on_examples_scored") is not None:
            guard_calls += 1
            return (0.7, 1.0) if reject and guard_calls % 2 == 0 else (0.8, 0.5)
        return result

    monkeypatch.setattr(matched, "_evaluate_head", evaluate)
    _seed_process()
    control_head = _head(config)
    control = _run(control_head, config)
    expected_draw = _draw_process()

    guard_calls = 0
    _seed_process()
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "cuda.ckpt")
    with pytest.raises(InterruptedCudaRun):
        _run(_head(config), config, store=InterruptingStore(store, stage))
    saved = store.load()
    assert saved.combined.torch_cuda_device == "cuda:0"
    assert torch.is_tensor(saved.combined.torch_cuda_random_state)
    torch.rand((100,), device="cuda:0")  # The checkpoint must replace later process draws.

    resumed_head = _head(config)
    resumed = _run(resumed_head, config, store=store, resume=True)
    _assert_head_equal(resumed_head, control_head)
    for field in fields(resumed):
        if field.name == "sleep_events":
            left = [asdict(event) for event in resumed.sleep_events]
            right = [asdict(event) for event in control.sleep_events]
            for event in (*left, *right):
                event.pop("durations")
            assert left == right
            continue
        if field.name not in {"head", "train_seconds"}:
            assert getattr(resumed, field.name) == getattr(control, field.name), field.name
    assert _draw_process() == expected_draw
    assert resumed.total_rollbacks >= 1 if reject else resumed.total_splits >= 1
    assert resumed.sleep_events[0].outcome == ("rolled_back" if reject else "accepted")
    assert resumed.sleep_events[0].guard is not None
    assert resumed.sleep_events[0].guard.role == "inner_guard"
    assert resumed.sleep_events[0].guard.examples_scored == 2
    if reject:
        assert resumed.sleep_events[0].changes.proposed_split_pairs
        assert resumed.sleep_events[0].changes.applied_split_pairs == ()


@pytest.mark.parametrize("failure_stage", ["pre", "core", "post"])
@pytest.mark.parametrize("metric_name", ["cross_entropy", "accuracy"])
def test_cuda_failed_fixed_feature_sleep_resumes_with_error_and_rng(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, failure_stage: str, metric_name: str
) -> None:
    config = replace(_config(), circadian_sleep_rollback_metric=metric_name)
    failure_enabled = False
    guard_calls = 0

    def disturb_random_and_head(head: Any) -> None:
        head._chemical[0] += 0.5
        random.random()
        np.random.random()
        torch.rand(())
        torch.rand((), device="cuda:0")
        torch.rand((), device="cuda:0", generator=head._split_generator)

    original_evaluate = matched._evaluate_head

    def evaluate(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal failure_enabled, guard_calls
        result = original_evaluate(*args, **kwargs)
        if kwargs.get("on_examples_scored") is not None:
            guard_calls += 1
            if failure_enabled and (
                (failure_stage == "pre" and guard_calls == 1)
                or (failure_stage == "post" and guard_calls == 2)
            ):
                disturb_random_and_head(args[2].__self__)
                failure_enabled = False
                raise RuntimeError(f"CUDA fixed-feature {failure_stage} failed")
            return 0.8, 0.5
        return result

    monkeypatch.setattr(matched, "_evaluate_head", evaluate)
    original_sleep = CircadianPredictiveCodingHead.sleep_event

    def sleep_or_fail(self: Any, **kwargs: Any) -> Any:
        nonlocal failure_enabled
        if failure_enabled and failure_stage == "core":
            disturb_random_and_head(self)
            failure_enabled = False
            raise RuntimeError("CUDA fixed-feature core failed")
        return original_sleep(self, **kwargs)

    monkeypatch.setattr(CircadianPredictiveCodingHead, "sleep_event", sleep_or_fail)
    _seed_process()
    control_head = _head(config)
    control = _run(control_head, config)
    expected_draw = _draw_process()

    _seed_process()
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "cuda-error.ckpt")
    recording = CaptureBeforeSleepStore(store)
    failed_head = _head(config)
    failure_enabled = True
    guard_calls = 0
    with pytest.raises(RuntimeError, match="CUDA fixed-feature .* failed"):
        _run(failed_head, config, store=recording)
    saved = store.load()
    entry = recording.entry
    assert entry is not None
    assert saved.combined.position.stage == "before_sleep"
    assert saved.progress.sleep_attempts == 1
    assert saved.combined.python_random_state == entry.combined.python_random_state
    np.testing.assert_equal(saved.combined.numpy_random_state, entry.combined.numpy_random_state)
    assert torch.equal(saved.combined.torch_cpu_random_state, entry.combined.torch_cpu_random_state)
    assert torch.equal(
        saved.combined.torch_cuda_random_state, entry.combined.torch_cuda_random_state
    )
    assert isinstance(saved.combined.model_state, dict)
    assert isinstance(entry.combined.model_state, dict)
    assert torch.equal(
        saved.combined.model_state["split_generator_state"],
        entry.combined.model_state["split_generator_state"],
    )
    assert len(saved.sleep_events) == 1
    error = saved.sleep_events[0]
    assert error.outcome == "error"
    assert error.reason == (
        "sleep_core_exception"
        if failure_stage == "core"
        else f"inner_guard_{failure_stage}_exception"
    )
    assert error.guard is not None
    assert error.guard.role == "inner_guard"
    assert error.guard.metric_name == metric_name
    assert error.guard.examples_scored == (2 if failure_stage == "post" else 1)
    assert (error.guard.pre_accuracy is not None) == (failure_stage != "pre")
    assert bool(error.changes.proposed_split_pairs) == (failure_stage == "post")
    assert error.changes.applied_split_pairs == ()

    resumed_head = _head(config)
    resumed = _run(resumed_head, config, store=store, resume=True)
    _assert_head_equal(resumed_head, control_head)
    assert resumed.sleep_attempts == control.sleep_attempts + 1
    assert [event.outcome for event in resumed.sleep_events[:2]] == ["error", "accepted"]
    for actual_event, expected_event in zip(
        resumed.sleep_events[1:], control.sleep_events, strict=True
    ):
        actual_facts, expected_facts = asdict(actual_event), asdict(expected_event)
        actual_facts.pop("durations")
        expected_facts.pop("durations")
        assert actual_facts == expected_facts
    for field in fields(resumed):
        if field.name not in {"head", "train_seconds", "sleep_attempts", "sleep_events"}:
            assert getattr(resumed, field.name) == getattr(control, field.name)
    assert _draw_process() == expected_draw


@pytest.mark.parametrize("change", ["missing_cuda_rng", "wrong_cuda_device", "bad_cuda_rng"])
def test_cuda_resume_rejects_bad_process_state_before_head_mutation(
    tmp_path: Any, change: str
) -> None:
    config = _config()
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "bad-cuda.ckpt")
    with pytest.raises(InterruptedCudaRun):
        _run(_head(config), config, store=InterruptingStore(store, "wake"))
    saved = store.load()
    if change == "missing_cuda_rng":
        combined = replace(saved.combined, torch_cuda_random_state=None)
    elif change == "wrong_cuda_device":
        combined = replace(saved.combined, torch_cuda_device="cuda:1")
    else:
        combined = replace(
            saved.combined, torch_cuda_random_state=torch.zeros(2, dtype=torch.int32)
        )
    store.save(replace(saved, combined=combined))

    head = _head(config)
    before = head.snapshot_state()
    process_before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        torch.cuda.get_rng_state(0).clone(),
    )
    with pytest.raises(ValueError, match="checkpoint|CUDA"):
        _run(head, config, store=store, resume=True)
    after = head.snapshot_state()
    for name, value in before.items():
        assert torch.equal(value, after[name]) if torch.is_tensor(value) else value == after[name]
    assert random.getstate() == process_before[0]
    assert np.array_equal(cast(Any, np.random.get_state())[1], cast(Any, process_before[1])[1])
    assert torch.equal(torch.get_rng_state(), process_before[2])
    assert torch.equal(torch.cuda.get_rng_state(0), process_before[3])


def test_cuda_public_route_keeps_final_test_sealed_until_resume(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    train, guard, validation = _batches()
    completed = [False]
    test_reads = [0]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            if not completed[0]:
                raise AssertionError("CUDA final test opened before resumed training")
            test_reads[0] += 1
            return iter(validation)

    monkeypatch.setattr(
        matched,
        "_build_benchmark_loaders",
        lambda _config: SimpleNamespace(
            train_loader=list(train),
            guard_loader=list(guard),
            validation_loader=list(validation),
            test_loader=SealedTestLoader(),
            num_classes=2,
            split_hashes={role: role for role in ("train", "guard", "validation", "test")},
        ),
    )
    monkeypatch.setattr(
        matched, "_build_resnet50_backbone", lambda **kwargs: (torch.nn.Identity(), 3)
    )
    original_train = matched._train_circadian_head

    def tracked_train(*args: Any, **kwargs: Any) -> Any:
        outcome = original_train(*args, **kwargs)
        completed[0] = True
        return outcome

    monkeypatch.setattr(matched, "_train_circadian_head", tracked_train)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "public-cuda.ckpt")
    with pytest.raises(InterruptedCudaRun):
        matched.run_three_head_fixed_feature_benchmark(
            config, checkpoint_store=InterruptingStore(store, "before_sleep")
        )
    assert test_reads == [0]
    resumed = matched.run_three_head_fixed_feature_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert test_reads == [1]
    assert resumed.circadian.sleep_attempts >= 1
    assert resumed.memory_telemetry_enabled is False
    assert resumed.circadian.sleep_events
    assert resumed.circadian.sleep_events[0].guard is not None
    assert resumed.circadian.sleep_events[0].guard.role == "inner_guard"
    result_path = tmp_path / "cuda-fixed-feature-result.json"
    write_local_result_json(resumed, result_path)
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_result["circadian"]["sleep_events"][0]["guard"]["role"] == "inner_guard"
    assert saved_result["backprop"]["sleep_events"] == []
    assert saved_result["predictive_coding"]["sleep_events"] == []


def test_cuda_checkpoint_memory_reports_device_segments(tmp_path: Any) -> None:
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "fixed-epoch-memory.ckpt")
    result = matched.run_three_head_fixed_feature_benchmark(
        _config(), checkpoint_store=store, measure_memory=True
    )
    assert result.protocol_id == matched.THREE_HEAD_FIXED_FEATURE_CUDA_CHECKPOINT_MEMORY_PROTOCOL
    assert result.memory_observation_scope == matched.CHECKPOINT_CUDA_MEMORY_SCOPE
    assert all(
        len(report.cuda_allocator_segments) == 1
        for report in (result.backprop, result.predictive_coding, result.circadian)
    )


def test_cuda_alias_rejects_a_different_current_device_before_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _head(_config())
    before = head.snapshot_state()
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "wrong-alias.ckpt")
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "current_device", lambda: 1)
        with pytest.raises(ValueError, match="head/device mismatch"):
            matched._train_circadian_head(
                torch, torch.device("cuda"), head, *_batches(), _config(), checkpoint_store=store
            )
    _assert_head_equal(head, SimpleNamespace(snapshot_state=lambda: before))
    assert not store.path.exists()


def test_cuda_capacity_checkpoint_without_memory_preserves_control(tmp_path: Any) -> None:
    config = replace(
        _config(),
        circadian_min_hidden_dim=4,
        circadian_max_hidden_dim=4,
        circadian_sleep_enable_split=True,
        circadian_sleep_enable_prune=True,
        circadian_max_prune_per_sleep=1,
    )
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "capacity-only.ckpt")
    result = matched.run_three_head_fixed_width_capacity_benchmark(config, checkpoint_store=store)
    assert result.capacity_control is not None
    assert result.memory_telemetry_enabled is False
    assert store.path.exists()


def test_cuda_wall_time_checkpoint_without_memory_reaches_deadlines(tmp_path: Any) -> None:
    config = replace(_config(), epochs=1000)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "wall-time-only.ckpt")
    result = matched.run_three_head_fixed_feature_wall_time_benchmark(
        config, wall_time_budget_seconds=0.1, checkpoint_store=store
    )
    assert all(
        report.stop_reason == "deadline"
        for report in (result.backprop, result.predictive_coding, result.circadian)
    )
    assert result.memory_telemetry_enabled is False
