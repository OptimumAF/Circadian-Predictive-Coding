"""Trusted local checkpoints resume the real fixed-feature guarded head loop."""

from __future__ import annotations

from dataclasses import fields, replace
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.core.resnet50_variants import CircadianPredictiveCodingHead  # noqa: E402
from src.infra.circadian_checkpoint_files import (  # noqa: E402
    TrustedLocalCircadianCheckpointStore,
)


class IntentionalInterruption(Exception):
    pass


class InterruptingStore:
    def __init__(
        self, store: TrustedLocalCircadianCheckpointStore, *, stage: str, epoch: int
    ) -> None:
        self.store = store
        self.stage = stage
        self.epoch = epoch
        self.interrupted = False

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        position = checkpoint.combined.position
        if (
            not self.interrupted
            and position.stage == self.stage
            and (position.completed_epoch == self.epoch)
        ):
            self.interrupted = True
            raise IntentionalInterruption()


@pytest.fixture(autouse=True)
def preserve_random_streams() -> Any:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        epochs=4,
        target_accuracy=None,
        device="cpu",
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
    )


def _head(config: ResNet50BenchmarkConfig) -> CircadianPredictiveCodingHead:
    head = CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=241,
        config=matched._build_circadian_head_config(config),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )
    head._chemical = torch.tensor([0.95, 0.0, 0.0, 0.0])
    return head


def _batches() -> tuple[Any, Any, Any]:
    train = (
        (torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]]), torch.tensor([1, 0])),
        (torch.tensor([[0.1, 0.2, -0.4], [0.3, -0.2, 0.1]]), torch.tensor([0, 1])),
    )
    guard = ((torch.tensor([[0.2, 0.1, -0.1]]), torch.tensor([1])),)
    validation = ((torch.tensor([[0.1, -0.2, 0.3]]), torch.tensor([0])),)
    return train, guard, validation


def _same_head(
    actual: CircadianPredictiveCodingHead, expected: CircadianPredictiveCodingHead
) -> None:
    left = actual.snapshot_state()
    right = expected.snapshot_state()
    assert left.keys() == right.keys()
    for name, value in left.items():
        if torch.is_tensor(value):
            assert torch.equal(value, right[name]), name
        else:
            assert value == right[name], name


def _same_report(actual: Any, expected: Any) -> None:
    for field in fields(actual):
        if field.name not in {"head", "train_seconds"}:
            assert getattr(actual, field.name) == getattr(expected, field.name), field.name


def _run(
    head: CircadianPredictiveCodingHead,
    config: ResNet50BenchmarkConfig,
    *,
    store: Any = None,
    resume: bool = False,
    train: Any = None,
    time_budget_seconds: float | None = None,
    clock: Any = None,
) -> Any:
    train_batches, guard, validation = _batches()
    return matched._train_circadian_head(
        torch,
        torch.device("cpu"),
        head,
        train_batches if train is None else train,
        guard,
        validation,
        config,
        time_budget_seconds=time_budget_seconds,
        clock=clock,
        checkpoint_store=store,
        resume_from_checkpoint=resume,
    )


@pytest.mark.parametrize(
    ("stage", "epoch", "reject"),
    [
        ("wake", 0, False),
        ("before_sleep", 1, False),
        ("after_sleep", 1, False),
        ("after_sleep", 1, True),
    ],
)
def test_file_checkpoint_matches_uninterrupted_guarded_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, stage: str, epoch: int, reject: bool
) -> None:
    config = _config()
    monkeypatch.setattr(
        matched,
        "_evaluate_head",
        lambda _torch, _device, predict, *_args: (
            0.8,
            0.6 if reject and predict.__self__.hidden_dim > 4 else 0.5,
        ),
    )
    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    control_head = _head(config)
    control = _run(control_head, config)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "head.ckpt")
    interrupting = InterruptingStore(store, stage=stage, epoch=epoch)
    with pytest.raises(IntentionalInterruption):
        _run(_head(config), config, store=interrupting)
    assert (tmp_path / "head.ckpt").is_file()
    assert store.load().combined.position.stage == stage

    resumed_head = _head(config)
    resumed = _run(resumed_head, config, store=store, resume=True)
    _same_report(resumed, control)
    _same_head(resumed_head, control_head)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw
    assert resumed.total_splits >= 1 if not reject else resumed.total_rollbacks >= 1


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "protocol",
        "features",
        "corrupt",
        "checksum",
        "counters",
        "initial_head",
        "retry",
        "cursor",
    ],
)
def test_bad_file_rejects_before_training_or_head_mutation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    config = _config()
    monkeypatch.setattr(matched, "_evaluate_head", lambda *args: (0.8, 0.5))
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "head.ckpt")
    with pytest.raises(IntentionalInterruption):
        _run(
            _head(config),
            config,
            store=InterruptingStore(store, stage="before_sleep", epoch=1),
        )
    head = _head(config)
    before = head.snapshot_state()
    train = None
    if change == "config":
        config = replace(config, circadian_learning_rate=0.04)
        head = _head(config)
        before = head.snapshot_state()
    elif change == "protocol":
        config = replace(config, protocol_id="other_protocol")
    elif change == "features":
        train, _, _ = _batches()
        changed = train[0][0].clone()
        changed[0, 0] += 0.1
        train = ((changed, train[0][1]), train[1])
    elif change == "corrupt":
        (tmp_path / "head.ckpt").write_bytes(b"corrupt file")
    elif change == "checksum":
        path = tmp_path / "head.ckpt"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    elif change == "counters":
        saved = store.load()
        store.save(replace(saved, progress=replace(saved.progress, wake_batches=999)))
    elif change == "initial_head":
        saved = store.load()
        store.save(replace(saved, initial_head_hash="0" * 64))
    elif change == "retry":
        saved = store.load()
        assert saved.combined.retry_state is not None
        retry_state = replace(saved.combined.retry_state, rejected_attempts=1)
        store.save(replace(saved, combined=replace(saved.combined, retry_state=retry_state)))
    else:
        saved = store.load()
        position = replace(
            saved.combined.position, completed_epoch=0, stage="wake", next_batch_index=1
        )
        store.save(replace(saved, combined=replace(saved.combined, position=position)))
    with pytest.raises((TypeError, ValueError), match="checkpoint|corrupt|incompatible"):
        _run(head, config, store=store, resume=True, train=train)
    left = head.snapshot_state()
    assert left.keys() == before.keys()
    for name, value in left.items():
        if torch.is_tensor(value):
            assert torch.equal(value, before[name]), name
        else:
            assert value == before[name], name


def test_public_matched_route_resumes_before_final_test(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(),
        epochs=2,
        train_samples=4,
        guard_samples=2,
        validation_samples=2,
        test_samples=2,
        batch_size=2,
        predictive_head_hidden_dim=4,
        predictive_inference_steps=2,
        backprop_freeze_backbone=True,
        backbone_weights="none",
    )
    features, guard, validation = _batches()
    finished = [False]
    test_accesses = [0]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            if not finished[0]:
                raise AssertionError("final test opened before the circadian head finished")
            test_accesses[0] += 1
            return iter(validation)

    def loaders(_config: Any) -> Any:
        return SimpleNamespace(
            train_loader=list(features),
            guard_loader=list(guard),
            validation_loader=list(validation),
            test_loader=SealedTestLoader(),
            num_classes=2,
            split_hashes={role: role for role in ("train", "guard", "validation", "test")},
        )

    monkeypatch.setattr(matched, "_build_benchmark_loaders", loaders)
    monkeypatch.setattr(
        matched,
        "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 3),
    )
    monkeypatch.setattr(matched, "_evaluate_head", lambda *args: (0.8, 0.5))
    original_train = matched._train_circadian_head

    def tracked_train(*args: Any, **kwargs: Any) -> Any:
        outcome = original_train(*args, **kwargs)
        finished[0] = True
        return outcome

    monkeypatch.setattr(matched, "_train_circadian_head", tracked_train)
    control = matched.run_three_head_fixed_feature_benchmark(config)
    assert test_accesses == [1]

    finished[0] = False
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "public.ckpt")
    with pytest.raises(IntentionalInterruption):
        matched.run_three_head_fixed_feature_benchmark(
            config,
            checkpoint_store=InterruptingStore(store, stage="before_sleep", epoch=1),
        )
    assert test_accesses == [1]

    resumed = matched.run_three_head_fixed_feature_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert test_accesses == [2]
    assert resumed.feature_hashes == control.feature_hashes
    assert resumed.trained_head_hashes == control.trained_head_hashes
    assert resumed.circadian.epochs_ran == control.circadian.epochs_ran
    assert resumed.circadian.wake_batches == control.circadian.wake_batches
    assert resumed.circadian.sleep_attempts == control.circadian.sleep_attempts
    assert resumed.circadian.test_accuracy == control.circadian.test_accuracy
    assert resumed.backprop.test_accuracy == control.backprop.test_accuracy
    assert resumed.predictive_coding.test_accuracy == control.predictive_coding.test_accuracy


def test_checkpoint_mode_rejects_cuda_before_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_config(), predictive_head_hidden_dim=4, backprop_freeze_backbone=True)
    monkeypatch.setattr(
        matched,
        "_build_benchmark_loaders",
        lambda _config: pytest.fail("checkpoint mode reached data loading"),
    )
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "head.ckpt")
    head = _head(config)
    before = head.snapshot_state()
    with pytest.raises(ValueError, match="CPU device"):
        matched._train_circadian_head(
            torch,
            "cuda:0",
            head,
            *_batches(),
            config,
            time_budget_seconds=1.0,
            checkpoint_store=store,
        )
    for name, value in head.snapshot_state().items():
        if torch.is_tensor(value):
            assert torch.equal(value, before[name]), name
        else:
            assert value == before[name], name


@pytest.mark.parametrize(
    ("stage", "reject"),
    [("before_sleep", False), ("after_sleep", False), ("after_sleep", True)],
)
def test_wall_time_file_resume_uses_remaining_active_deadline(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, stage: str, reject: bool
) -> None:
    class TrainingClock:
        def __init__(self) -> None:
            self.time = 0.0

        def read(self) -> float:
            return self.time

        def advance(self, seconds: float) -> None:
            self.time += seconds

    class TimedStore:
        def __init__(
            self,
            store: TrustedLocalCircadianCheckpointStore,
            clock: TrainingClock,
            interrupt: bool = False,
            target_stage: str = "before_sleep",
        ) -> None:
            self.store = store
            self.clock = clock
            self.interrupt = interrupt
            self.target_stage = target_stage

        def load(self) -> Any:
            return self.store.load()

        def save(self, checkpoint: Any) -> None:
            self.store.save(checkpoint)
            self.clock.advance(0.25)
            position = checkpoint.combined.position
            if self.interrupt and position.stage == self.target_stage:
                raise IntentionalInterruption()

    config = _config()
    monkeypatch.setattr(
        matched,
        "_evaluate_head",
        lambda _torch, _device, predict, *_args: (
            0.8,
            0.6 if reject and predict.__self__.hidden_dim > 4 else 0.5,
        ),
    )

    def timed_head(clock: TrainingClock) -> CircadianPredictiveCodingHead:
        head = _head(config)
        original = head.train_step

        def step(**kwargs: Any) -> Any:
            result = original(**kwargs)
            clock.advance(1.0)
            return result

        monkeypatch.setattr(head, "train_step", step)
        return head

    control_clock = TrainingClock()
    control_head = timed_head(control_clock)
    control = _run(control_head, config, time_budget_seconds=5.0, clock=control_clock.read)
    assert control.stop_reason == "deadline"
    assert control.wake_batches == 5
    assert control.train_seconds == 5.0

    store = TrustedLocalCircadianCheckpointStore(tmp_path / "wall.ckpt")
    interrupted_clock = TrainingClock()
    with pytest.raises(IntentionalInterruption):
        _run(
            timed_head(interrupted_clock),
            config,
            store=TimedStore(store, interrupted_clock, interrupt=True, target_stage=stage),
            time_budget_seconds=5.0,
            clock=interrupted_clock.read,
        )
    saved = store.load()
    assert saved.combined.position.stage == stage
    assert saved.protocol_id == matched.THREE_HEAD_FIXED_FEATURE_WALL_TIME_PROTOCOL
    assert saved.progress.elapsed_seconds == 2.0

    resumed_clock = TrainingClock()
    resumed_head = timed_head(resumed_clock)
    resumed = _run(
        resumed_head,
        config,
        store=TimedStore(store, resumed_clock),
        resume=True,
        time_budget_seconds=5.0,
        clock=resumed_clock.read,
    )
    _same_report(resumed, control)
    _same_head(resumed_head, control_head)
    assert resumed.train_seconds == control.train_seconds == 5.0
    assert resumed_clock.read() > 3.0  # File I/O advanced wall time outside the active budget.

    incompatible_head = timed_head(TrainingClock())
    before = incompatible_head.snapshot_state()
    with pytest.raises(ValueError, match="checkpoint|incompatible"):
        _run(
            incompatible_head,
            config,
            store=store,
            resume=True,
            time_budget_seconds=6.0,
            clock=TrainingClock().read,
        )
    for name, value in incompatible_head.snapshot_state().items():
        if torch.is_tensor(value):
            assert torch.equal(value, before[name]), name
        else:
            assert value == before[name], name


def test_public_wall_time_checkpoint_keeps_final_test_sealed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(),
        epochs=1000,
        predictive_head_hidden_dim=4,
        predictive_inference_steps=2,
        backprop_freeze_backbone=True,
        train_samples=4,
        guard_samples=2,
        validation_samples=2,
        test_samples=2,
        batch_size=2,
    )
    train, guard, validation = _batches()
    test_opened = [False]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            test_opened[0] = True
            raise AssertionError("final test opened before interrupted wall-time training finished")

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
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "wall-public.ckpt")
    forwarded: list[tuple[Any, bool]] = []

    def interrupting_trainer(*args: Any, **kwargs: Any) -> Any:
        forwarded.append((kwargs["checkpoint_store"], kwargs["resume_from_checkpoint"]))
        raise IntentionalInterruption()

    monkeypatch.setattr(matched, "_train_circadian_head", interrupting_trainer)
    with pytest.raises(IntentionalInterruption):
        matched.run_three_head_fixed_feature_wall_time_benchmark(
            config, wall_time_budget_seconds=0.02, checkpoint_store=store
        )
    assert forwarded == [(store, False)]
    assert test_opened == [False]

    with pytest.raises(IntentionalInterruption):
        matched.run_three_head_fixed_feature_wall_time_benchmark(
            config,
            wall_time_budget_seconds=0.02,
            measure_memory=True,
            checkpoint_store=store,
        )
    assert forwarded == [(store, False), (store, False)]
    assert test_opened == [False]
