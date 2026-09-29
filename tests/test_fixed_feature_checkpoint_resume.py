"""Trusted local checkpoints resume the real fixed-feature guarded head loop."""

from __future__ import annotations

from dataclasses import fields, replace
from dataclasses import asdict
import json
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.sleep_schedule import decide_sleep_attempt  # noqa: E402
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
        if field.name in {"head", "train_seconds"}:
            continue
        if field.name == "sleep_events":
            assert len(actual.sleep_events) == len(expected.sleep_events)
            for actual_event, expected_event in zip(
                actual.sleep_events, expected.sleep_events, strict=True
            ):
                actual_facts, expected_facts = asdict(actual_event), asdict(expected_event)
                actual_facts.pop("durations")
                expected_facts.pop("durations")
                assert actual_facts == expected_facts
            continue
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


def _run(
    head: CircadianPredictiveCodingHead,
    config: ResNet50BenchmarkConfig,
    *,
    store: Any = None,
    resume: bool = False,
    train: Any = None,
    guard_batches: Any = None,
    time_budget_seconds: float | None = None,
    clock: Any = None,
) -> Any:
    train_batches, guard, validation = _batches()
    return matched._train_circadian_head(
        torch,
        torch.device("cpu"),
        head,
        train_batches if train is None else train,
        guard if guard_batches is None else guard_batches,
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
        lambda _torch, _device, predict, *_args, **_kwargs: (
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


def test_fixed_feature_sleep_history_survives_guarded_after_sleep_resume(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(), epochs=3, evaluation_batches=2, circadian_sleep_rollback_cooldown_epochs=1
    )
    scoring = {"pairs": iter([(0.8, 0.5), (0.8, 0.5), (0.8, 0.5), (0.7, 1.0)])}

    def evaluate(
        _torch: Any, _device: Any, _predict: Any, _batches: Any, limit: Any, **_kwargs: Any
    ) -> Any:
        return next(scoring["pairs"]) if limit == 1 else (0.8, 0.5)

    monkeypatch.setattr(matched, "_evaluate_head", evaluate)
    control = _run(_head(config), config)
    assert [event.outcome for event in control.sleep_events] == [
        "accepted",
        "rolled_back",
        "skipped",
    ]
    assert [event.trigger_reason for event in control.sleep_events] == [
        "periodic",
        "periodic",
        "cooldown_suppressed",
    ]
    assert control.sleep_attempts == 2
    assert control.total_rollbacks == 1
    assert control.sleep_cooldown_suppressions == 1
    assert control.guard_examples_scored == 7
    assert all(event.guard is not None for event in control.sleep_events[:2])
    assert [event.guard.examples_scored for event in control.sleep_events[:2]] == [2, 2]
    assert all(event.guard.role == "inner_guard" for event in control.sleep_events[:2])
    assert all(len(event.guard.role_hash or "") == 64 for event in control.sleep_events[:2])
    json.dumps([asdict(event) for event in control.sleep_events], allow_nan=False)

    scoring["pairs"] = iter([(0.8, 0.5), (0.8, 0.5), (0.8, 0.5), (0.7, 1.0)])
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "head.ckpt")
    with pytest.raises(IntentionalInterruption):
        _run(
            _head(config),
            config,
            store=InterruptingStore(store, stage="after_sleep", epoch=1),
        )
    assert len(store.load().sleep_events) == 1
    resumed = _run(_head(config), config, store=store, resume=True)
    assert [event.outcome for event in resumed.sleep_events] == [
        "accepted",
        "rolled_back",
        "skipped",
    ]
    _same_report(resumed, control)


@pytest.mark.parametrize(
    ("failure_stage", "reason", "scored", "has_proposal"),
    [
        ("pre", "inner_guard_pre_exception", 0, False),
        ("core", "sleep_core_exception", 1, False),
        ("post", "inner_guard_post_exception", 1, True),
        ("pre_nonfinite", "inner_guard_pre_nonfinite", 1, False),
        ("post_nonfinite", "inner_guard_post_nonfinite", 2, True),
    ],
)
@pytest.mark.parametrize("metric_name", ["cross_entropy", "accuracy"])
def test_failed_fixed_feature_attempt_is_saved_and_resumed_in_same_epoch(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    reason: str,
    scored: int,
    has_proposal: bool,
    metric_name: str,
) -> None:
    config = replace(
        _config(), epochs=2, evaluation_batches=2, circadian_sleep_rollback_metric=metric_name
    )
    failure = {"enabled": False, "guard_calls": 0}
    failing_head: CircadianPredictiveCodingHead | None = None

    def evaluate(
        _torch: Any, _device: Any, _predict: Any, _batches: Any, limit: Any, **_kwargs: Any
    ) -> Any:
        if limit == 1:
            failure["guard_calls"] += 1
            should_fail = failure["enabled"] and (
                (failure_stage.startswith("pre") and failure["guard_calls"] == 1)
                or (failure_stage.startswith("post") and failure["guard_calls"] == 2)
            )
            if should_fail:
                failure["enabled"] = False
                assert failing_head is not None
                failing_head._chemical[0] += 0.5
                random.random()
                np.random.random()
                torch.rand(())
                if failure_stage.endswith("nonfinite"):
                    return float("nan"), 0.5
                raise RuntimeError(f"{failure_stage} guard failed")
        return 0.8, 0.5

    monkeypatch.setattr(matched, "_evaluate_head", evaluate)
    random.seed(904)
    np.random.seed(905)
    torch.manual_seed(906)
    control_head = _head(config)
    control = _run(control_head, config)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    class RecordingStore:
        def __init__(self, wrapped: TrustedLocalCircadianCheckpointStore) -> None:
            self.wrapped = wrapped
            self.before: Any = None

        def load(self) -> Any:
            return self.wrapped.load()

        def save(self, checkpoint: Any) -> None:
            if checkpoint.combined.position.stage == "before_sleep" and self.before is None:
                self.before = checkpoint
            self.wrapped.save(checkpoint)

    random.seed(904)
    np.random.seed(905)
    torch.manual_seed(906)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "failed-sleep.ckpt")
    recording = RecordingStore(store)
    failing_head = _head(config)
    failure["enabled"] = True
    failure["guard_calls"] = 0
    original_sleep = CircadianPredictiveCodingHead.sleep_event

    def sleep_or_fail(self: CircadianPredictiveCodingHead, **kwargs: Any) -> Any:
        if failure_stage == "core" and failure["enabled"]:
            failure["enabled"] = False
            self._chemical[0] += 0.5
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError("core failed")
        return original_sleep(self, **kwargs)

    monkeypatch.setattr(CircadianPredictiveCodingHead, "sleep_event", sleep_or_fail)
    with pytest.raises(
        (RuntimeError, FloatingPointError), match="guard failed|core failed|nonfinite"
    ):
        _run(failing_head, config, store=recording)

    saved = store.load()
    assert saved.combined.position.stage == "before_sleep"
    assert len(saved.sleep_events) == 1
    assert saved.progress.sleep_attempts == 1
    error = saved.sleep_events[0]
    assert error.outcome == "error" and error.reason == reason
    assert error.completed_epoch == 1
    assert error.guard is not None
    assert error.guard.role == "inner_guard"
    assert error.guard.metric_name == metric_name
    assert error.guard.examples_scored == scored
    assert error.guard.pre_cross_entropy == (None if failure_stage.startswith("pre") else 0.5)
    assert error.guard.post_cross_entropy is None
    assert bool(error.changes.proposed_split_pairs) == has_proposal
    assert error.changes.applied_split_pairs == ()
    assert error.final_width == error.before_width == 4
    assert error.durations.attempt_seconds >= error.durations.core_seconds
    assert recording.before is not None
    expected_head = _head(config)
    expected_head.restore_state(recording.before.combined.model_state)
    _same_head(failing_head, expected_head)
    assert random.getstate() == recording.before.combined.python_random_state
    np.testing.assert_equal(np.random.get_state(), recording.before.combined.numpy_random_state)
    assert torch.equal(torch.get_rng_state(), recording.before.combined.torch_cpu_random_state)

    resumed_head = _head(config)
    resumed = _run(resumed_head, config, store=store, resume=True)
    _same_head(resumed_head, control_head)
    assert resumed.sleep_attempts == control.sleep_attempts + 1
    assert resumed.guard_examples_scored == control.guard_examples_scored
    assert resumed.total_splits == control.total_splits
    assert [event.outcome for event in resumed.sleep_events] == ["error", "accepted", "accepted"]
    assert [event.completed_epoch for event in resumed.sleep_events] == [1, 1, 2]
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw
    json.dumps([asdict(event) for event in resumed.sleep_events], allow_nan=False)


def test_wall_time_resume_charges_failed_sleep_without_repeating_wake(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TrainingClock:
        def __init__(self) -> None:
            self.time = 0.0

        def read(self) -> float:
            return self.time

        def advance(self, seconds: float) -> None:
            self.time += seconds

    config = replace(_config(), epochs=2, evaluation_batches=2)
    active_clock = TrainingClock()
    fail_once = False
    guard_calls = 0

    def evaluate(*args: Any, **_kwargs: Any) -> tuple[float, float]:
        nonlocal fail_once, guard_calls
        if args[-1] == 1:
            guard_calls += 1
            if fail_once and guard_calls == 2:
                fail_once = False
                active_clock.advance(0.25)
                raise RuntimeError("post guard failed")
        return 0.8, 0.5

    def timed_head(clock: TrainingClock) -> CircadianPredictiveCodingHead:
        head = _head(config)
        original_step = head.train_step

        def step(**kwargs: Any) -> Any:
            result = original_step(**kwargs)
            clock.advance(1.0)
            return result

        monkeypatch.setattr(head, "train_step", step)
        return head

    monkeypatch.setattr(matched, "_evaluate_head", evaluate)
    control_head = timed_head(active_clock)
    control = _run(control_head, config, time_budget_seconds=10.0, clock=active_clock.read)
    assert control.train_seconds == 4.0

    active_clock = TrainingClock()
    fail_once = True
    guard_calls = 0
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "wall-error.ckpt")
    with pytest.raises(RuntimeError, match="post guard failed"):
        _run(
            timed_head(active_clock),
            config,
            store=store,
            time_budget_seconds=10.0,
            clock=active_clock.read,
        )
    saved = store.load()
    assert saved.protocol_id == matched.THREE_HEAD_FIXED_FEATURE_WALL_TIME_PROTOCOL
    assert saved.combined.position.stage == "before_sleep"
    assert saved.progress.elapsed_seconds == 2.25
    assert saved.progress.wake_batches == 2
    assert [event.outcome for event in saved.sleep_events] == ["error"]

    active_clock = TrainingClock()
    resumed_head = timed_head(active_clock)
    resumed = _run(
        resumed_head,
        config,
        store=store,
        resume=True,
        time_budget_seconds=10.0,
        clock=active_clock.read,
    )
    assert resumed.train_seconds == 4.25
    assert resumed.wake_batches == control.wake_batches == 4
    assert resumed.sleep_attempts == control.sleep_attempts + 1
    assert [event.outcome for event in resumed.sleep_events] == ["error", "accepted", "accepted"]
    _same_head(resumed_head, control_head)
    for field in fields(control):
        if field.name not in {"head", "train_seconds", "sleep_attempts", "sleep_events"}:
            assert getattr(resumed, field.name) == getattr(control, field.name), field.name
    for actual, expected in zip(resumed.sleep_events[1:], control.sleep_events, strict=True):
        actual_facts, expected_facts = asdict(actual), asdict(expected)
        actual_facts.pop("durations")
        expected_facts.pop("durations")
        assert actual_facts == expected_facts


@pytest.mark.parametrize(
    "change", ["missing", "epoch", "role", "metric", "reason", "counter", "nested", "resolved"]
)
def test_pending_error_history_tamper_rejects_before_resume(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    config = replace(_config(), epochs=2, evaluation_batches=2)
    fail_once = True

    def evaluate(*args: Any, **_kwargs: Any) -> tuple[float, float]:
        nonlocal fail_once
        if args[-1] == 1 and fail_once:
            fail_once = False
            raise RuntimeError("pre guard failed")
        return 0.8, 0.5

    monkeypatch.setattr(matched, "_evaluate_head", evaluate)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "pending.ckpt")
    with pytest.raises(RuntimeError, match="pre guard failed"):
        _run(_head(config), config, store=store)
    saved = store.load()
    assert saved.combined.position.stage == "before_sleep"
    assert len(saved.sleep_events) == 1
    error = saved.sleep_events[0]
    if change == "missing":
        saved = replace(saved, sleep_events=())
    elif change == "epoch":
        saved = replace(saved, sleep_events=(replace(error, completed_epoch=2),))
    elif change == "role":
        assert error.guard is not None
        saved = replace(
            saved, sleep_events=(replace(error, guard=replace(error.guard, role_hash="0" * 64)),)
        )
    elif change == "metric":
        assert error.guard is not None
        saved = replace(
            saved,
            sleep_events=(replace(error, guard=replace(error.guard, metric_name="accuracy")),),
        )
    elif change == "reason":
        saved = replace(saved, sleep_events=(replace(error, reason="inner_guard_post_exception"),))
    elif change == "counter":
        saved = replace(saved, progress=replace(saved.progress, sleep_attempts=0))
    elif change == "nested":
        assert error.guard is not None
        object.__setattr__(error.guard, "examples_scored", -1)
    else:
        clean = _run(_head(config), config)
        saved = replace(saved, sleep_events=(error, clean.sleep_events[0]))
    store.save(saved)
    head = _head(config)
    before = head.snapshot_state()
    monkeypatch.setattr(
        matched, "_guarded_sleep_event", lambda *args, **kwargs: pytest.fail("sleep resumed")
    )

    with pytest.raises(ValueError, match="checkpoint"):
        _run(head, config, store=store, resume=True)

    for name, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, head.snapshot_state()[name]), name
        else:
            assert value == head.snapshot_state()[name], name


def test_two_failed_attempts_keep_order_before_same_epoch_resolution(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_config(), epochs=1, evaluation_batches=2)
    stage = "none"
    guard_calls = 0

    def evaluate(*args: Any, **_kwargs: Any) -> tuple[float, float]:
        nonlocal stage, guard_calls
        if args[-1] == 1:
            guard_calls += 1
            if stage == "pre":
                stage = "post"
                raise RuntimeError("pre guard failed")
            if stage == "post" and guard_calls == 2:
                stage = "none"
                raise RuntimeError("post guard failed")
        return 0.8, 0.5

    monkeypatch.setattr(matched, "_evaluate_head", evaluate)
    random.seed(936)
    np.random.seed(937)
    torch.manual_seed(938)
    control_head = _head(config)
    control = _run(control_head, config)

    random.seed(936)
    np.random.seed(937)
    torch.manual_seed(938)
    stage = "pre"
    guard_calls = 0
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "two-errors.ckpt")
    with pytest.raises(RuntimeError, match="pre guard failed"):
        _run(_head(config), config, store=store)
    assert [event.reason for event in store.load().sleep_events] == ["inner_guard_pre_exception"]

    guard_calls = 0
    with pytest.raises(RuntimeError, match="post guard failed"):
        _run(_head(config), config, store=store, resume=True)
    assert [event.reason for event in store.load().sleep_events] == [
        "inner_guard_pre_exception",
        "inner_guard_post_exception",
    ]
    assert store.load().progress.sleep_attempts == 2

    guard_calls = 0
    resumed_head = _head(config)
    resumed = _run(resumed_head, config, store=store, resume=True)
    _same_head(resumed_head, control_head)
    assert resumed.sleep_attempts == 3
    assert resumed.guard_examples_scored == control.guard_examples_scored
    assert [event.outcome for event in resumed.sleep_events] == ["error", "error", "accepted"]
    assert [event.completed_epoch for event in resumed.sleep_events] == [1, 1, 1]
    json.dumps([asdict(event) for event in resumed.sleep_events], allow_nan=False)


def test_unguarded_core_error_is_restored_and_kept_on_explicit_resume(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_config(), epochs=1, circadian_enable_sleep_rollback=False)
    monkeypatch.setattr(matched, "_evaluate_head", lambda *args, **kwargs: (0.8, 0.5))
    random.seed(961)
    np.random.seed(962)
    torch.manual_seed(963)
    control_head = _head(config)
    control = _run(control_head, config)

    random.seed(961)
    np.random.seed(962)
    torch.manual_seed(963)
    original_sleep = CircadianPredictiveCodingHead.sleep_event
    fail_once = True

    def sleep_or_fail(self: CircadianPredictiveCodingHead, **kwargs: Any) -> Any:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            self._chemical[0] += 0.5
            random.random()
            np.random.random()
            torch.rand(())
            raise RuntimeError("unguarded core failed")
        return original_sleep(self, **kwargs)

    monkeypatch.setattr(CircadianPredictiveCodingHead, "sleep_event", sleep_or_fail)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "unguarded-error.ckpt")
    with pytest.raises(RuntimeError, match="unguarded core failed"):
        _run(_head(config), config, store=store)
    saved = store.load()
    assert saved.combined.position.stage == "before_sleep"
    assert saved.sleep_events[0].outcome == "error"
    assert saved.sleep_events[0].reason == "sleep_core_exception"
    assert saved.sleep_events[0].guard is None

    resumed_head = _head(config)
    resumed = _run(resumed_head, config, store=store, resume=True)
    _same_head(resumed_head, control_head)
    assert resumed.sleep_attempts == control.sleep_attempts + 1
    assert resumed.guard_examples_scored == control.guard_examples_scored
    assert [event.outcome for event in resumed.sleep_events] == ["error", "applied"]
    assert all(event.guard is None for event in resumed.sleep_events)


@pytest.mark.parametrize(
    ("failure_stage", "expected_scored", "expected_reason"),
    [
        ("pre_exception", 1, "inner_guard_pre_exception"),
        ("post_exception", 3, "inner_guard_post_exception"),
        ("pre_nonfinite", 2, "inner_guard_pre_nonfinite"),
        ("post_nonfinite", 4, "inner_guard_post_nonfinite"),
    ],
)
def test_guard_error_counts_each_completed_batch_in_partial_pass(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_scored: int,
    expected_reason: str,
) -> None:
    config = replace(_config(), epochs=1, circadian_sleep_rollback_eval_batches=2)
    head = _head(config)
    guard = (
        (torch.tensor([[0.2, 0.1, -0.1]]), torch.tensor([1])),
        (torch.tensor([[0.1, -0.2, 0.3]]), torch.tensor([0])),
    )
    original_predict = head.predict_logits
    calls = 0

    def predict(features: Any) -> Any:
        nonlocal calls
        calls += 1
        fail_on = 2 if failure_stage.startswith("pre") else 4
        if calls == fail_on:
            if failure_stage.endswith("exception"):
                raise RuntimeError("guard batch failed")
            return torch.full((len(features), 2), float("nan"))
        return original_predict(features)

    monkeypatch.setattr(head, "predict_logits", predict)
    decision = decide_sleep_attempt(
        sleep_mode="components",
        completed_epochs=1,
        interval_epochs=1,
        adaptive_due=False,
        force_periodic=True,
    )
    before = head.snapshot_state()
    errors: list[Any] = []

    with pytest.raises((RuntimeError, FloatingPointError), match="guard batch failed|nonfinite"):
        matched._guarded_sleep_event(
            torch,
            torch.device("cpu"),
            head,
            guard,
            config,
            1,
            True,
            decision=decision,
            guard_role_hash=matched._hash_batches(guard),
            on_error=errors.append,
        )

    assert len(errors) == 1
    assert errors[0].reason == expected_reason
    assert errors[0].guard is not None
    assert errors[0].guard.examples_scored == expected_scored
    if failure_stage.startswith("pre"):
        assert errors[0].guard.pre_accuracy is None
    else:
        assert errors[0].guard.pre_accuracy is not None
    for name, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, head.snapshot_state()[name]), name
        else:
            assert value == head.snapshot_state()[name], name


def test_partial_guard_batch_failure_persists_and_rejects_impossible_exposure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_config(), epochs=1, circadian_sleep_rollback_eval_batches=2)
    guard_batches = (
        (torch.tensor([[0.2, 0.1, -0.1]]), torch.tensor([1])),
        (
            torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]]),
            torch.tensor([0, 1]),
        ),
    )
    random.seed(977)
    np.random.seed(978)
    torch.manual_seed(979)
    control_head = _head(config)
    control = _run(control_head, config, guard_batches=guard_batches)

    random.seed(977)
    np.random.seed(978)
    torch.manual_seed(979)
    failed_head = _head(config)
    original_predict = failed_head.predict_logits
    fail_once = True

    def predict(features: Any) -> Any:
        nonlocal fail_once
        if fail_once and torch.equal(features.cpu(), guard_batches[1][0]):
            fail_once = False
            raise RuntimeError("second guard batch failed")
        return original_predict(features)

    monkeypatch.setattr(failed_head, "predict_logits", predict)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "partial-guard.ckpt")
    with pytest.raises(RuntimeError, match="second guard batch failed"):
        _run(failed_head, config, store=store, guard_batches=guard_batches)
    saved = store.load()
    assert saved.combined.position.stage == "before_sleep"
    assert len(saved.sleep_events) == 1
    error = saved.sleep_events[0]
    assert error.reason == "inner_guard_pre_exception"
    assert error.guard is not None
    assert error.guard.examples_scored == 1
    assert error.guard.pre_accuracy is None

    impossible = replace(error, guard=replace(error.guard, examples_scored=2))
    store.save(replace(saved, sleep_events=(impossible,)))
    with pytest.raises(ValueError, match="sleep error exposure"):
        _run(_head(config), config, store=store, resume=True, guard_batches=guard_batches)

    store.save(saved)
    resumed_head = _head(config)
    resumed = _run(resumed_head, config, store=store, resume=True, guard_batches=guard_batches)
    _same_head(resumed_head, control_head)
    assert resumed.sleep_attempts == control.sleep_attempts + 1
    assert [event.outcome for event in resumed.sleep_events] == ["error", "accepted"]
    assert resumed.sleep_events[0].guard.examples_scored == 1
    assert resumed.guard_examples_scored == control.guard_examples_scored


@pytest.mark.parametrize(
    ("changes", "outcomes", "reasons", "attempts"),
    [
        ({"circadian_sleep_mode": "disabled"}, ["skipped", "skipped"], ["sleep_disabled"] * 2, 0),
        ({"circadian_sleep_interval": 3}, ["skipped", "skipped"], ["schedule_not_due"] * 2, 0),
        ({"circadian_sleep_warmup_steps": 99}, ["skipped", "skipped"], ["warmup"] * 2, 2),
    ],
)
def test_fixed_feature_runner_records_unattempted_and_core_skips(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    outcomes: list[str],
    reasons: list[str],
    attempts: int,
) -> None:
    config = replace(_config(), epochs=2, **changes)
    monkeypatch.setattr(matched, "_evaluate_head", lambda *args, **kwargs: (0.8, 0.5))

    report = _run(_head(config), config)

    assert [event.outcome for event in report.sleep_events] == outcomes
    assert [event.reason for event in report.sleep_events] == reasons
    assert report.sleep_attempts == attempts
    assert all(event.guard is None for event in report.sleep_events) == (attempts == 0)
    assert all(event.changes.proposed_split_pairs == () for event in report.sleep_events)
    json.dumps([asdict(event) for event in report.sleep_events], allow_nan=False)


@pytest.mark.parametrize("change", ["format", "missing", "role", "clock", "attempts"])
def test_sleep_history_tamper_rejects_before_fixed_feature_update(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    config = _config()
    monkeypatch.setattr(matched, "_evaluate_head", lambda *args, **kwargs: (0.8, 0.5))
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "history.ckpt")
    with pytest.raises(IntentionalInterruption):
        _run(
            _head(config),
            config,
            store=InterruptingStore(store, stage="after_sleep", epoch=1),
        )
    saved = store.load()
    assert len(saved.sleep_events) == 1
    event = saved.sleep_events[0]
    if change == "format":
        saved = replace(saved, format_version=1)
    elif change == "missing":
        saved = replace(saved, sleep_events=())
    elif change == "role":
        assert event.guard is not None
        saved = replace(
            saved,
            sleep_events=(replace(event, guard=replace(event.guard, role_hash="0" * 64)),),
        )
    elif change == "clock":
        saved = replace(saved, sleep_events=(replace(event, wake_batches=999),))
    else:
        saved = replace(saved, progress=replace(saved.progress, sleep_attempts=0))
    store.save(saved)
    head = _head(config)
    before = head.snapshot_state()
    monkeypatch.setattr(head, "train_step", lambda **kwargs: pytest.fail("resumed update ran"))

    with pytest.raises(ValueError, match="checkpoint"):
        _run(head, config, store=store, resume=True)

    for name, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, head.snapshot_state()[name]), name
        else:
            assert value == head.snapshot_state()[name], name


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
    monkeypatch.setattr(matched, "_evaluate_head", lambda *args, **kwargs: (0.8, 0.5))
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
    monkeypatch.setattr(matched, "_evaluate_head", lambda *args, **kwargs: (0.8, 0.5))
    original_train = matched._train_circadian_head

    def tracked_train(*args: Any, **kwargs: Any) -> Any:
        outcome = original_train(*args, **kwargs)
        finished[0] = True
        return outcome

    monkeypatch.setattr(matched, "_train_circadian_head", tracked_train)
    control = matched.run_three_head_fixed_feature_benchmark(config)
    assert test_accesses == [1]
    assert len(control.circadian.sleep_events) == 2
    assert control.backprop.sleep_events == control.predictive_coding.sleep_events == ()
    encoded = json.loads(json.dumps(asdict(control), allow_nan=False))
    assert len(encoded["circadian"]["sleep_events"]) == 2
    assert encoded["backprop"]["sleep_events"] == []

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
    assert [event.outcome for event in resumed.circadian.sleep_events] == [
        event.outcome for event in control.circadian.sleep_events
    ]


def test_checkpoint_mode_rejects_head_device_mismatch_before_training(
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
    with pytest.raises(ValueError, match="head/device mismatch"):
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
        lambda _torch, _device, predict, *_args, **_kwargs: (
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
