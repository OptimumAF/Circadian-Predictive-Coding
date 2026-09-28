"""Runner guard rejection and scoring errors preserve future head behavior."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from src.app import matched_head_benchmark, resnet50_benchmark  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
    CircadianPredictiveCodingResNet50Classifier,
)


def _head() -> CircadianPredictiveCodingHead:
    head = CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=241,
        config=CircadianHeadConfig(
            sleep_mode="components",
            split_threshold=0.8,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_noise_scale=0.1,
            sleep_enable_homeostasis=False,
        ),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )
    head._chemical = torch.tensor([0.95, 0.6, 0.3, 0.1])
    return head


def _same_state(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if torch.is_tensor(value):
            assert torch.equal(value, expected[name]), name
        else:
            assert value == expected[name], name


class _FrozenVisionModel:
    def __init__(self, head: CircadianPredictiveCodingHead) -> None:
        self.head = head

    def snapshot_state(self) -> dict[str, Any]:
        return self.head.snapshot_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        self.head.restore_state(state)

    def sleep_event(self, *, force_sleep: bool, epoch_progress: Any) -> Any:
        return self.head.sleep_event(force_sleep=force_sleep, epoch_progress=epoch_progress)


def _run_guard(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    scores: Sequence[tuple[float, float] | BaseException],
) -> tuple[CircadianPredictiveCodingHead, Any]:
    head = _head()
    config = replace(
        ResNet50BenchmarkConfig(),
        epochs=4,
        circadian_enable_sleep_rollback=True,
        circadian_sleep_rollback_metric="cross_entropy",
        circadian_sleep_rollback_tolerance=0.002,
    )
    score_iter = iter(scores)

    def score(*args: Any, **kwargs: Any) -> tuple[float, float]:
        result = next(score_iter)
        if isinstance(result, BaseException):
            head._chemical[0] += 1.0
            raise result
        return result

    if backend == "matched":
        monkeypatch.setattr(matched_head_benchmark, "_evaluate_head", score)
        run = lambda: matched_head_benchmark._guarded_sleep_event(  # noqa: E731
            torch, torch.device("cpu"), head, (), config, 1, True
        )
    else:
        monkeypatch.setattr(resnet50_benchmark, "_compute_pc_metrics", score)
        model = _FrozenVisionModel(head)
        run = lambda: resnet50_benchmark._guarded_circadian_sleep_event(  # noqa: E731
            torch,
            torch.device("cpu"),
            cast(CircadianPredictiveCodingResNet50Classifier, model),
            [],
            config,
            1,
            True,
            2,
        )
    return head, run


@pytest.mark.parametrize("backend", ["matched", "vision"])
def test_rejected_guard_restores_state_and_next_seeded_sleep(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    head, run = _run_guard(backend, monkeypatch, [(0.8, 0.5), (0.7, 1.0)])
    before = head.snapshot_state()
    control = _head()
    control.restore_state(before)
    event, rolled_back = run()
    assert rolled_back
    assert not event.performed
    assert event.split_indices == ()
    _same_state(head.snapshot_state(), before)
    assert head.sleep_event(force_sleep=True) == control.sleep_event(force_sleep=True)
    _same_state(head.snapshot_state(), control.snapshot_state())
    features = torch.tensor([[0.1, 0.2, -0.3], [-0.2, 0.4, 0.3]])
    targets = torch.tensor([0, 1], dtype=torch.long)
    assert head.train_step(features, targets, 0.03, 2, 0.2) == control.train_step(
        features, targets, 0.03, 2, 0.2
    )
    _same_state(head.snapshot_state(), control.snapshot_state())


@pytest.mark.parametrize("backend", ["matched", "vision"])
def test_accepted_guard_keeps_sleep_result(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    head, run = _run_guard(backend, monkeypatch, [(0.8, 0.5), (0.8, 0.5)])
    control = _head()
    event, rolled_back = run()
    assert not rolled_back
    assert event.performed
    assert event.split_indices == (0,)
    assert event == control.sleep_event(force_sleep=True, current_step=1, total_steps=4)
    _same_state(head.snapshot_state(), control.snapshot_state())


@pytest.mark.parametrize("backend", ["matched", "vision"])
@pytest.mark.parametrize("position", ["pre", "post"])
def test_guard_error_restores_state(
    backend: str, position: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores: Sequence[tuple[float, float] | BaseException] = (
        [RuntimeError("guard failed")]
        if position == "pre"
        else [(0.8, 0.5), RuntimeError("guard failed")]
    )
    head, run = _run_guard(backend, monkeypatch, scores)
    before = head.snapshot_state()
    with pytest.raises(RuntimeError, match="guard failed"):
        run()
    _same_state(head.snapshot_state(), before)


@pytest.mark.parametrize("backend", ["matched", "vision"])
@pytest.mark.parametrize("position", ["pre", "post"])
def test_nonfinite_guard_score_rejects_without_mutation(
    backend: str, position: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = (
        [(0.8, float("nan")), (0.8, 0.5)]
        if position == "pre"
        else [(0.8, 0.5), (0.8, float("inf"))]
    )
    head, run = _run_guard(backend, monkeypatch, scores)
    before = head.snapshot_state()
    with pytest.raises(FloatingPointError, match="nonfinite guard"):
        run()
    _same_state(head.snapshot_state(), before)


def test_matched_runner_counts_attempt_and_rollback_outside_head_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = iter([(0.8, 0.5), (0.7, 1.0), (0.8, 0.5), (0.8, 0.5)])
    monkeypatch.setattr(
        matched_head_benchmark,
        "_evaluate_head",
        lambda *args, **kwargs: next(scores),
    )
    features = torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]])
    labels = torch.tensor([1, 0], dtype=torch.long)
    batches = ((features, labels),)
    head = _head()
    config = replace(
        ResNet50BenchmarkConfig(),
        epochs=1,
        circadian_sleep_mode="components",
        circadian_sleep_interval=1,
        circadian_inference_steps=2,
        target_accuracy=None,
    )
    result = matched_head_benchmark._train_circadian_head(
        torch, torch.device("cpu"), head, batches, batches, batches, config
    )
    assert result.sleep_attempts == 1
    assert result.total_rollbacks == 1
    assert result.total_splits == 0
    assert result.total_prunes == 0
    assert head.get_sleep_clocks().sleep_events == 0
    assert head.get_sleep_clocks().wake_batches == 1


def test_vision_runner_counts_rollback_outside_head_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClassifier(_FrozenVisionModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(_head())

        def train_step(self, **kwargs: Any) -> float:
            return 0.5

        def should_trigger_sleep(self) -> bool:
            return False

    monkeypatch.setattr(
        resnet50_benchmark, "CircadianPredictiveCodingResNet50Classifier", FakeClassifier
    )
    scores = iter([(0.8, 0.5), (0.7, 1.0), (0.8, 0.5), (0.8, 0.5)])
    monkeypatch.setattr(
        resnet50_benchmark,
        "_compute_pc_metrics",
        lambda *args, **kwargs: next(scores),
    )
    images = torch.zeros((2, 3, 8, 8))
    labels = torch.tensor([1, 0], dtype=torch.long)
    loaders = resnet50_benchmark._TrainingLoaders(
        train_loader=[(images, labels)],
        guard_loader=[(images, labels)],
        validation_loader=[(images, labels)],
        num_classes=2,
    )
    config = replace(
        ResNet50BenchmarkConfig(),
        epochs=1,
        circadian_sleep_mode="components",
        circadian_sleep_interval=1,
        target_accuracy=None,
    )
    result = resnet50_benchmark._train_circadian(torch, torch.device("cpu"), loaders, config)
    assert result.circadian_total_rollbacks == 1
    assert result.circadian_total_splits == 0
    assert result.circadian_total_prunes == 0
    assert result.model.head.get_sleep_clocks().sleep_events == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rollback_tolerance_rejects_before_training(invalid: float) -> None:
    config = replace(ResNet50BenchmarkConfig(), circadian_sleep_rollback_tolerance=invalid)
    with pytest.raises(ValueError, match="tolerance must be finite"):
        resnet50_benchmark._validate_benchmark_config(config)
