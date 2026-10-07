"""Guarded runners count rejected attempts and suppress identical retries."""

from __future__ import annotations

from dataclasses import replace
import sys
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from src.app import matched_head_benchmark, resnet50_benchmark  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.app.torch_sleep_decisions import describe_guarded_torch_sleep_decision  # noqa: E402
from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)
from src.core.sleep_clocks import SleepEpochProgress  # noqa: E402


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


class _VisionModel:
    def __init__(self, **kwargs: Any) -> None:
        self.head = _head()

    def should_trigger_sleep(self) -> bool:
        return False

    def predict_logits(self, features: Any) -> Any:
        return self.head.predict_logits(features)

    def parameter_count(self) -> int:
        return self.head.parameter_count()

    def trainable_parameter_count(self) -> int:
        return self.head.parameter_count()

    def train_step(self, **kwargs: Any) -> float:
        return self.head.train_step(
            torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]]),
            kwargs["targets"],
            kwargs["learning_rate"],
            kwargs["inference_steps"],
            kwargs["inference_learning_rate"],
        )

    def snapshot_state(self) -> dict[str, Any]:
        return self.head.snapshot_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        self.head.restore_state(state)

    def sleep_event(self, *, force_sleep: bool, epoch_progress: SleepEpochProgress) -> Any:
        return self.head.sleep_event(force_sleep=force_sleep, epoch_progress=epoch_progress)


def _run_schedule(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    cooldown_epochs: int | None,
    reject: bool,
    adaptive_due: bool = False,
    wake_batches_per_epoch: int = 0,
) -> tuple[Any, CircadianPredictiveCodingHead, list[tuple[int, dict[str, Any]]]]:
    config = replace(
        ResNet50BenchmarkConfig(),
        epochs=4,
        circadian_sleep_mode=mode,
        circadian_sleep_interval=0 if adaptive_due else 1,
        circadian_force_sleep=not adaptive_due,
        circadian_use_adaptive_sleep_trigger=adaptive_due,
        circadian_sleep_rollback_cooldown_epochs=cooldown_epochs,
        circadian_inference_steps=2,
        target_accuracy=None,
    )
    attempted_states: list[tuple[int, dict[str, Any]]] = []
    outcome: Any

    def guarded_head_sleep(
        head: CircadianPredictiveCodingHead, epoch: int, force_sleep: bool
    ) -> tuple[Any, bool, Any]:
        before = head.snapshot_state()
        attempted_states.append((epoch, before))
        event = head.sleep_event(
            force_sleep=force_sleep, epoch_progress=SleepEpochProgress(epoch, config.epochs)
        )
        if reject:
            head.restore_state(before)
            return (
                type(event)(
                    old_hidden_dim=head.hidden_dim,
                    new_hidden_dim=head.hidden_dim,
                    split_indices=(),
                    pruned_indices=(),
                ),
                True,
                event,
            )
        return event, False, event

    if backend == "matched":
        head = _head()
        if adaptive_due:
            monkeypatch.setattr(
                CircadianPredictiveCodingHead, "should_trigger_sleep", lambda self: True
            )
        monkeypatch.setattr(
            matched_head_benchmark, "_evaluate_head", lambda *args, **kwargs: (0.8, 0.5)
        )

        def guarded(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
            result, rolled_back, core = guarded_head_sleep(head, args[5], args[6])
            telemetry = describe_guarded_torch_sleep_decision(
                kwargs["decision"],
                core,
                completed_epoch=args[5],
                guard_role_hash=kwargs["guard_role_hash"],
                pre_accuracy=0.8,
                post_accuracy=0.7 if rolled_back else 0.8,
                pre_cross_entropy=0.5,
                post_cross_entropy=1.0 if rolled_back else 0.5,
                metric_name=config.circadian_sleep_rollback_metric,
                tolerance=config.circadian_sleep_rollback_tolerance,
                guard_examples=1,
                accepted=not rolled_back,
                attempt_seconds=1.0,
            )
            return replace(result, telemetry=telemetry), rolled_back

        monkeypatch.setattr(matched_head_benchmark, "_guarded_sleep_event", guarded)
        features = torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]])
        targets = torch.tensor([1, 0], dtype=torch.long)
        train = ((features, targets),) if wake_batches_per_epoch else ()
        guard_batches = ((features[:1], targets[:1]),)
        outcome = matched_head_benchmark._train_circadian_head(
            torch, torch.device("cpu"), head, train, guard_batches, (), config
        )
    else:
        monkeypatch.setattr(
            resnet50_benchmark, "CircadianPredictiveCodingResNet50Classifier", _VisionModel
        )
        if adaptive_due:
            monkeypatch.setattr(_VisionModel, "should_trigger_sleep", lambda self: True)
        monkeypatch.setattr(
            resnet50_benchmark, "_compute_pc_metrics", lambda *args, **kwargs: (0.8, 0.5)
        )

        def guarded(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
            model = args[2]
            result, rolled_back, core = guarded_head_sleep(model.head, args[5], args[6])
            telemetry = describe_guarded_torch_sleep_decision(
                kwargs["decision"],
                core,
                completed_epoch=args[5],
                guard_role_hash=kwargs["guard_role_hash"],
                pre_accuracy=0.8,
                post_accuracy=0.7 if rolled_back else 0.8,
                pre_cross_entropy=0.5,
                post_cross_entropy=1.0 if rolled_back else 0.5,
                metric_name=config.circadian_sleep_rollback_metric,
                tolerance=config.circadian_sleep_rollback_tolerance,
                guard_examples=1,
                accepted=not rolled_back,
                attempt_seconds=0.0,
            )
            return replace(result, telemetry=telemetry), rolled_back

        monkeypatch.setattr(resnet50_benchmark, "_guarded_circadian_sleep_event", guarded)
        images = torch.zeros((2, 3, 8, 8))
        targets = torch.tensor([1, 0], dtype=torch.long)
        train_loader = [(images, targets)] if wake_batches_per_epoch else []
        loaders = resnet50_benchmark._TrainingLoaders(
            train_loader=train_loader,
            guard_loader=[(images[:1], targets[:1])],
            validation_loader=[],
            num_classes=2,
        )
        outcome = resnet50_benchmark._train_circadian(torch, torch.device("cpu"), loaders, config)
        head = outcome.model.head
    return outcome, head, attempted_states


@pytest.mark.parametrize("backend", ["matched", "vision"])
def test_component_rejection_waits_for_new_wake_before_retry(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, head, attempts = _run_schedule(
        backend, monkeypatch, mode="components", cooldown_epochs=None, reject=True
    )
    if backend == "matched":
        assert outcome.sleep_attempts == 1
        assert outcome.total_rollbacks == 1
        assert outcome.sleep_cooldown_suppressions == 3
        assert outcome.sleep_retry_cooldown_epochs == 1
    else:
        assert outcome.circadian_sleep_attempts == 1
        assert outcome.circadian_total_rollbacks == 1
        assert outcome.circadian_sleep_cooldown_suppressions == 3
        assert outcome.circadian_sleep_retry_cooldown_epochs == 1
    assert len(attempts) == 1
    assert [epoch for epoch, _ in attempts] == [1]
    _same_state(head.snapshot_state(), attempts[0][1])
    control = _head()
    assert head.sleep_event(force_sleep=True) == control.sleep_event(force_sleep=True)
    _same_state(head.snapshot_state(), control.snapshot_state())


@pytest.mark.parametrize("backend", ["matched", "vision"])
@pytest.mark.parametrize(("mode", "cooldown"), [("legacy", None), ("components", 0)])
def test_legacy_or_explicit_zero_preserves_every_due_attempt(
    backend: str, mode: str, cooldown: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, head, attempts = _run_schedule(
        backend, monkeypatch, mode=mode, cooldown_epochs=cooldown, reject=True
    )
    if backend == "matched":
        assert outcome.sleep_attempts == 4
        assert outcome.total_rollbacks == 4
        assert outcome.sleep_cooldown_suppressions == 0
    else:
        assert outcome.circadian_sleep_attempts == 4
        assert outcome.circadian_total_rollbacks == 4
        assert outcome.circadian_sleep_cooldown_suppressions == 0
    assert len(attempts) == 4
    assert [epoch for epoch, _ in attempts] == [1, 2, 3, 4]
    for _, state in attempts[1:]:
        _same_state(state, attempts[0][1])
    _same_state(head.snapshot_state(), attempts[0][1])


@pytest.mark.parametrize("backend", ["matched", "vision"])
@pytest.mark.parametrize(("mode", "attempts"), [("components", 4), ("disabled", 0)])
def test_accepted_and_disabled_routes_have_no_cooldown_suppression(
    backend: str, mode: str, attempts: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, _, observed = _run_schedule(
        backend, monkeypatch, mode=mode, cooldown_epochs=None, reject=False
    )
    if backend == "matched":
        assert outcome.sleep_attempts == attempts
        assert outcome.sleep_cooldown_suppressions == 0
    else:
        assert outcome.circadian_sleep_attempts == attempts
        assert outcome.circadian_sleep_cooldown_suppressions == 0
    assert len(observed) == attempts


@pytest.mark.parametrize("backend", ["matched", "vision"])
def test_adaptive_due_attempts_obey_the_same_rejection_gate(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, _, observed = _run_schedule(
        backend,
        monkeypatch,
        mode="components",
        cooldown_epochs=None,
        reject=True,
        adaptive_due=True,
    )
    if backend == "matched":
        assert outcome.sleep_attempts == 1
        assert outcome.sleep_cooldown_suppressions == 3
    else:
        assert outcome.circadian_sleep_attempts == 1
        assert outcome.circadian_sleep_cooldown_suppressions == 3
    assert len(observed) == 1


@pytest.mark.parametrize("backend", ["matched", "vision"])
def test_new_wake_batches_allow_seeded_retry_after_epoch_cooldown(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, first_head, first_attempts = _run_schedule(
        backend,
        monkeypatch,
        mode="components",
        cooldown_epochs=None,
        reject=True,
        wake_batches_per_epoch=1,
    )
    second, second_head, second_attempts = _run_schedule(
        backend,
        monkeypatch,
        mode="components",
        cooldown_epochs=None,
        reject=True,
        wake_batches_per_epoch=1,
    )
    if backend == "matched":
        assert first.sleep_attempts == second.sleep_attempts == 2
        assert first.total_rollbacks == second.total_rollbacks == 2
        assert first.sleep_cooldown_suppressions == second.sleep_cooldown_suppressions == 2
    else:
        assert first.circadian_sleep_attempts == second.circadian_sleep_attempts == 2
        assert first.circadian_total_rollbacks == second.circadian_total_rollbacks == 2
        assert (
            first.circadian_sleep_cooldown_suppressions
            == second.circadian_sleep_cooldown_suppressions
            == 2
        )
    assert [epoch for epoch, _ in first_attempts] == [1, 3]
    assert [epoch for epoch, _ in second_attempts] == [1, 3]
    _same_state(first_head.snapshot_state(), second_head.snapshot_state())


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_invalid_cooldown_config_rejects_before_data_loading(invalid: object) -> None:
    config = replace(
        ResNet50BenchmarkConfig(),
        circadian_sleep_rollback_cooldown_epochs=cast(int | None, invalid),
    )
    with pytest.raises(ValueError, match="cooldown_epochs"):
        resnet50_benchmark._validate_benchmark_config(config)


def test_cli_passes_explicit_retry_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.adapters import resnet_benchmark_cli

    seen: list[ResNet50BenchmarkConfig] = []
    monkeypatch.setattr(
        sys, "argv", ["resnet-benchmark", "--circ-sleep-rollback-cooldown-epochs", "3"]
    )
    monkeypatch.setattr(
        resnet_benchmark_cli, "run_resnet50_benchmark", lambda config: seen.append(config)
    )
    monkeypatch.setattr(resnet_benchmark_cli, "format_resnet50_benchmark_result", lambda _: "ok")
    resnet_benchmark_cli.main()
    assert len(seen) == 1
    assert seen[0].circadian_sleep_rollback_cooldown_epochs == 3


@pytest.mark.parametrize("backend", ["matched", "vision"])
def test_public_report_exposes_resolved_policy_and_suppression_count(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, _, _ = _run_schedule(
        backend, monkeypatch, mode="components", cooldown_epochs=None, reject=True
    )
    if backend == "matched":
        report = matched_head_benchmark._finalize_head(
            torch, torch.device("cpu"), outcome, (), 0, "none", None
        )
        assert report.sleep_attempts == 1
        assert report.sleep_cooldown_suppressions == 3
        assert report.sleep_retry_cooldown_epochs == 1
    else:
        monkeypatch.setattr(
            resnet50_benchmark,
            "_benchmark_inference",
            lambda **kwargs: {
                "latency_mean_ms": 0.0,
                "latency_p95_ms": 0.0,
                "samples_per_second": 0.0,
            },
        )
        config = replace(ResNet50BenchmarkConfig(), circadian_sleep_mode="components")
        vision_report = resnet50_benchmark._finalize_test_report(
            torch, torch.device("cpu"), outcome, (), config
        )
        development = resnet50_benchmark._finalize_validation_report(
            torch, torch.device("cpu"), outcome, (), config
        )
        for result in (vision_report, development):
            assert result.circadian_sleep_attempts == 1
            assert result.circadian_sleep_cooldown_suppressions == 3
            assert result.circadian_sleep_retry_cooldown_epochs == 1
            assert [event.outcome for event in result.sleep_events] == [
                "rolled_back",
                "skipped",
                "skipped",
                "skipped",
            ]
            assert [event.reason for event in result.sleep_events[1:]] == ["rollback_cooldown"] * 3
            assert all(event.guard is None for event in result.sleep_events[1:])
