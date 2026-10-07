"""Fixed-feature Torch runner decisions retain core proposals and guard facts."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.app.sleep_schedule import decide_sleep_attempt  # noqa: E402
from src.app.torch_sleep_decisions import (  # noqa: E402
    describe_guarded_torch_sleep_decision,
    describe_skipped_torch_sleep_decision,
    describe_unguarded_torch_sleep_decision,
)
from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)


def _head(*, sleep_mode: str = "components") -> CircadianPredictiveCodingHead:
    head = CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=1361,
        config=CircadianHeadConfig(
            sleep_mode=sleep_mode,
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
            sleep_enable_split=True,
            sleep_enable_prune=False,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_threshold=0.8,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            split_noise_scale=0.0,
        ),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )
    head._chemical = torch.tensor([0.95, 0.5, 0.5, 0.5])
    return head


def _decision(*, epoch: int = 1, adaptive: bool = False, mode: str = "components") -> Any:
    return decide_sleep_attempt(
        sleep_mode=mode,
        completed_epochs=epoch,
        interval_epochs=1 if not adaptive else 0,
        adaptive_due=adaptive,
        force_periodic=True,
    )


@pytest.mark.parametrize(
    ("accepted", "expected_outcome", "expected_delta"),
    [(True, "accepted", 0.0), (False, "rolled_back", 0.5)],
)
def test_guarded_torch_decision_retains_proposal_and_exact_guard_facts(
    accepted: bool, expected_outcome: str, expected_delta: float
) -> None:
    head = _head()
    decision = _decision()
    result = head.sleep_event(force_sleep=decision.force_sleep, current_step=1, total_steps=3)
    before_adapter = head.snapshot_state()

    event = describe_guarded_torch_sleep_decision(
        decision,
        result,
        completed_epoch=1,
        guard_role_hash="a" * 64,
        pre_accuracy=0.8,
        post_accuracy=0.8 if accepted else 0.7,
        pre_cross_entropy=0.5,
        post_cross_entropy=0.5 if accepted else 1.0,
        metric_name="cross_entropy",
        tolerance=0.1,
        guard_examples=3,
        accepted=accepted,
        attempt_seconds=0.1,
    )

    assert event.trigger_reason == "periodic"
    assert event.outcome == expected_outcome
    assert event.reason == ("guard_accepted" if accepted else "guard_rejected")
    assert event.guard is not None
    assert event.guard.role == "inner_guard"
    assert event.guard.role_hash == "a" * 64
    assert event.guard.metric_name == "cross_entropy"
    assert event.guard.examples_scored == 6
    assert event.guard.delta == pytest.approx(expected_delta)
    assert event.changes.proposed_split_pairs == ((0, 4),)
    assert event.changes.applied_split_pairs == (((0, 4),) if accepted else ())
    assert event.proposed_width == 5
    assert event.final_width == (5 if accepted else 4)
    assert event.budgets.time_limit_seconds is None
    assert event.durations.attempt_seconds >= event.durations.core_seconds
    for name, value in before_adapter.items():
        if torch.is_tensor(value):
            assert torch.equal(value, head.snapshot_state()[name]), name
        else:
            assert value == head.snapshot_state()[name], name
    json.dumps(asdict(event), allow_nan=False)


@pytest.mark.parametrize(
    ("mode", "epoch", "adaptive", "cooldown", "trigger", "reason"),
    [
        ("components", 2, False, False, "not_due", "schedule_not_due"),
        ("disabled", 1, False, False, "disabled", "sleep_disabled"),
        ("components", 1, False, True, "cooldown_suppressed", "rollback_cooldown"),
    ],
)
def test_unattempted_torch_decision_has_no_guard_or_proposed_work(
    mode: str, epoch: int, adaptive: bool, cooldown: bool, trigger: str, reason: str
) -> None:
    head = _head(sleep_mode=mode)
    decision = decide_sleep_attempt(
        sleep_mode=mode,
        completed_epochs=epoch,
        interval_epochs=3 if not cooldown else 1,
        adaptive_due=adaptive,
        force_periodic=True,
    )
    before = head.snapshot_state()

    event = describe_skipped_torch_sleep_decision(
        head, decision, completed_epoch=epoch, cooldown_suppressed=cooldown
    )

    assert event.trigger_reason == trigger
    assert event.outcome == "skipped"
    assert event.reason == reason
    assert event.guard is None
    assert event.changes.proposed_split_pairs == ()
    assert event.durations.attempt_seconds == 0.0
    assert event.chemistry_before == event.chemistry_final
    for name, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, head.snapshot_state()[name]), name
        else:
            assert value == head.snapshot_state()[name], name
    json.dumps(asdict(event), allow_nan=False)


@pytest.mark.parametrize(
    ("periodic_due", "adaptive_due", "trigger"),
    [(True, False, "periodic"), (False, True, "adaptive"), (True, True, "periodic_and_adaptive")],
)
def test_unguarded_torch_decision_records_runner_trigger_without_guard(
    periodic_due: bool, adaptive_due: bool, trigger: str
) -> None:
    head = _head()
    decision = decide_sleep_attempt(
        sleep_mode="components",
        completed_epochs=1,
        interval_epochs=1 if periodic_due else 0,
        adaptive_due=adaptive_due,
        force_periodic=True,
    )
    result = head.sleep_event(force_sleep=True, current_step=1, total_steps=3)

    event = describe_unguarded_torch_sleep_decision(
        decision, result, completed_epoch=1, attempt_seconds=0.1
    )

    assert event.trigger_reason == trigger
    assert event.outcome == "applied"
    assert event.guard is None
    assert event.changes.proposed_split_pairs == ((0, 4),)
    assert event.durations.attempt_seconds >= event.durations.core_seconds
    json.dumps(asdict(event), allow_nan=False)


def test_accuracy_guard_uses_accuracy_drop_even_when_cross_entropy_improves() -> None:
    head = _head()
    decision = _decision()
    result = head.sleep_event(force_sleep=True, current_step=1, total_steps=3)

    event = describe_guarded_torch_sleep_decision(
        decision,
        result,
        completed_epoch=1,
        guard_role_hash="b" * 64,
        pre_accuracy=0.8,
        post_accuracy=0.6,
        pre_cross_entropy=0.5,
        post_cross_entropy=0.4,
        metric_name="accuracy",
        tolerance=0.1,
        guard_examples=2,
        accepted=False,
        attempt_seconds=0.1,
    )

    assert event.outcome == "rolled_back"
    assert event.guard is not None
    assert event.guard.delta == pytest.approx(0.2)
    assert event.guard.examples_scored == 4
    assert event.changes.proposed_split_pairs == ((0, 4),)
