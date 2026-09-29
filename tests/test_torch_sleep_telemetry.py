"""Torch head sleep reports core facts even when post-split pruning removes a child."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)


def _head(*, minimum: int = 3, device: Any = None, **changes: Any) -> CircadianPredictiveCodingHead:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_chemical_reset=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
        split_noise_scale=0.0,
        use_dual_chemical=True,
    )
    options.update(changes)
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu") if device is None else device,
        seed=1361,
        config=CircadianHeadConfig(**options),
        min_hidden_dim=minimum,
        max_hidden_dim=7,
    )


def _same_state(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if torch.is_tensor(value):
            assert torch.equal(value, expected[name]), name
        else:
            assert value == expected[name], name


def test_forced_torch_split_records_stable_ids_chemistry_and_zero_replay() -> None:
    head = _head(sleep_enable_split=True, max_split_per_sleep=1)
    head._chemical = torch.tensor([0.95, 0.5, 0.5, 0.5])
    head._chemical_fast = torch.tensor([0.8, 0.4, 0.3, 0.2])
    head._chemical_slow = torch.tensor([0.7, 0.3, 0.2, 0.1])

    event = head.sleep_event(force_sleep=True)

    telemetry = event.telemetry
    assert telemetry is not None
    assert telemetry.trigger_reason == "forced"
    assert telemetry.outcome == "applied"
    assert telemetry.reason == "core_executed"
    assert telemetry.budgets.split_limit == 1
    assert telemetry.budgets.prune_limit == 0
    assert telemetry.budgets.replay_update_limit == 0
    assert telemetry.changes.proposed_split_pairs == ((0, 4),)
    assert telemetry.changes.applied_split_pairs == ((0, 4),)
    assert telemetry.before_width == 4
    assert telemetry.proposed_width == telemetry.final_width == 5
    assert telemetry.chemistry_before.primary.maximum == pytest.approx(0.95)
    assert telemetry.chemistry_before.fast.maximum == pytest.approx(0.8)
    assert telemetry.chemistry_before.slow.maximum == pytest.approx(0.7)
    assert telemetry.chemistry_final.primary.count == 5
    assert telemetry.chemistry_final.primary.minimum == pytest.approx(
        float(head._chemical.min().item())
    )
    assert telemetry.chemistry_final.primary.mean == pytest.approx(
        float(head._chemical.mean().item())
    )
    assert telemetry.chemistry_final.fast.mean == pytest.approx(
        float(head._chemical_fast.mean().item())
    )
    assert telemetry.chemistry_final.slow.maximum == pytest.approx(
        float(head._chemical_slow.max().item())
    )
    assert telemetry.replay.proposed_examples == telemetry.replay.proposed_updates == 0
    assert telemetry.replay.applied_examples == telemetry.replay.applied_updates == 0
    assert telemetry.guard is None
    assert telemetry.durations.core_seconds >= 0.0
    assert telemetry.durations.attempt_seconds == telemetry.durations.core_seconds
    assert event == replace(event, telemetry=None)
    json.dumps(asdict(telemetry), allow_nan=False)


@pytest.mark.parametrize("removed_index,removed_id", [(0, 0), (4, 4)])
def test_post_split_prune_keeps_parent_child_proposal_when_either_is_removed(
    monkeypatch: pytest.MonkeyPatch, removed_index: int, removed_id: int
) -> None:
    head = _head(
        sleep_enable_split=True,
        sleep_enable_prune=True,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        prune_threshold=0.6,
    )
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])
    monkeypatch.setattr(head, "_select_prune_indices", lambda **_: (removed_index,))

    event = head.sleep_event(force_sleep=True)

    assert event.telemetry is not None
    changes = event.telemetry.changes
    assert changes.proposed_split_pairs == ((0, 4),)
    assert changes.applied_split_pairs == ((0, 4),)
    assert changes.proposed_prune_ids == (removed_id,)
    assert changes.proposed_scheduled_prune_ids == ()
    assert changes.proposed_removed_prune_ids == (removed_id,)
    assert changes.applied_removed_prune_ids == (removed_id,)
    assert event.telemetry.before_width == event.telemetry.final_width == 4
    assert event.telemetry.chemistry_final.primary.count == 4


def test_torch_prune_only_and_executed_noop_report_actual_changes() -> None:
    pruned = _head(sleep_enable_prune=True, max_prune_per_sleep=1)
    pruned._chemical = torch.tensor([0.5, 0.5, 0.5, 0.0])
    prune_event = pruned.sleep_event(force_sleep=True)
    assert prune_event.telemetry is not None
    assert prune_event.telemetry.changes.proposed_prune_ids == (3,)
    assert prune_event.telemetry.changes.applied_removed_prune_ids == (3,)
    assert prune_event.telemetry.final_width == 3

    noop = _head()
    noop_event = noop.sleep_event(force_sleep=True)
    assert noop_event.performed
    assert noop_event.telemetry is not None
    assert noop_event.telemetry.outcome == "applied"
    assert noop_event.telemetry.changes.proposed_split_pairs == ()
    assert noop_event.telemetry.replay.applied_updates == 0
    assert noop_event.telemetry.chemistry_before == noop_event.telemetry.chemistry_final

    protected = _head(minimum=4, sleep_enable_prune=True, max_prune_per_sleep=1)
    protected._chemical = torch.zeros(4)
    protected_event = protected.sleep_event(force_sleep=True)
    assert protected_event.telemetry is not None
    assert protected_event.telemetry.budgets.prune_limit == 1
    assert protected_event.telemetry.changes.proposed_prune_ids == ()
    assert protected_event.telemetry.final_width == 4


@pytest.mark.parametrize(
    "config_changes,force_sleep,current_step,total_steps,trigger,reason",
    [
        ({"sleep_mode": "disabled"}, True, None, None, "disabled", "sleep_disabled"),
        ({"use_adaptive_sleep_trigger": True}, False, None, None, "not_due", "adaptive_not_due"),
        (
            {
                "sleep_mode": "legacy",
                "sleep_enable_chemical_reset": True,
                "sleep_enable_homeostasis": True,
                "sleep_enable_split": True,
                "sleep_enable_prune": True,
            },
            True,
            None,
            None,
            "budget_skipped",
            "zero_structural_budget",
        ),
        ({"sleep_warmup_steps": 2}, True, 1, 4, "budget_skipped", "warmup"),
    ],
)
def test_skipped_torch_routes_record_reason_and_keep_state(
    config_changes: dict[str, Any],
    force_sleep: bool,
    current_step: int | None,
    total_steps: int | None,
    trigger: str,
    reason: str,
) -> None:
    head = _head(**config_changes)
    before = head.snapshot_state()

    event = head.sleep_event(
        force_sleep=force_sleep, current_step=current_step, total_steps=total_steps
    )

    assert not event.performed
    assert event.telemetry is not None
    assert event.telemetry.trigger_reason == trigger
    assert event.telemetry.reason == reason
    assert event.telemetry.outcome == "skipped"
    assert event.telemetry.before_width == event.telemetry.final_width == 4
    assert event.telemetry.chemistry_before == event.telemetry.chemistry_final
    assert event.telemetry.guard is None
    _same_state(head.snapshot_state(), before)


def test_torch_telemetry_failure_is_atomic_and_preserves_next_seeded_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _head(sleep_enable_split=True, max_split_per_sleep=1, split_noise_scale=0.1)
    head._chemical = torch.tensor([0.95, 0.5, 0.5, 0.5])
    before = head.snapshot_state()
    control = _head(sleep_enable_split=True, max_split_per_sleep=1, split_noise_scale=0.1)
    control.restore_state(before)

    def fail_telemetry(self: CircadianPredictiveCodingHead, **_: Any) -> None:
        raise RuntimeError("injected telemetry failure")

    with monkeypatch.context() as patch:
        patch.setattr(CircadianPredictiveCodingHead, "_executed_sleep_telemetry", fail_telemetry)
        with pytest.raises(RuntimeError, match="injected telemetry failure"):
            head.sleep_event(force_sleep=True)

    _same_state(head.snapshot_state(), before)
    assert head.sleep_event(force_sleep=True) == control.sleep_event(force_sleep=True)
    _same_state(head.snapshot_state(), control.snapshot_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an actual CUDA device")
def test_cuda_head_emits_finite_json_safe_core_telemetry() -> None:
    device = torch.device("cuda:0")
    head = _head(device=device, sleep_enable_split=True, max_split_per_sleep=1)
    head._chemical = torch.tensor([0.95, 0.5, 0.5, 0.5], device=device)

    event = head.sleep_event(force_sleep=True)

    assert event.telemetry is not None
    assert event.telemetry.outcome == "applied"
    assert event.telemetry.changes.proposed_split_pairs == ((0, 4),)
    assert event.telemetry.chemistry_final.primary.count == 5
    assert event.telemetry.durations.core_seconds >= 0.0
    json.dumps(asdict(event.telemetry), allow_nan=False)
