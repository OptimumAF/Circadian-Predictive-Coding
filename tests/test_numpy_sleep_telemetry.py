"""NumPy sleep returns complete core facts without changing learning outcomes."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork


def _model(**changes: Any) -> CircadianPredictiveCodingNetwork:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        replay_steps=0,
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
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=1261,
        min_hidden_dim=3,
        max_hidden_dim=6,
        circadian_config=CircadianConfig(**options),
    )


def _train_batch(model: CircadianPredictiveCodingNetwork, rows: int = 2) -> None:
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])[:rows]
    targets = np.array([[1.0], [0.0]])[:rows]
    model.train_epoch(features, targets, 0.03, 2, 0.2)


def test_forced_split_records_budget_ids_chemistry_and_duration() -> None:
    model = _model(sleep_enable_split=True, max_split_per_sleep=1)
    model.set_chemical_state(np.array([0.95, 0.5, 0.5, 0.5]))
    model._hidden_chemical_fast[:] = [0.8, 0.4, 0.3, 0.2]
    model._hidden_chemical_slow[:] = [0.7, 0.3, 0.2, 0.1]

    event = model.sleep_event(force_sleep=True)

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
        float(model._hidden_chemical.min())
    )
    assert telemetry.chemistry_final.primary.mean == pytest.approx(
        float(model._hidden_chemical.mean())
    )
    assert telemetry.chemistry_final.fast.mean == pytest.approx(
        float(model._hidden_chemical_fast.mean())
    )
    assert telemetry.chemistry_final.slow.maximum == pytest.approx(
        float(model._hidden_chemical_slow.max())
    )
    assert telemetry.chemistry_final == telemetry.chemistry_proposed
    assert telemetry.guard is None
    assert telemetry.replay.applied_examples == telemetry.replay.applied_updates == 0
    assert telemetry.durations.core_seconds >= 0.0
    assert telemetry.durations.attempt_seconds == telemetry.durations.core_seconds
    assert event == replace(event, telemetry=None)
    json.dumps(asdict(telemetry), allow_nan=False)


def test_immediate_prune_records_selected_and_removed_stable_id() -> None:
    model = _model(sleep_enable_prune=True, max_prune_per_sleep=1)
    model.set_chemical_state(np.array([0.5, 0.5, 0.5, 0.0]))

    event = model.sleep_event(force_sleep=True)

    assert event.telemetry is not None
    changes = event.telemetry.changes
    assert changes.proposed_prune_ids == (3,)
    assert changes.proposed_scheduled_prune_ids == ()
    assert changes.proposed_removed_prune_ids == (3,)
    assert changes.applied_removed_prune_ids == (3,)
    assert event.telemetry.before_width == 4
    assert event.telemetry.final_width == 3


def test_replay_finalizes_older_pending_prune_and_counts_exact_examples() -> None:
    model = _model(
        sleep_enable_prune=True,
        sleep_enable_replay=True,
        max_prune_per_sleep=1,
        prune_decay_steps=2,
        replay_steps=1,
        replay_memory_size=2,
    )
    _train_batch(model)
    model.set_chemical_state(np.array([0.5, 0.5, 0.5, 0.0]))

    first = model.sleep_event(force_sleep=True)
    assert first.telemetry is not None
    assert first.telemetry.changes.proposed_prune_ids == (3,)
    assert first.telemetry.changes.proposed_scheduled_prune_ids == (3,)
    assert first.telemetry.changes.proposed_removed_prune_ids == ()
    assert first.telemetry.replay.proposed_examples == 2
    assert first.telemetry.replay.proposed_updates == 1
    assert first.telemetry.replay.applied_examples == 2
    assert first.telemetry.replay.applied_updates == 1
    assert first.telemetry.final_width == 4

    model.set_chemical_state(np.array([0.5, 0.5, 0.5, 0.0]))
    second = model.sleep_event(force_sleep=True)

    assert second.telemetry is not None
    assert second.telemetry.changes.proposed_prune_ids == ()
    assert second.telemetry.changes.proposed_removed_prune_ids == (3,)
    assert second.telemetry.changes.applied_removed_prune_ids == (3,)
    assert second.telemetry.replay.proposed_examples == 2
    assert second.telemetry.replay.proposed_updates == 1
    assert second.telemetry.before_width == 4
    assert second.telemetry.final_width == 3


def test_replay_only_and_executed_noop_have_distinct_actual_work() -> None:
    replay = _model(sleep_enable_replay=True, replay_steps=2, replay_memory_size=2)
    _train_batch(replay, rows=2)
    _train_batch(replay, rows=1)
    replay_event = replay.sleep_event(force_sleep=True)

    assert replay_event.telemetry is not None
    assert replay_event.telemetry.outcome == "applied"
    assert replay_event.telemetry.budgets.replay_update_limit == 2
    assert replay_event.telemetry.replay.proposed_examples == 3
    assert replay_event.telemetry.replay.proposed_updates == 2
    assert replay_event.telemetry.replay.applied_examples == 3
    assert replay_event.telemetry.replay.applied_updates == 2
    assert replay_event.telemetry.changes.proposed_split_pairs == ()

    limited = _model(sleep_enable_replay=True, replay_steps=3, replay_memory_size=2)
    _train_batch(limited, rows=1)
    limited_event = limited.sleep_event(force_sleep=True)
    assert limited_event.telemetry is not None
    assert limited_event.telemetry.budgets.replay_update_limit == 1
    assert limited_event.telemetry.replay.applied_examples == 1
    assert limited_event.telemetry.replay.applied_updates == 1

    noop = _model()
    noop_event = noop.sleep_event(force_sleep=True)
    assert noop_event.performed
    assert noop_event.telemetry is not None
    assert noop_event.telemetry.outcome == "applied"
    assert noop_event.telemetry.replay.applied_updates == 0
    assert noop_event.telemetry.chemistry_before == noop_event.telemetry.chemistry_final


@pytest.mark.parametrize(
    "config_changes,force_sleep,current_step,total_steps,trigger,reason",
    [
        ({"sleep_mode": "disabled"}, True, None, None, "disabled", "sleep_disabled"),
        ({"use_adaptive_sleep_trigger": True}, False, None, None, "not_due", "adaptive_not_due"),
        (
            {
                "sleep_mode": "legacy",
                "sleep_enable_chemical_reset": True,
                "sleep_enable_replay": True,
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
def test_skipped_routes_record_explicit_reason_and_unchanged_state(
    config_changes: dict[str, Any],
    force_sleep: bool,
    current_step: int | None,
    total_steps: int | None,
    trigger: str,
    reason: str,
) -> None:
    model = _model(**config_changes)
    before = model.snapshot_state()

    event = model.sleep_event(
        force_sleep=force_sleep, current_step=current_step, total_steps=total_steps
    )

    assert not event.performed
    assert event.telemetry is not None
    assert event.telemetry.trigger_reason == trigger
    assert event.telemetry.reason == reason
    assert event.telemetry.outcome == "skipped"
    assert event.telemetry.before_width == event.telemetry.final_width == 4
    assert event.telemetry.chemistry_before == event.telemetry.chemistry_final
    assert event.telemetry.changes.proposed_split_pairs == ()
    assert event.telemetry.guard is None
    assert pickle.dumps(model.snapshot_state()) == pickle.dumps(before)


def test_telemetry_failure_restores_model_and_next_seeded_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(sleep_enable_split=True, max_split_per_sleep=1, split_noise_scale=0.1)
    model.set_chemical_state(np.array([0.95, 0.5, 0.5, 0.5]))
    before = model.snapshot_state()
    control = _model(sleep_enable_split=True, max_split_per_sleep=1, split_noise_scale=0.1)
    control.restore_state(before)

    def fail_telemetry(self: CircadianPredictiveCodingNetwork, **_: Any) -> None:
        raise RuntimeError("injected telemetry failure")

    with monkeypatch.context() as patch:
        patch.setattr(CircadianPredictiveCodingNetwork, "_executed_sleep_telemetry", fail_telemetry)
        with pytest.raises(RuntimeError, match="injected telemetry failure"):
            model.sleep_event(force_sleep=True)

    assert pickle.dumps(model.snapshot_state()) == pickle.dumps(before)
    assert model.sleep_event(force_sleep=True) == control.sleep_event(force_sleep=True)
    assert pickle.dumps(model.snapshot_state()) == pickle.dumps(control.snapshot_state())
