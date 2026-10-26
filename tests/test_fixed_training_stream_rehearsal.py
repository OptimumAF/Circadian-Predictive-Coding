"""Explicit historical TRAIN episodes stay bounded and separate from new wake data."""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionStream
from src.app.stream_rehearsal import (
    CoverageStreamReplayPolicy,
    RecentStreamReplayPolicy,
    StreamRehearsal,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import episode_arrays
from test_coverage_stream_rehearsal import Model, ROW_BYTES, SCHEMA, batches
from test_demonstration_episode import episode


def fixed_bank():
    return tuple(
        episode(dim=2, steps=2, episode_id=f"old-{key}", context=key) for key in ("A", "B")
    )


def policy(**changes):
    values: dict[str, Any] = dict(
        every_updates=2,
        rows_per_event=4,
        retention=ReplayRetentionBudget(4, ROW_BYTES * 2),
    )
    values.update(changes)
    return RecentStreamReplayPolicy(**values)


def setup(*, bank=None, selected=None, extra=None, **faults):
    source = Model()
    source.control.update(faults)
    wake = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=10), lambda: 0.0
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=8, max_wall_seconds=10), lambda: 0.0
    )
    helper = StreamRehearsal(
        source,
        wake,
        policy() if selected is None else selected,
        replay_budget=replay,
        total_updates=4,
        input_dim=2,
        episode_schema=SCHEMA,
        fixed_training_episodes=fixed_bank() if bank is None else bank,
        **({} if extra is None else extra),
    )
    return source, wake, replay, helper


def stream(source, wake, helper, training=None):
    return AttributionStream(
        source,
        batches(("F",) * 4) if training is None else training,
        (),
        checkpoint_updates=(0, 2, 4),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )


def test_should_retain_complete_historical_bank_without_claiming_any_construction_updates():
    source, wake, replay, helper = setup()
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == wake.replay_examples_completed == 0
    assert replay.updates_completed == replay.replay_examples_completed == 0
    assert helper.retention.example_count == 4
    assert helper.retention.retained_bytes == ROW_BYTES * 2
    report = helper.report
    assert report["control"] == "fixed_train_episodes_v1"
    assert report["fixed_bank"] == {
        "episode_ids": ["old-A", "old-B"],
        "episode_count": 2,
        "native_transition_count": 4,
        "encoded_array_bytes": ROW_BYTES * 2,
        "construction_updates": 0,
    }
    report["fixed_bank"]["episode_ids"].clear()
    assert helper.report["fixed_bank"]["episode_ids"] == ["old-A", "old-B"]


def test_should_replay_entire_original_bank_in_order_without_retaining_new_wake_episodes():
    bank = fixed_bank()
    training = batches(("F",) * 4)
    before = pickle.dumps((bank, training))
    source, wake, replay, helper = setup(bank=bank, mutate_inputs=True)
    run = stream(source, wake, helper, training)
    run.run_until(4)
    report = run.finish()["rehearsal"]
    assert pickle.dumps((bank, training)) == before
    assert wake.updates_completed == 4 and wake.replay_examples_completed == 0
    assert replay.updates_completed == 0 and replay.replay_examples_completed == 8
    assert source.state[0] == run._runtime.actor.snapshot_state().state[0] == 0
    assert run._runtime._candidate.snapshot_state()[0] == 8
    assert len(source.control["calls"]) == 8
    for call_index, packet in zip((2, 3, 6, 7), bank * 2, strict=True):
        expected_features, expected_targets = episode_arrays(packet)
        features, targets = source.control["calls"][call_index]
        np.testing.assert_array_equal(features, expected_features)
        np.testing.assert_array_equal(targets, expected_targets)
    for event in report["events"]:
        assert event["selected_episodes"] == [asdict(packet) for packet in bank]
        assert event["charged_exposures"] == 4
        assert event["attempted_updates"] == event["returned_updates"] == 2
        assert event["committed_updates"] == 2
        assert event["retention"]["example_count"] == 4
        assert event["retention"]["retained_bytes"] == ROW_BYTES * 2
    assert len(run._runtime.applied_consolidations) == 2


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_every_reserved_role_before_native_work_or_any_charge(role):
    source = Model()
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=8), lambda: 0.0)
    bank = (fixed_bank()[0], replace(fixed_bank()[1], role=role))
    with pytest.raises(ValueError):
        StreamRehearsal(
            source,
            wake,
            policy(),
            replay_budget=replay,
            total_updates=4,
            input_dim=2,
            episode_schema=SCHEMA,
            fixed_training_episodes=bank,
        )
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "duplicate_id",
        "wrong_width",
        "wrong_steps",
        "episode_extra",
        "transition_extra",
        "transition_cap",
        "byte_cap",
        "event_cap",
        "distillation",
        "incident",
    ],
)
def test_should_refuse_incomplete_nonexact_or_nonfitting_bank_atomically(fault):
    bank: Any = fixed_bank()
    selected = policy()
    extra: dict[str, Any] = {}
    if fault == "list":
        bank = list(bank)
    elif fault == "duplicate_id":
        bank = (bank[0], replace(bank[1], episode_id=bank[0].episode_id))
    elif fault == "wrong_width":
        bank = (bank[0], episode(dim=3, steps=2, episode_id="wrong"))
    elif fault == "wrong_steps":
        bank = (bank[0], episode(dim=2, steps=3, episode_id="wrong"))
    elif fault == "episode_extra":
        object.__setattr__(bank[1], "unobserved", True)
    elif fault == "transition_extra":
        object.__setattr__(bank[1].transitions[0], "unobserved", True)
    elif fault == "transition_cap":
        selected = policy(retention=ReplayRetentionBudget(3, ROW_BYTES * 2))
    elif fault == "byte_cap":
        selected = policy(retention=ReplayRetentionBudget(4, ROW_BYTES * 2 - 1))
    elif fault == "event_cap":
        selected = policy(rows_per_event=3)
    elif fault == "distillation":
        extra["distillation"] = object()
    else:
        extra["incident_channel"] = "fixed-bank"
    with pytest.raises(ValueError):
        setup(bank=bank, selected=selected, extra=extra)


def test_should_preserve_default_recent_report_when_explicit_fixed_bank_is_empty():
    source, wake, replay, helper = setup(bank=())
    assert helper.report["control"] == "recent_fifo_v1"
    assert "fixed_bank" not in helper.report
    assert helper.retention.example_count == 0
    assert (
        source.control["forks"] == wake.updates_completed == replay.replay_examples_completed == 0
    )


@pytest.mark.parametrize("fault", ["missing_schema", "coverage_policy", "replaced_array"])
def test_should_require_original_recent_schema_and_owned_encoded_arrays(fault):
    if fault == "replaced_array":
        source, wake, replay, helper = setup()
        key = next(iter(helper._buffer._rows))
        features, targets = helper._buffer._rows[key]
        helper._buffer._rows[key] = (features.copy(), targets.copy())
        with pytest.raises(ValueError):
            stream(source, wake, helper)
    else:
        source = Model()
        wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), lambda: 0.0)
        replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=8), lambda: 0.0)
        selected = (
            CoverageStreamReplayPolicy(2, 4, ReplayRetentionBudget(4, ROW_BYTES * 2), ("A", "B"))
            if fault == "coverage_policy"
            else policy()
        )
        with pytest.raises(ValueError):
            StreamRehearsal(
                source,
                wake,
                selected,
                replay_budget=replay,
                total_updates=4,
                input_dim=2,
                episode_schema=None if fault == "missing_schema" else SCHEMA,
                fixed_training_episodes=fixed_bank(),
            )
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


def test_should_resume_owned_fixed_bank_without_repeating_events_or_renewing_original_budget():
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper)
    run.run_until(2)
    first = deepcopy(helper.report["events"][0])
    identities = (id(wake), id(replay), wake.started_at, replay.started_at)
    pause = run.pause()
    run.resume(pause)
    assert helper.report["events"][0] == first
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    run.run_until(4)
    report = run.finish()["rehearsal"]
    assert report["events"][0] == first
    assert identities == (id(wake), id(replay), wake.started_at, replay.started_at)
    assert wake.updates_completed == 4 and replay.replay_examples_completed == 8
    assert len(run._runtime.applied_consolidations) == 2
    assert len(source.control["calls"]) == 8


def test_should_charge_failed_second_episode_without_publishing_partial_replay():
    source, wake, replay, helper = setup(fail_call=4)
    run = stream(source, wake, helper)
    with pytest.raises(RuntimeError, match="coverage replay native failure"):
        run.run_until(4)
    event = helper.report["events"][0]
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    assert event["status"] == "failed" and event["charged_exposures"] == 4
    assert event["attempted_updates"] == 2 and event["returned_updates"] == 1
    assert event["committed_updates"] == 0
    assert run._runtime._candidate.snapshot_state()[0] == 2
    assert run._runtime.applied_consolidations == ()


@pytest.mark.parametrize("fault", ["contents", "membership"])
def test_should_refuse_corrupted_or_replaced_bank_before_admitting_any_native_work(fault):
    source, wake, replay, helper = setup()
    key = next(iter(helper._buffer._rows))
    features, targets = helper._buffer._rows[key]
    if fault == "contents":
        features[0, -3] += 0.125
    else:
        helper._buffer._rows["replacement"] = helper._buffer._rows.pop(key)
    with pytest.raises(ValueError):
        stream(source, wake, helper)
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


def test_should_refuse_corrupted_paused_bank_and_keep_preceding_success_charges():
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper)
    run.run_until(2)
    first = deepcopy(helper.report["events"][0])
    pause = run.pause()
    saved = pause._snapshot["rehearsal_state"].buffer
    next(iter(saved._rows.values()))[0][0, -3] += 0.125
    with pytest.raises(ValueError):
        run.resume(pause)
    assert helper.report["events"][0] == first
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    assert len(source.control["calls"]) == 4


def test_should_validate_new_wake_episode_role_even_when_it_is_not_retained():
    source, wake, replay, helper = setup()
    training = list(batches(("F",) * 4))
    training[0] = replace(training[0], episode=replace(training[0].episode, role="probe_test"))
    with pytest.raises(ValueError):
        stream(source, wake, helper, tuple(training))
    assert wake.updates_completed == replay.replay_examples_completed == 0
    assert source.control["calls"] == []
    assert helper.retention.example_count == 4


def test_should_refuse_changed_live_bank_before_more_wake_work_and_preserve_paid_success():
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper)
    run.run_until(2)
    preceding = deepcopy(run._runtime.applied_consolidations)
    next(iter(helper._buffer._rows.values()))[0][0, -3] += 0.125
    with pytest.raises(ValueError, match="fixed training bank content changed"):
        run.run_until(4)
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    assert len(source.control["calls"]) == 4
    assert run._runtime.applied_consolidations == preceding
