"""Fixed TRAIN membership, observed coverage cursor, no genuine models."""

from dataclasses import replace

import pytest

from src.app.coupled_attribution import AttributionStream
from src.app.stream_rehearsal import CoverageStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from test_coverage_stream_rehearsal import Model, SCHEMA, ROW_BYTES, batches
from test_demonstration_episode import episode


def setup():
    source = Model()
    wake = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=10), lambda: 0.0
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=4, max_wall_seconds=10), lambda: 0.0
    )
    bank = tuple(
        episode(dim=2, steps=2, episode_id=f"old-{key}", context=key) for key in ("A", "B", "C")
    )
    helper = StreamRehearsal(
        source,
        wake,
        CoverageStreamReplayPolicy(2, 2, ReplayRetentionBudget(6, 3 * ROW_BYTES), ("A", "B", "C")),
        replay_budget=replay,
        total_updates=4,
        input_dim=2,
        episode_schema=SCHEMA,
        fixed_training_episodes=bank,
        wake_role="probe_train",
    )
    packets = tuple(
        replace(p, role="probe_train", episode=replace(p.episode, role="probe_train"))
        for p in batches(("F",) * 4)
    )
    stream = AttributionStream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 2, 4),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
        training_role="probe_train",
    )
    return source, wake, replay, helper, stream


def test_fixed_coverage_selects_part_of_bank_and_never_stores_probe_wakes():
    source, wake, replay, helper, stream = setup()
    ids = helper._buffer.retained_order_ids
    stream.run_until(4)
    assert wake.updates_completed == 4 and replay.replay_examples_completed == 4
    assert len(source.control["calls"]) == 6
    assert helper._buffer.retained_order_ids == ids
    report = helper.report
    assert report["control"] == "fixed_train_coverage_v1"
    assert report["fixed_bank"]["episode_count"] == 3
    assert report["coverage_keys"] == ["A", "B", "C"]
    assert [r["coverage_selected_keys"] for r in report["events"]] == [["A"], ["B"]]
    assert [r["coverage_cursor_before"] for r in report["events"]] == [None, "A"]
    assert report["coverage_cursor"] == "B"


@pytest.mark.parametrize(
    "fault", ["cursor", "membership", "replace_membership", "schema", "contexts", "array"]
)
def test_fixed_coverage_refuses_unobserved_changes_before_more_paid_work(fault):
    source, wake, replay, helper, stream = setup()
    stream.run_until(2)
    paid = len(source.control["calls"])
    buffer = helper._buffer
    if fault == "cursor":
        buffer._coverage_cursor = "C"
    if fault == "membership":
        buffer._coverage_ids["A"] = buffer._coverage_ids["B"]
    if fault == "replace_membership":
        buffer._coverage_ids = buffer._coverage_ids.copy()
    if fault == "schema":
        buffer._schema = replace(buffer._schema)
    if fault == "contexts":
        buffer._contexts = tuple(list(buffer._contexts))
    if fault == "array":
        next(iter(buffer._rows.values()))[0][0, 0] += 1
    with pytest.raises(ValueError):
        stream.run_until(4)
    assert len(source.control["calls"]) == paid
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 2


def test_fixed_coverage_continuation_keeps_paid_cursor_and_original_budgets():
    source, wake, replay, helper, stream = setup()
    stream.run_until(2)
    births = wake.started_at, replay.started_at
    saved = stream.pause()
    stream.resume(saved)
    stream.run_until(4)
    assert len(source.control["calls"]) == 6
    assert helper.report["coverage_cursor"] == "B"
    assert (wake.started_at, replay.started_at) == births
    assert replay.replay_examples_completed == 4


def test_fixed_coverage_selector_cannot_admit_forged_transition(monkeypatch):
    source, wake, replay, helper, stream = setup()
    select = helper._buffer.select_coverage

    def bad(maximum):
        result = select(maximum)
        helper._buffer._coverage_cursor = "C"
        return result

    monkeypatch.setattr(helper._buffer, "select_coverage", bad)
    with pytest.raises(ValueError, match="coverage"):
        stream.run_until(2)
    # Wake2 paid; no replay update/row admitted after a forged selector result.
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 0
    assert len(source.control["calls"]) == 2
