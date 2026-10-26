"""Endpoint budget and paid-report behavior using scripted, typed native ports."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from src.adapters import minigrid_probe_rollout as live
from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_action_memory_learner import InvariantMemorySnapshot
from src.core.action_memory import ActionMemoryEntry, ActionMemorySnapshot
from test_minigrid_memory_control import (
    FLAGS,
    POLICY,
    NATIVE_POLICY,
    memory_environment,
)
from test_minigrid_learning_pilot import evaluator as evaluator
from test_torch_feature_rehearsal_pilot import forbidden


class Learner:
    def __init__(self, updates=16):
        self.state = InvariantMemorySnapshot(
            POLICY,
            ActionMemorySnapshot(
                NATIVE_POLICY,
                (ActionMemoryEntry((-1.0,) * 296, (9 * updates, 0, 0, 0, 0, 0, 0)),),
                updates,
                9 * updates,
            ),
        )

    def snapshot_state(self):
        return deepcopy(self.state)


def report():
    return dict(
        status="complete",
        work=dict(
            environment_resets_completed=4,
            live_prediction_calls_returned=4,
            environment_steps_returned=4,
            internal_projection_rows_returned=4,
            internal_projection_array_bytes_returned=9472,
        ),
    )


@pytest.fixture
def ports(monkeypatch):
    work: dict[str, Any] = dict(calls=[], callback=None)
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)

    def evaluate(learner, state, **options):
        work["calls"].append((learner, state, options))
        assert options["budget"].budget.max_training_updates == 0
        assert options["budget"].budget.max_replay_examples == 0
        assert options["budget"].budget.max_wall_seconds == 20
        if work["callback"]:
            return work["callback"](learner, state, options)
        options["check_boundary"]()
        return report()

    monkeypatch.setattr(live, "evaluate_memory_probe_checkpoint", evaluate)
    return work


def test_should_read_both_actual_endpoints_with_distinct_zero_update_budgets(ports):
    observer, learner = live.ProbeRollouts(0), Learner()
    observer.observe(learner, "after_probe", lambda: None)
    learner.state = Learner(32).state
    observer.observe(learner, "after_return", lambda: None)
    observer.check()
    assert [r["status"] for r in observer.records] == ["complete", "complete"]
    assert ports["calls"][0][2]["budget"] is not ports["calls"][1][2]["budget"]
    assert all(r["charged_updates"] == r["replay_examples"] == 0 for r in observer.records)


def test_should_report_expiry_separately_from_a_changed_budget_binding(monkeypatch, ports):
    now = [1.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])

    def expire(*args):
        now[0] = 22.0
        return report()

    ports["callback"] = expire
    observer = live.ProbeRollouts(0)
    with pytest.raises(RuntimeError, match="max_wall_seconds"):
        observer.observe(Learner(), "after_probe", lambda: None)
    saved = observer.records[0]
    assert saved["timing_complete"] is False
    assert saved["budget_binding_preserved"] is True
    assert saved["charged_updates"] == saved["replay_examples"] == 0
    assert saved["report"]["work"]["live_prediction_calls_returned"] == 4


@pytest.mark.parametrize("failure", ["partial", "lost", "state", "budget", "clock", "joint"])
def test_should_keep_paid_or_unknown_work_on_a_failed_endpoint(ports, failure):
    observer, learner = live.ProbeRollouts(0), Learner()

    def fail(owner, state, options):
        if failure == "partial":
            partial = report()
            partial.update(status="failed", reason="physics port interrupted")
            raise live.MiniGridEvaluationFailed(partial)
        if failure == "lost":
            raise RuntimeError("admitted evaluator lost its report")
        if failure == "state":
            owner.state = Learner(17).state
        elif failure == "joint":
            entry = replace(state.native_state.entries[0], counts=(0, 144, 0, 0, 0, 0, 0))
            object.__setattr__(state.native_state, "entries", (entry,))
            owner.state = state
        elif failure == "clock":
            options["budget"].last_clock += 0.5
        else:
            options["budget"].budget = replace(options["budget"].budget, max_training_updates=1)
        return report()

    ports["callback"] = fail
    with pytest.raises(RuntimeError):
        observer.observe(learner, "after_probe", lambda: None)
    saved = observer.records[0]
    assert saved["status"] == "failed"
    work = dict(probes=[dict(probe_rollouts=observer.records)])
    totals: dict[str, Any] = dict(
        total_score_forwards_completed=94,
        total_score_rows_completed=846,
        source_environment_steps_returned=90,
        failure_work_may_be_unobserved=False,
    )
    live.add_work(totals, [work])
    if failure == "lost":
        assert totals["learner_live_episodes"] is None
        assert totals["total_prediction_forwards_completed"] is None
        assert (
            totals["probe_rollouts"]["known_returned_work_lower_bounds"][
                "live_prediction_calls_returned"
            ]
            == 0
        )
    else:
        assert saved["report"]["work"]["live_prediction_calls_returned"] == 4
        assert (
            totals["probe_rollouts"]["known_returned_work_lower_bounds"][
                "live_prediction_calls_returned"
            ]
            == 4
        )


def test_should_refuse_replaced_paid_endpoint_before_admitting_more_work(ports):
    observer = live.ProbeRollouts(0)
    observer.observe(Learner(), "after_probe", lambda: None)
    paid = deepcopy(observer.records)
    observer.records[0]["report"] = deepcopy(observer.records[0]["report"])
    with pytest.raises(ValueError, match="replaced"):
        observer.observe(Learner(32), "after_return", lambda: None)
    assert len(ports["calls"]) == 1
    observer.restore_paid()
    assert observer.records == paid


def test_should_admit_next_phase_only_from_captured_status_after_callbacks(ports):
    returned = report()
    returned.update(status="failed", reason="original readout failure")
    ready = [False]

    def evaluate(*args):
        ready[0] = True
        return returned

    def parent():
        if ready[0]:
            returned["status"] = "complete"

    # Arm the mutation only after serializing the delegated result: callback
    # invocations before the return would otherwise change the source fixture.
    ports["callback"] = evaluate
    observer = live.ProbeRollouts(0)
    with pytest.raises(RuntimeError, match="did not complete"):
        observer.observe(Learner(), "after_probe", parent)
    assert returned["status"] == "complete"
    assert observer.records[0]["report"]["status"] == "failed"
    assert observer.records[0]["status"] == "failed"


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        {},
        "lost",
        {"live_prediction_calls_returned": True},
        {"live_prediction_calls_returned": 3},
    ],
)
def test_should_publish_unknown_costs_for_admitted_malformed_work(ports, malformed):
    returned = dict(status="complete", work=malformed)
    ports["callback"] = lambda *args: returned
    observer = live.ProbeRollouts(0)
    with pytest.raises(RuntimeError, match="invalid paid work"):
        observer.observe(Learner(), "after_probe", lambda: None)
    work: dict[str, Any] = dict(
        total_score_forwards_completed=94,
        total_score_rows_completed=846,
        source_environment_steps_returned=90,
        failure_work_may_be_unobserved=False,
    )
    live.add_work(work, [dict(probes=[dict(probe_rollouts=observer.records)])])
    assert observer.records[0]["status"] == "failed"
    assert len(ports["calls"]) == 1
    assert work["learner_live_episodes"] is None
    assert work["total_prediction_forwards_completed"] is None
    assert work["failure_work_may_be_unobserved"] is True
    expected = 3 if malformed == {"live_prediction_calls_returned": 3} else 0
    assert (
        work["probe_rollouts"]["known_returned_work_lower_bounds"]["live_prediction_calls_returned"]
        == expected
    )


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_should_refuse_invalid_rollout_flag_before_allocating_resources(monkeypatch, value):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError, match="probe_rollout"):
            operation(**FLAGS, probe_rollouts=value)


def test_should_route_live_observers_at_all_four_genuine_scripted_endpoints(
    monkeypatch, evaluator, ports
):
    case = memory_environment(monkeypatch, evaluator)
    outcome = pilot.run_pilot(**FLAGS, probe_rollouts=True)
    assert outcome["status"] == "complete", outcome["reason"]
    assert [call[1].completed_updates for call in ports["calls"]] == [16, 32, 144, 160]
    assert [call[2]["phase"] for call in ports["calls"]] == list(live.PHASES) * 2
    assert len(case.work["trains"]) == 64  # Scripted, not native count updates.
    assert outcome["work"]["native_train_calls"] == 192
    assert outcome["work"]["probe_rollouts"]["endpoint_records"] == 4
    assert outcome["work"]["learner_live_episodes"] == 16
    assert outcome["work"]["total_prediction_forwards_completed"] == 110
    assert outcome["work"]["total_environment_steps_returned"] == 106
