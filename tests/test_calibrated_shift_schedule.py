"""Existing runtime/ACK/detector composition on literal scripted probabilities."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from time import monotonic
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import candidate_training_feedback as candidate_feedback
from scripts import run_bp_ticket_serving as cli
from src.app.actor_shadow import StableActor
from src.app.coupled_attribution import AttributionBatch
from src.app.resource_sharing import ServingPriorityGate
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration
from src.core.shift_calibration import ShiftCalibration


@dataclass(frozen=True)
class State:
    completed_updates: int = 12
    learned: tuple[int, ...] = (0,)


def literal_prediction(state, features):
    result = np.zeros((len(features), 6))
    for i, row in enumerate(features):
        result[i, int(row[0] > 0 and int(4 * row[0]) in state.learned)] = 1.0
    return result


class ScriptedLearner:
    def __init__(self):
        self.state = State()

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        child = ScriptedLearner()
        child.restore_state(self.state)
        return child

    def predict(self, features):
        return literal_prediction(self.state, features)

    def train_batch(self, features, targets):
        self.state = replace(
            self.state,
            completed_updates=self.state.completed_updates + 1,
            learned=tuple(sorted(set(self.state.learned) | {int(4 * v) for v in features[:, 0]})),
        )
        return TrainingDiagnostic("scripted_test_only", 0.0)


def packets(values=(0, 0, 0, 0, 0, 0, 0.5, 1)):
    return tuple(
        AttributionBatch(
            "train",
            "stationary" if v == 0 else "changed",
            (f"row-{i}",),
            np.array([[float(v)]]),
            np.array([[float(v > 0)]]),
            num_classes=6,
        )
        for i, v in enumerate(values)
    )


def run(
    *,
    values=(0, 0, 0, 0, 0, 0, 0.5, 1),
    delay=1,
    opportunities=12,
    available=lambda: True,
    on_tick=lambda: None,
    capture=None,
    promotable=False,
    **options,
):
    source = ScriptedLearner()
    actor = (
        PromotableActor(
            source,
            version="actor",
            configuration=ServingConfiguration(1, 1),
            feature_digest=lambda features: "literal",
            metadata={},
        )
        if promotable
        else StableActor(source, version="actor")
    )
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=8, max_wall_seconds=30), monotonic
    )
    gate = ServingPriorityGate(SharingLimits(1, 8, 1), resource_available=available)
    timeline = SimpleNamespace(
        gate=gate,
        plan=SimpleNamespace(opportunities=opportunities),
        opportunity=lambda tick: nullcontext(),
    )
    report: dict[str, Any] = {}
    if capture is not None:
        capture.update(source=source, actor=actor, budget=budget, report=report, gate=gate)
    args = dict(
        label_delay=delay,
        sharing_gate=gate,
        live=timeline,
        arrived_shift_predict=literal_prediction,
        candidate_predict=literal_prediction,
        acknowledge_correct=True,
        shift_calibration=ShiftCalibration(),
        shift_model_id="scripted-initial",
        shift_prefix_rows=tuple((f"row-{i}",) for i in range(6)),
    )
    args.update(options)

    def require_input():
        budget.before_final()
        on_tick()

    runtime, horizon = cli._reference_schedule(
        source,
        actor,
        packets(values),
        budget,
        "drift",
        report,
        require_input,
        lambda: None,
        **args,
    )
    return source, actor, budget, runtime, report, horizon


@pytest.mark.parametrize("delay", [1, 2])
def test_should_ack_prefix_then_fit_only_current_errors_within_incident(delay):
    source, actor, budget, runtime, report, horizon = run(delay=delay)
    assert horizon == 12
    assert budget.updates_completed == 2
    assert source.snapshot_state() == actor.snapshot_state().state == State()
    assert runtime.candidate_snapshot().state == State(14, (0, 2, 4))
    assert [r.sample_id for r in runtime.acknowledged_experiences] == [
        f"batch-{i}" for i in range(6)
    ]
    assert report["receipt_progress"]["complete"]
    assert report["stream"]["pending_unapplied"] == 0
    records = report["arrived_shift_feedback"]["records"]
    assert len(records) == 8
    assert all(r["decision"]["alarms"] == (False,) * 3 for r in records[:6])
    assert all(r["prediction_native_updates"] == 12 for r in records)
    assert report["calibrated_shift_schedule"]["episodes"] == [
        dict(opened_at=7 + delay, through_arrived=8)
    ]
    wakes = [e for e in report["controller_events"] if e["action_id"] == "wake1"]
    assert [e["observation"]["tick"] for e in wakes] == [7 + delay, 8 + delay]
    assert [r["index"] for r in report["candidate_feedback"]["records"]] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        7,
    ]


def test_should_leave_wrong_packets_unfitted_without_a_shift_incident():
    # Constant features/labels but wrong current candidate: fixed diagnostic is
    # already correct, so a supervised error alone must not open the episode.
    def wrong_current(state, features):
        result = np.zeros((len(features), 6))
        result[:, 1] = 1.0
        return result

    _, _, budget, _, report, _ = run(values=(0,) * 8, candidate_predict=wrong_current)
    assert budget.updates_completed == 0
    assert not report["receipt_progress"]["complete"]
    assert report["calibrated_shift_schedule"]["episodes"] == []


def test_should_not_extend_original_horizon_to_drain_an_incident():
    _, _, budget, _, report, _ = run(opportunities=9)
    assert budget.updates_completed == 1
    assert report["stream"]["pending_unapplied"] == 1
    assert not report["receipt_progress"]["complete"]


def test_should_ack_after_a_fit_using_current_candidate_instead_of_frozen_reference():
    _, _, budget, runtime, report, _ = run(values=(0, 0, 0, 0, 0, 0, 0.5, 0.5))
    assert budget.updates_completed == 1
    assert len(runtime.acknowledged_experiences) == 7
    last = report["candidate_feedback"]["records"][-1]
    assert last["index"] == 7 and last["native_updates"] == 13 and last["mistakes"] == 0
    assert report["arrived_shift_feedback"]["records"][-1]["prediction_native_updates"] == 12
    assert report["receipt_progress"]["complete"]


def test_should_resume_deferred_incident_with_fresh_current_feedback():
    availability = iter((False, True, True))
    _, _, budget, _, report, _ = run(available=lambda: next(availability))
    assert budget.updates_completed == 2
    assert report["receipt_progress"]["complete"]
    wakes = [e for e in report["controller_events"] if e["action_id"] == "wake1"]
    assert [e["observation"]["tick"] for e in wakes] == [8, 9, 10]
    assert wakes[0]["poll"]["training"]["deferred_reason"] is not None
    assert [r["index"] for r in report["candidate_feedback"]["records"]][-3:] == [6, 6, 7]
    assert len(report["arrived_shift_feedback"]["records"]) == 8


def test_should_not_renew_arrived_boundary_from_continued_alarms():
    _, _, budget, _, report, _ = run(
        values=(0, 0, 0, 0, 0, 0.25, 0.5, 0.75),
        shift_calibration=ShiftCalibration(prefix_batches=5),
        shift_prefix_rows=tuple((f"row-{i}",) for i in range(5)),
    )
    assert budget.updates_completed == 2
    assert report["calibrated_shift_schedule"]["episodes"] == [dict(opened_at=7, through_arrived=7)]
    assert "continued" in report["arrived_shift_feedback"]["records"][-1]["incident"]["states"]
    assert report["stream"]["pending_unapplied"] == 1
    assert not report["receipt_progress"]["complete"]


@pytest.mark.parametrize("fault", ["candidate", "episode", "paid"])
def test_should_refuse_selection_side_changes_before_fit(fault, monkeypatch):
    captured: dict[str, Any] = {}
    original_runtime, original_choose = cli.ActorShadowRuntime, cli.choose_reference

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        captured["runtime"] = result
        return result

    def choose(policy, observation):
        decision = original_choose(policy, observation)
        if observation.tick == 8:
            if fault == "candidate":
                captured["runtime"]._candidate.restore_state(State(12, (0, 9)))
            elif fault == "episode":
                captured["report"]["calibrated_shift_schedule"]["through_arrived"] = 99
            else:
                captured["report"]["candidate_feedback"]["attempts"] = 0
        return decision

    monkeypatch.setattr(cli, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(cli, "choose_reference", choose)
    with pytest.raises(ValueError, match="changed"):
        run(capture=captured)
    assert captured["budget"].updates_completed == 0
    assert captured["report"]["candidate_feedback"]["attempts"] == 8
    assert captured["report"]["calibrated_shift_schedule"]["through_arrived"] == 8


def test_should_preserve_failed_current_diagnostic_costs(monkeypatch):
    captured: dict[str, Any] = {}

    def fail_current(state, features):
        if features[0, 0] > 0:
            raise RuntimeError("lost current diagnostic")
        return literal_prediction(state, features)

    with pytest.raises(RuntimeError, match="lost current"):
        run(candidate_predict=fail_current, capture=captured)
    paid = captured["report"]["candidate_feedback"]
    assert paid["attempts"] == 7 and paid["returns"] == 6
    assert paid["records"][-1]["complete"] is False
    assert "lost current" in paid["records"][-1]["error"]
    assert captured["budget"].updates_completed == 0


def test_should_refuse_unobserved_state_change_during_wait_poll(monkeypatch):
    captured: dict[str, Any] = {}
    original = cli.ResourceSharedRuntime.train_action

    def train_action(shared, action):
        result = original(shared, action)
        shared._runtime._candidate.state = State(12, (0, 9))
        return result

    monkeypatch.setattr(cli.ResourceSharedRuntime, "train_action", train_action)
    with pytest.raises(ValueError, match="without a completed update"):
        run(capture=captured)
    assert captured["budget"].updates_completed == 0


def test_should_require_complete_fit_and_ack_drain_before_one_publication_callback():
    calls = []

    def decline(runtime, tick):
        calls.append(
            (tick, runtime.candidate_snapshot().state, len(runtime.acknowledged_experiences))
        )
        return None

    _, _, budget, _, report, _ = run(promotable=True, on_drain=decline)
    assert calls == [(9, State(14, (0, 2, 4)), 6)]
    assert budget.updates_completed == 2
    assert report["drain_publication"]["completed"]
    assert report["drain_publication"]["accepted"] is False


@pytest.mark.parametrize("fault", ["candidate_during_actor", "replace_paid", "erase_paid"])
def test_should_refuse_changed_evidence_inside_observation(fault, monkeypatch):
    captured: dict[str, Any] = {}
    original_runtime, original_predict = cli.ActorShadowRuntime, StableActor.predict

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        captured["runtime"] = result
        return result

    def actor_predict(actor, features):
        result = original_predict(actor, features)
        if fault == "candidate_during_actor":
            captured["runtime"]._candidate.restore_state(State(12, (0, 9)))
        return result

    def current_prediction(state, features):
        if features[0, 0] > 0:
            if fault == "replace_paid":
                captured["report"]["candidate_feedback"] = {}
            elif fault == "erase_paid":
                captured["report"]["candidate_feedback"]["records"].clear()
        return literal_prediction(state, features)

    monkeypatch.setattr(cli, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(StableActor, "predict", actor_predict)
    with pytest.raises(ValueError, match="changed"):
        run(capture=captured, candidate_predict=current_prediction)
    assert captured["budget"].updates_completed == 0
    if fault != "candidate_during_actor":
        paid = captured["report"]["candidate_feedback"]
        assert paid["attempts"] == paid["returns"] == 7
        assert len(paid["records"]) == 7 and paid["records"][-1]["complete"] is False


def test_should_refuse_pre_ack_input_hook_state_change_before_consuming_receipt(monkeypatch):
    captured: dict[str, Any] = {}
    original_runtime, original_read = (
        cli.ActorShadowRuntime,
        candidate_feedback.read_candidate_feedback,
    )

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        captured["runtime"] = result
        return result

    def read(*args, **kwargs):
        captured["in_read"] = kwargs["arrived"] > 0
        try:
            return original_read(*args, **kwargs)
        finally:
            captured["in_read"] = False

    def input_hook():
        if captured.get("in_read"):
            captured["runtime"]._candidate.restore_state(State(12, (0, 9)))

    monkeypatch.setattr(cli, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(candidate_feedback, "read_candidate_feedback", read)
    with pytest.raises(ValueError, match="pinned ACK candidate changed"):
        run(capture=captured, on_tick=input_hook)
    assert captured["runtime"].acknowledged_experiences == ()
    assert captured["budget"].updates_completed == 0
    assert captured["report"]["candidate_feedback"]["attempts"] == 0


def test_should_preserve_existing_supervised_ack_path_without_calibration():
    _, _, budget, runtime, report, _ = run(
        shift_calibration=None,
        shift_model_id=None,
        shift_prefix_rows=None,
        arrived_shift_predict=None,
    )
    assert budget.updates_completed == 2
    assert len(runtime.acknowledged_experiences) == 6
    assert report["receipt_progress"]["complete"]
    assert "calibrated_shift_schedule" not in report
    assert "arrived_shift_feedback" not in report


@pytest.mark.parametrize(
    "options",
    [
        dict(shift_model_id=None),
        dict(shift_prefix_rows=()),
        dict(shift_calibration=None),
        dict(acknowledge_correct=False),
        dict(candidate_predict=None),
        dict(shift_calibration=True),
        dict(arrived_shift_predict=None),
    ],
)
def test_should_refuse_incomplete_controls_before_runtime(options, monkeypatch):
    monkeypatch.setattr(cli, "ActorShadowRuntime", lambda *a, **kw: pytest.fail("runtime entered"))
    with pytest.raises(ValueError):
        run(**options)
