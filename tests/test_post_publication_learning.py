"""Fresh post-publication labels stay causal on the original lifetime owner."""

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts.ticket_serving_timeline import FollowupTrainingPlan, LiveServingPlan
from src.app.actor_shadow import StableActor
from src.app.coupled_attribution import AttributionBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.learner_ports import TrainingDiagnostic
from src.core.serving_ports import PromotionReceipt


class LiteralLearner:
    def __init__(self, value=0):
        self.value = value

    def fork(self):
        return LiteralLearner(self.value)

    def snapshot_state(self):
        return self.value

    def restore_state(self, value):
        self.value = value

    def train_batch(self, x, y):
        self.value += 1
        return TrainingDiagnostic("scripted_train", float(self.value))

    def predict(self, x):
        return np.tile([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], (len(x), 1))


class LiteralActor(StableActor):
    def __init__(self, source):
        super().__init__(source, version="old")
        self.generation = 0

    def serving_snapshot(self):
        return SimpleNamespace(actor_version=self.version, generation=self.generation)

    def publish(self, runtime):
        self._learner.restore_state(runtime.candidate_snapshot().state)
        previous = self.version
        self._version = "c0"
        self.generation += 1
        return PromotionReceipt(self.generation, previous, self.version)


class LiteralLive:
    def __init__(self, gate):
        self.gate = gate
        self.plan = LiveServingPlan(10, ())
        self.ticks = []

    @contextmanager
    def opportunity(self, tick):
        self.ticks.append(tick)
        yield

    def snapshot(self):
        return dict(ticks=self.ticks[:], fixture=True)


def batches():
    return tuple(
        AttributionBatch(
            "train",
            f"context-{i}",
            (f"row-{i}",),
            np.zeros((1, 7)),
            np.zeros((1, 1)),
            num_classes=6,
        )
        for i in range(2)
    )


def schedule(monkeypatch, *, publish=True, budget_limit=2, require_input=None, plan=None):
    source = LiteralLearner()
    actor = LiteralActor(source)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=budget_limit), lambda: 0.0)
    gate = demo.ServingPriorityGate(demo.SharingLimits(1, 2, 1), resource_available=lambda: True)
    live = LiteralLive(gate)
    report: dict[str, Any] = {}
    owners = []
    original = demo.ActorShadowRuntime

    def runtime(*args, **kw):
        r = original(*args, **kw)
        owners.append(r)
        return r

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)

    def commit(owner, tick):
        return actor.publish(owner) if publish else None

    def run():
        return demo._reference_schedule(
            source,
            actor,
            batches(),
            budget,
            "periodic",
            report,
            (lambda: require_input(actor, report, owners)) if require_input else (lambda: None),
            lambda: None,
            sharing_gate=gate,
            live=live,
            on_drain=commit,
            followup_plan=plan or FollowupTrainingPlan(1, 4),
        )

    return run, report, source, actor, budget, gate, owners


def test_two_windows_keep_one_runtime_clock_budget_gate_envelope_and_truthful_versions(monkeypatch):
    run, report, source, actor, budget, gate, owners = schedule(monkeypatch)
    runtime, horizon = run()
    assert owners == [runtime] and runtime._budget is budget and horizon == 10
    assert (
        source.snapshot_state() == 0
        and actor.snapshot_state().state == 1
        and runtime.candidate_snapshot().state == 2
    )
    assert budget.updates_completed == gate.snapshot().admitted_updates == 2
    assert [
        (r.actor_version, r.observed_at, r.arrived_at, r.applied_at, r.update_number)
        for r in runtime.applied_updates
    ] == [("old", 0, 1, 2, 1), ("c0", 4, 5, 6, 2)]
    assert report["feedback_invalidations"][0]["feedback"]["actor_version"] == "old"
    assert report["controller_events"][4]["observation"]["feedback"] is None
    fresh = report["controller_events"][5]["observation"]["feedback"]
    assert fresh["actor_version"] == "c0" and fresh["observed_at"] == 4 and fresh["arrived_at"] == 5
    assert report["action_snapshot"]["envelope"]["max_wake_updates"] == 2
    assert report["action_snapshot"]["progress"]["decisions_attempted"] == 10
    assert (
        report["stream"]["completed_updates"] == 2
        and report["followup_training"]["status"] == "complete"
    )


def test_refused_publication_never_opens_second_window_or_predicts_new_data(monkeypatch):
    run, report, source, actor, budget, gate, owners = schedule(monkeypatch, publish=False)
    with pytest.raises(ValueError, match="publication"):
        run()
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1
    assert len(owners[0].applied_updates) == 1 and actor.version == "old"
    assert report["controller_work"]["prediction_rows"] == 1


def test_original_budget_is_not_reset_for_second_window(monkeypatch):
    run, report, source, actor, budget, gate, owners = schedule(monkeypatch, budget_limit=1)
    with pytest.raises(ToyExecutionStopped):
        run()
    assert budget.updates_completed == 1 and len(owners[0].applied_updates) == 1
    assert gate.snapshot().admitted_updates == 2 and actor.version == "c0"


def test_unobserved_later_generation_refuses_before_fresh_label_or_native_use(monkeypatch):
    changed = [False]

    def require(actor, report, owners):
        if owners and owners[0]._inbox._clock.now() == 4 and not changed[0]:
            changed[0] = True
            actor.generation = 2

    run, report, source, actor, budget, gate, owners = schedule(monkeypatch, require_input=require)
    with pytest.raises(ValueError, match="publication"):
        run()
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1
    assert report["controller_work"]["prediction_records_remaining"] == 1
    assert report["followup_training"]["pending_prediction_indices"] == [1]
    assert report["followup_training"]["status"] == "incomplete"
    assert len(report["applied_updates"]) == 1


def test_generation_change_after_selection_refuses_before_second_admission(monkeypatch):
    def require(actor, report, owners):
        if len(report.get("controller_events", ())) == 7:
            actor.generation = 2

    run, report, source, actor, budget, gate, owners = schedule(monkeypatch, require_input=require)
    with pytest.raises(ValueError, match="publication"):
        run()
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1
    assert len(report["applied_updates"]) == 1
    assert report["stream"]["pending_unapplied"] == 1


def test_mutated_plan_with_equal_coerced_values_refuses_before_new_work(monkeypatch):
    plan = FollowupTrainingPlan(1, 4)

    def require(actor, report, owners):
        if owners and owners[0]._inbox._clock.now() == 4:
            object.__setattr__(plan, "start_tick", 4.0)

    run, report, source, actor, budget, gate, owners = schedule(
        monkeypatch, require_input=require, plan=plan
    )
    with pytest.raises(ValueError, match="start"):
        run()
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1


@pytest.mark.parametrize("value", [None, False, 0, 1])
def test_unaccepted_publication_flags_refuse_before_actor_model_copy(value):
    actor = SimpleNamespace(serving_snapshot=lambda: pytest.fail("serving payload read"))
    report = {
        "drain_publication": dict(
            accepted=value, completed=True, publication_unknown=False, generation_before=0
        )
    }
    with pytest.raises(ValueError, match="publication"):
        demo._require_followup_publication(actor, report)


@pytest.mark.parametrize(
    "changes",
    [
        {"first_batches": True},
        {"first_batches": 0},
        {"first_batches": 9},
        {"start_tick": True},
        {"start_tick": 1},
        {"start_tick": 64},
    ],
)
def test_followup_plan_is_exact_finite_and_not_outcome_dependent(changes):
    with pytest.raises(ValueError):
        replace(FollowupTrainingPlan(), **changes)


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard", "outer_selection"])
def test_window_helpers_refuse_nontrain_before_payload_access(role):
    b = SimpleNamespace(role=role)
    for helper in (demo._record_window_source, demo._record_window_label):
        with pytest.raises(ValueError, match="TRAIN"):
            helper(object(), b, 0, 0, "v")


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"reference_mode": "drift"},
        {"publish_on_drain": False},
        {"contextual_policy": object()},
        {"base_snapshot": object()},
        {"counterfactual": True},
        {"deferred_evaluation": True},
        {"stream_recipe": object()},
        {"live_training_recipe": object()},
        {"serving_plan": LiveServingPlan(24)},
        {"followup_plan": FollowupTrainingPlan(1, 4)},
    ],
)
def test_invalid_followup_mode_refuses_before_source_or_model_access(changes, monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source read"))
    kwargs: dict[str, Any] = dict(
        reference_mode="periodic",
        publish_on_drain=True,
        serving_plan=LiveServingPlan(48),
        followup_plan=FollowupTrainingPlan(),
    )
    if not changes:
        kwargs["serving_plan"] = None
    kwargs.update(changes)
    with pytest.raises(ValueError):
        demo.run_case(Path("not-read.json"), **kwargs)


def test_new_cli_is_explicit_and_exclusive_without_changing_old_mode(tmp_path, monkeypatch):
    calls = []

    def run_case(source, **kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(demo, "run_case", run_case)
    output = tmp_path / "new.json"
    assert (
        demo.main(
            [
                "--source",
                "saved.json",
                "--output",
                str(output),
                "--continue-training-after-publication",
            ]
        )
        == 0
    )
    assert (
        calls[0]["reference_mode"] == "periodic"
        and calls[0]["followup_plan"] == FollowupTrainingPlan()
    )
    assert calls[0]["serving_plan"].opportunities == 48
    with pytest.raises(FileExistsError):
        demo.main(
            [
                "--source",
                "saved.json",
                "--output",
                str(output),
                "--continue-training-after-publication",
            ]
        )


def test_native_fixed_post_publication_case():
    output = os.environ.get("POST_PUBLICATION_NATIVE_REPORT")
    if not output:
        pytest.skip("separate frozen native case after prerequisite gates")
    result = demo.run_case(
        Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
        reference_mode="periodic",
        serving_plan=LiveServingPlan(48),
        publish_on_drain=True,
        followup_plan=FollowupTrainingPlan(),
    )
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    assert result["complete"], result.get("error")
    assert (
        result["budget"]["updates_completed"] == result["training_work"]["returned_updates"] == 16
    )
    assert result["training_work"]["attempted_rows"] == 96
    assert (
        result["drain_publication"]["accepted"] and result["drain_publication"]["wake_updates"] == 8
    )
    assert result["followup_training"]["status"] == "complete"
    receipts = result["applied_updates"]
    assert len(receipts) == 16 and [r["update_number"] for r in receipts] == list(range(1, 17))
    assert all(r["actor_version"] == "qualified-96" for r in receipts[:8])
    assert all(r["actor_version"] == "c0" and r["observed_at"] >= 24 for r in receipts[8:])
    assert result["candidate_checkpoint"]["state"]["completed_updates"] == 112
    published = next(p for p in result["publications"] if p["phase"] == "stream-promoted")
    assert published["model"]["completed_updates"] == 104
    assert result["publications"][-1]["phase"] == "rollback"
    assert result["publications"][-1]["model"]["completed_updates"] == 96
    assert (
        result["source_unchanged"]
        and result["numpy_rng_unchanged"]
        and result["torch_rng_unchanged"]
    )
