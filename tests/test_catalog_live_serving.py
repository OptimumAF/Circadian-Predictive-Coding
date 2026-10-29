"""Existing live workers and catalog admissions with literal public/native ports."""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_catalog_control as control
from scripts.ticket_serving_timeline import LiveServingPlan
from scripts.ticket_controller_streams import TicketStreamRecipe
from test_ticket_catalog_control import FakeBP, batches, request
from test_ticket_catalog_control import driver as driver
from test_catalog_ticket_stream import policy
from test_ticket_workflow import decision


@pytest.fixture
def serving(driver, monkeypatch):
    monkeypatch.setattr(demo.context, "BATCHES_PER_PHASE", 8)
    monkeypatch.setattr(demo.context, "_teacher_batches", lambda s: (None, None, batches(8)))
    ports = []

    class PublicPort:
        def __init__(self, actor, spec):
            self.actor = actor
            self.work = dict(attempts=0, returns=0, readouts=0, native_forwards=0)
            self.after = lambda: None
            self.respond = decision
            self.release = Event()
            ports.append(self)

        def serve(self, item):
            self.work["attempts"] += 1
            assert self.release.wait(5), "scripted service was not released after admission"
            outcome = self.respond(item)
            snapshot = self.actor.serving_snapshot()
            self.work["returns"] += 1
            self.after()
            return SimpleNamespace(
                generation=snapshot.generation,
                actor_version=snapshot.actor_version,
                native_updates=snapshot.model_state.completed_updates,
                result=outcome,
            )

    monkeypatch.setattr(demo, "TorchServingPredictor", PublicPort)
    opportunity = demo.TicketServingTimeline.opportunity

    @contextmanager
    def held_service(live, tick):
        # Hold the real serving lease across the complete action poll. This
        # deterministically exercises replay success followed by wake deferral.
        live.predictor.release = Event()
        with opportunity(live, tick):
            try:
                yield
            finally:
                live.predictor.release.set()

    monkeypatch.setattr(demo.TicketServingTimeline, "opportunity", held_service)
    monkeypatch.setattr(demo, "_sealed_cases", lambda: pytest.fail("final source opened"))
    driver.ports = ports
    return driver


@pytest.mark.parametrize("action", ["wait", "wake1", "replay_wake"])
@pytest.mark.parametrize("contention", [(), (2, 6)])
def test_should_serve_on_original_actor_while_catalog_learns_privately(serving, action, contention):
    fitted = policy(action)
    pin = deepcopy(fitted)
    r = demo.run_case(
        serving.path, catalog_policy=fitted, serving_plan=LiveServingPlan(9, contention)
    )
    assert r["complete"], r.get("error")
    assert r["definition"] == "live_catalog_bp_ticket_private_v1"
    wakes = 8 - len(contention) if action == "wake1" else 0
    replay = 8 - len(contention) if action == "replay_wake" else 0
    assert r["training_work"] == dict(
        attempted_updates=wakes + replay,
        attempted_rows=6 * wakes + 3 * replay,
        returned_updates=wakes + replay,
        unreturned_work_unknown=False,
    )
    assert r["budget"]["updates_completed"] == wakes
    assert r["replay_examples"] == 3 * replay
    assert r["catalog_control"]["replay_budget"]["replay_examples_completed"] == 3 * replay
    assert r["stream"]["opportunities"] == 9 and r["stream"]["arrived_labels"] == 8
    assert r["stream"]["pending_unapplied"] == 8 - wakes
    assert r["action_snapshot"]["progress"]["decisions_attempted"] == 9
    receipts = r["applied_updates"]
    assert [v["sample_id"] for v in receipts] == [f"batch-{i}" for i in range(wakes)]
    assert all(v["arrived_at"] <= v["applied_at"] < 9 for v in receipts)
    if action != "wait":
        for tick in contention:
            assert r["controller_events"][tick]["poll"]["training"]["deferred_reason"] == (
                "serving_active"
            )
        assert r["sharing_snapshot"]["admitted_updates"] == wakes + replay
    live = r["live_serving"]
    assert live["closed"] and not live["stopped"] and not live["unreturned_work_unknown"]
    assert len(live["records"]) == 9 and all(v["complete"] for v in live["records"])
    assert sum(v["reserved_before_admission"] for v in live["records"]) == len(contention)
    assert sum(v["training_active_at_lease"] for v in live["records"]) == wakes + replay
    public = [p for row in live["records"] for p in row["requests"]]
    assert all(
        p["complete"]
        and p["actor_version"] == "qualified-96"
        and p["native_updates"] == 96
        and p["generation"] == 0
        for p in public
    )
    assert live["latency"]["attempts"] == len(public)
    assert r["serving_work"]["attempts"] == r["serving_work"]["returns"] == len(public)
    assert fitted == pin and serving.made[0].snapshot_state() == serving.initial
    assert r["catalog_actor_unchanged"] and r["source_unchanged"]
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["catalog_promotions"] == r["final_examples"] == r["reserved_probe_updates"] == 0
    assert len(r["publications"]) == 1 and len(serving.readouts) == 1
    assert "promotion_receipt" not in r and "guards" not in r
    costs = control.case_work(r)
    assert costs.attempted_updates == wakes + replay and costs.replay_examples_charged == 3 * replay
    with pytest.raises(ValueError, match="live catalog"):
        demo._case_measures(r, 1.0)


@pytest.mark.parametrize(
    "change",
    [
        dict(reference_mode="drift"),
        dict(catalog_trial=request(), catalog_policy=None, base_snapshot=object()),
        dict(serving_plan=LiveServingPlan(8)),
        dict(serving_plan=LiveServingPlan(10)),
        dict(serving_plan=object()),
        dict(training_context="old_return"),
        dict(deferred_evaluation=True),
        dict(publish_on_drain=True),
        dict(live_training_recipe=object()),
        dict(followup_plan=object()),
        dict(policy_change=True),
        dict(base_snapshot=object()),
        dict(counterfactual=True),
        dict(decision_limit=9),
        dict(
            stream_recipe=TicketStreamRecipe("isolated", ("old_return",) * 8, 1),
            deferred_evaluation=True,
        ),
    ],
)
def test_should_refuse_unsupported_live_combinations_before_source(monkeypatch, change):
    monkeypatch.setattr(demo.context, "_source", lambda p: pytest.fail("source opened"))
    args = dict(catalog_policy=policy(), serving_plan=LiveServingPlan(9)) | change
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), **args)


def test_should_keep_public_task_results_out_of_training_and_controller_feedback(serving):
    first = demo.run_case(
        serving.path, catalog_policy=policy("wake1"), serving_plan=LiveServingPlan(9)
    )
    assert first["complete"], first.get("error")
    public: Any = demo.TorchServingPredictor

    def wrong(actor, spec):
        port = public(actor, spec)
        port.respond = lambda item: decision(
            item, "route" if item.query.split()[0] == "ready" else None
        )
        return port

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "TorchServingPredictor", wrong)
        second = demo.run_case(
            serving.path, catalog_policy=policy("wake1"), serving_plan=LiveServingPlan(9)
        )
    assert second["complete"], second.get("error")
    assert first["candidate_checkpoint"] == second["candidate_checkpoint"]
    assert first["applied_updates"] == second["applied_updates"]
    assert first["training_work"] == second["training_work"]
    assert [v["observation"] for v in first["controller_events"]] == [
        v["observation"] for v in second["controller_events"]
    ]
    assert [v["action_id"] for v in first["controller_events"]] == [
        v["action_id"] for v in second["controller_events"]
    ]
    assert any(v["episode"]["success"] for v in first["live_serving"]["records"])
    assert not any(v["episode"]["success"] for v in second["live_serving"]["records"])


@pytest.mark.parametrize("action,rows,charge", [("wake1", 6, 0), ("replay_wake", 3, 3)])
def test_should_stop_after_lost_native_return_with_permanent_costs_and_closed_service(
    serving, monkeypatch, action, rows, charge
):
    train = FakeBP.train_batch

    def lose(owner, features, targets):
        result = train(owner, features, targets)
        if len(features) == rows:
            raise RuntimeError("lost native return")
        return result

    monkeypatch.setattr(FakeBP, "train_batch", lose)
    r = demo.run_case(serving.path, catalog_policy=policy(action), serving_plan=LiveServingPlan(9))
    assert not r["complete"] and r["training_work"]["unreturned_work_unknown"]
    assert serving.made[0].calls == [rows] and r["replay_examples"] == charge
    assert r["catalog_control"]["replay"]["committed_events"] == 0
    assert r["live_serving"]["closed"] and len(r["live_serving"]["records"]) == 2
    assert serving.readouts == []
    with pytest.raises(ValueError):
        control.case_work(r)


def test_should_preserve_successful_update_if_its_concurrent_service_fails(serving, monkeypatch):
    returned = Event()
    train = FakeBP.train_batch

    def completed(owner, features, targets):
        result = train(owner, features, targets)
        returned.set()
        return result

    monkeypatch.setattr(FakeBP, "train_batch", completed)
    public: Any = demo.TorchServingPredictor

    def broken(actor, spec):
        port = public(actor, spec)
        respond = port.respond

        def fail(item):
            if port.work["attempts"] > 3:
                assert returned.wait(5), "scripted native update never returned"
                raise RuntimeError("lost public response")
            return respond(item)

        port.respond = fail
        return port

    monkeypatch.setattr(demo, "TorchServingPredictor", broken)
    r = demo.run_case(serving.path, catalog_policy=policy("wake1"), serving_plan=LiveServingPlan(9))
    assert not r["complete"] and "lost public response" in r["error"]
    assert r["training_work"]["attempted_updates"] == r["training_work"]["returned_updates"] == 1
    assert serving.made[0].calls == [6] and r["budget"]["updates_completed"] == 1
    live = r["live_serving"]
    assert live["closed"] and live["stopped"] and live["unreturned_work_unknown"]
    assert len(live["records"]) == 2 and live["records"][0]["complete"]
    assert not live["records"][1]["complete"] and serving.readouts == []


def test_should_refuse_corrupted_paid_replay_before_another_fit(serving, monkeypatch):
    choose = control.CatalogTicketControl.choose

    def corrupt(owner, observation, progress):
        if progress.replay_commits:
            row = owner.replay._runtime._consolidations[0]
            owner.replay._runtime._consolidations[0] = replace(row, event_id="changed")
        return choose(owner, observation, progress)

    monkeypatch.setattr(control.CatalogTicketControl, "choose", corrupt)
    r = demo.run_case(serving.path, catalog_policy=policy(), serving_plan=LiveServingPlan(9))
    assert not r["complete"] and serving.made[0].calls == [3]
    assert r["replay_examples"] == 3 and serving.readouts == []
    assert r["live_serving"]["closed"]


@pytest.mark.parametrize("fault", ["source", "policy", "plan", "actor"])
def test_should_refuse_worker_side_changes_before_more_work(serving, monkeypatch, fault):
    fitted, plan = policy("wake1"), LiveServingPlan(9)
    public: Any = demo.TorchServingPredictor

    def corrupt(actor, spec):
        port = public(actor, spec)

        def after():
            if port.work["returns"] != 1:
                return
            if fault == "source":
                serving.path.write_bytes(b"changed-source")
            elif fault == "policy":
                object.__setattr__(fitted, "fitted_at", fitted.fitted_at + 1)
            elif fault == "plan":
                object.__setattr__(plan, "wait_seconds", 1.0)
            else:
                with actor._payload_exclusive():
                    model = actor._payload_references().models[0]
                    model.state = replace(model.state, momentum=999.0)

        port.after = after
        return port

    monkeypatch.setattr(demo, "TorchServingPredictor", corrupt)
    r = demo.run_case(serving.path, catalog_policy=fitted, serving_plan=plan)
    assert not r["complete"] and r["live_serving"]["closed"]
    assert serving.made[0].calls == [] and serving.readouts == []
    assert len(r["live_serving"]["records"]) == 1
    assert serving.ports[0].work["returns"] == 1


@pytest.mark.parametrize("field", ["closed", "unreturned_work_unknown"])
def test_should_refuse_unknown_final_service_capture(serving, monkeypatch, field):
    snapshot = demo.TicketServingTimeline.snapshot

    def corrupt(owner):
        record = snapshot(owner)
        record[field] = field == "unreturned_work_unknown"
        return record

    monkeypatch.setattr(demo.TicketServingTimeline, "snapshot", corrupt)
    r = demo.run_case(serving.path, catalog_policy=policy("wait"), serving_plan=LiveServingPlan(9))
    assert not r["complete"] and "live catalog" in r["error"]
    assert serving.made[0].calls == []
