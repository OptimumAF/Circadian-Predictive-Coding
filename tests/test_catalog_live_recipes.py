"""Actual mixed/noisy TRAIN builders and live workers, with literal learner ports."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_catalog_control as control
from scripts.ticket_controller_streams import LiveTrainingRecipe, TicketStreamRecipe
from scripts.ticket_serving_timeline import LiveServingPlan
from src.core.ticket_workflow import ACTION_IDS
from test_catalog_live_serving import serving as serving
from test_catalog_ticket_stream import policy
from test_ticket_catalog_control import FakeBP, batches, request
from test_ticket_catalog_control import driver as driver
from test_ticket_workflow import decision


def recipe(noise="none", delay=2):
    return LiveTrainingRecipe(
        TicketStreamRecipe(
            "literal-live-mixed",
            ("old_return" if noise == "selected" else "new_contexts",) * 8,
            delay,
            noise == "selected",
        ),
        noise == "retained",
    )


@pytest.fixture
def recipes(serving, monkeypatch):
    packets = list(batches(8))
    for i in range(8, 16):
        targets = packets[i].targets.copy()
        targets[1, 0] = ACTION_IDS.index("inspect")
        packets[i] = replace(packets[i], targets=targets)
    monkeypatch.setattr(demo.context, "_teacher_batches", lambda spec: (None, None, tuple(packets)))
    serving.targets, serving.owners = [], []
    train = FakeBP.train_batch
    bind = control.CatalogTicketControl.bind

    def counted(owner, features, targets):
        serving.targets.append(targets.copy())
        return train(owner, features, targets)

    def bound(owner, runtime, source, gate):
        result = bind(owner, runtime, source, gate)
        serving.owners.append(owner)
        return result

    monkeypatch.setattr(FakeBP, "train_batch", counted)
    monkeypatch.setattr(control.CatalogTicketControl, "bind", bound)
    return serving


def run(recipes, **changes):
    return demo.run_case(
        recipes.path,
        **(
            dict(
                catalog_policy=policy("wake1"),
                serving_plan=LiveServingPlan(9, ()),
                live_training_recipe=recipe(),
            )
            | changes
        ),
    )


@pytest.mark.parametrize("noise", ["none", "selected", "retained"])
@pytest.mark.parametrize("action", ["wait", "wake1", "replay_wake"])
@pytest.mark.parametrize("contention", [(), (2, 6)])
def test_should_use_actual_recipe_without_extending_live_or_learning_limits(
    recipes, noise, action, contention
):
    declared, fitted = recipe(noise), policy(action)
    pin = deepcopy(declared)
    r = run(
        recipes,
        catalog_policy=fitted,
        live_training_recipe=declared,
        serving_plan=LiveServingPlan(9, contention),
    )
    assert r["complete"], r.get("error")
    admitted = 7 - len(contention)
    wakes = admitted if action == "wake1" else 0
    replays = admitted if action == "replay_wake" else 0
    assert r["training_work"] == dict(
        attempted_updates=wakes + replays,
        returned_updates=wakes + replays,
        attempted_rows=12 * wakes + 3 * replays,
        unreturned_work_unknown=False,
    )
    assert r["budget"]["updates_completed"] == wakes
    assert r["replay_examples"] == 3 * replays
    assert r["catalog_control"]["stream"] == dict(
        natural_horizon=10, observed_horizon=9, decision_cap=9, horizon_limited=True
    )
    assert r["training_stream"]["label_delay"] == 2
    assert r["training_stream"]["row_counts"] == [12] * 8
    stream = r["stream"]
    assert stream["opportunities"] == 9 and stream["registered_batches"] == 8
    assert stream["arrived_labels"] == 7 and stream["unarrived_labels"] == 1
    assert stream["arrived_unapplied"] == 7 - wakes and stream["pending_unapplied"] == 8 - wakes
    assert r["action_snapshot"]["progress"]["decisions_returned"] == 9
    assert [u["sample_id"] for u in r["applied_updates"]] == [f"batch-{i}" for i in range(wakes)]
    assert all(
        u["arrived_at"] == int(u["sample_id"].split("-")[1]) + 2 <= u["applied_at"] < 9
        for u in r["applied_updates"]
    )
    for targets in recipes.targets:
        if len(targets) != 12:
            continue
        assert targets[7, 0] == ACTION_IDS.index("defer" if noise == "retained" else "inspect")
        if noise == "selected":
            assert targets[1, 0] == ACTION_IDS.index("defer")
    assert r["training_corruption"]["rows_per_batch"] == (
        [] if noise == "none" else [1 if noise == "selected" else 7]
    )
    owner = recipes.owners[0]
    assert owner.live_recipe is declared and declared == pin
    assert len(owner.replay._runtime._inbox._labels) == 7
    assert r["catalog_control"]["native_history"]["training_rows"] == 576 + 12 * wakes + 3 * replays
    assert len(r["live_serving"]["records"]) == 9 and r["live_serving"]["closed"]
    assert all(e["complete"] for e in r["live_serving"]["records"])
    assert all(
        p["generation"] == 0 and p["actor_version"] == "qualified-96" and p["native_updates"] == 96
        for e in r["live_serving"]["records"]
        for p in e["requests"]
    )
    assert (
        r["catalog_actor_unchanged"]
        and r["source_unchanged"]
        and r["numpy_rng_unchanged"]
        and r["torch_rng_unchanged"]
    )
    assert recipes.made[0].snapshot_state() == recipes.initial and len(recipes.readouts) == 1
    assert r["catalog_promotions"] == r["final_examples"] == r["reserved_probe_updates"] == 0
    assert "guards" not in r and "promotion_receipt" not in r
    assert control.case_work(r).attempted_rows == 12 * wakes + 3 * replays
    with pytest.raises(ValueError, match="live catalog"):
        demo._case_measures(r, 1.0)


def test_should_preserve_delay_one_and_its_original_complete_arrivals(recipes):
    r = run(recipes, live_training_recipe=recipe(delay=1))
    assert r["complete"], r.get("error")
    assert r["stream"]["arrived_labels"] == r["training_work"]["returned_updates"] == 8
    assert r["stream"]["unarrived_labels"] == r["stream"]["pending_unapplied"] == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(serving_plan=None),
        dict(serving_plan=LiveServingPlan(10)),
        dict(live_training_recipe=object()),
        dict(deferred_evaluation=True),
        dict(
            stream_recipe=TicketStreamRecipe("overlap", ("new_contexts",) * 8, 2),
            deferred_evaluation=True,
        ),
        dict(reference_mode="drift"),
        dict(publish_after_window=True),
        dict(publish_on_drain=True),
        dict(followup_plan=object()),
        dict(policy_change=True),
        dict(training_context="old_return"),
        dict(catalog_trial=request(n=8), catalog_policy=None, base_snapshot=object()),
        dict(
            live_training_recipe=LiveTrainingRecipe(
                TicketStreamRecipe("short", ("new_contexts",), 2)
            )
        ),
    ],
)
def test_should_refuse_incompatible_recipe_modes_before_io(monkeypatch, change):
    monkeypatch.setattr(demo.context, "_source", lambda *args: pytest.fail("source opened"))
    args = (
        dict(
            catalog_policy=policy(), serving_plan=LiveServingPlan(9), live_training_recipe=recipe()
        )
        | change
    )
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), **args)


@pytest.mark.parametrize("fault", ["flag", "delay", "subclass"])
def test_should_validate_exact_live_recipe_metadata_before_io(monkeypatch, fault):
    class DerivedRecipe(LiveTrainingRecipe):
        pass

    declared = recipe()
    if fault == "flag":
        object.__setattr__(declared, "retained_invoice_noise", 0)
    elif fault == "delay":
        object.__setattr__(declared.recipe, "label_delay", True)
    else:
        declared = DerivedRecipe(declared.recipe)
    monkeypatch.setattr(demo.context, "_source", lambda *args: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        demo.run_case(
            Path("unused"),
            catalog_policy=policy(),
            serving_plan=LiveServingPlan(9),
            live_training_recipe=declared,
        )


def test_should_refuse_non_train_builder_output_before_model_construction(recipes, monkeypatch):
    build = demo.build_live_stream

    def wrong(*args):
        packets = build(*args)
        return (replace(packets[0], role="final_test"), *packets[1:])

    monkeypatch.setattr(demo, "build_live_stream", wrong)
    r = run(recipes)
    assert not r["complete"] and recipes.made == [] and recipes.readouts == []


@pytest.mark.parametrize(
    "fault", ["value", "root", "underlying", "delay", "packet", "schedule", "receipt", "budget"]
)
def test_should_stop_changed_evidence_after_paid_update_before_more_work(
    recipes, monkeypatch, fault
):
    choose = control.CatalogTicketControl.choose

    def corrupt(owner, observation, progress):
        result = choose(owner, observation, progress)
        if observation.tick == 3:
            if fault == "value":
                object.__setattr__(owner.live_recipe, "retained_invoice_noise", True)
            elif fault == "root":
                owner.live_recipe = deepcopy(owner.live_recipe)
            elif fault == "underlying":
                owner.stream_recipe = deepcopy(owner.stream_recipe)
            elif fault == "delay":
                object.__setattr__(owner.live_recipe.recipe, "label_delay", 1)
            elif fault == "packet":
                np.copyto(owner._packets[0].targets, 5)
            elif fault == "schedule":
                owner._schedule_packets = tuple(list(owner._schedule_packets))
            elif fault == "receipt":
                inbox = owner.replay._runtime._inbox
                key = next(iter(inbox._applied))
                inbox._applied[key] = replace(inbox._applied[key])
            else:
                owner._wake.budget = replace(owner._wake.budget, max_training_updates=9)
        return result

    monkeypatch.setattr(control.CatalogTicketControl, "choose", corrupt)
    r = run(recipes)
    assert not r["complete"] and recipes.made[0].calls == [12]
    assert r["training_work"]["attempted_updates"] == 1 and recipes.readouts == []
    assert r["live_serving"]["closed"]


@pytest.mark.parametrize("fault", ["value", "root", "packet", "source"])
def test_should_refuse_worker_changes_before_another_prediction(recipes, monkeypatch, fault):
    public: Any = demo.TorchServingPredictor

    def changed(actor, spec):
        port = public(actor, spec)

        def after():
            if port.work["returns"] != 1:
                return
            owner = recipes.owners[0]
            if fault == "value":
                object.__setattr__(owner.live_recipe, "retained_invoice_noise", True)
            elif fault == "root":
                owner.live_recipe = deepcopy(owner.live_recipe)
            elif fault == "packet":
                np.copyto(owner._packets[0].targets, 5)
            else:
                recipes.path.write_bytes(b"changed")

        port.after = after
        return port

    monkeypatch.setattr(demo, "TorchServingPredictor", changed)
    r = run(recipes)
    assert not r["complete"] and recipes.ports[0].work["returns"] == 1
    assert recipes.made[0].calls == [] and recipes.readouts == [] and r["live_serving"]["closed"]


@pytest.mark.parametrize("action,rows,charge", [("wake1", 12, 0), ("replay_wake", 3, 3)])
def test_should_preserve_lost_native_return_and_stop_service(
    recipes, monkeypatch, action, rows, charge
):
    train = FakeBP.train_batch

    def lost(owner, features, targets):
        train(owner, features, targets)
        raise RuntimeError("lost native return")

    monkeypatch.setattr(FakeBP, "train_batch", lost)
    r = run(recipes, catalog_policy=policy(action))
    assert not r["complete"] and r["training_work"]["unreturned_work_unknown"]
    assert recipes.made[0].calls == [rows] and r["replay_examples"] == charge
    assert recipes.readouts == [] and r["live_serving"]["closed"]


def test_should_keep_live_outcomes_out_of_recipe_feedback_and_choices(recipes, monkeypatch):
    a = run(recipes)
    public: Any = demo.TorchServingPredictor

    def wrong(actor, spec):
        port = public(actor, spec)
        port.respond = lambda item: decision(
            item, "route" if item.query.split()[0] == "ready" else None
        )
        return port

    monkeypatch.setattr(demo, "TorchServingPredictor", wrong)
    b = run(recipes)
    assert a["complete"] and b["complete"]
    assert a["candidate_checkpoint"] == b["candidate_checkpoint"]
    assert a["applied_updates"] == b["applied_updates"] and a["training_work"] == b["training_work"]
    assert [(e["observation"], e["action_id"]) for e in a["controller_events"]] == [
        (e["observation"], e["action_id"]) for e in b["controller_events"]
    ]
    assert any(e["episode"]["success"] for e in a["live_serving"]["records"])
    assert not any(e["episode"]["success"] for e in b["live_serving"]["records"])
