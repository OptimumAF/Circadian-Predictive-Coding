"""Catalog ticket driver wiring with explicit scripted models and task readouts."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_controller_learning as learning
from scripts import ticket_catalog_control as catalog_driver
from src.app.counterfactual_controller import CatalogCounterfactualRequest
from src.core.catalog_contextual_controller import (
    ContextualArmCatalog,
    CatalogEstimate,
    FittedCatalogController,
)
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.controller_objective import ControllerObjective
from src.app.coupled_attribution import AttributionBatch
from src.core.learner_ports import TrainingDiagnostic
from src.adapters.torch_feature_learner import TORCH_CE_ID
from test_categorical_replay_action import ScalarLearner, State, episode


def catalog(n=8, replay_rows=6):
    return ContextualArmCatalog(
        AdaptationEnvelope(
            (
                AdaptationAction("wait"),
                AdaptationAction("wake1", 1),
                AdaptationAction("replay_wake", 1, 3),
            ),
            n + 1,
            n,
            replay_rows,
        )
    )


def digest(state):
    return json.dumps(asdict(state), sort_keys=True)


def request(action="replay_wake", n=8, context="mistakes"):
    return CatalogCounterfactualRequest(
        "qualified-96",
        digest(State(completed_updates=96, training_examples=576)),
        context,
        action,
        0,
        catalog=catalog(n),
    )


def fitted(n=8):
    c = catalog(n)
    rows = tuple(
        CatalogEstimate(ctx, arm, float(arm == "replay_wake"))
        for ctx in ("clean", "mistakes")
        for arm in c.arms
    )
    return FittedCatalogController(
        "qualified-96",
        digest(State(completed_updates=96, training_examples=576)),
        ControllerObjective(),
        10,
        rows,
        ("replay_wake", "replay_wake"),
        catalog=c,
    )


def test_should_refuse_mixed_catalog_modes_before_source_io(monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda *args: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), catalog_trial=request(), catalog_policy=fitted())


class FakeBP(ScalarLearner):
    def __init__(self, before=None, after=None, calls=None, faults=None):
        super().__init__(calls, faults)
        self.state = State(completed_updates=96, training_examples=576)
        self.before, self.after = before, after

    def fork(self):
        child = FakeBP(self.before, self.after, self.calls, self.faults)
        child.restore_state(self.state)
        return child

    def train_batch(self, features, targets):
        if self.before:
            self.before(features)
        result = super().train_batch(features, targets)
        if self.after:
            self.after()
        return TrainingDiagnostic(TORCH_CE_ID, result.value)


def batches(n=2):
    return tuple(
        AttributionBatch(
            "train",
            "new" if i < n else "old",
            tuple(f"row-{i}-{j}" for j in range(6)),
            np.ones((6, 7)),
            np.ones((6, 1)) if i < n else np.zeros((6, 1)),
            num_classes=6,
        )
        for i in range(2 * n)
    )


@pytest.fixture
def driver(monkeypatch, tmp_path):
    source_path = tmp_path / "literal-source.json"
    source_path.write_bytes(b"literal-source")
    initial = State(completed_updates=96, training_examples=576)
    made, readouts = [], []

    def factory(method, width, classes, before, after):
        model = FakeBP(before, after)
        made.append(model)
        return model

    monkeypatch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
    monkeypatch.setattr(
        demo.context,
        "_source",
        lambda p: (
            b"literal-source",
            dict(
                checkpoints=[dict(state=initial)],
                training_work=dict(scripted=True),
                costs=dict(scripted=True),
            ),
        ),
    )
    monkeypatch.setattr(demo.context, "bp_snapshot_from_record", lambda v: v)
    monkeypatch.setattr(demo.context, "_teacher_batches", lambda spec: (None, None, batches()))
    monkeypatch.setattr(demo.native, "counted_learner", factory)
    monkeypatch.setattr(demo.native, "same_state", lambda a, b: a == b)
    monkeypatch.setattr(demo.native, "snapshot_record", asdict)
    monkeypatch.setattr(
        demo.native,
        "checkpoint",
        lambda point, state: dict(
            source_updates=point, source_rows=state.training_examples, state=asdict(state)
        ),
    )
    monkeypatch.setattr(demo, "snapshot_digest", digest)
    rng = SimpleNamespace(clone=lambda: (0,))
    monkeypatch.setattr(
        demo.native,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: True),
    )
    monkeypatch.setattr(
        catalog_driver,
        "original_bank",
        lambda: (
            (episode(),),
            dict(prepared_episodes=1, prepared_rows=3, retained_rows=3, scripted=True),
        ),
    )
    monkeypatch.setattr(
        demo,
        "TorchServingPredictor",
        lambda *args: SimpleNamespace(
            work=dict(attempts=0, returns=0, readouts=0, native_forwards=0)
        ),
    )
    monkeypatch.setattr(
        demo,
        "ServingPromotionController",
        lambda *args, **kwargs: SimpleNamespace(
            prepare=lambda *a, **kw: pytest.fail("catalog promotion"),
            commit=lambda *a, **kw: pytest.fail("catalog commit"),
        ),
    )
    monkeypatch.setattr(
        demo,
        "run_workflow_episode",
        lambda *args, **kw: dict(complete=True, success=False, steps=[]),
    )

    def readout(spec, runtime, report, require):
        require()
        readouts.append(runtime.candidate_snapshot().state)
        report["candidate_checkpoint"] = demo.native.checkpoint(
            runtime.candidate_snapshot().wake_updates, runtime.candidate_snapshot().state
        )
        report["candidate_readout_work"] = dict(
            work=dict(native_forward_attempts=0, native_forward_returns=0)
        )
        report["candidate_episodes"] = []
        report["reference_outcomes"] = dict(new_before=0, new_after=0, old_before=2, old_after=2)

    monkeypatch.setattr(demo, "_candidate_tasks", readout)
    return SimpleNamespace(path=source_path, initial=initial, made=made, readouts=readouts)


@pytest.mark.parametrize(
    "action,updates,rows,replayed,pending",
    [("wait", 0, 0, 0, 2), ("wake1", 2, 12, 0, 0), ("replay_wake", 4, 18, 6, 0)],
)
def test_should_run_actual_private_catalog_schedule_with_original_separate_charges(
    driver, action, updates, rows, replayed, pending
):
    r = demo.run_case(driver.path, catalog_trial=request(action, 2), base_snapshot=driver.initial)
    assert r["complete"], r.get("error")
    assert r["definition"] == "catalog_bp_ticket_private_v1"
    assert r["catalog_control"]["phase"] == "trial" and r["catalog_promotions"] == 0
    assert r["training_work"] == dict(
        attempted_updates=updates,
        attempted_rows=rows,
        returned_updates=updates,
        unreturned_work_unknown=False,
    )
    assert r["replay_examples"] == replayed and r["stream"]["pending_unapplied"] == pending
    assert r["budget"]["updates_completed"] == (0 if action == "wait" else 2)
    assert r["catalog_control"]["native_history"]["completed_updates"] == 96 + updates
    assert r["catalog_control"]["replay_budget"]["limits"]["max_training_updates"] == 0
    assert r["catalog_actor_unchanged"] and r["source_unchanged"]
    assert len(driver.readouts) == 1 and driver.made[0].snapshot_state() == driver.initial
    costs = catalog_driver.case_work(r)
    costs.require_complete(catalog(2), action)
    assert costs.attempted_updates == updates and costs.attempted_rows == rows
    assert costs.replay_examples_charged == replayed


def test_should_apply_frozen_catalog_privately_without_rebase_or_publication(driver):
    r = demo.run_case(driver.path, catalog_policy=fitted(2))
    assert r["complete"], r.get("error")
    assert r["catalog_control"]["phase"] == "private_application"
    assert r["catalog_actor_unchanged"] and r["catalog_promotions"] == 0
    assert "promotion_receipt" not in r and "guards" not in r
    assert driver.made[0].calls == [3, 6, 3, 6]


@pytest.mark.parametrize(
    "change",
    [
        {"reference_mode": "drift"},
        {"counterfactual": True},
        {"deferred_evaluation": True},
        {"publish_on_drain": True},
        {"policy_change": True},
        {"serving_plan": object()},
        {"stream_recipe": object()},
        {"live_training_recipe": object()},
        {"followup_plan": object()},
        {"training_context": "other"},
    ],
)
def test_should_refuse_unsupported_catalog_combinations_before_io(change, monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda *args: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), catalog_trial=request(), base_snapshot=object(), **change)


@pytest.mark.parametrize("fault", ["digest", "base", "version", "role", "catalog"])
def test_should_bind_original_complete_source_before_model_construction(driver, fault):
    req, base = request(n=2), driver.initial
    if fault == "digest":
        req = replace(req, state_digest="wrong")
    elif fault == "base":
        base = replace(base, momentum=999.0)
    elif fault == "version":
        req = replace(req, learner_version="new")
    elif fault == "role":
        req = replace(req, role="final_test")
    else:
        req = replace(req, catalog=catalog(8))
    try:
        r = demo.run_case(driver.path, catalog_trial=req, base_snapshot=base)
    except ValueError:
        r = dict(complete=False)
    assert not r["complete"] and driver.made == [] and driver.readouts == []


def test_should_keep_replay_and_lost_wake_return_paid_and_refuse_task_scoring(driver, monkeypatch):
    original = FakeBP.train_batch

    def lose(owner, features, targets):
        result = original(owner, features, targets)
        if len(features) == 6:
            raise RuntimeError("lost wake return")
        return result

    monkeypatch.setattr(FakeBP, "train_batch", lose)
    r = demo.run_case(driver.path, catalog_trial=request(n=2), base_snapshot=driver.initial)
    assert not r["complete"] and driver.readouts == []
    assert r["catalog_control"]["replay"]["committed_events"] == 1
    assert r["replay_examples"] == 3 and r["catalog_control"]["replay"]["stopped"]
    with pytest.raises(ValueError):
        catalog_driver.case_work(r)


def test_should_refuse_changed_original_record_before_more_native_work(driver, monkeypatch):
    req = request(n=2)
    original = FakeBP.train_batch

    def change(owner, features, targets):
        result = original(owner, features, targets)
        object.__setattr__(req, "action_id", "wake1")
        return result

    monkeypatch.setattr(FakeBP, "train_batch", change)
    r = demo.run_case(driver.path, catalog_trial=req, base_snapshot=driver.initial)
    assert not r["complete"] and "record changed" in r["error"]
    assert driver.made[0].calls == [3] and driver.readouts == []


def test_should_compile_only_the_original_complete_train_bank(monkeypatch):
    from scripts import ticket_workflow_rehearsal as bank

    dispatches = []
    dispatch = bank.dispatch_workflow

    def count(*args, **kwargs):
        dispatches.append(1)
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(bank, "dispatch_workflow", count)
    episodes, metadata = catalog_driver.original_bank()
    assert len(episodes) == 1 and episodes[0].role == "train" and len(episodes[0].transitions) == 3
    assert metadata["prepared_episodes"] == 12 and metadata["prepared_rows"] == 36
    assert len(dispatches) == 36


def test_should_run_six_real_scripted_trials_then_two_frozen_private_applications(
    driver, monkeypatch
):
    original_measures = demo._case_measures
    # Literal unit costs make the negative/select-wait fixture deterministic;
    # production utility, timing, metrics and all outcomes stay unchanged.
    monkeypatch.setattr(
        demo,
        "_case_measures",
        lambda result, seconds: replace(
            original_measures(result, seconds),
            execution_seconds=1.0 + result["training_work"]["attempted_updates"],
            peak_rss_bytes=1024,
        ),
    )
    r = learning.run_study(driver.path, catalog=catalog(2))
    assert r["complete"], r.get("error")
    assert len(r["trials"]) == 6 and len(r["applications"]) == 2
    assert [row["request"]["action_id"] for row in r["trials"]] == [
        "wait",
        "wake1",
        "replay_wake",
    ] * 2
    assert r["training_state"]["discarded_trials"] == 6
    assert all(
        app["catalog_control"]["phase"] == "private_application"
        for app in r["applications"].values()
    )
    assert r["total_training_work"]["attempted_updates"] == 12
    assert r["total_replay_examples"] == 12
    assert r["total_catalog_work"]["replay_examples_charged"] == 12
    assert r["final_source_reads"] == r["probe_updates"] == 0
    assert r["fitted_policy"]["choices"] == ("wait", "wait")
    assert json.dumps(r, allow_nan=False)


def test_should_leave_fresh_packets_pending_when_replay_credit_is_exhausted(driver):
    policy = replace(fitted(2), catalog=catalog(2, replay_rows=3))
    r = demo.run_case(driver.path, catalog_policy=policy)
    assert r["complete"], r.get("error")
    assert driver.made[0].calls == [3, 6] and r["stream"]["pending_unapplied"] == 1
    assert r["budget"]["updates_completed"] == 1 and r["replay_examples"] == 3
    assert r["controller_events"][-1]["action_id"] == "wait"
    assert r["controller_events"][-1]["reason"] == "insufficient_declared_replay_credit"


def test_should_preserve_all_original_resource_deferrals_without_replaying(driver, monkeypatch):
    gate_class = demo.ServingPriorityGate

    def paused(*args, **kw):
        gate = gate_class(*args, **kw)
        gate.pause()
        return gate

    monkeypatch.setattr(demo, "ServingPriorityGate", paused)
    r = demo.run_case(driver.path, catalog_policy=fitted(2))
    assert r["complete"], r.get("error")
    assert driver.made[0].calls == [] and r["replay_examples"] == 0
    assert r["stream"]["pending_unapplied"] == 2
    assert r["controller_events"][-1]["poll"]["training"]["deferred_reason"] == "paused"


def test_should_refuse_replaced_replay_receipt_before_another_native_update(driver, monkeypatch):
    choose = catalog_driver.CatalogTicketControl.choose

    def change(owner, observation, progress):
        if progress.replay_commits:
            r = owner.replay._runtime
            r._consolidations[0] = replace(r._consolidations[0], event_id="replaced")
        return choose(owner, observation, progress)

    monkeypatch.setattr(catalog_driver.CatalogTicketControl, "choose", change)
    r = demo.run_case(driver.path, catalog_policy=fitted(2))
    assert not r["complete"] and driver.readouts == []
    assert driver.made[0].calls == [3, 6] and r["replay_examples"] == 3
    assert r["catalog_control"]["replay"]["stopped"]


@pytest.mark.parametrize(
    "fault", ["native", "replay", "history", "actor", "controller", "serving", "readout"]
)
def test_should_refuse_unknown_or_corrupted_case_work_before_metrics(driver, fault):
    good = demo.run_case(driver.path, catalog_policy=fitted(2))
    assert good["complete"]
    bad = deepcopy(good)
    if fault == "native":
        bad["training_work"]["returned_updates"] = 0
    elif fault == "replay":
        bad["replay_examples"] = 0
    elif fault == "history":
        bad["catalog_control"]["native_history"]["completed_updates"] += 1
    elif fault == "actor":
        bad["catalog_actor_unchanged"] = False
    elif fault == "controller":
        bad["controller_work"]["prediction_returns"] = 0
    elif fault == "serving":
        bad["serving_work"]["attempts"] = 1
    else:
        bad["candidate_readout_work"]["work"]["native_forward_returns"] = 1
    with pytest.raises(ValueError):
        catalog_driver.case_work(bad)


def test_should_stop_study_on_first_unknown_paid_case_without_applications(driver, monkeypatch):
    run = demo.run_case
    calls = []

    def failed(*args, **kw):
        calls.append(kw["catalog_trial"].action_id)
        result = run(*args, **kw)
        if kw["catalog_trial"].action_id == "wake1":
            result["training_work"]["unreturned_work_unknown"] = True
        return result

    monkeypatch.setattr(demo, "run_case", failed)
    r = learning.run_study(driver.path, catalog=catalog(2))
    assert not r["complete"] and r["applications"] == {}
    assert calls == ["wait", "wake1"] and r["training_state"]["stopped"]
    assert r["training_state"]["attempts"] == 2
    assert r["trials"][-1]["result"]["training_work"]["attempted_updates"] == 2


def test_should_refuse_changed_preceding_paid_case_before_opening_another_case(driver, monkeypatch):
    run = demo.run_case
    prior = []
    original_measures = demo._case_measures

    def keep(*args, **kw):
        result = run(*args, **kw)
        prior.append(result)
        return result

    def mutate(result, seconds):
        measured = original_measures(result, seconds)
        if len(prior) == 2:
            prior[0]["training_work"] = deepcopy(prior[0]["training_work"])
        return measured

    monkeypatch.setattr(demo, "run_case", keep)
    monkeypatch.setattr(demo, "_case_measures", mutate)
    r = learning.run_study(driver.path, catalog=catalog(2))
    assert not r["complete"] and len(prior) == 2 and r["applications"] == {}
    assert "preceding paid catalog" in r["error"]


def test_should_preserve_original_v1_trial_and_study_shapes(driver, monkeypatch):
    from src.core.contextual_controller import ContextualChoice

    r = demo.run_case(
        driver.path,
        contextual_policy=ContextualChoice("wait", "wait"),
        base_snapshot=driver.initial,
        counterfactual=True,
    )
    assert r["complete"], r.get("error")
    assert r["definition"] == "contextual_bp_ticket_serving_v1"
    assert "catalog_control" not in r and r["replay_examples"] == 0
    monkeypatch.setattr(demo, "_publish_candidate", lambda *args: None)
    studied = learning.run_study(driver.path)
    assert studied["complete"], studied.get("error")
    assert studied["definition"] == "learned_contextual_controller_v1"
    assert len(studied["trials"]) == 4 and "catalog" not in studied
    assert studied["max_total_training_updates"] == 32


def test_should_preserve_the_actual_eight_batch_horizon_and_pending_prefix(driver, monkeypatch):
    monkeypatch.setattr(demo.context, "BATCHES_PER_PHASE", 8)
    monkeypatch.setattr(demo.context, "_teacher_batches", lambda spec: (None, None, batches(8)))
    r = demo.run_case(driver.path, catalog_policy=fitted(8))
    assert r["complete"], r.get("error")
    assert r["stream"]["opportunities"] == 9 and r["stream"]["registered_batches"] == 8
    assert r["budget"]["limits"]["max_training_updates"] == 8
    assert r["stream"]["pending_unapplied"] == 6
    assert r["catalog_control"]["native_history"]["completed_updates"] == 100
    assert driver.made[0].calls == [3, 6, 3, 6]


@pytest.mark.parametrize("fault", ["candidate", "source", "actor"])
def test_should_validate_paid_success_before_admitting_private_task_readouts(
    driver, monkeypatch, fault
):
    run = demo._reference_schedule

    def corrupt(*args, **kwargs):
        runtime, now = run(*args, **kwargs)
        target = {
            "candidate": runtime._candidate,
            "source": args[0],
            "actor": runtime.actor._slot.bundle.learner,
        }[fault]
        target.state = replace(target.state, weight=999.0)
        return runtime, now

    monkeypatch.setattr(demo, "_reference_schedule", corrupt)
    r = demo.run_case(driver.path, catalog_policy=fitted(2))
    assert not r["complete"] and driver.readouts == []
    assert driver.made[0].calls == [3, 6, 3, 6]
    assert r["replay_examples"] == 6 and r["catalog_control"]["replay"]["stopped"], r.get("error")
