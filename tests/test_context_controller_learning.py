"""Source-bound contextual application and disposable TRAIN trials, not a study."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import context_controller_learning as learning
from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as cli
from scripts.context_head_probe import head_states
from scripts.context_head_serving import prepare_data
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters.context_serving import snapshot_digest
from src.core.contextual_controller import ContextualChoice, ContextualEstimate, FittedController
from src.core.controller_objective import ControllerObjective
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")


def initial():
    return head_states(json.loads(SOURCE.read_bytes()))[0]


def fitted():
    estimates = tuple(
        ContextualEstimate(c, a, float(c == "mistakes" and a == "wake1"))
        for c in ("clean", "mistakes")
        for a in ("wait", "wake1")
    )
    return FittedController(
        "qualified-12",
        snapshot_digest(initial()),
        ControllerObjective(),
        4,
        estimates,
        ContextualChoice("wait", "wake1"),
    )


@pytest.mark.parametrize("policy", [True, {}, [], "wake1"])
def test_should_refuse_invalid_contextual_policy_before_io(policy, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source read"))
    with pytest.raises(ValueError):
        live.run_case("unused", contextual_policy=policy)


def test_should_refuse_both_policy_modes_before_io(monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source read"))
    with pytest.raises(ValueError, match="either"):
        live.run_case(
            "unused", reference_mode="drift", contextual_policy=ContextualChoice("wait", "wait")
        )


@pytest.mark.parametrize(
    "field,value", [("learner_version", "qualified-96"), ("state_digest", "other")]
)
def test_should_refuse_fit_for_other_complete_source_before_models(field, value, monkeypatch):
    policy = replace(fitted(), **{field: value})
    monkeypatch.setattr(
        live, "ContextPerceptronLearner", lambda *a: pytest.fail("model constructed")
    )
    r = live.run_case(SOURCE, contextual_policy=policy)
    assert not r["complete"] and "version/state" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


def test_should_pass_frozen_policy_to_original_schedule_and_allow_pending(monkeypatch):
    p = fitted()
    seen = []

    def schedule(*args, **kwargs):
        seen.append(kwargs["contextual_policy"])
        return SimpleNamespace(applied_updates=[]), 14

    monkeypatch.setattr(cli, "_reference_schedule", schedule)
    r = live.run_case(SOURCE, contextual_policy=p)
    assert r["complete"] and not r["training_complete"] and seen == [p]
    assert r["source_unchanged"] and r["budget"]["updates"] == 0


def test_should_stop_on_original_policy_mutation(monkeypatch):
    p = ContextualChoice("wait", "wake1")

    def schedule(*args, **kwargs):
        object.__setattr__(p, "clean", "wake1")
        return SimpleNamespace(applied_updates=[]), 14

    monkeypatch.setattr(cli, "_reference_schedule", schedule)
    r = live.run_case(SOURCE, contextual_policy=p)
    assert not r["complete"] and "original contextual policy changed" in r["error"]
    assert r["budget"]["updates"] == 0


@pytest.mark.parametrize("index,action", [(0, "wait"), (0, "wake1"), (1, "wait"), (1, "wake1")])
def test_should_execute_one_causal_disposable_trial_without_changing_source(index, action):
    state = initial()
    data = prepare_data(DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG))
    r = learning._trial(state, data[0][index], data[1], action, lambda: None)
    assert r["complete"], r.get("error")
    n = int(action == "wake1")
    assert r["training_work"] == dict(
        attempted_updates=n, attempted_rows=n * (3, 6)[index], returned_updates=n
    )
    assert r["before"] == [5, 12] and r["source_unchanged"] and r["actor_unchanged"]
    assert r["after"] == ([6, 11] if index == 0 and n else [5, 12])
    assert r["budget"]["updates"] == n and r["stream"]["pending_unapplied"] == 1 - n
    assert r["controller_events"][0]["observation"]["feedback"] is None
    assert r["controller_events"][1]["observation"]["feedback"]["arrived_at"] == 1
    assert state == initial()


def scripted_trial(state, batch, guards, action, require):
    require()
    new = batch.context == "new-v1" and action == "wake1"
    return dict(
        complete=True,
        before=[5, 12],
        after=[6, 11] if new else [5, 12],
        elapsed_seconds=1,
        memory=dict(peak_bytes=1024),
        training_work=dict(attempted_updates=int(action == "wake1")),
    )


def test_should_freeze_complete_train_fit_before_application_without_task_feedback(monkeypatch):
    monkeypatch.setattr(learning, "_trial", scripted_trial)
    applied = []

    def application(path, *, contextual_policy):
        applied.append(deepcopy(contextual_policy))
        return dict(complete=True, success=False)

    monkeypatch.setattr(live, "run_case", application)
    r = learning.run_study(SOURCE)
    assert r["complete"], r.get("error")
    assert [(t["request"]["context"], t["request"]["action_id"]) for t in r["trials"]] == [
        ("mistakes", "wait"),
        ("mistakes", "wake1"),
        ("clean", "wait"),
        ("clean", "wake1"),
    ]
    assert len(applied) == 1 and applied[0].choices == ContextualChoice("wait", "wake1")
    assert r["fitted_policy"]["choices"] == dict(clean="wait", mistakes="wake1")
    assert r["training_state"]["discarded_trials"] == r["training_state"]["completed"] == 4
    assert all(
        t["request"]["started_at"] < t["labels_arrived_at"] <= t["completed_at"] <= r["fitted_at"]
        for t in r["trials"]
    )
    assert r["final_examples"] == r["probe_updates"] == r["replay_examples"] == 0


def test_should_preserve_failed_paid_trial_and_admit_nothing_more(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(1)
        return dict(complete=False, error="lost return", native_calls=[dict(complete=False)])

    monkeypatch.setattr(learning, "_trial", fail)
    monkeypatch.setattr(live, "run_case", lambda *a, **k: pytest.fail("application entered"))
    r = learning.run_study(SOURCE)
    assert not r["complete"] and calls == [1]
    assert r["trials"][0]["result"]["native_calls"] == [dict(complete=False)]
    assert r["training_state"]["stopped"]


def test_should_restore_corrupted_preceding_trial_before_next_admission(monkeypatch):
    retained = []
    original = learning.paid_report_guard

    def seal(row):
        retained.append(row)
        return original(row)

    calls = []

    def corrupt(*args):
        calls.append(1)
        if len(calls) == 2:
            retained[0]["result"]["after"][0] = 999
        return scripted_trial(*args)

    monkeypatch.setattr(learning, "paid_report_guard", seal)
    monkeypatch.setattr(learning, "_trial", corrupt)
    monkeypatch.setattr(live, "run_case", lambda *a, **k: pytest.fail("application entered"))
    r = learning.run_study(SOURCE)
    assert not r["complete"] and "corrupted" in r["error"] and len(calls) == 2
    assert r["trials"][0]["result"]["after"] == [5, 12]


def test_should_dispatch_new_cli_without_entering_legacy_fit(tmp_path, monkeypatch):
    seen = []

    def study(path):
        seen.append(path)
        return dict(complete=True)

    monkeypatch.setattr(learning, "run_study", study)
    output = tmp_path / "out.json"
    assert (
        cli.main(
            ["--source", str(SOURCE), "--context-controller-learning", "--output", str(output)]
        )
        == 0
    )
    assert seen == [SOURCE] and json.loads(output.read_bytes())["complete"]
