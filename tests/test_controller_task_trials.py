"""Private controller-development outcomes; scalar fixtures are not experiments."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import context_controller_learning as learning
from scripts import context_head_serving as serving
from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as cli
from src.core.controller_objective import ControllerObjective, controller_utility
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG
from test_context_controller_learning import initial

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")


def task_rows(phase, native, *, new=1, old=4):
    rows: list[dict[str, Any]] = []
    for i, (version, concept) in enumerate(serving.TASKS):
        success = i < old if i < 4 else i - 4 < new
        rows.append(
            dict(
                role="controller_train",
                phase=phase,
                policy_version=version,
                concept=concept,
                complete=True,
                reads=[
                    dict(
                        generation=0, actor_version=f"controller-{phase}", completed_updates=native
                    )
                ]
                * 3,
                episode=dict(
                    complete=True,
                    success=success,
                    scenario="ordinary",
                    training_updates=0,
                    external_actions=0,
                    initial_state=dict(policy_version=version, concept=concept),
                    unreturned_prediction_work_unknown=False,
                    unvalidated_return_work_unknown=False,
                    work={
                        k: 3
                        for k in (
                            "prediction_calls_attempted",
                            "prediction_calls_returned",
                            "prediction_calls_validated",
                        )
                    },
                    steps=[],
                ),
            )
        )
    return rows


def scripted(state, batch, guards, action, require, *, task_outcomes=False):
    assert task_outcomes is True
    require()
    n = int(action == "wake1")
    new = batch.context == "new-v1" and n == 1
    return dict(
        complete=True,
        role="controller_train",
        before=[1, 4],
        after=[2, 3] if new else [1, 4],
        task_outcomes=dict(
            before=task_rows("before", 12),
            after=task_rows("after", 12 + n, new=2 if new else 1, old=3 if new else 4),
        ),
        task_readout_work={
            p: dict(attempts=18, returns=18, readouts=18, typed_attempts=18)
            for p in ("before", "after")
        },
        native_calls=[dict(complete=True)] * n,
        budget=dict(updates=n),
        source_unchanged=True,
        actor_unchanged=True,
        elapsed_seconds=1,
        memory=dict(peak_bytes=1024),
        training_work=dict(
            attempted_updates=n, returned_updates=n, attempted_rows=n * len(batch.features)
        ),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
    )


@pytest.mark.parametrize(
    "role",
    [
        False,
        "train",
        "probe_train",
        "inner_guard",
        "outer_selection",
        "final_test",
        "sealed_evaluation",
    ],
)
def test_should_refuse_non_controller_train_role_before_predictor(role):
    class Poison:
        def serve(self, request):
            pytest.fail("predictor entered")

    rows: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="controller_train"):
        serving._episodes(
            Poison(), "before", lambda: pytest.fail("source entered"), rows, role=role
        )
    assert rows == []


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_should_refuse_nonboolean_study_optin_before_source(flag, monkeypatch):
    monkeypatch.setattr(learning, "_source_bytes", lambda p: pytest.fail("source entered"))
    with pytest.raises(ValueError):
        learning.run_study("unused", task_outcomes=flag)


def test_should_execute_private_wait_tasks_without_classifier_guards_or_update(monkeypatch):
    state = initial()
    batches, _ = serving.prepare_data(DecisionFeatureSpec(serving.CONTEXT_FEATURES, CATALOG))
    monkeypatch.setattr(learning, "_correct", lambda *a: pytest.fail("classifier metric entered"))
    r = learning._trial(state, batches[0], object(), "wait", lambda: None, task_outcomes=True)
    assert r["complete"], r.get("error")
    assert r["before"] == r["after"] == [1, 4]
    assert r["training_work"] == dict(attempted_updates=0, returned_updates=0, attempted_rows=0)
    assert r["budget"]["updates"] == 0 and r["source_unchanged"] and r["actor_unchanged"]
    assert len(r["task_outcomes"]["before"]) == len(r["task_outcomes"]["after"]) == 6
    assert all(v["attempts"] == v["returns"] == 18 for v in r["task_readout_work"].values())
    measures = learning._task_measures(r)
    assert (measures.new_count, measures.old_count) == (2, 4)
    assert controller_utility(ControllerObjective(), measures).normalized_new_gain == 0
    assert state == initial()


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard"])
def test_should_refuse_unsupported_training_role_before_private_models(role, monkeypatch):
    state = initial()
    batches, _ = serving.prepare_data(DecisionFeatureSpec(serving.CONTEXT_FEATURES, CATALOG))
    monkeypatch.setattr(
        learning, "ContextPerceptronLearner", lambda *a: pytest.fail("learner entered")
    )
    with pytest.raises(ValueError, match="ordinary TRAIN"):
        learning._trial(
            state, replace(batches[0], role=role), (), "wait", lambda: None, task_outcomes=True
        )


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "task",
        "incomplete",
        "unknown",
        "version",
        "native",
        "readwork",
        "modelwork",
        "counts",
        "extra_phase",
        "extra_work",
        "cost",
    ],
)
def test_should_refuse_unknown_or_changed_task_evidence_before_utility(fault):
    state = initial()
    batches, _ = serving.prepare_data(DecisionFeatureSpec(serving.CONTEXT_FEATURES, CATALOG))
    r = scripted(state, batches[0], (), "wait", lambda: None, task_outcomes=True)
    row = r["task_outcomes"]["after"][0]
    if fault == "role":
        row["role"] = "final_test"
    elif fault == "task":
        row["concept"] = "legacy"
    elif fault == "incomplete":
        row["episode"]["complete"] = False
    elif fault == "unknown":
        row["episode"]["unreturned_prediction_work_unknown"] = True
    elif fault == "version":
        row["reads"][0]["actor_version"] = "deployed"
    elif fault == "native":
        row["reads"][0]["completed_updates"] = 13
    elif fault == "readwork":
        r["task_readout_work"]["after"]["returns"] = 0
    elif fault == "modelwork":
        r["training_work"]["attempted_updates"] = 1
    elif fault == "counts":
        r["after"][0] = 2
    elif fault == "extra_phase":
        r["task_outcomes"]["final_test"] = object()
    elif fault == "extra_work":
        r["task_readout_work"]["final_test"] = object()
    else:
        r["memory"]["peak_bytes"] = None
    with pytest.raises(ValueError):
        controller_utility(ControllerObjective(), learning._task_measures(r))


def test_should_retain_partial_before_episode_and_stop_before_optimizer(monkeypatch):
    state = initial()
    batches, guards = serving.prepare_data(DecisionFeatureSpec(serving.CONTEXT_FEATURES, CATALOG))

    def incomplete(predictor, phase, require, rows, *, role):
        rows.append(dict(role=role, complete=False, phase=phase))
        raise RuntimeError("lost development episode")

    monkeypatch.setattr(learning, "_episodes", incomplete)
    r = learning._trial(state, batches[0], guards, "wake1", lambda: None, task_outcomes=True)
    assert not r["complete"] and "lost development episode" in r["error"]
    assert r["task_outcomes"]["before"] == [
        dict(role="controller_train", complete=False, phase="before")
    ]
    assert r["training_work"]["attempted_updates"] == 0 and r["native_calls"] == []


def test_should_fit_only_after_complete_task_trials_without_any_live_application(monkeypatch):
    monkeypatch.setattr(learning, "_trial", scripted)
    monkeypatch.setattr(live, "run_case", lambda *a, **k: pytest.fail("application entered"))
    r = learning.run_study(SOURCE, task_outcomes=True)
    assert r["complete"], r.get("error")
    assert r["definition"] == "context_controller_task_trials_v1" and "application" not in r
    assert [(t["request"]["context"], t["request"]["action_id"]) for t in r["trials"]] == [
        ("mistakes", "wait"),
        ("mistakes", "wake1"),
        ("clean", "wait"),
        ("clean", "wake1"),
    ]
    assert r["fitted_policy"]["choices"] == dict(clean="wait", mistakes="wake1")
    assert r["training_state"]["completed"] == r["training_state"]["discarded_trials"] == 4
    assert all(
        t["request"]["started_at"] < t["labels_arrived_at"] <= t["completed_at"] <= r["fitted_at"]
        for t in r["trials"]
    )
    assert all(
        t["measures"]["new_count"] == 2 and t["measures"]["old_count"] == 4 for t in r["trials"]
    )
    assert r["final_examples"] == r["probe_updates"] == r["replay_examples"] == 0


def test_should_retain_malformed_paid_outcome_and_stop_before_next_trial(monkeypatch):
    calls = []

    def malformed(*a, **k):
        calls.append(1)
        r = scripted(*a, **k)
        r["task_outcomes"]["after"][0]["role"] = "final_test"
        return r

    monkeypatch.setattr(learning, "_trial", malformed)
    r = learning.run_study(SOURCE, task_outcomes=True)
    assert not r["complete"] and calls == [1] and len(r["trials"]) == 1
    assert r["training_state"]["stopped"] and r["training_state"]["completed"] == 0
    assert r["trials"][0]["result"]["task_outcomes"]["after"][0]["role"] == "final_test"


def test_should_restore_equal_replaced_preceding_task_evidence_and_stop(monkeypatch):
    retained = []
    original = learning.paid_report_guard

    def seal(row):
        retained.append(row)
        return original(row)

    calls = []

    def corrupt(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            row = retained[0]["result"]
            row["task_outcomes"] = deepcopy(row["task_outcomes"])
        return scripted(*a, **k)

    monkeypatch.setattr(learning, "paid_report_guard", seal)
    monkeypatch.setattr(learning, "_trial", corrupt)
    r = learning.run_study(SOURCE, task_outcomes=True)
    assert not r["complete"] and len(calls) == 2 and "corrupted" in r["error"]
    assert len(r["trials"][0]["result"]["task_outcomes"]["after"]) == 6


def test_should_dispatch_task_cli_without_entering_classifier_mode(tmp_path, monkeypatch):
    seen: list[tuple[Path, dict[str, Any]]] = []

    def study(path, **kwargs):
        seen.append((path, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(learning, "run_study", study)
    output = tmp_path / "task.json"
    assert (
        cli.main(
            ["--source", str(SOURCE), "--context-controller-task-trials", "--output", str(output)]
        )
        == 0
    )
    assert seen == [(SOURCE, dict(task_outcomes=True))]
