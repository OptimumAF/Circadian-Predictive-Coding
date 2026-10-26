"""Fixed acquired-return metric arithmetic and bounded live adapter boundaries."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.adapters import acquired_task_return as acquired
from src.adapters import key_door_pilot as shared
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import _same_value
from src.app.key_door_policy_evaluation import KeyDoorEvaluationFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.context_return_metrics import summarize_acquired_return
from test_train_policy_diagnostics import Learner, state


def score(ce, accuracy, sequence):
    return dict(ce=ce, accuracy=accuracy, exact_sequence_accuracy=sequence)


def points():
    return (
        dict(updates=192, A=score(0.1, 1.0, 1.0), C_train=score(2.0, 0.2, 0.0)),
        dict(updates=256, A=score(0.8, 0.4, 0.25), C_train=score(0.5, 0.8, 0.5)),
        dict(updates=320, A=score(0.2, 0.9, 0.75), C_train=score(0.9, 0.5, 0.25)),
    )


def test_should_preserve_fixed_signed_raw_metrics_and_detached_endpoints():
    original = points()
    before = deepcopy(original)
    summary = summarize_acquired_return(original)
    assert summary["metrics"]["ce"] == pytest.approx(
        dict(
            new_C_gain=1.5,
            old_A_loss=0.7,
            return_A_gain=0.6,
            return_gap=0.1,
            new_C_loss_after_return=0.4,
        )
    )
    assert summary["metrics"]["accuracy"] == pytest.approx(
        dict(
            new_C_gain=0.6,
            old_A_loss=0.6,
            return_A_gain=0.5,
            return_gap=0.1,
            new_C_loss_after_return=0.3,
        )
    )
    assert summary["metrics"]["exact_sequence_accuracy"] == pytest.approx(
        dict(
            new_C_gain=0.5,
            old_A_loss=0.75,
            return_A_gain=0.5,
            return_gap=0.25,
            new_C_loss_after_return=0.25,
        )
    )
    assert (
        summary["incomplete_initial_A_fit"] is False
        and summary["useful_skill_established"] is False
    )
    assert original == before
    original[0]["A"]["ce"] = 7
    assert summary["endpoint_values"][0]["A"]["ce"] == 0.1
    summary["endpoint_values"][1]["C_train"]["ce"] = 8
    assert original[1]["C_train"]["ce"] == 0.5


def test_should_keep_negative_results_and_incomplete_initial_fit_without_clipping():
    original = points()
    original[0]["A"]["exact_sequence_accuracy"] = 0.5
    original[1]["C_train"]["ce"] = 3.0
    original[2]["A"]["ce"] = 1.2
    original[1]["A"]["accuracy"] = 1.0
    original[0]["A"]["accuracy"] = 0.5
    summary = summarize_acquired_return(original)
    assert summary["metrics"]["ce"]["new_C_gain"] == -1.0
    assert summary["metrics"]["ce"]["return_A_gain"] == pytest.approx(-0.4)
    assert summary["metrics"]["accuracy"]["old_A_loss"] == -0.5
    assert summary["incomplete_initial_A_fit"] is True
    assert "inconclusive" in summary["interpretation"]


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "length",
        "extra_context",
        "missing_context",
        "update",
        "bool_update",
        "score_extra",
        "score_missing",
        "nan",
        "infinity",
        "negative_ce",
        "accuracy",
        "sequence",
        "bool_score",
        "numpy_score",
        "overflow",
    ],
)
def test_should_reject_noncanonical_raw_endpoints_without_changing_input(fault):
    original: Any = points()
    if fault == "list":
        original = list(original)
    elif fault == "length":
        original = original[:2]
    elif fault == "extra_context":
        original[0]["B"] = score(0, 0, 0)
    elif fault == "missing_context":
        del original[0]["C_train"]
    elif fault == "update":
        original[0]["updates"] = 191
    elif fault == "bool_update":
        original[0]["updates"] = True
    elif fault == "score_extra":
        original[1]["A"]["selected"] = True
    elif fault == "score_missing":
        del original[1]["A"]["accuracy"]
    elif fault == "nan":
        original[0]["A"]["ce"] = float("nan")
    elif fault == "infinity":
        original[0]["A"]["ce"] = float("inf")
    elif fault == "negative_ce":
        original[0]["A"]["ce"] = -0.1
    elif fault == "accuracy":
        original[0]["A"]["accuracy"] = 1.1
    elif fault == "sequence":
        original[0]["A"]["exact_sequence_accuracy"] = -0.1
    elif fault == "bool_score":
        original[0]["A"]["ce"] = False
    elif fault == "numpy_score":
        original[0]["A"]["ce"] = np.float64(0.1)
    else:
        original[0]["A"]["ce"] = 10**1000
    before = deepcopy(original)
    with pytest.raises(ValueError):
        summarize_acquired_return(original)
    assert repr(original) == repr(before)


@pytest.fixture(autouse=True)
def fake_state_comparison(monkeypatch):
    monkeypatch.setattr(acquired, "_same_state", _same_value)
    monkeypatch.setattr(shared, "_same_state", _same_value)


def budget(maximum=320, completed=320):
    session = ToyBudgetSession(ToyExecutionBudget(max_training_updates=maximum), lambda: 0.0)
    session.updates_completed = completed
    return session


def states():
    return [(point, state(point)) for point in (192, 256, 320)]


def test_should_read_all_twelve_train_slots_without_prior_rollouts_or_native_updates(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no data collection or prior permitted")

    monkeypatch.setattr(shared, "_collect", forbidden)
    report, source, checkpoints, session = acquired.new_diagnostic(), Learner(), states(), budget()
    before = deepcopy((source.state, checkpoints))
    acquired.evaluate_acquired_tasks(
        report, source, checkpoints, session, check_boundary=lambda: None
    )
    assert report["status"] == "complete" and session.updates_completed == 320
    assert _same_value((source.state, checkpoints), before)
    assert source.trace.count("fork") == 3 and source.trace.count("session") == 12
    assert source.trace.count("predict") == 144
    assert report["work"]["live_callback_returns"] == report["work"]["live_applied_steps"] == 144
    assert report["initial_A_acquired"] is False and "inconclusive" in report["interpretation"]
    assert len(report["outcomes"]) == 3 and len(report["episodes"]) == 12
    assert all(
        slot["status"] == "complete"
        and slot["source_role"] == "train"
        and slot["state_unchanged"] is True
        for slot in report["episodes"]
    )
    assert all(
        slot["identity"]["task_sequence"] == ("A", "B", "A", "C_train", "A")
        for slot in report["episodes"]
    )


def test_should_keep_all_live_differences_with_fixed_gain_and_loss_directions(monkeypatch):
    values = {
        192: {"A": (1, 1), "C_train": (0, 0)},
        256: {"A": (1, 0), "C_train": (1, 0)},
        320: {"A": (1, 1), "C_train": (-1, 0)},
    }

    def episode(predict, **options):
        value = values[options["identity"].checkpoint_updates][options["context"]][
            0 if options["cue"] == 1 else 1
        ]
        return dict(
            status="complete",
            success=value == 1,
            **{"return": value},
            prediction_calls_attempted=0,
            prediction_calls_returned=0,
            steps_executed=0,
        )

    monkeypatch.setattr(shared, "run_key_door_episode", episode)
    report = acquired.new_diagnostic()
    acquired.evaluate_acquired_tasks(
        report, Learner(), states(), budget(), check_boundary=lambda: None
    )
    assert report["initial_A_acquired"] is True
    assert report["metrics"]["success_rate"] == dict(
        new_C_gain=0.5,
        old_A_loss=0.5,
        return_A_gain=0.5,
        return_gap=0.0,
        new_C_loss_after_return=0.5,
    )
    assert report["metrics"]["mean_return"] == dict(
        new_C_gain=0.5,
        old_A_loss=0.5,
        return_A_gain=0.5,
        return_gap=0.0,
        new_C_loss_after_return=1.0,
    )


def test_should_publish_observed_acquisition_and_metrics_after_final_callback_mutation():
    report = acquired.new_diagnostic()
    attempts = []

    def forge_derived_claims():
        if all(slot["state_unchanged"] is True for slot in report["episodes"]):
            attempts.append(report["status"])
            report.update(
                initial_A_acquired=True,
                outcomes=[{"forged": True}],
                metrics={"forged": True},
                interpretation="forged_acquisition",
            )

    acquired.evaluate_acquired_tasks(
        report, Learner(), states(), budget(), check_boundary=forge_derived_claims
    )
    assert attempts and all(status == "running" for status in attempts)
    assert report["status"] == "complete" and report["initial_A_acquired"] is False
    assert report["interpretation"] == "inconclusive_initial_A_not_acquired"
    assert [point["updates"] for point in report["outcomes"]] == [192, 256, 320]
    assert set(report["metrics"]) == {"success_rate", "mean_return"}
    assert all(
        value == 0 for differences in report["metrics"].values() for value in differences.values()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "container",
        "point",
        "bool_point",
        "rows",
        "native_count",
        "policy",
        "budget_max",
        "budget_committed",
        "report",
        "bool_cue",
        "callback",
    ],
)
def test_should_refuse_mismatched_plan_or_native_counters_before_source_work(fault):
    checkpoints: Any = states()
    report, source, session = acquired.new_diagnostic(), Learner(), budget()
    callback: Any = lambda: None
    if fault == "container":
        checkpoints = tuple(checkpoints)
    elif fault == "point":
        checkpoints[0] = (128, state(128))
    elif fault == "bool_point":
        checkpoints[0] = (True, state(192))
    elif fault == "rows":
        checkpoints[-1][1].training_examples = 2559
    elif fault == "native_count":
        checkpoints[-1][1].completed_updates = 319
    elif fault == "policy":
        checkpoints[-1][1].policy = TorchRecurrentPolicy(8, 9, 5, 17)
    elif fault == "budget_max":
        session = budget(321, 320)
    elif fault == "budget_committed":
        session = budget(320, 319)
    elif fault == "report":
        report["episodes"].pop()
    elif fault == "bool_cue":
        report["episodes"][0]["cue"] = True
    else:
        callback = None
    with pytest.raises(ValueError):
        acquired.evaluate_acquired_tasks(
            report, source, checkpoints, session, check_boundary=callback
        )
    assert source.trace == []


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "state",
        "state_replacement",
        "future_slot",
        "slot_replacement",
        "slot_container",
        "clock",
        "charges",
        "preceding",
    ],
)
def test_should_stop_on_replaced_or_corrupted_evidence_before_admitting_more_work(fault):
    report, checkpoints, session = acquired.new_diagnostic(), states(), budget()
    source = Learner()
    changed = False

    def corrupt(event, actor):
        nonlocal changed
        if event == "predict" and not changed:
            changed = True
            if fault == "source":
                source.state.parameters["weight"] = 2.0
            elif fault == "state":
                checkpoints[-1][1].training_examples = 2559
            elif fault == "state_replacement":
                checkpoints[-1] = deepcopy(checkpoints[-1])
            elif fault == "future_slot":
                report["episodes"][-1]["cue"] = 1
            elif fault == "slot_replacement":
                report["episodes"][-1] = deepcopy(report["episodes"][-1])
            elif fault == "slot_container":
                report["episodes"] = list(report["episodes"])
            elif fault == "clock":
                session.clock = lambda: 0.0
            elif fault == "charges":
                session.updates_completed = 319

    source.hook = corrupt

    def guard():
        if fault == "preceding" and report["episodes"][0]["state_unchanged"] is True:
            report["episodes"][0]["return"] = 7.0

    expected = ValueError if fault == "preceding" else KeyDoorEvaluationFailed
    with pytest.raises(expected, match="changed"):
        acquired.evaluate_acquired_tasks(report, source, checkpoints, session, check_boundary=guard)
    assert report["status"] == "failed" and source.trace.count("session") < 12
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert report["metrics"] is None
