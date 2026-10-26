"""Bounded explicit64/128 diagnostic endpoints with actual toy dynamics."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.adapters import key_door_pilot as shared
from src.adapters import train_policy_diagnostics as diagnostic
from src.app.coupled_attribution import _same_value
from src.app.key_door_policy_evaluation import KeyDoorEvaluationFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_train_policy_diagnostics import Learner, reference, state


@pytest.fixture(autouse=True)
def fake_state_comparison(monkeypatch):
    monkeypatch.setattr(diagnostic, "_same_state", _same_value)
    monkeypatch.setattr(shared, "_same_state", _same_value)


def budget(maximum=128, completed=128):
    session = ToyBudgetSession(ToyExecutionBudget(max_training_updates=maximum), lambda: 0.0)
    session.updates_completed = completed
    return session


def execute(*, endpoint=128, prior=None):
    report = diagnostic.new_diagnostic(endpoint=endpoint)
    source, states = Learner(), [(0, state(0)), (endpoint, state(endpoint))]
    prior = reference() if prior is None else prior
    before = deepcopy((source.state, states, prior))
    session = budget(endpoint, endpoint)
    diagnostic.evaluate_train_policy(
        report, source, states, prior, session, check_boundary=lambda: None, endpoint=endpoint
    )
    assert _same_value((source.state, states, prior), before)
    assert session.updates_completed == endpoint
    return report, source


def test_should_preserve_default64_schema_behavior_and_error_text_when_explicit():
    assert diagnostic.new_diagnostic() == diagnostic.new_diagnostic(endpoint=64)
    reports = []
    for options in ({}, {"endpoint": 64}):
        report = diagnostic.new_diagnostic(**options)
        diagnostic.evaluate_train_policy(
            report,
            Learner(),
            [(0, state(0)), (64, state(64))],
            reference(),
            budget(64, 64),
            check_boundary=lambda: None,
            **options,
        )
        reports.append(report)
    assert reports[0] == reports[1]
    errors = []
    for options in ({}, {"endpoint": 64}):
        with pytest.raises(ValueError) as caught:
            diagnostic.evaluate_train_policy(
                diagnostic.new_diagnostic(**options),
                Learner(),
                [(0, state(0)), (128, state(128))],
                reference(),
                budget(64, 64),
                check_boundary=lambda: None,
                **options,
            )
        errors.append(str(caught.value))
    assert errors == ["train diagnostic requires exact owned checkpoints0/64"] * 2


def test_should_run_only_four_owned128_endpoint_episodes_and_retain_both_prior_refusals():
    report, source = execute()
    assert report["status"] == "complete"
    assert [(slot["checkpoint_updates"], slot["cue"]) for slot in report["episodes"]] == [
        (0, 1),
        (0, -1),
        (128, 1),
        (128, -1),
    ]
    assert [point["updates"] for point in report["outcomes"]] == [0, 128]
    assert report["learner_gains"] == {"success_rate": 0.0, "mean_return": 0.0}
    assert source.trace.count("fork") == 2 and source.trace.count("session") == 4
    assert report["work"]["learner_callback_returns"] == 48
    assert report["work"]["prior_callback_returns"] == 8
    assert report["work"]["prior_applied_steps"] == 6
    assert report["comparison"] is None and report["reference"]["outcomes"] is None
    for slot in report["reference"]["episodes"]:
        assert slot["status"] == "refused" and slot["refusal"] == "zero_legal_mass"
        assert slot["success"] is None and slot["transitions"][-1]["applied"] is False
    assert all(slot["state_unchanged"] is True for slot in report["episodes"])


def test_should_publish128_comparison_only_when_both_prior_episodes_complete():
    report, _ = execute(prior=reference([0.2] * 5))
    assert report["comparison"] == [
        dict(updates=point, success_rate=0.0, mean_return=0.0) for point in (0, 128)
    ]
    assert report["comparison_reason"] is None
    assert report["reference"]["outcomes"] == {"success_rate": 0.0, "mean_return": 0.0}
    assert report["work"]["prior_callback_returns"] == 24


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "endpoint",
    [True, False, 64.0, 128.0, None, "128", 0, 63, 127, 129, np.int64(128), IntSubclass(128)],
)
@pytest.mark.parametrize("entry", ["allocate", "evaluate"])
def test_should_refuse_invalid_endpoint_before_callback_source_or_budget_work(endpoint, entry):
    source, checks = Learner(), []
    if entry == "allocate":
        with pytest.raises(ValueError, match="endpoint must be an exact native integer64 or128"):
            diagnostic.new_diagnostic(endpoint=endpoint)
    else:
        invalid_budget: Any = object()
        with pytest.raises(ValueError, match="endpoint must be an exact native integer64 or128"):
            diagnostic.evaluate_train_policy(
                {},
                source,
                [],
                {},
                invalid_budget,
                check_boundary=lambda: checks.append(1),
                endpoint=endpoint,
            )
    assert source.trace == [] and checks == []


@pytest.mark.parametrize("maximum,completed", [(64, 128), (128, 64), (192, 128)])
def test_should_require_the_original_completed128_update_budget_before_source_work(
    maximum, completed
):
    source, session = Learner(), budget(maximum, completed)
    with pytest.raises(ValueError, match="original128update budget"):
        diagnostic.evaluate_train_policy(
            diagnostic.new_diagnostic(endpoint=128),
            source,
            [(0, state(0)), (128, state(128))],
            reference(),
            session,
            check_boundary=lambda: None,
            endpoint=128,
        )
    assert source.trace == [] and session.updates_completed == completed


@pytest.mark.parametrize("fault", ["point", "rows", "native_count", "report"])
def test_should_refuse_mismatched128_checkpoint_evidence_before_any_native_work(fault):
    source, states, report = (
        Learner(),
        [(0, state(0)), (128, state(128))],
        diagnostic.new_diagnostic(endpoint=128),
    )
    if fault == "point":
        states[-1] = (64, state(64))
    elif fault == "rows":
        states[-1][1].training_examples = 1023
    elif fault == "native_count":
        states[-1][1].completed_updates = 127
    else:
        report = diagnostic.new_diagnostic()
    with pytest.raises(ValueError):
        diagnostic.evaluate_train_policy(
            report, source, states, reference(), budget(), check_boundary=lambda: None, endpoint=128
        )
    assert source.trace == []


@pytest.mark.parametrize("fault", ["future_point", "slot_replacement", "counter", "prior"])
def test_should_stop128_readout_on_slot_or_checkpoint_mutation_without_spending_more_updates(fault):
    report, states, session = (
        diagnostic.new_diagnostic(endpoint=128),
        [(0, state(0)), (128, state(128))],
        budget(),
    )
    prior = reference()
    changed = False

    def corrupt(event, actor):
        nonlocal changed
        if event == "predict" and not changed:
            changed = True
            if fault == "future_point":
                report["episodes"][-1]["checkpoint_updates"] = 64
            elif fault == "slot_replacement":
                report["episodes"][-1] = deepcopy(report["episodes"][-1])
            elif fault == "counter":
                states[-1][1].completed_updates = 129
            else:
                prior["probabilities"][0] = 0.4

    source = Learner(hook=corrupt)
    with pytest.raises(KeyDoorEvaluationFailed, match="changed") as caught:
        diagnostic.evaluate_train_policy(
            report, source, states, prior, session, check_boundary=lambda: None, endpoint=128
        )
    assert type(caught.value.__cause__) is ValueError
    assert report["status"] == "failed" and session.updates_completed == 128
    assert source.trace.count("session") == 1 and source.trace.count("predict") == 1
    assert all(slot["status"] == "not_run" for slot in report["reference"]["episodes"])
    assert report["comparison"] is None
