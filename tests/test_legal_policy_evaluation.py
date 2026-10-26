"""Explicit greedy legal conditioning with real toy dynamics and fake policies."""

from copy import deepcopy
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import key_door_pilot as shared
from src.app import coupled_attribution as attribution
from src.app import key_door_policy_evaluation as evaluation
from src.app.line_policy_evaluation import EvaluationIdentity
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation
from src.core.legal_policy_gradient import condition_action_probabilities
from test_return_evaluation_boundaries import Learner, row, state, fake_episode


RAW = (0.1, 0.15, 0.2, 0.25, 0.3)


def identity():
    return EvaluationIdentity(
        "recurrent_bc", 17, "fixed-saved-checkpoint", ("A", "B", "A"), 8, False
    )


def run(predict=None, **options):
    if predict is None:
        predict = lambda features: np.array([RAW], dtype=np.float64)
    return evaluation.run_key_door_episode(
        predict, identity=identity(), context="A", layout=KeyDoorLayout("A"), cue=1, **options
    )


def test_should_condition_actual_legal_support_before_one_greedy_decision_and_preserve_raw():
    before = pickle.dumps(np.random.get_state())
    report = run(max_steps=3, condition_on_legal_actions=True)
    assert report["action_conditioning"] == "legal_actions_v1"
    assert report["status"] == "complete" and report["outcome"] == "step_limit"
    assert report["success"] is False and report["return"] == 0
    environment = KeyDoorNavigation(KeyDoorLayout("A"), 1)
    for slot in report["transitions"]:
        legal = environment.legal_actions
        expected = condition_action_probabilities(RAW, legal)
        action = max(range(5), key=expected.__getitem__)
        assert slot["raw_probabilities"] == RAW
        assert slot["probabilities"] == expected
        assert slot["legal_actions"] == legal and slot["observation"] == environment.observation
        assert slot["action"] == action and action in legal and slot["applied"] is True
        result = environment.step(action)
        assert slot["next_observation"] == result.observation
    assert report["transitions"][0]["action"] == 0
    assert int(np.argmax(RAW)) == 4 and 4 not in report["transitions"][0]["legal_actions"]
    assert (
        report["steps_executed"]
        == report["prediction_calls_attempted"]
        == report["prediction_calls_returned"]
        == 3
    )
    assert pickle.dumps(np.random.get_state()) == before


def test_should_choose_first_conditional_argmax_tie_without_randomness_or_retry():
    raw = (0.2, 0.4, 0.2, 0.1, 0.1)
    calls = []

    def predict(features):
        calls.append(features.copy())
        return np.array([raw], dtype=np.float64)

    report = run(predict, max_steps=2, condition_on_legal_actions=True)
    second = report["transitions"][1]
    assert second["legal_actions"] == (0, 2)
    assert second["probabilities"] == (0.5, 0.0, 0.5, 0.0, 0.0)
    assert second["action"] == 0 and len(calls) == 2


def test_should_preserve_exact_default_report_and_raw_illegal_argmax_when_disabled():
    implicit = run(max_steps=3)
    explicit = run(max_steps=3, condition_on_legal_actions=False)
    assert pickle.dumps(implicit) == pickle.dumps(explicit)
    assert implicit["outcome"] == "illegal_action" and implicit["steps_executed"] == 0
    assert implicit["transitions"][0]["action"] == 4
    assert implicit["transitions"][0]["probabilities"] == RAW
    assert "action_conditioning" not in implicit
    assert "raw_probabilities" not in implicit["transitions"][0]


@pytest.mark.parametrize("mode", [0, 1, None, "true", np.bool_(True)])
@pytest.mark.parametrize("endpoint", ["episode", "live"])
def test_should_refuse_nonbool_mode_before_environment_source_or_prediction_work(
    monkeypatch, mode, endpoint
):
    def forbidden(*args, **kwargs):
        raise AssertionError("external work reached")

    monkeypatch.setattr(evaluation, "KeyDoorNavigation", forbidden)
    blocked: Any = object()
    with pytest.raises(ValueError, match="exact bool"):
        if endpoint == "episode":
            run(forbidden, condition_on_legal_actions=mode)
        else:
            shared._live({}, blocked, [], blocked, condition_on_legal_actions=mode)


def test_should_keep_zero_mass_raw_returned_count_and_unapplied_transition_on_refusal(monkeypatch):
    def forbidden(*args):
        raise AssertionError("no physical step or fallback permitted")

    monkeypatch.setattr(KeyDoorNavigation, "step", forbidden)
    raw = (0.0, 1.0, 0.0, 0.0, 0.0)
    with pytest.raises(evaluation.KeyDoorEvaluationFailed) as caught:
        run(lambda _: np.array([raw]), condition_on_legal_actions=True)
    report = caught.value.report
    assert report["status"] == "failed" and "positive finite probability mass" in report["reason"]
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert report["steps_executed"] == report["return"] == 0
    slot = report["transitions"][0]
    assert slot["raw_probabilities"] == raw and slot["probabilities"] is None
    assert slot["action"] is None and slot["applied"] is False
    assert slot["observation"] == slot["next_observation"] and slot["observation"][6] == 1


@pytest.mark.parametrize("fault", ["before", "predict", "invalid_raw", "physics"])
def test_should_preserve_attempted_returned_and_physical_work_for_each_failure_boundary(
    monkeypatch, fault
):
    def fail(*args):
        raise RuntimeError("injected fault")

    predict = fail if fault == "predict" else lambda _: np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    if fault == "invalid_raw":
        predict = lambda _: np.array([[np.nan, 0, 0, 0, 0]])
    if fault == "physics":
        monkeypatch.setattr(KeyDoorNavigation, "step", fail)
    with pytest.raises(evaluation.KeyDoorEvaluationFailed) as caught:
        run(
            predict,
            before_prediction=fail if fault == "before" else None,
            condition_on_legal_actions=True,
        )
    report = caught.value.report
    assert report["status"] == "failed" and report["steps_executed"] == 0
    assert report["prediction_calls_attempted"] == (0 if fault == "before" else 1)
    assert report["prediction_calls_returned"] == (1 if fault in ("invalid_raw", "physics") else 0)
    if fault == "physics":
        slot = report["transitions"][0]
        assert slot["action"] == 0 and slot["probabilities"] == slot["raw_probabilities"] == (
            1,
            0,
            0,
            0,
            0,
        )
        assert slot["applied"] is False
    elif fault != "before":
        assert report["transitions"][0]["raw_probabilities"] is None


def test_should_own_recorded_raw_values_after_predictor_mutates_its_returned_array():
    raw = np.array([RAW])
    report = run(lambda _: raw, max_steps=1, condition_on_legal_actions=True)
    before = deepcopy(report)
    raw.fill(0)
    assert report == before


@pytest.mark.parametrize("selected", ["omitted", False, True])
def test_should_forward_only_requested_live_flag_and_preserve_old_evaluator_kwargs(
    monkeypatch, selected
):
    source, report, seen = Learner(), row(), []
    monkeypatch.setattr(shared, "_same_state", attribution._same_value)

    def evaluate(*args, **kwargs):
        seen.append(kwargs.copy())
        result = fake_episode(*args, **kwargs)
        if kwargs.get("condition_on_legal_actions") is True:
            result["action_conditioning"] = "legal_actions_v1"
        return result

    monkeypatch.setattr(shared, "run_key_door_episode", evaluate)
    options = {} if selected == "omitted" else {"condition_on_legal_actions": selected}
    shared._live(
        report, source, [(192, state())], budget(), task_sequence=("A", "B", "A"), **options
    )
    assert len(seen) == 4 and report["live_sessions_started"] == 4
    if selected is True:
        assert report["action_conditioning"] == "legal_actions_v1"
        assert all(kwargs["condition_on_legal_actions"] is True for kwargs in seen)
        assert all(slot["action_conditioning"] == "legal_actions_v1" for slot in report["episodes"])
    else:
        assert "action_conditioning" not in report
        assert all("condition_on_legal_actions" not in kwargs for kwargs in seen)
        assert all("action_conditioning" not in slot for slot in report["episodes"])
    assert all(slot["state_unchanged"] is True for slot in report["episodes"])


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=8), lambda: 0.0)


def test_should_retain_actual_partial_failure_in_shared_live_without_admitting_next_episode(
    monkeypatch,
):
    source, report = Learner(), row()
    monkeypatch.setattr(shared, "_same_state", attribution._same_value)
    monkeypatch.setattr(Learner, "predict", lambda *args: np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(evaluation.KeyDoorEvaluationFailed):
        shared._live(
            report,
            source,
            [(192, state())],
            budget(),
            task_sequence=("A", "B", "A"),
            condition_on_legal_actions=True,
        )
    first = report["episodes"][0]
    assert report["action_conditioning"] == first["action_conditioning"] == "legal_actions_v1"
    assert first["status"] == "failed" and first["prediction_calls_returned"] == 1
    assert first["transitions"][0]["raw_probabilities"] == (0.0, 1.0, 0.0, 0.0, 0.0)
    assert first["transitions"][0]["probabilities"] is None
    assert report["live_sessions_started"] == 1
    assert report["episodes"][1]["status"] == "not_run"
