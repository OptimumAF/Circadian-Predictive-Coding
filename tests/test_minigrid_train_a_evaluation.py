"""Known TRAIN A plan routing through the real target-free evaluator.

Native state admission is an explicit fake port here; the previously accepted
full native snapshot validator is reused as evidence. Predictions and two-step
engineering environments perform no neural work or physics.
"""

from copy import deepcopy
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_invariant_evaluation import DiscreteEnvironment
from test_minigrid_learning_pilot import Environment, Learner, same_state
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    monkeypatch.setattr(Learner, "work", work)
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(Learner, "train_batch", forbidden)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", DiscreteEnvironment)
    monkeypatch.setattr(evaluation, "copy_invariant_torch_feature_snapshot", deepcopy)
    monkeypatch.setattr(evaluation, "_same_state", same_state)

    def build(method="bp"):
        source = Learner(
            InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))
        )
        state = source.snapshot_state()
        state.completed_updates, state.training_examples = 128, 1152
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=0, max_replay_examples=0), lambda: 0.0
        )
        return SimpleNamespace(source=source, state=state, budget=budget)

    return SimpleNamespace(work=work, build=build)


def readout(parts, **options):
    return evaluation.evaluate_neural_representation_checkpoint(
        parts.source,
        parts.state,
        updates=128,
        budget=parts.budget,
        check_boundary=options.pop("check_boundary", lambda: None),
        **options,
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonboolean_plan_selection_before_snapshot_or_environment_ports(
    monkeypatch, value
):
    monkeypatch.setattr(evaluation, "copy_invariant_torch_feature_snapshot", forbidden)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", forbidden)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    with pytest.raises(ValueError, match="known_a_only"):
        evaluation.evaluate_neural_representation_checkpoint(
            None, None, updates=128, budget=budget, check_boundary=forbidden, known_a_only=value
        )


@pytest.mark.parametrize("method", METHODS)
def test_should_read_only_two_train_cues_with_raw_argmax_and_fresh_public_history(
    case, monkeypatch, method
):
    monkeypatch.setattr(evaluation, "predefined_probe_layouts", forbidden)
    parts = case.build(method)
    before = pickle.dumps((parts.source.snapshot_state(), parts.state))
    report = readout(parts, known_a_only=True)
    assert report["status"] == "complete" and report["role"] == "knownTRAIN_A_only"
    assert report["method"] == f"public_invariant_{method}"
    assert report["updates"] == 128 and report["native_training_examples"] == 1152
    assert report["scores"] == [] and report["state_unchanged"] is True
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
    ]
    assert report["work"]["raw_score_forwards_completed"] == 0
    assert (
        report["work"]["live_prediction_calls_returned"]
        == report["work"]["environment_steps_returned"]
        == 4
    )
    assert report["work"]["environment_resets_completed"] == 2
    assert report["work"]["internal_projection_rows_returned"] == 4
    assert report["work"]["internal_projection_array_bytes_returned"] == 4 * 2368
    assert report["work"]["history_episodes_started"] == 2
    for slot in report["episodes"]:
        assert slot["status"] == "complete"
        initial = tuple(slot["predictions"][0]["observation"])
        assert len(slot["predictions"]) == 2
        for decision in slot["predictions"]:
            assert tuple(decision["features"]) == initial + tuple(decision["observation"])
            assert decision["probabilities"] == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
            assert decision["action"] == 0
    assert len(case.work["forks"]) == 1 and len(case.work["environments"]) == 2
    assert all(
        environment.role == "train" and environment.layout.context == "A"
        for environment in case.work["environments"]
    )
    assert all(environment.closed for environment in case.work["environments"])
    assert parts.budget.updates_completed == parts.budget.replay_examples_completed == 0
    assert pickle.dumps((parts.source.snapshot_state(), parts.state)) == before


def test_should_keep_the_default_a_e_report_identical_when_false_is_explicit(case):
    first = readout(case.build())
    second = readout(case.build(), known_a_only=False)
    assert first == second
    assert [slot["context"] for slot in first["episodes"]] == ["A", "A", "E-blue", "E-red"]


def test_should_validate_preceding_training_evidence_before_any_fork_or_reset(case, monkeypatch):
    parts = case.build()
    monkeypatch.setattr(parts.source, "fork", forbidden)

    def changed():
        raise ValueError("preceding TRAIN evidence changed")

    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="preceding TRAIN") as caught:
        readout(parts, known_a_only=True, check_boundary=changed)
    report = caught.value.report
    assert len(report["episodes"]) == 2 and all(
        slot["status"] == "not_run" for slot in report["episodes"]
    )
    assert report["work"]["live_prediction_calls_attempted"] == 0
    assert not case.work.get("environments")


@pytest.mark.parametrize(
    "fault,predictions,attempts,steps",
    [
        ("prediction", 0, 0, 0),
        ("step", 1, 1, 0),
        ("cleanup", 2, 2, 2),
        ("scorer_state", 1, 0, 0),
        ("budget_clock", 1, 0, 0),
    ],
)
def test_should_keep_paid_partial_work_and_the_unreached_second_cue(
    case, fault, predictions, attempts, steps
):
    parts = case.build()
    before = pickle.dumps((parts.source.snapshot_state(), parts.state))
    if fault == "prediction":
        case.work["prediction_error"] = True
    elif fault == "step":
        case.work["step_error"] = True
    elif fault == "cleanup":
        case.work["close_error"] = True
    else:

        def corrupt(owner):
            if fault == "scorer_state":
                owner.state.parameters["weight"][0, 0] += 0.25
            else:
                parts.budget.clock = lambda: 0.0

        case.work["on_predict"] = corrupt
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        readout(parts, known_a_only=True)
    report = caught.value.report
    assert report["status"] == "failed" and len(report["episodes"]) == 2
    assert (
        report["episodes"][0]["status"] == "failed" and report["episodes"][1]["status"] == "not_run"
    )
    assert report["work"]["live_prediction_calls_returned"] == predictions
    assert report["work"]["environment_steps_attempted"] == attempts
    assert report["work"]["environment_steps_returned"] == steps
    assert parts.budget.updates_completed == parts.budget.replay_examples_completed == 0
    assert pickle.dumps((parts.source.snapshot_state(), parts.state)) == before
