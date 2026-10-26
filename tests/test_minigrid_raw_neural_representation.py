"""Matched raw-neural admission and actual frozen forwards on fake game frames."""

from copy import deepcopy
from time import monotonic
from typing import Any

import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_evaluation import DiscreteEnvironment, Environment
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.fixture
def case(monkeypatch):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", DiscreteEnvironment)
    monkeypatch.setattr(Environment, "work", {})
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", forbidden)


def policy(method="pc", **options):
    values = dict(method=method, feature_dim=296, hidden_dim=8, seed=17, num_classes=7)
    values.update(options)
    return TorchFeaturePolicy(**values)


def evaluate(source, state=None, *, updates=0, budget=None, guard=lambda: None):
    if state is None:
        state = source.snapshot_state()
    return evaluation.evaluate_raw_neural_representation_checkpoint(
        source,
        state,
        updates=updates,
        budget=budget or ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), monotonic),
        check_boundary=guard,
    )


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_apply_matched_games_without_projecting_or_changing_raw_head(case, method):
    source = TorchFeatureLearner(policy(method))
    state = source.snapshot_state()
    original = deepcopy(state)
    report = evaluate(source, state)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == report["native_method"] == method
    assert report["model_encoding"] == "original_raw_history" and report["scores"] == []
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
        ("E-blue", "blue", "probe_test"),
        ("E-red", "red", "probe_test"),
    ]
    assert report["work"]["internal_projection_rows_returned"] == 0
    assert report["work"]["internal_projection_array_bytes_returned"] == 0
    assert report["work"]["live_prediction_calls_returned"] == 8
    for slot in report["episodes"]:
        initial = slot["predictions"][0]["observation"]
        for decision in slot["predictions"]:
            assert decision["features"] == initial + decision["observation"]
    assert _same_state(state, original) and _same_state(source.snapshot_state(), original)


@pytest.mark.parametrize(
    "fault", ["width", "hidden", "seed", "rate", "budget", "clock", "bound", "label"]
)
def test_should_refuse_unmatched_inputs_before_fork_or_physics(case, monkeypatch, fault):
    choices: dict[str, dict[str, Any]] = {
        "width": dict(feature_dim=148),
        "hidden": dict(hidden_dim=4),
        "seed": dict(seed=18),
        "rate": dict(learning_rate=0.04),
    }
    source = TorchFeatureLearner(policy(**choices.get(fault, {})))
    state: Any = source.snapshot_state()
    point = 0
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=int(fault == "budget")), monotonic
    )
    if fault == "clock":
        object.__setattr__(state, "training_examples", 1)
    elif fault == "bound":
        state = InvariantTorchFeatureLearner(InvariantTorchFeaturePolicy(policy())).snapshot_state()
    elif fault == "label":
        point = True
    monkeypatch.setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        evaluate(source, state, updates=point, budget=budget)


def test_should_stop_on_changed_native_payload_after_one_paid_forward(case, monkeypatch):
    predict = TorchFeatureLearner.predict

    def corrupt(owner, features):
        result = predict(owner, features)
        state = owner.snapshot_state()
        state.head_state["bias_hidden"][0, 0] += 0.1
        owner.restore_state(state)
        return result

    monkeypatch.setattr(TorchFeatureLearner, "predict", corrupt)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(TorchFeatureLearner(policy()))
    report = rejected.value.report
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert all(slot["status"] == "not_run" for slot in report["episodes"][1:])


def test_should_validate_original_completed_guard_before_any_work(case, monkeypatch):
    source = TorchFeatureLearner(policy())
    monkeypatch.setattr(source, "fork", forbidden)

    def refuse():
        raise ValueError("original training evidence changed")

    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="original training") as rejected:
        evaluate(source, guard=refuse)
    assert rejected.value.report["work"]["live_prediction_calls_attempted"] == 0
