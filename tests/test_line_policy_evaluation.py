"""Actual bounded line dynamics with fake prediction ports; no learner updates."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

import src.app.line_policy_evaluation as evaluation
from src.core.line_navigation import LineNavigation


def identity(**changes):
    values: dict[str, Any] = dict(
        method="pc",
        seed=23,
        actor_version="frozen-actor-6",
        task_sequence=("A", "B", "A"),
        checkpoint_updates=6,
        replay_enabled=False,
    )
    values.update(changes)
    return evaluation.EvaluationIdentity(**values)


class Predictor:
    """Expose exactly the prediction-only port used by live evaluation."""

    def __init__(self, probabilities=(0.75,), *, mutate=False):
        self.probabilities = probabilities
        self.inputs: list[np.ndarray] = []
        self.mutate = mutate

    def __call__(self, features):
        assert type(features) is np.ndarray
        assert features.dtype == np.dtype(np.float64)
        assert features.shape == (1, 2)
        self.inputs.append(features.copy())
        value = self.probabilities[min(len(self.inputs) - 1, len(self.probabilities) - 1)]
        if self.mutate:
            features.fill(999)
        return np.full((1, 1), value, dtype=np.float64)


def run(predict=None, **changes):
    options: dict[str, Any] = dict(identity=identity(), context="A", start=-2, goal=2, max_steps=4)
    options.update(changes)
    return evaluation.run_line_episode(Predictor() if predict is None else predict, **options)


@pytest.mark.parametrize(
    ("changes", "label"),
    [
        ({"method": "unknown"}, "method"),
        ({"seed": True}, "bool seed"),
        ({"seed": -1}, "negative seed"),
        ({"seed": 2**31}, "overflow seed"),
        ({"actor_version": "   "}, "empty actor"),
        ({"actor_version": "a" * 129}, "long actor"),
        ({"actor_version": "é" * 65}, "utf8 actor"),
        ({"task_sequence": []}, "mutable sequence"),
        ({"task_sequence": ()}, "empty sequence"),
        ({"task_sequence": ("A",) * 33}, "long sequence"),
        ({"task_sequence": ("é" * 17,)}, "utf8 context"),
        ({"checkpoint_updates": True}, "bool updates"),
        ({"checkpoint_updates": 513}, "overflow updates"),
        ({"replay_enabled": 1}, "nonbool replay"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_identity_should_reject_invalid_frozen_controls(changes, label):
    with pytest.raises((TypeError, ValueError)):
        identity(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"context": "unseen"},
        {"start": True},
        {"start": -3},
        {"goal": 3},
        {"start": 2, "goal": 2},
        {"max_steps": True},
        {"max_steps": 0},
        {"max_steps": 17},
        {"before_prediction": 1},
    ],
)
def test_episode_should_validate_controls_before_prediction(changes):
    predictor = Predictor()
    with pytest.raises((TypeError, ValueError)):
        run(predictor, **changes)
    assert predictor.inputs == []


def test_episode_should_revalidate_identity_after_external_tampering():
    config = identity()
    object.__setattr__(config, "seed", -1)
    predictor = Predictor()
    with pytest.raises((TypeError, ValueError)):
        run(predictor, identity=config)
    assert predictor.inputs == []


@pytest.mark.parametrize(
    ("start", "goal", "expected"),
    [(-2, 2, (1,)), (2, -2, (0,)), (0, 2, (0, 1))],
)
def test_environment_should_publish_only_current_legal_actions(start, goal, expected):
    environment = LineNavigation(start, goal)
    assert environment.legal_actions == expected
    action = int(goal > start)
    for _ in range(abs(goal - start)):
        environment.step(action)
    assert environment.legal_actions == ()
    with pytest.raises(ValueError, match="terminal"):
        environment.step(action)


@pytest.mark.parametrize(("start", "goal", "probability"), [(-2, 2, 0.75), (2, -2, 0.25)])
def test_policy_should_act_through_complete_actual_episode(start, goal, probability):
    predictor = Predictor((probability,))
    report = run(predictor, start=start, goal=goal)
    assert report["status"] == "complete"
    assert report["outcome"] == "success"
    assert report["success"] is True
    assert report["return"] == 1.0
    assert report["steps_executed"] == 4
    assert report["prediction_calls_attempted"] == 4
    assert report["prediction_calls_returned"] == 4
    assert len(predictor.inputs) == len(report["transitions"]) == 4
    direction = 1 if goal > start else -1
    for index, transition in enumerate(report["transitions"]):
        position = start + index * direction
        assert transition["observation"] == (position / 2, goal / 2)
        assert transition["next_observation"] == ((position + direction) / 2, goal / 2)
        assert transition["action"] == int(direction > 0)
        assert transition["probability"] == probability
        assert transition["applied"] is True
        assert transition["reward"] == float(index == 3)
        assert transition["terminal"] is (index == 3)
        np.testing.assert_array_equal(predictor.inputs[index], [[position / 2, goal / 2]])
    assert report["execution_origin"] == "live_environment"
    assert report["checkpoint_origin"] == "recorded_parameters"
    assert report["phase"] == "evaluation"
    assert report["learning_active"] is False


def test_policy_should_select_right_at_exact_half_and_stop_at_terminal():
    predictor = Predictor((0.5,))
    report = run(predictor, start=1, goal=2)
    assert report["outcome"] == "success"
    assert report["steps_executed"] == 1
    assert report["prediction_calls_attempted"] == 1
    assert len(predictor.inputs) == 1
    assert report["transitions"][0]["action"] == 1


@pytest.mark.parametrize(("start", "goal", "probability"), [(-2, 2, 0.0), (2, -2, 1.0)])
def test_policy_should_record_illegal_action_without_clipping_or_retry(start, goal, probability):
    predictor = Predictor((probability,))
    report = run(predictor, start=start, goal=goal)
    assert report["status"] == "complete"
    assert report["outcome"] == "illegal_action"
    assert report["success"] is False
    assert report["return"] == 0
    assert report["steps_executed"] == 0
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert len(predictor.inputs) == len(report["transitions"]) == 1
    transition = report["transitions"][0]
    assert transition["action"] == int(probability >= 0.5)
    assert transition["action"] not in transition["legal_actions"]
    assert transition["applied"] is False
    assert transition["observation"] == transition["next_observation"] == (start / 2, goal / 2)
    assert transition["reward"] == 0
    assert transition["terminal"] is False


def test_policy_should_stop_oscillation_at_declared_step_limit():
    predictor = Predictor((0.75, 0.25, 0.75, 0.25))
    report = run(predictor, start=0, goal=2)
    assert report["status"] == "complete"
    assert report["outcome"] == "step_limit"
    assert report["success"] is False
    assert report["return"] == 0
    assert report["steps_executed"] == 4
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 4
    assert [step["action"] for step in report["transitions"]] == [1, 0, 1, 0]
    assert all(step["applied"] and not step["terminal"] for step in report["transitions"])


def test_mutating_predictor_should_not_change_actual_dynamics_or_recorded_observations():
    predictor = Predictor(mutate=True)
    report = run(predictor)
    assert report["success"] is True
    assert [step["observation"][0] for step in report["transitions"]] == [-1, -0.5, 0, 0.5]
    assert [float(features[0, 0]) for features in predictor.inputs] == [-1, -0.5, 0, 0.5]


def test_report_should_own_identity_and_transitions_across_fresh_episodes():
    config = identity(method="gated_cpc", replay_enabled=True)
    first = run(identity=config)
    saved = deepcopy(first)
    first["identity"]["actor_version"] = "changed"
    first["transitions"][0]["observation"] = (999, 999)
    second = run(identity=config)
    assert config.actor_version == "frozen-actor-6"
    assert second == saved
    assert second["identity"] == dict(
        method="gated_cpc",
        seed=23,
        actor_version="frozen-actor-6",
        task_sequence=("A", "B", "A"),
        checkpoint_updates=6,
        replay_enabled=True,
    )


@pytest.mark.parametrize(
    "result",
    [
        [[0.5]],
        np.array([[0.5]], dtype=np.float32),
        np.array([0.5], dtype=np.float64),
        np.array([[np.nan]], dtype=np.float64),
        np.array([[1.01]], dtype=np.float64),
    ],
    ids=["list", "float32", "shape", "nan", "range"],
)
def test_invalid_return_should_preserve_returned_cost_without_applying_action(result):
    calls = []

    def predict(features):
        calls.append(features.copy())
        return result

    with pytest.raises(evaluation.LineEvaluationFailed) as failure:
        run(predict)
    report = failure.value.report
    assert isinstance(failure.value.__cause__, ValueError)
    assert report["status"] == "failed"
    assert report["success"] is False
    assert report["steps_executed"] == 0
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert len(calls) == len(report["transitions"]) == 1
    assert report["transitions"][0]["action"] is None
    assert report["transitions"][0]["applied"] is False


def test_prediction_fault_should_preserve_preceding_steps_and_permanent_attempt():
    predictor = Predictor()
    primary = RuntimeError("prediction fault")

    def predict(features):
        if len(predictor.inputs) == 1:
            raise primary
        return predictor(features)

    with pytest.raises(evaluation.LineEvaluationFailed) as failure:
        run(predict)
    report = failure.value.report
    assert failure.value.__cause__ is primary
    assert "prediction fault" in report["reason"]
    assert report["steps_executed"] == 1
    assert report["prediction_calls_attempted"] == 2
    assert report["prediction_calls_returned"] == 1
    assert len(report["transitions"]) == 2
    assert report["transitions"][0]["applied"] is True
    assert report["transitions"][1]["action"] is None


def test_budget_fault_should_stop_before_admitting_another_prediction():
    predictor = Predictor()
    checkpoints = []
    primary = RuntimeError("original budget exhausted")

    def before():
        checkpoints.append(len(predictor.inputs))
        if len(checkpoints) == 2:
            raise primary

    with pytest.raises(evaluation.LineEvaluationFailed) as failure:
        run(predictor, before_prediction=before)
    report = failure.value.report
    assert failure.value.__cause__ is primary
    assert checkpoints == [0, 1]
    assert report["steps_executed"] == 1
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert len(report["transitions"]) == 1


def test_environment_fault_should_preserve_predictions_and_primary_error(monkeypatch):
    original = LineNavigation.step
    steps = []
    primary = RuntimeError("environment fault")

    def step(environment, action):
        steps.append(action)
        if len(steps) == 2:
            raise primary
        return original(environment, action)

    monkeypatch.setattr(LineNavigation, "step", step)
    with pytest.raises(evaluation.LineEvaluationFailed) as failure:
        run()
    report = failure.value.report
    assert failure.value.__cause__ is primary
    assert report["steps_executed"] == 1
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 2
    assert len(report["transitions"]) == 2
    assert report["transitions"][1]["action"] == 1
    assert report["transitions"][1]["applied"] is False


def test_failure_should_take_owned_bounded_report_snapshot():
    source: dict[str, Any] = dict(reason="fault", transitions=[dict(observation=(0, 1))])
    failure = evaluation.LineEvaluationFailed(source)
    source["transitions"][0]["observation"] = (999, 999)
    assert failure.report["transitions"] == [dict(observation=(0, 1))]
