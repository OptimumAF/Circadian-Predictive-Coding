"""Real key/door dynamics and prediction-only episodes, without native learners."""

from typing import Any

import numpy as np
import pytest

import src.app.key_door_policy_evaluation as evaluation
from src.app.line_policy_evaluation import EvaluationIdentity
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation, scripted_actions


def layout(**changes):
    values: dict[str, Any] = dict(
        layout_id="A", rotation=0, branch_row=2, swap_keys=False, appearance=0.0
    )
    values.update(changes)
    return KeyDoorLayout(**values)


def identity():
    return EvaluationIdentity("pc", 17, "owned-checkpoint-8", ("A", "B", "C", "A"), 8, True)


class Predictor:
    def __init__(self, actions, *, mutate=False):
        self.actions = tuple(actions)
        self.inputs: list[np.ndarray] = []
        self.mutate = mutate

    def __call__(self, features):
        assert type(features) is np.ndarray
        assert features.dtype == np.float64 and features.shape == (1, 8)
        assert np.isfinite(features).all() and (np.abs(features) <= 1).all()
        action = self.actions[min(len(self.inputs), len(self.actions) - 1)]
        self.inputs.append(features.copy())
        if self.mutate:
            features.fill(999)
        result = np.zeros((1, 5), dtype=np.float64)
        result[0, action] = 1
        return result


def run(predict=None, **changes):
    options: dict[str, Any] = dict(identity=identity(), context="A", layout=layout(), cue=1)
    options.update(changes)
    if predict is None:
        predict = Predictor(scripted_actions(options["layout"], options["cue"]))
    return evaluation.run_key_door_episode(predict, **options)


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("branch_row", [1, 2, 3])
@pytest.mark.parametrize("swap_keys", [False, True])
@pytest.mark.parametrize("cue", [1, -1])
def test_should_execute_eight_actual_teacher_steps_for_every_declared_geometry(
    rotation, branch_row, swap_keys, cue
):
    config = layout(rotation=rotation, branch_row=branch_row, swap_keys=swap_keys)
    environment = KeyDoorNavigation(config, cue)
    actions = scripted_actions(config, cue)
    assert type(actions) is tuple and len(actions) == 8
    key_side = 3 if (cue == 1) != swap_keys else 1
    canonical = (
        (0,) * (4 - branch_row)
        + (key_side, 4, (key_side + 2) % 4)
        + (0,) * (branch_row - 1)
        + (4, 0)
    )
    expected = tuple((action + rotation) % 4 if action < 4 else 4 for action in canonical)
    assert actions == expected
    rewards = []
    for index, action in enumerate(actions):
        assert action in environment.legal_actions
        result = environment.step(action)
        rewards.append(result.reward)
        assert result.observation == environment.observation
        assert type(result.observation) is tuple and len(result.observation) == 8
        assert all(type(value) is float and -1 <= value <= 1 for value in result.observation)
        assert result.observation[6] == 0
        assert result.terminal is (index == 7)
        assert result.success is (index == 7)
    assert rewards == [0] * 7 + [1]
    assert environment.terminal is True and environment.success is True
    assert environment.legal_actions == ()
    before = environment.observation
    with pytest.raises(ValueError):
        environment.step(0)
    assert environment.observation == before


@pytest.mark.parametrize(
    "changes",
    [
        {"layout_id": ""},
        {"rotation": True},
        {"rotation": 4},
        {"branch_row": 0},
        {"branch_row": 4},
        {"swap_keys": 1},
        {"appearance": np.nan},
        {"appearance": 1.1},
    ],
)
def test_should_refuse_invalid_layout_controls(changes):
    with pytest.raises(ValueError):
        layout(**changes)


@pytest.mark.parametrize("cue", [True, 0, 2])
def test_should_refuse_non_native_or_non_signed_cue(cue):
    with pytest.raises(ValueError):
        KeyDoorNavigation(layout(), cue)


@pytest.mark.parametrize("cue", [1, -1])
def test_should_terminate_with_negative_reward_when_teacher_uses_opposite_cue(cue):
    config = layout()
    environment = KeyDoorNavigation(config, cue)
    for action in scripted_actions(config, -cue)[:7]:
        assert action in environment.legal_actions
        result = environment.step(action)
    assert result.reward == -1 and result.terminal is True and result.success is False
    assert environment.terminal is True and environment.success is False
    assert environment.legal_actions == ()


@pytest.mark.parametrize("branch_row", [1, 2, 3])
def test_should_erase_cue_and_merge_observations_before_opposite_required_branch_actions(
    branch_row,
):
    config = layout(branch_row=branch_row)
    positive, negative = KeyDoorNavigation(config, 1), KeyDoorNavigation(config, -1)
    assert positive.observation[:6] == negative.observation[:6]
    assert positive.observation[6] == 1 and negative.observation[6] == -1
    assert positive.observation[7] == negative.observation[7]
    assert positive.legal_actions == negative.legal_actions
    for _ in range(4 - branch_row):
        positive.step(0)
        negative.step(0)
        assert positive.observation == negative.observation
        assert positive.legal_actions == negative.legal_actions
    index = 4 - branch_row
    assert scripted_actions(config, 1)[index] != scripted_actions(config, -1)[index]
    # Returning to the start must not recreate the transient cue.
    for _ in range(4 - branch_row):
        positive.step(2)
        negative.step(2)
    assert positive.observation == negative.observation
    assert positive.observation[6] == 0


def test_should_keep_visible_held_key_color_after_pickup_and_unlock():
    config = layout()
    environment = KeyDoorNavigation(config, -1)
    actions = scripted_actions(config, -1)
    for index, action in enumerate(actions):
        result = environment.step(action)
        if index >= 3:
            assert result.observation[5] == -1
    assert environment.success is True


def test_should_refuse_illegal_move_without_mutating_state_or_erasing_initial_cue():
    environment = KeyDoorNavigation(layout(), 1)
    before = environment.observation
    assert 3 not in environment.legal_actions
    with pytest.raises(ValueError):
        environment.step(3)
    assert environment.observation == before and environment.observation[6] == 1


@pytest.mark.parametrize("action", [True, 0.0, -1, 5])
def test_should_refuse_non_native_or_out_of_range_action_without_mutation(action):
    environment = KeyDoorNavigation(layout(), 1)
    before = environment.observation
    with pytest.raises(ValueError):
        environment.step(action)
    assert environment.observation == before and environment.terminal is False


def test_should_own_layout_before_caller_mutates_original_record():
    config = layout()
    environment = KeyDoorNavigation(config, 1)
    actions = scripted_actions(config, 1)
    object.__setattr__(config, "rotation", 3)
    object.__setattr__(config, "appearance", 1.0)
    for action in actions:
        environment.step(action)
    assert environment.success is True and environment.observation[7] == 0


@pytest.mark.parametrize("cue", [1, -1])
def test_should_evaluate_actual_eight_step_success_with_target_free_predictions(cue):
    config = layout()
    predictor = Predictor(scripted_actions(config, cue))
    report = run(predictor, layout=config, cue=cue)
    assert report["status"] == "complete" and report["outcome"] == "success"
    assert report["success"] is True and report["terminal"] is True
    assert report["return"] == 1 and report["steps_executed"] == 8
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 8
    assert len(report["transitions"]) == len(predictor.inputs) == 8
    for index, transition in enumerate(report["transitions"]):
        assert transition["action"] in transition["legal_actions"]
        assert transition["applied"] is True
        assert transition["terminal"] is (index == 7)
        np.testing.assert_array_equal(transition["observation"], predictor.inputs[index][0])
        assert len(transition["probabilities"]) == 5
        assert transition["probabilities"][transition["action"]] == 1
    assert report["execution_origin"] == "live_environment"
    assert report["checkpoint_origin"] == "owned_native_checkpoint"
    assert report["phase"] == "evaluation" and report["learning_active"] is False


def test_should_preserve_wrong_key_as_complete_negative_policy_outcome():
    predictor = Predictor(scripted_actions(layout(), -1))
    report = run(predictor)
    assert report["status"] == "complete" and report["outcome"] == "wrong_key"
    assert report["success"] is False and report["terminal"] is True
    assert report["return"] == -1 and report["steps_executed"] == 7
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 7
    assert len(predictor.inputs) == 7


def test_should_not_mask_illegal_greedy_action_when_lower_probability_legal_action_exists():
    calls = []

    def predict(features):
        calls.append(features.copy())
        return np.array([[0.49, 0.0, 0.0, 0.51, 0.0]])

    report = run(predict)
    assert report["status"] == "complete" and report["outcome"] == "illegal_action"
    assert report["success"] is False and report["terminal"] is False
    assert report["return"] == report["steps_executed"] == 0
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    transition = report["transitions"][0]
    assert transition["action"] == 3 and transition["applied"] is False
    assert transition["observation"] == transition["next_observation"]
    assert len(calls) == 1


def test_should_stop_oscillating_policy_at_fixed_step_limit():
    report = run(Predictor((0, 2)), max_steps=2)
    assert report["status"] == "complete" and report["outcome"] == "step_limit"
    assert report["success"] is False and report["terminal"] is False
    assert report["steps_executed"] == 2 and report["return"] == 0
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 2


def test_should_choose_lowest_index_for_probability_tie():
    report = run(lambda features: np.full((1, 5), 0.2), max_steps=1)
    assert report["transitions"][0]["action"] == 0
    assert report["transitions"][0]["applied"] is True
    assert report["outcome"] == "step_limit"


def test_mutating_predictor_should_not_change_environment_observations_or_dynamics():
    predictor = Predictor(scripted_actions(layout(), 1), mutate=True)
    report = run(predictor)
    assert report["success"] is True
    assert all(
        max(abs(value) for value in row["observation"]) <= 1 for row in report["transitions"]
    )
    assert predictor.inputs[0][0, 6] == 1
    assert all(features[0, 6] == 0 for features in predictor.inputs[1:])


def test_should_admit_declared_heldout_context_without_adding_it_to_training_task_sequence():
    config = layout(layout_id="D", branch_row=1)
    report = run(layout=config, context="D", layout_partition="heldout_layout")
    assert report["success"] is True and report["context"] == "D"
    assert report["layout_partition"] == "heldout_layout"
    assert "D" not in report["identity"]["task_sequence"]


def test_should_refuse_unknown_known_layout_context_before_prediction():
    predictor = Predictor((0,))
    with pytest.raises(ValueError):
        run(predictor, context="D")
    assert predictor.inputs == []


@pytest.mark.parametrize("fault", ["context_layout", "known_marked_heldout"])
def test_should_refuse_inconsistent_layout_identity_or_partition_before_prediction(fault):
    predictor = Predictor((0,))
    options = (
        dict(context="A", layout=layout(layout_id="D"))
        if fault == "context_layout"
        else dict(context="A", layout_partition="heldout_layout")
    )
    with pytest.raises(ValueError):
        run(predictor, **options)
    assert predictor.inputs == []


@pytest.mark.parametrize("fault", ["dtype", "shape", "nan", "sum", "negative"])
def test_should_charge_returned_malformed_probabilities_without_environment_step(fault):
    result = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    if fault == "dtype":
        result = result.astype(np.float32)
    elif fault == "shape":
        result = result[:, :4]
    elif fault == "nan":
        result[0, 0] = np.nan
    elif fault == "sum":
        result[0, 0] = 0.1
    else:
        result[0, :2] = [-0.1, 0.5]
    with pytest.raises(evaluation.KeyDoorEvaluationFailed) as failure:
        run(lambda features: result)
    report = failure.value.report
    assert isinstance(failure.value.__cause__, ValueError)
    assert report["status"] == "failed"
    assert report["steps_executed"] == 0
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert len(report["transitions"]) == 1 and report["transitions"][0]["applied"] is False


def test_should_preserve_prior_success_and_attempt_cost_when_predictor_raises():
    predictor = Predictor((0,))
    primary = RuntimeError("prediction failed")

    def predict(features):
        if len(predictor.inputs) == 1:
            raise primary
        return predictor(features)

    with pytest.raises(evaluation.KeyDoorEvaluationFailed) as failure:
        run(predict)
    report = failure.value.report
    assert failure.value.__cause__ is primary
    assert report["steps_executed"] == 1
    assert report["prediction_calls_attempted"] == 2 and report["prediction_calls_returned"] == 1
    assert len(report["transitions"]) == 2


def test_should_stop_at_original_budget_before_another_prediction_attempt():
    predictor = Predictor((0,))
    callbacks = []
    primary = RuntimeError("original budget exhausted")

    def before():
        callbacks.append(len(predictor.inputs))
        if len(callbacks) == 3:
            raise primary

    with pytest.raises(evaluation.KeyDoorEvaluationFailed) as failure:
        run(predictor, before_prediction=before)
    report = failure.value.report
    assert failure.value.__cause__ is primary and callbacks == [0, 1, 2]
    assert report["steps_executed"] == 2
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 2
    assert len(report["transitions"]) == 2


def test_should_preserve_primary_environment_fault_and_returned_prediction_cost(monkeypatch):
    original = KeyDoorNavigation.step
    calls = []
    primary = RuntimeError("environment step failed")

    def step(environment, action):
        calls.append(action)
        if len(calls) == 2:
            raise primary
        return original(environment, action)

    monkeypatch.setattr(KeyDoorNavigation, "step", step)
    with pytest.raises(evaluation.KeyDoorEvaluationFailed) as failure:
        run(Predictor((0,)))
    report = failure.value.report
    assert failure.value.__cause__ is primary
    assert report["steps_executed"] == 1
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 2
    assert len(report["transitions"]) == 2 and report["transitions"][1]["applied"] is False


def test_should_revalidate_tampered_identity_before_prediction():
    config = identity()
    object.__setattr__(config, "seed", -1)
    predictor = Predictor((0,))
    with pytest.raises(ValueError):
        run(predictor, identity=config)
    assert predictor.inputs == []


def test_should_revalidate_tampered_layout_before_prediction():
    config = layout()
    object.__setattr__(config, "rotation", 4)
    predictor = Predictor((0,))
    with pytest.raises(ValueError):
        run(predictor, layout=config)
    assert predictor.inputs == []


def test_failure_should_own_partial_transition_report():
    original: dict[str, Any] = dict(reason="fault", transitions=[dict(applied=False)])
    failure = evaluation.KeyDoorEvaluationFailed(original)
    original["transitions"][0]["applied"] = True
    assert failure.report["transitions"] == [dict(applied=False)]


def test_should_return_owned_report_and_config_records_for_fresh_episodes():
    config = layout()
    first = run(layout=config)
    first["layout"]["rotation"] = 3
    first["identity"]["actor_version"] = "changed"
    first["transitions"][0]["observation"] = (999,) * 8
    second = run(layout=config)
    assert second["success"] is True and second["layout"]["rotation"] == config.rotation == 0
    assert second["identity"]["actor_version"] == identity().actor_version
    assert max(abs(value) for value in second["transitions"][0]["observation"]) <= 1


@pytest.mark.parametrize("max_steps", [True, 0])
def test_should_refuse_invalid_step_limit_before_prediction(max_steps):
    predictor = Predictor((0,))
    with pytest.raises(ValueError):
        run(predictor, max_steps=max_steps)
    assert predictor.inputs == []
