"""Mapped controls compose with existing frozen collection and live evaluation."""

from copy import deepcopy
import pickle

import numpy as np
import pytest

import src.app.key_door_policy_evaluation as evaluation
import src.app.key_door_training as training
from src.app.line_policy_evaluation import EvaluationIdentity
from src.core.key_door_navigation import (
    KeyDoorLayout,
    KeyDoorNavigation,
    key_door_layout_record,
    scripted_actions,
)
from test_key_door_training import Source, collect


REMAPPED = (1, 2, 3, 0)


def mapped_layout(rotation=0):
    return KeyDoorLayout("B", rotation, 2, False, 0.0, REMAPPED)


def evaluate(predict, layout, cue=1, **options):
    return evaluation.run_key_door_episode(
        predict,
        identity=EvaluationIdentity("pc", 17, "owned-checkpoint-8", ("A", "B", "C", "A"), 8, True),
        context=layout.layout_id,
        layout=layout,
        cue=cue,
        **options,
    )


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("cue", (1, -1))
def test_should_collect_mapped_policy_actions_with_a_fresh_frozen_session_each_episode(
    rotation, cue
):
    layout = mapped_layout(rotation)
    actions = scripted_actions(layout, cue)
    source = Source(actions)
    sampler = np.random.Generator(np.random.PCG64(29))
    expected_sampler = np.random.Generator(np.random.PCG64(29))
    source_before = pickle.dumps(source.state)
    global_before = pickle.dumps(np.random.get_state())

    reports = [
        collect(source, sampler, layout=layout, cue=cue, episode_id=f"mapped-{index}")
        for index in range(2)
    ]

    assert len(source.sessions) == source.snapshots == 2
    assert source.sessions[0] is not source.sessions[1]
    for session, report in zip(source.sessions, reports, strict=True):
        assert report["status"] == "complete" and report["termination"] == "terminal"
        assert report["total_return"] == 1.0 and report["episode"].role == "train"
        assert report["layout"] == key_door_layout_record(layout)
        assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 8
        assert (
            report["sampling_draws"]
            == report["environment_steps_attempted"]
            == report["environment_steps"]
            == 8
        )
        assert len(session.inputs) == 8
        environment = KeyDoorNavigation(layout, cue)
        for index, (action, slot) in enumerate(zip(actions, report["transitions"], strict=True)):
            expected_sampler.random()
            assert slot["observation"] == environment.observation
            np.testing.assert_array_equal(
                session.inputs[index], np.array([environment.observation])
            )
            assert slot["action"] == action and action in slot["legal_actions"]
            assert slot["probabilities"] == tuple(float(i == action) for i in range(5))
            step = environment.step(action)
            assert slot["applied"] is True and slot["next_observation"] == step.observation
            assert slot["terminal"] is (index == 7)
        assert session.inputs[0][0, 6] == cue
        assert all(features[0, 6] == 0 for features in session.inputs[1:])
    assert reports[0]["sampler_after"] == reports[1]["sampler_before"]
    assert (
        sampler.bit_generator.state
        == reports[1]["sampler_after"]
        == expected_sampler.bit_generator.state
    )
    assert pickle.dumps(source.state) == source_before
    assert pickle.dumps(np.random.get_state()) == global_before
    reports[0]["behavior_state"]["parameters"].fill(99)
    np.testing.assert_array_equal(source.state["parameters"], np.array([0.25]))
    np.testing.assert_array_equal(reports[1]["behavior_state"]["parameters"], np.array([0.25]))


def test_should_preserve_actual_illegal_mapped_decision_without_teacher_repair():
    layout = mapped_layout()
    assert KeyDoorNavigation(layout, 1).legal_actions == (3,)
    raw = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    source = Source(rows=[raw])
    sampler = np.random.Generator(np.random.PCG64(29))
    expected = np.random.Generator(np.random.PCG64(29))
    expected.random()
    report = collect(source, sampler, layout=layout)
    assert report["termination"] == "illegal_action" and report["total_return"] == 0
    assert (
        report["prediction_calls_attempted"]
        == report["prediction_calls_returned"]
        == report["sampling_draws"]
        == 1
    )
    assert report["environment_steps_attempted"] == report["environment_steps"] == 0
    slot = report["transitions"][0]
    assert slot["action"] == 0 and slot["probabilities"] == tuple(raw[0])
    assert slot["applied"] is False and slot["terminal"] is False
    assert slot["next_observation"] == slot["observation"]
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state
    assert report["layout"] == key_door_layout_record(layout)


def test_should_charge_only_returned_draws_and_applied_steps_before_mapped_prediction_failure():
    layout = mapped_layout()
    source = Source(scripted_actions(layout, 1), fail_at=2)
    before = pickle.dumps(source.state)
    sampler = np.random.Generator(np.random.PCG64(29))
    expected = np.random.Generator(np.random.PCG64(29))
    expected.random()
    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, layout=layout)
    report = caught.value.report
    assert "frozen prediction fault" in str(caught.value.__cause__)
    assert report["status"] == "failed" and report["episode"] is None
    assert report["layout"] == key_door_layout_record(layout)
    assert report["prediction_calls_attempted"] == 2 and report["prediction_calls_returned"] == 1
    assert (
        report["sampling_draws"]
        == report["environment_steps_attempted"]
        == report["environment_steps"]
        == 1
    )
    assert len(report["transitions"]) == 2
    assert report["transitions"][0]["action"] == 3 and report["transitions"][0]["applied"] is True
    assert report["transitions"][1]["action"] is None
    assert report["transitions"][1]["probabilities"] is None
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state
    assert pickle.dumps(source.state) == before


def test_should_condition_only_on_actual_mapped_legal_actions_using_one_owned_draw():
    layout = mapped_layout(2)
    raw = np.array([[0.1, 0.15, 0.2, 0.25, 0.3]])
    source = Source(rows=[raw])
    sampler = np.random.Generator(np.random.PCG64(29))
    expected = np.random.Generator(np.random.PCG64(29))
    report = collect(source, sampler, layout=layout, max_steps=3, condition_on_legal_actions=True)
    environment = KeyDoorNavigation(layout, 1)
    for slot in report["transitions"]:
        legal = environment.legal_actions
        distribution = np.zeros(5)
        distribution[list(legal)] = raw[0, list(legal)] / sum(raw[0, list(legal)])
        action = int(
            np.searchsorted(
                np.cumsum(distribution), expected.random() * distribution.sum(), side="right"
            )
        )
        assert slot["legal_actions"] == legal and slot["raw_probabilities"] == tuple(raw[0])
        np.testing.assert_allclose(slot["probabilities"], distribution, rtol=0, atol=2e-16)
        assert slot["action"] == action and slot["applied"] is True
        assert slot["next_observation"] == environment.step(action).observation
    assert report["sampling_draws"] == report["environment_steps"] == len(report["transitions"])
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state


@pytest.mark.parametrize("role", ("final_test", "probe_train", "probe_test"))
def test_should_reject_non_training_mapped_collection_before_source_or_random_work(role):
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    with pytest.raises(ValueError):
        collect(source, sampler, layout=mapped_layout(), role=role)
    assert source.snapshots == 0 and source.sessions == []
    assert sampler.bit_generator.state == before


def test_should_reject_bypassed_invalid_map_before_collection_or_evaluation_calls():
    layout = mapped_layout()
    object.__setattr__(layout, "movement_action_map", (0, 0, 2, 3))
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    with pytest.raises(ValueError):
        collect(source, sampler, layout=layout)
    assert source.snapshots == 0 and source.sessions == []
    assert sampler.bit_generator.state == before
    calls = []

    def predict(features):
        calls.append(features)
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        evaluate(predict, layout)
    assert calls == []


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("cue", (1, -1))
def test_should_evaluate_actual_mapped_actions_and_record_spatial_provenance(rotation, cue):
    layout = mapped_layout(rotation)
    source = Source(scripted_actions(layout, cue))
    predictor = source.start_episode()
    report = evaluate(predictor.predict, layout, cue)
    assert report["status"] == "complete" and report["outcome"] == "success"
    assert report["success"] is True and report["terminal"] is True
    assert report["return"] == 1.0 and report["steps_executed"] == 8
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 8
    assert report["layout"] == key_door_layout_record(layout)
    assert report["learning_active"] is False and report["phase"] == "evaluation"
    assert tuple(slot["action"] for slot in report["transitions"]) == scripted_actions(layout, cue)
    for features, slot in zip(predictor.inputs, report["transitions"], strict=True):
        np.testing.assert_array_equal(features, np.array([slot["observation"]]))
        assert slot["action"] in slot["legal_actions"] and slot["applied"] is True


def test_should_preserve_evaluator_mapped_failure_without_legal_action_repair():
    layout = mapped_layout()
    predictor = Source((0,)).start_episode()
    report = evaluate(predictor.predict, layout)
    assert report["outcome"] == "illegal_action" and report["steps_executed"] == 0
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert report["transitions"][0]["action"] == 0
    assert report["transitions"][0]["probabilities"] == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert report["layout"] == key_door_layout_record(layout)


def test_should_preserve_legacy_identity_collection_and_evaluator_layout_shapes():
    layout = KeyDoorLayout("A")
    actions = scripted_actions(layout, 1)
    report = collect(Source(actions), layout=layout)
    assert "layout" not in report
    evaluated = evaluate(Source(actions).start_episode().predict, layout)
    assert evaluated["layout"] == {
        "layout_id": "A",
        "rotation": 0,
        "branch_row": 2,
        "swap_keys": False,
        "appearance": 0.0,
    }
