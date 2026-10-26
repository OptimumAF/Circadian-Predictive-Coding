"""Legal-conditioned actual game collection with frozen fake probability sources."""

from copy import deepcopy
import pickle
from typing import Any

import numpy as np
import pytest

import src.app.key_door_training as training
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation
from src.core.legal_policy_gradient import LegalPolicyGradientEpisode
from test_key_door_training import Source, collect


LAYOUTS = (
    KeyDoorLayout("A"),
    KeyDoorLayout("B", 1, 2, True, 0.5),
    KeyDoorLayout("C_train", 2, 3, True, -0.75),
    KeyDoorLayout("D", 0, 1, False, 0.25),
)


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda value: value.layout_id)
@pytest.mark.parametrize("cue", [1, -1])
def test_should_record_actual_legal_support_and_one_draw_per_conditioned_decision(layout, cue):
    raw = np.array([[0.1, 0.15, 0.2, 0.25, 0.3]], dtype=np.float64)
    source = Source(rows=[raw])
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    source_before, global_before = pickle.dumps(source.state), pickle.dumps(np.random.get_state())
    report = collect(source, sampler, layout=layout, cue=cue, condition_on_legal_actions=True)
    assert report["status"] == "complete" and report["action_conditioning"] == "legal_actions_v1"
    packet = report["episode"]
    assert type(packet) is LegalPolicyGradientEpisode and packet.episode.role == "train"
    assert report["termination"] in ("terminal", "truncated")
    observed = KeyDoorNavigation(layout, cue)
    for index, slot in enumerate(report["transitions"]):
        legal = tuple(observed.legal_actions)
        probabilities = np.zeros(5, dtype=np.float64)
        probabilities[list(legal)] = raw[0, list(legal)] / sum(raw[0, list(legal)])
        draw = float(expected.random())
        action = int(
            np.searchsorted(np.cumsum(probabilities), draw * probabilities.sum(), side="right")
        )
        assert slot["observation"] == observed.observation
        assert slot["legal_actions"] == packet.legal_actions[index] == legal
        assert tuple(slot["raw_probabilities"]) == packet.raw_probabilities[index] == tuple(raw[0])
        np.testing.assert_allclose(slot["probabilities"], probabilities, rtol=0, atol=2e-16)
        assert packet.episode.behavior_probabilities[index] == slot["probabilities"]
        assert slot["action"] == action and action in legal and slot["applied"] is True
        step = observed.step(action)
        assert slot["next_observation"] == step.observation
        assert slot["reward"] == step.reward and slot["terminal"] is step.terminal
    count = len(report["transitions"])
    assert report["sampling_draws"] == report["environment_steps"] == count
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == count
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state
    assert pickle.dumps(source.state) == source_before
    assert pickle.dumps(np.random.get_state()) == global_before
    assert packet.episode.transitions[0].observation[6] == cue
    assert all(step.observation[6] == 0 for step in packet.episode.transitions[1:])


def test_should_refuse_zero_legal_mass_after_one_returned_prediction_before_draw_or_physics():
    layout = KeyDoorLayout("A")
    legal = KeyDoorNavigation(layout, 1).legal_actions
    illegal = next(action for action in range(5) if action not in legal)
    source = Source((illegal,))
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, layout=layout, condition_on_legal_actions=True)
    report: dict[str, Any] = caught.value.report
    assert report["status"] == "failed" and report["episode"] is None
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert (
        report["sampling_draws"]
        == report["environment_steps"]
        == report["environment_steps_attempted"]
        == 0
    )
    assert sampler.bit_generator.state == report["sampler_after"] == before
    assert report["transitions"][0]["raw_probabilities"][illegal] == 1
    assert report["transitions"][0]["probabilities"] is None


@pytest.mark.parametrize("invalid", [None, 0, "true"])
def test_should_refuse_non_bool_conditioning_before_source_or_sampler_work(invalid):
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    with pytest.raises(ValueError):
        collect(source, sampler, condition_on_legal_actions=invalid)
    assert source.snapshots == 0 and source.sessions == []
    assert sampler.bit_generator.state == before


def test_should_preserve_old_default_report_and_rng_when_conditioning_is_explicitly_disabled():
    raw = np.array([[0.1, 0.15, 0.2, 0.25, 0.3]])
    implicit = collect(Source(rows=[raw]), max_steps=3)
    explicit = collect(Source(rows=[raw]), max_steps=3, condition_on_legal_actions=False)
    assert pickle.dumps(implicit) == pickle.dumps(explicit)
    assert "action_conditioning" not in implicit
    assert all("raw_probabilities" not in slot for slot in implicit["transitions"])


def test_should_retain_first_conditioned_draw_and_step_when_second_prediction_fails():
    source = Source(rows=[np.array([[0.2] * 5])], fail_at=2)
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    expected.random()
    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, condition_on_legal_actions=True)
    report = caught.value.report
    assert report["prediction_calls_attempted"] == 2 and report["prediction_calls_returned"] == 1
    assert report["sampling_draws"] == report["environment_steps"] == 1
    assert report["transitions"][0]["action"] in report["transitions"][0]["legal_actions"]
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state


def test_should_retain_spent_conditioned_draw_when_environment_step_raises(monkeypatch):
    source = Source(rows=[np.array([[0.2] * 5])])
    primary = RuntimeError("conditioned physical step failed")

    def fail(*args):
        raise primary

    monkeypatch.setattr(KeyDoorNavigation, "step", fail)
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    expected.random()
    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, condition_on_legal_actions=True)
    report = caught.value.report
    assert caught.value.__cause__ is primary
    assert report["sampling_draws"] == report["environment_steps_attempted"] == 1
    assert report["environment_steps"] == 0
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state


def test_should_own_conditioned_evidence_without_aliasing_model_rows_or_source_snapshot():
    source = Source(rows=[np.array([[0.2] * 5])])
    report = collect(source, max_steps=1, condition_on_legal_actions=True)
    packet = deepcopy(report["episode"])
    source.rows[0].fill(0)
    report["behavior_state"]["parameters"].fill(99)
    report["transitions"][0]["raw_probabilities"] = (1.0, 0.0, 0.0, 0.0, 0.0)
    assert report["episode"] == packet
    np.testing.assert_array_equal(source.state["parameters"], np.array([0.25]))
