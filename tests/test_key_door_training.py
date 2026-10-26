"""Actual sampled game trajectories with owned fake frozen prediction sessions."""

from copy import deepcopy
import pickle
from typing import Any

import numpy as np
import pytest

import src.app.key_door_training as training
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation, scripted_actions


class Session:
    def __init__(self, source):
        self.rows = deepcopy(source.rows)
        self.state = deepcopy(source.state)
        self.inputs: list[np.ndarray] = []
        self.fail_at = source.fail_at

    def predict(self, features):
        self.inputs.append(features.copy())
        if len(self.inputs) == self.fail_at:
            raise RuntimeError("frozen prediction fault")
        return self.rows[min(len(self.inputs) - 1, len(self.rows) - 1)].copy()

    def snapshot_state(self):
        return deepcopy(self.state)


class Source:
    def __init__(self, actions=(0,), *, rows=None, fail_at=-1):
        self.rows = []
        for action in actions:
            value = np.zeros((1, 5), dtype=np.float64)
            value[0, action] = 1
            self.rows.append(value)
        if rows is not None:
            self.rows = rows
        self.state = dict(parameters=np.array([0.25]), updates=8)
        self.fail_at = fail_at
        self.snapshots = 0
        self.sessions: list[Session] = []

    def snapshot_state(self):
        self.snapshots += 1
        return deepcopy(self.state)

    def start_episode(self):
        result = Session(self)
        self.sessions.append(result)
        return result


def collect(source=None, sampler=None, **options):
    return training.collect_key_door_reward_episode(
        Source() if source is None else source,
        options.pop("layout", KeyDoorLayout("A")),
        options.pop("cue", 1),
        episode_id=options.pop("episode_id", "sampled-train-A"),
        sampler=np.random.Generator(np.random.PCG64(29)) if sampler is None else sampler,
        **options,
    )


@pytest.mark.parametrize("last_mass", [0.2, 0.2000005])
def test_should_draw_one_uniform_from_raw_mass_cdf_without_masking_or_normalizing(last_mass):
    probabilities = np.array([[0.2, 0.2, 0.2, 0.2, last_mass]])
    source = Source(rows=[probabilities])
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    before = deepcopy(sampler.bit_generator.state)
    global_before = pickle.dumps(np.random.get_state())
    uniform = expected.random()
    action = int(
        np.searchsorted(np.cumsum(probabilities[0]), uniform * probabilities.sum(), side="right")
    )
    report = collect(source, sampler, max_steps=1)
    assert report["status"] == "complete"
    assert (
        report["sampling_draws"]
        == report["prediction_calls_attempted"]
        == report["prediction_calls_returned"]
        == 1
    )
    assert report["transitions"][0]["action"] == action
    np.testing.assert_array_equal(report["transitions"][0]["probabilities"], probabilities[0])
    assert report["sampler_before"] == before
    assert report["sampler_after"] == expected.bit_generator.state == sampler.bit_generator.state
    assert pickle.dumps(np.random.get_state()) == global_before


def test_should_collect_eight_actual_sampled_teacher_actions_and_own_behavior_rng_state():
    config = KeyDoorLayout("A")
    source = Source(scripted_actions(config, 1))
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    before = pickle.dumps(source.state)
    expected.random(8)
    report = collect(source, sampler)
    assert report["status"] == "complete" and report["termination"] == "terminal"
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 8
    assert report["sampling_draws"] == report["environment_steps"] == 8
    assert report["total_return"] == 1
    packet = report["episode"]
    assert packet.role == "train" and len(packet.transitions) == 8
    assert [step.action for step in packet.transitions] == list(scripted_actions(config, 1))
    assert [step.terminal for step in packet.transitions] == [False] * 7 + [True]
    assert packet.transitions[0].observation[6] == 1
    assert all(step.observation[6] == 0 for step in packet.transitions[1:])
    assert len(source.sessions) == 1 and len(source.sessions[0].inputs) == 8
    assert pickle.dumps(source.state) == before
    assert report["sampler_after"] == expected.bit_generator.state
    report["sampler_before"]["state"]["state"] = 0
    report["sampler_after"]["state"]["state"] = 0
    report["behavior_state"]["parameters"].fill(99)
    assert sampler.bit_generator.state == expected.bit_generator.state
    assert pickle.dumps(source.state) == before


def test_should_report_step_cap_as_truncated_without_fake_terminal():
    report = collect(Source((0,)), max_steps=1)
    assert report["status"] == "complete" and report["termination"] == "truncated"
    assert report["environment_steps"] == report["sampling_draws"] == 1
    assert report["total_return"] == 0
    assert report["episode"].termination == "truncated"
    assert report["episode"].transitions[0].terminal is False


def test_should_keep_sampled_illegal_action_unapplied_with_zero_reward_and_spent_draw():
    report = collect(Source((3,)))
    assert report["status"] == "complete" and report["termination"] == "illegal_action"
    assert report["environment_steps"] == 0 and report["sampling_draws"] == 1
    assert report["total_return"] == 0
    transition = report["transitions"][0]
    assert transition["action"] == 3 and transition["applied"] is False
    assert transition["terminal"] is False and transition["reward"] == 0
    assert transition["observation"] == transition["next_observation"]
    assert report["episode"].termination == "illegal_action"


@pytest.mark.parametrize(
    "fault",
    [
        "final_test",
        "probe_train",
        "probe_test",
        "step_zero",
        "step_bool",
        "cue",
        "layout",
        "sampler",
    ],
)
def test_should_refuse_invalid_public_controls_before_source_prediction_or_sampling(fault):
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    options: dict[str, Any] = {}
    if fault in ("final_test", "probe_train", "probe_test"):
        options["role"] = fault
    elif fault == "step_zero":
        options["max_steps"] = 0
    elif fault == "step_bool":
        options["max_steps"] = True
    elif fault == "cue":
        options["cue"] = 0
    elif fault == "layout":
        config = KeyDoorLayout("A")
        object.__setattr__(config, "rotation", 4)
        options["layout"] = config
    else:
        sampler = np.random.Generator(np.random.MT19937(29))
    with pytest.raises(ValueError):
        collect(source, sampler, **options)
    assert source.snapshots == 0 and source.sessions == []
    if fault != "sampler":
        assert sampler.bit_generator.state == before


@pytest.mark.parametrize("fault", ["dtype", "shape", "mass"])
def test_should_count_returned_invalid_prediction_without_spending_a_uniform(fault):
    value = np.array([[0.2] * 5])
    if fault == "dtype":
        value = value.astype(np.float32)
    elif fault == "shape":
        value = value[:, :4]
    else:
        value[0, 0] = 0.1
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    with pytest.raises(training.KeyDoorTrainingFailed) as failure:
        collect(Source(rows=[value]), sampler)
    report = failure.value.report
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert report["sampling_draws"] == report["environment_steps"] == 0
    assert sampler.bit_generator.state == before
    assert isinstance(failure.value.__cause__, ValueError)


def test_should_stop_original_budget_callback_before_admitting_prediction():
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    primary = RuntimeError("original training budget exhausted")

    def stop():
        raise primary

    with pytest.raises(training.KeyDoorTrainingFailed) as failure:
        collect(source, sampler, before_prediction=stop)
    report = failure.value.report
    assert failure.value.__cause__ is primary
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 0
    assert report["sampling_draws"] == report["environment_steps"] == 0
    assert sampler.bit_generator.state == before


def test_should_preserve_one_spent_draw_and_prior_step_when_second_prediction_raises():
    source = Source((0,), fail_at=2)
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    expected.random()
    with pytest.raises(training.KeyDoorTrainingFailed) as failure:
        collect(source, sampler)
    report = failure.value.report
    assert "frozen prediction fault" in str(failure.value.__cause__)
    assert report["prediction_calls_attempted"] == 2 and report["prediction_calls_returned"] == 1
    assert report["sampling_draws"] == report["environment_steps"] == 1
    assert sampler.bit_generator.state == expected.bit_generator.state == report["sampler_after"]
    assert len(report["transitions"]) == 2


def test_should_preserve_spent_uniform_on_environment_fault_and_keep_primary_error(monkeypatch):
    primary = RuntimeError("physical environment fault")

    def fail(environment, action):
        raise primary

    monkeypatch.setattr(KeyDoorNavigation, "step", fail)
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    expected.random()
    with pytest.raises(training.KeyDoorTrainingFailed) as failure:
        collect(Source((0,)), sampler)
    report = failure.value.report
    assert failure.value.__cause__ is primary
    assert report["prediction_calls_attempted"] == report["prediction_calls_returned"] == 1
    assert report["sampling_draws"] == 1 and report["environment_steps"] == 0
    assert report["environment_steps_attempted"] == 1
    assert report["transitions"][0]["action"] == 0 and report["transitions"][0]["applied"] is False
    assert sampler.bit_generator.state == expected.bit_generator.state == report["sampler_after"]
    report["sampler_after"]["state"]["state"] = 0
    assert sampler.bit_generator.state == expected.bit_generator.state


def test_should_reproduce_private_sampling_after_explicit_caller_rng_restore():
    source = Source(rows=[np.array([[0.2] * 5])])
    sampler = np.random.Generator(np.random.PCG64(29))
    first = collect(source, sampler, max_steps=1)
    sampler.bit_generator.state = deepcopy(first["sampler_before"])
    second = collect(source, sampler, max_steps=1)
    assert second["episode"] == first["episode"]
    assert second["sampler_after"] == first["sampler_after"]
    assert second["transitions"] == first["transitions"]


def test_should_capture_original_owned_layout_context_before_source_callback_mutation(monkeypatch):
    config = KeyDoorLayout("A")
    source = Source(scripted_actions(config, 1))
    original = source.start_episode

    def start():
        object.__setattr__(config, "layout_id", "changed")
        object.__setattr__(config, "rotation", 3)
        return original()

    monkeypatch.setattr(source, "start_episode", start)
    result = collect(source, layout=config)
    assert result["episode"].context == "A" and result["total_return"] == 1
    assert result["environment_steps"] == 8


@pytest.mark.parametrize("phase", ["before_prediction", "predict"])
def test_should_refuse_unexpected_external_sampler_changes_without_counting_them_as_own_draws(
    monkeypatch, phase
):
    source = Source()
    sampler, expected = (
        np.random.Generator(np.random.PCG64(29)),
        np.random.Generator(np.random.PCG64(29)),
    )
    expected.random()
    options = {}
    if phase == "before_prediction":
        options["before_prediction"] = lambda: sampler.random()
    else:
        original = Session.predict

        def predict(owner, features):
            result = original(owner, features)
            sampler.random()
            return result

        monkeypatch.setattr(Session, "predict", predict)
    with pytest.raises(training.KeyDoorTrainingFailed) as failure:
        collect(source, sampler, **options)
    report = failure.value.report
    assert report["status"] == "failed"
    assert report["prediction_calls_attempted"] == (0 if phase == "before_prediction" else 1)
    assert report["prediction_calls_returned"] == (0 if phase == "before_prediction" else 1)
    assert report["sampling_draws"] == report["environment_steps"] == 0
    assert sampler.bit_generator.state == expected.bit_generator.state == report["sampler_after"]
