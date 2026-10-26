"""Raw recurrent reward collection with owned fake sessions and native reports."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from src.adapters import minigrid_reward_collection as collection
from src.core.minigrid_key_door import predefined_layouts, predefined_probe_layouts
from src.core.policy_gradient_episode import PolicyGradientEpisode


def frame(step=0):
    return (-1.0,) * 147 + (float(2 * (step % 4) / 3 - 1),)


class Source:
    def __init__(self, work):
        self.work = work
        self.state = {
            "parameters": np.array([0.25]),
            "optimizer": {"momentum": 0.5},
            "gradients": None,
        }
        self.rows = np.array([[0.05, 0.1, 0.0, 0.15, 0.2, 0.0, 0.5]], dtype=np.float64)
        self.snapshots = 0
        self.sessions = []

    def snapshot_state(self):
        self.snapshots += 1
        return deepcopy(self.state)

    def start_episode(self):
        result = Session(self)
        self.sessions.append(result)
        if self.work.get("initial_counter"):
            result.completed_steps = 1
        if self.work.get("initial_state"):
            result.state["parameters"] += 1
        return result


class Session:
    def __init__(self, source):
        self.source, self.completed_steps = source, 0
        self.state = deepcopy(source.state)

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, values):
        assert values.dtype == np.float64 and values.shape == (1, 148)
        work = self.source.work
        work.setdefault("predictions", []).append(values.copy())
        if work.get("predict_fail_at") == len(work["predictions"]):
            raise RuntimeError("primary prediction failed")
        self.completed_steps += 1
        callback = work.get("on_predict")
        if callback:
            callback(self)
        if work.get("bad_shape"):
            return np.zeros((1, 6), dtype=np.float64)
        if work.get("bad_mass"):
            return np.zeros((1, 7), dtype=np.float64)
        return self.source.rows.copy()


class Environment:
    work: ClassVar[dict[str, Any]]

    def __init__(self, layout, cue, seed=17, max_steps=16, *, role="train", episode_id="fake"):
        self.layout, self.cue, self.seed, self.max_steps = layout, cue, seed, max_steps
        self.role, self.episode_id = role, episode_id
        self.observation, self.terminal, self.closed = frame(), False, False
        self.actions, self.transitions = [], []
        self.returned, self.terminated, self.truncated, self.unknown = 0, False, False, False
        self.work.setdefault("environments", []).append(self)

    def step(self, action):
        assert type(action) is int and 0 <= action < 7
        self.actions.append(action)
        if self.work.get("step_error"):
            self.observation, self.unknown = frame(1), True
            raise RuntimeError("primary physics failed")
        before = self.observation
        self.returned += 1
        self.observation = frame(self.returned)
        ending = self.work.get("ending", "terminal")
        self.terminal = self.returned >= min(2, self.max_steps)
        self.terminated = self.terminal and ending != "truncated"
        self.truncated = self.terminal and ending in ("truncated", "both")
        reward = (-1.0 if self.work.get("negative") else 0.5) if self.terminated else 0.0
        result = dict(
            observation=self.observation,
            reward=reward,
            terminated=self.terminated,
            truncated=self.truncated,
        )
        self.transitions.append(
            dict(
                observation=before,
                action=action,
                attempted=True,
                applied=True,
                next_observation=self.observation,
                reward=reward,
                terminated=self.terminated,
                truncated=self.truncated,
            )
        )
        if self.work.get("malformed_step"):
            result["observation"] = self.observation[:-1]
        callback = self.work.get("on_step")
        if callback:
            callback(self)
        return result

    def close(self):
        self.closed = True
        if self.work.get("close_error"):
            raise RuntimeError("cleanup close failed")

    @property
    def report(self):
        if self.work.get("readback_error"):
            raise RuntimeError("cleanup readback failed")
        return deepcopy(
            dict(
                environment_id="fake-minigrid",
                context=self.layout.context,
                cue=self.cue,
                seed=self.seed,
                max_steps=self.max_steps,
                role=self.role,
                episode_id=self.episode_id,
                observation_dim=148,
                num_classes=7,
                reset_calls=1,
                closed=self.closed,
                environment_steps_attempted=len(self.actions),
                environment_steps=self.returned,
                terminated=self.terminated,
                truncated=self.truncated,
                total_return=sum(row["reward"] for row in self.transitions),
                status="complete" if self.terminal else ("failed" if self.unknown else "running"),
                transitions=self.transitions,
                failure_work_may_be_unobserved=self.unknown,
            )
        )


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    source = Source(work)
    sampler = np.random.Generator(np.random.PCG64(29))
    guards = []

    def guard():
        guards.append(True)
        if work.get("guard_before") or (
            work.get("guard_after_predict") and work.get("predictions")
        ):
            raise ValueError("original completed boundary refused")

    monkeypatch.setattr(Environment, "work", work, raising=False)
    monkeypatch.setattr(collection, "MiniGridKeyDoor", Environment)
    monkeypatch.setattr(
        collection, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    return SimpleNamespace(work=work, source=source, sampler=sampler, guard=guard, guards=guards)


def collect(case, **options):
    arguments = dict(
        episode_id="reward-collected",
        sampler=case.sampler,
        check_boundary=case.guard,
        clock=lambda: 10.0,
        max_steps=2,
    )
    arguments.update(options)
    return collection.collect_minigrid_reward_episode(
        case.source,
        arguments.pop("layout", predefined_layouts()[0]),
        arguments.pop("cue", "blue"),
        **arguments,
    )


@pytest.mark.parametrize("layout", predefined_layouts()[:3], ids=lambda value: value.context)
@pytest.mark.parametrize("cue", ["blue", "red"])
def test_should_sample_one_raw_seven_action_cdf_draw_per_observed_step(case, layout, cue):
    before, global_rng = pickle.dumps(case.source.state), pickle.dumps(np.random.get_state())
    expected = np.random.Generator(np.random.PCG64(29))
    report = collect(case, layout=layout, cue=cue)
    assert report["status"] == "complete" and type(report["episode"]) is PolicyGradientEpisode
    episode = report["episode"]
    assert (
        episode.context == layout.context and episode.role == "train" and episode.num_classes == 7
    )
    assert len(episode.transitions) == len(report["predictions"]) == 2
    for index, prediction in enumerate(report["predictions"]):
        draw = float(expected.random())
        probabilities = case.source.rows[0]
        action = int(np.searchsorted(np.cumsum(probabilities), draw, side="right"))
        assert prediction["observation"] == frame(index)
        assert prediction["uniform"] == draw and prediction["action"] == action
        assert (
            tuple(prediction["probabilities"])
            == episode.behavior_probabilities[index]
            == tuple(probabilities)
        )
        transition = episode.transitions[index]
        assert transition.action == action and transition.observation == frame(index)
        assert transition.next_observation == frame(index + 1)
    assert (
        report["sampler_after"] == case.sampler.bit_generator.state == expected.bit_generator.state
    )
    assert report["sampler_binding_preserved"] is report["state_unchanged"] is True
    assert (
        pickle.dumps(case.source.state) == before
        and pickle.dumps(np.random.get_state()) == global_rng
    )
    assert len(case.source.sessions) == 1 and case.source.sessions[0].completed_steps == 2
    assert case.work["environments"][0].closed and report["environment"]["closed"] is True
    work = report["work"]
    assert (
        work["prediction_calls_attempted"]
        == work["prediction_calls_returned"]
        == work["sampling_draws"]
        == 2
    )
    assert work["environment_steps_attempted"] == work["environment_steps_returned"] == 2
    assert work["close_calls_attempted"] == work["close_calls_returned"] == 1
    assert report["elapsed_seconds"] == 0


@pytest.mark.parametrize("ending", ["terminal", "truncated", "both"])
def test_should_preserve_native_ending_flags_and_terminal_precedence(case, ending):
    case.work.update(ending=ending, negative=True)
    report = collect(case)
    assert report["episode"].termination == ("truncated" if ending == "truncated" else "terminal")
    assert report["episode"].transitions[-1].terminal is (ending != "truncated")
    assert report["environment"]["terminated"] is (ending != "truncated")
    assert report["environment"]["truncated"] is (ending in ("truncated", "both"))
    assert report["episode"].transitions[-1].reward == (0 if ending == "truncated" else -1)


def test_should_accept_zero_probability_actions_and_native_noop_without_mask_or_retry(case):
    case.source.rows[:] = 0
    case.source.rows[0, 6] = 1
    report = collect(case)
    assert case.work["environments"][0].actions == [6, 6]
    assert report["work"]["sampling_draws"] == 2
    assert all(row["action"] == 6 for row in report["predictions"])


@pytest.mark.parametrize(
    "fault",
    [
        "D",
        "E",
        "geometry",
        "cue",
        "role",
        "seed",
        "steps_low",
        "steps_high",
        "identity",
        "sampler",
        "generator",
    ],
)
def test_should_refuse_invalid_collection_admission_before_source_or_native_work(case, fault):
    options: dict[str, Any] = {}
    if fault == "D":
        options["layout"] = predefined_layouts()[-1]
    elif fault == "E":
        options["layout"] = predefined_probe_layouts()[0]
    elif fault == "geometry":
        options["layout"] = replace(predefined_layouts()[0], rotation=1)
    elif fault == "cue":
        options["cue"] = "green"
    elif fault == "role":
        options["role"] = "final_test"
    elif fault == "seed":
        options["seed"] = True
    elif fault == "steps_low":
        options["max_steps"] = 0
    elif fault == "steps_high":
        options["max_steps"] = 17
    elif fault == "identity":
        options["episode_id"] = "x" * 65
    elif fault == "sampler":
        options["sampler"] = None
    else:
        options["sampler"] = np.random.Generator(np.random.MT19937(29))
    before = deepcopy(case.sampler.bit_generator.state)
    with pytest.raises(ValueError):
        collect(case, **options)
    assert case.source.snapshots == 0 and case.source.sessions == []
    assert not case.work.get("environments") and case.sampler.bit_generator.state == before


@pytest.mark.parametrize("fault", ["prediction", "shape", "mass", "step", "malformed_step"])
def test_should_retain_paid_prediction_draw_and_physics_counts_on_failure(case, fault):
    case.work[
        {
            "prediction": "predict_fail_at",
            "shape": "bad_shape",
            "mass": "bad_mass",
            "step": "step_error",
            "malformed_step": "malformed_step",
        }[fault]
    ] = 1
    expected = np.random.Generator(np.random.PCG64(29))
    spent = fault in ("step", "malformed_step")
    if spent:
        expected.random()
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert report["status"] == "failed" and report["episode"] is None
    assert len(report["predictions"]) == 1 and case.work["environments"][0].closed
    work = report["work"]
    assert work["prediction_calls_attempted"] == 1
    assert work["prediction_calls_returned"] == (0 if fault == "prediction" else 1)
    assert work["sampling_draws"] == work["environment_steps_attempted"] == int(spent)
    assert work["environment_steps_returned"] == int(fault == "malformed_step")
    assert report["sampler_after"] == expected.bit_generator.state
    assert len(case.work["predictions"]) == 1


def test_should_keep_one_successful_prefix_when_second_prediction_fails(case):
    case.work["predict_fail_at"] = 2
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert report["work"]["prediction_calls_attempted"] == 2
    assert report["work"]["prediction_calls_returned"] == report["work"]["sampling_draws"] == 1
    assert report["work"]["environment_steps_returned"] == 1
    assert len(report["environment"]["transitions"]) == 1 and len(report["predictions"]) == 2


@pytest.mark.parametrize("target", ["source", "session", "counter", "sampler"])
def test_should_refuse_unobserved_state_counter_or_sampler_changes_before_draw(case, target):
    def mutate(session):
        if target == "source":
            case.source.state["parameters"] += 1
        elif target == "session":
            session.state["optimizer"]["momentum"] += 1
        elif target == "counter":
            session.completed_steps += 1
        else:
            case.sampler.random()

    case.work["on_predict"] = mutate
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert report["work"]["prediction_calls_returned"] == 1
    assert report["work"]["sampling_draws"] == report["work"]["environment_steps_attempted"] == 0
    assert report["episode"] is None and case.work["environments"][0].closed


@pytest.mark.parametrize("fault", ["initial_counter", "initial_state"])
def test_should_refuse_unclean_session_before_environment(case, fault):
    case.work[fault] = True
    with pytest.raises(collection.MiniGridRewardCollectionFailed):
        collect(case)
    assert not case.work.get("environments") and not case.work.get("predictions")


@pytest.mark.parametrize("fault", ["guard_before", "guard_after_predict"])
def test_should_preserve_original_guard_authority_and_stop_before_draw(case, fault):
    case.work[fault] = True
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert report["work"]["sampling_draws"] == report["work"]["environment_steps_attempted"] == 0
    assert report["work"]["prediction_calls_returned"] == int(fault == "guard_after_predict")
    if fault == "guard_before":
        assert case.source.snapshots == 0 and not case.work.get("environments")


@pytest.mark.parametrize("cleanup", ["close_error", "readback_error"])
def test_should_preserve_primary_physics_failure_alongside_cleanup_failure(case, cleanup):
    case.work.update(step_error=True)
    case.work[cleanup] = True
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert "primary physics failed" in report["reason"]
    assert report[cleanup] is not None and report["work"]["sampling_draws"] == 1
    assert report["work"]["environment_steps_attempted"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert case.work["environments"][0].closed


@pytest.mark.parametrize("cleanup", ["close_error", "readback_error"])
def test_should_fail_cleanup_only_fault_without_publishing_a_trainable_episode(case, cleanup):
    case.work[cleanup] = True
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert "cleanup" in report["reason"] and report[cleanup]
    assert report["episode"] is None and report["work"]["environment_steps_returned"] == 2


def test_should_refuse_clock_reversal_after_returned_prediction_before_sampling(case):
    now = [10.0]
    case.work["on_predict"] = lambda session: now.__setitem__(0, 9.0)
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case, clock=lambda: now[0])
    report = raised.value.report
    assert report["work"]["prediction_calls_returned"] == 1
    assert report["work"]["sampling_draws"] == 0
    assert report["elapsed_seconds"] is None or report["elapsed_seconds"] >= 0


def test_should_own_trajectory_state_probability_and_native_report_evidence(case):
    report = collect(case)
    episode = deepcopy(report["episode"])
    state = deepcopy(case.source.state)
    case.source.rows.fill(0)
    report["behavior_state"]["parameters"].fill(99)
    report["environment"]["transitions"][0]["action"] = 99
    assert report["episode"] == episode
    assert pickle.dumps(case.source.state) == pickle.dumps(state)
    assert case.work["environments"][0].transitions[0]["action"] != 99


def test_should_preserve_unknown_constructor_attempt_without_a_fabricated_environment(
    monkeypatch, case
):
    def fail(*args, **kwargs):
        raise RuntimeError("native constructor failed")

    monkeypatch.setattr(collection, "MiniGridKeyDoor", fail)
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert "native constructor failed" in report["reason"] and report["environment"] is None
    assert report["work"]["environment_constructions_attempted"] == 1
    assert report["work"]["environment_constructions_returned"] == 0
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert report["predictions"] == [] and report["work"]["sampling_draws"] == 0


def test_should_refuse_source_identity_as_a_session_before_native_construction(monkeypatch, case):
    monkeypatch.setattr(case.source, "start_episode", lambda: case.source)
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert "independent" in report["reason"]
    assert report["work"]["sessions_started"] == 1
    assert report["work"]["environment_constructions_attempted"] == 0
    assert not case.work.get("environments") and report["predictions"] == []


@pytest.mark.parametrize("corrupted", [0.75, float("nan"), True, None])
def test_should_refuse_corrupted_native_total_without_losing_paid_work(
    monkeypatch, case, corrupted
):
    original = vars(Environment)["report"].fget

    def corrupt(environment):
        report = original(environment)
        report["total_return"] = corrupted
        return report

    monkeypatch.setattr(Environment, "report", property(corrupt))
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert "total_return" in report["reason"]
    assert report["episode"] is None and report["work"]["sampling_draws"] == 2
    assert report["work"]["environment_steps_returned"] == 2
    assert report["environment"]["closed"] is True
    json.dumps(
        {key: value for key, value in report.items() if key != "behavior_state"}, allow_nan=False
    )
    if type(corrupted) is float and np.isnan(corrupted):
        assert report["environment"]["total_return"] is None
        assert report["environment"]["invalid_total_return"] == "nan"


def test_should_refuse_missing_behavior_checkpoint_before_session_or_physics(monkeypatch, case):
    monkeypatch.setattr(case.source, "snapshot_state", lambda: None)
    with pytest.raises(collection.MiniGridRewardCollectionFailed) as raised:
        collect(case)
    report = raised.value.report
    assert "non-null full behavior checkpoint" in report["reason"]
    assert report["episode"] is None and report["work"]["sessions_attempted"] == 0
    assert report["work"]["environment_constructions_attempted"] == 0
    assert report["work"]["sampling_draws"] == 0
