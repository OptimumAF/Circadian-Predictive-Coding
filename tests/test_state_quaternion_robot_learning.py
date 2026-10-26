"""Canonical state features before A fit and frozen fake native prediction.

One BP initializer2944 normals, three updates/six rows, one fit/two rows/84
coordinates. Eight conditioning/residual/actor calls each read16rows. Two fake
scenes use four prediction rows/four controls; both reset/close/read joint names
and limits, sampling six raw qpos packets/fifty-four coordinates in total.
All other learners/native calls are cached RNG-free forks or stubs; no physics.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_learnability_demo as demo
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.continuous_episode_clock import EpisodeClock
from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.infra import maniskill_continuous as native
from test_continuous_robot_learning import assert_state
from test_maniskill_joint_positions import StateEnvironment


def observation(index=0, *, offset=0.0):
    values = [0.0] * 42
    values[0], values[7], values[8] = offset + index, 0.02, 0.02
    sign = -1.0 if index % 2 else 1.0
    values[22:26] = [sign * value for value in (-0.5, 0.5, -0.5, 0.5)]
    values[32:36] = [sign * value for value in (0.0, 0.0, -1.0, 0.0)]
    return tuple(values)


def episodes():
    spec = ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)

    def source(seed):
        actions = ((0.5, -0.5, 0.0, -1.0), (-0.5, 0.5, 0.5, 1.0))
        offset = 0.0 if seed == 17 else 10.0
        return ContinuousEpisode(
            f"pick-cube-readiness-teacher-{seed}",
            "pick-cube" if seed == 17 else "pick-cube-train-19",
            "train",
            spec,
            tuple(
                ContinuousTransition(
                    i,
                    observation(i, offset=offset),
                    actions[i],
                    float(i == 1),
                    observation(i + 1, offset=offset),
                    i == 1,
                    False,
                )
                for i in range(2)
            ),
        )

    return source(17), source(19)


class QuaternionEnvironment(StateEnvironment):
    def __init__(self):
        super().__init__()
        self.reset_observation = np.asarray([observation()], dtype=np.float64)

    def step(self, command):
        returned = super().step(command)
        return (np.asarray([observation(len(self.commands))], dtype=np.float64), *returned[1:])


def forbidden(*args, **kwargs):
    pytest.fail("invalid or stub request admitted model/native work")


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    folder = tmp_path_factory.mktemp("canonical-state-learning")
    warm, new = episodes()
    original_inputs = asdict(warm), asdict(new)
    original_train, original_fit, original_predict = (
        entry.run_continuous_learnability,
        demo.fit_training_standardization,
        NumpyContinuousBCLearner.predict,
    )
    captured: dict[str, Any] = dict(predictions=[], fits=[])
    environment = QuaternionEnvironment()

    def train(given_warm, given_new, **options):
        captured["training_inputs"] = given_warm, given_new
        result = original_train(given_warm, given_new, **options)
        assert result.candidate is not None
        captured["result"], captured["before_native"] = result, result.candidate.snapshot_state()
        return result

    def fit(value, **options):
        captured["fits"].append((asdict(value), options))
        return original_fit(value, **options)

    def predict(owner, features):
        if len(features) == 1:
            captured["predictions"].append(features.copy())
        return original_predict(owner, features)

    def factory(**options):
        assert options == {}
        return environment

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(entry, "run_continuous_learnability", train)
        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(NumpyContinuousBCLearner, "predict", predict)
        patch.setattr(native, "_make_environment", factory)
        patch.setattr(smoke, "version", lambda _: "fake-canonical-state")
        report = entry.run_learning_rollout(
            folder / "case.json",
            warm,
            new,
            phase_updates=1,
            a_only_control=True,
            canonical_state_quaternions=True,
        )
    assert (asdict(warm), asdict(new)) == original_inputs
    return report, captured, environment, warm, new


def test_should_project_both_train_sources_before_the_single_a_fit_without_relabelling(learned):
    report, captured, _, warm, new = learned
    assert report["complete"] is True
    projected_warm, projected_new = captured["training_inputs"]
    assert (projected_warm.episode_id, projected_new.episode_id) == (
        warm.episode_id,
        new.episode_id,
    )
    assert all(
        row.observation[22:26] == (0.5, -0.5, 0.5, -0.5)
        for value in (projected_warm, projected_new)
        for row in value.transitions
    )
    assert all(
        row.next_observation[32:36] == (0.0, 0.0, 1.0, 0.0)
        for value in (projected_warm, projected_new)
        for row in value.transitions
    )
    assert captured["fits"] == [(asdict(projected_warm), {})]
    policy = captured["result"].candidate.policy
    assert policy.observation_standardization.means[22:26] == (0.5, -0.5, 0.5, -0.5)
    assert policy.observation_standardization.scales[22:26] == (1.0,) * 4
    assert report["observation_projection"] == asdict(state_quaternion_projection())
    assert report["input_source_episodes"]["warm"]["episode_id"] == warm.episode_id
    assert report["input_source_episodes"]["new"]["episode_id"] == new.episode_id
    assert report["training"]["work"]["training_rows_by_context"] == dict(warm=6, new=0)
    assert all(
        receipt["episode_id"] == warm.episode_id
        for receipt in report["training"]["applied_updates"]
    )
    assert report["input_projection_work"] == dict(
        episode_calls=2,
        observation_attempts=8,
        observation_returns=8,
        source_values_upper_bound=336,
        source_values_validated=336,
        projected_values=336,
        projected_numeric_bytes=2688,
    )


def test_should_predict_canonical42_while_preserving_signed_raw_trace_and_full_candidate_state(
    learned,
):
    report, captured, environment, _, _ = learned
    returned = report["rollout"]
    assert (
        returned["complete"]
        is returned["projection_exact"]
        is report["candidate_unchanged"]
        is True
    )
    rows = returned["episode"]["transitions"]
    assert rows[0]["observation"][22:26] == (-0.5, 0.5, -0.5, 0.5)
    assert rows[0]["next_observation"][22:26] == (0.5, -0.5, 0.5, -0.5)
    assert rows[-1]["next_observation"][22:26] == (-0.5, 0.5, -0.5, 0.5)
    assert len(captured["predictions"]) == 2
    for value in captured["predictions"]:
        assert value.shape == (1, 42)
        assert tuple(value[0, 22:26]) == (0.5, -0.5, 0.5, -0.5)
        assert tuple(value[0, 32:36]) == (0.0, 0.0, 1.0, 0.0)
    assert returned["projection_work"] == dict(
        observation_attempts=2,
        observation_returns=2,
        source_values_upper_bound=84,
        source_values_validated=84,
        projected_values=84,
        projected_numeric_bytes=672,
    )
    assert (
        returned["joint_position_exact"] is True
        and returned["joint_position_validation_samples"] == 3
    )
    assert (
        environment.reset_calls == [17]
        and len(environment.commands) == 2
        and environment.close_calls == 1
    )
    assert_state(captured["before_native"], captured["result"].candidate.snapshot_state())


@pytest.mark.parametrize("value,visual", [(None, False), (1, False), ("true", False), (True, True)])
def test_should_refuse_malformed_or_visual_canonical_request_before_io_and_fit(
    monkeypatch, tmp_path, value, visual
):
    monkeypatch.setattr(entry, "_admit_inputs", forbidden)
    monkeypatch.setattr(entry, "run_continuous_learnability", forbidden)
    output = tmp_path / "absent.json"
    with pytest.raises(ValueError):
        entry.run_learning_rollout(
            output, *episodes(), visual_observation=visual, canonical_state_quaternions=value
        )
    assert not output.exists()


@pytest.mark.parametrize("selected", [None, False])
def test_should_preserve_default_training_kwargs_and_outer_report_without_projection(
    monkeypatch, tmp_path, selected
):
    calls = []

    def train(*args, **options):
        calls.append((args, options))
        return demo.ContinuousLearnabilityResult(dict(complete=False), None)

    monkeypatch.setattr(entry, "run_continuous_learnability", train)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    options = {} if selected is None else dict(canonical_state_quaternions=selected)
    warm, new = episodes()
    report = entry.run_learning_rollout(
        tmp_path / "partial.json", warm, new, phase_updates=1, **options
    )
    assert calls == [((warm, new), dict(learner_family="bp", phase_updates=1))]
    assert "observation_projection" not in report and "input_projection_work" not in report


@pytest.mark.parametrize(
    "fault",
    [
        "missing_recipe",
        "recipe_changed",
        "float_recipe",
        "false_exact",
        "bool_attempt",
        "float_returns",
        "source_count",
        "missing_work",
    ],
)
def test_should_preserve_paid_training_and_return_but_refuse_unbound_native_projection(
    learned, monkeypatch, tmp_path, fault
):
    report, captured, _, warm, new = learned
    result = demo.ContinuousLearnabilityResult(
        deepcopy(captured["result"].report), captured["result"].candidate.fork()
    )
    raw = deepcopy(report["rollout"])
    if fault == "missing_recipe":
        del raw["observation_projection"]
    elif fault == "recipe_changed":
        raw["observation_projection"]["quaternion_starts"] = (21, 32)
    elif fault == "float_recipe":
        raw["observation_projection"]["source_dim"] = 42.0
    elif fault == "false_exact":
        raw["projection_exact"] = False
    elif fault == "bool_attempt":
        raw["projection_work"]["observation_attempts"] = True
    elif fault == "float_returns":
        raw["projection_work"]["observation_returns"] = 2.0
    elif fault == "source_count":
        raw["projection_work"]["source_values_validated"] = 42
    else:
        del raw["projection_work"]
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: result)
    monkeypatch.setattr(
        entry, "run_frozen_rollout", lambda *args, **options: smoke.FrozenRolloutResult(raw, None)
    )
    observed = entry.run_learning_rollout(
        tmp_path / "tampered.json",
        warm,
        new,
        phase_updates=1,
        a_only_control=True,
        canonical_state_quaternions=True,
    )
    assert observed["complete"] is False and observed["rollout_attempted"] is True
    assert observed["training"] == result.report and observed["rollout"] == raw
    assert observed["candidate_unchanged"] is True


def test_should_refuse_final_telemetry_mutation_after_paid_controls_without_releasing_success(
    learned, monkeypatch, tmp_path
):
    _, captured, _, _, _ = learned
    owner = captured["result"].candidate.fork()
    before = owner.snapshot_state()
    projection = state_quaternion_projection()
    environment = QuaternionEnvironment()
    original_telemetry = smoke._read_joint_position_report

    def telemetry(port, report, **options):
        returned = original_telemetry(port, report, **options)
        if environment.close_calls:
            report["observation_projection"]["unit_tolerance"] = 0.001
        return returned

    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(smoke, "_read_joint_position_report", telemetry)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-canonical-mutation")
    result = smoke.run_frozen_rollout(
        tmp_path / "refused.json",
        learner=owner,
        observation_projection=projection,
        monitor_joint_positions=True,
    )
    assert result.report["complete"] is False and result.report["projection_exact"] is None
    assert result.report["prediction_returns"] == 2 and len(environment.commands) == 2
    assert result.report["projection_work"]["observation_returns"] == 2
    assert result.report["joint_position_work"]["samples"] == 3
    assert environment.close_calls == 1 and result.corrections is None
    assert_state(before, owner.snapshot_state())


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_cli_canonical_flag_only_when_selected(monkeypatch, tmp_path, selected):
    warm, new = episodes()
    calls = []
    monkeypatch.setattr(
        entry, "_read_teacher", lambda path, seed, **options: warm if seed == 17 else new
    )

    def run(*args, **options):
        calls.append(options)
        return dict(complete=False)

    monkeypatch.setattr(entry, "run_learning_rollout", run)
    code = entry.main(
        [
            "--output",
            str(tmp_path / "cli.json"),
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
            *(["--canonical-state-quaternions"] if selected else []),
        ]
    )
    assert code == 1
    expected = dict(learner_family="bp", phase_updates=8, visual_observation=False)
    if selected:
        expected["canonical_state_quaternions"] = True
    assert calls == [expected]


def test_should_refuse_visual_canonical_cli_combination_before_file_access(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "_read_teacher", forbidden)
    monkeypatch.setattr(entry, "run_learning_rollout", forbidden)
    output = tmp_path / "absent.json"
    with pytest.raises(SystemExit) as error:
        entry.main(
            [
                "--output",
                str(output),
                "--warm-teacher",
                "missing-A",
                "--new-teacher",
                "missing-B",
                "--canonical-state-quaternions",
                "--visual-observation",
            ]
        )
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("fault", ["visual", "actor", "clock", "corrections", "camera"])
def test_should_refuse_unsupported_state_projection_combinations_before_fork_or_sdk(
    learned, monkeypatch, tmp_path, fault
):
    _, captured, _, _, _ = learned
    owner = captured["result"].candidate.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    monkeypatch.setattr(smoke, "version", forbidden)
    options: dict[str, Any] = dict(
        learner=owner, observation_projection=state_quaternion_projection()
    )
    fault_options: dict[str, dict[str, Any]] = {
        "visual": dict(visual_observation=True),
        "actor": dict(actor=object()),
        "clock": dict(observation_clock=EpisodeClock(50, True)),
        "corrections": dict(corrective_queries=True),
        "camera": dict(camera_context=pick_cube_camera_context()),
    }
    options.update(fault_options[fault])
    result = smoke.run_frozen_rollout(tmp_path / "unsupported.json", **options)
    assert result.report["complete"] is False and result.report["creation_attempts"] == 0
    assert result.report["prediction_attempts"] == 0 and result.corrections is None
