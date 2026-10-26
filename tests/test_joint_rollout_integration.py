"""Owned joint telemetry in frozen rollout reports, using fake IO only.

Per module invocation: four constructors, five explicit restores, 186 seeded
normal draws and 32 prediction rows (28 MLP, four nearest). No fits, SGD, appends
or native work. Fake boundary calls total139: scene/reset/close each20, names19,
limits19, controls32, mass reads8/setter1; two fake geometry calls are additional.
Telemetry accepts47 samples/423 coordinates.
"""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import sys
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import StableActor
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_episode_clock import EpisodeClock
from src.core.joint_position_limits import (
    JointPositionLimits,
    JointPositionMonitor,
    check_joint_position_samples,
    copy_joint_position_summary,
)
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as port
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import assert_same_state
from test_maniskill_joint_positions import Robot, StateEnvironment
from test_maniskill_rgb_observations import RGBEnvironment, packet, public_feature_oracle
from test_pick_cube_mass_context import FakeCube


def summary():
    monitor = JointPositionMonitor(
        JointPositionLimits(("arm", "finger"), ("rad", "m"), (-1.0, 0.0), (1.0, 0.04))
    )
    monitor.observe((-1.25, 0.05))
    monitor.observe((1.5, 0.02))
    return monitor.summary()


def joint_work(samples=3):
    return port.ManiSkillJointPositionWork(1, 1, samples, 9 * samples, 1, 1)


def test_should_own_strict_summary_and_reconstruct_signed_peaks_from_raw_samples():
    declared = summary()
    owned = copy_joint_position_summary(declared)
    assert owned == declared and owned is not declared and owned.limits is not declared.limits
    checked = check_joint_position_samples(declared, ((-1.25, 0.05), (1.5, 0.02)))
    assert checked == declared and checked is not declared
    assert checked.lower_peak == (-0.25, 0.0) and checked.upper_peak == pytest.approx((0.5, 0.01))
    assert checked.violations_per_joint == (2, 1)
    with pytest.raises(FrozenInstanceError):
        setattr(owned, "samples", 0)
    object.__setattr__(declared.limits, "upper", (99.0, 99.0))
    assert owned.limits.upper == (1.0, 0.04)
    empty = JointPositionMonitor(owned.limits).summary()
    assert check_joint_position_samples(empty, ()) == empty


@pytest.mark.parametrize(
    "changes",
    [
        dict(samples=True),
        dict(samples=52),
        dict(samples=-1),
        dict(violating_samples=True),
        dict(violating_samples=3),
        dict(violating_samples=0),
        dict(lower_peak=[-0.25, 0.0]),
        dict(lower_peak=(0.25, 0.0)),
        dict(upper_peak=(-0.5, 0.01)),
        dict(upper_peak=(float("inf"), 0.01)),
        dict(violations_per_joint=(True, 1)),
        dict(violations_per_joint=(-1, 1)),
        dict(violations_per_joint=(3, 1)),
        dict(violations_per_joint=(1, 1)),
        dict(violations_per_joint=(2, 0)),
        dict(upper_peak=(0.5, 0.0)),
    ],
)
def test_should_refuse_malformed_or_incoherent_summary_without_coercion(changes: dict[str, Any]):
    with pytest.raises(ValueError):
        copy_joint_position_summary(replace(summary(), **changes))


@pytest.mark.parametrize("value", [None, {}, "summary"])
def test_should_require_the_exact_summary_record_type(value: Any):
    with pytest.raises(ValueError):
        copy_joint_position_summary(value)


@pytest.mark.parametrize(
    "samples",
    [
        [(-1.25, 0.05), (1.5, 0.02)],
        ((-1.25, 0.05),),
        ((-1.0, 0.05), (1.5, 0.02)),
        ((-1.25, 0.05), (1.5,)),
        ((-1.25, 0.05), (1, 0.02)),
        ((-1.25, 0.05), (float("nan"), 0.02)),
        ((0.0, 0.02),) * 52,
    ],
)
def test_should_refuse_wrong_raw_trace_shape_values_count_or_peaks(samples: Any):
    value = summary()
    before = asdict(value)
    with pytest.raises(ValueError):
        check_joint_position_samples(value, samples)
    assert asdict(value) == before


def test_should_copy_valid_partial_work_and_complete_bounded_work():
    for value in (
        port.ManiSkillJointPositionWork(0, 0, 0, 0, 0, 0),
        port.ManiSkillJointPositionWork(0, 0, 0, 0, 1, 0),
        port.ManiSkillJointPositionWork(0, 0, 0, 0, 1, 1),
        port.ManiSkillJointPositionWork(1, 0, 0, 0, 1, 1),
        port.ManiSkillJointPositionWork(1, 1, 0, 0, 1, 1),
        joint_work(1),
        joint_work(51),
    ):
        owned = port.validate_joint_position_work(value)
        assert owned == value and owned is not value
    with pytest.raises(ValueError):
        port.validate_joint_position_work(cast(Any, asdict(joint_work())))


@pytest.mark.parametrize(
    "changes",
    [
        dict(limit_read_attempts=True),
        dict(limit_read_returns=1.0),
        dict(limit_read_attempts=2),
        dict(limit_read_attempts=0),
        dict(name_read_attempts=0),
        dict(name_read_returns=0),
        dict(samples=52, sampled_coordinates=468),
        dict(samples=True),
        dict(sampled_coordinates=26),
        dict(sampled_coordinates=-1),
    ],
)
def test_should_refuse_untyped_unbounded_or_impossible_retained_joint_work(changes: dict[str, Any]):
    with pytest.raises(ValueError):
        port.validate_joint_position_work(replace(joint_work(), **changes))


def test_should_bound_failed_run_sample_claims_by_reset_plus_returned_controls():
    class InflatedPartialPort:
        joint_position_work = joint_work(2)

        @property
        def joint_position_summary(self):
            pytest.fail("impossible sample cost reached summary read")

    result = dict(
        joint_position_exact=None,
        joint_position_work=asdict(joint_work(1)),
        work=asdict(port.ManiSkillWork(1, 0, 0, 5)),
    )
    with pytest.raises(ValueError, match="reset plus returned controls"):
        smoke._read_joint_position_report(InflatedPartialPort(), result)
    assert result["joint_position_work"] == asdict(joint_work(2))
    assert result["joint_position_exact"] is None


@pytest.fixture(scope="module")
def source():
    spec = ContinuousActionSpec(42, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(spec, hidden_dim=2, seed=17))
    initial = model.snapshot_state()
    parameters = tuple(np.zeros_like(value) for value in initial.parameters)
    parameters[3][:] = (0.2, -0.3, 0.4, -0.5)
    model.restore_state(
        replace(
            initial,
            parameters=(parameters[0], parameters[1], parameters[2], parameters[3]),
            velocity=(
                np.full_like(initial.velocity[0], 0.125),
                np.full_like(initial.velocity[1], 0.125),
                np.full_like(initial.velocity[2], 0.125),
                np.full_like(initial.velocity[3], 0.125),
            ),
            completed_updates=13,
            training_examples=377,
        )
    )
    return model


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment, calls = StateEnvironment(), []

    def factory(**options):
        calls.append(options)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-joint-rollout")
    yield environment, calls, tmp_path
    if calls and environment.close_calls == 0:
        environment.close()


def test_should_publish_exact_joint_trace_and_keep_all_frozen_model_arrays(boundary, source):
    environment, calls, directory = boundary
    before = source.snapshot_state()
    result = smoke.run_frozen_rollout(
        directory / "joint.json", learner=source, monitor_joint_positions=True
    )
    report = result.report
    assert report["complete"] is True and report["joint_position_exact"] is True
    assert report["joint_position_work"] == asdict(joint_work())
    observed = report["joint_position_summary"]
    assert observed["samples"] == 3 and observed["violating_samples"] == 2
    assert tuple(observed["violations_per_joint"]) == (2, 0, 0, 0, 0, 0, 0, 2, 0)
    assert observed["lower_peak"][0] == -0.25 and observed["upper_peak"][0] == 0.5
    assert (
        report["joint_position_validation_samples"] == 3
        and report["joint_position_validation_coordinates"] == 27
    )
    assert (
        report["prediction_attempts"]
        == report["prediction_returns"]
        == report["validated_prediction_rows"]
        == 2
    )
    assert report["source_updates"] == 13 and report["source_training_rows"] == 377
    assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    assert result.corrections is None and environment.close_calls == 1 and calls == [{}]
    expected = np.tanh(np.array([[0.2, -0.3, 0.4, -0.5]])).astype(np.float32)
    for command in environment.commands:
        np.testing.assert_array_equal(command, expected)
    assert json.loads((directory / "joint.json").read_text()) == json.loads(json.dumps(report))
    assert_same_state(before, source.snapshot_state())


def test_should_keep_default_calls_and_reports_free_of_joint_telemetry_access(
    boundary, source, monkeypatch
):
    environment, calls, directory = boundary

    def forbidden(_owner):
        pytest.fail("default frozen rollout accessed joint telemetry")

    monkeypatch.setattr(
        port.ManiSkillPickCubeEpisode, "joint_position_summary", property(forbidden)
    )
    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "joint_position_work", property(forbidden))
    original = port.ManiSkillPickCubeEpisode.__init__
    options = []

    def initialize(owner, *args, **kwargs):
        options.append(kwargs)
        original(owner, *args, **kwargs)

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "__init__", initialize)
    report = smoke.run_frozen_rollout(directory / "default.json", learner=source).report
    assert report["complete"] is True and len(environment.commands) == 2 and calls == [{}]
    assert not any(name.startswith("joint_position_") for name in report)
    assert len(options) == 1 and "monitor_joint_positions" not in options[0]


@pytest.mark.parametrize("bad", [None, 0, 1, "yes"])
def test_should_refuse_untyped_monitor_flag_before_scene_or_prediction(boundary, source, bad: Any):
    _, calls, directory = boundary
    report = smoke.run_frozen_rollout(
        directory / "bad-flag.json", learner=source, monitor_joint_positions=bad
    ).report
    assert (
        report["complete"] is False
        and report["creation_attempts"] == report["prediction_attempts"] == 0
    )
    assert calls == []


@pytest.mark.parametrize("failure", ["constructor", "step", "reward", "cleanup"])
def test_should_retain_valid_partial_joint_costs_without_claiming_complete_trace(
    boundary, source, failure
):
    environment, _, directory = boundary
    if failure == "constructor":
        environment.agent.robot.limit_error = RuntimeError("joint limits unavailable")
    elif failure == "step":
        environment.step_error = RuntimeError("native step unavailable")
    elif failure == "reward":
        environment.bad_reward = np.array([np.nan], dtype=np.float32)
    else:
        environment.close_error = RuntimeError("cleanup unavailable")
    before = source.snapshot_state()
    report = smoke.run_frozen_rollout(
        directory / "partial.json", learner=source, monitor_joint_positions=True
    ).report
    assert report["complete"] is False and report["joint_position_exact"] is not True
    assert environment.close_calls == 1
    expected_samples = dict(constructor=0, step=1, reward=2, cleanup=3)[failure]
    observed = report["joint_position_work"]
    assert (
        observed["samples"] == expected_samples
        and observed["sampled_coordinates"] == 9 * expected_samples
    )
    assert observed["limit_read_attempts"] == 1 and observed["limit_read_returns"] == int(
        failure != "constructor"
    )
    predictions = dict(constructor=0, step=1, reward=1, cleanup=2)[failure]
    assert report["prediction_attempts"] == report["prediction_returns"] == predictions
    assert len(environment.commands) == predictions
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize("damage", ["bounds", "peaks", "work"])
def test_should_refuse_late_valid_metadata_that_no_longer_matches_initial_bounds_or_trace(
    boundary, source, monkeypatch, damage
):
    environment, _, directory = boundary
    name = "joint_position_work" if damage == "work" else "joint_position_summary"
    original = vars(port.ManiSkillPickCubeEpisode)[name].fget

    def getter(owner):
        value = original(owner)
        if environment.close_calls:
            if damage == "bounds":
                return replace(
                    value, limits=replace(value.limits, upper=(11.0,) + value.limits.upper[1:])
                )
            if damage == "peaks":
                return replace(value, upper_peak=(0.75,) + value.upper_peak[1:])
            return replace(value, samples=2, sampled_coordinates=18)
        return value

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, name, property(getter))
    before = source.snapshot_state()
    result = smoke.run_frozen_rollout(
        directory / "late.json", learner=source, monitor_joint_positions=True
    )
    assert result.report["complete"] is False and result.report["joint_position_exact"] is not True
    assert result.corrections is None and len(environment.commands) == 2
    assert result.report["prediction_returns"] == 2
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize("kind", ["summary", "work"])
def test_should_refuse_equal_valued_wrong_native_counter_types_before_prediction(
    boundary, source, monkeypatch, kind
):
    environment, _, directory = boundary
    name = "joint_position_" + kind
    original = vars(port.ManiSkillPickCubeEpisode)[name].fget
    monkeypatch.setattr(
        port.ManiSkillPickCubeEpisode,
        name,
        property(lambda owner: replace(original(owner), samples=True)),
    )
    report = smoke.run_frozen_rollout(
        directory / "wrong-type.json", learner=source, monitor_joint_positions=True
    ).report
    assert report["complete"] is False and report["joint_position_exact"] is not True
    assert report["prediction_attempts"] == 0 and environment.commands == []


def test_should_reconstruct_from_the_retained_raw_episode_instead_of_trusting_monitor_totals(
    boundary, source, monkeypatch
):
    environment, _, directory = boundary
    original = port.ManiSkillPickCubeEpisode.episode

    def episode(owner):
        value = original(owner)
        first = value.transitions[0]
        return replace(
            value,
            transitions=(replace(first, observation=(99.0,) + first.observation[1:]),)
            + value.transitions[1:],
        )

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "episode", episode)
    report = smoke.run_frozen_rollout(
        directory / "trace.json", learner=source, monitor_joint_positions=True
    ).report
    assert report["complete"] is False and report["joint_position_exact"] is not True
    assert report["prediction_returns"] == len(environment.commands) == 2


def test_should_keep_increased_paid_work_when_final_summary_getter_throws(
    boundary, source, monkeypatch
):
    environment, _, directory = boundary
    original = vars(port.ManiSkillPickCubeEpisode)["joint_position_summary"].fget

    def getter(owner):
        if environment.close_calls:
            raise ValueError("final joint summary unavailable")
        return original(owner)

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "joint_position_summary", property(getter))
    before = source.snapshot_state()
    report = smoke.run_frozen_rollout(
        directory / "throwing-summary.json", learner=source, monitor_joint_positions=True
    ).report
    assert report["complete"] is False and report["joint_position_exact"] is not True
    assert report["joint_position_work"] == asdict(joint_work())
    assert "final joint summary unavailable" in report["joint_position_error"]
    assert report["prediction_returns"] == len(environment.commands) == 2
    assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    assert_same_state(before, source.snapshot_state())


def test_should_check_frozen_source_after_final_telemetry_getter_callbacks(
    boundary, source, monkeypatch
):
    environment, _, directory = boundary
    before = source.snapshot_state()
    original = vars(port.ManiSkillPickCubeEpisode)["joint_position_summary"].fget
    mutated: list[bool] = []

    def getter(owner):
        value = original(owner)
        if environment.close_calls and not mutated:
            getattr(source, "_state").parameters[0][0, 0] = 99.0
            mutated.append(True)
        return value

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "joint_position_summary", property(getter))
    try:
        result = smoke.run_frozen_rollout(
            directory / "callback.json", learner=source, monitor_joint_positions=True
        )
        assert mutated == [True] and result.report["complete"] is False
        assert result.report["joint_position_exact"] is not True and result.corrections is None
        assert result.report["original_owner_exact"] is not True
        assert len(environment.commands) == result.report["prediction_returns"] == 2
    finally:
        source.restore_state(before)
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize("projected", [False, True])
def test_should_serve_actual_rgb_actor_with_owned_joint_report_and_unchanged_prototypes(
    tmp_path, monkeypatch, projected
):
    calls, directory = [], tmp_path
    environment = RGBEnvironment()
    environment.agent.robot = Robot([])

    def factory(**options):
        calls.append(options)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-joint-rgb")
    dimension = 21 if projected else 789
    spec = ContinuousActionSpec(dimension, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    policy = NearestContinuousBCPolicy(
        spec, ObservationStandardization((0.0,) * dimension, (1.0,) * dimension, 2)
    )
    owner = NumpyNearestContinuousBCLearner(policy)
    features = np.asarray(
        [public_feature_oracle(packet(index)) for index in (0, 1)], dtype=np.float64
    )
    features = features[:, :dimension].copy()
    actions = np.array([[0.125, -0.25, 0.5, -1.0], [-0.5, 0.25, 0.0, 1.0]], dtype=np.float64)
    owner.restore_state(NearestContinuousBCSnapshot(policy, features, actions, 1, 2))
    initial = owner.snapshot_state()
    actor = StableActor(owner, version="joint-actor-0")
    report = smoke.run_frozen_rollout(
        directory / "rgb.json",
        actor=actor,
        visual_observation=True,
        monitor_joint_positions=True,
        observation_projection=public_vector_projection() if projected else None,
    ).report
    assert report["complete"] is True and report["joint_position_exact"] is True
    assert report["joint_position_work"] == asdict(joint_work())
    assert report["joint_position_summary"]["violating_samples"] == 3
    assert report["episode"]["spec"]["observation_dim"] == 789
    assert (
        report["joint_position_validation_samples"] == 3
        and report["joint_position_validation_coordinates"] == 27
    )
    if projected:
        assert (
            report["projection_exact"] is True
            and report["observation_projection"]["feature_dim"] == 21
        )
    else:
        assert "observation_projection" not in report
    assert report["actor_prediction_attempts"] == report["actor_prediction_returns"] == 2
    assert report["actor_version_exact"] is report["actor_backend_exact"] is True
    assert report["frozen_state_exact"] is report["original_owner_exact"] is True
    for command, action in zip(environment.commands, actions, strict=True):
        np.testing.assert_array_equal(command[0], action.astype(np.float32))
    for observed in (owner.snapshot_state(), actor.snapshot_state().state):
        assert (
            observed.policy == initial.policy
            and observed.training_examples == 2
            and observed.completed_updates == 1
        )
        np.testing.assert_array_equal(observed.prototype_features, initial.prototype_features)
        np.testing.assert_array_equal(observed.prototype_actions, initial.prototype_actions)
    assert environment.close_calls == 1


def test_should_forward_cli_opt_in_and_preserve_exclusive_output(boundary, source, monkeypatch):
    environment, calls, directory = boundary
    output = directory / "cli.json"
    monkeypatch.setattr(
        sys, "argv", ["smoke", "--output", str(output), "--monitor-joint-positions"]
    )
    assert smoke.main(learner=source) == 0
    report = json.loads(output.read_text())
    assert report["joint_position_exact"] is True and report["joint_position_work"] == asdict(
        joint_work()
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.run_frozen_rollout(output, learner=source, monitor_joint_positions=True)
    assert output.read_bytes() == before and len(calls) == environment.close_calls == 1


def test_should_revalidate_owned_mass_binding_after_final_joint_summary_callback(
    boundary, source, monkeypatch
):
    environment, _, directory = boundary
    environment.cube = FakeCube(environment.events)
    original = vars(port.ManiSkillPickCubeEpisode)["joint_position_summary"].fget
    before = source.snapshot_state()
    mutated: list[bool] = []

    def getter(owner):
        value = original(owner)
        if environment.close_calls and not mutated:
            setattr(owner, "_mass_applied_mass", 0.5)
            mutated.append(True)
        return value

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "joint_position_summary", property(getter))
    result = smoke.run_frozen_rollout(
        directory / "mass-joints.json",
        learner=source,
        mass_context=pick_cube_mass_context(),
        monitor_joint_positions=True,
    )
    report = result.report
    assert mutated == [True] and report["complete"] is False
    assert report["mass_context_exact"] is None and report["joint_position_exact"] is None
    assert report["mass_error"] and result.corrections is None
    assert (
        report["prediction_returns"]
        == len(environment.commands)
        == report["work"]["completed_steps"]
        == 2
    )
    assert report["joint_position_work"] == asdict(joint_work())
    assert report["joint_position_summary"]["samples"] == 3
    assert report["mass_work"]["read_returns"] == environment.cube.reads == 8
    assert len(environment.cube.writes) == 1 and environment.close_calls == 1
    assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    assert_same_state(before, source.snapshot_state())


def test_should_withhold_an_actually_formed_corrective_batch_after_final_joint_reporting_failure(
    boundary, source, monkeypatch
):
    environment, _, directory = boundary
    original_summary = vars(port.ManiSkillPickCubeEpisode)["joint_position_summary"].fget
    original_finish = queries.PickCubeCorrectionQuery.finish
    formed, geometry_calls = [], []

    def geometry(_owner):
        geometry_calls.append(True)
        return PickCubeGeometry((0.0,) * 3, (0.05, 0.0, 0.02), (0.1, 0.0, 0.2))

    def finish(owner, episode):
        batch = original_finish(owner, episode)
        formed.append(batch)
        return batch

    def summary_getter(owner):
        if environment.close_calls:
            raise ValueError("joint report failed after corrective batch formation")
        return original_summary(owner)

    monkeypatch.setattr(queries, "_geometry", geometry)
    monkeypatch.setattr(queries.PickCubeCorrectionQuery, "finish", finish)
    monkeypatch.setattr(
        port.ManiSkillPickCubeEpisode, "joint_position_summary", property(summary_getter)
    )
    before = source.snapshot_state()
    result = smoke.run_frozen_rollout(
        directory / "corrective-joints.json",
        learner=source,
        corrective_queries=True,
        monitor_joint_positions=True,
    )
    report = result.report
    assert len(formed) == 1 and len(formed[0].labels) == len(geometry_calls) == 2
    assert report["complete"] is False and result.corrections is None
    assert report["joint_position_exact"] is None and report["joint_position_work"] == asdict(
        joint_work()
    )
    assert len(report["corrections"]["labels"]) == report["corrections"]["geometry_returns"] == 2
    assert report["prediction_returns"] == len(environment.commands) == 2
    assert report["optimizer_updates"] == 0 and environment.close_calls == 1
    assert_same_state(before, source.snapshot_state())


def test_should_reconstruct_native_joint_trace_while_clocked_model_predicts_43_coordinates(
    boundary, monkeypatch
):
    environment, _, directory = boundary
    spec = ContinuousActionSpec(43, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(spec, hidden_dim=2, seed=17))
    initial = model.snapshot_state()
    parameters = tuple(np.zeros_like(value) for value in initial.parameters)
    parameters[3][:] = (0.2, -0.3, 0.4, -0.5)
    model.restore_state(
        replace(
            initial,
            parameters=(parameters[0], parameters[1], parameters[2], parameters[3]),
            velocity=(
                np.full_like(initial.velocity[0], 0.125),
                np.full_like(initial.velocity[1], 0.125),
                np.full_like(initial.velocity[2], 0.125),
                np.full_like(initial.velocity[3], 0.125),
            ),
        )
    )
    before = model.snapshot_state()
    original_predict = NumpyContinuousBCLearner.predict
    observed = []

    def predict(owner, features):
        observed.append(features.copy())
        return original_predict(owner, features)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", predict)
    report = smoke.run_frozen_rollout(
        directory / "clocked-joints.json",
        learner=model,
        observation_clock=EpisodeClock(50, True),
        monitor_joint_positions=True,
    ).report
    assert report["complete"] is True and report["joint_position_exact"] is True
    assert report["episode"]["spec"]["observation_dim"] == 42
    assert (
        report["joint_position_validation_samples"] == 3
        and report["joint_position_validation_coordinates"] == 27
    )
    assert report["joint_position_work"] == asdict(joint_work())
    assert report["clock_values"] == [0.0, 0.02] and len(observed) == 2
    assert [value.shape for value in observed] == [(1, 43), (1, 43)]
    assert [float(value[0, -1]) for value in observed] == [0.0, 0.02]
    assert report["prediction_returns"] == len(environment.commands) == 2
    assert_same_state(before, model.snapshot_state())
