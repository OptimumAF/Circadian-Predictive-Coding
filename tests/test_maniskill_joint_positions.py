"""Read-only joint telemetry from deterministic fake state/RGB IO; no models.

Per full module: 35 fake scenes/reset/close attempts, 32 name-access calls,
25 limit-access calls, 66 controls, and 72 accepted samples (648 coordinates).
These 228 boundary calls include failed accessor/cleanup attempts. No extra qpos
query, simulator setter, learner, fitting, RNG or native execution is admitted.
"""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.infra import maniskill_continuous as port
from test_maniskill_continuous import FakeEnvironment, ForeignArray
from test_maniskill_rgb_observations import RGBEnvironment, public_feature_oracle


NAMES = tuple(f"panda_joint{index}" for index in range(1, 8)) + (
    "panda_finger_joint1",
    "panda_finger_joint2",
)
ACTION = (0.125, -0.25, 0.5, -1.0)


def native_limits(dtype=np.float64):
    return np.array([[[-1.0, 1.0]] * 7 + [[0.0, 0.04]] * 2], dtype=dtype)


def state(index, dtype=np.float64):
    qpos = [0.0] * 7 + [0.02, 0.02]
    if index == 1:
        qpos[0], qpos[7] = -1.25, 0.05
    elif index == 2:
        qpos[0], qpos[7] = 1.5, -0.02
    return np.asarray([qpos + [1000.0 + value for value in range(33)]], dtype=dtype)


class Robot:
    def __init__(self, events, dtype=np.float64):
        self.events = events
        self.limits: Any = native_limits(dtype)
        self.joints: Any = [SimpleNamespace(name=name) for name in NAMES]
        self.names_error: Any = None
        self.limit_error: Any = None
        self.after_names: Any = None
        self.after_limits: Any = None

    def get_active_joints(self):
        self.events.append("names")
        if self.names_error is not None:
            raise self.names_error
        if self.after_names is not None:
            self.after_names()
        return self.joints

    def get_qlimits(self):
        self.events.append("limits")
        if self.limit_error is not None:
            raise self.limit_error
        if self.after_limits is not None:
            self.after_limits()
        return self.limits

    def get_qpos(self):
        pytest.fail("telemetry made an extra qpos query")

    def set_qpos(self, *_args):
        pytest.fail("telemetry changed joint positions")


class StateEnvironment(FakeEnvironment):
    def __init__(self, dtype=np.float64):
        super().__init__()
        self.events: list[str] = []
        self.dtype = dtype
        self.agent.robot = Robot(self.events, dtype)
        self.reset_observation = state(0, dtype)
        self.next_state: Any = None
        self.after_step: Any = None
        self.bad_reward: Any = None
        self.close_error: Any = None

    def reset(self, *, seed):
        self.events.append("reset")
        return super().reset(seed=seed)

    def step(self, command):
        returned = super().step(command)
        self.last_observation = (
            state(len(self.commands), self.dtype) if self.next_state is None else self.next_state
        )
        if self.after_step is not None:
            self.after_step()
        return (
            self.last_observation,
            returned[1] if self.bad_reward is None else self.bad_reward,
            *returned[2:],
        )

    def close(self):
        super().close()
        if self.close_error is not None:
            raise self.close_error


def install(monkeypatch, environment):
    calls = []

    def factory(**options):
        calls.append(options)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    return calls


def open_episode(**options):
    return port.ManiSkillPickCubeEpisode("joint-test", "pick-cube", seed=17, **options)


def work(session, reads, returns, samples, names=1, name_returns=1):
    assert asdict(session.joint_position_work) == dict(
        limit_read_attempts=reads,
        limit_read_returns=returns,
        samples=samples,
        sampled_coordinates=9 * samples,
        name_read_attempts=names,
        name_read_returns=name_returns,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_should_sample_reset_and_each_state_without_changing_features_or_commands(
    monkeypatch, dtype
):
    environment = StateEnvironment(dtype)
    calls = install(monkeypatch, environment)
    robot = environment.agent.robot
    session = open_episode(monitor_joint_positions=True)
    try:
        assert environment.events == ["reset", "names", "limits"]
        assert session.observation == tuple(
            float(value) for value in environment.reset_observation[0]
        )
        for _ in range(2):
            record = session.step(ACTION)
            assert environment.last_observation is not None
            assert record.next_observation == tuple(
                float(value) for value in environment.last_observation[0]
            )
        episode = session.episode()
        report = session.joint_position_summary
        assert report.samples == 3 and report.violating_samples == 2
        assert report.violations_per_joint == (2, 0, 0, 0, 0, 0, 0, 2, 0)
        assert report.lower_peak[0] == -0.25 and report.upper_peak[0] == 0.5
        assert report.lower_peak[7] == pytest.approx(-0.02) and report.upper_peak[
            7
        ] == pytest.approx(0.01)
        assert report.limits.names == NAMES and report.limits.units == ("rad",) * 7 + ("m",) * 2
        assert episode.spec.observation_dim == 42 and episode.transitions[-1].terminated is True
        assert episode.transitions[-1].reward == 37.5
        work(session, 1, 1, 3)
        assert environment.events.count("names") == environment.events.count("limits") == 1
        for command in environment.commands:
            np.testing.assert_array_equal(command, np.array([ACTION], dtype=np.float32))
        robot.limits.fill(99.0)
        assert session.joint_position_summary.limits.lower == (-1.0,) * 7 + (0.0,) * 2
    finally:
        session.close()
    assert (
        session.joint_position_summary == report and environment.close_calls == 1 and calls == [{}]
    )


def test_should_sample_the_same_rgb_prefix_and_preserve_full_public_pixels(monkeypatch):
    environment = RGBEnvironment()
    environment.agent.robot = Robot([])
    calls = install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True, visual_observation=True)
    try:
        assert session.observation == tuple(public_feature_oracle(environment.reset_packet))
        for _ in range(2):
            record = session.step(ACTION)
            assert record.next_observation == tuple(public_feature_oracle(environment.last_packet))
        summary = session.joint_position_summary
        assert summary.samples == summary.violating_samples == 3
        assert summary.upper_peak[0] == 1.0 and summary.violations_per_joint[0] == 1
        assert summary.upper_peak[8] == pytest.approx(10.0 - 0.04)
        assert summary.violations_per_joint[8] == 3
        assert session.episode().spec.observation_dim == 789
        work(session, 1, 1, 3)
        assert session.visual_work.observation_returns == 3
        assert calls == [dict(visual_observation=True)]
    finally:
        session.close()


@pytest.mark.parametrize("visual", [False, True])
def test_should_leave_default_telemetry_absent_without_accessing_a_robot(monkeypatch, visual):
    class ForbiddenAgent:
        uid = "panda"

        @property
        def robot(self):
            pytest.fail("default port accessed telemetry robot")

    environment = RGBEnvironment() if visual else StateEnvironment()
    setattr(environment, "agent", ForbiddenAgent())
    install(monkeypatch, environment)
    session = open_episode(visual_observation=visual)
    try:
        assert session.joint_position_summary is session.joint_position_work is None
        session.step(ACTION)
        session.step(ACTION)
        assert len(session.episode().transitions) == 2
    finally:
        session.close()


@pytest.mark.parametrize("bad", [None, 0, 1, "yes"])
def test_should_require_exact_opt_in_before_environment_creation(monkeypatch, bad: Any):
    monkeypatch.setattr(
        port,
        "_make_environment",
        lambda **_options: pytest.fail("invalid telemetry flag created scene"),
    )
    with pytest.raises(ValueError):
        open_episode(monitor_joint_positions=bad)


@pytest.mark.parametrize(
    "damage", ["shape", "integer", "float16", "subclass", "nan", "reversed", "equal"]
)
def test_should_retain_returned_limit_read_cost_for_malformed_native_arrays(monkeypatch, damage):
    environment = StateEnvironment()
    robot = environment.agent.robot
    if damage == "shape":
        robot.limits = robot.limits[0]
    elif damage == "integer":
        robot.limits = robot.limits.astype(np.int64)
    elif damage == "float16":
        robot.limits = robot.limits.astype(np.float16)
    elif damage == "subclass":
        robot.limits = robot.limits.view(ForeignArray)
    elif damage == "nan":
        robot.limits[0, 0, 0] = np.nan
    else:
        robot.limits[0, 0] = (2.0, 1.0) if damage == "reversed" else (1.0, 1.0)
    install(monkeypatch, environment)
    with pytest.raises(ValueError) as caught:
        open_episode(monitor_joint_positions=True)
    failure = getattr(caught.value, "joint_position_work")
    assert asdict(failure) == dict(
        limit_read_attempts=1,
        limit_read_returns=1,
        samples=0,
        sampled_coordinates=0,
        name_read_attempts=1,
        name_read_returns=1,
    )
    assert environment.close_calls == 1 and environment.commands == []


@pytest.mark.parametrize(
    "damage", ["raises", "tuple", "eight", "order", "missing_name", "wrong_name"]
)
def test_should_record_name_read_cost_and_refuse_before_limits(monkeypatch, damage):
    environment = StateEnvironment()
    robot = environment.agent.robot
    if damage == "raises":
        robot.names_error = RuntimeError("names unavailable")
    elif damage == "tuple":
        robot.joints = tuple(robot.joints)
    elif damage == "eight":
        robot.joints = robot.joints[:8]
    elif damage == "order":
        robot.joints.reverse()
    elif damage == "missing_name":
        robot.joints[0] = SimpleNamespace()
    else:
        robot.joints[0].name = "other_joint"
    install(monkeypatch, environment)
    with pytest.raises((ValueError, AttributeError, RuntimeError)) as caught:
        open_episode(monitor_joint_positions=True)
    failure = getattr(caught.value, "joint_position_work")
    assert failure.name_read_attempts == 1 and failure.name_read_returns == int(damage != "raises")
    assert failure.limit_read_attempts == failure.limit_read_returns == failure.samples == 0
    assert environment.events == ["reset", "names"] and environment.close_calls == 1


@pytest.mark.parametrize("close_fails", [False, True])
def test_should_preserve_original_native_limit_exception_and_paid_partial_work(
    monkeypatch, close_fails
):
    environment = StateEnvironment()
    original = RuntimeError("limits unavailable")
    environment.agent.robot.limit_error = original
    if close_fails:
        environment.close_error = RuntimeError("cleanup unavailable")
    install(monkeypatch, environment)
    with pytest.raises(RuntimeError, match="limits unavailable") as caught:
        open_episode(monitor_joint_positions=True)
    assert caught.value is original
    receipt = getattr(original, "joint_position_work")
    assert receipt.limit_read_attempts == 1 and receipt.limit_read_returns == receipt.samples == 0
    assert receipt.name_read_attempts == receipt.name_read_returns == 1
    assert bool(getattr(original, "joint_position_cleanup_error", None)) is close_fails
    assert environment.close_calls == 1


@pytest.mark.parametrize("phase", ["names", "limits"])
def test_should_refuse_robot_replaced_during_an_external_accessor(monkeypatch, phase):
    environment = StateEnvironment()
    robot = environment.agent.robot
    setattr(robot, "after_" + phase, lambda: setattr(environment.agent, "robot", Robot([])))
    install(monkeypatch, environment)
    with pytest.raises(ValueError) as caught:
        open_episode(monitor_joint_positions=True)
    receipt = getattr(caught.value, "joint_position_work")
    assert receipt.limit_read_attempts == receipt.limit_read_returns == int(phase == "limits")
    assert receipt.name_read_attempts == receipt.name_read_returns == 1
    assert receipt.samples == 0 and environment.close_calls == 1


def test_should_permanently_refuse_replaced_robot_before_sending_control(monkeypatch):
    environment = StateEnvironment()
    robot = environment.agent.robot
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)
    environment.agent.robot = Robot([])
    try:
        with pytest.raises(ValueError):
            session.step(ACTION)
        environment.agent.robot = robot
        with pytest.raises(ValueError):
            session.step(ACTION)
        with pytest.raises(ValueError):
            session.episode()
        assert environment.commands == [] and session.work.step_attempts == 0
        work(session, 1, 1, 1)
    finally:
        session.close()


def test_should_preserve_paid_control_when_robot_changes_before_next_encoding(monkeypatch):
    environment = StateEnvironment()
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)
    environment.after_step = lambda: setattr(environment.agent, "robot", Robot([]))
    try:
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert (
            session.work.step_attempts
            == session.work.completed_steps
            == len(environment.commands)
            == 1
        )
        work(session, 1, 1, 1)
        with pytest.raises(ValueError):
            session.episode()
    finally:
        session.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_should_keep_prior_samples_on_nonfinite_native_observation(monkeypatch, value):
    environment = StateEnvironment()
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)
    before = session.joint_position_summary
    environment.next_state = state(1)
    environment.next_state[0, 0] = value
    try:
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert session.joint_position_summary == before
        assert session.work.completed_steps == 1
        work(session, 1, 1, 1)
    finally:
        session.close()


def test_should_retain_a_sampled_encoding_even_when_later_reward_is_invalid(monkeypatch):
    environment = StateEnvironment()
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)
    environment.bad_reward = np.array([np.nan], dtype=np.float32)
    try:
        with pytest.raises(ValueError):
            session.step(ACTION)
        work(session, 1, 1, 2)
        assert session.joint_position_summary.violating_samples == 1
        assert session.work.completed_steps == 1 and len(environment.commands) == 1
        with pytest.raises(ValueError):
            session.episode()
    finally:
        session.close()


def test_should_retain_limit_work_without_a_sample_for_wrong_reset_dimension(monkeypatch):
    environment = StateEnvironment()
    environment.reset_observation = np.zeros((1, 3), dtype=np.float64)
    install(monkeypatch, environment)
    with pytest.raises(ValueError) as caught:
        open_episode(monitor_joint_positions=True)
    receipt = getattr(caught.value, "joint_position_work")
    assert receipt.limit_read_attempts == receipt.limit_read_returns == 1
    assert receipt.samples == receipt.sampled_coordinates == 0 and environment.close_calls == 1


def test_should_collect_exactly_reset_plus_50_encodings_at_official_horizon(monkeypatch):
    environment = StateEnvironment()
    environment.steps = 50
    environment.terminated, environment.truncated = False, True
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)
    try:
        for _ in range(50):
            session.step(ACTION)
        assert len(session.episode().transitions) == 50
        assert session.episode().transitions[-1].truncated is True
        work(session, 1, 1, 51)
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert len(environment.commands) == 50
        work(session, 1, 1, 51)
    finally:
        session.close()


def test_should_refuse_robot_replacement_during_native_limit_conversion(monkeypatch):
    environment = StateEnvironment()
    robot = environment.agent.robot
    original_array = port._array

    def array(value):
        converted = original_array(value)
        if value is robot.limits:
            environment.agent.robot = Robot([])
        return converted

    install(monkeypatch, environment)
    monkeypatch.setattr(port, "_array", array)
    with pytest.raises(ValueError) as caught:
        open_episode(monitor_joint_positions=True)
    receipt = getattr(caught.value, "joint_position_work")
    assert receipt.limit_read_attempts == receipt.limit_read_returns == 1
    assert receipt.samples == 0 and environment.close_calls == 1


@pytest.mark.parametrize("damage", ["robot", "admission"])
def test_should_recheck_after_other_callbacks_immediately_before_control(monkeypatch, damage):
    environment = StateEnvironment()
    robot = environment.agent.robot
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)

    def callback():
        if damage == "robot":
            environment.agent.robot = Robot([])
        else:
            setattr(session, "_monitor_joint_positions", False)

    monkeypatch.setattr(session, "_check_lighting_binding", callback)
    try:
        with pytest.raises(ValueError):
            session.step(ACTION)
        monkeypatch.setattr(session, "_check_lighting_binding", lambda: None)
        environment.agent.robot = robot
        setattr(session, "_monitor_joint_positions", True)
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert environment.commands == [] and session.work.step_attempts == 0
        work(session, 1, 1, 1)
    finally:
        session.close()


def test_should_refuse_adding_telemetry_to_an_unmonitored_session_late(monkeypatch):
    environment = StateEnvironment()
    install(monkeypatch, environment)
    session = open_episode()
    monkeypatch.setattr(
        session, "_check_lighting_binding", lambda: setattr(session, "_joint_positions", object())
    )
    try:
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert environment.commands == [] and session.joint_position_work is None
        assert environment.events == ["reset"]
    finally:
        session.close()


def test_should_recheck_robot_after_all_callbacks_before_episode_export(monkeypatch):
    environment = StateEnvironment()
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)
    try:
        session.step(ACTION)
        session.step(ACTION)
        monkeypatch.setattr(
            session,
            "_check_lighting_binding",
            lambda: setattr(environment.agent, "robot", Robot([])),
        )
        with pytest.raises(ValueError):
            session.episode()
        assert len(environment.commands) == 2
        work(session, 1, 1, 3)
    finally:
        session.close()


def test_should_refuse_an_identity_getter_disabling_telemetry_while_returning_original_robot(
    monkeypatch,
):
    environment = StateEnvironment()
    original_agent = environment.agent
    robot = original_agent.robot
    install(monkeypatch, environment)
    session = open_episode(monitor_joint_positions=True)

    class MutatingAgent:
        uid = "panda"

        @property
        def robot(self):
            setattr(session, "_monitor_joint_positions", False)
            return robot

    setattr(environment, "agent", MutatingAgent())
    try:
        with pytest.raises(ValueError):
            session.step(ACTION)
        environment.agent = original_agent
        setattr(session, "_monitor_joint_positions", True)
        with pytest.raises(ValueError):
            session.step(ACTION)
        assert environment.commands == [] and session.work.step_attempts == 0
        work(session, 1, 1, 1)
    finally:
        session.close()
