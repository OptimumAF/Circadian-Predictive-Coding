"""Integration boundary doubles exercise the existing port, without native physics."""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as port_module
from src.infra import maniskill_task_teacher as collector


class RigidPose:
    """Independent rigid-transform double, including a rotated translated root."""

    def __init__(self, point, rotation=None):
        self.p = np.array([point], dtype=np.float64)
        self.rotation = np.eye(3) if rotation is None else np.array(rotation, dtype=np.float64)

    def inv(self):
        rotation = self.rotation.T
        return RigidPose(-(rotation @ self.p[0]), rotation)

    def __mul__(self, other):
        return RigidPose(self.rotation @ other.p[0] + self.p[0], self.rotation @ other.rotation)


class FakeEnvironment:
    def __init__(self):
        self.unwrapped = self
        self.num_envs = 1
        self.obs_mode = "state"
        self.control_mode = "pd_ee_delta_pos"
        self.device = SimpleNamespace(type="cpu")
        self.sim_freq = 100
        self.control_freq = 20
        self.single_action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32), high=np.ones(4, dtype=np.float32), shape=(4,)
        )
        root = RigidPose((2.0, -3.0, 1.0), ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        self.agent = SimpleNamespace(
            uid="panda",
            robot=SimpleNamespace(pose=root),
            tcp=SimpleNamespace(pose=root * RigidPose((0.0, 0.0, 0.0))),
        )
        self.cube = SimpleNamespace(pose=root * RigidPose((0.02, -0.03, 0.01)))
        self.goal_site = SimpleNamespace(pose=root * RigidPose((0.05, 0.02, 0.08)))
        self.reset_calls = []
        self.commands = []
        self.close_calls = 0
        self.done_at = 50
        self.terminated = False
        self.truncated = True
        self.rewards = {}
        self.default_reward = 1.0
        self.step_failure_at = None
        self.close_error = None
        self.reset_error = None

    def reset(self, *, seed):
        self.reset_calls.append(seed)
        if self.reset_error:
            raise self.reset_error
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32), {}

    def step(self, command):
        self.commands.append(command.copy())
        index = len(self.commands)
        if self.step_failure_at == index:
            raise RuntimeError("physics failed")
        return (
            np.array([[float(index), 0.0, 0.0]], dtype=np.float32),
            np.array([self.rewards.get(index, self.default_reward)], dtype=np.float32),
            np.array([index == self.done_at and self.terminated], dtype=np.bool_),
            np.array([index == self.done_at and self.truncated], dtype=np.bool_),
            {},
        )

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise self.close_error


@pytest.fixture
def environment(monkeypatch):
    value = FakeEnvironment()
    monkeypatch.setattr(port_module, "_make_environment", lambda: value)
    return value


def test_should_transform_all_three_world_points_into_rotated_robot_root(environment):
    session = port_module.ManiSkillPickCubeEpisode("transform-check", "pick-cube")
    try:
        actual = collector._geometry(session)
    finally:
        session.close()
    assert actual.tcp == pytest.approx((0.0, 0.0, 0.0))
    assert actual.cube == pytest.approx((0.02, -0.03, 0.01))
    assert actual.goal == pytest.approx((0.05, 0.02, 0.08))


def test_should_collect_fixed_train_seed_through_existing_port_and_account_all_work(environment):
    result = collector.collect_task_teacher()
    assert result.qualified and result.observed_success and result.final_success
    assert result.error is None and result.traceback is None
    assert environment.reset_calls == [17] and environment.close_calls == 1
    assert result.creation_attempts == 1
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 50
    assert (
        result.stages
        == ("align",) * 8 + ("descend",) * 8 + ("grasp",) * 6 + ("lift",) * 18 + ("settle",) * 10
    )
    assert result.work == port_module.ManiSkillWork(50, 50, 250, 0)
    episode = result.training_episode()
    assert episode.role == "train" and episode.context == "pick-cube"
    assert episode.episode_id == "pick-cube-task-teacher-17"
    assert episode is not result.candidate_episode
    assert len(episode.transitions) == len(environment.commands) == 50
    for transition, command in zip(episode.transitions, environment.commands, strict=True):
        assert command.dtype == np.float32 and command.shape == (1, 4)
        assert transition.action == tuple(float(item) for item in command[0])
    assert episode.transitions[0].action == pytest.approx((0.2, -0.3, 0.9, 1.0))
    assert episode.transitions[22].action == pytest.approx((0.3, 0.5, 0.7, -1.0))
    assert not episode.transitions[-1].terminated and episode.transitions[-1].truncated
    parameters = inspect.signature(collector.collect_task_teacher).parameters
    assert tuple(parameters) == (
        "train_seed",
        "visual_observation",
        "camera_context",
        "mass_context",
        "lighting_context",
    )
    assert parameters["train_seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["train_seed"].default == 17
    assert parameters["visual_observation"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["visual_observation"].default is False
    assert parameters["camera_context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["camera_context"].default is None
    assert parameters["lighting_context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["lighting_context"].default is None


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_should_preserve_early_completion_and_independent_native_flags(
    environment, terminated, truncated
):
    environment.done_at = 3
    environment.terminated, environment.truncated = terminated, truncated
    result = collector.collect_task_teacher()
    assert result.qualified and result.teacher_actions == 3
    assert result.geometry_attempts == result.geometry_returns == 3
    assert result.work == port_module.ManiSkillWork(3, 3, 15, 0)
    final = result.training_episode().transitions[-1]
    assert final.terminated is terminated and final.truncated is truncated


@pytest.mark.parametrize("earlier_success", [False, True])
def test_should_refuse_final_failure_even_if_success_was_observed_earlier(
    environment, earlier_success
):
    environment.default_reward = 0.0
    environment.rewards = {1: float(earlier_success)}
    result = collector.collect_task_teacher()
    assert result.candidate_episode is not None
    assert result.observed_success is earlier_success and not result.final_success
    assert result.error is None and not result.qualified
    with pytest.raises(ValueError, match="not qualified"):
        result.training_episode()


def test_should_retain_failed_geometry_read_cost_without_admitting_an_action(
    environment, monkeypatch
):
    reads = []

    def geometry_read(session):
        reads.append(session)
        if len(reads) == 2:
            raise RuntimeError("geometry unavailable")
        return PickCubeGeometry((0.0,) * 3, (0.0,) * 3, (0.0,) * 3)

    monkeypatch.setattr(collector, "_geometry", geometry_read)
    result = collector.collect_task_teacher()
    assert result.geometry_attempts == 2 and result.geometry_returns == 1
    assert result.teacher_actions == 1 and result.stages == ("align",)
    assert result.work == port_module.ManiSkillWork(1, 1, 5, 0)
    assert "geometry unavailable" in result.error and "RuntimeError" in result.traceback
    assert not result.qualified and result.candidate_episode is None
    assert environment.close_calls == 1


def test_should_count_returned_but_invalid_geometry_without_consuming_teacher_action(
    environment, monkeypatch
):
    monkeypatch.setattr(collector, "_geometry", lambda session: None)
    result = collector.collect_task_teacher()
    assert result.geometry_attempts == result.geometry_returns == 1
    assert result.teacher_actions == 0 and result.stages == ()
    assert result.work == port_module.ManiSkillWork(0, 0, 0, 0)
    assert "PickCubeGeometry" in result.error and not result.qualified
    assert environment.commands == [] and environment.close_calls == 1


@pytest.mark.parametrize("actor", ["tcp", "cube", "goal"])
def test_should_refuse_nonfinite_actual_pose_read_before_command_admission(environment, actor):
    poses = {
        "tcp": environment.agent.tcp.pose,
        "cube": environment.cube.pose,
        "goal": environment.goal_site.pose,
    }
    poses[actor].p[0, 0] = np.nan
    result = collector.collect_task_teacher()
    assert result.geometry_attempts == 1 and result.geometry_returns == 0
    assert result.teacher_actions == 0 and result.stages == ()
    assert result.work == port_module.ManiSkillWork(0, 0, 0, 0)
    assert "finite" in result.error and not result.qualified
    assert environment.commands == [] and environment.close_calls == 1


def test_should_keep_teacher_action_and_possible_physics_cost_when_step_raises(environment):
    environment.step_failure_at = 2
    result = collector.collect_task_teacher()
    assert result.teacher_actions == result.geometry_attempts == result.geometry_returns == 2
    assert result.stages == ("align", "align")
    assert result.work == port_module.ManiSkillWork(2, 1, 5, 5)
    assert len(environment.commands) == 2 and environment.close_calls == 1
    assert "physics failed" in result.error and not result.qualified
    assert result.candidate_episode is None


def test_should_refuse_non_sparse_reward_without_erasing_completed_work(environment):
    environment.default_reward = 0.5
    result = collector.collect_task_teacher()
    assert "sparse0/1" in result.error
    assert result.work == port_module.ManiSkillWork(1, 1, 5, 0)
    assert result.teacher_actions == 1 and result.candidate_episode is None
    assert not result.qualified and environment.close_calls == 1


def test_should_refuse_missing_time_limit_flag_and_stop_at_fifty(environment):
    environment.done_at = 51
    result = collector.collect_task_teacher()
    assert "time-limit flag" in result.error
    assert result.teacher_actions == 50 and len(environment.commands) == 50
    assert result.work == port_module.ManiSkillWork(50, 50, 250, 0)
    assert result.candidate_episode is None and not result.qualified
    assert environment.close_calls == 1


def test_should_refuse_cleanup_failure_without_erasing_successful_candidate(environment):
    environment.close_error = RuntimeError("close failed")
    result = collector.collect_task_teacher()
    assert result.candidate_episode is not None and result.final_success
    assert "cleanup RuntimeError: close failed" in result.error
    assert result.work == port_module.ManiSkillWork(50, 50, 250, 0)
    assert not result.qualified and environment.close_calls == 1
    with pytest.raises(ValueError, match="not qualified"):
        result.training_episode()


def test_should_preserve_primary_failure_when_cleanup_also_fails(environment):
    environment.step_failure_at = 1
    environment.close_error = RuntimeError("close failed")
    result = collector.collect_task_teacher()
    assert result.error == "RuntimeError: physics failed"
    assert "physics failed" in result.traceback
    assert result.work == port_module.ManiSkillWork(1, 0, 0, 5)
    assert environment.close_calls == 1 and not result.qualified


def test_should_refuse_constructor_failure_and_preserve_unknown_controller_work(environment):
    environment.reset_error = RuntimeError("reset failed")
    result = collector.collect_task_teacher()
    assert result.creation_attempts == 1 and result.work is None
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 0
    assert result.stages == () and result.candidate_episode is None
    assert "reset failed" in result.error and environment.close_calls == 1
    assert not result.qualified


def test_should_revalidate_candidate_when_training_episode_is_extracted(environment):
    result = collector.collect_task_teacher()
    candidate = result.candidate_episode
    object.__setattr__(candidate.transitions[-1], "action", (9.0,) * 4)
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_write_complete_cli_receipt_and_refuse_overwrite_before_collection(
    environment, monkeypatch, tmp_path
):
    from scripts import maniskill_task_teacher_smoke as cli

    result = collector.collect_task_teacher()
    calls = []
    monkeypatch.setattr(cli, "collect_task_teacher", lambda: calls.append(True) or result)
    monkeypatch.setattr(cli, "version", lambda name: "test-version")
    monkeypatch.setattr(cli.faulthandler, "enable", lambda: None)
    output = tmp_path / "teacher.json"
    monkeypatch.setattr("sys.argv", ["teacher", "--output", str(output)])
    assert cli.main() == 0
    payload = json.loads(output.read_text(encoding="utf8"))
    assert payload["qualified"] and payload["role"] == "train" and payload["seed"] == 17
    assert payload["optimizer_updates"] == 0 and payload["reset_physics_steps"] is None
    assert len(payload["candidate_episode"]["transitions"]) == 50
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_bytes() == before and calls == [True]


def test_should_return_nonzero_cli_status_for_unqualified_candidate(
    environment, monkeypatch, tmp_path
):
    from scripts import maniskill_task_teacher_smoke as cli

    environment.default_reward = 0.0
    result = collector.collect_task_teacher()
    monkeypatch.setattr(cli, "collect_task_teacher", lambda: result)
    monkeypatch.setattr(cli, "version", lambda name: "test-version")
    monkeypatch.setattr(cli.faulthandler, "enable", lambda: None)
    output = tmp_path / "failed-teacher.json"
    monkeypatch.setattr("sys.argv", ["teacher", "--output", str(output)])
    assert cli.main() == 1
    payload = json.loads(output.read_text(encoding="utf8"))
    assert not payload["qualified"] and payload["error"] is None
    assert payload["teacher_actions"] == 50 and not payload["final_success"]
