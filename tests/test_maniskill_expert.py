"""Expert boundary doubles prove admission and work, not robot skill."""

from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from src.infra import maniskill_continuous as boundary
from src.infra import maniskill_expert as expert


class FakeEnvironment:
    """Two independent reference phases or one target phase; no native physics."""

    def __init__(self, mode, *, horizon=2, successes=(True, True)):
        size = 8 if mode == "pd_joint_pos" else 4
        self.unwrapped = self
        self.num_envs = 1
        self.device = SimpleNamespace(type="cpu")
        self.obs_mode = "state"
        self.control_mode = mode
        self.agent = SimpleNamespace(uid="panda")
        self.sim_freq, self.control_freq = 100, 20
        self.single_action_space = SimpleNamespace(
            shape=(size,), low=np.full(size, -1.0), high=np.ones(size)
        )
        self.horizon = horizon
        self.successes = successes
        self.reset_calls = []
        self.commands = []
        self.phase_commands = []
        self.close_calls = 0
        self.output_override = None
        self.reset_error = self.step_error = self.close_error = None

    def reset(self, *, seed):
        self.reset_calls.append(seed)
        self.phase_commands.append([])
        if self.reset_error is not None:
            raise self.reset_error
        return np.array([[0.0, 100.0]], dtype=np.float32), {}

    def step(self, command):
        self.commands.append(command.copy())
        self.phase_commands[-1].append(command.copy())
        if self.step_error is not None:
            raise self.step_error
        if self.output_override is not None:
            return self.output_override
        position = len(self.phase_commands[-1])
        phase = len(self.reset_calls) - 1
        return (
            np.array([[float(position), 100.0]], dtype=np.float32),
            np.array([self.successes[phase]], dtype=np.bool_),
            np.array([False], dtype=np.bool_),
            np.array([position == self.horizon], dtype=np.bool_),
            {},
        )

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def scenes(monkeypatch):
    source = FakeEnvironment("pd_joint_pos")
    target = FakeEnvironment(boundary.CONTROL_MODE)
    creation = []

    def factory(*, control_mode=boundary.CONTROL_MODE):
        creation.append(control_mode)
        return source if control_mode == "pd_joint_pos" else target

    def solve(reference):
        reference.reset(seed=17)
        reference.step(np.full(8, 0.1, dtype=np.float64))
        return None

    def convert(reference, converted):
        for command in reference.actions:
            reference.step(np.array(command, dtype=np.float32))
        while True:
            converted.step(np.full(4, 0.2, dtype=np.float64))

    monkeypatch.setattr(boundary, "_make_environment", factory)
    monkeypatch.setattr(expert, "_require_expert_compatibility", lambda: None)
    monkeypatch.setattr(expert, "_solve", solve)
    monkeypatch.setattr(expert, "_convert", convert)
    return source, target, creation


def test_should_refuse_known_native_abi_crash_before_scene_creation(monkeypatch):
    monkeypatch.setattr(expert, "version", lambda package: "0.1.1")
    monkeypatch.setattr(expert.np, "__version__", "2.4.6")
    calls = []
    monkeypatch.setattr(boundary, "_make_environment", lambda **kwargs: calls.append(kwargs))
    result = expert.collect_pick_cube_expert()
    assert "NumPy<2" in result.error
    assert calls == []
    assert result.source_creation_attempts == result.target_creation_attempts == 0
    assert result.source_work.reset_attempts == 0
    assert result.source_work.collection_attempts == result.source_work.replay_attempts == 0
    assert result.candidate_episode is None
    assert not result.qualified


def test_should_admit_verified_numpy1_mplib_profile(monkeypatch):
    monkeypatch.setattr(expert, "version", lambda package: "0.1.1")
    monkeypatch.setattr(expert.np, "__version__", "1.26.4")
    expert._require_expert_compatibility()


def test_should_preserve_missing_planner_dependency_before_scene_creation(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def missing(package):
        raise PackageNotFoundError(package)

    monkeypatch.setattr(expert, "version", missing)
    result = expert.collect_pick_cube_expert()
    assert "PackageNotFoundError" in result.error
    assert result.source_creation_attempts == result.target_creation_attempts == 0
    assert result.source_work.collection_attempts == 0
    assert not result.qualified


def test_should_qualify_only_actual_target_commands_and_separate_reference_work(scenes):
    source, target, creation = scenes
    result = expert.collect_pick_cube_expert()

    assert result.qualified
    assert result.error is None
    assert creation == ["pd_joint_pos", boundary.CONTROL_MODE]
    assert source.reset_calls == [17, 17]
    assert target.reset_calls == [17]
    assert source.close_calls == target.close_calls == 1
    assert result.source_creation_attempts == result.target_creation_attempts == 1
    work = result.source_work
    assert (work.reset_attempts, work.reset_returns) == (2, 2)
    assert (work.collection_attempts, work.collection_returns) == (2, 2)
    assert (work.replay_attempts, work.replay_returns) == (2, 2)
    assert work.completed_control_physics_steps == 20
    assert work.possible_failed_physics_steps == 0
    assert result.target_work.completed_steps == 2
    assert result.target_work.completed_control_physics_steps == 10
    episode = result.training_episode()
    assert episode.role == "train"
    assert len(episode.transitions) == 2
    sent = tuple(float(item) for item in target.commands[0][0])
    assert episode.transitions[0].action == sent
    assert sent == (float(np.float32(0.2)),) * 4
    assert sent != (0.2,) * 4
    assert source.commands[0].dtype == target.commands[0].dtype == np.float32
    assert episode.transitions[-1].truncated
    assert not episode.transitions[-1].terminated
    assert episode is not result.candidate_episode


def test_should_hold_last_actual_joint_target_to_original_horizon(scenes):
    source, target, _ = scenes
    source.horizon = target.horizon = boundary.HORIZON
    result = expert.collect_pick_cube_expert()

    assert result.qualified
    assert [len(phase) for phase in source.phase_commands] == [50, 50]
    assert len(target.commands) == 50
    assert result.source_work.completed_control_physics_steps == 500
    assert result.target_work.completed_control_physics_steps == 250
    for command in source.commands:
        np.testing.assert_array_equal(command, source.commands[0])
    assert result.candidate_episode.transitions[-1].step == 49


def test_should_preserve_zero_success_horizon_as_negative_candidate(scenes):
    source, target, _ = scenes
    source.horizon = target.horizon = 50
    target.successes = (False, False)
    result = expert.collect_pick_cube_expert()

    assert not result.qualified
    assert result.collection_success and result.replay_success
    assert not result.target_success
    assert result.error is None
    assert len(result.candidate_episode.transitions) == 50
    assert all(record.reward == 0 for record in result.candidate_episode.transitions)
    assert result.target_work.completed_steps == 50
    with pytest.raises(ValueError, match="not qualified"):
        result.training_episode()


@pytest.mark.parametrize("phase", [0, 1])
def test_should_require_independent_collection_and_replay_success(scenes, phase):
    source, target, creation = scenes
    source.successes = tuple(index != phase for index in range(2))
    result = expert.collect_pick_cube_expert()

    assert not result.qualified
    assert (result.collection_success, result.replay_success)[phase] is False
    with pytest.raises(ValueError, match="not qualified"):
        result.training_episode()
    assert source.close_calls == 1
    if phase == 0:
        assert creation == ["pd_joint_pos"]
        assert target.close_calls == 0


def test_should_qualify_successful_prefix_replay_when_target_completes(scenes, monkeypatch):
    source, target, _ = scenes

    def convert(reference, converted):
        reference.step(np.array(reference.actions[0], dtype=np.float32))
        while True:
            converted.step(np.zeros(4, dtype=np.float32))

    monkeypatch.setattr(expert, "_convert", convert)
    result = expert.collect_pick_cube_expert()
    assert len(source.phase_commands[1]) == 1
    assert len(target.commands) == 2
    assert result.replay_success and result.target_success
    assert result.qualified
    assert result.source_work.replay_returns == 1
    assert result.source_work.completed_control_physics_steps == 15
    assert result.training_episode().role == "train"


def test_should_stop_upstream_solver_at_terminal_without_an_extra_native_step(scenes, monkeypatch):
    source, _, _ = scenes
    reached = []

    def solve(reference):
        reference.reset(seed=17)
        for index in range(100):
            reached.append(index)
            reference.step(np.zeros(8, dtype=np.float32))

    monkeypatch.setattr(expert, "_solve", solve)
    result = expert.collect_pick_cube_expert()
    assert result.qualified
    assert reached == [0, 1, 2]
    assert len(source.phase_commands[0]) == 2
    assert not result.solver_completed


def test_should_stop_conversion_at_target_terminal_before_more_reference_work(scenes, monkeypatch):
    source, target, _ = scenes
    reached = []

    def convert(reference, converted):
        for command in reference.actions:
            reference.step(np.array(command, dtype=np.float32))
        for index in range(100):
            reached.append(index)
            converted.step(np.zeros(4, dtype=np.float32))

    monkeypatch.setattr(expert, "_convert", convert)
    result = expert.collect_pick_cube_expert()
    assert result.qualified
    assert reached == [0, 1]
    assert len(source.phase_commands[1]) == len(target.commands) == 2


@pytest.mark.parametrize(
    "command",
    [
        np.zeros(7, dtype=np.float32),
        np.zeros((2, 8), dtype=np.float32),
        np.zeros(8, dtype=np.int64),
        np.full(8, np.nan, dtype=np.float64),
        np.full(8, 1.00000001, dtype=np.float64),
        np.full(8, -1.00000001, dtype=np.float64),
    ],
)
def test_should_refuse_invalid_joint_commands_before_native_admission(scenes, monkeypatch, command):
    source, _, _ = scenes

    def solve(reference):
        reference.reset(seed=17)
        reference.step(command)

    monkeypatch.setattr(expert, "_solve", solve)
    result = expert.collect_pick_cube_expert()
    assert not result.qualified
    assert result.error is not None
    assert source.commands == []
    assert result.source_work.collection_attempts == 0
    assert source.close_calls == 1


@pytest.mark.parametrize("field", ["observation", "reward", "terminated", "truncated"])
def test_should_poison_reference_after_invalid_native_output(scenes, field):
    source, _, _ = scenes
    reference = expert._Reference()
    reference.open()
    reference.reset(seed=17)
    output = [
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
        {},
    ]
    index = ["observation", "reward", "terminated", "truncated"].index(field)
    output[index] = (
        np.zeros((2, 2), dtype=np.float32)
        if field == "observation"
        else np.array([0.5], dtype=np.float32)
    )
    source.output_override = tuple(output)

    with pytest.raises(ValueError):
        reference.step(np.zeros(8, dtype=np.float32))
    source.output_override = None
    with pytest.raises(ValueError, match="failed"):
        reference.step(np.zeros(8, dtype=np.float32))
    assert len(source.commands) == 1
    assert reference.work().collection_attempts == 1
    assert reference.work().collection_returns == 1


@pytest.mark.parametrize("output", [(), (None,) * 4, (None,) * 6])
def test_should_poison_reference_after_malformed_native_output_arity(scenes, output):
    source, _, _ = scenes
    reference = expert._Reference()
    reference.open()
    reference.reset(seed=17)
    source.output_override = output
    with pytest.raises(ValueError):
        reference.step(np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="failed"):
        reference.step(np.zeros(8, dtype=np.float32))
    assert len(source.commands) == 1


def test_should_charge_failed_native_reference_call_and_refuse_retry(scenes):
    source, _, _ = scenes
    reference = expert._Reference()
    reference.open()
    reference.reset(seed=17)
    source.step_error = RuntimeError("native reference failure")
    with pytest.raises(RuntimeError, match="native reference failure"):
        reference.step(np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="failed"):
        reference.step(np.zeros(8, dtype=np.float32))
    assert reference.work().collection_attempts == 1
    assert reference.work().collection_returns == 0
    assert reference.work().possible_failed_physics_steps == 5
    assert len(source.commands) == 1


def test_should_refuse_missing_reference_terminal_at_fifty_without_fifty_first_call(scenes):
    source, _, _ = scenes
    source.horizon = 51
    result = expert.collect_pick_cube_expert()
    assert not result.qualified
    assert "terminal" in result.error
    assert len(source.commands) == 50
    assert result.source_work.collection_returns == 50
    assert result.target_creation_attempts == 0


@pytest.mark.parametrize("bad", [np.ones(3, dtype=np.float32), np.full(4, 1.00000001)])
def test_should_route_target_validation_before_any_native_step(scenes, monkeypatch, bad):
    _, target, _ = scenes

    def convert(reference, converted):
        for command in reference.actions:
            reference.step(np.array(command, dtype=np.float32))
        converted.step(bad)

    monkeypatch.setattr(expert, "_convert", convert)
    result = expert.collect_pick_cube_expert()
    assert not result.qualified
    assert result.error is not None
    assert target.commands == []
    assert result.target_work.step_attempts == 0
    assert target.close_calls == 1


def test_should_refuse_incomplete_conversion_without_fabricated_terminal(scenes, monkeypatch):
    _, target, _ = scenes

    def convert(reference, converted):
        for command in reference.actions:
            reference.step(np.array(command, dtype=np.float32))
        converted.step(np.zeros(4, dtype=np.float32))

    monkeypatch.setattr(expert, "_convert", convert)
    result = expert.collect_pick_cube_expert()
    assert not result.qualified
    assert result.candidate_episode is None
    assert result.error is not None
    assert len(target.commands) == 1
    assert target.close_calls == 1
    with pytest.raises(ValueError, match="not qualified"):
        result.training_episode()


def test_should_preserve_source_factory_failure_and_avoid_target_creation(scenes, monkeypatch):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("source creation failed")

    monkeypatch.setattr(boundary, "_make_environment", factory)
    result = expert.collect_pick_cube_expert()
    assert result.source_creation_attempts == 1
    assert result.target_creation_attempts == 0
    assert result.source_work.collection_attempts == 0
    assert result.source_work.reset_attempts == 0
    assert result.target_work is None
    assert calls == [{"control_mode": "pd_joint_pos"}]
    assert "source creation failed" in result.error
    assert "RuntimeError" in result.traceback


def test_should_close_source_when_target_factory_fails_after_source_work(scenes, monkeypatch):
    source, _, _ = scenes

    def factory(*, control_mode=boundary.CONTROL_MODE):
        if control_mode == "pd_joint_pos":
            return source
        raise RuntimeError("target creation failed")

    monkeypatch.setattr(boundary, "_make_environment", factory)
    result = expert.collect_pick_cube_expert()
    assert not result.qualified
    assert result.source_creation_attempts == result.target_creation_attempts == 1
    assert result.source_work.collection_returns == 2
    assert result.source_work.reset_returns == 2
    assert result.target_work is None
    assert source.close_calls == 1
    assert "target creation failed" in result.error


def test_should_preserve_failed_reference_step_work_and_cleanup(scenes):
    source, _, _ = scenes
    source.step_error = RuntimeError("reference step failed")
    result = expert.collect_pick_cube_expert()
    assert result.source_work.collection_attempts == 1
    assert result.source_work.collection_returns == 0
    assert result.source_work.possible_failed_physics_steps == 5
    assert result.source_work.completed_control_physics_steps == 0
    assert result.target_creation_attempts == 0
    assert "reference step failed" in result.error
    assert source.close_calls == 1


def test_should_preserve_failed_target_step_work_and_cleanup_both_scenes(scenes):
    source, target, _ = scenes
    target.step_error = RuntimeError("target step failed")
    result = expert.collect_pick_cube_expert()
    assert not result.qualified
    assert result.source_work.replay_returns == 2
    assert result.target_work.step_attempts == 1
    assert result.target_work.completed_steps == 0
    assert result.target_work.possible_failed_physics_steps == 5
    assert "target step failed" in result.error
    assert source.close_calls == target.close_calls == 1


def test_should_refuse_official_planning_failure_without_holds_or_target(scenes, monkeypatch):
    source, _, _ = scenes

    def solve(reference):
        reference.reset(seed=17)
        reference.step(np.zeros(8, dtype=np.float32))
        return -1

    monkeypatch.setattr(expert, "_solve", solve)
    result = expert.collect_pick_cube_expert()
    assert not result.qualified
    assert result.solver_completed
    assert "planning failure" in result.error
    assert len(source.commands) == 1
    assert result.target_creation_attempts == 0


def test_should_fail_qualification_on_cleanup_failure_but_close_both_scenes(scenes):
    source, target, _ = scenes
    target.close_error = RuntimeError("target close failed")
    result = expert.collect_pick_cube_expert()
    assert source.close_calls == target.close_calls == 1
    assert result.candidate_episode is not None
    assert not result.qualified
    assert "target close failed" in result.error


@pytest.mark.parametrize(
    "changes",
    [
        {"collection_success": False},
        {"replay_success": False},
        {"target_success": False},
        {"candidate_episode": None},
        {"error": "retained failure"},
    ],
)
def test_should_refuse_training_for_any_unqualified_result(scenes, changes):
    result = replace(expert.collect_pick_cube_expert(), **changes)
    assert not result.qualified
    with pytest.raises(ValueError, match="not qualified"):
        result.training_episode()


def test_should_expose_no_seed_role_final_or_training_controls():
    assert not inspect.signature(expert.collect_pick_cube_expert).parameters
