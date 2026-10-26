"""Geometry-readiness decisions and existing fake-port collection, no native skill.

Core probes use deterministic scalar geometry. Fake environments only return
declared observations/flags and record commands; no learner/RNG/fit/physics runs.

Prospective per invocation: 50 cases, 316 readiness decisions attempted (309
returned), 52 legacy actions. Eight fake scenes/reset/close calls, 79 attempted
controls (78 completed), 80 geometry attempts (79 returned); three CLI stub
dispatches reuse cached receipts. RGB encoding is three fixed fake frames.
"""

from dataclasses import FrozenInstanceError, replace
import json
import math
import sys
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_task_teacher_smoke as cli
from src.core.pick_cube_readiness_teacher import (
    DEFINITION,
    ReadinessDecision,
    ReadinessPickCubeTeacher,
)
from src.core.pick_cube_teacher import PickCubeGeometry, PickCubeTeacher
from src.infra import maniskill_continuous as native
from src.infra import maniskill_task_teacher as collector
from test_maniskill_task_teacher import FakeEnvironment
from test_maniskill_rgb_observations import RGBEnvironment


def geometry(tcp=(0.0, 0.0, 0.0), cube=(0.0, 0.0, 0.0), goal=(0.1, 0.0, 0.2)):
    return PickCubeGeometry(tcp, cube, goal)


def hover():
    return geometry((0.0, 0.0, 0.08))


def close_teacher(count=6):
    teacher = ReadinessPickCubeTeacher()
    assert teacher.decide(hover()).stage == "descend"
    for _ in range(count):
        assert teacher.decide(geometry()).stage == "grasp"
    return teacher


def lift_teacher():
    teacher = close_teacher()
    assert teacher.decide(geometry()).stage == "lift"
    return teacher


def test_should_stay_aligned_for_fifty_stuck_calls_and_refuse_fifty_first():
    teacher = ReadinessPickCubeTeacher()
    stuck = geometry((-0.2, 0.0, 0.0))
    for step in range(50):
        decision = teacher.decide(stuck)
        assert decision.stage == "align"
        assert decision.action == pytest.approx((1.0, 0.0, 0.8, 1.0))
        assert teacher.steps == step + 1
        assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in decision.action)
    with pytest.raises(ValueError, match="horizon"):
        teacher.decide(stuck)
    assert teacher.steps == 50


def test_should_return_frozen_stage_and_action_from_the_same_decision():
    teacher = ReadinessPickCubeTeacher()
    decision = teacher.decide(hover())
    assert type(decision) is ReadinessDecision
    assert decision.stage == "descend" and decision.action == pytest.approx((0.0, 0.0, -0.8, 1.0))
    with pytest.raises(FrozenInstanceError):
        setattr(decision, "stage", "lift")
    assert type(decision.action) is tuple and teacher.steps == 1


@pytest.mark.parametrize("offset,stage", [(0.01, "descend"), (0.010001, "align")])
def test_should_use_hover_geometry_threshold_instead_of_elapsed_calls(offset, stage):
    teacher = ReadinessPickCubeTeacher()
    assert teacher.decide(geometry((offset, 0.0, 0.08))).stage == stage


@pytest.mark.parametrize("offset,stage", [(0.012, "grasp"), (0.012001, "descend")])
def test_should_close_only_after_current_three_axis_descent_error_is_ready(offset, stage):
    teacher = ReadinessPickCubeTeacher()
    teacher.decide(hover())
    assert teacher.decide(geometry((offset, 0.0, 0.0))).stage == stage


@pytest.mark.parametrize("near_goal", [False, True])
def test_should_issue_six_close_commands_before_lift_or_settle(near_goal):
    teacher = ReadinessPickCubeTeacher()
    teacher.decide(hover())
    current = geometry(goal=(0.0, 0.0, 0.0)) if near_goal else geometry()
    for index in range(6):
        decision = teacher.decide(current)
        assert decision.stage == "grasp" and decision.action == (0.0, 0.0, 0.0, -1.0)
        assert teacher.steps == index + 2
    following = teacher.decide(current)
    assert following.stage == ("settle" if near_goal else "lift")
    assert following.action == ((0.0, 0.0, 0.0, -1.0) if near_goal else (1.0, 0.0, 1.0, -1.0))


def test_should_not_advance_stuck_descent_by_time():
    teacher = ReadinessPickCubeTeacher()
    teacher.decide(hover())
    for _ in range(10):
        decision = teacher.decide(geometry((0.0, 0.0, 0.04)))
        assert decision.stage == "descend" and decision.action == pytest.approx(
            (0.0, 0.0, -0.4, 1.0)
        )


def test_should_restart_alignment_when_lateral_descent_drifts_then_require_readiness_again():
    teacher = ReadinessPickCubeTeacher()
    teacher.decide(hover())
    decision = teacher.decide(geometry((0.021, 0.0, 0.04)))
    assert decision.stage == "align" and decision.action[-1] == 1.0
    assert teacher.decide(hover()).stage == "descend"
    for _ in range(6):
        assert teacher.decide(geometry()).stage == "grasp"
    assert teacher.decide(geometry()).stage == "lift"


def test_should_reset_close_count_when_grasp_alignment_is_lost():
    teacher = close_teacher(2)
    decision = teacher.decide(geometry((0.020001, 0.0, 0.0)))
    assert decision.stage == "align" and decision.action[-1] == 1.0
    assert teacher.decide(hover()).stage == "descend"
    for _ in range(6):
        assert teacher.decide(geometry()).stage == "grasp"
    assert teacher.decide(geometry()).stage == "lift"


@pytest.mark.parametrize("settle", [False, True])
def test_should_reopen_after_losing_cube_proximity_during_transport_or_settle(settle):
    teacher = lift_teacher()
    if settle:
        assert teacher.decide(geometry(goal=(0.0, 0.0, 0.0))).stage == "settle"
    decision = teacher.decide(geometry((0.061, 0.0, 0.0), goal=(0.0, 0.0, 0.0)))
    assert decision.stage == "align" and decision.action[-1] == 1.0


def test_should_use_goal_hysteresis_and_resume_lift_without_reclosing():
    teacher = lift_teacher()
    for offset in (0.02, 0.03):
        decision = teacher.decide(geometry((offset, 0.0, 0.0), (offset, 0.0, 0.0), (0.0, 0.0, 0.0)))
        assert decision.stage == "settle" and decision.action == (0.0, 0.0, 0.0, -1.0)
    decision = teacher.decide(geometry((0.030001, 0.0, 0.0), (0.030001, 0.0, 0.0), (0.0, 0.0, 0.0)))
    assert decision.stage == "lift" and decision.action[0] == pytest.approx(-0.30001)


@pytest.mark.parametrize("fault", ["foreign", "list", "bool", "nonfinite", "shape", "overflow"])
def test_should_not_consume_a_decision_or_close_state_when_geometry_copy_or_command_fails(fault):
    teacher, reference = close_teacher(2), close_teacher(2)
    invalid: Any = geometry()
    if fault == "foreign":
        invalid = object()
    elif fault == "list":
        object.__setattr__(invalid, "tcp", [0.0, 0.0, 0.0])
    elif fault == "bool":
        object.__setattr__(invalid, "cube", (True, 0.0, 0.0))
    elif fault == "nonfinite":
        object.__setattr__(invalid, "goal", (float("nan"), 0.0, 0.0))
    elif fault == "shape":
        object.__setattr__(invalid, "tcp", (0.0, 0.0))
    else:
        invalid = geometry((-1e308, 0.0, 0.0), (1e308, 0.0, 0.0))
    before = teacher.steps
    with pytest.raises(ValueError):
        teacher.decide(invalid)
    assert teacher.steps == before
    for _ in range(5):
        assert teacher.decide(geometry()) == reference.decide(geometry())


def test_should_preserve_original_fixed_teacher_schedule():
    original = PickCubeTeacher()
    stages = []
    for _ in range(50):
        stages.append(original.stage)
        command = original.action(geometry())
        assert len(command) == 4
    assert (
        stages == ["align"] * 8 + ["descend"] * 8 + ["grasp"] * 6 + ["lift"] * 18 + ["settle"] * 10
    )


class StateEnvironment(FakeEnvironment):
    """Keep existing fake IO while matching the real state42 codec dimension."""

    def reset(self, *, seed):
        _, info = super().reset(seed=seed)
        return np.zeros((1, 42), dtype=np.float32), info

    def step(self, command):
        _, reward, terminated, truncated, info = super().step(command)
        values = np.zeros((1, 42), dtype=np.float32)
        values[0, 0] = len(self.commands)
        return values, reward, terminated, truncated, info


def install_environment(patch, environment, geometries):
    calls = []
    values = iter(geometries)

    def factory(**options):
        calls.append(options)
        return environment

    patch.setattr(native, "_make_environment", factory)
    patch.setattr(collector, "_geometry", lambda _: next(values))
    return calls


def successful_environment(steps=10):
    environment = StateEnvironment()
    environment.done_at = steps
    environment.terminated, environment.truncated = True, False
    environment.default_reward = 0.0
    environment.rewards[steps] = 1.0
    return environment


def route_geometry():
    return [hover()] + [geometry()] * 7 + [geometry((0.1, 0.0, 0.2), (0.1, 0.0, 0.2))] * 2


@pytest.fixture(scope="module")
def collected():
    readiness, legacy = successful_environment(), successful_environment(2)
    with pytest.MonkeyPatch.context() as patch:
        calls = install_environment(patch, readiness, route_geometry())
        result = collector.collect_readiness_teacher()
        install_environment(patch, legacy, [hover(), geometry()])
        default = collector.collect_task_teacher()
    return result, readiness, calls, default, legacy


def test_should_collect_exact_decision_stages_actions_and_costs_without_geometry_in_observations(
    collected,
):
    result, environment, calls, _, _ = collected
    assert result.qualified and result.teacher_definition == DEFINITION
    assert calls == [{}] and environment.reset_calls == [17] and environment.close_calls == 1
    assert result.stages == ("descend",) + ("grasp",) * 6 + ("lift", "settle", "settle")
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 10
    assert result.work == native.ManiSkillWork(10, 10, 50, 0)
    value = result.training_episode()
    assert value.episode_id == "pick-cube-readiness-teacher-17" and value.role == "train"
    assert value.context == "pick-cube" and value.spec.observation_dim == 42
    commands = (
        [(0.0, 0.0, -0.8, 1.0)]
        + [(0.0, 0.0, 0.0, -1.0)] * 6
        + [(1.0, 0.0, 1.0, -1.0)]
        + [(0.0, 0.0, 0.0, -1.0)] * 2
    )
    for index, (row, command) in enumerate(zip(value.transitions, commands, strict=True)):
        assert row.action == pytest.approx(command)
        np.testing.assert_allclose(
            environment.commands[index], np.array([command], dtype=np.float32)
        )
        assert row.observation == (float(index),) + (0.0,) * 41
    assert result.training_episode() is not value


def test_should_keep_default_collector_provenance_and_elapsed_schedule_unchanged(collected):
    _, _, _, result, environment = collected
    assert result.qualified and result.teacher_definition is None
    assert result.candidate_episode.episode_id == "pick-cube-task-teacher-17"
    assert result.stages == ("align", "align") and result.teacher_actions == 2
    assert len(environment.commands) == 2


def test_should_collect_separate_train19_context(monkeypatch):
    environment = successful_environment()
    install_environment(monkeypatch, environment, route_geometry())
    result = collector.collect_readiness_teacher(train_seed=19)
    assert result.qualified and environment.reset_calls == [19]
    assert result.training_episode().context == "pick-cube-train-19"
    assert result.training_episode().episode_id == "pick-cube-readiness-teacher-19"


def test_should_preserve_decision_and_possible_step_cost_on_physics_failure(monkeypatch):
    environment = successful_environment()
    environment.step_failure_at = 3
    install_environment(monkeypatch, environment, route_geometry())
    result = collector.collect_readiness_teacher()
    assert not result.qualified and result.error is not None and "physics failed" in result.error
    assert result.stages == ("descend", "grasp", "grasp")
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 3
    assert result.work == native.ManiSkillWork(3, 2, 10, 5)
    assert result.candidate_episode is None and environment.close_calls == 1


def test_should_retain_successful_candidate_and_paid_work_if_cleanup_fails(monkeypatch):
    environment = successful_environment(2)
    environment.close_error = RuntimeError("readiness cleanup failed")
    install_environment(monkeypatch, environment, [hover(), geometry()])
    result = collector.collect_readiness_teacher()
    assert result.final_success and result.candidate_episode is not None and not result.qualified
    assert result.teacher_actions == 2 and result.work == native.ManiSkillWork(2, 2, 10, 0)
    assert result.error is not None and "readiness cleanup failed" in result.error
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_stop_geometry_failure_without_admitting_a_teacher_action(monkeypatch):
    environment = successful_environment()
    install_environment(monkeypatch, environment, [])
    result = collector.collect_readiness_teacher()
    assert result.geometry_attempts == 1 and result.geometry_returns == result.teacher_actions == 0
    assert result.stages == () and result.work == native.ManiSkillWork(0, 0, 0, 0)
    assert not result.qualified and environment.commands == [] and environment.close_calls == 1


def test_should_stop_at_horizon_without_time_based_descent_or_grasp(monkeypatch):
    environment = successful_environment(50)
    stuck = geometry((-0.2, 0.0, 0.0))
    install_environment(monkeypatch, environment, [stuck] * 50)
    result = collector.collect_readiness_teacher()
    assert result.qualified and result.stages == ("align",) * 50
    assert result.teacher_actions == 50 and result.work == native.ManiSkillWork(50, 50, 250, 0)
    assert all(float(command[0, 3]) == 1.0 for command in environment.commands)


def test_should_support_fixed_rgb_recipe_without_exposing_privileged_geometry(monkeypatch):
    environment = RGBEnvironment()
    environment.default_reward, environment.rewards = 0.0, {2: 1.0}
    calls = install_environment(monkeypatch, environment, [hover(), geometry()])
    result = collector.collect_readiness_teacher(visual_observation=True)
    assert result.qualified and result.training_episode().spec.observation_dim == 789
    assert calls == [{"visual_observation": True}]
    assert result.stages == ("descend", "grasp") and result.teacher_actions == 2
    assert result.visual_work is not None
    assert result.visual_work.observation_returns == 3


@pytest.mark.parametrize("seed", [True, 17.0, 23, 31, 37])
def test_should_refuse_reserved_or_non_native_train_seeds_before_port_creation(monkeypatch, seed):
    monkeypatch.setattr(
        collector,
        "ManiSkillPickCubeEpisode",
        lambda *a, **k: pytest.fail("invalid seed created port"),
    )
    with pytest.raises(ValueError):
        collector.collect_readiness_teacher(train_seed=seed)


@pytest.mark.parametrize(
    "fault", ["definition", "id", "role", "state21", "stage", "work_bool", "counts", "guard"]
)
def test_should_refuse_relabeling_or_corrupted_readiness_qualification(collected, fault):
    result = collected[0]
    if fault == "definition":
        result = replace(result, teacher_definition=None)
    elif fault == "id":
        result = replace(
            result,
            candidate_episode=replace(
                result.candidate_episode, episode_id="pick-cube-task-teacher-17"
            ),
        )
    elif fault == "role":
        result = replace(
            result, candidate_episode=replace(result.candidate_episode, role="final_test")
        )
    elif fault == "state21":
        episode = result.candidate_episode
        episode = replace(
            episode,
            spec=replace(episode.spec, observation_dim=21),
            transitions=tuple(
                replace(
                    row,
                    observation=row.observation[:21],
                    next_observation=row.next_observation[:21],
                )
                for row in episode.transitions
            ),
        )
        result = replace(result, candidate_episode=episode)
    elif fault == "stage":
        result = replace(result, stages=("grasp",) + result.stages[1:])
    elif fault == "work_bool":
        result = replace(result, work=replace(result.work, possible_failed_physics_steps=False))
    elif fault == "counts":
        result = replace(result, teacher_actions=9)
    else:
        result = replace(result, guard_seed=31)
    assert result.qualified is False
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize("readiness", [False, True])
def test_should_route_cli_explicitly_and_keep_default_json_fieldset(
    collected, monkeypatch, tmp_path, readiness
):
    calls = []
    result = collected[0] if readiness else collected[3]

    def selected(**options):
        calls.append(options)
        return result

    monkeypatch.setattr(
        cli, "collect_readiness_teacher" if readiness else "collect_task_teacher", selected
    )
    monkeypatch.setattr(cli, "version", lambda _: "fake-readiness-cli")
    output = tmp_path / "teacher.json"
    arguments = ["teacher", "--output", str(output)] + (
        ["--geometry-readiness"] if readiness else []
    )
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0 and calls == [{}]
    payload = json.loads(output.read_text(encoding="utf8"))
    assert payload["qualified"] is True
    if readiness:
        assert payload["teacher_definition"] == DEFINITION
    else:
        assert "teacher_definition" not in payload
    with pytest.raises(FileExistsError):
        cli.main()
    assert calls == [{}]


def test_should_refuse_legacy_teacher_returned_for_explicit_readiness_request_without_losing_paid_payload(
    collected, monkeypatch, tmp_path
):
    legacy = collected[3]
    calls = []

    def wrong_teacher(**options):
        calls.append(options)
        return legacy

    monkeypatch.setattr(cli, "collect_readiness_teacher", wrong_teacher)
    monkeypatch.setattr(
        cli,
        "collect_task_teacher",
        lambda **options: pytest.fail("readiness request fell back to legacy"),
    )
    monkeypatch.setattr(cli, "version", lambda _: "fake-readiness-cli")
    output = tmp_path / "mismatched.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), "--geometry-readiness"])
    assert cli.main() == 1 and calls == [{}]
    payload = json.loads(output.read_text(encoding="utf8"))
    assert payload["qualified"] is False and payload["readiness_request_error"]
    assert payload["candidate_episode"]["episode_id"] == "pick-cube-task-teacher-17"
    assert payload["teacher_actions"] == payload["work"]["completed_steps"] == 2
    assert legacy.qualified is True


@pytest.mark.parametrize(
    "extra",
    [
        ["--guard-seed", "31", "--labels-arrived-at", "1"],
        ["--labels-arrived-at", "1"],
        ["--camera-drift"],
        ["--mass-drift"],
        ["--lighting-drift"],
    ],
)
def test_should_refuse_readiness_with_other_collection_scopes_before_output(
    monkeypatch, tmp_path, extra
):
    def forbidden(**options):
        pytest.fail("mixed readiness scope collected data")

    monkeypatch.setattr(cli, "collect_readiness_teacher", forbidden)
    monkeypatch.setattr(cli, "collect_task_teacher", forbidden)
    monkeypatch.setattr(cli, "collect_guard_teacher", forbidden)
    output = tmp_path / "uncreated.json"
    monkeypatch.setattr(
        sys, "argv", ["teacher", "--output", str(output), "--geometry-readiness", *extra]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and not output.exists()
