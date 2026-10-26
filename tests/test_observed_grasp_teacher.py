"""Observed-contact teacher qualification through fake state42 IO, no learner.

Prospective per platform: eight scenes/reset/close attempts,17 control attempts
/16 returns,19 geometry attempts/returns,18 teacher decisions/returns. Model,
RNG, fit, update, forward and real physics counts are zero. Qualification and
CLI cases reuse cached receipts; existing50 v1 cases run separately by root.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace
import inspect
import json
import sys
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_task_teacher_smoke as cli
from scripts import run_continuous_robot_learning as robot
from src.core.pick_cube_readiness_teacher import (
    DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    ObservedGraspPickCubeTeacher,
)
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_task_teacher as collector
from test_maniskill_task_teacher import FakeEnvironment


OFF_HOVER = PickCubeGeometry((-0.3, 0.0, 0.25), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2))


class ObservedEnvironment(FakeEnvironment):
    def __init__(self, flags=(1.0, 0.0, 0.0)):
        super().__init__()
        self.flags = flags
        self.done_at = len(flags) - 1
        self.terminated, self.truncated = True, False
        self.default_reward = 0.0
        self.rewards[self.done_at] = 1.0

    def observation(self, index):
        values = np.zeros((1, 42), dtype=np.float32)
        values[0, 0], values[0, 18] = index, self.flags[index]
        values[0, 22], values[0, 32] = 1.0, 1.0
        return values

    def reset(self, *, seed):
        _, info = super().reset(seed=seed)
        return self.observation(0), info

    def step(self, command):
        _, reward, terminated, truncated, info = super().step(command)
        return self.observation(len(self.commands)), reward, terminated, truncated, info


def install(patch, environment, geometry=None):
    calls = []

    def factory(**options):
        calls.append(options)
        return environment

    patch.setattr(native, "_make_environment", factory)
    patch.setattr(collector, "_geometry", geometry or (lambda _: OFF_HOVER))
    return calls


def forbidden(*args, **kwargs):
    pytest.fail("invalid selected scope admitted collection or learner work")


@pytest.fixture(scope="module")
def collected():
    records = {}
    for seed in (17, 19):
        environment = ObservedEnvironment()
        with pytest.MonkeyPatch.context() as patch:
            calls = install(patch, environment)
            result = collector.collect_readiness_teacher(train_seed=seed, observed_grasp=True)
        assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
        assert result.qualified is True, result
        records[seed] = (result, environment, calls)
    return records


@pytest.mark.parametrize("seed", [17, 19])
def test_should_collect_held_off_hover_then_reopen_after_current_grasp_loss(collected, seed):
    result, environment, calls = collected[seed]
    assert type(result) is collector.ObservedGraspTaskTeacherResult
    assert result.teacher_definition == OBSERVED_GRASP_DEFINITION
    assert result.observed_grasp == (True, False)
    assert result.stages == ("lift", "align")
    assert result.creation_attempts == 1
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    assert result.work == native.ManiSkillWork(2, 2, 10, 0)
    assert calls == [{}] and environment.reset_calls == [seed] and environment.close_calls == 1
    episode = result.training_episode()
    assert episode.episode_id == f"pick-cube-observed-grasp-teacher-{seed}"
    assert episode.context == ("pick-cube" if seed == 17 else "pick-cube-train-19")
    assert episode.role == "train" and episode.spec.observation_dim == 42
    assert episode.transitions[0].action == (1.0, 0.0, 1.0, -1.0)
    assert episode.transitions[1].action == (1.0, 0.0, -1.0, 1.0)
    assert tuple(row.observation[18] == 1.0 for row in episode.transitions) == result.observed_grasp
    assert episode.transitions[0].next_observation[18] == 0.0
    assert episode.transitions[-1].terminated and episode.transitions[-1].reward == 1.0
    assert result.observation_recipe is result.visual_work is None
    assert result.error is result.traceback is None
    for row, command in zip(episode.transitions, environment.commands, strict=True):
        np.testing.assert_array_equal(command, np.asarray([row.action], dtype=np.float32))
    assert result.training_episode() is not episode


def test_should_never_infer_contact_from_seven_executed_closing_commands(monkeypatch):
    environment = ObservedEnvironment((0.0,) * 9)
    count = 0

    def geometry(_):
        nonlocal count
        count += 1
        return PickCubeGeometry(
            (0.0, 0.0, 0.08) if count == 1 else (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2)
        )

    install(monkeypatch, environment, geometry)
    result = collector.collect_readiness_teacher(observed_grasp=True)
    assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
    assert result.qualified is True
    assert result.stages == ("descend",) + ("grasp",) * 7
    assert result.observed_grasp == (False,) * 8
    assert result.work == native.ManiSkillWork(8, 8, 40, 0)
    assert result.teacher_actions == result.geometry_attempts == result.geometry_returns == 8
    assert [float(command[0, 3]) for command in environment.commands] == [1.0] + [-1.0] * 7


def test_should_refuse_initial_invalid_flag_before_geometry_or_teacher_decision(monkeypatch):
    environment = ObservedEnvironment((0.5, 0.0, 0.0))
    install(monkeypatch, environment, forbidden)
    original_init = native.ManiSkillPickCubeEpisode.__init__
    original_info = collector._LOG.info
    captured: list[native.ManiSkillPickCubeEpisode] = []
    returned_logs: list[str] = []

    def initialize(port, *args, **options):
        original_init(port, *args, **options)
        captured.append(port)

    def info(message, *args, **options):
        if message == "teacher_phase=create state=returned":
            returned_logs.append(message)
            values = list(captured[0]._observation)
            values[18] = 0.0
            captured[0]._observation = tuple(values)
        return original_info(message, *args, **options)

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "__init__", initialize)
    monkeypatch.setattr(collector._LOG, "info", info)
    result = collector.collect_readiness_teacher(observed_grasp=True)
    assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
    assert result.qualified is False and result.error
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 0
    assert result.observed_grasp == () and result.stages == ()
    assert result.work == native.ManiSkillWork(0, 0, 0, 0)
    assert environment.commands == [] and environment.close_calls == 1
    assert len(captured) == 1 and returned_logs == []


def test_should_preserve_first_paid_feedback_when_later_geometry_changes_current_flag(monkeypatch):
    environment = ObservedEnvironment()

    def geometry(port):
        if port._records:
            values = list(port._observation)
            values[18] = 1.0
            port._observation = tuple(values)
        return OFF_HOVER

    install(monkeypatch, environment, geometry)
    result = collector.collect_readiness_teacher(observed_grasp=True)
    assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
    assert result.qualified is False and result.error
    assert result.observed_grasp == (True,) and result.stages == ("lift",)
    assert result.geometry_attempts == result.geometry_returns == 2 and result.teacher_actions == 1
    assert result.work == native.ManiSkillWork(1, 1, 5, 0)
    assert result.candidate_episode is None
    assert len(environment.commands) == environment.close_calls == 1


def test_should_keep_decision_and_possible_failed_step_cost_without_inventing_returned_feedback(
    monkeypatch,
):
    environment = ObservedEnvironment()
    environment.step_failure_at = 2
    install(monkeypatch, environment)
    original_close = native.ManiSkillPickCubeEpisode.close
    erased: dict[str, tuple[int, int]] = {}

    def close(port):
        original_close(port)
        erased["before"] = (port._attempts, port._completed)
        port._attempts = port._completed
        erased["after"] = (port._attempts, port._completed)

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "close", close)
    result = collector.collect_readiness_teacher(observed_grasp=True)
    assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
    assert result.error is not None
    assert result.qualified is False and "physics failed" in result.error
    assert result.stages == ("lift", "align") and result.teacher_actions == 2
    assert result.observed_grasp == (True,)
    assert result.geometry_attempts == result.geometry_returns == 2
    assert result.work == native.ManiSkillWork(2, 1, 5, 5)
    assert result.candidate_episode is None and environment.close_calls == 1
    assert erased == {"before": (2, 1), "after": (1, 1)}


def test_should_retain_complete_feedback_and_candidate_when_cleanup_fails(monkeypatch):
    environment = ObservedEnvironment()
    environment.close_error = RuntimeError("observed cleanup failed")
    install(monkeypatch, environment)
    result = collector.collect_readiness_teacher(observed_grasp=True)
    assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
    assert result.error is not None
    assert result.observed_success is result.final_success is True
    assert result.candidate_episode is not None and result.observed_grasp == (True, False)
    assert result.work == native.ManiSkillWork(2, 2, 10, 0)
    assert result.qualified is False and "observed cleanup failed" in result.error
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_count_returned_teacher_decision_before_refusing_its_wrong_type(monkeypatch):
    environment = ObservedEnvironment()
    install(monkeypatch, environment)
    original = ObservedGraspPickCubeTeacher.decide

    def wrong(teacher, geometry, *, grasped, executed_gripper=None):
        decision = original(teacher, geometry, grasped=grasped, executed_gripper=executed_gripper)
        return decision.stage, decision.action

    monkeypatch.setattr(ObservedGraspPickCubeTeacher, "decide", wrong)
    result = collector.collect_readiness_teacher(observed_grasp=True)
    assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
    assert result.qualified is False and result.error
    assert result.teacher_actions == result.geometry_attempts == result.geometry_returns == 1
    assert result.observed_grasp == () and result.stages == ()
    assert result.work == native.ManiSkillWork(0, 0, 0, 0)
    assert environment.commands == [] and environment.close_calls == 1


def as_base(result, *, readiness=False):
    values = {
        field.name: getattr(result, field.name) for field in fields(collector.TaskTeacherResult)
    }
    value = collector.TaskTeacherResult(**values)
    assert value.candidate_episode is not None
    return replace(
        value,
        teacher_definition=DEFINITION if readiness else None,
        candidate_episode=replace(
            value.candidate_episode,
            episode_id="pick-cube-readiness-teacher-17"
            if readiness
            else "pick-cube-task-teacher-17",
        ),
    )


def test_should_require_new_feedback_keyword_without_changing_the_base_result_schema(collected):
    result = collected[17][0]
    parameter = inspect.signature(collector.ObservedGraspTaskTeacherResult).parameters[
        "observed_grasp"
    ]
    assert (
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    )
    values = {
        field.name: getattr(result, field.name) for field in fields(collector.TaskTeacherResult)
    }
    with pytest.raises(TypeError):
        collector.ObservedGraspTaskTeacherResult(**values)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "observed_grasp", ())
    base = as_base(result)
    assert type(base) is collector.TaskTeacherResult and base.qualified is True
    assert "observed_grasp" not in asdict(base)
    assert set(asdict(result)) == set(asdict(base)) | {"observed_grasp"}


@pytest.mark.parametrize(
    "damage",
    [
        "feedback_list",
        "feedback_length",
        "feedback_integer",
        "feedback_mismatch",
        "current_bool",
        "next_fraction",
        "true_approach",
        "false_transport",
        "definition",
        "role",
        "work_bool",
        "counts",
        "erased_family",
    ],
)
def test_should_refuse_corrupted_selected_qualification_and_family_erasure(collected, damage):
    result: Any = deepcopy(collected[17][0])
    if damage.startswith("feedback_"):
        feedback = {
            "feedback_list": [True, False],
            "feedback_length": (True,),
            "feedback_integer": (1, False),
            "feedback_mismatch": (False, False),
        }[damage]
        object.__setattr__(result, "observed_grasp", feedback)
    elif damage in ("current_bool", "next_fraction"):
        row = result.candidate_episode.transitions[0 if damage == "current_bool" else -1]
        name = "observation" if damage == "current_bool" else "next_observation"
        values = list(getattr(row, name))
        values[18] = True if damage == "current_bool" else 0.5
        object.__setattr__(row, name, tuple(values))
    elif damage in ("true_approach", "false_transport"):
        episode = result.candidate_episode
        index = 0 if damage == "true_approach" else 1
        rows = list(episode.transitions)
        rows[index] = replace(rows[index], action=(0.0, 0.0, 0.0, -1.0))
        stages = list(result.stages)
        stages[index] = "grasp" if damage == "true_approach" else "settle"
        result = replace(
            result,
            candidate_episode=replace(episode, transitions=tuple(rows)),
            stages=tuple(stages),
        )
    elif damage == "definition":
        result = replace(result, teacher_definition=DEFINITION)
    elif damage == "role":
        result = replace(
            result, candidate_episode=replace(result.candidate_episode, role="final_test")
        )
    elif damage == "work_bool":
        result = replace(result, work=replace(result.work, possible_failed_physics_steps=False))
    elif damage == "counts":
        result = replace(result, geometry_returns=1)
    else:
        result = collector.TaskTeacherResult(
            **{
                field.name: getattr(result, field.name)
                for field in fields(collector.TaskTeacherResult)
            }
        )
        result = replace(result, teacher_definition=None)
    assert result.qualified is False
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize("fault", ["integer", "none", "visual", "seed"])
def test_should_refuse_invalid_selected_api_scope_before_port_creation(monkeypatch, fault):
    monkeypatch.setattr(collector, "ManiSkillPickCubeEpisode", forbidden)
    options: dict[str, Any] = dict(observed_grasp=True)
    if fault == "integer":
        options["observed_grasp"] = 1
    elif fault == "none":
        options["observed_grasp"] = None
    elif fault == "visual":
        options["visual_observation"] = True
    else:
        options["train_seed"] = 31
    with pytest.raises(ValueError):
        collector.collect_readiness_teacher(**options)


@pytest.fixture(scope="module")
def cli_receipts(collected, tmp_path_factory):
    directory = tmp_path_factory.mktemp("observed-teacher-cli")
    records = {}
    for mode in ("observed17", "observed19", "readiness", "legacy"):
        seed = 19 if mode == "observed19" else 17
        selected = mode.startswith("observed")
        result = (
            collected[seed][0]
            if selected
            else as_base(collected[17][0], readiness=mode == "readiness")
        )
        output = directory / f"{mode}.json"
        calls = []

        def collect(**options):
            calls.append(options)
            return result

        args = ["teacher", "--output", str(output)]
        if mode != "legacy":
            args.append("--geometry-readiness")
        if selected:
            args.append("--observed-grasp")
        if seed == 19:
            args += ["--train-seed", "19"]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                cli,
                "collect_task_teacher" if mode == "legacy" else "collect_readiness_teacher",
                collect,
            )
            patch.setattr(cli, "version", lambda _: "fake-observed-cli")
            patch.setattr(sys, "argv", args)
            status = cli.main()
        records[mode] = (status, json.loads(output.read_text(encoding="utf8")), calls, output)
    return records


@pytest.mark.parametrize("mode", ["observed17", "observed19", "readiness", "legacy"])
def test_should_forward_only_selected_true_and_preserve_default_cli_fieldsets(cli_receipts, mode):
    status, payload, calls, _ = cli_receipts[mode]
    assert status == 0 and payload["qualified"] is True
    expected: dict[str, Any] = {"observed_grasp": True} if mode.startswith("observed") else {}
    if mode == "observed19":
        expected["train_seed"] = 19
    assert calls == [expected]
    assert payload["optimizer_updates"] == 0 and payload["teacher_actions"] == 2
    if mode.startswith("observed"):
        assert payload["observed_grasp"] == [True, False]
        assert payload["teacher_definition"] == OBSERVED_GRASP_DEFINITION
    else:
        assert "observed_grasp" not in payload
        if mode == "legacy":
            assert "teacher_definition" not in payload
        else:
            assert payload["teacher_definition"] == DEFINITION


def test_should_refuse_selected_receipt_in_the_old_geometry_readiness_learning_reader(cli_receipts):
    _, payload, _, path = cli_receipts["observed17"]
    assert payload["qualified"] is True
    with pytest.raises(ValueError):
        robot._read_readiness_teacher(path, 17)


@pytest.mark.parametrize("selected_request", [False, True])
def test_should_retain_paid_payload_but_reject_the_wrong_requested_teacher_family(
    collected, monkeypatch, tmp_path, selected_request
):
    result = as_base(collected[17][0], readiness=True) if selected_request else collected[17][0]
    calls = []

    def collect(**options):
        calls.append(options)
        return result

    monkeypatch.setattr(cli, "collect_readiness_teacher", collect)
    monkeypatch.setattr(cli, "version", lambda _: "fake-observed-cli")
    output = tmp_path / "wrong-family.json"
    args = ["teacher", "--output", str(output), "--geometry-readiness"]
    if selected_request:
        args.append("--observed-grasp")
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 1
    payload = json.loads(output.read_text(encoding="utf8"))
    assert calls == ([{"observed_grasp": True}] if selected_request else [{}])
    assert payload["qualified"] is False and payload["readiness_request_error"]
    assert payload["candidate_episode"] is not None
    assert payload["teacher_actions"] == payload["work"]["completed_steps"] == 2
    assert result.qualified is True


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--geometry-readiness", "--visual-observation"],
        ["--geometry-readiness", "--camera-drift"],
        ["--geometry-readiness", "--mass-drift"],
        ["--geometry-readiness", "--lighting-drift"],
        ["--geometry-readiness", "--guard-seed", "31", "--labels-arrived-at", "1"],
        ["--geometry-readiness", "--labels-arrived-at", "1"],
    ],
)
def test_should_refuse_cli_cross_scope_before_output_or_collection(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(cli, "collect_task_teacher", forbidden)
    monkeypatch.setattr(cli, "collect_readiness_teacher", forbidden)
    monkeypatch.setattr(cli, "collect_guard_teacher", forbidden)
    output = tmp_path / "not-created.json"
    monkeypatch.setattr(
        sys, "argv", ["teacher", "--output", str(output), "--observed-grasp", *extra]
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2 and not output.exists()
