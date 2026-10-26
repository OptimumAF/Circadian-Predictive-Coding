"""Explicit Euclidean goal readiness; legacy observed behavior stays unchanged.

Source ledger per platform:20 pure decision attempts/17 returns plus four
returned fake-collector decisions; two scenes/reset/close, four geometry and
control calls. No learner, RNG, fitting, update, forward or real physics work.
Qualification and CLI cases reuse the two cached complete result records.
"""

from copy import deepcopy
from dataclasses import fields, replace
import json
import sys
from typing import Any

import pytest

from scripts import maniskill_task_teacher_smoke as cli
from scripts import run_native_stream_demo as reader
from src.core.pick_cube_readiness_teacher import (
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    GoalConsistentObservedGraspPickCubeTeacher,
    ObservedGraspPickCubeTeacher,
)
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_task_teacher as collector
from test_observed_grasp_teacher import ObservedEnvironment, install


def geometry(goal=(0.018, 0.018, 0.0)):
    return PickCubeGeometry((-0.3, 0.0, 0.25), (0.0, 0.0, 0.0), goal)


def forbidden(*args, **kwargs):
    pytest.fail("invalid goal-consistent scope admitted collection or learner work")


@pytest.mark.parametrize("selected", [False, True])
def test_should_distinguish_the_diagonal_goal_corner_from_legacy_max_axis_readiness(selected):
    teacher = (
        GoalConsistentObservedGraspPickCubeTeacher() if selected else ObservedGraspPickCubeTeacher()
    )
    decision = teacher.decide(geometry(), grasped=True)
    assert decision.stage == ("lift" if selected else "settle")
    assert decision.action == pytest.approx(
        (0.18, 0.18, 0.0, -1.0) if selected else (0.0, 0.0, 0.0, -1.0)
    )
    assert teacher.steps == 1


def test_should_enter_at_two_centimetres_and_retain_only_through_two_point_five_centimetres():
    teacher = GoalConsistentObservedGraspPickCubeTeacher()
    stages = [
        teacher.decide(geometry((distance, 0.0, 0.0)), grasped=True).stage
        for distance in (0.02, 0.025, 0.025001, 0.021, 0.019)
    ]
    assert stages == ["settle", "settle", "lift", "lift", "settle"]
    assert teacher.steps == 5


def test_should_preserve_legacy_corner_and_three_centimetre_retention_without_silently_changing_v1():
    legacy, selected = ObservedGraspPickCubeTeacher(), GoalConsistentObservedGraspPickCubeTeacher()
    assert legacy.decide(geometry(), grasped=True).stage == "settle"
    assert selected.decide(geometry(), grasped=True).stage == "lift"
    assert selected.decide(geometry((0.0, 0.0, 0.0)), grasped=True).stage == "settle"
    assert legacy.decide(geometry((0.028, 0.0, 0.0)), grasped=True).stage == "settle"
    assert selected.decide(geometry((0.028, 0.0, 0.0)), grasped=True).stage == "lift"


def test_should_reopen_on_lost_current_grasp_even_after_goal_consistent_transport():
    teacher = GoalConsistentObservedGraspPickCubeTeacher()
    assert teacher.decide(geometry(), grasped=True).stage == "lift"
    lost = teacher.decide(geometry(), grasped=False, executed_gripper=-1.0)
    assert lost.stage == "align" and lost.action[3] == 1.0


@pytest.mark.parametrize("fault", ["geometry", "feedback", "executed"])
def test_should_preserve_atomic_validation_before_a_goal_consistent_decision(fault):
    teacher = GoalConsistentObservedGraspPickCubeTeacher()
    current = geometry()
    options: dict[str, Any] = dict(grasped=True)
    if fault == "geometry":
        object.__setattr__(current, "goal", (float("nan"), 0.0, 0.0))
    elif fault == "feedback":
        options["grasped"] = None
    else:
        options["executed_gripper"] = float("nan")
    with pytest.raises(ValueError):
        teacher.decide(current, **options)
    assert teacher.steps == 0
    assert teacher.decide(geometry((0.019, 0.0, 0.0)), grasped=True).stage == "settle"


@pytest.fixture(scope="module")
def collected():
    results = {}
    for selected in (False, True):
        environment = ObservedEnvironment((1.0, 1.0, 1.0))

        def current(port):
            return geometry((0.01, 0.0, 0.0)) if port._records else geometry()

        with pytest.MonkeyPatch.context() as patch:
            calls = install(patch, environment, current)
            options = dict(observed_grasp=True, **({"goal_consistent": True} if selected else {}))
            result = collector.collect_readiness_teacher(**options)
        assert result.qualified is True, result
        results[selected] = (result, environment, calls)
    return results


@pytest.mark.parametrize("selected", [False, True])
def test_should_collect_the_requested_version_with_original_raw_feedback_and_paid_work(
    collected, selected
):
    result, environment, calls = collected[selected]
    expected_type = (
        collector.GoalConsistentObservedGraspTaskTeacherResult
        if selected
        else collector.ObservedGraspTaskTeacherResult
    )
    assert type(result) is expected_type
    assert isinstance(result, collector.ObservedGraspTaskTeacherResult)
    assert result.teacher_definition == (
        GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION if selected else OBSERVED_GRASP_DEFINITION
    )
    assert result.observed_grasp == (True, True)
    assert result.stages == (("lift", "settle") if selected else ("settle", "settle"))
    assert result.work == native.ManiSkillWork(2, 2, 10, 0)
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    episode = result.training_episode()
    family = "goal-consistent-observed-grasp" if selected else "observed-grasp"
    assert episode.episode_id == f"pick-cube-{family}-teacher-17"
    assert episode.role == "train" and episode.context == "pick-cube"
    assert episode.spec.observation_dim == 42
    assert result.final_success is result.observed_success is True
    assert all(
        row.observation[18] == 1.0 and row.next_observation[18] == 1.0
        for row in episode.transitions
    )
    assert episode.transitions[-1].reward == 1.0 and episode.transitions[-1].terminated
    assert result.error is result.traceback is None
    assert calls == [{}] and environment.reset_calls == [17]
    assert len(environment.commands) == 2 and environment.close_calls == 1
    assert [
        field.name for field in fields(collector.GoalConsistentObservedGraspTaskTeacherResult)
    ] == [field.name for field in fields(collector.ObservedGraspTaskTeacherResult)]


@pytest.mark.parametrize(
    "damage",
    [
        "old_definition",
        "old_id",
        "base_erasure",
        "observed_erasure",
        "v1_relabel",
        "final_false",
        "final_integer",
        "feedback",
    ],
)
def test_should_refuse_family_erasure_relabeling_and_missing_actual_final_success(
    collected, damage
):
    selected: Any = deepcopy(collected[True][0])
    if damage == "old_definition":
        selected = replace(selected, teacher_definition=OBSERVED_GRASP_DEFINITION)
    elif damage == "old_id":
        selected = replace(
            selected,
            candidate_episode=replace(
                selected.candidate_episode, episode_id="pick-cube-observed-grasp-teacher-17"
            ),
        )
    elif damage in ("base_erasure", "observed_erasure"):
        cls = (
            collector.TaskTeacherResult
            if damage == "base_erasure"
            else collector.ObservedGraspTaskTeacherResult
        )
        selected = cls(**{field.name: getattr(selected, field.name) for field in fields(cls)})
    elif damage == "v1_relabel":
        old = deepcopy(collected[False][0])
        selected = replace(
            old,
            teacher_definition=GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
            candidate_episode=replace(
                old.candidate_episode,
                episode_id="pick-cube-goal-consistent-observed-grasp-teacher-17",
            ),
        )
    elif damage in ("final_false", "final_integer"):
        object.__setattr__(selected, "final_success", False if damage == "final_false" else 1)
    else:
        selected = replace(selected, observed_grasp=(False, True))
    assert selected.qualified is False
    with pytest.raises(ValueError):
        selected.training_episode()


@pytest.mark.parametrize("fault", ["integer", "none", "unobserved", "visual"])
def test_should_refuse_goal_mode_without_exact_observed_nonvisual_admission(monkeypatch, fault):
    monkeypatch.setattr(collector, "ManiSkillPickCubeEpisode", forbidden)
    options: dict[str, Any] = dict(observed_grasp=True, goal_consistent=True)
    if fault == "integer":
        options["goal_consistent"] = 1
    elif fault == "none":
        options["goal_consistent"] = None
    elif fault == "unobserved":
        options["observed_grasp"] = False
    else:
        options["visual_observation"] = True
    with pytest.raises(ValueError):
        collector.collect_readiness_teacher(**options)


@pytest.fixture(scope="module")
def cli_receipts(collected, tmp_path_factory):
    directory = tmp_path_factory.mktemp("goal-consistent-cli")
    records = {}
    for selected in (False, True):
        result = collected[selected][0]
        output = directory / f"selected-{selected}.json"
        calls = []

        def collect(**options):
            calls.append(options)
            return result

        args = ["teacher", "--output", str(output), "--geometry-readiness", "--observed-grasp"]
        if selected:
            args.append("--goal-consistent")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(cli, "collect_readiness_teacher", collect)
            patch.setattr(cli, "version", lambda _: "fake-goal-consistent")
            patch.setattr(sys, "argv", args)
            status = cli.main()
        assert output.stat().st_size < 1024**2
        records[selected] = (status, calls, output, json.loads(output.read_text(encoding="utf8")))
    return records


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_goal_true_only_when_selected_and_keep_paid_artifact_version(
    cli_receipts, selected
):
    status, calls, _, payload = cli_receipts[selected]
    expected = dict(observed_grasp=True, **({"goal_consistent": True} if selected else {}))
    assert status == 0 and payload["qualified"] is True and calls == [expected]
    assert payload["observed_grasp"] == [True, True]
    assert payload["teacher_definition"] == (
        GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION if selected else OBSERVED_GRASP_DEFINITION
    )
    assert payload["teacher_actions"] == payload["work"]["completed_steps"] == 2


def test_should_keep_new_v2_artifact_out_of_the_existing_v1_selected_reader(cli_receipts):
    _, _, path, payload = cli_receipts[True]
    assert payload["qualified"] is True
    with pytest.raises(ValueError):
        reader._read_readiness_teacher(path, 17, observed_grasp=True)


@pytest.mark.parametrize("requested_goal", [False, True])
def test_should_retain_paid_wrong_family_output_without_qualifying_a_different_request(
    collected, monkeypatch, tmp_path, requested_goal
):
    wrong = collected[not requested_goal][0]
    monkeypatch.setattr(cli, "collect_readiness_teacher", lambda **_: wrong)
    monkeypatch.setattr(cli, "version", lambda _: "fake-goal-consistent")
    output = tmp_path / "wrong-family.json"
    args = ["teacher", "--output", str(output), "--geometry-readiness", "--observed-grasp"]
    if requested_goal:
        args.append("--goal-consistent")
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 1
    assert output.stat().st_size < 1024**2
    payload = json.loads(output.read_text(encoding="utf8"))
    assert payload["qualified"] is False and payload["readiness_request_error"]
    assert payload["candidate_episode"] is not None and payload["teacher_actions"] == 2
    assert wrong.qualified is True


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--geometry-readiness"],
        ["--geometry-readiness", "--observed-grasp", "--visual-observation"],
        ["--geometry-readiness", "--observed-grasp", "--mass-drift"],
        [
            "--geometry-readiness",
            "--observed-grasp",
            "--guard-seed",
            "31",
            "--labels-arrived-at",
            "1",
        ],
    ],
)
def test_should_refuse_goal_cli_cross_scope_before_output_or_collection(
    monkeypatch, tmp_path, extra
):
    monkeypatch.setattr(cli, "collect_readiness_teacher", forbidden)
    monkeypatch.setattr(cli, "collect_task_teacher", forbidden)
    monkeypatch.setattr(cli, "collect_guard_teacher", forbidden)
    output = tmp_path / "not-created.json"
    monkeypatch.setattr(
        sys, "argv", ["teacher", "--output", str(output), "--goal-consistent", *extra]
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2 and not output.exists()
