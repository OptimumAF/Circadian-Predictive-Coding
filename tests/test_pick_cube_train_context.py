"""Declared TRAIN reset contexts through fake native IO; no task generalization claim."""

import inspect
import json
import sys
from typing import Any

import numpy as np
import pytest

from test_maniskill_bc_smoke import FakeEnvironment as ActorEnvironment, learner as mlp_learner
from test_maniskill_task_teacher import FakeEnvironment as TeacherEnvironment
from test_nearest_continuous_bc import native_learner as nearest_learner
from scripts import maniskill_continuous_smoke as actor
from scripts import maniskill_task_teacher_smoke as teacher_cli
from src.core.pick_cube_train_context import pick_cube_training_context
from src.infra import maniskill_continuous as port
from src.infra import maniskill_task_teacher as teacher


@pytest.mark.parametrize("seed,context", [(17, "pick-cube"), (19, "pick-cube-train-19")])
def test_should_preserve_the_original_declared_training_contexts(seed, context):
    assert pick_cube_training_context(seed) == context


@pytest.mark.parametrize(
    "seed", [-1, 0, 18, 21, 23, 29, 2**32, True, 17.0, "17", np.int64(17), None]
)
def test_should_refuse_reserved_or_non_native_training_seeds(seed):
    with pytest.raises(ValueError):
        pick_cube_training_context(seed)


@pytest.fixture
def teacher_environment(monkeypatch):
    environment = TeacherEnvironment()
    environment.done_at = 3
    creations = []

    def factory():
        creations.append(True)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    return environment, creations


@pytest.mark.parametrize("seed,context", [(17, "pick-cube"), (19, "pick-cube-train-19")])
def test_should_collect_actual_declared_reset_and_preserve_teacher_controls_and_train_role(
    teacher_environment, seed, context
):
    environment, creations = teacher_environment
    result = (
        teacher.collect_task_teacher()
        if seed == 17
        else teacher.collect_task_teacher(train_seed=seed)
    )
    assert result.qualified and result.final_success and result.error is None
    assert (
        environment.reset_calls == [seed] and creations == [True] and environment.close_calls == 1
    )
    data = result.training_episode()
    assert (data.episode_id, data.context, data.role) == (
        f"pick-cube-task-teacher-{seed}",
        context,
        "train",
    )
    assert result.teacher_actions == result.geometry_attempts == result.geometry_returns == 3
    assert result.stages == ("align",) * 3
    assert result.work == port.ManiSkillWork(3, 3, 15, 0)
    assert data.transitions[0].action == pytest.approx((0.2, -0.3, 0.9, 1.0))
    assert data.transitions[-1].truncated is True
    for transition, command in zip(data.transitions, environment.commands, strict=True):
        assert transition.action == tuple(float(value) for value in command[0])


@pytest.mark.parametrize("seed", [23, 29, 18, True, 19.0, "19", np.int64(19)])
def test_should_refuse_invalid_teacher_seed_before_port_creation(monkeypatch, seed):
    monkeypatch.setattr(
        teacher,
        "ManiSkillPickCubeEpisode",
        lambda *_args, **_kwargs: pytest.fail("invalid TRAIN seed reached port creation"),
    )
    with pytest.raises(ValueError):
        teacher.collect_task_teacher(train_seed=seed)


def test_should_expose_bounded_training_seed_visual_mode_and_explicit_camera_keywords():
    parameters = inspect.signature(teacher.collect_task_teacher).parameters
    assert list(parameters) == [
        "train_seed",
        "visual_observation",
        "camera_context",
        "mass_context",
        "lighting_context",
    ]
    assert parameters["train_seed"].default == 17
    assert parameters["train_seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["visual_observation"].default is False
    assert parameters["visual_observation"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["camera_context"].default is None
    assert parameters["camera_context"].kind is inspect.Parameter.KEYWORD_ONLY
    for name in ("mass_context", "lighting_context"):
        assert parameters[name].default is None
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_should_preserve_failed_teacher19_work_without_qualifying_data(teacher_environment):
    environment, creations = teacher_environment
    environment.step_failure_at = 2
    result = teacher.collect_task_teacher(train_seed=19)
    assert not result.qualified and result.candidate_episode is None
    assert result.error is not None and "physics failed" in result.error
    assert environment.reset_calls == [19] and creations == [True] and environment.close_calls == 1
    assert result.work == port.ManiSkillWork(2, 1, 5, 5)
    assert result.teacher_actions == result.geometry_attempts == result.geometry_returns == 2


@pytest.mark.parametrize("seed", [17, 19])
def test_should_emit_teacher_seed_context_and_preserve_default_mock_call(
    teacher_environment, monkeypatch, tmp_path, seed
):
    result = teacher.collect_task_teacher(train_seed=seed)
    calls = []

    def collect(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(teacher_cli, "collect_task_teacher", collect)
    monkeypatch.setattr(teacher_cli, "version", lambda _: "test-version")
    monkeypatch.setattr(teacher_cli.faulthandler, "enable", lambda: None)
    output = tmp_path / "teacher.json"
    arguments = ["teacher", "--output", str(output)]
    if seed == 19:
        arguments += ["--train-seed", "19"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert teacher_cli.main() == 0
    payload = json.loads(output.read_text(encoding="utf8"))
    assert calls == [((), {} if seed == 17 else {"train_seed": 19})]
    assert payload["seed"] == seed and payload["context"] == pick_cube_training_context(seed)
    assert payload["role"] == payload["candidate_episode"]["role"] == "train"
    assert payload["candidate_episode"]["context"] == payload["context"]
    assert str(seed) in payload["explicit_reset"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        teacher_cli.main()
    assert output.read_bytes() == before and len(calls) == 1


@pytest.mark.parametrize("seed", ["23", "29", "18"])
def test_should_refuse_reserved_teacher_cli_seeds_before_output_or_collection(
    monkeypatch, tmp_path, seed
):
    output = tmp_path / "forbidden-teacher.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), "--train-seed", seed])
    monkeypatch.setattr(
        teacher_cli,
        "collect_task_teacher",
        lambda *_args, **_kwargs: pytest.fail("reserved CLI seed reached teacher collection"),
    )
    with pytest.raises(SystemExit) as caught:
        teacher_cli.main()
    assert caught.value.code == 2 and not output.exists()


@pytest.fixture
def actor_environment(monkeypatch, tmp_path):
    environment = ActorEnvironment()
    creations = []

    def factory():
        creations.append(True)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(actor, "version", lambda _: "fake-native-boundary")
    output = tmp_path / "actor.json"
    monkeypatch.setattr(sys, "argv", ["actor", "--output", str(output)])
    return environment, creations, output


@pytest.mark.parametrize("seed", [17, 19])
@pytest.mark.parametrize("family", ["zero", "mlp", "nearest"])
def test_should_bind_actual_actor_reset_context_and_trace_for_each_existing_policy(
    actor_environment, seed, family
):
    environment, creations, output = actor_environment
    model = {"zero": lambda: None, "mlp": mlp_learner, "nearest": nearest_learner}[family]()
    assert actor.main(learner=model, train_seed=seed) == 0
    result = json.loads(output.read_text(encoding="utf8"))
    assert (
        environment.reset_calls == [seed] and creations == [True] and environment.close_calls == 1
    )
    assert result["seed"] == seed and result["context"] == pick_cube_training_context(seed)
    assert result["role"] == result["episode"]["role"] == "train"
    assert result["episode"]["episode_id"] == f"pick-cube-smoke-{seed}"
    assert result["episode"]["context"] == result["context"]
    assert str(seed) in result["explicit_reset"]
    assert result["complete"] is True and result["optimizer_updates"] == 0
    assert result["work"] == dict(
        step_attempts=2,
        completed_steps=2,
        completed_control_physics_steps=10,
        possible_failed_physics_steps=0,
    )
    if model is not None:
        assert result["frozen_state_exact"] is True and result["original_owner_exact"] is True


@pytest.mark.parametrize("seed", [23, 29, 18, True, 19.0, "17", np.int64(19), None])
def test_should_preserve_invalid_programmatic_seed_receipt_before_version_or_scene(
    actor_environment, monkeypatch, seed
):
    environment, creations, output = actor_environment
    monkeypatch.setattr(
        actor, "version", lambda _: pytest.fail("invalid seed reached runtime metadata query")
    )
    assert actor.main(train_seed=seed) == 1
    result = json.loads(output.read_text(encoding="utf8"))
    assert result["seed"] is None and result["context"] is None
    assert result["complete"] is False and result["error"]
    assert result["creation_attempts"] == result["creation_returns"] == 0
    assert result["prediction_attempts"] == 0 and result["work"] is None
    assert creations == [] and environment.reset_calls == [] and environment.commands == []


@pytest.mark.parametrize("keyword_seed,cli_seed", [(17, 19), (19, 17)])
def test_should_use_explicit_actor_cli_seed_over_valid_programmatic_default(
    actor_environment, monkeypatch, keyword_seed, cli_seed
):
    environment, _, output = actor_environment
    monkeypatch.setattr(
        sys, "argv", ["actor", "--output", str(output), "--train-seed", str(cli_seed)]
    )
    assert actor.main(train_seed=keyword_seed) == 0
    result = json.loads(output.read_text(encoding="utf8"))
    assert result["seed"] == cli_seed and result["context"] == pick_cube_training_context(cli_seed)
    assert environment.reset_calls == [cli_seed]


@pytest.mark.parametrize("seed", ["23", "29"])
def test_should_refuse_reserved_actor_cli_seed_before_output_or_native_work(
    actor_environment, monkeypatch, seed
):
    environment, creations, output = actor_environment
    monkeypatch.setattr(sys, "argv", ["actor", "--output", str(output), "--train-seed", seed])
    with pytest.raises(SystemExit) as caught:
        actor.main()
    assert caught.value.code == 2 and not output.exists()
    assert creations == [] and environment.reset_calls == []


def test_should_validate_seed_before_clock_or_learner_configuration(actor_environment, monkeypatch):
    environment, creations, output = actor_environment
    monkeypatch.setattr(
        actor, "_prepare_learner", lambda *_: pytest.fail("invalid seed reached learner validation")
    )
    monkeypatch.setattr(
        actor.clocks, "_clock", lambda *_: pytest.fail("invalid seed reached clock validation")
    )
    invalid: Any = object()
    assert actor.main(train_seed=23, learner=invalid, observation_clock=invalid) == 1
    result = json.loads(output.read_text(encoding="utf8"))
    assert "TRAIN PickCube seed" in result["error"]
    assert result["seed"] is None and result["context"] is None
    assert creations == [] and environment.reset_calls == []


def test_should_preserve_reset19_failure_identity_and_unknown_controller_work(actor_environment):
    environment, creations, output = actor_environment

    # The existing fake has no reset-failure option, so fail at the native boundary.
    def reset(*, seed):
        environment.reset_calls.append(seed)
        raise RuntimeError("seed19 reset failed")

    environment.reset = reset
    assert actor.main(train_seed=19) == 1
    result = json.loads(output.read_text(encoding="utf8"))
    assert result["seed"] == 19 and result["context"] == "pick-cube-train-19"
    assert result["creation_attempts"] == 1 and result["creation_returns"] == 0
    assert result["work"] is None and result["prediction_attempts"] == 0
    assert "seed19 reset failed" in result["error"]
    assert creations == [True] and environment.reset_calls == [19] and environment.close_calls == 1


def test_should_reserve_existing_actor_receipt_before_seed19_native_work(actor_environment):
    environment, creations, output = actor_environment
    output.write_bytes(b"preserved evidence\n")
    with pytest.raises(FileExistsError):
        actor.main(train_seed=19)
    assert output.read_bytes() == b"preserved evidence\n"
    assert creations == [] and environment.reset_calls == [] and environment.close_calls == 0
