"""One declared camera displacement through fake native IO, with zero learning.

All models are initialized/restored, never trained; no nearest prototype append.
Fake frames verify encoding and routing, not an actual rendered camera effect.
"""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import maniskill_task_teacher_smoke as teacher_cli
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.pick_cube_camera_context import (
    PickCubeCameraContext,
    camera_context_recipe,
    pick_cube_camera_context,
    require_camera_binding,
    validate_camera_recipe,
    validate_pick_cube_camera_context,
)
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe, validate_pick_cube_rgb_recipe
from src.infra import maniskill_continuous as port
from src.infra import maniskill_task_teacher as teacher
from test_maniskill_bc_smoke import assert_same_state, learner
from test_maniskill_rgb_observations import RGBEnvironment, public_feature_oracle


CONTEXT = "pick-cube-camera-y005-train-17"
RECIPE = "pick_cube_rgb_pool16_qpos_qvel_goal_camera_y005_v1"


def options(**changes):
    values = dict(context=CONTEXT, seed=17, role="train", visual_observation=True)
    values.update(changes)
    return values


def model():
    spec = ContinuousActionSpec(789, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    return learner(spec=spec)


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment, calls = RGBEnvironment(), []

    def factory(**kwargs):
        calls.append(kwargs)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-camera-context")
    monkeypatch.setattr(teacher_cli, "version", lambda _: "fake-camera-context")
    monkeypatch.setattr(teacher_cli.faulthandler, "enable", lambda: None)
    return environment, calls, tmp_path


def test_should_freeze_exact_known_context_and_change_only_camera_eye_and_recipe_id():
    context = pick_cube_camera_context()
    assert (
        type(context) is PickCubeCameraContext and context.context == CONTEXT and context.seed == 17
    )
    assert pick_cube_camera_context() is not context
    owned = validate_pick_cube_camera_context(context)
    assert owned == context and owned is not context
    with pytest.raises(FrozenInstanceError):
        context.seed = 19
    recipe = camera_context_recipe(context)
    assert recipe.recipe_id == RECIPE
    assert recipe.camera_eye == (0.3, 0.05, 0.6)
    assert recipe.camera_target == (-0.1, 0.0, 0.1)
    assert (
        replace(recipe, recipe_id=pick_cube_rgb_recipe().recipe_id, camera_eye=(0.3, 0.0, 0.6))
        == pick_cube_rgb_recipe()
    )
    assert validate_camera_recipe(recipe, context) == recipe
    assert validate_camera_recipe(pick_cube_rgb_recipe()) == pick_cube_rgb_recipe()
    with pytest.raises(ValueError):
        validate_pick_cube_rgb_recipe(recipe)
    with pytest.raises(ValueError):
        validate_camera_recipe(recipe)
    with pytest.raises(ValueError):
        validate_camera_recipe(pick_cube_rgb_recipe(), context)


@pytest.mark.parametrize(
    "changes",
    [
        dict(seed=True),
        dict(seed=17.0),
        dict(seed=np.int64(17)),
        dict(seed=19),
        dict(context="pick-cube"),
        dict(context=CONTEXT + " "),
        dict(context=None),
    ],
)
def test_should_refuse_arbitrary_or_untyped_camera_context(changes):
    with pytest.raises(ValueError):
        validate_pick_cube_camera_context(PickCubeCameraContext(**changes))


@pytest.mark.parametrize(
    "damage", ["none", "mapping", "subclass", "mutated_seed", "mutated_context"]
)
def test_should_revalidate_exact_context_records(damage):
    context = pick_cube_camera_context()
    if damage == "none":
        context = None
    elif damage == "mapping":
        context = asdict(context)
    elif damage == "subclass":

        class Derived(PickCubeCameraContext):
            pass

        context = Derived()
    elif damage == "mutated_seed":
        object.__setattr__(context, "seed", 19)
    else:
        object.__setattr__(context, "context", "pick-cube")
    with pytest.raises(ValueError):
        validate_pick_cube_camera_context(context)


@pytest.mark.parametrize(
    "changes",
    [
        dict(seed=19),
        dict(seed=True),
        dict(context="pick-cube"),
        dict(role="inner_guard"),
        dict(role="probe_train"),
        dict(role="probe_test"),
        dict(role="final_test"),
        dict(visual_observation=False),
        dict(visual_observation=1),
    ],
)
def test_should_bind_only_explicit_visual_train17_context(changes):
    with pytest.raises(ValueError):
        require_camera_binding(pick_cube_camera_context(), **options(**changes))


def test_should_require_explicit_metadata_for_reserved_context_and_preserve_defaults():
    assert (
        require_camera_binding(None, **options(context="pick-cube", visual_observation=False))
        is None
    )
    with pytest.raises(ValueError):
        require_camera_binding(None, **options())
    original = pick_cube_camera_context()
    owned = require_camera_binding(original, **options())
    object.__setattr__(original, "seed", 19)
    assert owned.seed == 17 and owned.context == CONTEXT


@pytest.mark.parametrize(
    "changes",
    [
        dict(camera_eye=(0.3, 0.051, 0.6)),
        dict(camera_target=(-0.1, 0.01, 0.1)),
        dict(feature_dim=789.0),
        dict(pixel_divisor=255),
        dict(recipe_id=pick_cube_rgb_recipe().recipe_id),
    ],
)
def test_should_refuse_changed_or_equal_wrong_type_camera_recipe(changes):
    context = pick_cube_camera_context()
    with pytest.raises(ValueError):
        validate_camera_recipe(replace(camera_context_recipe(context), **changes), context)


@pytest.mark.parametrize(
    "changes",
    [
        dict(seed=19),
        dict(role="inner_guard"),
        dict(role="probe_train"),
        dict(role="probe_test"),
        dict(role="final_test"),
        dict(context="pick-cube"),
        dict(visual_observation=False),
    ],
)
def test_should_refuse_invalid_port_camera_binding_before_factory(boundary, changes):
    environment, calls, _ = boundary
    with pytest.raises(ValueError):
        port.ManiSkillPickCubeEpisode(
            "camera-port", camera_context=pick_cube_camera_context(), **options(**changes)
        )
    assert calls == [] and environment.reset_calls == []


def test_should_forward_owned_camera_and_preserve_public_pixels_actual_controls_and_work(boundary):
    environment, calls, _ = boundary
    original = pick_cube_camera_context()
    session = port.ManiSkillPickCubeEpisode("camera-port", camera_context=original, **options())
    assert calls == [dict(visual_observation=True, camera_context=pick_cube_camera_context())]
    assert calls[0]["camera_context"] is not original
    assert environment.reset_calls == [17]
    assert session.observation_recipe == camera_context_recipe(pick_cube_camera_context())
    initial = session.observation
    assert initial == tuple(public_feature_oracle(environment.reset_packet))
    assert session.spec.observation_dim == 789 and session.spec.action_units == port.ACTION_UNITS
    object.__setattr__(original, "seed", 19)
    first = session.step((0.123456789, -1.0, 1.0, 0.0))
    second = session.step((0.0,) * 4)
    assert first.observation == initial and first.next_observation == second.observation
    assert first.action == tuple(float(value) for value in environment.commands[0][0])
    assert first.action[0] != 0.123456789
    assert environment.commands[0].dtype == np.float32
    assert session.episode().context == CONTEXT and session.episode().role == "train"
    assert session.visual_work.observation_attempts == session.visual_work.observation_returns == 3
    assert session.work == port.ManiSkillWork(2, 2, 10, 0)
    session.close()
    assert environment.close_calls == 1


@pytest.mark.parametrize("phase", ["before_control", "after_return", "extraction"])
@pytest.mark.parametrize("damage", ["seed", "removed_binding", "equal_untyped_getter"])
def test_should_refuse_changed_direct_port_binding_and_keep_paid_work(
    boundary, monkeypatch, phase, damage
):
    environment, _, _ = boundary
    session = port.ManiSkillPickCubeEpisode(
        "camera-port", camera_context=pick_cube_camera_context(), **options()
    )

    def corrupt():
        if damage == "seed":
            session._seed = 19
        elif damage == "removed_binding":
            session._camera_context = None
        else:
            malformed = replace(pick_cube_camera_context(), seed=17.0)
            monkeypatch.setattr(
                port.ManiSkillPickCubeEpisode, "camera_context", property(lambda _: malformed)
            )

    def repair():
        session._seed = 17
        session._camera_context = pick_cube_camera_context()
        if damage == "equal_untyped_getter":
            monkeypatch.setattr(
                port.ManiSkillPickCubeEpisode,
                "camera_context",
                property(lambda _: pick_cube_camera_context()),
            )

    if phase == "extraction":
        session.step((0.0,) * 4)
        session.step((0.0,) * 4)
        corrupt()
        with pytest.raises(ValueError):
            session.episode()
        repair()
        with pytest.raises(ValueError):
            session.episode()
        expected = 2
    else:
        if phase == "before_control":
            corrupt()
        else:
            original = environment.step

            def returned(command):
                result = original(command)
                corrupt()
                return result

            monkeypatch.setattr(environment, "step", returned)
        with pytest.raises(ValueError):
            session.step((0.0,) * 4)
        repair()
        with pytest.raises(ValueError):
            session.step((0.0,) * 4)
        with pytest.raises(ValueError):
            session.episode()
        expected = int(phase == "after_return")
    assert len(environment.commands) == expected
    assert session.work == port.ManiSkillWork(expected, expected, expected * 5, 0)
    session.close()
    assert environment.close_calls == 1


def test_should_pass_exact_frozen_camera_pose_to_existing_sdk_factory(monkeypatch):
    names = (
        "mani_skill",
        "mani_skill.envs",
        "mani_skill.envs.tasks",
        "mani_skill.envs.tasks.tabletop",
        "mani_skill.envs.tasks.tabletop.pick_cube",
        "mani_skill.utils",
        "mani_skill.utils.registration",
        "mani_skill.utils.sapien_utils",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)

    class Registered:
        pass

    modules["mani_skill.envs.tasks.tabletop.pick_cube"].PickCubeEnv = Registered
    calls, poses = [], []
    environment = RGBEnvironment()
    modules["mani_skill.utils.registration"].REGISTERED_ENVS = {
        port.TASK_ID: SimpleNamespace(
            cls=Registered,
            asset_download_ids=(),
            max_episode_steps=50,
            make=lambda **kwargs: calls.append(kwargs) or environment,
        )
    }
    modules["mani_skill.utils.registration"].TimeLimitWrapper = lambda value, horizon: value
    modules["mani_skill.utils.sapien_utils"].look_at = lambda eye, target: (
        poses.append((eye, target)) or "declared-camera-pose"
    )
    monkeypatch.setattr(port, "_require_cpu_kinematics", lambda: None)
    monkeypatch.setattr(
        port, "version", lambda name: {"mani_skill": "3.0.1", "sapien": "3.0.3"}[name]
    )
    assert (
        port._make_environment(visual_observation=True, camera_context=pick_cube_camera_context())
        is environment
    )
    assert poses == [((0.3, 0.05, 0.6), (-0.1, 0.0, 0.1))]
    assert calls[0]["control_mode"] == port.CONTROL_MODE
    assert calls[0]["obs_mode"] == "rgb" and calls[0]["render_backend"] == "sapien_cpu"
    assert calls[0]["sensor_configs"]["base_camera"]["pose"] == "declared-camera-pose"
    assert calls[0]["sensor_configs"]["base_camera"]["width"] == 128


def test_should_collect_owned_camera_teacher_with_same_train17_physics_and_public_features(
    boundary,
):
    environment, calls, _ = boundary
    context = pick_cube_camera_context()
    result = teacher.collect_task_teacher(visual_observation=True, camera_context=context)
    assert result.qualified and result.error is None and result.final_success is True
    assert (
        result.camera_context == pick_cube_camera_context() and result.camera_context is not context
    )
    assert result.observation_recipe == camera_context_recipe(pick_cube_camera_context())
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    assert calls == [dict(visual_observation=True, camera_context=pick_cube_camera_context())]
    assert environment.reset_calls == [17] and environment.close_calls == 1
    extracted = result.training_episode()
    assert extracted.episode_id == f"{CONTEXT}-teacher" and extracted.context == CONTEXT
    assert extracted.role == "train" and extracted.spec.observation_dim == 789
    assert extracted.transitions[0].observation == tuple(
        public_feature_oracle(environment.reset_packet)
    )
    assert extracted.transitions[0].action == pytest.approx((0.2, -0.3, 0.9, 1.0))
    assert extracted is not result.candidate_episode
    object.__setattr__(context, "seed", 19)
    assert result.qualified and result.training_episode() == extracted
    object.__setattr__(extracted.transitions[0], "action", (0.0,) * 4)
    assert result.training_episode().transitions[0].action != (0.0,) * 4


@pytest.mark.parametrize(
    "damage",
    [
        "lost_context",
        "lost_context_relabel_context",
        "lost_context_relabel_identity",
        "wrong_recipe",
        "identity",
        "context",
        "role",
        "bounds",
        "lost_recipe",
        "terminal_rgb",
        "guard_metadata",
        "observed_type",
        "final_type",
    ],
)
def test_should_refuse_relabeling_or_corrupted_camera_teacher_qualification(boundary, damage):
    result = teacher.collect_task_teacher(
        visual_observation=True, camera_context=pick_cube_camera_context()
    )
    data = result.candidate_episode
    if damage == "lost_context":
        result = replace(result, camera_context=None)
    elif damage in ("lost_context_relabel_context", "lost_context_relabel_identity"):
        data = replace(data, context="pick-cube")
        if damage == "lost_context_relabel_identity":
            data = replace(data, episode_id="pick-cube-task-teacher-17")
        result = replace(result, camera_context=None, candidate_episode=data)
    elif damage == "wrong_recipe":
        result = replace(result, observation_recipe=pick_cube_rgb_recipe())
    elif damage == "identity":
        result = replace(
            result, candidate_episode=replace(data, episode_id="pick-cube-task-teacher-17")
        )
    elif damage == "context":
        result = replace(result, candidate_episode=replace(data, context="pick-cube"))
    elif damage == "role":
        result = replace(result, candidate_episode=replace(data, role="probe_train"))
    elif damage == "bounds":
        result = replace(
            result,
            candidate_episode=replace(data, spec=replace(data.spec, upper_bounds=(2.0,) * 4)),
        )
    elif damage == "lost_recipe":
        result = replace(result, observation_recipe=None)
    elif damage == "terminal_rgb":
        after = list(data.transitions[-1].next_observation)
        after[-1] = -0.01
        result = replace(
            result,
            candidate_episode=replace(
                data,
                transitions=data.transitions[:-1]
                + (replace(data.transitions[-1], next_observation=tuple(after)),),
            ),
        )
    elif damage == "guard_metadata":
        result = replace(result, guard_seed=31)
    elif damage == "observed_type":
        result = replace(result, observed_success=1)
    else:
        result = replace(result, final_success=1)
    assert not result.qualified
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize(
    "changes",
    [
        dict(train_seed=19),
        dict(visual_observation=False),
        dict(train_seed=True),
        dict(camera_context={"seed": 17, "context": CONTEXT}),
    ],
)
def test_should_refuse_invalid_teacher_camera_request_before_port(boundary, changes):
    environment, calls, _ = boundary
    arguments = dict(visual_observation=True, camera_context=pick_cube_camera_context())
    arguments.update(changes)
    with pytest.raises(ValueError):
        teacher.collect_task_teacher(**arguments)
    assert calls == [] and environment.reset_calls == []


@pytest.mark.parametrize("phase", ["after_geometry", "after_step", "cleanup"])
def test_should_stop_teacher_on_late_camera_or_recipe_swap_and_keep_paid_work(
    boundary, monkeypatch, phase
):
    environment, _, _ = boundary
    geometry = teacher._geometry
    recipe_property = port.ManiSkillPickCubeEpisode.observation_recipe

    if phase == "after_geometry":

        def changed_geometry(session):
            result = geometry(session)
            session._camera_context = None
            return result

        monkeypatch.setattr(teacher, "_geometry", changed_geometry)
    else:

        def observed_recipe(session):
            value = recipe_property.fget(session)
            drift = (
                bool(environment.commands)
                if phase == "after_step"
                else bool(environment.close_calls)
            )
            return pick_cube_rgb_recipe() if drift else value

        monkeypatch.setattr(
            port.ManiSkillPickCubeEpisode, "observation_recipe", property(observed_recipe)
        )
    result = teacher.collect_task_teacher(
        visual_observation=True, camera_context=pick_cube_camera_context()
    )
    assert not result.qualified and result.error
    assert environment.close_calls == 1
    assert len(environment.commands) == {"after_geometry": 0, "after_step": 1, "cleanup": 2}[phase]
    assert (
        result.geometry_attempts
        == result.geometry_returns
        == result.teacher_actions
        == {"after_geometry": 1, "after_step": 1, "cleanup": 2}[phase]
    )
    assert result.visual_work.observation_returns == len(environment.commands) + 1
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_preserve_default_visual_factory_and_teacher_metadata(boundary):
    environment, calls, _ = boundary
    result = teacher.collect_task_teacher(visual_observation=True)
    assert result.qualified and result.camera_context is None
    assert result.observation_recipe == pick_cube_rgb_recipe()
    assert result.training_episode().episode_id == "pick-cube-task-teacher-17"
    assert calls == [dict(visual_observation=True)] and environment.reset_calls == [17]


@pytest.mark.parametrize("entry", ["teacher", "rollout"])
@pytest.mark.parametrize("phase", ["after_step", "cleanup"])
def test_should_reject_equal_but_untyped_returned_camera_metadata(
    boundary, monkeypatch, entry, phase
):
    environment, _, directory = boundary
    getter = port.ManiSkillPickCubeEpisode.camera_context

    def observed(session):
        changed = (
            bool(environment.commands) if phase == "after_step" else bool(environment.close_calls)
        )
        return replace(pick_cube_camera_context(), seed=17.0) if changed else getter.fget(session)

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "camera_context", property(observed))
    if entry == "teacher":
        result = teacher.collect_task_teacher(
            visual_observation=True, camera_context=pick_cube_camera_context()
        )
        assert not result.qualified and result.error
        with pytest.raises(ValueError):
            result.training_episode()
        assert result.work.completed_steps == (1 if phase == "after_step" else 2)
    else:
        result = smoke.run_frozen_rollout(
            directory / "typed-metadata.json",
            visual_observation=True,
            camera_context=pick_cube_camera_context(),
        )
        assert result.report["complete"] is False and result.report["visual_recipe_exact"] is None
        assert result.report["visual_recipe_error"]
        assert result.report["work"]["completed_steps"] == (1 if phase == "after_step" else 2)
    assert environment.close_calls == 1


def test_should_refuse_reserved_camera_context_without_declared_port_metadata(boundary):
    environment, calls, _ = boundary
    with pytest.raises(ValueError):
        port.ManiSkillPickCubeEpisode("missing-camera", **options())
    assert calls == [] and environment.reset_calls == []


def test_should_run_actual_frozen_learner_on_public_camera_features_without_training(
    boundary, monkeypatch
):
    environment, calls, directory = boundary
    owner = model()
    before = owner.snapshot_state()
    predict, features = NumpyContinuousBCLearner.predict, []

    def observed(child, values):
        features.append(values.copy())
        return predict(child, values)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", observed)
    monkeypatch.setattr(
        NumpyContinuousBCLearner,
        "train_batch",
        lambda *_a, **_k: pytest.fail("camera rollout trained"),
    )
    output = directory / "camera-rollout.json"
    outcome = smoke.run_frozen_rollout(
        output, learner=owner, visual_observation=True, camera_context=pick_cube_camera_context()
    )
    result = outcome.report
    assert result["complete"] is True and outcome.corrections is None
    assert result["camera_context"] == asdict(pick_cube_camera_context())
    assert result["context"] == CONTEXT and result["seed"] == 17 and result["role"] == "train"
    assert result["observation_recipe"] == asdict(camera_context_recipe(pick_cube_camera_context()))
    assert (
        result["visual_recipe_exact"]
        is result["frozen_state_exact"]
        is result["original_owner_exact"]
        is True
    )
    assert (
        result["optimizer_updates"] == 0
        and result["prediction_attempts"] == result["prediction_returns"] == 2
    )
    assert result["validated_prediction_bytes"] == 64
    assert result["visual_work"]["observation_returns"] == 3
    assert result["visual_work"]["returned_feature_values"] == 3 * 789
    assert result["work"]["completed_control_physics_steps"] == 10
    assert calls == [dict(visual_observation=True, camera_context=pick_cube_camera_context())]
    assert_same_state(before, owner.snapshot_state())
    expected = np.tanh(np.array([[0.2, -0.3, 0.4, -0.5]])).astype(np.float32)
    for index, command in enumerate(environment.commands):
        np.testing.assert_array_equal(command, expected)
        row = result["episode"]["transitions"][index]
        assert row["action"] == tuple(float(value) for value in command[0])
        np.testing.assert_array_equal(
            features[index], np.asarray([row["observation"]], dtype=np.float64)
        )
    assert features[0][0].tolist() == public_feature_oracle(environment.reset_packet)
    preserved = output.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.run_frozen_rollout(
            output,
            learner=owner,
            visual_observation=True,
            camera_context=pick_cube_camera_context(),
        )
    assert output.read_bytes() == preserved and environment.close_calls == 1


@pytest.mark.parametrize("phase", ["after_step", "cleanup", "camera_context"])
def test_should_reject_late_rollout_camera_binding_and_keep_completed_work(
    boundary, monkeypatch, phase
):
    environment, _, directory = boundary
    original_step, original_close = environment.step, environment.close
    recipe_property = port.ManiSkillPickCubeEpisode.observation_recipe
    context_property = port.ManiSkillPickCubeEpisode.camera_context
    changed = []
    if phase in ("after_step", "camera_context"):

        def step(command):
            result = original_step(command)
            changed.append(True)
            return result

        monkeypatch.setattr(environment, "step", step)
    else:

        def close():
            original_close()
            changed.append(True)

        monkeypatch.setattr(environment, "close", close)

    if phase == "camera_context":

        def camera(session):
            return None if changed else context_property.fget(session)

        monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "camera_context", property(camera))
    else:

        def recipe(session):
            return pick_cube_rgb_recipe() if changed else recipe_property.fget(session)

        monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "observation_recipe", property(recipe))
    owner = model()
    before = owner.snapshot_state()
    outcome = smoke.run_frozen_rollout(
        directory / "late.json",
        learner=owner,
        visual_observation=True,
        camera_context=pick_cube_camera_context(),
    )
    assert outcome.report["complete"] is False and outcome.corrections is None
    assert outcome.report["visual_recipe_exact"] is None
    assert outcome.report["visual_recipe_error"]
    expected_steps = 2 if phase == "cleanup" else 1
    assert len(environment.commands) == outcome.report["work"]["completed_steps"] == expected_steps
    assert environment.close_calls == 1
    assert_same_state(before, owner.snapshot_state())


@pytest.mark.parametrize(
    "changes",
    [
        dict(train_seed=19),
        dict(visual_observation=False),
        dict(camera_context={"seed": 17, "context": CONTEXT}),
        dict(corrective_queries=True),
    ],
)
def test_should_refuse_camera_rollout_before_fork_version_or_scene(boundary, monkeypatch, changes):
    environment, calls, directory = boundary
    owner = model()
    monkeypatch.setattr(owner, "fork", lambda: pytest.fail("invalid camera forked learner"))
    monkeypatch.setattr(smoke, "version", lambda _: pytest.fail("invalid camera queried runtime"))
    arguments = dict(
        learner=owner, visual_observation=True, camera_context=pick_cube_camera_context()
    )
    arguments.update(changes)
    outcome = smoke.run_frozen_rollout(directory / "refused.json", **arguments)
    assert outcome.report["complete"] is False and outcome.report["creation_attempts"] == 0
    assert outcome.corrections is None and calls == [] and environment.commands == []


def test_should_preserve_default_visual_rollout_report_without_camera_metadata(boundary):
    _, calls, directory = boundary
    outcome = smoke.run_frozen_rollout(directory / "default.json", visual_observation=True)
    assert outcome.report["complete"] is True and "camera_context" not in outcome.report
    assert outcome.report["observation_recipe"] == asdict(pick_cube_rgb_recipe())
    assert calls == [dict(visual_observation=True)]


@pytest.mark.parametrize("entry", ["teacher", "rollout"])
def test_should_forward_cli_opt_in_and_write_known_camera_metadata(boundary, monkeypatch, entry):
    environment, calls, directory = boundary
    output = directory / f"{entry}.json"
    monkeypatch.setattr(
        sys, "argv", [entry, "--output", str(output), "--visual-observation", "--camera-drift"]
    )
    assert (teacher_cli.main() if entry == "teacher" else smoke.main()) == 0
    payload = json.loads(output.read_text(encoding="utf8"))
    assert payload["camera_context"] == asdict(pick_cube_camera_context())
    assert payload["context"] == CONTEXT and payload["seed"] == 17
    assert payload["observation_recipe"] == json.loads(
        json.dumps(asdict(camera_context_recipe(pick_cube_camera_context())))
    )
    assert calls == [dict(visual_observation=True, camera_context=pick_cube_camera_context())]
    assert environment.reset_calls == [17] and environment.close_calls == 1


def test_should_keep_default_teacher_cli_schema_without_optional_camera_field(
    boundary, monkeypatch
):
    _, calls, directory = boundary
    output = directory / "default-teacher.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), "--visual-observation"])
    assert teacher_cli.main() == 0
    payload = json.loads(output.read_text(encoding="utf8"))
    assert "camera_context" not in payload
    assert payload["context"] == "pick-cube" and payload["seed"] == 17
    assert payload["observation_recipe"] == json.loads(json.dumps(asdict(pick_cube_rgb_recipe())))
    assert calls == [dict(visual_observation=True)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--camera-drift"],
        ["--camera-drift", "--visual-observation", "--train-seed", "19"],
        [
            "--camera-drift",
            "--visual-observation",
            "--guard-seed",
            "31",
            "--labels-arrived-at",
            "2",
        ],
    ],
)
def test_should_reject_invalid_teacher_cli_camera_selection_before_output_or_scene(
    boundary, monkeypatch, arguments
):
    _, calls, directory = boundary
    output = directory / "invalid-teacher.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), *arguments])
    with pytest.raises(SystemExit) as error:
        teacher_cli.main()
    assert error.value.code == 2 and not output.exists() and calls == []


@pytest.mark.parametrize(
    "arguments",
    [["--camera-drift"], ["--camera-drift", "--visual-observation", "--train-seed", "19"]],
)
def test_should_capture_invalid_rollout_cli_camera_without_creating_scene(
    boundary, monkeypatch, arguments
):
    environment, calls, directory = boundary
    output = directory / "invalid-rollout.json"
    monkeypatch.setattr(sys, "argv", ["rollout", "--output", str(output), *arguments])
    assert smoke.main() == 1
    payload = json.loads(output.read_text(encoding="utf8"))
    assert payload["complete"] is False and payload["creation_attempts"] == 0
    assert calls == [] and environment.reset_calls == []
