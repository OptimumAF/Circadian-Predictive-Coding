"""Public nested camera boundary with fake native IO; no simulator skill claim."""

from dataclasses import asdict, replace
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_maniskill_task_teacher import FakeEnvironment as TeacherEnvironment
from scripts import maniskill_continuous_smoke as smoke
from scripts import maniskill_task_teacher_smoke as teacher_cli
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import StableActor
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_episode_clock import EpisodeClock
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra import maniskill_continuous as port
from src.infra import maniskill_task_teacher as teacher


class ForbiddenGeometry:
    def __array__(self, *_args, **_kwargs):
        pytest.fail("visual learner encoding accessed privileged geometry")


def packet(index=0):
    frame = np.empty((1, 128, 128, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 10 + index, 20 + index, 30 + index
    return dict(
        agent=dict(
            qpos=np.arange(9, dtype=np.float32)[None] + index,
            qvel=(np.arange(9, dtype=np.float32) * 0.1)[None],
            hidden=ForbiddenGeometry(),
        ),
        extra=dict(
            goal_pos=np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
            tcp_pose=ForbiddenGeometry(),
            obj_pose=ForbiddenGeometry(),
            is_grasped=ForbiddenGeometry(),
            relative_vector=ForbiddenGeometry(),
        ),
        sensor_data=dict(
            base_camera=dict(rgb=frame, depth=ForbiddenGeometry()),
            ignored_camera=ForbiddenGeometry(),
        ),
        sensor_param=ForbiddenGeometry(),
    )


class RGBEnvironment(TeacherEnvironment):
    def __init__(self):
        super().__init__()
        self.obs_mode = "rgb"
        self.backend = SimpleNamespace(render_backend="sapien_cpu")
        self.done_at = 2
        self.reset_packet = packet()
        self.last_packet = None
        self.next_packet = None

    def reset(self, *, seed):
        super().reset(seed=seed)
        return self.reset_packet, {}

    def step(self, command):
        _, reward, terminated, truncated, info = super().step(command)
        self.last_packet = (
            packet(len(self.commands)) if self.next_packet is None else self.next_packet
        )
        return self.last_packet, reward, terminated, truncated, info


@pytest.fixture
def visual_environment(monkeypatch):
    environment = RGBEnvironment()
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    return environment, calls


def open_visual():
    return port.ManiSkillPickCubeEpisode("visual-17", "pick-cube", visual_observation=True)


def test_should_encode_selected_public_nested_arrays_and_own_only_current_frame(visual_environment):
    environment, calls = visual_environment
    session = open_visual()
    assert calls == [dict(visual_observation=True)] and environment.reset_calls == [17]
    initial = session.observation
    assert len(initial) == session.spec.observation_dim == 789
    assert initial[:9] == tuple(float(value) for value in range(9))
    assert initial[9:18] == tuple(
        float(value) for value in environment.reset_packet["agent"]["qvel"][0]
    )
    assert initial[18:21] == tuple(
        float(value) for value in environment.reset_packet["extra"]["goal_pos"][0]
    )
    assert initial[21:24] == pytest.approx((10 / 255, 20 / 255, 30 / 255))
    assert session.observation_recipe == pick_cube_rgb_recipe()
    exported = session.rgb_frame
    exported.fill(255)
    environment.reset_packet["sensor_data"]["base_camera"]["rgb"].fill(0)
    environment.reset_packet["agent"]["qpos"].fill(999.0)
    assert session.observation == initial
    assert session.rgb_frame[0, 0].tolist() == [10, 20, 30]
    first = session.step((0.0,) * 4)
    second = session.step((0.0,) * 4)
    assert first.observation == initial and first.next_observation == second.observation
    assert len(second.next_observation) == 789 and second.truncated and not second.terminated
    assert session.rgb_frame[0, 0].tolist() == [12, 22, 32]
    work = asdict(session.visual_work)
    assert work == dict(
        observation_attempts=3,
        observation_returns=3,
        input_frame_bytes_upper_bound=3 * 49152,
        feature_values_upper_bound=3 * 789,
        returned_frame_bytes=3 * 49152,
        returned_feature_values=3 * 789,
        returned_feature_numeric_bytes=3 * 6312,
        current_owned_frame_bytes=49152,
        frame_export_copies=3,
        frame_export_bytes=3 * 49152,
    )
    assert session.work == port.ManiSkillWork(2, 2, 10, 0)
    assert all(len(row.observation) == 789 for row in session.episode().transitions)
    assert not any(isinstance(value, np.ndarray) for value in session.episode().transitions)
    session.close()
    assert environment.close_calls == 1


def test_should_ignore_changes_to_every_privileged_or_unselected_observation_field(
    visual_environment,
):
    environment, _ = visual_environment
    session = open_visual()
    initial = session.observation
    unchanged = packet()
    unchanged["extra"].update(
        obj_pose=np.full((1, 7), np.nan), tcp_pose=object(), is_grasped="changed"
    )
    unchanged["sensor_param"] = {"extrinsics": np.full((100, 100), np.inf)}
    environment.next_packet = unchanged
    record = session.step((0.0,) * 4)
    assert record.next_observation == initial
    session.close()


@pytest.mark.parametrize("bad", [0, 1, None, "rgb", np.bool_(True)])
def test_should_refuse_non_boolean_visual_option_before_factory(monkeypatch, bad):
    monkeypatch.setattr(
        port, "_make_environment", lambda **_: pytest.fail("invalid visual flag created a scene")
    )
    with pytest.raises(ValueError):
        port.ManiSkillPickCubeEpisode("bad", "pick-cube", visual_observation=bad)


def test_should_preserve_default_state_factory_call_and_refuse_final_visual_role(monkeypatch):
    environment = TeacherEnvironment()
    calls = []
    monkeypatch.setattr(
        port, "_make_environment", lambda **kwargs: calls.append(kwargs) or environment
    )
    session = port.ManiSkillPickCubeEpisode("state", "pick-cube")
    assert calls == [{}] and session.spec.observation_dim == 3
    assert session.observation_recipe is None and session.visual_work is None
    with pytest.raises(ValueError):
        _ = session.rgb_frame
    session.close()
    with pytest.raises(ValueError):
        port.ManiSkillPickCubeEpisode(
            "held", "pick-cube", role="final_test", visual_observation=True
        )
    assert calls == [{}]


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "top_type",
        "agent_type",
        "camera_type",
        "rgb_shape",
        "rgb_dtype",
        "qpos_shape",
        "qvel_dtype",
        "goal_shape",
        "nonfinite",
    ],
)
def test_should_charge_failed_late_encoding_without_publishing_new_features(
    visual_environment, bad
):
    environment, _ = visual_environment
    session = open_visual()
    before = session.observation
    value = packet(1)
    if bad == "missing":
        del value["extra"]["goal_pos"]
    elif bad == "top_type":
        value = []
    elif bad == "agent_type":
        value["agent"] = []
    elif bad == "camera_type":
        value["sensor_data"]["base_camera"] = []
    elif bad == "rgb_shape":
        value["sensor_data"]["base_camera"]["rgb"] = np.zeros((128, 128, 3), dtype=np.uint8)
    elif bad == "rgb_dtype":
        value["sensor_data"]["base_camera"]["rgb"] = np.zeros((1, 128, 128, 3), dtype=np.float32)
    elif bad == "qpos_shape":
        value["agent"]["qpos"] = np.zeros((1, 8), dtype=np.float32)
    elif bad == "qvel_dtype":
        value["agent"]["qvel"] = np.zeros((1, 9), dtype=np.int64)
    elif bad == "goal_shape":
        value["extra"]["goal_pos"] = np.zeros((1, 4), dtype=np.float32)
    else:
        value["extra"]["goal_pos"][0, 0] = np.nan
    environment.next_packet = value
    with pytest.raises(ValueError):
        session.step((0.0,) * 4)
    assert session.observation == before and session.work == port.ManiSkillWork(1, 1, 5, 0)
    assert (
        session.visual_work.observation_attempts == 2
        and session.visual_work.observation_returns == 1
    )
    assert session.visual_work.input_frame_bytes_upper_bound == 2 * 49152
    assert session.visual_work.returned_feature_values == 789
    assert session.visual_work.current_owned_frame_bytes == 49152
    assert session.rgb_frame[0, 0].tolist() == [10, 20, 30]
    with pytest.raises(ValueError):
        session.step((0.0,) * 4)
    with pytest.raises(ValueError):
        session.episode()
    session.close()
    assert len(environment.commands) == environment.close_calls == 1


def test_should_separate_unreturned_native_step_from_observation_encoding_work(visual_environment):
    environment, _ = visual_environment
    session = open_visual()
    environment.step_failure_at = 1
    with pytest.raises(RuntimeError, match="physics failed"):
        session.step((0.0,) * 4)
    assert session.work == port.ManiSkillWork(1, 0, 0, 5)
    assert session.visual_work.observation_attempts == session.visual_work.observation_returns == 1
    session.close()


def test_should_accept_detached_cpu_tensors_for_nested_rgb_and_public_vectors(visual_environment):
    import torch

    environment, _ = visual_environment
    value = packet()
    for name in ("qpos", "qvel"):
        value["agent"][name] = torch.from_numpy(value["agent"][name])
    value["extra"]["goal_pos"] = torch.from_numpy(value["extra"]["goal_pos"])
    value["sensor_data"]["base_camera"]["rgb"] = torch.from_numpy(
        value["sensor_data"]["base_camera"]["rgb"]
    )
    environment.reset_packet = value
    session = open_visual()
    assert len(session.observation) == 789 and session.rgb_frame.dtype == np.uint8
    session.close()


def test_should_collect_visual_teacher_without_exposing_geometry_in_learner_features(
    visual_environment,
):
    environment, calls = visual_environment
    result = teacher.collect_task_teacher(visual_observation=True)
    assert result.qualified and result.error is None and environment.close_calls == 1
    data = result.training_episode()
    assert data.role == "train" and data.spec.observation_dim == 789
    assert data.transitions[0].observation[:9] == tuple(float(value) for value in range(9))
    assert data.transitions[0].action == pytest.approx((0.2, -0.3, 0.9, 1.0))
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    assert calls == [dict(visual_observation=True)]
    assert result.observation_recipe == pick_cube_rgb_recipe()
    assert result.visual_work.observation_attempts == result.visual_work.observation_returns == 3


def test_should_preserve_failed_visual_teacher_encoding_and_partial_costs(visual_environment):
    environment, _ = visual_environment
    environment.next_packet = packet(1)
    environment.next_packet["agent"]["qpos"] = np.zeros((1, 8), dtype=np.float32)
    result = teacher.collect_task_teacher(visual_observation=True)
    assert not result.qualified and result.candidate_episode is None and result.error
    assert result.teacher_actions == 1 and result.work == port.ManiSkillWork(1, 1, 5, 0)
    assert (
        result.visual_work.observation_attempts == 2 and result.visual_work.observation_returns == 1
    )
    assert environment.close_calls == 1


@pytest.mark.parametrize("bad", [1, None, "rgb"])
def test_should_refuse_invalid_teacher_visual_flag_before_port(monkeypatch, bad):
    monkeypatch.setattr(
        teacher,
        "ManiSkillPickCubeEpisode",
        lambda *_args, **_kwargs: pytest.fail("invalid teacher visual flag created a port"),
    )
    with pytest.raises(ValueError):
        teacher.collect_task_teacher(visual_observation=bad)


def public_feature_oracle(value):
    # The fixture has uniform RGB planes, so block means are known analytically.
    prefix = [
        float(item)
        for source in (value["agent"]["qpos"], value["agent"]["qvel"], value["extra"]["goal_pos"])
        for item in source[0]
    ]
    pixel = [
        float(channel) / 255 for channel in value["sensor_data"]["base_camera"]["rgb"][0, 0, 0]
    ]
    return prefix + pixel * 256


def visual_learner(*, clock=False):
    dimension = 790 if clock else 789
    declared = ContinuousActionSpec(dimension, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    frozen = ObservationStandardization((0.0,) * dimension, (1.0,) * dimension, 2)
    model = NumpyNearestContinuousBCLearner(NearestContinuousBCPolicy(declared, frozen))
    rows = [public_feature_oracle(packet(0)), public_feature_oracle(packet(1))]
    if clock:
        rows = [rows[0] + [0.0], rows[1] + [0.02]]
    model.train_batch(
        np.asarray(rows, dtype=np.float64),
        np.array([[0.125, -1.0, 1.0, 0.0], [-0.5, 0.5, 0.0, -1.0]]),
    )
    return model


@pytest.mark.parametrize("family", ["zero", "nearest", "actor_clock"])
def test_should_integrate_visual_frames_with_existing_smoke_and_actual_actor_clock(
    visual_environment, monkeypatch, tmp_path, family
):
    environment, calls = visual_environment
    monkeypatch.setattr(smoke, "version", lambda _: "test-version")
    output = tmp_path / "visual-smoke.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output), "--visual-observation"])
    options = {}
    model = None
    if family == "nearest":
        model = visual_learner()
        options["learner"] = model
    elif family == "actor_clock":
        model = visual_learner(clock=True)
        options.update(
            actor=StableActor(model, version="rgb-serving"),
            observation_clock=EpisodeClock(50, True),
        )
    before = None if model is None else model.snapshot_state()
    assert smoke.main(**options) == 0
    result = json.loads(output.read_text(encoding="utf8"))
    assert calls == [dict(visual_observation=True)] and environment.close_calls == 1
    assert result["complete"] is True and result["visual_recipe_exact"] is True
    assert result["observation_recipe"] == json.loads(json.dumps(asdict(pick_cube_rgb_recipe())))
    assert result["episode"]["spec"]["observation_dim"] == 789
    assert (
        result["visual_work"]["observation_attempts"]
        == result["visual_work"]["observation_returns"]
        == 3
    )
    assert result["visual_work"]["returned_feature_values"] == 3 * 789
    assert result["visual_work"]["current_owned_frame_bytes"] == 49152
    assert result["visual_work"]["frame_export_copies"] == 0
    assert (
        result["optimizer_updates"] == 0 and result["work"]["completed_control_physics_steps"] == 10
    )
    if model is not None:
        assert result["frozen_state_exact"] is result["original_owner_exact"] is True
        after = model.snapshot_state()
        assert before.policy == after.policy and before.completed_updates == after.completed_updates
        np.testing.assert_array_equal(before.prototype_features, after.prototype_features)
        np.testing.assert_array_equal(before.prototype_actions, after.prototype_actions)
        for index, command in enumerate(environment.commands):
            np.testing.assert_array_equal(
                command[0], before.prototype_actions[index].astype(np.float32)
            )
    if family == "actor_clock":
        assert result["actor_version"] == "rgb-serving" and result["actor_prediction_returns"] == 2
        assert result["clock_values"] == [0.0, 0.02] and result["clock_encoding_returns"] == 2


@pytest.mark.parametrize("phase", ["after_step", "cleanup"])
def test_should_refuse_late_visual_recipe_drift_and_clear_stale_exact_flag(
    visual_environment, monkeypatch, tmp_path, phase
):
    environment, _ = visual_environment
    original_recipe = port.ManiSkillPickCubeEpisode.observation_recipe
    drift = []

    def observed(session):
        value = original_recipe.fget(session)
        return replace(value, pixel_divisor=255) if drift else value

    if phase == "after_step":
        original = environment.step

        def step(command):
            result = original(command)
            drift.append(True)
            return result

        monkeypatch.setattr(environment, "step", step)
    else:
        original = environment.close

        def close():
            original()
            drift.append(True)

        monkeypatch.setattr(environment, "close", close)
    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "observation_recipe", property(observed))
    monkeypatch.setattr(smoke, "version", lambda _: "test-version")
    output = tmp_path / "drift.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    assert smoke.main(visual_observation=True) == 1
    result = json.loads(output.read_text(encoding="utf8"))
    assert result["complete"] is False and result["visual_recipe_exact"] is None
    assert result["visual_recipe_error"]
    assert len(environment.commands) == (1 if phase == "after_step" else 2)
    assert environment.close_calls == 1


def test_should_write_visual_teacher_cli_metadata_and_forward_only_selected_flag(
    visual_environment, monkeypatch, tmp_path
):
    result = teacher.collect_task_teacher(visual_observation=True)
    calls = []
    monkeypatch.setattr(
        teacher_cli, "collect_task_teacher", lambda **kwargs: calls.append(kwargs) or result
    )
    monkeypatch.setattr(teacher_cli, "version", lambda _: "test-version")
    monkeypatch.setattr(teacher_cli.faulthandler, "enable", lambda: None)
    output = tmp_path / "teacher.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), "--visual-observation"])
    assert teacher_cli.main() == 0
    payload = json.loads(output.read_text(encoding="utf8"))
    assert calls == [dict(visual_observation=True)]
    assert (
        payload["qualified"] is True
        and payload["candidate_episode"]["spec"]["observation_dim"] == 789
    )
    assert payload["observation_recipe"] == json.loads(json.dumps(asdict(pick_cube_rgb_recipe())))
    assert payload["visual_work"]["observation_returns"] == 3
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        teacher_cli.main()
    assert output.read_bytes() == before and len(calls) == 1


def test_should_refuse_non_boolean_smoke_visual_flag_and_existing_output_before_scene(
    visual_environment, monkeypatch, tmp_path
):
    environment, calls = visual_environment
    monkeypatch.setattr(
        smoke, "version", lambda _: pytest.fail("invalid visual flag reached runtime query")
    )
    output = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    assert smoke.main(visual_observation=1) == 1
    result = json.loads(output.read_text(encoding="utf8"))
    assert result["creation_attempts"] == 0 and result["visual_work"] is None
    assert calls == [] and environment.reset_calls == []
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.main(visual_observation=True)
    assert output.read_bytes() == before and calls == []


@pytest.mark.parametrize("visual", [False, True])
def test_should_select_frozen_camera_and_cpu_renderer_only_for_visual_factory(monkeypatch, visual):
    # Fake SDK registration checks constructor options without native imports/work.
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

    class RegisteredTask:
        pass

    modules["mani_skill.envs.tasks.tabletop.pick_cube"].PickCubeEnv = RegisteredTask
    created, poses, wrappers = [], [], []
    environment = RGBEnvironment()

    def make(**options):
        created.append(options)
        return environment

    def wrap(value, horizon):
        wrappers.append((value, horizon))
        return value

    def look_at(eye, target):
        poses.append((eye, target))
        return "frozen-fake-pose"

    modules["mani_skill.utils.registration"].REGISTERED_ENVS = {
        port.TASK_ID: SimpleNamespace(
            cls=RegisteredTask, asset_download_ids=(), max_episode_steps=50, make=make
        ),
    }
    modules["mani_skill.utils.registration"].TimeLimitWrapper = wrap
    modules["mani_skill.utils.sapien_utils"].look_at = look_at
    monkeypatch.setattr(port, "_require_cpu_kinematics", lambda: None)
    monkeypatch.setattr(
        port, "version", lambda name: {"mani_skill": "3.0.1", "sapien": "3.0.3"}[name]
    )
    assert port._make_environment(visual_observation=visual) is environment
    assert wrappers == [(environment, 50)] and len(created) == 1
    options = created[0]
    assert options["sim_backend"] == "physx_cpu" and options["robot_uids"] == "panda"
    assert options["control_mode"] == port.CONTROL_MODE and options["reward_mode"] == "sparse"
    assert options["num_envs"] == 1 and options["render_mode"] is None
    if visual:
        recipe = pick_cube_rgb_recipe()
        assert options["obs_mode"] == "rgb" and options["render_backend"] == "sapien_cpu"
        assert options["sensor_configs"] == {
            "base_camera": dict(
                width=128,
                height=128,
                fov=recipe.camera_fov,
                near=0.01,
                far=100.0,
                pose="frozen-fake-pose",
                shader_pack="minimal",
            )
        }
        assert poses == [((0.3, 0.0, 0.6), (-0.1, 0.0, 0.1))]
    else:
        assert options["obs_mode"] == "state" and options["render_backend"] == "none"
        assert "sensor_configs" not in options and poses == []


def test_should_refuse_visual_teacher_cleanup_recipe_corruption_without_erasing_candidate(
    visual_environment, monkeypatch
):
    environment, _ = visual_environment
    original = port.ManiSkillPickCubeEpisode.observation_recipe

    def changed_after_close(session):
        recipe = original.fget(session)
        return replace(recipe, block_size=8.0) if environment.close_calls else recipe

    monkeypatch.setattr(
        port.ManiSkillPickCubeEpisode, "observation_recipe", property(changed_after_close)
    )
    result = teacher.collect_task_teacher(visual_observation=True)
    assert result.candidate_episode is not None and result.final_success
    assert not result.qualified and result.error and "visual metadata" in result.error
    assert result.visual_work.observation_returns == 3 and environment.close_calls == 1
