"""Single CPU ManiSkill PickCube episode to owned continuous records (R4.4).

Inputs are native normalized controller actions; outputs preserve actual float32
commands, state or explicit frozen RGB features and independent end flags.
Optional simulator imports are lazy. No teacher, learner, role conversion, final
evaluation, replay, promotion or new orchestration framework is provided here.
"""

from dataclasses import dataclass, fields
from importlib import import_module
from importlib.metadata import version
from typing import Any

import numpy as np

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_rgb_features import (
    PickCubeRGBRecipe,
    encode_pick_cube_rgb,
    pick_cube_rgb_recipe,
)
from src.core.pick_cube_guard_context import pick_cube_guard_context
from src.core.joint_position_limits import (
    JointPositionLimits,
    JointPositionMonitor,
    JointPositionSummary,
)
from src.core.pick_cube_mass_context import (
    PickCubeMassContext,
    require_mass_binding,
    validate_pick_cube_mass_context,
)
from src.core.pick_cube_camera_context import (
    PickCubeCameraContext,
    camera_context_recipe,
    require_camera_binding,
    validate_pick_cube_camera_context,
)
from src.core.pick_cube_lighting_context import (
    PickCubeLightingContext,
    require_lighting_binding,
    validate_pick_cube_lighting_context,
)
from src.infra.maniskill_lighting import (
    ManiSkillAmbientLight,
    ManiSkillLightingReadback,
    ManiSkillLightingWork,
    validate_lighting_readback,
    validate_lighting_work,
)

TASK_ID = "PickCube-v1"
CONTROL_MODE = "pd_ee_delta_pos"
MANISKILL_VERSION = "3.0.1"
HORIZON = 50
ACTION_UNITS = (
    "normalized_root_dx",
    "normalized_root_dy",
    "normalized_root_dz",
    "normalized_gripper",
)
PANDA_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8)) + (
    "panda_finger_joint1",
    "panda_finger_joint2",
)


def _array(value: Any) -> np.ndarray:
    if type(value) is np.ndarray:
        return value
    # Why: accept the real CPU tensor boundary, not arbitrary duck-typed objects
    # that could hide device transfers, squeezing or conversions of dictionaries.
    try:
        # The optional tensor library stays behind this runtime boundary; the
        # public record API does not depend on Torch's large static type graph.
        torch = import_module("torch")
    except ImportError as error:
        raise ValueError("expected an exact NumPy array or detached CPU torch.Tensor") from error

    if type(value) is torch.Tensor and value.device.type == "cpu" and not value.requires_grad:
        return value.detach().numpy()
    raise ValueError("expected an exact NumPy array or detached CPU torch.Tensor")


def encode_state_observation(value: Any) -> tuple[float, ...]:
    """Keep the official flattened state order; never flatten batches/images/dicts."""
    array = _array(value)
    if (
        array.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or array.ndim != 2
        or array.shape[0] != 1
        or not 1 <= array.shape[1] <= 1024
        or not np.isfinite(array).all()
    ):
        raise ValueError("state must be one finite float32/64 row with1..1024 coordinates")
    return tuple(float(item) for item in array[0])


def _scalar(value: Any, *, boolean: bool = False) -> float | bool:
    array = _array(value)
    valid_dtype = (
        array.dtype == np.dtype("bool")
        if boolean
        else array.dtype in (np.dtype("float32"), np.dtype("float64"), np.dtype("bool"))
    )
    if array.shape != (1,) or not valid_dtype or not np.isfinite(array).all():
        raise ValueError("reward/flags must retain the single native environment axis and dtype")
    # The official sparse PickCube reward is the boolean success tensor. Preserve
    # that objective as exact0/1; flags still require an actual boolean dtype.
    return bool(array[0]) if boolean else float(array[0])


def _make_environment(
    *,
    control_mode: str = CONTROL_MODE,
    visual_observation: bool = False,
    camera_context: PickCubeCameraContext | None = None,
) -> Any:
    """Use the pinned official task and time limit, with no auto-reset wrapper."""
    if control_mode not in (CONTROL_MODE, "pd_joint_pos"):
        raise ValueError("unsupported PickCube controller")
    if type(visual_observation) is not bool:
        raise ValueError("visual_observation must be an exact bool")
    if visual_observation and control_mode != CONTROL_MODE:
        raise ValueError("visual observations require the normalized four-axis controller")
    if camera_context is not None:
        require_camera_binding(
            camera_context,
            context=camera_context.context if type(camera_context) is PickCubeCameraContext else "",
            seed=17,
            role="train",
            visual_observation=visual_observation,
        )
    for package, expected in (("mani_skill", MANISKILL_VERSION), ("sapien", "3.0.3")):
        if version(package) != expected:
            raise ValueError(f"this port requires {package}=={expected}")
    _require_cpu_kinematics()
    import mani_skill.envs  # noqa: F401 -- register the real optional tasks
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    from mani_skill.utils.registration import REGISTERED_ENVS, TimeLimitWrapper

    registered = REGISTERED_ENVS[TASK_ID]
    if (
        registered.cls is not PickCubeEnv
        or registered.asset_download_ids
        or registered.max_episode_steps != HORIZON
    ):
        raise ValueError("unexpected task registration or external asset requirement")
    options: dict[str, Any] = dict(obs_mode="state", render_backend="none")
    if visual_observation:
        from mani_skill.utils import sapien_utils

        recipe = (
            pick_cube_rgb_recipe()
            if camera_context is None
            else camera_context_recipe(camera_context)
        )
        options = dict(
            obs_mode="rgb",
            render_backend=recipe.render_backend,
            sensor_configs={
                recipe.camera: dict(
                    width=128,
                    height=128,
                    fov=recipe.camera_fov,
                    near=recipe.camera_near,
                    far=recipe.camera_far,
                    pose=sapien_utils.look_at(recipe.camera_eye, recipe.camera_target),
                    shader_pack=recipe.shader_pack,
                )
            },
        )
    env = registered.make(
        robot_uids="panda",
        control_mode=control_mode,
        num_envs=1,
        sim_backend="physx_cpu",
        render_mode=None,
        reward_mode="sparse",
        **options,
    )
    try:
        return TimeLimitWrapper(env, HORIZON)
    except BaseException:
        env.close()
        raise


def _require_cpu_kinematics() -> None:
    # On Windows, SAPIEN can import successfully while replacing this optional
    # CPU IK class with None. Check before scene creation, not after a costly
    # constructor reset fails with the uninformative NoneType-callable error.
    module = import_module("sapien.wrapper.pinocchio_model")
    if not callable(module.PinocchioModel):
        raise RuntimeError(
            "CPU pd_ee_delta_pos requires a usable SAPIEN PinocchioModel; "
            "supply a compatible Pinocchio installation in the simulator environment "
            "before creating a robot scene"
        )


def _validate_environment(env: Any, *, visual_observation: bool = False) -> int:
    base = env.unwrapped
    space = base.single_action_space
    if (
        base.num_envs != 1
        or base.obs_mode != ("rgb" if visual_observation else "state")
        or base.control_mode != CONTROL_MODE
        or base.agent.uid != "panda"
        or base.device.type != "cpu"
        or space.shape != (4,)
        or not np.array_equal(space.low, np.full(4, -1.0))
        or not np.array_equal(space.high, np.full(4, 1.0))
    ):
        raise ValueError(
            "expected single CPU Panda state/pd_ee_delta_pos normalized4-axis contract"
        )
    if visual_observation and base.backend.render_backend != "sapien_cpu":
        raise ValueError("visual observation requires the frozen CPU renderer")
    sim, control = base.sim_freq, base.control_freq
    if type(sim) is not int or type(control) is not int or control <= 0 or sim % control:
        raise ValueError("expected an integral positive controller physics ratio")
    if not 1 <= sim // control <= 100:
        raise ValueError("controller physics ratio must be in1..100")
    return sim // control


@dataclass(frozen=True)
class ManiSkillWork:
    """Controller work only; constructor/reset physics are explicitly unmeasured."""

    step_attempts: int
    completed_steps: int
    completed_control_physics_steps: int
    possible_failed_physics_steps: int


@dataclass(frozen=True)
class ManiSkillJointPositionWork:
    """One limit snapshot and accepted encodings; not physics-substep coverage."""

    limit_read_attempts: int
    limit_read_returns: int
    samples: int
    sampled_coordinates: int
    name_read_attempts: int
    name_read_returns: int


def validate_joint_position_work(value: ManiSkillJointPositionWork) -> ManiSkillJointPositionWork:
    """Own the one-snapshot paid lifecycle, including failed constructor reads."""
    if type(value) is not ManiSkillJointPositionWork or any(
        type(getattr(value, field.name)) is not int or getattr(value, field.name) < 0
        for field in fields(ManiSkillJointPositionWork)
    ):
        raise ValueError("joint-position work requires exact nonnegative integer counters")
    if (
        not 0 <= value.limit_read_returns <= value.limit_read_attempts <= 1
        or not 0 <= value.name_read_returns <= value.name_read_attempts <= 1
        or not 0 <= value.samples <= HORIZON + 1
        or value.sampled_coordinates != 9 * value.samples
        or (value.limit_read_attempts > 0 and value.name_read_returns != 1)
        or (value.samples > 0 and value.limit_read_returns != 1)
    ):
        raise ValueError("joint-position work differs from its bounded snapshot/sample lifecycle")
    return ManiSkillJointPositionWork(
        *(getattr(value, field.name) for field in fields(ManiSkillJointPositionWork))
    )


@dataclass(frozen=True)
class ManiSkillVisualWork:
    observation_attempts: int
    observation_returns: int
    input_frame_bytes_upper_bound: int
    feature_values_upper_bound: int
    returned_frame_bytes: int
    returned_feature_values: int
    returned_feature_numeric_bytes: int
    current_owned_frame_bytes: int
    frame_export_copies: int
    frame_export_bytes: int


def validate_lighting_visual_work(
    value: ManiSkillVisualWork, *, steps: int | None = None
) -> ManiSkillVisualWork:
    """Own lighting encoding costs; optional complete lifecycle is stricter."""
    if type(value) is not ManiSkillVisualWork or any(
        type(getattr(value, field.name)) is not int or not 0 <= getattr(value, field.name) < 2**63
        for field in fields(ManiSkillVisualWork)
    ):
        raise ValueError("lighting visual work requires exact bounded integer counters")
    attempts, returns = value.observation_attempts, value.observation_returns
    if (
        not 0 <= returns <= attempts <= HORIZON + 1
        or value.input_frame_bytes_upper_bound != attempts * 49152
        or value.feature_values_upper_bound != attempts * 789
        or value.returned_frame_bytes != returns * 49152
        or value.returned_feature_values != returns * 789
        or value.returned_feature_numeric_bytes != returns * 6312
        or value.current_owned_frame_bytes != (49152 if returns else 0)
        or value.frame_export_bytes != value.frame_export_copies * 49152
    ):
        raise ValueError("lighting visual work differs from its bounded encoding lifecycle")
    if steps is not None and (
        type(steps) is not int
        or not 0 <= steps <= HORIZON
        or attempts != steps + 1
        or returns != steps + 1
        or value.frame_export_copies != 0
    ):
        raise ValueError("lighting visual work differs from the complete control lifecycle")
    return ManiSkillVisualWork(*(getattr(value, field.name) for field in fields(value)))


@dataclass(frozen=True)
class ManiSkillMassReadback:
    """Owned after-reset mass measurements; inertia/COM effects are unmeasured."""

    context: PickCubeMassContext
    source_mass_kg: float
    applied_mass_kg: float


@dataclass(frozen=True)
class ManiSkillMassWork:
    read_attempts: int
    read_returns: int
    write_attempts: int
    write_returns: int


def validate_mass_readback(
    value: ManiSkillMassReadback, context: PickCubeMassContext
) -> ManiSkillMassReadback:
    """Own exact original native measurements without reading simulator state."""
    expected = validate_pick_cube_mass_context(context)
    if type(value) is not ManiSkillMassReadback:
        raise ValueError("expected an exact owned native mass readback")
    observed = validate_pick_cube_mass_context(value.context)
    source, target = value.source_mass_kg, value.applied_mass_kg
    if (
        observed != expected
        or type(source) is not float
        or type(target) is not float
        or not 0.0 < source <= 0.5
        or not 0.0 < target <= 1.0
        or float(np.float32(source)) != source
        or float(np.float32(target)) != target
        or target != source * expected.mass_multiplier
    ):
        raise ValueError("mass readback differs from its declared native float32 doubling")
    return ManiSkillMassReadback(expected, source, target)


def validate_mass_work(value: ManiSkillMassWork) -> ManiSkillMassWork:
    """Own bounded scalar counters; consumers bind them to their own lifecycle."""
    if type(value) is not ManiSkillMassWork:
        raise ValueError("expected exact native mass work")
    counts = (value.read_attempts, value.read_returns, value.write_attempts, value.write_returns)
    if (
        any(type(count) is not int or not 0 <= count <= 2**63 - 1 for count in counts)
        or value.read_returns > value.read_attempts
        or not 0 <= value.write_returns <= value.write_attempts <= 1
    ):
        raise ValueError("invalid native mass read/write work counters")
    return ManiSkillMassWork(*counts)


class ManiSkillPickCubeEpisode:
    """One serialized episode: invalid native output permanently stops admission.

    Construction performs one explicit reset(seed), in addition to the pinned
    simulator constructor's seed2022 reconfiguration reset. No reset or retry is
    exposed. Storage is bounded by the official50-step horizon.
    """

    def __init__(
        self,
        episode_id: str,
        context: str,
        *,
        role: str = "train",
        seed: int = 17,
        visual_observation: bool = False,
        camera_context: PickCubeCameraContext | None = None,
        mass_context: PickCubeMassContext | None = None,
        lighting_context: PickCubeLightingContext | None = None,
        monitor_joint_positions: bool = False,
    ):
        # Reuse the core record validators before admitting expensive creation.
        spec = ContinuousActionSpec(1, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
        dummy = ContinuousTransition(0, (0.0,), (0.0,) * 4, 0.0, (0.0,), False, True)
        ContinuousEpisode(episode_id, context, role, spec, (dummy,))
        if role == "final_test":
            raise ValueError("final evaluation requires a separate release protocol")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a native integer in0..2**32-1")
        # Why: reserved evaluation resets must not enter TRAIN by changing only
        # the role label. Generic caller-declared probe episodes remain supported.
        if seed == 29:
            raise ValueError("final reset29 requires a separate release protocol")
        if seed == 24 and role != "probe_test":
            raise ValueError("reserved probe test reset24 is score-only")
        if seed == 23 and role not in ("probe_train", "probe_test"):
            raise ValueError("probe reset23 is reserved for probe_train/probe_test")
        if role == "inner_guard":
            if context != pick_cube_guard_context(seed):
                raise ValueError("guard context must match its declared reset seed")
        elif seed in (31, 37):
            raise ValueError("guard seeds31/37 are reserved for inner_guard collection")
        if type(visual_observation) is not bool:
            raise ValueError("visual_observation must be an exact bool")
        if type(monitor_joint_positions) is not bool:
            raise ValueError("monitor_joint_positions must be an exact bool")
        self._monitor_joint_positions = monitor_joint_positions
        self._joint_position_admitted = monitor_joint_positions
        self._joint_positions: JointPositionMonitor | None = None
        self._joint_robot: Any = None
        self._joint_owner: tuple[Any, JointPositionMonitor] | None = None
        self._joint_limit_reads = self._joint_limit_returns = 0
        self._joint_name_reads = self._joint_name_returns = 0
        self._camera_context = require_camera_binding(
            camera_context,
            context=context,
            seed=seed,
            role=role,
            visual_observation=visual_observation,
        )
        self._camera_context_active = self._camera_context is not None
        self._mass_context = require_mass_binding(
            mass_context, context=context, seed=seed, role=role
        )
        self._lighting_context = require_lighting_binding(
            lighting_context,
            context=context,
            seed=seed,
            role=role,
            visual_observation=visual_observation,
        )
        if self._lighting_context is not None and (
            self._camera_context is not None or self._mass_context is not None
        ):
            raise ValueError("lighting, camera and mass require separate declared contexts")
        self._lighting_context_active = self._lighting_context is not None
        self._lighting: ManiSkillAmbientLight | None = None
        self._lighting_owner: tuple[Any, Any, Any, ManiSkillAmbientLight] | None = None
        if self._mass_context is not None and self._camera_context is not None:
            raise ValueError("mass and camera interventions require separate declared contexts")
        self._mass_context_active = self._mass_context is not None
        self._mass_actor: Any = None
        self._mass_source_mass = self._mass_applied_mass = 0.0
        self._mass_reads = self._mass_read_returns = 0
        self._mass_writes = self._mass_write_returns = 0
        self._episode_id, self._context, self._role = episode_id, context, role
        # Retain the admitted reset seed so corrective supervision cannot bind a
        # held-out reset by merely supplying a TRAIN-looking episode identity.
        self._seed = seed
        self._closed = False
        self._failed = False
        self._records: list[ContinuousTransition] = []
        self._attempts = self._completed = 0
        self._visual_observation = visual_observation
        self._visual_attempts = self._visual_returns = self._frame_exports = 0
        self._rgb_frame: np.ndarray | None = None
        self._env = (
            _make_environment(visual_observation=True, camera_context=self._camera_context)
            if self._camera_context is not None
            else _make_environment(visual_observation=True)
            if visual_observation
            else _make_environment()
        )
        try:
            self._physics_ratio = _validate_environment(
                self._env, visual_observation=visual_observation
            )
            observation, _ = self._env.reset(seed=seed)
            if self._mass_context is not None:
                self._apply_mass_context()
            if self._lighting_context is not None:
                observation = self._apply_lighting_context()
            if monitor_joint_positions:
                self._start_joint_positions()
            self._observation = self._encode_observation(observation)
            self._spec = ContinuousActionSpec(
                len(self._observation), ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4
            )
        except BaseException as error:
            if self._joint_position_admitted:
                setattr(error, "joint_position_work", self.joint_position_work)
            if self._lighting is not None:
                self._lighting.preserve_failure(error)
                try:
                    self.close()
                except BaseException as cleanup_error:
                    setattr(error, "lighting_cleanup_error", cleanup_error)
            else:
                try:
                    self.close()
                except BaseException as cleanup_error:
                    if not self._joint_position_admitted:
                        raise
                    setattr(error, "joint_position_cleanup_error", cleanup_error)
            raise

    def _start_joint_positions(self) -> None:
        robot = self._env.unwrapped.agent.robot
        self._joint_robot = robot
        self._joint_name_reads += 1
        joints = robot.get_active_joints()
        self._joint_name_returns += 1
        if type(joints) is not list or len(joints) != 9:
            raise ValueError("joint telemetry requires nine active Panda joints")
        names = tuple(joint.name for joint in joints)
        if self._env.unwrapped.agent.robot is not robot or names != PANDA_JOINT_NAMES:
            raise ValueError("joint telemetry requires the fixed Panda active-joint order")
        self._joint_limit_reads += 1
        limits = robot.get_qlimits()
        self._joint_limit_returns += 1
        if self._env.unwrapped.agent.robot is not robot:
            raise ValueError("joint telemetry robot was replaced during limit read")
        array = _array(limits)
        if self._env.unwrapped.agent.robot is not robot:
            raise ValueError("joint telemetry robot was replaced during limit conversion")
        if (
            array.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or array.shape != (1, 9, 2)
            or not np.isfinite(array).all()
        ):
            raise ValueError("Panda limits must be one finite float32/64 (1,9,2) array")
        self._joint_positions = JointPositionMonitor(
            JointPositionLimits(
                names,
                ("rad",) * 7 + ("m",) * 2,
                tuple(float(item) for item in array[0, :, 0]),
                tuple(float(item) for item in array[0, :, 1]),
            )
        )
        self._joint_owner = (robot, self._joint_positions)

    def _check_joint_position_binding(self) -> None:
        try:
            if self._monitor_joint_positions is not self._joint_position_admitted:
                raise ValueError("joint telemetry admission was changed")
            if not self._joint_position_admitted:
                if (
                    self._joint_positions is not None
                    or self._joint_robot is not None
                    or self._joint_owner is not None
                ):
                    raise ValueError("joint telemetry was added after admission")
                return
            owner = self._joint_owner
            if (
                owner is None
                or self._joint_robot is not owner[0]
                or self._joint_positions is not owner[1]
            ):
                raise ValueError("joint telemetry robot binding was removed or replaced")
            robot = self._env.unwrapped.agent.robot
            if (
                robot is not owner[0]
                or self._joint_owner is not owner
                or self._joint_robot is not owner[0]
                or self._joint_positions is not owner[1]
                or self._monitor_joint_positions is not True
                or self._joint_position_admitted is not True
            ):
                raise ValueError("joint telemetry robot binding changed during identity read")
        except BaseException:
            self._failed = True
            raise

    def _sample_joint_positions(self, encoded: tuple[float, ...]) -> None:
        self._check_joint_position_binding()
        if self._joint_positions is not None:
            # The pinned SDK flattens agent.qpos first; RGB uses that same owned
            # prefix. No extra get_qpos/read/render or hidden geometry access.
            if len(encoded) != (789 if self._visual_observation else 42):
                raise ValueError("joint telemetry requires the pinned observation dimension")
            self._joint_positions.observe(encoded[:9])

    @property
    def joint_position_summary(self) -> JointPositionSummary | None:
        return None if self._joint_positions is None else self._joint_positions.summary()

    @property
    def joint_position_work(self) -> ManiSkillJointPositionWork | None:
        if not self._joint_position_admitted:
            return None
        # Keep paid work readable even if a reporting property is instrumented
        # or fails; the owned core monitor needs no virtual/native getter.
        summary = None if self._joint_positions is None else self._joint_positions.summary()
        samples = 0 if summary is None else summary.samples
        return ManiSkillJointPositionWork(
            self._joint_limit_reads,
            self._joint_limit_returns,
            samples,
            samples * 9,
            self._joint_name_reads,
            self._joint_name_returns,
        )

    @property
    def observation(self) -> tuple[float, ...]:
        return self._observation

    @property
    def spec(self) -> ContinuousActionSpec:
        return ContinuousActionSpec(
            self._spec.observation_dim,
            self._spec.action_units,
            self._spec.lower_bounds,
            self._spec.upper_bounds,
        )

    @property
    def observation_recipe(self) -> PickCubeRGBRecipe | None:
        if self._camera_context is not None:
            camera = self.camera_context
            if camera is None:
                raise ValueError("declared camera binding was removed")
            return camera_context_recipe(camera)
        return pick_cube_rgb_recipe() if self._visual_observation else None

    @property
    def camera_context(self) -> PickCubeCameraContext | None:
        return require_camera_binding(
            self._camera_context,
            context=self._context,
            seed=self._seed,
            role=self._role,
            visual_observation=self._visual_observation,
        )

    def _check_camera_binding(self) -> None:
        if self._camera_context_active or self._camera_context is not None:
            try:
                camera = self.camera_context
                if camera is None:
                    raise ValueError("admitted camera binding was removed")
                validate_pick_cube_camera_context(camera)
                if not self._camera_context_active:
                    raise ValueError("admitted camera binding was removed or replaced")
            except BaseException:
                self._failed = True
                raise

    @property
    def mass_context(self) -> PickCubeMassContext | None:
        return require_mass_binding(
            self._mass_context, context=self._context, seed=self._seed, role=self._role
        )

    @property
    def mass_readback(self) -> ManiSkillMassReadback | None:
        """Return an owned original measurement, not an implicit native reread."""
        context = self.mass_context
        if context is None:
            if self._mass_context_active:
                raise ValueError("admitted mass binding was removed")
            return None
        if not self._mass_context_active:
            raise ValueError("mass binding was added after admission")
        return ManiSkillMassReadback(context, self._mass_source_mass, self._mass_applied_mass)

    @property
    def mass_work(self) -> ManiSkillMassWork | None:
        if not self._mass_context_active:
            return None
        return ManiSkillMassWork(
            self._mass_reads,
            self._mass_read_returns,
            self._mass_writes,
            self._mass_write_returns,
        )

    def check_mass_metadata(
        self,
        context: PickCubeMassContext,
        original: ManiSkillMassReadback,
        observed_work: ManiSkillMassWork,
    ) -> None:
        """Check owned values after external callbacks; no virtual/native reads.

        Why: an owner or metadata getter can change another binding while
        returning its own unchanged value. This final pure check closes that
        ordering gap without a hidden physics read or a callback cycle.
        """
        expected = validate_pick_cube_mass_context(context)
        owned = require_mass_binding(
            self._mass_context, context=self._context, seed=self._seed, role=self._role
        )
        if owned != expected or self._mass_context_active is not True:
            raise ValueError("admitted mass metadata changed after external callbacks")
        current = ManiSkillMassReadback(owned, self._mass_source_mass, self._mass_applied_mass)
        if validate_mass_readback(current, expected) != validate_mass_readback(original, expected):
            raise ValueError("original mass readback changed after external callbacks")
        work = ManiSkillMassWork(
            self._mass_reads, self._mass_read_returns, self._mass_writes, self._mass_write_returns
        )
        if validate_mass_work(work) != validate_mass_work(observed_work):
            raise ValueError("observed mass work changed after external callbacks")

    def _require_mass_actor(self, actor: Any) -> None:
        if self.mass_context is None or not self._mass_context_active:
            raise ValueError("admitted mass binding was removed or replaced")
        if self._mass_actor is not actor or self._env.unwrapped.cube is not actor:
            raise ValueError("admitted cube actor was replaced")

    def _read_mass(self) -> float:
        actor = self._mass_actor
        self._require_mass_actor(actor)
        self._mass_reads += 1
        array = _array(actor.mass)
        # A native accessor is an external call: recheck the same identity after
        # it returns, before its result can admit a setter, control or export.
        self._require_mass_actor(actor)
        if (
            array.dtype != np.dtype("float32")
            or array.shape != (1,)
            or not np.isfinite(array).all()
            or not 0.0 < float(array[0]) <= 1.0
        ):
            raise ValueError("cube mass must be one finite positive CPU float32 value <=1kg")
        self._mass_read_returns += 1
        return float(array[0])

    def _apply_mass_context(self) -> None:
        context = self.mass_context
        if context is None:
            raise ValueError("missing declared mass binding")
        actor = self._env.unwrapped.cube
        self._mass_actor = actor
        source = self._read_mass()
        if source > 0.5:
            raise ValueError("source cube mass must be <=0.5kg before the declared doubling")
        target = float(np.float32(source * context.mass_multiplier))
        self._mass_source_mass, self._mass_applied_mass = source, target
        # Why: intervene after explicit reset, which may replace/reconfigure the
        # actor. Only mass is set; inertia/COM are neither set nor measured here.
        self._mass_writes += 1
        actor.set_mass(target)
        self._mass_write_returns += 1
        self._require_mass_actor(actor)
        if self._read_mass() != target:
            raise ValueError("cube mass setter did not return the exact declared float32 target")

    def _check_mass_binding(self) -> None:
        if self._mass_context_active or self._mass_context is not None:
            try:
                target = self._mass_applied_mass
                if self._read_mass() != target:
                    raise ValueError("observed cube mass differs from its admitted target")
            except BaseException:
                self._failed = True
                raise

    @property
    def lighting_context(self) -> PickCubeLightingContext | None:
        return require_lighting_binding(
            self._lighting_context,
            context=self._context,
            seed=self._seed,
            role=self._role,
            visual_observation=self._visual_observation,
        )

    @property
    def lighting_readback(self) -> ManiSkillLightingReadback | None:
        """Owned original measurements, including after cleanup; no native reread."""
        try:
            if self._lighting_context_active:
                if self.lighting_context is None or self._lighting is None:
                    raise ValueError("admitted lighting binding was removed")
                return self._lighting.readback
            if self._lighting_context is not None:
                raise ValueError("lighting binding was added after admission")
            return None
        except BaseException:
            self._failed = True
            raise

    @property
    def lighting_work(self) -> ManiSkillLightingWork | None:
        return self._lighting.work if self._lighting is not None else None

    def check_lighting_metadata(
        self,
        context: PickCubeLightingContext,
        original: ManiSkillLightingReadback,
        observed_work: ManiSkillLightingWork,
        *,
        observed_visual_work: ManiSkillVisualWork | None = None,
    ) -> None:
        """Final pure check after external owner/metadata getters; no native IO.

        Why: getters can change other stored fields while returning old values.
        Read the admitted helper's own storage, avoiding another virtual getter.
        Closed-scene state is not inspected or represented as a live invariant.
        """
        expected = validate_pick_cube_lighting_context(context)
        owned = require_lighting_binding(
            self._lighting_context,
            context=self._context,
            seed=self._seed,
            role=self._role,
            visual_observation=self._visual_observation,
        )
        helper, owner = self._lighting, self._lighting_owner
        if (
            owned != expected
            or self._lighting_context_active is not True
            or self._camera_context is not None
            or self._camera_context_active is not False
            or self._mass_context is not None
            or self._mass_context_active is not False
            or type(helper) is not ManiSkillAmbientLight
            or owner is None
            or self._env is not owner[0]
            or helper is not owner[3]
            or helper._base is not owner[1]
            or helper._scene is not owner[2]
        ):
            raise ValueError("admitted lighting metadata or IO owner changed")
        if helper._original is None:
            raise ValueError("original lighting readback was removed")
        value = validate_lighting_readback(helper._original, expected)
        if (
            validate_pick_cube_lighting_context(helper._context) != expected
            or helper._source != value.source_ambient_rgb
            or helper._target != value.applied_ambient_rgb
            or value != validate_lighting_readback(original, expected)
        ):
            raise ValueError("original lighting readback changed after external callbacks")
        work = ManiSkillLightingWork(
            helper._reads,
            helper._read_returns,
            helper._writes,
            helper._write_returns,
            helper._refreshes,
            helper._refresh_returns,
            1,
        )
        if validate_lighting_work(work) != validate_lighting_work(observed_work):
            raise ValueError("observed lighting work changed after external callbacks")
        if observed_visual_work is not None:
            expected_visual = validate_lighting_visual_work(observed_visual_work)
            if any(
                type(count) is not int
                for count in (self._visual_attempts, self._visual_returns, self._frame_exports)
            ) or (self._rgb_frame is not None and type(self._rgb_frame) is not np.ndarray):
                raise ValueError("owned visual work storage changed")
            current = ManiSkillVisualWork(
                self._visual_attempts,
                self._visual_returns,
                self._visual_attempts * 49152,
                self._visual_attempts * 789,
                self._visual_returns * 49152,
                self._visual_returns * 789,
                self._visual_returns * 6312,
                self._rgb_frame.nbytes if self._rgb_frame is not None else 0,
                self._frame_exports,
                self._frame_exports * 49152,
            )
            if validate_lighting_visual_work(current) != expected_visual:
                raise ValueError("observed visual work changed after external callbacks")

    def _apply_lighting_context(self) -> Any:
        context = self.lighting_context
        if context is None:
            raise ValueError("missing declared lighting context")
        base = self._env.unwrapped
        scene = base.scene

        def require_identity(observed_base: Any, observed_scene: Any) -> None:
            if self.lighting_context != context or self._lighting_context_active is not True:
                raise ValueError("admitted lighting binding was removed or replaced")
            if (
                observed_base is not base
                or observed_scene is not scene
                or self._env.unwrapped is not base
                or base.scene is not scene
            ):
                raise ValueError("admitted lighting base or scene was replaced")
            # A native identity accessor is external too; recheck the declaration.
            if self.lighting_context != context:
                raise ValueError("admitted lighting binding changed during identity access")

        self._lighting = ManiSkillAmbientLight(context, base, scene, require_identity)
        self._lighting_owner = (self._env, base, scene, self._lighting)
        return self._lighting.apply_and_refresh()

    def _check_lighting_binding(self) -> None:
        if self._lighting_context_active or self._lighting_context is not None:
            try:
                if self.lighting_context is None or self._lighting is None:
                    raise ValueError("admitted lighting binding was removed or replaced")
                self._lighting.verify()
            except BaseException:
                self._failed = True
                raise

    @property
    def visual_work(self) -> ManiSkillVisualWork | None:
        if not self._visual_observation:
            return None
        return ManiSkillVisualWork(
            self._visual_attempts,
            self._visual_returns,
            self._visual_attempts * 49152,
            self._visual_attempts * 789,
            self._visual_returns * 49152,
            self._visual_returns * 789,
            self._visual_returns * 6312,
            self._rgb_frame.nbytes if self._rgb_frame is not None else 0,
            self._frame_exports,
            self._frame_exports * 49152,
        )

    @property
    def rgb_frame(self) -> np.ndarray:
        if self._rgb_frame is None:
            raise ValueError("this episode has no owned RGB frame")
        frame = self._rgb_frame.copy(order="C")
        self._frame_exports += 1
        return frame

    def _encode_observation(self, observation: Any) -> tuple[float, ...]:
        self._check_camera_binding()
        self._check_mass_binding()
        self._check_lighting_binding()
        if not self._visual_observation:
            encoded_state = encode_state_observation(observation)
            self._sample_joint_positions(encoded_state)
            return encoded_state
        self._visual_attempts += 1
        try:
            if type(observation) is not dict:
                raise ValueError("RGB observation must be an exact nested dictionary")
            agent, extra, sensors = (observation[key] for key in ("agent", "extra", "sensor_data"))
            if any(type(value) is not dict for value in (agent, extra, sensors)):
                raise ValueError("RGB observation must keep exact nested dictionaries")
            camera = sensors["base_camera"]
            if type(camera) is not dict:
                raise ValueError("base_camera must keep its RGB dictionary")
            frame = _array(camera["rgb"])
            if frame.dtype != np.dtype("uint8") or frame.shape != (1, 128, 128, 3):
                raise ValueError("camera RGB must be one CPU uint8 NHWC128x128x3 frame")
            vectors = [
                np.asarray(encode_state_observation(value), dtype=np.float64)
                for value in (agent["qpos"], agent["qvel"], extra["goal_pos"])
            ]
            encoded = encode_pick_cube_rgb(frame[0], *vectors)
        except (KeyError, TypeError) as error:
            raise ValueError("RGB observation is missing a declared public input") from error
        self._visual_returns += 1
        self._rgb_frame = encoded.frame
        self._sample_joint_positions(encoded.features)
        return encoded.features

    @property
    def work(self) -> ManiSkillWork:
        return ManiSkillWork(
            self._attempts,
            self._completed,
            self._completed * self._physics_ratio,
            (self._attempts - self._completed) * self._physics_ratio,
        )

    def step(self, action: tuple[float, ...]) -> ContinuousTransition:
        if (
            self._closed
            or self._failed
            or len(self._records) >= HORIZON
            or (self._records and (self._records[-1].terminated or self._records[-1].truncated))
        ):
            raise ValueError("episode is closed, failed, complete or at its50-step bound")
        self._check_joint_position_binding()
        self._check_camera_binding()
        self._check_mass_binding()
        self._check_lighting_binding()
        # Validate before quantizing: an invalid action must never be rounded back
        # into bounds or passed to ManiSkill's internal clip-and-scale operation.
        trial = ContinuousTransition(
            len(self._records), self._observation, action, 0.0, self._observation, False, True
        )
        ContinuousEpisode(
            self._episode_id,
            self._context,
            self._role,
            self.spec,
            (
                ContinuousTransition(
                    0, trial.observation, trial.action, 0.0, trial.next_observation, False, True
                ),
            ),
        )
        command = np.array([trial.action], dtype=np.float32)
        sent = tuple(float(item) for item in command[0])
        self._check_joint_position_binding()
        self._attempts += 1
        try:
            output = self._env.step(command)
            self._completed += 1
            observation, reward, terminated, truncated, _ = output
            after = self._encode_observation(observation)
            if len(after) != self._spec.observation_dim:
                raise ValueError("state observation shape changed during episode")
            record = ContinuousTransition(
                len(self._records),
                self._observation,
                sent,
                float(_scalar(reward)),
                after,
                bool(_scalar(terminated, boolean=True)),
                bool(_scalar(truncated, boolean=True)),
            )
            if len(self._records) == HORIZON - 1 and not (record.terminated or record.truncated):
                raise ValueError("official50-step time-limit flag is missing")
            self._records.append(record)
            self._observation = after
            return record
        except BaseException:
            self._failed = True
            raise

    def episode(self) -> ContinuousEpisode:
        if self._failed:
            raise ValueError("failed interaction cannot produce an admissible episode")
        self._check_joint_position_binding()
        self._check_camera_binding()
        self._check_mass_binding()
        self._check_lighting_binding()
        self._check_joint_position_binding()
        return ContinuousEpisode(
            self._episode_id, self._context, self._role, self.spec, tuple(self._records)
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._env.close()

    def __enter__(self) -> "ManiSkillPickCubeEpisode":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
