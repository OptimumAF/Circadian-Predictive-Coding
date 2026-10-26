"""Collect TRAIN, guard or fixed reserved-probe teacher episodes through the port.

Privileged geometry is read only here and never given to a deployed learner.
No scoring, optimizer, planner, retry or new stream runner is supplied.
"""

from dataclasses import dataclass
import logging
import struct
import traceback
from typing import Any, cast

from src.core.continuous_demonstration import (
    ContinuousEpisode,
    ContinuousTransition,
    _copy_spec,
    _copy_transition,
    _vector,
    copy_continuous_episode,
)
from src.core.experience import require_tick
from src.core.pick_cube_guard_context import pick_cube_guard_context
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_lighting_context import (
    PickCubeLightingContext,
    pick_cube_lighting_context,
    require_lighting_binding,
    validate_pick_cube_lighting_context,
)
from src.core.pick_cube_mass_context import (
    PickCubeMassContext,
    pick_cube_mass_context,
    require_mass_binding,
    validate_pick_cube_mass_context,
)
from src.core.pick_cube_teacher import PickCubeGeometry, PickCubeTeacher
from src.core.pick_cube_readiness_teacher import (
    DEFINITION as READINESS_DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
    GoalConsistentObservedGraspPickCubeTeacher,
    ObservedGraspPickCubeTeacher,
    ReadinessDecision,
    ReadinessPickCubeTeacher,
)
from src.core.pick_cube_train_context import TRAIN_SEEDS, pick_cube_training_context
from src.core.pick_cube_readiness_corrections import ReadinessCorrectionLabel, _observed_grasp_flags
from src.core.pick_cube_rgb_features import PickCubeRGBRecipe, validate_pick_cube_rgb_recipe
from src.core.pick_cube_camera_context import (
    PickCubeCameraContext,
    camera_context_recipe,
    pick_cube_camera_context,
    require_camera_binding,
    validate_camera_recipe,
    validate_pick_cube_camera_context,
)
from src.infra.maniskill_continuous import (
    HORIZON,
    ACTION_UNITS,
    ManiSkillPickCubeEpisode,
    ManiSkillMassReadback,
    ManiSkillMassWork,
    ManiSkillLightingReadback,
    ManiSkillLightingWork,
    ManiSkillWork,
    ManiSkillVisualWork,
    encode_state_observation,
    validate_mass_readback,
    validate_mass_work,
    validate_lighting_visual_work,
    validate_lighting_readback,
    validate_lighting_work,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskTeacherResult:
    candidate_episode: ContinuousEpisode | None
    observed_success: bool
    final_success: bool
    creation_attempts: int
    geometry_attempts: int
    geometry_returns: int
    teacher_actions: int
    stages: tuple[str, ...]
    work: ManiSkillWork | None
    error: str | None
    traceback: str | None
    observation_recipe: PickCubeRGBRecipe | None = None
    visual_work: ManiSkillVisualWork | None = None
    guard_seed: int | None = None
    labels_arrived_at: int | None = None
    camera_context: PickCubeCameraContext | None = None
    mass_context: PickCubeMassContext | None = None
    mass_readback: ManiSkillMassReadback | None = None
    mass_work: ManiSkillMassWork | None = None
    lighting_context: PickCubeLightingContext | None = None
    lighting_readback: ManiSkillLightingReadback | None = None
    lighting_work: ManiSkillLightingWork | None = None
    teacher_definition: str | None = None

    @property
    def qualified(self) -> bool:
        if self.candidate_episode is None:
            return False
        if (
            type(self)
            in (
                ObservedGraspTaskTeacherResult,
                GoalConsistentObservedGraspTaskTeacherResult,
                ReservedProbeTaskTeacherResult,
            )
            or hasattr(self, "observed_grasp")
        ) and (type(self.candidate_episode) is not ContinuousEpisode):
            return False
        basic = self.final_success and self.error is None
        if basic and (
            type(self)
            in (
                ObservedGraspTaskTeacherResult,
                GoalConsistentObservedGraspTaskTeacherResult,
                ReservedProbeTaskTeacherResult,
            )
            or self.teacher_definition is not None
            or self.candidate_episode.episode_id.startswith("pick-cube-readiness-teacher-")
            or self.candidate_episode.episode_id.startswith("pick-cube-observed-grasp-teacher-")
            or self.candidate_episode.episode_id.startswith(
                "pick-cube-goal-consistent-observed-grasp-teacher-"
            )
            or self.candidate_episode.episode_id.startswith("pick-cube-probe-")
            or self.candidate_episode.context == "pick-cube-reserved-probe-23-24"
            or hasattr(self, "probe_seed")
            or hasattr(self, "observed_grasp")
        ):
            try:
                self._check_readiness_episode()
            except (ValueError, TypeError, AttributeError):
                return False
        known_lighting = pick_cube_lighting_context()
        if basic and (
            self.lighting_context is not None
            or self.lighting_readback is not None
            or self.lighting_work is not None
            or self.candidate_episode.context == known_lighting.context
            or self.candidate_episode.episode_id == f"{known_lighting.context}-teacher"
        ):
            try:
                self._check_lighting_episode()
            except (ValueError, TypeError, AttributeError):
                return False
        known_mass = pick_cube_mass_context()
        if basic and (
            self.mass_context is not None
            or self.mass_readback is not None
            or self.mass_work is not None
            or self.candidate_episode.context == known_mass.context
            or self.candidate_episode.episode_id == f"{known_mass.context}-teacher"
        ):
            try:
                self._check_mass_episode()
            except (ValueError, TypeError, AttributeError):
                return False
        known_camera = pick_cube_camera_context()
        if basic and (
            self.camera_context is not None
            or self.candidate_episode.context == known_camera.context
            or self.candidate_episode.episode_id == f"{known_camera.context}-teacher"
            or (
                self.observation_recipe is not None
                and self.observation_recipe.recipe_id
                == camera_context_recipe(known_camera).recipe_id
            )
        ):
            try:
                self._check_camera_episode()
            except ValueError:
                return False
        return basic

    def _check_readiness_episode(self) -> None:
        reserved = type(self) is ReservedProbeTaskTeacherResult
        goal_consistent = type(self) in (
            GoalConsistentObservedGraspTaskTeacherResult,
            ReservedProbeTaskTeacherResult,
        )
        selected = type(self) in (
            ObservedGraspTaskTeacherResult,
            GoalConsistentObservedGraspTaskTeacherResult,
            ReservedProbeTaskTeacherResult,
        )
        expected_definition = (
            GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
            if goal_consistent
            else OBSERVED_GRASP_DEFINITION
            if selected
            else READINESS_DEFINITION
        )
        if (
            type(self)
            not in (
                TaskTeacherResult,
                ObservedGraspTaskTeacherResult,
                GoalConsistentObservedGraspTaskTeacherResult,
                ReservedProbeTaskTeacherResult,
            )
            or self.teacher_definition != expected_definition
            or self.candidate_episode is None
            or (not selected and hasattr(self, "observed_grasp"))
            or (not reserved and hasattr(self, "probe_seed"))
        ):
            raise ValueError("readiness teacher lost its distinct definition or family")
        observed_feedback = (
            cast(ObservedGraspTaskTeacherResult, self).observed_grasp if selected else None
        )
        if selected:
            flags = _observed_grasp_flags(self.candidate_episode)
            if (
                type(observed_feedback) is not tuple
                or any(type(flag) is not bool for flag in observed_feedback)
                or observed_feedback != flags
                or self.observation_recipe is not None
                or self.visual_work is not None
            ):
                raise ValueError("observed-grasp teacher feedback differs from raw state")
        value = copy_continuous_episode(self.candidate_episode)
        if reserved:
            allocation = pick_cube_probe_binding(value.role)
            probe_seed = cast(ReservedProbeTaskTeacherResult, self).probe_seed
            if type(probe_seed) is not int or probe_seed != allocation.seed:
                raise ValueError("reserved probe result lost its original reset binding")
            expected_context, expected_id, expected_role = (
                allocation.context,
                allocation.episode_id,
                allocation.role,
            )
        else:
            # Why: match an explicit allocation; an unknown context must never
            # inherit the last known reset's identity or qualification authority.
            seeds = [
                seed for seed in TRAIN_SEEDS if value.context == pick_cube_training_context(seed)
            ]
            if len(seeds) != 1:
                raise ValueError("readiness teacher has no declared TRAIN context")
            seed = seeds[0]
            expected_context = pick_cube_training_context(seed)
            expected_id = (
                f"pick-cube-goal-consistent-observed-grasp-teacher-{seed}"
                if goal_consistent
                else f"pick-cube-observed-grasp-teacher-{seed}"
                if selected
                else f"pick-cube-readiness-teacher-{seed}"
            )
            expected_role = "train"
        if (
            value.context != expected_context
            or value.episode_id != expected_id
            or value.role != expected_role
            or self.observed_success is not True
            or self.final_success is not True
            or self.error is not None
            or self.traceback is not None
            or any(
                metadata is not None
                for metadata in (
                    self.guard_seed,
                    self.labels_arrived_at,
                    self.camera_context,
                    self.mass_context,
                    self.mass_readback,
                    self.mass_work,
                    self.lighting_context,
                    self.lighting_readback,
                    self.lighting_work,
                )
            )
            or value.spec.action_units != ACTION_UNITS
            or value.spec.lower_bounds != (-1.0,) * 4
            or value.spec.upper_bounds != (1.0,) * 4
            or len(value.transitions) > HORIZON
            or value.transitions[-1].reward != 1.0
            or any(row.reward not in (0.0, 1.0) for row in value.transitions)
        ):
            raise ValueError("readiness teacher differs from its declared TRAIN binding")
        steps = len(value.transitions)
        if (
            type(self.creation_attempts) is not int
            or self.creation_attempts != 1
            or any(
                type(count) is not int or count != steps
                for count in (self.geometry_attempts, self.geometry_returns, self.teacher_actions)
            )
            or type(self.stages) is not tuple
            or len(self.stages) != steps
            or any(
                type(stage) is not str
                or stage not in ("align", "descend", "grasp", "lift", "settle")
                for stage in self.stages
            )
            or self.work is None
            or _copy_mass_teacher_work(self.work) != ManiSkillWork(steps, steps, steps * 5, 0)
        ):
            raise ValueError("readiness teacher work differs from the complete collection")
        for index, (stage, row) in enumerate(zip(self.stages, value.transitions, strict=True)):
            if (
                selected
                and observed_feedback is not None
                and ((stage in ("lift", "settle")) is not observed_feedback[index])
            ):
                raise ValueError("observed-grasp teacher stage differs from current feedback")
            if row.action[3] != (1.0 if stage in ("align", "descend") else -1.0):
                raise ValueError("readiness stage differs from its recorded gripper command")
            if stage in ("grasp", "settle") and row.action[:3] != (0.0, 0.0, 0.0):
                raise ValueError("readiness hold stage has a translation command")
        if self.observation_recipe is None:
            if self.visual_work is not None or value.spec.observation_dim != 42:
                raise ValueError(
                    "readiness state teacher requires the original42-state observation"
                )
        else:
            recipe = validate_pick_cube_rgb_recipe(self.observation_recipe)
            if value.spec.observation_dim != recipe.feature_dim or any(
                not 0.0 <= pixel <= 1.0
                for row in value.transitions
                for observation in (row.observation, row.next_observation)
                for pixel in observation[21:]
            ):
                raise ValueError("readiness visual teacher differs from the base RGB recipe")
            _copy_lighting_visual_work(self.visual_work, steps=steps)

    def _check_lighting_episode(self) -> None:
        if (
            self.lighting_context is None
            or self.lighting_readback is None
            or self.lighting_work is None
            or self.observation_recipe is None
            or self.candidate_episode is None
        ):
            raise ValueError("lighting teacher lost its declared binding")
        lighting = validate_pick_cube_lighting_context(self.lighting_context)
        value = copy_continuous_episode(self.candidate_episode)
        require_lighting_binding(
            lighting,
            context=value.context,
            seed=lighting.seed,
            role=value.role,
            visual_observation=True,
        )
        validate_lighting_readback(self.lighting_readback, lighting)
        lighting_work = validate_lighting_work(self.lighting_work)
        recipe = validate_pick_cube_rgb_recipe(self.observation_recipe)
        if (
            value.episode_id != f"{lighting.context}-teacher"
            or self.observed_success is not True
            or self.final_success is not True
            or self.error is not None
            or self.traceback is not None
            or self.camera_context is not None
            or self.mass_context is not None
            or self.mass_readback is not None
            or self.mass_work is not None
            or self.guard_seed is not None
            or self.labels_arrived_at is not None
            or value.spec.observation_dim != recipe.feature_dim
            or value.spec.action_units != ACTION_UNITS
            or value.spec.lower_bounds != (-1.0,) * 4
            or value.spec.upper_bounds != (1.0,) * 4
            or len(value.transitions) > HORIZON
            or value.transitions[-1].reward != 1.0
            or any(row.reward not in (0.0, 1.0) for row in value.transitions)
            or any(
                not 0.0 <= pixel <= 1.0
                for row in value.transitions
                for observation in (row.observation, row.next_observation)
                for pixel in observation[21:]
            )
        ):
            raise ValueError("lighting teacher differs from its declared visual TRAIN binding")
        steps = len(value.transitions)
        expected_stages = (
            ("align",) * 8 + ("descend",) * 8 + ("grasp",) * 6 + ("lift",) * 18 + ("settle",) * 10
        )[:steps]
        if (
            lighting_work.read_attempts != lighting_work.read_returns
            or lighting_work.read_returns != 4 + 2 * steps
            or lighting_work.write_attempts != 1
            or lighting_work.write_returns != 1
            or lighting_work.refresh_attempts != 1
            or lighting_work.refresh_returns != 1
            or lighting_work.discarded_reset_observations != 1
            or type(self.creation_attempts) is not int
            or self.creation_attempts != 1
            or any(
                type(count) is not int or count != steps
                for count in (self.geometry_attempts, self.geometry_returns, self.teacher_actions)
            )
            or type(self.stages) is not tuple
            or len(self.stages) != steps
            or any(type(stage) is not str for stage in self.stages)
            or self.stages != expected_stages
            or self.work is None
            or _copy_mass_teacher_work(self.work) != ManiSkillWork(steps, steps, steps * 5, 0)
        ):
            raise ValueError("lighting teacher work differs from its complete one-export lifecycle")
        _copy_lighting_visual_work(self.visual_work, steps=steps)

    def _check_mass_episode(self) -> None:
        if (
            self.mass_context is None
            or self.mass_readback is None
            or self.mass_work is None
            or self.candidate_episode is None
        ):
            raise ValueError("mass teacher lost its declared binding")
        mass = validate_pick_cube_mass_context(self.mass_context)
        value = copy_continuous_episode(self.candidate_episode)
        require_mass_binding(mass, context=value.context, seed=mass.seed, role=value.role)
        validate_mass_readback(self.mass_readback, mass)
        mass_work = validate_mass_work(self.mass_work)
        if (
            value.episode_id != f"{mass.context}-teacher"
            or self.observed_success is not True
            or self.final_success is not True
            or self.camera_context is not None
            or self.guard_seed is not None
            or self.labels_arrived_at is not None
            or value.spec.action_units != ACTION_UNITS
            or value.spec.lower_bounds != (-1.0,) * 4
            or value.spec.upper_bounds != (1.0,) * 4
            or len(value.transitions) > HORIZON
            or value.transitions[-1].reward != 1.0
            or any(row.reward not in (0.0, 1.0) for row in value.transitions)
        ):
            raise ValueError("mass teacher episode differs from its declared TRAIN binding")
        self._check_mass_observations(value)
        steps = len(value.transitions)
        # Why: the fixed port reads at reset3, before/after each control2,
        # and once at the sole episode export; metadata getters read no physics.
        if (
            mass_work.read_attempts != mass_work.read_returns
            or mass_work.read_returns != 4 + 2 * steps
            or mass_work.write_attempts != 1
            or mass_work.write_returns != 1
            or type(self.creation_attempts) is not int
            or self.creation_attempts != 1
            or any(
                type(count) is not int or count != steps
                for count in (self.geometry_attempts, self.geometry_returns, self.teacher_actions)
            )
            or type(self.stages) is not tuple
            or len(self.stages) != steps
            or type(self.work) is not ManiSkillWork
            or type(self.work.step_attempts) is not int
            or type(self.work.completed_steps) is not int
            or type(self.work.completed_control_physics_steps) is not int
            or type(self.work.possible_failed_physics_steps) is not int
            or self.work.step_attempts != steps
            or self.work.completed_steps != steps
            or self.work.completed_control_physics_steps != steps * 5
            or self.work.possible_failed_physics_steps != 0
        ):
            raise ValueError("mass teacher work differs from its complete one-export lifecycle")

    def _check_mass_observations(self, value: ContinuousEpisode) -> None:
        if self.observation_recipe is None:
            if value.spec.observation_dim != 42 or self.visual_work is not None:
                raise ValueError("mass state teacher requires the original42-state observation")
        else:
            recipe = validate_pick_cube_rgb_recipe(self.observation_recipe)
            if value.spec.observation_dim != recipe.feature_dim or any(
                not 0.0 <= pixel <= 1.0
                for row in value.transitions
                for observation in (row.observation, row.next_observation)
                for pixel in observation[21:]
            ):
                raise ValueError("mass visual teacher requires the original base RGB recipe")

    def _check_camera_episode(self) -> None:
        if (
            self.camera_context is None
            or self.observation_recipe is None
            or self.candidate_episode is None
        ):
            raise ValueError("camera teacher lost its declared binding")
        camera = validate_pick_cube_camera_context(self.camera_context)
        value = copy_continuous_episode(self.candidate_episode)
        require_camera_binding(
            camera,
            context=value.context,
            seed=camera.seed,
            role=value.role,
            visual_observation=True,
        )
        validate_camera_recipe(self.observation_recipe, camera)
        if (
            value.episode_id != f"{camera.context}-teacher"
            or self.observed_success is not True
            or self.final_success is not True
            or self.guard_seed is not None
            or self.labels_arrived_at is not None
            or value.spec.observation_dim != 789
            or value.spec.action_units != ACTION_UNITS
            or value.spec.lower_bounds != (-1.0,) * 4
            or value.spec.upper_bounds != (1.0,) * 4
            or len(value.transitions) > HORIZON
            or value.transitions[-1].reward != 1.0
            or any(row.reward not in (0.0, 1.0) for row in value.transitions)
            or any(
                not 0.0 <= pixel <= 1.0
                for row in value.transitions
                for observation in (row.observation, row.next_observation)
                for pixel in observation[21:]
            )
        ):
            raise ValueError("camera teacher episode differs from its declared TRAIN binding")

    def training_episode(self) -> ContinuousEpisode:
        if not self.qualified or self.candidate_episode is None:
            raise ValueError("task teacher has not qualified for BC training")
        value = self.candidate_episode
        if (
            type(value) is not ContinuousEpisode
            or value.role != "train"
            or self.guard_seed is not None
            or self.labels_arrived_at is not None
        ):
            raise ValueError("BC training extraction requires TRAIN data without guard metadata")
        return copy_continuous_episode(value)

    def guard_episode(self, *, now: int) -> ContinuousEpisode:
        """Return arrived owned guard data; no training or promotion authority."""
        require_tick(now, "guard observation time")
        if self.labels_arrived_at is None or self.guard_seed is None:
            raise ValueError("guard extraction requires declared seed and arrival metadata")
        require_tick(self.labels_arrived_at, "guard label arrival")
        if self.labels_arrived_at > now:
            raise ValueError("guard labels have not arrived")
        context = pick_cube_guard_context(self.guard_seed)
        if (
            self.observed_success is not True
            or self.final_success is not True
            or not self.qualified
            or self.candidate_episode is None
        ):
            raise ValueError("task teacher has not qualified for guard evaluation")
        value = copy_continuous_episode(self.candidate_episode)
        if (
            value.role != "inner_guard"
            or value.context != context
            or value.episode_id != f"pick-cube-guard-teacher-{self.guard_seed}"
            or value.transitions[-1].reward != 1.0
            or any(row.reward not in (0.0, 1.0) for row in value.transitions)
            or len(value.transitions) > HORIZON
            or value.spec.action_units != ACTION_UNITS
            or value.spec.lower_bounds != (-1.0,) * 4
            or value.spec.upper_bounds != (1.0,) * 4
        ):
            raise ValueError("guard episode role/context/identity differs from declared allocation")
        if self.observation_recipe is not None:
            recipe = validate_pick_cube_rgb_recipe(self.observation_recipe)
            if value.spec.observation_dim != recipe.feature_dim:
                raise ValueError("guard observation dimension differs from visual recipe")
            if any(
                not 0.0 <= pixel <= 1.0
                for row in value.transitions
                for observation in (row.observation, row.next_observation)
                for pixel in observation[21:]
            ):
                raise ValueError("guard RGB features must lie in range [0,1]")
        elif value.spec.observation_dim == PickCubeRGBRecipe().feature_dim:
            raise ValueError("visual guard lost its observation recipe")
        return value


def _geometry(port: ManiSkillPickCubeEpisode) -> PickCubeGeometry:
    # Why: this privileged read stays within infra, on the same role-bound teacher
    # owner. Every actual action still passes through port.step and its checks.
    base = port._env.unwrapped
    root_from_world = base.agent.robot.pose.inv()
    points = [
        encode_state_observation((root_from_world * pose).p)
        for pose in (base.agent.tcp.pose, base.cube.pose, base.goal_site.pose)
    ]
    return PickCubeGeometry(*points)


@dataclass(frozen=True, kw_only=True)
class ObservedGraspTaskTeacherResult(TaskTeacherResult):
    """Selected feedback only; base teacher result fields and default JSON stay unchanged."""

    observed_grasp: tuple[bool, ...]


@dataclass(frozen=True, kw_only=True)
class GoalConsistentObservedGraspTaskTeacherResult(ObservedGraspTaskTeacherResult):
    """Explicit v2 policy family; existing serialized v1 fields remain unchanged."""


@dataclass(frozen=True, kw_only=True)
class ReservedProbeTaskTeacherResult(GoalConsistentObservedGraspTaskTeacherResult):
    """Qualified reserved-reset data; never grants benchmark TRAIN extraction."""

    probe_seed: int

    def probe_episode(self) -> ContinuousEpisode:
        if (
            type(self) is not ReservedProbeTaskTeacherResult
            or not self.qualified
            or self.candidate_episode is None
        ):
            raise ValueError("reserved probe teacher has not qualified")
        return copy_continuous_episode(self.candidate_episode)


def _official_teacher_grasp(observation) -> bool:
    if (
        type(observation) is not tuple
        or len(observation) != 42
        or type(observation[18]) is not float
        or observation[18] not in (0.0, 1.0)
    ):
        raise ValueError("observed-grasp teacher requires current official state42 float0/1")
    _vector(observation, "observed-grasp teacher state", 42)
    return observation[18] == 1.0


class _ObservedTeacherBinding:
    """Bounded original port/teacher pairing; no native reads beyond existing getters."""

    def __init__(self, port, teacher, seed, context, *, goal_consistent=False, probe_role=None):
        self.port, self.teacher, self.seed, self.context = port, teacher, seed, context
        self.role = "train" if probe_role is None else probe_role
        allocation = pick_cube_probe_binding(probe_role) if probe_role is not None else None
        if allocation is not None and (seed != allocation.seed or context != allocation.context):
            raise ValueError("observed probe teacher differs from the fixed reserved split")
        self.teacher_type = (
            GoalConsistentObservedGraspPickCubeTeacher
            if goal_consistent
            else ObservedGraspPickCubeTeacher
        )
        self.episode_id = (
            allocation.episode_id
            if allocation is not None
            else f"pick-cube-goal-consistent-observed-grasp-teacher-{seed}"
            if goal_consistent
            else f"pick-cube-observed-grasp-teacher-{seed}"
        )
        self.records = port._records
        self.env = port._env
        self.spec = _copy_spec(port._spec)
        self.observation = port._observation
        self.originals: tuple[ContinuousTransition, ...] = ()
        self.copies: tuple[ContinuousTransition, ...] = ()
        self.teacher_state = (0, "align", 0)
        self.work: ManiSkillWork | None = None
        self.observe(port)

    def _teacher(self):
        teacher = self.teacher
        if type(teacher) is not self.teacher_type:
            raise ValueError("observed-grasp teacher owner changed")
        state = (teacher._steps, teacher._phase, teacher._closing_steps)
        if (
            type(state[0]) is not int
            or not 0 <= state[0] <= HORIZON
            or type(state[1]) is not str
            or state[1] not in ("align", "descend", "grasp", "lift", "settle")
            or type(state[2]) is not int
            or not 0 <= state[2] <= 6
        ):
            raise ValueError("observed-grasp teacher scalar state changed")
        return state

    def native(self, port, *, closed=False, check_teacher=True):
        if (
            type(port) is not ManiSkillPickCubeEpisode
            or port is not self.port
            or port._env is not self.env
            or port._records is not self.records
            or type(port._seed) is not int
            or port._seed != self.seed
            or port._context != self.context
            or port._role != self.role
            or port._episode_id != self.episode_id
            or port._visual_observation is not False
            or port._camera_context is not None
            or port._mass_context is not None
            or port._lighting_context is not None
            or port._camera_context_active is not False
            or port._mass_context_active is not False
            or port._lighting_context_active is not False
            or port._closed is not closed
            or port._failed is not False
            or _copy_spec(port._spec) != self.spec
            or self.spec.observation_dim != 42
            or self.spec.action_units != ACTION_UNITS
            or self.spec.lower_bounds != (-1.0,) * 4
            or self.spec.upper_bounds != (1.0,) * 4
            or type(port._attempts) is not int
            or port._attempts != len(self.originals)
            or type(port._completed) is not int
            or port._completed != len(self.originals)
            or type(port._physics_ratio) is not int
            or port._physics_ratio != 5
            or len(port._records) != len(self.originals)
            or port._observation != self.observation
        ):
            raise ValueError("observed-grasp teacher port binding changed")
        _official_teacher_grasp(port._observation)
        for row, original, saved in zip(port._records, self.originals, self.copies, strict=True):
            _official_teacher_grasp(row.observation)
            _official_teacher_grasp(row.next_observation)
            if row is not original or _copy_transition(row) != saved:
                raise ValueError("preceding observed-grasp teacher record changed")
        if check_teacher and self._teacher() != self.teacher_state:
            raise ValueError("observed-grasp teacher preceding decision changed")

    def remember_work(self, work):
        value = _copy_mass_teacher_work(work)
        if self.work is not None and any(
            getattr(value, name) < getattr(self.work, name)
            for name in (
                "step_attempts",
                "completed_steps",
                "completed_control_physics_steps",
                "possible_failed_physics_steps",
            )
        ):
            raise ValueError("paid observed-grasp teacher work decreased")
        self.work = value

    def observe(self, port, *, closed=False) -> bool:
        observation, spec, work = port.observation, port.spec, port.work
        if observation != self.observation or spec != self.spec:
            raise ValueError("observed-grasp teacher public pairing changed")
        value = _copy_mass_teacher_work(work)
        self.remember_work(value)
        if value != ManiSkillWork(
            len(self.originals), len(self.originals), 5 * len(self.originals), 0
        ):
            raise ValueError("observed-grasp teacher work changed before another action")
        self.native(port, closed=closed)
        return _official_teacher_grasp(self.observation)

    def decision(self, port, decision):
        self.native(port, check_teacher=False)
        if type(decision) is not ReadinessDecision:
            raise ValueError("observed-grasp teacher returned an unexpected decision")
        label = ReadinessCorrectionLabel(
            len(self.originals), self.observation, decision.action, decision.stage
        )
        grasped = _official_teacher_grasp(self.observation)
        if (label.stage in ("lift", "settle")) is not grasped:
            raise ValueError("observed-grasp decision differs from admitted current feedback")
        expected_close = (
            min(6, self.teacher_state[2] + 1)
            if label.stage == "grasp"
            else self.teacher_state[2]
            if label.stage in ("lift", "settle")
            else 0
        )
        expected = (self.teacher_state[0] + 1, label.stage, expected_close)
        if self._teacher() != expected:
            raise ValueError("observed-grasp teacher did not advance its original scalar state")
        self.teacher_state = expected
        self.observe(port)
        return label.stage, label.target

    def control(self, port, record, command):
        if type(record) is not ContinuousTransition:
            raise ValueError("observed-grasp teacher requires an exact returned transition")
        sent = tuple(struct.unpack("f", struct.pack("f", value))[0] for value in command)
        if (
            record.step != len(self.originals)
            or record.observation != self.observation
            or record.action != sent
            or len(port._records) != len(self.originals) + 1
            or port._records[-1] is not record
            or port._observation != record.next_observation
        ):
            raise ValueError("returned teacher control differs from admitted observation/action")
        _official_teacher_grasp(record.observation)
        _official_teacher_grasp(record.next_observation)
        self.originals += (record,)
        self.copies += (_copy_transition(record),)
        self.observation = record.next_observation
        self.observe(port)

    def episode(self, port, episode):
        self.observe(port)
        if (
            type(episode) is not ContinuousEpisode
            or episode.episode_id != port._episode_id
            or episode.context != self.context
            or episode.role != self.role
            or episode.spec != self.spec
            or episode.transitions != self.copies
        ):
            raise ValueError("exported observed-grasp teacher episode changed")
        _observed_grasp_flags(episode)


class _MassTeacherBinding:
    """Bind original measurements while retaining the last valid paid counters.

    These are metadata reads only. Live actor/mass checks remain on the real
    port's step and episode export, where their native reads are charged.
    """

    def __init__(self, context: PickCubeMassContext):
        self.context = validate_pick_cube_mass_context(context)
        self.original: ManiSkillMassReadback | None = None
        self.readback: ManiSkillMassReadback | None = None
        self.work: ManiSkillMassWork | None = None

    def observe(self, port: ManiSkillPickCubeEpisode) -> None:
        self.remember_work(port)
        context = port.mass_context
        if context is None or validate_pick_cube_mass_context(context) != self.context:
            raise ValueError("declared teacher mass context changed")
        readback = port.mass_readback
        if readback is None:
            raise ValueError("declared teacher mass readback was removed")
        value = validate_mass_readback(readback, self.context)
        if self.original is None:
            self.original = validate_mass_readback(value, self.context)
        elif value != self.original:
            raise ValueError("original teacher mass readback changed")
        self.readback = value
        context = port.mass_context
        if context is None or validate_pick_cube_mass_context(context) != self.context:
            raise ValueError("declared teacher mass context changed after metadata read")
        if self.original is None or self.work is None:
            raise ValueError("mass teacher has not observed its original binding and work")
        # Why: the final nonvirtual check catches a getter changing another
        # stored binding while returning its own unchanged metadata value.
        ManiSkillPickCubeEpisode.check_mass_metadata(port, self.context, self.original, self.work)

    def remember_work(self, port: ManiSkillPickCubeEpisode) -> None:
        work = port.mass_work
        if work is None:
            raise ValueError("declared teacher mass work was removed")
        value = validate_mass_work(work)
        if self.work is not None and any(
            getattr(value, name) < getattr(self.work, name)
            for name in ("read_attempts", "read_returns", "write_attempts", "write_returns")
        ):
            raise ValueError("paid teacher mass work counters decreased")
        self.work = value


class _LightingTeacherBinding:
    """Keep original lighting receipts separate from later metadata callbacks."""

    def __init__(self, context: PickCubeLightingContext):
        self.context = validate_pick_cube_lighting_context(context)
        self.original: ManiSkillLightingReadback | None = None
        self.readback: ManiSkillLightingReadback | None = None
        self.work: ManiSkillLightingWork | None = None

    def observe(self, port: ManiSkillPickCubeEpisode) -> None:
        self._remember_work(port.lighting_work)
        context = port.lighting_context
        if context is None or validate_pick_cube_lighting_context(context) != self.context:
            raise ValueError("declared teacher lighting context changed")
        self._remember_readback(port.lighting_readback)
        context = port.lighting_context
        if context is None or validate_pick_cube_lighting_context(context) != self.context:
            raise ValueError("teacher lighting context changed after metadata read")
        if self.original is None or self.work is None:
            raise ValueError("lighting teacher has not observed its original binding and work")
        ManiSkillPickCubeEpisode.check_lighting_metadata(
            port, self.context, self.original, self.work
        )

    def preserve_failure(self, error: BaseException) -> None:
        # Why: a constructor may pay source/setter/refresh work before returning
        # a port. Its original exception carries the already observed receipts.
        work = getattr(error, "lighting_work", None)
        if work is not None:
            self._remember_work(work)
        readback = getattr(error, "lighting_readback", None)
        if readback is not None:
            self._remember_readback(readback)

    def _remember_readback(self, readback: ManiSkillLightingReadback | None) -> None:
        if readback is None:
            raise ValueError("declared teacher lighting readback was removed")
        value = validate_lighting_readback(readback, self.context)
        if self.original is None:
            self.original = validate_lighting_readback(value, self.context)
        elif value != self.original:
            raise ValueError("original teacher lighting readback changed")
        self.readback = value

    def _remember_work(self, work: ManiSkillLightingWork | None) -> None:
        if work is None:
            raise ValueError("declared teacher lighting work was removed")
        value = validate_lighting_work(work)
        if self.work is not None and any(
            getattr(value, name) < getattr(self.work, name)
            for name in (
                "read_attempts",
                "read_returns",
                "write_attempts",
                "write_returns",
                "refresh_attempts",
                "refresh_returns",
                "discarded_reset_observations",
            )
        ):
            raise ValueError("paid teacher lighting work counters decreased")
        self.work = value


def _copy_lighting_visual_work(
    value: ManiSkillVisualWork | None, *, steps: int | None = None
) -> ManiSkillVisualWork:
    if value is None:
        raise ValueError("lighting teacher requires native visual work")
    return validate_lighting_visual_work(value, steps=steps)


def _copy_mass_teacher_work(value: ManiSkillWork) -> ManiSkillWork:
    if type(value) is not ManiSkillWork:
        raise ValueError("mass teacher requires exact native controller work")
    counts = (
        value.step_attempts,
        value.completed_steps,
        value.completed_control_physics_steps,
        value.possible_failed_physics_steps,
    )
    if (
        any(type(count) is not int or not 0 <= count <= HORIZON * 5 for count in counts)
        or not 0 <= value.completed_steps <= value.step_attempts <= HORIZON
        or value.completed_control_physics_steps != value.completed_steps * 5
        or value.possible_failed_physics_steps != (value.step_attempts - value.completed_steps) * 5
    ):
        raise ValueError("mass teacher controller work differs from the fixed port lifecycle")
    return ManiSkillWork(*counts)


def collect_task_teacher(
    *,
    train_seed: int = 17,
    visual_observation: bool = False,
    camera_context: PickCubeCameraContext | None = None,
    mass_context: PickCubeMassContext | None = None,
    lighting_context: PickCubeLightingContext | None = None,
) -> TaskTeacherResult:
    """One declared TRAIN reset; only complete final-success data can qualify."""
    context = pick_cube_training_context(train_seed)
    lighting = None
    if lighting_context is not None:
        if camera_context is not None or mass_context is not None:
            raise ValueError("lighting, mass and camera teachers require separate collections")
        lighting = validate_pick_cube_lighting_context(lighting_context)
        context = lighting.context
        require_lighting_binding(
            lighting,
            context=context,
            seed=train_seed,
            role="train",
            visual_observation=visual_observation,
        )
    mass = None
    if mass_context is not None:
        if camera_context is not None:
            raise ValueError("mass and camera teacher contexts require separate collections")
        mass = validate_pick_cube_mass_context(mass_context)
        context = mass.context
        require_mass_binding(mass, context=context, seed=train_seed, role="train")
    camera = None
    if camera_context is not None:
        camera = validate_pick_cube_camera_context(camera_context)
        context = camera.context
        require_camera_binding(
            camera,
            context=context,
            seed=train_seed,
            role="train",
            visual_observation=visual_observation,
        )
    lighting_options: dict[str, Any] = {}
    if lighting is not None:
        lighting_options["lighting_context"] = lighting
    return _collect_teacher(
        train_seed,
        context,
        "train",
        visual_observation,
        camera_context=camera,
        mass_context=mass,
        **lighting_options,
    )


def collect_guard_teacher(
    *, guard_seed: int, labels_arrived_at: int, visual_observation: bool = False
) -> TaskTeacherResult:
    """One prospectively allocated guard reset, never a training episode."""
    context = pick_cube_guard_context(guard_seed)
    require_tick(labels_arrived_at, "guard label arrival")
    return _collect_teacher(
        guard_seed, context, "inner_guard", visual_observation, labels_arrived_at
    )


def collect_readiness_teacher(
    *,
    train_seed: int = 17,
    visual_observation: bool = False,
    observed_grasp: bool = False,
    goal_consistent: bool = False,
) -> TaskTeacherResult:
    """Collect one distinct geometry-readiness TRAIN teacher, with no retry."""
    if type(observed_grasp) is not bool:
        raise ValueError("observed_grasp must be an exact bool")
    if type(goal_consistent) is not bool or (goal_consistent and not observed_grasp):
        raise ValueError("goal_consistent must be an exact bool and requires observed_grasp")
    context = pick_cube_training_context(train_seed)
    options: dict[str, Any] = {"observed_grasp": True} if observed_grasp else {}
    if goal_consistent:
        options["goal_consistent"] = True
    return _collect_teacher(
        train_seed, context, "train", visual_observation, geometry_readiness=True, **options
    )


def collect_probe_teacher(*, role: str) -> ReservedProbeTaskTeacherResult:
    """One fixed reserved reset using the existing v2 teacher loop, without retry.

    Why: genuine probe inputs retain their original roles and reset allocation;
    TRAIN demonstrations cannot be relabeled into a fresh probe after observation.
    """
    allocation = pick_cube_probe_binding(role)
    return cast(
        ReservedProbeTaskTeacherResult,
        _collect_teacher(
            allocation.seed,
            allocation.context,
            allocation.role,
            False,
            geometry_readiness=True,
            observed_grasp=True,
            goal_consistent=True,
            reserved_probe=True,
        ),
    )


def _collect_teacher(
    seed: int,
    context: str,
    role: str,
    visual_observation: bool,
    labels_arrived_at: int | None = None,
    camera_context: PickCubeCameraContext | None = None,
    mass_context: PickCubeMassContext | None = None,
    lighting_context: PickCubeLightingContext | None = None,
    *,
    geometry_readiness: bool = False,
    observed_grasp: bool = False,
    goal_consistent: bool = False,
    reserved_probe: bool = False,
) -> TaskTeacherResult:
    if type(visual_observation) is not bool:
        raise ValueError("visual_observation must be an exact bool")
    if type(geometry_readiness) is not bool:
        raise ValueError("geometry_readiness must be an exact bool")
    if type(observed_grasp) is not bool:
        raise ValueError("observed_grasp must be an exact bool")
    if type(goal_consistent) is not bool or (goal_consistent and not observed_grasp):
        raise ValueError("goal_consistent must be an exact bool and requires observed_grasp")
    if type(reserved_probe) is not bool:
        raise ValueError("reserved_probe must be an exact bool")
    allocation = pick_cube_probe_binding(role) if reserved_probe else None
    if allocation is not None and (
        type(seed) is not int
        or seed != allocation.seed
        or context != allocation.context
        or not geometry_readiness
        or not observed_grasp
        or not goal_consistent
        or visual_observation
        or any(
            value is not None
            for value in (labels_arrived_at, camera_context, mass_context, lighting_context)
        )
    ):
        raise ValueError("reserved probe collection requires its fixed raw42 v2 binding")
    if observed_grasp and (not geometry_readiness or visual_observation):
        raise ValueError("observed-grasp teacher requires readiness nonvisual default TRAIN")
    if (
        geometry_readiness
        and not reserved_probe
        and (
            role != "train"
            or context != pick_cube_training_context(seed)
            or any(
                value is not None
                for value in (labels_arrived_at, camera_context, mass_context, lighting_context)
            )
        )
    ):
        raise ValueError("readiness teacher requires a default TRAIN context without interventions")
    if lighting_context is not None and (
        camera_context is not None or mass_context is not None or labels_arrived_at is not None
    ):
        raise ValueError("lighting teacher requires visual TRAIN without mass/camera/guard labels")
    lighting = require_lighting_binding(
        lighting_context,
        context=context,
        seed=seed,
        role=role,
        visual_observation=visual_observation,
    )
    mass = require_mass_binding(mass_context, context=context, seed=seed, role=role)
    if mass is not None and (camera_context is not None or labels_arrived_at is not None):
        raise ValueError("mass teacher requires TRAIN without camera or guard labels")
    mass_binding = _MassTeacherBinding(mass) if mass is not None else None
    lighting_binding = _LightingTeacherBinding(lighting) if lighting is not None else None
    binding = mass_binding if mass_binding is not None else lighting_binding
    binding_name = (
        "lighting"
        if lighting_binding is not None
        else "readiness"
        if geometry_readiness
        else "mass"
    )
    port = None
    episode = None
    teacher: PickCubeTeacher | ReadinessPickCubeTeacher | ObservedGraspPickCubeTeacher = (
        GoalConsistentObservedGraspPickCubeTeacher()
        if goal_consistent
        else ObservedGraspPickCubeTeacher()
        if observed_grasp
        else ReadinessPickCubeTeacher()
        if geometry_readiness
        else PickCubeTeacher()
    )
    geometry_attempts = geometry_returns = 0
    observed_success = final_success = False
    stages = []
    error = trace = None
    observation_recipe = visual_work = None
    work = None
    observed_binding = None
    feedback: list[bool] = []
    observed_decisions_returned = 0
    try:
        _LOG.info("teacher_phase=create state=attempt")
        visual_options: dict[str, Any] = {}
        if visual_observation:
            visual_options["visual_observation"] = True
        if camera_context is not None:
            visual_options["camera_context"] = camera_context
        if mass is not None:
            visual_options["mass_context"] = validate_pick_cube_mass_context(mass)
        if lighting is not None:
            visual_options["lighting_context"] = validate_pick_cube_lighting_context(lighting)
        port = ManiSkillPickCubeEpisode(
            allocation.episode_id
            if allocation is not None
            else f"pick-cube-goal-consistent-observed-grasp-teacher-{seed}"
            if goal_consistent
            else f"pick-cube-observed-grasp-teacher-{seed}"
            if observed_grasp
            else f"pick-cube-readiness-teacher-{seed}"
            if geometry_readiness
            else f"pick-cube-guard-teacher-{seed}"
            if role == "inner_guard"
            else f"{context}-teacher"
            if camera_context is not None or mass is not None or lighting is not None
            else f"pick-cube-task-teacher-{seed}",
            context,
            role=role,
            seed=seed,
            **visual_options,
        )
        if observed_grasp:
            observed_binding = _ObservedTeacherBinding(
                port,
                teacher,
                seed,
                context,
                goal_consistent=goal_consistent,
                probe_role=allocation.role if allocation is not None else None,
            )
        _LOG.info("teacher_phase=create state=returned")
        for _ in range(HORIZON):
            _check_camera_port(port, camera_context)
            if binding is not None:
                binding.observe(port)
            current_grasp = observed_binding.observe(port) if observed_binding is not None else None
            geometry_attempts += 1
            _LOG.info("teacher_phase=geometry attempt=%s", geometry_attempts)
            if observed_binding is not None:
                observed_binding.native(port)
            geometry = _geometry(port)
            geometry_returns += 1
            _LOG.info("teacher_phase=geometry returned=%s", geometry_returns)
            if binding is not None:
                binding.observe(port)
            if observed_binding is not None:
                observed_binding.observe(port)
                decision = cast(ObservedGraspPickCubeTeacher, teacher).decide(
                    geometry, grasped=cast(bool, current_grasp)
                )
                observed_decisions_returned += 1
                stage, command = observed_binding.decision(port, decision)
            elif type(teacher) is ReadinessPickCubeTeacher:
                decision = teacher.decide(geometry)
                if type(decision) is not ReadinessDecision:
                    raise ValueError("readiness teacher returned an unexpected decision")
                stage, command = decision.stage, decision.action
            elif isinstance(teacher, PickCubeTeacher):
                stage = teacher.stage
                command = teacher.action(geometry)
            else:
                raise ValueError("unexpected task teacher owner")
            stages.append(stage)
            _check_camera_port(port, camera_context)
            if binding is not None:
                binding.observe(port)
            _LOG.info("teacher_phase=control stage=%s attempt=%s", stage, teacher.steps)
            if observed_binding is not None:
                observed_binding.observe(port)
            record = port.step(command)
            if observed_binding is not None:
                feedback.append(cast(bool, current_grasp))
                observed_binding.control(port, record, command)
            _check_camera_port(port, camera_context)
            if binding is not None:
                binding.observe(port)
            _LOG.info("teacher_phase=control returned=%s", record.step)
            if record.reward not in (0.0, 1.0):
                raise ValueError("task teacher requires the pinned sparse0/1 reward")
            observed_success |= bool(record.reward)
            final_success = bool(record.reward)
            if record.terminated or record.truncated:
                break
        if binding is not None:
            binding.observe(port)
        if observed_binding is not None:
            observed_binding.observe(port)
        episode = port.episode()
        if observed_binding is not None:
            observed_binding.episode(port, episode)
        if binding is not None:
            binding.observe(port)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        trace = traceback.format_exc()
        if port is None and lighting_binding is not None:
            try:
                lighting_binding.preserve_failure(failure)
            except Exception as metadata_failure:
                error += f"; lighting failure metadata {type(metadata_failure).__name__}: {metadata_failure}"
    finally:
        if port is not None:
            if observed_binding is not None:
                try:
                    # Cleanup cannot erase a paid failed-step charge before it is retained.
                    observed_binding.remember_work(port.work)
                except Exception as failure:
                    error = error or f"readiness work {type(failure).__name__}: {failure}"
                    trace = trace or traceback.format_exc()
            if binding is not None:
                try:
                    binding.observe(port)
                except Exception as failure:
                    error = error or f"{binding_name} metadata {type(failure).__name__}: {failure}"
                    trace = trace or traceback.format_exc()
            try:
                port.close()
            except Exception as failure:
                error = error or f"cleanup {type(failure).__name__}: {failure}"
                trace = trace or traceback.format_exc()
            if binding is not None or geometry_readiness:
                try:
                    work = _copy_mass_teacher_work(port.work)
                    if observed_binding is not None:
                        observed_binding.remember_work(work)
                    if binding is not None:
                        binding.observe(port)
                except Exception as failure:
                    error = error or f"{binding_name} work {type(failure).__name__}: {failure}"
                    trace = trace or traceback.format_exc()
            try:
                recipe_metadata, visual_metadata = port.observation_recipe, port.visual_work
                if lighting_binding is not None:
                    if recipe_metadata is None:
                        raise ValueError("lighting teacher lost its base RGB recipe")
                    observation_recipe = validate_pick_cube_rgb_recipe(recipe_metadata)
                    visual_work = _copy_lighting_visual_work(visual_metadata)
                else:
                    observation_recipe, visual_work = recipe_metadata, visual_metadata
                if visual_observation:
                    if observation_recipe is None:
                        raise ValueError("visual teacher lost its observation recipe")
                    recipe = validate_camera_recipe(observation_recipe, camera_context)
                    if binding is not None:
                        observation_recipe = recipe
                    _check_camera_port(port, camera_context)
                elif mass_binding is not None and observation_recipe is not None:
                    raise ValueError("mass state teacher unexpectedly acquired an RGB recipe")
                if binding is not None:
                    binding.observe(port)
            except Exception as failure:
                error = error or f"visual metadata {type(failure).__name__}: {failure}"
                trace = trace or traceback.format_exc()
            if binding is not None:
                try:
                    binding.observe(port)
                except Exception as failure:
                    error = error or f"{binding_name} metadata {type(failure).__name__}: {failure}"
                    trace = trace or traceback.format_exc()
            if lighting_binding is not None:
                try:
                    if (
                        lighting_binding.original is None
                        or lighting_binding.work is None
                        or visual_work is None
                    ):
                        raise ValueError("lighting teacher lacks final observed receipts")
                    ManiSkillPickCubeEpisode.check_lighting_metadata(
                        port,
                        lighting_binding.context,
                        lighting_binding.original,
                        lighting_binding.work,
                        observed_visual_work=visual_work,
                    )
                except Exception as failure:
                    error = error or f"lighting final metadata {type(failure).__name__}: {failure}"
                    trace = trace or traceback.format_exc()
            if observed_binding is not None:
                try:
                    observed_binding.observe(port, closed=True)
                except Exception as failure:
                    error = (
                        error
                        or f"observed-grasp final metadata {type(failure).__name__}: {failure}"
                    )
                    trace = trace or traceback.format_exc()
    result_type = (
        ReservedProbeTaskTeacherResult
        if reserved_probe
        else GoalConsistentObservedGraspTaskTeacherResult
        if goal_consistent
        else ObservedGraspTaskTeacherResult
        if observed_grasp
        else TaskTeacherResult
    )
    selected_options: dict[str, Any] = {"observed_grasp": tuple(feedback)} if observed_grasp else {}
    if allocation is not None:
        selected_options["probe_seed"] = allocation.seed
    return result_type(
        episode,
        observed_success,
        final_success,
        1,
        geometry_attempts,
        geometry_returns,
        observed_decisions_returned if observed_grasp else teacher.steps,
        tuple(stages),
        observed_binding.work
        if observed_binding is not None
        else work
        if binding is not None or geometry_readiness
        else port.work
        if port
        else None,
        error,
        trace,
        observation_recipe,
        visual_work,
        seed if role == "inner_guard" else None,
        labels_arrived_at,
        camera_context,
        mass,
        mass_binding.readback if mass_binding is not None else None,
        mass_binding.work if mass_binding is not None else None,
        lighting,
        lighting_binding.readback if lighting_binding is not None else None,
        lighting_binding.work if lighting_binding is not None else None,
        GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
        if goal_consistent
        else OBSERVED_GRASP_DEFINITION
        if observed_grasp
        else READINESS_DEFINITION
        if geometry_readiness
        else None,
        **selected_options,
    )


def _check_camera_port(port, camera_context: PickCubeCameraContext | None) -> None:
    if camera_context is not None:
        expected = validate_pick_cube_camera_context(camera_context)
        if validate_pick_cube_camera_context(port.camera_context) != expected:
            raise ValueError("declared teacher camera context changed")
        validate_camera_recipe(port.observation_recipe, expected)
