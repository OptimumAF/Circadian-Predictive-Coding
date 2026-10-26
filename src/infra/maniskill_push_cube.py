"""One explicit PushCube TRAIN episode with owned physical task-space state.

The SDK boundary reads public positions and the same Panda grasp query used by
PickCube, then appends actual command history. No teacher, learner, role
conversion, reset retry, policy fitting, file IO or deployment occurs here.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from typing import Any, cast

import numpy as np

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.cube_taskspace_state import (
    CUBE_TASKSPACE_DEFINITION,
    CubeTaskSpaceState,
    copy_cube_taskspace_state,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.infra.maniskill_continuous import _array, _scalar, _require_cpu_kinematics

TASK_ID = "PushCube-v1"
CONTROL_MODE = "pd_ee_delta_pos"
HORIZON = 50


@dataclass(frozen=True)
class PushCubeWork:
    construction_attempts: int
    construction_returns: int
    reset_attempts: int
    reset_returns: int
    step_attempts: int
    completed_steps: int
    completed_control_physics_steps: int
    possible_failed_physics_steps: int
    state_encoding_attempts: int
    state_encoding_returns: int
    grasp_query_attempts: int
    grasp_query_returns: int
    source_contact_force_query_attempts_upper_bound: int
    source_contact_force_query_returns: int
    close_attempts: int
    close_returns: int


def _make_environment() -> Any:
    """Check optional dependencies before the one real factory invocation."""
    for package, expected in (("mani_skill", "3.0.1"), ("sapien", "3.0.3")):
        if version(package) != expected:
            raise ValueError(f"PushCube requires {package}=={expected}")
    _require_cpu_kinematics()
    import mani_skill.envs  # noqa: F401 -- register pinned official tasks lazily
    from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
    from mani_skill.utils.registration import REGISTERED_ENVS, TimeLimitWrapper

    registered = REGISTERED_ENVS[TASK_ID]
    if (
        registered.cls is not PushCubeEnv
        or registered.max_episode_steps != HORIZON
        or registered.asset_download_ids
    ):
        raise ValueError("unexpected PushCube registration or external asset requirement")
    env = registered.make(
        robot_uids="panda",
        control_mode=CONTROL_MODE,
        num_envs=1,
        sim_backend="physx_cpu",
        obs_mode="state_dict",
        render_backend="none",
        render_mode=None,
        reward_mode="sparse",
    )
    try:
        return TimeLimitWrapper(env, HORIZON)
    except BaseException:
        env.close()
        raise


def _vector(value: Any, width: int) -> tuple[float, ...]:
    array = _array(value)
    if (
        array.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or array.shape != (1, width)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"PushCube public state requires one finite CPU row of width {width}")
    return tuple(float(item) for item in array[0])


def _seal(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _identity_root_quaternion(quaternion: tuple[float, ...]) -> bool:
    # Why: native float32 qw can be one ULP off unit magnitude. Exactly zero
    # vector components still require identity axes; allow no angular tolerance.
    norm_error = abs(abs(quaternion[0]) - 1.0)
    return quaternion[1:] == (0.0, 0.0, 0.0) and norm_error <= 2 * float(np.finfo(np.float32).eps)


class ManiSkillPushCubeEpisode:
    """One single-CPU TRAIN reset; failed paid work never grants another control."""

    _COUNTERS = (
        "_construction_attempts",
        "_construction_returns",
        "_reset_attempts",
        "_reset_returns",
        "_attempts",
        "_completed",
        "_encoding_attempts",
        "_encoding_returns",
        "_query_attempts",
        "_query_returns",
        "_close_attempts",
        "_close_returns",
    )

    def __init__(self, episode_id: str, *, seed: int = 17, role: str = "train") -> None:
        if (
            type(seed) is not int
            or seed not in (17, 19, 20)
            or type(role) is not str
            or role != "train"
        ):
            raise ValueError("PushCube admits only explicit TRAIN seeds17/19/20")
        context = f"push-cube-train-{seed}"
        spec = cube_taskspace_spec()
        # Domain validation precedes optional imports, factory, reset and queries.
        dummy = ContinuousTransition(0, (0.0,) * 12, (0.0,) * 4, 0.0, (0.0,) * 12, False, True)
        ContinuousEpisode(episode_id, context, role, spec, (dummy,))
        self._episode_id, self._seed, self._context, self._role = episode_id, seed, context, role
        self._namespace = (episode_id, seed, context, role)
        self._spec = spec
        self._history = ActionHistoryState()
        self._records: list[ContinuousTransition] = []
        self._record_owner = self._records
        self._attempted_actions: list[tuple[float, ...]] = []
        self._attempted_action_owner = self._attempted_actions
        self._attempted_action_seal = _seal(self._attempted_actions)
        self._last_returned_action: tuple[float, ...] | None = None
        self._returned_actions: list[tuple[float, ...]] = []
        self._returned_action_owner = self._returned_actions
        self._returned_action_seal = _seal(self._returned_actions)
        self._record_seals: list[tuple[ContinuousTransition, str]] = []
        self._history_seal = _seal(asdict(self._history))
        self._failed = self._closed = False
        self._construction_attempts = self._construction_returns = 0
        self._reset_attempts = self._reset_returns = 0
        self._attempts = self._completed = 0
        self._encoding_attempts = self._encoding_returns = 0
        self._query_attempts = self._query_returns = 0
        self._close_attempts = self._close_returns = 0
        self._paid_counts = tuple(getattr(self, name) for name in self._COUNTERS)
        self._physics_ratio = 0
        self._env: Any = None
        self._original_env: Any = None
        try:
            self._increment_counter("_construction_attempts")
            self._env = _make_environment()
            self._original_env = self._env
            self._increment_counter("_construction_returns", native_return=True)
            self._bind()
            self._check()
            self._increment_counter("_reset_attempts")
            result = self._env.reset(seed=seed)
            self._increment_counter("_reset_returns", native_return=True)
            self._check()
            if type(result) is not tuple or len(result) != 2:
                raise ValueError("PushCube reset must return observation and info")
            self._state = self._encode(result[0])
            self._observation = cube_taskspace_observation(self._state, self._history)
            self._increment_counter("_encoding_returns")
            self._metadata = self._describe()
            self._metadata_seal = _seal(self._metadata)
            self._state_seal = _seal(asdict(self._state))
            self._check()
        except BaseException as error:
            self._failed = True
            try:
                self.close()
            except BaseException as cleanup:
                try:
                    error.add_note(f"PushCube cleanup failed: {type(cleanup).__name__}: {cleanup}")
                except AttributeError:
                    pass
            self._attach_work(error)
            raise

    def _bind(self) -> None:
        base = self._env.unwrapped
        agent: Any = base.agent
        if agent is None:
            raise ValueError("PushCube requires the native Panda agent")
        controller = agent.controller
        arm = controller.controllers["arm"]
        robot = agent.robot
        query = agent.is_grasping
        if (
            not callable(query)
            or getattr(query, "__self__", None) is not agent
            or getattr(query, "__func__", None) is None
        ):
            raise ValueError("PushCube requires the bound native Panda grasp query")
        self._grasp_query = query
        self._owners: tuple[Any, ...] = (
            self._env,
            base,
            agent,
            robot,
            base.obj,
            cast(Any, agent).tcp,
            base.goal_region,
            controller,
            controller.controllers,
            arm,
            arm.config,
            controller.controllers["gripper"],
            robot.root,
        )
        sim, control = base.sim_freq, base.control_freq
        if (
            type(sim) is not int
            or type(control) is not int
            or control <= 0
            or sim % control
            or not 1 <= sim // control <= 100
        ):
            raise ValueError("PushCube requires an integral positive physics/control ratio")
        self._physics_ratio = sim // control
        self._frequencies = (sim, control)

    def _identity(self) -> None:
        (
            env,
            base,
            agent,
            robot,
            obj,
            tcp,
            goal,
            controller,
            controllers,
            arm,
            config,
            gripper,
            root,
        ) = self._owners
        query = agent.is_grasping
        if (
            self._env is not env
            or env.unwrapped is not base
            or base.agent is not agent
            or agent.robot is not robot
            or base.obj is not obj
            or agent.tcp is not tcp
            or base.goal_region is not goal
            or agent.controller is not controller
            or controller.controllers is not controllers
            or controllers.get("arm") is not arm
            or controllers.get("gripper") is not gripper
            or arm.config is not config
            or robot.root is not root
            or arm.root_link is not root
            or arm.articulation is not robot
            or getattr(query, "__self__", None) is not agent
            or getattr(query, "__func__", None) is not self._grasp_query.__func__
        ):
            raise ValueError("PushCube native owner or controller was replaced")

    def _counter_mismatch(self) -> bool:
        return any(
            type(getattr(self, name, None)) is not int or getattr(self, name, None) != paid
            for name, paid in zip(self._COUNTERS, self._paid_counts, strict=True)
        )

    def _preserve_counter_mismatch(self) -> None:
        if not hasattr(self, "_counter_corruption"):
            self._counter_corruption = {
                "observed": {
                    name: value
                    if type(value := getattr(self, name, None)) is int
                    else type(value).__name__
                    for name in self._COUNTERS
                },
                "preceding_paid": dict(zip(self._COUNTERS, self._paid_counts, strict=True)),
            }

    def _increment_counter(self, field: str, *, native_return=False, cleanup=False) -> None:
        # A native return is paid even if its callback corrupted working counters.
        # Keep the last known scalar charges separately; never bless the mutation.
        changed = self._counter_mismatch()
        if changed:
            self._preserve_counter_mismatch()
            if not native_return and not cleanup:
                raise ValueError("PushCube original paid counters changed")
        index = self._COUNTERS.index(field)
        paid = list(self._paid_counts)
        paid[index] += 1
        self._paid_counts = tuple(paid)
        if not changed:
            setattr(self, field, paid[index])
        if changed and not cleanup:
            raise ValueError("PushCube original paid counters changed after native return")

    def _prefix(self) -> None:
        if self._counter_mismatch():
            self._preserve_counter_mismatch()
            raise ValueError("PushCube original paid counters changed")
        if (
            any(
                type(value) is not expected
                for value, expected in (
                    (self._episode_id, str),
                    (self._seed, int),
                    (self._context, str),
                    (self._role, str),
                )
            )
            or self._namespace != (self._episode_id, self._seed, self._context, self._role)
            or self._spec != cube_taskspace_spec()
            or self._records is not self._record_owner
            or len(self._records) != len(self._record_seals)
            or self._attempted_actions is not self._attempted_action_owner
            or _seal(self._attempted_actions) != self._attempted_action_seal
            or len(self._attempted_actions) != self._attempts
            or not self._completed <= self._attempts <= self._completed + 1
            or self._returned_actions is not self._returned_action_owner
            or _seal(self._returned_actions) != self._returned_action_seal
            or len(self._returned_actions) != self._completed
            or not len(self._records) <= self._completed <= len(self._records) + 1
            or _seal(asdict(self._history)) != self._history_seal
        ):
            raise ValueError("PushCube declared binding or executed history changed")
        for index, (record, (owner, payload)) in enumerate(
            zip(self._records, self._record_seals, strict=True)
        ):
            if (
                record is not owner
                or _seal(asdict(record)) != payload
                or record.action != self._returned_actions[index]
            ):
                raise ValueError("PushCube preceding paid record changed")
        if hasattr(self, "_state_seal") and (
            _seal(asdict(self._state)) != self._state_seal
            or self._observation != cube_taskspace_observation(self._state, self._history)
            or _seal(self._metadata) != self._metadata_seal
        ):
            raise ValueError("PushCube current state or declared metadata changed")

    def _check(self) -> None:
        self._identity()
        self._prefix()
        _, base, agent, robot, _, _, _, _, _, arm, config, _, root = self._owners
        space = base.single_action_space
        pose = _vector(robot.pose.raw_pose, 7)
        root_pose = _vector(root.pose.raw_pose, 7)
        if (
            base.num_envs != 1
            or base.obs_mode != "state_dict"
            or base.control_mode != CONTROL_MODE
            or agent.uid != "panda"
            or base.device.type != "cpu"
            or space.shape != (4,)
            or type(base.reconfiguration_freq) is not int
            or base.reconfiguration_freq != 0
            or not np.array_equal(space.low, np.full(4, -1.0))
            or not np.array_equal(space.high, np.full(4, 1.0))
            or (base.sim_freq, base.control_freq) != self._frequencies
            or type(self._physics_ratio) is not int
            or self._physics_ratio != self._frequencies[0] // self._frequencies[1]
            or not _identity_root_quaternion(pose[3:])
            or root_pose != pose
            or config.frame != "root_translation"
            or config.root_link_name is not None
            or config.use_delta is not True
            or config.use_target is not False
            or config.normalize_action is not True
            or arm._normalize_action is not True
            or not np.array_equal(np.broadcast_to(config.pos_lower, (3,)), np.full(3, -0.1))
            or not np.array_equal(np.broadcast_to(config.pos_upper, (3,)), np.full(3, 0.1))
        ):
            # Why: a live SDK refusal needs bounded leaf values to diagnose its
            # cause without guessing, weakening admission or recollecting controls.
            readback = dict(
                num_envs=repr(base.num_envs),
                obs_mode=repr(base.obs_mode),
                control_mode=repr(base.control_mode),
                agent_uid=repr(agent.uid),
                device_type=repr(base.device.type),
                action_shape=repr(space.shape),
                action_low=repr(space.low)[:160],
                action_high=repr(space.high)[:160],
                reconfiguration_freq=repr(base.reconfiguration_freq),
                reconfiguration_type=type(base.reconfiguration_freq).__name__,
                frequencies=repr((base.sim_freq, base.control_freq)),
                physics_ratio=repr(self._physics_ratio),
                robot_pose=pose,
                root_pose=root_pose,
                frame=repr(config.frame),
                root_link_name=repr(config.root_link_name),
                use_delta=repr(config.use_delta),
                use_target=repr(config.use_target),
                normalize_action=repr(config.normalize_action),
                arm_normalize_action=repr(arm._normalize_action),
                pos_lower=repr(config.pos_lower)[:160],
                pos_upper=repr(config.pos_upper)[:160],
            )
            readback = {
                key: value[:160] if type(value) is str else value for key, value in readback.items()
            }
            raise ValueError(
                "PushCube CPU/controller/world-root frame contract changed; observed="
                + _seal(readback)
            )
        self._identity()
        self._prefix()

    def _encode(self, observation: Any) -> CubeTaskSpaceState:
        self._increment_counter("_encoding_attempts")
        self._check()
        if type(observation) is not dict or type(observation.get("extra")) is not dict:
            raise ValueError("PushCube requires structured public state_dict extra")
        extra = observation["extra"]
        if any(key not in extra for key in ("tcp_pose", "obj_pose", "goal_pos")):
            raise ValueError("PushCube structured state is missing public pose fields")
        tcp, obj, goal = (
            _vector(extra[key], width)
            for key, width in (("tcp_pose", 7), ("obj_pose", 7), ("goal_pos", 3))
        )
        _, _, agent, _, native_obj, native_tcp, native_goal, *_ = self._owners
        positions = (
            _vector(native_tcp.pose.raw_pose, 7),
            _vector(native_obj.pose.raw_pose, 7),
            _vector(native_goal.pose.p, 3),
        )
        self._check()
        if positions != (tcp, obj, goal):
            raise ValueError("PushCube observation differs from its current native pose owners")
        self._increment_counter("_query_attempts")
        signal = self._grasp_query(native_obj, min_force=0.5, max_angle=85)
        self._increment_counter("_query_returns", native_return=True)
        self._check()
        # An opaque grasp query may change public poses while returning a valid flag.
        current = (
            _vector(native_tcp.pose.raw_pose, 7),
            _vector(native_obj.pose.raw_pose, 7),
            _vector(native_goal.pose.p, 3),
        )
        self._check()
        if current != positions:
            raise ValueError("PushCube public positions changed during the grasp query")
        state = CubeTaskSpaceState(bool(_scalar(signal, boolean=True)), tcp[:3], obj[:3], goal)
        return state

    def _describe(self) -> dict[str, Any]:
        return dict(
            task_id=TASK_ID,
            episode_id=self._episode_id,
            seed=self._seed,
            context=self._context,
            role=self._role,
            recipe=dict(
                definition=CUBE_TASKSPACE_DEFINITION,
                feature_dim=12,
                displacement_scale=0.1,
                displacement_frame="world",
                history="executed",
                closing_limit=6,
            ),
            controller=dict(
                mode=CONTROL_MODE, frame="root_translation", normalized_delta=True, no_target=True
            ),
            grasp_source=dict(
                method="panda.is_grasping",
                min_force=0.5,
                max_angle=85,
                source_force_queries_per_call=2,
                force_count_kind="source-derived",
            ),
            reward=dict(
                mode="sparse",
                source="info.success",
                rule="xy_distance_lt_0.1_and_object_z_lt_0.025",
            ),
            horizon=HORIZON,
            constructor_reset_physics_steps=None,
            control_physics_steps_per_control=self._physics_ratio,
        )

    def _attach_work(self, error: BaseException) -> None:
        try:
            if hasattr(self, "_counter_corruption"):
                error.push_cube_counter_corruption = deepcopy(self._counter_corruption)  # type: ignore[attr-defined]
            error.push_cube_work = self.work  # type: ignore[attr-defined]
            attempted = tuple(tuple(row) for row in json.loads(self._attempted_action_seal))
            returned = tuple(tuple(row) for row in json.loads(self._returned_action_seal))
            if self.work.completed_steps > len(returned) and self._last_returned_action is not None:
                returned += (self._last_returned_action,)
            error.push_cube_attempted_actions = attempted  # type: ignore[attr-defined]
            error.push_cube_returned_actions = returned  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    @property
    def work(self) -> PushCubeWork:
        paid = dict(zip(self._COUNTERS, self._paid_counts, strict=True))
        ratio = self._frequencies[0] // self._frequencies[1] if hasattr(self, "_frequencies") else 0
        return PushCubeWork(
            paid["_construction_attempts"],
            paid["_construction_returns"],
            paid["_reset_attempts"],
            paid["_reset_returns"],
            paid["_attempts"],
            paid["_completed"],
            paid["_completed"] * ratio,
            (paid["_attempts"] - paid["_completed"]) * ratio,
            paid["_encoding_attempts"],
            paid["_encoding_returns"],
            paid["_query_attempts"],
            paid["_query_returns"],
            2 * paid["_query_attempts"],
            2 * paid["_query_returns"],
            paid["_close_attempts"],
            paid["_close_returns"],
        )

    @property
    def metadata(self) -> dict[str, Any]:
        # Paid failure reporting retains the admitted declaration even if a
        # callback corrupted the mutable working metadata (checks still refuse it).
        return json.loads(self._metadata_seal)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def role(self) -> str:
        return self._role

    @property
    def context(self) -> str:
        return self._context

    @property
    def spec(self) -> ContinuousActionSpec:
        self._available()
        return cube_taskspace_spec()

    def _available(self) -> None:
        if self._failed or self._closed:
            raise ValueError("PushCube episode is failed or closed")
        try:
            self._check()
            _, _, _, _, obj, tcp, goal, *_ = self._owners
            current = (
                _vector(tcp.pose.raw_pose, 7)[:3],
                _vector(obj.pose.raw_pose, 7)[:3],
                _vector(goal.pose.p, 3),
            )
            self._check()
            if current != (
                self._state.tcp_position,
                self._state.object_position,
                self._state.goal_position,
            ):
                raise ValueError(
                    "PushCube current public positions changed without a native control"
                )
        except BaseException as error:
            self._failed = True
            self._attach_work(error)
            raise

    @property
    def state(self) -> CubeTaskSpaceState:
        self._available()
        return copy_cube_taskspace_state(self._state)

    @property
    def observation(self) -> tuple[float, ...]:
        self._available()
        return tuple(self._observation)

    def step(self, action: tuple[float, ...]) -> ContinuousTransition:
        self._available()
        if len(self._records) >= HORIZON or (
            self._records and (self._records[-1].terminated or self._records[-1].truncated)
        ):
            raise ValueError("PushCube episode is already complete")
        try:
            trial = ContinuousTransition(
                len(self._records), self._observation, action, 0.0, self._observation, False, True
            )
            ContinuousEpisode(
                self._episode_id,
                self._context,
                self._role,
                self._spec,
                (
                    ContinuousTransition(
                        0, trial.observation, trial.action, 0.0, trial.next_observation, False, True
                    ),
                ),
            )
            command = np.array([trial.action], dtype=np.float32)
            sent = tuple(float(value) for value in command[0])
            self._check()
            self._increment_counter("_attempts")
            self._attempted_actions.append(sent)
            self._attempted_action_seal = _seal(self._attempted_actions)
            output = self._env.step(command)
            self._last_returned_action = sent
            self._increment_counter("_completed", native_return=True)
            if (
                self._returned_actions is not self._returned_action_owner
                or _seal(self._returned_actions) != self._returned_action_seal
            ):
                raise ValueError("PushCube preceding returned command receipt changed")
            self._returned_actions.append(sent)
            self._returned_action_seal = _seal(self._returned_actions)
            self._check()
            if type(output) is not tuple or len(output) != 5:
                raise ValueError("PushCube step must return all five native outputs")
            observation, reward, terminated, truncated, info = output
            if type(info) is not dict or "success" not in info:
                raise ValueError("PushCube sparse reward requires native info.success")
            score = float(_scalar(reward))
            success = bool(_scalar(info["success"], boolean=True))
            end, limit = (
                bool(_scalar(terminated, boolean=True)),
                bool(_scalar(truncated, boolean=True)),
            )
            if score not in (0.0, 1.0) or score != float(success):
                raise ValueError("PushCube reward differs from native sparse task success")
            # Advance only after the actual native control has returned, never from targets.
            history = advance_action_history(self._history, sent)
            state = self._encode(observation)
            after = cube_taskspace_observation(state, history)
            self._increment_counter("_encoding_returns")
            if len(self._records) == HORIZON - 1 and not (end or limit):
                raise ValueError("PushCube50-step native horizon flag is missing")
            record = ContinuousTransition(
                len(self._records), self._observation, sent, score, after, end, limit
            )
            self._check()
            self._records.append(record)
            self._record_seals.append((record, _seal(asdict(record))))
            self._history, self._state, self._observation = history, state, after
            self._history_seal, self._state_seal = _seal(asdict(history)), _seal(asdict(state))
            self._check()
            return record
        except BaseException as error:
            self._failed = True
            self._attach_work(error)
            raise

    def episode(self) -> ContinuousEpisode:
        self._available()
        if not self._records or not (self._records[-1].terminated or self._records[-1].truncated):
            raise ValueError("PushCube exports only complete native episodes")
        try:
            result = ContinuousEpisode(
                self._episode_id, self._context, self._role, self._spec, tuple(self._records)
            )
            self._check()
            return result
        except BaseException as error:
            self._failed = True
            self._attach_work(error)
            raise

    def close(self) -> None:
        if self._closed:
            return
        already_failed = self._failed
        newly_corrupted = self._counter_mismatch() and not already_failed
        self._closed = True
        if self._original_env is not None:
            # Cleanup belongs to the original paid factory result, even when an
            # opaque callback replaced the working environment reference.
            self._increment_counter("_close_attempts", cleanup=True)
            try:
                self._original_env.close()
                self._increment_counter("_close_returns", native_return=True, cleanup=True)
                if not already_failed and (newly_corrupted or self._counter_mismatch()):
                    raise ValueError("PushCube original paid counters changed before cleanup")
                # Native owners may be destroyed by close; closed=True already
                # prevents subsequent controls/exports. Do not read them again.
            except BaseException as error:
                self._failed = True
                self._attach_work(error)
                raise

    def __enter__(self) -> "ManiSkillPushCubeEpisode":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
