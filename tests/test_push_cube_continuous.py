"""Physical PushCube features and fake native IO, never task-learning evidence.

Only deterministic arrays and scripted commands run here: no learner, SDK,
physics, teacher or RNG. The module ledger separates fake call attempts/returns.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.core.cube_taskspace_state import (
    CubeTaskSpaceState,
    copy_cube_taskspace_state,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.pick_cube_action_history import ActionHistoryState, action_history_observation
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_push_cube as port


def physical_state():
    return CubeTaskSpaceState(False, (0.0, 0.0, 0.0), (0.025, -0.05, 0.075), (0.125, 0.0, 0.025))


def test_should_use_owned_world_displacements_and_executed_history():
    source = physical_state()
    history = ActionHistoryState((0.125, -0.25, 0.5, -1.0), 2)
    actual = cube_taskspace_observation(source, history)
    assert actual[:7] == pytest.approx((0.0, 0.25, -0.5, 0.75, 1.0, 0.5, -0.5))
    assert actual[7:] == (*history.previous_action, 2 / 6)
    assert cube_taskspace_observation(source)[7:] == (0.0,) * 5
    translated = CubeTaskSpaceState(
        False, (1.0, 1.0, 1.0), (1.025, 0.95, 1.075), (1.125, 1.0, 1.025)
    )
    assert cube_taskspace_observation(translated, history) == pytest.approx(actual)
    owned = copy_cube_taskspace_state(source)
    assert owned == source and owned is not source
    object.__setattr__(source, "object_position", (999.0,) * 3)
    assert owned == physical_state()
    spec = cube_taskspace_spec()
    assert spec.observation_dim == 12
    assert spec.action_units == (
        "normalized_root_dx",
        "normalized_root_dy",
        "normalized_root_dz",
        "normalized_gripper",
    )
    assert spec.lower_bounds == (-1.0,) * 4 and spec.upper_bounds == (1.0,) * 4


def test_should_match_pick_taskspace_on_exact_synthetic_displacements():
    source = CubeTaskSpaceState(True, (0.0,) * 3, (0.25, -0.5, 0.125), (0.75, 0.0, 0.25))
    raw = [0.0] * 42
    raw[18] = 1.0
    raw[36:42] = [0.25, -0.5, 0.125, 0.5, 0.5, 0.125]
    history = ActionHistoryState((0.5, 0.0, -0.25, -1.0), 6)
    assert cube_taskspace_observation(source, history) == action_history_observation(
        tuple(raw), history, PickCubeTaskSpaceHistoryRecipe()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "flag_int",
        "flag_missing",
        "list",
        "short",
        "bool_coordinate",
        "numpy_coordinate",
        "nan",
        "overflow",
        "history",
    ],
)
def test_should_refuse_untyped_missing_or_nonfinite_physical_features(fault):
    source = physical_state()
    history: Any = ActionHistoryState()
    if fault == "flag_int":
        object.__setattr__(source, "is_grasped", 1)
    elif fault == "flag_missing":
        object.__delattr__(source, "is_grasped")
    elif fault == "list":
        object.__setattr__(source, "tcp_position", [0.0] * 3)
    elif fault == "short":
        object.__setattr__(source, "goal_position", (0.0,) * 2)
    elif fault == "bool_coordinate":
        object.__setattr__(source, "tcp_position", (True, 0.0, 0.0))
    elif fault == "numpy_coordinate":
        object.__setattr__(source, "tcp_position", (np.float32(0), 0.0, 0.0))
    elif fault == "nan":
        object.__setattr__(source, "goal_position", (float("nan"), 0.0, 0.0))
    elif fault == "overflow":
        object.__setattr__(source, "tcp_position", (-1e308, 0.0, 0.0))
        object.__setattr__(source, "object_position", (1e308, 0.0, 0.0))
    else:
        history = object()
    with pytest.raises((ValueError, AttributeError)):
        cube_taskspace_observation(source, history)


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        factory_attempts=0,
        factory_returns=0,
        reset_attempts=0,
        reset_returns=0,
        control_attempts=0,
        control_returns=0,
        grasp_attempts=0,
        grasp_returns=0,
        close_attempts=0,
        close_returns=0,
    )
    yield counts
    if os.environ.get("PUS_FIXTURE_LEDGER"):
        with Path(os.environ["PUS_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["factory_attempts"] <= 64 and counts["reset_attempts"] <= 64
    assert counts["control_attempts"] <= 256 and counts["grasp_attempts"] <= 512


class FakeAgent(SimpleNamespace):
    def is_grasping(self, obj, *, min_force=0.5, max_angle=85):
        assert min_force == 0.5 and max_angle == 85
        return self.environment.is_grasping(obj)


class FakePushEnvironment:
    """Scripted native-shaped arrays; no dynamics or task solver."""

    def __init__(self, counts, *, horizon=2, terminated=True, truncated=False, success=False):
        self.counts = counts
        self.unwrapped = self
        self.num_envs, self.obs_mode, self.control_mode = 1, "state_dict", "pd_ee_delta_pos"
        self.device = SimpleNamespace(type="cpu")
        self.sim_freq, self.control_freq = 100, 20
        self.reconfiguration_freq = 0
        self.spec = SimpleNamespace(id="PushCube-v1", max_episode_steps=50)
        self.single_action_space = SimpleNamespace(
            low=np.full(4, -1, np.float32), high=np.ones(4, np.float32), shape=(4,)
        )
        root = SimpleNamespace(
            pose=SimpleNamespace(raw_pose=np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
        )
        robot = SimpleNamespace(root=root, pose=root.pose)
        config = SimpleNamespace(
            frame="root_translation",
            use_delta=True,
            use_target=False,
            normalize_action=True,
            pos_lower=-0.1,
            pos_upper=0.1,
            root_link_name=None,
        )
        arm = SimpleNamespace(
            config=config, _normalize_action=True, root_link=root, articulation=robot
        )
        self.agent = FakeAgent(
            environment=self,
            uid="panda",
            robot=robot,
            tcp=SimpleNamespace(pose=SimpleNamespace(raw_pose=np.zeros((1, 7)))),
            controller=SimpleNamespace(controllers={"arm": arm, "gripper": SimpleNamespace()}),
        )
        self.obj = SimpleNamespace(pose=SimpleNamespace(raw_pose=np.zeros((1, 7))))
        self.goal_region = SimpleNamespace(pose=SimpleNamespace(p=np.zeros((1, 3))))
        self.horizon, self.terminated, self.truncated, self.success = (
            horizon,
            terminated,
            truncated,
            success,
        )
        self.commands: list[Any] = []
        self.reset_seeds: list[int] = []
        self.close_calls = 0
        self.step_error: Any = None
        self.grasp_error: Any = None
        self.grasp_override: Any = None
        self.packet_fault: str | None = None
        self.output_fault: str | None = None
        self.after_grasp: Any = None
        self.after_step: Any = None
        self.last_packet: Any = None

    def packet(self):
        index = len(self.commands)
        tcp = np.array([[index * 0.01, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]], np.float64)
        obj = np.array([[0.05, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]], np.float64)
        goal = np.array([[0.05 if self.success else 0.3, 0.0, 0.02]], np.float64)
        self.agent.tcp.pose.raw_pose, self.obj.pose.raw_pose, self.goal_region.pose.p = (
            tcp.copy(),
            obj.copy(),
            goal.copy(),
        )
        packet = {"extra": {"tcp_pose": tcp, "obj_pose": obj, "goal_pos": goal}}
        if self.packet_fault == "missing":
            del packet["extra"]["tcp_pose"]
        elif self.packet_fault == "shape":
            packet["extra"]["goal_pos"] = goal[0]
        elif self.packet_fault == "dtype":
            packet["extra"]["tcp_pose"] = tcp.astype(np.int64)
        elif self.packet_fault == "nonfinite":
            packet["extra"]["obj_pose"][0, 0] = np.nan
        self.last_packet = packet
        return packet

    def reset(self, *, seed):
        self.counts["reset_attempts"] += 1
        self.reset_seeds.append(seed)
        returned: tuple[Any, ...] = self.packet(), {}
        self.counts["reset_returns"] += 1
        return returned

    def is_grasping(self, obj):
        assert obj is self.obj
        self.counts["grasp_attempts"] += 1
        if self.grasp_error is not None:
            raise self.grasp_error
        returned = (
            np.array([bool(len(self.commands) % 2)], dtype=np.bool_)
            if self.grasp_override is None
            else self.grasp_override
        )
        if self.after_grasp is not None:
            self.after_grasp()
        self.counts["grasp_returns"] += 1
        return returned

    def step(self, action):
        self.counts["control_attempts"] += 1
        self.commands.append(action.copy())
        if self.step_error is not None:
            raise self.step_error
        done = len(self.commands) == self.horizon
        result = (
            self.packet(),
            np.array([float(self.success)], np.float32),
            np.array([done and self.terminated], bool),
            np.array([done and self.truncated], bool),
            {"success": np.array([self.success], bool)},
        )
        if self.output_fault == "reward":
            result = (result[0], np.array([0.5]), *result[2:])
        elif self.output_fault == "success":
            result[4]["success"] = np.array([not self.success], bool)
        elif self.output_fault == "flag":
            result = (*result[:2], np.array([1], np.int64), *result[3:])
        if self.after_step is not None:
            self.after_step()
        self.counts["control_returns"] += 1
        return result

    def close(self):
        self.counts["close_attempts"] += 1
        self.close_calls += 1
        self.counts["close_returns"] += 1


@pytest.fixture
def boundary(monkeypatch, ledger):
    environment = FakePushEnvironment(ledger)
    calls: list[bool] = []

    def factory():
        ledger["factory_attempts"] += 1
        calls.append(True)
        ledger["factory_returns"] += 1
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    return environment, calls


def open_episode(**options):
    return port.ManiSkillPushCubeEpisode("push-test", **options)


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("success", [False, True])
def test_should_capture_causal_native_commands_sparse_rewards_and_independent_endflags(
    boundary, terminated, truncated, success
):
    environment, calls = boundary
    environment.terminated, environment.truncated, environment.success = (
        terminated,
        truncated,
        success,
    )
    with open_episode() as session:
        initial = session.observation
        first = session.step((0.123456789, -0.25, 0.5, -1.0))
        second = session.step((0.0, 0.0, 0.0, 1.0))
        packet = session.episode()
        assert first.observation == initial and initial[7:] == (0.0,) * 5
        assert first.action == tuple(float(value) for value in environment.commands[0][0])
        assert first.action[0] != 0.123456789
        assert (
            environment.commands[0].shape == (1, 4) and environment.commands[0].dtype == np.float32
        )
        assert first.next_observation == second.observation
        assert first.next_observation[0] == 1.0 and second.next_observation[0] == 0.0
        assert first.next_observation[7:] == (*first.action, 1 / 6)
        assert second.next_observation[7:] == (*second.action, 0.0)
        assert first.reward == second.reward == float(success)
        assert not first.terminated and not first.truncated
        assert second.terminated is terminated and second.truncated is truncated
        assert (packet.episode_id, packet.context, packet.role) == (
            "push-test",
            "push-cube-train-17",
            "train",
        )
        assert packet.transitions == (first, second) and packet.spec == cube_taskspace_spec()
        work = session.work
        assert work.step_attempts == work.completed_steps == 2
        assert (
            work.completed_control_physics_steps == 10 and work.possible_failed_physics_steps == 0
        )
        assert work.state_encoding_attempts == work.state_encoding_returns == 3
        assert work.grasp_query_attempts == work.grasp_query_returns == 3
        assert (
            work.source_contact_force_query_attempts_upper_bound
            == work.source_contact_force_query_returns
            == 6
        )
    assert calls == [True] and environment.reset_seeds == [17] and environment.close_calls == 1
    session.close()
    assert environment.close_calls == 1


@pytest.mark.parametrize("seed", [17, 19, 20])
def test_should_derive_only_explicit_train_binding_and_detach_metadata(boundary, seed):
    environment, _ = boundary
    with open_episode(seed=seed) as session:
        assert (session.seed, session.context, session.role) == (
            seed,
            f"push-cube-train-{seed}",
            "train",
        )
        metadata = session.metadata
        assert metadata["task_id"] == "PushCube-v1" and metadata["seed"] == seed
        metadata["context"] = "spoof"
        assert session.metadata["context"] == f"push-cube-train-{seed}"
        assert session.work.reset_attempts == session.work.reset_returns == 1
    assert environment.reset_seeds == [seed]


@pytest.mark.parametrize(
    "options",
    [
        {"seed": True},
        {"seed": 18},
        {"seed": 23},
        {"seed": 24},
        {"seed": 31},
        {"seed": 37},
        {"role": "probe_train"},
        {"role": "probe_test"},
        {"role": "final_test"},
        {"role": True},
    ],
)
def test_should_refuse_reserved_or_ambiguous_authority_before_factory(boundary, options):
    _, calls = boundary
    with pytest.raises(ValueError):
        open_episode(**options)
    assert calls == []


@pytest.mark.parametrize(
    "action",
    [
        [0.0] * 4,
        (0.0,) * 3,
        (True, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0),
        (1.000000001, 0.0, 0.0, 0.0),
    ],
)
def test_should_reject_invalid_action_before_float32_conversion_or_native_control(boundary, action):
    environment, _ = boundary
    with open_episode() as session:
        with pytest.raises(ValueError):
            session.step(action)
        assert environment.commands == [] and session.work.step_attempts == 0


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "nonfinite"])
def test_should_close_and_preserve_returned_reset_cost_for_malformed_structured_state(
    boundary, fault
):
    environment, _ = boundary
    environment.packet_fault = fault
    with pytest.raises((ValueError, RuntimeError)):
        open_episode()
    assert environment.reset_seeds == [17] and environment.commands == []
    assert environment.close_calls == 1


@pytest.mark.parametrize("fault", ["missing", "integer", "shape"])
def test_should_query_real_grasp_signal_and_refuse_a_missing_or_untyped_return(boundary, fault):
    environment, _ = boundary
    if fault == "missing":
        environment.grasp_error = RuntimeError("sensor unavailable")
    elif fault == "integer":
        environment.grasp_override = np.array([1], np.int64)
    else:
        environment.grasp_override = np.array([[True]], bool)
    with pytest.raises((ValueError, RuntimeError)):
        open_episode()
    assert environment.commands == [] and environment.close_calls == 1


@pytest.mark.parametrize("fault", ["owner", "frame", "quaternion", "controller"])
def test_should_poison_changed_native_owner_or_frame_before_control(boundary, fault):
    environment, _ = boundary
    with open_episode() as session:
        if fault == "owner":
            environment.obj = SimpleNamespace(
                pose=SimpleNamespace(raw_pose=environment.obj.pose.raw_pose.copy())
            )
        elif fault == "frame":
            environment.agent.controller.controllers["arm"].config.frame = "body_translation"
        elif fault == "quaternion":
            environment.agent.robot.root.pose.raw_pose[0, 3:] = [0.0, 1.0, 0.0, 0.0]
        else:
            environment.agent.controller.controllers["arm"].config.use_target = True
        with pytest.raises((ValueError, RuntimeError)) as caught:
            session.step((0.0,) * 4)
        if fault != "owner":
            readback = json.loads(str(caught.value).split("observed=", 1)[1])
            assert readback["action_shape"] == "(4,)"
            assert readback["reconfiguration_freq"] == "0"
            assert len(readback["robot_pose"]) == len(readback["root_pose"]) == 7
        assert environment.commands == [] and session.work.step_attempts == 0
        with pytest.raises((ValueError, RuntimeError)):
            session.episode()


@pytest.mark.parametrize(
    "fault", ["native", "reward", "success", "flag", "grasp", "sensor_callback"]
)
def test_should_keep_paid_control_cost_and_fail_stop_after_untrusted_native_return(boundary, fault):
    environment, _ = boundary
    with open_episode() as session:
        if fault == "native":
            environment.step_error = RuntimeError("possible native physics")
        elif fault in ("reward", "success", "flag"):
            environment.output_fault = fault
        elif fault == "grasp":
            environment.grasp_error = RuntimeError("paid sensor failure")
        else:
            environment.after_grasp = lambda: setattr(environment.agent, "robot", SimpleNamespace())
        with pytest.raises((ValueError, RuntimeError)) as caught:
            session.step((0.0,) * 4)
        assert getattr(caught.value, "push_cube_attempted_actions") == ((0.0,) * 4,)
        assert getattr(caught.value, "push_cube_returned_actions") == (
            () if fault == "native" else ((0.0,) * 4,)
        )
        work = session.work
        assert work.step_attempts == 1
        assert work.completed_steps == (0 if fault == "native" else 1)
        assert work.completed_control_physics_steps == (0 if fault == "native" else 5)
        assert work.possible_failed_physics_steps == (5 if fault == "native" else 0)
        for operation in (session.episode, lambda: session.step((0.0,) * 4)):
            with pytest.raises((ValueError, RuntimeError)):
                operation()
        assert len(environment.commands) == 1


@pytest.mark.parametrize("truncate", [False, True])
def test_should_bound_trace_at_official_horizon_without_inventing_completion(boundary, truncate):
    environment, _ = boundary
    environment.horizon, environment.terminated, environment.truncated = 50, False, truncate
    with open_episode() as session:
        for _ in range(49):
            session.step((0.0,) * 4)
        if truncate:
            final = session.step((0.0,) * 4)
            assert final.truncated and not final.terminated
            assert len(session.episode().transitions) == 50
        else:
            with pytest.raises(ValueError):
                session.step((0.0,) * 4)
            with pytest.raises(ValueError):
                session.episode()
        with pytest.raises((ValueError, RuntimeError)):
            session.step((0.0,) * 4)
        assert len(environment.commands) == session.work.completed_steps == 50
        assert session.work.completed_control_physics_steps == 250


def test_should_own_state_and_receipts_without_extra_sensor_queries(boundary, ledger):
    environment, _ = boundary
    with open_episode() as session:
        initial, before, queries = session.state, session.work, ledger["grasp_attempts"]
        object.__setattr__(initial, "is_grasped", True)
        environment.last_packet["extra"]["tcp_pose"].fill(999.0)
        assert session.state.is_grasped is False
        assert session.observation[0] == 0.0
        assert ledger["grasp_attempts"] == queries
        with pytest.raises((ValueError, RuntimeError)):
            session.episode()
        session.step((0.0, 0.0, 0.0, -1.0))
        session.step((0.0, 0.0, 0.0, 1.0))
        assert before.step_attempts == 0 and session.work.completed_steps == 2
        assert session.episode().transitions[0].observation[7:] == (0.0,) * 5


@pytest.mark.parametrize("deleted", [False, True])
def test_should_refuse_counter_corruption_and_preserve_observed_paid_query_return(
    boundary, deleted
):
    environment, _ = boundary
    with open_episode() as session:
        environment.after_grasp = lambda: (
            delattr(session, "_query_attempts")
            if deleted
            else setattr(session, "_query_attempts", 999)
        )
        with pytest.raises(ValueError, match="paid counters") as caught:
            session.step((0.0,) * 4)
        work = session.work
        assert work.step_attempts == work.completed_steps == 1
        assert work.grasp_query_attempts == work.grasp_query_returns == 2
        assert work.source_contact_force_query_returns == 4
        assert getattr(caught.value, "push_cube_counter_corruption")["observed"][
            "_query_attempts"
        ] == ("NoneType" if deleted else 999)
        assert (
            not hasattr(session, "_query_attempts") if deleted else session._query_attempts == 999
        )
        with pytest.raises(ValueError):
            session.step((0.0,) * 4)
        assert len(environment.commands) == 1


@pytest.mark.parametrize("owner", [False, True])
def test_should_charge_returned_control_before_refusing_corrupted_receipt(boundary, owner):
    environment, _ = boundary
    with open_episode() as session:
        native_step = environment.step

        def changed_step(command):
            result = native_step(command)
            if owner:
                session._returned_actions = []
            else:
                session._returned_actions.append((1.0,) * 4)
            return result

        environment.step = changed_step
        with pytest.raises(ValueError) as caught:
            session.step((0.0,) * 4)
        assert session.work.completed_steps == 1
        assert getattr(caught.value, "push_cube_returned_actions") == ((0.0,) * 4,)
        assert len(environment.commands) == 1


@pytest.mark.parametrize("during", [False, True])
def test_should_close_original_environment_when_accounting_was_corrupted(boundary, during):
    environment, _ = boundary
    session = open_episode()
    if during:
        native_close = environment.close

        def changed_close():
            native_close()
            session._query_attempts = 999

        environment.close = changed_close
    else:
        session._query_attempts = 999
    with pytest.raises(ValueError, match="paid counters") as caught:
        session.close()
    assert environment.close_calls == 1
    assert session.work.close_attempts == session.work.close_returns == 1
    assert getattr(caught.value, "push_cube_work").grasp_query_attempts == 1
    session.close()
    assert environment.close_calls == 1


@pytest.mark.parametrize(
    "quaternion,accepted",
    [
        ((1.0 + 2**-23, 0.0, 0.0, 0.0), True),
        ((-1.0 - 2**-23, 0.0, 0.0, 0.0), True),
        ((1.0 + 2 * 2**-23, 0.0, 0.0, 0.0), True),
        ((1.0 - 2 * 2**-23, 0.0, 0.0, 0.0), True),
        ((1.0 + 3 * 2**-23, 0.0, 0.0, 0.0), False),
        ((0.0, 0.0, 0.0, 0.0), False),
        ((1.0, 2**-126, 0.0, 0.0), False),
        ((float("nan"), 0.0, 0.0, 0.0), False),
    ],
)
def test_should_allow_bounded_scalar_norm_error_but_no_nonzero_root_rotation(
    boundary, quaternion, accepted
):
    environment, _ = boundary
    environment.agent.robot.root.pose.raw_pose[0, 3:] = quaternion
    if accepted:
        with open_episode() as session:
            record = session.step((0.0, 0.0, 0.0, 1.0))
            assert record.action == (0.0, 0.0, 0.0, 1.0)
            assert session.work.completed_steps == 1
            assert session.work.grasp_query_returns == 2
        assert environment.reset_seeds == [17]
    else:
        with pytest.raises(ValueError):
            open_episode()
        assert environment.reset_seeds == [] and environment.commands == []
    assert environment.close_calls == 1
