"""Bounded TRAIN-only official Panda expert conversion into continuous records.

The joint-position reference is never used as four-axis training labels. Only
actual converted commands through the existing episode port can qualify. No
learner, final evaluation, state teleport, horizon extension or retry is added.
"""

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
import logging
import traceback
from typing import Any

import numpy as np

from src.core.continuous_demonstration import ContinuousEpisode
from src.infra import maniskill_continuous as boundary

_LOG = logging.getLogger(__name__)


class _EpisodeComplete(Exception):
    """Stop upstream loops that otherwise ignore a returned terminal flag."""


@dataclass(frozen=True)
class ReferenceWork:
    reset_attempts: int
    reset_returns: int
    collection_attempts: int
    collection_returns: int
    replay_attempts: int
    replay_returns: int
    completed_control_physics_steps: int
    possible_failed_physics_steps: int


@dataclass(frozen=True)
class PickCubeExpertResult:
    candidate_episode: ContinuousEpisode | None
    collection_success: bool
    replay_success: bool
    target_success: bool
    source_creation_attempts: int
    target_creation_attempts: int
    source_work: ReferenceWork | None
    target_work: boundary.ManiSkillWork | None
    solver_completed: bool
    error: str | None
    traceback: str | None

    @property
    def qualified(self) -> bool:
        return (
            self.candidate_episode is not None
            and self.collection_success
            and self.replay_success
            and self.target_success
            and self.error is None
        )

    def training_episode(self) -> ContinuousEpisode:
        """Refuse incomplete or unsuccessful conversions as expert BC data."""
        if not self.qualified or self.candidate_episode is None:
            raise ValueError("expert conversion has not qualified for training")
        value = self.candidate_episode
        return ContinuousEpisode(
            value.episode_id, value.context, value.role, value.spec, value.transitions
        )


def _command(value: Any, size: int) -> np.ndarray:
    array = boundary._array(value)
    if array.shape == (size,):
        array = array.reshape(1, size)
    values = boundary.encode_state_observation(array)
    if len(values) != size:
        raise ValueError(f"expected one {size}-coordinate command")
    return np.array([values], dtype=np.float64)


class _Reference:
    """Own one joint-reference scene, two explicit resets and two bounded phases."""

    def __init__(self) -> None:
        self.reset_attempts = self.reset_returns = 0
        self.attempts = [0, 0]
        self.returns = [0, 0]
        self.success = [False, False]
        self.actions: list[tuple[float, ...]] = []
        self.phase = 0
        self.done = False
        self.failed = False
        self.ratio = 0
        self.observation_dim = 0
        self.env: Any = None

    def open(self) -> None:
        _LOG.info("expert_phase=source_create state=attempt")
        self.env = boundary._make_environment(control_mode="pd_joint_pos")
        _LOG.info("expert_phase=source_create state=returned")
        base = self.env.unwrapped
        space = base.single_action_space
        if (
            base.num_envs != 1
            or base.device.type != "cpu"
            or base.obs_mode != "state"
            or base.control_mode != "pd_joint_pos"
            or base.agent.uid != "panda"
            or space.shape != (8,)
            or np.asarray(space.low).shape != (8,)
            or np.asarray(space.high).shape != (8,)
            or not np.isfinite(space.low).all()
            or not np.isfinite(space.high).all()
            or np.any(space.low > space.high)
            or space.low[-1] != -1
            or space.high[-1] != 1
        ):
            raise ValueError("expected single CPU Panda state/joint-position reference")
        sim, control = base.sim_freq, base.control_freq
        if (
            type(sim) is not int
            or type(control) is not int
            or control <= 0
            or sim % control
            or not 1 <= sim // control <= 100
        ):
            raise ValueError("invalid reference controller physics ratio")
        self.ratio = sim // control

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, *, seed: int) -> Any:
        if self.failed or seed != 17 or self.reset_attempts != self.phase:
            raise ValueError("one seed17 reset per reference phase is required")
        self.reset_attempts += 1
        _LOG.info("expert_phase=source_reset phase=%s state=attempt", self.phase)
        try:
            output = self.env.reset(seed=seed)
            self.reset_returns += 1
            _LOG.info("expert_phase=source_reset phase=%s state=returned", self.phase)
            self.observation_dim = len(boundary.encode_state_observation(output[0]))
            self.done = False
            return output
        except BaseException:
            self.failed = True
            raise

    def begin_replay(self) -> None:
        if self.phase != 0 or not self.done or not self.success[0] or self.failed:
            raise ValueError("a complete successful source is required before replay")
        self.phase = 1
        self.reset(seed=17)

    def step(self, value: Any) -> Any:
        if self.failed or self.reset_returns != self.phase + 1:
            raise ValueError("reference is failed or has not reset")
        if self.done:
            raise _EpisodeComplete()
        if self.attempts[self.phase] >= boundary.HORIZON:
            raise ValueError("reference50-step bound reached without a terminal flag")
        command = _command(value, 8)
        space = self.unwrapped.single_action_space
        if np.any(command < space.low) or np.any(command > space.high):
            raise ValueError("joint-reference command is outside native controller bounds")
        command = command.astype(np.float32)
        if np.any(command < space.low) or np.any(command > space.high):
            raise ValueError("quantized joint-reference command is outside bounds")
        self.attempts[self.phase] += 1
        _LOG.info(
            "expert_phase=source_step phase=%s attempt=%s", self.phase, self.attempts[self.phase]
        )
        try:
            output = self.env.step(command)
            self.returns[self.phase] += 1
            _LOG.info(
                "expert_phase=source_step phase=%s returned=%s",
                self.phase,
                self.returns[self.phase],
            )
            if type(output) not in (tuple, list) or len(output) != 5:
                raise ValueError("reference step must return five fields")
            if len(boundary.encode_state_observation(output[0])) != self.observation_dim:
                raise ValueError("reference state observation shape changed")
            reward = boundary._scalar(output[1])
            if reward not in (0.0, 1.0):
                raise ValueError("reference sparse reward must be0/1")
            terminated = boundary._scalar(output[2], boolean=True)
            truncated = boundary._scalar(output[3], boolean=True)
            self.done = bool(terminated or truncated)
            if self.returns[self.phase] == boundary.HORIZON and not self.done:
                raise ValueError("reference50-step terminal flag is missing")
            self.success[self.phase] |= bool(reward)
            if self.phase == 0:
                self.actions.append(tuple(float(item) for item in command[0]))
            return output
        except BaseException:
            self.failed = True
            raise

    def finish_collection(self) -> None:
        if not self.actions:
            raise ValueError("official expert produced no joint commands")
        # Why: maintain the official horizon with actual last-target teacher
        # holds, rather than fabricating completion or adding zero-action data.
        while not self.done:
            self.step(np.array(self.actions[-1], dtype=np.float32))
        if not self.success[0]:
            raise ValueError("source expert did not obtain sparse task success")

    def work(self) -> ReferenceWork:
        return ReferenceWork(
            self.reset_attempts,
            self.reset_returns,
            self.attempts[0],
            self.returns[0],
            self.attempts[1],
            self.returns[1],
            sum(self.returns) * self.ratio,
            (sum(self.attempts) - sum(self.returns)) * self.ratio,
        )


class _Target:
    """Let official conversion inspect controllers but route every step via port."""

    def __init__(self, port: boundary.ManiSkillPickCubeEpisode) -> None:
        self.port = port
        self.success = False

    @property
    def unwrapped(self) -> Any:
        # This private coupling stays within infra: no native environment is
        # returned by the public collector/record API or given to a learner.
        return self.port._env.unwrapped

    def step(self, value: Any) -> Any:
        command = tuple(float(item) for item in _command(value, 4)[0])
        _LOG.info("expert_phase=target_step state=attempt")
        record = self.port.step(command)
        _LOG.info("expert_phase=target_step state=returned step=%s", record.step)
        if record.reward not in (0.0, 1.0):
            raise ValueError("target sparse reward must be0/1")
        self.success |= bool(record.reward)
        if record.terminated or record.truncated:
            raise _EpisodeComplete()
        return None, record.reward, False, False, {}


def _solve(reference: _Reference) -> Any:
    _LOG.info("expert_phase=solver_import state=attempt")
    module = import_module("mani_skill.examples.motionplanning.panda.solutions.pick_cube")
    _LOG.info("expert_phase=solver_call state=attempt")
    return module.solve(reference, seed=17, debug=False, vis=False)


def _convert(reference: _Reference, target: _Target) -> None:
    _LOG.info("expert_phase=converter_import state=attempt")
    module = import_module("mani_skill.trajectory.utils.actions.conversion")
    _LOG.info("expert_phase=converter_call state=attempt")
    module.from_pd_joint_pos(
        boundary.CONTROL_MODE,
        np.array(reference.actions, dtype=np.float32),
        reference,
        target,
        render=False,
        pbar=None,
        verbose=False,
    )


def _require_expert_compatibility() -> None:
    # Why: this native constructor crashes rather than raising Python errors.
    # Official issue1100 documents the NumPy boundary; the isolated compatible
    # profile passes our identical constructor cut. Never mutate the caller's env.
    if version("mplib") == "0.1.1" and int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError(
            "MPlib0.1.1 expert planning requires NumPy<2 in a compatible isolated "
            "runtime; the NumPy2 profile can crash during ArticulatedModel creation. "
            "See docs/maniskill-expert.md for the verified compatibility profile"
        )


def collect_pick_cube_expert() -> PickCubeExpertResult:
    """One fixed seed17 TRAIN qualification; failures and work remain observable.

    Final evaluation, probes, teacher tuning and learner updates are unavailable.
    Both scenes retain the official50-step time limit and independent state.
    """
    reference = _Reference()
    port = None
    target = None
    episode = None
    source_attempts = 0
    target_attempts = 0
    solver_completed = False
    error = trace = None
    try:
        _require_expert_compatibility()
        source_attempts += 1
        reference.open()
        try:
            outcome = _solve(reference)
            solver_completed = True
            if type(outcome) is int and outcome == -1:
                raise ValueError("official expert reported planning failure")
        except _EpisodeComplete:
            pass
        reference.finish_collection()
        reference.begin_replay()
        target_attempts += 1
        _LOG.info("expert_phase=target_create state=attempt")
        port = boundary.ManiSkillPickCubeEpisode("pick-cube-expert-17", "pick-cube")
        _LOG.info("expert_phase=target_create state=returned")
        target = _Target(port)
        try:
            _convert(reference, target)
        except _EpisodeComplete:
            pass
        episode = port.episode()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        trace = traceback.format_exc()
    finally:
        for owned in (port, reference.env):
            if owned is not None:
                try:
                    owned.close()
                except Exception as failure:
                    error = error or f"cleanup {type(failure).__name__}: {failure}"
                    trace = trace or traceback.format_exc()
    return PickCubeExpertResult(
        episode,
        reference.success[0],
        reference.success[1],
        target.success if target else False,
        source_attempts,
        target_attempts,
        reference.work(),
        port.work if port else None,
        solver_completed,
        error,
        trace,
    )
