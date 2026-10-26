"""Collect one bounded TRAIN MiniGrid trajectory under a frozen recurrent policy.

Inputs are an owned-session source, predefined TRAIN layout and caller PCG64.
Outputs retain raw categorical decisions, native physics, behavior state and
permanent work on failures. No teacher, SGD, masking, replay, final/probe data,
budget creation, selection or IO belongs here. Source ports are checked for
unchanged learned state; this boundary does not attest arbitrary implementations.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields, replace
from math import fsum, isfinite
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.key_door_training import RewardPolicySource
from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.demonstration_episode import DemonstrationTransition
from src.core.minigrid_key_door import MiniGridKeyDoorLayout, predefined_layouts
from src.core.policy_gradient_episode import PolicyGradientEpisode
from src.infra.minigrid_key_door import MiniGridKeyDoor


class MiniGridRewardCollectionFailed(RuntimeError):
    """A detached observed prefix, never authority for a completed reward update."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        super().__init__(str(self.report["reason"]))


def _admit(source, layout, cue, episode_id, sampler, check_boundary, seed, max_steps, role, clock):
    if type(layout) is not MiniGridKeyDoorLayout or set(vars(layout)) != {
        field.name for field in fields(MiniGridKeyDoorLayout)
    }:
        raise ValueError("layout must be an exact complete MiniGridKeyDoorLayout")
    owned = replace(layout)
    if owned not in predefined_layouts()[:3]:
        raise ValueError("reward collection requires a predefined A/B/C_train layout")
    if type(cue) is not str or cue not in ("blue", "red"):
        raise ValueError("cue must be an exact string blue or red")
    if type(role) is not str or role != "train":
        raise ValueError("reward collection accepts only the train role")
    if (
        type(episode_id) is not str
        or not episode_id
        or episode_id != episode_id.strip()
        or len(episode_id.encode("utf8")) > 64
    ):
        raise ValueError("episode_id must be trimmed nonempty text within64 UTF-8 bytes")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a native integer in0..2**31-1")
    if type(max_steps) is not int or not 1 <= max_steps <= 16:
        raise ValueError("max_steps must be a native integer in1..16")
    if (
        type(sampler) is not np.random.Generator
        or type(sampler.bit_generator) is not np.random.PCG64
    ):
        raise ValueError("sampling requires an exact caller-owned Generator(PCG64)")
    if not callable(check_boundary) or not callable(clock):
        raise ValueError("check_boundary and clock must be callable")
    if not callable(getattr(source, "snapshot_state", None)) or not callable(
        getattr(source, "start_episode", None)
    ):
        raise ValueError("source requires snapshot_state and start_episode")
    return owned


def _observation(value: Any) -> tuple[float, ...]:
    if (
        type(value) is not tuple
        or len(value) != 148
        or any(type(v) not in (int, float) or not -1 <= v <= 1 for v in value)
    ):
        raise ValueError("observation requires148 native finite normalized coordinates")
    return tuple(float(v) for v in value)


def _transition(result: Any, observation, action: int, step: int) -> DemonstrationTransition:
    if type(result) is not dict or set(result) != {
        "observation",
        "reward",
        "terminated",
        "truncated",
    }:
        raise ValueError("native result requires observation/reward/terminated/truncated")
    if type(result["terminated"]) is not bool or type(result["truncated"]) is not bool:
        raise ValueError("native end flags must be exact bools")
    return DemonstrationTransition(
        step,
        observation,
        action,
        result["reward"],
        _observation(result["observation"]),
        result["terminated"],
        7,
    )


def _physical_readback(
    native: Any, transitions: list[DemonstrationTransition], flags, work, complete: bool
):
    if type(native) is not dict:
        raise ValueError("environment report must be a dict")
    if complete:
        if native.get("closed") is not True or native.get("status") != "complete":
            raise ValueError("completed collection requires a completed closed environment")
        if (
            type(native.get("environment_steps")) is not int
            or native["environment_steps"] != work["environment_steps_returned"]
            or type(native.get("environment_steps_attempted")) is not int
            or native["environment_steps_attempted"] != work["environment_steps_attempted"]
        ):
            raise ValueError("native returned physics count differs from collector")
        if (
            native.get("terminated") is not flags[-1][0]
            or native.get("truncated") is not flags[-1][1]
        ):
            raise ValueError("native final end flags differ from the captured result")
        rows = native.get("transitions")
        if type(rows) is not list or len(rows) != len(transitions):
            raise ValueError("native transitions differ from observed trajectory")
        total: Any = native.get("total_return")
        if (
            type(total) not in (int, float)
            or not -16 <= total <= 16
            or abs(total - fsum(row.reward for row in transitions)) > 1e-12
        ):
            raise ValueError("native total_return differs from observed rewards")
        for row, transition, (_, truncated) in zip(rows, transitions, flags, strict=True):
            if (
                type(row) is not dict
                or row.get("applied") is not True
                or row.get("action") != transition.action
                or row.get("observation") != transition.observation
                or row.get("next_observation") != transition.next_observation
                or row.get("reward") != transition.reward
                or row.get("terminated") is not transition.terminal
                or row.get("truncated") is not truncated
            ):
                raise ValueError("native transition contradicts the captured result")
    return deepcopy(native)


def collect_minigrid_reward_episode(
    source: RewardPolicySource,
    layout: MiniGridKeyDoorLayout,
    cue: str,
    *,
    episode_id: str,
    sampler: np.random.Generator,
    check_boundary: Callable[[], None],
    seed: int = 17,
    max_steps: int = 16,
    role: str = "train",
    clock: Callable[[], float] = monotonic,
) -> dict[str, Any]:
    """Sample raw seven-action probabilities once per actual native TRAIN step.

    Why: reward rollouts often truncate, so build the existing reward DTO from
    actual returns instead of the demonstration conversion requiring termination.
    All seven native inputs remain available, including blocked moves and no-ops.
    """
    owned = _admit(
        source, layout, cue, episode_id, sampler, check_boundary, seed, max_steps, role, clock
    )
    work: dict[str, Any] = dict(
        snapshots_attempted=0,
        snapshots_returned=0,
        sessions_attempted=0,
        sessions_started=0,
        environment_constructions_attempted=0,
        environment_constructions_returned=0,
        prediction_calls_attempted=0,
        prediction_calls_returned=0,
        sampling_draws=0,
        environment_steps_attempted=0,
        environment_steps_returned=0,
        close_calls_attempted=0,
        close_calls_returned=0,
        failure_work_may_be_unobserved=False,
    )
    report: dict[str, Any] = dict(
        status="running",
        reason=None,
        episode=None,
        behavior_state=None,
        sampler_before=deepcopy(sampler.bit_generator.state),
        sampler_after=None,
        sampler_binding_preserved=None,
        state_unchanged=None,
        elapsed_seconds=None,
        predictions=[],
        environment=None,
        work=work,
        close_error=None,
        readback_error=None,
    )
    expected_sampler = deepcopy(report["sampler_before"])
    behavior = session = environment = None
    transitions: list[DemonstrationTransition] = []
    flags: list[tuple[bool, bool]] = []
    started: float | None = None
    last_time: float | None = None
    ending = None
    problem: Exception | None = None

    def time_now() -> float:
        nonlocal started, last_time
        value = clock()
        if (
            type(value) not in (int, float)
            or not isfinite(value)
            or (last_time is not None and value < last_time)
        ):
            raise ValueError("clock must return finite nondecreasing native seconds")
        last_time = float(value)
        if started is None:
            started = last_time
        report["elapsed_seconds"] = last_time - started
        return last_time

    def snapshot(owner):
        work["snapshots_attempted"] += 1
        value = owner.snapshot_state()
        work["snapshots_returned"] += 1
        return deepcopy(value)

    def verify() -> None:
        if sampler.bit_generator.state != expected_sampler:
            raise ValueError("sampler changed outside recorded policy draws")
        if behavior is not None and not _same_state(behavior, snapshot(source)):
            raise ValueError("source learned state changed during reward collection")
        if session is not None:
            if not _same_state(behavior, snapshot(session)):
                raise ValueError("session learned state changed during reward collection")
            if (
                type(session.completed_steps) is not int
                or session.completed_steps != work["prediction_calls_returned"]
            ):
                raise ValueError("session completed_steps differs from returned predictions")
        if sampler.bit_generator.state != expected_sampler:
            raise ValueError("sampler changed during state verification")

    def guard() -> None:
        verify()
        time_now()
        check_boundary()
        verify()

    def fail(error: Exception, field: str | None = None) -> None:
        nonlocal problem
        text = f"{type(error).__name__}: {error}"
        if field is not None:
            previous = report[field]
            report[field] = text if previous is None else f"{previous}; {text}"
        if problem is None:
            problem = error
            report.update(status="failed", reason=text, episode=None)

    try:
        guard()
        behavior = snapshot(source)
        if behavior is None:
            raise ValueError("source requires a non-null full behavior checkpoint")
        report["behavior_state"] = deepcopy(behavior)
        guard()
        work["sessions_attempted"] += 1
        session = source.start_episode()
        work["sessions_started"] += 1
        if (
            session is source
            or not callable(getattr(session, "snapshot_state", None))
            or not callable(getattr(session, "predict", None))
        ):
            raise ValueError("source requires an independent frozen recurrent session")
        guard()
        work["environment_constructions_attempted"] += 1
        environment = MiniGridKeyDoor(owned, cue, seed, max_steps, role=role, episode_id=episode_id)
        work["environment_constructions_returned"] += 1
        guard()
        for step in range(max_steps):
            guard()
            observation = _observation(environment.observation)
            slot: dict[str, Any] = dict(
                observation=observation, probabilities=None, uniform=None, action=None
            )
            report["predictions"].append(slot)
            work["prediction_calls_attempted"] += 1
            probabilities = session.predict(np.array([observation], dtype=np.float64))
            work["prediction_calls_returned"] += 1
            validate_class_probabilities(probabilities, 7)
            if probabilities.shape != (1, 7):
                raise ValueError("predictor must return exactly one seven-action row")
            masses = probabilities[0].copy()
            slot["probabilities"] = tuple(float(p) for p in masses)
            guard()
            uniform = float(sampler.random())
            work["sampling_draws"] += 1
            expected_sampler = deepcopy(sampler.bit_generator.state)
            slot["uniform"] = uniform
            cumulative = np.cumsum(masses)
            action = int(np.searchsorted(cumulative, uniform * cumulative[-1], side="right"))
            if not 0 <= action < 7:
                raise FloatingPointError("categorical sample exceeded raw cumulative mass")
            slot["action"] = action
            guard()
            work["environment_steps_attempted"] += 1
            result = environment.step(action)
            work["environment_steps_returned"] += 1
            transition = _transition(result, observation, action, step)
            transitions.append(transition)
            flags.append((result["terminated"], result["truncated"]))
            guard()
            if environment.observation != transition.next_observation:
                raise ValueError("native observation differs from its returned transition")
            if result["terminated"] or result["truncated"]:
                ending = "terminal" if result["terminated"] else "truncated"
                break
        if ending is None:
            raise ValueError("native environment failed to signal max_steps ending")
        guard()
    except Exception as error:
        fail(error)
    finally:
        # Cleanup is mandatory even when a guard already refused further work.
        if environment is not None:
            try:
                guard()
            except Exception as error:
                fail(error, "readback_error")
            try:
                work["close_calls_attempted"] += 1
                environment.close()
                work["close_calls_returned"] += 1
            except Exception as error:
                work["failure_work_may_be_unobserved"] = True
                fail(error, "close_error")
            try:
                native = environment.report
                report["environment"] = deepcopy(native)
                if type(native) is dict:
                    total = native.get("total_return")
                    if type(total) is float and not isfinite(total):
                        # Keep corrupt-header failures publishable without inventing
                        # a valid return or altering the observed paid transitions.
                        report["environment"]["total_return"] = None
                        report["environment"]["invalid_total_return"] = repr(total)
                _physical_readback(native, transitions, flags, work, problem is None)
            except Exception as error:
                fail(error, "readback_error")
        try:
            guard()
            report["state_unchanged"] = True
        except Exception as error:
            report["state_unchanged"] = False
            fail(error, "readback_error")
        report["sampler_after"] = deepcopy(sampler.bit_generator.state)
        report["sampler_binding_preserved"] = report["sampler_after"] == expected_sampler
        for attempted, returned in (
            ("snapshots_attempted", "snapshots_returned"),
            ("sessions_attempted", "sessions_started"),
            ("environment_constructions_attempted", "environment_constructions_returned"),
            ("prediction_calls_attempted", "prediction_calls_returned"),
            ("environment_steps_attempted", "environment_steps_returned"),
        ):
            if work[attempted] != work[returned]:
                work["failure_work_may_be_unobserved"] = True
        if type(report["environment"]) is dict and report["environment"].get(
            "failure_work_may_be_unobserved"
        ):
            work["failure_work_may_be_unobserved"] = True
    if problem is None:
        try:
            if ending is None:
                raise ValueError("completed collection requires an observed native ending")
            episode = PolicyGradientEpisode(
                episode_id,
                owned.context,
                role,
                tuple(transitions),
                tuple(slot["probabilities"] for slot in report["predictions"]),
                ending,
                7,
            )
            report.update(episode=episode, status="complete", reason=None)
        except Exception as error:
            fail(error)
    if problem is not None:
        raise MiniGridRewardCollectionFailed(report) from problem
    return report
