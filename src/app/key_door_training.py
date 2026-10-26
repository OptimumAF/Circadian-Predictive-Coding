"""Collect bounded policy-generated key/door reward trajectories.

Inputs are a frozen-session source, train-only layout and caller-owned PCG64.
Outputs retain actual decisions/transitions, full behavior state and spent RNG/
prediction/physics work, including failures. No SGD, teacher, action repair,
replay, final evaluation, budget creation, promotion or filesystem belongs here.
An optional caller-owned quota charges physical attempts across fresh episodes;
predictions and illegal sampled actions consume no physical allowance.

The source port must provide owned frozen sessions. Collection records behavior;
it does not attest arbitrary caller implementations. Native reward admission
rechecks the full current checkpoint before SGD, refusing a changed source.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, Callable, Protocol

import numpy as np

from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.coupled_energy import Array
from src.core.demonstration_episode import DemonstrationTransition
from src.core.key_door_navigation import (
    KeyDoorLayout,
    KeyDoorNavigation,
    key_door_layout_record,
)
from src.core.policy_gradient_episode import PolicyGradientEpisode
from src.core.physical_interaction_quota import PhysicalInteractionQuota
from src.core.legal_policy_gradient import (
    LegalPolicyGradientEpisode,
    condition_action_probabilities,
)


class RewardEpisodePredictor(Protocol):
    def predict(self, features: Array) -> Array: ...


class RewardPolicySource(Protocol):
    def snapshot_state(self) -> Any: ...

    def start_episode(self) -> RewardEpisodePredictor: ...


class KeyDoorTrainingFailed(RuntimeError):
    """An operational fault with owned observed work, never a completed episode."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        super().__init__(str(self.report["reason"]))


def _admit(
    source: RewardPolicySource,
    layout: KeyDoorLayout,
    cue: int,
    episode_id: str,
    sampler: np.random.Generator,
    role: str,
    max_steps: int,
    before_prediction: Callable[[], None] | None,
) -> tuple[KeyDoorNavigation, str]:
    if type(role) is not str or role != "train":
        raise ValueError("reward collection accepts only the train role")
    if type(episode_id) is not str or not episode_id or episode_id != episode_id.strip():
        raise ValueError("episode_id must be nonempty trimmed text")
    if len(episode_id.encode("utf8")) > 64:
        raise ValueError("episode_id exceeds 64 UTF-8 bytes")
    if type(max_steps) is not int or not 1 <= max_steps <= 16:
        raise ValueError("max_steps must be a native integer in 1..16")
    if (
        type(sampler) is not np.random.Generator
        or type(sampler.bit_generator) is not np.random.PCG64
    ):
        raise ValueError("sampling requires an exact caller-owned Generator(PCG64)")
    if not callable(getattr(source, "snapshot_state", None)) or not callable(
        getattr(source, "start_episode", None)
    ):
        raise ValueError("source requires snapshot_state and start_episode")
    if before_prediction is not None and not callable(before_prediction):
        raise ValueError("before_prediction must be callable")
    if type(layout) is not KeyDoorLayout:
        raise ValueError("layout must be an exact KeyDoorLayout")
    owned = replace(layout)
    return KeyDoorNavigation(owned, cue), owned.layout_id


def collect_key_door_reward_episode(
    source: RewardPolicySource,
    layout: KeyDoorLayout,
    cue: int,
    *,
    episode_id: str,
    sampler: np.random.Generator,
    role: str = "train",
    max_steps: int = 12,
    before_prediction: Callable[[], None] | None = None,
    condition_on_legal_actions: bool = False,
    interaction_quota: PhysicalInteractionQuota | None = None,
) -> dict[str, Any]:
    if type(condition_on_legal_actions) is not bool:
        raise ValueError("condition_on_legal_actions must be an exact bool")
    quota_before = _admit_quota(interaction_quota)
    environment, context = _admit(
        source, layout, cue, episode_id, sampler, role, max_steps, before_prediction
    )
    report: dict[str, Any] = dict(
        status="running",
        reason=None,
        episode=None,
        behavior_state=None,
        sampler_before=deepcopy(sampler.bit_generator.state),
        sampler_after=None,
        sampler_binding_preserved=None,
        readback_error=None,
        prediction_calls_attempted=0,
        prediction_calls_returned=0,
        sampling_draws=0,
        environment_steps_attempted=0,
        environment_steps=0,
        total_return=0.0,
        transitions=[],
        termination=None,
    )
    if condition_on_legal_actions:
        report["action_conditioning"] = "legal_actions_v1"
    if quota_before is not None:
        # Why: preserve the original wire when disabled; enabled episodes carry
        # both cumulative boundaries without resetting or refunding prior work.
        report["interaction_quota_before"] = deepcopy(quota_before)
        report["interaction_quota_after"] = None
    # Keep legacy records exact; a changed control law needs explicit provenance.
    layout_record = key_door_layout_record(layout)
    if "movement_action_map" in layout_record:
        report["layout"] = layout_record
    problem = None
    expected_sampler = deepcopy(report["sampler_before"])
    expected_quota = deepcopy(quota_before)
    episode_steps = (
        max_steps
        if quota_before is None
        else min(max_steps, quota_before["maximum_steps"] - quota_before["attempted_steps"])
    )
    try:
        _check_quota(interaction_quota, expected_quota)
        report["behavior_state"] = deepcopy(source.snapshot_state())
        _check_quota(interaction_quota, expected_quota)
        _check_sampler(sampler, expected_sampler)
        predictor = source.start_episode()
        _check_quota(interaction_quota, expected_quota)
        _check_sampler(sampler, expected_sampler)
        for _ in range(episode_steps):
            arguments = (
                predictor,
                environment,
                sampler,
                expected_sampler,
                report,
                before_prediction,
                condition_on_legal_actions,
            )
            ending = (
                _decision(*arguments)
                if interaction_quota is None
                else _decision(
                    *arguments, interaction_quota=interaction_quota, expected_quota=expected_quota
                )
            )
            if ending is not None:
                break
        else:
            ending = "truncated"
        _check_quota(interaction_quota, expected_quota)
        report["termination"] = ending
        episode = _episode(report, episode_id, context, ending)
        report["episode"] = (
            LegalPolicyGradientEpisode(
                episode,
                tuple(slot["legal_actions"] for slot in report["transitions"]),
                tuple(slot["raw_probabilities"] for slot in report["transitions"]),
            )
            if condition_on_legal_actions
            else episode
        )
        report.update(status="complete", reason=None)
    except Exception as error:
        problem = error
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        # Caller RNG advances permanently per draw, even if later work fails.
        report["sampler_after"] = deepcopy(sampler.bit_generator.state)
        report["sampler_binding_preserved"] = report["sampler_after"] == expected_sampler
        if not report["sampler_binding_preserved"]:
            readback_problem = ValueError("sampler changed outside recorded policy draws")
            report["readback_error"] = (
                f"readback {type(readback_problem).__name__}: {readback_problem}"
            )
            if problem is None:
                problem = readback_problem
                report.update(status="failed", reason=report["readback_error"], episode=None)
        if interaction_quota is not None:
            try:
                report["interaction_quota_after"] = asdict(interaction_quota.snapshot())
                _check_quota(interaction_quota, expected_quota)
            except Exception as error:
                report["interaction_quota_readback_error"] = f"{type(error).__name__}: {error}"
                if problem is None:
                    problem = error
                    report.update(
                        status="failed",
                        reason=report["interaction_quota_readback_error"],
                        episode=None,
                    )
    if problem is not None:
        raise KeyDoorTrainingFailed(report) from problem
    return report


def _decision(
    predictor: RewardEpisodePredictor,
    environment: KeyDoorNavigation,
    sampler: np.random.Generator,
    expected_sampler: dict[str, Any],
    report: dict[str, Any],
    before_prediction: Callable[[], None] | None,
    condition_on_legal_actions: bool = False,
    *,
    interaction_quota: PhysicalInteractionQuota | None = None,
    expected_quota: dict[str, int] | None = None,
) -> str | None:
    _check_quota(interaction_quota, expected_quota)
    _check_sampler(sampler, expected_sampler)
    if before_prediction is not None:
        before_prediction()
    _check_quota(interaction_quota, expected_quota)
    _check_sampler(sampler, expected_sampler)
    observation = environment.observation
    slot: dict[str, Any] = dict(
        step=len(report["transitions"]),
        observation=observation,
        legal_actions=environment.legal_actions,
        probabilities=None,
        action=None,
        applied=False,
        reward=0.0,
        next_observation=observation,
        terminal=False,
    )
    report["transitions"].append(slot)
    report["prediction_calls_attempted"] += 1
    probabilities = predictor.predict(np.array([observation], dtype=np.float64))
    report["prediction_calls_returned"] += 1
    _check_quota(interaction_quota, expected_quota)
    _check_sampler(sampler, expected_sampler)
    validate_class_probabilities(probabilities, 5)
    if probabilities.shape != (1, 5):
        raise ValueError("training predictor must return exactly one five-action row")
    masses = probabilities[0].copy()
    if condition_on_legal_actions:
        slot["raw_probabilities"] = tuple(float(p) for p in masses)
        masses = np.array(
            condition_action_probabilities(slot["raw_probabilities"], slot["legal_actions"]),
            dtype=np.float64,
        )
    slot["probabilities"] = tuple(float(p) for p in masses)
    uniform = float(sampler.random())
    report["sampling_draws"] += 1
    expected_sampler.clear()
    expected_sampler.update(deepcopy(sampler.bit_generator.state))
    cumulative = np.cumsum(masses)
    # Categorical sampling uses actual positive total mass to accommodate native
    # roundoff. Opt-in conditioning happens before this single draw; no resampling.
    # The default preserves the original unmasked raw-mass CDF exactly.
    action = int(np.searchsorted(cumulative, uniform * cumulative[-1], side="right"))
    if not 0 <= action < 5:
        raise FloatingPointError("categorical sampling exceeded its raw cumulative mass")
    slot["action"] = action
    if action not in slot["legal_actions"]:
        return "illegal_action"
    if interaction_quota is not None:
        _check_quota(interaction_quota, expected_quota)
        interaction_quota.begin_step()
        assert expected_quota is not None
        expected_quota.update(asdict(interaction_quota.snapshot()))
    report["environment_steps_attempted"] += 1
    try:
        result = environment.step(action)
    except BaseException:
        if interaction_quota is not None:
            _check_quota(interaction_quota, expected_quota)
            interaction_quota.fail_step()
            assert expected_quota is not None
            expected_quota.update(asdict(interaction_quota.snapshot()))
        raise
    report["environment_steps"] += 1
    slot.update(
        applied=True,
        reward=result.reward,
        next_observation=result.observation,
        terminal=result.terminal,
    )
    report["total_return"] += result.reward
    if interaction_quota is not None:
        _check_quota(interaction_quota, expected_quota)
        interaction_quota.complete_step()
        assert expected_quota is not None
        expected_quota.update(asdict(interaction_quota.snapshot()))
    _check_sampler(sampler, expected_sampler)
    return "terminal" if result.terminal else None


def _episode(
    report: dict[str, Any], episode_id: str, context: str, ending: str
) -> PolicyGradientEpisode:
    transitions = tuple(
        DemonstrationTransition(
            slot["step"],
            slot["observation"],
            slot["action"],
            slot["reward"],
            slot["next_observation"],
            slot["terminal"],
            5,
        )
        for slot in report["transitions"]
    )
    return PolicyGradientEpisode(
        episode_id,
        context,
        "train",
        transitions,
        tuple(slot["probabilities"] for slot in report["transitions"]),
        ending,
        5,
    )


def _check_sampler(sampler: np.random.Generator, expected: dict[str, Any]) -> None:
    if sampler.bit_generator.state != expected:
        raise ValueError("sampler changed outside recorded policy draws")


def _admit_quota(quota: PhysicalInteractionQuota | None) -> dict[str, int] | None:
    if quota is None:
        return None
    if type(quota) is not PhysicalInteractionQuota:
        raise ValueError("interaction_quota must be an exact PhysicalInteractionQuota")
    snapshot = asdict(quota.snapshot())
    if (
        snapshot["failed_steps"]
        or snapshot["attempted_steps"] != snapshot["completed_steps"]
        or snapshot["attempted_steps"] == snapshot["maximum_steps"]
    ):
        raise ValueError("interaction quota is failed, pending or exhausted")
    return snapshot


def _check_quota(quota: PhysicalInteractionQuota | None, expected: dict[str, int] | None) -> None:
    if quota is not None and asdict(quota.snapshot()) != expected:
        raise ValueError("interaction quota changed outside recorded physical steps")
