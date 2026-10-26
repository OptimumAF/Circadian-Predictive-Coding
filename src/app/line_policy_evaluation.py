"""Bounded, live line-policy episodes through a prediction-only callable.

Inputs are an explicit evaluation identity and a fresh environment configuration.
Outputs retain every attempted prediction and its actual environment outcome.
This module does not train, select checkpoints, repair actions, or access IO/RNG.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.line_navigation import LineNavigation

Array = NDArray[np.float64]
_METHODS = (
    "bp",
    "pc",
    "neutral_cpc",
    "gated_cpc",
    "recurrent_bc",
    "recurrent_rl",
    "observation_memory",
)


def _text(value: object, name: str, byte_limit: int) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > byte_limit
    ):
        raise ValueError(
            f"{name} must be nonempty trimmed text of at most {byte_limit} UTF-8 bytes"
        )


def _integer(value: object, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a native integer in {minimum}..{maximum}")


@dataclass(frozen=True)
class EvaluationIdentity:
    method: str
    seed: int
    actor_version: str
    task_sequence: tuple[str, ...]
    checkpoint_updates: int
    replay_enabled: bool

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        _integer(self.seed, "seed", 0, 2**31 - 1)
        _text(self.actor_version, "actor_version", 128)
        if type(self.task_sequence) is not tuple or not 1 <= len(self.task_sequence) <= 32:
            raise ValueError("task_sequence must be a tuple containing 1..32 context identifiers")
        for context in self.task_sequence:
            _text(context, "task_sequence context", 32)
        _integer(self.checkpoint_updates, "checkpoint_updates", 0, 512)
        if type(self.replay_enabled) is not bool:
            raise ValueError("replay_enabled must be a native bool")


class LineEvaluationFailed(RuntimeError):
    """Unexpected execution failure with an owned, bounded partial report."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        super().__init__(str(self.report["reason"]))


def _admit(
    predict: Callable[[Array], Array],
    identity: EvaluationIdentity,
    context: str,
    start: int,
    goal: int,
    max_steps: int,
    before_prediction: Callable[[], None] | None,
) -> LineNavigation:
    if type(identity) is not EvaluationIdentity:
        raise ValueError("identity must be an EvaluationIdentity")
    # Frozen dataclasses can still be bypassed with object.__setattr__; recheck admission.
    identity.__post_init__()
    _text(context, "context", 32)
    if context not in identity.task_sequence:
        raise ValueError("context must occur in identity.task_sequence")
    _integer(max_steps, "max_steps", 1, 16)
    if not callable(predict):
        raise ValueError("predict must be callable")
    if before_prediction is not None and not callable(before_prediction):
        raise ValueError("before_prediction must be callable or None")
    return LineNavigation(start, goal)


def _report(
    identity: EvaluationIdentity, context: str, start: int, goal: int, max_steps: int
) -> dict[str, Any]:
    return {
        "identity": asdict(identity),
        "execution_origin": "live_environment",
        "checkpoint_origin": "recorded_parameters",
        "phase": "evaluation",
        "learning_active": False,
        "context": context,
        "start": start,
        "goal": goal,
        "max_steps": max_steps,
        "status": "running",
        "outcome": None,
        "success": False,
        "return": 0.0,
        "steps_executed": 0,
        "prediction_calls_attempted": 0,
        "prediction_calls_returned": 0,
        "transitions": [],
    }


def _probability(result: Array) -> float:
    if (
        type(result) is not np.ndarray
        or result.dtype != np.dtype(np.float64)
        or result.shape != (1, 1)
        or not bool(np.isfinite(result).all())
        or not 0.0 <= float(result[0, 0]) <= 1.0
    ):
        raise ValueError("predict must return a finite float64 ndarray of shape (1, 1) in [0, 1]")
    return float(result[0, 0])


def _attempt(
    predict: Callable[[Array], Array],
    environment: LineNavigation,
    report: dict[str, Any],
    before_prediction: Callable[[], None] | None,
) -> str | None:
    if before_prediction is not None:
        before_prediction()
    observation = environment.observation
    transition: dict[str, Any] = {
        "observation": observation,
        "next_observation": observation,
        "legal_actions": environment.legal_actions,
        "action": None,
        "probability": None,
        "applied": False,
        "reward": 0.0,
        "terminal": False,
    }
    report["transitions"].append(transition)
    report["prediction_calls_attempted"] += 1
    result = predict(np.array([observation], dtype=np.float64))
    report["prediction_calls_returned"] += 1
    probability = _probability(result)
    action = int(probability >= 0.5)
    transition.update(action=action, probability=probability)
    # Why: illegal decisions are policy failures, not invitations to clip or retry.
    if action not in environment.legal_actions:
        return "illegal_action"
    step = environment.step(action)
    transition.update(
        next_observation=step.observation,
        applied=True,
        reward=step.reward,
        terminal=step.terminal,
    )
    report["steps_executed"] += 1
    report["return"] += step.reward
    return "success" if step.terminal else None


def run_line_episode(
    predict: Callable[[Array], Array],
    *,
    identity: EvaluationIdentity,
    context: str,
    start: int,
    goal: int,
    max_steps: int = 4,
    before_prediction: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Evaluate actual policy decisions; unexpected faults raise with partial work.

    Why: successful action classification does not establish closed-loop success.
    Expected negative outcomes remain complete evaluations, separate from faults.
    The caller supplies checkpoint ownership and any permanent budget callback.
    """
    environment = _admit(predict, identity, context, start, goal, max_steps, before_prediction)
    report = _report(identity, context, start, goal, max_steps)
    try:
        outcome = None
        for _ in range(max_steps):
            outcome = _attempt(predict, environment, report, before_prediction)
            if outcome is not None:
                break
        report.update(
            status="complete", outcome=outcome or "step_limit", success=outcome == "success"
        )
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise LineEvaluationFailed(report) from error
    return report
