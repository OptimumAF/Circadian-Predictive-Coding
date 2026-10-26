"""Bounded actual key/door episodes from a target-free categorical predictor.

Inputs are the common evaluation identity, owned environment config and a
prediction callback. Outputs are actual decisions, transitions, returns and
bounded partial failures. No training, action repair, checkpoint selection or IO.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.app.line_policy_evaluation import EvaluationIdentity, _integer, _text
from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.key_door_navigation import (
    KeyDoorLayout,
    KeyDoorNavigation,
    key_door_layout_record,
)
from src.core.legal_policy_gradient import condition_action_probabilities

Array = NDArray[np.float64]


class KeyDoorEvaluationFailed(RuntimeError):
    """Unexpected fault retaining the owned, bounded observed partial episode."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        super().__init__(str(self.report["reason"]))


def _admit(
    predict: Callable[[Array], Array],
    identity: EvaluationIdentity,
    context: str,
    layout: KeyDoorLayout,
    cue: int,
    partition: str,
    max_steps: int,
    before_prediction: Callable[[], None] | None,
) -> tuple[KeyDoorNavigation, KeyDoorLayout]:
    if type(identity) is not EvaluationIdentity:
        raise ValueError("identity must be an exact EvaluationIdentity")
    identity.__post_init__()
    _text(context, "context", 32)
    if type(partition) is not str or partition not in ("known_layout", "heldout_layout"):
        raise ValueError("layout_partition must be known_layout or heldout_layout")
    if type(layout) is not KeyDoorLayout:
        raise ValueError("layout must be an exact KeyDoorLayout")
    owned = replace(layout)
    if context != owned.layout_id:
        raise ValueError("context must match the declared layout_id")
    if (context in identity.task_sequence) != (partition == "known_layout"):
        raise ValueError("layout partition must agree with the declared training sequence")
    _integer(max_steps, "max_steps", 1, 16)
    if not callable(predict) or (before_prediction is not None and not callable(before_prediction)):
        raise ValueError("predict and optional before_prediction must be callable")
    return KeyDoorNavigation(owned, cue), owned


def _attempt(
    predict: Callable[[Array], Array],
    environment: KeyDoorNavigation,
    report: dict[str, Any],
    before_prediction: Callable[[], None] | None,
    *,
    condition_on_legal_actions: bool = False,
) -> str | None:
    if before_prediction is not None:
        before_prediction()
    observation, legal = environment.observation, environment.legal_actions
    transition: dict[str, Any] = dict(
        observation=observation,
        next_observation=observation,
        legal_actions=legal,
        action=None,
        probabilities=None,
        applied=False,
        reward=0.0,
        terminal=False,
    )
    if condition_on_legal_actions:
        transition["raw_probabilities"] = None
    report["transitions"].append(transition)
    report["prediction_calls_attempted"] += 1
    result = predict(np.array([observation], dtype=np.float64))
    report["prediction_calls_returned"] += 1
    validate_class_probabilities(result, 5)
    if result.shape != (1, 5):
        raise ValueError("live predictor must return exactly one row of five probabilities")
    probabilities = tuple(float(v) for v in result[0])
    if condition_on_legal_actions:
        transition["raw_probabilities"] = probabilities
        probabilities = condition_action_probabilities(probabilities, legal)
    action = int(np.argmax(probabilities))
    transition.update(action=action, probabilities=probabilities)
    # The default uses raw argmax. Explicit conditioning changes the declared
    # distribution before choosing one action, without retrying or repairing it.
    if action not in legal:
        return "illegal_action"
    step = environment.step(action)
    transition.update(
        next_observation=step.observation, applied=True, reward=step.reward, terminal=step.terminal
    )
    report["steps_executed"] += 1
    report["return"] += step.reward
    report["terminal"] = step.terminal
    if step.terminal:
        return "success" if step.success else "wrong_key"
    return None


def run_key_door_episode(
    predict: Callable[[Array], Array],
    *,
    identity: EvaluationIdentity,
    context: str,
    layout: KeyDoorLayout,
    cue: int,
    layout_partition: str = "known_layout",
    max_steps: int = 12,
    before_prediction: Callable[[], None] | None = None,
    condition_on_legal_actions: bool = False,
) -> dict[str, Any]:
    """Execute actual greedy decisions; never train or repair a failed policy."""
    if type(condition_on_legal_actions) is not bool:
        raise ValueError("condition_on_legal_actions must be an exact bool")
    environment, owned_layout = _admit(
        predict, identity, context, layout, cue, layout_partition, max_steps, before_prediction
    )
    report: dict[str, Any] = dict(
        identity=asdict(identity),
        execution_origin="live_environment",
        checkpoint_origin="owned_native_checkpoint",
        phase="evaluation",
        learning_active=False,
        context=context,
        layout=key_door_layout_record(owned_layout),
        cue=cue,
        layout_partition=layout_partition,
        num_classes=5,
        observation_dim=8,
        max_steps=max_steps,
        status="running",
        outcome=None,
        success=False,
        terminal=False,
        **{"return": 0.0},
        steps_executed=0,
        prediction_calls_attempted=0,
        prediction_calls_returned=0,
        transitions=[],
    )
    if condition_on_legal_actions:
        report["action_conditioning"] = "legal_actions_v1"
    try:
        outcome = None
        for _ in range(max_steps):
            outcome = _attempt(
                predict,
                environment,
                report,
                before_prediction,
                **({"condition_on_legal_actions": True} if condition_on_legal_actions else {}),
            )
            if outcome is not None:
                break
        report.update(
            status="complete", outcome=outcome or "step_limit", success=outcome == "success"
        )
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise KeyDoorEvaluationFailed(report) from error
    return report
