"""Bounded train-role live readout of two owned acquisition checkpoints.

Inputs are completed training evidence, its unchanged empirical label prior and
the original budget/guard. Outputs retain four learner and two prior episodes,
including refusal prefixes. No training, final/probe data, selection or IO.
"""

from copy import deepcopy
from math import fsum, isfinite
from typing import Any, Callable

import numpy as np

from src.adapters import key_door_pilot as shared
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.key_door_policy_evaluation import KeyDoorEvaluationFailed, run_key_door_episode
from src.app.line_policy_evaluation import EvaluationIdentity
from src.core.legal_policy_gradient import condition_action_probabilities


def _slot(point: int, cue: int) -> dict[str, Any]:
    return dict(
        checkpoint_updates=point,
        context="A",
        cue=cue,
        layout_partition="known_layout",
        source_role="train",
        status="not_run",
        reason="not reached",
        state_unchanged=None,
    )


def _endpoint(value: int) -> None:
    if type(value) is not int or value not in (64, 128):
        raise ValueError("endpoint must be an exact native integer64 or128")


def new_diagnostic(*, endpoint: int = 64) -> dict[str, Any]:
    """Allocate only six fixed slots; no environment or learner is constructed."""
    _endpoint(endpoint)
    return dict(
        source_role="train",
        classification="train_resubstitution",
        status="not_run",
        reason="not reached",
        method="recurrent_bc",
        replay_enabled=False,
        action_conditioning="legal_actions_v1",
        episodes=[_slot(point, cue) for point in (0, endpoint) for cue in (1, -1)],
        live_sessions_started=0,
        reference=dict(
            source_role="train",
            kind="fixed_pooled_train_label_prior",
            probabilities=None,
            episodes=[_slot(0, cue) for cue in (1, -1)],
            outcomes=None,
        ),
        outcomes=None,
        learner_gains=None,
        comparison=None,
        comparison_reason="not reached",
        work={},
        scientific_advantage_verified=False,
        limitation="train resubstitution; same legal information, unequal computation; no final promotion or generalization claim",
    )


def _observed_support(observation: Any) -> tuple[int, ...]:
    """Read the evaluator's visible support without constructing an environment."""
    if (
        type(observation) is not tuple
        or len(observation) != 8
        or any(type(value) not in (int, float) for value in observation)
        or any(value not in (-1, -0.5, 0, 0.5, 0.75, 1) for value in observation[:5])
        or any(value not in (-1, 0, 1) for value in observation[5:7])
        or not -1 <= observation[7] <= 1
        or not all(isfinite(value) for value in observation)
    ):
        raise ValueError("prior refusal requires a valid native eight-feature observation")
    support = tuple(action for action in range(4) if observation[action] not in (-1, 0.75))
    interaction = (abs(observation[4]) == 0.5 and observation[5] == 0) or (
        0.75 in observation[:4] and observation[5] != 0
    )
    return (*support, 4) if interaction else support


def _zero_legal_mass(record: dict[str, Any], probabilities: tuple[float, ...]) -> bool:
    """Recognize only an observed valid raw return with zero declared support."""
    transitions = record.get("transitions")
    if type(transitions) is not list or not transitions or len(transitions) > 12:
        return False
    last = transitions[-1]
    if (
        type(last) is not dict
        or any(
            type(record.get(field)) is not int
            for field in (
                "prediction_calls_attempted",
                "prediction_calls_returned",
                "steps_executed",
            )
        )
        or record.get("status") != "failed"
        or record.get("terminal") is not False
        or record.get("success") is not False
        or record.get("prediction_calls_attempted") != len(transitions)
        or record.get("prediction_calls_returned") != len(transitions)
        or record.get("steps_executed") != len(transitions) - 1
        or last.get("raw_probabilities") != probabilities
        or last.get("probabilities") is not None
        or last.get("action") is not None
        or last.get("applied") is not False
        or last.get("observation") != last.get("next_observation")
        or last.get("reward") != 0
        or last.get("terminal") is not False
    ):
        return False
    try:
        # Validate both the raw distribution and support independently of the
        # exception text; a different predictor/physics/guard fault must stop.
        condition_action_probabilities(probabilities, tuple(range(5)))
        legal = last["legal_actions"]
        condition_action_probabilities((0.2,) * 5, legal)
        if (
            legal != _observed_support(last["observation"])
            or fsum(probabilities[action] for action in legal) != 0
        ):
            return False
        previous = None
        rewards = []
        for transition in transitions[:-1]:
            conditional = condition_action_probabilities(probabilities, transition["legal_actions"])
            action = max(range(5), key=conditional.__getitem__)
            if (
                transition["raw_probabilities"] != probabilities
                or transition["legal_actions"] != _observed_support(transition["observation"])
                or transition["probabilities"] != conditional
                or transition["action"] != action
                or transition["applied"] is not True
                or transition["terminal"] is not False
                or type(transition["reward"]) not in (int, float)
                or transition["reward"] != 0
                or (previous is not None and transition["observation"] != previous)
            ):
                return False
            previous = transition["next_observation"]
            rewards.append(transition["reward"])
        return (previous is None or last["observation"] == previous) and record["return"] == fsum(
            rewards
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _outcome(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if len(episodes) != 2 or any(slot["status"] != "complete" for slot in episodes):
        raise ValueError("train policy aggregate requires both complete cue episodes")
    return dict(
        success_rate=sum(slot["success"] for slot in episodes) / 2,
        mean_return=sum(slot["return"] for slot in episodes) / 2,
    )


def evaluate_train_policy(
    report: dict[str, Any],
    source: Any,
    states: list[tuple[int, Any]],
    reference: dict[str, Any],
    budget: Any,
    *,
    check_boundary: Callable[[], None],
    endpoint: int = 64,
) -> None:
    """Read fixed train checkpoints without updating their learner or prior."""
    _endpoint(endpoint)
    if not callable(check_boundary):
        raise ValueError("check_boundary must be callable")
    if type(report) is not dict or report != new_diagnostic(endpoint=endpoint):
        raise ValueError("train diagnostic requires its original unused six fixed slots")
    if (
        type(states) is not list
        or len(states) != 2
        or any(type(entry) is not tuple or len(entry) != 2 for entry in states)
        or tuple(point for point, _ in states) != (0, endpoint)
        or any(type(point) is not int for point, _ in states)
    ):
        raise ValueError(f"train diagnostic requires exact owned checkpoints0/{endpoint}")
    for point, state in states:
        if (
            type(state.policy) is not TorchRecurrentPolicy
            or state.policy != TorchRecurrentPolicy(8, 8, 5, 17)
            or type(state.completed_updates) is not int
            or state.completed_updates != point
            or type(state.training_examples) is not int
            or state.training_examples != 8 * point
        ):
            raise ValueError("train diagnostic checkpoint policy or native counters differ")
    if (
        type(reference) is not dict
        or reference.get("source_role") != "train"
        or type(reference.get("probabilities")) is not list
        or len(reference["probabilities"]) != 5
        or budget.updates_completed != endpoint
        or budget.budget.max_training_updates != endpoint
    ):
        raise ValueError(
            f"train diagnostic requires unchanged train prior and original{endpoint}update budget"
        )
    probabilities = tuple(reference["probabilities"])
    condition_action_probabilities(probabilities, tuple(range(5)))
    frozen_states, frozen_reference = deepcopy(states), deepcopy(reference)
    report.update(status="running", reason=None)
    report["reference"]["probabilities"] = list(probabilities)
    frozen_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    source_state: Any = None
    learner_slots, prior_report = report["episodes"], report["reference"]
    prior_slots = prior_report["episodes"]
    fields = ("checkpoint_updates", "context", "cue", "layout_partition", "source_role")
    bindings = tuple(
        (slot, tuple(slot[field] for field in fields)) for slot in (*learner_slots, *prior_slots)
    )

    def check_slots() -> None:
        if (
            report["episodes"] is not learner_slots
            or report["reference"] is not prior_report
            or prior_report["episodes"] is not prior_slots
            or len(learner_slots) != 4
            or len(prior_slots) != 2
            or any(
                actual is not bound or tuple(actual.get(field) for field in fields) != metadata
                for actual, (bound, metadata) in zip(
                    (*learner_slots, *prior_slots), bindings, strict=True
                )
            )
            or report["method"] != "recurrent_bc"
            or report["replay_enabled"] is not False
            or report["source_role"] != "train"
            or report["classification"] != "train_resubstitution"
            or report["action_conditioning"] != "legal_actions_v1"
            or prior_report["source_role"] != "train"
            or prior_report["kind"] != "fixed_pooled_train_label_prior"
            or prior_report["probabilities"] != list(probabilities)
        ):
            raise ValueError("train diagnostic fixed slots or metadata changed")

    def guard() -> None:
        check_slots()
        for slot in report["episodes"]:
            if slot["status"] == "complete" and not any(
                record is slot for record, _ in frozen_records
            ):
                frozen_records.append((slot, deepcopy(slot)))
        check_boundary()
        budget.before_final()
        check_slots()
        for record, frozen in frozen_records:
            # _live publishes the returned episode before its unchanged-state
            # verification; only that one pending confirmation may advance.
            if frozen.get("state_unchanged") is None and record.get("state_unchanged") is True:
                frozen["state_unchanged"] = True
        if (
            not _same_state(states, frozen_states)
            or reference != frozen_reference
            or (source_state is not None and not _same_state(source_state, source.snapshot_state()))
            or any(record != frozen for record, frozen in frozen_records)
        ):
            raise ValueError("train diagnostic checkpoint, prior or preceding evidence changed")

    prior_attempts = prior_returns = 0

    def predict_prior(features: Any) -> Any:
        nonlocal prior_attempts, prior_returns
        guard()
        prior_attempts += 1
        result = np.tile(np.array([probabilities], dtype=np.float64), (len(features), 1))
        prior_returns += 1
        guard()
        return result

    try:
        guard()
        source_state = deepcopy(source.snapshot_state())
        guard()
        shared._live(
            report,
            source,
            states,
            budget,
            task_sequence=("A",),
            declared_layouts=(shared.layouts()[0],),
            condition_on_legal_actions=True,
            check_boundary=guard,
        )
        guard()
        report["outcomes"] = [
            dict(
                updates=point,
                **_outcome([s for s in report["episodes"] if s["checkpoint_updates"] == point]),
            )
            for point in (0, endpoint)
        ]
        report["learner_gains"] = {
            key: report["outcomes"][1][key] - report["outcomes"][0][key]
            for key in ("success_rate", "mean_return")
        }
        for slot in report["reference"]["episodes"]:
            guard()
            try:
                result = run_key_door_episode(
                    predict_prior,
                    identity=EvaluationIdentity(
                        "recurrent_bc", 17, "train:fixed-label-prior", ("A",), 0, False
                    ),
                    context="A",
                    layout=shared.layouts()[0],
                    cue=slot["cue"],
                    max_steps=12,
                    before_prediction=guard,
                    condition_on_legal_actions=True,
                )
                slot.update(result, state_unchanged=True, reason=None)
            except KeyDoorEvaluationFailed as error:
                slot.update(error.report, state_unchanged=None)
                if not _zero_legal_mass(error.report, probabilities):
                    raise
                guard()
                # A refusal has an observed return prefix, not a completed score.
                slot.update(
                    status="refused", refusal="zero_legal_mass", success=None, state_unchanged=True
                )
            slot["checkpoint_origin"] = "fixed_train_label_prior"
            frozen_records.append((slot, deepcopy(slot)))
            guard()
        prior = report["reference"]
        if all(slot["status"] == "complete" for slot in prior["episodes"]):
            prior["outcomes"] = _outcome(prior["episodes"])
            report["comparison"] = [
                dict(
                    updates=point["updates"],
                    **{
                        key: point[key] - prior["outcomes"][key]
                        for key in ("success_rate", "mean_return")
                    },
                )
                for point in report["outcomes"]
            ]
            report["comparison_reason"] = None
        else:
            report["comparison_reason"] = (
                "prior_zero_legal_mass; incomplete prior episodes have no aggregate or numeric comparison"
            )
        guard()
        report.update(status="complete", reason=None)
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
        raise
    finally:
        report["work"] = dict(
            learner_callback_attempts=sum(
                s.get("prediction_calls_attempted", 0) for s in report["episodes"]
            ),
            learner_callback_returns=sum(
                s.get("prediction_calls_returned", 0) for s in report["episodes"]
            ),
            learner_applied_steps=sum(s.get("steps_executed", 0) for s in report["episodes"]),
            learner_sessions_started=report["live_sessions_started"],
            prior_callback_attempts=prior_attempts,
            prior_callback_returns=prior_returns,
            prior_applied_steps=sum(
                s.get("steps_executed", 0) for s in report["reference"]["episodes"]
            ),
            failure_work_may_be_unobserved=report["status"] != "complete",
        )
