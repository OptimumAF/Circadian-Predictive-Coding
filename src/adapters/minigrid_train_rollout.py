"""Known TRAIN A games on one genuine frozen benchmark checkpoint.

Inputs are an existing completed arm and its boundary guard. Outputs retain the
two target-free game reports, including partial costs, and a zero-update budget
guard. This adapter performs no training, checkpoint reconstruction, selection,
final access or persistence; the existing evaluator owns all episode behavior.
"""

from collections.abc import Callable
from copy import deepcopy
import pickle
from time import monotonic
from typing import Any

from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    _return_budget_guard,
    evaluate_neural_representation_checkpoint,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

LIVE_SECONDS = 20.0


def validate_option(flag: bool, acquisition_only: bool, *, joint_catalogue: bool = False) -> None:
    if type(flag) is not bool or (
        flag and acquisition_only is not True and joint_catalogue is not True
    ):
        raise ValueError(
            "train_a_rollout requires an exact bool and acquisition_only or joint_catalogue"
        )


def describe_rollout(protocol: dict[str, Any], *, joint_catalogue: bool = False) -> None:
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_train_a_rollout_v1"
    protocol["train_a_rollout"] = dict(
        checkpoint_updates=128,
        cues=["blue", "red"],
        role="train",
        context="A",
        order="after all TRAIN fit and disposable probes/returns; genuine owned benchmark128",
        policy="frozen raw seven-action first argmax, no mask/repair; no teacher/targets in actions",
        budget="separate zero training/replay updates; max20s and original whole180s/RSS ceiling",
        metrics="complete episodes, success, return, legal/illegal actions, all predictions/steps",
        scope="known TRAIN games only; no E/F/final, selection, learning or generalization",
    )
    protocol["planned_work"].update(
        learner_live_episodes=2,
        maximum_live_prediction_forwards=32,
        maximum_live_environment_steps=32,
        maximum_total_prediction_forwards=protocol["planned_work"]["score_forwards"] + 32,
    )
    protocol["evaluation"] = (
        protocol["evaluation"].replace("no live/final evaluation", "no final evaluation")
        + "; then two target-free known TRAIN A games from frozen benchmark128"
    )
    protocol["metric_direction"] = protocol["metric_direction"].replace(
        "; no live success/return measured", ""
    )
    protocol["metric_direction"] += (
        "; live success/return higher, legal/illegal actions reported separately"
    )
    protocol["limitations"] += (
        "; known TRAIN live behavior is not novel-task/held-out generalization or circadian benefit"
    )
    if joint_catalogue:
        protocol["train_a_rollout"].update(
            checkpoint_training_rows=6912,
            catalogue_rows_per_update=54,
            order="after all joint TRAIN readouts and disposable F/A returns; genuine owned joint128/6912 checkpoint",
        )


def empty_record() -> dict[str, Any]:
    return dict(
        status="not_run",
        reason=None,
        checkpoint_updates=128,
        report=None,
        charged_updates=0,
        replay_examples=0,
        budget_binding_preserved=None,
        elapsed_seconds=None,
        timing_complete=None,
        evaluator_admitted=False,
    )


def run_train_rollout(
    arm: dict[str, Any],
    *,
    check_boundary: Callable[[], None],
    catalogue_method: str | None = None,
    wide_method: str | None = None,
) -> tuple[dict[str, Any], Callable[[], None]]:
    """Preserve a paid evaluator report before any later boundary can refuse it."""
    record = empty_record()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0, max_replay_examples=0, max_wall_seconds=LIVE_SECONDS
        ),
        monotonic,
    )
    binding = _return_budget_guard(budget)
    observed_elapsed = 0.0

    def guard() -> None:
        nonlocal observed_elapsed
        check_boundary()
        binding()
        budget.before_final()
        binding()
        observed_elapsed = budget.last_clock - budget.started_at

    payload = None
    try:
        guard()
        point, state = arm["states"][-1]
        if type(point) is not int or point != 128:
            raise ValueError("TRAIN A rollout requires the genuine benchmark128 endpoint")
        try:
            record["evaluator_admitted"] = True
            catalogue_options: dict[str, Any] = (
                {} if catalogue_method is None else {"catalogue_method": catalogue_method}
            )
            if wide_method is not None:
                catalogue_options["wide_method"] = wide_method
            result = evaluate_neural_representation_checkpoint(
                arm["source"],
                state,
                updates=128,
                budget=budget,
                check_boundary=guard,
                known_a_only=True,
                **catalogue_options,
            )
        except MiniGridEvaluationFailed as error:
            payload = pickle.dumps(error.report, protocol=5)
            raise
        # Why: a failing final boundary cannot erase already returned work.
        payload = pickle.dumps(result, protocol=5)
        guard()
        if result["status"] != "complete":
            raise ValueError("TRAIN A rollout did not complete")
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        if payload is not None:
            record["report"] = pickle.loads(payload)
        try:
            binding()
            budget.before_final()
            binding()
            observed_elapsed = budget.last_clock - budget.started_at
            record["timing_complete"] = True
        except Exception as error:
            record.update(
                status="failed", reason=f"{type(error).__name__}: {error}", timing_complete=False
            )
        try:
            # A final clock callback may alter preceding parent evidence.
            check_boundary()
        except Exception as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        # The last clock port is also a callback; qualify the binding afterward.
        try:
            binding()
            record["budget_binding_preserved"] = True
        except Exception as error:
            record.update(
                status="failed",
                reason=f"{type(error).__name__}: {error}",
                budget_binding_preserved=False,
            )
        record["charged_updates"] = budget.updates_completed
        record["replay_examples"] = budget.replay_examples_completed
        record["elapsed_seconds"] = observed_elapsed
    return record, binding


def add_work(work: dict[str, Any], record: dict[str, Any]) -> None:
    """Keep live prediction/physics counts separate from teacher-forced scores."""
    report = record["report"]
    paid = deepcopy(report["work"]) if report is not None else {}
    unknown = record.get("evaluator_admitted") is True and report is None
    if unknown:
        paid.update(
            environment_resets_completed=None,
            live_prediction_calls_returned=None,
            environment_steps_returned=None,
            internal_projection_rows_returned=None,
            internal_projection_array_bytes_returned=None,
            failure_work_may_be_unobserved=True,
            scope="evaluator admitted but no paid report returned; live work unknown",
        )
    paid.update(
        elapsed_seconds=record["elapsed_seconds"],
        charged_updates=record["charged_updates"],
        replay_examples=record["replay_examples"],
        timing_complete=record.get("timing_complete"),
    )
    work["train_a_rollout"] = paid
    work["learner_live_episodes"] = paid.get("environment_resets_completed", 0)
    work["live_prediction_forwards_completed"] = paid.get("live_prediction_calls_returned", 0)
    work.setdefault(
        "total_score_forwards_completed",
        work.get("known_raw_score_forwards_completed", 0)
        + work.get("probe_score_forwards_completed", 0),
    )
    work.setdefault(
        "total_score_rows_completed",
        work.get("known_raw_score_rows_completed", 0) + work.get("probe_score_rows_completed", 0),
    )
    work["total_prediction_forwards_completed"] = (
        None
        if unknown
        else (work["total_score_forwards_completed"] + work["live_prediction_forwards_completed"])
    )
    work["total_environment_steps_returned"] = (
        None
        if unknown
        else (work["source_environment_steps_returned"] + paid.get("environment_steps_returned", 0))
    )
    work["total_prediction_rows_completed"] = (
        None
        if unknown
        else (work["total_score_rows_completed"] + work["live_prediction_forwards_completed"])
    )
    if unknown:
        work["known_prediction_forwards_lower_bound"] = work["total_score_forwards_completed"]
        work["known_environment_steps_returned_lower_bound"] = work[
            "source_environment_steps_returned"
        ]
    work["failure_work_may_be_unobserved"] = work.get("failure_work_may_be_unobserved", False) or (
        paid.get("failure_work_may_be_unobserved", False)
        or record.get("timing_complete") is False
        or (record["status"] == "failed" and report is None)
    )
