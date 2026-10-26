"""Bounded TRAIN diagnostics after the completed warm/reward runtime.

Inputs are two owned128/136 snapshots, original TRAIN sources and a completed
training guard. Outputs retain two existing evaluator reports and separately
charged raw/live work. This module grants no update, final/probe access, replay,
selection, recovery or causal reward-effect claim.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from math import fsum
import pickle
from time import monotonic
from typing import Any

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.key_door_reward_pilot import _binding, _check_budget
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed, evaluate_checkpoint
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_key_door import predefined_layouts

RSS_LIMIT = 1024**3
WORK_KEYS = (
    "raw_score_forwards_attempted",
    "raw_score_forwards_completed",
    "raw_score_rows_attempted",
    "raw_score_rows_completed",
    "live_prediction_calls_attempted",
    "live_prediction_calls_returned",
    "live_sessions_started",
    "environment_resets_attempted",
    "environment_resets_completed",
    "environment_steps_attempted",
    "environment_steps_returned",
)


def new_train_diagnostics(*, momentum_control: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = dict(
        status="not_run",
        reason="not reached",
        training_phase=None,
        readout_budget=None,
        readouts=[
            dict(method="recurrent_warm_reward", updates=point, role="train", status="not_run")
            for point in (128, 136)
        ],
        work=None,
        endpoint_changes=None,
        endpoint_conditions=None,
    )

    if momentum_control:
        report["readouts"].append(
            dict(
                method="recurrent_warm_momentum_control",
                updates=136,
                role="train",
                status="not_run",
            )
        )
        report.update(
            control_endpoint_changes=None,
            control_endpoint_conditions=None,
            control_minus_reward=None,
        )
    return report


def _work(records, failed: bool) -> dict[str, Any]:
    work = {key: sum(record.get("work", {}).get(key, 0) for record in records) for key in WORK_KEYS}
    work["failure_work_may_be_unobserved"] = failed or any(
        record.get("work", {}).get("failure_work_may_be_unobserved", False) for record in records
    )
    work["observed_recurrent_unroll_steps"] = (
        work["raw_score_rows_completed"] + work["live_prediction_calls_returned"]
    )
    return work


def _endpoints(records):
    changes, conditions = {}, {}
    for context in ("A", "B", "C_train"):
        raw = [
            [score for score in record["scores"] if score["context"] == context]
            for record in records
        ]
        live = [
            [episode for episode in record["episodes"] if episode["context"] == context]
            for record in records
        ]
        if any(len(values) != 2 for values in (*raw, *live)):
            raise ValueError("TRAIN endpoints require both cues in every declared context")
        changes[context] = {
            "ce_gain": fsum(s["ce"] for s in raw[0]) / 2 - fsum(s["ce"] for s in raw[1]) / 2,
            **{
                f"{key}_gain": fsum(s[key] for s in raw[1]) / 2 - fsum(s[key] for s in raw[0]) / 2
                for key in ("accuracy", "exact_sequence_accuracy")
            },
            "live_success_gain": (
                sum(e["success"] for e in live[1]) - sum(e["success"] for e in live[0])
            )
            / 2,
            "mean_return_gain": fsum(e["return"] for e in live[1]) / 2
            - fsum(e["return"] for e in live[0]) / 2,
        }
        initial_raw = all(s["exact_sequence_accuracy"] == 1 for s in raw[0])
        initial_live = all(e["success"] is True for e in live[0])
        final_raw = all(s["exact_sequence_accuracy"] == 1 for s in raw[1])
        final_live = all(e["success"] is True for e in live[1])
        conditions[context] = dict(
            initial_raw_both_cues_exact=initial_raw,
            initial_live_both_cues_goals=initial_live,
            initial_joint_acquisition=initial_raw and initial_live,
            final_raw_both_cues_exact=final_raw,
            final_live_both_cues_goals=final_live,
            final_joint_acquisition=final_raw and final_live,
            interpretation="endpoint_only; initial joint acquisition is a required retention premise; no within-stream recovery or causal RL benefit inferred",
        )
    return changes, conditions


def run_train_diagnostics(
    report: dict[str, Any],
    source: Any,
    states: tuple[tuple[int, Any], ...],
    sources: tuple,
    memory: Any,
    deadline: float,
    *,
    training_phase: dict[str, Any],
    check_boundary: Callable[[], None],
    control_state: Any = None,
    extended_control_readouts: bool = False,
) -> dict[str, Any]:
    """Keep the completed training clock frozen while a zero-update budget scores."""
    if (
        type(report) is not dict
        or type(states) is not tuple
        or len(states) != 2
        or any(type(pair) is not tuple or len(pair) != 2 for pair in states)
        or tuple(point for point, _ in states) != (128, 136)
        or any(type(point) is not int for point, _ in states)
        or not callable(check_boundary)
    ):
        raise ValueError("TRAIN diagnostics require two owned128/136 endpoints and a caller guard")
    if type(extended_control_readouts) is not bool or (
        extended_control_readouts and control_state is None
    ):
        raise ValueError(
            "extended_control_readouts requires an owned control endpoint and exact bool"
        )
    if control_state is not None:
        states = (*states, (136, control_state))
    report.clear()
    report.update(new_train_diagnostics(momentum_control=control_state is not None))
    report.update(status="running", reason=None, training_phase=deepcopy(training_phase))
    phase = report["training_phase"]
    phase_payload = pickle.dumps(phase, protocol=5)
    records: list[dict[str, Any]] = report["readouts"]
    owners = list(records)
    payloads = [pickle.dumps(record, protocol=5) for record in records]
    expected_states = deepcopy(states)
    source_signature = shared._signature(sources)
    source_before = None
    budget: Any = None
    birth = None
    last_clock = None
    frozen_clock = None
    completed_payload = None
    primary_error = None

    def evidence() -> None:
        if (
            report.get("training_phase") is not phase
            or pickle.dumps(phase, protocol=5) != phase_payload
            or report.get("readouts") is not records
            or len(records) != len(states)
            or any(
                record is not owner or pickle.dumps(record, protocol=5) != payload
                for record, owner, payload in zip(records, owners, payloads, strict=True)
            )
            or not _same_state(expected_states, states)
            or not shared._check_signature(sources, source_signature)
            or (
                completed_payload is not None
                and pickle.dumps(report, protocol=5) != completed_payload
            )
        ):
            raise ValueError("TRAIN diagnostic checkpoint/source/preceding evidence changed")
        if budget is not None:
            _check_budget(budget, birth, 0, last_clock)
            if budget.process_rss_sampler is not memory or (
                frozen_clock is not None and budget.last_clock != frozen_clock
            ):
                raise ValueError("original readout budget or completed clock changed")

    def proof() -> None:
        evidence()
        if source_before is not None and not _same_state(source_before, source.snapshot_state()):
            raise ValueError("TRAIN diagnostic source full learned state changed")
        # The snapshot port may invoke callbacks. Repeat the pure record/input/
        # state/budget proof after that last external read, before admitting work.
        evidence()

    def guard() -> None:
        nonlocal last_clock
        check_boundary()
        if monotonic() >= deadline:
            raise ValueError("original whole-case TRAIN readout deadline exhausted")
        proof()
        if budget is not None and frozen_clock is None:
            budget.before_final()
            _check_budget(budget, birth, 0, last_clock)
            last_clock = budget.last_clock
        check_boundary()
        proof()

    def budget_record():
        if budget is None:
            return None
        return dict(
            limits=asdict(budget.budget),
            started_at=budget.started_at,
            last_clock=budget.last_clock,
            elapsed_seconds=budget.last_clock - budget.started_at,
            updates_completed=budget.updates_completed,
            replay_examples_completed=budget.replay_examples_completed,
            process_rss=None
            if budget.process_rss_segment is None
            else asdict(budget.process_rss_segment),
        )

    try:
        guard()
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_wall_seconds=min(
                    210.0
                    if extended_control_readouts
                    else (90.0 if control_state is not None else 60.0),
                    max(0.0, deadline - monotonic()),
                ),
                max_process_rss_bytes=RSS_LIMIT,
            ),
            monotonic,
        )
        budget.attach_memory(memory)
        birth = _binding(budget)
        last_clock = budget.last_clock
        source_before = deepcopy(source.snapshot_state())
        guard()
        for index, (point, state) in enumerate(states):
            guard()
            try:
                current = evaluate_checkpoint(
                    source,
                    state,
                    method=records[index]["method"],
                    updates=point,
                    native_training_examples=state.training_examples,
                    layouts=predefined_layouts()[:3],
                    role="train",
                    sources=sources,
                    budget=budget,
                    check_boundary=guard,
                )
            except MiniGridEvaluationFailed as error:
                current = error.report
            records[index] = owners[index] = current
            payloads[index] = pickle.dumps(current, protocol=5)
            if current["status"] != "complete":
                raise ValueError("TRAIN diagnostic readout refused; paid prefix retained")
            guard()
        budget.complete_memory()
        guard()
        frozen_clock = budget.last_clock
    except Exception as error:
        primary_error = f"{type(error).__name__}: {error}"
    try:
        if budget is not None:
            budget.record_memory_snapshot()
        recorded_budget = budget_record()
        guard()
    except Exception as error:
        recorded_budget = budget_record()
        primary_error = primary_error or f"{type(error).__name__}: {error}"
    # All native/caller readbacks finish before publishing private paid records.
    canonical = [pickle.loads(payload) for payload in payloads]
    changes = conditions = None
    control_changes = control_conditions = comparison = None
    if primary_error is None:
        try:
            changes, conditions = _endpoints(canonical[:2])
            if control_state is not None:
                control_changes, control_conditions = _endpoints((canonical[0], canonical[2]))
                comparison, _ = _endpoints((canonical[1], canonical[2]))
        except Exception as error:
            changes = conditions = control_changes = control_conditions = comparison = None
            primary_error = f"{type(error).__name__}: {error}"
    report.update(
        status="complete" if primary_error is None else "incomplete",
        reason=primary_error,
        training_phase=pickle.loads(phase_payload),
        readout_budget=recorded_budget,
        readouts=canonical,
        work=_work(canonical, primary_error is not None),
        endpoint_changes=changes,
        endpoint_conditions=conditions,
    )
    if control_state is not None:
        report.update(
            control_endpoint_changes=control_changes,
            control_endpoint_conditions=control_conditions,
            control_minus_reward=comparison,
        )
    if primary_error is not None:
        raise RuntimeError(primary_error)
    records = report["readouts"]
    owners = list(records)
    phase = report["training_phase"]
    completed_payload = pickle.dumps(report, protocol=5)
    return dict(guard=guard)
