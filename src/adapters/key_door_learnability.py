"""Bounded reserved checkpoint probes for the existing key/door recipe.

Inputs are an owned checkpoint, reserved packets and original benchmark budgets.
Outputs retain probe curves, independent charges, isolation and partial failures.
This helper never drives the benchmark stream, selects data or promotes a model.
"""

from copy import deepcopy
from math import isfinite
from time import monotonic
from typing import Any, Callable

from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def probe_slots(*, checkpoint_updates: tuple[int, int] = (0, 8)) -> list[dict[str, Any]]:
    if (
        type(checkpoint_updates) is not tuple
        or len(checkpoint_updates) != 2
        or any(type(point) is not int for point in checkpoint_updates)
        or checkpoint_updates[0] != 0
        or not 1 <= checkpoint_updates[1] <= 512
    ):
        raise ValueError("probe checkpoints require an exact tuple (0, native integer in 1..512)")
    return [
        dict(
            checkpoint_updates=point,
            status="not_run",
            reason="not reached",
            report=None,
            charged_updates=0,
            source_unchanged=None,
            state_unchanged=None,
            budget_binding_preserved=None,
            total_seconds=None,
            readback_error=None,
            outcomes=None,
            native_updates_before=None,
            native_training_examples_before=None,
        )
        for point in checkpoint_updates
    ]


def _bindings(wake: ToyBudgetSession, replay: ToyBudgetSession | None) -> tuple[Any, ...]:
    return tuple(
        (
            id(session),
            id(session.budget),
            deepcopy(session.budget),
            id(session.clock),
            id(session.progress),
            session.started_at,
            session.updates_completed,
            session.replay_examples_completed,
            session.last_clock,
            type(session.updates_completed),
            type(session.replay_examples_completed),
            type(session.last_clock),
        )
        for session in (wake, replay)
        if session is not None
    )


def probe_outcomes(report: dict[str, Any]) -> dict[str, float]:
    points = {p["updates"]: p["scores"] for p in report["checkpoints"]}

    def mean(point: int, metric: str) -> float:
        return sum(points[point][f"E-{color}"][metric] for color in ("red", "blue")) / 2

    start, end = mean(0, "ce"), mean(16, "ce")
    return dict(
        initial_ce=start,
        final_ce=end,
        ce_gain=start - end,
        initial_accuracy=mean(0, "accuracy"),
        final_accuracy=mean(16, "accuracy"),
    )


def _require_decision_diagnostics(report: dict[str, Any], evaluation: Any) -> None:
    if report.get("decision_diagnostics_enabled") is not True:
        raise ValueError("requested probe decision diagnostics were not explicitly enabled")
    contexts = {batch.context for batch in evaluation}
    points = report.get("checkpoints")
    if type(points) is not list or not points:
        raise ValueError("requested probe decision diagnostics have no reported checkpoints")
    for point in points:
        details = point.get("decision_diagnostics") if type(point) is dict else None
        if type(details) is not dict or any(
            context not in details or type(details[context]) is not dict for context in contexts
        ):
            raise ValueError("requested probe decision diagnostics lack reserved context maps")


def run_checkpoint_probe(
    row: dict[str, Any],
    source: Any,
    state: Any,
    data: tuple[Any, Any],
    *,
    checkpoint: int,
    wake: ToyBudgetSession,
    replay: ToyBudgetSession | None,
    deadline: float,
    sampler: Any,
    same_state: Callable[[Any, Any], bool],
    probe_runner: Callable[..., dict[str, Any]],
    decision_diagnostics: bool = False,
    check_boundary: Callable[[], None] | None = None,
) -> None:
    if type(decision_diagnostics) is not bool:
        raise ValueError("decision_diagnostics must be an exact bool")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    slot = next(s for s in row["probes"] if s["checkpoint_updates"] == checkpoint)
    started = monotonic()
    budget, problem = None, None
    frozen, original, binding, probe_birth = None, None, None, None
    captured = False
    slot.update(status="running", reason=None)
    try:
        if check_boundary is not None:
            check_boundary()
        frozen, original = deepcopy(state), source.snapshot_state()
        if check_boundary is not None:
            check_boundary()
        binding = _bindings(wake, replay)
        captured = True
        slot.update(
            native_updates_before=state.completed_updates,
            native_training_examples_before=state.training_examples,
        )
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=16,
                max_wall_seconds=max(0.0, deadline - monotonic()),
                max_process_rss_bytes=768 * 1024**2,
            ),
            monotonic,
        )
        probe_birth = _bindings(budget, None)[0]
        budget.attach_memory(sampler)
        budget.before_final()
        probe_kwargs: dict[str, Any] = dict(budget=budget, checkpoint_updates=(0, 4, 8, 16))
        # Why: old injected runners keep their original keyword contract by default.
        if decision_diagnostics:
            probe_kwargs["decision_diagnostics"] = True
        if check_boundary is not None:
            probe_kwargs["check_boundary"] = check_boundary
            check_boundary()
        slot["report"] = probe_runner(source, state, *data, **probe_kwargs)
        if check_boundary is not None:
            check_boundary()
        if slot["report"]["status"] != "complete":
            raise RuntimeError("fresh-task probe failed: " + str(slot["report"]["reason"]))
        if decision_diagnostics:
            _require_decision_diagnostics(slot["report"], data[1])
        slot.update(status="complete", outcomes=probe_outcomes(slot["report"]))
    except Exception as error:
        problem = error
        slot.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        slot["charged_updates"] = 0 if budget is None else budget.updates_completed
        slot["total_seconds"] = monotonic() - started
        try:
            if captured:
                if check_boundary is not None:
                    check_boundary()
                slot["source_unchanged"] = same_state(original, source.snapshot_state())
                if check_boundary is not None:
                    check_boundary()
                slot["state_unchanged"] = same_state(frozen, state)
                slot["budget_binding_preserved"] = binding == _bindings(
                    wake, replay
                ) and _probe_budget_preserved(budget, probe_birth, slot["report"])
            if captured and not all(
                slot[k] is True
                for k in ("source_unchanged", "state_unchanged", "budget_binding_preserved")
            ):
                raise ValueError(
                    "probe changed source/checkpoint/original probe or benchmark budget"
                )
            if check_boundary is not None:
                check_boundary()
        except Exception as error:
            slot["readback_error"] = f"readback {type(error).__name__}: {error}"
            if problem is None:
                problem = error
                slot.update(status="failed", reason=slot["readback_error"])
    if problem is not None:
        raise problem


def _probe_budget_preserved(budget: ToyBudgetSession | None, birth: Any, report: Any) -> bool:
    if budget is None:
        return True  # No probe budget was created or used.
    return (
        birth is not None
        and birth[:6] == _bindings(budget, None)[0][:6]
        and type(budget.last_clock) is float
        and isfinite(budget.last_clock)
        and budget.last_clock >= birth[8]
        and type(budget.updates_completed) is int
        and budget.updates_completed >= 0
        and type(budget.replay_examples_completed) is int
        and budget.replay_examples_completed == 0
        and (
            report is None
            or (
                type(report["completed_updates"]) is int
                and budget.updates_completed == report["completed_updates"]
            )
        )
    )


def valid_probe(p: dict[str, Any]) -> bool:
    """The existing completed-cohort contract, shared with preceding-work admission."""
    return (
        p["status"] == "complete"
        and p["charged_updates"] == 16
        and p["source_unchanged"] is True
        and p["state_unchanged"] is True
        and p["budget_binding_preserved"] is True
        and p["report"]["num_classes"] == 5
        and p["report"]["completed_updates"] == 16
        and p["report"]["completed_training_examples"] == 128
        and p["report"]["score_forward_calls_completed"] == 8
        and [c["updates"] for c in p["report"]["checkpoints"]] == [0, 4, 8, 16]
    )


def valid_probes(row: dict[str, Any]) -> bool:
    return len(row["probes"]) == 2 and all(valid_probe(p) for p in row["probes"])
