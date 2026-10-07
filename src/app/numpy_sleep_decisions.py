"""Attach NumPy runner scheduling decisions to model-owned sleep facts.

Inputs are the immutable schedule decision and optional core result. The
output is one typed event per epoch; this module neither mutates the model nor
chooses evaluation roles.
"""

from __future__ import annotations

from dataclasses import replace

from src.app.sleep_schedule import SleepAttemptDecision
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    SleepEventResult,
)
from src.core.sleep_telemetry import (
    SleepBudgets,
    SleepDurations,
    SleepEventTelemetry,
    SleepGuardMetrics,
    SleepReplayUsage,
    SleepStructuralChanges,
)


def describe_unguarded_numpy_sleep_decision(
    model: CircadianPredictiveCodingNetwork,
    decision: SleepAttemptDecision,
    *,
    completed_epoch: int,
    result: SleepEventResult | None = None,
    attempt_seconds: float = 0.0,
) -> SleepEventTelemetry:
    """Record an unguarded trigger without changing the core event or model."""
    if decision.attempted:
        if result is None or result.telemetry is None:
            raise ValueError("attempted NumPy sleep requires core telemetry")
        if result.telemetry.completed_epoch != completed_epoch:
            raise ValueError("NumPy sleep core telemetry has a different epoch")
        trigger = (
            "periodic_and_adaptive"
            if decision.periodic_due and decision.adaptive_due
            else "periodic"
            if decision.periodic_due
            else "adaptive"
        )
        core = result.telemetry
        return replace(
            core,
            trigger_reason=trigger,
            durations=SleepDurations(
                core.durations.core_seconds,
                max(attempt_seconds, core.durations.core_seconds),
            ),
        )
    if result is not None:
        raise ValueError("unscheduled NumPy sleep cannot have a core result")
    # Why this: a schedule miss must not call sleep_event, since doing so can
    # trigger adaptive sleep and change the historical training trajectory.
    chemistry = model.get_sleep_chemical_summaries()
    trigger = "disabled" if model.config.sleep_mode == "disabled" else "not_due"
    return SleepEventTelemetry(
        format_version=1,
        trigger_reason=trigger,
        outcome="skipped",
        reason="sleep_disabled" if trigger == "disabled" else "schedule_not_due",
        completed_epoch=completed_epoch,
        wake_batches=model.get_sleep_clocks().wake_batches,
        budgets=SleepBudgets(0, 0, 0, None),
        changes=SleepStructuralChanges((), (), (), (), (), (), ()),
        before_width=model.hidden_dim,
        proposed_width=model.hidden_dim,
        final_width=model.hidden_dim,
        guard=None,
        replay=SleepReplayUsage(0, 0, 0, 0),
        chemistry_before=chemistry,
        chemistry_proposed=chemistry,
        chemistry_final=chemistry,
        durations=SleepDurations(0.0, 0.0),
    )


def describe_guarded_numpy_sleep_decision(
    decision: SleepAttemptDecision,
    result: SleepEventResult,
    *,
    completed_epoch: int,
    guard_role_hash: str,
    accuracy_before: float,
    accuracy_after: float,
    tolerance: float,
    guard_examples: int,
    accepted: bool,
    attempt_seconds: float,
    model: CircadianPredictiveCodingNetwork,
) -> SleepEventTelemetry:
    """Keep the complete core proposal while reflecting an outer guard rollback."""
    core = describe_unguarded_numpy_sleep_decision(
        model,
        decision,
        completed_epoch=completed_epoch,
        result=result,
        attempt_seconds=attempt_seconds,
    )
    guard = SleepGuardMetrics(
        role="inner_guard",
        metric_name="accuracy",
        pre_accuracy=accuracy_before,
        post_accuracy=accuracy_after,
        pre_cross_entropy=None,
        post_cross_entropy=None,
        delta=accuracy_before - accuracy_after,
        tolerance=tolerance,
        examples_scored=2 * guard_examples,
        role_hash=guard_role_hash,
    )
    if accepted:
        if core.outcome == "skipped":
            return replace(core, guard=guard)
        return replace(core, outcome="accepted", reason="guard_accepted", guard=guard)
    return replace(
        core,
        outcome="rolled_back",
        reason="guard_rejected",
        changes=replace(
            core.changes,
            applied_split_pairs=(),
            applied_scheduled_prune_ids=(),
            applied_removed_prune_ids=(),
        ),
        final_width=core.before_width,
        replay=replace(core.replay, applied_examples=0, applied_updates=0),
        chemistry_final=core.chemistry_before,
        guard=guard,
    )


def describe_failed_guarded_numpy_sleep_decision(
    model: CircadianPredictiveCodingNetwork,
    decision: SleepAttemptDecision,
    *,
    completed_epoch: int,
    role_hash: str,
    accuracy_before: float | None,
    examples_scored: int,
    tolerance: float,
    reason: str,
    attempt_seconds: float,
    result: SleepEventResult | None = None,
) -> SleepEventTelemetry:
    """Record a restored failure, retaining any proposal the core returned."""
    if not decision.attempted:
        raise ValueError("a failed guarded attempt requires a due sleep decision")
    if result is None:
        # Why this: a core exception exposes no completed proposal or measured
        # core duration; zero budgets/work describe only what can be observed.
        chemistry = model.get_sleep_chemical_summaries()
        trigger = (
            "periodic_and_adaptive"
            if decision.periodic_due and decision.adaptive_due
            else "periodic"
            if decision.periodic_due
            else "adaptive"
        )
        core = SleepEventTelemetry(
            format_version=1,
            trigger_reason=trigger,
            outcome="error",
            reason=reason,
            completed_epoch=completed_epoch,
            wake_batches=model.get_sleep_clocks().wake_batches,
            budgets=SleepBudgets(0, 0, 0, None),
            changes=SleepStructuralChanges((), (), (), (), (), (), ()),
            before_width=model.hidden_dim,
            proposed_width=model.hidden_dim,
            final_width=model.hidden_dim,
            guard=None,
            replay=SleepReplayUsage(0, 0, 0, 0),
            chemistry_before=chemistry,
            chemistry_proposed=chemistry,
            chemistry_final=chemistry,
            durations=SleepDurations(0.0, attempt_seconds),
        )
    else:
        core = describe_unguarded_numpy_sleep_decision(
            model,
            decision,
            completed_epoch=completed_epoch,
            result=result,
            attempt_seconds=attempt_seconds,
        )
    guard = SleepGuardMetrics(
        role="inner_guard",
        metric_name="accuracy",
        pre_accuracy=accuracy_before,
        post_accuracy=None,
        pre_cross_entropy=None,
        post_cross_entropy=None,
        delta=None,
        tolerance=tolerance,
        examples_scored=examples_scored,
        role_hash=role_hash,
    )
    return replace(
        core,
        outcome="error",
        reason=reason,
        changes=replace(
            core.changes,
            applied_split_pairs=(),
            applied_scheduled_prune_ids=(),
            applied_removed_prune_ids=(),
        ),
        final_width=core.before_width,
        replay=replace(core.replay, applied_examples=0, applied_updates=0),
        chemistry_final=core.chemistry_before,
        guard=guard,
    )
