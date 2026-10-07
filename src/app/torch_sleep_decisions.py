"""Describe Torch runner sleep scheduling and guard outcomes.

Inputs are an immutable schedule decision, core result, and measured guard
facts. Outputs are typed events; this module neither mutates the head nor
chooses guard/validation data or performs a rollback.
"""

from __future__ import annotations

from dataclasses import replace

from src.app.sleep_schedule import SleepAttemptDecision
from src.core.resnet50_variants import CircadianPredictiveCodingHead, SleepEventResult
from src.core.sleep_telemetry import (
    SleepBudgets,
    SleepDurations,
    SleepEventTelemetry,
    SleepGuardMetrics,
    SleepReplayUsage,
    SleepStructuralChanges,
)


def _attempt_trigger(decision: SleepAttemptDecision) -> str:
    if not decision.attempted:
        raise ValueError("attempted Torch sleep requires a due schedule decision")
    if decision.periodic_due and decision.adaptive_due:
        return "periodic_and_adaptive"
    return "periodic" if decision.periodic_due else "adaptive"


def describe_skipped_torch_sleep_decision(
    head: CircadianPredictiveCodingHead,
    decision: SleepAttemptDecision,
    *,
    completed_epoch: int,
    cooldown_suppressed: bool = False,
) -> SleepEventTelemetry:
    """Record a decision that never calls core sleep, with no proposed work."""
    if cooldown_suppressed:
        _attempt_trigger(decision)
        trigger, reason = "cooldown_suppressed", "rollback_cooldown"
    elif decision.attempted:
        raise ValueError("due Torch sleep must run or be cooldown-suppressed")
    elif head.config.sleep_mode == "disabled":
        trigger, reason = "disabled", "sleep_disabled"
    else:
        trigger, reason = "not_due", "schedule_not_due"
    chemistry = head.get_sleep_chemical_summaries()
    return SleepEventTelemetry(
        format_version=1,
        trigger_reason=trigger,
        outcome="skipped",
        reason=reason,
        completed_epoch=completed_epoch,
        wake_batches=head.get_sleep_clocks().wake_batches,
        budgets=SleepBudgets(0, 0, 0, None),
        changes=SleepStructuralChanges((), (), (), (), (), (), ()),
        before_width=head.hidden_dim,
        proposed_width=head.hidden_dim,
        final_width=head.hidden_dim,
        guard=None,
        replay=SleepReplayUsage(0, 0, 0, 0),
        chemistry_before=chemistry,
        chemistry_proposed=chemistry,
        chemistry_final=chemistry,
        durations=SleepDurations(0.0, 0.0),
    )


def describe_unguarded_torch_sleep_decision(
    decision: SleepAttemptDecision,
    result: SleepEventResult,
    *,
    completed_epoch: int,
    attempt_seconds: float,
) -> SleepEventTelemetry:
    """Attach the runner trigger and total attempt time to a core result."""
    trigger = _attempt_trigger(decision)
    core = result.telemetry
    if core is None or core.completed_epoch != completed_epoch:
        raise ValueError("Torch core telemetry must match the completed epoch")
    return replace(
        core,
        trigger_reason=trigger,
        durations=SleepDurations(
            core.durations.core_seconds,
            max(attempt_seconds, core.durations.core_seconds),
        ),
    )


def describe_guarded_torch_sleep_decision(
    decision: SleepAttemptDecision,
    result: SleepEventResult,
    *,
    completed_epoch: int,
    guard_role_hash: str | None,
    guard_role: str = "inner_guard",
    pre_accuracy: float,
    post_accuracy: float,
    pre_cross_entropy: float,
    post_cross_entropy: float,
    metric_name: str,
    tolerance: float,
    guard_examples: int,
    guard_examples_scored: int | None = None,
    accepted: bool,
    attempt_seconds: float,
) -> SleepEventTelemetry:
    """Keep core proposals while reflecting a two-pass outer guard decision."""
    core = describe_unguarded_torch_sleep_decision(
        decision, result, completed_epoch=completed_epoch, attempt_seconds=attempt_seconds
    )
    delta = (
        pre_accuracy - post_accuracy
        if metric_name == "accuracy"
        else post_cross_entropy - pre_cross_entropy
    )
    guard = SleepGuardMetrics(
        role=guard_role,
        metric_name=metric_name,
        pre_accuracy=pre_accuracy,
        post_accuracy=post_accuracy,
        pre_cross_entropy=pre_cross_entropy,
        post_cross_entropy=post_cross_entropy,
        delta=delta,
        tolerance=tolerance,
        examples_scored=(
            2 * guard_examples if guard_examples_scored is None else guard_examples_scored
        ),
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


def describe_failed_torch_sleep_decision(
    head: CircadianPredictiveCodingHead,
    decision: SleepAttemptDecision,
    *,
    completed_epoch: int,
    guard_role_hash: str | None,
    guard_role: str = "inner_guard",
    metric_name: str,
    tolerance: float,
    pre_accuracy: float | None,
    pre_cross_entropy: float | None,
    post_accuracy: float | None,
    post_cross_entropy: float | None,
    examples_scored: int,
    reason: str,
    attempt_seconds: float,
    result: SleepEventResult | None = None,
) -> SleepEventTelemetry:
    """Describe a restored failure, retaining any completed core proposal."""
    if reason not in {
        "inner_guard_pre_exception",
        "inner_guard_pre_nonfinite",
        "sleep_core_exception",
        "inner_guard_post_exception",
        "inner_guard_post_nonfinite",
        "inner_guard_delta_exception",
    }:
        raise ValueError("unknown Torch sleep failure reason")
    trigger = _attempt_trigger(decision)
    if result is None:
        # Why this: an interrupted core exposes no completed proposal or
        # measured core duration, even if it did transient work before restore.
        chemistry = head.get_sleep_chemical_summaries()
        core = SleepEventTelemetry(
            format_version=1,
            trigger_reason=trigger,
            outcome="error",
            reason=reason,
            completed_epoch=completed_epoch,
            wake_batches=head.get_sleep_clocks().wake_batches,
            budgets=SleepBudgets(0, 0, 0, None),
            changes=SleepStructuralChanges((), (), (), (), (), (), ()),
            before_width=head.hidden_dim,
            proposed_width=head.hidden_dim,
            final_width=head.hidden_dim,
            guard=None,
            replay=SleepReplayUsage(0, 0, 0, 0),
            chemistry_before=chemistry,
            chemistry_proposed=chemistry,
            chemistry_final=chemistry,
            durations=SleepDurations(0.0, attempt_seconds),
        )
    else:
        core = describe_unguarded_torch_sleep_decision(
            decision, result, completed_epoch=completed_epoch, attempt_seconds=attempt_seconds
        )
    guard = None
    if guard_role_hash is not None:
        delta = None
        if post_accuracy is not None:
            assert pre_accuracy is not None
            if metric_name == "accuracy":
                delta = pre_accuracy - post_accuracy
            else:
                assert pre_cross_entropy is not None and post_cross_entropy is not None
                delta = post_cross_entropy - pre_cross_entropy
        guard = SleepGuardMetrics(
            role=guard_role,
            metric_name=metric_name,
            pre_accuracy=pre_accuracy,
            post_accuracy=post_accuracy,
            pre_cross_entropy=pre_cross_entropy,
            post_cross_entropy=post_cross_entropy,
            delta=delta,
            tolerance=tolerance,
            examples_scored=examples_scored,
            role_hash=guard_role_hash,
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
