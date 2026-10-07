"""Validate completed unmatched-vision sleep history before checkpoint restore.

Inputs are a trusted payload's typed events, bounded runner counters, and the
current guard identity. Outputs are validation errors or no value. This module
does not open loaders, score guard data, restore models, or write checkpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.sleep_telemetry import SleepEventTelemetry

if TYPE_CHECKING:
    from src.app.resnet50_benchmark import ResNet50BenchmarkConfig


def selected_vision_guard_batch_sizes(loader: Any, max_batches: int | None) -> tuple[int, ...]:
    """Use the fixed evaluation loader's shape without consuming its RNG."""
    batch_size = loader.batch_size
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("vision sleep history requires fixed guard batches")
    batch_count = len(loader)
    sample_count = len(loader.dataset)
    if loader.drop_last:
        sizes = (batch_size,) * batch_count
    else:
        sizes = tuple(
            min(batch_size, sample_count - index * batch_size) for index in range(batch_count)
        )
    selected = sizes if max_batches is None else sizes[:max_batches]
    if not selected or any(size <= 0 for size in selected):
        raise ValueError("vision sleep history requires nonempty guard batches")
    return selected


def validate_vision_sleep_history(
    events: object,
    *,
    config: ResNet50BenchmarkConfig,
    guard_role: str,
    guard_role_hash: str,
    guard_batch_sizes: tuple[int, ...],
    resolved_epochs: int,
    pending_epoch: int | None = None,
    batches_per_epoch: int,
    sleep_attempts: int,
    sleep_rollbacks: int,
    sleep_splits: int,
    sleep_prunes: int,
    cooldown_suppressions: int,
) -> None:
    """Bind resolved decisions and preceding errors to runner counters."""
    if type(events) is not tuple or any(type(event) is not SleepEventTelemetry for event in events):
        raise ValueError("incompatible vision checkpoint sleep history length")
    if pending_epoch is not None and pending_epoch != resolved_epochs + 1:
        raise ValueError("incompatible vision checkpoint sleep history cursor")
    assert isinstance(events, tuple)
    resolved = 0
    for event in events:
        epoch = resolved + 1
        if (
            event.completed_epoch != epoch
            or event.wake_batches != epoch * batches_per_epoch
            or epoch > resolved_epochs + int(pending_epoch is not None)
        ):
            raise ValueError("incompatible vision checkpoint sleep history clock")
        _validate_event_contract(event)
        _validate_event_decision(
            event,
            config=config,
            guard_role=guard_role,
            guard_role_hash=guard_role_hash,
            guard_batch_sizes=guard_batch_sizes,
        )
        if event.outcome != "error":
            resolved += 1
            if resolved > resolved_epochs:
                raise ValueError("incompatible vision checkpoint sleep history resolution")
    if resolved != resolved_epochs or (
        pending_epoch is None and events and events[-1].outcome == "error"
    ):
        raise ValueError("incompatible vision checkpoint sleep history cursor")
    due_triggers = {"periodic", "adaptive", "periodic_and_adaptive"}
    if (
        sleep_attempts != sum(event.trigger_reason in due_triggers for event in events)
        or sleep_rollbacks != sum(event.outcome == "rolled_back" for event in events)
        or sleep_splits != sum(len(event.changes.applied_split_pairs) for event in events)
        or sleep_prunes != sum(len(event.changes.applied_removed_prune_ids) for event in events)
        or cooldown_suppressions
        != sum(event.trigger_reason == "cooldown_suppressed" for event in events)
    ):
        raise ValueError("incompatible vision checkpoint sleep history counters")


def _validate_event_contract(event: SleepEventTelemetry) -> None:
    """Pickle bypasses dataclass constructors, so recheck nested invariants."""
    try:
        event.budgets.__post_init__()
        event.changes.__post_init__()
        event.replay.__post_init__()
        event.durations.__post_init__()
        for chemistry in (
            event.chemistry_before,
            event.chemistry_proposed,
            event.chemistry_final,
        ):
            chemistry.__post_init__()
            for summary in (chemistry.primary, chemistry.fast, chemistry.slow):
                summary.__post_init__()
        if event.guard is not None:
            event.guard.__post_init__()
        event.__post_init__()
    except (AssertionError, AttributeError, TypeError, ValueError) as error:
        raise ValueError("incompatible vision checkpoint sleep history facts") from error


def _validate_event_decision(
    event: SleepEventTelemetry,
    *,
    config: ResNet50BenchmarkConfig,
    guard_role: str,
    guard_role_hash: str,
    guard_batch_sizes: tuple[int, ...],
) -> None:
    epoch = event.completed_epoch
    assert epoch is not None
    periodic_due = (
        config.circadian_sleep_mode != "disabled"
        and config.circadian_sleep_interval > 0
        and epoch % config.circadian_sleep_interval == 0
    )
    allowed_triggers = (
        {"disabled"}
        if config.circadian_sleep_mode == "disabled"
        else {"periodic", "periodic_and_adaptive", "cooldown_suppressed"}
        if periodic_due
        else {"adaptive", "not_due", "cooldown_suppressed"}
    )
    if event.trigger_reason not in allowed_triggers:
        raise ValueError("incompatible vision checkpoint sleep history trigger")
    absent_guard_reason = {
        "disabled": "sleep_disabled",
        "not_due": "schedule_not_due",
        "cooldown_suppressed": "rollback_cooldown",
    }.get(event.trigger_reason)
    if absent_guard_reason is not None:
        if (
            event.outcome != "skipped"
            or event.reason != absent_guard_reason
            or event.guard is not None
        ):
            raise ValueError("incompatible vision checkpoint sleep history skip")
        return
    guard = event.guard
    guard_examples_per_pass = sum(guard_batch_sizes)
    if config.circadian_enable_sleep_rollback:
        if (
            guard is None
            or guard.role != guard_role
            or guard.role_hash != guard_role_hash
            or guard.metric_name != config.circadian_sleep_rollback_metric
            or guard.tolerance != config.circadian_sleep_rollback_tolerance
        ):
            raise ValueError("incompatible vision checkpoint sleep history guard")
        if event.outcome == "error":
            _validate_failed_event(event, guard_batch_sizes)
            return
        if guard.examples_scored != 2 * guard_examples_per_pass:
            raise ValueError("incompatible vision checkpoint sleep history guard exposure")
        resolved_reason = {"accepted": "guard_accepted", "rolled_back": "guard_rejected"}.get(
            event.outcome
        )
        if resolved_reason is not None and event.reason != resolved_reason:
            raise ValueError("incompatible vision checkpoint sleep history reason")
        if resolved_reason is None and (
            event.outcome != "skipped"
            or event.reason not in {"warmup", "zero_structural_budget", "adaptive_not_due"}
        ):
            raise ValueError("incompatible vision checkpoint sleep history outcome")
    elif event.outcome == "error":
        if (
            guard is not None
            or event.reason != "sleep_core_exception"
            or _has_completed_core_proposal(event)
        ):
            raise ValueError("incompatible vision checkpoint sleep history unguarded error")
    elif (
        guard is not None
        or (event.outcome == "applied" and event.reason != "core_executed")
        or (
            event.outcome == "skipped"
            and event.reason not in {"warmup", "zero_structural_budget", "adaptive_not_due"}
        )
        or event.outcome not in {"applied", "skipped"}
    ):
        raise ValueError("incompatible vision checkpoint sleep history unguarded outcome")


def _validate_failed_event(event: SleepEventTelemetry, guard_batch_sizes: tuple[int, ...]) -> None:
    guard = event.guard
    guard_examples = sum(guard_batch_sizes)
    prefixes = {0}
    scored = 0
    for batch_size in guard_batch_sizes:
        scored += batch_size
        prefixes.add(scored)
    expected_counts = {
        "inner_guard_pre_exception": prefixes,
        "inner_guard_pre_nonfinite": {guard_examples},
        "sleep_core_exception": {guard_examples},
        "inner_guard_post_exception": {guard_examples + prefix for prefix in prefixes},
        "inner_guard_post_nonfinite": {2 * guard_examples},
        "inner_guard_delta_exception": {2 * guard_examples},
    }.get(event.reason)
    if guard is None or expected_counts is None or guard.examples_scored not in expected_counts:
        raise ValueError("incompatible vision checkpoint sleep history error exposure")
    has_pre_score = event.reason not in {"inner_guard_pre_exception", "inner_guard_pre_nonfinite"}
    has_post_score = event.reason == "inner_guard_delta_exception"
    if (
        (guard.pre_accuracy is not None) != has_pre_score
        or (guard.pre_cross_entropy is not None) != has_pre_score
        or (guard.post_accuracy is not None) != has_post_score
        or (guard.post_cross_entropy is not None) != has_post_score
        or (
            event.reason
            in {"inner_guard_pre_exception", "inner_guard_pre_nonfinite", "sleep_core_exception"}
            and _has_completed_core_proposal(event)
        )
    ):
        raise ValueError("incompatible vision checkpoint sleep history partial error facts")


def _has_completed_core_proposal(event: SleepEventTelemetry) -> bool:
    changes = event.changes
    budgets = event.budgets
    return bool(
        budgets.split_limit
        or budgets.prune_limit
        or budgets.replay_update_limit
        or budgets.time_limit_seconds is not None
        or changes.proposed_split_pairs
        or changes.proposed_prune_ids
        or changes.proposed_scheduled_prune_ids
        or changes.proposed_removed_prune_ids
        or event.replay.proposed_examples
        or event.replay.proposed_updates
        or event.proposed_width != event.before_width
        or event.chemistry_proposed != event.chemistry_before
        or event.durations.core_seconds
    )
