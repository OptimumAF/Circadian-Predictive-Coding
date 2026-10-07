"""Validate v6 sleep history against phase roles and the existing guard ledger.

The inputs are trusted checkpoint facts, arrived development roles, and phase
progress. This module does not train, score final tests, or choose a sleep policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.app.continual_arrived_checkpoint import arrived_sleep_history_digest
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles

if TYPE_CHECKING:
    from src.app.continual_arrived_benchmark import GuardDecision


def validate_arrived_sleep_history(
    events: tuple[SleepEventTelemetry, ...],
    decisions: tuple[GuardDecision, ...],
    *,
    role_event_digest: str,
    history_digest: str,
    phases: tuple[tuple[str, PhaseDecisionRoles, int, int, int], ...],
    tolerance: float,
    sleep_mode: str,
    pending_epoch: tuple[str, int] | None = None,
) -> None:
    """Require zero or more errors followed by one decision per resolved epoch."""
    if type(events) is not tuple or any(type(event) is not SleepEventTelemetry for event in events):
        raise ValueError("incompatible v6 sleep event history")
    for event in events:
        _validate_event_contract(event)
    try:
        digest = arrived_sleep_history_digest(events, role_event_digest)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("incompatible v6 sleep event history") from error
    if history_digest != digest:
        raise ValueError("incompatible v6 sleep history digest")

    guard_decisions = {(item.phase, item.epoch): item for item in decisions}
    event_index = 0
    for phase, roles, offset, completed, interval in phases:
        for local_epoch in range(1, completed + 1):
            event_index = _validate_epoch_attempts(
                events,
                event_index,
                phase=phase,
                roles=roles,
                expected_epoch=offset + local_epoch,
                local_epoch=local_epoch,
                interval=interval,
                tolerance=tolerance,
                sleep_mode=sleep_mode,
                decision=guard_decisions.get((phase, local_epoch)),
                resolved=True,
            )
        if pending_epoch is not None and pending_epoch[0] == phase:
            local_epoch = pending_epoch[1]
            if local_epoch != completed + 1:
                raise ValueError("incompatible v6 pending sleep cursor")
            event_index = _validate_epoch_attempts(
                events,
                event_index,
                phase=phase,
                roles=roles,
                expected_epoch=offset + local_epoch,
                local_epoch=local_epoch,
                interval=interval,
                tolerance=tolerance,
                sleep_mode=sleep_mode,
                decision=None,
                resolved=False,
            )
    if event_index != len(events):
        raise ValueError("incompatible v6 sleep event history")


def _validate_epoch_attempts(
    events: tuple[SleepEventTelemetry, ...],
    start: int,
    *,
    phase: str,
    roles: PhaseDecisionRoles,
    expected_epoch: int,
    local_epoch: int,
    interval: int,
    tolerance: float,
    sleep_mode: str,
    decision: GuardDecision | None,
    resolved: bool,
) -> int:
    index = start
    while index < len(events) and events[index].completed_epoch == expected_epoch:
        event = events[index]
        _validate_attempt_schedule(event, expected_epoch, local_epoch, interval, sleep_mode)
        if event.outcome != "error":
            if not resolved:
                raise ValueError("incompatible v6 pending sleep decision")
            _validate_resolved_guard(event, decision, roles, tolerance)
            return index + 1
        _validate_error_guard(event, roles, tolerance)
        index += 1
    if resolved:
        raise ValueError(f"incompatible v6 missing phase {phase} sleep decision")
    return index


def _validate_attempt_schedule(
    event: SleepEventTelemetry,
    expected_epoch: int,
    local_epoch: int,
    interval: int,
    sleep_mode: str,
) -> None:
    periodic_due = interval > 0 and local_epoch % interval == 0
    if (
        event.format_version != 1
        or event.completed_epoch != expected_epoch
        or event.wake_batches != expected_epoch
        or (
            sleep_mode == "disabled"
            and (event.trigger_reason != "disabled" or event.guard is not None)
        )
        or (
            sleep_mode != "disabled"
            and periodic_due
            and event.trigger_reason not in {"periodic", "periodic_and_adaptive"}
        )
        or (
            sleep_mode != "disabled"
            and not periodic_due
            and event.trigger_reason not in {"adaptive", "not_due"}
        )
    ):
        raise ValueError("incompatible v6 sleep phase or epoch")


def _validate_error_guard(
    event: SleepEventTelemetry, roles: PhaseDecisionRoles, tolerance: float
) -> None:
    guard = event.guard
    if (
        guard is None
        or event.trigger_reason in {"disabled", "not_due"}
        or event.reason
        not in {
            "inner_guard_pre_exception",
            "inner_guard_pre_nonfinite",
            "sleep_core_exception",
            "inner_guard_post_exception",
            "inner_guard_post_nonfinite",
        }
        or guard.role != "inner_guard"
        or guard.role_hash != roles.split_hashes["inner_guard"]
        or guard.metric_name != "accuracy"
        or guard.pre_cross_entropy is not None
        or guard.post_cross_entropy is not None
        or guard.post_accuracy is not None
        or guard.delta is not None
        or guard.tolerance != tolerance
        or guard.examples_scored
        != (len(roles.inner_guard.input) if guard.pre_accuracy is not None else 0)
        or (
            event.reason in {"inner_guard_pre_exception", "inner_guard_pre_nonfinite"}
            and guard.pre_accuracy is not None
        )
        or (
            event.reason
            in {"sleep_core_exception", "inner_guard_post_exception", "inner_guard_post_nonfinite"}
            and guard.pre_accuracy is None
        )
        or (
            event.reason
            in {"inner_guard_pre_exception", "inner_guard_pre_nonfinite", "sleep_core_exception"}
            and (
                event.changes.proposed_split_pairs
                or event.changes.proposed_prune_ids
                or event.changes.proposed_scheduled_prune_ids
                or event.changes.proposed_removed_prune_ids
                or event.replay.proposed_examples
                or event.replay.proposed_updates
            )
        )
    ):
        raise ValueError("incompatible v6 sleep error guard facts")


def _validate_resolved_guard(
    event: SleepEventTelemetry,
    decision: GuardDecision | None,
    roles: PhaseDecisionRoles,
    tolerance: float,
) -> None:
    guard = event.guard
    if guard is None:
        expected_reason = {
            "disabled": "sleep_disabled",
            "not_due": "schedule_not_due",
        }.get(event.trigger_reason)
        if decision is not None or event.outcome != "skipped" or event.reason != expected_reason:
            raise ValueError("incompatible v6 sleep guard absence")
        return
    expected_reason = {
        "accepted": "guard_accepted",
        "rolled_back": "guard_rejected",
    }.get(event.outcome)
    reason_valid = (
        event.reason == expected_reason
        if expected_reason is not None
        else event.outcome == "skipped"
        and event.reason in {"warmup", "zero_structural_budget", "adaptive_not_due"}
    )
    if (
        decision is None
        or event.trigger_reason in {"disabled", "not_due"}
        or guard.role != "inner_guard"
        or guard.role_hash != roles.split_hashes["inner_guard"]
        or guard.metric_name != "accuracy"
        or guard.pre_cross_entropy is not None
        or guard.post_cross_entropy is not None
        or guard.examples_scored != 2 * len(roles.inner_guard.input)
        or guard.tolerance != tolerance
        or guard.pre_accuracy != decision.accuracy_before
        or guard.post_accuracy != decision.accuracy_after
        or guard.delta != decision.accuracy_before - decision.accuracy_after
        or decision.accepted != (decision.accuracy_after + tolerance >= decision.accuracy_before)
        or (event.outcome == "rolled_back") != (not decision.accepted)
        or event.outcome not in {"accepted", "rolled_back", "skipped"}
        or not reason_valid
    ):
        raise ValueError("incompatible v6 sleep guard facts")


def _validate_event_contract(event: SleepEventTelemetry) -> None:
    """Recheck dataclass invariants that pickle does not run on loading."""
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
        raise ValueError("incompatible v6 sleep event facts") from error
