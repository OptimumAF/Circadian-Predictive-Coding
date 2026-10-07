"""Runner-owned state and compatibility checks for fixed-feature resume.

Inputs are a combined circadian checkpoint and hashes of materialized
training, guard, and validation batches. File encoding belongs to infra;
this module does not read final-test batches or train a head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from math import isfinite
from typing import Any, Protocol

from src.app.circadian_checkpoint import CircadianResumePosition, CircadianRunCheckpoint
from src.app.sleep_schedule import SleepRollbackCooldownState
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig
from src.core.sleep_telemetry import SleepEventTelemetry
from src.shared.process_memory import ProcessRssSegment


@dataclass(frozen=True)
class FixedFeatureCircadianProgress:
    """Counters needed to continue the current circadian training report."""

    completed_epochs: int = 0
    seen_samples: int = 0
    wake_batches: int = 0
    sleep_attempts: int = 0
    total_splits: int = 0
    total_prunes: int = 0
    total_rollbacks: int = 0
    rollback_guard_examples: int = 0
    initial_width: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class CudaAllocatorSegment:
    """One process/device invocation's allocator baseline and high-water marks."""

    pid: int
    device: str
    allocated_start_bytes: int
    reserved_start_bytes: int
    allocated_peak_bytes: int
    reserved_peak_bytes: int


@dataclass(frozen=True)
class FixedFeatureCircadianCheckpoint:
    """One trusted-file payload for a fixed-feature circadian head."""

    format_version: int
    protocol_id: str
    runner_config_digest: str
    initial_head_hash: str
    feature_hashes: tuple[tuple[str, str], ...]
    split_hashes: tuple[tuple[str, str], ...]
    progress: FixedFeatureCircadianProgress
    combined: CircadianRunCheckpoint
    sleep_events: tuple[SleepEventTelemetry, ...] = ()
    memory_segments: tuple[ProcessRssSegment, ...] = ()
    cuda_allocator_segments: tuple[CudaAllocatorSegment, ...] = ()


class FixedFeatureCheckpointStore(Protocol):
    """App port for one replaceable local checkpoint file."""

    def load(self) -> FixedFeatureCircadianCheckpoint: ...

    def save(self, checkpoint: FixedFeatureCircadianCheckpoint) -> None: ...


def fixed_feature_config_digest(
    config: ResNet50BenchmarkConfig,
    protocol_id: str,
    *,
    wall_time_budget_seconds: float | None = None,
) -> str:
    """Bind all runner settings, including settings outside the head config."""
    identity: dict[str, Any] = {"protocol_id": protocol_id, "runner_config": asdict(config)}
    if wall_time_budget_seconds is not None:
        identity["wall_time_budget_seconds"] = wall_time_budget_seconds
    payload = dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def fixed_feature_data_digest(
    feature_hashes: tuple[tuple[str, str], ...],
    split_hashes: tuple[tuple[str, str], ...],
) -> str:
    """Bind the cached training roles without opening final-test features."""
    payload = dumps(
        {"feature_hashes": feature_hashes, "split_hashes": split_hashes},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_fixed_feature_checkpoint(
    checkpoint: FixedFeatureCircadianCheckpoint,
    *,
    protocol_id: str,
    runner_config_digest: str,
    initial_head_hash: str,
    feature_hashes: tuple[tuple[str, str], ...],
    split_hashes: tuple[tuple[str, str], ...],
    batch_sizes: tuple[int, ...],
    guard_batch_sizes: tuple[int, ...],
    epochs: int,
    initial_width: int,
    config: ResNet50BenchmarkConfig,
    memory_sample_interval_seconds: float | None = None,
    cuda_memory_device: str | None = None,
) -> FixedFeatureCircadianProgress:
    """Check runner identity and counters before restoring the live head."""
    if (
        not isinstance(checkpoint, FixedFeatureCircadianCheckpoint)
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != 2
    ):
        raise ValueError("incompatible fixed-feature checkpoint format")
    if (
        checkpoint.protocol_id != protocol_id
        or checkpoint.runner_config_digest != runner_config_digest
        or checkpoint.initial_head_hash != initial_head_hash
        or checkpoint.feature_hashes != feature_hashes
        or checkpoint.split_hashes != split_hashes
    ):
        raise ValueError("incompatible fixed-feature checkpoint config or features")
    memory_segments = getattr(checkpoint, "memory_segments", None)
    if memory_sample_interval_seconds is None:
        if memory_segments != ():
            raise ValueError("incompatible fixed-feature checkpoint memory segments")
    elif (
        not isinstance(memory_segments, tuple)
        or not memory_segments
        or any(
            not _valid_memory_segment(segment, memory_sample_interval_seconds)
            for segment in memory_segments
        )
    ):
        raise ValueError("incompatible fixed-feature checkpoint memory segments")
    cuda_segments = getattr(checkpoint, "cuda_allocator_segments", None)
    if cuda_memory_device is None:
        if cuda_segments != ():
            raise ValueError("incompatible fixed-feature checkpoint CUDA allocator segments")
    elif (
        not isinstance(cuda_segments, tuple)
        or not cuda_segments
        or not isinstance(memory_segments, tuple)
        or len(cuda_segments) != len(memory_segments)
        or any(
            not _valid_cuda_segment(segment, cuda_memory_device) or segment.pid != rss.pid
            for segment, rss in zip(cuda_segments, memory_segments, strict=True)
        )
    ):
        raise ValueError("incompatible fixed-feature checkpoint CUDA allocator segments")
    progress = checkpoint.progress
    if not isinstance(progress, FixedFeatureCircadianProgress):
        raise ValueError("incompatible fixed-feature checkpoint progress")
    counters = (
        progress.completed_epochs,
        progress.seen_samples,
        progress.wake_batches,
        progress.sleep_attempts,
        progress.total_splits,
        progress.total_prunes,
        progress.total_rollbacks,
        progress.rollback_guard_examples,
    )
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("incompatible fixed-feature checkpoint counters")
    if (
        type(progress.initial_width) is not int
        or progress.initial_width != initial_width
        or type(progress.elapsed_seconds) not in {int, float}
        or not isfinite(progress.elapsed_seconds)
        or progress.elapsed_seconds < 0.0
        or progress.total_rollbacks > progress.sleep_attempts
    ):
        raise ValueError("incompatible fixed-feature checkpoint report state")
    if not isinstance(checkpoint.combined, CircadianRunCheckpoint):
        raise ValueError("incompatible fixed-feature combined checkpoint")
    retry_state = checkpoint.combined.retry_state
    if (
        not isinstance(retry_state, SleepRollbackCooldownState)
        or retry_state.rejected_attempts != progress.total_rollbacks
    ):
        raise ValueError("incompatible fixed-feature checkpoint rollback counters")
    position = checkpoint.combined.position
    if not isinstance(position, CircadianResumePosition):
        raise ValueError("incompatible fixed-feature checkpoint position")
    if position.stage == "wake":
        if not 0 <= position.completed_epoch < epochs or not (
            1 <= position.next_batch_index <= len(batch_sizes)
        ):
            raise ValueError("incompatible fixed-feature checkpoint wake cursor")
        complete_wake_epochs = position.completed_epoch
        prefix_batches = position.next_batch_index
        report_epochs = complete_wake_epochs
    elif position.stage in {"before_sleep", "after_sleep"}:
        if not 1 <= position.completed_epoch <= epochs or position.next_batch_index != 0:
            raise ValueError("incompatible fixed-feature checkpoint sleep cursor")
        complete_wake_epochs = position.completed_epoch
        prefix_batches = 0
        report_epochs = complete_wake_epochs - 1
    else:
        raise ValueError("incompatible fixed-feature checkpoint stage")
    expected_batches = complete_wake_epochs * len(batch_sizes) + prefix_batches
    expected_samples = complete_wake_epochs * sum(batch_sizes) + sum(batch_sizes[:prefix_batches])
    if (
        progress.completed_epochs != report_epochs
        or progress.wake_batches != expected_batches
        or progress.seen_samples != expected_samples
        or position.wake_batches != expected_batches
    ):
        raise ValueError("incompatible fixed-feature checkpoint batch or report counters")
    _validate_sleep_events(
        checkpoint.sleep_events,
        position=position,
        progress=progress,
        retry_state=retry_state,
        batches_per_epoch=len(batch_sizes),
        guard_hash=dict(feature_hashes)["guard"],
        selected_guard_batch_sizes=(
            guard_batch_sizes[: config.circadian_sleep_rollback_eval_batches]
            if config.circadian_sleep_rollback_eval_batches > 0
            else guard_batch_sizes
        ),
        config=config,
    )
    return progress


def _validate_sleep_events(
    events: object,
    *,
    position: CircadianResumePosition,
    progress: FixedFeatureCircadianProgress,
    retry_state: SleepRollbackCooldownState,
    batches_per_epoch: int,
    guard_hash: str,
    selected_guard_batch_sizes: tuple[int, ...],
    config: ResNet50BenchmarkConfig,
) -> None:
    """Bind resolved and failed attempts to a retryable cursor before restore."""
    expected_resolved = position.completed_epoch - int(position.stage == "before_sleep")
    if type(events) is not tuple or any(type(event) is not SleepEventTelemetry for event in events):
        raise ValueError("incompatible fixed-feature checkpoint sleep history")
    due_triggers = {"periodic", "adaptive", "periodic_and_adaptive"}
    assert isinstance(events, tuple)
    resolved = 0
    for event in events:
        epoch = resolved + 1
        if (
            event.completed_epoch != epoch
            or event.wake_batches != epoch * batches_per_epoch
            or epoch > position.completed_epoch
        ):
            raise ValueError("incompatible fixed-feature checkpoint sleep event clock or order")
        _validate_sleep_event_contract(event)
        _validate_sleep_event_decision(
            event,
            config=config,
            guard_hash=guard_hash,
            selected_guard_batch_sizes=selected_guard_batch_sizes,
        )
        if event.outcome != "error":
            resolved += 1
            if resolved > expected_resolved:
                raise ValueError("incompatible fixed-feature checkpoint resolved sleep history")
    if resolved != expected_resolved or (
        position.stage != "before_sleep" and events and events[-1].outcome == "error"
    ):
        raise ValueError("incompatible fixed-feature checkpoint sleep cursor history")
    attempts = sum(event.trigger_reason in due_triggers for event in events)
    rollbacks = sum(event.outcome == "rolled_back" for event in events)
    suppressions = sum(event.trigger_reason == "cooldown_suppressed" for event in events)
    guard_examples = sum(
        event.guard.examples_scored
        for event in events
        if event.outcome != "error" and event.guard is not None
    )
    splits = sum(len(event.changes.applied_split_pairs) for event in events)
    prunes = sum(len(event.changes.applied_removed_prune_ids) for event in events)
    if (
        progress.sleep_attempts != attempts
        or progress.total_rollbacks != rollbacks
        or progress.rollback_guard_examples != guard_examples
        or progress.total_splits != splits
        or progress.total_prunes != prunes
        or retry_state.suppressed_due_attempts != suppressions
    ):
        raise ValueError("incompatible fixed-feature checkpoint sleep counters")


def _validate_sleep_event_contract(event: SleepEventTelemetry) -> None:
    """Recheck invariants bypassed when a trusted pickle rehydrates a dataclass."""
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
        raise ValueError("incompatible fixed-feature checkpoint sleep event facts") from error


def _validate_sleep_event_decision(
    event: SleepEventTelemetry,
    *,
    config: ResNet50BenchmarkConfig,
    guard_hash: str,
    selected_guard_batch_sizes: tuple[int, ...],
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
        raise ValueError("incompatible fixed-feature checkpoint sleep trigger")
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
            raise ValueError("incompatible fixed-feature checkpoint sleep skip")
        return
    guard = event.guard
    if guard is not None and (
        not config.circadian_enable_sleep_rollback
        or guard.role != "inner_guard"
        or guard.role_hash != guard_hash
        or guard.metric_name != config.circadian_sleep_rollback_metric
        or guard.tolerance != config.circadian_sleep_rollback_tolerance
    ):
        raise ValueError("incompatible fixed-feature checkpoint sleep guard role or metric")
    guard_examples_per_pass = sum(selected_guard_batch_sizes)
    if event.outcome == "error":
        _validate_failed_sleep_event(event, config, selected_guard_batch_sizes)
        return
    if config.circadian_enable_sleep_rollback:
        if guard is None or guard.examples_scored != 2 * guard_examples_per_pass:
            raise ValueError("incompatible fixed-feature checkpoint completed guard exposure")
        reason = {"accepted": "guard_accepted", "rolled_back": "guard_rejected"}.get(event.outcome)
        if reason is None:
            if event.outcome != "skipped" or event.reason not in {
                "warmup",
                "zero_structural_budget",
                "adaptive_not_due",
            }:
                raise ValueError("incompatible fixed-feature checkpoint guarded sleep outcome")
        elif event.reason != reason:
            raise ValueError("incompatible fixed-feature checkpoint guarded sleep reason")
    elif (
        guard is not None
        or (event.outcome == "applied" and event.reason != "core_executed")
        or (
            event.outcome == "skipped"
            and event.reason not in {"warmup", "zero_structural_budget", "adaptive_not_due"}
        )
        or event.outcome not in {"applied", "skipped"}
    ):
        raise ValueError("incompatible fixed-feature checkpoint unguarded sleep outcome")


def _validate_failed_sleep_event(
    event: SleepEventTelemetry,
    config: ResNet50BenchmarkConfig,
    selected_guard_batch_sizes: tuple[int, ...],
) -> None:
    guard = event.guard
    if not config.circadian_enable_sleep_rollback:
        if (
            guard is not None
            or event.reason != "sleep_core_exception"
            or _has_completed_core_proposal(event)
        ):
            raise ValueError("incompatible fixed-feature checkpoint unguarded sleep error")
        return
    guard_examples_per_pass = sum(selected_guard_batch_sizes)
    prefix_counts = {0}
    scored = 0
    for batch_size in selected_guard_batch_sizes:
        scored += batch_size
        prefix_counts.add(scored)
    expected_counts = {
        "inner_guard_pre_exception": prefix_counts,
        "inner_guard_pre_nonfinite": {guard_examples_per_pass},
        "sleep_core_exception": {guard_examples_per_pass},
        "inner_guard_post_exception": {
            guard_examples_per_pass + prefix for prefix in prefix_counts
        },
        "inner_guard_post_nonfinite": {2 * guard_examples_per_pass},
        "inner_guard_delta_exception": {2 * guard_examples_per_pass},
    }.get(event.reason)
    if guard is None or expected_counts is None or guard.examples_scored not in expected_counts:
        raise ValueError("incompatible fixed-feature checkpoint sleep error exposure")
    has_pre_score = event.reason not in {"inner_guard_pre_exception", "inner_guard_pre_nonfinite"}
    has_post_score = event.reason == "inner_guard_delta_exception"
    if (
        (guard.pre_accuracy is not None) != has_pre_score
        or (guard.post_accuracy is not None) != has_post_score
        or (guard.pre_cross_entropy is not None) != has_pre_score
        or (guard.post_cross_entropy is not None) != has_post_score
        or (
            event.reason
            in {"inner_guard_pre_exception", "inner_guard_pre_nonfinite", "sleep_core_exception"}
            and _has_completed_core_proposal(event)
        )
    ):
        raise ValueError("incompatible fixed-feature checkpoint partial sleep error facts")


def _has_completed_core_proposal(event: SleepEventTelemetry) -> bool:
    """A pre/core exception cannot claim a core result it never returned."""
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


def _valid_memory_segment(segment: object, interval_seconds: float) -> bool:
    return (
        isinstance(segment, ProcessRssSegment)
        and type(segment.pid) is int
        and segment.pid > 0
        and type(segment.start_bytes) is int
        and segment.start_bytes >= 0
        and type(segment.peak_bytes) is int
        and segment.peak_bytes >= segment.start_bytes
        and type(segment.sample_count) is int
        and segment.sample_count >= 1
        and type(segment.interval_seconds) is float
        and segment.interval_seconds == interval_seconds
    )


def _valid_cuda_segment(segment: object, device: str) -> bool:
    if not isinstance(segment, CudaAllocatorSegment):
        return False
    values = (
        segment.allocated_start_bytes,
        segment.reserved_start_bytes,
        segment.allocated_peak_bytes,
        segment.reserved_peak_bytes,
    )
    return (
        type(segment.pid) is int
        and segment.pid > 0
        and segment.device == device
        and all(type(value) is int and value >= 0 for value in values)
        and segment.allocated_peak_bytes >= segment.allocated_start_bytes
        and segment.reserved_peak_bytes >= segment.reserved_start_bytes
        and segment.reserved_start_bytes >= segment.allocated_start_bytes
        and segment.reserved_peak_bytes >= segment.allocated_peak_bytes
    )
