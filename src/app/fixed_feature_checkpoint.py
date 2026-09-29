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
class FixedFeatureCircadianCheckpoint:
    """One trusted-file payload for a CPU fixed-feature circadian head."""

    format_version: int
    protocol_id: str
    runner_config_digest: str
    initial_head_hash: str
    feature_hashes: tuple[tuple[str, str], ...]
    split_hashes: tuple[tuple[str, str], ...]
    progress: FixedFeatureCircadianProgress
    combined: CircadianRunCheckpoint
    memory_segments: tuple[ProcessRssSegment, ...] = ()


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
    epochs: int,
    initial_width: int,
    memory_sample_interval_seconds: float | None = None,
) -> FixedFeatureCircadianProgress:
    """Check runner identity and counters before restoring the live head."""
    if (
        not isinstance(checkpoint, FixedFeatureCircadianCheckpoint)
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != 1
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
    return progress


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
