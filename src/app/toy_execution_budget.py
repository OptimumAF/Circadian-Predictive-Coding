"""Opt-in toy runner limits at complete wake/sleep boundaries.

Inputs are total wake-call/replay-example, circadian width, and
per-invocation wall-time/process-RSS ceilings plus a monotonic clock. Outputs are
typed incomplete-stop facts. This module does not train, score,
checkpoint, write files, or choose a research metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Literal, NoReturn

from src.app.circadian_checkpoint import CircadianResumePosition
from src.shared.process_memory import ProcessRssSampler, ProcessRssSegment


ToyStopReason = Literal[
    "max_training_updates",
    "max_wall_seconds",
    "max_replay_examples",
    "max_hidden_width",
    "max_process_rss_bytes",
]


@dataclass
class ToyExecutionProgress:
    """Observed wake/replay work and last durable cursor, including on error."""

    updates_completed: int = 0
    replay_examples_completed: int = 0
    checkpoint_position: CircadianResumePosition | None = None
    hidden_width_observed: int | None = None
    peak_hidden_width_observed: int | None = None
    rejected_proposed_hidden_width: int | None = None
    process_rss_segment: ProcessRssSegment | None = None


@dataclass(frozen=True)
class ToyExecutionBudget:
    max_training_updates: int | None = None
    max_wall_seconds: float | None = None
    max_replay_examples: int | None = None
    max_hidden_width: int | None = None
    max_process_rss_bytes: int | None = None

    def __post_init__(self) -> None:
        updates = self.max_training_updates
        seconds = self.max_wall_seconds
        replay_examples = self.max_replay_examples
        hidden_width = self.max_hidden_width
        process_rss = self.max_process_rss_bytes
        if all(
            limit is None
            for limit in (updates, seconds, replay_examples, hidden_width, process_rss)
        ):
            raise ValueError(
                "toy execution budget requires an update, wall-time, replay, width, or RSS limit"
            )
        if updates is not None and (type(updates) is not int or updates < 0):
            raise ValueError("toy execution budget max_training_updates must be non-negative")
        if seconds is not None and (
            type(seconds) not in {int, float} or not isfinite(seconds) or seconds < 0
        ):
            raise ValueError(
                "toy execution budget max_wall_seconds must be finite and non-negative"
            )
        if replay_examples is not None and (
            type(replay_examples) is not int or replay_examples < 0
        ):
            raise ValueError("toy execution budget max_replay_examples must be non-negative")
        if hidden_width is not None and (type(hidden_width) is not int or hidden_width <= 0):
            raise ValueError("toy execution budget max_hidden_width must be positive")
        if process_rss is not None and (type(process_rss) is not int or process_rss <= 0):
            raise ValueError("toy execution budget max_process_rss_bytes must be positive")


@dataclass(frozen=True)
class ToyExecutionStop:
    reason: ToyStopReason
    updates_completed: int
    elapsed_seconds: float
    checkpoint_position: CircadianResumePosition | None
    replay_examples_completed: int = 0
    hidden_width_observed: int | None = None
    peak_hidden_width_observed: int | None = None
    proposed_hidden_width: int | None = None
    process_rss_segment: ProcessRssSegment | None = None
    status: Literal["incomplete"] = "incomplete"

    @property
    def resumable(self) -> bool:
        return self.checkpoint_position is not None


class ToyExecutionStopped(RuntimeError):
    """A limit stopped training before a partial result could be scored."""

    def __init__(self, stop: ToyExecutionStop) -> None:
        self.stop = stop
        cursor = "checked checkpoint" if stop.resumable else "no resumable checkpoint"
        super().__init__(
            f"toy run incomplete: {stop.reason} after {stop.updates_completed} "
            f"wake updates ({stop.elapsed_seconds:.3f}s); {cursor}"
        )


class ToyProcessRssUnavailable(RuntimeError):
    """The host did not provide a usable process-RSS sample."""


class ToyBudgetSession:
    """Count committed wake and replay work; check a monotonic wall clock."""

    def __init__(
        self,
        budget: ToyExecutionBudget,
        clock: Callable[[], float],
        progress: ToyExecutionProgress | None = None,
    ) -> None:
        if (
            type(budget) is not ToyExecutionBudget
            or not callable(clock)
            or (progress is not None and type(progress) is not ToyExecutionProgress)
        ):
            raise ValueError("toy execution budget requires a budget, callable clock, and progress")
        self.budget = budget
        self.clock = clock
        self.progress = progress
        self.started_at = self._read_clock()
        self.last_clock = self.started_at
        self.updates_completed = 0
        self.replay_examples_completed = 0
        self.hidden_width_observed: int | None = None
        self.peak_hidden_width_observed: int | None = None
        self.rejected_proposed_hidden_width: int | None = None
        self.checkpoint_position: CircadianResumePosition | None = None
        self.process_rss_sampler: ProcessRssSampler | None = None
        self.process_rss_segment: ProcessRssSegment | None = None

    def attach_memory(self, sampler: ProcessRssSampler) -> None:
        """Attach a started sampler before the toy dataset is constructed."""
        if self.budget.max_process_rss_bytes is None or type(sampler) is not ProcessRssSampler:
            raise ValueError("toy RSS sampler requires a process RSS budget")
        self.process_rss_sampler = sampler
        if sampler.start_bytes is None:
            raise ToyProcessRssUnavailable("toy process RSS is unavailable on this host")
        self._check_memory()

    def record_memory_snapshot(self) -> None:
        """Keep measured attempt work even if training or publication fails."""
        sampler = self.process_rss_sampler
        if sampler is None or sampler.start_bytes is None:
            return
        self.process_rss_segment = sampler.snapshot()
        if self.progress is not None:
            self.progress.process_rss_segment = self.process_rss_segment

    def complete_memory(self) -> None:
        """Check the sampler's final observation before returning a result."""
        self.record_memory_snapshot()
        self._stop_if_memory_over_cap()

    def _check_memory(self) -> None:
        sampler = self.process_rss_sampler
        if sampler is None:
            return
        if sampler.sample() is None:
            raise ToyProcessRssUnavailable(
                "toy process RSS became unavailable during this invocation"
            )
        self.record_memory_snapshot()
        self._stop_if_memory_over_cap()

    def _stop_if_memory_over_cap(self) -> None:
        limit = self.budget.max_process_rss_bytes
        segment = self.process_rss_segment
        if limit is not None and segment is not None and segment.peak_bytes > limit:
            self._stop("max_process_rss_bytes", self._elapsed())

    def restore_progress(
        self,
        updates: int,
        position: CircadianResumePosition,
        replay_examples: int = 0,
        hidden_width: int | None = None,
        peak_hidden_width: int | None = None,
    ) -> None:
        """Install only progress already checked by the toy checkpoint validator."""
        if hidden_width is not None and (
            type(hidden_width) is not int
            or hidden_width <= 0
            or type(peak_hidden_width) is not int
            or peak_hidden_width < hidden_width
        ):
            raise ValueError("validated toy checkpoint width progress is incompatible")
        self.updates_completed = updates
        self.replay_examples_completed = replay_examples
        self.hidden_width_observed = hidden_width
        self.peak_hidden_width_observed = peak_hidden_width
        self.checkpoint_position = position
        if self.progress is not None:
            self.progress.updates_completed = updates
            self.progress.replay_examples_completed = replay_examples
            self.progress.hidden_width_observed = hidden_width
            self.progress.peak_hidden_width_observed = peak_hidden_width
            self.progress.checkpoint_position = position
        replay_limit = self.budget.max_replay_examples
        if replay_limit is not None and replay_examples > replay_limit:
            self.stop_replay()
        width_limit = self.budget.max_hidden_width
        if (
            width_limit is not None
            and peak_hidden_width is not None
            and peak_hidden_width > width_limit
        ):
            self.stop_hidden_width(peak_hidden_width)

    def record_initial_width(self, width: int) -> None:
        self._record_hidden_width(width, width)
        limit = self.budget.max_hidden_width
        if limit is not None and width > limit:
            self.stop_hidden_width(width)

    def record_hidden_width(self, width: int, transient_peak: int | None = None) -> None:
        self._record_hidden_width(width, width if transient_peak is None else transient_peak)
        limit = self.budget.max_hidden_width
        if limit is not None and self.peak_hidden_width_observed is not None:
            if self.peak_hidden_width_observed > limit:
                raise AssertionError("observed hidden width exceeded the preflighted toy limit")

    def _record_hidden_width(self, width: int, transient_peak: int) -> None:
        if (
            type(width) is not int
            or width <= 0
            or type(transient_peak) is not int
            or transient_peak < width
        ):
            raise ValueError("observed toy hidden widths must be positive and ordered")
        self.hidden_width_observed = width
        self.peak_hidden_width_observed = max(self.peak_hidden_width_observed or 0, transient_peak)
        if self.progress is not None:
            self.progress.hidden_width_observed = self.hidden_width_observed
            self.progress.peak_hidden_width_observed = self.peak_hidden_width_observed

    def stop_hidden_width(self, proposed_width: int) -> NoReturn:
        self.rejected_proposed_hidden_width = proposed_width
        if self.progress is not None:
            self.progress.rejected_proposed_hidden_width = proposed_width
        self._stop("max_hidden_width", self._elapsed())

    def record_update(self) -> None:
        self.updates_completed += 1
        if self.progress is not None:
            self.progress.updates_completed = self.updates_completed

    def record_checkpoint(self, position: CircadianResumePosition) -> None:
        self.checkpoint_position = position
        if self.progress is not None:
            self.progress.checkpoint_position = position

    def record_replay(self, examples: int) -> None:
        if type(examples) is not int or examples < 0:
            raise ValueError("applied replay examples must be non-negative")
        self.replay_examples_completed += examples
        if self.progress is not None:
            self.progress.replay_examples_completed = self.replay_examples_completed
        limit = self.budget.max_replay_examples
        if limit is not None and self.replay_examples_completed > limit:
            raise AssertionError("applied replay exceeded the preflighted toy limit")

    @property
    def remaining_replay_examples(self) -> int | None:
        limit = self.budget.max_replay_examples
        return None if limit is None else max(0, limit - self.replay_examples_completed)

    def stop_replay(self) -> NoReturn:
        self._stop("max_replay_examples", self._elapsed())

    def before_update(self) -> None:
        self._check_memory()
        elapsed = self._elapsed()
        limit = self.budget.max_training_updates
        if limit is not None and self.updates_completed >= limit:
            self._stop("max_training_updates", elapsed)
        self._check_wall(elapsed)

    def before_sleep(self) -> None:
        self._check_memory()
        self._check_wall(self._elapsed())

    def before_final(self) -> None:
        self._check_memory()
        self._check_wall(self._elapsed())

    def _before_final_leased(self, observe: Callable[[], ProcessRssSegment | None] | None) -> None:
        """Internal capture uses the original sampler's already-held lease.

        Preserve cumulative segment/progress before cap refusal, just like normal
        final admission. A supplied port cannot replace an absent original sampler.
        """
        if self.process_rss_sampler is None:
            if observe is not None:
                raise ValueError("foreign RSS port cannot replace original sampler")
        else:
            if observe is None:
                raise ValueError("original sampler requires its leased observation")
            segment = observe()
            if segment is None:
                raise ToyProcessRssUnavailable("toy process RSS became unavailable during capture")
            if type(segment) is not ProcessRssSegment:
                raise ValueError("original RSS port requires a complete segment")
            self.process_rss_segment = segment
            if self.progress is not None:
                self.progress.process_rss_segment = segment
            self._stop_if_memory_over_cap()
        self._check_wall(self._elapsed())

    def _check_wall(self, elapsed: float) -> None:
        limit = self.budget.max_wall_seconds
        if limit is not None and elapsed >= limit:
            self._stop("max_wall_seconds", elapsed)

    def _stop(self, reason: ToyStopReason, elapsed: float) -> NoReturn:
        raise ToyExecutionStopped(
            ToyExecutionStop(
                reason,
                self.updates_completed,
                elapsed,
                self.checkpoint_position,
                self.replay_examples_completed,
                self.hidden_width_observed,
                self.peak_hidden_width_observed,
                self.rejected_proposed_hidden_width,
                self.process_rss_segment,
            )
        )

    def _elapsed(self) -> float:
        now = self._read_clock()
        if now < self.last_clock:
            raise ValueError("toy execution clock moved backwards")
        self.last_clock = now
        return now - self.started_at

    def _read_clock(self) -> float:
        raw = self.clock()
        if type(raw) not in {int, float} or not isfinite(raw):
            raise ValueError("toy execution clock must return finite seconds")
        return float(raw)
