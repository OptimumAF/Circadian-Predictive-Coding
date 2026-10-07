"""Immutable facts for one sleep attempt, independent of runners and storage.

Inputs are already measured model/runner facts. The output is a JSON-safe
record; this module neither performs sleep nor chooses evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite


def _require_count(name: str, value: int, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {bound} integer")


def _require_finite(name: str, value: float, *, minimum: float | None = None) -> None:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_ids(name: str, ids: tuple[int, ...]) -> None:
    if type(ids) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    for neuron_id in ids:
        _require_count(f"{name} neuron ID", neuron_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate neuron IDs")


def _require_pairs(name: str, pairs: tuple[tuple[int, int], ...]) -> None:
    if type(pairs) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError(f"{name} must contain (parent ID, child ID) pairs")
        _require_count(f"{name} parent ID", pair[0])
        _require_count(f"{name} child ID", pair[1])
        if pair[0] == pair[1]:
            raise ValueError(f"{name} parent and child IDs must differ")
    if len({parent for parent, _ in pairs}) != len(pairs):
        raise ValueError(f"{name} contains duplicate parent IDs")
    if len({child for _, child in pairs}) != len(pairs):
        raise ValueError(f"{name} contains duplicate child IDs")


@dataclass(frozen=True)
class ChemicalSummary:
    """Finite scalar summary of one per-neuron chemical array."""

    count: int
    minimum: float
    mean: float
    maximum: float

    def __post_init__(self) -> None:
        _require_count("chemical count", self.count, positive=True)
        for name in ("minimum", "mean", "maximum"):
            _require_finite(f"chemical {name}", getattr(self, name))
        lower_ordered = self.minimum <= self.mean or isclose(
            self.minimum, self.mean, rel_tol=1e-12, abs_tol=1e-15
        )
        upper_ordered = self.mean <= self.maximum or isclose(
            self.mean, self.maximum, rel_tol=1e-12, abs_tol=1e-15
        )
        if not lower_ordered or not upper_ordered:
            raise ValueError("chemical minimum, mean, and maximum must be ordered")


@dataclass(frozen=True)
class ChemicalSummaries:
    """Aligned primary, fast, and slow chemical vectors at one boundary."""

    primary: ChemicalSummary
    fast: ChemicalSummary
    slow: ChemicalSummary

    def __post_init__(self) -> None:
        if not all(
            isinstance(summary, ChemicalSummary) for summary in (self.primary, self.fast, self.slow)
        ):
            raise TypeError("chemical summaries must contain ChemicalSummary values")
        if len({self.primary.count, self.fast.count, self.slow.count}) != 1:
            raise ValueError("chemical vector counts must align")

    @property
    def count(self) -> int:
        return self.primary.count


@dataclass(frozen=True)
class SleepBudgets:
    """Resolved limits for one sleep attempt, in events, updates, and seconds."""

    split_limit: int
    prune_limit: int
    replay_update_limit: int
    time_limit_seconds: float | None

    def __post_init__(self) -> None:
        for name in ("split_limit", "prune_limit", "replay_update_limit"):
            _require_count(name, getattr(self, name))
        if self.time_limit_seconds is not None:
            _require_finite("time_limit_seconds", self.time_limit_seconds, minimum=0.0)


@dataclass(frozen=True)
class SleepStructuralChanges:
    """Stable-ID proposals and committed split/prune effects.

    A scheduled prune retains its neuron and width. A removed ID changes
    width. Proposed effects survive a runner rollback; applied effects do not.
    Removed IDs may include an older pending prune finalized during replay.
    """

    proposed_split_pairs: tuple[tuple[int, int], ...]
    applied_split_pairs: tuple[tuple[int, int], ...]
    proposed_prune_ids: tuple[int, ...]
    proposed_scheduled_prune_ids: tuple[int, ...]
    proposed_removed_prune_ids: tuple[int, ...]
    applied_scheduled_prune_ids: tuple[int, ...]
    applied_removed_prune_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_pairs("proposed_split_pairs", self.proposed_split_pairs)
        _require_pairs("applied_split_pairs", self.applied_split_pairs)
        for name in (
            "proposed_prune_ids",
            "proposed_scheduled_prune_ids",
            "proposed_removed_prune_ids",
            "applied_scheduled_prune_ids",
            "applied_removed_prune_ids",
        ):
            _require_ids(name, getattr(self, name))
        if not set(self.applied_split_pairs).issubset(self.proposed_split_pairs):
            raise ValueError("applied splits must have been proposed")
        if not set(self.proposed_scheduled_prune_ids).issubset(self.proposed_prune_ids):
            raise ValueError("scheduled prunes must have been proposed")
        if not set(self.applied_scheduled_prune_ids).issubset(self.proposed_scheduled_prune_ids):
            raise ValueError("applied scheduled prunes must have been proposed")
        if not set(self.applied_removed_prune_ids).issubset(self.proposed_removed_prune_ids):
            raise ValueError("applied removals must have been proposed")


@dataclass(frozen=True)
class SleepGuardMetrics:
    """Scores on the designated guard; failed attempts may lack a score."""

    role: str
    metric_name: str
    pre_accuracy: float | None
    post_accuracy: float | None
    pre_cross_entropy: float | None
    post_cross_entropy: float | None
    delta: float | None
    tolerance: float
    examples_scored: int
    role_hash: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"inner_guard", "validation"}:
            raise ValueError("guard role must be inner_guard or validation")
        if self.metric_name not in {"accuracy", "cross_entropy"}:
            raise ValueError("guard metric_name must be accuracy or cross_entropy")
        if self.pre_accuracy is None and self.post_accuracy is not None:
            raise ValueError("post guard accuracy requires pre accuracy")
        for name in ("pre_accuracy", "post_accuracy"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value, minimum=0.0)
            if value is not None and value > 1.0:
                raise ValueError(f"{name} must not exceed 1")
        if self.delta is not None and (
            (self.pre_cross_entropy is None) != (self.post_cross_entropy is None)
        ):
            raise ValueError("guard cross-entropy scores must both be present or absent")
        if self.pre_cross_entropy is None and self.post_cross_entropy is not None:
            raise ValueError("post guard cross-entropy requires pre cross-entropy")
        if (
            self.metric_name == "cross_entropy"
            and self.delta is not None
            and (self.pre_cross_entropy is None or self.post_cross_entropy is None)
        ):
            raise ValueError("cross-entropy guard requires cross-entropy scores")
        for name in ("pre_cross_entropy", "post_cross_entropy"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value, minimum=0.0)
        _require_finite("tolerance", self.tolerance, minimum=0.0)
        if self.post_accuracy is None:
            if self.delta is not None:
                raise ValueError("guard delta requires a post score")
        else:
            if self.delta is None:
                raise ValueError("complete guard scores require a delta")
            _require_finite("guard delta", self.delta)
        _require_count(
            "examples_scored", self.examples_scored, positive=self.pre_accuracy is not None
        )
        if self.role_hash is not None and (
            type(self.role_hash) is not str
            or len(self.role_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.role_hash)
        ):
            raise ValueError("guard role hash must be a lowercase SHA-256 digest")
        if self.delta is not None:
            assert self.pre_accuracy is not None and self.post_accuracy is not None
            expected = (
                self.pre_accuracy - self.post_accuracy
                if self.metric_name == "accuracy"
                else self._cross_entropy_delta()
            )
            if not isclose(self.delta, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("guard delta does not match the selected metric")

    def _cross_entropy_delta(self) -> float:
        assert self.pre_cross_entropy is not None and self.post_cross_entropy is not None
        return self.post_cross_entropy - self.pre_cross_entropy


@dataclass(frozen=True)
class SleepReplayUsage:
    """Exact replay examples and updates attempted and retained."""

    proposed_examples: int
    proposed_updates: int
    applied_examples: int
    applied_updates: int

    def __post_init__(self) -> None:
        for name in (
            "proposed_examples",
            "proposed_updates",
            "applied_examples",
            "applied_updates",
        ):
            _require_count(name, getattr(self, name))
        if self.applied_examples > self.proposed_examples:
            raise ValueError("applied replay examples exceed proposal")
        if self.applied_updates > self.proposed_updates:
            raise ValueError("applied replay updates exceed proposal")
        if self.proposed_updates and not self.proposed_examples:
            raise ValueError("replay updates require replay examples")


@dataclass(frozen=True)
class SleepDurations:
    """Core sleep time and total runner attempt time in seconds."""

    core_seconds: float
    attempt_seconds: float

    def __post_init__(self) -> None:
        _require_finite("core_seconds", self.core_seconds, minimum=0.0)
        _require_finite("attempt_seconds", self.attempt_seconds, minimum=0.0)
        if self.attempt_seconds < self.core_seconds:
            raise ValueError("attempt_seconds cannot be shorter than core_seconds")


@dataclass(frozen=True)
class SleepEventTelemetry:
    """Complete, versioned facts for an applied, rejected, or skipped attempt."""

    format_version: int
    trigger_reason: str
    outcome: str
    reason: str
    completed_epoch: int | None
    wake_batches: int
    budgets: SleepBudgets
    changes: SleepStructuralChanges
    before_width: int
    proposed_width: int
    final_width: int
    guard: SleepGuardMetrics | None
    replay: SleepReplayUsage
    chemistry_before: ChemicalSummaries
    chemistry_proposed: ChemicalSummaries
    chemistry_final: ChemicalSummaries
    durations: SleepDurations

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("unsupported sleep telemetry format version")
        if self.trigger_reason not in {
            "direct",
            "forced",
            "periodic",
            "adaptive",
            "periodic_and_adaptive",
            "disabled",
            "not_due",
            "cooldown_suppressed",
            "budget_skipped",
        }:
            raise ValueError("unknown sleep trigger reason")
        if self.outcome not in {"applied", "accepted", "rolled_back", "skipped", "error"}:
            raise ValueError("unknown sleep outcome")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("sleep outcome reason must be nonempty")
        if self.completed_epoch is not None:
            _require_count("completed_epoch", self.completed_epoch)
        _require_count("wake_batches", self.wake_batches)
        for name in ("before_width", "proposed_width", "final_width"):
            _require_count(name, getattr(self, name), positive=True)
        for name, expected in (
            ("chemistry_before", self.before_width),
            ("chemistry_proposed", self.proposed_width),
            ("chemistry_final", self.final_width),
        ):
            summary = getattr(self, name)
            if not isinstance(summary, ChemicalSummaries) or summary.count != expected:
                raise ValueError(f"{name} count must match its model width")
        if not isinstance(self.budgets, SleepBudgets) or not isinstance(
            self.changes, SleepStructuralChanges
        ):
            raise TypeError("sleep budgets and changes must be typed records")
        if not isinstance(self.replay, SleepReplayUsage) or not isinstance(
            self.durations, SleepDurations
        ):
            raise TypeError("sleep replay and durations must be typed records")
        if self.guard is not None and not isinstance(self.guard, SleepGuardMetrics):
            raise TypeError("guard must be a SleepGuardMetrics record or None")
        self._validate_widths()
        self._validate_outcome()

    def _validate_widths(self) -> None:
        proposed = (
            self.before_width
            + len(self.changes.proposed_split_pairs)
            - len(self.changes.proposed_removed_prune_ids)
        )
        applied = (
            self.before_width
            + len(self.changes.applied_split_pairs)
            - len(self.changes.applied_removed_prune_ids)
        )
        if self.proposed_width != proposed or self.final_width != applied:
            raise ValueError("sleep widths disagree with stable-ID split/removal counts")

    def _validate_outcome(self) -> None:
        if self.outcome != "error" and self.guard is not None and self.guard.delta is None:
            raise ValueError("completed guarded sleep requires full guard scores")
        if self.outcome in {"rolled_back", "error"}:
            if self.outcome == "rolled_back" and self.guard is None:
                raise ValueError("rolled-back sleep requires guard metrics")
            if (
                self.outcome == "rolled_back"
                and self.guard is not None
                and self.guard.delta is not None
                and self.guard.delta <= self.guard.tolerance
            ):
                raise ValueError("rolled-back guard delta must exceed tolerance")
            if (
                self.final_width != self.before_width
                or self.chemistry_final != self.chemistry_before
            ):
                raise ValueError("aborted sleep must retain the entry model state")
            if (
                self.changes.applied_split_pairs
                or self.changes.applied_scheduled_prune_ids
                or self.changes.applied_removed_prune_ids
                or self.replay.applied_examples
                or self.replay.applied_updates
            ):
                raise ValueError("aborted sleep cannot retain applied changes")
        elif self.outcome == "skipped":
            if (
                self.changes.proposed_split_pairs
                or self.changes.proposed_prune_ids
                or self.changes.proposed_scheduled_prune_ids
                or self.changes.proposed_removed_prune_ids
                or self.changes.applied_split_pairs
                or self.changes.applied_scheduled_prune_ids
                or self.changes.applied_removed_prune_ids
                or self.replay.proposed_examples
                or self.replay.proposed_updates
                or self.replay.applied_examples
                or self.replay.applied_updates
                or self.chemistry_proposed != self.chemistry_before
                or self.chemistry_final != self.chemistry_before
            ):
                raise ValueError("skipped sleep cannot have changes or a proposal")
        else:
            if self.proposed_width != self.final_width:
                raise ValueError("accepted sleep must retain its proposed width")
            if (
                self.guard is not None
                and self.guard.delta is not None
                and self.guard.delta > self.guard.tolerance
            ):
                raise ValueError("accepted sleep exceeds its guard tolerance")
            if self.replay.proposed_examples != self.replay.applied_examples or (
                self.replay.proposed_updates != self.replay.applied_updates
            ):
                raise ValueError("accepted sleep must retain its proposed replay")
            if (
                self.changes.proposed_split_pairs != self.changes.applied_split_pairs
                or self.changes.proposed_scheduled_prune_ids
                != self.changes.applied_scheduled_prune_ids
                or self.changes.proposed_removed_prune_ids != self.changes.applied_removed_prune_ids
                or self.chemistry_proposed != self.chemistry_final
            ):
                raise ValueError("accepted sleep must retain its proposed state")
