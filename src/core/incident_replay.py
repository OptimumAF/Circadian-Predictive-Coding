"""Bounded, arrived-only incident admission for existing replay boundaries.

Inputs are ordered incident records and wake/boundary clocks. Outputs are
detached scalar admission records. This module owns no models, data, budgets,
training, IO or randomness; admitting an onset consumes it before caller work.
"""

from typing import Any

from src.core.shift_incidents import ShiftIncidentDecision

_CHANNELS = ("uncertainty", "feature", "residual")
_STATES = {"inactive", "onset", "continued", "cleared"}


def _integer(value: object, lower: int, upper: int) -> bool:
    return type(value) is int and lower <= value <= upper


class IncidentReplayGate:
    """Admit each still-active selected-channel onset at most once.

    Why this: the current wake's observer runs after its replay boundary, so a
    newly observed onset can affect only a later boundary. Clear observations
    cancel pending work, and continued levels never manufacture new onsets.
    """

    __slots__ = (
        "channel",
        "every_updates",
        "total_updates",
        "_birth_config",
        "_observed_updates",
        "_last_boundary_updates",
        "_active_onsets",
        "_pending_onset",
        "_last_consumed_onset",
        "_observed_onsets",
        "_expired_onsets",
        "_admitted_events",
        "_skipped_boundaries",
    )

    def __init__(self, channel: str, every_updates: int, total_updates: int) -> None:
        if type(channel) is not str or channel not in _CHANNELS:
            raise ValueError("incident replay channel must be uncertainty, feature or residual")
        if not _integer(total_updates, 1, 512) or not _integer(every_updates, 1, total_updates):
            raise ValueError("incident replay intervals require 1 <= every <= total <= 512")
        self.channel = channel
        self.every_updates = every_updates
        self.total_updates = total_updates
        self._birth_config = (channel, every_updates, total_updates)
        self._observed_updates = 0
        self._last_boundary_updates = 0
        self._active_onsets: tuple[int | None, int | None, int | None] = (None, None, None)
        self._pending_onset: int | None = None
        self._last_consumed_onset: int | None = None
        self._observed_onsets = 0
        self._expired_onsets = 0
        self._admitted_events = 0
        self._skipped_boundaries = 0

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"channel", "every_updates", "total_updates", "_birth_config"} and hasattr(
            self, name
        ):
            raise AttributeError("incident replay birth configuration is immutable")
        object.__setattr__(self, name, value)

    def _validate_state(self) -> None:
        birth = self._birth_config
        if (
            type(birth) is not tuple
            or len(birth) != 3
            or type(birth[0]) is not str
            or birth[0] not in _CHANNELS
            or not _integer(birth[2], 1, 512)
            or not _integer(birth[1], 1, birth[2])
            or type(self.channel) is not str
            or type(self.every_updates) is not int
            or type(self.total_updates) is not int
            or (self.channel, self.every_updates, self.total_updates) != birth
        ):
            raise ValueError("incident replay birth configuration was changed")
        count, boundary = self._observed_updates, self._last_boundary_updates
        if (
            not _integer(count, 0, self.total_updates)
            or not _integer(boundary, 0, self.total_updates)
            or boundary % self.every_updates
            or boundary < (count // self.every_updates) * self.every_updates
            or boundary > count + 1
        ):
            raise ValueError("incident replay clocks are inconsistent")
        self._validate_onsets(count, boundary)

    def _validate_onsets(self, count: int, boundary: int) -> None:
        active = self._active_onsets
        counters = (
            self._observed_onsets,
            self._expired_onsets,
            self._admitted_events,
            self._skipped_boundaries,
        )
        if (
            type(active) is not tuple
            or len(active) != 3
            or any(value is not None and not _integer(value, 1, count) for value in active)
            or any(not _integer(value, 0, 512) for value in counters)
            or self._observed_onsets > count
            or self._admitted_events + self._skipped_boundaries != boundary // self.every_updates
        ):
            raise ValueError("incident replay state or counters are invalid")
        pending, consumed = self._pending_onset, self._last_consumed_onset
        selected = active[_CHANNELS.index(self.channel)]
        if (
            (pending is not None and not _integer(pending, 1, count))
            or (consumed is not None and not _integer(consumed, 1, min(count, boundary - 1)))
            or (consumed is None) != (self._admitted_events == 0)
            or (pending is not None and pending != selected)
            or (pending is not None and consumed is not None and pending <= consumed)
            or (selected is not None and selected not in (pending, consumed))
            or self._observed_onsets
            != self._expired_onsets + self._admitted_events + int(pending is not None)
        ):
            raise ValueError("incident replay onset ownership is inconsistent")

    def _validate_decision(self, decision: ShiftIncidentDecision, completed_updates: int) -> None:
        if (
            type(decision) is not ShiftIncidentDecision
            or not _integer(completed_updates, 1, self.total_updates)
            or completed_updates != self._observed_updates + 1
            or type(decision.batch_index) is not int
            or decision.batch_index != completed_updates
        ):
            raise ValueError("incident observations require the next completed wake exactly once")
        if (
            self._last_boundary_updates
            != (completed_updates // self.every_updates) * self.every_updates
        ):
            raise ValueError(
                "incident observation requires its preceding configured replay boundary"
            )
        if (
            type(decision.states) is not tuple
            or len(decision.states) != 3
            or type(decision.onset_batches) is not tuple
            or len(decision.onset_batches) != 3
        ):
            raise ValueError("incident records require exactly three state and onset tuples")
        for state, onset, previous in zip(
            decision.states, decision.onset_batches, self._active_onsets
        ):
            if type(state) is not str or state not in _STATES:
                raise ValueError("incident record state is invalid")
            if onset is not None and not _integer(onset, 1, completed_updates):
                raise ValueError("incident record onset must be an arrived native integer")
            valid = (
                state == "inactive"
                and previous is None
                and onset is None
                or state == "onset"
                and previous is None
                and onset == completed_updates
                or state in {"continued", "cleared"}
                and previous is not None
                and onset == previous
            )
            if not valid:
                raise ValueError(
                    "incident record transition does not match the previous observation"
                )

    def observe(self, decision: ShiftIncidentDecision, *, completed_updates: int) -> None:
        self._validate_state()
        self._validate_decision(decision, completed_updates)
        index = _CHANNELS.index(self.channel)
        state, onset = decision.states[index], decision.onset_batches[index]
        if state == "onset":
            self._pending_onset = onset
            self._observed_onsets += 1
        elif state == "cleared" and self._pending_onset is not None:
            self._pending_onset = None
            self._expired_onsets += 1
        active = [
            onset if state in {"onset", "continued"} else None
            for state, onset in zip(decision.states, decision.onset_batches)
        ]
        self._active_onsets = (active[0], active[1], active[2])
        self._observed_updates = completed_updates

    def admit_boundary(self, updates: int) -> dict[str, Any]:
        self._validate_state()
        if (
            not _integer(updates, 1, self.total_updates)
            or updates != self._last_boundary_updates + self.every_updates
            or self._observed_updates != updates - 1
        ):
            raise ValueError(
                "incident admission requires the next boundary before its wake observer"
            )
        onset = self._pending_onset
        admitted = onset is not None
        # Consume before returning: a later failed replay must never refund this onset.
        if admitted:
            self._pending_onset = None
            self._last_consumed_onset = onset
            self._admitted_events += 1
        else:
            self._skipped_boundaries += 1
        self._last_boundary_updates = updates
        return {
            "admitted": admitted,
            "channel": self.channel,
            "onset_batch": onset,
            "observed_updates": self._observed_updates,
            "boundary_updates": updates,
            "reason": "active_prior_onset" if admitted else "no_pending_active_onset",
        }

    def report(self) -> dict[str, Any]:
        self._validate_state()
        return {
            "channel": self.channel,
            "every_updates": self.every_updates,
            "total_updates": self.total_updates,
            "observed_updates": self._observed_updates,
            "last_boundary_updates": self._last_boundary_updates,
            "active_onset": self._active_onsets[_CHANNELS.index(self.channel)],
            "pending_onset": self._pending_onset,
            "last_consumed_onset": self._last_consumed_onset,
            "observed_onsets": self._observed_onsets,
            "expired_onsets": self._expired_onsets,
            "admitted_events": self._admitted_events,
            "skipped_boundaries": self._skipped_boundaries,
        }
