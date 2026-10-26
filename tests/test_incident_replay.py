"""Temporal replay admission consumes arrived incidents without retaining history."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import pytest

from src.core.incident_replay import IncidentReplayGate
from src.core.shift_incidents import ShiftIncidentDecision


def row(batch: int, state: str = "inactive", onset: int | None = None) -> ShiftIncidentDecision:
    return ShiftIncidentDecision(batch, ("inactive", state, "inactive"), (None, onset, None))


def observe(
    gate: IncidentReplayGate, batch: int, state: str = "inactive", onset: int | None = None
):
    gate.observe(row(batch, state, onset), completed_updates=batch)


def test_should_skip_first_boundary_admit_prior_onset_once_and_skip_continuation():
    gate = IncidentReplayGate("feature", 4, 12)
    decisions = []
    for batch in range(1, 13):
        if batch % 4 == 0:
            decisions.append(gate.admit_boundary(batch))
        observe(
            gate,
            batch,
            "inactive" if batch < 5 else "onset" if batch == 5 else "continued",
            None if batch < 5 else 5,
        )
    assert [value["admitted"] for value in decisions] == [False, True, False]
    assert decisions[1] == {
        "admitted": True,
        "channel": "feature",
        "onset_batch": 5,
        "observed_updates": 7,
        "boundary_updates": 8,
        "reason": "active_prior_onset",
    }
    assert gate.report()["admitted_events"] == 1
    assert gate.report()["last_consumed_onset"] == 5


def test_should_defer_onset_observed_at_boundary_until_next_boundary():
    gate = IncidentReplayGate("feature", 2, 4)
    observe(gate, 1)
    assert not gate.admit_boundary(2)["admitted"]
    observe(gate, 2, "onset", 2)
    observe(gate, 3, "continued", 2)
    assert gate.admit_boundary(4)["onset_batch"] == 2
    observe(gate, 4, "continued", 2)
    assert gate.report()["pending_onset"] is None


def test_should_expire_cleared_pending_incident_and_allow_a_later_distinct_onset():
    gate = IncidentReplayGate("feature", 4, 8)
    observe(gate, 1, "onset", 1)
    observe(gate, 2, "cleared", 1)
    observe(gate, 3)
    assert not gate.admit_boundary(4)["admitted"]
    observe(gate, 4, "onset", 4)
    for batch in range(5, 8):
        observe(gate, batch, "continued", 4)
    assert gate.admit_boundary(8)["onset_batch"] == 4
    observe(gate, 8, "cleared", 4)
    assert gate.report()["expired_onsets"] == 1
    assert gate.report()["observed_onsets"] == 2


@pytest.mark.parametrize("channel,index", [("uncertainty", 0), ("feature", 1), ("residual", 2)])
def test_should_use_only_the_frozen_selected_channel(channel, index):
    gate = IncidentReplayGate(channel, 2, 2)
    states = ["inactive"] * 3
    onsets: list[int | None] = [None] * 3
    states[(index + 1) % 3], onsets[(index + 1) % 3] = "onset", 1
    gate.observe(
        ShiftIncidentDecision(
            1, (states[0], states[1], states[2]), (onsets[0], onsets[1], onsets[2])
        ),
        completed_updates=1,
    )
    assert not gate.admit_boundary(2)["admitted"]
    assert gate.report()["observed_onsets"] == 0


@pytest.mark.parametrize(
    "channel,every,total",
    [
        ("unknown", 4, 12),
        (1, 4, 12),
        ("feature", True, 12),
        ("feature", 0, 12),
        ("feature", 13, 12),
        ("feature", 4.0, 12),
        ("feature", 4, True),
        ("feature", 4, 513),
        ("feature", 1, 0),
    ],
)
def test_should_refuse_invalid_birth_configuration(channel, every, total):
    with pytest.raises(ValueError):
        IncidentReplayGate(channel, every, total)


@pytest.mark.parametrize("batch,clock", [(0, 0), (2, 2), (1, 2), (True, 1), (1, True), (1, 1.0)])
def test_should_refuse_unobserved_future_or_mismatched_wake_clocks_transactionally(batch, clock):
    gate = IncidentReplayGate("feature", 4, 12)
    before = gate.report()
    with pytest.raises(ValueError):
        gate.observe(row(batch), completed_updates=clock)
    assert gate.report() == before


def test_should_refuse_missing_boundary_duplicate_observation_and_same_wake_admission():
    gate = IncidentReplayGate("feature", 2, 4)
    observe(gate, 1, "onset", 1)
    before = gate.report()
    with pytest.raises(ValueError):
        observe(gate, 1, "continued", 1)
    with pytest.raises(ValueError):
        observe(gate, 2, "continued", 1)
    with pytest.raises(ValueError):
        gate.admit_boundary(4)
    assert gate.report() == before
    assert gate.admit_boundary(2)["admitted"]
    with pytest.raises(ValueError):
        gate.admit_boundary(2)
    observe(gate, 2, "continued", 1)
    with pytest.raises(ValueError):
        gate.admit_boundary(2)


@pytest.mark.parametrize(
    "changes",
    [
        {"states": ["inactive", "inactive", "inactive"]},
        {"states": ("inactive", "inactive")},
        {"states": ("inactive", "bad", "inactive")},
        {"states": ("inactive", True, "inactive")},
        {"onset_batches": [None, None, None]},
        {"onset_batches": (None,)},
        {"onset_batches": (None, True, None)},
        {"onset_batches": (None, 2, None)},
        {"onset_batches": (None, 1.0, None)},
        {"onset_batches": (None, 1, None)},
        {"states": ("continued", "inactive", "inactive"), "onset_batches": (1, None, None)},
    ],
)
def test_should_refuse_malformed_records_and_other_channel_transitions(changes):
    gate = IncidentReplayGate("feature", 4, 12)
    before = gate.report()
    with pytest.raises(ValueError):
        gate.observe(replace(row(1), **changes), completed_updates=1)
    assert gate.report() == before
    observe(gate, 1)


@pytest.mark.parametrize(
    "state,onset", [("inactive", None), ("onset", 2), ("continued", 2), ("cleared", 2)]
)
def test_should_refuse_invalid_transition_from_an_active_incident(state, onset):
    gate = IncidentReplayGate("feature", 4, 12)
    observe(gate, 1, "onset", 1)
    before = gate.report()
    with pytest.raises(ValueError):
        observe(gate, 2, state, onset)
    assert gate.report() == before
    observe(gate, 2, "continued", 1)


def test_should_keep_consumption_permanent_without_refund_or_retry_interface():
    gate = IncidentReplayGate("feature", 2, 4)
    observe(gate, 1, "onset", 1)
    admission = gate.admit_boundary(2)
    admission["admitted"] = False
    admission["onset_batch"] = 999
    # Caller-native failure changes no gate-owned state; ordinary continuation cannot retry.
    observe(gate, 2, "continued", 1)
    observe(gate, 3, "continued", 1)
    assert not gate.admit_boundary(4)["admitted"]
    assert gate.report()["last_consumed_onset"] == 1


@pytest.mark.parametrize("clone", [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_should_copy_all_pending_and_consumed_state_without_resetting_birth_or_clocks(clone):
    gate = IncidentReplayGate("feature", 2, 6)
    observe(gate, 1, "onset", 1)
    pending = clone(gate)
    assert pending.report() == gate.report()
    assert pending.admit_boundary(2)["admitted"]
    assert gate.report()["admitted_events"] == 0
    consumed = clone(pending)
    assert consumed.report() == pending.report()
    observe(consumed, 2, "continued", 1)
    observe(consumed, 3, "continued", 1)
    assert not consumed.admit_boundary(4)["admitted"]
    assert pending.report()["observed_updates"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("channel", "residual"),
        ("every_updates", 2),
        ("total_updates", 10),
        ("_observed_updates", True),
        ("_observed_updates", 4),
        ("_last_boundary_updates", 4),
        ("_active_onsets", [None, None, None]),
        ("_active_onsets", (None, 9, None)),
        ("_pending_onset", 1),
        ("_last_consumed_onset", 1),
        ("_observed_onsets", 1),
        ("_expired_onsets", True),
        ("_admitted_events", 1),
        ("_skipped_boundaries", 1),
        ("_birth_config", ("feature", True, 12)),
    ],
)
@pytest.mark.parametrize("method", ["report", "observe", "admit_boundary"])
def test_should_refuse_corrupted_owned_state_at_every_public_boundary(field, value, method):
    gate = IncidentReplayGate("feature", 4, 12)
    object.__setattr__(gate, field, value)
    with pytest.raises(ValueError):
        if method == "report":
            gate.report()
        elif method == "observe":
            observe(gate, 1)
        else:
            gate.admit_boundary(4)


def test_should_hold_only_bounded_scalars_and_detach_reports_after_512_observations():
    gate = IncidentReplayGate("feature", 1, 512)
    size = len(pickle.dumps(gate))
    for batch in range(1, 513):
        gate.admit_boundary(batch)
        observe(gate, batch, "onset" if batch == 1 else "continued", 1)
    report = gate.report()
    assert all(type(value) in (str, int, type(None)) for value in report.values())
    assert report["admitted_events"] == 1 and report["skipped_boundaries"] == 511
    assert len(pickle.dumps(gate)) < size + 128
    report["observed_updates"] = 0
    assert gate.report()["observed_updates"] == 512
    with pytest.raises(ValueError):
        observe(gate, 513, "continued", 1)
    with pytest.raises(ValueError):
        gate.admit_boundary(513)
    with pytest.raises(AttributeError):
        gate.channel = "residual"
    with pytest.raises(AttributeError):
        setattr(gate, "history", [])


def test_should_not_retain_caller_owned_decisions():
    gate = IncidentReplayGate("feature", 2, 2)
    source = row(1, "onset", 1)
    gate.observe(source, completed_updates=1)
    object.__setattr__(source, "onset_batches", (None, 999, None))
    assert gate.admit_boundary(2)["onset_batch"] == 1


def test_should_reject_foreign_record_without_consuming_observation():
    gate = IncidentReplayGate("feature", 2, 2)
    bad: Any = object()
    with pytest.raises(ValueError):
        gate.observe(bad, completed_updates=1)
    assert gate.report()["observed_updates"] == 0
