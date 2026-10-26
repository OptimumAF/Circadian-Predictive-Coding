"""Incident edges preserve descriptive detector levels and never select learner work."""

from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any

import pytest

from src.core.shift_calibration import CalibratedShiftDetector, ShiftCalibration
from src.core.shift_incidents import ShiftIncidentDecision, ShiftIncidentTracker
from src.core.shift_signals import ShiftDecision, ShiftDetector, ShiftMetrics, ShiftPolicy


def decision(
    batch: int, alarms: tuple[bool, bool, bool] = (False, False, False), **changes: Any
) -> ShiftDecision:
    row = ShiftDecision(
        batch,
        "monitor",
        0.25,
        0.5,
        0.25,
        (int(alarms[0]), int(alarms[1]), int(alarms[2])),
        alarms,
    )
    return replace(row, **changes)


def test_should_track_independent_channels_and_rearm_only_after_a_clear():
    tracker = ShiftIncidentTracker()
    rows = [
        tracker.observe(decision(1, (True, False, True))),
        tracker.observe(decision(2, (True, True, False))),
        tracker.observe(decision(3, (False, True, False))),
        tracker.observe(decision(4, (True, False, True))),
        tracker.observe(decision(5)),
        tracker.observe(decision(6)),
    ]
    assert [row.batch_index for row in rows] == list(range(1, 7))
    assert [row.states for row in rows] == [
        ("onset", "inactive", "onset"),
        ("continued", "onset", "cleared"),
        ("cleared", "continued", "inactive"),
        ("onset", "cleared", "onset"),
        ("cleared", "inactive", "cleared"),
        ("inactive", "inactive", "inactive"),
    ]
    assert [row.onset_batches for row in rows] == [
        (1, None, 1),
        (1, 2, 1),
        (1, 2, None),
        (4, 2, 4),
        (4, None, 4),
        (None, None, None),
    ]


def test_should_leave_prefix_and_below_persistence_levels_inactive():
    tracker = ShiftIncidentTracker()
    rows = [
        decision(
            1, phase="reference", uncertainty_change=None, feature_drift=None, residual_change=None
        ),
        decision(
            2, phase="window", uncertainty_change=None, feature_drift=None, residual_change=None
        ),
        decision(
            3,
            phase="calibration",
            uncertainty_change=None,
            feature_drift=None,
            residual_change=None,
        ),
        decision(4, phase="calibration"),
        decision(5, streaks=(1, 2, 3)),
    ]
    for source in rows:
        before = asdict(source)
        result = tracker.observe(source)
        assert result.states == ("inactive",) * 3
        assert result.onset_batches == (None,) * 3
        assert asdict(source) == before


def test_should_accept_inclusive_score_bounds_and_native_numbers_without_deriving_alarms():
    tracker = ShiftIncidentTracker()
    low = tracker.observe(
        decision(
            1, uncertainty_change=-1, feature_drift=0, residual_change=-1, streaks=(512, 512, 512)
        )
    )
    high = tracker.observe(
        decision(
            2,
            uncertainty_change=1.0,
            feature_drift=2.0,
            residual_change=1.0,
            streaks=(512, 512, 512),
        )
    )
    assert low.states == high.states == ("inactive",) * 3


def test_should_return_frozen_detached_records_without_retaining_caller_records():
    tracker = ShiftIncidentTracker()
    source = decision(1, (True, False, False))
    first = tracker.observe(source)
    assert type(first) is ShiftIncidentDecision
    assert type(first.states) is type(first.onset_batches) is tuple
    with pytest.raises(FrozenInstanceError):
        setattr(first, "batch_index", 999)
    object.__setattr__(source, "alarms", (False, False, False))
    second = tracker.observe(decision(2, (True, False, False)))
    assert first.states == ("onset", "inactive", "inactive")
    assert first.onset_batches == second.onset_batches == (1, None, None)
    assert second.states == ("continued", "inactive", "inactive")
    # Forced caller corruption of a returned frozen record cannot become tracker state.
    object.__setattr__(second, "onset_batches", (999, None, None))
    third = tracker.observe(decision(3, (True, False, False)))
    assert third.onset_batches == (1, None, None)
    assert first.onset_batches == (1, None, None)


def test_should_refuse_duplicate_or_skipped_batches_without_changing_an_active_incident():
    tracker = ShiftIncidentTracker()
    tracker.observe(decision(1, (True, False, False)))
    for batch in (1, 3, 512):
        with pytest.raises(ValueError):
            tracker.observe(decision(batch))
    valid = tracker.observe(decision(2, (True, False, False)))
    assert valid.states == ("continued", "inactive", "inactive")
    assert valid.onset_batches == (1, None, None)


@pytest.mark.parametrize("phase", ("reference", "window", "calibration"))
def test_should_refuse_returning_to_prefix_after_monitoring_without_clearing_incidents(phase):
    tracker = ShiftIncidentTracker()
    tracker.observe(decision(1, (True, False, False)))
    prefix = decision(
        2, phase=phase, uncertainty_change=None, feature_drift=None, residual_change=None
    )
    with pytest.raises(ValueError):
        tracker.observe(prefix)
    valid = tracker.observe(decision(2, (True, False, False)))
    assert valid.states == ("continued", "inactive", "inactive")
    assert valid.onset_batches == (1, None, None)


def test_should_admit_exactly_512_observations_and_refuse_further_work():
    tracker = ShiftIncidentTracker()
    for batch in range(1, 513):
        result = tracker.observe(decision(batch, (True, False, False)))
        assert result.batch_index == batch
        assert result.onset_batches == (1, None, None)
        assert result.states[0] == ("onset" if batch == 1 else "continued")
    for batch in (513, 512):
        with pytest.raises(ValueError):
            tracker.observe(decision(batch))


class ForeignDecision(ShiftDecision):
    pass


class ForeignTuple(tuple):
    pass


class ForeignString(str):
    pass


class ForeignFloat(float):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "foreign_record",
        "subclass_record",
        "batch_bool",
        "batch_float",
        "batch_zero",
        "phase_unknown",
        "phase_subclass",
        "alarms_list",
        "alarms_short",
        "alarm_integer",
        "alarms_subclass",
        "streaks_list",
        "streaks_short",
        "streak_bool",
        "streak_negative",
        "streak_over_cap",
        "streak_float",
        "streaks_subclass",
        "alarm_without_streak",
        "reference_score",
        "window_score",
        "calibration_mixed_scores",
        "prefix_alarm",
        "prefix_streak",
        "score_nan",
        "score_infinite",
        "monitor_missing_score",
        "uncertainty_outside",
        "feature_outside",
        "residual_outside",
        "score_subclass",
    ],
)
def test_should_refuse_malformed_decisions_transactionally(fault):
    tracker = ShiftIncidentTracker()
    tracker.observe(decision(1, (True, False, False)))
    bad: Any = decision(2, (True, False, False))
    replacements: dict[str, Any] = {
        "batch_bool": {"batch_index": True},
        "batch_float": {"batch_index": 2.0},
        "batch_zero": {"batch_index": 0},
        "phase_unknown": {"phase": "final"},
        "phase_subclass": {"phase": ForeignString("monitor")},
        "alarms_list": {"alarms": [True, False, False]},
        "alarms_short": {"alarms": (True, False)},
        "alarm_integer": {"alarms": (1, False, False)},
        "alarms_subclass": {"alarms": ForeignTuple((True, False, False))},
        "streaks_list": {"streaks": [1, 0, 0]},
        "streaks_short": {"streaks": (1, 0)},
        "streak_bool": {"streaks": (True, 0, 0)},
        "streak_negative": {"streaks": (1, -1, 0)},
        "streak_over_cap": {"streaks": (1, 513, 0)},
        "streak_float": {"streaks": (1.0, 0, 0)},
        "streaks_subclass": {"streaks": ForeignTuple((1, 0, 0))},
        "alarm_without_streak": {"streaks": (0, 0, 0)},
        "reference_score": {"phase": "reference", "alarms": (False,) * 3, "streaks": (0,) * 3},
        "window_score": {"phase": "window", "alarms": (False,) * 3, "streaks": (0,) * 3},
        "calibration_mixed_scores": {
            "phase": "calibration",
            "uncertainty_change": None,
            "alarms": (False,) * 3,
            "streaks": (0,) * 3,
        },
        "prefix_alarm": {"phase": "calibration"},
        "prefix_streak": {"phase": "calibration", "alarms": (False,) * 3},
        "score_nan": {"uncertainty_change": float("nan")},
        "score_infinite": {"feature_drift": float("inf")},
        "monitor_missing_score": {"residual_change": None},
        "uncertainty_outside": {"uncertainty_change": -1.01},
        "feature_outside": {"feature_drift": -0.01},
        "residual_outside": {"residual_change": 1.01},
        "score_subclass": {"residual_change": ForeignFloat(0.25)},
    }
    if fault == "foreign_record":
        bad = object()
    elif fault == "subclass_record":
        bad = ForeignDecision(**asdict(bad))
    else:
        bad = replace(bad, **replacements[fault])
    with pytest.raises(ValueError):
        tracker.observe(bad)
    valid = tracker.observe(decision(2, (True, False, False)))
    assert valid.states == ("continued", "inactive", "inactive")
    assert valid.onset_batches == (1, None, None)


@pytest.mark.parametrize("calibrated", (False, True))
def test_should_preserve_all_real_detector_decisions_and_prefix_fit(calibrated):
    policy = ShiftPolicy(reference_batches=1, window_batches=1, persistence_batches=2)
    plain: CalibratedShiftDetector | ShiftDetector
    observed: CalibratedShiftDetector | ShiftDetector
    if calibrated:
        plain = CalibratedShiftDetector(policy, ShiftCalibration(prefix_batches=2))
        observed = CalibratedShiftDetector(policy, ShiftCalibration(prefix_batches=2))
    else:
        plain, observed = ShiftDetector(policy), ShiftDetector(policy)
    tracker = ShiftIncidentTracker()
    incident_rows = []
    for level in (0.0, 0.0, 0.75, 0.75, 0.75, 0.0, 0.0, 0.75, 0.75):
        metrics = ShiftMetrics(1, level, (level,), level)
        expected = plain.observe(metrics)
        actual = observed.observe(metrics)
        before = asdict(actual)
        incident_rows.append(tracker.observe(actual))
        assert actual == expected and asdict(actual) == before
        if calibrated:
            assert isinstance(plain, CalibratedShiftDetector)
            assert isinstance(observed, CalibratedShiftDetector)
            assert plain.calibration_result == observed.calibration_result
    assert [row.states[0] for row in incident_rows] == [
        "inactive",
        "inactive",
        "inactive",
        "onset",
        "continued",
        "cleared",
        "inactive",
        "inactive",
        "onset",
    ]
    assert [row.onset_batches[0] for row in incident_rows] == [
        None,
        None,
        None,
        4,
        4,
        4,
        None,
        None,
        9,
    ]
