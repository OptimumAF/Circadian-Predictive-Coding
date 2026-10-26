"""Bounded prefix fitting with independent algebraic thresholds and ceilings."""

from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass
import pickle
from typing import Any

import pytest

from src.core.shift_calibration import (
    CalibratedShiftDetector,
    CalibrationResult,
    ShiftCalibration,
)
from src.core.shift_signals import ShiftDecision, ShiftMetrics, ShiftPolicy


def metric(uncertainty=0.0, feature=(0.0,), residual=0.0, rows=1):
    return ShiftMetrics(rows, uncertainty, feature, residual)


def make_detector(*, prefix=2, margin=0.125, persistence=2):
    return CalibratedShiftDetector(
        ShiftPolicy(reference_batches=1, window_batches=1, persistence_batches=persistence),
        ShiftCalibration(prefix_batches=prefix, margin=margin),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefix_batches", 0),
        ("prefix_batches", True),
        ("prefix_batches", 33),
        ("prefix_batches", 2.0),
        ("margin", 0),
        ("margin", True),
        ("margin", float("nan")),
        ("margin", float("inf")),
        ("margin", 1.01),
    ],
)
def test_should_refuse_invalid_calibration_configuration(field, value):
    with pytest.raises(ValueError):
        ShiftCalibration(**{field: value})


@pytest.mark.parametrize(
    "fault",
    ["policy_type", "calibration_type", "policy_tamper", "calibration_tamper", "short_prefix"],
)
def test_should_validate_exact_copied_controls_and_prefix_geometry_before_observations(fault):
    policy: Any = ShiftPolicy(reference_batches=1, window_batches=1)
    calibration: Any = ShiftCalibration(prefix_batches=2)
    if fault == "policy_type":
        policy = object()
    elif fault == "calibration_type":
        calibration = object()
    elif fault == "policy_tamper":
        object.__setattr__(policy, "reference_batches", True)
    elif fault == "calibration_tamper":
        object.__setattr__(calibration, "margin", 0)
    else:
        calibration = ShiftCalibration(prefix_batches=1)
    with pytest.raises(ValueError):
        CalibratedShiftDetector(policy, calibration)


def test_should_fit_full_window_maxima_plus_margin_and_report_batch_weighted_reference():
    detector = CalibratedShiftDetector(
        ShiftPolicy(reference_batches=2, window_batches=2, persistence_batches=2),
        ShiftCalibration(prefix_batches=6, margin=0.125),
    )
    prefix = (
        metric(0.25, (-0.5, 0.5), 0.125, 1),
        metric(0.75, (0.5, 0.5), 0.375, 3),
        metric(0.5, (0.25, 0.5), 0.25, 2),
        metric(0.75, (0.75, 1.0), 0.5, 4),
        metric(0.25, (-0.25, 0.0), 0.0, 1),
        metric(1.0, (1.0, 1.0), 1.0, 5),
    )
    decisions = []
    for item in prefix:
        assert detector.calibration_result is None
        decisions.append(detector.observe(item))
    assert [row.phase for row in decisions] == ["calibration"] * 6
    assert [row.batch_index for row in decisions] == list(range(1, 7))
    assert all(
        row.streaks == (0, 0, 0) and row.alarms == (False, False, False) for row in decisions
    )
    assert all(row.uncertainty_change is None for row in decisions[:3])
    assert [
        (row.uncertainty_change, row.feature_drift, row.residual_change) for row in decisions[3:]
    ] == [
        (0.125, 0.375, 0.125),
        (0.0, 0.125, 0.0),
        (0.125, 0.1875, 0.25),
    ]
    result = detector.calibration_result
    assert type(result) is CalibrationResult
    assert asdict(result) == {
        "policy": {
            "reference_batches": 2,
            "window_batches": 2,
            "persistence_batches": 2,
            "uncertainty_threshold": 0.25,
            "feature_threshold": 0.5,
            "residual_threshold": 0.375,
        },
        "reference_uncertainty": 0.5,
        "reference_feature_mean": (0.0, 0.5),
        "reference_residual": 0.25,
        "score_ceilings": (0.5, 1.25, 0.75),
        "reachable": (True, True, True),
        "calibration_batches": 6,
        "calibration_rows": 16,
        "calibration_windows": 3,
        "replayed_observations": 6,
    }


def test_should_freeze_before_monitoring_and_start_persistence_without_calibration_streaks():
    detector = make_detector(prefix=3)
    for _ in range(3):
        assert detector.observe(metric(0.25, (0.0,), 0.25)).alarms == (False, False, False)
    result = detector.calibration_result
    assert result is not None
    fitted = asdict(result)
    first = detector.observe(metric(1.0, (1.0,), 1.0))
    second = detector.observe(metric(1.0, (1.0,), 1.0))
    reset = detector.observe(metric(0.25, (0.0,), 0.25))
    assert (first.batch_index, second.batch_index, reset.batch_index) == (4, 5, 6)
    assert first.phase == second.phase == reset.phase == "monitor"
    assert first.streaks == (1, 1, 1) and first.alarms == (False, False, False)
    assert second.streaks == (2, 2, 2) and second.alarms == (True, True, True)
    assert reset.streaks == (0, 0, 0)
    result = detector.calibration_result
    assert result is not None and asdict(result) == fitted


def test_should_cap_at_mathematical_domains_without_forcing_unreachable_thresholds_down():
    detector = make_detector(persistence=1)
    detector.observe(metric(0.0, (-1.0,), 0.0))
    detector.observe(metric(1.0, (1.0,), 1.0))
    result = detector.calibration_result
    assert result is not None
    assert (
        result.policy.uncertainty_threshold,
        result.policy.feature_threshold,
        result.policy.residual_threshold,
    ) == (1.0, 2.0, 1.0)
    assert result.score_ceilings == (1.0, 2.0, 1.0) and result.reachable == (False, False, False)
    maximum = detector.observe(metric(1.0, (1.0,), 1.0))
    assert maximum.alarms == (False, False, False) and maximum.streaks == (0, 0, 0)


@pytest.mark.parametrize("channel", ["uncertainty", "feature", "residual"])
def test_should_report_strict_unreachability_when_threshold_equals_its_reference_ceiling(channel):
    baseline = metric(
        0.75 if channel == "uncertainty" else 0.0, (0.0,), 0.75 if channel == "residual" else 0.0
    )
    detector = make_detector(margin=1.0 if channel == "feature" else 0.25, persistence=1)
    detector.observe(baseline)
    detector.observe(baseline)
    result = detector.calibration_result
    assert result is not None
    index = {"uncertainty": 0, "feature": 1, "residual": 2}[channel]
    thresholds = (
        result.policy.uncertainty_threshold,
        result.policy.feature_threshold,
        result.policy.residual_threshold,
    )
    assert thresholds[index] == result.score_ceilings[index]
    assert result.reachable[index] is False


def test_should_expose_high_entropy_headroom_and_zero_ceilings_instead_of_claiming_reachability():
    detector = make_detector(margin=0.25)
    baseline = metric(1.0, (0.0,), 1.0)
    detector.observe(baseline)
    detector.observe(baseline)
    result = detector.calibration_result
    assert result is not None
    assert result.score_ceilings == (0.0, 1.0, 0.0)
    assert result.reachable == (False, True, False)
    assert result.policy.uncertainty_threshold == result.policy.residual_threshold == 0.25
    lower = detector.observe(metric(0.0, (0.0,), 0.0))
    assert lower.uncertainty_change == lower.residual_change == -1.0
    assert lower.alarms == (False, False, False)


def test_should_own_controls_and_return_frozen_results_without_exposing_fitted_policy():
    policy = ShiftPolicy(reference_batches=1, window_batches=1, persistence_batches=1)
    calibration = ShiftCalibration(prefix_batches=2, margin=0.125)
    detector = CalibratedShiftDetector(policy, calibration)
    object.__setattr__(policy, "window_batches", 32)
    object.__setattr__(calibration, "prefix_batches", 32)
    detector.observe(metric())
    detector.observe(metric())
    result = detector.calibration_result
    assert result is not None and result.policy.window_batches == 1
    with pytest.raises(FrozenInstanceError):
        setattr(result, "calibration_rows", 999)
    with pytest.raises(FrozenInstanceError):
        setattr(result.policy, "feature_threshold", 2.0)
    object.__setattr__(result.policy, "feature_threshold", 2.0)
    object.__setattr__(result, "reachable", (False, False, False))
    fresh = detector.calibration_result
    assert fresh is not None and fresh.policy.feature_threshold == 0.125
    assert fresh.reachable == (True, True, True)
    assert detector.observe(metric(0.0, (1.0,), 0.0)).alarms == (False, True, False)


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("fault", ["record", "width", "nan", "rows", "feature", "residual"])
def test_should_refuse_invalid_metrics_atomically_before_and_after_fitting(fitted, fault):
    detector = make_detector()
    detector.observe(metric())
    if fitted:
        detector.observe(metric())
    invalid: Any = metric()
    if fault == "record":
        invalid = object()
    else:
        field, value = {
            "width": ("feature_mean", (0.0, 0.0)),
            "nan": ("uncertainty", float("nan")),
            "rows": ("rows", True),
            "feature": ("feature_mean", (1.01,)),
            "residual": ("residual", -0.1),
        }[fault]
        object.__setattr__(invalid, field, value)
    before = pickle.dumps(detector)
    with pytest.raises(ValueError):
        detector.observe(invalid)
    assert pickle.dumps(detector) == before
    assert detector.observe(metric()).batch_index == (3 if fitted else 2)


def records_in(root, kind):
    seen: set[int] = set()
    found: set[int] = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if type(value) is kind:
            found.add(id(value))
        if is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif type(value) is dict:
            for child in value.values():
                visit(child)
        elif type(value) in (tuple, list):
            for child in value:
                visit(child)
        elif hasattr(value, "__dict__"):
            visit(vars(value))

    visit(root)
    return len(found)


def test_should_drop_prefix_records_after_fit_and_keep_only_bounded_reference_and_window():
    detector = CalibratedShiftDetector(
        ShiftPolicy(reference_batches=16, window_batches=16), ShiftCalibration(prefix_batches=32)
    )
    for number in range(31):
        detector.observe(metric(number / 64))
    assert records_in(detector, ShiftMetrics) <= 64
    detector.observe(metric(31 / 64))
    assert records_in(detector, ShiftMetrics) <= 32
    for number in range(100):
        detector.observe(metric((number % 8) / 8))
    assert records_in(detector, ShiftMetrics) <= 32
    assert records_in(detector, ShiftDecision) <= 1


def test_should_count_original_observations_once_and_refuse_513_without_changing_fitted_state():
    detector = make_detector()
    for index in range(512):
        decision = detector.observe(metric((index % 4) / 4))
    assert decision.batch_index == 512
    result = detector.calibration_result
    assert result is not None and result.replayed_observations == 2
    before = pickle.dumps(detector)
    with pytest.raises(ValueError):
        detector.observe(metric())
    assert pickle.dumps(detector) == before
