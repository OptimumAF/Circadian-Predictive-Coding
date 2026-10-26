"""Analytic arrived-batch signals and bounded, descriptive persistence rules."""

from collections import deque
from dataclasses import FrozenInstanceError, fields, is_dataclass
from math import log
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.shift_signals import (
    ShiftDecision,
    ShiftDetector,
    ShiftMetrics,
    ShiftPolicy,
    summarize_shift_batch,
)


def metrics(uncertainty=0.0, feature=0.0, residual=0.0, rows=1):
    return ShiftMetrics(rows, uncertainty, (feature,), residual)


def test_should_match_binary_entropy_absolute_residual_and_raw_input_column_means():
    features = np.array([[-1.0, 0.0], [1.0, 0.5], [0.0, -0.5], [0.0, 1.0]])
    prediction, targets = (
        np.array([[0.0], [0.25], [0.5], [1.0]]),
        np.array([[1.0], [0.0], [1.0], [0.0]]),
    )
    result = summarize_shift_batch(features, prediction, targets)
    entropy_quarter = -(0.25 * log(0.25) + 0.75 * log(0.75)) / log(2)
    assert type(result) is ShiftMetrics and result.rows == 4
    assert result.uncertainty == pytest.approx((entropy_quarter + 1) / 4)
    assert result.feature_mean == (0.0, 0.25)
    assert result.residual == 0.6875
    endpoints = summarize_shift_batch(
        np.zeros((2, 1)), np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]])
    )
    assert endpoints.uncertainty == endpoints.residual == 0.0


def test_should_accept_inclusive_matrix_caps_and_tanh_probability_endpoints():
    result = summarize_shift_batch(np.ones((128, 16)), np.full((128, 1), 0.5), np.zeros((128, 1)))
    assert result.rows == 128 and result.feature_mean == (1.0,) * 16
    assert result.uncertainty == 1.0 and result.residual == 0.5


def test_should_keep_maximum_entropy_in_its_declared_domain_for_every_admitted_batch_length():
    detector = ShiftDetector()
    for rows in range(1, 129):
        result = summarize_shift_batch(
            np.zeros((rows, 1)), np.full((rows, 1), 0.5), np.zeros((rows, 1))
        )
        assert 0.0 <= result.uncertainty <= 1.0
        assert result.uncertainty == pytest.approx(1.0)
        assert detector.observe(result).batch_index == rows


def test_should_preserve_input_bytes_and_flags_and_return_frozen_detached_python_values():
    features, prediction, targets = np.array([[0.25, -0.5]]), np.array([[0.5]]), np.array([[1.0]])
    for value in (features, prediction, targets):
        value.flags.writeable = False
    before = tuple(
        (value.tobytes(), value.flags.writeable) for value in (features, prediction, targets)
    )
    result = summarize_shift_batch(features, prediction, targets)
    assert (
        tuple((value.tobytes(), value.flags.writeable) for value in (features, prediction, targets))
        == before
    )
    for value in (features, prediction, targets):
        value.flags.writeable = True
        value.fill(0)
    assert result.feature_mean == (0.25, -0.5) and result.residual == 0.5
    with pytest.raises(FrozenInstanceError):
        setattr(result, "rows", 2)


class ForeignArray(np.ndarray):
    pass


@pytest.mark.parametrize(
    "bad",
    (
        "list",
        "subclass",
        "dtype",
        "rank",
        "rows_zero",
        "width_zero",
        "rows",
        "width",
        "feature_nan",
        "feature_range",
        "prediction_dtype",
        "prediction_shape",
        "prediction_nan",
        "prediction_range",
        "target_dtype",
        "target_shape",
        "target_nan",
        "target_range",
    ),
)
def test_should_refuse_bad_matrix_envelopes_values_and_caps_without_unsafe_scans(monkeypatch, bad):
    x: Any = np.zeros((1, 2))
    p: Any = np.full((1, 1), 0.5)
    y: Any = np.zeros((1, 1))
    if bad == "list":
        x = [[0.0, 0.0]]
    elif bad == "subclass":
        x = x.view(ForeignArray)
    elif bad == "dtype":
        x = x.astype(np.float32)
    elif bad == "rank":
        x = np.zeros(2)
    elif bad in ("rows_zero", "width_zero", "rows", "width"):
        shape = {"rows_zero": (0, 2), "width_zero": (1, 0), "rows": (129, 2), "width": (1, 17)}[bad]
        x = np.zeros(shape)
        p, y = np.zeros((shape[0], 1)), np.zeros((shape[0], 1))
    elif bad.startswith("feature"):
        x[0, 0] = np.nan if bad.endswith("nan") else 1.01
    elif bad.startswith("prediction"):
        p = (
            np.zeros((1, 1), dtype=np.float32)
            if bad.endswith("dtype")
            else np.zeros((1, 2))
            if bad.endswith("shape")
            else np.array([[np.nan if bad.endswith("nan") else 1.01]])
        )
    else:
        y = (
            np.zeros((1, 1), dtype=np.float32)
            if bad.endswith("dtype")
            else np.zeros((1, 2))
            if bad.endswith("shape")
            else np.array([[np.nan if bad.endswith("nan") else -0.01]])
        )
    if bad == "target_shape":

        def forbidden(*args, **kwargs):
            pytest.fail("later invalid metadata was found after an earlier value scan")

        monkeypatch.setattr(np, "isfinite", forbidden)
    with pytest.raises(ValueError):
        summarize_shift_batch(x, p, y)


@pytest.mark.parametrize(
    "field,bad",
    (
        ("reference_batches", 0),
        ("reference_batches", True),
        ("reference_batches", 33),
        ("window_batches", 0),
        ("window_batches", 33),
        ("persistence_batches", 0),
        ("persistence_batches", 33),
        ("uncertainty_threshold", 0.0),
        ("uncertainty_threshold", np.nan),
        ("uncertainty_threshold", 1.01),
        ("uncertainty_threshold", True),
        ("feature_threshold", 0.0),
        ("feature_threshold", 2.01),
        ("residual_threshold", 0.0),
        ("residual_threshold", np.inf),
    ),
)
def test_should_revalidate_tampered_policy_fields_before_owning_any_observations(field, bad):
    policy = ShiftPolicy()
    object.__setattr__(policy, field, bad)
    with pytest.raises(ValueError):
        ShiftDetector(policy)


class ForeignPolicy(ShiftPolicy):
    pass


@pytest.mark.parametrize("bad", ("none", "object", "subclass"))
def test_should_require_the_exact_declared_policy_record(bad):
    policy: Any = None if bad == "none" else object()
    if bad == "subclass":
        policy = object.__new__(ForeignPolicy)
        for name, value in vars(ShiftPolicy()).items():
            object.__setattr__(policy, name, value)
    with pytest.raises(ValueError):
        ShiftDetector(policy)


def test_should_freeze_batch_weighted_reference_and_keep_postreference_warmup_separate():
    detector = ShiftDetector(ShiftPolicy(reference_batches=2, window_batches=2))
    first = detector.observe(metrics(0, 0, 0, rows=1))
    second = detector.observe(metrics(1, 1, 1, rows=128))
    warmup = detector.observe(metrics(0.75, 0.75, 0.75))
    monitor = detector.observe(metrics(0.75, 0.75, 0.75))
    assert [(item.batch_index, item.phase) for item in (first, second, warmup, monitor)] == [
        (1, "reference"),
        (2, "reference"),
        (3, "window"),
        (4, "monitor"),
    ]
    for item in (first, second, warmup):
        assert item.uncertainty_change is item.feature_drift is item.residual_change is None
        assert item.streaks == (0, 0, 0) and item.alarms == (False, False, False)
    assert (monitor.uncertainty_change, monitor.feature_drift, monitor.residual_change) == (
        0.25,
        0.25,
        0.25,
    )


def test_should_require_strict_threshold_exceedance_and_keep_directional_changes_signed():
    detector = ShiftDetector(
        ShiftPolicy(
            reference_batches=1,
            window_batches=1,
            persistence_batches=1,
            uncertainty_threshold=0.25,
            feature_threshold=0.25,
            residual_threshold=0.25,
        )
    )
    detector.observe(metrics(0.5, 0, 0.5))
    exact = detector.observe(metrics(0.75, 0.25, 0.75))
    assert exact.streaks == (0, 0, 0) and exact.alarms == (False, False, False)
    lower = detector.observe(metrics(0, -0.5, 0))
    assert lower.uncertainty_change == lower.residual_change == -0.5
    assert lower.feature_drift == 0.5 and lower.alarms == (False, True, False)


def test_should_persist_and_reset_each_channel_independently_without_accumulating_noise():
    detector = ShiftDetector(
        ShiftPolicy(
            reference_batches=1,
            window_batches=1,
            persistence_batches=2,
            uncertainty_threshold=0.25,
            feature_threshold=0.25,
            residual_threshold=0.25,
        )
    )
    detector.observe(metrics())
    observed = [
        detector.observe(value)
        for value in (
            metrics(0.5),
            metrics(0.5, 0.5),
            metrics(0, 0.5, 0.5),
            metrics(0, 0, 0.5),
            metrics(0.5),
        )
    ]
    assert [item.streaks for item in observed] == [
        (1, 0, 0),
        (2, 1, 0),
        (0, 2, 1),
        (0, 0, 2),
        (1, 0, 0),
    ]
    assert [item.alarms for item in observed] == [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ]


def test_should_keep_the_original_reference_during_stationary_noise_shift_and_recurrence():
    detector = ShiftDetector()
    baseline, changed = metrics(0.25, 0, 0.125), metrics(0.5, 1, 0.5)
    for _ in range(3):
        detector.observe(baseline)
    decisions = [
        detector.observe(value)
        for value in (changed, changed, changed, baseline, baseline, changed, changed)
    ]
    assert [item.alarms for item in decisions] == [
        (False, False, False),
        (False, False, False),
        (True, True, True),
        (True, True, True),
        (False, False, False),
        (False, False, False),
        (True, True, True),
    ]
    assert decisions[-1].feature_drift == 1.0
    assert decisions[-1].uncertainty_change == 0.25


def test_should_own_a_detached_policy_and_return_frozen_decisions():
    policy = ShiftPolicy(
        reference_batches=1, window_batches=1, persistence_batches=1, uncertainty_threshold=0.25
    )
    detector = ShiftDetector(policy)
    object.__setattr__(policy, "uncertainty_threshold", 1.0)
    detector.observe(metrics())
    decision = detector.observe(metrics(0.5))
    assert decision.alarms[0]
    with pytest.raises(FrozenInstanceError):
        setattr(decision, "phase", "foreign")


def reachable_records(root, kind):
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
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list, deque)):
            for item in value:
                visit(item)

    visit(vars(root))
    return len(found)


def test_should_bound_retained_reference_window_and_refuse_observation_513_atomically():
    detector = ShiftDetector()
    for number in range(512):
        decision = detector.observe(metrics((number % 4) / 4))
    assert decision.batch_index == 512
    assert reachable_records(detector, ShiftMetrics) <= 5
    assert reachable_records(detector, ShiftDecision) <= 1
    before = pickle.dumps(detector)
    with pytest.raises(ValueError):
        detector.observe(metrics())
    assert pickle.dumps(detector) == before


@pytest.mark.parametrize(
    "bad",
    (
        "type",
        "rows_bool",
        "rows_zero",
        "rows_cap",
        "unc_nan",
        "unc_range",
        "feature_list",
        "feature_nan",
        "feature_range",
        "width",
        "residual_negative",
        "residual_inf",
    ),
)
def test_should_refuse_invalid_or_changed_width_metrics_without_advancing_detector(bad):
    detector = ShiftDetector(ShiftPolicy(reference_batches=1, window_batches=1))
    detector.observe(metrics())
    value: Any = metrics()
    if bad == "type":
        value = object()
    else:
        field, changed = {
            "rows_bool": ("rows", True),
            "rows_zero": ("rows", 0),
            "rows_cap": ("rows", 129),
            "unc_nan": ("uncertainty", np.nan),
            "unc_range": ("uncertainty", 1.01),
            "feature_list": ("feature_mean", [0.0]),
            "feature_nan": ("feature_mean", (np.nan,)),
            "feature_range": ("feature_mean", (-1.01,)),
            "width": ("feature_mean", (0.0, 0.0)),
            "residual_negative": ("residual", -0.01),
            "residual_inf": ("residual", np.inf),
        }[bad]
        object.__setattr__(value, field, changed)
    before = pickle.dumps(detector)
    with pytest.raises(ValueError):
        detector.observe(value)
    assert pickle.dumps(detector) == before
    assert detector.observe(metrics()).batch_index == 2
