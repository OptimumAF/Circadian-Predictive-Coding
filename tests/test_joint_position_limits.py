"""Signed, unit-preserving joint telemetry without simulator or learner work."""

from dataclasses import FrozenInstanceError, asdict
import math
from typing import Any

import pytest

from src.core.joint_position_limits import JointPositionLimits, JointPositionMonitor


def limits():
    return JointPositionLimits(("shoulder", "finger"), ("rad", "m"), (-1.0, 0.0), (1.0, 0.04))


def test_should_report_per_joint_signed_peaks_and_count_a_violating_sample_once():
    monitor = JointPositionMonitor(limits())
    initial = monitor.summary()
    assert asdict(initial) == dict(
        limits=asdict(limits()),
        samples=0,
        violating_samples=0,
        lower_peak=(0.0, 0.0),
        upper_peak=(0.0, 0.0),
        violations_per_joint=(0, 0),
    )
    for qpos in ((-1.0, 0.0), (1.0, 0.04), (-1.25, 0.05), (1.5, -0.02)):
        monitor.observe(qpos)
    report = monitor.summary()
    assert report.samples == 4 and report.violating_samples == 2
    assert report.lower_peak == pytest.approx((-0.25, -0.02))
    assert report.upper_peak == pytest.approx((0.5, 0.01))
    assert report.violations_per_joint == (2, 2)
    assert report.limits.units == ("rad", "m")


def test_should_use_zero_epsilon_for_the_smallest_representable_exceedance():
    monitor = JointPositionMonitor(limits())
    lower, upper = math.nextafter(-1.0, -math.inf), math.nextafter(0.04, math.inf)
    monitor.observe((lower, upper))
    report = monitor.summary()
    assert report.violating_samples == 1 and report.violations_per_joint == (1, 1)
    assert report.lower_peak == (lower + 1.0, 0.0)
    assert report.upper_peak == (0.0, upper - 0.04)
    assert report.lower_peak[0] < 0.0 < report.upper_peak[1]


def test_should_own_declared_limits_and_every_immutable_summary():
    declared = limits()
    monitor = JointPositionMonitor(declared)
    first = monitor.summary()
    assert first.limits == declared and first.limits is not declared
    with pytest.raises(FrozenInstanceError):
        setattr(first, "samples", 99)
    object.__setattr__(declared, "lower", (-99.0, -99.0))
    object.__setattr__(first.limits, "units", ("m", "rad"))
    monitor.observe((-2.0, 0.02))
    again = monitor.summary()
    assert again.limits == limits() and again.limits is not first.limits
    assert again.lower_peak == (-1.0, 0.0)
    assert first.samples == 0 and first.lower_peak == (0.0, 0.0)
    assert monitor.summary() == again


@pytest.mark.parametrize(
    "changes",
    [
        dict(names=["shoulder", "finger"]),
        dict(names=()),
        dict(names=("joint", "joint")),
        dict(names=("", "finger")),
        dict(names=("x" * 65, "finger")),
        dict(names=(1, "finger")),
        dict(units=["rad", "m"]),
        dict(units=("degree", "m")),
        dict(units=("rad",)),
        dict(lower=[-1.0, 0.0]),
        dict(lower=(-1, 0.0)),
        dict(lower=(-1.0,)),
        dict(lower=(float("nan"), 0.0)),
        dict(upper=(float("inf"), 0.04)),
        dict(upper=(1.0, float("-inf"))),
        dict(upper=(1.0, 0.0)),
        dict(lower=(2.0, 0.0)),
    ],
)
def test_should_refuse_ambiguous_units_shapes_nonfinite_or_non_strict_limits(
    changes: dict[str, Any],
):
    values: dict[str, Any] = dict(
        names=("shoulder", "finger"), units=("rad", "m"), lower=(-1.0, 0.0), upper=(1.0, 0.04)
    )
    values.update(changes)
    with pytest.raises(ValueError):
        JointPositionLimits(**values)


def test_should_require_native_float_coordinates_instead_of_equal_valued_subclasses():
    class Float(float):
        pass

    with pytest.raises(ValueError):
        JointPositionLimits(("joint",), ("rad",), (Float(-1.0),), (1.0,))
    monitor = JointPositionMonitor(limits())
    with pytest.raises(ValueError):
        monitor.observe((Float(0.0), 0.02))
    assert monitor.summary().samples == 0


def test_should_accept_the_documented_dimension_and_name_size_boundaries():
    declared = JointPositionLimits(
        tuple("x" * 62 + f"{index:02d}" for index in range(32)),
        ("rad",) * 32,
        (-1.0,) * 32,
        (1.0,) * 32,
    )
    monitor = JointPositionMonitor(declared, maximum_samples=1)
    monitor.observe((0.0,) * 32)
    assert monitor.summary().samples == 1 and monitor.summary().violations_per_joint == (0,) * 32
    with pytest.raises(ValueError):
        JointPositionLimits(
            tuple(f"joint{index}" for index in range(33)), ("rad",) * 33, (-1.0,) * 33, (1.0,) * 33
        )


@pytest.mark.parametrize("maximum", [None, True, 0, 52, 1.0, "2"])
def test_should_require_an_exact_bounded_native_sample_limit(maximum: Any):
    with pytest.raises(ValueError):
        JointPositionMonitor(limits(), maximum_samples=maximum)


@pytest.mark.parametrize(
    "qpos",
    [
        [0.0, 0.02],
        (),
        (0.0,),
        (0.0, 0.02, 1.0),
        (0, 0.02),
        (True, 0.02),
        ("0", 0.02),
        (float("nan"), 0.02),
        (0.0, float("inf")),
    ],
)
def test_should_refuse_invalid_observation_atomically(qpos: Any):
    monitor = JointPositionMonitor(limits())
    monitor.observe((-1.25, 0.05))
    before = monitor.summary()
    with pytest.raises(ValueError):
        monitor.observe(qpos)
    assert monitor.summary() == before


def test_should_refuse_unrepresentable_differences_without_publishing_partial_counts():
    monitor = JointPositionMonitor(JointPositionLimits(("joint",), ("rad",), (-1e308,), (1e308,)))
    monitor.observe((0.0,))
    before = monitor.summary()
    with pytest.raises(ValueError):
        monitor.observe((1e308,))
    assert monitor.summary() == before


def test_should_stop_at_51_samples_and_keep_prior_report_available():
    monitor = JointPositionMonitor(limits())
    for _ in range(51):
        monitor.observe((0.0, 0.02))
    before = monitor.summary()
    assert before.samples == 51 and before.violating_samples == 0
    with pytest.raises(ValueError, match="sample"):
        monitor.observe((-2.0, 0.02))
    assert monitor.summary() == before
