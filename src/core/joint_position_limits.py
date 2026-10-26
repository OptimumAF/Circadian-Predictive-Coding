"""Bounded position-limit observations, without simulator IO or safety control.

Inputs are owned joint metadata and finite positions in rad/meters. Outputs keep
signed per-joint extremes and counts; no history, unit mixing, clipping, velocity,
collision, interpolation or learning decisions belong here.
"""

from dataclasses import dataclass
from math import isfinite


def _finite_vector(value: tuple[float, ...], dimension: int) -> None:
    if (
        type(value) is not tuple
        or len(value) != dimension
        or any(type(item) is not float or not isfinite(item) for item in value)
    ):
        raise ValueError("joint values must be an exact tuple of finite native floats")


@dataclass(frozen=True)
class JointPositionLimits:
    names: tuple[str, ...]
    units: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.names) is not tuple
            or not 1 <= len(self.names) <= 32
            or any(type(name) is not str or not 1 <= len(name) <= 64 for name in self.names)
            or len(set(self.names)) != len(self.names)
            or type(self.units) is not tuple
            or len(self.units) != len(self.names)
            or any(type(unit) is not str or unit not in ("rad", "m") for unit in self.units)
        ):
            raise ValueError("expected 1..32 unique named joints with explicit rad/m units")
        _finite_vector(self.lower, len(self.names))
        _finite_vector(self.upper, len(self.names))
        if any(low >= high for low, high in zip(self.lower, self.upper)):
            raise ValueError("each lower joint bound must be strictly below its upper bound")


def _own_limits(value: JointPositionLimits) -> JointPositionLimits:
    if type(value) is not JointPositionLimits:
        raise ValueError("expected exact joint-position limits")
    return JointPositionLimits(value.names, value.units, value.lower, value.upper)


@dataclass(frozen=True)
class JointPositionSummary:
    limits: JointPositionLimits
    samples: int
    violating_samples: int
    lower_peak: tuple[float, ...]
    upper_peak: tuple[float, ...]
    violations_per_joint: tuple[int, ...]


def copy_joint_position_summary(value: JointPositionSummary) -> JointPositionSummary:
    """Own coherent bounded metadata at a reporting boundary, with no IO."""
    if type(value) is not JointPositionSummary:
        raise ValueError("expected an exact joint-position summary")
    limits = _own_limits(value.limits)
    dimension = len(limits.names)
    if (
        type(value.samples) is not int
        or not 0 <= value.samples <= 51
        or type(value.violating_samples) is not int
        or not 0 <= value.violating_samples <= value.samples
        or type(value.violations_per_joint) is not tuple
        or len(value.violations_per_joint) != dimension
        or any(
            type(count) is not int or not 0 <= count <= value.samples
            for count in value.violations_per_joint
        )
    ):
        raise ValueError("joint-position summary requires bounded native integer counts")
    _finite_vector(value.lower_peak, dimension)
    _finite_vector(value.upper_peak, dimension)
    for lower, upper, count in zip(value.lower_peak, value.upper_peak, value.violations_per_joint):
        directions = int(lower < 0.0) + int(upper > 0.0)
        if lower > 0.0 or upper < 0.0 or (count > 0) != (directions > 0) or count < directions:
            raise ValueError("joint-position signed peaks differ from their violation counts")
    if (
        not max(value.violations_per_joint)
        <= value.violating_samples
        <= sum(value.violations_per_joint)
    ):
        raise ValueError("joint-position sample counts differ from the per-joint union bounds")
    return JointPositionSummary(
        limits,
        value.samples,
        value.violating_samples,
        value.lower_peak,
        value.upper_peak,
        value.violations_per_joint,
    )


def check_joint_position_samples(
    value: JointPositionSummary, samples: tuple[tuple[float, ...], ...]
) -> JointPositionSummary:
    """Reconstruct a reported summary from bounded raw positions, never labels."""
    owned = copy_joint_position_summary(value)
    if type(samples) is not tuple or len(samples) > 51:
        raise ValueError("joint-position verification requires an exact tuple of <=51 samples")
    monitor = JointPositionMonitor(owned.limits)
    for sample in samples:
        monitor.observe(sample)
    if monitor.summary() != owned:
        raise ValueError("joint-position summary differs from the retained raw positions")
    return owned


class JointPositionMonitor:
    """Accumulate at most51 observations with O(joints) retained numeric storage."""

    def __init__(self, limits: JointPositionLimits, *, maximum_samples: int = 51):
        if type(maximum_samples) is not int or not 1 <= maximum_samples <= 51:
            raise ValueError("maximum_samples must be a native integer in1..51")
        self._limits = _own_limits(limits)
        self._maximum_samples = maximum_samples
        self._samples = self._violating_samples = 0
        self._lower = (0.0,) * len(self._limits.names)
        self._upper = (0.0,) * len(self._limits.names)
        self._counts = (0,) * len(self._limits.names)

    def observe(self, qpos: tuple[float, ...]) -> None:
        if self._samples >= self._maximum_samples:
            raise ValueError("joint-position sample bound exhausted")
        _finite_vector(qpos, len(self._limits.names))
        # Why: check all differences before committing, including overflow from
        # individually finite extremes. Invalid samples must leave totals intact.
        lower = tuple(q - low for q, low in zip(qpos, self._limits.lower))
        upper = tuple(q - high for q, high in zip(qpos, self._limits.upper))
        if not all(isfinite(item) for item in lower + upper):
            raise ValueError("joint-position difference must remain finite")
        violations = tuple(low < 0.0 or high > 0.0 for low, high in zip(lower, upper))
        self._lower = tuple(min(old, new) for old, new in zip(self._lower, lower))
        self._upper = tuple(max(old, new) for old, new in zip(self._upper, upper))
        self._counts = tuple(old + int(bad) for old, bad in zip(self._counts, violations))
        self._samples += 1
        self._violating_samples += int(any(violations))

    def summary(self) -> JointPositionSummary:
        return JointPositionSummary(
            _own_limits(self._limits),
            self._samples,
            self._violating_samples,
            self._lower,
            self._upper,
            self._counts,
        )
