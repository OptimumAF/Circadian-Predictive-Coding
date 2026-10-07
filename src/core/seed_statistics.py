"""Pure summaries for complete declared seed vectors and explicit nulls.

Inputs are ordered seed observations and an optional predeclared interval
policy. Outputs retain every seed/reason and conditional observed summaries.
This module owns no data, models, RNG, quantile selection, IO or experiment
scope; app contracts supply independently verified critical values.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt


@dataclass(frozen=True)
class SeedObservation:
    seed: int
    value: float | None
    reason: str | None = None


@dataclass(frozen=True)
class IntervalSpec:
    sample_count: int
    confidence_level: float
    marginal_critical: float
    statement_count: int | None = None
    simultaneous_critical: float | None = None
    zero_deviation_tolerance: float = 0.0


@dataclass(frozen=True)
class MeanInterval:
    lower: float
    upper: float
    confidence_level: float
    statement_count: int


@dataclass(frozen=True)
class SeedSummary:
    observations: tuple[SeedObservation, ...]
    planned_seed_count: int
    observed_seed_count: int
    observed_mean: float | None
    observed_sample_standard_deviation: float | None
    observed_minimum: float | None
    observed_maximum: float | None
    mean: float | None
    standard_error: float | None
    marginal_interval: MeanInterval | None
    simultaneous_interval: MeanInterval | None
    interval_status: str


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite numeric value")
    try:
        if not isfinite(float(value)):
            raise ValueError(f"{label} must be a finite numeric value")
        return float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite numeric value") from error


def _validate_observations(
    observations: tuple[SeedObservation, ...], planned_seeds: tuple[int, ...]
) -> tuple[float, ...]:
    if (
        type(planned_seeds) is not tuple
        or not planned_seeds
        or any(type(seed) is not int or seed < 0 for seed in planned_seeds)
        or len(set(planned_seeds)) != len(planned_seeds)
        or type(observations) is not tuple
        or len(observations) != len(planned_seeds)
    ):
        raise ValueError("seed observations require the complete unique planned seed order")
    values = []
    for seed, row in zip(planned_seeds, observations, strict=True):
        if type(row) is not SeedObservation or type(row.seed) is not int or row.seed != seed:
            raise ValueError("seed observations differ from the declared seed order")
        if row.value is None:
            if type(row.reason) is not str or not row.reason.strip():
                raise ValueError("null seed observations require a nonempty reason")
        else:
            if row.reason is not None:
                raise ValueError("numeric seed observations cannot also have a null reason")
            values.append(_finite_number(row.value, f"seed {seed}"))
    return tuple(values)


def _validate_spec(spec: IntervalSpec | None, count: int) -> None:
    if spec is None:
        return
    if type(spec) is not IntervalSpec or type(spec.sample_count) is not int:
        raise ValueError("interval policy requires an exact integer sample count")
    if spec.sample_count != count or count < 2:
        raise ValueError("interval policy sample count differs from the planned seeds")
    level = _finite_number(spec.confidence_level, "interval confidence")
    critical = _finite_number(spec.marginal_critical, "interval critical")
    tolerance = _finite_number(spec.zero_deviation_tolerance, "interval numeric tolerance")
    if not 0.0 < level < 1.0 or critical <= 0.0 or tolerance < 0.0:
        raise ValueError("interval policy confidence/critical is invalid")
    if spec.statement_count is None and spec.simultaneous_critical is None:
        return
    if type(spec.statement_count) is not int or spec.statement_count < 1:
        raise ValueError("interval policy statement count must be a positive integer")
    simultaneous = _finite_number(spec.simultaneous_critical, "interval simultaneous critical")
    if simultaneous < critical:
        raise ValueError("interval simultaneous critical cannot be below the marginal critical")


def _observed_statistics(values: tuple[float, ...]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    try:
        mean = _finite_number(fsum(values) / len(values), "observed mean")
        deviation = None
        if len(values) > 1:
            variance = fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
            deviation = _finite_number(sqrt(variance), "observed sample standard deviation")
        return mean, deviation
    except (OverflowError, ArithmeticError) as error:
        raise ValueError("seed statistics must remain finite") from error


def _interval(mean: float, error: float, critical: float, level: float, count: int) -> MeanInterval:
    return MeanInterval(
        _finite_number(mean - critical * error, "interval lower endpoint"),
        _finite_number(mean + critical * error, "interval upper endpoint"),
        level,
        count,
    )


def _intervals(
    mean: float | None, deviation: float | None, count: int, spec: IntervalSpec | None
) -> tuple[float | None, MeanInterval | None, MeanInterval | None, str]:
    if mean is None:
        return None, None, None, "incomplete_observations"
    if deviation is None:
        return None, None, None, "insufficient_observations"
    error = _finite_number(deviation / sqrt(count), "standard error")
    if spec is None:
        return error, None, None, "descriptive_only"
    # Why this: constant discrete observations do not estimate population uncertainty.
    if deviation <= spec.zero_deviation_tolerance:
        return error, None, None, "zero_observed_variance"
    marginal = _interval(mean, error, spec.marginal_critical, spec.confidence_level, 1)
    simultaneous = None
    if spec.simultaneous_critical is not None and spec.statement_count is not None:
        simultaneous = _interval(
            mean, error, spec.simultaneous_critical, spec.confidence_level, spec.statement_count
        )
    return error, marginal, simultaneous, "estimated"


def summarize_seeds(
    observations: tuple[SeedObservation, ...],
    planned_seeds: tuple[int, ...],
    intervals: IntervalSpec | None,
) -> SeedSummary:
    """Keep the whole planned vector; missing values never become fewer replications."""
    values = _validate_observations(observations, planned_seeds)
    _validate_spec(intervals, len(planned_seeds))
    observed_mean, deviation = _observed_statistics(values)
    complete_mean = observed_mean if len(values) == len(planned_seeds) else None
    error, marginal, simultaneous, status = _intervals(
        complete_mean, deviation, len(planned_seeds), intervals
    )
    return SeedSummary(
        observations,
        len(planned_seeds),
        len(values),
        observed_mean,
        deviation,
        min(values) if values else None,
        max(values) if values else None,
        complete_mean,
        error,
        marginal,
        simultaneous,
        status,
    )
