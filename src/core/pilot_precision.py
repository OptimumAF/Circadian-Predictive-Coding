"""Project conditional SE from a complete ordered pilot vector.

Inputs are seed observations, a proposed replication count and numeric
tolerance. Output retains observed spread and explicit unavailable forecasts.
This module owns no IO, critical values, power/coverage or sample-size decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from src.core.seed_statistics import SeedObservation, SeedSummary, summarize_seeds


@dataclass(frozen=True)
class PilotPrecisionProjection:
    pilot_summary: SeedSummary
    confirmation_seed_count: int
    zero_deviation_tolerance: float
    projected_standard_error: float | None
    status: str


def _validate_projection_settings(count: int, tolerance: float) -> tuple[float, float]:
    if type(count) is not int or count < 2:
        raise ValueError("confirmation seed count must be an integer of at least two")
    try:
        denominator = sqrt(count)
    except (OverflowError, ValueError) as error:
        raise ValueError("confirmation seed count must be numerically representable") from error
    if type(tolerance) not in (int, float):
        raise ValueError("pilot numeric tolerance must be finite and nonnegative")
    try:
        numeric_tolerance = float(tolerance)
    except OverflowError as error:
        raise ValueError("pilot numeric tolerance must be finite and nonnegative") from error
    if not isfinite(numeric_tolerance) or numeric_tolerance < 0:
        raise ValueError("pilot numeric tolerance must be finite and nonnegative")
    return denominator, numeric_tolerance


def _project_error(
    summary: SeedSummary, denominator: float, tolerance: float
) -> tuple[float | None, str]:
    if summary.observed_seed_count != summary.planned_seed_count:
        return None, "incomplete_pilot_observations"
    deviation = summary.observed_sample_standard_deviation
    if deviation is None:
        return None, "insufficient_pilot_observations"
    # Why this: a constant discrete pilot cannot establish population precision.
    if deviation <= tolerance:
        return None, "zero_observed_variance"
    error = deviation / denominator
    if not isfinite(error) or error <= 0:
        raise ValueError("projected standard error must remain finite and positive")
    return error, "conditional_projection"


def project_pilot_precision(
    observations: tuple[SeedObservation, ...],
    planned_pilot_seeds: tuple[int, ...],
    confirmation_seed_count: int,
    zero_deviation_tolerance: float,
) -> PilotPrecisionProjection:
    """Assume future spread equals pilot SD; never infer a sufficient count."""
    denominator, tolerance = _validate_projection_settings(
        confirmation_seed_count, zero_deviation_tolerance
    )
    summary = summarize_seeds(observations, planned_pilot_seeds, None)
    error, status = _project_error(summary, denominator, tolerance)
    return PilotPrecisionProjection(summary, confirmation_seed_count, tolerance, error, status)
