"""Pure three-seed precision sensitivity and bounded-mean feasibility arithmetic.

Inputs are complete pilot observations and an explicit planning policy. Outputs
retain the pilot plus conditional normal-variance and variance-independent bounds.
No IO, experiment scope, quantile search, count selection or coverage validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log, log1p, sqrt

from src.core.pilot_precision import PilotPrecisionProjection, project_pilot_precision
from src.core.seed_statistics import SeedObservation


@dataclass(frozen=True)
class PrecisionBudgetSpec:
    target_half_width: float
    mean_family_alpha: float
    pilot_variance_family_alpha: float
    statement_count: int
    lower_bound: float
    upper_bound: float
    candidate_seed_count: int
    simultaneous_critical: float
    zero_deviation_tolerance: float


@dataclass(frozen=True)
class SeedPrecisionBudget:
    spec: PrecisionBudgetSpec
    pilot: PilotPrecisionProjection
    pilot_variance_lower_quantile: float
    conditional_normal_sd_upper: float | None
    conditional_normal_half_width: float | None
    minimum_seeds_at_fixed_critical: int | None
    conditional_normal_target_met: bool | None
    bounded_mean_half_width: float
    bounded_mean_required_seeds: int
    bounded_mean_target_met: bool


def _finite(value: float, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be finite numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite numeric") from error
    if not isfinite(result):
        raise ValueError(f"{label} must be finite numeric")
    return result


def _validate_spec(spec: PrecisionBudgetSpec) -> None:
    if type(spec) is not PrecisionBudgetSpec:
        raise ValueError("precision requires its explicit spec")
    if type(spec.statement_count) is not int or spec.statement_count < 1:
        raise ValueError("statement count must be a positive integer")
    if type(spec.candidate_seed_count) is not int or spec.candidate_seed_count < 2:
        raise ValueError("candidate seed count must be an integer of at least two")
    for label, value in (
        ("target half width", spec.target_half_width),
        ("critical", spec.simultaneous_critical),
    ):
        if _finite(value, label) <= 0:
            raise ValueError(f"{label} must be positive")
    for alpha in (spec.mean_family_alpha, spec.pilot_variance_family_alpha):
        if not 0 < _finite(alpha, "family alpha") < 1:
            raise ValueError("family alpha must be strictly between zero and one")
    lower, upper = (
        _finite(spec.lower_bound, "range lower"),
        _finite(spec.upper_bound, "range upper"),
    )
    if lower >= upper or not isfinite(upper - lower):
        raise ValueError("observation range must have positive finite width")


def _normal_sensitivity(
    pilot: PilotPrecisionProjection, spec: PrecisionBudgetSpec
) -> tuple[float, float | None, float | None, int | None, bool | None]:
    # Why df2: exactly three declared pilot seeds; invert the integrated density.
    lower_quantile = -2 * log1p(-spec.pilot_variance_family_alpha / spec.statement_count)
    if lower_quantile <= 0 or not isfinite(lower_quantile):
        raise ValueError("pilot variance quantile must be positive finite")
    if pilot.status != "conditional_projection":
        return lower_quantile, None, None, None, None
    deviation = pilot.pilot_summary.observed_sample_standard_deviation
    assert deviation is not None
    upper = _finite(deviation * sqrt(2 / lower_quantile), "conditional normal SD upper")
    half_width = _finite(
        upper * spec.simultaneous_critical / sqrt(spec.candidate_seed_count), "normal half width"
    )
    ratio_squared = _finite(
        (upper * spec.simultaneous_critical / spec.target_half_width) ** 2, "normal seed count"
    )
    required = max(2, ceil(ratio_squared))
    return lower_quantile, upper, half_width, required, half_width <= spec.target_half_width


def _bounded_mean(spec: PrecisionBudgetSpec) -> tuple[float, int, bool]:
    width = spec.upper_bound - spec.lower_bound
    log_tail = _finite(log(2 * spec.statement_count / spec.mean_family_alpha), "bounded tail")
    half_width = _finite(
        width * sqrt(log_tail / (2 * spec.candidate_seed_count)), "bounded half width"
    )
    required_float = _finite(
        (width / spec.target_half_width) ** 2 * log_tail / 2, "bounded seed count"
    )
    return half_width, max(2, ceil(required_float)), half_width <= spec.target_half_width


def plan_seed_precision_budget(
    observations: tuple[SeedObservation, ...],
    planned_pilot_seeds: tuple[int, ...],
    spec: PrecisionBudgetSpec,
) -> SeedPrecisionBudget:
    """Keep all pilot rows; a forecast is conditional, never a stopping rule."""
    _validate_spec(spec)
    if type(planned_pilot_seeds) is not tuple or len(planned_pilot_seeds) != 3:
        raise ValueError("normal sensitivity requires exactly three planned pilot seeds")
    pilot = project_pilot_precision(
        observations, planned_pilot_seeds, spec.candidate_seed_count, spec.zero_deviation_tolerance
    )
    for row in observations:
        if row.value is not None and not spec.lower_bound <= row.value <= spec.upper_bound:
            raise ValueError("pilot observation lies outside the declared range")
    try:
        normal = _normal_sensitivity(pilot, spec)
        bounded = _bounded_mean(spec)
        return SeedPrecisionBudget(spec, pilot, *normal, *bounded)
    except (OverflowError, ArithmeticError) as error:
        raise ValueError("precision planning arithmetic must remain finite") from error
