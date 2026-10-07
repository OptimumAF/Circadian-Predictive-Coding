"""Known pilot spread, covariance, missing values and conditional forecasts."""

from dataclasses import replace
from math import sqrt
from typing import Any

import pytest

from src.core.pilot_precision import project_pilot_precision
from src.core.seed_statistics import SeedObservation


SEEDS = (41, 43, 59)
ROWS = tuple(SeedObservation(seed, value) for seed, value in zip(SEEDS, (-0.1, 0.0, 0.1)))


def test_should_project_complete_sample_spread_without_selecting_a_seed_count() -> None:
    result = project_pilot_precision(ROWS, SEEDS, 10, 1e-12)

    # Hand calculation: squared deviations .01 + 0 + .01, divisor 3-1.
    assert result.pilot_summary.observations == ROWS
    assert result.pilot_summary.observed_sample_standard_deviation == pytest.approx(0.1)
    assert result.projected_standard_error == pytest.approx(0.1 / sqrt(10))
    assert result.confirmation_seed_count == 10
    assert result.status == "conditional_projection"
    assert result.pilot_summary.marginal_interval is None
    assert result.pilot_summary.simultaneous_interval is None
    assert result == project_pilot_precision(ROWS, SEEDS, 10, 1e-12)


def test_should_preserve_precision_when_pair_direction_or_mean_changes() -> None:
    original = project_pilot_precision(ROWS, SEEDS, 10, 1e-12)
    reverse = tuple(replace(row, value=-row.value) for row in ROWS if row.value is not None)
    shifted = tuple(replace(row, value=row.value + 0.5) for row in ROWS if row.value is not None)
    assert project_pilot_precision(
        reverse, SEEDS, 10, 1e-12
    ).projected_standard_error == pytest.approx(original.projected_standard_error)
    assert project_pilot_precision(
        shifted, SEEDS, 10, 1e-12
    ).projected_standard_error == pytest.approx(original.projected_standard_error)


def test_should_keep_constant_paired_values_without_a_zero_uncertainty_forecast() -> None:
    # Two varying arms can have a constant paired difference.
    rows = tuple(
        SeedObservation(seed, left - right)
        for seed, left, right in zip(SEEDS, (1, 2, 3), (0, 1, 2))
    )
    result = project_pilot_precision(rows, SEEDS, 10, 1e-12)
    assert result.pilot_summary.observed_sample_standard_deviation == 0
    assert result.projected_standard_error is None
    assert result.status == "zero_observed_variance"
    assert result.pilot_summary.observed_mean == 1


def test_should_retain_raw_roundoff_but_suppress_its_precision_forecast() -> None:
    rows = tuple(
        SeedObservation(seed, value)
        for seed, value in zip(SEEDS, (0.35 - 0.30, 0.20 - 0.15, 0.35 - 0.30))
    )
    result = project_pilot_precision(rows, SEEDS, 10, 1e-12)
    assert result.pilot_summary.observed_sample_standard_deviation is not None
    assert 0 < result.pilot_summary.observed_sample_standard_deviation < 1e-12
    assert result.projected_standard_error is None
    assert result.status == "zero_observed_variance"


@pytest.mark.parametrize("observed", [0, 1, 2])
def test_should_keep_failed_pilot_seeds_instead_of_reducing_sample_count(observed: int) -> None:
    rows = tuple(
        row if i < observed else SeedObservation(row.seed, None, "failed fixture")
        for i, row in enumerate(ROWS)
    )
    result = project_pilot_precision(rows, SEEDS, 10, 1e-12)
    assert result.pilot_summary.planned_seed_count == 3
    assert result.pilot_summary.observed_seed_count == observed
    assert result.pilot_summary.observations == rows
    assert result.projected_standard_error is None
    assert result.status == "incomplete_pilot_observations"


def test_should_suppress_forecast_from_only_one_planned_pilot_seed() -> None:
    result = project_pilot_precision((ROWS[0],), (41,), 10, 1e-12)
    assert result.status == "insufficient_pilot_observations"
    assert result.projected_standard_error is None


@pytest.mark.parametrize("count", [True, 1, 0, -1, 10.0, None, 10**400])
def test_should_reject_invalid_or_unrepresentable_confirmation_counts(count: Any) -> None:
    with pytest.raises(ValueError, match="confirmation seed count"):
        project_pilot_precision(ROWS, SEEDS, count, 1e-12)


@pytest.mark.parametrize("tolerance", [True, -1, "0", None, float("nan"), float("inf"), 10**400])
def test_should_reject_invalid_numeric_tolerance(tolerance: Any) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        project_pilot_precision(ROWS, SEEDS, 10, tolerance)


@pytest.mark.parametrize(
    "rows",
    [
        ROWS[:-1],
        tuple(reversed(ROWS)),
        ROWS[:2] + (ROWS[0],),
        ROWS[:2] + (SeedObservation(59, float("inf")),),
    ],
)
def test_should_reject_partial_reordered_duplicate_or_nonfinite_pilot_vectors(rows: Any) -> None:
    with pytest.raises(ValueError):
        project_pilot_precision(rows, SEEDS, 10, 1e-12)


def test_should_refuse_overflowed_dispersion_instead_of_publishing_a_forecast() -> None:
    rows = tuple(replace(row, value=value) for row, value in zip(ROWS, (-1e308, 0.0, 1e308)))
    with pytest.raises(ValueError, match="finite"):
        project_pilot_precision(rows, SEEDS, 10, 1e-12)
