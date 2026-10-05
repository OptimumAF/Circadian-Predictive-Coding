"""Known quantile, range, covariance and conservative planning behavior."""

from dataclasses import replace
from math import exp, log, sqrt
from typing import Any

import pytest

from src.core.seed_precision_budget import PrecisionBudgetSpec, plan_seed_precision_budget
from src.core.seed_statistics import SeedObservation


SEEDS = (41, 43, 59)
ROWS = tuple(SeedObservation(seed, value) for seed, value in zip(SEEDS, (-0.1, 0.0, 0.1)))
SPEC = PrecisionBudgetSpec(0.05, 0.05, 0.05, 116, -1.0, 1.0, 10, 5.403490569214909, 1e-12)


def test_should_invert_df_two_cdf_and_keep_variance_uncertainty_separate() -> None:
    result = plan_seed_precision_budget(ROWS, SEEDS, SPEC)
    assert 1 - exp(-result.pilot_variance_lower_quantile / 2) == pytest.approx(0.05 / 116)
    assert result.conditional_normal_sd_upper is not None
    assert result.conditional_normal_sd_upper > 40 * 0.1
    assert result.conditional_normal_half_width is not None
    assert result.conditional_normal_half_width > SPEC.target_half_width
    assert result.conditional_normal_target_met is False
    assert result.minimum_seeds_at_fixed_critical is not None
    assert result.minimum_seeds_at_fixed_critical > 10
    assert result.pilot.pilot_summary.observations == ROWS


def test_should_match_exact_df_two_median_without_a_quantile_library() -> None:
    result = plan_seed_precision_budget(
        ROWS, SEEDS, replace(SPEC, pilot_variance_family_alpha=0.5, statement_count=1)
    )
    assert result.pilot_variance_lower_quantile == pytest.approx(2 * log(2))
    assert result.conditional_normal_sd_upper == pytest.approx(0.1 / sqrt(log(2)))


def test_should_plan_range_based_precision_for_constant_pilots_without_zero_uncertainty() -> None:
    rows = tuple(SeedObservation(seed, 0.0) for seed in SEEDS)
    result = plan_seed_precision_budget(rows, SEEDS, SPEC)
    assert result.conditional_normal_half_width is result.conditional_normal_sd_upper is None
    assert result.conditional_normal_target_met is None
    assert result.minimum_seeds_at_fixed_critical is None
    assert result.bounded_mean_half_width > 1
    assert result.bounded_mean_target_met is False
    assert result.bounded_mean_required_seeds == 6754
    wider = plan_seed_precision_budget(
        rows, SEEDS, replace(SPEC, lower_bound=-2.0, upper_bound=2.0)
    )
    assert wider.bounded_mean_half_width == pytest.approx(2 * result.bounded_mean_half_width)
    assert wider.bounded_mean_required_seeds == 27016


def test_should_make_bounded_required_count_sufficient_and_preceding_count_insufficient() -> None:
    result = plan_seed_precision_budget(ROWS, SEEDS, SPEC)
    sufficient = plan_seed_precision_budget(
        ROWS, SEEDS, replace(SPEC, candidate_seed_count=result.bounded_mean_required_seeds)
    )
    previous = plan_seed_precision_budget(
        ROWS, SEEDS, replace(SPEC, candidate_seed_count=result.bounded_mean_required_seeds - 1)
    )
    assert sufficient.bounded_mean_half_width <= SPEC.target_half_width
    assert sufficient.bounded_mean_target_met is True
    assert previous.bounded_mean_half_width > SPEC.target_half_width
    assert previous.bounded_mean_target_met is False


def test_should_keep_precision_independent_of_mean_sign_and_paired_covariance() -> None:
    original = plan_seed_precision_budget(ROWS, SEEDS, SPEC)
    reversed_rows = tuple(replace(r, value=-r.value) for r in ROWS if r.value is not None)
    shifted_rows = tuple(replace(r, value=r.value + 0.5) for r in ROWS if r.value is not None)
    for rows in (reversed_rows, shifted_rows):
        result = plan_seed_precision_budget(rows, SEEDS, SPEC)
        assert result.conditional_normal_half_width == pytest.approx(
            original.conditional_normal_half_width
        )
        assert result.bounded_mean_required_seeds == original.bounded_mean_required_seeds
    constant_pairs = tuple(
        SeedObservation(seed, left - right)
        for seed, left, right in zip(SEEDS, (0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
    )
    assert (
        plan_seed_precision_budget(constant_pairs, SEEDS, SPEC).conditional_normal_half_width
        is None
    )


@pytest.mark.parametrize("observed", [0, 1, 2])
def test_should_retain_missing_pilot_rows_and_bounds_without_a_normal_forecast(
    observed: int,
) -> None:
    rows = tuple(
        r if i < observed else SeedObservation(r.seed, None, "failed pilot")
        for i, r in enumerate(ROWS)
    )
    result = plan_seed_precision_budget(rows, SEEDS, SPEC)
    assert result.pilot.pilot_summary.observations == rows
    assert result.conditional_normal_half_width is None
    assert result.bounded_mean_required_seeds == 6754


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_half_width", 0),
        ("target_half_width", True),
        ("target_half_width", float("inf")),
        ("mean_family_alpha", 0),
        ("mean_family_alpha", 1),
        ("pilot_variance_family_alpha", float("nan")),
        ("statement_count", 0),
        ("statement_count", True),
        ("statement_count", 116.0),
        ("candidate_seed_count", 1),
        ("candidate_seed_count", True),
        ("candidate_seed_count", 10**400),
        ("lower_bound", 1.0),
        ("upper_bound", float("inf")),
        ("lower_bound", None),
        ("simultaneous_critical", 0),
        ("simultaneous_critical", True),
        ("zero_deviation_tolerance", -1),
        ("zero_deviation_tolerance", float("nan")),
    ],
)
def test_should_reject_invalid_types_bounds_policy_and_overflow(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        plan_seed_precision_budget(ROWS, SEEDS, replace(SPEC, **{field: value}))


@pytest.mark.parametrize(
    "change", ["missing", "reordered", "duplicate", "out_of_range", "nonfinite"]
)
def test_should_reject_changed_full_pilot_order_or_invalid_observations(change: str) -> None:
    rows = ROWS
    if change == "missing":
        rows = rows[:-1]
    if change == "reordered":
        rows = tuple(reversed(rows))
    if change == "duplicate":
        rows = rows[:2] + (rows[0],)
    if change == "out_of_range":
        rows = rows[:2] + (SeedObservation(59, 1.1),)
    if change == "nonfinite":
        rows = rows[:2] + (SeedObservation(59, float("nan")),)
    with pytest.raises(ValueError):
        plan_seed_precision_budget(rows, SEEDS, SPEC)


def test_should_reject_a_different_pilot_size_instead_of_reusing_df_two() -> None:
    with pytest.raises(ValueError, match="three"):
        plan_seed_precision_budget(ROWS[:2], SEEDS[:2], SPEC)


def test_should_refuse_numerical_underflow_or_overflow_before_output() -> None:
    for spec in (replace(SPEC, target_half_width=1e-300), replace(SPEC, statement_count=10**400)):
        with pytest.raises(ValueError):
            plan_seed_precision_budget(ROWS, SEEDS, spec)
