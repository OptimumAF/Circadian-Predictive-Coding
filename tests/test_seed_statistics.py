"""Known seed arithmetic and explicit missing/constant inference behavior."""

from dataclasses import replace
from math import sqrt
from typing import Any

import pytest

from src.core.seed_statistics import IntervalSpec, SeedObservation, summarize_seeds


SEEDS = tuple(range(1, 11))
SPEC = IntervalSpec(10, 0.95, 2.2621571627982053, 116, 5.403490569214909)


def _observations(values: tuple[float | None, ...]) -> tuple[SeedObservation, ...]:
    return tuple(
        SeedObservation(seed, value, "failed fixture" if value is None else None)
        for seed, value in zip(SEEDS, values, strict=True)
    )


def test_should_use_sample_dispersion_and_nine_degree_of_freedom_intervals() -> None:
    result = summarize_seeds(_observations(tuple(float(i) for i in SEEDS)), SEEDS, SPEC)

    # Hand sums for 1..10: mean 5.5 and sum of squared deviations 82.5.
    assert result.planned_seed_count == result.observed_seed_count == 10
    assert result.mean == result.observed_mean == 5.5
    assert result.observed_sample_standard_deviation == pytest.approx(sqrt(55 / 6))
    assert result.standard_error == pytest.approx(sqrt(11 / 12))
    assert (result.observed_minimum, result.observed_maximum) == (1.0, 10.0)
    assert result.marginal_interval is not None
    assert result.simultaneous_interval is not None
    assert SPEC.simultaneous_critical is not None
    assert result.marginal_interval.lower == pytest.approx(
        5.5 - SPEC.marginal_critical * sqrt(11 / 12)
    )
    assert result.simultaneous_interval.upper == pytest.approx(
        5.5 + SPEC.simultaneous_critical * sqrt(11 / 12)
    )
    assert result.simultaneous_interval.statement_count == 116
    assert result.interval_status == "estimated"
    assert result == summarize_seeds(result.observations, SEEDS, SPEC)


def test_should_match_the_published_nist_ten_observation_interval_example() -> None:
    # Symmetric fabricated data with NIST's mean 53.7 and sample SD 6.567.
    spread = 6.567 * sqrt(9 / 10)
    values = (53.7 - spread,) * 5 + (53.7 + spread,) * 5
    result = summarize_seeds(_observations(values), SEEDS, SPEC)

    assert result.mean == pytest.approx(53.7)
    assert result.observed_sample_standard_deviation == pytest.approx(6.567)
    assert result.marginal_interval is not None
    assert result.marginal_interval.lower == pytest.approx(49.002470182, abs=0.002)
    assert result.marginal_interval.upper == pytest.approx(58.397529818, abs=0.002)


def test_should_keep_failed_seed_and_suppress_full_planned_mean_and_intervals() -> None:
    values = tuple(float(i) for i in range(1, 10)) + (None,)
    result = summarize_seeds(_observations(values), SEEDS, SPEC)

    assert (result.planned_seed_count, result.observed_seed_count) == (10, 9)
    assert result.observed_mean == 5.0
    assert result.observed_sample_standard_deviation == pytest.approx(sqrt(7.5))
    assert result.mean is result.standard_error is None
    assert result.marginal_interval is result.simultaneous_interval is None
    assert result.observations[-1] == SeedObservation(10, None, "failed fixture")
    assert result.interval_status == "incomplete_observations"


@pytest.mark.parametrize("observed", [0, 1])
def test_should_handle_all_failed_or_one_observed_without_invented_dispersion(
    observed: int,
) -> None:
    values = (2.0,) * observed + (None,) * (10 - observed)
    result = summarize_seeds(_observations(values), SEEDS, SPEC)

    assert result.observed_seed_count == observed
    assert result.observed_mean == (2.0 if observed else None)
    assert result.observed_sample_standard_deviation is None
    assert result.mean is result.standard_error is None
    assert result.marginal_interval is result.simultaneous_interval is None


def test_should_record_zero_variance_without_claiming_a_point_confidence_interval() -> None:
    result = summarize_seeds(_observations((0.0,) * 10), SEEDS, SPEC)

    assert result.mean == result.observed_sample_standard_deviation == result.standard_error == 0.0
    assert result.marginal_interval is result.simultaneous_interval is None
    assert result.interval_status == "zero_observed_variance"


def test_should_distinguish_roundoff_from_inferential_variation_at_declared_precision() -> None:
    # These exact decimal count differences are all .05; binary subtraction differs.
    values = (0.35 - 0.30, 0.20 - 0.15) * 5
    policy = replace(SPEC, zero_deviation_tolerance=1e-12)
    result = summarize_seeds(_observations(values), SEEDS, policy)

    assert result.observed_sample_standard_deviation is not None
    assert 0 < result.observed_sample_standard_deviation < 1e-12
    assert result.marginal_interval is result.simultaneous_interval is None
    assert result.interval_status == "zero_observed_variance"


def test_should_return_descriptive_only_summary_without_an_interval_policy() -> None:
    result = summarize_seeds(_observations(tuple(float(i) for i in SEEDS)), SEEDS, None)
    assert result.mean == 5.5
    assert result.marginal_interval is result.simultaneous_interval is None
    assert result.interval_status == "descriptive_only"


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), -float("inf")])
def test_should_reject_nonfinite_or_non_numeric_values(value: object) -> None:
    rows = list(_observations((1.0,) * 10))
    rows[-1] = SeedObservation(10, value, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite numeric"):
        summarize_seeds(tuple(rows), SEEDS, SPEC)


@pytest.mark.parametrize(
    "reason,value", [(None, None), ("", None), ("  ", None), (3, None), ("failure", 1.0)]
)
def test_should_require_exactly_a_value_or_nonempty_null_reason(
    reason: object, value: float | None
) -> None:
    rows = list(_observations((1.0,) * 10))
    rows[-1] = SeedObservation(10, value, reason)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason"):
        summarize_seeds(tuple(rows), SEEDS, SPEC)


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "unknown", "boolean"])
def test_should_require_complete_declared_seed_order(change: str) -> None:
    rows = list(_observations((1.0,) * 10))
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "order":
        rows.reverse()
    elif change == "unknown":
        rows[-1] = replace(rows[-1], seed=11)
    else:
        rows[0] = replace(rows[0], seed=True)
    with pytest.raises(ValueError, match="seed"):
        summarize_seeds(tuple(rows), SEEDS, SPEC)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_count", 9),
        ("sample_count", True),
        ("confidence_level", 1.0),
        ("confidence_level", float("nan")),
        ("marginal_critical", 0.0),
        ("marginal_critical", True),
        ("statement_count", 0),
        ("statement_count", True),
        ("simultaneous_critical", float("inf")),
        ("simultaneous_critical", 1.0),
        ("zero_deviation_tolerance", -1e-12),
        ("zero_deviation_tolerance", float("nan")),
    ],
)
def test_should_reject_invalid_interval_policies(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="interval"):
        summarize_seeds(_observations((1.0,) * 10), SEEDS, replace(SPEC, **{field: value}))


def test_should_reject_overflow_instead_of_emitting_nonfinite_summary() -> None:
    values = (1e308,) * 5 + (-1e308,) * 5
    with pytest.raises(ValueError, match="finite"):
        summarize_seeds(_observations(values), SEEDS, SPEC)
