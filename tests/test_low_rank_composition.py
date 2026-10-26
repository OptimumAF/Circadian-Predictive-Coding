"""Literal operator oracles for bounded, nontransposed low-rank composition."""

from typing import Any

import numpy as np
import pytest

from src.core import low_rank_composition as composition
from src.core.low_rank_composition import LowRankTerm, compose_low_rank


def terms() -> tuple[LowRankTerm, ...]:
    return (
        LowRankTerm(
            np.array([[1.0, 2.0, -1.0, 0.5]]),
            np.array([[2.0], [-1.0], [0.5]]),
            coefficient=-2.0,
            scaling=0.5,
        ),
        LowRankTerm(
            np.array([[0.0, 1.0, 2.0, -1.0], [1.0, 0.0, 0.5, 2.0]]),
            np.array([[1.0, 0.0], [0.5, 2.0], [-1.0, 1.0]]),
            coefficient=0.25,
            scaling=2.0,
        ),
    )


def dense(items: tuple[LowRankTerm, ...]) -> np.ndarray:
    result = np.zeros((items[0].b.shape[0], items[0].a.shape[1]))
    for term in items:
        result += term.coefficient * term.scaling * (term.b @ term.a)
    return result


def saved(items: tuple[LowRankTerm, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(term.a.copy(), term.b.copy()) for term in items]


def assert_unchanged(
    items: tuple[LowRankTerm, ...], before: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    for term, (a, b) in zip(items, before, strict=True):
        np.testing.assert_array_equal(term.a, a)
        np.testing.assert_array_equal(term.b, b)


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("refused input reached a scan, allocation, or factorization")


def test_should_compose_signed_scaled_operators_without_factorwise_cross_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = terms()
    before = saved(items)
    monkeypatch.setattr(composition.np.linalg, "qr", forbidden)
    monkeypatch.setattr(composition.np.linalg, "svd", forbidden)
    result = compose_low_rank(items)

    np.testing.assert_allclose(result.b @ result.a, dense(items), rtol=1e-14, atol=1e-14)
    assert result.input_rank == result.retained_rank == 3
    assert result.singular_values is None
    assert result.numerical_rank is None
    assert result.discarded_frobenius == 0.0
    assert result.work["qr_calls"] == result.work["svd_calls"] == 0
    assert_unchanged(items, before)


def test_should_own_float64_outputs_independently_of_all_borrowed_inputs() -> None:
    items = terms()
    result = compose_low_rank(items)
    expected = (result.a.copy(), result.b.copy())
    assert result.a.dtype == result.b.dtype == np.dtype("float64")
    assert result.a.flags.owndata and result.b.flags.owndata
    assert not np.shares_memory(result.a, result.b)
    for term in items:
        assert not np.shares_memory(result.a, term.a)
        assert not np.shares_memory(result.b, term.b)
        term.a.fill(100.0)
        term.b.fill(-100.0)
    np.testing.assert_array_equal(result.a, expected[0])
    np.testing.assert_array_equal(result.b, expected[1])


def test_should_accept_readonly_noncontiguous_float32_factors() -> None:
    a = np.array([[1.0, 9.0, 2.0, 9.0, -1.0, 9.0]], dtype=np.float32)[:, ::2]
    b = np.array([[1.0, 9.0], [2.0, 9.0]], dtype=np.float32)[:, ::2]
    a.flags.writeable = b.flags.writeable = False
    items = (LowRankTerm(a, b, coefficient=0.5, scaling=2.0),)
    before = saved(items)
    result = compose_low_rank(items)
    np.testing.assert_array_equal(result.b @ result.a, [[1.0, 2.0, -1.0], [2.0, 4.0, -2.0]])
    assert_unchanged(items, before)
    assert not a.flags.writeable and not b.flags.writeable


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_should_match_independent_best_rank_error_and_spectrum(rank: int) -> None:
    items = terms()
    expected = dense(items)
    spectrum = np.linalg.svd(expected, compute_uv=False)
    before = saved(items)
    result = compose_low_rank(items, rank=rank)
    assert result.singular_values is not None
    np.testing.assert_allclose(result.singular_values, spectrum, rtol=1e-12, atol=1e-12)
    expected_tail = float(np.hypot.reduce(spectrum[rank:], initial=0.0))
    actual_error = float(np.linalg.norm(expected - result.b @ result.a))
    assert result.discarded_frobenius == pytest.approx(expected_tail, rel=1e-12, abs=1e-12)
    assert actual_error == pytest.approx(expected_tail, rel=1e-12, abs=1e-12)
    assert result.a.shape == (rank, 4) and result.b.shape == (3, rank)
    assert result.a.flags.owndata and result.b.flags.owndata
    for term in items:
        assert not np.shares_memory(result.a, term.a)
        assert not np.shares_memory(result.b, term.b)
    threshold = np.finfo(np.float64).eps * 4 * spectrum[0]
    assert result.numerical_rank == int(np.count_nonzero(spectrum > threshold))
    assert result.retained_rank == rank
    assert_unchanged(items, before)


def test_should_factor_small_core_when_input_rank_exceeds_both_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = (
        LowRankTerm(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]),
            np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, -1.0, 1.0]]),
        ),
    )
    original_svd = composition.np.linalg.svd
    seen: list[tuple[int, ...]] = []

    def record_svd(value: np.ndarray, **kwargs: Any) -> Any:
        seen.append(value.shape)
        return original_svd(value, **kwargs)

    monkeypatch.setattr(composition.np.linalg, "svd", record_svd)
    result = compose_low_rank(items, rank=2)
    np.testing.assert_allclose(result.b @ result.a, dense(items), atol=1e-12)
    assert result.input_rank == 4 and result.retained_rank == 2
    assert result.work["core_shape"] == (2, 2)
    assert result.work["qr_shapes"] == ((2, 4), (2, 4))
    assert seen == [(2, 2)]


@pytest.mark.parametrize("kind", ["rank_deficient", "cancellation", "zero_coefficient"])
def test_should_report_degeneracy_without_inventing_retained_information(kind: str) -> None:
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    items: tuple[LowRankTerm, ...] = (LowRankTerm(a, b),)
    expected_rank = 1
    if kind == "cancellation":
        items = (LowRankTerm(a, b), LowRankTerm(a, b, coefficient=-1.0))
        expected_rank = 0
    elif kind == "zero_coefficient":
        items = (LowRankTerm(a, b, coefficient=0.0),)
        expected_rank = 0
    result = compose_low_rank(items, rank=1)
    np.testing.assert_allclose(result.b @ result.a, dense(items), atol=1e-12)
    # QR/SVD roundoff can leave tiny cancellation singular values; compare the
    # operator and tail while only asserting exact numerical rank for zero input.
    if kind != "cancellation":
        assert result.numerical_rank == expected_rank
    assert result.discarded_frobenius == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("rank", [None, 0])
def test_should_skip_linear_algebra_for_zero_total_input_rank(
    monkeypatch: pytest.MonkeyPatch, rank: int | None
) -> None:
    item = LowRankTerm(np.empty((0, 3)), np.empty((2, 0)))
    monkeypatch.setattr(composition.np.linalg, "qr", forbidden)
    monkeypatch.setattr(composition.np.linalg, "svd", forbidden)
    result = compose_low_rank((item,), rank=rank)
    assert result.a.shape == (0, 3) and result.b.shape == (2, 0)
    assert result.input_rank == result.retained_rank == 0
    assert result.discarded_frobenius == 0.0
    assert result.work["qr_calls"] == result.work["svd_calls"] == 0
    if rank is None:
        assert result.singular_values is None and result.numerical_rank is None
    else:
        assert result.singular_values is not None and len(result.singular_values) == 0
        assert result.numerical_rank == 0


def test_should_report_coordinate_reserves_and_only_explicit_product_costs() -> None:
    items = terms()
    exact = compose_low_rank(items)
    projected = compose_low_rank(items, rank=2)
    e, r, incoming, outgoing, retained = 21, 3, 4, 3, 2
    assert exact.work["term_count"] == projected.work["term_count"] == 2
    assert exact.work["input_elements"] == projected.work["input_elements"] == e
    assert exact.work["workspace_elements_reserved"] == 3 * e
    assert exact.work["output_factor_elements"] == e
    assert exact.work["matmul_scalar_products"] == 0
    assert (
        projected.work["workspace_elements_reserved"]
        == 8 * e + 24 * r * r + 4 * (outgoing + incoming) * retained + 16 * r
    )
    assert projected.work["output_factor_elements"] == retained * (outgoing + incoming)
    assert projected.work["qr_calls"] == 2 and projected.work["svd_calls"] == 1
    assert projected.work["qr_shapes"] == ((3, 3), (4, 3))
    assert projected.work["core_shape"] == (3, 3)
    assert projected.work["matmul_scalar_products"] == 3 * 3 * 3 + 3 * 3 * 2 + 4 * 3 * 2


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "empty",
        "too_many",
        "foreign_term",
        "array_subclass",
        "integer_dtype",
        "complex_dtype",
        "dimension",
        "rank_join",
        "zero_dimension",
        "bool_coefficient",
        "bad_scaling",
        "rank_bool",
        "rank_negative",
        "rank_too_large",
        "budget_bool",
        "budget_too_large",
    ],
)
def test_should_refuse_malformed_metadata_before_any_finite_scan(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    item = terms()[0]
    items: Any = (item,)
    options: dict[str, Any] = {}
    if fault == "list":
        items = [item]
    elif fault == "empty":
        items = ()
    elif fault == "too_many":
        items = (item,) * 9
    elif fault == "foreign_term":
        items = (object(),)
    elif fault == "array_subclass":

        class ArraySubclass(np.ndarray):
            pass

        items = (LowRankTerm(item.a.view(ArraySubclass), item.b),)
    elif fault == "integer_dtype":
        items = (LowRankTerm(item.a.astype(np.int64), item.b),)
    elif fault == "complex_dtype":
        items = (LowRankTerm(item.a.astype(np.complex128), item.b),)
    elif fault == "dimension":
        items = (LowRankTerm(item.a.reshape(4), item.b),)
    elif fault == "rank_join":
        items = (LowRankTerm(item.a, np.zeros((3, 2))),)
    elif fault == "zero_dimension":
        items = (LowRankTerm(np.empty((1, 0)), item.b),)
    elif fault == "bool_coefficient":
        items = (LowRankTerm(item.a, item.b, coefficient=True),)
    elif fault == "bad_scaling":
        bad_scaling: Any = "1"
        items = (LowRankTerm(item.a, item.b, scaling=bad_scaling),)
    elif fault.startswith("rank_"):
        options["rank"] = {"rank_bool": True, "rank_negative": -1, "rank_too_large": 2}[fault]
    else:
        options["max_elements"] = True if fault == "budget_bool" else 20_000_001
    monkeypatch.setattr(composition.np, "isfinite", forbidden)
    monkeypatch.setattr(composition.np, "empty", forbidden)
    monkeypatch.setattr(composition.np.linalg, "qr", forbidden)
    monkeypatch.setattr(composition.np.linalg, "svd", forbidden)
    with pytest.raises(ValueError):
        compose_low_rank(items, **options)


def test_should_validate_later_term_dimensions_before_scanning_earlier_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = terms()[0]
    later = LowRankTerm(np.zeros((1, 5)), np.zeros((3, 1)))
    monkeypatch.setattr(composition.np, "isfinite", forbidden)
    with pytest.raises(ValueError):
        compose_low_rank((first, later))


@pytest.mark.parametrize("fault", ["nan_a", "inf_b", "nan_coefficient", "inf_scaling"])
def test_should_refuse_nonfinite_inputs_without_changing_them(fault: str) -> None:
    item = terms()[0]
    if fault == "nan_a":
        item.a[0, 0] = np.nan
    elif fault == "inf_b":
        item.b[0, 0] = np.inf
    elif fault == "nan_coefficient":
        item = LowRankTerm(item.a, item.b, coefficient=float("nan"))
    else:
        item = LowRankTerm(item.a, item.b, scaling=float("inf"))
    items = (item,)
    before = saved(items)
    with pytest.raises(ValueError):
        compose_low_rank(items)
    assert_unchanged(items, before)


@pytest.mark.parametrize("fault", ["weight_product", "scaled_factor"])
def test_should_refuse_finite_arithmetic_overflow_without_mutation(fault: str) -> None:
    item = LowRankTerm(
        np.array([[1.0, 2.0]]),
        np.array([[2.0], [1.0]]),
        coefficient=1e308,
        scaling=1e308 if fault == "weight_product" else 1.0,
    )
    before = saved((item,))
    with pytest.raises((ValueError, FloatingPointError, OverflowError)):
        compose_low_rank((item,))
    assert_unchanged((item,), before)


@pytest.mark.parametrize("stage", ["qr", "svd"])
def test_should_propagate_factorization_failure_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    items = terms()
    before = saved(items)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise np.linalg.LinAlgError("deliberate factorization failure")

    monkeypatch.setattr(composition.np.linalg, stage, fail)
    with pytest.raises(ValueError, match="deliberate"):
        compose_low_rank(items, rank=1)
    assert_unchanged(items, before)


@pytest.mark.parametrize("fault", ["rank", "dimension", "elements"])
def test_should_refuse_large_metadata_views_before_scanning_or_allocating(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    scalar = np.array([1.0])
    shape = {"rank": (65, 1, 1), "dimension": (1, 131073, 1), "elements": (64, 40000, 40000)}[fault]
    rank, incoming, outgoing = shape
    a = np.lib.stride_tricks.as_strided(
        scalar, shape=(rank, incoming), strides=(0, 0), writeable=False
    )
    b = np.lib.stride_tricks.as_strided(
        scalar, shape=(outgoing, rank), strides=(0, 0), writeable=False
    )
    monkeypatch.setattr(composition.np, "isfinite", forbidden)
    monkeypatch.setattr(composition.np, "empty", forbidden)
    monkeypatch.setattr(composition.np.linalg, "qr", forbidden)
    monkeypatch.setattr(composition.np.linalg, "svd", forbidden)
    with pytest.raises(ValueError):
        compose_low_rank((LowRankTerm(a, b),))


@pytest.mark.parametrize("rank,budget", [(None, 62), (2, 487)])
def test_should_refuse_insufficient_workspace_before_finite_scans(
    monkeypatch: pytest.MonkeyPatch, rank: int | None, budget: int
) -> None:
    items = terms()
    monkeypatch.setattr(composition.np, "isfinite", forbidden)
    monkeypatch.setattr(composition.np, "empty", forbidden)
    with pytest.raises(ValueError):
        compose_low_rank(items, rank=rank, max_elements=budget)
