"""Small analytic feature-health references; no learner or scientific comparison."""

from dataclasses import asdict
from math import exp, log
from typing import Any

import numpy as np
import pytest

from src.core.learnability_diagnostics import ActivationHealth, diagnose_activations


FIELDS = {
    "layer_index",
    "rows",
    "units",
    "active_fraction",
    "dead_fraction",
    "saturated_fraction",
    "mean_absolute_activity",
    "min_mean_absolute_activity",
    "max_mean_absolute_activity",
    "unit_mean_absolute_activity",
    "feature_effective_rank",
}


def test_should_measure_inclusive_dead_and_saturation_thresholds_as_overlapping_properties():
    values = np.array([[0.001, 0.0010001, 0.0, -0.99], [0.001, 0.0010001, 0.0, 1.0]])
    result = diagnose_activations((values,))
    assert type(result) is tuple and len(result) == 1 and type(result[0]) is ActivationHealth
    health = result[0]
    assert set(asdict(health)) == FIELDS
    assert (health.layer_index, health.rows, health.units) == (0, 2, 4)
    expected = (0.001, 0.0010001, 0.0, 0.995)
    assert health.unit_mean_absolute_activity == pytest.approx(expected)
    assert health.mean_absolute_activity == pytest.approx(sum(expected) / 4)
    assert health.min_mean_absolute_activity == 0.0
    assert health.max_mean_absolute_activity == pytest.approx(0.995)
    assert health.dead_fraction == health.active_fraction == 0.5
    assert health.saturated_fraction == 0.25
    assert health.feature_effective_rank == pytest.approx(1.0)


@pytest.mark.parametrize(
    "case,rank",
    (
        ("zero", 0.0),
        ("constant", 0.0),
        ("rank_one", 1.0),
        ("rank_two", 2.0),
        ("unequal", exp(-log(1 / 3) / 3 - 2 * log(2 / 3) / 3)),
    ),
)
def test_should_match_centered_singular_value_entropy_references(case, rank):
    if case == "zero":
        values = np.zeros((4, 2), dtype=np.float64)
    elif case == "constant":
        values = np.full((4, 2), 0.5, dtype=np.float64)
    elif case == "rank_one":
        values = np.array([[0.25, 0.25], [-0.25, -0.25], [0.0, 0.0], [0.0, 0.0]])
    else:
        second = 0.5 if case == "unequal" else 0.25
        values = np.array([[0.25, 0.0], [-0.25, 0.0], [0.0, second], [0.0, -second]])
    health = diagnose_activations((values,))[0]
    assert health.feature_effective_rank == pytest.approx(rank, abs=1e-12)
    if case == "constant":
        assert (
            health.active_fraction == 1.0
            and health.dead_fraction == health.saturated_fraction == 0.0
        )
    if case == "zero":
        assert health.dead_fraction == 1.0 and health.active_fraction == 0.0


def test_should_discard_numerical_singular_values_below_the_declared_absolute_floor():
    epsilon = np.finfo(np.float64).eps
    values = np.array([[epsilon / 16], [-epsilon / 16]])
    assert diagnose_activations((values,))[0].feature_effective_rank == 0.0


def test_should_keep_callers_arrays_order_flags_and_random_state_unchanged(monkeypatch):
    first = np.asfortranarray([[0.1, -0.2], [-0.1, 0.2]])
    first.flags.writeable = False
    second = np.array([[0.2], [-0.2]])
    before = tuple(
        (item.tobytes(order="A"), item.strides, item.flags.writeable) for item in (first, second)
    )
    original_svd = np.linalg.svd
    calls: list[tuple[int, ...]] = []

    def svd(value, *args, **kwargs):
        calls.append(value.shape)
        return original_svd(value, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("feature diagnostics accessed a random generator")

    monkeypatch.setattr(np.linalg, "svd", svd)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    result = diagnose_activations((first, second))
    assert [item.layer_index for item in result] == [0, 1]
    assert calls == [(2, 2), (2, 1)]
    assert (
        tuple(
            (item.tobytes(order="A"), item.strides, item.flags.writeable)
            for item in (first, second)
        )
        == before
    )


@pytest.mark.parametrize("boundary", ("layers", "rows", "units", "cells"))
def test_should_accept_each_inclusive_declared_geometry_boundary(boundary):
    shapes = {
        "layers": ((1, 1),) * 8,
        "rows": ((1024, 1),),
        "units": ((1, 128),),
        "cells": ((512, 128),),
    }
    values = tuple(np.zeros(shape, dtype=np.float64) for shape in shapes[boundary])
    result = diagnose_activations(values)
    assert len(result) == len(values)
    assert all(item.feature_effective_rank == 0.0 and item.dead_fraction == 1.0 for item in result)


class ForeignArray(np.ndarray):
    pass


@pytest.mark.parametrize(
    "bad",
    (
        "list",
        "empty",
        "too_many_layers",
        "none",
        "subclass",
        "float32",
        "integer",
        "object",
        "rank1",
        "rank3",
        "empty_rows",
        "empty_units",
        "nan",
        "inf",
        "low",
        "high",
        "rows",
        "units",
        "cells",
    ),
)
def test_should_refuse_invalid_inputs_and_caps_before_any_singular_value_work(monkeypatch, bad):
    values: Any = (np.zeros((1, 1), dtype=np.float64),)
    if bad == "list":
        values = list(values)
    elif bad == "empty":
        values = ()
    elif bad == "too_many_layers":
        values = values * 9
    elif bad == "none":
        values = (None,)
    elif bad == "subclass":
        values = (values[0].view(ForeignArray),)
    elif bad in ("float32", "integer", "object"):
        dtype = {"float32": np.float32, "integer": np.int64, "object": object}[bad]
        values = (np.zeros((1, 1), dtype=dtype),)
    elif bad in ("rank1", "rank3", "empty_rows", "empty_units", "rows", "units", "cells"):
        shape = {
            "rank1": (2,),
            "rank3": (1, 1, 1),
            "empty_rows": (0, 1),
            "empty_units": (1, 0),
            "rows": (1025, 1),
            "units": (1, 129),
            "cells": (513, 128),
        }[bad]
        values = (np.zeros(shape, dtype=np.float64),)
    else:
        values = (np.array([[{"nan": np.nan, "inf": np.inf, "low": -1.01, "high": 1.01}[bad]]]),)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid activation metadata or values reached SVD")

    monkeypatch.setattr(np.linalg, "svd", forbidden)
    with pytest.raises(ValueError):
        diagnose_activations(values)


def test_should_validate_every_layer_before_diagnosing_the_first_one(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a later invalid layer was discovered after SVD started")

    monkeypatch.setattr(np.linalg, "svd", forbidden)
    with pytest.raises(ValueError):
        diagnose_activations((np.zeros((2, 2)), np.array([[np.nan]])))
