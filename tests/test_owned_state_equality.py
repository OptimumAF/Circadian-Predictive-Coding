"""Exact state comparison preserves every nested value after scalar dispatch."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.app.coupled_attribution import _same_value


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (1.0, 1.0, True),
        (1.0, 2.0, False),
        (1, 1, True),
        (1, True, False),
        (1, 1.0, False),
        (True, True, True),
        (True, False, False),
        ("receipt", "receipt", True),
        ("receipt", "replaced", False),
        (None, None, True),
        (None, 0, False),
        (float("nan"), float("nan"), False),
        (float("inf"), float("inf"), True),
    ],
)
def test_should_keep_exact_primitive_comparison(left, right, expected):
    assert _same_value(left, right) is expected


def test_should_refuse_unsupported_values_even_when_identical_or_equal():
    class ForeignFloat(float):
        pass

    value = ForeignFloat(1.0)
    assert not _same_value(value, value)
    assert not _same_value(value, 1.0)
    assert not _same_value({1.0}, {1.0})


def test_should_recheck_nested_dataclass_values_without_identity_shortcut_or_cache():
    @dataclass
    class Snapshot:
        heads: tuple
        receipt: dict

    left = Snapshot(((0.0, 1.0), (2.0, 3.0)), {"updates": [1, 2]})
    right = Snapshot(((0.0, 1.0), (2.0, 3.0)), {"updates": [1, 2]})
    assert _same_value(left, right)
    right.receipt["updates"][1] = True
    assert not _same_value(left, right)
    right.receipt["updates"][1] = 2
    assert _same_value(left, right)
    right.heads = ((0.0, 1.0), (2.0, 4.0))
    assert not _same_value(left, right)
    shared = float("nan")
    assert not _same_value((shared,), (shared,))
    assert not _same_value(Snapshot, Snapshot)


def test_should_keep_numpy_shape_dtype_and_value_checks_on_every_call():
    left = np.array([[1.0, 2.0]])
    right = left.copy()
    assert _same_value(left, right)
    assert not _same_value(left, right.astype(np.float32))
    assert not _same_value(left, right.reshape(2, 1))
    right[0, 1] = 3.0
    assert not _same_value(left, right)
    right[0, 1] = 2.0
    assert _same_value(left, right)
    assert not _same_value({"a": [1]}, {"b": [1]})
    assert not _same_value([1], (1,))
