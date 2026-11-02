"""Shared packet algebra, without fitting or experiment work."""

from copy import deepcopy

import numpy as np
import pytest

from src.core.shared_linear_replay import compose_shared_packet


def packets():
    x: np.ndarray = np.eye(4, dtype=np.float64)
    y = np.array([[0.0], [1.0], [2.0], [0.0]])
    return (x[:2], y[:2]), ((x[2:], y[2:]),)


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_joint_and_control_have_equal_rows_owned_arrays_and_unchanged_inputs(control):
    current, retained = packets()
    pin = deepcopy((current, retained))
    x, y = compose_shared_packet(current, retained, 3, control=control)
    expected = (
        (np.eye(4), np.array([[0.0], [1.0], [2.0], [0.0]]))
        if control == "joint"
        else (np.tile(current[0], (2, 1)), np.tile(current[1], (2, 1)))
    )
    assert x.shape == (4, 4) and y.shape == (4, 1)
    assert np.array_equal(x, expected[0]) and np.array_equal(y, expected[1])
    assert not any(
        np.shares_memory(x, a) or np.shares_memory(y, a) for p in (current, *retained) for a in p
    )
    assert all(
        np.array_equal(a, b)
        for pairs in zip((current, *retained), (pin[0], *pin[1]), strict=True)
        for a, b in zip(*pairs, strict=True)
    )


@pytest.mark.parametrize("control", [True, None, "latest"])
def test_packet_refuses_foreign_controls(control):
    with pytest.raises(ValueError):
        current, retained = packets()
        compose_shared_packet(current, retained, 3, control=control)


def test_packet_refuses_conflicting_retained_labels_for_both_controls():
    current, _ = packets()
    retained = ((current[0].copy(), np.array([[2.0], [1.0]])),)
    for control in ("joint", "current_only"):
        with pytest.raises(ValueError, match="conflict"):
            compose_shared_packet(current, retained, 3, control=control)


def test_packet_refuses_oversized_or_incomplete_control_exposure():
    x: np.ndarray = np.zeros((128, 4), dtype=np.float64)
    y: np.ndarray = np.zeros((128, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="128-row"):
        compose_shared_packet((x, y), ((x[:1], y[:1]),), 3, control="joint")
    with pytest.raises(ValueError, match="complete current"):
        compose_shared_packet((x[:2], y[:2]), ((x[:1], y[:1]),), 3, control="current_only")


def test_mixed_public_context_rows_do_not_require_context_routed_heads():
    current = (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0.0], [1.0]]))
    retained = ((np.array([[0.0, 0.0, 1.0]]), np.array([[2.0]])),)
    assert len(compose_shared_packet(current, retained, 3, control="joint")[0]) == 3
