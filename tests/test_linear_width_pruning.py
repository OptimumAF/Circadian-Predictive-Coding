"""Independent selector, disconnected-channel and executable pruning controls."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from src.core.linear_pc_reference import evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    initialize_linear_scaling,
    step_linear_pc,
)
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
    initialize_residual_linear,
    step_residual_linear_pc,
)
from src.core.linear_width_pruning import LinearPruneChange, LinearPruneLineage, prune_linear_width

SCHEMES = (
    ("legacy_gaussian", 1.0),
    ("fan_in", 1.0),
    ("mean_field", 0.5),
    ("mean_field", 1.0),
    ("mean_field", 2.0),
)


def initialize(family, scheme="mean_field", gamma=1.0, hidden=2, width=3):
    if family == "mlp":
        return initialize_linear_scaling(LinearScalingConfig(scheme, 2, width, hidden, gamma), 41)
    return initialize_residual_linear(LinearResidualConfig(scheme, 2, width, hidden, gamma), 41)


def effective(state):
    return (
        effective_linear_weights(state)
        if type(state) is LinearScalingState
        else effective_residual_weights(state)
    )


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("hidden", (1, 2, 4))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_match_independent_dense_selector_and_rebind_raw_factors(
    family, hidden, scheme, gamma
):
    source = initialize(family, scheme, gamma, hidden)
    originals = tuple(w.copy() for w in source.raw_weights)
    selector = np.eye(3)[:, (0, 2)]
    old = effective(source)
    expected_middle = []
    for matrix in old[1:-1]:
        expected_middle.append(
            np.eye(2) + selector.T @ (matrix - np.eye(3)) @ selector
            if family == "residual"
            else selector.T @ matrix @ selector
        )
    expected = (old[0] @ selector, *expected_middle, selector.T @ old[-1])
    change = prune_linear_width(source, 1)
    assert type(change) is LinearPruneChange
    assert type(change.lineage) is LinearPruneLineage
    assert type(change.lineage.format_version) is int and change.lineage.format_version == 1
    assert change.lineage.family == family
    assert change.lineage.source_config == source.config
    assert change.lineage.prune_index == 1
    assert change.state.config.width == 2
    assert change.state.config.hidden_layers == hidden
    assert change.state.config.scheme == scheme and change.state.config.gamma0 == gamma
    for layer, (actual, raw, reference, scale) in enumerate(
        zip(
            effective(change.state),
            change.state.raw_weights,
            expected,
            change.state.config.forward_scales,
            strict=True,
        )
    ):
        np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=2e-13)
        branch = reference - (
            np.eye(2) if family == "residual" and 0 < layer < len(expected) - 1 else 0
        )
        np.testing.assert_allclose(raw, branch / scale, atol=2e-14, rtol=2e-13)
        for original in source.raw_weights:
            assert not np.shares_memory(raw, original)
    if scheme == "mean_field":
        assert (
            change.state.config.effective_learning_rate_factors
            != source.config.effective_learning_rate_factors
        )
    change.state.raw_weights[0].fill(999)
    for actual, original in zip(source.raw_weights, originals, strict=True):
        np.testing.assert_array_equal(actual, original)
    with pytest.raises(FrozenInstanceError):
        setattr(change.lineage, "prune_index", 0)
    with pytest.raises(FrozenInstanceError):
        setattr(change, "state", source)


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("hidden", (1, 2, 4))
@pytest.mark.parametrize("scheme", ("legacy_gaussian", "fan_in", "mean_field"))
def test_should_preserve_full_equilibrium_for_zero_disconnected_coordinate(family, hidden, scheme):
    initial = initialize(family, scheme, hidden=hidden)
    raw = tuple(w.copy() for w in initial.raw_weights)
    raw[0][:, 1] = 0
    for middle in raw[1:-1]:
        middle[1, :] = middle[:, 1] = 0
    raw[-1][1, :] = 0
    source = replace(initial, raw_weights=raw)
    candidate = prune_linear_width(source, 1).state
    x, y = np.array([[0.3, -0.2], [-0.7, 0.8]]), np.array([[-0.4], [1.2]])
    old = evaluate_linear_equilibrium(effective(source), x, y)
    new = evaluate_linear_equilibrium(effective(candidate), x, y)
    np.testing.assert_allclose(new.predictions, old.predictions, atol=2e-13, rtol=2e-12)
    assert new.mse == pytest.approx(old.mse, abs=2e-13, rel=2e-12)
    assert new.rescaling == pytest.approx(old.rescaling, abs=2e-13, rel=2e-12)
    assert new.energy == pytest.approx(old.energy, abs=2e-13, rel=2e-12)
    for before, after in zip(old.latents, new.latents, strict=True):
        np.testing.assert_allclose(after, before[:, (0, 2)], atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(before[:, 1], 0, atol=2e-13)


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_show_arbitrary_pruning_can_change_prediction_and_energy(family):
    config: Any = (
        LinearScalingConfig("legacy_gaussian", 1, 2, 1)
        if family == "mlp"
        else LinearResidualConfig("legacy_gaussian", 1, 2, 1)
    )
    raw = (np.array([[1.0, 2.0]]), np.array([[3.0], [4.0]]))
    source: Any = (
        LinearScalingState(config, raw) if family == "mlp" else LinearResidualState(config, raw)
    )
    candidate = prune_linear_width(source, 1).state
    old = evaluate_linear_equilibrium(effective(source), np.ones((1, 1)), np.zeros((1, 1)))
    new = evaluate_linear_equilibrium(effective(candidate), np.ones((1, 1)), np.zeros((1, 1)))
    np.testing.assert_array_equal(old.predictions, [[11]])
    np.testing.assert_array_equal(new.predictions, [[3]])
    assert old.energy != new.energy


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("index", (True, -1, 3, 1.0, np.int64(1), "1"))
def test_should_refuse_non_native_or_out_of_range_indices(family, index: Any):
    with pytest.raises(ValueError):
        prune_linear_width(initialize(family), index)


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_refuse_width_one(family):
    with pytest.raises(ValueError):
        prune_linear_width(initialize(family, width=1), 0)


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("kind", ("tuple", "dtype", "shape", "nan", "infinity", "config_family"))
def test_should_refuse_mutated_schema_and_nonfinite_arrays_atomically(family, kind):
    source = initialize(family)
    raw: Any = tuple(w.copy() for w in source.raw_weights)
    if kind == "tuple":
        object.__setattr__(source, "raw_weights", list(raw))
    elif kind == "dtype":
        object.__setattr__(source, "raw_weights", (raw[0].astype(np.float32), *raw[1:]))
    elif kind == "shape":
        object.__setattr__(source, "raw_weights", (np.ones((2, 4)), *raw[1:]))
    elif kind == "config_family":
        wrong = (
            LinearResidualConfig("mean_field", 2, 3, 2)
            if family == "mlp"
            else LinearScalingConfig("mean_field", 2, 3, 2)
        )
        object.__setattr__(source, "config", wrong)
    else:
        source.raw_weights[0][0, 0] = np.nan if kind == "nan" else np.inf
    before = tuple(w.copy() for w in source.raw_weights)
    with pytest.raises((ValueError, FloatingPointError)):
        prune_linear_width(source, 1)
    for actual, original in zip(source.raw_weights, before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_should_refuse_foreign_state_family():
    invalid: Any
    for invalid in (None, {}, (np.ones((1, 1)),)):
        with pytest.raises(ValueError):
            prune_linear_width(invalid, 0)


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_run_exactly_four_sgd_continuation_calls(family):
    source = initialize(family)
    candidate = prune_linear_width(source, 1).state
    step: Any = step_linear_pc if family == "mlp" else step_residual_linear_pc
    for state in (source, candidate):
        before = tuple(w.copy() for w in state.raw_weights)
        result = step(state, np.array([[0.2, -0.4]]), np.array([[0.7]]), 0.01)
        assert np.isfinite(result.evaluation.energy)
        assert all(np.isfinite(w).all() for w in result.state.raw_weights)
        for actual, original in zip(state.raw_weights, before, strict=True):
            np.testing.assert_array_equal(actual, original)
