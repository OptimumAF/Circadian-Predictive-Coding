"""Independent dense-isometry oracles for tiny width changes and their limits.

Compatible merge tolerances constrain represented branch coordinates. They do
not provide a global prediction-error bound under arbitrary amplification.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from math import sqrt
from typing import Any, cast

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
from src.core.linear_width_structure import (
    LinearWidthLineage,
    merge_linear_width,
    split_linear_width,
)

SCHEMES = (
    ("legacy_gaussian", 1.0),
    ("fan_in", 1.0),
    ("mean_field", 0.5),
    ("mean_field", 1.0),
    ("mean_field", 2.0),
)


def initialize(family, scheme="mean_field", gamma=1.0, hidden=2, width=3, inputs=2):
    if family == "mlp":
        return initialize_linear_scaling(
            LinearScalingConfig(scheme, inputs, width, hidden, gamma), 41
        )
    return initialize_residual_linear(
        LinearResidualConfig(scheme, inputs, width, hidden, gamma), 41
    )


def effective(state):
    return (
        effective_linear_weights(state)
        if type(state) is LinearScalingState
        else effective_residual_weights(state)
    )


def isometry(width, index):
    matrix = np.zeros((width, width + 1))
    matrix[:, :width] = np.eye(width)
    matrix[index, index] = matrix[index, -1] = 1 / sqrt(2)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(width), atol=3e-16)
    return matrix


def expected_effective(state, embedding):
    old = effective(state)
    middle = []
    for weight in old[1:-1]:
        if type(state) is LinearResidualState:
            middle.append(
                np.eye(state.config.width + 1)
                + embedding.T @ (weight - np.eye(state.config.width)) @ embedding
            )
        else:
            middle.append(embedding.T @ weight @ embedding)
    return (old[0] @ embedding, *middle, embedding.T @ old[-1])


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("hidden", (1, 2, 4))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_match_dense_map_preserve_complete_equilibrium_and_roundtrip(
    family, hidden, scheme, gamma
):
    source = initialize(family, scheme, gamma, hidden)
    originals = tuple(weight.copy() for weight in source.raw_weights)
    change = split_linear_width(source, 1)
    assert change.state.config.width == 4
    assert change.state.config.hidden_layers == hidden
    assert change.lineage.format_version == 1 and type(change.lineage.format_version) is int
    assert type(change.lineage) is LinearWidthLineage
    assert change.lineage.family == family
    assert change.lineage.source_config == source.config
    assert change.lineage.split_index == 1
    embedding = isometry(3, 1)
    expected = expected_effective(source, embedding)
    observed = effective(change.state)
    for index, (actual, reference, raw, scale) in enumerate(
        zip(
            observed,
            expected,
            change.state.raw_weights,
            change.state.config.forward_scales,
            strict=True,
        )
    ):
        np.testing.assert_allclose(actual, reference, atol=3e-14, rtol=3e-13)
        branch = reference - (
            np.eye(4) if family == "residual" and 0 < index < len(observed) - 1 else 0
        )
        np.testing.assert_allclose(raw, branch / scale, atol=3e-14, rtol=3e-13)
        for old in source.raw_weights:
            assert not np.shares_memory(raw, old)
    x, y = np.array([[0.2, -0.3], [-0.4, 0.7], [1.1, 0.2]]), np.array([[-0.8], [1.3], [0.2]])
    old_evaluation = evaluate_linear_equilibrium(effective(source), x, y)
    wide_evaluation = evaluate_linear_equilibrium(observed, x, y)
    for name in ("energy", "mse", "rescaling"):
        assert getattr(wide_evaluation, name) == pytest.approx(
            getattr(old_evaluation, name), abs=2e-12, rel=2e-11
        )
    np.testing.assert_allclose(
        wide_evaluation.predictions, old_evaluation.predictions, atol=2e-12, rtol=2e-11
    )
    for old, wide in zip(old_evaluation.latents, wide_evaluation.latents, strict=True):
        np.testing.assert_allclose(wide, old @ embedding, atol=2e-12, rtol=2e-11)
    for old, wide in zip(
        old_evaluation.residuals[:-1], wide_evaluation.residuals[:-1], strict=True
    ):
        np.testing.assert_allclose(wide, old @ embedding, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(
        wide_evaluation.residuals[-1], old_evaluation.residuals[-1], atol=2e-12, rtol=2e-11
    )
    if hidden > 1:
        complement = np.zeros(4)
        complement[1], complement[-1] = 1 / sqrt(2), -1 / sqrt(2)
        for matrix in observed[1:-1]:
            np.testing.assert_allclose(
                matrix @ complement, complement if family == "residual" else 0, atol=3e-14
            )
    merged = merge_linear_width(change.state, change.lineage)
    merged_evaluation = evaluate_linear_equilibrium(effective(merged), x, y)
    assert merged_evaluation.energy == pytest.approx(old_evaluation.energy, abs=2e-12, rel=2e-11)
    for actual, original in zip(merged.raw_weights, originals, strict=True):
        np.testing.assert_allclose(actual, original, atol=3e-13, rtol=3e-12)
    change.state.raw_weights[0].fill(999)
    for actual, original in zip(source.raw_weights, originals, strict=True):
        np.testing.assert_array_equal(actual, original)
    for actual, original in zip(merged.raw_weights, originals, strict=True):
        np.testing.assert_allclose(actual, original, atol=3e-13, rtol=3e-12)
    with pytest.raises(FrozenInstanceError):
        setattr(change.lineage, "split_index", 0)
    with pytest.raises(FrozenInstanceError):
        setattr(change, "state", source)


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("hidden", (1, 2))
def test_should_support_single_coordinate_width(family, hidden):
    source = initialize(family, hidden=hidden, width=1)
    change = split_linear_width(source, 0)
    x, y = np.array([[0.2, 0.4]]), np.array([[0.7]])
    old = evaluate_linear_equilibrium(effective(source), x, y)
    wide = evaluate_linear_equilibrium(effective(change.state), x, y)
    np.testing.assert_allclose(wide.predictions, old.predictions, atol=1e-13)
    assert wide.energy == pytest.approx(old.energy, abs=1e-13)
    merged = merge_linear_width(change.state, change.lineage)
    assert merged.config.width == 1


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_reject_off_image_scaled_branch_components_without_mutation(family):
    change = split_linear_width(initialize(family), 1)
    change.state.raw_weights[0][0, 1] += 0.1
    change.state.raw_weights[0][0, -1] -= 0.1
    originals = tuple(w.copy() for w in change.state.raw_weights)
    with pytest.raises(ValueError):
        merge_linear_width(change.state, change.lineage)
    for observed, original in zip(change.state.raw_weights, originals, strict=True):
        np.testing.assert_array_equal(observed, original)


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("position", ("output", "middle"))
def test_should_reject_off_image_output_or_middle_without_mutating_input(family, position):
    change = split_linear_width(initialize(family, hidden=2), 1)
    layer = -1 if position == "output" else 1
    # Set a clearly incompatible component in effective BRANCH coordinates.
    delta = 0.1 / change.state.config.forward_scales[layer]
    change.state.raw_weights[layer][1, 0] += delta
    change.state.raw_weights[layer][-1, 0] -= delta
    originals = tuple(weight.copy() for weight in change.state.raw_weights)
    with pytest.raises(ValueError):
        merge_linear_width(change.state, change.lineage)
    for actual, original in zip(change.state.raw_weights, originals, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_merge_mixed_zero_subnormal_pair_without_premature_rounding(family):
    gamma = float.fromhex("0x1p-530")
    change = split_linear_width(
        initialize(family, "mean_field", gamma=gamma, hidden=1, width=1, inputs=1), 0
    )
    minimum = float.fromhex("0x0.0000000000001p-1022")
    small = 2 * minimum
    wide = replace(
        change.state,
        raw_weights=(np.zeros((1, 2)), np.array([[0.0], [small]])),
    )
    originals = tuple(weight.copy() for weight in wide.raw_weights)
    # Independent high precision projection: new/old output scale is exactly
    # 1/2. Keep all factors until final float64 rounding; the answer is one ulp.
    with localcontext() as context:
        context.prec = 100
        projected = Decimal.from_float(small) / Decimal(2).sqrt() / Decimal(2)
        expected = float(projected)
    assert expected == minimum and expected > 0
    merged = merge_linear_width(wide, change.lineage)
    assert merged.raw_weights[-1][0, 0] == expected
    for actual, original in zip(wide.raw_weights, originals, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("index", (True, -1, 3, 1.0, np.int64(1), "1"))
def test_should_refuse_invalid_split_indices(index: Any):
    with pytest.raises(ValueError):
        split_linear_width(initialize("mlp"), index)


@pytest.mark.parametrize(
    "updates",
    (
        {"format_version": True},
        {"format_version": 2},
        {"family": "other"},
        {"family": "residual"},
        {"source_config": None},
        {"split_index": True},
        {"split_index": -1},
        {"split_index": 3},
        {"source_config": LinearScalingConfig("mean_field", 2, 4, 2)},
        {"source_config": LinearScalingConfig("mean_field", 2, 3, 1)},
    ),
)
def test_should_refuse_incompatible_lineage_schema_family_and_config(updates: Any):
    change = split_linear_width(initialize("mlp"), 1)
    with pytest.raises(ValueError):
        merge_linear_width(change.state, replace(change.lineage, **updates))


def test_should_refuse_foreign_state_and_lineage_families():
    invalid: Any
    for invalid in (None, {}, (np.ones((1, 1)),)):
        with pytest.raises(ValueError):
            split_linear_width(invalid, 0)
    change = split_linear_width(initialize("mlp"), 1)
    for invalid in (None, {}, change.state.config):
        with pytest.raises(ValueError):
            merge_linear_width(change.state, invalid)


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_refuse_nonfinite_mutated_arrays_atomically(family):
    source = initialize(family)
    source.raw_weights[0][0, 0] = np.nan
    originals = tuple(w.copy() for w in source.raw_weights)
    with pytest.raises((ValueError, FloatingPointError)):
        split_linear_width(source, 1)
    for observed, original in zip(source.raw_weights, originals, strict=True):
        np.testing.assert_array_equal(observed, original)
    change = split_linear_width(initialize(family), 1)
    change.state.raw_weights[-1][0, 0] = np.inf
    with pytest.raises((ValueError, FloatingPointError)):
        merge_linear_width(change.state, change.lineage)
    assert np.isinf(change.state.raw_weights[-1][0, 0])


def test_should_support_large_h1_width_without_quadratic_allocation_and_preflight_cap(monkeypatch):
    # This one large H1 case reaches the accepted one-million-entry cap exactly.
    source = initialize("mlp", "fan_in", hidden=1, width=499_999, inputs=1)
    for name in ("zeros", "ones", "empty", "full", "eye"):
        original = getattr(np, name)

        def guarded(shape, *args, _original=original, _is_eye=name == "eye", **kwargs):
            if isinstance(shape, (tuple, list)):
                dimensions = tuple(shape)
            elif _is_eye:
                dimensions = (shape, args[0] if args else shape)
            else:
                dimensions = (shape,)
            assert not (len(dimensions) == 2 and min(dimensions) > 2048), (
                "quadratic width allocation"
            )
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, name, guarded)
    change = split_linear_width(source, 123)
    assert change.state.config.width == 500_000
    merged = merge_linear_width(change.state, change.lineage)
    np.testing.assert_allclose(merged.raw_weights[0], source.raw_weights[0], atol=1e-13, rtol=1e-13)
    with pytest.raises(ValueError):
        split_linear_width(change.state, 123)


def test_should_document_tolerance_compatible_merge_can_amplify_prediction_error():
    source = LinearScalingState(
        LinearScalingConfig("legacy_gaussian", 1, 1, 1), (np.zeros((1, 1)), np.zeros((1, 1)))
    )
    change = split_linear_width(source, 0)
    eps = 1e-13
    wide = LinearScalingState(
        cast(LinearScalingConfig, change.state.config),
        (np.array([[eps, -eps]]), np.array([[1e20 + 1e9], [1e20 - 1e9]])),
    )
    merged = merge_linear_width(wide, change.lineage)
    wide_prediction = float((wide.raw_weights[0] @ wide.raw_weights[1])[0, 0])
    merged_prediction = float((merged.raw_weights[0] @ merged.raw_weights[1])[0, 0])
    assert abs(merged_prediction) < 1e-12
    assert 1e-5 < abs(wide_prediction - merged_prediction) < 1e-3


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("magnitude", (1.6e308, 1e-310))
def test_should_preserve_representable_large_and_subnormal_branches_with_zero_neighbors(
    family, magnitude
):
    source: LinearScalingState | LinearResidualState
    if family == "mlp":
        mlp_config = LinearScalingConfig("legacy_gaussian", 1, 1, 2)
        source = LinearScalingState(
            mlp_config, (np.array([[magnitude]]), np.array([[magnitude]]), np.zeros((1, 1)))
        )
    else:
        residual_config = LinearResidualConfig("legacy_gaussian", 1, 1, 2)
        source = LinearResidualState(
            residual_config, (np.array([[magnitude]]), np.array([[magnitude]]), np.zeros((1, 1)))
        )
    change = split_linear_width(source, 0)
    merged = merge_linear_width(change.state, change.lineage)
    for actual, original in zip(merged.raw_weights, source.raw_weights, strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, original, rtol=2e-13, atol=1e-323)
    assert merged.raw_weights[0][0, 0] != 0


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_execute_four_total_sgd_calls_with_new_width_dependent_factors(family):
    source = initialize(family, hidden=2)
    change = split_linear_width(source, 1)
    assert (
        source.config.effective_learning_rate_factors
        != change.state.config.effective_learning_rate_factors
    )
    # Parameterization chooses the corresponding exact state family above.
    step: Any = step_linear_pc if family == "mlp" else step_residual_linear_pc
    x, y = np.array([[0.2, -0.3], [-0.4, 0.7]]), np.array([[-0.8], [1.3]])
    for initial in (source, change.state):
        originals = tuple(w.copy() for w in initial.raw_weights)
        result = step(initial, x, y, 0.01)
        assert np.isfinite(result.evaluation.energy)
        assert all(np.isfinite(w).all() for w in result.state.raw_weights)
        assert result.state.config == initial.config
        for observed, original in zip(initial.raw_weights, originals, strict=True):
            np.testing.assert_array_equal(observed, original)
