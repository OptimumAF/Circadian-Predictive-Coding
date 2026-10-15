"""Independent selector/new-coordinate references for uniform raw PC/CPC prune.

Preservation is tested only for an explicitly disconnected fixture. Ordinary
selection changes the function; the reported error bounds tensor remapping only.
"""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import (
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
    predict_coupled,
)
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    effective_coupled_parameters,
    initialize_coupled_scaling,
    step_coupled_scaling,
)
from src.core.coupled_scaling_pruning import CoupledScalingPruneLineage, prune_coupled_scaling_width

SCHEMES = (
    ("legacy_gaussian", 1.0),
    ("fan_in", 1.0),
    ("mean_field", 0.5),
    ("mean_field", 1.0),
    ("mean_field", 2.0),
)


def arrays(p):
    return (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)


def state(scheme="mean_field", hidden=2, gamma=2.0, family="pc", width=3):
    policy = (
        None
        if family == "pc"
        else (
            CoupledCircadianConfig.matched_pc_control()
            if family == "neutral"
            else CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2)
        )
    )
    config = CoupledScalingConfig(
        scheme,
        2,
        width,
        hidden,
        gamma,
        precisions=tuple(0.7 + 0.3 * i for i in range(hidden)),
        circadian=policy,
    )
    initial = initialize_coupled_scaling(config, 41)
    raw = replace(
        initial.raw_parameters,
        hidden_biases=tuple(
            np.full_like(b, 0.1 * (i + 1))
            for i, b in enumerate(initial.raw_parameters.hidden_biases)
        ),
        output_bias=np.array([[-0.15]]),
    )
    wake = initial.wake_state
    if wake is not None:
        chemical = np.linspace(0.3, 1.1, width)
        assert policy is not None
        wake = replace(
            wake,
            chemical_state=chemical,
            plasticity_state=np.clip(
                np.exp(-policy.plasticity_sensitivity * chemical), policy.min_plasticity, 1
            ),
            wake_updates=4,
        )
    return CoupledScalingState(config, raw, wake)


def selected_reference(p, removed):
    width = p.hidden_weights[0].shape[1]
    keeps = tuple(tuple(j for j in range(width) if j != unit) for unit in removed)
    weights = []
    for layer, w in enumerate(p.hidden_weights):
        rows = tuple(range(w.shape[0])) if layer == 0 else keeps[layer - 1]
        weights.append(w[np.ix_(rows, keeps[layer])])
    return CoupledParameters(
        tuple(weights),
        tuple(b[:, keep] for b, keep in zip(p.hidden_biases, keeps, strict=True)),
        p.output_weight[list(keeps[-1]), :],
        p.output_bias.copy(),
    ), keeps


@pytest.mark.parametrize("hidden", (1, 2, 3))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_slice_effective_axes_then_reexpress_all_raw_weights_and_biases(
    hidden, scheme, gamma
):
    initial = state(scheme, hidden, gamma)
    before = asdict(initial)
    removed = tuple(i % 3 for i in range(hidden))
    expected, keeps = selected_reference(effective_coupled_parameters(initial), removed)
    result = prune_coupled_scaling_width(initial, removed)
    target = result.state.config
    assert target == replace(initial.config, width=2)
    assert result.lineage.source_config == initial.config
    assert result.lineage.target_config == target
    assert result.lineage.removed_units == removed and result.lineage.retained_indices == keeps
    assert 0 <= result.max_effective_relative_error <= 8 * np.finfo(np.float64).eps
    factors = (
        *target.forward_scales[:-1],
        *((target.bias_scale,) * hidden),
        target.forward_scales[-1],
        target.bias_scale,
    )
    for actual, effective, scale in zip(
        arrays(result.state.raw_parameters), arrays(expected), factors, strict=True
    ):
        np.testing.assert_allclose(
            actual, effective / scale, rtol=2 * np.finfo(np.float64).eps, atol=0
        )
    measured_error = 0.0
    for actual, reference in zip(
        arrays(effective_coupled_parameters(result.state)), arrays(expected), strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=8 * np.finfo(np.float64).eps, atol=0)
        nonzero = reference != 0
        if nonzero.any():
            measured_error = max(
                measured_error,
                float(
                    np.max(
                        np.abs(actual[nonzero] - reference[nonzero]) / np.abs(reference[nonzero])
                    )
                ),
            )
    assert result.max_effective_relative_error == pytest.approx(
        measured_error, rel=1e-12, abs=1e-30
    )
    np.testing.assert_equal(asdict(initial), before)
    if scheme == "mean_field":
        assert target.raw_learning_rate_factor != initial.config.raw_learning_rate_factor
        assert target.bias_scale != initial.config.bias_scale
        assert not np.array_equal(
            result.state.raw_parameters.hidden_biases[0],
            initial.raw_parameters.hidden_biases[0][:, keeps[0]],
        )


@pytest.mark.parametrize(
    "scheme,gamma", (("legacy_gaussian", 1.0), ("fan_in", 1.0), ("mean_field", 2.0))
)
def test_should_keep_owned_full_policy_and_select_only_last_chemistry_coordinates(scheme, gamma):
    initial = state(scheme, 3, gamma, "active")
    before = asdict(initial)
    result = prune_coupled_scaling_width(initial, (0, 1, 2))
    assert initial.wake_state is not None and result.state.wake_state is not None
    assert result.state.wake_state.wake_updates == 4
    np.testing.assert_array_equal(
        result.state.wake_state.chemical_state, initial.wake_state.chemical_state[:2]
    )
    np.testing.assert_array_equal(
        result.state.wake_state.plasticity_state, initial.wake_state.plasticity_state[:2]
    )
    assert result.state.config.circadian == initial.config.circadian
    result.state.raw_parameters.output_bias[:] = 99
    result.state.wake_state.chemical_state[:] = 99
    assert result.lineage.source_config.circadian is not None
    object.__setattr__(result.lineage.source_config.circadian, "chemical_decay", 0.1)
    np.testing.assert_equal(asdict(initial), before)
    assert result.state.config.circadian is not None
    assert result.state.config.circadian.chemical_decay == 0.9


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
def test_should_preserve_predictions_and_full_energy_only_for_explicitly_disconnected_units(family):
    initial = state(family=family)
    raw = initial.raw_parameters
    for layer, (w, b) in enumerate(zip(raw.hidden_weights, raw.hidden_biases, strict=True)):
        w[:, 1] = 0
        b[:, 1] = 0
        if layer:
            w[1, :] = 0
    raw.output_weight[1, :] = 0
    result = prune_coupled_scaling_width(initial, (1, 1))
    x, y = np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])
    values = []
    for item in (initial, result.state):
        p = effective_coupled_parameters(item)
        z = feedforward_latents(p, x)
        assert item.config.precisions is not None
        values.append(
            (predict_coupled(p, x), evaluate_coupled(p, x, y, z, item.config.precisions).energy)
        )
    np.testing.assert_allclose(values[0][0], values[1][0], rtol=1e-14, atol=1e-15)
    assert values[0][1] == pytest.approx(values[1][1], abs=1e-14)


def test_should_report_tensor_roundtrip_without_promising_function_preservation():
    initial = state("legacy_gaussian", 1, 1.0)
    initial.raw_parameters.hidden_weights[0][:] = np.array([[1, 2, 3], [2, 3, 4]])
    initial.raw_parameters.output_weight[:] = 1
    changed = prune_coupled_scaling_width(initial, (1,))
    x = np.array([[0.2, 0.4]])
    assert changed.max_effective_relative_error == 0
    assert (
        np.max(
            np.abs(
                predict_coupled(effective_coupled_parameters(initial), x)
                - predict_coupled(effective_coupled_parameters(changed.state), x)
            )
        )
        > 1e-3
    )


@pytest.mark.parametrize(
    "scheme,gamma", (("legacy_gaussian", 1.0), ("fan_in", 1.0), ("mean_field", 2.0))
)
def test_should_continue_pruned_pc_and_real_neutral_cpc_with_exact_parameter_parity(scheme, gamma):
    pc = prune_coupled_scaling_width(state(scheme, 2, gamma, "pc"), (0, 2)).state
    neutral = prune_coupled_scaling_width(state(scheme, 2, gamma, "neutral"), (0, 2)).state
    x, y = np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])
    for _ in range(2):
        a = step_coupled_scaling(pc, x, y, 0.03, 2, 0.05)
        b = step_coupled_scaling(neutral, x, y, 0.03, 2, 0.05)
        assert a.evaluation.energy == b.evaluation.energy
        np.testing.assert_equal(asdict(a.state.raw_parameters), asdict(b.state.raw_parameters))
        pc, neutral = a.state, b.state
    assert neutral.wake_state is not None and neutral.wake_state.wake_updates == 6


@pytest.mark.parametrize("gamma", (float.fromhex("0x1p-500"), float.fromhex("0x1p500")))
def test_should_preserve_representable_extreme_selected_effective_coordinates(gamma):
    initial = state("mean_field", 3, gamma)
    for index, a in enumerate(arrays(initial.raw_parameters)):
        a[:] = float.fromhex("0x1p-500") if index % 2 else float.fromhex("0x1p500")
    expected, _ = selected_reference(effective_coupled_parameters(initial), (0, 1, 2))
    result = prune_coupled_scaling_width(initial, (0, 1, 2))
    for actual, wanted in zip(
        arrays(effective_coupled_parameters(result.state)), arrays(expected), strict=True
    ):
        np.testing.assert_allclose(actual, wanted, rtol=8 * np.finfo(float).eps, atol=0)


def test_should_refuse_nonzero_effective_coordinate_when_new_raw_value_underflows():
    initial = initialize_coupled_scaling(CoupledScalingConfig("mean_field", 1, 2, 1, 0.5), 41)
    initial.raw_parameters.hidden_weights[0][:] = 0
    initial.raw_parameters.output_weight[:] = np.array(
        [[float.fromhex("0x0.0000000000001p-1022")], [0.0]]
    )
    before = asdict(initial)
    assert effective_coupled_parameters(initial).output_weight[0, 0] > 0
    with pytest.raises((ValueError, FloatingPointError)):
        prune_coupled_scaling_width(initial, (1,))
    np.testing.assert_equal(asdict(initial), before)


def test_should_refuse_unrepresentable_target_rate_configuration_before_publication():
    gamma = float.fromhex("0x1p-537") / np.sqrt(2.0)
    initial = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 1, 2, 1, float(gamma)), 41
    )
    before = asdict(initial)
    with pytest.raises((ValueError, FloatingPointError)):
        prune_coupled_scaling_width(initial, (1,))
    np.testing.assert_equal(asdict(initial), before)


@pytest.mark.parametrize(
    "removed", ([0, 1], (), (0,), (0, 1, 2), (True, 1), (np.int64(0), 1), (-1, 1), (3, 1), (0, 1.0))
)
def test_should_refuse_invalid_removed_units_without_touching_source(removed: Any):
    initial = state()
    before = asdict(initial)
    with pytest.raises(ValueError):
        prune_coupled_scaling_width(initial, removed)
    np.testing.assert_equal(asdict(initial), before)


@pytest.mark.parametrize(
    "changes",
    (
        {"format_version": True},
        {"format_version": 2},
        {"removed_units": [0, 1]},
        {"removed_units": (1,)},
        {"source_config": {}},
    ),
)
def test_should_validate_exact_lineage_schema_and_copy_configuration(changes):
    values: dict[str, Any] = dict(
        format_version=1, source_config=state().config, removed_units=(0, 1)
    )
    values.update(changes)
    with pytest.raises(ValueError):
        CoupledScalingPruneLineage(**values)


@pytest.mark.parametrize("invalid", ("family", "dtype", "nan", "width_one", "cap", "wake"))
def test_should_refuse_bad_source_state_and_geometry_without_allocating_a_candidate(invalid):
    initial: Any = state()
    if invalid == "family":
        initial = asdict(initial)
    elif invalid == "dtype":
        object.__setattr__(
            initial.raw_parameters,
            "output_weight",
            initial.raw_parameters.output_weight.astype(np.float32),
        )
    elif invalid == "nan":
        initial.raw_parameters.output_bias[:] = np.nan
    elif invalid == "width_one":
        initial = state(width=1)
    elif invalid == "cap":
        object.__setattr__(initial.config, "width", 1001)
    else:
        object.__setattr__(initial, "wake_state", {})
    with pytest.raises(ValueError):
        prune_coupled_scaling_width(initial, (0, 0))
