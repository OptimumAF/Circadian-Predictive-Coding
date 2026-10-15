"""Independent dyadic nonlinear energy identities and exact compatibility guards.

The tests do not claim source/split SGD trajectories agree or raw roundtrips are
bitwise exact. Fixed latent inference uses the explicitly doubled step size.
"""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.core.coupled_scaling_structure as structure
from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.adapters.topology_coupled_scaling_learner import TopologyCoupledScalingLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledParameters, evaluate_coupled, predict_coupled
from src.core.coupled_inference import CoupledInferencePolicy, infer_coupled
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.coupled_scaling_structure import (
    CoupledScalingSplitLineage,
    merge_coupled_scaling_width,
    split_coupled_scaling_width,
)
from src.core.coupled_scaling_topology import CoupledScalingTopologyState

SCHEMES = (
    ("legacy_gaussian", 1.0),
    ("fan_in", 1.0),
    ("mean_field", 0.5),
    ("mean_field", 1.0),
    ("mean_field", 2.0),
)


def arrays(p):
    return (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)


def initial(scheme="mean_field", hidden=2, gamma=2.0, family="pc", width=3, precisions=None):
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
        precisions=tuple(0.7 + 0.6 * i for i in range(hidden))
        if precisions is None
        else precisions,
        circadian=policy,
    )
    state = initialize_coupled_scaling(config, 41)
    raw = replace(
        state.raw_parameters,
        hidden_biases=tuple(
            np.full_like(b, 0.1 * (i + 1)) for i, b in enumerate(state.raw_parameters.hidden_biases)
        ),
        output_bias=np.array([[-0.15]]),
    )
    wake = state.wake_state
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


def repeat_columns(a):
    return a[:, np.arange(a.shape[1]).repeat(2)]


def split_reference(p):
    weights = []
    for i, w in enumerate(p.hidden_weights):
        rows = np.arange(w.shape[0]) if i == 0 else np.arange(w.shape[0]).repeat(2)
        weights.append(w[np.ix_(rows, np.arange(w.shape[1]).repeat(2))] / (1 if i == 0 else 2))
    return CoupledParameters(
        tuple(weights),
        tuple(repeat_columns(b) for b in p.hidden_biases),
        p.output_weight[np.arange(len(p.output_weight)).repeat(2), :] / 2,
        p.output_bias.copy(),
    )


def merge_reference(p):
    return CoupledParameters(
        (p.hidden_weights[0][:, ::2].copy(), *(w[::2, ::2] * 2 for w in p.hidden_weights[1:])),
        tuple(b[:, ::2].copy() for b in p.hidden_biases),
        p.output_weight[::2, :] * 2,
        p.output_bias.copy(),
    )


def assert_encoding(result, expected):
    config = result.state.config
    factors = (
        *config.forward_scales[:-1],
        *((config.bias_scale,) * config.hidden_layers),
        config.forward_scales[-1],
        config.bias_scale,
    )
    for raw, value, factor in zip(
        arrays(result.state.raw_parameters), arrays(expected), factors, strict=True
    ):
        np.testing.assert_allclose(raw, value / factor, rtol=2 * np.finfo(float).eps, atol=0)
    error = 0.0
    for actual, value in zip(
        arrays(effective_coupled_parameters(result.state)), arrays(expected), strict=True
    ):
        np.testing.assert_allclose(actual, value, rtol=8 * np.finfo(float).eps, atol=0)
        mask = value != 0
        if mask.any():
            error = max(
                error, float(np.max(np.abs(actual[mask] - value[mask]) / np.abs(value[mask])))
            )
    assert result.max_effective_relative_error == pytest.approx(error, rel=1e-12, abs=1e-30)


@pytest.mark.parametrize("hidden", (1, 2, 3))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_map_every_effective_tensor_and_preserve_arbitrary_latent_energy_and_all_gradients(
    hidden, scheme, gamma
):
    source = initial(scheme, hidden, gamma)
    before = asdict(source)
    p = effective_coupled_parameters(source)
    change = split_coupled_scaling_width(source)
    assert source.config.precisions is not None and change.state.config.precisions is not None
    assert change.lineage.replication_factor == 2
    expected_config = replace(
        source.config, width=6, precisions=tuple(beta / 2 for beta in source.config.precisions)
    )
    assert change.state.config == change.lineage.target_config == expected_config
    assert_encoding(change, split_reference(p))
    x, y = np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])
    z = tuple(np.array([[0.4, -0.2, 0.6], [-0.3, 0.7, 0.1]]) + 0.1 * i for i in range(hidden))
    doubled = tuple(repeat_columns(a) for a in z)
    a = evaluate_coupled(p, x, y, z, source.config.precisions)
    q = effective_coupled_parameters(change.state)
    b = evaluate_coupled(q, x, y, doubled, change.state.config.precisions)
    assert b.energy == pytest.approx(a.energy, rel=2e-13, abs=2e-14)
    assert b.task_loss == pytest.approx(a.task_loss, rel=2e-13, abs=2e-14)
    np.testing.assert_allclose(b.hidden_penalties, a.hidden_penalties, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(
        doubled[-1] @ q.output_weight + q.output_bias,
        z[-1] @ p.output_weight + p.output_bias,
        rtol=2e-13,
        atol=2e-14,
    )
    for g, h in zip(a.latent_gradients, b.latent_gradients, strict=True):
        np.testing.assert_allclose(h, repeat_columns(g) / 2, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(predict_coupled(q, x), predict_coupled(p, x), rtol=2e-13, atol=2e-14)
    merged = merge_coupled_scaling_width(change.state, change.lineage)
    assert merged.state.config == source.config
    assert_encoding(merged, merge_reference(q))
    np.testing.assert_equal(asdict(source), before)


@pytest.mark.parametrize("hidden", (1, 3))
@pytest.mark.parametrize(
    "scheme,gamma", (("legacy_gaussian", 1.0), ("fan_in", 1.0), ("mean_field", 2.0))
)
def test_should_match_fixed_two_step_inference_only_with_doubled_replica_learning_rate(
    hidden, scheme, gamma
):
    source = initial(scheme, hidden, gamma)
    change = split_coupled_scaling_width(source)
    assert source.config.precisions is not None and change.state.config.precisions is not None
    x, y = np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])
    policy = CoupledInferencePolicy(2, None, None)
    a = infer_coupled(
        effective_coupled_parameters(source),
        x,
        y,
        source.config.precisions,
        max_steps=2,
        learning_rate=0.02,
        policy=policy,
    )
    b = infer_coupled(
        effective_coupled_parameters(change.state),
        x,
        y,
        change.state.config.precisions,
        max_steps=2,
        learning_rate=0.04,
        policy=policy,
    )
    assert a.diagnostics.reason == b.diagnostics.reason == "max_steps"
    assert a.diagnostics.accepted_steps == b.diagnostics.accepted_steps == 2
    assert not a.diagnostics.used_fallback and not b.diagnostics.used_fallback
    for z, w in zip(a.latents, b.latents, strict=True):
        np.testing.assert_allclose(w, repeat_columns(z), rtol=2e-13, atol=2e-14)
    assert a.evaluation.energy == pytest.approx(b.evaluation.energy, rel=2e-13, abs=2e-14)


@pytest.mark.parametrize(
    "scheme,gamma", (("legacy_gaussian", 1.0), ("fan_in", 1.0), ("mean_field", 2.0))
)
def test_should_continue_symmetric_owned_forks_merge_trained_pairs_and_keep_pc_neutral_exact(
    scheme, gamma
):
    completed = {}
    for family in ("pc", "neutral", "active"):
        source = initial(scheme, 2, gamma, family)
        before = asdict(source)
        change = split_coupled_scaling_width(source)
        learner = CoupledScalingLearner(
            change.state, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.03
        )
        fork = learner.fork()
        x, y = np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])
        learner.train_batch(x, y)
        fork.train_batch(x, y)
        saved = learner.snapshot_state()
        np.testing.assert_equal(asdict(saved), asdict(fork.snapshot_state()))
        merged = merge_coupled_scaling_width(saved, change.lineage)
        assert merged.state.config == source.config
        assert_encoding(merged, merge_reference(effective_coupled_parameters(saved)))
        if saved.wake_state is not None:
            assert merged.state.wake_state is not None
            assert saved.wake_state.wake_updates == merged.state.wake_state.wake_updates == 5
            np.testing.assert_array_equal(
                merged.state.wake_state.chemical_state, saved.wake_state.chemical_state[::2]
            )
        np.testing.assert_equal(asdict(source), before)
        completed[family] = saved
    np.testing.assert_equal(
        asdict(completed["pc"].raw_parameters), asdict(completed["neutral"].raw_parameters)
    )


def test_should_clone_full_cpc_trace_and_own_states_policy_and_lineage_on_both_operations():
    source = initial(family="active")
    before = asdict(source)
    change = split_coupled_scaling_width(source)
    assert source.wake_state is not None and change.state.wake_state is not None
    np.testing.assert_array_equal(
        change.state.wake_state.chemical_state, np.repeat(source.wake_state.chemical_state, 2)
    )
    np.testing.assert_array_equal(
        change.state.wake_state.plasticity_state, np.repeat(source.wake_state.plasticity_state, 2)
    )
    assert change.state.wake_state.wake_updates == 4
    merged = merge_coupled_scaling_width(change.state, change.lineage)
    change.state.raw_parameters.output_bias[:] = 99
    change.state.wake_state.chemical_state[:] = 99
    object.__setattr__(change.lineage.source_config.circadian, "min_plasticity", 1.0)
    assert (
        merged.state.config.circadian is not None
        and merged.state.config.circadian.min_plasticity == 0.2
    )
    np.testing.assert_equal(asdict(source), before)


@pytest.mark.parametrize(
    "group",
    (
        "first",
        "middle1_rows",
        "middle1_cols",
        "middle2_rows",
        "middle2_cols",
        "bias0",
        "bias1",
        "bias2",
        "output",
        "chemistry",
        "gates",
    ),
)
def test_should_require_exact_pairs_for_every_raw_coordinate_group_and_full_wake_state(group):
    source = initial(hidden=3, family="neutral")
    change = split_coupled_scaling_width(source)
    p = change.state.raw_parameters
    if group == "first":
        p.hidden_weights[0][0, 0] = np.nextafter(p.hidden_weights[0][0, 0], np.inf)
    elif group.startswith("middle"):
        layer = int(group[6])
        if group.endswith("rows"):
            p.hidden_weights[layer][0, :] = np.nextafter(p.hidden_weights[layer][0, :], np.inf)
        else:
            p.hidden_weights[layer][:, 0] = np.nextafter(p.hidden_weights[layer][:, 0], np.inf)
    elif group.startswith("bias"):
        b = p.hidden_biases[int(group[-1])]
        b[0, 0] = np.nextafter(b[0, 0], np.inf)
    elif group == "output":
        p.output_weight[0, 0] = np.nextafter(p.output_weight[0, 0], np.inf)
    else:
        assert change.state.wake_state is not None
        if group == "chemistry":
            change.state.wake_state.chemical_state[0] += 0.1
        else:
            change.state.wake_state.plasticity_state[0] = 0.5
    before = asdict(change.state)
    with pytest.raises((ValueError, FloatingPointError)):
        merge_coupled_scaling_width(change.state, change.lineage)
    np.testing.assert_equal(asdict(change.state), before)


def tiny_state(hidden=1):
    config = CoupledScalingConfig("legacy_gaussian", 1, 1, hidden)
    return CoupledScalingState(
        config,
        CoupledParameters(
            tuple(np.zeros((1, 1)) for _ in range(hidden)),
            tuple(np.zeros((1, 1)) for _ in range(hidden)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
        ),
    )


@pytest.mark.parametrize("kind", ("output", "middle"))
@pytest.mark.parametrize("subnormal", (1, 3))
def test_should_refuse_irreversible_effective_halves_before_raw_encoding(
    kind, subnormal, monkeypatch
):
    source = tiny_state(2 if kind == "middle" else 1)
    value = subnormal * np.nextafter(0.0, 1.0)
    if kind == "middle":
        source.raw_parameters.hidden_weights[1][:] = value
    else:
        source.raw_parameters.output_weight[:] = value
    before = asdict(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("irreversible half reached encoding")

    monkeypatch.setattr(structure, "_encode", forbidden)
    with pytest.raises((ValueError, FloatingPointError)):
        split_coupled_scaling_width(source)
    np.testing.assert_equal(asdict(source), before)


def test_should_accept_reversible_subnormal_halves_and_retain_zero_signs():
    source = tiny_state(2)
    unit = np.nextafter(0.0, 1.0)
    source.raw_parameters.hidden_weights[0][:] = 3 * unit
    source.raw_parameters.hidden_weights[1][:] = -2 * unit
    source.raw_parameters.hidden_biases[0][:] = -0.0
    source.raw_parameters.hidden_biases[1][:] = 3 * unit
    source.raw_parameters.output_weight[:] = 2 * unit
    source.raw_parameters.output_bias[:] = -0.0
    change = split_coupled_scaling_width(source)
    merged = merge_coupled_scaling_width(change.state, change.lineage)
    for a, b in zip(
        arrays(merged.state.raw_parameters), arrays(source.raw_parameters), strict=True
    ):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.signbit(a), np.signbit(b))


@pytest.mark.parametrize("kind", ("output", "middle"))
def test_should_refuse_merge_effective_doubling_overflow_before_raw_encoding(kind, monkeypatch):
    source = tiny_state(2 if kind == "middle" else 1)
    change = split_coupled_scaling_width(source)
    if kind == "middle":
        change.state.raw_parameters.hidden_weights[1][:] = 1e308
    else:
        change.state.raw_parameters.output_weight[:] = 1e308
    before = asdict(change.state)

    def forbidden(*args, **kwargs):
        raise AssertionError("overflowing double reached encoding")

    monkeypatch.setattr(structure, "_encode", forbidden)
    with pytest.raises((ValueError, FloatingPointError)):
        merge_coupled_scaling_width(change.state, change.lineage)
    np.testing.assert_equal(asdict(change.state), before)


def test_should_refuse_unrepresentable_expanded_raw_bias_without_touching_valid_source():
    config = CoupledScalingConfig("mean_field", 1, 1, 1, 2.0)
    source = CoupledScalingState(
        config,
        CoupledParameters(
            (np.zeros((1, 1)),), (np.full((1, 1), 1.5e308),), np.zeros((1, 1)), np.zeros((1, 1))
        ),
    )
    before = asdict(source)
    with pytest.raises((ValueError, FloatingPointError)):
        split_coupled_scaling_width(source)
    np.testing.assert_equal(asdict(source), before)


@pytest.mark.parametrize("kind", ("target_cap", "target_precision"))
def test_should_validate_target_configuration_before_source_reconstruction_or_expansion(
    kind, monkeypatch
):
    source = initial(width=500) if kind == "target_cap" else initial(hidden=1, precisions=(1e-6,))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid target allocated or reconstructed")

    monkeypatch.setattr(structure, "_validated_model", forbidden)
    monkeypatch.setattr(structure.np, "repeat", forbidden)
    with pytest.raises(ValueError):
        split_coupled_scaling_width(source)


@pytest.mark.parametrize("operation", ("split", "merge"))
def test_should_keep_source_and_incoming_lineage_owned_after_late_mutating_encoder_fault(
    operation, monkeypatch
):
    old = initial()
    change = split_coupled_scaling_width(old)
    source = old if operation == "split" else change.state
    before, metadata = asdict(source), asdict(change.lineage)

    def fail(config, parameters):
        for a in arrays(parameters):
            a[:] = 999
        object.__setattr__(config, "gamma0", 3.0)
        raise RuntimeError("late encoder fault")

    monkeypatch.setattr(structure, "_encode", fail)
    with pytest.raises(RuntimeError, match="late encoder fault"):
        if operation == "split":
            split_coupled_scaling_width(source)
        else:
            merge_coupled_scaling_width(source, change.lineage)
    np.testing.assert_equal(asdict(source), before)
    np.testing.assert_equal(asdict(change.lineage), metadata)


@pytest.mark.parametrize(
    "kind", ("source", "gamma", "precision", "scheme", "input", "policy", "format", "family")
)
def test_should_bind_merge_to_exact_complete_source_and_target_metadata(kind):
    source = initial()
    change = split_coupled_scaling_width(source)
    lineage: Any = change.lineage
    wider = change.state
    if kind == "source":
        wider = source
    elif kind == "format":
        object.__setattr__(lineage, "format_version", True)
    elif kind == "family":
        lineage = asdict(lineage)
    else:
        updates: dict[str, dict[str, Any]] = {
            "gamma": {"gamma0": 1.0},
            "precision": {"precisions": (1.0, 1.0)},
            "scheme": {"scheme": "fan_in", "gamma0": 1.0},
            "input": {"input_dim": 3},
            "policy": {"circadian": CoupledCircadianConfig()},
        }
        lineage = CoupledScalingSplitLineage(1, replace(source.config, **updates[kind]))
    before = asdict(wider)
    with pytest.raises(ValueError):
        merge_coupled_scaling_width(wider, lineage)
    np.testing.assert_equal(asdict(wider), before)


@pytest.mark.parametrize(
    "bad", ("format_bool", "format_version", "config_family", "width", "precision", "depth")
)
def test_should_revalidate_lineage_configuration_and_native_format(bad):
    source = initial()
    lineage = CoupledScalingSplitLineage(1, source.config)
    if bad == "format_bool":
        object.__setattr__(lineage, "format_version", True)
    elif bad == "format_version":
        object.__setattr__(lineage, "format_version", 2)
    elif bad == "config_family":
        object.__setattr__(lineage, "source_config", {})
    else:
        object.__setattr__(
            lineage.source_config,
            {"width": "width", "precision": "precisions", "depth": "hidden_layers"}[bad],
            {"width": True, "precision": (1e-6, 1e-6), "depth": 17}[bad],
        )
    with pytest.raises(ValueError):
        _ = lineage.target_config


@pytest.mark.parametrize("bad", ("family", "dtype", "nan", "wake", "raw_tuple", "envelope"))
def test_should_refuse_foreign_or_mutated_full_states_before_transform(bad):
    source: Any = initial(family="active")
    if bad == "family":
        source = asdict(source)
    elif bad == "envelope":
        source = CoupledScalingTopologyState(1, source)
    elif bad == "dtype":
        object.__setattr__(source.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32))
    elif bad == "nan":
        source.raw_parameters.output_bias[:] = np.nan
    elif bad == "wake":
        source.wake_state.plasticity_state[:] = 0
    else:
        object.__setattr__(
            source.raw_parameters, "hidden_weights", list(source.raw_parameters.hidden_weights)
        )
    with pytest.raises(ValueError):
        split_coupled_scaling_width(source)


def test_should_keep_existing_strict_restore_and_topology_precision_pin_unchanged():
    source = initial()
    strict = CoupledScalingLearner(
        source, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.03
    )
    topology = TopologyCoupledScalingLearner(
        source, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.03
    )
    a, b = asdict(strict.snapshot_state()), asdict(topology.snapshot_state())
    change = split_coupled_scaling_width(source)
    with pytest.raises(ValueError):
        strict.restore_state(change.state)
    envelope = CoupledScalingTopologyState(1, change.state)
    with pytest.raises(ValueError):
        topology.restore_state(envelope)
    with pytest.raises(ValueError):
        topology.build_from_state(envelope)
    np.testing.assert_equal(asdict(strict.snapshot_state()), a)
    np.testing.assert_equal(asdict(topology.snapshot_state()), b)
