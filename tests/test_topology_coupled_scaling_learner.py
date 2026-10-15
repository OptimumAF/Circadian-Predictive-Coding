"""Width-aware raw ports preserve fixed bindings, policy and complete ownership."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_scaling_learner as strict_adapter
from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.adapters.topology_coupled_scaling_learner import TopologyCoupledScalingLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledEvaluation
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingStep,
    copy_coupled_scaling_state,
    initialize_coupled_scaling,
    step_coupled_scaling,
)
from src.core.coupled_scaling_pruning import prune_coupled_scaling_width
from src.core.coupled_scaling_topology import CoupledScalingTopologyState


def raw(family="pc", **changes):
    policy = (
        None
        if family == "pc"
        else (
            CoupledCircadianConfig.matched_pc_control()
            if family == "neutral"
            else CoupledCircadianConfig()
        )
    )
    config: dict[str, Any] = dict(
        scheme="mean_field",
        input_dim=2,
        width=3,
        hidden_layers=2,
        gamma0=2.0,
        precisions=(0.7, 1.3),
        circadian=policy,
    )
    config.update(changes)
    return initialize_coupled_scaling(CoupledScalingConfig(**config), 41)


def learner(initial=None, **changes):
    settings: dict[str, Any] = dict(
        learning_rate=0.03, inference_steps=2, inference_learning_rate=0.05
    )
    settings.update(changes)
    return TopologyCoupledScalingLearner(raw() if initial is None else initial, **settings)


def pruned(initial):
    change = prune_coupled_scaling_width(initial, (1, 1))
    return CoupledScalingTopologyState(1, change.state, (change.lineage,))


def batch():
    return np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
def test_should_continue_changed_width_with_manual_raw_reference_and_exact_fork_restore(family):
    original = raw(family)
    source = learner(original)
    original_envelope = source.snapshot_state()
    target = pruned(original)
    built = source.build_from_state(target)
    x, y = batch()
    diagnostic = built.train_batch(x, y)
    reference = step_coupled_scaling(target.raw_state, x, y, 0.03, 2, 0.05)
    assert diagnostic.value == reference.evaluation.energy
    np.testing.assert_equal(asdict(built.snapshot_state().raw_state), asdict(reference.state))
    saved = built.snapshot_state()
    fork = built.fork()
    fork.train_batch(x, y)
    restored = source.build_from_state(saved)
    restored.train_batch(x, y)
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(fork.snapshot_state()))
    np.testing.assert_equal(asdict(built.snapshot_state()), asdict(saved))
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(original_envelope))
    assert restored.snapshot_state().prune_lineage == target.prune_lineage
    source.restore_state(target)
    assert source.snapshot_state().raw_state.config.width == 2
    source.restore_state(original_envelope)
    assert (
        source.snapshot_state().raw_state.config.width == 3
        and source.snapshot_state().prune_lineage == ()
    )
    target.raw_state.raw_parameters.output_bias[:] = 999
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(original_envelope))


def test_should_own_constructor_snapshots_history_and_policy_across_fork_without_updates():
    initial = raw("active")
    source = learner(initial)
    expected = asdict(source.snapshot_state())
    initial.raw_parameters.output_bias[:] = 99
    assert initial.config.circadian is not None
    object.__setattr__(initial.config.circadian, "min_plasticity", 1.0)
    fork = source.fork()
    snapshot = fork.snapshot_state()
    snapshot.raw_state.raw_parameters.hidden_weights[0][:] = 99
    assert snapshot.raw_state.wake_state is not None
    snapshot.raw_state.wake_state.chemical_state[:] = 99
    np.testing.assert_equal(asdict(fork.snapshot_state()), expected)
    np.testing.assert_equal(asdict(source.snapshot_state()), expected)


def test_should_copy_fixed_inference_policy_and_rates_through_builder_restore_and_fork(monkeypatch):
    policy = CoupledInferencePolicy(gradient_tolerance=0.3)
    source = learner(inference_policy=policy)
    target = pruned(raw())
    built = source.build_from_state(target)
    calls = []

    def fake_step(state, x, y, rate, steps, inference_rate, *, inference_policy=None):
        calls.append((rate, steps, inference_rate, deepcopy(inference_policy)))
        candidate = copy_coupled_scaling_state(state)
        candidate.raw_parameters.output_bias[:] += 1
        p = candidate.raw_parameters
        gradients = replace(
            p,
            hidden_weights=tuple(np.zeros_like(w) for w in p.hidden_weights),
            hidden_biases=tuple(np.zeros_like(b) for b in p.hidden_biases),
            output_weight=np.zeros_like(p.output_weight),
            output_bias=np.zeros_like(p.output_bias),
        )
        z = tuple(np.zeros((len(x), state.config.width)) for _ in range(state.config.hidden_layers))
        return CoupledScalingStep(
            candidate, CoupledEvaluation(0.7, 0.7, (0.0, 0.0), z, gradients), z, None
        )

    monkeypatch.setattr(strict_adapter, "step_coupled_scaling", fake_step)
    object.__setattr__(policy, "gradient_tolerance", 99.0)
    fork = built.fork()
    fork.restore_state(target)
    fork.train_batch(*batch())
    assert calls == [(0.03, 2, 0.05, CoupledInferencePolicy(gradient_tolerance=0.3))]
    np.testing.assert_array_equal(built.snapshot_state().raw_state.raw_parameters.output_bias, 0)
    assert fork.snapshot_state().prune_lineage == target.prune_lineage


@pytest.mark.parametrize(
    "changes",
    (
        {"scheme": "fan_in", "gamma0": 1.0},
        {"gamma0": 1.0},
        {"input_dim": 3},
        {"hidden_layers": 1, "precisions": (0.7,)},
        {"precisions": (1.0, 1.0)},
        {"circadian": CoupledCircadianConfig()},
    ),
)
def test_should_pin_all_non_width_configuration_on_restore_and_builder(changes):
    source = learner()
    before = asdict(source.snapshot_state())
    wrong = CoupledScalingTopologyState(1, raw(**changes))
    with pytest.raises(ValueError):
        source.restore_state(wrong)
    with pytest.raises(ValueError):
        source.build_from_state(wrong)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


def test_should_pin_complete_active_policy_and_keep_default_raw_adapter_restore_strict():
    initial = raw("active")
    source = learner(initial)
    wrong = CoupledScalingTopologyState(
        1, raw("active", circadian=CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2))
    )
    before = asdict(source.snapshot_state())
    with pytest.raises(ValueError):
        source.restore_state(wrong)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
    strict = CoupledScalingLearner(
        initial, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.05
    )
    strict_before = asdict(strict.snapshot_state())
    with pytest.raises(ValueError):
        strict.restore_state(pruned(initial).raw_state)
    np.testing.assert_equal(asdict(strict.snapshot_state()), strict_before)


def test_should_distinguish_exact_large_native_gamma_values_even_when_float_factors_match():
    gamma = 2**53
    source = learner(raw(gamma0=gamma))
    before = asdict(source.snapshot_state())
    different = CoupledScalingTopologyState(1, raw(gamma0=gamma + 1))
    assert float(gamma) == float(gamma + 1)
    with pytest.raises(ValueError):
        source.restore_state(different)
    with pytest.raises(ValueError):
        source.build_from_state(different)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


def test_should_bind_valid_small_gamma_without_constructing_unrepresentable_width_one():
    gamma = float(float.fromhex("0x1p-537") / np.sqrt(2.0))
    initial = raw(gamma0=gamma)
    source = learner(initial)
    changed = pruned(initial)
    built = source.build_from_state(changed)
    assert built.snapshot_state().raw_state.config.width == 2
    assert built.snapshot_state().raw_state.config.gamma0 == gamma


@pytest.mark.parametrize("kind", ("format", "raw", "nan", "geometry", "history", "wake"))
def test_should_prepare_complete_payload_before_publishing_geometry_or_history(kind):
    source = learner(raw("active"))
    before = asdict(source.snapshot_state())
    target = pruned(raw("active"))
    if kind == "format":
        object.__setattr__(target, "format_version", True)
    elif kind == "raw":
        object.__setattr__(target, "raw_state", {})
    elif kind == "nan":
        target.raw_state.raw_parameters.output_bias[:] = np.nan
    elif kind == "geometry":
        object.__setattr__(target.raw_state.config, "width", 3)
    elif kind == "history":
        object.__setattr__(target, "prune_lineage", target.prune_lineage * 2)
    else:
        assert target.raw_state.wake_state is not None
        target.raw_state.wake_state.plasticity_state[:] = 0
    with pytest.raises(ValueError):
        source.restore_state(target)
    with pytest.raises(ValueError):
        source.build_from_state(target)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize(
    "settings",
    (
        {"learning_rate": True},
        {"learning_rate": 0},
        {"inference_steps": True},
        {"inference_steps": 0},
        {"inference_steps": 10001},
        {"inference_learning_rate": np.float64(0.1)},
        {"inference_policy": {}},
        {"inference_policy": CoupledInferencePolicy(min_steps=3)},
    ),
)
def test_should_validate_fixed_training_policy_before_use(settings):
    with pytest.raises(ValueError):
        learner(**settings)
