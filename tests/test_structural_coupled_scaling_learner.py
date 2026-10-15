"""Root-pinned mixed topology ports and atomic private reconstruction controls."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_scaling_learner as strict_adapter
from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
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
from src.core.coupled_scaling_structure import split_coupled_scaling_width
from src.core.coupled_scaling_topology import CoupledScalingTopologyState
from src.core.coupled_structural_state import CoupledStructuralState, CoupledStructuralTransition


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
    values: dict[str, Any] = dict(
        scheme="mean_field",
        input_dim=2,
        width=3,
        hidden_layers=2,
        gamma0=2.0,
        precisions=(0.7, 1.3),
        circadian=policy,
    )
    values.update(changes)
    return initialize_coupled_scaling(CoupledScalingConfig(**values), 41)


def learner(initial=None, **changes):
    settings: dict[str, Any] = dict(
        learning_rate=0.03, inference_steps=2, inference_learning_rate=0.05
    )
    settings.update(changes)
    return StructuralCoupledScalingLearner(raw() if initial is None else initial, **settings)


def target(source):
    split = split_coupled_scaling_width(source)
    return CoupledStructuralState(
        1, source.config, split.state, (CoupledStructuralTransition("split", split.lineage),)
    )


def batch():
    return np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
def test_should_build_full_split_chain_then_match_raw_reference_and_exact_owned_fork_continuation(
    family,
):
    initial = raw(family)
    source = learner(initial)
    before = source.snapshot_state()
    changed = target(initial)
    built = source.build_from_state(changed)
    x, y = batch()
    diagnostic = built.train_batch(x, y)
    reference = step_coupled_scaling(changed.raw_state, x, y, 0.03, 2, 0.05)
    assert diagnostic.value == reference.evaluation.energy
    np.testing.assert_equal(asdict(built.snapshot_state().raw_state), asdict(reference.state))
    saved = built.snapshot_state()
    fork = built.fork()
    fork.train_batch(x, y)
    restored = source.build_from_state(saved)
    restored.train_batch(x, y)
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(fork.snapshot_state()))
    np.testing.assert_equal(asdict(built.snapshot_state()), asdict(saved))
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(before))
    source.restore_state(saved)
    assert len(source.snapshot_state().transitions) == 1
    source.restore_state(before)
    assert source.snapshot_state().transitions == ()
    changed.raw_state.raw_parameters.output_bias[:] = 99
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(before))


def test_should_copy_origin_policy_and_all_arrays_across_constructor_snapshot_and_fork():
    initial = raw("active")
    source = learner(initial)
    before = asdict(source.snapshot_state())
    initial.raw_parameters.output_bias[:] = 99
    assert initial.config.circadian is not None
    object.__setattr__(initial.config.circadian, "min_plasticity", 1.0)
    fork = source.fork()
    snapshot = fork.snapshot_state()
    snapshot.raw_state.raw_parameters.output_weight[:] = 99
    object.__setattr__(snapshot.origin_config, "gamma0", 3.0)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
    np.testing.assert_equal(asdict(fork.snapshot_state()), before)


def test_should_pin_fixed_rates_and_owned_inference_policy_through_build_fork_restore(monkeypatch):
    policy = CoupledInferencePolicy(gradient_tolerance=0.3)
    source = learner(inference_policy=policy)
    changed = target(raw())
    built = source.build_from_state(changed)
    calls = []

    def fake_step(state, x, y, rate, steps, inference_rate, *, inference_policy=None):
        calls.append((rate, steps, inference_rate, deepcopy(inference_policy)))
        candidate = copy_coupled_scaling_state(state)
        candidate.raw_parameters.output_bias[:] += 1
        p = candidate.raw_parameters
        gradients = replace(
            p,
            hidden_weights=tuple(np.zeros_like(a) for a in p.hidden_weights),
            hidden_biases=tuple(np.zeros_like(a) for a in p.hidden_biases),
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
    fork.restore_state(changed)
    fork.train_batch(*batch())
    assert calls == [(0.03, 2, 0.05, CoupledInferencePolicy(gradient_tolerance=0.3))]
    assert fork.snapshot_state().transitions == changed.transitions
    np.testing.assert_array_equal(built.snapshot_state().raw_state.raw_parameters.output_bias, 0)


@pytest.mark.parametrize(
    "changes",
    (
        {"width": 4},
        {"gamma0": 3.0},
        {"input_dim": 3},
        {"hidden_layers": 1, "precisions": (0.7,)},
        {"precisions": (1.0, 1.0)},
        {"scheme": "fan_in", "gamma0": 1.0},
        {"circadian": CoupledCircadianConfig()},
    ),
)
def test_should_refuse_unrelated_complete_origin_even_with_valid_empty_reachability(changes):
    source = learner()
    before = asdict(source.snapshot_state())
    foreign_raw = raw(**changes)
    foreign = CoupledStructuralState(1, foreign_raw.config, foreign_raw)
    with pytest.raises(ValueError):
        source.restore_state(foreign)
    with pytest.raises(ValueError):
        source.build_from_state(foreign)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


def test_should_distinguish_adjacent_large_native_integer_origin_gamma_values():
    source = learner(raw(gamma0=2**53))
    before = asdict(source.snapshot_state())
    other = raw(gamma0=2**53 + 1)
    assert float(other.config.gamma0) == float(2**53)
    with pytest.raises(ValueError):
        source.restore_state(CoupledStructuralState(1, other.config, other))
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize(
    "kind", ("no_chain", "changed_origin", "invalid_tag", "dtype", "nan", "wake")
)
def test_should_refuse_precision_or_geometry_change_without_its_full_rooted_chain(kind):
    source = learner(raw("active"))
    before = asdict(source.snapshot_state())
    changed = target(raw("active"))
    if kind == "no_chain":
        object.__setattr__(changed, "transitions", ())
    elif kind == "changed_origin":
        object.__setattr__(changed, "origin_config", changed.raw_state.config)
    elif kind == "invalid_tag":
        object.__setattr__(changed.transitions[0], "kind", "merge")
    elif kind == "dtype":
        object.__setattr__(
            changed.raw_state.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
        )
    elif kind == "nan":
        changed.raw_state.raw_parameters.output_bias[:] = np.nan
    else:
        assert changed.raw_state.wake_state is not None
        changed.raw_state.wake_state.plasticity_state[:] = 0
    with pytest.raises(ValueError):
        source.restore_state(changed)
    with pytest.raises(ValueError):
        source.build_from_state(changed)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


def test_should_not_publish_any_origin_or_history_during_late_private_learner_construction_fault(
    monkeypatch,
):
    source = learner()
    before = asdict(source.snapshot_state())
    changed = target(raw())
    seen = []

    def broken(*args, **kwargs):
        seen.append(asdict(source.snapshot_state()))
        raise RuntimeError("private build fault")

    monkeypatch.setattr(CoupledScalingLearner, "__init__", broken)
    with pytest.raises(RuntimeError, match="private build fault"):
        source.restore_state(changed)
    with pytest.raises(RuntimeError, match="private build fault"):
        source.build_from_state(changed)
    assert len(seen) == 2
    for snapshot in seen:
        np.testing.assert_equal(snapshot, before)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


def test_should_keep_existing_raw_and_prune_topology_precision_bindings_strict():
    initial = raw()
    strict = CoupledScalingLearner(
        initial, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.05
    )
    topology = TopologyCoupledScalingLearner(
        initial, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.05
    )
    a, b = asdict(strict.snapshot_state()), asdict(topology.snapshot_state())
    changed = target(initial)
    with pytest.raises(ValueError):
        strict.restore_state(changed.raw_state)
    with pytest.raises(ValueError):
        topology.restore_state(CoupledScalingTopologyState(1, changed.raw_state))
    np.testing.assert_equal(asdict(strict.snapshot_state()), a)
    np.testing.assert_equal(asdict(topology.snapshot_state()), b)
