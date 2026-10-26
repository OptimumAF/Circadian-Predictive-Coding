"""Rooted structural BP data, exact native updates, and owned reconstruction.

Configuration histories describe reachable tensor layouts; they do not certify
which algorithm or operation produced compatible PC-only native data.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

import src.adapters.scaled_matched_backprop_learner as strict_module
import src.adapters.structural_scaled_matched_backprop_learner as structural_module
from src.adapters.scaled_matched_backprop_learner import ScaledMatchedBackpropLearner
from src.adapters.structural_scaled_matched_backprop_learner import (
    StructuralScaledMatchedBackpropLearner,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import predict_coupled
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.coupled_scaling_pruning import prune_coupled_scaling_width
from src.core.coupled_scaling_structure import (
    merge_coupled_scaling_width,
    split_coupled_scaling_width,
)
from src.core.coupled_structural_state import CoupledStructuralState, CoupledStructuralTransition
from src.core.learner_ports import TrainingDiagnostic
from src.core.scaled_matched_backprop import step_scaled_matched_backprop


RATE = 0.01


def raw(depth=1, **changes):
    arguments: dict[str, Any] = dict(
        scheme="mean_field",
        input_dim=2,
        width=2,
        hidden_layers=depth,
        gamma0=1.5,
        precisions=(1.0,) * depth,
    )
    arguments.update(changes)
    return initialize_coupled_scaling(CoupledScalingConfig(**arguments), seed=23)


def batch():
    return np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[0.2], [0.8]])


def same(actual, expected):
    np.testing.assert_equal(asdict(actual), asdict(expected))


@pytest.fixture(scope="module")
def graphs():
    # Two actual split kernels only. Refusal cases borrow detached value copies
    # of these bounded fixtures rather than repeating structural operations.
    result: dict[int, Any] = {}
    for depth in (1, 2):
        initial = raw(depth)
        split = split_coupled_scaling_width(initial)
        changed = CoupledStructuralState(
            1,
            initial.config,
            split.state,
            (CoupledStructuralTransition("split", split.lineage),),
        )
        result[depth] = (initial, changed, split.lineage)
    return result


def learner(graphs, depth=1):
    return StructuralScaledMatchedBackpropLearner(deepcopy(graphs[depth][0]), learning_rate=RATE)


def assert_native_update(port, source):
    x, y = batch()
    x_before, y_before = x.copy(), y.copy()
    expected = step_scaled_matched_backprop(source, x, y, RATE)
    diagnostic = port.train_batch(x, y)
    assert diagnostic == TrainingDiagnostic(
        expected.evaluation.loss_definition, expected.evaluation.loss
    )
    assert diagnostic.definition == "numpy_matched_binary_bce_preupdate_v1"
    same(port.snapshot_state().raw_state, expected.state)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)
    return expected.state


@pytest.mark.parametrize("depth", [1, 2])
def test_should_match_the_exact_unmodified_raw_bp_kernel_with_empty_structural_history(
    graphs, depth
):
    initial = deepcopy(graphs[depth][0])
    port = StructuralScaledMatchedBackpropLearner(initial, learning_rate=RATE)
    before = deepcopy(initial)
    expected = assert_native_update(port, initial)
    assert port.snapshot_state().transitions == ()
    assert port.snapshot_state().origin_config == initial.config
    x, _ = batch()
    np.testing.assert_array_equal(
        port.predict(x), predict_coupled(effective_coupled_parameters(expected), x)
    )
    same(initial, before)
    saved = port.snapshot_state()
    initial.raw_parameters.output_weight.fill(999)
    object.__setattr__(initial.config, "gamma0", 2.0)
    same(port.snapshot_state(), saved)


@pytest.mark.parametrize("depth", [1, 2])
def test_should_continue_after_real_split_compatible_merge_prune_and_earlier_rollback(
    graphs, depth
):
    initial, changed, lineage = deepcopy(graphs[depth])
    prototype = learner(graphs, depth)
    original = prototype.snapshot_state()
    port = prototype.build_from_state(changed)
    after_split = assert_native_update(port, changed.raw_state)
    merged = merge_coupled_scaling_width(after_split, lineage)
    history = (*changed.transitions, CoupledStructuralTransition("merge", lineage))
    merged_state = CoupledStructuralState(1, initial.config, merged.state, history)
    port.restore_state(merged_state)
    after_merge = assert_native_update(port, merged_state.raw_state)
    pruned = prune_coupled_scaling_width(after_merge, (1,) * depth)
    pruned_state = CoupledStructuralState(
        1,
        initial.config,
        pruned.state,
        (*history, CoupledStructuralTransition("prune", pruned.lineage)),
    )
    port.restore_state(pruned_state)
    assert_native_update(port, pruned_state.raw_state)
    assert [item.kind for item in port.snapshot_state().transitions] == ["split", "merge", "prune"]
    port.restore_state(original)
    assert port.snapshot_state().transitions == ()
    assert_native_update(port, initial)
    same(prototype.snapshot_state(), original)


@pytest.mark.parametrize("rate", [True, 0, -0.1, float("nan"), float("inf")])
def test_should_refuse_invalid_fixed_bp_learning_rate(graphs, rate):
    with pytest.raises(ValueError):
        StructuralScaledMatchedBackpropLearner(graphs[1][0], learning_rate=rate)


@pytest.mark.parametrize("initial", [None, {}])
def test_should_require_the_exact_initial_raw_state(initial):
    with pytest.raises(ValueError):
        StructuralScaledMatchedBackpropLearner(initial, learning_rate=RATE)


def test_should_keep_the_original_fixed_configuration_bp_port_strict(graphs):
    initial, changed, _ = deepcopy(graphs[1])
    fixed = ScaledMatchedBackpropLearner(initial, learning_rate=RATE)
    before = fixed.snapshot_state()
    with pytest.raises(ValueError):
        fixed.restore_state(changed.raw_state)
    same(fixed.snapshot_state(), before)
    structural = learner(graphs).build_from_state(changed)
    same(structural.snapshot_state(), changed)


def test_should_own_every_root_history_and_array_across_build_fork_restore_and_continuation(graphs):
    source = learner(graphs)
    changed = deepcopy(graphs[1][1])
    built = source.build_from_state(changed)
    before = built.snapshot_state()
    fork = built.fork()
    restored = source.build_from_state(before)
    x, y = batch()
    assert built.train_batch(x, y) == fork.train_batch(x, y) == restored.train_batch(x, y)
    continued = built.snapshot_state()
    same(fork.snapshot_state(), continued)
    same(restored.snapshot_state(), continued)
    restored.restore_state(before)
    restored.train_batch(x, y)
    same(restored.snapshot_state(), continued)
    changed.raw_state.raw_parameters.output_weight.fill(999)
    before.raw_state.raw_parameters.hidden_weights[0].fill(999)
    object.__setattr__(before.origin_config, "gamma0", 3.0)
    object.__setattr__(before.transitions[0].lineage.source_config, "width", 1)
    same(built.snapshot_state(), continued)
    same(fork.snapshot_state(), continued)


def test_should_preserve_the_fixed_owner_rate_during_build_fork_and_restore(graphs, monkeypatch):
    source = learner(graphs)
    changed = deepcopy(graphs[1][1])
    expected_rate = source._learning_rate
    seen: list[float] = []
    original = strict_module.step_scaled_matched_backprop

    def observe(state, features, targets, rate):
        seen.append(rate)
        return original(state, features, targets, rate)

    monkeypatch.setattr(strict_module, "step_scaled_matched_backprop", observe)
    fork = source.build_from_state(changed).fork()
    fork.restore_state(changed)
    fork.train_batch(*batch())
    assert seen == [expected_rate] == [RATE]


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 1},
        {"gamma0": 2.0},
        {"precisions": (0.7,)},
        {"hidden_layers": 2, "precisions": (1.0, 1.0)},
        {"scheme": "fan_in", "gamma0": 1.0},
        {"circadian": CoupledCircadianConfig()},
    ],
)
def test_should_refuse_a_different_complete_root_even_when_its_empty_chain_is_valid(
    graphs, changes
):
    source = learner(graphs)
    before = source.snapshot_state()
    other = raw(**changes)
    foreign = CoupledStructuralState(1, other.config, other)
    for operation in (source.restore_state, source.build_from_state):
        with pytest.raises(ValueError):
            operation(foreign)
    same(source.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault",
    ["unrooted", "changed_origin", "tag", "format", "dtype", "nan", "raw", "list", "lineage"],
)
def test_should_refuse_malformed_or_unrooted_states_without_publication(graphs, fault):
    source = learner(graphs)
    before = source.snapshot_state()
    changed = deepcopy(graphs[1][1])
    if fault == "unrooted":
        object.__setattr__(changed, "transitions", ())
    elif fault == "changed_origin":
        object.__setattr__(changed, "origin_config", changed.raw_state.config)
    elif fault == "tag":
        object.__setattr__(changed.transitions[0], "kind", "merge")
    elif fault == "format":
        object.__setattr__(changed, "format_version", True)
    elif fault == "dtype":
        object.__setattr__(
            changed.raw_state.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
        )
    elif fault == "nan":
        changed.raw_state.raw_parameters.output_weight.fill(np.nan)
    elif fault == "raw":
        object.__setattr__(changed, "raw_state", {})
    elif fault == "list":
        object.__setattr__(changed, "transitions", list(changed.transitions))
    else:
        object.__setattr__(changed.transitions[0], "lineage", {})
    payload = source._payload
    for operation in (source.restore_state, source.build_from_state):
        with pytest.raises(ValueError):
            operation(changed)
    assert source._payload is payload
    same(source.snapshot_state(), before)


@pytest.mark.parametrize("where", ["constructor", "restore"])
def test_should_refuse_cpc_configuration_and_wake_arrays(graphs, where):
    cpc = raw(circadian=CoupledCircadianConfig())
    if where == "constructor":
        with pytest.raises(ValueError):
            StructuralScaledMatchedBackpropLearner(cpc, learning_rate=RATE)
    else:
        source = learner(graphs)
        before = source.snapshot_state()
        cpc_state = CoupledStructuralState(1, cpc.config, cpc)
        with pytest.raises(ValueError):
            source.restore_state(cpc_state)
        same(source.snapshot_state(), before)


def test_should_not_publish_origin_or_history_when_private_reconstruction_raises(
    graphs, monkeypatch
):
    source = learner(graphs)
    before, payload = source.snapshot_state(), source._payload
    changed = deepcopy(graphs[1][1])
    observed: list[CoupledStructuralState] = []

    def fail(*args, **kwargs):
        observed.append(source.snapshot_state())
        raise RuntimeError("private BP construction failed")

    monkeypatch.setattr(structural_module.ScaledMatchedBackpropLearner, "__init__", fail)
    for operation in (source.restore_state, source.build_from_state):
        with pytest.raises(RuntimeError, match="private BP construction failed"):
            operation(changed)
    assert len(observed) == 2 and source._payload is payload
    for state in observed:
        same(state, before)
    same(source.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["before_kernel", "after_kernel", "nonfinite_input"])
def test_should_preserve_complete_live_state_on_native_update_failure(graphs, monkeypatch, fault):
    source = learner(graphs).build_from_state(deepcopy(graphs[1][1]))
    before = source.snapshot_state()
    x, y = batch()
    if fault == "nonfinite_input":
        x[0, 0] = np.inf
    else:
        original = strict_module.step_scaled_matched_backprop

        def fail(*args, **kwargs):
            if fault == "after_kernel":
                original(*args, **kwargs)
            raise RuntimeError("injected native kernel failure")

        monkeypatch.setattr(strict_module, "step_scaled_matched_backprop", fail)
    with pytest.raises((RuntimeError, ValueError, FloatingPointError)):
        source.train_batch(x, y)
    same(source.snapshot_state(), before)


def test_should_accept_compatible_pc_only_tensor_data_without_claiming_algorithm_provenance(graphs):
    source = learner(graphs)
    compatible = source.snapshot_state()
    compatible.raw_state.raw_parameters.output_bias[:] = 0.2
    source.restore_state(compatible)
    same(source.snapshot_state(), compatible)
    assert not hasattr(compatible, "algorithm") and not hasattr(compatible.raw_state, "algorithm")
