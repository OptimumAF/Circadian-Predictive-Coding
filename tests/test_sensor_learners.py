"""Sensor composition contracts; native SGD is forbidden throughout this module."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.adapters.sensor_learners import METHODS, build_sensor_learner, validate_sensor_method
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import COUPLED_ENERGY_ID, CoupledParameters, predict_coupled
from src.core.coupled_learner_state import CoupledUpdatePolicy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledTrainResult,
)
from src.core.matched_backprop import (
    MATCHED_BACKPROP_LOSS_ID,
    MatchedBackpropNetwork,
    MatchedBackpropEvaluation,
)


@pytest.fixture(autouse=True)
def forbid_native_updates(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native SGD is outside this engineering test allowance")

    for kind in (
        CoupledPredictiveCodingNetwork,
        CoupledCircadianPredictiveCodingNetwork,
        MatchedBackpropNetwork,
    ):
        monkeypatch.setattr(kind, "train_epoch", forbidden)


def _initial() -> CoupledParameters:
    return CoupledPredictiveCodingNetwork(16, (8,), seed=17).snapshot_parameters()


def _arrays(parameters: CoupledParameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def _same_parameters(left: CoupledParameters, right: CoupledParameters) -> None:
    for left_array, right_array in zip(_arrays(left), _arrays(right), strict=True):
        np.testing.assert_array_equal(left_array, right_array)
        assert not np.shares_memory(left_array, right_array)


class StringSubclass(str):
    pass


@pytest.mark.parametrize("method", [None, True, 0, [], "BP", "active_cpc", StringSubclass("pc")])
def test_should_reject_nonexact_or_unknown_method_before_reading_parameters(method):
    with pytest.raises(ValueError, match="method must be an exact string"):
        build_sensor_learner(method, cast(Any, None))
    with pytest.raises(ValueError, match="method must be an exact string"):
        validate_sensor_method(method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("geometry", [(2, (8,)), (16, (7,)), (16, (8, 8))])
def test_should_reject_parameters_outside_fixed_sensor_geometry(method, geometry):
    input_dim, hidden_dims = geometry
    wrong = CoupledPredictiveCodingNetwork(input_dim, hidden_dims, seed=17).snapshot_parameters()
    with pytest.raises(ValueError):
        build_sensor_learner(method, wrong)


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_common_initial_prediction_and_own_all_parameters(method):
    initial = _initial()
    learner = build_sensor_learner(method, initial)
    snapshot = learner.snapshot_state()
    _same_parameters(initial, snapshot.parameters)
    features = np.arange(64, dtype=np.float64).reshape(4, 16) / 64
    expected = predict_coupled(initial, features)
    np.testing.assert_array_equal(learner.predict(features), expected)
    for array in _arrays(initial):
        array.fill(77.0)
    for array in _arrays(snapshot.parameters):
        array.fill(-77.0)
    np.testing.assert_array_equal(learner.predict(features), expected)


@pytest.mark.parametrize("method", METHODS)
def test_should_fork_and_restore_owned_parameters_without_changing_the_source(method):
    learner = build_sensor_learner(method, _initial())
    original = learner.snapshot_state()
    changed = learner.snapshot_state()
    changed.parameters.output_bias.fill(0.75)
    learner.restore_state(changed)
    fork = learner.fork()
    _same_parameters(fork.snapshot_state().parameters, changed.parameters)
    changed.parameters.output_bias.fill(99.0)
    learner.restore_state(original)
    np.testing.assert_array_equal(learner.snapshot_state().parameters.output_bias, [[0.0]])
    np.testing.assert_array_equal(fork.snapshot_state().parameters.output_bias, [[0.75]])


@pytest.mark.parametrize("method", METHODS)
def test_should_dispatch_fixed_sensor_policy_without_native_training(method, monkeypatch):
    learner = build_sensor_learner(method, _initial())
    calls = []

    def fake_native(model, features, targets, *args, **kwargs):
        calls.append((type(model), features, targets, args, kwargs))
        if method == "bp":
            return MatchedBackpropEvaluation(0.25, _initial())
        return CoupledTrainResult(0.25)

    kind = (
        MatchedBackpropNetwork
        if method == "bp"
        else CoupledPredictiveCodingNetwork
        if method == "pc"
        else CoupledCircadianPredictiveCodingNetwork
    )
    monkeypatch.setattr(kind, "train_epoch", fake_native)
    features = np.zeros((4, 16), dtype=np.float64)
    targets = np.zeros((4, 1), dtype=np.float64)
    diagnostic = learner.train_batch(features, targets)
    fork_diagnostic = learner.fork().train_batch(features, targets)
    expected_args = (0.1,) if method == "bp" else (0.1, 4, 0.2)
    assert len(calls) == 2
    for actual_kind, actual_features, actual_targets, args, kwargs in calls:
        assert actual_kind is kind
        assert actual_features is features and actual_targets is targets
        assert args == expected_args and kwargs == {}
    assert diagnostic == fork_diagnostic
    assert diagnostic.value == 0.25
    assert diagnostic.definition == (
        MATCHED_BACKPROP_LOSS_ID if method == "bp" else COUPLED_ENERGY_ID
    )
    if method == "bp":
        assert type(learner) is MatchedBackpropLearner
    else:
        assert isinstance(learner, CoupledLearner)
        assert learner.checkpoint_policy() == CoupledUpdatePolicy(0.1, 4, 0.2, None)
        assert learner.snapshot_state().precisions == (1.0,)


@pytest.mark.parametrize("method", ["neutral_cpc", "gated_cpc"])
def test_should_preserve_full_chemistry_and_configuration_on_restore_and_fork(method):
    learner = build_sensor_learner(method, _initial())
    original = learner.snapshot_state()
    assert isinstance(original, CoupledSnapshot)
    wake = original.wake_state
    assert wake is not None
    config = (
        CoupledCircadianConfig.matched_pc_control()
        if method == "neutral_cpc"
        else CoupledCircadianConfig()
    )
    assert (
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
    ) == (
        config.chemical_decay,
        config.chemical_buildup_rate,
        config.plasticity_sensitivity,
        config.min_plasticity,
    )
    chemicals = np.linspace(0.1, 0.8, 8, dtype=np.float64)
    gates = np.clip(np.exp(-config.plasticity_sensitivity * chemicals), config.min_plasticity, 1.0)
    restored = replace(
        original,
        wake_state=replace(wake, chemical_state=chemicals, plasticity_state=gates, wake_updates=7),
    )
    learner.restore_state(restored)
    fork = learner.fork()
    for owner in (learner, fork):
        state = owner.snapshot_state()
        assert state.wake_state is not None
        assert state.wake_state.wake_updates == 7
        np.testing.assert_array_equal(state.wake_state.chemical_state, chemicals)
        np.testing.assert_array_equal(state.wake_state.plasticity_state, gates)
        _same_parameters(state.parameters, restored.parameters)
    chemicals.fill(99.0)
    gates.fill(99.0)
    learner.restore_state(original)
    fork_state = fork.snapshot_state()
    assert fork_state.wake_state is not None
    assert fork_state.wake_state.wake_updates == 7
    assert np.all(fork_state.wake_state.chemical_state < 1.0)
    assert np.all(fork_state.wake_state.plasticity_state <= 1.0)
    assert learner.snapshot_state().wake_state.wake_updates == 0
