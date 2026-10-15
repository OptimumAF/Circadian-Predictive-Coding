"""Full raw-state continuation and fixed adapter settings, without topology transfer."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_scaling_learner as adapter
from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledEvaluation
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingStep,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def initial(family="pc"):
    policy = (
        None
        if family == "pc"
        else (
            CoupledCircadianConfig.matched_pc_control()
            if family == "neutral"
            else CoupledCircadianConfig()
        )
    )
    return initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, precisions=(0.7, 1.3), circadian=policy),
        41,
    )


def learner(state=None, **settings):
    options: dict[str, Any] = dict(
        learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
    )
    options.update(settings)
    return CoupledScalingLearner(initial() if state is None else state, **options)


def batch():
    return np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.2], [0.8]])


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
def test_should_restore_fork_and_continue_complete_owned_raw_state_exactly(family):
    state = initial(family)
    source = learner(state)
    x, y = batch()
    source.train_batch(x, y)
    saved = source.snapshot_state()
    before = asdict(saved)
    fork = source.fork()
    fork.train_batch(x, y)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
    restored = learner(state)
    restored.restore_state(saved)
    restored.train_batch(x, y)
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(fork.snapshot_state()))
    if family != "pc":
        assert restored.snapshot_state().wake_state is not None
        assert restored.snapshot_state().wake_state.wake_updates == 2
    effective = effective_coupled_parameters(saved)
    z = x
    for w, b in zip(effective.hidden_weights, effective.hidden_biases, strict=True):
        z = np.tanh(z @ w + b)
    expected = 1 / (1 + np.exp(-(z @ effective.output_weight + effective.output_bias)))
    np.testing.assert_allclose(source.predict(x), expected, rtol=1e-15, atol=1e-15)
    source.restore_state(state)
    saved.raw_parameters.hidden_weights[0][:] = 789
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(state))
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(fork.snapshot_state()))


def test_should_detach_constructor_state_snapshot_policy_and_fork_without_training():
    state = initial("active")
    source = learner(state)
    before = asdict(source.snapshot_state())
    fork = source.fork()
    state.raw_parameters.output_bias[:] = 99
    assert state.wake_state is not None and state.config.circadian is not None
    state.wake_state.chemical_state[:] = 99
    object.__setattr__(state.config.circadian, "min_plasticity", 1.0)
    snapshot = source.snapshot_state()
    snapshot.raw_parameters.output_weight[:] = 99
    assert snapshot.wake_state is not None
    snapshot.wake_state.plasticity_state[:] = 99
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
    np.testing.assert_equal(asdict(fork.snapshot_state()), before)
    altered = fork.snapshot_state()
    altered.raw_parameters.output_bias[:] = 0.4
    fork.restore_state(altered)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
    assert fork.snapshot_state().raw_parameters.output_bias[0, 0] == 0.4


def test_should_keep_rates_and_inference_policy_owned_across_fork_and_restore(monkeypatch):
    policy = CoupledInferencePolicy(gradient_tolerance=0.3)
    source = learner(inference_policy=policy)
    calls = []

    def fake_step(
        state,
        x,
        y,
        learning_rate,
        inference_steps,
        inference_learning_rate,
        *,
        inference_policy=None,
    ):
        calls.append(
            (learning_rate, inference_steps, inference_learning_rate, asdict(inference_policy))
        )
        candidate = copy_coupled_scaling_state(state)
        candidate.raw_parameters.output_bias[:] += 1
        zero = replace(
            candidate.raw_parameters,
            hidden_weights=tuple(np.zeros_like(w) for w in candidate.raw_parameters.hidden_weights),
            hidden_biases=tuple(np.zeros_like(b) for b in candidate.raw_parameters.hidden_biases),
            output_weight=np.zeros_like(candidate.raw_parameters.output_weight),
            output_bias=np.zeros_like(candidate.raw_parameters.output_bias),
        )
        evaluation = CoupledEvaluation(
            0.7, 0.7, (0.0, 0.0), (np.zeros((2, 3)), np.zeros((2, 3))), zero
        )
        return CoupledScalingStep(candidate, evaluation, (np.zeros((2, 3)), np.zeros((2, 3))), None)

    monkeypatch.setattr(adapter, "step_coupled_scaling", fake_step)
    object.__setattr__(policy, "gradient_tolerance", 99.0)
    fork = source.fork()
    fork.restore_state(source.snapshot_state())
    fork.train_batch(*batch())
    assert calls == [
        (0.03, 2, 0.1, dict(min_steps=1, gradient_tolerance=0.3, relative_energy_tolerance=None))
    ]
    np.testing.assert_array_equal(source.snapshot_state().raw_parameters.output_bias, 0)
    np.testing.assert_array_equal(fork.snapshot_state().raw_parameters.output_bias, 1)


@pytest.mark.parametrize(
    "changes",
    (
        {"scheme": "fan_in", "gamma0": 1.0},
        {"gamma0": 1.0},
        {"width": 4},
        {"input_dim": 3},
        {"hidden_layers": 1, "precisions": (0.7,)},
        {"precisions": (1.0, 1.0)},
        {"circadian": CoupledCircadianConfig()},
    ),
)
def test_should_refuse_different_config_without_publishing_any_state(changes):
    source = learner()
    before = asdict(source.snapshot_state())
    config = replace(initial().config, **changes)
    different = initialize_coupled_scaling(config, 41)
    with pytest.raises(ValueError):
        source.restore_state(different)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize("invalid", ("native", "effective", "dtype", "nonfinite", "config"))
def test_should_refuse_effective_substitution_and_mutated_snapshot_atomically(invalid):
    source = learner()
    before = asdict(source.snapshot_state())
    payload: Any = source.snapshot_state()
    if invalid == "native":
        payload = CoupledPredictiveCodingNetwork(2, (3, 3), 41).snapshot_state()
    elif invalid == "effective":
        payload = effective_coupled_parameters(payload)
    elif invalid == "dtype":
        object.__setattr__(
            payload.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
        )
    elif invalid == "nonfinite":
        payload.raw_parameters.output_bias[:] = np.nan
    else:
        object.__setattr__(payload.config, "width", True)
    with pytest.raises(ValueError):
        source.restore_state(payload)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize(
    "settings",
    (
        {"learning_rate": True},
        {"learning_rate": 0.0},
        {"inference_steps": True},
        {"inference_steps": 0},
        {"inference_steps": 10001},
        {"inference_learning_rate": np.float64(0.1)},
        {"inference_policy": {}},
        {"inference_policy": CoupledInferencePolicy(min_steps=3)},
    ),
)
def test_should_refuse_invalid_fixed_training_settings(settings):
    with pytest.raises(ValueError):
        learner(**settings)
