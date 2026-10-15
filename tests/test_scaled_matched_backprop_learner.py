"""Owned raw BP ports and delayed actor/shadow continuation at fixed rates."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.adapters.scaled_matched_backprop_learner as adapter
from src.adapters.scaled_matched_backprop_learner import ScaledMatchedBackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.matched_backprop import MatchedBackpropEvaluation, MatchedBackpropNetwork
from src.core.scaled_matched_backprop import ScaledMatchedBackpropStep


def initial(scheme="mean_field"):
    return initialize_coupled_scaling(
        CoupledScalingConfig(
            scheme, 2, 3, 2, 2.0 if scheme == "mean_field" else 1.0, precisions=(0.7, 1.3)
        ),
        41,
    )


def learner(state=None, rate=0.03):
    return ScaledMatchedBackpropLearner(initial() if state is None else state, learning_rate=rate)


def batch():
    return np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.2], [0.8]])


@pytest.mark.parametrize("scheme", ("legacy_gaussian", "fan_in", "mean_field"))
def test_should_continue_full_raw_state_exactly_after_fork_and_restore(scheme):
    state = initial(scheme)
    source = learner(state)
    x, y = batch()
    diagnostic = source.train_batch(x, y)
    assert diagnostic.definition == "numpy_matched_binary_bce_preupdate_v1"
    assert np.isfinite(diagnostic.value)
    saved = source.snapshot_state()
    before = asdict(saved)
    fork = source.fork()
    fork.train_batch(x, y)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
    restored = learner(state)
    restored.restore_state(saved)
    restored.train_batch(x, y)
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(fork.snapshot_state()))
    native = MatchedBackpropNetwork(effective_coupled_parameters(saved))
    np.testing.assert_array_equal(source.predict(x), native.predict_proba(x))
    assert source.snapshot_state().config.precisions == (0.7, 1.3)
    source.restore_state(state)
    saved.raw_parameters.output_bias[:] = 99
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(state))


def test_should_detach_constructor_snapshot_and_fork_without_training():
    state = initial()
    source = learner(state)
    before = asdict(source.snapshot_state())
    fork = source.fork()
    state.raw_parameters.hidden_weights[0][:] = 99
    object.__setattr__(state.config, "gamma0", 3.0)
    snapshot = source.snapshot_state()
    snapshot.raw_parameters.output_weight[:] = 99
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
    np.testing.assert_equal(asdict(fork.snapshot_state()), before)
    altered = fork.snapshot_state()
    altered.raw_parameters.output_bias[:] = 0.4
    fork.restore_state(altered)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


def test_should_keep_fixed_learning_rate_across_fork_and_raw_restore(monkeypatch):
    source = learner(rate=0.07)
    calls = []

    def fake_step(state, x, y, learning_rate):
        calls.append(learning_rate)
        candidate = copy_coupled_scaling_state(state)
        candidate.raw_parameters.output_bias[:] += 1
        return ScaledMatchedBackpropStep(
            candidate, MatchedBackpropEvaluation(0.7, candidate.raw_parameters)
        )

    monkeypatch.setattr(adapter, "step_scaled_matched_backprop", fake_step)
    fork = source.fork()
    fork.restore_state(source.snapshot_state())
    fork.train_batch(*batch())
    assert calls == [0.07]
    np.testing.assert_array_equal(source.snapshot_state().raw_parameters.output_bias, 0)
    np.testing.assert_array_equal(fork.snapshot_state().raw_parameters.output_bias, 1)


def test_should_apply_delayed_native_bp_wake_only_to_candidate_and_keep_actor_stable():
    source = learner()
    before = asdict(source.snapshot_state())
    clock = LogicalClock()
    resource = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=resource,
        max_experiences=2,
    )
    actor_before = asdict(runtime.actor.snapshot_state())
    x, y = batch()
    prediction = runtime.actor.predict(x).prediction
    for index in range(2):
        runtime.record_experience(
            Experience(
                f"row-{index}", "episode", 0, "a0", x, "train", ExperiencePermissions(training=True)
            )
        )
        runtime.record_label(
            LabelArrival(f"label-{index}", f"row-{index}", "episode", index + 1, "a0", y)
        )
    assert runtime.train_ready() == ()
    clock.advance_to(1)
    first = runtime.train_ready(max_updates=1)
    assert (
        len(first) == 1
        and first[0].diagnostic.definition == "numpy_matched_binary_bce_preupdate_v1"
    )
    clock.advance_to(2)
    second = runtime.train_ready(max_updates=1)
    assert len(second) == 1
    candidate = runtime.candidate_snapshot()
    assert candidate.wake_updates == resource.updates_completed == 2
    assert not np.array_equal(
        candidate.state.raw_parameters.output_bias,
        source.snapshot_state().raw_parameters.output_bias,
    )
    assert runtime.train_ready() == ()
    np.testing.assert_equal(asdict(runtime.actor.snapshot_state()), actor_before)
    np.testing.assert_array_equal(runtime.actor.predict(x).prediction, prediction)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize(
    "changes",
    (
        {"scheme": "fan_in", "gamma0": 1.0},
        {"gamma0": 1.0},
        {"width": 4},
        {"input_dim": 3},
        {"hidden_layers": 1, "precisions": (0.7,)},
        {"precisions": (1.0, 1.0)},
    ),
)
def test_should_pin_full_config_on_restore_without_publication(changes):
    source = learner()
    before = asdict(source.snapshot_state())
    other = initialize_coupled_scaling(replace(initial().config, **changes), 41)
    with pytest.raises(ValueError):
        source.restore_state(other)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize("invalid", ("native", "effective", "dtype", "nan", "config", "wake"))
def test_should_refuse_substituted_or_mutated_snapshot_atomically(invalid):
    source = learner()
    before = asdict(source.snapshot_state())
    payload: Any = source.snapshot_state()
    if invalid == "native":
        payload = MatchedBackpropNetwork(effective_coupled_parameters(payload)).snapshot_state()
    elif invalid == "effective":
        payload = effective_coupled_parameters(payload)
    elif invalid == "dtype":
        object.__setattr__(
            payload.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
        )
    elif invalid == "nan":
        payload.raw_parameters.output_bias[:] = np.nan
    elif invalid == "config":
        object.__setattr__(payload.config, "width", True)
    else:
        object.__setattr__(payload, "wake_state", {})
    with pytest.raises(ValueError):
        source.restore_state(payload)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize(
    "policy", (CoupledCircadianConfig(), CoupledCircadianConfig.matched_pc_control())
)
def test_should_refuse_cpc_constructor_and_restore_even_for_neutral_gates(policy):
    state = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, precisions=(0.7, 1.3), circadian=policy),
        41,
    )
    with pytest.raises(ValueError):
        learner(state)
    source = learner()
    before = asdict(source.snapshot_state())
    with pytest.raises(ValueError):
        source.restore_state(state)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)


@pytest.mark.parametrize("rate", (True, 0, -1, np.float64(0.03), float("inf")))
def test_should_refuse_invalid_fixed_learning_rate(rate):
    with pytest.raises(ValueError):
        learner(rate=rate)


def test_should_keep_adapter_state_after_failed_actual_training_attempt():
    source = learner()
    before = asdict(source.snapshot_state())
    x, y = batch()
    y[0, 0] = np.inf
    with pytest.raises(ValueError):
        source.train_batch(x, y)
    np.testing.assert_equal(asdict(source.snapshot_state()), before)
