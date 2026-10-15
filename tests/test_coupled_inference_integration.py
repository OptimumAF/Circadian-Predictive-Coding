"""Bounded native updates, adapter continuation and fixed-path compatibility.

Full file: 28 train calls and 28 inference calls (including nested calls).
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters, evaluate_coupled, feedforward_latents
from src.core.coupled_inference import CoupledInferencePolicy, infer_coupled
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def batch():
    return np.array([[0.4, -0.2], [-0.3, 0.1], [0.2, 0.6]]), np.array([[1.0], [0.0], [1.0]])


def model(kind, widths=(3, 2)):
    if kind == "pc":
        return CoupledPredictiveCodingNetwork(2, widths, seed=23)
    config = (
        CoupledCircadianConfig.matched_pc_control()
        if kind == "neutral"
        else CoupledCircadianConfig()
    )
    return CoupledCircadianPredictiveCodingNetwork(2, widths, seed=23, config=config)


def arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def assert_parameters(left, right):
    for a, b in zip(arrays(left), arrays(right)):
        np.testing.assert_array_equal(a, b)


def assert_snapshot(left, right):
    assert left.format_version == right.format_version == 1
    assert left.precisions == right.precisions
    assert_parameters(left.parameters, right.parameters)
    if left.wake_state is None:
        assert right.wake_state is None
    else:
        assert right.wake_state is not None
        assert left.wake_state.wake_updates == right.wake_state.wake_updates
        np.testing.assert_array_equal(
            left.wake_state.chemical_state, right.wake_state.chemical_state
        )
        np.testing.assert_array_equal(
            left.wake_state.plasticity_state, right.wake_state.plasticity_state
        )


@pytest.mark.parametrize("widths", [(2,), (2, 3), (2, 3, 2, 1)])
def test_should_match_fixed_steps_and_neutral_parameters_at_guarded_cap(widths):
    fixed, pc, neutral = model("pc", widths), model("pc", widths), model("neutral", widths)
    x, y = batch()
    policy = CoupledInferencePolicy(gradient_tolerance=None)
    baseline = fixed.train_epoch(x, y, 0.01, 3, 0.03)
    ordinary = pc.train_epoch(x, y, 0.01, 3, 0.03, inference_policy=policy)
    circadian = neutral.train_epoch(x, y, 0.01, 3, 0.03, inference_policy=policy)
    assert baseline.inference is None
    assert ordinary.inference == circadian.inference
    assert ordinary.inference.reason == "max_steps" and ordinary.inference.accepted_steps == 3
    assert baseline.energy == ordinary.energy == circadian.energy
    assert_parameters(fixed.snapshot_parameters(), pc.snapshot_parameters())
    assert_parameters(neutral.snapshot_parameters(), pc.snapshot_parameters())
    assert neutral.wake_updates == 1 and np.any(neutral.get_chemical_state() > 0)
    np.testing.assert_array_equal(neutral.get_plasticity_state(), np.ones(widths[-1]))


@pytest.mark.parametrize("kind", ["pc", "neutral", "gated"])
def test_should_commit_gradients_and_chemistry_from_the_actual_early_stop(kind):
    learner = model(kind)
    x, y = batch()
    initial = learner.snapshot_parameters()
    policy = CoupledInferencePolicy(min_steps=2, gradient_tolerance=1.0)
    inferred = infer_coupled(
        initial, x, y, learner.precisions, max_steps=6, learning_rate=0.1, policy=policy
    )
    result = learner.train_epoch(x, y, 0.01, 6, 0.1, inference_policy=policy)
    assert result.inference == inferred.diagnostics
    assert result.inference.reason == "gradient_tolerance" and result.inference.accepted_steps == 2
    gradients = inferred.evaluation.parameter_gradients
    if kind != "pc":
        chemicals = learner.config.chemical_buildup_rate * np.mean(
            np.abs(inferred.latents[-1]), axis=0
        )
        gates = np.clip(
            np.exp(-learner.config.plasticity_sensitivity * chemicals),
            learner.config.min_plasticity,
            1.0,
        )
        np.testing.assert_array_equal(learner.get_chemical_state(), chemicals)
        np.testing.assert_array_equal(learner.get_plasticity_state(), gates)
        gradients = CoupledParameters(
            (*gradients.hidden_weights[:-1], gradients.hidden_weights[-1] * gates),
            (*gradients.hidden_biases[:-1], gradients.hidden_biases[-1] * gates),
            gradients.output_weight * gates[:, None],
            gradients.output_bias,
        )
    expected = CoupledParameters(
        tuple(w - 0.01 * g for w, g in zip(initial.hidden_weights, gradients.hidden_weights)),
        tuple(b - 0.01 * g for b, g in zip(initial.hidden_biases, gradients.hidden_biases)),
        initial.output_weight - 0.01 * gradients.output_weight,
        initial.output_bias - 0.01 * gradients.output_bias,
    )
    assert_parameters(learner.snapshot_parameters(), expected)
    assert result.energy == inferred.evaluation.energy


@pytest.mark.parametrize("kind", ["pc", "neutral"])
def test_should_use_feedforward_fallback_when_the_first_proposal_increases_energy(kind):
    learner = model(kind)
    x, y = batch()
    initial = learner.snapshot_parameters()
    latents = feedforward_latents(initial, x)
    initial_evaluation = evaluate_coupled(initial, x, y, latents, learner.precisions)
    result = learner.train_epoch(
        x,
        y,
        0.01,
        4,
        1e6,
        inference_policy=CoupledInferencePolicy(min_steps=2, gradient_tolerance=None),
    )
    assert result.inference.reason == "energy_increase" and result.inference.used_fallback
    assert result.inference.accepted_steps == 0 and result.inference.attempted_steps == 1
    assert result.inference.energy_evaluations == 2
    assert result.energy == initial_evaluation.energy
    gradients = initial_evaluation.parameter_gradients
    expected = CoupledParameters(
        tuple(w - 0.01 * g for w, g in zip(initial.hidden_weights, gradients.hidden_weights)),
        tuple(b - 0.01 * g for b, g in zip(initial.hidden_biases, gradients.hidden_biases)),
        initial.output_weight - 0.01 * gradients.output_weight,
        initial.output_bias - 0.01 * gradients.output_bias,
    )
    assert_parameters(learner.snapshot_parameters(), expected)


@pytest.mark.parametrize("kind", ["pc", "gated"])
def test_should_publish_no_parameters_or_chemistry_if_candidate_preparation_fails(
    monkeypatch, kind
):
    learner = model(kind)
    before = learner.snapshot_state()

    def fail(*args):
        raise FloatingPointError("injected candidate failure")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "_prepare_parameters", fail)
    x, y = batch()
    with pytest.raises(FloatingPointError, match="candidate failure"):
        learner.train_epoch(
            x, y, 0.01, 4, 0.1, inference_policy=CoupledInferencePolicy(gradient_tolerance=1.0)
        )
    assert_snapshot(learner.snapshot_state(), before)


@pytest.mark.parametrize("kind", ["pc", "neutral"])
def test_should_copy_adapter_policy_and_continue_it_across_fork_and_restore(kind):
    source, direct = model(kind), model(kind)
    policy = CoupledInferencePolicy(min_steps=2, gradient_tolerance=1.0)
    learner = CoupledLearner(
        source,
        learning_rate=0.01,
        inference_steps=4,
        inference_learning_rate=0.1,
        inference_policy=policy,
    )
    object.__setattr__(policy, "min_steps", 1000)
    owned = CoupledInferencePolicy(min_steps=2, gradient_tolerance=1.0)
    x, y = batch()
    diagnostic = learner.train_batch(x, y)
    reference = direct.train_epoch(x, y, 0.01, 4, 0.1, inference_policy=owned)
    assert diagnostic.value == reference.energy
    assert_snapshot(learner.snapshot_state(), direct.snapshot_state())
    saved = learner.snapshot_state()
    fork = learner.fork()
    learner.train_batch(x[:2], y[:2])
    fork.train_batch(x[:2], y[:2])
    direct.restore_state(saved)
    direct.train_epoch(x[:2], y[:2], 0.01, 4, 0.1, inference_policy=owned)
    assert_snapshot(learner.snapshot_state(), fork.snapshot_state())
    assert_snapshot(learner.snapshot_state(), direct.snapshot_state())
    assert_snapshot(source.snapshot_state(), model(kind).snapshot_state())


@pytest.mark.parametrize("kind", ["pc", "neutral"])
def test_should_refuse_foreign_policy_before_publishing_state(kind):
    learner = model(kind)
    before = learner.snapshot_state()
    x, y = batch()
    invalid: Any = object()
    with pytest.raises(ValueError):
        learner.train_epoch(x, y, 0.01, 4, 0.1, inference_policy=invalid)
    assert_snapshot(learner.snapshot_state(), before)


@pytest.mark.parametrize("policy", [object(), CoupledInferencePolicy(min_steps=5)])
def test_should_refuse_foreign_or_over_cap_adapter_policy(policy: Any):
    source = model("pc")
    before = source.snapshot_state()
    with pytest.raises(ValueError):
        CoupledLearner(
            source,
            learning_rate=0.01,
            inference_steps=4,
            inference_learning_rate=0.1,
            inference_policy=policy,
        )
    assert_snapshot(source.snapshot_state(), before)


def test_should_refuse_mutated_adapter_policy_before_model_copy():
    policy = replace(CoupledInferencePolicy(), gradient_tolerance=0.0)
    object.__setattr__(policy, "gradient_tolerance", float("nan"))
    with pytest.raises(ValueError):
        CoupledLearner(
            model("pc"),
            learning_rate=0.01,
            inference_steps=4,
            inference_learning_rate=0.1,
            inference_policy=policy,
        )
