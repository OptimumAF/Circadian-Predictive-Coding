"""Independent bounded-inference dynamics, ownership and stopping-work controls.

Hand-solved quadratic hooks test stopping logic. Real coupled-energy controls
write simultaneous latent updates directly and never train model parameters.
"""

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

import src.core.coupled_inference as inference
from src.core.coupled_energy import (
    CoupledEvaluation,
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
)
from src.core.coupled_inference import CoupledInferencePolicy, infer_coupled
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def parameters():
    return CoupledParameters(
        (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.zeros((1, 1)), np.zeros((1, 1))
    )


def test_should_expose_prospective_default_policy():
    default = CoupledInferencePolicy()
    assert default.min_steps == 1
    assert default.gradient_tolerance == 1e-6
    assert default.relative_energy_tolerance is None
    with pytest.raises(FrozenInstanceError):
        setattr(default, "min_steps", 2)


def policy(**changes):
    values: dict[str, Any] = dict(
        min_steps=1, gradient_tolerance=None, relative_energy_tolerance=None
    )
    values.update(changes)
    return CoupledInferencePolicy(**values)


def run(*, problem=None, **options):
    p, x, y, precisions = (
        (parameters(), np.ones((1, 1)), np.zeros((1, 1)), (1.0,)) if problem is None else problem
    )
    return infer_coupled(
        p,
        x,
        y,
        precisions,
        max_steps=options.pop("max_steps", 4),
        learning_rate=options.pop("learning_rate", 0.5),
        policy=options.pop("policy", policy()),
        **options,
    )


class Quadratic:
    def __init__(self, monkeypatch):
        self.calls = []
        self.initial = np.array([[2.0]])
        self.fail_evaluation = None
        self.increase_evaluation = None
        self.gradient_override = None
        self.failure_type = FloatingPointError
        monkeypatch.setattr(inference, "feedforward_latents", self.feedforward)
        monkeypatch.setattr(inference, "evaluate_coupled", self.evaluate)

    def feedforward(self, parameters, features):
        return (self.initial.copy(),)

    def evaluate(self, p, x, y, latents, precisions):
        self.calls.append(tuple(z.copy() for z in latents))
        if len(self.calls) == self.fail_evaluation:
            raise self.failure_type("deliberate evaluation failure")
        gradient = (
            latents[0].copy()
            if self.gradient_override is None
            else np.full_like(latents[0], self.gradient_override)
        )
        energy = float(np.sum(latents[0] ** 2) / 2)
        if len(self.calls) == self.increase_evaluation:
            energy += 100
        return CoupledEvaluation(energy, energy, (0.0,), (gradient,), parameters())


@pytest.mark.parametrize(
    "settings,cap,rate,reason,steps,value",
    [
        ({}, 3, 0.5, "max_steps", 3, 0.25),
        ({"gradient_tolerance": 1.0}, 4, 0.5, "gradient_tolerance", 1, 1.0),
        ({"min_steps": 3, "gradient_tolerance": 100.0}, 4, 0.5, "gradient_tolerance", 3, 0.25),
        ({"gradient_tolerance": 0.0}, 4, 1.0, "gradient_tolerance", 1, 0.0),
        ({"relative_energy_tolerance": 0.75}, 4, 0.5, "relative_energy_tolerance", 1, 1.0),
        ({"relative_energy_tolerance": 0.4}, 4, 0.5, "relative_energy_tolerance", 2, 0.5),
        (
            {"min_steps": 2, "relative_energy_tolerance": 100.0},
            4,
            0.5,
            "relative_energy_tolerance",
            2,
            0.5,
        ),
        (
            {"min_steps": 2, "gradient_tolerance": 100.0, "relative_energy_tolerance": 100.0},
            4,
            0.5,
            "gradient_tolerance",
            2,
            0.5,
        ),
    ],
)
def test_should_match_hand_solved_min_cap_thresholds_precedence_and_counts(
    monkeypatch, settings, cap, rate, reason, steps, value
):
    quadratic = Quadratic(monkeypatch)
    result = run(max_steps=cap, learning_rate=rate, policy=policy(**settings))
    diagnostic = result.diagnostics
    np.testing.assert_array_equal(result.latents[0], [[value]])
    assert diagnostic.reason == reason
    assert diagnostic.attempted_steps == diagnostic.accepted_steps == steps
    assert diagnostic.energy_evaluations == len(quadratic.calls) == steps + 1
    assert diagnostic.gradient_norm_evaluations == steps + 1
    assert diagnostic.energy_comparisons == steps
    assert diagnostic.initial_energy == 2.0
    assert diagnostic.final_energy == result.evaluation.energy == value * value / 2
    assert diagnostic.final_gradient_inf_norm == abs(value)
    assert diagnostic.used_fallback is False
    for index, latents in enumerate(quadratic.calls):
        np.testing.assert_array_equal(latents[0], [[2 * (1 - rate) ** index]])
    with pytest.raises(FrozenInstanceError):
        setattr(diagnostic, "accepted_steps", 99)


@pytest.mark.parametrize(
    "failure",
    ("energy_increase", "second_energy_increase", "nonfinite_candidate", "numerical_failure"),
)
def test_should_return_last_accepted_state_without_retry_on_unsafe_proposal(monkeypatch, failure):
    quadratic = Quadratic(monkeypatch)
    options: dict[str, Any] = dict(policy=policy(min_steps=3))
    if failure == "energy_increase":
        options["learning_rate"] = 3.0
    elif failure == "second_energy_increase":
        quadratic.increase_evaluation = 3
    elif failure == "nonfinite_candidate":
        quadratic.gradient_override = 1e308
        options["learning_rate"] = 2.0
    else:
        quadratic.fail_evaluation = 2
    result = run(**options)
    diagnostic = result.diagnostics
    accepted = 1 if failure == "second_energy_increase" else 0
    expected = 1.0 if accepted else 2.0
    np.testing.assert_array_equal(result.latents[0], [[expected]])
    assert diagnostic.reason == (
        "energy_increase" if failure == "second_energy_increase" else failure
    )
    assert diagnostic.attempted_steps == accepted + 1
    assert diagnostic.accepted_steps == accepted
    evaluations = 1 if failure == "nonfinite_candidate" else accepted + 2
    assert diagnostic.energy_evaluations == len(quadratic.calls) == evaluations
    assert diagnostic.gradient_norm_evaluations == 1 + accepted
    assert diagnostic.energy_comparisons == (accepted + 1 if "energy_increase" in failure else 0)
    assert diagnostic.final_energy == result.evaluation.energy == expected**2 / 2
    expected_gradient = 1e308 if failure == "nonfinite_candidate" else expected
    assert diagnostic.final_gradient_inf_norm == expected_gradient
    np.testing.assert_array_equal(result.evaluation.latent_gradients[0], [[expected_gradient]])
    assert diagnostic.used_fallback is True


@pytest.mark.parametrize(
    "failure_index,failure_type", ((1, FloatingPointError), (1, ValueError), (2, ValueError))
)
def test_should_raise_when_no_initial_fallback_or_programmer_error(
    monkeypatch, failure_index, failure_type
):
    quadratic = Quadratic(monkeypatch)
    quadratic.fail_evaluation, quadratic.failure_type = failure_index, failure_type
    with pytest.raises(failure_type, match="deliberate"):
        run()
    assert len(quadratic.calls) == failure_index


@pytest.mark.parametrize("depth", ((3,), (3, 2), (3, 2, 3, 2)))
def test_should_match_manual_real_simultaneous_updates_at_monotonic_prefixes(depth):
    p = CoupledPredictiveCodingNetwork(2, depth, seed=41).snapshot_parameters()
    x = np.array([[0.3, -0.4], [0.7, 0.2], [-0.2, 0.8]])
    y = np.array([[0.0], [0.35], [1.0]])
    beta = tuple(0.7 + 0.2 * i for i in range(len(depth)))
    sources = (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias, x, y)
    originals = tuple(a.copy() for a in sources)
    manual = feedforward_latents(p, x)
    evaluated = evaluate_coupled(p, x, y, manual, beta)
    for step in range(1, 5):
        proposed = tuple(
            z - 0.05 * g for z, g in zip(manual, evaluated.latent_gradients, strict=True)
        )
        new_evaluation = evaluate_coupled(p, x, y, proposed, beta)
        assert new_evaluation.energy <= evaluated.energy
        manual, evaluated = proposed, new_evaluation
        if step not in (1, 2, 4):
            continue
        result = run(problem=(p, x, y, beta), max_steps=step, learning_rate=0.05)
        assert result.diagnostics.reason == "max_steps"
        assert result.diagnostics.accepted_steps == result.diagnostics.attempted_steps == step
        assert (
            result.diagnostics.energy_evaluations
            == result.diagnostics.gradient_norm_evaluations
            == step + 1
        )
        assert result.diagnostics.energy_comparisons == step
        assert result.diagnostics.used_fallback is False
        assert result.evaluation.energy == result.diagnostics.final_energy == evaluated.energy
        assert result.diagnostics.final_gradient_inf_norm == max(
            float(np.max(np.abs(g))) for g in evaluated.latent_gradients
        )
        outputs = (
            *result.latents,
            *result.evaluation.latent_gradients,
            *result.evaluation.parameter_gradients.hidden_weights,
            *result.evaluation.parameter_gradients.hidden_biases,
            result.evaluation.parameter_gradients.output_weight,
            result.evaluation.parameter_gradients.output_bias,
        )
        for observed, expected in zip(result.latents, manual, strict=True):
            np.testing.assert_array_equal(observed, expected)
        for index, a in enumerate(outputs):
            for original in sources:
                assert not np.shares_memory(a, original)
            for b in outputs[index + 1 :]:
                assert not np.shares_memory(a, b)
        result.latents[0].fill(999)
    for observed, expected in zip(sources, originals, strict=True):
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "changes",
    (
        {"min_steps": True},
        {"min_steps": 0},
        {"min_steps": 10001},
        {"min_steps": 1.0},
        {"min_steps": np.int64(1)},
        {"gradient_tolerance": True},
        {"gradient_tolerance": -1.0},
        {"gradient_tolerance": float("nan")},
        {"gradient_tolerance": np.float64(0.1)},
        {"relative_energy_tolerance": float("inf")},
        {"relative_energy_tolerance": -1},
        {"relative_energy_tolerance": False},
        {"gradient_tolerance": 10**1000},
    ),
)
def test_should_refuse_invalid_policy_settings(changes: Any):
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize("cap", (True, 0, 10001, 2.0, np.int64(2), -1))
def test_should_refuse_invalid_caps_before_energy_work(monkeypatch, cap: Any):
    quadratic = Quadratic(monkeypatch)
    with pytest.raises(ValueError):
        run(max_steps=cap)
    assert quadratic.calls == []


@pytest.mark.parametrize("rate", (True, 0, -0.1, float("inf"), np.float64(0.1), 10**1000))
def test_should_refuse_invalid_rates_before_energy_work(monkeypatch, rate: Any):
    quadratic = Quadratic(monkeypatch)
    with pytest.raises(ValueError):
        run(learning_rate=rate)
    assert quadratic.calls == []


@pytest.mark.parametrize("bad_policy", (None, {}, "adaptive"))
def test_should_refuse_foreign_policy_before_energy_work(monkeypatch, bad_policy: Any):
    quadratic = Quadratic(monkeypatch)
    with pytest.raises(ValueError):
        run(policy=bad_policy)
    assert quadratic.calls == []


@pytest.mark.parametrize("field,value", (("min_steps", True), ("gradient_tolerance", float("nan"))))
def test_should_revalidate_mutated_policy_before_energy_work(monkeypatch, field, value):
    quadratic = Quadratic(monkeypatch)
    bad = policy()
    object.__setattr__(bad, field, value)
    with pytest.raises(ValueError):
        run(policy=bad)
    assert quadratic.calls == []


def test_should_refuse_minimum_above_cap_before_energy_work(monkeypatch):
    quadratic = Quadratic(monkeypatch)
    with pytest.raises(ValueError):
        run(max_steps=2, policy=policy(min_steps=3))
    assert quadratic.calls == []


@pytest.mark.parametrize(
    "problem",
    (
        (None, np.ones((1, 1)), np.zeros((1, 1)), (1.0,)),
        (parameters(), np.ones((1, 1), dtype=np.float32), np.zeros((1, 1)), (1.0,)),
        (parameters(), np.ones((1, 1)), np.array([[1.1]]), (1.0,)),
        (parameters(), np.ones((1, 1)), np.zeros((1, 1)), [1.0]),
        (parameters(), np.ones((1, 1)), np.zeros((1, 1)), (True,)),
        (parameters(), np.ones((1, 1)), np.zeros((2, 1)), (1.0,)),
    ),
)
def test_should_raise_for_invalid_initial_problem_instead_of_fallback(problem: Any):
    with pytest.raises(ValueError):
        run(problem=problem)
