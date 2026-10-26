"""Continuous latent inference: hand-solved guards and independent tiny prefixes.

No RNG, model construction, parameter fit, simulator or deployment update occurs.
Inputs use fixed float64 arrays; the manual four-axis oracle calls no production
energy or gradient helper.
"""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from src.core import continuous_pc as energy
from src.core import continuous_pc_inference as inference
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy


def policy(**changes):
    values: dict[str, Any] = dict(
        min_steps=1, gradient_tolerance=None, relative_energy_tolerance=None
    )
    values.update(changes)
    return CoupledInferencePolicy(**values)


def scalar_problem(output_weight=1.0):
    params = CoupledParameters(
        (np.zeros((1, 1)),),
        (np.zeros((1, 1)),),
        np.array([[output_weight]], dtype=np.float64),
        np.zeros((1, 1)),
    )
    return (
        params,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        (1.0,),
        ContinuousActionSpec(1, ("normalized",), (-1.0,), (1.0,)),
    )


def run(*, problem=None, **options):
    params, features, targets, precisions, spec = scalar_problem() if problem is None else problem
    return inference.infer_continuous_pc(
        params,
        features,
        targets,
        precisions,
        spec,
        max_steps=options.pop("max_steps", 4),
        learning_rate=options.pop("learning_rate", 0.5),
        policy=options.pop("policy", policy()),
        **options,
    )


class Quadratic:
    """A scalar E=z²/2 hook isolates the shared guard from task calculus."""

    def __init__(self, monkeypatch):
        self.initial = np.array([[2.0]])
        self.calls = []
        self.evaluations = []
        self.initializations = 0
        self.fail_on: int | None = None
        self.increase_on: int | None = None
        self.nonfinite_on: int | None = None
        self.failure_type: Any = FloatingPointError
        self.gradient_override: Any = None
        self.constant_energy = False
        monkeypatch.setattr(inference.energy, "feedforward_continuous_latents", self.feedforward)
        monkeypatch.setattr(inference.energy, "evaluate_continuous_pc", self.evaluate)

    def feedforward(self, _params, _features, _spec):
        self.initializations += 1
        return (self.initial.copy(),)

    def evaluate(self, _params, _features, _targets, latents, precisions, _spec):
        self.calls.append(tuple(value.copy() for value in latents))
        if len(self.calls) == self.fail_on:
            raise self.failure_type("deliberate inference energy failure")
        value = 2.0 if self.constant_energy else float(np.square(latents[0]).sum() / 2)
        if len(self.calls) == self.increase_on:
            value += 100.0
        if len(self.calls) == self.nonfinite_on:
            value = float("nan")
        gradient = (
            latents[0].copy()
            if self.gradient_override is None
            else np.full_like(latents[0], self.gradient_override)
        )
        zeros = CoupledParameters(
            (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.zeros((1, 1)), np.zeros((1, 1))
        )
        evaluated = energy.ContinuousPCEvaluation(
            value, value, (0.0,), np.zeros((1, 1)), (gradient,), zeros, tuple(precisions)
        )
        self.evaluations.append(evaluated)
        return evaluated


@pytest.mark.parametrize(
    "settings,cap,rate,reason,steps,latent",
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
def test_should_match_quadratic_minimum_cap_threshold_precedence_and_exact_work(
    monkeypatch, settings, cap, rate, reason, steps, latent
):
    quadratic = Quadratic(monkeypatch)
    result = run(max_steps=cap, learning_rate=rate, policy=policy(**settings))
    diagnostic = result.diagnostics
    np.testing.assert_array_equal(result.latents[0], [[latent]])
    assert diagnostic.reason == reason and diagnostic.used_fallback is False
    assert diagnostic.attempted_steps == diagnostic.accepted_steps == steps
    assert diagnostic.energy_evaluations == diagnostic.gradient_norm_evaluations == steps + 1
    assert diagnostic.energy_comparisons == steps and quadratic.initializations == 1
    assert len(quadratic.calls) == steps + 1 and result.evaluation is quadratic.evaluations[-1]
    assert diagnostic.initial_energy == 2.0 and diagnostic.final_energy == latent**2 / 2
    assert diagnostic.final_gradient_inf_norm == abs(latent)
    for index, values in enumerate(quadratic.calls):
        np.testing.assert_array_equal(values[0], [[2 * (1 - rate) ** index]])
    with pytest.raises(FrozenInstanceError):
        setattr(diagnostic, "accepted_steps", 99)


def test_should_accept_equal_energy_with_nonzero_gradient_and_apply_relative_drop_after_minimum(
    monkeypatch,
):
    quadratic = Quadratic(monkeypatch)
    quadratic.constant_energy = True
    result = run(policy=policy(min_steps=2, relative_energy_tolerance=0.0))
    diagnostic = result.diagnostics
    assert diagnostic.reason == "relative_energy_tolerance" and diagnostic.used_fallback is False
    assert (
        diagnostic.accepted_steps
        == diagnostic.attempted_steps
        == diagnostic.energy_comparisons
        == 2
    )
    assert (
        diagnostic.energy_evaluations
        == diagnostic.gradient_norm_evaluations
        == len(quadratic.calls)
        == 3
    )
    assert diagnostic.initial_energy == diagnostic.final_energy == 2.0
    assert diagnostic.final_gradient_inf_norm == 0.5
    np.testing.assert_array_equal(result.latents[0], [[0.5]])


def test_should_own_the_validated_policy_before_initializer_callbacks(monkeypatch):
    quadratic = Quadratic(monkeypatch)
    declared = policy(min_steps=3, gradient_tolerance=100.0)

    def initialize(params, features, spec):
        object.__setattr__(declared, "min_steps", 1)
        return quadratic.feedforward(params, features, spec)

    monkeypatch.setattr(inference.energy, "feedforward_continuous_latents", initialize)
    result = run(policy=declared)
    assert declared.min_steps == 1 and result.diagnostics.accepted_steps == 3
    assert result.diagnostics.reason == "gradient_tolerance"
    assert result.diagnostics.energy_evaluations == len(quadratic.calls) == 4


@pytest.mark.parametrize(
    "failure",
    [
        "increase",
        "late_increase",
        "candidate_nonfinite",
        "energy_failure",
        "energy_nonfinite",
        "late_energy_failure",
    ],
)
def test_should_refuse_unsafe_work_without_retry_and_return_the_last_accepted_evaluation(
    monkeypatch, failure
):
    quadratic = Quadratic(monkeypatch)
    options: dict[str, Any] = dict(policy=policy(min_steps=3))
    if failure == "increase":
        options["learning_rate"] = 3.0
    elif failure == "late_increase":
        quadratic.increase_on = 3
    elif failure == "candidate_nonfinite":
        quadratic.gradient_override = 1e308
        options["learning_rate"] = 2.0
    elif failure == "energy_nonfinite":
        quadratic.nonfinite_on = 2
    else:
        quadratic.fail_on = 3 if failure == "late_energy_failure" else 2
    result = run(**options)
    diagnostic = result.diagnostics
    accepted = int(failure.startswith("late_"))
    expected = 1.0 if accepted else 2.0
    comparisons = accepted + 1 if failure in ("increase", "late_increase") else accepted
    calls = 1 if failure == "candidate_nonfinite" else accepted + 2
    expected_reason = (
        "energy_increase"
        if "increase" in failure
        else "nonfinite_candidate"
        if failure == "candidate_nonfinite"
        else "numerical_failure"
    )
    assert diagnostic.reason == expected_reason and diagnostic.used_fallback is True
    assert diagnostic.attempted_steps == accepted + 1 and diagnostic.accepted_steps == accepted
    assert diagnostic.energy_evaluations == len(quadratic.calls) == calls
    assert (
        diagnostic.gradient_norm_evaluations == accepted + 1
        and diagnostic.energy_comparisons == comparisons
    )
    assert result.evaluation is quadratic.evaluations[accepted]
    assert diagnostic.final_energy == result.evaluation.energy == expected**2 / 2
    np.testing.assert_array_equal(result.latents[0], [[expected]])
    assert quadratic.initializations == 1


@pytest.mark.parametrize(
    "index,error,nonfinite",
    [
        (1, FloatingPointError, False),
        (1, ValueError, False),
        (2, ValueError, False),
        (1, FloatingPointError, True),
    ],
)
def test_should_propagate_initial_or_programmer_failures_instead_of_inventing_a_fallback(
    monkeypatch, index, error, nonfinite
):
    quadratic = Quadratic(monkeypatch)
    if nonfinite:
        quadratic.nonfinite_on = index
    else:
        quadratic.fail_on, quadratic.failure_type = index, error
    with pytest.raises(error):
        run()
    assert quadratic.initializations == 1 and len(quadratic.calls) == index


@pytest.mark.parametrize(
    "settings",
    [
        dict(max_steps=0),
        dict(max_steps=True),
        dict(max_steps=10001),
        dict(learning_rate=True),
        dict(learning_rate=0.0),
        dict(learning_rate=float("inf")),
        dict(policy=None),
        dict(max_steps=2, policy=CoupledInferencePolicy(min_steps=3)),
    ],
)
def test_should_refuse_settings_before_initialization_or_energy(
    monkeypatch, settings: dict[str, Any]
):
    quadratic = Quadratic(monkeypatch)
    with pytest.raises(ValueError):
        run(**settings)
    assert quadratic.initializations == 0 and quadratic.calls == []


@pytest.mark.parametrize("field,value", [("min_steps", True), ("gradient_tolerance", float("nan"))])
def test_should_revalidate_a_corrupted_policy_before_numerical_work(monkeypatch, field, value):
    quadratic = Quadratic(monkeypatch)
    declared = policy()
    object.__setattr__(declared, field, value)
    with pytest.raises(ValueError):
        run(policy=declared)
    assert quadratic.initializations == 0 and quadratic.calls == []


@pytest.mark.parametrize(
    "damage",
    [
        "target_bound",
        "target_nonfinite",
        "precision_bool",
        "precision_count",
        "feature_dtype",
        "parameter_nonfinite",
        "output_shape",
    ],
)
def test_should_validate_initial_targets_precisions_and_geometry_before_feedforward(
    monkeypatch, damage
):
    quadratic = Quadratic(monkeypatch)
    params, x, y, beta, spec = scalar_problem()
    if damage == "target_bound":
        y[0, 0] = 2.0
    elif damage == "target_nonfinite":
        y[0, 0] = np.nan
    elif damage == "precision_bool":
        beta = (True,)
    elif damage == "precision_count":
        beta = ()
    elif damage == "feature_dtype":
        x = x.astype(np.float32)
    elif damage == "parameter_nonfinite":
        params.hidden_weights[0][0, 0] = np.inf
    else:
        params = replace(params, output_weight=np.zeros((1, 2)))
    with pytest.raises(ValueError):
        run(problem=(params, x, y, beta, spec))
    assert quadratic.initializations == 0 and quadratic.calls == []


def deep_problem(depth):
    params = CoupledParameters(
        tuple(
            np.array([[0.12 + 0.01 * layer, -0.08], [0.05, -0.17 + 0.02 * layer]])
            for layer in range(depth)
        ),
        tuple(np.array([[0.03, -0.02]]) * (layer + 1) for layer in range(depth)),
        np.array([[0.2, -0.15, 0.3, -0.2], [-0.1, 0.25, -0.2, 0.1]]),
        np.array([[0.01, -0.04, 0.07, 0.03]]),
    )
    x = np.array([[0.2, -0.3], [0.7, 0.4]])
    y = np.array([[-0.5, 0.25, 0.75, -1.0], [0.4, -0.6, 0.2, 1.0]])
    beta = tuple(0.8 + 0.1 * layer for layer in range(depth))
    spec = ContinuousActionSpec(
        2,
        ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    return params, x, y, beta, spec


def arrays(params):
    return (*params.hidden_weights, *params.hidden_biases, params.output_weight, params.output_bias)


def manual_evaluation(params, x, y, z, beta):
    """Independent scalar E and partials for the declared four normalized axes."""
    previous = [x, *z[:-1]]
    predictions = [
        np.tanh(lower @ weight + bias)
        for lower, weight, bias in zip(
            previous, params.hidden_weights, params.hidden_biases, strict=True
        )
    ]
    residuals = [value - prediction for value, prediction in zip(z, predictions, strict=True)]
    hidden_energy = sum(
        float(precision * np.square(residual).sum() / (2 * len(x)))
        for precision, residual in zip(beta, residuals, strict=True)
    )
    output = np.tanh(z[-1] @ params.output_weight + params.output_bias)
    task = float(np.square(output - y).sum() / (2 * len(x) * 4))
    head_delta = (output - y) * (1 - output**2) / (len(x) * 4)
    gradients = []
    for index, (precision, residual) in enumerate(zip(beta, residuals, strict=True)):
        local = precision * residual / len(x)
        if index + 1 < len(z):
            next_delta = (
                -beta[index + 1] * residuals[index + 1] * (1 - predictions[index + 1] ** 2) / len(x)
            )
            local = local + next_delta @ params.hidden_weights[index + 1].T
        else:
            local = local + head_delta @ params.output_weight.T
        gradients.append(local)
    return task + hidden_energy, tuple(gradients)


@pytest.fixture(scope="module", params=[1, 2, 4])
def manual_case(request):
    problem = deep_problem(request.param)
    params, x, y, beta, _ = problem
    current, initialized = x, []
    for weight, bias in zip(params.hidden_weights, params.hidden_biases, strict=True):
        current = np.tanh(current @ weight + bias)
        initialized.append(current)
    states = [tuple(initialized)]
    energy_value, gradients = manual_evaluation(params, x, y, states[0], beta)
    energies, norms = [energy_value], [max(float(np.abs(value).max()) for value in gradients)]
    for _ in range(4):
        candidate = tuple(
            value - 0.05 * gradient for value, gradient in zip(states[-1], gradients, strict=True)
        )
        next_energy, gradients = manual_evaluation(params, x, y, candidate, beta)
        assert next_energy <= energies[-1]
        states.append(candidate)
        energies.append(next_energy)
        norms.append(max(float(np.abs(value).max()) for value in gradients))
    return problem, states, energies, norms


@pytest.mark.parametrize("cap", [1, 2, 4])
def test_should_match_manual_simultaneous_four_axis_updates_at_every_frozen_prefix(
    manual_case, cap
):
    problem, states, energies, norms = manual_case
    params, x, y, beta, spec = problem
    protected = (*arrays(params), x, y)
    before = tuple(value.copy() for value in protected)
    result = run(problem=problem, max_steps=cap, learning_rate=0.05)
    diagnostic = result.diagnostics
    assert diagnostic.reason == "max_steps" and diagnostic.used_fallback is False
    assert (
        diagnostic.accepted_steps
        == diagnostic.attempted_steps
        == diagnostic.energy_comparisons
        == cap
    )
    assert diagnostic.energy_evaluations == diagnostic.gradient_norm_evaluations == cap + 1
    assert diagnostic.initial_energy == pytest.approx(energies[0], abs=1e-14, rel=1e-14)
    assert diagnostic.final_energy == pytest.approx(energies[cap], abs=1e-14, rel=1e-14)
    assert diagnostic.final_gradient_inf_norm == pytest.approx(norms[cap], abs=1e-14, rel=1e-14)
    assert result.evaluation.energy == diagnostic.final_energy
    assert (
        result.evaluation.task_definition
        == "continuous_bc_half_mean_active_axis_normalized_squared_error_v1"
    )
    for actual, expected in zip(result.latents, states[cap], strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    assert result.evaluation.predictions.shape == (2, 4)
    for actual, original in zip(protected, before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_should_own_results_and_leave_target_free_deployment_unchanged_by_inference():
    problem = deep_problem(2)
    params, x, y, beta, spec = problem
    protected = (*arrays(params), x, y)
    before = tuple(value.copy() for value in protected)
    deployed = energy.predict_continuous_pc(params, x, spec)
    first = run(problem=problem, max_steps=2, learning_rate=0.05)
    second = run(problem=(params, x, -y, beta, spec), max_steps=2, learning_rate=0.05)
    assert np.linalg.norm(first.latents[-1] - second.latents[-1]) > 1e-8
    np.testing.assert_array_equal(energy.predict_continuous_pc(params, x, spec), deployed)
    outputs = (
        *first.latents,
        first.evaluation.predictions,
        *first.evaluation.latent_gradients,
        *arrays(first.evaluation.parameter_gradients),
    )
    for value in outputs:
        assert all(not np.shares_memory(value, original) for original in protected)
    for index, value in enumerate(outputs):
        assert all(not np.shares_memory(value, other) for other in outputs[index + 1 :])
    for value in outputs:
        value.fill(999.0)
    for value, original in zip(protected, before, strict=True):
        np.testing.assert_array_equal(value, original)
    assert first.diagnostics.final_energy == first.evaluation.energy


@pytest.mark.parametrize("mode", ["zero", "constant", "saturated"])
def test_should_accept_equal_energy_without_exiting_before_the_minimum(mode):
    params, x, y, beta, spec = scalar_problem()
    if mode == "constant":
        spec = replace(spec, lower_bounds=(2.0,), upper_bounds=(2.0,))
        y.fill(2.0)
    elif mode == "saturated":
        params.output_bias.fill(1000.0)
        y.fill(-1.0)
    result = run(
        problem=(params, x, y, beta, spec),
        max_steps=4,
        learning_rate=0.05,
        policy=policy(min_steps=3, gradient_tolerance=0.0, relative_energy_tolerance=0.0),
    )
    diagnostic = result.diagnostics
    assert diagnostic.reason == "gradient_tolerance" and diagnostic.used_fallback is False
    assert (
        diagnostic.accepted_steps
        == diagnostic.attempted_steps
        == diagnostic.energy_comparisons
        == 3
    )
    assert diagnostic.energy_evaluations == diagnostic.gradient_norm_evaluations == 4
    assert (
        diagnostic.initial_energy
        == diagnostic.final_energy
        == (2.0 if mode == "saturated" else 0.0)
    )
    np.testing.assert_array_equal(result.latents[0], [[0.0]])


@pytest.mark.parametrize(
    "head,reason,entries", [(1.0, "numerical_failure", 2), (2.0, "nonfinite_candidate", 1)]
)
def test_should_preserve_real_initial_state_when_extreme_rate_cannot_produce_safe_work(
    head, reason, entries
):
    params, x, y, beta, spec = scalar_problem(head)
    y.fill(1.0)
    result = run(problem=(params, x, y, beta, spec), max_steps=1, learning_rate=1e308)
    diagnostic = result.diagnostics
    assert diagnostic.reason == reason and diagnostic.used_fallback is True
    assert diagnostic.attempted_steps == 1 and diagnostic.accepted_steps == 0
    assert diagnostic.energy_evaluations == entries and diagnostic.gradient_norm_evaluations == 1
    assert (
        diagnostic.energy_comparisons == 0
        and diagnostic.final_energy == diagnostic.initial_energy == 0.5
    )
    np.testing.assert_array_equal(result.latents[0], [[0.0]])


@pytest.mark.parametrize("stage", ["initializer", "energy"])
def test_should_raise_for_real_initial_overflow_without_claiming_an_energy_fallback(
    monkeypatch, stage
):
    params, x, y, beta, spec = scalar_problem()
    if stage == "initializer":
        params.hidden_weights[0].fill(1e308)
        x.fill(1e308)
    else:
        params.hidden_biases[0].fill(1000.0)
        params.output_weight.fill(1e308)
        params.output_bias.fill(1e308)
    protected = (*arrays(params), x, y)
    before = tuple(value.copy() for value in protected)
    original_energy = inference.energy.evaluate_continuous_pc
    calls = []

    def evaluate(*args):
        calls.append(True)
        if stage == "initializer":
            pytest.fail("failed initializer evaluated energy")
        return original_energy(*args)

    monkeypatch.setattr(inference.energy, "evaluate_continuous_pc", evaluate)
    with pytest.raises(FloatingPointError):
        run(problem=(params, x, y, beta, spec))
    assert len(calls) == int(stage == "energy")
    for actual, original in zip(protected, before, strict=True):
        np.testing.assert_array_equal(actual, original)
