"""Detached TRAIN-only bounded inference mathematics, safety, and paid work."""

from dataclasses import FrozenInstanceError
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters.torch_feature_learner import TorchFeaturePolicy, _components
from src.core import categorical_pc_inference as bounded
from src.core import categorical_pc_weighted as weighted
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.learner_ports import TrainingUpdateFailed
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from test_categorical_pc_weighted import NAMES, WEIGHTS, fixture, objective, torch as torch


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        bounded_proposal_attempts=0,
        bounded_proposals_returned=0,
        bounded_configured_nominal_latent_row_iterations=0,
        bounded_attempted_latent_row_iterations=0,
        bounded_accepted_latent_row_iterations=0,
        fixed_proposal_attempts=0,
        fixed_nominal_latent_row_iterations=0,
        native_calls=0,
        native_nominal_latent_row_iterations=0,
    )
    original = bounded.bounded_class_weighted_pc_update

    def invoke(features, targets, parameters, **options):
        work["bounded_proposal_attempts"] += 1
        steps = options.get("inference_steps")
        if type(steps) is int and 1 <= steps <= 64:
            work["bounded_configured_nominal_latent_row_iterations"] += len(features) * steps
        try:
            result = original(features, targets, parameters, **options)
        except TrainingUpdateFailed as error:
            work["bounded_attempted_latent_row_iterations"] += error.work.get(
                "attempted_latent_row_iterations", 0
            )
            work["bounded_accepted_latent_row_iterations"] += error.work.get(
                "accepted_latent_row_iterations", 0
            )
            raise
        work["bounded_proposals_returned"] += 1
        work["bounded_attempted_latent_row_iterations"] += (
            result.diagnostics.attempted_latent_row_iterations
        )
        work["bounded_accepted_latent_row_iterations"] += (
            result.diagnostics.accepted_latent_row_iterations
        )
        return result

    monkeypatch.setattr(bounded, "bounded_class_weighted_pc_update", invoke)
    fixed = weighted.class_weighted_pc_update

    def fixed_invoke(features, targets, parameters, **options):
        work["fixed_proposal_attempts"] += 1
        work["fixed_nominal_latent_row_iterations"] += len(features) * options["inference_steps"]
        return fixed(features, targets, parameters, **options)

    monkeypatch.setattr(weighted, "class_weighted_pc_update", fixed_invoke)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        train = kind.train_step

        def native(*args, _train=train, **options):
            work["native_calls"] += 1
            work["native_nominal_latent_row_iterations"] += len(args[1]) * args[4]
            return _train(*args, **options)

        monkeypatch.setattr(kind, "train_step", native)
    yield
    for name, value in work.items():
        record_property(name, value)


def policy(**changes):
    values: dict[str, Any] = dict(
        min_steps=1, gradient_tolerance=None, relative_energy_tolerance=None
    )
    values.update(changes)
    return CoupledInferencePolicy(**values)


def invoke(features, targets, parameters, **changes):
    options: dict[str, Any] = dict(
        class_weights=WEIGHTS,
        learning_rate=0.03,
        inference_steps=3,
        inference_learning_rate=0.04,
        role="train",
        policy=policy(),
    )
    options.update(changes)
    return bounded.bounded_class_weighted_pc_update(features, targets, parameters, **options)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_should_match_independent_autograd_latents_partial_gradients_and_candidate(
    torch, dtype_name
):
    dtype = getattr(torch, dtype_name)
    features, targets, parameters = fixture(torch, dtype)
    result = invoke(features, targets, parameters)
    hidden = (features @ parameters[0] + parameters[1]).tanh().detach()
    energies = [objective(torch, features, targets, parameters, hidden).item()]
    for _ in range(3):
        hidden.requires_grad_()
        direction = (
            len(targets)
            * torch.autograd.grad(objective(torch, features, targets, parameters, hidden), hidden)[
                0
            ]
        )
        hidden = (hidden - 0.04 * direction).detach()
        energies.append(objective(torch, features, targets, parameters, hidden).item())
    independent_parameters = tuple(value.clone().requires_grad_() for value in parameters)
    energy = objective(torch, features, targets, independent_parameters, hidden)
    gradients = torch.autograd.grad(energy, independent_parameters)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(result.update.hidden_state, hidden, rtol=tolerance, atol=tolerance)
    assert all(after <= before for before, after in zip(energies, energies[1:]))
    assert result.diagnostics.initial_energy == pytest.approx(energies[0], abs=tolerance)
    assert result.diagnostics.final_energy == pytest.approx(energies[-1], abs=tolerance)
    assert result.update.energy == pytest.approx(energy.item(), abs=tolerance)
    for old, actual, gradient, candidate in zip(
        parameters, result.update.gradients, gradients, result.update.parameters, strict=True
    ):
        torch.testing.assert_close(actual, gradient, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(candidate, old - 0.03 * gradient, rtol=tolerance, atol=tolerance)
    independent_hidden = hidden.clone().requires_grad_()
    direction = (
        len(targets)
        * torch.autograd.grad(
            objective(torch, features, targets, parameters, independent_hidden), independent_hidden
        )[0]
    )
    assert result.diagnostics.final_scaled_gradient_inf_norm == pytest.approx(
        direction.abs().max().item(), abs=tolerance
    )


def test_should_match_finite_difference_for_every_final_parameter_partial(torch):
    features, targets, parameters = fixture(torch, torch.float64)
    result = invoke(features, targets, parameters)
    epsilon = 1e-6
    for parameter_index, gradient in enumerate(result.update.gradients):
        for coordinate in range(gradient.numel()):
            energies = []
            for sign in (-1, 1):
                changed = tuple(value.clone() for value in parameters)
                changed[parameter_index].view(-1)[coordinate] += sign * epsilon
                energies.append(
                    objective(torch, features, targets, changed, result.update.hidden_state).item()
                )
            assert gradient.view(-1)[coordinate].item() == pytest.approx(
                (energies[1] - energies[0]) / (2 * epsilon), abs=3e-9
            )


@pytest.mark.parametrize("steps", [1, 3, 64])
@pytest.mark.parametrize("weights", [WEIGHTS, (1.0, 1.0, 1.0)])
def test_should_preserve_fixed_loop_outputs_when_every_step_is_accepted(torch, steps, weights):
    features, targets, parameters = fixture(torch, torch.float64)
    result = invoke(features, targets, parameters, class_weights=weights, inference_steps=steps)
    original = weighted.class_weighted_pc_update(
        features,
        targets,
        parameters,
        class_weights=weights,
        learning_rate=0.03,
        inference_steps=steps,
        inference_learning_rate=0.04,
        role="train",
    )
    assert result.diagnostics.reason == "max_steps" and not result.diagnostics.used_fallback
    assert result.diagnostics.accepted_steps == result.diagnostics.attempted_steps == steps
    assert result.update.energy == original.energy
    assert result.update.legacy_diagnostic == original.legacy_diagnostic
    assert torch.equal(result.update.hidden_state, original.hidden_state)
    for actual, expected in zip(
        (*result.update.parameters, *result.update.gradients),
        (*original.parameters, *original.gradients),
        strict=True,
    ):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps", [2, 64])
def test_should_match_actual_native_all_ones_pc_and_neutral_parameters(torch, method, steps):
    features, targets, _ = fixture(torch, torch.float32)
    head, _ = _components(TorchFeaturePolicy(method, 2, 2, 17, num_classes=3))
    parameters = tuple(getattr(head, name).clone() for name in NAMES)
    result = invoke(features, targets, parameters, class_weights=(1.0,) * 3, inference_steps=steps)
    legacy = head.train_step(features, targets, 0.03, steps, 0.04)
    assert result.diagnostics.accepted_steps == steps
    assert result.update.legacy_diagnostic == legacy
    for name, actual in zip(NAMES, result.update.parameters, strict=True):
        assert torch.equal(actual, getattr(head, name))


@pytest.mark.parametrize(
    "settings,reason,accepted",
    [
        ({}, "max_steps", 4),
        ({"gradient_tolerance": 0.0}, "gradient_tolerance", 1),
        ({"min_steps": 3, "gradient_tolerance": 0.0}, "gradient_tolerance", 3),
        ({"min_steps": 2, "relative_energy_tolerance": 0.0}, "relative_energy_tolerance", 2),
        ({"gradient_tolerance": 0.0, "relative_energy_tolerance": 0.0}, "gradient_tolerance", 1),
    ],
)
def test_should_respect_minimum_cap_and_tolerance_precedence_on_stationary_latents(
    torch, settings, reason, accepted
):
    features, targets, parameters = fixture(torch, torch.float64)
    parameters[2].zero_()
    result = invoke(features, targets, parameters, inference_steps=4, policy=policy(**settings))
    diagnostic = result.diagnostics
    assert diagnostic.rows == 3
    assert diagnostic.reason == reason
    assert diagnostic.attempted_steps == diagnostic.accepted_steps == accepted
    assert diagnostic.rejected_steps == 0 and diagnostic.used_fallback is False
    assert diagnostic.final_energy == diagnostic.initial_energy
    assert diagnostic.final_scaled_gradient_inf_norm == 0.0
    assert result.update.nominal_latent_row_iterations == 3 * accepted
    assert diagnostic.attempted_latent_row_iterations == 3 * accepted
    assert diagnostic.accepted_latent_row_iterations == 3 * accepted
    assert (
        diagnostic.energy_evaluations_attempted
        == diagnostic.energy_evaluations_completed
        == accepted + 1
    )
    assert (
        diagnostic.latent_gradient_evaluations_attempted
        == diagnostic.latent_gradient_evaluations_completed
        == accepted + 1
    )
    assert diagnostic.gradient_norm_evaluations == accepted + 1
    assert diagnostic.energy_comparisons == accepted
    for name in (
        "parameter_gradient_evaluations",
        "parameter_proposals",
        "final_energy_evaluations",
        "legacy_diagnostic_evaluations",
        "initial_prior_evaluations",
    ):
        assert getattr(diagnostic, name + "_attempted") == 1
        assert getattr(diagnostic, name + "_completed") == 1
    assert diagnostic.failure_work_may_be_unobserved is False
    saved = diagnostic.work
    saved["accepted_steps"] = 999
    assert diagnostic.work["accepted_steps"] == accepted
    with pytest.raises(FrozenInstanceError):
        diagnostic.accepted_steps = 999


@pytest.mark.parametrize("failure", ["energy_increase", "nonfinite_candidate"])
def test_should_keep_prior_state_on_first_unsafe_attempt_without_retry(torch, failure):
    features, targets, parameters = fixture(torch, torch.float64)
    if failure == "nonfinite_candidate":
        parameters[2].mul_(1e200)
    rate = 1e200 if failure == "nonfinite_candidate" else 1e6
    prior = (features @ parameters[0] + parameters[1]).tanh()
    result = invoke(
        features, targets, parameters, inference_learning_rate=rate, policy=policy(min_steps=3)
    )
    diagnostic = result.diagnostics
    assert diagnostic.reason == failure and diagnostic.used_fallback is True
    assert diagnostic.attempted_steps == diagnostic.rejected_steps == 1
    assert diagnostic.accepted_steps == result.update.nominal_latent_row_iterations == 0
    assert diagnostic.attempted_latent_row_iterations == 3
    assert diagnostic.accepted_latent_row_iterations == 0
    assert torch.equal(result.update.hidden_state, prior)
    assert diagnostic.final_energy == diagnostic.initial_energy
    assert (
        diagnostic.latent_gradient_evaluations_completed
        == diagnostic.gradient_norm_evaluations
        == 1
    )
    expected_energy_calls = 1 if failure == "nonfinite_candidate" else 2
    assert (
        diagnostic.energy_evaluations_attempted
        == diagnostic.energy_evaluations_completed
        == expected_energy_calls
    )
    assert diagnostic.energy_comparisons == (failure == "energy_increase")


@pytest.mark.parametrize(
    "changes",
    [
        dict(role="final_test"),
        dict(role="probe"),
        dict(role=True),
        dict(inference_steps=0),
        dict(inference_steps=65),
        dict(inference_steps=True),
        dict(inference_learning_rate=0.0),
        dict(learning_rate=float("nan")),
        dict(class_weights=(1.0, 0.0, 1.0)),
        dict(class_weights=[1.0, 1.0, 1.0]),
        dict(policy=None),
        dict(policy=object()),
        dict(policy=policy(min_steps=4)),
    ],
)
def test_should_reject_invalid_role_policy_weights_or_rates_before_proposals(torch, changes):
    features, targets, parameters = fixture(torch, torch.float64)
    before = tuple(value.clone() for value in (features, targets, *parameters))
    with pytest.raises(ValueError):
        invoke(features, targets, parameters, **changes)
    for actual, expected in zip((features, targets, *parameters), before, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "name,value",
    [
        ("min_steps", True),
        ("min_steps", 0),
        ("gradient_tolerance", float("nan")),
        ("relative_energy_tolerance", -1.0),
    ],
)
def test_should_revalidate_mutated_policy_values_before_any_numeric_work(torch, name, value):
    features, targets, parameters = fixture(torch, torch.float64)
    settings = policy()
    object.__setattr__(settings, name, value)
    with pytest.raises(ValueError):
        invoke(features, targets, parameters, policy=settings)


def test_should_return_detached_independent_outputs_and_preserve_donor_gradients_rng_and_mode(
    torch,
):
    features, targets, parameters = fixture(torch, torch.float64)
    features.requires_grad_()
    for parameter in parameters:
        parameter.requires_grad_()
        parameter.grad = torch.ones_like(parameter)
    donors = (features, targets, *parameters)
    before = tuple(value.detach().clone() for value in donors)
    gradients = tuple(parameter.grad for parameter in parameters)
    torch_rng, numpy_rng, python_rng = (
        torch.get_rng_state().clone(),
        pickle.dumps(np.random.get_state()),
        random.getstate(),
    )
    with torch.no_grad():
        result = invoke(features, targets, parameters)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    outputs = (*result.update.parameters, *result.update.gradients, result.update.hidden_state)
    assert len({value.data_ptr() for value in outputs}) == len(outputs)
    for output in outputs:
        assert output.grad_fn is None and not output.requires_grad
        assert all(output.data_ptr() != donor.data_ptr() for donor in donors)
        output.add_(10)
    for actual, expected in zip(donors, before, strict=True):
        assert torch.equal(actual.detach(), expected)
    assert all(
        parameter.grad is original
        for parameter, original in zip(parameters, gradients, strict=True)
    )
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in parameters)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert numpy_rng == pickle.dumps(np.random.get_state()) and python_rng == random.getstate()


def test_should_charge_failed_initial_prior_work_without_mutating_donors(torch):
    features, targets, parameters = fixture(torch, torch.float64)
    features.fill_(1e308)
    parameters[0].fill_(1.0)
    before = tuple(value.clone() for value in (features, targets, *parameters))
    with pytest.raises(TrainingUpdateFailed) as caught:
        invoke(features, targets, parameters)
    work = caught.value.work
    assert work["initial_prior_evaluations_attempted"] == 1
    assert work["initial_prior_evaluations_completed"] == 0
    assert work["attempted_steps"] == 0 and work["parameter_proposals_attempted"] == 0
    assert work["failure_work_may_be_unobserved"] is False
    work["attempted_steps"] = 999
    assert caught.value.work["attempted_steps"] == 0
    for actual, expected in zip((features, targets, *parameters), before, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("fault", ["second_candidate_energy", "first_candidate_direction"])
def test_should_keep_last_fully_qualified_state_when_later_numeric_evaluation_fails(
    torch, monkeypatch, fault
):
    features, targets, parameters = fixture(torch, torch.float64)
    helper = "_energy" if fault == "second_candidate_energy" else "_direction"
    original = getattr(bounded, helper)
    calls = []
    stop = 3 if fault == "second_candidate_energy" else 2

    def fail(*args, **options):
        calls.append("attempt")
        if len(calls) == stop:
            raise FloatingPointError("deliberate candidate numeric failure")
        return original(*args, **options)

    monkeypatch.setattr(bounded, helper, fail)
    result = invoke(features, targets, parameters)
    diagnostic = result.diagnostics
    accepted = 1 if fault == "second_candidate_energy" else 0
    assert len(calls) == stop
    assert diagnostic.reason == "numerical_failure" and diagnostic.used_fallback is True
    assert diagnostic.attempted_steps == accepted + 1
    assert diagnostic.accepted_steps == accepted and diagnostic.rejected_steps == 1
    assert diagnostic.gradient_norm_evaluations == accepted + 1
    assert result.update.nominal_latent_row_iterations == 3 * accepted
    assert diagnostic.failure_work_may_be_unobserved is False
    prior = (features @ parameters[0] + parameters[1]).tanh()
    if accepted:
        variable = prior.detach().requires_grad_()
        direction = (
            3
            * torch.autograd.grad(
                objective(torch, features, targets, parameters, variable), variable
            )[0]
        )
        expected = (prior - 0.04 * direction).detach()
    else:
        expected = prior
    torch.testing.assert_close(result.update.hidden_state, expected, rtol=2e-12, atol=2e-12)
    if fault == "second_candidate_energy":
        assert diagnostic.energy_evaluations_attempted == 3
        assert diagnostic.energy_evaluations_completed == 2
        assert diagnostic.energy_comparisons == 1
    else:
        assert (
            diagnostic.energy_evaluations_attempted == diagnostic.energy_evaluations_completed == 2
        )
        assert diagnostic.latent_gradient_evaluations_attempted == 2
        assert diagnostic.latent_gradient_evaluations_completed == 1
        assert diagnostic.energy_comparisons == 1


def test_should_charge_unexpected_initial_port_failure_as_unknown_and_publish_no_candidate(
    torch, monkeypatch
):
    features, targets, parameters = fixture(torch, torch.float64)

    def lost(*args, **options):
        raise OSError("initial energy port interrupted")

    monkeypatch.setattr(bounded, "_energy", lost)
    with pytest.raises(TrainingUpdateFailed) as caught:
        invoke(features, targets, parameters)
    work = caught.value.work
    assert work["energy_evaluations_attempted"] == 1
    assert work["energy_evaluations_completed"] == 0
    assert work["initial_prior_evaluations_completed"] == 1
    assert work["attempted_steps"] == 0 and work["parameter_proposals_attempted"] == 0
    assert work["failure_work_may_be_unobserved"] is True


def test_should_retain_all_returned_finalization_work_when_parameter_candidate_overflows(torch):
    features, targets, parameters = fixture(torch, torch.float64)
    features.fill_(1e100)
    parameters[0].fill_(1e-100)
    donors = (features, targets, *parameters)
    before = tuple(value.clone() for value in donors)
    with pytest.raises(TrainingUpdateFailed) as caught:
        invoke(features, targets, parameters, learning_rate=1e308, inference_steps=1)
    work = caught.value.work
    assert work["attempted_steps"] == work["accepted_steps"] == 1
    for name in (
        "parameter_gradient_evaluations",
        "parameter_proposals",
        "final_energy_evaluations",
        "legacy_diagnostic_evaluations",
    ):
        assert work[name + "_attempted"] == work[name + "_completed"] == 1
    assert work["failure_work_may_be_unobserved"] is False
    for actual, expected in zip(donors, before, strict=True):
        assert torch.equal(actual, expected)
