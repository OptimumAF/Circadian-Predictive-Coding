"""Independent weighted PC mathematics and owned-candidate engineering gates.

Small literal TRAIN fixtures check partial derivatives at the detached final
latent state. They do not assess acquisition, retention, or circadian benefit.
"""

from dataclasses import FrozenInstanceError
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters.torch_feature_learner import TorchFeaturePolicy, _components
from src.core import categorical_pc_weighted as weighted
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from src.shared.torch_runtime import require_torch


NAMES = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
WEIGHTS = (0.25, 1.5, 3.0)


@pytest.fixture(scope="module")
def torch():
    return require_torch()


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        primitive_attempts=0,
        primitive_successes=0,
        primitive_rows=0,
        primitive_nominal_latent_row_iterations=0,
        native_calls=0,
        native_rows=0,
        native_nominal_latent_row_iterations=0,
    )
    original = weighted.class_weighted_pc_update

    def invoke(features, targets, parameters, **kwargs):
        work["primitive_attempts"] += 1
        result = original(features, targets, parameters, **kwargs)
        work["primitive_successes"] += 1
        work["primitive_rows"] += len(features)
        work["primitive_nominal_latent_row_iterations"] += result.nominal_latent_row_iterations
        return result

    monkeypatch.setattr(weighted, "class_weighted_pc_update", invoke)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        train = kind.train_step

        def count_native(*args, _train=train, **kwargs):
            work["native_calls"] += 1
            work["native_rows"] += len(args[1])
            work["native_nominal_latent_row_iterations"] += len(args[1]) * args[4]
            return _train(*args, **kwargs)

        monkeypatch.setattr(kind, "train_step", count_native)
    yield
    for name, value in work.items():
        record_property(name, value)


def fixture(torch, dtype):
    features = torch.tensor([[0.2, -0.3], [-0.7, 0.4], [0.5, 0.8]], dtype=dtype)
    targets = torch.tensor([0, 0, 2], dtype=torch.int64)
    parameters = (
        torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=dtype),
        torch.tensor([[0.04, -0.07]], dtype=dtype),
        torch.tensor([[0.2, -0.4, 0.3], [-0.1, 0.5, 0.2]], dtype=dtype),
        torch.tensor([[0.03, -0.02, 0.05]], dtype=dtype),
    )
    return features, targets, parameters


def invoke(features, targets, parameters, **changes):
    options: dict[str, Any] = dict(
        class_weights=WEIGHTS,
        learning_rate=0.03,
        inference_steps=3,
        inference_learning_rate=0.04,
        role="train",
    )
    options.update(changes)
    return weighted.class_weighted_pc_update(features, targets, parameters, **options)


def objective(torch, features, targets, parameters, hidden, weights=WEIGHTS):
    """Use log probabilities directly, independent of production error formulas."""
    prior = (features @ parameters[0] + parameters[1]).tanh()
    logits = hidden @ parameters[2] + parameters[3]
    row_ce = -logits.log_softmax(dim=1)[torch.arange(len(targets)), targets]
    row_penalty = 0.5 * (hidden - prior).square().sum(dim=1)
    mass = torch.tensor(weights, dtype=features.dtype)[targets]
    return (mass * (row_ce + row_penalty)).sum() / mass.sum()


def autograd_relaxation(torch, features, targets, parameters, steps, weights=WEIGHTS):
    hidden = (features @ parameters[0] + parameters[1]).tanh().detach()
    for _ in range(steps):
        hidden.requires_grad_(True)
        gradient = torch.autograd.grad(
            objective(torch, features, targets, parameters, hidden, weights), hidden
        )[0]
        hidden = (hidden - 0.04 * len(targets) * gradient).detach()
    return hidden


@pytest.mark.parametrize("dtype_name", ("float32", "float64"))
def test_should_match_independent_autograd_at_detached_relaxed_state(torch, dtype_name):
    dtype = getattr(torch, dtype_name)
    features, targets, parameters = fixture(torch, dtype)
    result = invoke(features, targets, parameters)
    hidden = autograd_relaxation(torch, features, targets, parameters, 3)
    independent_parameters = tuple(value.clone().requires_grad_() for value in parameters)
    energy = objective(torch, features, targets, independent_parameters, hidden)
    gradients = torch.autograd.grad(energy, independent_parameters)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(result.hidden_state, hidden, rtol=tolerance, atol=tolerance)
    assert result.energy == pytest.approx(energy.item(), rel=tolerance, abs=tolerance)
    for old, actual, gradient, candidate in zip(
        parameters, result.gradients, gradients, result.parameters, strict=True
    ):
        torch.testing.assert_close(actual, gradient, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(candidate, old - 0.03 * gradient, rtol=tolerance, atol=tolerance)
    logits = hidden @ parameters[2] + parameters[3]
    one_hot = torch.nn.functional.one_hot(targets, num_classes=3).to(dtype)
    prior = (features @ parameters[0] + parameters[1]).tanh()
    legacy = 0.5 * (
        (logits.softmax(dim=1) - one_hot).square().mean() + (hidden - prior).square().mean()
    )
    assert result.legacy_diagnostic == pytest.approx(legacy.item(), rel=tolerance, abs=tolerance)
    assert result.nominal_latent_row_iterations == 9


def test_should_match_finite_differences_for_every_parameter_and_latent_coordinate(torch):
    features, targets, parameters = fixture(torch, torch.float64)
    result = invoke(features, targets, parameters)
    epsilon = 1e-6
    for parameter_index, (parameter, gradient) in enumerate(
        zip(parameters, result.gradients, strict=True)
    ):
        for coordinate in range(parameter.numel()):
            energies = []
            for direction in (-1.0, 1.0):
                changed = tuple(value.clone() for value in parameters)
                changed[parameter_index].view(-1)[coordinate] += direction * epsilon
                energies.append(
                    objective(torch, features, targets, changed, result.hidden_state).item()
                )
            numerical = (energies[1] - energies[0]) / (2 * epsilon)
            assert gradient.view(-1)[coordinate].item() == pytest.approx(numerical, abs=3e-9)
    prior = (features @ parameters[0] + parameters[1]).tanh()
    first = invoke(features, targets, parameters, inference_steps=1)
    for coordinate in range(prior.numel()):
        energies = []
        for direction in (-1.0, 1.0):
            changed_hidden = prior.clone()
            changed_hidden.view(-1)[coordinate] += direction * epsilon
            energies.append(objective(torch, features, targets, parameters, changed_hidden).item())
        numerical = len(targets) * (energies[1] - energies[0]) / (2 * epsilon)
        observed = (prior - first.hidden_state).view(-1)[coordinate].item() / 0.04
        assert observed == pytest.approx(numerical, abs=1e-8)


@pytest.mark.parametrize("method", ("pc", "neutral_cpc"))
@pytest.mark.parametrize("steps", (2, 64))
def test_should_reproduce_native_parameters_exactly_with_all_one_weights(torch, method, steps):
    policy = TorchFeaturePolicy(method, 2, 2, 17, num_classes=3)
    head, _ = _components(policy)
    features, targets, _ = fixture(torch, torch.float32)
    parameters = tuple(getattr(head, name).clone() for name in NAMES)
    result = invoke(features, targets, parameters, class_weights=(1.0,) * 3, inference_steps=steps)
    energy = head.train_step(features, targets, 0.03, steps, 0.04)
    assert result.legacy_diagnostic == energy
    for name, actual in zip(NAMES, result.parameters, strict=True):
        assert torch.equal(actual, getattr(head, name))


def test_should_preserve_pc_neutral_parameter_parity_with_nonuniform_weights(torch):
    features, targets, _ = fixture(torch, torch.float32)
    results = []
    for method in ("pc", "neutral_cpc"):
        head, _ = _components(TorchFeaturePolicy(method, 2, 2, 17, num_classes=3))
        results.append(invoke(features, targets, tuple(getattr(head, name) for name in NAMES)))
    assert results[0].energy == results[1].energy
    for left, right in zip(results[0].parameters, results[1].parameters, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize("weights", ((1.0, 17.0, 1.0), (2.0, 17.0, 2.0)))
def test_should_preserve_native_identity_when_only_absent_class_or_uniform_mass_changes(
    torch, weights
):
    head, _ = _components(TorchFeaturePolicy("pc", 2, 2, 17, num_classes=3))
    features, targets, _ = fixture(torch, torch.float32)
    assert 1 not in targets.tolist()
    result = invoke(
        features, targets, tuple(getattr(head, name) for name in NAMES), class_weights=weights
    )
    energy = head.train_step(features, targets, 0.03, 3, 0.04)
    assert result.legacy_diagnostic == energy
    for name, actual in zip(NAMES, result.parameters, strict=True):
        assert torch.equal(actual, getattr(head, name))


def test_should_make_global_class_weight_scale_irrelevant(torch):
    features, targets, parameters = fixture(torch, torch.float64)
    original = invoke(features, targets, parameters)
    scaled = invoke(
        features, targets, parameters, class_weights=tuple(8 * value for value in WEIGHTS)
    )
    assert original.energy == scaled.energy
    for left, right in zip(original.parameters, scaled.parameters, strict=True):
        assert torch.equal(left, right)
    assert torch.equal(original.hidden_state, scaled.hidden_state)


def test_should_return_owned_detached_candidates_without_touching_donor_or_rng(torch):
    features, targets, parameters = fixture(torch, torch.float64)
    for parameter in parameters:
        parameter.requires_grad_()
        parameter.grad = torch.ones_like(parameter)
    features.requires_grad_()
    donors = (features, targets, *parameters)
    before = tuple(value.detach().clone() for value in donors)
    gradients = tuple(parameter.grad for parameter in parameters)
    tensor_rng = torch.get_rng_state().clone()
    numpy_rng = pickle.dumps(np.random.get_state())
    python_rng = random.getstate()
    with torch.no_grad():
        result = invoke(features, targets, parameters)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert torch.equal(torch.get_rng_state(), tensor_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    assert random.getstate() == python_rng
    outputs = (*result.parameters, *result.gradients, result.hidden_state)
    for output in outputs:
        assert not output.requires_grad and output.grad_fn is None
        assert all(output.data_ptr() != donor.data_ptr() for donor in donors)
    assert len({value.data_ptr() for value in outputs}) == len(outputs)
    result.parameters[0].fill_(23)
    result.gradients[0].fill_(-23)
    result.hidden_state.fill_(42)
    for actual, expected in zip(donors, before, strict=True):
        assert torch.equal(actual, expected)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        assert parameter.requires_grad and parameter.grad is gradient
        assert torch.equal(parameter.grad, torch.ones_like(parameter))
    with pytest.raises(FrozenInstanceError):
        setattr(result, "energy", 0.0)


@pytest.mark.parametrize(
    "change",
    (
        "empty",
        "too_many_rows",
        "feature_width",
        "integer_features",
        "feature_nan",
        "float_targets",
        "target_shape",
        "negative_target",
        "large_target",
        "parameter_list",
        "missing_parameter",
        "bias_shape",
        "parameter_dtype",
        "parameter_nan",
        "weight_list",
        "zero_weight",
        "nan_weight",
        "weight_count",
        "role",
        "bool_steps",
        "too_many_steps",
        "zero_steps",
        "nan_lr",
        "zero_lr",
        "huge_integer_lr",
        "infinite_inference_lr",
    ),
)
def test_should_refuse_invalid_or_nontrain_inputs_without_mutation(torch, change):
    features, targets, parameters = fixture(torch, torch.float64)
    options: dict[str, Any] = {}
    if change == "empty":
        features, targets = features[:0], targets[:0]
    elif change == "too_many_rows":
        features, targets = features[:1].repeat(55, 1), targets[:1].repeat(55)
    elif change == "feature_width":
        features = features[:, :1]
    elif change == "integer_features":
        features = features.to(torch.int64)
    elif change == "feature_nan":
        features[0, 0] = float("nan")
    elif change == "float_targets":
        targets = targets.to(torch.float64)
    elif change == "target_shape":
        targets = targets[:, None]
    elif change == "negative_target":
        targets[0] = -1
    elif change == "large_target":
        targets[0] = 3
    elif change == "parameter_list":
        parameters = list(parameters)
    elif change == "missing_parameter":
        parameters = parameters[:3]
    elif change == "bias_shape":
        parameters = (parameters[0], parameters[1].reshape(2), *parameters[2:])
    elif change == "parameter_dtype":
        parameters = (*parameters[:3], parameters[3].to(torch.float32))
    elif change == "parameter_nan":
        parameters[0][0, 0] = float("nan")
    else:
        invalid_options: dict[str, dict[str, Any]] = {
            "weight_list": dict(class_weights=list(WEIGHTS)),
            "zero_weight": dict(class_weights=(0.0, 1.0, 1.0)),
            "nan_weight": dict(class_weights=(float("nan"), 1.0, 1.0)),
            "weight_count": dict(class_weights=(1.0, 1.0)),
            "role": dict(role="final"),
            "bool_steps": dict(inference_steps=True),
            "too_many_steps": dict(inference_steps=65),
            "zero_steps": dict(inference_steps=0),
            "nan_lr": dict(learning_rate=float("nan")),
            "zero_lr": dict(learning_rate=0.0),
            "huge_integer_lr": dict(learning_rate=10**1000),
            "infinite_inference_lr": dict(inference_learning_rate=float("inf")),
        }
        options = invalid_options[change]
    donors = (features, targets, *parameters)
    before = tuple(value.clone() for value in donors)
    with pytest.raises((ValueError, TypeError)):
        invoke(features, targets, parameters, **options)
    for actual, expected in zip(donors, before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_should_refuse_nonfinite_arithmetic_without_publishing_candidate(torch):
    features, targets, parameters = fixture(torch, torch.float32)
    features.fill_(1e30)
    parameters[0].fill_(1e30)
    donors = (features, targets, *parameters)
    before = tuple(value.clone() for value in donors)
    with pytest.raises(FloatingPointError):
        invoke(features, targets, parameters)
    for actual, expected in zip(donors, before, strict=True):
        assert torch.equal(actual, expected)
