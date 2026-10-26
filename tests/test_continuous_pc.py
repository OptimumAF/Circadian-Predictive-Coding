"""Independent continuous-PC calculus oracles; fixed tiny arrays and no RNG.

Central differences cover all97 latent/parameter coordinates at depths1/2/4.
Neither the scalar NumPy oracle nor the three optional CPU Torch graphs calls a
production helper. No model constructor, fitting, optimizer or simulator runs.
Per full module with Torch: 82 PC entry attempts (66 evaluate, nine feedforward,
seven predict), two BC calls and two expected legacy-binary refusals; 197 oracle
energy probes, three CPU autodiff graphs and zero candidate-update proposals.
"""

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pytest

from src.core import continuous_bc as bc
from src.core import continuous_pc as pc
from src.core import coupled_energy as binary
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.coupled_energy import CoupledParameters


EPSILON, ATOL, RTOL = 1e-6, 2e-8, 2e-6


def problem(depth=1):
    parameters = CoupledParameters(
        tuple(
            np.array([[0.12 + 0.01 * layer, -0.08], [0.05, -0.17 + 0.02 * layer]], dtype=np.float64)
            for layer in range(depth)
        ),
        tuple(np.array([[0.03, -0.02]], dtype=np.float64) * (layer + 1) for layer in range(depth)),
        np.array([[0.2, -0.15, 0.3], [-0.1, 0.25, -0.2]], dtype=np.float64),
        np.array([[0.01, -0.04, 0.07]], dtype=np.float64),
    )
    features = np.array([[0.2, -0.3], [0.7, 0.4]], dtype=np.float64)
    targets = np.array([[1.0, 5.0, 1.5], [4.0, 7.0, 1.5]], dtype=np.float64)
    latents = tuple(
        np.array([[0.08 * (layer + 1), -0.05], [0.12, -0.09 * (layer + 1)]], dtype=np.float64)
        for layer in range(depth)
    )
    precisions = tuple(0.4 + 0.15 * layer for layer in range(depth))
    spec = ContinuousActionSpec(2, ("rad/s", "N", "fixed_m"), (-2.0, 4.0, 1.5), (6.0, 8.0, 1.5))
    return parameters, features, targets, latents, precisions, spec


def parameter_arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def oracle(parameters, features, targets, latents, precisions, spec):
    """Write E directly, without any production forward or geometry helper."""
    previous, penalties = features, []
    for weight, bias, latent, precision in zip(
        parameters.hidden_weights, parameters.hidden_biases, latents, precisions, strict=True
    ):
        residual = latent - np.tanh(previous @ weight + bias)
        penalties.append(float(precision * np.square(residual).sum() / (2 * len(features))))
        previous = latent
    output = np.tanh(latents[-1] @ parameters.output_weight + parameters.output_bias)
    lower, upper = np.asarray(spec.lower_bounds), np.asarray(spec.upper_bounds)
    active = upper > lower
    center, half = (upper + lower) / 2, (upper - lower) / 2
    normalized_targets = (targets[:, active] - center[active]) / half[active]
    task = (
        0.0
        if not active.any()
        else float(
            np.square(output[:, active] - normalized_targets).sum()
            / (2 * len(features) * active.sum())
        )
    )
    physical = lower + (output + 1) * (upper - lower) / 2
    return task + sum(penalties), task, tuple(penalties), physical


def difference(array, energy):
    result = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = float(array[index])
        try:
            array[index] = original + EPSILON
            plus = energy()
            array[index] = original - EPSILON
            minus = energy()
        finally:
            array[index] = original
        result[index] = (plus - minus) / (2 * EPSILON)
    return result


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_should_match_handwritten_energy_and_every_partial_central_difference(depth):
    params, x, y, z, beta, spec = problem(depth)
    protected = (*parameter_arrays(params), x, y, *z)
    original = tuple(value.copy() for value in protected)
    actual = pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    energy, task, penalties, predictions = oracle(params, x, y, z, beta, spec)
    assert (
        actual.energy_definition
        == pc.CONTINUOUS_PC_ENERGY_ID
        == "numpy_coupled_continuous_energy_v1"
    )
    assert actual.task_definition == bc.OBJECTIVE
    assert actual.energy == pytest.approx(energy, abs=1e-14, rel=1e-14)
    assert actual.task_loss == pytest.approx(task, abs=1e-14, rel=1e-14)
    np.testing.assert_allclose(actual.hidden_penalties, penalties, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(actual.predictions, predictions, atol=1e-14, rtol=1e-14)
    assert all(value > 0.0 for value in actual.hidden_penalties)
    if depth > 1:
        local = (
            beta[0]
            * (z[0] - np.tanh(x @ params.hidden_weights[0] + params.hidden_biases[0]))
            / len(x)
        )
        assert np.linalg.norm(actual.latent_gradients[0] - local) > 1e-5

    def scalar():
        return oracle(params, x, y, z, beta, spec)[0]

    for value, gradient in zip(
        (*z, *parameter_arrays(params)),
        (*actual.latent_gradients, *parameter_arrays(actual.parameter_gradients)),
        strict=True,
    ):
        np.testing.assert_allclose(gradient, difference(value, scalar), atol=ATOL, rtol=RTOL)
    for value, before in zip(protected, original, strict=True):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_should_match_an_independent_cpu_float64_torch_graph(depth):
    torch = pytest.importorskip("torch")
    params, x, y, z, beta, spec = problem(depth)
    tensors = [
        torch.tensor(value, dtype=torch.float64, device="cpu", requires_grad=True)
        for value in parameter_arrays(params)
    ]
    free = [
        torch.tensor(value, dtype=torch.float64, device="cpu", requires_grad=True) for value in z
    ]
    weights, biases = tensors[:depth], tensors[depth : 2 * depth]
    previous = torch.tensor(x, dtype=torch.float64, device="cpu")
    penalties = []
    for weight, bias, latent, precision in zip(weights, biases, free, beta, strict=True):
        residual = latent - torch.tanh(previous @ weight + bias)
        penalties.append(precision * (residual * residual).sum() / (2 * len(x)))
        previous = latent
    normalized = torch.tanh(free[-1] @ tensors[-2] + tensors[-1])
    labels = torch.tensor(y[:, :2], dtype=torch.float64, device="cpu")
    center = torch.tensor([2.0, 6.0], dtype=torch.float64, device="cpu")
    half = torch.tensor([4.0, 2.0], dtype=torch.float64, device="cpu")
    task = ((normalized[:, :2] - (labels - center) / half) ** 2).sum() / (2 * len(x) * 2)
    energy = task + torch.stack(penalties).sum()
    energy.backward()
    actual = pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    assert actual.energy == pytest.approx(float(energy.detach()), abs=1e-14, rel=1e-14)
    assert actual.task_loss == pytest.approx(float(task.detach()), abs=1e-14, rel=1e-14)
    np.testing.assert_allclose(
        actual.hidden_penalties,
        [float(value.detach()) for value in penalties],
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        actual.predictions[:, :2],
        (center + half * normalized[:, :2]).detach().numpy(),
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_array_equal(actual.predictions[:, 2], [1.5, 1.5])
    for gradient, tensor in zip(
        (*actual.latent_gradients, *parameter_arrays(actual.parameter_gradients)),
        (*free, *tensors),
        strict=True,
    ):
        assert (
            tensor.device.type == "cpu"
            and tensor.dtype == torch.float64
            and tensor.grad is not None
        )
        np.testing.assert_allclose(gradient, tensor.grad.detach().numpy(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("precision", [1e-6, 2, 1e6])
def test_should_sum_hidden_neurons_and_average_only_rows_at_native_precision_endpoints(precision):
    spec = ContinuousActionSpec(2, ("rad", "m"), (-1.0, -1.0), (1.0, 1.0))
    params = CoupledParameters(
        (np.zeros((2, 2)),), (np.zeros((1, 2)),), np.zeros((2, 2)), np.zeros((1, 2))
    )
    z = (np.array([[1.0, 2.0], [3.0, 4.0]]),)
    result = pc.evaluate_continuous_pc(
        params, np.zeros((2, 2)), np.zeros((2, 2)), z, (precision,), spec
    )
    assert result.task_loss == 0.0
    assert result.energy == result.hidden_penalties[0]
    assert result.energy == pytest.approx(7.5 * precision, abs=1e-14, rel=1e-14)
    np.testing.assert_allclose(
        result.latent_gradients[0], precision * z[0] / 2, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_array_equal(result.parameter_gradients.hidden_weights[0], np.zeros((2, 2)))
    np.testing.assert_allclose(
        result.parameter_gradients.hidden_biases[0],
        [[-2.0 * precision, -3.0 * precision]],
        atol=1e-14,
        rtol=1e-14,
    )
    assert result.precisions == (float(precision),) and type(result.precisions[0]) is float


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_should_keep_energy_and_parameter_gradients_when_batch_is_duplicated(depth):
    params, x, y, z, beta, spec = problem(depth)
    result = pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    repeated = pc.evaluate_continuous_pc(
        params,
        np.tile(x, (2, 1)),
        np.tile(y, (2, 1)),
        tuple(np.tile(value, (2, 1)) for value in z),
        beta,
        spec,
    )
    assert repeated.energy == pytest.approx(result.energy, abs=1e-14, rel=1e-14)
    for first, second in zip(
        parameter_arrays(result.parameter_gradients),
        parameter_arrays(repeated.parameter_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(second, first, atol=1e-14, rtol=1e-14)
    for first, second in zip(result.latent_gradients, repeated.latent_gradients, strict=True):
        np.testing.assert_allclose(second, np.tile(first / 2, (2, 1)), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("robot", [False, True])
def test_should_match_bc_task_predictions_and_output_head_but_not_hidden_bp_partials(robot):
    params, x, y, _, beta, spec = problem()
    if robot:
        spec = ContinuousActionSpec(
            2,
            (
                "normalized_root_dx",
                "normalized_root_dy",
                "normalized_root_dz",
                "normalized_gripper",
            ),
            (-1.0,) * 4,
            (1.0,) * 4,
        )
        params = replace(
            params,
            output_weight=np.array([[0.2, -0.15, 0.3, -0.2], [-0.1, 0.25, -0.2, 0.1]]),
            output_bias=np.array([[0.01, -0.04, 0.07, 0.03]]),
        )
        y = np.array([[-0.5, 0.25, 0.75, -1.0], [0.4, -0.6, 0.2, 1.0]])
    latents = pc.feedforward_continuous_latents(params, x, spec)
    prediction = pc.predict_continuous_pc(params, x, spec)
    actual = pc.evaluate_continuous_pc(params, x, y, latents, beta, spec)
    bc_parameters = (
        params.hidden_weights[0],
        params.hidden_biases[0][0],
        params.output_weight,
        params.output_bias[0],
    )
    expected = bc.continuous_bc_loss_and_gradients(x, y, bc_parameters, spec)
    np.testing.assert_array_equal(prediction, expected.predictions)
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
    assert (
        actual.task_loss == expected.loss
        and actual.energy == actual.task_loss
        and actual.hidden_penalties == (0.0,)
    )
    np.testing.assert_array_equal(actual.parameter_gradients.output_weight, expected.gradients[2])
    np.testing.assert_array_equal(actual.parameter_gradients.output_bias[0], expected.gradients[3])
    np.testing.assert_array_equal(
        actual.parameter_gradients.hidden_weights[0], np.zeros_like(params.hidden_weights[0])
    )
    assert (
        np.linalg.norm(expected.gradients[0]) > 1e-6
        and np.linalg.norm(actual.latent_gradients[0]) > 1e-6
    )


def test_should_ignore_constant_axes_without_disabling_hidden_penalties():
    params, x, _, z, beta, spec = problem()
    spec = replace(spec, lower_bounds=(2.0, 6.0, 1.5), upper_bounds=(2.0, 6.0, 1.5))
    y = np.tile(np.array(spec.lower_bounds), (2, 1))
    result = pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    assert result.task_loss == 0.0 and result.energy == sum(result.hidden_penalties) > 0.0
    np.testing.assert_array_equal(result.predictions, y)
    np.testing.assert_array_equal(
        result.parameter_gradients.output_weight, np.zeros_like(params.output_weight)
    )
    np.testing.assert_array_equal(
        result.parameter_gradients.output_bias, np.zeros_like(params.output_bias)
    )
    assert np.linalg.norm(result.latent_gradients[0]) > 0.0


def test_should_average_active_axes_and_exclude_a_constant_third_axis():
    params, x, y, z, beta, spec = problem()
    baseline = pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    reduced = replace(
        params,
        output_weight=params.output_weight[:, :2].copy(),
        output_bias=params.output_bias[:, :2].copy(),
    )
    small_spec = ContinuousActionSpec(
        2, spec.action_units[:2], spec.lower_bounds[:2], spec.upper_bounds[:2]
    )
    result = pc.evaluate_continuous_pc(reduced, x, y[:, :2].copy(), z, beta, small_spec)
    assert baseline.energy == pytest.approx(result.energy, abs=1e-14, rel=1e-14)
    assert baseline.task_loss == pytest.approx(result.task_loss, abs=1e-14, rel=1e-14)
    np.testing.assert_allclose(
        baseline.parameter_gradients.output_weight[:, :2],
        result.parameter_gradients.output_weight,
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_array_equal(baseline.parameter_gradients.output_weight[:, 2], [0.0, 0.0])


def test_should_return_owned_arrays_without_changing_inputs_or_retaining_result_aliases():
    params, x, y, z, beta, spec = problem(2)
    inputs = (*parameter_arrays(params), x, y, *z)
    before = tuple(value.copy() for value in inputs)
    hidden = pc.feedforward_continuous_latents(params, x, spec)
    prediction = pc.predict_continuous_pc(params, x, spec)
    result = pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    outputs = (
        *hidden,
        prediction,
        result.predictions,
        *result.latent_gradients,
        *parameter_arrays(result.parameter_gradients),
    )
    assert all(type(value) is np.ndarray and value.dtype == np.float64 for value in outputs)
    for value in outputs:
        assert all(not np.shares_memory(value, source) for source in inputs)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "energy", 0.0)
    for value in outputs:
        value.fill(99.0)
    for value, original in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(value, original)
    again = pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    assert again.energy == result.energy and not np.any(again.predictions == 99.0)


@pytest.mark.parametrize(
    "precision", [True, np.float64(1.0), 0.0, 1e-7, 1e7, float("nan"), float("inf"), "1"]
)
def test_should_refuse_non_native_or_out_of_range_precisions(precision: Any):
    params, x, y, z, _, spec = problem()
    with pytest.raises(ValueError):
        pc.evaluate_continuous_pc(params, x, y, z, (precision,), spec)


@pytest.mark.parametrize(
    "kind",
    [
        "parameter_record",
        "weight_list",
        "no_layers",
        "deep",
        "wide",
        "flat_bias",
        "bias_count",
        "wrong_output",
        "output_float32",
        "weight_nonfinite",
        "feature_float32",
        "feature_subclass",
        "feature_rows",
        "feature_empty",
        "feature_width",
        "target_shape",
        "target_bound",
        "inactive_target_bound",
        "target_nonfinite",
        "latent_list",
        "latent_shape",
        "latent_float32",
        "latent_nonfinite",
        "precision_list",
        "precision_count",
        "spec_type",
    ],
)
def test_should_refuse_invalid_problem_before_any_numeric_result(kind):
    class Foreign(np.ndarray):
        pass

    params, x, y, z, beta, spec = problem()
    if kind == "parameter_record":
        params = None
    elif kind == "weight_list":
        params = replace(params, hidden_weights=list(params.hidden_weights))
    elif kind == "no_layers":
        params = replace(params, hidden_weights=(), hidden_biases=())
    elif kind == "deep":
        params = replace(
            params,
            hidden_weights=params.hidden_weights * 17,
            hidden_biases=params.hidden_biases * 17,
        )
    elif kind == "wide":
        params = replace(
            params,
            hidden_weights=(np.zeros((2, 257)),),
            hidden_biases=(np.zeros((1, 257)),),
            output_weight=np.zeros((257, 3)),
        )
    elif kind == "flat_bias":
        params = replace(params, hidden_biases=(params.hidden_biases[0][0].copy(),))
    elif kind == "bias_count":
        params = replace(params, hidden_biases=())
    elif kind == "wrong_output":
        params = replace(params, output_weight=np.zeros((2, 1)))
    elif kind == "output_float32":
        params = replace(params, output_bias=params.output_bias.astype(np.float32))
    elif kind == "weight_nonfinite":
        params.hidden_weights[0][0, 0] = np.inf
    elif kind == "feature_float32":
        x = x.astype(np.float32)
    elif kind == "feature_subclass":
        x = x.view(Foreign)
    elif kind == "feature_rows":
        x = np.zeros((257, 2))
    elif kind == "feature_empty":
        x = np.zeros((0, 2))
    elif kind == "feature_width":
        x = np.zeros((2, 1))
    elif kind == "target_shape":
        y = y[:, :2].copy()
    elif kind == "target_bound":
        y[0, 0] = 7.0
    elif kind == "inactive_target_bound":
        y[0, 2] = 1.6
    elif kind == "target_nonfinite":
        y[0, 0] = np.nan
    elif kind == "latent_list":
        z = list(z)
    elif kind == "latent_shape":
        z = (np.zeros((1, 2)),)
    elif kind == "latent_float32":
        z = (z[0].astype(np.float32),)
    elif kind == "latent_nonfinite":
        z[0][0, 0] = np.nan
    elif kind == "precision_list":
        beta = list(beta)
    elif kind == "precision_count":
        beta = ()
    else:
        spec = None
    with pytest.raises(ValueError):
        pc.evaluate_continuous_pc(params, x, y, z, beta, spec)


@pytest.mark.parametrize("dimension", ["depth", "width", "batch", "input", "actions"])
def test_should_accept_each_declared_upper_dimension_with_other_dimensions_small(dimension):
    depth, width, rows, inputs, actions = dict(
        depth=(16, 1, 1, 1, 1),
        width=(1, 256, 1, 1, 1),
        batch=(1, 1, 256, 1, 1),
        input=(1, 1, 1, 1024, 1),
        actions=(1, 1, 1, 1, 16),
    )[dimension]
    spec = ContinuousActionSpec(inputs, ("unit",) * actions, (-1.0,) * actions, (1.0,) * actions)
    weights = (np.zeros((inputs, width)),) + tuple(
        np.zeros((width, width)) for _ in range(depth - 1)
    )
    params = CoupledParameters(
        weights,
        tuple(np.zeros((1, width)) for _ in range(depth)),
        np.zeros((width, actions)),
        np.zeros((1, actions)),
    )
    x = np.zeros((rows, inputs))
    latents = pc.feedforward_continuous_latents(params, x, spec)
    result = pc.evaluate_continuous_pc(
        params, x, np.zeros((rows, actions)), latents, (1.0,) * depth, spec
    )
    assert result.energy == 0.0 and result.predictions.shape == (rows, actions)


def test_should_preserve_exact_saturated_endpoints_and_finite_extreme_bound_geometry():
    spec = ContinuousActionSpec(1, ("large", "m"), (-1e308, 4.0), (1e308, 8.0))
    params = CoupledParameters(
        (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.zeros((1, 2)), np.array([[1e308, -1e308]])
    )
    x, z, y = np.zeros((1, 1)), (np.zeros((1, 1)),), np.array([[-1e308, 8.0]])
    result = pc.evaluate_continuous_pc(params, x, y, z, (1.0,), spec)
    np.testing.assert_array_equal(result.predictions, [[1e308, 4.0]])
    np.testing.assert_array_equal(pc.predict_continuous_pc(params, x, spec), result.predictions)
    assert result.task_loss == result.energy == 2.0
    assert all(np.all(value == 0.0) for value in parameter_arrays(result.parameter_gradients))


@pytest.mark.parametrize("where", ["hidden_forward", "output_forward", "residual_square"])
def test_should_fail_loudly_on_unrepresentable_arithmetic_without_changing_inputs(where):
    params, x, y, z, beta, spec = problem()
    if where == "hidden_forward":
        x.fill(1e308)
        params.hidden_weights[0].fill(1e308)
    elif where == "output_forward":
        params.hidden_weights[0].fill(0.0)
        params.hidden_biases[0].fill(1000.0)
        z[0].fill(1.0)
        params.output_weight.fill(1e308)
    else:
        z[0].fill(1e308)
        params.output_weight.fill(0.0)
    before = tuple(value.copy() for value in (*parameter_arrays(params), x, y, *z))
    with pytest.raises(FloatingPointError):
        pc.evaluate_continuous_pc(params, x, y, z, beta, spec)
    for value, original in zip((*parameter_arrays(params), x, y, *z), before, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("operation", ["feedforward", "predict_hidden", "predict_output"])
def test_should_refuse_overflow_before_tanh_can_hide_a_nonfinite_preactivation(operation):
    params, x, _, _, _, spec = problem()
    if operation == "predict_output":
        params.hidden_weights[0].fill(0.0)
        params.hidden_biases[0].fill(1000.0)
        params.output_weight.fill(1e308)
    else:
        x.fill(1e308)
        params.hidden_weights[0].fill(1e308)
    function = (
        pc.feedforward_continuous_latents
        if operation == "feedforward"
        else pc.predict_continuous_pc
    )
    with pytest.raises(FloatingPointError):
        function(params, x, spec)


def test_should_refuse_unrepresentable_nonconstant_halfwidth_instead_of_clipping_it():
    spec = ContinuousActionSpec(1, ("tiny",), (0.0,), (5e-324,))
    params = CoupledParameters(
        (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.zeros((1, 1)), np.zeros((1, 1))
    )
    with pytest.raises(ValueError):
        pc.predict_continuous_pc(params, np.zeros((1, 1)), spec)


def test_should_keep_legacy_binary_multiaxis_refusal_and_exclude_binary_gripper_objective():
    params, x, y, z, beta, spec = problem()
    with pytest.raises(ValueError):
        binary.evaluate_coupled(params, x, y[:, :1].copy(), z, beta)
    with pytest.raises(ValueError):
        binary.predict_coupled(params, x)
    with pytest.raises(TypeError):
        cast(Any, pc.evaluate_continuous_pc)(params, x, y, z, beta, spec, binary_gripper_axis=0)
