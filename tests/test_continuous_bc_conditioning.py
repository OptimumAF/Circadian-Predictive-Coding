"""Independent activation/gradient conditioning oracles; synthetic TRAIN only."""

from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
import math

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as adapter
from src.core import continuous_bc as core
from src.core import continuous_bc_diagnostics as diagnostics
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization


def spec():
    return ContinuousActionSpec(
        2, ("m", "normalized_gripper", "fixed"), (-2.0, -1.0, 4.0), (6.0, 1.0, 4.0)
    )


def data():
    return np.asarray([[3.0, 4.0], [-3.0, -4.0]]), np.asarray([[-2.0, -1.0, 4.0], [6.0, 1.0, 4.0]])


def weights():
    return (
        np.asarray([[0.1, -0.2], [0.3, 0.15]]),
        np.asarray([0.02, -0.03]),
        np.asarray([[0.2, -0.1, 0.12], [-0.3, 0.25, -0.15]]),
        np.asarray([0.05, -0.04, 0.07]),
    )


def zero_weights():
    return tuple(np.zeros_like(value) for value in weights())


def episode(role="train"):
    features, targets = data()
    return ContinuousEpisode(
        "synthetic-conditioning",
        "synthetic",
        role,
        spec(),
        (
            ContinuousTransition(
                0,
                tuple(features[0].tolist()),
                tuple(targets[0].tolist()),
                0.0,
                tuple(features[1].tolist()),
                False,
                False,
            ),
            ContinuousTransition(
                1,
                tuple(features[1].tolist()),
                tuple(targets[1].tolist()),
                0.0,
                (0.0, 0.0),
                True,
                False,
            ),
        ),
    )


def scalar_loss(features, targets, values, *, binary=False):
    """Scalar-loop loss oracle independent of production kernels and reduction."""
    first, bias_first, second, bias_second = values
    total = 0.0
    for feature, target in zip(features.tolist(), targets.tolist(), strict=True):
        hidden = [
            math.tanh(sum(feature[d] * first[d, h] for d in range(2)) + bias_first[h])
            for h in range(2)
        ]
        logits = [
            sum(hidden[h] * second[h, axis] for h in range(2)) + bias_second[axis]
            for axis in range(3)
        ]
        position_error = math.tanh(logits[0]) - (target[0] - 2.0) / 4.0
        grip = (
            math.log1p(math.exp(-2 * target[1] * logits[1]))
            if binary
            else 0.5 * (math.tanh(logits[1]) - target[1]) ** 2
        )
        total += 0.5 * position_error**2 + grip
    return total / (2 * len(features))


def finite_difference(features, targets, values, *, binary=False):
    output = []
    for block, parameter in enumerate(values):
        result = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            plus, minus = (
                tuple(value.copy() for value in values),
                tuple(value.copy() for value in values),
            )
            plus[block][index] += 1e-6
            minus[block][index] -= 1e-6
            result[index] = (
                scalar_loss(features, targets, plus, binary=binary)
                - scalar_loss(features, targets, minus, binary=binary)
            ) / 2e-6
        output.append(result)
    return tuple(output)


def same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for name in ("parameters", "velocity"):
        for expected, observed in zip(getattr(first, name), getattr(second, name), strict=True):
            np.testing.assert_array_equal(expected, observed)


def learner(*, transform=None, binary=False, role="train"):
    owner = adapter.NumpyContinuousBCLearner(
        adapter.ContinuousBCPolicy(
            spec(),
            hidden_dim=2,
            learning_rate=0.03,
            momentum=0.8,
            training_role=role,
            observation_standardization=transform,
            binary_gripper_axis=1 if binary else None,
        )
    )
    original = owner.snapshot_state()
    owner.restore_state(
        replace(
            original,
            parameters=weights(),
            velocity=tuple(np.full_like(value, 0.125) for value in original.velocity),
            completed_updates=3,
            training_examples=6,
        )
    )
    return owner


def assert_plain_frozen(value):
    if is_dataclass(value):
        assert value.__dataclass_params__.frozen
        for field in fields(value):
            assert_plain_frozen(getattr(value, field.name))
    elif type(value) is tuple:
        for item in value:
            assert_plain_frozen(item)
    else:
        assert type(value) in (type(None), bool, int, float, str)
        if type(value) is float:
            assert math.isfinite(value)


def test_should_match_hand_zero_network_active_axis_norms_and_all_work_counts():
    features, targets = data()
    report = diagnostics.continuous_bc_conditioning(features, targets, zero_weights(), spec())
    assert report.rows == 2 and report.objective_definition == core.OBJECTIVE
    assert report.objective_loss == 0.5
    assert report.feature_magnitude.values == 4
    assert report.feature_magnitude.maximum_absolute == 4.0
    assert report.feature_magnitude.rms == pytest.approx(math.sqrt(12.5))
    assert report.feature_magnitude.zero_values == 0
    assert asdict(report.hidden_saturation) == dict(
        values=4,
        saturated_values=0,
        saturated_fraction=0.0,
        minimum_derivative=1.0,
        mean_derivative=1.0,
    )
    assert [(row.index, row.unit, row.active) for row in report.output_saturation] == [
        (0, "m", True),
        (1, "normalized_gripper", True),
        (2, "fixed", False),
    ]
    assert report.output_saturation[2].saturation is None
    for row in report.output_saturation[:2]:
        assert asdict(row.saturation) == dict(
            values=2,
            saturated_values=0,
            saturated_fraction=0.0,
            minimum_derivative=1.0,
            mean_derivative=1.0,
        )
    assert [row.name for row in report.gradients] == ["w1", "b1", "w2", "b2"]
    assert [row.values for row in report.gradients] == [4, 2, 6, 3]
    assert all(
        row.l2_norm == row.maximum_absolute == 0.0 and row.zero_values == row.values
        for row in report.gradients
    )
    assert asdict(report.work) == dict(
        forward_passes=2,
        backward_passes=1,
        input_rows=2,
        input_dimension=2,
        hidden_dimension=2,
        output_dimension=3,
        input_values=4,
        hidden_values=4,
        output_values=6,
        parameter_values=15,
        gradient_values=15,
        gradient_numeric_bytes=120,
        dense_multiply_accumulates=72,
    )


def test_should_match_hand_nonzero_derivatives_with_physical_asymmetric_bounds():
    declared = ContinuousActionSpec(1, ("N",), (10.0,), (14.0,))
    values = (
        np.asarray([[2.0]]),
        np.asarray([math.atanh(0.5)]),
        np.asarray([[1.0]]),
        np.asarray([math.atanh(0.25) - 0.5]),
    )
    report = diagnostics.continuous_bc_conditioning(
        np.zeros((1, 1)), np.asarray([[13.0]]), values, declared
    )
    assert report.objective_loss == pytest.approx(0.03125)
    assert report.hidden_saturation.minimum_derivative == pytest.approx(0.75)
    assert report.hidden_saturation.mean_derivative == pytest.approx(0.75)
    assert report.output_saturation[0].saturation.minimum_derivative == pytest.approx(0.9375)
    assert report.output_saturation[0].saturation.mean_derivative == pytest.approx(0.9375)
    expected = [0.0, 0.17578125, 0.1171875, 0.234375]
    for row, magnitude in zip(report.gradients, expected, strict=True):
        assert row.values == 1
        assert row.l2_norm == pytest.approx(magnitude)
        assert row.maximum_absolute == pytest.approx(magnitude)
        assert row.zero_values == int(magnitude == 0)


@pytest.mark.parametrize("binary", [False, True])
def test_should_match_every_block_independent_finite_difference_summary(binary):
    features, targets = data()
    values = weights()
    expected = finite_difference(features, targets, values, binary=binary)
    report = diagnostics.continuous_bc_conditioning(
        features, targets, values, spec(), binary_gripper_axis=1 if binary else None
    )
    assert report.objective_definition == (core.MIXED_OBJECTIVE if binary else core.OBJECTIVE)
    assert report.objective_loss == pytest.approx(
        scalar_loss(features, targets, values, binary=binary), abs=1e-14
    )
    for summary, gradient in zip(report.gradients, expected, strict=True):
        assert summary.values == gradient.size
        assert summary.l2_norm == pytest.approx(float(np.linalg.norm(gradient)), rel=2e-6, abs=1e-9)
        assert summary.maximum_absolute == pytest.approx(
            float(np.max(np.abs(gradient))), rel=2e-6, abs=1e-9
        )
    # The constant physical output's three parameter entries remain exact zero.
    assert report.gradients[2].zero_values == 2 and report.gradients[3].zero_values == 1


@pytest.mark.parametrize("logit", [-1000.0, 1000.0])
def test_should_keep_binary_logit_gradient_visible_despite_identical_saturated_tanh_readout(logit):
    declared = ContinuousActionSpec(1, ("normalized_gripper",), (-1.0,), (1.0,))
    values = (np.zeros((1, 1)), np.zeros(1), np.zeros((1, 1)), np.asarray([logit]))
    features, targets = np.zeros((1, 1)), np.asarray([[-math.copysign(1.0, logit)]])
    mse = diagnostics.continuous_bc_conditioning(features, targets, values, declared)
    binary = diagnostics.continuous_bc_conditioning(
        features, targets, values, declared, binary_gripper_axis=0
    )
    assert mse.output_saturation == binary.output_saturation
    saturation = binary.output_saturation[0].saturation
    assert saturation.saturated_values == 1 and saturation.saturated_fraction == 1.0
    assert saturation.minimum_derivative == saturation.mean_derivative == 0.0
    assert mse.objective_loss == 2.0 and binary.objective_loss == 2000.0
    assert mse.gradients[3].l2_norm == 0.0 and mse.gradients[3].zero_values == 1
    assert binary.gradients[3].l2_norm == binary.gradients[3].maximum_absolute == 2.0
    assert binary.gradients[3].zero_values == 0


def test_should_use_fixed_threshold_and_keep_duplicate_output_units_distinct():
    declared = ContinuousActionSpec(1, ("m", "m"), (-1.0, -1.0), (1.0, 1.0))
    logits = np.asarray([math.atanh(0.991), math.atanh(0.989)])
    values = (np.zeros((1, 2)), logits.copy(), np.zeros((2, 2)), logits.copy())
    report = diagnostics.continuous_bc_conditioning(
        np.zeros((1, 1)), np.zeros((1, 2)), values, declared
    )
    assert diagnostics.SATURATION_THRESHOLD == 0.99
    assert report.hidden_saturation.saturated_values == 1
    assert report.hidden_saturation.saturated_fraction == 0.5
    assert report.hidden_saturation.minimum_derivative == pytest.approx(1 - 0.991**2)
    assert report.hidden_saturation.mean_derivative == pytest.approx(1 - (0.991**2 + 0.989**2) / 2)
    assert [(row.index, row.unit) for row in report.output_saturation] == [(0, "m"), (1, "m")]
    assert report.output_saturation[0].saturation.saturated_values == 1
    assert report.output_saturation[1].saturation.saturated_values == 0


def test_should_report_no_active_output_saturation_and_zero_gradients_for_constant_bounds():
    declared = ContinuousActionSpec(1, ("fixed", "fixed"), (4.0, -3.0), (4.0, -3.0))
    values = (np.ones((1, 1)), np.zeros(1), np.ones((1, 2)), np.ones(2))
    report = diagnostics.continuous_bc_conditioning(
        np.asarray([[1.0]]), np.asarray([[4.0, -3.0]]), values, declared
    )
    assert report.objective_loss == 0.0
    assert all(not row.active and row.saturation is None for row in report.output_saturation)
    assert all(row.l2_norm == 0.0 and row.zero_values == row.values for row in report.gradients)


def test_should_keep_large_representable_rms_and_l2_finite_without_squaring_overflow():
    declared = ContinuousActionSpec(2, ("m",), (-1.0,), (1.0,))
    values = (np.zeros((2, 1)), np.zeros(1), np.ones((1, 1)), np.zeros(1))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        report = diagnostics.continuous_bc_conditioning(
            np.asarray([[1e308, -1e308]]), np.asarray([[-1.0]]), values, declared
        )
    assert report.feature_magnitude.maximum_absolute == report.feature_magnitude.rms == 1e308
    assert report.gradients[0].l2_norm == pytest.approx(math.sqrt(2) * 1e308)
    assert report.gradients[0].maximum_absolute == 1e308 and report.gradients[0].zero_values == 0
    assert_plain_frozen(report)


def test_should_refuse_unrepresentable_l2_after_finite_actual_gradients():
    declared = ContinuousActionSpec(3, ("m",), (-1.0,), (1.0,))
    values = (np.zeros((3, 1)), np.zeros(1), np.ones((1, 1)), np.zeros(1))
    with pytest.raises(FloatingPointError, match="L2|representable"):
        diagnostics.continuous_bc_conditioning(
            np.full((1, 3), 1.1e308), np.asarray([[-1.0]]), values, declared
        )


@pytest.mark.parametrize(
    "bad",
    [
        "features_dtype",
        "features_shape",
        "features_empty",
        "features_rows",
        "features_nonfinite",
        "targets_shape",
        "targets_bounds",
        "targets_nonfinite",
        "parameter_tuple",
        "parameter_shape",
        "parameter_dtype",
        "parameter_nonfinite",
        "hidden_width",
        "binary_type",
        "binary_label",
        "geometry",
    ],
)
def test_should_reject_invalid_metadata_arrays_and_objective_before_any_kernel(monkeypatch, bad):
    features, targets, values, declared, axis = *data(), weights(), spec(), None
    if bad == "features_dtype":
        features = features.astype(np.float32)
    elif bad == "features_shape":
        features = features[:, :1]
    elif bad == "features_empty":
        features, targets = features[:0], targets[:0]
    elif bad == "features_rows":
        features, targets = np.zeros((257, 2)), np.tile(targets[0], (257, 1))
    elif bad == "features_nonfinite":
        features[0, 0] = np.inf
    elif bad == "targets_shape":
        targets = targets[:, :2]
    elif bad == "targets_bounds":
        targets[0, 0] = 6.1
    elif bad == "targets_nonfinite":
        targets[0, 0] = np.nan
    elif bad == "parameter_tuple":
        values = list(values)
    elif bad == "parameter_shape":
        values = values[:3] + (np.zeros(2),)
    elif bad == "parameter_dtype":
        values = (values[0].astype(np.float32),) + values[1:]
    elif bad == "parameter_nonfinite":
        values[1][0] = np.inf
    elif bad == "hidden_width":
        values = (np.zeros((2, 257)), np.zeros(257), np.zeros((257, 3)), np.zeros(3))
    elif bad == "binary_type":
        axis = True
    elif bad == "binary_label":
        axis, targets[0, 1] = 1, 0.5
    else:
        declared = ContinuousActionSpec(2, ("m",), (0.0,), (5e-324,))
        targets = np.zeros((2, 1))
        values = (np.zeros((2, 1)), np.zeros(1), np.zeros((1, 1)), np.zeros(1))
    monkeypatch.setattr(
        core, "_forward", lambda *_a, **_k: pytest.fail("invalid input reached forward")
    )
    monkeypatch.setattr(
        core,
        "continuous_bc_loss_and_gradients",
        lambda *_a, **_k: pytest.fail("invalid input reached loss/backward"),
    )
    with pytest.raises(ValueError):
        diagnostics.continuous_bc_conditioning(
            features, targets, values, declared, binary_gripper_axis=axis
        )


def test_should_count_two_actual_forward_calls_and_one_actual_loss_backward(monkeypatch):
    features, targets = data()
    calls = []
    forward, loss_gradient = core._forward, core.continuous_bc_loss_and_gradients

    def counted_forward(*args, **kwargs):
        calls.append("forward")
        return forward(*args, **kwargs)

    def counted_gradient(*args, **kwargs):
        calls.append("loss_backward")
        return loss_gradient(*args, **kwargs)

    monkeypatch.setattr(core, "_forward", counted_forward)
    monkeypatch.setattr(core, "continuous_bc_loss_and_gradients", counted_gradient)
    report = diagnostics.continuous_bc_conditioning(features, targets, weights(), spec())
    assert calls == ["loss_backward", "forward", "forward"]
    assert report.work.forward_passes == 2 and report.work.backward_passes == 1


@pytest.mark.parametrize(
    "rows,dimension,hidden,axes,parameters,macs",
    [
        (256, 1, 1, 16, 34, 17152),
        (1, 1024, 1, 1, 1027, 3076),
        (1, 1, 256, 1, 769, 1792),
    ],
)
def test_should_support_declared_individual_size_limits_with_bounded_scalar_output(
    rows, dimension, hidden, axes, parameters, macs
):
    declared = ContinuousActionSpec(dimension, ("m",) * axes, (-1.0,) * axes, (1.0,) * axes)
    values = (
        np.zeros((dimension, hidden)),
        np.zeros(hidden),
        np.zeros((hidden, axes)),
        np.zeros(axes),
    )
    report = diagnostics.continuous_bc_conditioning(
        np.zeros((rows, dimension)), np.zeros((rows, axes)), values, declared
    )
    assert report.work.parameter_values == report.work.gradient_values == parameters
    assert report.work.gradient_numeric_bytes == parameters * 8
    assert report.work.dense_multiply_accumulates == macs
    assert report.work.input_values == rows * dimension
    assert report.work.hidden_values == rows * hidden and report.work.output_values == rows * axes
    assert len(report.output_saturation) == axes and len(report.gradients) == 4
    assert_plain_frozen(report)


def test_should_fail_finite_inputs_whose_real_forward_is_unrepresentable_without_mutation():
    declared = ContinuousActionSpec(1, ("m",), (-1.0,), (1.0,))
    features, targets = np.asarray([[1e308]]), np.zeros((1, 1))
    values = (np.asarray([[2.0]]), np.zeros(1), np.ones((1, 1)), np.zeros(1))
    originals = tuple(value.copy() for value in (features, targets, *values))
    with pytest.raises(FloatingPointError):
        diagnostics.continuous_bc_conditioning(features, targets, values, declared)
    for expected, observed in zip(originals, (features, targets, *values), strict=True):
        np.testing.assert_array_equal(expected, observed)


def test_should_return_owned_frozen_plain_summaries_without_mutating_input_payloads():
    features, targets, values = *data(), weights()
    original = [value.copy() for value in (features, targets, *values)]
    report = diagnostics.continuous_bc_conditioning(features, targets, values, spec())
    assert_plain_frozen(report)
    retained = asdict(report)
    for expected, value in zip(original, (features, targets, *values), strict=True):
        np.testing.assert_array_equal(expected, value)
        value.fill(99.0)
    assert asdict(report) == retained
    with pytest.raises(FrozenInstanceError):
        report.objective_loss = 99.0
    with pytest.raises(FrozenInstanceError):
        report.gradients[0].values = 99


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("binary", [False, True])
def test_should_standardize_once_call_real_kernels_and_preserve_next_update_exactly(
    monkeypatch, normalized, binary
):
    transform = ObservationStandardization((1.0, -2.0), (2.0, 4.0), 2) if normalized else None
    owner = learner(transform=transform, binary=binary)
    control = owner.fork()
    record = episode()
    state, original_episode = owner.snapshot_state(), asdict(record)
    calls, normalized_inputs = [], []
    original_standardize = adapter.observations.standardize_observations
    original_forward = core._forward

    def standardize(*args):
        calls.append("standardize")
        return original_standardize(*args)

    def forward(features, *args):
        normalized_inputs.append(features.copy())
        return original_forward(features, *args)

    monkeypatch.setattr(adapter.observations, "standardize_observations", standardize)
    monkeypatch.setattr(core, "_forward", forward)
    monkeypatch.setattr(
        adapter.observations,
        "fit_observation_standardization",
        lambda *_a, **_k: pytest.fail("diagnostic refitted statistics"),
    )
    monkeypatch.setattr(
        np.random, "default_rng", lambda *_a, **_k: pytest.fail("diagnostic consumed RNG")
    )
    report = owner.diagnose_training_conditioning(record)
    assert calls == ["standardize"] and len(normalized_inputs) == 2
    expected = np.asarray([[1.0, 1.5], [-2.0, -0.5]]) if normalized else data()[0]
    for values in normalized_inputs:
        np.testing.assert_array_equal(values, expected)
    assert report.standardization_calls == 1 and report.standardization_rows == 2
    assert report.standardization_values == 4 and report.standardization_applied is normalized
    assert report.conditioning.feature_magnitude.rms == pytest.approx(
        math.sqrt(float(np.mean(expected**2)))
    )
    assert report.conditioning.objective_definition == (
        core.MIXED_OBJECTIVE if binary else core.OBJECTIVE
    )
    assert_plain_frozen(report)
    same_state(state, owner.snapshot_state())
    assert asdict(record) == original_episode
    expected_update = control.train_episode(record)
    actual_update = owner.train_episode(record)
    assert actual_update == expected_update
    same_state(control.snapshot_state(), owner.snapshot_state())


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_refuse_non_train_episode_before_normalization_or_core(monkeypatch, role):
    owner = learner()
    before = owner.snapshot_state()
    monkeypatch.setattr(
        adapter.observations,
        "standardize_observations",
        lambda *_a, **_k: pytest.fail("held-out episode reached normalization"),
    )
    monkeypatch.setattr(
        diagnostics,
        "continuous_bc_conditioning",
        lambda *_a, **_k: pytest.fail("held-out episode reached conditioning"),
    )
    with pytest.raises(ValueError):
        owner.diagnose_training_conditioning(episode(role))
    same_state(before, owner.snapshot_state())


def test_should_refuse_probe_policy_and_role_override_before_any_normalization(monkeypatch):
    owner = learner(role="probe_train")
    monkeypatch.setattr(
        adapter.observations,
        "standardize_observations",
        lambda *_a, **_k: pytest.fail("probe policy reached normalization"),
    )
    with pytest.raises(ValueError, match="TRAIN"):
        owner.diagnose_training_conditioning(episode())
    with pytest.raises(TypeError):
        owner.diagnose_training_conditioning(episode("probe_train"), role="train")


@pytest.mark.parametrize("bad", ["momentum_dtype", "counter_type"])
def test_should_validate_complete_owned_momentum_and_counters_before_normalization(
    monkeypatch, bad
):
    owner = learner()
    before, stored = owner.snapshot_state(), owner._state
    if bad == "momentum_dtype":
        corrupt = replace(
            stored, velocity=(stored.velocity[0].astype(np.float32),) + stored.velocity[1:]
        )
    else:
        corrupt = replace(stored, completed_updates=3.0)
    owner._state = corrupt
    monkeypatch.setattr(
        adapter.observations,
        "standardize_observations",
        lambda *_a, **_k: pytest.fail("corrupt complete state reached normalization"),
    )
    try:
        with pytest.raises(ValueError):
            owner.diagnose_training_conditioning(episode())
    finally:
        owner._state = stored
    same_state(before, owner.snapshot_state())


@pytest.mark.parametrize("bad", ["spec", "corrupt_state", "corrupt_episode", "binary_label"])
def test_should_refuse_invalid_adapter_inputs_before_forward_and_leave_owner_exact(
    monkeypatch, bad
):
    owner = learner(binary=bad == "binary_label")
    before = owner.snapshot_state()
    record = episode()
    if bad == "spec":
        changed = ContinuousActionSpec(
            2, spec().action_units, (-3.0, -1.0, 4.0), spec().upper_bounds
        )
        record = replace(record, spec=changed)
    elif bad == "corrupt_state":
        # Corrupt an observation of state, without altering the owner's payload.
        original = owner.snapshot_state

        def snapshot():
            value = original()
            value.parameters[0][0, 0] = np.inf
            return value

        monkeypatch.setattr(owner, "snapshot_state", snapshot)
    elif bad == "corrupt_episode":
        object.__setattr__(record.transitions[0], "action", (99.0, -1.0, 4.0))
    else:
        rows = (replace(record.transitions[0], action=(-2.0, 0.5, 4.0)),) + record.transitions[1:]
        record = replace(record, transitions=rows)
    monkeypatch.setattr(
        core, "_forward", lambda *_a, **_k: pytest.fail("invalid adapter input reached forward")
    )
    monkeypatch.setattr(
        core,
        "continuous_bc_loss_and_gradients",
        lambda *_a, **_k: pytest.fail("invalid adapter input reached backward"),
    )
    with pytest.raises(ValueError):
        owner.diagnose_training_conditioning(record)
    if bad == "corrupt_state":
        monkeypatch.setattr(owner, "snapshot_state", original)
    same_state(before, owner.snapshot_state())


def test_should_preserve_existing_residual_schema_and_behavior_after_conditioning():
    owner = learner()
    record = episode()
    before = owner.diagnose_training_episode(record, ("first", "second"))
    state = owner.snapshot_state()
    owner.diagnose_training_conditioning(record)
    after = owner.diagnose_training_episode(record, ("first", "second"))
    assert before == after
    assert tuple(field.name for field in fields(after)) == (
        "residuals",
        "prediction_calls",
        "prediction_rows",
        "prediction_values",
        "prediction_bytes",
    )
    assert after.prediction_calls == 1 and after.prediction_rows == 2
    assert after.prediction_values == 6 and after.prediction_bytes == 48
    same_state(state, owner.snapshot_state())
