"""Independent observation-conditioning oracles; synthetic data, no robot evidence."""

from dataclasses import replace
from dataclasses import asdict
import json
import math
import sys

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as adapter
from src.core import continuous_bc as core
from src.core import observation_standardization as normalization
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def test_should_apply_frozen_transform_through_existing_robot_caller(monkeypatch, tmp_path):
    from scripts import maniskill_continuous_smoke as smoke
    from src.infra import maniskill_continuous as port
    from tests.test_maniskill_bc_smoke import FakeEnvironment

    environment = FakeEnvironment()
    monkeypatch.setattr(port, "_make_environment", lambda: environment)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native-boundary")
    output = tmp_path / "normalized-robot.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    action_spec = ContinuousActionSpec(3, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    transform = normalization.ObservationStandardization((0.1, -0.1, 0.2), (0.5, 2.0, 0.25), 2)
    model = adapter.NumpyContinuousBCLearner(
        adapter.ContinuousBCPolicy(action_spec, hidden_dim=2, observation_standardization=transform)
    )
    before = model.snapshot_state()
    raw = np.array([[0.2, -0.3, 0.4], [1.2, -0.3, 0.4]], dtype=np.float32).astype(np.float64)
    manual = (raw - np.array(transform.means)) / np.array(transform.scales)
    expected = core.continuous_bc_predict(manual, before.parameters, action_spec).astype(np.float32)
    assert smoke.main(learner=model) == 0
    result = json.loads(output.read_text())
    np.testing.assert_array_equal(np.concatenate(environment.commands), expected)
    assert result["frozen_state_exact"] is True and result["original_owner_exact"] is True
    assert result["learner_policy"]["observation_standardization"] == json.loads(
        json.dumps(asdict(transform))
    )
    assert result["prediction_returns"] == result["validated_prediction_rows"] == 2
    assert result["validated_prediction_bytes"] == 64 and result["optimizer_updates"] == 0
    after = model.snapshot_state()
    assert before.policy == after.policy
    for a, b in zip(
        (*before.parameters, *before.velocity), (*after.parameters, *after.velocity), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def spec(dimension=2):
    return ContinuousActionSpec(
        dimension, ("normalized_x", "normalized_y"), (-1.0, -1.0), (1.0, 1.0)
    )


def raw_rows():
    return np.array([[10.0, 100.0], [12.0, 100.0], [14.0, 100.0]], dtype=np.float64)


def targets():
    return np.array([[-0.2, 0.4], [0.1, -0.3], [0.5, 0.2]], dtype=np.float64)


def episode(role="train"):
    observations = raw_rows().tolist()
    actions = targets().tolist()
    return ContinuousEpisode(
        "conditioning",
        "synthetic",
        role,
        spec(),
        tuple(
            ContinuousTransition(
                index,
                tuple(observation),
                tuple(action),
                0.0,
                tuple(observations[index + 1] if index < 2 else [16.0, 100.0]),
                index == 2,
                False,
            )
            for index, (observation, action) in enumerate(zip(observations, actions, strict=True))
        ),
    )


def learner(transform=None):
    return adapter.NumpyContinuousBCLearner(
        adapter.ContinuousBCPolicy(
            spec(),
            hidden_dim=2,
            seed=17,
            learning_rate=0.03,
            momentum=0.8,
            observation_standardization=transform,
        )
    )


def assert_arrays_and_counters_equal(first, second):
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(first, group), getattr(second, group), strict=True):
            np.testing.assert_array_equal(left, right)


def test_should_fit_population_mean_and_scale_from_hand_computed_rows():
    values = np.array([[2.0, 10.0], [4.0, 14.0], [6.0, 18.0]], dtype=np.float64)
    fitted = normalization.fit_observation_standardization(values)
    assert fitted.means == pytest.approx((4.0, 14.0))
    assert fitted.scales == pytest.approx((math.sqrt(8 / 3), math.sqrt(32 / 3)))
    assert fitted.fitted_rows == 3
    transformed = normalization.standardize_observations(values, fitted, spec())
    expected = np.array(
        [[-math.sqrt(1.5), -math.sqrt(1.5)], [0.0, 0.0], [math.sqrt(1.5), math.sqrt(1.5)]]
    )
    np.testing.assert_allclose(transformed, expected, rtol=2e-15, atol=1e-15)
    # The population standard deviation, rather than sample ddof=1, is one.
    np.testing.assert_allclose(np.mean(transformed**2, axis=0), [1.0, 1.0], rtol=2e-15)


def test_should_assign_constant_scale_one_and_preserve_later_deviations():
    fitted = normalization.fit_observation_standardization(raw_rows())
    assert fitted.means == pytest.approx((12.0, 100.0))
    assert fitted.scales == pytest.approx((math.sqrt(8 / 3), 1.0))
    changed = np.array([[12.0, 102.5]], dtype=np.float64)
    transformed = normalization.standardize_observations(changed, fitted, spec())
    np.testing.assert_array_equal(transformed, [[0.0, 2.5]])


def test_should_fit_single_row_without_removing_any_coordinate():
    values = np.array([[1.5, -4.0]], dtype=np.float64)
    fitted = normalization.fit_observation_standardization(values)
    assert fitted.means == (1.5, -4.0) and fitted.scales == (1.0, 1.0)
    assert fitted.fitted_rows == 1
    np.testing.assert_array_equal(
        normalization.standardize_observations(values, fitted, spec()), [[0.0, 0.0]]
    )


def test_should_fit_huge_finite_values_without_overflowing_sum_or_squares():
    values = np.array([[-1e308, 1e308], [1e308, 1e308]], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        fitted = normalization.fit_observation_standardization(values)
    assert fitted.means == (0.0, 1e308)
    assert fitted.scales == (1e308, 1.0)
    np.testing.assert_array_equal(
        normalization.standardize_observations(values, fitted, spec()), [[-1.0, 0.0], [1.0, 0.0]]
    )


def test_should_refuse_unrepresentable_nonconstant_standard_deviation():
    # The true standard deviation is half the smallest float64 subnormal.
    minimum = np.nextafter(np.float64(0.0), np.float64(1.0))
    values = np.array([[0.0], [minimum]], dtype=np.float64)
    with pytest.raises((ValueError, FloatingPointError)):
        normalization.fit_observation_standardization(values)


def test_should_accept_maximum_fitting_shape_and_retain_native_immutable_metadata():
    values = np.ones((256, 1024), dtype=np.float64)
    fitted = normalization.fit_observation_standardization(values)
    assert type(fitted.means) is tuple and type(fitted.scales) is tuple
    assert len(fitted.means) == len(fitted.scales) == 1024 and fitted.fitted_rows == 256
    assert all(type(value) is float for value in (*fitted.means, *fitted.scales))
    assert fitted.__dataclass_params__.frozen


@pytest.mark.parametrize(
    "bad",
    [
        "list",
        "float32",
        "integer",
        "subclass",
        "nan",
        "inf",
        "flat",
        "empty_rows",
        "many_rows",
        "empty_columns",
        "many_columns",
    ],
)
def test_should_refuse_ambiguous_nonfinite_or_unbounded_fit_matrices(bad):
    values = raw_rows()
    if bad == "list":
        values = values.tolist()
    elif bad in ("float32", "integer"):
        values = values.astype(np.float32 if bad == "float32" else np.int64)
    elif bad == "subclass":

        class ForeignArray(np.ndarray):
            pass

        values = values.view(ForeignArray)
    elif bad in ("nan", "inf"):
        values[0, 0] = np.nan if bad == "nan" else np.inf
    elif bad == "flat":
        values = values.ravel()
    else:
        values = np.zeros(
            {
                "empty_rows": (0, 2),
                "many_rows": (257, 2),
                "empty_columns": (1, 0),
                "many_columns": (1, 1025),
            }[bad],
            dtype=np.float64,
        )
    with pytest.raises(ValueError):
        normalization.fit_observation_standardization(values)


@pytest.mark.parametrize(
    "means,scales,rows",
    [
        ([0.0], (1.0,), 1),
        ((0.0,), [1.0], 1),
        ((), (), 1),
        ((0.0,), (1.0, 1.0), 1),
        ((True,), (1.0,), 1),
        ((float("nan"),), (1.0,), 1),
        ((float("inf"),), (1.0,), 1),
        ((0.0,), (0.0,), 1),
        ((0.0,), (-1.0,), 1),
        ((0.0,), (True,), 1),
        ((0.0,), (float("inf"),), 1),
        ((0.0,), (float("nan"),), 1),
        ((0.0,), (1.0,), True),
        ((0.0,), (1.0,), 0),
        ((0.0,), (1.0,), 257),
        ((0.0,), (1.0,), 1.0),
        ((0.0,) * 1025, (1.0,) * 1025, 1),
        ((np.float64(0.0),), (1.0,), 1),
        ((0.0,), (np.float64(1.0),), 1),
    ],
)
def test_should_refuse_invalid_transform_metadata(means, scales, rows):
    with pytest.raises(ValueError):
        normalization.ObservationStandardization(means, scales, rows)


@pytest.mark.parametrize(
    "transform", [None, normalization.ObservationStandardization((12.0, 100.0), (2.0, 1.0), 3)]
)
def test_should_return_owned_arrays_and_leave_raw_input_untouched(transform):
    values = raw_rows()
    before = values.copy()
    result = normalization.standardize_observations(values, transform, spec())
    assert result.dtype == np.float64 and result.shape == values.shape
    assert not np.shares_memory(values, result)
    np.testing.assert_array_equal(values, before)
    if transform is None:
        np.testing.assert_array_equal(result, before)
    result.fill(999.0)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("case", ["subtraction", "division"])
def test_should_fail_unrepresentable_application_without_clipping_or_refitting(case):
    if case == "subtraction":
        transform = normalization.ObservationStandardization((-1e308,), (1.0,), 1)
    else:
        transform = normalization.ObservationStandardization((0.0,), (1e-308,), 1)
    with pytest.raises((ValueError, FloatingPointError)):
        normalization.standardize_observations(np.array([[1e308]]), transform, spec(1))


@pytest.mark.parametrize(
    "case", ["dimension", "raw_dimension", "float32", "nan", "corrupt", "foreign"]
)
def test_should_revalidate_transform_and_raw_inputs_at_application(case):
    transform = normalization.ObservationStandardization((12.0, 100.0), (2.0, 1.0), 3)
    values = raw_rows()
    if case == "dimension":
        transform = normalization.ObservationStandardization((12.0,), (2.0,), 3)
    elif case == "raw_dimension":
        values = values[:, :1]
    elif case == "float32":
        values = values.astype(np.float32)
    elif case == "nan":
        values[0, 0] = np.nan
    elif case == "corrupt":
        object.__setattr__(transform, "scales", (0.0, 1.0))
    else:
        transform = object()
    with pytest.raises(ValueError):
        normalization.standardize_observations(values, transform, spec())


def test_should_revalidate_raw_inputs_even_without_a_transform():
    with pytest.raises(ValueError):
        normalization.standardize_observations(raw_rows().astype(np.float32), None, spec())


def test_should_fit_only_training_observations_without_mutating_episode():
    data = episode()
    before = repr(data)
    fitted = adapter.fit_training_standardization(data)
    assert fitted.means == pytest.approx((12.0, 100.0))
    assert fitted.scales == pytest.approx((math.sqrt(8 / 3), 1.0))
    assert fitted.fitted_rows == 3 and repr(data) == before


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_non_train_fitting_before_computing_statistics(monkeypatch, role):
    def forbidden(*_):
        pytest.fail("non-TRAIN role reached statistics fitting")

    monkeypatch.setattr(normalization, "fit_observation_standardization", forbidden)
    with pytest.raises(ValueError):
        adapter.fit_training_standardization(episode(role))


def test_should_revalidate_corrupted_training_observations_before_fit(monkeypatch):
    data = episode()
    object.__setattr__(data.transitions[0], "observation", (float("nan"), 100.0))
    monkeypatch.setattr(
        normalization,
        "fit_observation_standardization",
        lambda *_: pytest.fail("corrupt episode reached fit"),
    )
    with pytest.raises(ValueError):
        adapter.fit_training_standardization(data)


def test_should_preserve_default_positional_policy_and_exact_raw_prediction_and_update():
    policy = adapter.ContinuousBCPolicy(spec(), 2, 17, 0.03, 0.8, "train")
    assert policy.observation_standardization is None
    model = adapter.NumpyContinuousBCLearner(policy)
    before = model.snapshot_state()
    expected_prediction = core.continuous_bc_predict(raw_rows(), before.parameters, spec())
    np.testing.assert_array_equal(model.predict(raw_rows()), expected_prediction)
    derivative = core.continuous_bc_loss_and_gradients(
        raw_rows(), targets(), before.parameters, spec()
    )
    actual = model.train_batch(raw_rows(), targets())
    assert actual.value == derivative.loss
    after = model.snapshot_state()
    for old, gradient, velocity, parameter in zip(
        before.parameters, derivative.gradients, after.velocity, after.parameters, strict=True
    ):
        np.testing.assert_array_equal(velocity, gradient)
        np.testing.assert_array_equal(parameter, old - 0.03 * gradient)
    assert after.completed_updates == 1 and after.training_examples == 3


def test_should_match_manual_preprocessing_for_predictions_losses_momentum_and_updates(monkeypatch):
    transform = normalization.fit_observation_standardization(raw_rows())
    model = learner(transform)
    manual = learner()
    # Independent known transform; do not call the transform under test for the oracle.
    preprocessed = np.array([[-math.sqrt(1.5), 0.0], [0.0, 0.0], [math.sqrt(1.5), 0.0]])
    before = model.snapshot_state()
    assert_arrays_and_counters_equal(before, manual.snapshot_state())
    monkeypatch.setattr(
        normalization,
        "fit_observation_standardization",
        lambda *_: pytest.fail("inference/update refitted statistics"),
    )
    for _ in range(3):
        np.testing.assert_allclose(
            model.predict(raw_rows()), manual.predict(preprocessed), rtol=2e-14, atol=2e-15
        )
        observed = model.train_batch(raw_rows(), targets())
        expected = manual.train_batch(preprocessed, targets())
        assert observed.value == pytest.approx(expected.value, rel=2e-14, abs=2e-15)
        actual_state, manual_state = model.snapshot_state(), manual.snapshot_state()
        for group in ("parameters", "velocity"):
            for actual_array, expected_array in zip(
                getattr(actual_state, group), getattr(manual_state, group), strict=True
            ):
                np.testing.assert_allclose(actual_array, expected_array, rtol=2e-13, atol=2e-15)
    assert actual_state.completed_updates == 3 and actual_state.training_examples == 9
    assert actual_state.policy.observation_standardization == transform


def test_should_apply_same_frozen_transform_to_training_episode_and_diagnosis():
    transform = adapter.fit_training_standardization(episode())
    model = learner(transform)
    raw_model = learner()
    preprocessed = np.array([[-math.sqrt(1.5), 0.0], [0.0, 0.0], [math.sqrt(1.5), 0.0]])
    expected = raw_model.train_batch(preprocessed, targets())
    observed = model.train_episode(episode())
    assert observed.value == pytest.approx(expected.value, rel=2e-14)
    before = model.snapshot_state()
    manual_predictions = raw_model.predict(preprocessed)
    report = model.diagnose_training_episode(episode(), ("first", "second", "third"))
    independent_loss = float(np.mean((manual_predictions - targets()) ** 2) / 2)
    assert report.residuals.overall.normalized_half_mse == pytest.approx(
        independent_loss, rel=2e-14
    )
    assert_arrays_and_counters_equal(before, model.snapshot_state())
    assert model.snapshot_state().policy == before.policy


def test_should_own_transform_in_policy_snapshot_and_fork_and_continue_exactly():
    transform = normalization.ObservationStandardization((12.0, 100.0), (2.0, 1.0), 3)
    model = learner(transform)
    expected = model.policy.observation_standardization
    object.__setattr__(transform, "means", (999.0, 100.0))
    assert model.policy.observation_standardization == expected
    model.train_batch(raw_rows(), targets())
    snapshot = model.snapshot_state()
    forked = model.fork()
    restored = learner(expected)
    restored.restore_state(snapshot)
    external_policy = model.policy
    object.__setattr__(external_policy.observation_standardization, "scales", (999.0, 1.0))
    assert model.policy.observation_standardization == expected
    for child in (model, forked, restored):
        child.train_batch(raw_rows(), targets())
    for child in (forked, restored):
        assert child.policy == model.policy
        assert_arrays_and_counters_equal(model.snapshot_state(), child.snapshot_state())
    saved = model.snapshot_state()
    object.__setattr__(snapshot.policy.observation_standardization, "means", (-999.0, 100.0))
    snapshot.parameters[0].fill(999.0)
    assert model.policy == saved.policy
    assert_arrays_and_counters_equal(saved, model.snapshot_state())


@pytest.mark.parametrize("difference", ["means", "scales", "fitted_rows", "missing"])
def test_should_refuse_transform_mismatch_restore_atomically(difference):
    fitted = normalization.ObservationStandardization((12.0, 100.0), (2.0, 1.0), 3)
    model = learner(fitted)
    model.train_batch(raw_rows(), targets())
    before = model.snapshot_state()
    changed = {
        "means": replace(fitted, means=(11.0, 100.0)),
        "scales": replace(fitted, scales=(3.0, 1.0)),
        "fitted_rows": replace(fitted, fitted_rows=2),
        "missing": None,
    }[difference]
    candidate = learner(changed).snapshot_state()
    with pytest.raises(ValueError):
        model.restore_state(candidate)
    after = model.snapshot_state()
    assert after.policy == before.policy
    assert_arrays_and_counters_equal(before, after)


def test_should_refuse_policy_transform_dimension_mismatch_before_model_creation():
    transform = normalization.ObservationStandardization((1.0,), (1.0,), 3)
    with pytest.raises(ValueError):
        adapter.ContinuousBCPolicy(spec(), observation_standardization=transform)


def test_should_refuse_preprocessing_overflow_before_gradient_without_changing_state(monkeypatch):
    transform = normalization.ObservationStandardization((-1e308, 0.0), (1.0, 1.0), 1)
    model = learner(transform)
    before = model.snapshot_state()
    raw = np.array([[1e308, 0.0]], dtype=np.float64)
    monkeypatch.setattr(
        core,
        "continuous_bc_loss_and_gradients",
        lambda *_: pytest.fail("overflow reached optimizer derivative"),
    )
    with pytest.raises((ValueError, FloatingPointError)):
        model.train_batch(raw, np.zeros((1, 2), dtype=np.float64))
    after = model.snapshot_state()
    assert after.policy == before.policy
    assert_arrays_and_counters_equal(before, after)
