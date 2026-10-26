"""Declared scale floors retain empirical statistics and frozen learner state.

Prospective per-module costs: two D2/H2/A2 initializers draw16 normal values;
four SGD updates consume8 rows; four predictions consume8 rows. Five successful
fits consume10 rows/20 values. No other learner kernels, RNG, native IO or fits.
The manual learner receives literal transformed rows, not the tested transform.
"""

from dataclasses import FrozenInstanceError, asdict, replace
from typing import cast

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as adapter
from src.core import observation_standardization as normalization
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe


def spec():
    return ContinuousActionSpec(2, ("x", "grip"), (-1.0, -1.0), (1.0, 1.0))


def episode(role="train"):
    return ContinuousEpisode(
        "scale-floor",
        "synthetic",
        role,
        spec(),
        (
            ContinuousTransition(0, (1.0, 10.0), (-0.2, 0.4), 0.0, (3.0, 10.0), False, False),
            ContinuousTransition(1, (3.0, 10.0), (0.5, -0.3), 1.0, (5.0, 14.0), True, False),
        ),
    )


def selected():
    return normalization.ScaleFlooredObservationStandardization(
        (2.0, 10.0), (0.5, 1.0), 2, (2.0, 4.0)
    )


def assert_state(first, second, *, policy=True):
    if policy:
        assert first.policy == second.policy
    assert (first.completed_updates, first.training_examples) == (
        second.completed_updates,
        second.training_examples,
    )
    for a, b in zip(
        (*first.parameters, *first.velocity), (*second.parameters, *second.velocity), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def forbidden(*args, **kwargs):
    pytest.fail("refused metadata or role reached a fit or learner kernel")


def test_should_retain_population_scales_and_constant_one_while_applying_only_declared_floor():
    raw = np.array([[-1.0, 10.0], [1.0, 10.0]], dtype=np.float64)
    before = raw.copy()
    fitted = normalization.fit_observation_standardization(raw, scale_floors=(0.5, 4.0))
    assert type(fitted) is normalization.ScaleFlooredObservationStandardization
    assert fitted.means == (0.0, 10.0) and fitted.scales == (1.0, 1.0)
    assert fitted.scale_floors == (0.5, 4.0) and fitted.effective_scales == (1.0, 4.0)
    assert fitted.fitted_rows == 2
    probe = np.array([[2.0, 14.0], [-3.0, 2.0]], dtype=np.float64)
    transformed = normalization.standardize_observations(probe, fitted, spec())
    np.testing.assert_array_equal(transformed, [[2.0, 1.0], [-3.0, -2.0]])
    transformed[0, 0] = 999.0
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(probe, [[2.0, 14.0], [-3.0, 2.0]])
    assert fitted.scales == (1.0, 1.0)


def test_should_preserve_default_three_field_serialization_and_own_selected_metadata():
    rows = np.array([[0.0, 7.0], [4.0, 7.0]], dtype=np.float64)
    legacy = normalization.fit_observation_standardization(rows)
    explicit_none = normalization.fit_observation_standardization(rows, scale_floors=None)
    floored = normalization.fit_observation_standardization(rows, scale_floors=(3.0, 0.25))
    assert type(floored) is normalization.ScaleFlooredObservationStandardization
    assert type(legacy) is normalization.ObservationStandardization
    assert (
        asdict(legacy)
        == asdict(explicit_none)
        == dict(means=(2.0, 7.0), scales=(2.0, 1.0), fitted_rows=2)
    )
    assert floored.means == legacy.means and floored.scales == legacy.scales
    assert floored.effective_scales == (3.0, 1.0)
    assert set(asdict(floored)) == {
        "means",
        "scales",
        "fitted_rows",
        "scale_floors",
        "effective_scales",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(floored, "scale_floors", (1.0, 1.0))
    policy = adapter.ContinuousBCPolicy(spec(), hidden_dim=2, observation_standardization=floored)
    owned = policy.observation_standardization
    assert owned == floored and owned is not floored
    object.__setattr__(floored, "effective_scales", (99.0, 99.0))
    assert getattr(owned, "effective_scales") == (3.0, 1.0)


def test_should_fit_adapter_only_from_training_observations_not_terminal_next_observation():
    source = episode()
    fitted = adapter.fit_training_standardization(source, scale_floors=(2.0, 4.0))
    assert fitted.means == (2.0, 10.0) and fitted.scales == (1.0, 1.0)
    assert getattr(fitted, "effective_scales") == (2.0, 4.0) and fitted.fitted_rows == 2
    assert source.transitions[-1].next_observation == (5.0, 14.0)


class FloatAlias(float):
    pass


class TupleAlias(tuple):
    pass


@pytest.mark.parametrize(
    "floors",
    [
        [1.0, 1.0],
        (),
        (1.0,),
        (1.0, 1.0, 1.0),
        (True, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (np.float64(1.0), 1.0),
        (np.int64(1), 1.0),
        (FloatAlias(1.0), 1.0),
        TupleAlias((1.0, 1.0)),
        ("1", 1.0),
    ],
)
def test_should_refuse_ambiguous_floor_types_dimensions_or_nonpositive_values(floors):
    with pytest.raises(ValueError):
        normalization.ScaleFlooredObservationStandardization((0.0, 0.0), (1.0, 1.0), 2, floors)


def test_should_accept_maximum_metadata_dimension_without_array_work():
    transform = normalization.ScaleFlooredObservationStandardization(
        (0.0,) * 1024, (1.0,) * 1024, 256, (2.0,) * 1024
    )
    assert transform.effective_scales == (2.0,) * 1024
    with pytest.raises(ValueError):
        normalization.ScaleFlooredObservationStandardization(
            (0.0,) * 1025, (1.0,) * 1025, 1, (2.0,) * 1025
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale_floors", (0.0, 4.0)),
        ("scale_floors", (np.float64(2.0), 4.0)),
        ("effective_scales", (2.0, 99.0)),
        ("effective_scales", (np.float64(2.0), 4.0)),
        ("effective_scales", [2.0, 4.0]),
        ("scales", (0.5, True)),
    ],
)
def test_should_refuse_corrupt_selected_transform_before_prediction_kernel(
    monkeypatch, field, value
):
    transform = selected()
    object.__setattr__(transform, field, value)
    monkeypatch.setattr(adapter.core, "continuous_bc_predict", forbidden)
    with pytest.raises(ValueError):
        adapter.ContinuousBCPolicy(spec(), hidden_dim=2, observation_standardization=transform)


@pytest.mark.parametrize(
    "field,value", [("scale_floors", (True, 4.0)), ("effective_scales", (2.0, 5.0))]
)
def test_should_validate_selected_metadata_before_feature_array_math(monkeypatch, field, value):
    transform = selected()
    object.__setattr__(transform, field, value)
    monkeypatch.setattr(normalization.bc, "_features", forbidden)
    with pytest.raises(ValueError):
        normalization.standardize_observations(cast(adapter.Array, object()), transform, spec())


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test", "inner_guard"])
def test_should_refuse_nontraining_fit_before_core_fitter(monkeypatch, role):
    monkeypatch.setattr(normalization, "fit_observation_standardization", forbidden)
    with pytest.raises(ValueError):
        adapter.fit_training_standardization(episode(role), scale_floors=(1.0, 1.0))


def test_should_refuse_combined_rgb_range_and_floor_before_either_fitter(monkeypatch):
    monkeypatch.setattr(normalization, "fit_observation_standardization", forbidden)
    monkeypatch.setattr(adapter, "fit_rgb_range_standardization", forbidden)
    with pytest.raises(ValueError):
        adapter.fit_training_standardization(
            episode(), scale_floors=(1.0, 1.0), rgb_range_recipe=pick_cube_rgb_recipe()
        )


@pytest.fixture(scope="module")
def paired():
    transform = selected()
    policy = adapter.ContinuousBCPolicy(
        spec(),
        hidden_dim=2,
        learning_rate=0.03,
        momentum=0.8,
        observation_standardization=transform,
    )
    source = adapter.NumpyContinuousBCLearner(policy)
    manual = adapter.NumpyContinuousBCLearner(replace(policy, observation_standardization=None))
    initial = source.snapshot_state()
    incoming = cast(
        adapter.Parameters, tuple(np.full_like(value, 0.125) for value in initial.velocity)
    )
    source.restore_state(replace(initial, velocity=incoming))
    manual.restore_state(
        replace(manual.snapshot_state(), parameters=initial.parameters, velocity=incoming)
    )
    raw = np.array([[0.0, 6.0], [4.0, 14.0]], dtype=np.float64)
    expected_features = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float64)
    targets = np.array([[-0.2, 0.4], [0.5, -0.3]], dtype=np.float64)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(normalization, "fit_observation_standardization", forbidden)
        np.testing.assert_array_equal(source.predict(raw), manual.predict(expected_features))
        diagnostics = []
        for _ in range(2):
            actual, expected = (
                source.train_batch(raw, targets),
                manual.train_batch(expected_features, targets),
            )
            assert actual == expected
            assert_state(source.snapshot_state(), manual.snapshot_state(), policy=False)
            diagnostics.append(actual)
        np.testing.assert_array_equal(source.predict(raw), manual.predict(expected_features))
    return source, manual, raw, expected_features, targets, diagnostics


def test_should_match_exact_manual_transform_predictions_updates_and_full_incoming_momentum(paired):
    source, manual, raw, expected, targets, diagnostics = paired
    assert_state(source.snapshot_state(), manual.snapshot_state(), policy=False)
    assert (
        source.snapshot_state().completed_updates == 2
        and source.snapshot_state().training_examples == 4
    )
    assert all(diagnostic.definition == adapter.OBJECTIVE for diagnostic in diagnostics)
    np.testing.assert_array_equal(raw, [[0.0, 6.0], [4.0, 14.0]])
    np.testing.assert_array_equal(expected, [[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(targets, [[-0.2, 0.4], [0.5, -0.3]])
    assert source.policy.observation_standardization == selected()


def test_should_own_selected_transform_through_fork_restore_and_refuse_different_floor_atomically(
    paired, monkeypatch
):
    source = paired[0]
    before = source.snapshot_state()
    child = source.fork()
    child.restore_state(before)
    assert_state(before, child.snapshot_state())
    assert child.policy.observation_standardization is not source.policy.observation_standardization
    child_snapshot = child.snapshot_state()
    child_snapshot.parameters[0][0, 0] += 10.0
    np.testing.assert_array_equal(child.snapshot_state().parameters[0], before.parameters[0])
    monkeypatch.setattr(normalization, "fit_observation_standardization", forbidden)
    foreign = replace(
        before,
        policy=replace(
            before.policy,
            observation_standardization=normalization.ScaleFlooredObservationStandardization(
                (2.0, 10.0), (0.5, 1.0), 2, (2.0, 5.0)
            ),
        ),
    )
    with pytest.raises(ValueError):
        child.restore_state(foreign)
    assert_state(before, child.snapshot_state())
    assert_state(before, source.snapshot_state())


@pytest.mark.parametrize(
    "field,value", [("scale_floors", (2.0, -1.0)), ("effective_scales", (2.0, 3.0))]
)
def test_should_refuse_mutated_snapshot_transform_without_partial_restore(paired, field, value):
    child = paired[0].fork()
    before = child.snapshot_state()
    damaged = child.snapshot_state()
    object.__setattr__(damaged.policy.observation_standardization, field, value)
    with pytest.raises(ValueError):
        child.restore_state(damaged)
    assert_state(before, child.snapshot_state())


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test", "inner_guard"])
def test_should_refuse_nontraining_update_before_transform_or_gradient(paired, monkeypatch, role):
    source = paired[0].fork()
    before = source.snapshot_state()
    monkeypatch.setattr(normalization, "standardize_observations", forbidden)
    monkeypatch.setattr(adapter.core, "continuous_bc_loss_and_gradients", forbidden)
    with pytest.raises(ValueError):
        source.train_episode(episode(role))
    assert_state(before, source.snapshot_state())
