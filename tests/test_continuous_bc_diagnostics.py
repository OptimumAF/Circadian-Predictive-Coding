"""Hand-calculated TRAIN residual checks; no simulator, updates or robot claims."""

from dataclasses import fields, is_dataclass, replace
import math

import numpy as np
import pytest

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_bc_diagnostics import continuous_bc_residuals
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def geometry():
    # Duplicate physical units are intentional: the axes remain distinct.
    return ContinuousActionSpec(2, ("m", "m", "fixed"), (-2.0, 10.0, 4.0), (6.0, 14.0, 4.0))


def action_rows():
    return (
        np.array([[6.0, 12.0, 4.0], [2.0, 14.0, 4.0], [-2.0, 10.0, 4.0]]),
        np.array([[2.0, 10.0, 4.0], [-2.0, 14.0, 4.0], [6.0, 12.0, 4.0]]),
    )


def episode(role="train", spec=None):
    declared = geometry() if spec is None else spec
    _, actions = action_rows()
    return ContinuousEpisode(
        "synthetic-train",
        "synthetic",
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                (float(index), 0.5),
                tuple(row),
                0.0,
                (float(index + 1), 0.5),
                index == 2,
                False,
            )
            for index, row in enumerate(actions.tolist())
        ),
    )


def assert_same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(first, group), getattr(second, group), strict=True):
            np.testing.assert_array_equal(left, right)


def test_should_match_hand_computed_axis_bias_mae_rmse_and_half_mse():
    predictions, targets = action_rows()
    report = continuous_bc_residuals(predictions, targets, geometry(), ("seek", "close", "seek"))
    first, second, constant = report.overall.axes
    assert report.overall.rows == 3 and report.overall.active_axes == 2
    assert report.overall.normalized_half_mse == pytest.approx(2 / 3)
    assert [(axis.index, axis.unit, axis.active) for axis in report.overall.axes] == [
        (0, "m", True),
        (1, "m", True),
        (2, "fixed", False),
    ]
    assert first.normalized_bias == 0.0
    assert first.normalized_mae == pytest.approx(4 / 3)
    assert first.normalized_rmse == pytest.approx(math.sqrt(2))
    assert first.normalized_half_mse == pytest.approx(1)
    assert first.maximum_absolute_normalized_error == 2.0
    assert second.normalized_bias == 0.0
    assert second.normalized_mae == pytest.approx(2 / 3)
    assert second.normalized_rmse == pytest.approx(math.sqrt(2 / 3))
    assert second.normalized_half_mse == pytest.approx(1 / 3)
    assert second.maximum_absolute_normalized_error == 1.0
    assert all(
        getattr(constant, name) == 0.0
        for name in (
            "normalized_bias",
            "normalized_mae",
            "normalized_rmse",
            "normalized_half_mse",
            "maximum_absolute_normalized_error",
        )
    )


def test_should_preserve_first_seen_stage_order_rows_and_weighted_objective():
    predictions, targets = action_rows()
    report = continuous_bc_residuals(predictions, targets, geometry(), ("seek", "close", "seek"))
    seek, close = report.stages
    assert (seek.label, seek.row_indices, seek.summary.rows) == ("seek", (0, 2), 2)
    assert (close.label, close.row_indices, close.summary.rows) == ("close", (1,), 1)
    assert seek.summary.normalized_half_mse == pytest.approx(7 / 8)
    assert close.summary.normalized_half_mse == pytest.approx(1 / 4)
    assert seek.summary.axes[0].normalized_bias == -0.5
    assert close.summary.axes[0].normalized_bias == 1.0
    weighted = sum(
        stage.summary.rows * stage.summary.normalized_half_mse for stage in report.stages
    )
    assert weighted / report.overall.rows == pytest.approx(report.overall.normalized_half_mse)
    # A mean of stage means would wrongly over-weight the single close row.
    assert report.overall.normalized_half_mse != pytest.approx((7 / 8 + 1 / 4) / 2)


def test_should_use_predicted_minus_target_sign_without_cancelling_absolute_error():
    spec = ContinuousActionSpec(1, ("N",), (10.0,), (14.0,))
    predictions = np.array([[14.0], [12.0]])
    targets = np.array([[10.0], [14.0]])
    axis = continuous_bc_residuals(predictions, targets, spec, ("all",) * 2).overall.axes[0]
    assert axis.normalized_bias == 0.5
    assert axis.normalized_mae == 1.5
    assert axis.normalized_rmse == pytest.approx(math.sqrt(2.5))
    assert axis.normalized_half_mse == 1.25
    assert axis.maximum_absolute_normalized_error == 2.0


def test_should_report_zero_objective_when_every_axis_is_constant():
    spec = ContinuousActionSpec(1, ("N", "m"), (4.0, -3.0), (4.0, -3.0))
    actions = np.array([[4.0, -3.0], [4.0, -3.0]])
    report = continuous_bc_residuals(actions, actions, spec, ("first", "second"))
    for summary in (report.overall, *(stage.summary for stage in report.stages)):
        assert summary.active_axes == 0
        assert summary.normalized_half_mse == 0.0
        assert all(not axis.active and axis.normalized_mae == 0.0 for axis in summary.axes)


def test_should_normalize_opposite_huge_finite_endpoints_without_overflow():
    spec = ContinuousActionSpec(1, ("m",), (-1e308,), (1e308,))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        report = continuous_bc_residuals(
            np.array([[1e308], [-1e308]]),
            np.array([[-1e308], [1e308]]),
            spec,
            ("up", "down"),
        )
    axis = report.overall.axes[0]
    assert axis.normalized_bias == 0.0
    assert axis.normalized_mae == axis.normalized_rmse == 2.0
    assert report.overall.normalized_half_mse == 2.0
    assert report.stages[0].summary.axes[0].normalized_bias == 2.0
    assert report.stages[1].summary.axes[0].normalized_bias == -2.0


def test_should_accept_maximum_rows_axes_stage_count_and_utf8_label_bytes():
    spec = ContinuousActionSpec(1, ("m",) * 16, (-1.0,) * 16, (1.0,) * 16)
    actions = np.zeros((256, 16), dtype=np.float64)
    labels = tuple("é" * 16 if index % 16 == 0 else str(index % 16) for index in range(256))
    report = continuous_bc_residuals(actions, actions, spec, labels)
    assert report.overall.rows == 256 and report.overall.active_axes == 16
    assert len(report.stages) == 16
    assert report.stages[0].row_indices == tuple(range(0, 256, 16))


@pytest.mark.parametrize(
    "labels",
    [
        ["a", "a", "a"],
        ("a",),
        (),
        ("", "a", "a"),
        (" a", "a", "a"),
        ("a ", "a", "a"),
        (1, "a", "a"),
        ("x" * 33, "a", "a"),
        ("é" * 17, "a", "a"),
        ("\ud800", "a", "a"),
    ],
)
def test_should_refuse_invalid_or_unbounded_stage_labels(labels):
    predictions, targets = action_rows()
    with pytest.raises(ValueError):
        continuous_bc_residuals(predictions, targets, geometry(), labels)


def test_should_refuse_more_than_sixteen_distinct_stage_labels():
    spec = ContinuousActionSpec(1, ("m",), (-1.0,), (1.0,))
    actions = np.zeros((17, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        continuous_bc_residuals(actions, actions, spec, tuple(str(index) for index in range(17)))


@pytest.mark.parametrize("field", ["predictions", "targets"])
@pytest.mark.parametrize(
    "bad", ["list", "float32", "integer", "object", "nan", "inf", "shape", "flat", "bounds"]
)
def test_should_refuse_malformed_or_out_of_bounds_action_matrices(field, bad):
    predictions, targets = action_rows()
    values = predictions if field == "predictions" else targets
    if bad == "list":
        values = values.tolist()
    elif bad in ("float32", "integer", "object"):
        values = values.astype({"float32": np.float32, "integer": np.int64, "object": object}[bad])
    elif bad in ("nan", "inf", "bounds"):
        values = values.copy()
        values[0, 0] = {"nan": np.nan, "inf": np.inf, "bounds": 6.0000000001}[bad]
    elif bad == "shape":
        values = values[:, :2]
    elif bad == "flat":
        values = values.ravel()
    if field == "predictions":
        predictions = values
    else:
        targets = values
    with pytest.raises(ValueError):
        continuous_bc_residuals(predictions, targets, geometry(), ("all",) * 3)


@pytest.mark.parametrize("rows", [0, 257])
def test_should_refuse_row_counts_outside_the_storage_bound(rows):
    spec = ContinuousActionSpec(1, ("m",), (-1.0,), (1.0,))
    actions = np.zeros((rows, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        continuous_bc_residuals(actions, actions, spec, ("all",) * rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_units", ("m",) * 17),
        ("action_units", ["m", "m", "fixed"]),
        ("lower_bounds", (-2.0, 15.0, 4.0)),
        ("upper_bounds", (float("nan"), 14.0, 4.0)),
        ("observation_dim", True),
    ],
)
def test_should_revalidate_corrupted_frozen_specs(field, value):
    spec = geometry()
    object.__setattr__(spec, field, value)
    predictions, targets = action_rows()
    with pytest.raises(ValueError):
        continuous_bc_residuals(predictions, targets, spec, ("all",) * 3)


def test_should_own_immutable_native_report_without_retaining_input_arrays():
    predictions, targets = action_rows()
    report = continuous_bc_residuals(predictions, targets, geometry(), ("all",) * 3)
    before = repr(report)
    predictions.fill(0.0)
    targets.fill(0.0)
    assert repr(report) == before

    def inspect(value):
        if is_dataclass(value):
            assert value.__dataclass_params__.frozen
            for field in fields(value):
                inspect(getattr(value, field.name))
        elif type(value) is tuple:
            for item in value:
                inspect(item)
        else:
            assert type(value) in (str, int, float, bool)

    inspect(report)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_non_train_episodes_before_any_prediction(monkeypatch, role):
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(geometry(), hidden_dim=2))

    def forbidden(*args, **kwargs):
        pytest.fail("non-TRAIN episode reached prediction")

    monkeypatch.setattr(model, "predict", forbidden)
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.diagnose_training_episode(episode(role), ("all",) * 3)
    assert_same_state(before, model.snapshot_state())


def test_should_refuse_probe_train_policy_before_any_prediction(monkeypatch):
    model = NumpyContinuousBCLearner(
        ContinuousBCPolicy(geometry(), hidden_dim=2, training_role="probe_train")
    )
    monkeypatch.setattr(model, "predict", lambda *_: pytest.fail("probe policy reached prediction"))
    with pytest.raises(ValueError):
        model.diagnose_training_episode(episode(), ("all",) * 3)


@pytest.mark.parametrize("bad", ["units", "bounds", "labels", "corrupt_episode"])
def test_should_validate_spec_labels_and_episode_before_prediction(monkeypatch, bad):
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(geometry(), hidden_dim=2))
    declared = geometry()
    if bad == "units":
        declared = replace(declared, action_units=("N", "m", "fixed"))
    elif bad == "bounds":
        declared = replace(declared, lower_bounds=(-3.0, 10.0, 4.0))
    data = episode(spec=declared)
    labels = ("all",) * 3
    if bad == "labels":
        labels = ("all",)
    elif bad == "corrupt_episode":
        object.__setattr__(data.transitions[0], "action", (float("nan"), 12.0, 4.0))
    monkeypatch.setattr(
        model, "predict", lambda *_: pytest.fail("invalid input reached prediction")
    )
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.diagnose_training_episode(data, labels)
    assert_same_state(before, model.snapshot_state())


def test_should_use_one_forward_and_preserve_all_parameters_momentum_policy_and_counters(
    monkeypatch,
):
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(geometry(), hidden_dim=2))
    initial = model.snapshot_state()
    # Populate momentum and counters without spending an optimizer proposal.
    state = replace(
        initial,
        velocity=tuple(np.full_like(value, 0.125) for value in initial.velocity),
        completed_updates=2,
        training_examples=6,
    )
    model.restore_state(state)
    before = model.snapshot_state()
    original = model.predict
    calls = []

    def counted(features):
        calls.append((features.shape, features.dtype))
        return original(features)

    def forbidden(*args, **kwargs):
        pytest.fail("residual diagnosis attempted an optimizer proposal")

    monkeypatch.setattr(model, "predict", counted)
    monkeypatch.setattr(model, "train_batch", forbidden)
    monkeypatch.setattr(model, "train_episode", forbidden)
    result = model.diagnose_training_episode(episode(), ("seek", "close", "seek"))
    assert calls == [((3, 2), np.dtype(np.float64))]
    assert (
        result.prediction_calls,
        result.prediction_rows,
        result.prediction_values,
        result.prediction_bytes,
    ) == (1, 3, 9, 72)
    assert result.residuals.overall.rows == 3
    assert_same_state(before, model.snapshot_state())
