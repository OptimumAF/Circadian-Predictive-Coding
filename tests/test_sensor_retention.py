"""Fixed heldout sensor data and signed retention summaries, without learners."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.core.sensor_retention import (
    describe_sensor_heldout,
    make_sensor_heldout,
    summarize_sensor_retention,
)
from src.core.sensor_stream import SensorBatch, make_sensor_stream


def _points() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "updates": updates,
            "training_context": None
            if updates == 0
            else "A"
            if updates <= 4 or updates > 8
            else "B",
            "scores": {
                "A": {"bce": 1.0, "accuracy": 0.5},
                "B": {"bce": 1.0, "accuracy": 0.5},
            },
        }
        for updates in range(13)
    )


def test_should_build_balanced_owned_32_row_float64_heldout_batches():
    batches = make_sensor_heldout()

    assert type(batches) is tuple and len(batches) == 2
    for batch in batches:
        assert type(batch) is SensorBatch
        assert batch.features.shape == (32, 16)
        assert batch.targets.shape == (32, 1)
        assert batch.features.dtype == batch.targets.dtype == np.dtype(np.float64)
        assert np.isfinite(batch.features).all()
        assert ((0 <= batch.features) & (batch.features <= 1)).all()
        np.testing.assert_array_equal(batch.targets[:, 0], np.arange(32) % 2)
    arrays = [value for batch in batches for value in (batch.features, batch.targets)]
    assert sum(value.nbytes for value in arrays) == 8704
    for index, value in enumerate(arrays):
        assert value.flags.owndata
        for other in arrays[index + 1 :]:
            assert not np.shares_memory(value, other)


def test_should_preserve_left_right_rule_and_paired_brightness_with_fixed_noise():
    baseline, shifted = make_sensor_heldout()
    base = np.full((32, 4, 4), 0.1, dtype=np.float64)
    base[0::2, :, :2] = 0.6
    base[1::2, :, 2:] = 0.6
    expected = base.reshape(32, 16) + np.random.default_rng(27031).uniform(-0.025, 0.025, (32, 16))

    np.testing.assert_array_equal(baseline.features, expected)
    np.testing.assert_array_equal(shifted.features, baseline.features + 0.25)
    for batch in (baseline, shifted):
        frames = batch.features.reshape(32, 4, 4)
        right = frames[:, :, 2:].mean(axis=(1, 2))
        left = frames[:, :, :2].mean(axis=(1, 2))
        np.testing.assert_array_equal((right > left).astype(np.float64), batch.targets[:, 0])


def test_should_reproduce_heldout_data_without_aliasing_other_calls_or_training():
    first = make_sensor_heldout()
    second = make_sensor_heldout()
    training = make_sensor_stream()

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.targets, right.targets)
        assert not np.shares_memory(left.features, right.features)
        assert not np.shares_memory(left.targets, right.targets)
        for batch in training:
            assert not np.shares_memory(left.features, batch.features)
            assert not np.shares_memory(left.targets, batch.targets)
            assert all(
                not np.array_equal(row, train_row)
                for row in left.features
                for train_row in batch.features
            )
    first[0].features[:] = -1
    first[0].targets[:] = -1
    assert (second[0].features > 0).all()
    np.testing.assert_array_equal(first[1].targets, second[1].targets)


def test_should_describe_frozen_fixture_costs_and_opposite_metric_directions():
    description = describe_sensor_heldout()

    assert description["seed"] == 27031
    assert description["noise_uniform"] == (-0.025, 0.025)
    assert description["rows_per_context"] == 32
    assert description["contexts"] == ("A", "B")
    assert description["numeric_bytes"] == 8704
    assert description["checkpoint_updates"] == tuple(range(13))
    assert description["readout_forward_calls"] == 26
    assert description["readout_forward_rows"] == 832
    assert description["metric_directions"] == {"bce": "lower", "accuracy": "higher"}
    description["metric_directions"]["bce"] = "changed"
    assert describe_sensor_heldout()["metric_directions"]["bce"] == "lower"


def test_should_report_signed_gains_losses_and_gap_without_clipping_negatives():
    points = _points()
    for index, (a_bce, b_bce, a_accuracy, b_accuracy) in {
        0: (1.0, 1.0, 0.5, 0.5),
        4: (0.6, 0.9, 0.8, 0.6),
        8: (0.5, 1.1, 0.9, 0.4),
        12: (0.7, 1.0, 0.7, 0.5),
    }.items():
        points[index]["scores"] = {
            "A": {"bce": a_bce, "accuracy": a_accuracy},
            "B": {"bce": b_bce, "accuracy": b_accuracy},
        }

    result = summarize_sensor_retention(points)

    assert tuple(point["updates"] for point in result["endpoints"]) == (0, 4, 8, 12)
    assert result["metrics"]["bce"] == pytest.approx(
        {
            "initial_A_gain": 0.4,
            "new_B_gain": -0.2,
            "old_A_loss": -0.1,
            "return_A_gain": -0.2,
            "return_gap": 0.1,
        }
    )
    assert result["metrics"]["accuracy"] == pytest.approx(
        {
            "initial_A_gain": 0.3,
            "new_B_gain": -0.2,
            "old_A_loss": -0.1,
            "return_A_gain": -0.2,
            "return_gap": 0.1,
        }
    )


def test_should_keep_summary_independent_of_input_mutation():
    points = _points()
    original = deepcopy(points)

    result = summarize_sensor_retention(points)

    assert points == original
    points[0]["scores"]["A"]["bce"] = 3.0
    assert result["endpoints"][0]["scores"]["A"]["bce"] == 1.0
    result["endpoints"][1]["scores"]["A"]["accuracy"] = 0.0
    assert points[4]["scores"]["A"]["accuracy"] == 0.5


def test_should_preserve_negative_return_gap_when_return_exceeds_first_phase_endpoint():
    points = _points()
    points[4]["scores"]["A"] = {"bce": 0.6, "accuracy": 0.8}
    points[8]["scores"]["A"] = {"bce": 0.7, "accuracy": 0.7}
    points[12]["scores"]["A"] = {"bce": 0.4, "accuracy": 1.0}

    result = summarize_sensor_retention(points)

    for metric in ("bce", "accuracy"):
        assert result["metrics"][metric]["old_A_loss"] == pytest.approx(0.1)
        assert result["metrics"][metric]["return_A_gain"] == pytest.approx(0.3)
        assert result["metrics"][metric]["return_gap"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "out_of_order",
        "bool_update",
        "bad_context",
        "missing_context",
        "extra_context",
        "missing_metric",
        "extra_metric",
        "extra_row_key",
        "bool_metric",
        "negative_bce",
        "nan",
        "infinite",
        "low_accuracy",
        "high_accuracy",
    ],
)
def test_should_refuse_incomplete_malformed_or_nonfinite_checkpoint_evidence(change):
    points = _points()
    if change == "missing":
        points = points[:-1]
    elif change == "duplicate":
        points[5]["updates"] = 4
    elif change == "out_of_order":
        points = points[:4] + (points[5], points[4]) + points[6:]
    elif change == "bool_update":
        points[0]["updates"] = False
    elif change == "bad_context":
        points[5]["training_context"] = "A"
    elif change == "missing_context":
        del points[3]["scores"]["B"]
    elif change == "extra_context":
        points[3]["scores"]["C"] = {"bce": 0.0, "accuracy": 1.0}
    elif change == "missing_metric":
        del points[3]["scores"]["A"]["bce"]
    elif change == "extra_metric":
        points[3]["scores"]["A"]["selected"] = 0.0
    elif change == "extra_row_key":
        points[3]["selected"] = True
    else:
        metric, value = {
            "bool_metric": ("bce", False),
            "negative_bce": ("bce", -0.1),
            "nan": ("bce", float("nan")),
            "infinite": ("accuracy", float("inf")),
            "low_accuracy": ("accuracy", -0.1),
            "high_accuracy": ("accuracy", 1.1),
        }[change]
        points[3]["scores"]["A"][metric] = value

    with pytest.raises(ValueError):
        summarize_sensor_retention(points)


def test_should_require_a_tuple_of_checkpoint_rows():
    with pytest.raises(ValueError):
        summarize_sensor_retention(list(_points()))  # type: ignore[arg-type]
