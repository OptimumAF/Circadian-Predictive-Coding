"""Fixed camera-negative sensor fixtures; no model calls or data-role authority."""

import json

import numpy as np
import pytest

import src.core.sensor_retention as retention
import src.core.sensor_stream as stream
from src.core.sensor_retention import describe_sensor_heldout, make_sensor_heldout
from src.core.sensor_stream import make_sensor_stream


def contrast_labels(features):
    frames = features.reshape(-1, 4, 4)
    return (
        (frames[:, :, 2:].mean(axis=(1, 2)) > frames[:, :, :2].mean(axis=(1, 2)))
        .astype(np.float64)
        .reshape(-1, 1)
    )


def assert_owned_bounded(batches):
    arrays = [value for batch in batches for value in (batch.features, batch.targets)]
    for index, value in enumerate(arrays):
        assert value.dtype == np.float64 and value.flags.owndata
        assert np.isfinite(value).all() and ((0 <= value) & (value <= 1)).all()
        for other in arrays[index + 1 :]:
            assert not np.shares_memory(value, other)


def test_should_invert_only_b_pixels_and_preserve_semantic_labels_and_exact_a_returns():
    original = make_sensor_stream()
    inverted = make_sensor_stream(contrast_inversion=True)
    assert len(inverted) == 12
    for tick, batch in enumerate(inverted):
        assert batch.features.shape == (4, 16) and batch.targets.shape == (4, 1)
        np.testing.assert_array_equal(batch.targets, original[tick].targets)
        if 4 <= tick < 8:
            np.testing.assert_array_equal(batch.features, 1.0 - original[0].features)
            np.testing.assert_array_equal(batch.targets, 1.0 - contrast_labels(batch.features))
            assert not np.array_equal(batch.features, original[tick].features)
        else:
            assert batch.features.tobytes() == original[tick].features.tobytes()
            np.testing.assert_array_equal(batch.targets, contrast_labels(batch.features))
    assert_owned_bounded((*original, *inverted))


def test_should_keep_frozen_noisy_a_and_pair_its_camera_negative_with_original_labels():
    original_a, original_b = make_sensor_heldout()
    a, b = make_sensor_heldout(contrast_inversion=True)
    assert a.features.shape == b.features.shape == (32, 16)
    assert a.targets.shape == b.targets.shape == (32, 1)
    assert a.features.tobytes() == original_a.features.tobytes()
    assert a.targets.tobytes() == original_a.targets.tobytes()
    np.testing.assert_array_equal(b.features, 1.0 - a.features)
    np.testing.assert_array_equal(b.targets, a.targets)
    np.testing.assert_array_equal(a.targets, contrast_labels(a.features))
    np.testing.assert_array_equal(b.targets, 1.0 - contrast_labels(b.features))
    np.testing.assert_array_equal(a.targets[:, 0], np.arange(32) % 2)
    assert not np.array_equal(b.features, original_b.features)
    assert_owned_bounded((original_a, original_b, a, b))
    assert (
        sum(value.nbytes for batch in (a, b) for value in (batch.features, batch.targets)) == 8704
    )


def test_should_make_independent_deterministic_training_and_final_arrays_across_calls():
    training = make_sensor_stream(contrast_inversion=True)
    repeated_training = make_sensor_stream(contrast_inversion=True)
    final = make_sensor_heldout(contrast_inversion=True)
    repeated_final = make_sensor_heldout(contrast_inversion=True)
    for first, second in zip((*training, *final), (*repeated_training, *repeated_final)):
        assert first.features.tobytes() == second.features.tobytes()
        assert first.targets.tobytes() == second.targets.tobytes()
    assert_owned_bounded((*training, *repeated_training, *final, *repeated_final))
    for batch in final:
        assert all(
            not np.array_equal(row, train_row)
            for row in batch.features
            for train in training
            for train_row in train.features
        )
    training[4].features.fill(99)
    training[4].targets.fill(99)
    final[0].features.fill(99)
    final[0].targets.fill(99)
    np.testing.assert_array_equal(repeated_training[4].features, repeated_training[5].features)
    np.testing.assert_array_equal(repeated_final[1].features, 1.0 - repeated_final[0].features)
    np.testing.assert_array_equal(final[1].targets, repeated_final[1].targets)


def test_should_preserve_default_brightness_arrays_and_complete_heldout_description():
    for builder in (make_sensor_stream, make_sensor_heldout):
        default, explicit = builder(), builder(contrast_inversion=False)
        for left, right in zip(default, explicit):
            assert left.features.tobytes() == right.features.tobytes()
            assert left.targets.tobytes() == right.targets.tobytes()
    for left, right in zip(
        make_sensor_stream(stationary=True),
        make_sensor_stream(stationary=True, contrast_inversion=False),
    ):
        assert left.features.tobytes() == right.features.tobytes()
        assert left.targets.tobytes() == right.targets.tobytes()
    frames = np.full((32, 4, 4), 0.1, dtype=np.float64)
    frames[0::2, :, :2] = 0.6
    frames[1::2, :, 2:] = 0.6
    expected_a = frames.reshape(32, 16) + np.random.default_rng(27031).uniform(
        -0.025, 0.025, (32, 16)
    )
    a, b = make_sensor_heldout()
    assert a.features.tobytes() == expected_a.tobytes()
    assert b.features.tobytes() == (expected_a + 0.25).tobytes()
    expected_descriptor = dict(
        seed=27031,
        noise_uniform=(-0.025, 0.025),
        contexts=("A", "B"),
        rows_per_context=32,
        feature_width=16,
        brightness_shift=0.25,
        numeric_bytes=8704,
        checkpoint_updates=tuple(range(13)),
        summary_updates=(0, 4, 8, 12),
        readout_forward_calls=26,
        readout_forward_rows=832,
        metric_directions={"bce": "lower", "accuracy": "higher"},
        gain_direction="positive_is_better",
        loss_and_gap_direction="positive_is_worse",
    )
    assert describe_sensor_heldout() == describe_sensor_heldout(contrast_inversion=False)
    assert json.dumps(describe_sensor_heldout()) == json.dumps(expected_descriptor)


def test_should_describe_camera_negative_without_brightness_claim_or_cost_changes():
    default = describe_sensor_heldout()
    inverted = describe_sensor_heldout(contrast_inversion=True)
    expected = dict(default)
    del expected["brightness_shift"]
    expected.update(
        contrast_inversion=True,
        transform="1 - pixels; semantic labels unchanged",
    )
    assert inverted == expected
    inverted["metric_directions"]["bce"] = "forged"
    assert describe_sensor_heldout(contrast_inversion=True) == expected


@pytest.mark.parametrize("invalid", [None, 0, 1, "yes", np.bool_(True), [], {}])
def test_should_reject_invalid_contrast_modes_before_allocating_or_using_rng(monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid fixture mode reached allocation or RNG")

    monkeypatch.setattr(stream.np, "array", forbidden)
    monkeypatch.setattr(stream.np, "full", forbidden)
    monkeypatch.setattr(retention.np.random, "default_rng", forbidden)
    for builder in (make_sensor_stream, make_sensor_heldout, describe_sensor_heldout):
        with pytest.raises(ValueError, match="exact boolean"):
            builder(contrast_inversion=invalid)


def test_should_reject_stationary_and_contrast_combination_before_allocating(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible fixture modes reached allocation")

    monkeypatch.setattr(stream.np, "array", forbidden)
    monkeypatch.setattr(stream.np, "full", forbidden)
    with pytest.raises(ValueError, match="cannot both"):
        make_sensor_stream(stationary=True, contrast_inversion=True)
