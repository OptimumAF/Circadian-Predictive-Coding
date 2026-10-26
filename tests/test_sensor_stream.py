"""The recurring public sensor fixture's shape, label law, and ownership."""

import numpy as np

from src.core.sensor_stream import SensorBatch, make_sensor_stream


def test_should_produce_twelve_finite_balanced_four_frame_batches():
    batches = make_sensor_stream()

    assert type(batches) is tuple
    assert len(batches) == 12
    for batch in batches:
        assert type(batch) is SensorBatch
        assert batch.features.shape == (4, 16)
        assert batch.targets.shape == (4, 1)
        assert batch.features.dtype == np.dtype(np.float64)
        assert batch.targets.dtype == np.dtype(np.float64)
        assert np.isfinite(batch.features).all()
        assert np.isfinite(batch.targets).all()
        assert ((0.0 <= batch.features) & (batch.features <= 1.0)).all()
        np.testing.assert_array_equal(batch.targets[:, 0], [0.0, 1.0, 0.0, 1.0])


def test_should_label_every_frame_by_its_public_left_right_contrast():
    for batch in make_sensor_stream():
        frames = batch.features.reshape(4, 4, 4)
        left = frames[:, :, :2].mean(axis=(1, 2))
        right = frames[:, :, 2:].mean(axis=(1, 2))

        assert np.all(np.abs(left - right) > 0.49)
        np.testing.assert_array_equal((right > left).astype(np.float64), batch.targets[:, 0])


def test_should_shift_only_brightness_for_four_middle_batches():
    batches = make_sensor_stream()

    for index in range(4):
        np.testing.assert_array_equal(batches[index].features, batches[0].features)
        np.testing.assert_array_equal(batches[index + 4].features, batches[index].features + 0.25)
        np.testing.assert_array_equal(batches[index + 4].targets, batches[index].targets)
    np.testing.assert_array_equal(np.unique(batches[0].features), [0.1, 0.6])
    np.testing.assert_array_equal(np.unique(batches[4].features), [0.35, 0.85])


def test_should_return_exactly_to_the_original_pixels_and_targets():
    batches = make_sensor_stream()

    for index in range(4):
        np.testing.assert_array_equal(batches[index + 8].features, batches[index].features)
        np.testing.assert_array_equal(batches[index + 8].targets, batches[index].targets)


def test_should_own_every_array_without_cross_batch_or_target_aliases():
    batches = make_sensor_stream()
    arrays = tuple(array for batch in batches for array in (batch.features, batch.targets))

    for index, array in enumerate(arrays):
        assert array.flags.owndata
        for other in arrays[index + 1 :]:
            assert not np.shares_memory(array, other)
    batches[0].features[:] = -1.0
    batches[0].targets[:] = -1.0
    np.testing.assert_array_equal(np.unique(batches[1].features), [0.1, 0.6])
    np.testing.assert_array_equal(batches[1].targets[:, 0], [0.0, 1.0, 0.0, 1.0])


def test_should_reproduce_the_fixture_with_independent_arrays_on_each_call():
    first = make_sensor_stream()
    second = make_sensor_stream()

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.targets, right.targets)
        assert not np.shares_memory(left.features, right.features)
        assert not np.shares_memory(left.targets, right.targets)
    first[4].features[:] = -1.0
    np.testing.assert_array_equal(np.unique(second[4].features), [0.35, 0.85])
