"""Independent fixed-data partitions; no native model or inference execution."""

import numpy as np
import pytest

import src.adapters.coupled_attribution_pilot as pilot


def assert_same_partition(left, right):
    assert len(left) == len(right)
    for first, second in zip(left, right):
        assert (first.role, first.context, first.row_ids) == (
            second.role,
            second.context,
            second.row_ids,
        )
        assert first.features.tobytes() == second.features.tobytes()
        assert first.targets.tobytes() == second.targets.tobytes()
        assert first.features.dtype == second.features.dtype == np.float64
        assert first.targets.dtype == second.targets.dtype == np.float64


@pytest.mark.parametrize("stream", ["stationary", "recurring"])
def test_should_preserve_default_arrays_labels_ids_and_order_against_manual_old_recipe(stream):
    training, final = pilot.build_stream(stream)
    explicit_training, explicit_final = pilot.build_stream(
        stream, training_rng=7301, heldout_rng=9101
    )
    assert_same_partition(training, explicit_training)
    assert_same_partition(final, explicit_final)
    train_generator = np.random.default_rng(7301)
    final_generator = np.random.default_rng(9101)
    contexts = ("A", "A", "A") if stream == "stationary" else ("A", "B", "A")
    assert len(training) == 18 and len(final) == 2
    for index, batch in enumerate(training):
        context = contexts[index // 6]
        features = train_generator.uniform(-1, 1, (8, 2))
        targets = (features[:, [0 if context == "A" else 1]] >= 0).astype(np.float64)
        assert batch.role == "train" and batch.context == context
        assert batch.row_ids == tuple(f"train-{stream}-{index}-{row}" for row in range(8))
        assert batch.features.tobytes() == features.tobytes()
        assert batch.targets.tobytes() == targets.tobytes()
    for column, batch in enumerate(final):
        context = ("A", "B")[column]
        features = final_generator.uniform(-1, 1, (64, 2))
        targets = (features[:, [column]] >= 0).astype(np.float64)
        assert batch.role == "final_test" and batch.context == context
        assert batch.row_ids == tuple(f"heldout-{context}-{row}" for row in range(64))
        assert batch.features.tobytes() == features.tobytes()
        assert batch.targets.tobytes() == targets.tobytes()


@pytest.mark.parametrize("stream", ["stationary", "recurring"])
def test_should_change_only_the_selected_partition_and_identify_its_rng(stream):
    original_training, original_final = pilot.build_stream(stream)
    new_training, stable_final = pilot.build_stream(stream, training_rng=7302)
    stable_training, new_final = pilot.build_stream(stream, heldout_rng=9102)
    both_training, both_final = pilot.build_stream(stream, training_rng=7302, heldout_rng=9102)
    assert_same_partition(stable_final, original_final)
    assert_same_partition(stable_training, original_training)
    assert_same_partition(both_training, new_training)
    assert_same_partition(both_final, new_final)
    for index, (before, after) in enumerate(zip(original_training, new_training)):
        assert not np.array_equal(before.features, after.features)
        assert after.row_ids == tuple(f"train-{stream}-rng7302-{index}-{row}" for row in range(8))
    for before, after in zip(original_final, new_final):
        assert not np.array_equal(before.features, after.features)
        assert after.row_ids == tuple(f"heldout-rng9102-{after.context}-{row}" for row in range(64))


def test_should_reproduce_declared_data_with_owned_disjoint_arrays_and_correct_labels():
    first = pilot.build_stream("recurring", training_rng=7302, heldout_rng=9102)
    second = pilot.build_stream("recurring", training_rng=7302, heldout_rng=9102)
    for left, right in zip(first, second):
        assert_same_partition(left, right)
    all_batches = (*first[0], *first[1], *second[0], *second[1])
    arrays = [value for batch in all_batches for value in (batch.features, batch.targets)]
    for index, array in enumerate(arrays):
        assert array.flags.owndata and array.dtype == np.float64
        assert np.isfinite(array).all()
        for other in arrays[index + 1 :]:
            assert not np.shares_memory(array, other)
    for batch in all_batches:
        assert batch.features.shape == (8 if batch.role == "train" else 64, 2)
        assert batch.targets.shape == (len(batch.features), 1)
        assert ((-1 <= batch.features) & (batch.features <= 1)).all()
        np.testing.assert_array_equal(
            batch.targets, batch.features[:, [0 if batch.context == "A" else 1]] >= 0
        )
    train_ids = {identifier for batch in first[0] for identifier in batch.row_ids}
    final_ids = {identifier for batch in first[1] for identifier in batch.row_ids}
    assert len(train_ids) == 144 and len(final_ids) == 128 and train_ids.isdisjoint(final_ids)
    first[0][0].features.fill(99)
    first[1][0].targets.fill(99)
    assert (second[0][0].features <= 1).all()
    assert (second[1][0].targets <= 1).all()


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
@pytest.mark.parametrize("name", ["training_rng", "heldout_rng"])
def test_should_accept_native_integer_rng_endpoints(name, seed):
    arguments = dict(training_rng=1, heldout_rng=1)
    arguments[name] = seed
    training, final = pilot.build_stream("recurring", **arguments)
    assert len(training) == 18 and len(final) == 2
    if name == "training_rng":
        assert training[0].row_ids[0] == f"train-recurring-rng{seed}-0-0"
    else:
        assert final[0].row_ids[0] == f"heldout-rng{seed}-A-0"


@pytest.mark.parametrize("name", ["training_rng", "heldout_rng"])
@pytest.mark.parametrize("invalid", [True, False, None, -1, 2**32, 1.0, "7302", np.int64(7302)])
def test_should_refuse_invalid_rng_before_generators_or_batches(monkeypatch, name, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid seed reached RNG or batch allocation")

    monkeypatch.setattr(pilot.np.random, "default_rng", forbidden)
    monkeypatch.setattr(pilot, "AttributionBatch", forbidden)
    with pytest.raises(ValueError, match="native integer"):
        pilot.build_stream("recurring", **{name: invalid})


@pytest.mark.parametrize("seed", [0, 7301, 9101, 2**32 - 1])
def test_should_refuse_shared_partition_seed_before_generators_or_batches(monkeypatch, seed):
    def forbidden(*args, **kwargs):
        pytest.fail("shared partition seed reached RNG or batch allocation")

    monkeypatch.setattr(pilot.np.random, "default_rng", forbidden)
    monkeypatch.setattr(pilot, "AttributionBatch", forbidden)
    with pytest.raises(ValueError, match="distinct"):
        pilot.build_stream("recurring", training_rng=seed, heldout_rng=seed)
