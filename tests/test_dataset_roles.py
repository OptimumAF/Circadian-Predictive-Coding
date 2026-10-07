"""NumPy split-role identities and holdout isolation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.infra.datasets import (
    generate_two_cluster_dataset,
    split_training_validation,
)


def test_validation_split_is_deterministic_disjoint_and_exhaustive() -> None:
    source = generate_two_cluster_dataset(sample_count=100, noise_scale=0.8, seed=7)
    first = split_training_validation(source, validation_fraction=0.2, seed=11)
    second = split_training_validation(source, validation_fraction=0.2, seed=11)
    changed_seed = split_training_validation(source, validation_fraction=0.2, seed=12)

    assert dict(first.split_hashes) == dict(second.split_hashes)
    assert first.split_hashes["validation"] != changed_seed.split_hashes["validation"]
    assert first.train.input.shape[0] + first.validation.input.shape[0] == source.train_input.shape[0]
    assert first.test.input.shape[0] == source.test_input.shape[0]
    assert len(set(first.split_hashes.values())) == 3
    train_rows = {tuple(row) for row in first.train.input.tolist()}
    validation_rows = {tuple(row) for row in first.validation.input.tolist()}
    test_rows = {tuple(row) for row in first.test.input.tolist()}
    assert train_rows.isdisjoint(validation_rows)
    assert train_rows.isdisjoint(test_rows)
    assert validation_rows.isdisjoint(test_rows)
    assert len(train_rows | validation_rows) == source.train_input.shape[0]
    assert set(first.train.target.reshape(-1)) == {0.0, 1.0}
    assert set(first.validation.target.reshape(-1)) == {0.0, 1.0}


def test_final_test_label_change_only_changes_final_test_hash() -> None:
    source = generate_two_cluster_dataset(sample_count=60, noise_scale=0.8, seed=7)
    original = split_training_validation(source, validation_fraction=0.2, seed=11)
    shifted_source = replace(source, test_target=1.0 - source.test_target)
    shifted = split_training_validation(shifted_source, validation_fraction=0.2, seed=11)

    assert original.split_hashes["train"] == shifted.split_hashes["train"]
    assert original.split_hashes["validation"] == shifted.split_hashes["validation"]
    assert original.split_hashes["test"] != shifted.split_hashes["test"]
    with pytest.raises(TypeError):
        cast(Any, original.split_hashes)["test"] = "overwritten"
    assert np.array_equal(original.train.input, shifted.train.input)


def test_validation_split_rejects_invalid_fraction() -> None:
    source = generate_two_cluster_dataset(sample_count=40, noise_scale=0.8, seed=7)
    with pytest.raises(ValueError, match="validation_fraction"):
        split_training_validation(source, validation_fraction=1.0, seed=11)
