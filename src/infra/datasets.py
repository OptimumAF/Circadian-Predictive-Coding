"""Synthetic dataset generation for baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test split for binary classification."""

    train_input: Array
    train_target: Array
    test_input: Array
    test_target: Array


def generate_two_cluster_dataset(
    sample_count: int,
    noise_scale: float,
    seed: int,
    test_ratio: float = 0.2,
) -> DatasetSplit:
    """Create a deterministic two-cluster binary classification dataset."""
    if sample_count < 20:
        raise ValueError("sample_count must be at least 20")
    if noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive")
    if test_ratio <= 0.0 or test_ratio >= 0.5:
        raise ValueError("test_ratio must be between 0 and 0.5")

    rng = np.random.default_rng(seed)
    class_size = sample_count // 2

    class_zero = rng.normal(loc=(-1.2, -1.0), scale=noise_scale, size=(class_size, 2))
    class_one = rng.normal(loc=(1.2, 1.0), scale=noise_scale, size=(class_size, 2))

    input_data = np.vstack([class_zero, class_one]).astype(np.float64)
    target_data = np.concatenate(
        [np.zeros(class_size, dtype=np.float64), np.ones(class_size, dtype=np.float64)]
    ).reshape(-1, 1)

    permutation = rng.permutation(input_data.shape[0])
    input_data = input_data[permutation]
    target_data = target_data[permutation]

    split_index = int((1.0 - test_ratio) * input_data.shape[0])
    return DatasetSplit(
        train_input=input_data[:split_index],
        train_target=target_data[:split_index],
        test_input=input_data[split_index:],
        test_target=target_data[split_index:],
    )

