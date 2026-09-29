"""Synthetic dataset generation for baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

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


@dataclass(frozen=True)
class LabeledData:
    """One explicit data role; pass only permitted roles to training code."""

    input: Array
    target: Array


@dataclass(frozen=True)
class DeferredFinalTestRole:
    """Carry a source reference without releasing final-test arrays early."""

    _source: DatasetSplit

    @property
    def input(self) -> Array:
        return self._source.test_input

    @property
    def target(self) -> Array:
        return self._source.test_target


@dataclass(frozen=True)
class RoleSeparatedDataset:
    """Deterministic train/validation/final-test roles and content hashes."""

    train: LabeledData
    validation: LabeledData
    test: LabeledData | DeferredFinalTestRole
    split_hashes: Mapping[str, str]


def make_role_separated_dataset(
    train: LabeledData,
    validation: LabeledData,
    test: LabeledData | DeferredFinalTestRole,
    *,
    hash_test: bool = True,
) -> RoleSeparatedDataset:
    """Build roles, optionally deferring final-test validation and hashing."""
    if type(hash_test) is not bool:
        raise ValueError("hash_test must be a bool")
    for role, samples in (("train", train), ("validation", validation), ("test", test)):
        if role == "test" and not hash_test:
            continue  # Why this: later-phase training must not inspect held-out labels.
        if samples.input.ndim != 2 or samples.target.shape != (samples.input.shape[0], 1):
            raise ValueError(f"{role} inputs and targets have incompatible shapes")
        if samples.input.shape[0] == 0:
            raise ValueError(f"{role} split must be nonempty")
    hashes = {
        "train": _split_hash("train", train),
        "validation": _split_hash("validation", validation),
    }
    if hash_test:
        hashes["test"] = _split_hash("test", test)
    return RoleSeparatedDataset(
        train=train, validation=validation, test=test, split_hashes=MappingProxyType(hashes)
    )


def split_training_validation(
    dataset: DatasetSplit,
    validation_fraction: float,
    seed: int,
    *,
    hash_test: bool = True,
    defer_test_access: bool = False,
) -> RoleSeparatedDataset:
    """Reserve stratified validation examples from an existing training split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if type(defer_test_access) is not bool or (defer_test_access and hash_test):
        raise ValueError("defer_test_access requires hash_test=False")
    labels = dataset.train_target.reshape(-1)
    if labels.shape[0] != dataset.train_input.shape[0]:
        raise ValueError("training inputs and targets have incompatible shapes")
    if not np.all(np.isin(labels, [0.0, 1.0])):
        raise ValueError("training targets must be binary")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_label in (0.0, 1.0):
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size < 2:
            raise ValueError("each class needs at least two training examples")
        shuffled = rng.permutation(class_indices)
        validation_count = max(
            1, min(class_indices.size - 1, round(class_indices.size * validation_fraction))
        )
        validation_indices.extend(shuffled[:validation_count].tolist())
        train_indices.extend(shuffled[validation_count:].tolist())

    # Preserve the original generated order within each role for replay/order audits.
    train_rows = np.array(sorted(train_indices), dtype=np.int64)
    validation_rows = np.array(sorted(validation_indices), dtype=np.int64)
    return make_role_separated_dataset(
        train=LabeledData(dataset.train_input[train_rows], dataset.train_target[train_rows]),
        validation=LabeledData(
            dataset.train_input[validation_rows], dataset.train_target[validation_rows]
        ),
        # Why this: a sealed source may withhold held-out values until every
        # training decision has completed; carrying its reference is safe.
        test=DeferredFinalTestRole(dataset)
        if defer_test_access
        else LabeledData(dataset.test_input, dataset.test_target),
        hash_test=hash_test,
    )


def _split_hash(role: str, samples: LabeledData | DeferredFinalTestRole) -> str:
    digest = sha256(role.encode("utf-8"))
    for values in (samples.input, samples.target):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def generate_two_cluster_dataset(
    sample_count: int,
    noise_scale: float,
    seed: int,
    test_ratio: float = 0.2,
) -> DatasetSplit:
    """Create a deterministic two-cluster binary classification dataset."""
    return generate_two_cluster_dataset_with_transform(
        sample_count=sample_count,
        noise_scale=noise_scale,
        seed=seed,
        test_ratio=test_ratio,
    )


def generate_two_cluster_dataset_with_transform(
    sample_count: int,
    noise_scale: float,
    seed: int,
    test_ratio: float = 0.2,
    class_zero_center: tuple[float, float] = (-1.2, -1.0),
    class_one_center: tuple[float, float] = (1.2, 1.0),
    rotation_degrees: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
) -> DatasetSplit:
    """Create a deterministic two-cluster dataset with optional affine transform."""
    if sample_count < 20:
        raise ValueError("sample_count must be at least 20")
    if noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive")
    if test_ratio <= 0.0 or test_ratio >= 0.5:
        raise ValueError("test_ratio must be between 0 and 0.5")

    rng = np.random.default_rng(seed)
    class_size = sample_count // 2

    class_zero = rng.normal(loc=class_zero_center, scale=noise_scale, size=(class_size, 2))
    class_one = rng.normal(loc=class_one_center, scale=noise_scale, size=(class_size, 2))

    input_data = np.vstack([class_zero, class_one]).astype(np.float64)
    if abs(rotation_degrees) > 1e-12:
        radians = np.deg2rad(rotation_degrees)
        rotation_matrix = np.array(
            [
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ],
            dtype=np.float64,
        )
        input_data = input_data @ rotation_matrix.T
    input_data += np.array(translation, dtype=np.float64)

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
