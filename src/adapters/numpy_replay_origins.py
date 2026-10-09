"""Bounded NumPy original-reference, copy-size and payload-integrity ports.

No model prediction/training/copy/restore or consent decisions. SHA256 verifies
an already identity-bound native copy; it never identifies a producer.
"""

from collections import deque
from hashlib import sha256
import json
from math import isfinite

import numpy as np

from src.adapters.numpy_learners import Array, CircadianLearner
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork, ReplaySnapshot


def replay_model_reference(learner: object) -> CircadianPredictiveCodingNetwork:
    if (
        type(learner) is not CircadianLearner
        or type(learner._model) is not CircadianPredictiveCodingNetwork
    ):
        raise ValueError("row origins require the original supported CPC learner/model")
    return learner._model


def retained_replay_references(model: object, maximum: int) -> tuple[object, ...]:
    if (
        type(model) is not CircadianPredictiveCodingNetwork
        or type(model._replay_memory) is not deque
    ):
        raise ValueError("row origins require the native replay deque")
    if type(maximum) is not int or maximum <= 0 or len(model._replay_memory) > maximum:
        raise ValueError("native replay exceeds original reference count bound")
    if any(type(s) is not ReplaySnapshot for s in model._replay_memory):
        raise ValueError("native replay snapshot type is unsupported")
    return tuple(model._replay_memory)


def _arrays(features, targets) -> None:
    if any(
        type(a) is not np.ndarray
        or a.ndim != 2
        or not np.issubdtype(a.dtype, np.number)
        or np.issubdtype(a.dtype, np.complexfloating)
        for a in (features, targets)
    ):
        raise ValueError("row origins require exact real numeric 2D arrays")
    if features.shape[0] == 0 or features.shape[0] != targets.shape[0] or targets.shape[1] != 1:
        raise ValueError("row origin arrays require a nonempty aligned binary batch")


def replay_copy_bytes(features, targets, start: int, count: int, maximum: int) -> int:
    _arrays(features, targets)
    if (
        type(start) is not int
        or type(count) is not int
        or start < 0
        or count <= 0
        or start + count > features.shape[0]
    ):
        raise ValueError("row origin copy range differs from original arrays")
    size = count * (features.shape[1] * features.dtype.itemsize + targets.dtype.itemsize)
    if type(maximum) is not int or not 0 < size <= maximum:
        raise ValueError("row origin copy exceeds payload byte bound")
    return size


def replay_payload_references(snapshot: object) -> tuple[Array, Array]:
    if type(snapshot) is not ReplaySnapshot:
        raise ValueError("row origins require the actual native ReplaySnapshot")
    _arrays(snapshot.input_batch, snapshot.target_batch)
    return snapshot.input_batch, snapshot.target_batch


def replay_payload_fingerprint(snapshot: object, maximum: int) -> tuple[int, str]:
    if type(snapshot) is not ReplaySnapshot:
        raise ValueError("row origins require the actual native ReplaySnapshot")
    features, targets = replay_payload_references(snapshot)
    if any(
        type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1
        for value in (snapshot.priority, snapshot.positive_fraction)
    ):
        raise ValueError("native replay scalar metadata is corrupt")
    size = features.nbytes + targets.nbytes
    if type(maximum) is not int or not 0 < size <= maximum:
        raise ValueError("row origin fingerprint exceeds payload byte bound")
    digest = sha256(b"identity_bound_replay_payload_integrity_v1")
    digest.update(
        json.dumps((snapshot.priority, snapshot.positive_fraction), separators=(",", ":")).encode()
    )
    for array in (features, targets):
        if not array.flags.c_contiguous:
            raise ValueError("native copied replay arrays must be C contiguous")
        metadata = (array.dtype.str, array.shape, array.strides, bool(array.flags.writeable))
        digest.update(json.dumps(metadata, separators=(",", ":")).encode())
        digest.update(array.data.cast("B"))
    return size, digest.hexdigest()
