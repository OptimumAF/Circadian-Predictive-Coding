"""Bounded local integrity stamps for already admitted native and inbox graphs.

Read exact supported records and array buffers; return a digest without payload
copies. This detects later mutation, not provenance or capture/restore authority.
No user callback, pickle, model method, persistence or consent check is invoked.
"""

from collections import deque
from dataclasses import dataclass, fields
from hashlib import sha256
from math import isfinite
from typing import Any

import numpy as np

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    ReplayRetentionBudget,
    ReplaySnapshot,
)
from src.core.data_erasure import ErasedExperience
from src.core.experience import AppliedExperience, Experience, ExperiencePermissions, LabelArrival
from src.core.inbox_cursor import InboxCursor
from src.core.learner_ports import TrainingDiagnostic
from src.core.replay_retention import ReplayRetentionPolicy

_RECORDS = (
    CircadianConfig,
    CircadianNetworkSnapshot,
    ReplayRetentionBudget,
    ReplaySnapshot,
    ReplayRetentionPolicy,
    InboxCursor,
    Experience,
    LabelArrival,
    AppliedExperience,
    ExperiencePermissions,
    TrainingDiagnostic,
    ErasedExperience,
)


def _exact_type(kind: type, *supported: type) -> bool:
    # A caller's metaclass may overload equality/hash. Identity never calls it.
    return any(kind is match for match in supported)


@dataclass(frozen=True)
class CheckpointContentLimits:
    max_nodes: int
    max_metadata_bytes: int
    max_array_bytes: int
    max_depth: int

    def __post_init__(self) -> None:
        data = vars(self)
        if (
            len(data) != 4
            or any(type(key) is not str for key in data)
            or data.keys() != {"max_nodes", "max_metadata_bytes", "max_array_bytes", "max_depth"}
            or any(type(n) is not int or not 0 < n < 2**63 for n in data.values())
        ):
            raise ValueError("checkpoint content limits require positive bounded integers")


def checkpoint_content_stamp(value: Any, limits: CheckpointContentLimits) -> str:
    """Stamp contents and nested identities, excluding the root wrapper identity.

    Why this: native restore installs the actual copied dictionary's values in
    the model's existing dictionary. Its wrapper differs; nested identities must
    remain the same. Bounds and exact types precede traversal and buffer hashing.
    """
    if type(limits) is not CheckpointContentLimits:
        raise ValueError("checkpoint content requires exact original limits")
    CheckpointContentLimits.__post_init__(limits)
    proof = _ContentProof(limits)
    proof.visit(value, 0, identities=False)
    return proof.digest.hexdigest()


class _ContentProof:
    def __init__(self, limits: CheckpointContentLimits):
        self.limits = limits
        self.digest = sha256(b"original_checkpoint_content_v1")
        self.nodes = self.metadata = self.array_bytes = 0
        self.active: set[int] = set()

    def metadata_token(self, raw: bytes) -> None:
        self.metadata += len(raw) + 8
        if self.metadata > self.limits.max_metadata_bytes:
            raise ValueError("checkpoint content metadata bound exceeded")
        self.digest.update(len(raw).to_bytes(8, "little"))
        self.digest.update(raw)

    def scalar(self, value: Any) -> None:
        kind = type(value)
        if kind is int and value.bit_length() > 1024:
            raise ValueError("checkpoint content integer bound exceeded")
        if kind is str and len(value) * 10 + 16 > self.limits.max_metadata_bytes - self.metadata:
            raise ValueError("checkpoint content string bound exceeded")
        if kind is float and not isfinite(value):
            raise ValueError("checkpoint content requires finite scalars")
        self.metadata_token((kind.__name__ + ":" + repr(value)).encode("utf8"))

    def visit(self, value: Any, depth: int, *, identities: bool = True) -> None:
        self.nodes += 1
        if self.nodes > self.limits.max_nodes or depth > self.limits.max_depth:
            raise ValueError("checkpoint content node/depth bound exceeded")
        kind = type(value)
        if value is None or _exact_type(kind, bool, int, float, str):
            self.scalar(value)
            return
        if not _exact_type(
            kind,
            dict,
            list,
            tuple,
            deque,
            set,
            frozenset,
            np.ndarray,
            np.random.Generator,
            *_RECORDS,
        ):
            raise ValueError("unsupported exact checkpoint content type")
        if id(value) in self.active:
            raise ValueError("cyclic checkpoint content is unsupported")
        self.metadata_token(kind.__name__.encode("ascii"))
        if identities or depth > 0:
            self.metadata_token(str(id(value)).encode("ascii"))
        self.active.add(id(value))
        try:
            if kind is np.ndarray:
                self.array(value)
            elif kind is np.random.Generator:
                self.generator(value, depth)
            elif _exact_type(kind, *_RECORDS):
                self.record(value, depth)
            else:
                self.collection(value, depth)
        finally:
            self.active.remove(id(value))

    def collection(self, value: Any, depth: int) -> None:
        if len(value) > self.limits.max_nodes - self.nodes:
            raise ValueError("checkpoint content collection bound exceeded")
        self.scalar(len(value))
        if type(value) is deque:
            self.scalar(value.maxlen)
        if type(value) is dict:
            for key, item in value.items():
                if not _exact_type(type(key), str, int, tuple):
                    raise ValueError("unsupported exact checkpoint dictionary key")
                self.visit(key, depth + 1)
                self.visit(item, depth + 1)
        elif _exact_type(type(value), set, frozenset):
            if any(type(item) is not str for item in value):
                raise ValueError("checkpoint content sets require exact strings")
            if (
                sum(len(item) * 10 + 16 for item in value)
                > self.limits.max_metadata_bytes - self.metadata
            ):
                raise ValueError("checkpoint content set string bound exceeded")
            for item in sorted(value):
                self.visit(item, depth + 1)
        else:
            for item in value:
                self.visit(item, depth + 1)

    def record(self, value: Any, depth: int) -> None:
        names = tuple(field.name for field in fields(type(value)))
        data = object.__getattribute__(value, "__dict__")
        if (
            type(data) is not dict
            or len(data) != len(names)
            or any(type(key) is not str for key in data)
            or data.keys() != set(names)
        ):
            raise ValueError("checkpoint content record fields changed")
        for name in names:
            self.scalar(name)
            self.visit(data[name], depth + 1)

    def array(self, value: Any) -> None:
        if (
            value.dtype.kind not in "biuf"
            or value.dtype.metadata is not None
            or not (value.flags.c_contiguous or value.flags.f_contiguous)
        ):
            raise ValueError("checkpoint content requires exact contiguous numeric arrays")
        self.array_bytes += value.nbytes
        if self.array_bytes > self.limits.max_array_bytes:
            raise ValueError("checkpoint content array byte bound exceeded")
        self.scalar(value.dtype.str)
        self.scalar(value.ndim)
        if value.ndim > self.limits.max_depth:
            raise ValueError("checkpoint content array dimension bound exceeded")
        for size in value.shape:
            self.scalar(size)
        for stride in value.strides:
            self.scalar(stride)
        self.scalar(value.flags.c_contiguous)
        self.scalar(value.flags.f_contiguous)
        self.scalar(value.flags.writeable)
        # Both layouts flatten to a view. No tobytes/array copy is permitted here.
        self.digest.update(memoryview(value.reshape(-1, order="A")).cast("B"))

    def generator(self, value: Any, depth: int) -> None:
        if type(value.bit_generator) is not np.random.PCG64:
            raise ValueError("checkpoint content requires exact PCG64 generator")
        self.metadata_token(str(id(value.bit_generator)).encode("ascii"))
        # NumPy returns a fresh scalar-only dictionary: stamp content, not its
        # transient dictionary identity, and never draw or change RNG state.
        self.rng_value(value.bit_generator.state, depth + 1)

    def rng_value(self, value: Any, depth: int) -> None:
        self.nodes += 1
        if self.nodes > self.limits.max_nodes or depth > self.limits.max_depth:
            raise ValueError("checkpoint content RNG node/depth bound exceeded")
        if type(value) is not dict:
            if not _exact_type(type(value), str, int):
                raise ValueError("unsupported exact PCG64 state scalar")
            self.scalar(value)
            return
        if len(value) > self.limits.max_nodes - self.nodes:
            raise ValueError("checkpoint content RNG collection bound exceeded")
        self.scalar(len(value))
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("unsupported exact PCG64 state key")
            self.scalar(key)
            self.rng_value(item, depth + 1)
