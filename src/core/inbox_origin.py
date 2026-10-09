"""Bounded metadata and borrowed numeric contents for observed inbox updates.

These checks establish integrity of already observed records. They grant no
consent, ownership, training, copy, checkpoint or restoration authority.
"""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from typing import Any, cast

import numpy as np

from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.experience import require_identifier, require_tick


@dataclass(frozen=True)
class InboxOriginData:
    key: tuple[str, str]
    event_id: str
    actor_version: str
    learner_version: str
    subject_id: str
    source_id: str
    observed_at: int
    arrived_at: int
    update_number: int
    payload_bytes: int
    features_digest: str
    targets_digest: str

    def __post_init__(self) -> None:
        if type(self.key) is not tuple or len(self.key) != 2:
            raise ValueError("inbox origin requires an exact paired identity")
        for value in (
            *self.key,
            self.event_id,
            self.actor_version,
            self.learner_version,
            self.subject_id,
            self.source_id,
        ):
            if type(value) is not str or len(value) > 128 * 1024:
                raise ValueError("inbox origin identities require exact strings")
            require_identifier(value, "inbox origin identity")
        for name in ("observed_at", "arrived_at", "update_number", "payload_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value >= 2**63:
                raise ValueError("inbox origin counters require bounded exact integers")
            require_tick(value, name)
        if self.update_number == 0 or self.payload_bytes == 0:
            raise ValueError("inbox origin work and payload bytes must be positive")
        for value in (self.features_digest, self.targets_digest):
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("inbox origin requires bounded SHA256 content stamps")


def inbox_origin_metadata(data: InboxOriginData, maximum: int) -> bytes:
    if type(data) is not InboxOriginData or type(maximum) is not int or not 0 < maximum < 2**63:
        raise ValueError("inbox origin metadata requires exact data and a positive bound")
    values = vars(data)
    names = {field.name for field in fields(InboxOriginData)}
    if (
        type(values) is not dict
        or len(values) != len(names)
        or any(type(key) is not str or len(key) > 64 for key in values)
        or values.keys() != names
    ):
        raise ValueError("inbox origin metadata fields changed")
    if type(data.key) is not tuple or len(data.key) != 2:
        raise ValueError("inbox origin requires an exact paired identity")
    strings = (
        *data.key,
        data.event_id,
        data.actor_version,
        data.learner_version,
        data.subject_id,
        data.source_id,
        data.features_digest,
        data.targets_digest,
    )
    if any(type(value) is not str for value in strings):
        raise ValueError("inbox origin metadata requires exact strings")
    # Bound aggregate escaped text before identifier normalization or JSON copies.
    if sum(len(value) * 6 for value in strings) > maximum:
        raise ValueError("inbox origin identifiers exceed metadata capacity")
    InboxOriginData.__post_init__(data)
    encoded = json.dumps(
        asdict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")
    if len(encoded) > maximum:
        raise ValueError("inbox origin metadata exceeds its original bound")
    return encoded


def inbox_numeric_stamp(value: object, limits: CheckpointContentLimits) -> str:
    if type(value) is not np.ndarray:
        raise ValueError("inbox history requires exact numeric arrays")
    checkpoint_content_stamp(value, limits)
    if not np.isfinite(value).all():
        raise ValueError("inbox history requires finite numeric input contents")
    # Why: native input copies may change writeability. Compare actual values,
    # shape/dtype/layout independently of pointer and access flags; identities
    # are bound separately by the actual update observer and copier memo.
    digest = sha256(repr((value.dtype.str, value.shape, value.strides)).encode("ascii"))
    # NumPy implements the buffer protocol; its installed typing stub omits it.
    digest.update(memoryview(cast(Any, value.reshape(-1, order="A"))).cast("B"))
    return digest.hexdigest()


def inbox_payload_binding(
    features: object,
    targets: object,
    consumed_features: object,
    consumed_targets: object,
    limits: CheckpointContentLimits,
) -> tuple[int, str, str]:
    if (
        type(limits) is not CheckpointContentLimits
        or type(vars(limits)) is not dict
        or len(vars(limits)) != 4
        or any(type(key) is not str for key in vars(limits))
        or set(vars(limits)) != {"max_nodes", "max_metadata_bytes", "max_array_bytes", "max_depth"}
        or any(type(value) is not int or not 0 < value < 2**63 for value in vars(limits).values())
    ):
        raise ValueError("inbox payload binding requires exact bounded content limits")
    if any(
        type(value) is not np.ndarray
        for value in (features, targets, consumed_features, consumed_targets)
    ):
        raise ValueError("inbox history requires exact numeric arrays")
    # Validate both paired bounds before any hash traversal or finite-check scratch.
    arrays = tuple(
        cast(np.ndarray, value)
        for value in (features, targets, consumed_features, consumed_targets)
    )
    if any(
        sum(value.nbytes for value in pair) > limits.max_array_bytes
        for pair in (arrays[:2], arrays[2:])
    ):
        raise ValueError("original paired payload exceeds its admitted byte bound")
    feature_digest = inbox_numeric_stamp(features, limits)
    target_digest = inbox_numeric_stamp(targets, limits)
    if feature_digest != inbox_numeric_stamp(
        consumed_features, limits
    ) or target_digest != inbox_numeric_stamp(consumed_targets, limits):
        raise ValueError("original inbox contents differ from actual consumed native inputs")
    size = arrays[0].nbytes + arrays[1].nbytes  # Exact ndarray types were checked above.
    return size, feature_digest, target_digest
