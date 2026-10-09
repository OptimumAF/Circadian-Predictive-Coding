"""Bounded component codec ports; no disk IO, model restoration or authority.

Bindings and expected byte digests must come from the independently known
original owner. Matching strings attest neither source closure nor ownership.
"""

from dataclasses import dataclass
import re
from typing import Protocol, TypeVar

State = TypeVar("State")


def require_codec_digest(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("codec binding requires lowercase SHA256")


@dataclass(frozen=True)
class CodecBinding:
    source_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        require_codec_digest(self.source_sha256)
        require_codec_digest(self.policy_sha256)


@dataclass(frozen=True)
class CodecLimits:
    max_encoded_bytes: int
    max_array_bytes: int
    max_layers: int
    max_dimension: int

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError("codec limits require bounded positive exact integers")


class CheckpointCodec(Protocol[State]):
    def encode(self, state: State, *, binding: CodecBinding, limits: CodecLimits) -> bytes: ...
    def decode(
        self,
        raw: bytes,
        *,
        binding: CodecBinding,
        expected_sha256: str,
        limits: CodecLimits,
    ) -> State: ...
