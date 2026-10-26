"""Bounded deterministic coupled managed-composite DATA bytes, format 1.

Encode/decode preserve complete capture fields, native array bytes and graph
aliases. Caller-owned copies are outside the source's retained-RAW ledger.
Decoded identity tokens and AuthorityPath records grant no live authority. No
model, callback, IO, native operation, reconstruction or recovery permission.
Ambiguous unordered structural-key ties are explicitly unsupported.
"""

from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from typing import Any

from src.adapters.coupled_composite_graph import GraphEncoder, validate_data
from src.adapters import coupled_composite_wire as wire
from src.adapters.numpy_checkpoint_frames import canonical_json, exact_object, unique_pairs
from src.adapters.lifecycle_checkpoint_codec import _json as bounded_json
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.managed_composite_state import CompositeCaptureLimits, ManagedCompositeState
from src.core.managed_lifecycle_state import LifecycleCaptureLimits

_ENVELOPE = {"codec_version", "kind", "binding", "capture_limits", "root", "nodes"}
_KIND = "coupled_managed_composite_v1"
_INPUT_ERRORS = (AttributeError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError)


def _copy_capture_limits(value: CompositeCaptureLimits) -> CompositeCaptureLimits:
    if type(value) is not CompositeCaptureLimits or vars(value).keys() != {
        field.name for field in fields(CompositeCaptureLimits)
    }:
        raise ValueError("composite codec requires exact complete capture limits")
    records = value.records
    if type(records) is not LifecycleCaptureLimits or vars(records).keys() != {
        "max_records",
        "max_identifier_bytes",
    }:
        raise ValueError("composite codec requires exact nested record limits")
    return CompositeCaptureLimits(
        LifecycleCaptureLimits(records.max_records, records.max_identifier_bytes),
        value.max_nodes,
        value.max_depth,
        value.max_array_bytes,
        value.max_dimension,
    )


def _controls(binding: CodecBinding, limits: CodecLimits) -> tuple[CodecBinding, CodecLimits]:
    if type(binding) is not CodecBinding or vars(binding).keys() != {
        "source_sha256",
        "policy_sha256",
    }:
        raise ValueError("composite codec requires exact CodecBinding")
    if type(limits) is not CodecLimits or vars(limits).keys() != {
        field.name for field in fields(CodecLimits)
    }:
        raise ValueError("composite codec requires exact CodecLimits")
    return CodecBinding(binding.source_sha256, binding.policy_sha256), CodecLimits(**vars(limits))


def _nonfinite(value: str) -> None:
    raise ValueError("nonfinite composite JSON constant")


class CoupledCompositeCheckpointCodec:
    """A fixed allowlisted data codec; constructor owns detached capture bounds."""

    def __init__(self, *, capture_limits: CompositeCaptureLimits) -> None:
        self._capture_limits = _copy_capture_limits(capture_limits)

    def _configuration(
        self, binding: CodecBinding, limits: CodecLimits
    ) -> tuple[CompositeCaptureLimits, CompositeCaptureLimits, CodecBinding, CodecLimits]:
        original = _copy_capture_limits(self._capture_limits)
        binding, limits = _controls(binding, limits)
        effective = replace(
            original,
            max_array_bytes=min(original.max_array_bytes, limits.max_array_bytes),
            max_dimension=min(original.max_dimension, limits.max_dimension),
        )
        return original, effective, binding, limits

    def encode(
        self, state: ManagedCompositeState, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        try:
            original, effective, binding, limits = self._configuration(binding, limits)
            validate_data(state, effective, limits.max_layers)
            graph = GraphEncoder(effective)
            root = graph.token(state)
            body = dict(
                codec_version=1,
                kind=_KIND,
                binding=asdict(binding),
                capture_limits=asdict(original),
                root=root,
                nodes=graph.nodes,
            )
            wire.preflight_wire(root, graph.nodes, effective, limits.max_layers, placeholders=True)
            # Empty data fields account for every JSON byte except base64 itself.
            size = len(bounded_json(body, limits.max_encoded_bytes)) + sum(
                4 * ((value.nbytes + 2) // 3) for value, _ in graph.arrays
            )
            if size > limits.max_encoded_bytes:
                raise ValueError("complete composite encoding exceeds byte bound")
            for value, node in graph.arrays:
                wire.encode_array(value, node)
            raw = bounded_json(body, limits.max_encoded_bytes)
            if len(raw) != size or len(raw) > limits.max_encoded_bytes:
                raise ValueError("complete composite encoded size changed")
            return raw
        except _INPUT_ERRORS as error:
            raise ValueError("unsupported complete composite encode input") from error

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> ManagedCompositeState:
        try:
            original, effective, binding, limits = self._configuration(binding, limits)
            if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
                raise ValueError("composite input must be bounded exact bytes")
            require_codec_digest(expected_sha256)
            if sha256(raw).hexdigest() != expected_sha256:
                raise ValueError("composite checkpoint SHA256 differs from expected digest")
            body = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=_nonfinite)
            exact_object(body, _ENVELOPE)
            if (
                type(body["codec_version"]) is not int
                or body["codec_version"] != 1
                or type(body["kind"]) is not str
                or body["kind"] != _KIND
            ):
                raise ValueError("unsupported complete composite codec version/kind")
            exact_object(body["binding"], {"source_sha256", "policy_sha256"})
            if CodecBinding(**body["binding"]) != binding:
                raise ValueError("composite binding differs from independent expectation")
            if canonical_json(body["capture_limits"]) != canonical_json(asdict(original)):
                raise ValueError("composite wire capture limits differ from owned limits")
            if bounded_json(body, limits.max_encoded_bytes) != raw:
                raise ValueError("composite JSON must have canonical bytes")
            wire.preflight_wire(body["root"], body["nodes"], effective, limits.max_layers)
            state: Any = wire.materialize(body["root"], body["nodes"])
            validate_data(state, effective, limits.max_layers)
            if self.encode(state, binding=binding, limits=limits) != raw:
                raise ValueError("noncanonical composite graph IDs or representations")
            return state
        except _INPUT_ERRORS as error:
            raise ValueError("unsupported complete composite decode input") from error
