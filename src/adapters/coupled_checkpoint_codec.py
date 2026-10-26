"""Bounded complete native PC/CPC snapshot bytes through the checkpoint port.

Topology and exact wire size are admitted before finite scans or array copies.
All wire frames are admitted before decoding any array. Why explicit frames:
native float64 bytes, storage order and signed zero survive without float lists.
Borrowed encode inputs must remain quiescent. Digests/bindings convey no source
provenance or authority. No models, live restore, IO or learner policy settings.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite, prod
import re

import numpy as np

from src.adapters.numpy_checkpoint_frames import (
    canonical_json,
    decode_frame,
    empty_frame,
    encode_frame,
    exact_object,
    native_order,
    unique_pairs,
    validate_frame,
    validate_native_array,
)
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    _plasticity_gates,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledWakeState

_ENVELOPE = {
    "codec_version",
    "kind",
    "binding",
    "snapshot_version",
    "input_dim",
    "hidden_dims",
    "precisions",
    "arrays",
    "wake_state",
}
_ARRAY_FIELDS = {"hidden_weights", "hidden_biases", "output_weight", "output_bias"}
_POLICY_FIELDS = (
    "chemical_decay",
    "chemical_buildup_rate",
    "plasticity_sensitivity",
    "min_plasticity",
)
_WAKE_FIELDS = {*_POLICY_FIELDS, "wake_updates", "chemical_state", "plasticity_state"}
_SINGLE_ARRAYS = ("output_weight", "output_bias")
_WAKE_ARRAYS = ("chemical_state", "plasticity_state")
_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}\Z")
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _configuration(binding: CodecBinding, limits: CodecLimits) -> None:
    if type(binding) is not CodecBinding or type(limits) is not CodecLimits:
        raise ValueError("codec requires exact CodecBinding and CodecLimits")
    exact_object(vars(binding), {"source_sha256", "policy_sha256"})
    exact_object(
        vars(limits), {"max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension"}
    )
    CodecBinding.__post_init__(binding)
    CodecLimits.__post_init__(limits)


def _shapes(input_dim, hidden_dims, limits):
    if (
        type(input_dim) is not int
        or not 0 < input_dim <= limits.max_dimension
        or not 1 <= len(hidden_dims) <= limits.max_layers
        or any(
            type(width) is not int or not 0 < width <= limits.max_dimension for width in hidden_dims
        )
    ):
        raise ValueError("coupled topology exceeds exact native codec bounds")
    widths = (input_dim, *hidden_dims)
    return {
        "hidden_weights": [(lower, upper) for lower, upper in zip(widths, hidden_dims)],
        "hidden_biases": [(1, width) for width in hidden_dims],
        "output_weight": [(hidden_dims[-1], 1)],
        "output_bias": [(1, 1)],
    }


def _precisions(values, depth, collection_type):
    if type(values) is not collection_type or len(values) != depth:
        raise ValueError("precisions must match the exact hidden topology")
    if any(
        type(value) not in (int, float) or not 1e-6 <= value <= 1e6 or not isfinite(value)
        for value in values
    ):
        raise ValueError("precisions require finite native numbers in [1e-6, 1e6]")
    return list(values)


def _wake_metadata(wake, limits):
    exact_object(wake, _WAKE_FIELDS)
    policy = CoupledCircadianConfig(*(wake[name] for name in _POLICY_FIELDS))
    count = wake["wake_updates"]
    if type(count) is not int or count < 0:
        raise ValueError("wake_updates must be a native nonnegative integer")
    # Refuse clearly oversized counters before decimal rendering. Python's own
    # integer JSON/parser limits may still refuse a smaller counter, never clip it.
    if count.bit_length() > 4 * limits.max_encoded_bytes:
        raise ValueError("wake counter alone exceeds the encoded byte bound")
    metadata = {name: wake[name] for name in _POLICY_FIELDS}
    metadata["wake_updates"] = count
    return metadata, policy


def _header(binding, input_dim, hidden_dims, precisions, wake):
    return dict(
        codec_version=1,
        kind="coupled_pc" if wake is None else "coupled_cpc",
        binding=asdict(binding),
        snapshot_version=1,
        input_dim=input_dim,
        hidden_dims=list(hidden_dims),
        precisions=precisions,
        arrays={},
        wake_state=wake,
    )


def _size_preflight(header, shapes, limits):
    array_bytes = sum(prod(shape) * 8 for group in shapes.values() for shape in group)
    if array_bytes > limits.max_array_bytes:
        raise ValueError("aggregate coupled arrays exceed codec byte bound")
    frames = {
        name: [empty_frame(shape, "<f8") for shape in group] for name, group in shapes.items()
    }
    header["arrays"] = {
        name: frames[name][0] if name in _SINGLE_ARRAYS else frames[name] for name in _ARRAY_FIELDS
    }
    if header["wake_state"] is not None:
        for name in _WAKE_ARRAYS:
            header["wake_state"][name] = frames[name][0]
    size = len(canonical_json(header)) + sum(
        4 * ((prod(shape) * 8 + 2) // 3) for group in shapes.values() for shape in group
    )
    if size > limits.max_encoded_bytes:
        raise ValueError("complete coupled encoding exceeds codec byte bound")


def _native_structure(state: CoupledSnapshot, limits: CodecLimits):
    """Borrow exact native metadata; bound aggregate arrays before any value scan."""
    if type(state) is not CoupledSnapshot:
        raise ValueError("codec requires exact CoupledSnapshot format 1")
    exact_object(vars(state), {"format_version", "parameters", "precisions", "wake_state"})
    if type(state.format_version) is not int or state.format_version != 1:
        raise ValueError("codec requires exact CoupledSnapshot format 1")
    parameters = state.parameters
    if type(parameters) is not CoupledParameters:
        raise ValueError("codec requires exact CoupledParameters")
    exact_object(vars(parameters), _ARRAY_FIELDS)
    weights, biases = parameters.hidden_weights, parameters.hidden_biases
    if (
        type(weights) is not tuple
        or not 1 <= len(weights) <= limits.max_layers
        or type(biases) is not tuple
        or len(biases) != len(weights)
    ):
        raise ValueError("native hidden arrays require exact matching nonempty tuples")
    if any(type(weight) is not np.ndarray or weight.ndim != 2 for weight in weights):
        raise ValueError("hidden weights must be exact matrices")
    input_dim, hidden_dims = weights[0].shape[0], tuple(weight.shape[1] for weight in weights)
    shapes = _shapes(input_dim, hidden_dims, limits)
    precisions = _precisions(state.precisions, len(hidden_dims), tuple)
    values = dict(
        hidden_weights=list(weights),
        hidden_biases=list(biases),
        output_weight=[parameters.output_weight],
        output_bias=[parameters.output_bias],
    )
    wake, policy = None, None
    if state.wake_state is not None:
        if type(state.wake_state) is not CoupledWakeState:
            raise ValueError("CPC requires exact complete CoupledWakeState")
        source = vars(state.wake_state)
        wake, policy = _wake_metadata(source, limits)
        for name in _WAKE_ARRAYS:
            shapes[name] = [(hidden_dims[-1],)]
            values[name] = [source[name]]
    if (
        sum(prod(shape) * 8 for group in shapes.values() for shape in group)
        > limits.max_array_bytes
    ):
        raise ValueError("aggregate coupled arrays exceed codec byte bound")
    return input_dim, hidden_dims, precisions, wake, shapes, values, policy


def _native_array_metadata(shapes, values):
    # No finite scan (which creates a temporary) occurs before aggregate sizing.
    for name, group in values.items():
        for value, shape in zip(group, shapes[name]):
            if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape:
                raise ValueError("native array dtype or shape differs from complete topology")
            native_order(value)


def _native_plan(state: CoupledSnapshot, binding: CodecBinding, limits: CodecLimits):
    input_dim, hidden_dims, precisions, wake, shapes, values, policy = _native_structure(
        state, limits
    )
    header = _header(binding, input_dim, hidden_dims, precisions, wake)
    _size_preflight(header, shapes, limits)
    _native_array_metadata(shapes, values)
    return header, shapes, values, policy


def _validate_wake(values, metadata, policy):
    if policy is None:
        return
    chemicals, gates = values["chemical_state"][0], values["plasticity_state"][0]
    if np.any(chemicals < 0) or np.any(gates < 0):
        raise ValueError("native wake vectors must be nonnegative")
    if metadata["wake_updates"] == 0 and (np.any(chemicals != 0) or np.any(gates != 1)):
        raise ValueError("zero wake count requires zero chemistry and unit gates")
    if not np.array_equal(gates, _plasticity_gates(chemicals, policy)):
        raise ValueError("plasticity gates disagree with stored chemistry and policy")


def _put_frames(header, frames):
    header["arrays"] = {
        name: frames[name][0] if name in _SINGLE_ARRAYS else frames[name] for name in _ARRAY_FIELDS
    }
    if header["wake_state"] is not None:
        for name in _WAKE_ARRAYS:
            header["wake_state"][name] = frames[name][0]


def _canonical_base64(frame, shape):
    data = frame["data"]
    padding = (3 - (prod(shape) * 8) % 3) % 3
    actual_padding = 2 if data.endswith("==") else 1 if data.endswith("=") else 0
    if _BASE64.fullmatch(data) is None or actual_padding != padding:
        raise ValueError("frame base64 is not canonical")
    if padding and _ALPHABET.find(data[-padding - 1]) & (15 if padding == 2 else 3):
        raise ValueError("frame base64 has noncanonical padding bits")


def _wire_plan(body, binding, limits):
    exact_object(body, _ENVELOPE)
    if (
        type(body["codec_version"]) is not int
        or body["codec_version"] != 1
        or type(body["snapshot_version"]) is not int
        or body["snapshot_version"] != 1
    ):
        raise ValueError("unsupported exact coupled codec/snapshot version")
    if type(body["kind"]) is not str or body["kind"] not in ("coupled_pc", "coupled_cpc"):
        raise ValueError("unsupported coupled checkpoint kind")
    encoded_binding = exact_object(body["binding"], {"source_sha256", "policy_sha256"})
    declared_binding = CodecBinding(
        encoded_binding["source_sha256"], encoded_binding["policy_sha256"]
    )
    if declared_binding != binding:
        raise ValueError("checkpoint differs from independently expected binding")
    if type(body["hidden_dims"]) is not list:
        raise ValueError("wire hidden_dims must be an exact list")
    shapes = _shapes(body["input_dim"], body["hidden_dims"], limits)
    precisions = _precisions(body["precisions"], len(body["hidden_dims"]), list)
    wake, policy = None, None
    if body["kind"] == "coupled_pc":
        if body["wake_state"] is not None:
            raise ValueError("PC must not contain wake state")
    else:
        wake, policy = _wake_metadata(body["wake_state"], limits)
        shapes.update({name: [(body["hidden_dims"][-1],)] for name in _WAKE_ARRAYS})
    _size_preflight(
        _header(binding, body["input_dim"], body["hidden_dims"], precisions, wake), shapes, limits
    )
    arrays = exact_object(body["arrays"], _ARRAY_FIELDS)
    frames = {}
    for name, group in shapes.items():
        source = body["wake_state"] if name in _WAKE_ARRAYS else arrays
        entries = source[name] if name in ("hidden_weights", "hidden_biases") else [source[name]]
        if type(entries) is not list or len(entries) != len(group):
            raise ValueError("wire array collection differs from topology")
        for frame, shape in zip(entries, group):
            validate_frame(frame, shape, "<f8")
            _canonical_base64(frame, shape)
        frames[name] = entries
    return shapes, frames, policy


def _snapshot(body, values):
    parameters = CoupledParameters(
        tuple(values["hidden_weights"]),
        tuple(values["hidden_biases"]),
        values["output_weight"][0],
        values["output_bias"][0],
    )
    wake = None
    if body["wake_state"] is not None:
        metadata = body["wake_state"]
        wake = CoupledWakeState(
            metadata["chemical_decay"],
            metadata["chemical_buildup_rate"],
            metadata["plasticity_sensitivity"],
            metadata["min_plasticity"],
            values["chemical_state"][0],
            values["plasticity_state"][0],
            metadata["wake_updates"],
        )
    return CoupledSnapshot(1, parameters, tuple(body["precisions"]), wake)


class CoupledCheckpointCodec:
    def encode(
        self, state: CoupledSnapshot, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        _configuration(binding, limits)
        try:
            header, shapes, values, policy = _native_plan(state, binding, limits)
            for name, group in values.items():
                for value, shape in zip(group, shapes[name]):
                    validate_native_array(value, shape, "<f8")
            _validate_wake(values, header["wake_state"], policy)
            frames = {
                name: [
                    encode_frame(value, shape, "<f8") for value, shape in zip(group, shapes[name])
                ]
                for name, group in values.items()
            }
            _put_frames(header, frames)
            return canonical_json(header)
        except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as error:
            raise ValueError("invalid/unsupported complete coupled snapshot") from error

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> CoupledSnapshot:
        _configuration(binding, limits)
        require_codec_digest(expected_sha256)
        if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
            raise ValueError("codec requires bounded exact bytes")
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("checkpoint differs from independently expected digest")
        try:
            body = json.loads(raw, object_pairs_hook=unique_pairs)
            shapes, frames, policy = _wire_plan(body, binding, limits)
            if canonical_json(body) != raw:
                raise ValueError("coupled checkpoint JSON must be canonical")
            values = {
                name: [
                    decode_frame(frame, shape, "<f8") for frame, shape in zip(group, shapes[name])
                ]
                for name, group in frames.items()
            }
            _validate_wake(values, body["wake_state"], policy)
            return _snapshot(body, values)
        except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as error:
            raise ValueError("invalid/corrupted complete coupled checkpoint bytes") from error
