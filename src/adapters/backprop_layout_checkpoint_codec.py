"""Explicit Backprop wire-v2 layout/alias codec through the inner typed port.

Inputs: complete detached native snapshots or independently bound bounded bytes.
Outputs: canonical bytes or detached owned snapshots. No learner policy, IO,
source/owner attestation, live restore, work admission or generic graph loading.
Legacy wire-v1 codec remains independently available with its original behavior.
"""

from base64 import b64decode, b64encode
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite, prod
from struct import iter_unpack
from typing import Any

import numpy as np

from src.adapters import backprop_checkpoint_codec as legacy
from src.adapters.numpy_checkpoint_frames import (
    empty_frame,
    encode_frame,
    validate_frame,
    validate_native_array,
)
from src.adapters.numpy_learners import BackpropSnapshot
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest


def _header(binding, input_dim, hidden_dims, steps):
    # Reuse the exact native/alias schema; select this codec explicitly on wire.
    header = legacy._header(binding, input_dim, hidden_dims, steps)
    header.update(codec_version=2, kind="backprop_layout_v2")
    return header


def _size_preflight(header, shapes, limits):
    frames: dict[str, Any] = {
        name: [empty_frame(shape, "<f8") for shape in group] for name, group in shapes.items()
    }
    for name in ("weight_hidden_output", "bias_output"):
        frames[name] = frames[name][0]
    header["arrays"] = frames
    size = len(legacy._json(header)) + sum(
        4 * ((prod(shape) * 8 + 2) // 3) for group in shapes.values() for shape in group
    )
    if size > limits.max_encoded_bytes:
        raise ValueError("complete layout wire exceeds original codec byte bound")


def _native_arrays(snapshot, limits):
    if type(snapshot) is not BackpropSnapshot:
        raise ValueError("layout codec requires exact native BackpropSnapshot")
    legacy._object(vars(snapshot), {"format_version", "input_dim", "hidden_dims", "state"})
    if type(snapshot.format_version) is not int or snapshot.format_version != 1:
        raise ValueError("unsupported native snapshot version")
    if type(snapshot.hidden_dims) is not tuple:
        raise ValueError("native topology requires immutable tuple")
    data = legacy._object(snapshot.state, legacy._FIELDS)
    if type(data["input_dim"]) is not int or type(data["hidden_dims"]) is not tuple:
        raise ValueError("native topology requires exact integer and tuple types")
    if any(type(width) is not int for width in data["hidden_dims"]):
        raise ValueError("native widths require exact integers")
    if data["input_dim"] != snapshot.input_dim or data["hidden_dims"] != snapshot.hidden_dims:
        raise ValueError("native topology differs from snapshot metadata")
    shapes = legacy._shapes(
        snapshot.input_dim, snapshot.hidden_dims, data["_traffic_steps"], limits
    )
    arrays = legacy._native_arrays(data, shapes)
    for name, group in arrays.items():
        for array, shape in zip(group, shapes[name]):
            validate_native_array(array, shape, "<f8")
    return shapes, arrays


def _encode_arrays(arrays):
    frames: dict[str, Any] = {
        name: [encode_frame(value, value.shape, "<f8") for value in group]
        for name, group in arrays.items()
    }
    for name in ("weight_hidden_output", "bias_output"):
        frames[name] = frames[name][0]
    return frames


def _frames(body, shapes):
    source = legacy._object(body["arrays"], legacy._ARRAYS)
    frames = {}
    for name, group in shapes.items():
        values = (
            source[name]
            if name in ("hidden_weights", "hidden_biases", "traffic_sums")
            else [source[name]]
        )
        if type(values) is not list or len(values) != len(group):
            raise ValueError("encoded array collection differs from complete topology")
        for frame, shape in zip(values, group):
            validate_frame(frame, shape, "<f8")
        frames[name] = values
    return frames


def _validated_payloads(frames):
    # Validate every bounded payload before allocating the first detached array.
    # Python scalar unpacking preserves this boundary even for a corrupt late frame.
    payloads = {}
    for name, group in frames.items():
        values = []
        for frame in group:
            raw = b64decode(frame["data"], validate=True)
            if (
                len(raw) != prod(frame["shape"]) * 8
                or b64encode(raw).decode("ascii") != frame["data"]
            ):
                raise ValueError("noncanonical or mismatched layout payload bytes")
            if any(
                not isfinite(value) or (name == "traffic_sums" and value < 0)
                for (value,) in iter_unpack("<d", raw)
            ):
                raise ValueError("invalid finite native array or traffic")
            values.append(raw)
        payloads[name] = values
    return payloads


def _snapshot(body, frames, payloads):
    arrays = {}
    for name, group in frames.items():
        arrays[name] = [
            np.frombuffer(raw, dtype="<f8")
            .reshape(frame["shape"], order=frame["order"])
            .astype(np.float64, order=frame["order"], copy=True)
            for frame, raw in zip(group, payloads[name])
        ]
    data = dict(
        input_dim=body["input_dim"],
        hidden_dims=tuple(body["hidden_dims"]),
        _traffic_steps=body["traffic_steps"],
        _hidden_weights=arrays["hidden_weights"],
        _hidden_biases=arrays["hidden_biases"],
        _traffic_sums=arrays["traffic_sums"],
        weight_hidden_output=arrays["weight_hidden_output"][0],
        bias_output=arrays["bias_output"][0],
        weight_input_hidden=arrays["hidden_weights"][0],
        bias_hidden=arrays["hidden_biases"][0],
    )
    return BackpropSnapshot(1, body["input_dim"], tuple(body["hidden_dims"]), data)


class LayoutBackpropCheckpointCodec:
    def encode(
        self, state: BackpropSnapshot, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        legacy._configuration(binding, limits)
        shapes, arrays = _native_arrays(state, limits)
        header = _header(binding, state.input_dim, state.hidden_dims, state.state["_traffic_steps"])
        _size_preflight(header, shapes, limits)
        header["arrays"] = _encode_arrays(arrays)
        return legacy._json(header)

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> BackpropSnapshot:
        legacy._configuration(binding, limits)
        require_codec_digest(expected_sha256)
        if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
            raise ValueError("layout codec requires bounded exact bytes")
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("content differs from independently expected digest")
        try:
            body = legacy._object(
                json.loads(raw, object_pairs_hook=legacy._pairs), legacy._ENVELOPE
            )
            if (
                type(body["codec_version"]) is not int
                or body["codec_version"] != 2
                or body["kind"] != "backprop_layout_v2"
                or type(body["snapshot_version"]) is not int
                or body["snapshot_version"] != 1
                or body["binding"] != asdict(binding)
                or body["aliases"] != legacy._ALIASES
                or type(body["hidden_dims"]) is not list
            ):
                raise ValueError("unsupported codec/native version, alias or independent binding")
            shapes = legacy._shapes(
                body["input_dim"], body["hidden_dims"], body["traffic_steps"], limits
            )
            _size_preflight(
                _header(binding, body["input_dim"], body["hidden_dims"], body["traffic_steps"]),
                shapes,
                limits,
            )
            frames = _frames(body, shapes)
            if legacy._json(body) != raw:
                raise ValueError("layout checkpoint bytes must be canonical")
            return _snapshot(body, frames, _validated_payloads(frames))
        except (
            ValueError,
            TypeError,
            RecursionError,
            UnicodeError,
            OverflowError,
            KeyError,
        ) as error:
            raise ValueError("invalid/corrupted complete layout Backprop bytes") from error
