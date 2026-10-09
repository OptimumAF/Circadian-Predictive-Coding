"""Explicit complete BackpropSnapshot byte schema with detached native arrays.

Inputs: exact owned native snapshots or bounded bytes, independent bindings and
limits. Output: canonical bytes or detached snapshot. No generic object loader,
learner policy construction, IO, work admission, live restore or source sealing.
"""

from base64 import b64decode, b64encode
from dataclasses import asdict
from hashlib import sha256
import json
from math import prod
from typing import Any

import numpy as np

from src.adapters.numpy_learners import BackpropSnapshot
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest

_FIELDS = {
    "input_dim",
    "hidden_dims",
    "_hidden_weights",
    "_hidden_biases",
    "_traffic_sums",
    "_traffic_steps",
    "weight_hidden_output",
    "bias_output",
    "weight_input_hidden",
    "bias_hidden",
}
_ARRAYS = {"hidden_weights", "hidden_biases", "traffic_sums", "weight_hidden_output", "bias_output"}
_ALIASES = {"weight_input_hidden": "hidden_weights/0", "bias_hidden": "hidden_biases/0"}
_ENVELOPE = {
    "codec_version",
    "kind",
    "binding",
    "snapshot_version",
    "input_dim",
    "hidden_dims",
    "traffic_steps",
    "arrays",
    "aliases",
}


def _configuration(binding, limits):
    if type(binding) is not CodecBinding or type(limits) is not CodecLimits:
        raise ValueError("codec requires exact binding and limits")
    CodecBinding.__post_init__(binding)
    CodecLimits.__post_init__(limits)


def _object(value, keys):
    if type(value) is not dict or value.keys() != keys:
        raise ValueError("codec fields differ from complete supported schema")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate codec JSON key")
        result[key] = value
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _shapes(input_dim, hidden_dims, steps, limits):
    if (
        type(input_dim) is not int
        or not 0 < input_dim <= limits.max_dimension
        or type(hidden_dims) not in (list, tuple)
        or not 0 < len(hidden_dims) <= limits.max_layers
        or any(type(w) is not int or not 0 < w <= limits.max_dimension for w in hidden_dims)
        or type(steps) is not int
        or not 0 <= steps < 2**63
    ):
        raise ValueError("codec topology or traffic counter exceeds supported bounds")
    widths = (input_dim, *hidden_dims)
    shapes: dict[str, list[tuple[int, ...]]] = {
        "hidden_weights": [(widths[i], w) for i, w in enumerate(hidden_dims)],
        "hidden_biases": [(1, w) for w in hidden_dims],
        "traffic_sums": [(w,) for w in hidden_dims],
        "weight_hidden_output": [(widths[-1], 1)],
        "bias_output": [(1, 1)],
    }
    if (
        sum(prod(shape) * 8 for group in shapes.values() for shape in group)
        > limits.max_array_bytes
    ):
        raise ValueError("complete native arrays exceed original codec byte bound")
    return shapes


def _header(binding, input_dim, hidden_dims, steps):
    return dict(
        codec_version=1,
        kind="backprop_full_v1",
        binding=asdict(binding),
        snapshot_version=1,
        input_dim=input_dim,
        hidden_dims=list(hidden_dims),
        traffic_steps=steps,
        arrays={},
        aliases=_ALIASES.copy(),
    )


def _size_preflight(header, shapes, limits):
    # Exact wire length is known before creating any raw/base64 array copies.
    empty: dict[str, Any] = {
        name: [dict(dtype="<f8", shape=list(s), data="") for s in group]
        for name, group in shapes.items()
    }
    for name in ("weight_hidden_output", "bias_output"):
        empty[name] = empty[name][0]
    header["arrays"] = empty
    size = len(_json(header)) + sum(
        4 * ((prod(s) * 8 + 2) // 3) for group in shapes.values() for s in group
    )
    if size > limits.max_encoded_bytes:
        raise ValueError("complete encoded snapshot exceeds original codec byte bound")


def _native_arrays(state, shapes):
    arrays = {}
    for name, group in shapes.items():
        values = (
            state["_" + name]
            if name in ("hidden_weights", "hidden_biases", "traffic_sums")
            else [state[name]]
        )
        if type(values) is not list or len(values) != len(group):
            raise ValueError("native array collection differs from topology")
        arrays[name] = values
        for value, shape in zip(values, group):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.float64
                or value.shape != shape
                or not np.all(np.isfinite(value))
            ):
                raise ValueError("native array requires exact finite float64 topology")
        if name == "traffic_sums" and any(np.any(v < 0) for v in values):
            raise ValueError("native traffic cannot be negative")
    flat = [v for group in arrays.values() for v in group]
    if any(np.shares_memory(left, right) for i, left in enumerate(flat) for right in flat[i + 1 :]):
        raise ValueError("unsupported native shared-memory edge")
    if (
        state["weight_input_hidden"] is not arrays["hidden_weights"][0]
        or state["bias_hidden"] is not arrays["hidden_biases"][0]
    ):
        raise ValueError("native legacy array aliases differ")
    return arrays


def _encode_arrays(arrays):
    frames: dict[str, Any] = {
        name: [
            dict(
                dtype="<f8",
                shape=list(v.shape),
                data=b64encode(v.astype("<f8", copy=False).tobytes(order="C")).decode("ascii"),
            )
            for v in group
        ]
        for name, group in arrays.items()
    }
    for name in ("weight_hidden_output", "bias_output"):
        frames[name] = frames[name][0]
    return frames


def _frames(body, shapes):
    source = _object(body["arrays"], _ARRAYS)
    frames = {}
    # Validate every frame before decoding/allocating the first native array.
    for name, group in shapes.items():
        values = (
            source[name]
            if name in ("hidden_weights", "hidden_biases", "traffic_sums")
            else [source[name]]
        )
        if type(values) is not list or len(values) != len(group):
            raise ValueError("encoded array collection differs from topology")
        frames[name] = values
        for value, shape in zip(values, group):
            _object(value, {"dtype", "shape", "data"})
            size = prod(shape) * 8
            if (
                type(value["shape"]) is not list
                or any(type(d) is not int for d in value["shape"])
                or value["shape"] != list(shape)
                or value["dtype"] != "<f8"
                or type(value["data"]) is not str
                or len(value["data"]) != 4 * ((size + 2) // 3)
            ):
                raise ValueError("encoded dtype/shape/byte length differs from topology")
    return frames


def _decode_arrays(frames):
    arrays = {}
    for name, group in frames.items():
        values = []
        for frame in group:
            raw = b64decode(frame["data"], validate=True)
            if (
                len(raw) != prod(frame["shape"]) * 8
                or b64encode(raw).decode("ascii") != frame["data"]
            ):
                raise ValueError("noncanonical or mismatched array bytes")
            value = (
                np.frombuffer(raw, dtype="<f8")
                .reshape(frame["shape"])
                .astype(np.float64, copy=True)
            )
            if not np.all(np.isfinite(value)) or (name == "traffic_sums" and np.any(value < 0)):
                raise ValueError("invalid finite native array or traffic")
            values.append(value)
        arrays[name] = values
    return arrays


class BackpropCheckpointCodec:
    def encode(
        self, state: BackpropSnapshot, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        _configuration(binding, limits)
        if (
            type(state) is not BackpropSnapshot
            or type(state.format_version) is not int
            or state.format_version != 1
            or type(state.hidden_dims) is not tuple
        ):
            raise ValueError("unsupported exact backprop snapshot")
        data = _object(state.state, _FIELDS)
        if data["input_dim"] != state.input_dim or data["hidden_dims"] != state.hidden_dims:
            raise ValueError("native topology differs from snapshot metadata")
        if type(data["input_dim"]) is not int or type(data["hidden_dims"]) is not tuple:
            raise ValueError("native topology types differ")
        if any(type(width) is not int for width in data["hidden_dims"]):
            raise ValueError("native hidden dimensions require exact integers")
        shapes = _shapes(state.input_dim, state.hidden_dims, data["_traffic_steps"], limits)
        header = _header(binding, state.input_dim, state.hidden_dims, data["_traffic_steps"])
        _size_preflight(header, shapes, limits)
        header["arrays"] = _encode_arrays(_native_arrays(data, shapes))
        return _json(header)

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> BackpropSnapshot:
        _configuration(binding, limits)
        require_codec_digest(expected_sha256)
        if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
            raise ValueError("codec requires bounded exact bytes")
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("codec content differs from independently expected digest")
        try:
            body = _object(json.loads(raw, object_pairs_hook=_pairs), _ENVELOPE)
            if (
                type(body["codec_version"]) is not int
                or body["codec_version"] != 1
                or type(body["snapshot_version"]) is not int
                or body["snapshot_version"] != 1
                or body["kind"] != "backprop_full_v1"
                or body["binding"] != asdict(binding)
                or body["aliases"] != _ALIASES
                or type(body["hidden_dims"]) is not list
            ):
                raise ValueError("unsupported codec version, alias or independent binding")
            shapes = _shapes(body["input_dim"], body["hidden_dims"], body["traffic_steps"], limits)
            _size_preflight(
                _header(binding, body["input_dim"], body["hidden_dims"], body["traffic_steps"]),
                shapes,
                limits,
            )
            frames = _frames(body, shapes)
            if _json(body) != raw:
                raise ValueError("codec JSON must be canonical")
            arrays = _decode_arrays(frames)
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
        except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as error:
            raise ValueError("invalid/corrupted complete backprop codec bytes") from error
