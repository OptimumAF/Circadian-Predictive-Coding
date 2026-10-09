"""Complete current circadian native byte variants through the inner codec port.

Inputs are exact detached snapshots or bounded canonical bytes with independently
expected bindings. Outputs are bytes or independently owned snapshots. No IO,
policy construction, updates, disk/live restoration or recovery admission.
"""

from collections import deque
from dataclasses import asdict
from hashlib import sha256
import json
from typing import TypeAlias

import numpy as np

from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    ReplayRetentionBudget,
    ReplaySnapshot,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.adapters.circadian_checkpoint_schema import (
    FLOAT_VECTORS,
    INT_VECTORS,
    IDS,
    CONFIG_TYPES,
    bounded_integer,
    finite_scalar,
    decode_config,
    state_fields,
    validate_configuration,
    validate_complete_native,
    validate_wire_scalars,
)
from src.adapters.numpy_checkpoint_frames import (
    canonical_json,
    exact_object,
    unique_pairs,
    frame_size,
    empty_frame,
    validate_native_array,
    validate_frame,
    encode_frame,
    decode_frame,
)

FrameSpec: TypeAlias = tuple[tuple[str | int, ...], tuple[int, ...], str]


def _at(data, path):
    value = data
    for key in path:
        value = value[key]
    return value


def _set(data, path, value):
    parent = _at(data, path[:-1])
    parent[path[-1]] = value


def _config_data(config):
    if type(config) is not CircadianConfig:
        raise ValueError("native configuration requires exact supported type")
    data = exact_object(vars(config), CONFIG_TYPES.keys()).copy()
    decode_config(data)
    return data


def _metadata(snapshot, limits):
    if (
        type(snapshot) is not CircadianNetworkSnapshot
        or type(snapshot.config) is not CircadianConfig
    ):
        raise ValueError("unsupported exact circadian native snapshot")
    exact_object(
        vars(snapshot),
        {
            "format_version",
            "input_dim",
            "initial_hidden_dims",
            "min_hidden_dim",
            "max_hidden_dim",
            "config",
            "state",
        },
    )
    if (
        type(snapshot.initial_hidden_dims) is not tuple
        or not 0 < len(snapshot.initial_hidden_dims) <= limits.max_layers
    ):
        raise ValueError("native snapshot requires bounded immutable topology")
    return dict(
        format_version=snapshot.format_version,
        input_dim=snapshot.input_dim,
        initial_hidden_dims=list(snapshot.initial_hidden_dims),
        min_hidden_dim=snapshot.min_hidden_dim,
        max_hidden_dim=snapshot.max_hidden_dim,
        config=_config_data(snapshot.config),
    )


def _native_replay(data, limits):
    rng = data["_rng"]
    if type(rng) is not np.random.Generator or type(rng.bit_generator) is not np.random.PCG64:
        raise ValueError("unsupported native RNG; current constructor uses exact PCG64")
    data["_rng"] = rng.bit_generator.state
    memory = data["_replay_memory"]
    if (
        type(memory) is not deque
        or len(memory) > limits.max_array_bytes // 16
        or any(type(item) is not ReplaySnapshot for item in memory)
    ):
        raise ValueError("native replay requires exact deque and row records")
    for item in memory:
        exact_object(vars(item), {"input_batch", "target_batch", "priority", "positive_fraction"})
    data["_replay_memory"] = dict(
        maxlen=memory.maxlen,
        items=[
            dict(
                input_batch=item.input_batch,
                target_batch=item.target_batch,
                priority=item.priority,
                positive_fraction=item.positive_fraction,
            )
            for item in memory
        ],
    )


def _native_optional(data, limits):
    for name in ("_replay_retention_budget", "_replay_retention_policy"):
        if name in data:
            kind = ReplayRetentionBudget if name.endswith("budget") else ReplayRetentionPolicy
            if type(data[name]) is not kind:
                raise ValueError("unsupported native replay policy record")
            if name.endswith("budget"):
                exact_object(vars(data[name]), {"max_examples", "max_bytes"})
                data[name].__post_init__()
            else:
                exact_object(vars(data[name]), {"name", "seed"})
                if type(data[name].name) is not str:
                    raise ValueError("native retention policy name requires exact string")
                data[name].__post_init__()
            data[name] = asdict(data[name])
    for name in IDS:
        if name in data:
            if type(data[name]) is not set:
                raise ValueError("native exposure identities require exact sets")
            if len(data[name]) * 66 > limits.max_encoded_bytes:
                raise ValueError("native exposure identities exceed original wire bound")
            for identity in data[name]:
                require_codec_digest(identity)
            data[name] = sorted(data[name])


def _native_body(snapshot, limits):
    meta = _metadata(snapshot, limits)
    state_fields(snapshot.state)
    data = snapshot.state.copy()
    for name in ("_pre_hidden_weights", "_pre_hidden_biases"):
        if type(data[name]) is not list or len(data[name]) > limits.max_layers:
            raise ValueError("native pre-hidden collections require exact lists")
        data[name] = data[name].copy()
    if type(data["config"]) is not CircadianConfig:
        raise ValueError("native configuration requires exact supported type")
    for name in ("hidden_dims", "pre_hidden_dims"):
        if type(data[name]) is not tuple or len(data[name]) > limits.max_layers:
            raise ValueError("native topology requires immutable tuples")
        data[name] = list(data[name])
    data["config"] = _config_data(data["config"])
    _native_replay(data, limits)
    _native_optional(data, limits)
    return dict(metadata=meta, state=data)


def _shape(value, encoded, limits):
    shape = value["shape"] if encoded else value.shape
    if type(shape) not in (list, tuple) or not 1 <= len(shape) <= 2:
        raise ValueError("unsupported array rank")
    for dimension in shape:
        bounded_integer(dimension, 1, limits.max_dimension)
    return tuple(shape)


def _specifications(body, encoded, limits):
    meta, data = body["metadata"], body["state"]
    validate_wire_scalars(meta, data, limits)
    dims, input_dim = meta["initial_hidden_dims"], meta["input_dim"]
    adaptive = _shape(data["weight_input_hidden"], encoded, limits)
    if len(adaptive) != 2:
        raise ValueError("adaptive weight requires rank two")
    width = adaptive[1]
    if width > meta["max_hidden_dim"]:
        raise ValueError("current adaptive width exceeds original maximum")
    specs: list[FrameSpec] = [(("state", name), (width,), "<f8") for name in FLOAT_VECTORS]
    specs += [(("state", name), (width,), dtype) for name, dtype in INT_VECTORS.items()]
    specs += [
        (
            ("state", "weight_input_hidden"),
            (dims[-2] if len(dims) > 1 else input_dim, width),
            "<f8",
        ),
        (("state", "bias_hidden"), (1, width), "<f8"),
        (("state", "weight_hidden_output"), (width, 1), "<f8"),
        (("state", "bias_output"), (1, 1), "<f8"),
    ]
    for name in ("_pre_hidden_weights", "_pre_hidden_biases"):
        if type(data[name]) is not list or len(data[name]) != len(dims) - 1:
            raise ValueError("pre-hidden array list differs from original topology")
    for index, layer in enumerate(dims[:-1]):
        specs += [
            (
                ("state", "_pre_hidden_weights", index),
                (input_dim if index == 0 else dims[index - 1], layer),
                "<f8",
            ),
            (("state", "_pre_hidden_biases", index), (1, layer), "<f8"),
        ]
    specs += _replay_specs(body, encoded, limits)
    if sum(frame_size(shape, dtype) for _, shape, dtype in specs) > limits.max_array_bytes:
        raise ValueError("aggregate native array bytes exceed original codec bound")
    return specs


def _replay_specs(body, encoded, limits):
    data, meta = body["state"], body["metadata"]
    memory = exact_object(data["_replay_memory"], {"maxlen", "items"})
    bounded = "_replay_retention_budget" in data
    expected = None if bounded else meta["config"]["replay_memory_size"]
    if type(memory["maxlen"]) is not type(expected) or memory["maxlen"] != expected:
        raise ValueError("native replay deque capacity differs from original policy")
    items = memory["items"]
    if type(items) is not list or len(items) > limits.max_encoded_bytes:
        raise ValueError("unsupported bounded replay records")
    if expected is not None and len(items) > expected:
        raise ValueError("native replay exceeds original deque capacity")
    specs: list[FrameSpec] = []
    for index, item in enumerate(items):
        exact_object(item, {"input_batch", "target_batch", "priority", "positive_fraction"})
        finite_scalar(item["priority"], minimum=0)
        finite_scalar(item["positive_fraction"], minimum=0, maximum=1)
        shape = _shape(item["input_batch"], encoded, limits)
        if len(shape) != 2 or shape[1] != meta["input_dim"]:
            raise ValueError("native replay input topology differs")
        if bounded and shape[0] != 1:
            raise ValueError("bounded native replay requires one distinct row per record")
        specs += [
            (("state", "_replay_memory", "items", index, "input_batch"), shape, "<f8"),
            (("state", "_replay_memory", "items", index, "target_batch"), (shape[0], 1), "<f8"),
        ]
    return specs


def _preflight(body, specs, limits, encoded):
    arrays = []
    for path, shape, dtype in specs:
        value = _at(body, path)
        if encoded:
            validate_frame(value, shape, dtype)
        else:
            validate_native_array(value, shape, dtype)
            arrays.append(value)
    if not encoded and any(
        np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :]
    ):
        raise ValueError("unsupported additional native array ownership edge")
    # Replace only declared arrays, then count exact future base64 wire length.
    saved = [_at(body, path) for path, _, _ in specs]
    try:
        for path, shape, dtype in specs:
            _set(body, path, empty_frame(shape, dtype))
        size = len(canonical_json(body)) + sum(
            4 * ((frame_size(s, d) + 2) // 3) for _, s, d in specs
        )
        if size > limits.max_encoded_bytes:
            raise ValueError("complete native wire exceeds original codec bound")
    finally:
        for (path, _, _), value in zip(specs, saved):
            _set(body, path, value)


def _snapshot(body):
    meta, data = body["metadata"], body["state"]
    data["config"] = decode_config(data["config"])
    for name in ("hidden_dims", "pre_hidden_dims"):
        data[name] = tuple(data[name])
    rng = np.random.Generator(np.random.PCG64(0))
    rng.bit_generator.state = data["_rng"]
    data["_rng"] = rng
    memory = data["_replay_memory"]
    data["_replay_memory"] = deque(
        (ReplaySnapshot(**item) for item in memory["items"]), maxlen=memory["maxlen"]
    )
    if "_replay_retention_budget" in data:
        data["_replay_retention_budget"] = ReplayRetentionBudget(**data["_replay_retention_budget"])
    if "_replay_retention_policy" in data:
        data["_replay_retention_policy"] = ReplayRetentionPolicy(**data["_replay_retention_policy"])
    for name in IDS:
        if name in data:
            data[name] = set(data[name])
    snapshot = CircadianNetworkSnapshot(
        meta["format_version"],
        meta["input_dim"],
        tuple(meta["initial_hidden_dims"]),
        meta["min_hidden_dim"],
        meta["max_hidden_dim"],
        decode_config(meta["config"]),
        data,
    )
    validate_complete_native(snapshot)
    return snapshot


class CircadianCheckpointCodec:
    def encode(
        self, state: CircadianNetworkSnapshot, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        validate_configuration(binding, limits)
        try:
            body = _native_body(state, limits)
            body.update(codec_version=2, kind="circadian_full_v2", binding=asdict(binding))
            specs = _specifications(body, False, limits)
            _preflight(body, specs, limits, False)
            validate_complete_native(state)
            for path, shape, dtype in specs:
                _set(body, path, encode_frame(_at(body, path), shape, dtype))
            return canonical_json(body)
        except (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            UnicodeError,
            OverflowError,
            FloatingPointError,
        ) as error:
            raise ValueError("invalid/unsupported complete circadian snapshot") from error

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> CircadianNetworkSnapshot:
        validate_configuration(binding, limits)
        require_codec_digest(expected_sha256)
        if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
            raise ValueError("circadian codec requires bounded exact bytes")
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("content differs from independently expected digest")
        try:
            body = exact_object(
                json.loads(raw, object_pairs_hook=unique_pairs),
                {"codec_version", "kind", "binding", "metadata", "state"},
            )
            if (
                type(body["codec_version"]) is not int
                or body["codec_version"] != 2
                or body["kind"] != "circadian_full_v2"
                or body["binding"] != asdict(binding)
            ):
                raise ValueError("unsupported version/kind or original binding differs")
            specs = _specifications(body, True, limits)
            _preflight(body, specs, limits, True)
            if canonical_json(body) != raw:
                raise ValueError("native checkpoint bytes must be canonical")
            for path, shape, dtype in specs:
                _set(body, path, decode_frame(_at(body, path), shape, dtype))
            return _snapshot(body)
        except (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            RecursionError,
            UnicodeError,
            OverflowError,
            FloatingPointError,
        ) as error:
            raise ValueError("invalid/corrupted complete circadian codec bytes") from error
