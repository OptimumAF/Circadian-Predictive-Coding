"""Complete native byte contract; no training or live recovery admission."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pytest

from src.adapters.backprop_checkpoint_codec import BackpropCheckpointCodec
from src.adapters.numpy_learners import BackpropLearner
from src.core.backprop_mlp import BackpropMLP
from src.core.checkpoint_codec import CodecBinding, CodecLimits


BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(32768, 8192, 8, 64)


def snapshot():
    value = BackpropLearner(
        BackpropMLP(3, 4, seed=0, hidden_dims=(4, 2)), learning_rate=0.1
    ).snapshot_state()
    value.state["_traffic_steps"] = 7
    for index, values in enumerate(value.state["_traffic_sums"], 1):
        values[:] = index + np.arange(values.size)
    value.state["bias_output"][:] = -0.0
    return value


def decode(raw, binding=BINDING, limits=LIMITS):
    return BackpropCheckpointCodec().decode(
        raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits
    )


def encoded(value=None):
    return BackpropCheckpointCodec().encode(
        snapshot() if value is None else value, binding=BINDING, limits=LIMITS
    )


def test_should_round_trip_every_native_field_and_required_alias_without_sharing():
    original = snapshot()
    raw = encoded(original)
    restored = decode(raw)
    assert restored.format_version == original.format_version
    assert restored.input_dim == original.input_dim
    assert restored.hidden_dims == original.hidden_dims
    assert restored.state.keys() == original.state.keys()
    assert encoded(restored) == raw
    for name, value in original.state.items():
        other = restored.state[name]
        if isinstance(value, np.ndarray):
            assert value.dtype == other.dtype and value.shape == other.shape
            assert value.tobytes() == other.tobytes()
            assert not np.shares_memory(value, other)
        elif type(value) is list:
            assert len(value) == len(other)
            for left, right in zip(value, other):
                assert left.dtype == right.dtype and left.shape == right.shape
                assert left.tobytes() == right.tobytes()
                assert not np.shares_memory(left, right)
        else:
            assert value == other
    assert restored.state["weight_input_hidden"] is restored.state["_hidden_weights"][0]
    assert restored.state["bias_hidden"] is restored.state["_hidden_biases"][0]
    restored.state["weight_input_hidden"][:] = 999
    assert encoded(original) == raw
    assert encoded(decode(raw)) == raw


@pytest.mark.parametrize("name", ["input_dim", "hidden_dims", "_traffic_steps", "bias_output"])
def test_should_refuse_missing_native_field(name):
    value = snapshot()
    del value.state[name]
    with pytest.raises(ValueError):
        encoded(value)


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "alias",
        "overlap",
        "dtype",
        "shape",
        "nan",
        "traffic",
        "steps",
        "metadata",
        "version",
    ],
)
def test_should_refuse_unsupported_or_inconsistent_native_graph(fault):
    value = snapshot()
    if fault == "extra":
        value.state["future_state"] = object()
    elif fault == "alias":
        value.state["weight_input_hidden"] = value.state["weight_input_hidden"].copy()
    elif fault == "overlap":
        value.state["_traffic_sums"][0] = value.state["_hidden_biases"][0].reshape(-1)
    elif fault == "dtype":
        value.state["bias_output"] = np.zeros((1, 1), dtype=np.float32)
    elif fault == "shape":
        value.state["bias_output"] = np.zeros((1, 2))
    elif fault == "nan":
        value.state["bias_output"][:] = np.nan
    elif fault == "traffic":
        value.state["_traffic_sums"][0][:] = -1
    elif fault == "steps":
        value.state["_traffic_steps"] = True
    elif fault == "metadata":
        value = replace(value, input_dim=2)
    else:
        value = replace(value, format_version=2)
    with pytest.raises(ValueError):
        encoded(value)


@pytest.mark.parametrize("field", ["source_sha256", "policy_sha256"])
def test_should_refuse_independently_changed_binding(field):
    raw = encoded()
    with pytest.raises(ValueError):
        decode(raw, binding=replace(BINDING, **{field: "c" * 64}))


def test_should_refuse_content_corruption_against_independent_digest():
    raw = encoded()
    with pytest.raises(ValueError):
        BackpropCheckpointCodec().decode(
            raw + b" ", binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "missing",
        "version",
        "dtype",
        "shape",
        "bytes",
        "nan",
        "traffic",
        "boolean_dimension",
        "aliases",
        "unknown_array",
    ],
)
def test_should_refuse_corrupted_schema_even_with_matching_content_digest(fault):
    body = json.loads(encoded())
    if fault == "extra":
        body["future"] = None
    elif fault == "missing":
        del body["traffic_steps"]
    elif fault == "version":
        body["codec_version"] = True
    elif fault == "boolean_dimension":
        body["input_dim"] = True
    elif fault == "aliases":
        body["aliases"]["bias_hidden"] = "traffic_sums/0"
    elif fault == "unknown_array":
        body["arrays"]["unknown"] = {}
    else:
        frame = body["arrays"]["traffic_sums"][0]
        if fault == "dtype":
            frame["dtype"] = "<f4"
        elif fault == "shape":
            frame["shape"] = [99999999999999]
        elif fault == "bytes":
            frame["data"] = "AA=="
        else:
            from base64 import b64encode

            values = np.full((4,), np.nan if fault == "nan" else -1.0, dtype="<f8")
            frame["data"] = b64encode(values.tobytes()).decode("ascii")
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        decode(raw)


@pytest.mark.parametrize("raw", [b"{}", b"\xff", b"{", b'{"a":1,"a":2}', b"[" * 1500])
def test_should_refuse_malformed_or_duplicate_json(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_should_enforce_exact_byte_limits_and_refuse_before_array_decode(monkeypatch):
    raw = encoded()
    arrays_bytes = sum(
        a.nbytes
        for name, value in snapshot().state.items()
        for a in (value if type(value) is list else [value])
        if isinstance(a, np.ndarray) and name not in ("weight_input_hidden", "bias_hidden")
    )
    limits = replace(LIMITS, max_encoded_bytes=len(raw), max_array_bytes=arrays_bytes)
    assert encoded(decode(raw, limits=limits)) == raw
    for small in [
        replace(limits, max_encoded_bytes=len(raw) - 1),
        replace(limits, max_array_bytes=arrays_bytes - 1),
        replace(limits, max_layers=1),
        replace(limits, max_dimension=3),
    ]:
        with pytest.raises(ValueError):
            BackpropCheckpointCodec().encode(snapshot(), binding=BINDING, limits=small)
        monkeypatch.setattr(
            np, "frombuffer", lambda *a, **kw: pytest.fail("array allocation before preflight")
        )
        with pytest.raises(ValueError):
            decode(raw, limits=small)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_encoded_bytes", True),
        ("max_array_bytes", 0),
        ("max_layers", -1),
        ("max_dimension", 2**63),
    ],
)
def test_should_refuse_invalid_resource_limits(field, value):
    with pytest.raises(ValueError):
        replace(LIMITS, **{field: value})


def test_should_refuse_noncanonical_bytes_and_wrong_exact_input_types():
    raw = encoded()
    with pytest.raises(ValueError):
        decode(raw + b" ")
    bad_inputs: tuple[Any, ...] = (bytearray(raw), None)
    for value in bad_inputs:
        with pytest.raises(ValueError):
            BackpropCheckpointCodec().decode(
                value, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
            )
    value = deepcopy(snapshot())
    value.state["_traffic_steps"] = 2**63
    with pytest.raises(ValueError):
        encoded(value)


def test_should_refuse_boolean_native_dimension_even_when_equal_to_metadata():
    value = BackpropLearner(BackpropMLP(3, 1, seed=0), learning_rate=0.1).snapshot_state()
    value.state["hidden_dims"] = (True,)
    with pytest.raises(ValueError):
        encoded(value)


@pytest.mark.parametrize("field", ["source_sha256", "policy_sha256"])
@pytest.mark.parametrize("value", ["A" * 64, "a" * 63, None, True])
def test_should_refuse_invalid_binding_digest(field, value):
    with pytest.raises(ValueError):
        replace(BINDING, **{field: value})


def test_should_refuse_wrong_binding_limits_and_expected_digest():
    codec = BackpropCheckpointCodec()
    raw = encoded()
    invalid: list[tuple[Any, Any, str]] = [
        (object(), LIMITS, "a" * 64),
        (BINDING, object(), "a" * 64),
        (BINDING, LIMITS, "A" * 64),
    ]
    for binding, limits, digest in invalid:
        with pytest.raises(ValueError):
            codec.decode(raw, binding=binding, limits=limits, expected_sha256=digest)


def test_should_round_trip_noncontiguous_values_with_required_alias():
    value = snapshot()
    original = value.state["_hidden_weights"][0]
    storage = np.zeros((original.shape[0], original.shape[1] * 2))
    storage[:, ::2] = original
    view = storage[:, ::2]
    assert not view.flags.c_contiguous
    value.state["_hidden_weights"][0] = value.state["weight_input_hidden"] = view
    restored = decode(encoded(value))
    assert restored.state["weight_input_hidden"].tobytes() == view.tobytes()
    assert restored.state["weight_input_hidden"] is restored.state["_hidden_weights"][0]
    assert not np.shares_memory(restored.state["weight_input_hidden"], view)
