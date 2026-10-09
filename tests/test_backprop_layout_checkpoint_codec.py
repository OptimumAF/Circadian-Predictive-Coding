"""Explicit storage/alias/budget/continuation controls; no recovery admission."""

from base64 import b64encode
from dataclasses import replace
from hashlib import sha256
import json
from typing import Any, Literal

import numpy as np
import pytest

from src.adapters.backprop_layout_checkpoint_codec import LayoutBackpropCheckpointCodec
from src.adapters.numpy_learners import BackpropLearner
from src.core.backprop_mlp import BackpropMLP
from src.core.checkpoint_codec import CodecBinding, CodecLimits

BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(32768, 8192, 8, 64)
FEATURES = np.array([[0.3, -0.2, 0.1], [-0.5, 0.4, 0.2], [0.1, 0.6, -0.3], [-0.7, -0.1, 0.5]])
TARGETS = np.array([[1.0], [0.0], [1.0], [0.0]])
WORK = dict(updates=0, restores=0, sleeps=0, structural=0)


@pytest.fixture(scope="session", autouse=True)
def should_record_fixed_native_work(tmp_path_factory):
    yield
    assert WORK["updates"] <= 36 and WORK["restores"] <= 12
    assert WORK["sleeps"] == 0 and WORK["structural"] == 0
    path = tmp_path_factory.mktemp("backprop-native-work") / "native-work.json"
    path.write_text(json.dumps(WORK, sort_keys=True), encoding="utf8")


def learner(hidden=(4, 2), order="F"):
    if order not in ("C", "F", "mixed"):
        raise ValueError("unsupported fixed fixture layout")
    native = BackpropMLP(3, hidden[0], seed=23, hidden_dims=hidden)
    for index, value in enumerate(native._hidden_weights):
        storage: Literal["C", "F"] = (
            "F" if order == "F" or (order == "mixed" and index % 2 == 0) else "C"
        )
        native._hidden_weights[index] = np.array(value, order=storage, copy=True)
    native.weight_input_hidden = native._hidden_weights[0]
    native.bias_output[:] = -0.0
    return BackpropLearner(native, learning_rate=0.03)


def encode(saved, limits=LIMITS):
    return LayoutBackpropCheckpointCodec().encode(saved, binding=BINDING, limits=limits)


def decode(raw, limits=LIMITS, binding=BINDING):
    return LayoutBackpropCheckpointCodec().decode(
        raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits
    )


def snapshot_arrays(saved):
    for name in sorted(saved.state):
        value = saved.state[name]
        if name in ("weight_input_hidden", "bias_hidden"):
            continue
        if type(value) is np.ndarray:
            yield value
        elif type(value) is list:
            yield from value


def assert_complete_preservation(original, decoded):
    assert original.state.keys() == decoded.state.keys()
    assert encode(decoded) == encode(original)
    assert decoded.state["weight_input_hidden"] is decoded.state["_hidden_weights"][0]
    assert decoded.state["bias_hidden"] is decoded.state["_hidden_biases"][0]
    for left, right in zip(snapshot_arrays(original), snapshot_arrays(decoded)):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
        assert left.flags.c_contiguous == right.flags.c_contiguous
        assert left.flags.f_contiguous == right.flags.f_contiguous
        assert right.flags.owndata and not np.shares_memory(left, right)


@pytest.mark.parametrize("order", ["C", "F"])
def test_should_preserve_each_supported_layout_alias_and_signed_zero(order):
    saved = learner(order=order).snapshot_state()
    raw = encode(saved)
    body = json.loads(raw)
    assert body["codec_version"] == 2 and body["kind"] == "backprop_layout_v2"
    assert body["arrays"]["hidden_weights"][0]["order"] == order
    detached = decode(raw)
    assert_complete_preservation(saved, detached)
    detached.state["weight_input_hidden"][:] = 999
    assert encode(saved) == raw


@pytest.mark.parametrize("hidden", [(4,), (4, 2), (1,), (3, 1, 2)])
@pytest.mark.parametrize("order", ["C", "F", "mixed"])
def test_should_continue_actual_native_state_with_exact_loss_and_predictions(hidden, order):
    original = learner(hidden, order)
    original._model.train_epoch(FEATURES, TARGETS, 0.03)
    WORK["updates"] += 1
    saved = original.snapshot_state()
    raw = encode(saved)
    detached = decode(raw)
    assert_complete_preservation(saved, detached)
    candidate = learner(hidden, order)
    candidate.restore_state(detached)
    WORK["restores"] += 1
    assert (
        original._model.predict_proba(FEATURES).tobytes()
        == candidate._model.predict_proba(FEATURES).tobytes()
    )
    first = original._model.train_epoch(FEATURES, TARGETS, 0.03)
    WORK["updates"] += 1
    second = candidate._model.train_epoch(FEATURES, TARGETS, 0.03)
    WORK["updates"] += 1
    assert first.loss == second.loss
    assert_complete_preservation(original.snapshot_state(), candidate.snapshot_state())
    assert (
        original._model.predict_proba(FEATURES).tobytes()
        == candidate._model.predict_proba(FEATURES).tobytes()
    )
    assert encode(saved) == raw


@pytest.mark.parametrize("name", sorted(learner().snapshot_state().state))
def test_should_refuse_every_missing_native_field(name):
    saved = learner().snapshot_state()
    del saved.state[name]
    with pytest.raises(ValueError):
        encode(saved)


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
        "record",
        "strided",
        "reversed",
    ],
)
def test_should_refuse_invalid_native_state_before_serialization(fault, monkeypatch):
    saved = learner().snapshot_state()
    if fault == "extra":
        saved.state["future"] = object()
    elif fault == "alias":
        saved.state["weight_input_hidden"] = saved.state["weight_input_hidden"].copy()
    elif fault == "overlap":
        saved.state["_traffic_sums"][0] = saved.state["bias_hidden"].reshape(-1)
    elif fault == "dtype":
        saved.state["bias_output"] = np.zeros((1, 1), dtype=np.float32)
    elif fault == "shape":
        saved.state["bias_output"] = np.zeros((2, 1))
    elif fault == "nan":
        saved.state["bias_output"][:] = np.nan
    elif fault == "traffic":
        saved.state["_traffic_sums"][0][:] = -1
    elif fault == "steps":
        saved.state["_traffic_steps"] = True
    elif fault == "metadata":
        saved = replace(saved, input_dim=2)
    elif fault == "record":
        object.__setattr__(saved, "future", object())
    else:
        value = (
            np.zeros((3, 8))[:, ::2]
            if fault == "strided"
            else saved.state["weight_input_hidden"][:, ::-1]
        )
        saved.state["_hidden_weights"][0] = saved.state["weight_input_hidden"] = value

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid native state reached serialization")

    monkeypatch.setattr("src.adapters.backprop_layout_checkpoint_codec.encode_frame", forbidden)
    with pytest.raises(ValueError):
        encode(saved)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "kind",
        "old",
        "extra",
        "missing",
        "aliases",
        "dtype",
        "shape",
        "order",
        "order_bool",
        "missing_order",
        "ambiguous",
        "bytes",
        "nan",
        "traffic",
    ],
)
def test_should_refuse_late_corrupt_frame_before_any_native_materialization(fault, monkeypatch):
    body = json.loads(encode(learner().snapshot_state()))
    frame = body["arrays"]["bias_output"]
    if fault == "version":
        body["codec_version"] = True
    elif fault == "kind":
        body["kind"] = "circadian_full_v2"
    elif fault == "old":
        body.update(codec_version=1, kind="backprop_full_v1")
    elif fault == "extra":
        frame["strides"] = [8, 8]
    elif fault == "missing":
        del body["traffic_steps"]
    elif fault == "aliases":
        body["aliases"]["bias_hidden"] = "traffic_sums/0"
    elif fault == "dtype":
        frame["dtype"] = "<f4"
    elif fault == "shape":
        frame["shape"] = [True, 1]
    elif fault == "order":
        frame["order"] = "K"
    elif fault == "order_bool":
        frame["order"] = True
    elif fault == "missing_order":
        del frame["order"]
    elif fault == "ambiguous":
        frame["order"] = "F"
    elif fault == "bytes":
        frame["data"] = "!!!!!!!!!!!!"
    elif fault == "nan":
        frame["data"] = b64encode(np.array([[np.nan]]).tobytes()).decode()
    else:
        frame = body["arrays"]["traffic_sums"][-1]
        frame["data"] = b64encode(np.full((2,), -1.0).tobytes()).decode()

    def forbidden(*args, **kwargs):
        raise AssertionError("corruption reached native materialization")

    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        decode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def test_should_enforce_exact_unique_array_and_order_inclusive_wire_bounds(monkeypatch):
    saved = learner().snapshot_state()
    raw = encode(saved)
    limits = replace(
        LIMITS,
        max_encoded_bytes=len(raw),
        max_array_bytes=sum(a.nbytes for a in snapshot_arrays(saved)),
    )
    assert encode(saved, limits) == raw and encode(decode(raw, limits)) == raw

    def forbidden(*args, **kwargs):
        raise AssertionError("insufficient budget reached materialization")

    monkeypatch.setattr(np, "frombuffer", forbidden)
    monkeypatch.setattr("src.adapters.backprop_layout_checkpoint_codec.encode_frame", forbidden)
    for small in (
        replace(limits, max_encoded_bytes=len(raw) - 1),
        replace(limits, max_array_bytes=limits.max_array_bytes - 1),
        replace(limits, max_layers=1),
        replace(limits, max_dimension=3),
    ):
        with pytest.raises(ValueError):
            encode(saved, small)
        with pytest.raises(ValueError):
            decode(raw, small)


@pytest.mark.parametrize("field", ["source_sha256", "policy_sha256"])
def test_should_refuse_independently_changed_bindings(field):
    raw = encode(learner().snapshot_state())
    with pytest.raises(ValueError):
        decode(raw, binding=replace(BINDING, **{field: "c" * 64}))


@pytest.mark.parametrize("raw", [b"{}", b"\xff", b"{", b'{"a":1,"a":2}', b"[" * 1500])
def test_should_refuse_malformed_duplicate_or_unbounded_json(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_should_refuse_content_noncanonical_bytes_and_exact_input_type():
    raw = encode(learner().snapshot_state())
    with pytest.raises(ValueError):
        decode(raw + b" ")
    with pytest.raises(ValueError):
        LayoutBackpropCheckpointCodec().decode(
            raw + b" ", binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )
    with pytest.raises(ValueError):
        mutable: Any = bytearray(raw)
        LayoutBackpropCheckpointCodec().decode(
            mutable, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )
