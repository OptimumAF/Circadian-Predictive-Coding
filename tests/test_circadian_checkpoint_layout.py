"""Storage and native structural continuity; no disk/live recovery admission."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.adapters.numpy_checkpoint_frames import decode_frame, encode_frame
from test_circadian_checkpoint_codec import FEATURES, TARGETS, LIMITS, model, encode, decode

WORK = dict(wake_updates=0, replay_updates=0, sleep_calls=0, restores=0, structural=0)


@pytest.fixture(scope="session", autouse=True)
def should_record_new_structural_fixture_allowance(tmp_path_factory):
    yield
    assert WORK["wake_updates"] + WORK["replay_updates"] <= 32
    assert WORK["sleep_calls"] <= 16 and WORK["restores"] <= 8 and WORK["structural"] <= 16
    path = tmp_path_factory.mktemp("layout-native-work") / "layout-native-work.json"
    path.write_text(json.dumps(WORK, sort_keys=True), encoding="utf8")


def assert_array_preserved(original, decoded):
    assert original.dtype == decoded.dtype and original.shape == decoded.shape
    assert original.tobytes(order="C") == decoded.tobytes(order="C")
    assert original.flags.c_contiguous == decoded.flags.c_contiguous
    assert original.flags.f_contiguous == decoded.flags.f_contiguous
    assert decoded.flags.owndata and not np.shares_memory(original, decoded)


@pytest.mark.parametrize("dtype", ["<f8", "<i8", "<i4", "|b1"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_should_preserve_each_supported_dtype_and_storage_order(dtype, order):
    original = np.array([[0, 1, 0], [1, 0, 1]], dtype=dtype, order=order)
    if dtype == "<f8":
        original[0, 0] = -0.0
    frame = encode_frame(original, original.shape, dtype)
    assert frame["order"] == order
    decoded = decode_frame(frame, original.shape, dtype)
    assert_array_preserved(original, decoded)
    assert encode_frame(decoded, original.shape, dtype) == frame
    decoded[:] = 0
    assert original[0, 1] == 1


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1), (1, 1)])
def test_should_canonicalize_ambiguous_storage_metadata_without_changing_flags(shape):
    original = np.zeros(shape, dtype="<f8", order="F")
    frame = encode_frame(original, shape, "<f8")
    assert frame["order"] == "C"
    assert_array_preserved(original, decode_frame(frame, shape, "<f8"))
    frame["order"] = "F"
    with pytest.raises(ValueError):
        decode_frame(frame, shape, "<f8")


@pytest.mark.parametrize("order", ["A", "K", "f", "", 1, True, None, []])
def test_should_refuse_invalid_order_before_any_native_allocation(order, monkeypatch):
    body = json.loads(encode(model().snapshot_state()))
    body["state"]["weight_input_hidden"]["order"] = order

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupt order reached native allocation")

    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        decode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize("fault", ["v1", "missing_order", "extra_frame"])
def test_should_refuse_old_or_unknown_frame_schema(fault):
    body = json.loads(encode(model().snapshot_state()))
    assert body["codec_version"] == 2 and body["kind"] == "circadian_full_v2"
    if fault == "v1":
        body.update(codec_version=1, kind="circadian_full_v1")
    elif fault == "missing_order":
        del body["state"]["weight_input_hidden"]["order"]
    else:
        body["state"]["weight_input_hidden"]["strides"] = [32, 8]
    with pytest.raises(ValueError):
        decode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize("layout", ["strided", "reversed"])
def test_should_refuse_unsupported_layout_before_frame_materialization(layout, monkeypatch):
    saved = model().snapshot_state()
    original = saved.state["weight_input_hidden"]
    array = np.zeros((3, 8))[:, ::2] if layout == "strided" else original[:, ::-1]
    saved.state["weight_input_hidden"] = array

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported layout reached serialization")

    monkeypatch.setattr("src.adapters.circadian_checkpoint_codec.encode_frame", forbidden)
    with pytest.raises(ValueError):
        encode(saved)
    assert saved.state["weight_input_hidden"] is array


def test_should_include_order_metadata_in_exact_prospective_wire_bound(monkeypatch):
    from src.adapters.circadian_checkpoint_codec import CircadianCheckpointCodec
    from test_circadian_checkpoint_codec import BINDING

    saved = model().snapshot_state()
    raw = encode(saved)
    limits = replace(LIMITS, max_encoded_bytes=len(raw))
    assert CircadianCheckpointCodec().encode(saved, binding=BINDING, limits=limits) == raw
    assert decode(raw, limits=limits).input_dim == 2

    def forbidden(*args, **kwargs):
        raise AssertionError("insufficient wire budget reached serialization")

    monkeypatch.setattr("src.adapters.circadian_checkpoint_codec.encode_frame", forbidden)
    with pytest.raises(ValueError):
        CircadianCheckpointCodec().encode(
            saved, binding=BINDING, limits=replace(limits, max_encoded_bytes=len(raw) - 1)
        )


def assert_snapshot_arrays_preserved(original, decoded):
    for name, value in original.state.items():
        other = decoded.state[name]
        if type(value) is np.ndarray:
            assert_array_preserved(value, other)
        elif type(value) is list and value and type(value[0]) is np.ndarray:
            for left, right in zip(value, other):
                assert_array_preserved(left, right)
    for left, right in zip(original.state["_replay_memory"], decoded.state["_replay_memory"]):
        assert_array_preserved(left.input_batch, right.input_batch)
        assert_array_preserved(left.target_batch, right.target_batch)


@pytest.mark.parametrize("retention", ["base", "content_hash", "recent_fifo", "seeded_reservoir"])
@pytest.mark.parametrize("wake_only", [False, True])
def test_should_continue_native_structural_state_with_exact_predictions(retention, wake_only):
    original = model(retention, wake_only)
    original._split_neurons((0,))
    WORK["structural"] += 1
    original._remove_neurons((0,))
    WORK["structural"] += 1
    saved = original.snapshot_state()
    weight = saved.state["weight_input_hidden"]
    assert weight.flags.f_contiguous and not weight.flags.c_contiguous
    raw = encode(saved)
    decoded = decode(raw)
    assert_snapshot_arrays_preserved(saved, decoded)
    assert encode(decoded) == raw
    candidate = model(retention, wake_only)
    candidate.restore_state(decoded)
    WORK["restores"] += 1
    assert original.predict_proba(FEATURES).tobytes() == candidate.predict_proba(FEATURES).tobytes()
    first = original.train_epoch(FEATURES, TARGETS, 0.03, 2, 0.2)
    WORK["wake_updates"] += 1
    second = candidate.train_epoch(FEATURES, TARGETS, 0.03, 2, 0.2)
    WORK["wake_updates"] += 1
    assert first.energy == second.energy
    assert encode(original.snapshot_state()) == encode(candidate.snapshot_state())
    for value in (original, candidate):
        before = value._replay_updates
        WORK["sleep_calls"] += 1
        value.sleep_event()
        WORK["replay_updates"] += value._replay_updates - before
    current = original.snapshot_state()
    continued = candidate.snapshot_state()
    assert encode(current) == encode(continued)
    assert_snapshot_arrays_preserved(current, continued)
    assert original.predict_proba(FEATURES).tobytes() == candidate.predict_proba(FEATURES).tobytes()
    assert encode(saved) == raw
