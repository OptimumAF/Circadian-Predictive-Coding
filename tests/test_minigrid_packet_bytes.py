"""Exact packet-bit comparison; no learner, forward, training or physics work."""

from dataclasses import replace

import numpy as np
import pytest

from src.adapters import minigrid_learning_pilot as pilot
from src.app.coupled_attribution import AttributionBatch


def large_array(layout):
    values = np.arange(54 * 296, dtype=np.float64).reshape(54, 296)
    if layout == "fortran":
        return np.asfortranarray(values)
    if layout == "negative":
        return values[::-1, ::-1]
    if layout == "strided":
        return np.repeat(values, 2, axis=1)[:, ::2]
    if layout == "readonly":
        values.setflags(write=False)
    return values


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "negative", "strided", "readonly"])
@pytest.mark.parametrize("change", ["none", "first", "middle", "last"])
def test_should_match_legacy_c_order_bytes_on_large_native_arrays(layout, change):
    values = large_array(layout)
    before = values.tobytes()
    if change != "none":
        if layout == "readonly":
            values.setflags(write=True)
        index = {"first": (0, 0), "middle": (27, 148), "last": (-1, -1)}[change]
        values[index] += 0.25
    assert pilot._same_packet_bytes(values, before) is (values.tobytes() == before)


@pytest.mark.parametrize("bits", [0x8000000000000000, 0x7FF8000000000001, 0x7FF8000000000002])
def test_should_compare_signed_zero_and_nan_payloads_as_bits(bits):
    values = np.zeros((54, 296), dtype=np.float64)
    words = values.view(np.uint64)
    words[27, 148] = bits
    frozen = values.tobytes()
    assert pilot._same_packet_bytes(values, frozen) is True
    words[27, 148] ^= np.uint64(1)
    assert pilot._same_packet_bytes(values, frozen) is False


@pytest.mark.parametrize("shape", [(1, 1), (1, 8191), (1, 8192), (1, 8193)])
def test_should_keep_comparison_exact_across_fixed_size_boundary(shape):
    values = np.zeros(shape, dtype=np.float64)
    assert pilot._same_packet_bytes(values, values.tobytes()) is True
    assert pilot._same_packet_bytes(values, values.tobytes()[:-1]) is False


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, ">f8"])
def test_should_keep_other_dtypes_on_legacy_comparison(dtype):
    values = np.zeros((54, 296), dtype=dtype)
    frozen = values.tobytes()
    assert pilot._same_packet_bytes(values, frozen) is True
    values[-1, -1] = 1
    assert pilot._same_packet_bytes(values, frozen) is False


def test_should_keep_subclass_callback_and_failure_behavior():
    class CallbackArray(np.ndarray):
        def tobytes(self):
            raise RuntimeError("original byte port")

    values = np.zeros((54, 296)).view(CallbackArray)
    with pytest.raises(RuntimeError, match="original byte port"):
        pilot._same_packet_bytes(values, bytes(values.nbytes))


def test_should_keep_object_array_comparison_free_of_payload_equality_callbacks():
    class Payload:
        def __eq__(self, other):
            pytest.fail("object array byte comparison invoked payload equality")

    values = np.empty((100, 100), dtype=object)
    values.fill(Payload())
    frozen = values.tobytes()
    assert pilot._same_packet_bytes(values, frozen) is True
    values[-1, -1] = None
    assert pilot._same_packet_bytes(values, frozen) is False


def test_should_preserve_opaque_frozen_value_equality_failure():
    class Frozen:
        def __eq__(self, other):
            raise RuntimeError("original frozen comparison")

    with pytest.raises(RuntimeError, match="original frozen comparison"):
        pilot._same_packet_bytes(large_array("contiguous"), Frozen())


def test_should_compare_large_operands_through_views_without_full_payload_copy(monkeypatch):
    values = large_array("contiguous")
    frozen = values.tobytes()
    compare = np.array_equal
    calls = []

    def observe(actual, expected):
        assert actual.dtype == expected.dtype == np.uint64
        assert np.shares_memory(actual, values)
        assert np.shares_memory(expected, np.frombuffer(frozen, np.uint64))
        assert not expected.flags.writeable
        calls.append(True)
        return compare(actual, expected)

    monkeypatch.setattr(np, "array_equal", observe)
    assert pilot._same_packet_bytes(values, frozen) is True
    assert calls == [True]


@pytest.mark.parametrize("field", ["dtype", "shape"])
def test_should_reject_metadata_before_reading_array_payload(monkeypatch, field):
    packet = AttributionBatch(
        "train",
        "train-catalogue",
        tuple(f"r{i}" for i in range(54)),
        np.zeros((54, 296)),
        np.zeros((54, 1)),
        num_classes=7,
    )
    binding = pilot._signature((packet,))
    value = packet.features.astype(np.float32) if field == "dtype" else packet.features.T
    object.__setattr__(packet, "features", value)

    def forbidden(*args):
        pytest.fail("changed metadata reached payload comparison")

    monkeypatch.setattr(pilot, "_same_packet_bytes", forbidden)
    assert pilot._check_signature((packet,), binding) is False


def test_should_recheck_original_large_packet_values_without_caching_validity():
    packet = AttributionBatch(
        "train",
        "train-catalogue",
        tuple(f"r{i}" for i in range(54)),
        np.zeros((54, 296)),
        np.zeros((54, 1)),
        num_classes=7,
    )
    binding = pilot._signature((packet,))
    assert pilot._check_signature((packet,), binding) is True
    packet.features[-1, -1] = 0.25
    assert pilot._check_signature((packet,), binding) is False
    packet.features[-1, -1] = 0.0
    assert pilot._check_signature((packet,), binding) is True
    packet.targets[-1, -1] = 1
    assert pilot._check_signature((packet,), binding) is False


def test_should_still_refuse_batch_owner_replacement_with_equal_values():
    packet = AttributionBatch(
        "train",
        "train-catalogue",
        tuple(f"r{i}" for i in range(54)),
        np.zeros((54, 296)),
        np.zeros((54, 1)),
        num_classes=7,
    )
    binding = pilot._signature((packet,))
    assert pilot._check_signature((replace(packet),), binding) is False
