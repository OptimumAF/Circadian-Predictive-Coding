"""Bounded pure metadata/numeric binding tests; no native or consent authority."""

from dataclasses import FrozenInstanceError, asdict, dataclass, replace
import json

import numpy as np
import pytest

from src.core.checkpoint_content import CheckpointContentLimits
from src.core.inbox_origin import (
    InboxOriginData,
    inbox_numeric_stamp,
    inbox_origin_metadata,
    inbox_payload_binding,
)
from src.core.replay_origin import RECORD_OVERHEAD_BYTES, ReplayOriginAdmission, ReplayOriginLimits


CONTENT = CheckpointContentLimits(64, 4096, 4096, 8)


@dataclass
class PureResources:
    arrays: int = 0
    bytes: int = 0
    fake_sequences: int = 0

    def array(self, values, dtype=float):
        size = len(values) * len(values[0]) * np.dtype(dtype).itemsize
        assert self.arrays + 1 <= 16 and self.bytes + size <= 4096
        value = np.array(values, dtype=dtype)
        assert value.nbytes == size
        self.arrays += 1
        self.bytes += size
        return value


@pytest.fixture(scope="module")
def resources(tmp_path_factory):
    work = PureResources()
    yield work
    assert work.arrays <= 16 and work.bytes <= 4096 and work.fake_sequences <= 16
    (tmp_path_factory.mktemp("pure-inbox-origin") / "resources.json").write_text(
        json.dumps(asdict(work), indent=2), encoding="utf8"
    )


@pytest.fixture(scope="module")
def arrays(resources):
    # Separate original and consumed allocations, with no payload-copy function.
    return (
        resources.array([[1.0, 0.0]]),
        resources.array([[0.0]]),
        resources.array([[1.0, 0.0]]),
        resources.array([[0.0]]),
    )


def metadata(**changes):
    values = dict(
        key=("e1", "s1"),
        event_id="label-s1",
        actor_version="actor-0",
        learner_version="candidate-0",
        subject_id="person-1",
        source_id="local",
        observed_at=1,
        arrived_at=3,
        update_number=1,
        payload_bytes=24,
        features_digest="a" * 64,
        targets_digest="b" * 64,
    )
    return InboxOriginData(**dict(values, **changes))


def test_should_encode_exact_immutable_metadata_without_native_authority():
    value = metadata()
    encoded = inbox_origin_metadata(value, 4096)
    result = json.loads(encoded)
    assert result["key"] == ["e1", "s1"] and result["update_number"] == 1
    assert result["features_digest"] == "a" * 64
    assert inbox_origin_metadata(value, 4096) == encoded
    with pytest.raises(FrozenInstanceError):
        setattr(value, "update_number", 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"key": ["e1", "s1"]},
        {"update_number": True},
        {"update_number": 0},
        {"payload_bytes": 0},
        {"features_digest": "a" * 63},
        {"targets_digest": "A" * 64},
        {"arrived_at": 2**63},
    ],
)
def test_should_reject_nonexact_or_unbounded_metadata(changes):
    with pytest.raises(ValueError):
        metadata(**changes)


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_should_refuse_mutated_metadata_schema(change):
    value = metadata()
    if change == "extra":
        object.__setattr__(value, "unexpected", 1)
    else:
        del vars(value)["event_id"]
    with pytest.raises(ValueError, match="fields changed"):
        inbox_origin_metadata(value, 4096)


def test_should_reject_metadata_subclass_before_custom_callbacks():
    calls = []

    class MetadataSubclass(InboxOriginData):
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("unsupported metadata callback ran")

    value = object.__new__(MetadataSubclass)
    with pytest.raises(ValueError, match="exact data"):
        inbox_origin_metadata(value, 4096)
    assert calls == []


def test_should_enforce_metadata_encoding_limit():
    value = metadata()
    size = len(inbox_origin_metadata(value, 4096))
    with pytest.raises(ValueError, match="bound|capacity"):
        inbox_origin_metadata(value, size - 1)


def test_should_bind_original_values_to_distinct_actual_consumed_buffers(arrays):
    features, targets, consumed_features, consumed_targets = arrays
    assert features is not consumed_features and targets is not consumed_targets
    size, feature_stamp, target_stamp = inbox_payload_binding(
        arrays[0], arrays[1], arrays[2], arrays[3], CONTENT
    )
    assert size == features.nbytes + targets.nbytes == 24
    assert feature_stamp == inbox_numeric_stamp(consumed_features, CONTENT)
    assert target_stamp == inbox_numeric_stamp(consumed_targets, CONTENT)


def test_should_exclude_pointer_and_writeability_from_numeric_stamp(arrays):
    features, _, consumed_features, _ = arrays
    original = inbox_numeric_stamp(features, CONTENT)
    features.flags.writeable = False
    try:
        assert inbox_numeric_stamp(features, CONTENT) == original
        assert inbox_numeric_stamp(consumed_features, CONTENT) == original
    finally:
        features.flags.writeable = True


@pytest.mark.parametrize("index", [2, 3])
def test_should_reject_source_and_consumed_value_mismatch(arrays, index):
    value = arrays[index]
    old = value[0, 0]
    value[0, 0] = old + 0.25
    try:
        with pytest.raises(ValueError, match="actual consumed native inputs"):
            inbox_payload_binding(arrays[0], arrays[1], arrays[2], arrays[3], CONTENT)
    finally:
        value[0, 0] = old


def test_should_detect_buffer_mutation_and_refuse_nonfinite_values(arrays):
    features = arrays[0]
    old = features[0, 0]
    original = inbox_numeric_stamp(features, CONTENT)
    try:
        features[0, 0] = old + 0.25
        assert inbox_numeric_stamp(features, CONTENT) != original
        for invalid in (float("nan"), float("inf")):
            features[0, 0] = invalid
            with pytest.raises(ValueError, match="finite"):
                inbox_numeric_stamp(features, CONTENT)
    finally:
        features[0, 0] = old


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.uint16, np.float64])
def test_should_stamp_exact_supported_numeric_dtypes(resources, dtype):
    value = resources.array([[0, 1]], dtype=dtype)
    assert len(inbox_numeric_stamp(value, CONTENT)) == 64


def test_should_reject_custom_sequence_without_callbacks():
    calls = []

    class Sequence:
        def __len__(self):
            calls.append("len")
            raise AssertionError("custom sequence was traversed")

        def __iter__(self):
            calls.append("iter")
            raise AssertionError("custom sequence was traversed")

    with pytest.raises(ValueError, match="exact numeric arrays"):
        inbox_numeric_stamp(Sequence(), CONTENT)
    assert calls == []


def test_should_reject_object_array_dtype(resources):
    value = resources.array([[1]], dtype=object)
    with pytest.raises(ValueError, match="contiguous numeric arrays"):
        inbox_numeric_stamp(value, CONTENT)


def test_should_enforce_array_capacity_before_numeric_finite_scratch(arrays, monkeypatch):
    import src.core.inbox_origin as module

    def forbidden(*args, **kwargs):
        pytest.fail("finite scratch allocated before original byte limit refusal")

    monkeypatch.setattr(module.np, "isfinite", forbidden)
    with pytest.raises(ValueError, match="array byte bound"):
        inbox_numeric_stamp(arrays[0], replace(CONTENT, max_array_bytes=1))


def test_should_charge_pair_under_original_shared_admission():
    limits = ReplayOriginLimits(4, 4, 4, 8192, 120, 4096)
    admission = ReplayOriginAdmission(limits)
    value = metadata()
    admission.reserve_inbox(value, 0)
    first = admission.accounting(1)
    assert first.records_created == 1 and first.invocations_started == 0
    assert (
        first.metadata_bytes_charged
        == len(inbox_origin_metadata(value, 8192)) + RECORD_OVERHEAD_BYTES
    )
    admission.start(1)
    second = admission.accounting(1)
    assert admission.limits is limits and admission._original_limits is limits
    assert second.records_created == 1 and second.invocations_started == 1
    assert second.metadata_bytes_charged == first.metadata_bytes_charged + RECORD_OVERHEAD_BYTES


@pytest.mark.parametrize("bound", ["live", "records", "metadata", "payload"])
def test_should_refuse_original_exhausted_pair_allowance_without_partial_charge(bound):
    value = metadata()
    encoded_size = len(inbox_origin_metadata(value, 8192))
    limits = ReplayOriginLimits(
        1 if bound == "live" else 4,
        1 if bound == "records" else 4,
        4,
        encoded_size + RECORD_OVERHEAD_BYTES - 1 if bound == "metadata" else 8192,
        120,
        23 if bound == "payload" else 4096,
    )
    admission = ReplayOriginAdmission(limits)
    live = 0
    if bound in ("live", "records"):
        admission.reserve_inbox(value, 0)
        live = 1
    before = admission.accounting(live)
    with pytest.raises(ValueError):
        admission.reserve_inbox(value, live)
    assert admission.accounting(live) == before
    assert admission.limits is limits


def test_should_reject_hostile_mutated_payload_counter_before_comparison():
    calls = []

    class HostileCounter:
        def __gt__(self, other):
            calls.append("comparison")
            raise AssertionError("hostile counter was compared")

        def __ge__(self, other):
            calls.append("comparison")
            raise AssertionError("hostile counter was compared")

    value = metadata()
    object.__setattr__(value, "payload_bytes", HostileCounter())
    admission = ReplayOriginAdmission(ReplayOriginLimits(4, 4, 4, 8192, 120, 4096))
    with pytest.raises(ValueError, match="bounded exact integers"):
        admission.reserve_inbox(value, 0)
    assert calls == []
    assert admission.accounting().records_created == 0


def test_should_reject_both_paired_byte_cap_before_finite_scratch(arrays, monkeypatch):
    import src.core.inbox_origin as module

    def forbidden(*args, **kwargs):
        pytest.fail("finite scratch allocated before paired byte bound refusal")

    monkeypatch.setattr(module.np, "isfinite", forbidden)
    limits = replace(CONTENT, max_array_bytes=20)
    with pytest.raises(ValueError, match="paired payload.*byte bound"):
        inbox_payload_binding(arrays[0], arrays[1], arrays[2], arrays[3], limits)
    # Same allocated arrays in reversed source/consumed pairs; no payload copies.
    with pytest.raises(ValueError, match="paired payload.*byte bound"):
        inbox_payload_binding(arrays[2], arrays[3], arrays[0], arrays[1], limits)


def test_should_bound_excess_metadata_before_json_copy(monkeypatch):
    import src.core.inbox_origin as module

    value = metadata()
    object.__setattr__(value, "subject_id", "x" * 32)

    def forbidden(*args, **kwargs):
        pytest.fail("JSON encoding ran before aggregate metadata capacity refusal")

    monkeypatch.setattr(module.json, "dumps", forbidden)
    with pytest.raises(ValueError, match="metadata capacity"):
        inbox_origin_metadata(value, 64)
