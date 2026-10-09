"""Pure bounded metadata and tiny NumPy descriptors; no native model calls."""

from dataclasses import replace
import numpy as np
import pytest

from src.adapters.numpy_replay_origins import replay_copy_bytes, replay_payload_fingerprint
from src.core.circadian_predictive_coding import ReplaySnapshot
from src.core.replay_origin import (
    ReplayOriginAdmission,
    ReplayOriginData,
    replay_origin_metadata_digest,
)
from test_managed_replay_origins import LIMITS


def data():
    return ReplayOriginData(
        ("e", "s"), "label", "actor", "candidate", "subject", "source", 0, 1, 1, 3, 1, 32, "0" * 64
    )


def test_should_charge_exact_metadata_accounting_and_keep_no_refund():
    admission = ReplayOriginAdmission(LIMITS)
    admission.start(0)
    admission.reserve(data(), 0)
    first = admission.accounting(1)
    assert first.invocations_started == first.records_created == 1
    assert first.metadata_bytes_charged > 2048
    assert admission.accounting(0).metadata_bytes_charged == first.metadata_bytes_charged
    assert replay_origin_metadata_digest(data(), 4096) == replay_origin_metadata_digest(
        data(), 4096
    )


def test_should_refuse_corrupt_or_renewed_accounting():
    admission = ReplayOriginAdmission(LIMITS)
    admission.start(0)
    spent = admission._progress
    admission._progress = replace(spent, invocations_started=0)
    with pytest.raises(ValueError, match="rewound"):
        admission.accounting()
    admission._progress = spent
    admission.limits = replace(LIMITS)
    with pytest.raises(ValueError, match="renewed"):
        admission.accounting()


@pytest.mark.parametrize(
    "field", ["max_metadata_bytes", "max_live_records", "max_records_created", "max_invocations"]
)
def test_should_refuse_exact_admission_boundary_without_refunding(field):
    limits = replace(LIMITS, **{field: 1})
    admission = ReplayOriginAdmission(limits)
    if field == "max_metadata_bytes":
        with pytest.raises(ValueError, match="limit"):
            admission.start(0)
        assert admission.accounting().metadata_bytes_charged == 0
    else:
        admission.start(0)
        admission.reserve(data(), 0)
        spent = admission.accounting(1)
        with pytest.raises(ValueError, match="limit"):
            admission.start(1) if field == "max_invocations" else admission.reserve(data(), 1)
        assert admission.accounting(1) == spent


def test_should_refuse_unbounded_identifier_rendering_before_charge():
    admission = ReplayOriginAdmission(replace(LIMITS, max_metadata_bytes=2048))
    admission.start(0)
    with pytest.raises(ValueError, match="identifier"):
        admission.reserve(replace(data(), subject_id="x" * 2048), 0)
    assert admission.accounting().records_created == 0


@pytest.mark.parametrize("start,count", [(True, 1), (-1, 1), (0, 0), (1, 2)])
def test_should_refuse_copy_range_before_array_access(start, count):
    features, targets = np.ones((2, 3)), np.zeros((2, 1))
    with pytest.raises(ValueError, match="range"):
        replay_copy_bytes(features, targets, start, count, 4096)


def test_should_measure_no_copy_and_fingerprint_bound_payload_integrity():
    features, targets = np.ones((1, 3)), np.zeros((1, 1))
    snapshot = ReplaySnapshot(features, targets, 0.5, 0.0)
    assert replay_copy_bytes(features, targets, 0, 1, 32) == 32
    size, original = replay_payload_fingerprint(snapshot, 32)
    assert size == 32
    features[0, 0] = 0.0
    assert replay_payload_fingerprint(snapshot, 32)[1] != original
    features.flags.writeable = False
    assert replay_payload_fingerprint(snapshot, 32)[1] != original
    with pytest.raises(ValueError, match="byte bound"):
        replay_copy_bytes(features, targets, 0, 1, 31)


def test_should_refuse_noncontiguous_or_oversized_payload_hashing_without_copy():
    features, targets = np.ones((1, 6))[:, ::2], np.zeros((1, 1))
    snapshot = ReplaySnapshot(features, targets, 0.5, 0.0)
    with pytest.raises(ValueError, match="contiguous"):
        replay_payload_fingerprint(snapshot, 32)
    with pytest.raises(ValueError, match="byte bound"):
        replay_payload_fingerprint(snapshot, 31)
