"""Bounded original app arrival proof and paid preparation after logical TTL.

Why this: a cleanup-only weak proof must preserve the original current arrival
across callbacks without granting expired access or inventing applied work.
No actual cleanup, native learner or history publication is exercised here.
"""

from dataclasses import asdict, dataclass, fields, replace
import json
from typing import Any
from weakref import ReferenceType, ref

import numpy as np
import pytest

import src.app.expiry_untrained_qualification as qualification
from src.app.managed_inbox_origins import ManagedInboxOrigins
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.untrained_inbox_erasure import _Arrival, _arrival_stamp, _require_arrival
from src.app.untrained_inbox_origins import UntrainedInboxOrigin, untrained_inbox_origin_stamp
from src.core.untrained_inbox_origin import untrained_inbox_origin_metadata
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.replay_origin import ReplayOriginLimits, ReplayOriginPorts
from src.core.replay_write_origin import ReplayWriteLimits
from src.core.resource_sharing import SharingLimits
from test_expiry_inbox_qualification import (
    FIRST,
    SECOND,
    FakeWork,
    Graph,
    NumericFakeLearner,
    _conservation,
    _fresh,
    _original_gates,
    _queue,
    _record,
    _train,
    _measure,
    _footprint,
    _erase,
    _auxiliary,
    _outside_copy,
    _growth,
    _model_reference,
    _retained,
    _copy_bytes,
    _payloads,
    _fingerprint,
)


@dataclass
class ArrivalWork(FakeWork):
    buffer_copies: int = 0
    buffer_copy_bytes: int = 0
    persistent_witnesses: int = 0
    metadata_preparation_attempts: int = 0
    metadata_preparation_successes: int = 0

    def _reserve_arrays(self, count, size):
        assert self.arrays + count <= 384 and self.array_bytes + size <= 48 * 1024
        self.arrays += count
        self.array_bytes += size

    def buffer_copy(self, value):
        self._reserve_arrays(1, value.nbytes)
        copied = value.copy()
        assert type(copied) is np.ndarray and copied is not value
        assert copied.nbytes == value.nbytes and not np.shares_memory(copied, value)
        self.buffer_copies += 1
        self.buffer_copy_bytes += copied.nbytes
        return copied


@pytest.fixture(scope="module")
def arrival_work(tmp_path_factory):
    work = ArrivalWork()
    try:
        yield work
    finally:
        (tmp_path_factory.mktemp("expiry-arrival-fake-work") / "resources.json").write_text(
            json.dumps(asdict(work), indent=2),
            encoding="utf8",
        )
    assert work.graphs <= 24 and work.updates <= 48
    assert work.arrays <= 384 and work.array_bytes <= 48 * 1024
    assert work.learners <= 72 and work.forks <= 48 and work.pending_consumed is None
    assert work.graphs == 23 and work.updates == 23
    assert work.learners == 69 and work.forks == 46
    assert work.arrays == 181 and work.array_bytes == 2192
    assert work.consumed_arrays == work.native_copy_arrays == 46
    assert work.consumed_bytes == work.native_copy_bytes == 552
    assert work.buffer_copies == 2 and work.buffer_copy_bytes == 24
    assert 1 <= work.metadata_preparation_successes < work.metadata_preparation_attempts <= 8
    assert work.persistent_witnesses == 4 + work.metadata_preparation_successes


def _ready(work, *, source=True, label=True, payload_limit=4096):
    graph = _fresh(work, payload_limit=payload_limit)
    _queue(graph, "s2", source=source, label=label)
    assert graph.history is graph.ledger._inbox_history_birth()
    assert graph.runtime._budget.updates_completed == 1
    assert SECOND not in graph.runtime._inbox._applied
    return graph


def _fresh_with_record_limit(
    work, *, record_limit, train=True, metadata_limit=128 * 1024, live_limit=16
):
    # Root-approved copy of the accepted factory. ONLY the original ledger
    # approved original record/metadata/live birth arguments differ; no runtime
    # ceiling changes. The zero-work path alone skips the original first update.
    assert work.graphs + 1 <= 24
    work.graphs += 1
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    runtime: Any = ActorShadowRuntime(
        NumericFakeLearner(work),
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
    )
    gate = ServingPriorityGate(SharingLimits(1, 2, 1), resource_available=lambda: True)
    owner = ManagedExperienceOwner(
        ResourceSharedRuntime(runtime, gate), limits=LifecycleLimits(4, 4)
    )
    life: Any = ManagedDataLifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096, 10, PayloadOwnershipLimits(16, 64), owned_payload_copies=PayloadCopyLimits(4096)
        ),
        measure_payload_bytes=_measure,
        native_footprint=_footprint,
        native_erase=_erase,
        measure_auxiliary_bytes=_auxiliary,
        measure_checkpoint_bytes=_outside_copy,
        native_growth_bytes=_growth,
        prepare_model_bytes=_outside_copy,
        prediction_cache_bytes=_outside_copy,
    )
    ledger: Any = ManagedReplayOrigins(
        owner,
        ReplayOriginPorts(_model_reference, _retained, _copy_bytes, _payloads, _fingerprint),
        ReplayOriginLimits(live_limit, record_limit, 8, metadata_limit, 120, 4096),
        ReplayWriteLimits(4, 2, 64),
        retain_inbox_origins=True,
    )
    graph = Graph(work, owner, life, runtime, ledger, clock, gate, ledger._history())
    _queue(graph)
    if not train:
        # Separately approved zero-update original ingress boundary. No receipt
        # or observed-work counters are manufactured to populate this history.
        assert runtime._budget.updates_completed == runtime._candidate.calls == 0
        assert life._admitted_bytes == life._copy_budget._charged == 24
        assert not runtime._candidate.model.rows and runtime._inbox._applied == {}
        assert graph.history is ledger._inbox_history_birth()
        assert graph.history._records == graph.history._sealed == {}
        assert ledger.accounting().records_created == ledger.accounting().invocations_started == 0
        return graph
    receipt = _train(graph)
    assert (
        receipt.update_number == runtime._budget.updates_completed == runtime._candidate.calls == 1
    )
    assert life._admitted_bytes == 24 and life._copy_budget._charged == 48
    assert len(runtime._candidate.model.rows) == 1
    assert graph.history is ledger._inbox_history_birth()
    records = ManagedInboxOrigins.verify(graph.history, ledger, owner, runtime, 3)
    assert len(records) == 1 and records[0].verified is True
    assert records[0].receipt is not None
    assert records[0].receipt() is receipt
    assert records[0].source() is runtime._inbox._experiences[FIRST]
    assert records[0].label() is runtime._inbox._labels[FIRST]
    assert ledger.accounting().records_created == 2
    assert ledger._admission.limits.max_records_created == record_limit
    assert ledger._admission.limits.max_metadata_bytes == metadata_limit
    assert ledger._admission.limits.max_live_records == live_limit
    return graph


def _observe(graph, now=None):
    now = graph.clock._time if now is None else now
    with _original_gates(graph):
        return qualification.observe_expiry_arrivals(
            graph.history,
            graph.ledger,
            graph.owner,
            graph.runtime,
            now,
        )


def _unchanged(graph, before, now=None):
    now = graph.clock._time if now is None else now
    with _original_gates(graph):
        return qualification.require_expiry_arrivals_unchanged(
            graph.history,
            graph.ledger,
            graph.owner,
            graph.runtime,
            now,
            before,
        )


def _prepare(graph, before, prepared, *, offset=0):
    with _original_gates(graph):
        return qualification.prepare_expiry_untrained_erasure(
            graph.history,
            graph.ledger,
            graph.owner,
            graph.runtime,
            graph.clock._time,
            before,
            prepared,
            offset=offset,
        )


def _refuse(graph, *, before=None, now=None):
    conserved = _conservation(graph)
    with pytest.raises(ValueError):
        _observe(graph, now) if before is None else _unchanged(graph, before, now)
    assert _conservation(graph) == conserved


def _assert_before(graph, before, *, source=True, label=True):
    assert type(before) is tuple and len(before) == 4
    assert before[0] == ManagedInboxOrigins.state_stamp(graph.history)
    arrivals = before[1]
    assert type(arrivals) is tuple and len(arrivals) == 1
    arrival = arrivals[0]
    assert type(arrival) is _Arrival and arrival.key == SECOND
    assert {field.name for field in fields(arrival)} == {
        "key",
        "source",
        "label",
        "declaration",
        "stamp",
        "tick",
        "work",
    }
    assert not hasattr(arrival, "__dict__") and arrival.work == 1
    assert arrival.tick == graph.clock._time
    assert type(arrival.declaration) is ReferenceType
    assert arrival.declaration() is graph.owner._catalog[SECOND]
    if source:
        assert type(arrival.source) is ReferenceType
        assert arrival.source() is graph.runtime._inbox._experiences[SECOND]
    else:
        assert arrival.source is None
    if label:
        assert type(arrival.label) is ReferenceType
        assert arrival.label() is graph.runtime._inbox._labels[SECOND]
    else:
        assert arrival.label is None
    assert type(before[2]) is tuple and type(before[3]) is tuple
    assert SECOND not in graph.runtime._inbox._applied
    return arrival


def _positive(work, *, source, label):
    graph = _ready(work, source=source, label=label)
    declared_at = graph.owner._declaration_ticks[SECOND]
    retention = graph.life._policy.max_retention_ticks
    assert declared_at == 3 and retention == 10
    expires_at = declared_at + retention
    # TTL starts at the original declaration, after FIRST's genuine training.
    graph.clock.advance_to(expires_at - 1)
    graph.owner._require_live(SECOND)
    for now in (expires_at, 122):
        graph.clock.advance_to(now)
        with pytest.raises(ValueError, match="expired"):
            graph.owner._require_live(SECOND)
        conserved = _conservation(graph)
        before = _observe(graph)
        arrival = _assert_before(graph, before, source=source, label=label)
        assert _unchanged(graph, before) is before[1]
        assert _unchanged(graph, before) is before[1]
        assert _conservation(graph) == conserved
        if now == 122 and source:
            with pytest.raises(ValueError, match="age"):
                _require_arrival(graph.history, graph.owner, graph.runtime, arrival)
    if not source:
        # The original label arrives at 3, so age 121 starts at 124, not 122.
        graph.clock.advance_to(124)
        conserved = _conservation(graph)
        before = _observe(graph)
        arrival = _assert_before(graph, before, source=False, label=True)
        assert _unchanged(graph, before) is before[1]
        assert _conservation(graph) == conserved
        with pytest.raises(ValueError, match="age"):
            _require_arrival(graph.history, graph.owner, graph.runtime, arrival)
    assert graph.runtime._budget.updates_completed == graph.ledger._last_work == 1
    assert graph.history._untrained_records == graph.history._untrained_sealed == {}


def test_should_qualify_original_source_only_after_ttl_and_source_age_without_live_grant(
    arrival_work,
):
    _positive(arrival_work, source=True, label=False)


def test_should_qualify_original_label_only_after_ttl_and_source_age_without_fake_receipt(
    arrival_work,
):
    _positive(arrival_work, source=False, label=True)


def test_should_qualify_original_pair_after_ttl_and_source_age_without_fake_work(arrival_work):
    _positive(arrival_work, source=True, label=True)


def test_should_avoid_opaque_callbacks_during_observe_and_repeat(arrival_work, monkeypatch):
    graph = _ready(arrival_work)
    graph.clock.advance_to(10)
    calls = []

    def poisoned(*arguments, **keywords):
        calls.append("opaque")
        raise AssertionError("opaque callback cannot grant expired arrival proof")

    with monkeypatch.context() as patch:
        for holder, name in (
            (graph.owner, "_require_live"),
            (graph.owner, "_eligible"),
            (graph.life, "_require_access"),
            (graph.life, "_require_live"),
            (graph.life, "_measure"),
            (graph.life, "_footprint"),
            (graph.life, "_erase"),
            (graph.runtime._budget, "clock"),
            (graph.gate, "_resource"),
            (graph.clock, "now"),
            (graph.ledger, "_history"),
            (graph.ledger, "_require_bindings"),
            (graph.ledger, "_live_records"),
        ):
            patch.setattr(holder, name, poisoned)
        conserved = _conservation(graph)
        before = _observe(graph)
        assert _unchanged(graph, before) is before[1]
        assert _conservation(graph) == conserved and calls == []


def test_should_refuse_late_equal_source_label_declaration_or_buffer_replacement(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    arrival = before[1][0]
    for mapping in (
        graph.runtime._inbox._experiences,
        graph.runtime._inbox._labels,
        graph.owner._catalog,
    ):
        original = mapping[SECOND]
        mapping[SECOND] = replace(original)
        try:
            _refuse(graph, before=before)
        finally:
            mapping[SECOND] = original
    assert arrival.source() is graph.runtime._inbox._experiences[SECOND]
    for record, name in (
        (graph.runtime._inbox._experiences[SECOND], "features"),
        (graph.runtime._inbox._labels[SECOND], "targets"),
    ):
        original = getattr(record, name)
        object.__setattr__(record, name, arrival_work.buffer_copy(original))
        try:
            _refuse(graph, before=before)
        finally:
            object.__setattr__(record, name, original)
    assert _unchanged(graph, before) is before[1]


def test_should_refuse_late_original_numeric_source_or_target_content_change(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    for array in (
        graph.runtime._inbox._experiences[SECOND].features,
        graph.runtime._inbox._labels[SECOND].targets,
    ):
        original = float(array[0, 0])
        array[0, 0] = original + 1.0
        try:
            _refuse(graph, before=before)
        finally:
            array[0, 0] = original
    assert _unchanged(graph, before) is before[1]


def test_should_refuse_rewritten_arrival_fields_under_immutable_original_own_seal(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    arrival = before[1][0]
    changes = (
        ("key", FIRST),
        ("tick", 4),
        ("work", 2),
        ("stamp", "0" * 64),
        ("source", None),
        ("label", None),
        ("declaration", ref(graph.owner._catalog[FIRST])),
    )
    for name, replacement in changes:
        original = getattr(arrival, name)
        object.__setattr__(arrival, name, replacement)
        try:
            _refuse(graph, before=before)
        finally:
            object.__setattr__(arrival, name, original)
    assert _unchanged(graph, before) is before[1]


def test_should_refuse_simultaneous_original_contents_and_arrival_stamp_rewrite(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    arrival = before[1][0]
    source = graph.runtime._inbox._experiences[SECOND]
    original_stamp = arrival.stamp
    original_value = float(source.features[0, 0])
    source.features[0, 0] = original_value + 1.0
    changed_stamp = _arrival_stamp(
        source,
        graph.runtime._inbox._labels[SECOND],
        graph.owner._catalog[SECOND],
        graph.history._content_limits,
    )
    assert changed_stamp != original_stamp
    object.__setattr__(arrival, "stamp", changed_stamp)
    try:
        _refuse(graph, before=before)
    finally:
        source.features[0, 0] = original_value
        object.__setattr__(arrival, "stamp", original_stamp)
    assert _unchanged(graph, before) is before[1]


def test_should_refuse_cloned_arrival_or_corrupted_observation_tuple(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    cloned = replace(before[1][0])
    assert cloned is not before[1][0]
    variants = (
        (before[0], (cloned,), before[2], before[3]),
        (before[0], before[1], (), before[3]),
        (before[0], before[1], before[2], ()),
        ((), before[1], before[2], before[3]),
        list(before),
    )
    for changed in variants:
        _refuse(graph, before=changed)
    assert _unchanged(graph, before) is before[1]


def test_should_refuse_changed_static_consent_permissions_roles_or_versions(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    source, label, declaration = (
        graph.runtime._inbox._experiences[SECOND],
        graph.runtime._inbox._labels[SECOND],
        graph.owner._catalog[SECOND],
    )
    changes = (
        (declaration.consent, "training", False),
        (declaration.consent, "replay", False),
        (source.permissions, "training", False),
        (source.permissions, "replay", False),
        (source, "role", "inner_guard"),
        (label, "role", "inner_guard"),
        (source, "model_version", "actor-foreign"),
        (label, "model_version", "actor-foreign"),
    )
    for record, name, replacement in changes:
        original = getattr(record, name)
        object.__setattr__(record, name, replacement)
        try:
            _refuse(graph, before=before)
        finally:
            object.__setattr__(record, name, original)
    graph.owner._revoked_keys.add(SECOND)
    try:
        _refuse(graph, before=before)
    finally:
        graph.owner._revoked_keys.remove(SECOND)
    graph.owner._opted_out.add("person-s2")
    try:
        _refuse(graph, before=before)
    finally:
        graph.owner._opted_out.remove("person-s2")


def test_should_refuse_current_time_work_or_active_state_changes_after_observation(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    for now in (True, -1, 2**63, 2, 4):
        _refuse(graph, before=before, now=now)
    for work in (0, 2):
        graph.runtime._budget.updates_completed = work
        try:
            _refuse(graph, before=before)
        finally:
            graph.runtime._budget.updates_completed = 1
    for holder, name in (
        (graph.ledger, "_busy"),
        (graph.ledger, "_poisoned"),
        (graph.runtime, "_stopped"),
        (graph.life, "_failed"),
    ):
        setattr(holder, name, True)
        try:
            _refuse(graph, before=before)
        finally:
            setattr(holder, name, False)
    assert _unchanged(graph, before) is before[1]


def test_should_refuse_missing_or_foreign_original_history_birth_binding(arrival_work):
    graph = _ready(arrival_work)
    before = _observe(graph)
    for name in ("_inbox_origins", "_original_inbox_origins", "_inbox_history_birth"):
        original = getattr(graph.ledger, name)
        setattr(graph.ledger, name, None)
        try:
            _refuse(graph, before=before)
        finally:
            setattr(graph.ledger, name, original)
    original = graph.ledger._owner
    graph.ledger._owner = ref(graph.life)
    try:
        _refuse(graph, before=before)
    finally:
        graph.ledger._owner = original


def test_should_refuse_provisional_original_trained_history_before_arrival_proof(arrival_work):
    graph = _ready(arrival_work)
    key, original = next(iter(graph.history._records.items()))
    graph.history._records[key] = graph.history._sealed[key] = replace(original, verified=False)
    try:
        _refuse(graph)
    finally:
        graph.history._records[key] = graph.history._sealed[key] = original


def test_should_refuse_genuine_untracked_receipt_without_arrival_or_history_fabrication(
    arrival_work,
):
    graph = _ready(arrival_work)
    receipt = _train(graph, observed=False, key=SECOND)
    assert receipt is graph.runtime._inbox._applied[SECOND] and receipt.update_number == 2
    assert len(graph.history._records) == 1
    _refuse(graph)


def test_should_reject_unsupported_schema_and_key_callbacks_before_traversal(arrival_work):
    graph = _ready(arrival_work)
    calls = []

    class Unsupported:
        def __getattribute__(self, name):
            calls.append("attribute")
            raise AssertionError("unsupported source callback ran")

        def __hash__(self):
            calls.append("hash")
            return 701

    mapping = graph.runtime._inbox._experiences
    original = mapping[SECOND]
    mapping[SECOND] = Unsupported()
    calls.clear()
    try:
        _refuse(graph)
        assert calls == []
    finally:
        mapping[SECOND] = original
    foreign_key = Unsupported()
    mapping[foreign_key] = original
    calls.clear()
    try:
        _refuse(graph)
        assert calls == []
    finally:
        del mapping[foreign_key]


def test_should_bound_trained_plus_untrained_payload_before_any_finite_probe(
    arrival_work, monkeypatch
):
    graph = _ready(arrival_work, payload_limit=24)
    assert graph.history._content_limits.max_array_bytes == 24
    assert _record(graph).data.payload_bytes == 24
    assert (
        graph.runtime._inbox._experiences[SECOND].features.nbytes
        + graph.runtime._inbox._labels[SECOND].targets.nbytes
        == 24
    )
    calls = []

    def cannot_probe(value):
        calls.append("finite")
        raise AssertionError("aggregate bound must precede payload traversal")

    with monkeypatch.context() as patch:
        patch.setattr(np, "isfinite", cannot_probe)
        _refuse(graph)
        assert calls == []


def _prepared_boundary(graph):
    graph.clock.advance_to(122)
    before = _observe(graph)
    inbox = graph.runtime._inbox
    keys = tuple(sorted(inbox._experiences.keys() | inbox._labels.keys()))
    prepared = inbox._prepare_erased_history(keys, 122, "expired")
    # This is ONLY the explicitly simulated original cleanup revocation boundary.
    # No lifecycle cleanup, payload pop, native erase or history commit occurs.
    graph.owner._revoked_keys.update(keys)
    return before, prepared


def _allocation_spy(graph, before, work, *, mutate=None):
    original = qualification.prepare_untrained_inbox_origin
    baseline = graph.ledger._admission._progress.records_created
    count = len(before[1])
    calls = []

    def allocate(data, tombstone, limits):
        # Test-only library hook: ALL original persistent record reservations
        # must precede the first actual weak witness allocation, not just its own.
        assert graph.ledger._admission._progress.records_created == baseline + count
        assert graph.ledger._admission.limits is graph.history._limits()
        record = original(data, tombstone, limits)
        work.persistent_witnesses += 1
        calls.append(record)
        if mutate is not None:
            mutate()
        return record

    return allocate, calls


def _assert_preparation_does_not_publish(graph, conserved):
    # Admission progress/minima are permitted to increase; every original raw,
    # map/storage/work/time authority and explicit allocation count stays put.
    assert _conservation(graph)[5:] == conserved[5:]
    assert graph.history._untrained_records == graph.history._untrained_sealed == {}
    assert set(graph.runtime._inbox._applied) == {FIRST}
    assert graph.runtime._budget.updates_completed == 1
    assert not graph.runtime._inbox._erased


def test_should_prepay_all_three_arrival_forms_before_persistent_weak_allocation(
    arrival_work, monkeypatch
):
    graph = _ready(arrival_work, label=False)
    _queue(graph, "s3", source=False)
    _queue(graph, "s4")
    before, prepared = _prepared_boundary(graph)
    assert len(before[1]) == 3 and set(prepared) == {FIRST, SECOND, ("e1", "s3"), ("e1", "s4")}
    conserved = _conservation(graph)
    admission = graph.ledger._admission
    progress = admission._progress
    allocation, allocated = _allocation_spy(graph, before, arrival_work)
    with monkeypatch.context() as patch:
        patch.setattr(qualification, "prepare_untrained_inbox_origin", allocation)
        records, sealed = _prepare(graph, before, prepared)
    assert type(records) is type(sealed) is dict and records is not sealed
    assert (
        records is not graph.history._untrained_records
        and sealed is not graph.history._untrained_sealed
    )
    assert len(allocated) == len(records) == len(sealed) == 3
    assert set(records) == set(sealed) == {SECOND, ("e1", "s3"), ("e1", "s4")}
    expected_metadata = 0
    for key, record in records.items():
        assert type(record) is UntrainedInboxOrigin and sealed[key] is record
        assert any(record is item for item in allocated)
        assert {field.name for field in fields(record)} == {"data", "tombstone", "tombstone_stamp"}
        assert type(record.tombstone) is ReferenceType and record.tombstone() is prepared[key]
        assert record.data.key == key and record.data.completed_updates == 1
        assert record.data.erased_at == 122 and record.data.reason == "expired"
        assert record.data.learner_version == "candidate-0"
        assert record.data.subject_id == "person-" + key[1]
        assert record.data.source_id == "local"
        assert all(
            not hasattr(record, name)
            for name in ("source", "label", "receipt", "features", "targets")
        )
        assert type(untrained_inbox_origin_stamp(record, graph.history._content_limits)) is tuple
        expected_metadata += len(untrained_inbox_origin_metadata(record.data, 128 * 1024)) + 1024
    assert (
        records[SECOND].data.observed_at,
        records[SECOND].data.event_id,
        records[SECOND].data.arrived_at,
    ) == (1, None, None)
    assert (
        records[("e1", "s3")].data.observed_at,
        records[("e1", "s3")].data.event_id,
        records[("e1", "s3")].data.arrived_at,
    ) == (None, "label-s3", 3)
    assert (
        records[("e1", "s4")].data.observed_at,
        records[("e1", "s4")].data.event_id,
        records[("e1", "s4")].data.arrived_at,
    ) == (1, "label-s4", 3)
    assert graph.ledger._admission is admission
    assert admission._progress.records_created == progress.records_created + 3
    assert admission._progress.invocations_started == progress.invocations_started
    assert (
        admission._progress.metadata_bytes_charged
        == progress.metadata_bytes_charged + expected_metadata
    )
    _assert_preparation_does_not_publish(graph, conserved)


def test_should_refuse_bad_preparation_boundary_and_late_paid_original_mutation(
    arrival_work, monkeypatch
):
    graph = _ready(arrival_work)
    graph.clock.advance_to(122)
    before = _observe(graph)
    prepared = graph.runtime._inbox._prepare_erased_history((FIRST, SECOND), 122, "expired")
    conserved = _conservation(graph)
    with pytest.raises(ValueError):
        _prepare(graph, before, prepared)
    assert _conservation(graph) == conserved
    graph.owner._revoked_keys.add(SECOND)
    conserved = _conservation(graph)
    with pytest.raises(ValueError):
        _prepare(graph, before, prepared)
    assert _conservation(graph) == conserved
    graph.owner._revoked_keys.add(FIRST)
    for malformed in (
        {SECOND: prepared[SECOND]},
        {FIRST: prepared[FIRST]},
        list(prepared.values()),
    ):
        conserved = _conservation(graph)
        with pytest.raises(ValueError):
            _prepare(graph, before, malformed)
        assert _conservation(graph) == conserved
    tombstone = prepared[SECOND]
    object.__setattr__(tombstone, "reason", "deleted")
    try:
        conserved = _conservation(graph)
        with pytest.raises(ValueError):
            _prepare(graph, before, prepared)
        assert _conservation(graph) == conserved
    finally:
        object.__setattr__(tombstone, "reason", "expired")
    for invalid in (True, -1, 2**63, 17):
        conserved = _conservation(graph)
        with pytest.raises(ValueError):
            _prepare(graph, before, prepared, offset=invalid)
        assert _conservation(graph) == conserved
    source = graph.runtime._inbox._experiences[SECOND]

    def mutate_original():
        object.__setattr__(source, "observed_at", 2)

    conserved = _conservation(graph)
    original_progress = graph.ledger._admission._progress
    allocation, allocated = _allocation_spy(graph, before, arrival_work, mutate=mutate_original)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(qualification, "prepare_untrained_inbox_origin", allocation)
            with pytest.raises(ValueError):
                _prepare(graph, before, prepared)
        assert source.observed_at == 2 and len(allocated) == 1
    finally:
        object.__setattr__(source, "observed_at", 1)
    assert (
        graph.ledger._admission._progress.records_created == original_progress.records_created + 1
    )
    expected_metadata = len(untrained_inbox_origin_metadata(allocated[0].data, 128 * 1024)) + 1024
    assert (
        graph.ledger._admission._progress.metadata_bytes_charged
        == original_progress.metadata_bytes_charged + expected_metadata
    )
    _assert_preparation_does_not_publish(graph, conserved)


def test_should_refuse_first_reserve_under_original_birth_record_ceiling_without_witness(
    arrival_work, monkeypatch
):
    graph = _fresh_with_record_limit(arrival_work, record_limit=2)
    _queue(graph, "s2", label=False)
    before, prepared = _prepared_boundary(graph)
    original_limit = graph.ledger._admission.limits
    conserved = _conservation(graph)
    allocation, allocated = _allocation_spy(graph, before, arrival_work)
    with monkeypatch.context() as patch:
        patch.setattr(qualification, "prepare_untrained_inbox_origin", allocation)
        with pytest.raises(ValueError, match="limit"):
            _prepare(graph, before, prepared)
    assert allocated == [] and _conservation(graph) == conserved
    assert (
        graph.ledger._admission.limits is original_limit and original_limit.max_records_created == 2
    )
    _assert_preparation_does_not_publish(graph, conserved)


def test_should_keep_partial_charges_without_witness_or_maps_at_original_birth_exhaustion(
    arrival_work, monkeypatch
):
    graph = _fresh_with_record_limit(arrival_work, record_limit=3)
    _queue(graph, "s2", label=False)
    _queue(graph, "s3", label=False)
    before, prepared = _prepared_boundary(graph)
    assert len(before[1]) == 2
    admission, original_limit = graph.ledger._admission, graph.ledger._admission.limits
    conserved, original_progress = _conservation(graph), admission._progress
    allocation, allocated = _allocation_spy(graph, before, arrival_work)
    with monkeypatch.context() as patch:
        patch.setattr(qualification, "prepare_untrained_inbox_origin", allocation)
        with pytest.raises(ValueError, match="limit"):
            _prepare(graph, before, prepared)
        assert allocated == []
        assert admission._progress.records_created == original_progress.records_created + 1 == 3
        spent = admission._progress
        with pytest.raises(ValueError, match="limit"):
            _prepare(graph, before, prepared)
        assert admission._progress is spent and allocated == []
    assert graph.ledger._admission is admission and admission.limits is original_limit
    assert admission._progress.invocations_started == original_progress.invocations_started
    assert admission._progress.metadata_bytes_charged > original_progress.metadata_bytes_charged
    assert admission._minimum == (
        spent.records_created,
        spent.invocations_started,
        spent.metadata_bytes_charged,
    )
    _assert_preparation_does_not_publish(graph, conserved)


def test_should_seal_zero_update_original_versions_and_inbox_map_identities(arrival_work):
    graph = _fresh_with_record_limit(arrival_work, record_limit=64, train=False)
    for now in (10, 122):
        graph.clock.advance_to(now)
        with pytest.raises(ValueError, match="expired"):
            graph.owner._require_live(FIRST)
        conserved = _conservation(graph)
        before = _observe(graph)
        assert len(before[1]) == 1 and before[1][0].key == FIRST and before[1][0].work == 0
        assert _unchanged(graph, before) is before[1]
        assert _conservation(graph) == conserved
    assert not graph.runtime._inbox._applied and graph.runtime._budget.updates_completed == 0
    before = _observe(graph)
    ledger_version, runtime_version = graph.ledger._version, graph.runtime._candidate_version
    graph.ledger._version = graph.runtime._candidate_version = "candidate-foreign"
    try:
        _refuse(graph, before=before)
    finally:
        graph.ledger._version, graph.runtime._candidate_version = ledger_version, runtime_version
    for name in ("_experiences", "_labels", "_applied", "_erased"):
        original = getattr(graph.runtime._inbox, name)
        setattr(graph.runtime._inbox, name, original.copy())
        try:
            _refuse(graph, before=before)
        finally:
            setattr(graph.runtime._inbox, name, original)
    assert _unchanged(graph, before) is before[1]
    assert not graph.runtime._inbox._applied and graph.runtime._budget.updates_completed == 0


def test_should_refuse_persistent_preparation_at_original_birth_live_record_ceiling(
    arrival_work, monkeypatch
):
    graph = _fresh_with_record_limit(arrival_work, record_limit=64, live_limit=2)
    _queue(graph, "s2", label=False)
    before, prepared = _prepared_boundary(graph)
    assert len(graph.ledger._rows) + len(graph.history._records) == 2
    limit = graph.ledger._admission.limits
    conserved = _conservation(graph)
    allocation, allocated = _allocation_spy(graph, before, arrival_work)
    with monkeypatch.context() as patch:
        patch.setattr(qualification, "prepare_untrained_inbox_origin", allocation)
        with pytest.raises(ValueError, match="live record limit"):
            _prepare(graph, before, prepared)
    assert allocated == [] and _conservation(graph) == conserved
    assert graph.ledger._admission.limits is limit and limit.max_live_records == 2
    _assert_preparation_does_not_publish(graph, conserved)


def test_should_exhaust_original_birth_metadata_with_bounded_paid_unpublished_preparations(
    arrival_work, monkeypatch
):
    graph = _fresh_with_record_limit(arrival_work, record_limit=64, metadata_limit=8192)
    _queue(graph, "s2", label=False)
    before, prepared = _prepared_boundary(graph)
    admission, limit = graph.ledger._admission, graph.ledger._admission.limits
    assert limit.max_metadata_bytes == 8192 and limit.max_live_records == 16
    original = admission._progress
    conserved = _conservation(graph)
    failed = False
    for attempt in range(8):
        assert arrival_work.metadata_preparation_attempts + 1 <= 8
        arrival_work.metadata_preparation_attempts += 1
        progress = admission._progress
        allocation, allocated = _allocation_spy(graph, before, arrival_work)
        with monkeypatch.context() as patch:
            patch.setattr(qualification, "prepare_untrained_inbox_origin", allocation)
            try:
                records, sealed = _prepare(graph, before, prepared)
            except ValueError as error:
                assert "metadata" in str(error)
                assert allocated == [] and admission._progress is progress
                failed = True
                break
        assert len(records) == len(sealed) == len(allocated) == 1
        assert records[SECOND] is sealed[SECOND] is allocated[0]
        charge = len(untrained_inbox_origin_metadata(records[SECOND].data, 8192)) + 1024
        assert admission._progress.records_created == progress.records_created + 1
        assert (
            admission._progress.metadata_bytes_charged == progress.metadata_bytes_charged + charge
        )
        arrival_work.metadata_preparation_successes += 1
        _assert_preparation_does_not_publish(graph, conserved)
    assert failed and arrival_work.metadata_preparation_successes > 0
    assert graph.ledger._admission is admission and admission.limits is limit
    assert original.metadata_bytes_charged < admission._progress.metadata_bytes_charged <= 8192
    assert admission._progress.invocations_started == original.invocations_started
    assert admission._minimum == (
        admission._progress.records_created,
        admission._progress.invocations_started,
        admission._progress.metadata_bytes_charged,
    )
    _assert_preparation_does_not_publish(graph, conserved)
