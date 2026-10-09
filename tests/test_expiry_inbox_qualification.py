"""Bounded app audit of originally committed numeric fake learner history.

Why this: TTL refusal remains an original raw-access rule, while erasure may
need the same admitted metadata proof. These fixtures run genuine app updates
and original replay-write observation; they establish no actual native result.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any
from weakref import ref

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.expiry_inbox_qualification import require_committed_erasure_ready
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_inbox_origins import ManagedInboxOrigins
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.replay_origin import ReplayOriginLimits, ReplayOriginPorts
from src.core.replay_write_origin import ReplayWriteLimits, begin_replay_write
from src.core.resource_sharing import SharingLimits


FIRST = ("e1", "s1")
SECOND = ("e1", "s2")


@dataclass
class FakeWork:
    graphs: int = 0
    learners: int = 0
    forks: int = 0
    updates: int = 0
    arrays: int = 0
    array_bytes: int = 0
    consumed_arrays: int = 0
    consumed_bytes: int = 0
    native_copy_arrays: int = 0
    native_copy_bytes: int = 0
    pending_consumed: tuple[int, int, int, int] | None = None

    def _reserve_arrays(self, count, size):
        assert self.arrays + count <= 128 and self.array_bytes + size <= 32 * 1024
        self.arrays += count
        self.array_bytes += size

    def array(self, values):
        expected = len(values) * len(values[0]) * np.dtype(float).itemsize
        self._reserve_arrays(1, expected)
        result = np.array(values, dtype=float)
        assert type(result) is np.ndarray and result.nbytes == expected
        return result

    def prepare_consumed(self, source, label):
        assert self.pending_consumed is None and self.updates + 1 <= 48
        features, targets = source.features, label.targets
        assert type(features) is type(targets) is np.ndarray
        self._reserve_arrays(2, features.nbytes + targets.nbytes)
        self.pending_consumed = (id(features), id(targets), features.nbytes, targets.nbytes)

    def accept_consumed(self, features, targets):
        assert self.pending_consumed is not None
        feature_id, target_id, feature_bytes, target_bytes = self.pending_consumed
        assert type(features) is type(targets) is np.ndarray
        assert id(features) != feature_id and id(targets) != target_id
        assert (features.nbytes, targets.nbytes) == (feature_bytes, target_bytes)
        self.pending_consumed = None
        self.consumed_arrays += 2
        self.consumed_bytes += feature_bytes + target_bytes
        assert self.updates + 1 <= 48
        self.updates += 1

    def native_copy(self, value):
        assert type(value) is np.ndarray
        self._reserve_arrays(1, value.nbytes)
        result = value.copy()
        assert type(result) is np.ndarray and result is not value
        assert result.nbytes == value.nbytes and not np.shares_memory(result, value)
        self.native_copy_arrays += 1
        self.native_copy_bytes += result.nbytes
        return result


@pytest.fixture(scope="module")
def fake_work(tmp_path_factory):
    work = FakeWork()
    try:
        yield work
    finally:
        # Preserve actual scalar work even if a case failed before its boundary.
        (tmp_path_factory.mktemp("expiry-inbox-fake-work") / "resources.json").write_text(
            json.dumps(asdict(work), indent=2),
            encoding="utf8",
        )
    assert work.graphs <= 24 and work.updates <= 48
    assert work.arrays <= 128 and work.array_bytes <= 32 * 1024
    assert work.graphs == 17 and work.updates == 19
    assert work.learners == 51 and work.forks == 34
    assert work.arrays == 118 and work.array_bytes == 1416
    assert work.consumed_arrays == work.native_copy_arrays == 38
    assert work.consumed_bytes == work.native_copy_bytes == 456
    assert work.pending_consumed is None


class NumericSnapshot:
    def __init__(self, features, targets, work):
        self.features = work.native_copy(features)
        self.targets = work.native_copy(targets)


class NumericModel:
    def __init__(self):
        self.rows = []


class NumericFakeLearner:
    def __init__(self, work):
        assert work.learners + 1 <= 72
        work.learners += 1
        self.work, self.model, self.calls = work, NumericModel(), 0

    def fork(self):
        assert not self.model.rows and self.calls == 0
        assert self.work.forks + 1 <= 48
        self.work.forks += 1
        return NumericFakeLearner(self.work)

    def train_batch(self, features, targets):
        self.work.accept_consumed(features, targets)
        self.calls += 1
        window = begin_replay_write(self.model, features, targets, self.model.rows, 1, "rows")
        if window is not None:
            window.before_copy(0, 1)
        snapshot = NumericSnapshot(features, targets, self.work)
        if window is not None:
            window.copied(snapshot)
        self.model.rows = (self.model.rows + [snapshot])[-2:]
        if window is not None:
            window.finish(self.model.rows)
        return TrainingDiagnostic("expiry_numeric_fake_v1", 1.0)

    def predict(self, features):
        raise AssertionError("prediction is outside this fake scope")

    def snapshot_state(self):
        raise AssertionError("raw snapshot implementation is outside this fake scope")

    def restore_state(self, state):
        raise AssertionError("restore is outside this fake scope")


def _model_reference(learner):
    return learner.model


def _retained(model, maximum):
    assert len(model.rows) <= maximum
    return tuple(model.rows)


def _copy_bytes(features, targets, start, count, maximum):
    assert type(features) is type(targets) is np.ndarray and (start, count) == (0, 1)
    size = features.nbytes + targets.nbytes
    assert size <= maximum
    return size


def _payloads(snapshot):
    return snapshot.features, snapshot.targets


def _fingerprint(snapshot, maximum):
    size = snapshot.features.nbytes + snapshot.targets.nbytes
    assert size <= maximum
    digest = sha256(snapshot.features.tobytes() + snapshot.targets.tobytes()).hexdigest()
    return size, digest


def _measure(value):
    assert type(value) is np.ndarray
    return value.nbytes


def _footprint(learner):
    rows = learner.model.rows
    return ReplayPayloadErasure(
        len(rows),
        len(rows),
        sum(row.features.nbytes + row.targets.nbytes for row in rows),
    )


def _erase(learner):
    raise AssertionError("cleanup is outside this fake qualification scope")


def _auxiliary(value):
    assert type(value) is dict and not value
    return 0


def _growth(learner, features, targets):
    return features.nbytes + targets.nbytes


def _outside_copy(*arguments):
    raise AssertionError("checkpoint/preparation/prediction copy is outside this scope")


@dataclass
class Graph:
    work: FakeWork
    owner: Any
    life: Any
    runtime: Any
    ledger: Any
    clock: Any
    gate: Any
    history: Any


def _declare(owner, sample):
    owner.declare(
        LifecycleDeclaration(
            ("e1", sample),
            DataProvenance("local", "person-" + sample, True, False),
            DataConsent(True, True),
            "replay",
        )
    )


def _queue(graph, sample="s1", *, source=True, label=True):
    _declare(graph.owner, sample)
    if source:
        graph.owner.record_experience(
            Experience(
                sample,
                "e1",
                1,
                "actor-0",
                graph.work.array([[1.0, 0.0]]),
                "train",
                ExperiencePermissions(True, True),
            )
        )
    if label:
        graph.owner.record_label(
            LabelArrival("label-" + sample, sample, "e1", 3, "actor-0", graph.work.array([[0.0]]))
        )
    graph.clock.advance_to(3)


def _train(graph, *, observed=True, key=FIRST):
    inbox = graph.runtime._inbox
    # Original app deepcopy creates consumed arrays BEFORE fake learner entry.
    # Reserve their finite count/bytes before calling that unchanged app path.
    graph.work.prepare_consumed(inbox._experiences[key], inbox._labels[key])
    poll = graph.ledger.train_ready() if observed else graph.owner.train_ready()
    assert graph.work.pending_consumed is None and len(poll.updates) == 1
    receipt = poll.updates[0]
    assert receipt is inbox._applied[key]
    assert receipt.update_number == graph.runtime._budget.updates_completed
    return receipt


def _fresh(work, *, history=True, payload_limit=4096):
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
        ReplayOriginLimits(16, 64, 8, 128 * 1024, 120, payload_limit),
        ReplayWriteLimits(4, 2, 64),
        retain_inbox_origins=history,
    )
    graph = Graph(work, owner, life, runtime, ledger, clock, gate, ledger._history())
    _queue(graph)
    receipt = _train(graph)
    assert (
        receipt.update_number == runtime._budget.updates_completed == runtime._candidate.calls == 1
    )
    assert life._admitted_bytes == 24 and life._copy_budget._charged == 48
    assert len(runtime._candidate.model.rows) == 1
    if history:
        assert graph.history is ledger._inbox_history_birth()
        records = ManagedInboxOrigins.verify(graph.history, ledger, owner, runtime, 3)
        assert len(records) == 1 and records[0].verified is True
        assert records[0].receipt is not None
        assert records[0].receipt() is receipt
        assert records[0].source() is runtime._inbox._experiences[FIRST]
        assert records[0].label() is runtime._inbox._labels[FIRST]
        assert ledger.accounting().records_created == 2
    else:
        assert graph.history is None and ledger.accounting().records_created == 1
    return graph


@contextmanager
def _original_gates(graph):
    with graph.ledger._exclusive(), graph.owner._operation(), graph.runtime._exclusive():
        yield


def _call(graph, now=None):
    now = graph.clock._time if now is None else now
    with _original_gates(graph):
        return require_committed_erasure_ready(
            graph.history, graph.ledger, graph.owner, graph.runtime, now
        )


def _conservation(graph):
    ledger, life, runtime, history = graph.ledger, graph.life, graph.runtime, graph.history
    admission = ledger._admission
    return (
        id(admission),
        id(admission.limits),
        id(admission._progress),
        admission._minimum,
        tuple(vars(admission._progress).values()),
        id(life._copy_budget),
        id(life._copy_budget._limits),
        life._copy_budget._charged,
        life._admitted_bytes,
        ledger._copy_slots,
        ledger._minimum_copy_slots,
        ledger._copy_charge_minimum,
        id(ledger._anchors),
        ledger._last_work,
        ledger._last_tick,
        life._last_tick,
        runtime._inbox._last_time,
        runtime._budget.updates_completed,
        runtime._candidate.calls,
        runtime._budget.started_at,
        runtime._budget.last_clock,
        ledger._last_budget_clock,
        graph.clock._time,
        id(runtime._budget),
        id(runtime._budget.budget),
        id(graph.owner._limits),
        graph.gate.snapshot().admitted_updates,
        life._registry._total,
        None
        if history is None
        else tuple(
            id(getattr(history, name))
            for name in (
                "_records",
                "_sealed",
                "_storage",
                "_erased_records",
                "_erased_sealed",
                "_erased_storage",
                "_untrained_records",
                "_untrained_sealed",
                "_untrained_storage",
            )
        ),
        None
        if history is None
        else tuple(
            tuple((id(key), id(value)) for key, value in getattr(history, name).items())
            for name in (
                "_records",
                "_sealed",
                "_erased_records",
                "_erased_sealed",
                "_untrained_records",
                "_untrained_sealed",
            )
        ),
        tuple(
            tuple((id(key), id(value)) for key, value in getattr(runtime._inbox, name).items())
            for name in ("_experiences", "_labels", "_applied", "_erased")
        ),
        graph.work.graphs,
        graph.work.updates,
        graph.work.arrays,
        graph.work.array_bytes,
    )


def _refuse(graph, *, now=None, match=None):
    before = _conservation(graph)
    with pytest.raises(ValueError, match=match):
        _call(graph, now)
    assert _conservation(graph) == before


def _record(graph):
    return next(iter(graph.history._records.values()))


def test_should_audit_same_committed_history_at_original_ttl_without_raw_access(fake_work):
    graph = _fresh(fake_work)
    original = ManagedInboxOrigins.state_stamp(graph.history)
    before = _conservation(graph)
    assert _call(graph) == (original, ())
    assert _conservation(graph) == before
    graph.clock.advance_to(10)
    with pytest.raises(ValueError, match="expired"):
        ManagedInboxOrigins.verify(graph.history, graph.ledger, graph.owner, graph.runtime, 10)
    with pytest.raises(ValueError, match="expired"):
        graph.runtime.actor.snapshot_state()
    before = _conservation(graph)
    assert _call(graph) == _call(graph) == (original, ())
    assert _conservation(graph) == before
    assert graph.runtime._inbox._experiences[FIRST].features is _record(graph).features()
    assert graph.runtime._inbox._applied[FIRST] is _record(graph).receipt()
    assert graph.runtime._inbox._erased == {} and not graph.owner._revoked_keys
    graph.clock.advance_to(122)
    with pytest.raises(ValueError, match="expired"):
        ManagedInboxOrigins.verify(graph.history, graph.ledger, graph.owner, graph.runtime, 122)
    with pytest.raises(ValueError, match="expired"):
        graph.ledger.origins()
    with pytest.raises(ValueError, match="expired"):
        graph.runtime.actor.snapshot_state()
    before = _conservation(graph)
    assert 122 - _record(graph).data.observed_at > graph.ledger._admission.limits.max_age_ticks
    assert _call(graph) == (original, ())
    assert _conservation(graph) == before


def test_should_refuse_history_disabled_at_original_birth(fake_work):
    graph = _fresh(fake_work, history=False)
    assert graph.ledger._inbox_history_birth is None
    _refuse(graph)
    assert graph.history is None and graph.ledger._inbox_origins is None


def test_should_refuse_equal_current_source_label_or_declaration_clones(fake_work):
    graph = _fresh(fake_work)
    for mapping in (
        graph.runtime._inbox._experiences,
        graph.runtime._inbox._labels,
        graph.owner._catalog,
    ):
        original = mapping[FIRST]
        cloned = replace(original)
        assert cloned is not original
        mapping[FIRST] = cloned
        try:
            _refuse(graph)
        finally:
            mapping[FIRST] = original
    assert _call(graph)[1] == ()


def test_should_refuse_original_numeric_content_mutation_without_new_allocation(fake_work):
    graph = _fresh(fake_work)
    for array in (
        graph.runtime._inbox._experiences[FIRST].features,
        graph.runtime._inbox._labels[FIRST].targets,
    ):
        original = float(array[0, 0])
        array[0, 0] = original + 1.0
        try:
            _refuse(graph)
        finally:
            array[0, 0] = original
    assert _call(graph)[1] == ()


def test_should_refuse_receipt_clone_or_changed_original_diagnostic(fake_work):
    graph = _fresh(fake_work)
    mapping = graph.runtime._inbox._applied
    original = mapping[FIRST]
    cloned = replace(original)
    assert cloned is not original and cloned == original
    mapping[FIRST] = cloned
    try:
        _refuse(graph)
    finally:
        mapping[FIRST] = original
    object.__setattr__(original.diagnostic, "value", 2.0)
    try:
        _refuse(graph)
    finally:
        object.__setattr__(original.diagnostic, "value", 1.0)
    assert _call(graph)[1] == ()


def test_should_preserve_original_consent_revocation_and_subject_refusal(fake_work):
    graph = _fresh(fake_work)
    consent = graph.owner._catalog[FIRST].consent
    for field in ("training", "replay"):
        object.__setattr__(consent, field, False)
        try:
            _refuse(graph)
        finally:
            object.__setattr__(consent, field, True)
    permissions = graph.runtime._inbox._experiences[FIRST].permissions
    for field in ("training", "replay"):
        object.__setattr__(permissions, field, False)
        try:
            _refuse(graph)
        finally:
            object.__setattr__(permissions, field, True)
    graph.owner._revoked_keys.add(FIRST)
    try:
        _refuse(graph)
    finally:
        graph.owner._revoked_keys.remove(FIRST)
    graph.owner._opted_out.add("person-s1")
    try:
        _refuse(graph)
    finally:
        graph.owner._opted_out.remove("person-s1")
    assert _call(graph)[1] == ()


def test_should_refuse_provisional_witness_even_with_matching_sealed_maps(fake_work):
    graph = _fresh(fake_work)
    key, original = next(iter(graph.history._records.items()))
    provisional = replace(original, verified=False)
    graph.history._records[key] = graph.history._sealed[key] = provisional
    try:
        _refuse(graph, match="provisional")
    finally:
        graph.history._records[key] = graph.history._sealed[key] = original
    assert _record(graph).verified is True


def test_should_refuse_real_untracked_second_receipt_without_retrospective_enrollment(fake_work):
    graph = _fresh(fake_work)
    original = _record(graph)
    _queue(graph, "s2")
    second_receipt = _train(graph, observed=False, key=SECOND)
    assert second_receipt.update_number == graph.runtime._budget.updates_completed == 2
    assert graph.runtime._inbox._applied[SECOND] is second_receipt
    assert len(graph.history._records) == 1 and _record(graph) is original
    _refuse(graph, match="complete original committed trained lineage")
    assert graph.ledger._admission._progress.invocations_started == 1


def test_should_explicitly_refuse_current_untrained_source_only_or_label_only(fake_work):
    graph = _fresh(fake_work)
    _queue(graph, "s2", label=False)
    _refuse(graph, match="current untrained arrivals are not yet supported")
    assert SECOND not in graph.runtime._inbox._applied
    source_only = graph.runtime._inbox._experiences.pop(SECOND)
    try:
        # Temporary test membership isolates the label-only refusal; this is
        # neither public cleanup nor a claim that original raw ownership expired.
        _queue(graph, "s3", source=False)
        assert set(graph.runtime._inbox._experiences) == {FIRST}
        assert set(graph.runtime._inbox._labels) == {FIRST, ("e1", "s3")}
        _refuse(graph, match="current untrained arrivals are not yet supported")
        assert ("e1", "s3") not in graph.runtime._inbox._applied
    finally:
        graph.runtime._inbox._experiences[SECOND] = source_only


def test_should_explicitly_refuse_current_untrained_pair(fake_work):
    graph = _fresh(fake_work)
    _queue(graph, "s2")
    _refuse(graph, match="current untrained arrivals are not yet supported")
    assert graph.runtime._budget.updates_completed == 1


def test_should_require_exact_current_original_time_and_monotone_minima(fake_work):
    graph = _fresh(fake_work)
    for now in (True, -1, 2**63, 2, 4):
        _refuse(graph, now=now)
    for holder, name in (
        (graph.ledger, "_last_tick"),
        (graph.runtime._inbox, "_last_time"),
        (graph.life, "_last_tick"),
    ):
        previous = getattr(holder, name)
        setattr(holder, name, 4)
        try:
            _refuse(graph)
        finally:
            setattr(holder, name, previous)
    assert _call(graph)[1] == ()


def test_should_refuse_active_pending_stopped_or_failed_original_work(fake_work):
    graph = _fresh(fake_work)
    flags = (
        (graph.ledger, "_busy"),
        (graph.ledger, "_poisoned"),
        (graph.runtime, "_stopped"),
        (graph.runtime, "_retired"),
        (graph.runtime._inbox, "_stopped"),
        (graph.life, "_failed"),
        (graph.life, "_retention_fault"),
    )
    for holder, name in flags:
        for invalid in (True, 1):
            setattr(holder, name, invalid)
            try:
                _refuse(graph)
            finally:
                setattr(holder, name, False)
    assert (
        graph.ledger._started is True
    )  # Original successful training retains this historical flag.
    graph.ledger._started = 1
    try:
        _refuse(graph)
    finally:
        graph.ledger._started = True
    graph.ledger._pending = next(iter(graph.ledger._rows.values()))
    try:
        _refuse(graph, match="pending")
    finally:
        graph.ledger._pending = None
    for name, reference in (
        ("_active_source", _record(graph).source),
        ("_active_label", _record(graph).label),
    ):
        setattr(graph.ledger, name, reference)
        try:
            _refuse(graph, match="pending")
        finally:
            setattr(graph.ledger, name, None)
    assert _call(graph)[1] == ()


def test_should_refuse_current_or_birth_history_pointer_replacement(fake_work):
    graph = _fresh(fake_work)
    ledger = graph.ledger
    for name in ("_inbox_origins", "_original_inbox_origins", "_inbox_history_birth"):
        previous = getattr(ledger, name)
        setattr(ledger, name, None)
        try:
            _refuse(graph)
        finally:
            setattr(ledger, name, previous)
    other_ref = ref(graph.history, lambda unused: None)
    assert other_ref is not ledger._inbox_history_birth and other_ref() is graph.history
    original = ledger._original_inbox_origins
    ledger._original_inbox_origins = other_ref
    try:
        _refuse(graph)
    finally:
        ledger._original_inbox_origins = original
    assert _call(graph)[1] == ()


def test_should_refuse_replacement_original_admission_limits_or_metadata_allowance(fake_work):
    graph = _fresh(fake_work)
    admission, history = graph.ledger._admission, graph.history
    original_limit = admission.limits
    admission.limits = replace(original_limit)
    try:
        _refuse(graph)
    finally:
        admission.limits = original_limit
    original_content = history._content_limits
    history._content_limits = replace(original_content)
    try:
        _refuse(graph)
    finally:
        history._content_limits = original_content
    object.__setattr__(original_limit, "max_invocations", original_limit.max_invocations + 1)
    try:
        _refuse(graph)
    finally:
        object.__setattr__(original_limit, "max_invocations", 8)
    assert _call(graph)[1] == ()


def test_should_refuse_changed_original_runtime_clock_budget_and_lineage_roots(fake_work):
    graph = _fresh(fake_work)
    changes = (
        (graph.runtime._inbox, "_clock", LogicalClock(3)),
        (graph.runtime, "_payload_lineage", object()),
        (graph.runtime, "_candidate_version", "candidate-foreign"),
        (graph.runtime, "_candidate", graph.runtime._actor._learner),
        (graph.runtime, "_actor", object()),
        (graph.owner, "_limits", replace(graph.owner._limits)),
        (graph.runtime._budget, "budget", replace(graph.runtime._budget.budget)),
        (graph.life, "_clock", LogicalClock(3)),
        (graph.ledger, "_owner", ref(graph.life)),
    )
    for holder, name, foreign in changes:
        previous = getattr(holder, name)
        setattr(holder, name, foreign)
        try:
            _refuse(graph)
        finally:
            setattr(holder, name, previous)
    old_work = graph.runtime._budget.updates_completed
    for unobserved_work in (0, 2):
        graph.runtime._budget.updates_completed = unobserved_work
        try:
            _refuse(graph)
        finally:
            graph.runtime._budget.updates_completed = old_work
    assert _call(graph)[1] == ()


def test_should_reject_exact_schema_changes_and_hostile_callbacks_before_comparison(fake_work):
    graph = _fresh(fake_work)
    calls = []

    class HostileMeta(type):
        def __eq__(cls, other):
            calls.append("equality")
            raise AssertionError("unsupported metaclass equality ran")

        def __hash__(cls):
            calls.append("hash")
            return 101

    class Foreign(metaclass=HostileMeta):
        def __hash__(self):
            calls.append("object hash")
            raise AssertionError("unsupported metadata key hashing ran")

        def __getattribute__(self, name):
            calls.append("attribute")
            raise AssertionError("unsupported object attribute ran")

    class ForeignSource(Experience):
        def __getattribute__(self, name):
            calls.append("source attribute")
            raise AssertionError("unsupported source attribute ran")

    mapping = graph.runtime._inbox._experiences
    original = mapping[FIRST]
    mapping[FIRST] = object.__new__(ForeignSource)
    calls.clear()
    try:
        _refuse(graph)
        assert calls == []
    finally:
        mapping[FIRST] = original
    mapping[Foreign] = original
    calls.clear()
    try:
        _refuse(graph)
        assert calls == []
    finally:
        del mapping[Foreign]
    data = _record(graph).data
    object.__setattr__(data, "unexpected", Foreign())
    calls.clear()
    try:
        _refuse(graph)
        assert calls == []
    finally:
        del vars(data)["unexpected"]
    original_key = data.key
    object.__setattr__(data, "key", Foreign())
    calls.clear()
    try:
        _refuse(graph)
        assert calls == []
    finally:
        object.__setattr__(data, "key", original_key)
    assert _call(graph)[1] == ()


def test_should_refuse_foreign_runtime_before_any_custom_attribute_callback():
    calls = []

    class ForeignRuntime:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("unsupported runtime attribute callback ran")

    value = ForeignRuntime()
    with pytest.raises(ValueError, match="exact original runtime"):
        require_committed_erasure_ready(None, None, None, value, 3)
    assert calls == []


def test_should_ignore_opaque_callbacks_and_bound_aggregate_arrays_before_payload_hash(
    fake_work, monkeypatch
):
    graph = _fresh(fake_work, payload_limit=24)
    stamp = ManagedInboxOrigins.state_stamp(graph.history)
    calls = []
    original_record = _record(graph)
    graph.runtime._candidate.model.rows.clear()  # Simulated fake canonical eviction only; no GC or cleanup.
    assert not graph.runtime._candidate.model.rows and _record(graph) is original_record

    def poisoned(*arguments, **keywords):
        calls.append("opaque")
        raise AssertionError("opaque callback is outside committed erasure qualification")

    # Replace callback surfaces only after genuine original training. A fixed
    # class audit must neither dispatch instance shadows nor inspect native rows.
    with monkeypatch.context() as patch:
        for holder, name in (
            (graph.owner, "_require_live"),
            (graph.owner, "_eligible"),
            (graph.life, "_measure"),
            (graph.life, "_footprint"),
            (graph.life, "_erase"),
            (graph.life, "_require_live"),
            (graph.life, "_require_access"),
            (graph.runtime._budget, "clock"),
            (graph.gate, "_resource"),
            (graph.clock, "now"),
            (graph.ledger, "_history"),
            (graph.ledger, "_live_records"),
            (graph.ledger, "_require_bindings"),
        ):
            patch.setattr(holder, name, poisoned)
        ports = graph.ledger._ports
        original_ports = tuple(vars(ports).items())
        try:
            for name, original in original_ports:
                object.__setattr__(ports, name, poisoned)
            before = _conservation(graph)
            assert _call(graph) == (stamp, ()) and calls == []
            assert _conservation(graph) == before
        finally:
            for name, original in original_ports:
                object.__setattr__(ports, name, original)
    _queue(graph, "s2")
    _train(graph, key=SECOND)
    assert len(graph.history._records) == graph.runtime._budget.updates_completed == 2
    assert sum(record.data.payload_bytes for record in graph.history._records.values()) == 48
    assert graph.history._content_limits.max_array_bytes == 24
    with monkeypatch.context() as patch:
        patch.setattr(np, "isfinite", poisoned)
        calls.clear()
        _refuse(graph, match="aggregate borrowed payload exceeds original bound")
        assert calls == []
