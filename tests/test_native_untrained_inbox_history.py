"""N13: original untrained erasure preserves paid, payload-free lineage.

Why this: an arrival can be erased without an applied receipt. Observation of
the genuine original commit must prove that absence without inventing work or
granting checkpoint-copy authority. This module exercises seven fresh graphs.
"""

from dataclasses import fields, replace
from weakref import ReferenceType, ref

import numpy as np
import pytest

from src.adapters import numpy_learners as learner_adapter
from src.app.managed_inbox_origins import ManagedInboxOrigins
from src.app.untrained_inbox_origins import (
    UntrainedInboxOrigin,
    untrained_inbox_origin_stamp,
)
from src.core.data_retention import DataCleanupReport
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.replay_origin import RECORD_OVERHEAD_BYTES
from src.core.untrained_inbox_origin import (
    UntrainedInboxOriginData,
    untrained_inbox_origin_metadata,
)
from test_managed_experience import declaration
from test_native_checkpoint_eviction import _queue_original_distinct_sample
from test_native_managed_replay_checkpoints import observe_native_work, setup


FIRST = ("e1", "s1")
SECOND = ("e1", "s2")
STORAGE_NAMES = ("_storage", "_erased_storage", "_untrained_storage")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_untrained_history_work(tmp_path_factory):
    limits = dict(
        models=7,
        learners=21,
        forks=14,
        wakes=4,
        steps=8,
        stores=4,
        predicts=4,
        array_copies=20,
        captures=1,
        preparations=0,
        native_restores=0,
        handoffs=0,
        graph_events=32,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=8,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _fresh(*, arrival="pair", history=True, records=64):
    result = setup(
        train_first=False,
        queue_first=arrival == "pair",
        records_created=records,
        replay_examples=1,
        retain_inbox_origins=history,
        live_records=32,
    )
    owner, _, runtime, ledger, controller, _ = result
    if arrival != "pair":
        owner.declare(declaration("s1", "person-1"))
        if arrival == "source":
            owner.record_experience(
                Experience(
                    "s1",
                    "e1",
                    1,
                    "actor-0",
                    np.array([[1.0, 0.0, 0.0]]),
                    "train",
                    ExperiencePermissions(True, True),
                )
            )
        else:
            assert arrival == "label"
            owner.record_label(
                LabelArrival("label-s1", "s1", "e1", 3, "actor-0", np.array([[0.0]]))
            )
        runtime._inbox._clock.advance_to(3)
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 0
    assert runtime._inbox._applied == {} and not runtime._candidate._model._replay_memory
    assert ledger.accounting().records_created == ledger.accounting().invocations_started == 0
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0
    return result


def _authorities(ledger, life, runtime, history, *, records=64):
    limits = ledger._admission.limits
    assert tuple(vars(limits).values()) == (32, records, 32, 128 * 1024, 120, 4096)
    assert life._copy_budget._limits.max_lifetime_owned_bytes == 4096
    assert runtime._budget.budget.max_training_updates == 2
    retention = runtime._candidate._model._replay_retention_budget
    assert (retention.max_examples, retention.max_bytes) == (1, 64)
    return dict(
        admission=ledger._admission,
        limits=limits,
        raw=life._copy_budget,
        raw_limits=life._copy_budget._limits,
        policy=life._policy,
        registry=life._registry,
        budget=runtime._budget,
        budget_policy=runtime._budget.budget,
        retention=retention,
        lineage=runtime._payload_lineage,
        anchors=ledger._anchors,
        history=history,
        storages=tuple(getattr(history, name) for name in STORAGE_NAMES)
        if history is not None
        else (),
        accounting=ledger.accounting(),
        raw_charge=life._copy_budget._charged,
        ingress=life._admitted_bytes,
        slots=ledger._copy_slots,
        minimum_slots=ledger._minimum_copy_slots,
        enrollments=life._registry._total,
        work=runtime._budget.updates_completed,
    )


def _assert_authorities(ledger, life, runtime, history, before, *, transitioned=False):
    assert ledger._admission is before["admission"]
    assert ledger._admission.limits is before["limits"]
    assert life._copy_budget is before["raw"] and life._copy_budget._limits is before["raw_limits"]
    assert life._policy is before["policy"]
    assert life._policy.owned_payload_copies is before["raw_limits"]
    assert life._registry is before["registry"]
    assert runtime._budget is before["budget"] and runtime._budget.budget is before["budget_policy"]
    assert runtime._candidate._model._replay_retention_budget is before["retention"]
    assert runtime._payload_lineage is before["lineage"] and ledger._anchors is before["anchors"]
    assert ledger._history() is history is before["history"]
    if history is not None:
        for name, previous in zip(STORAGE_NAMES, before["storages"]):
            current = getattr(history, name)
            assert type(current) is tuple and len(current) == 2
            assert all(type(mapping) is dict for mapping in current)
            assert current[0] is not current[1]
            if not transitioned:
                assert current is previous
            elif name == "_untrained_storage":
                assert current is not previous
                assert all(mapping is not old for mapping in current for old in previous)
        assert history._storage[0] is history._records and history._storage[1] is history._sealed
        assert history._erased_storage[0] is history._erased_records
        assert history._erased_storage[1] is history._erased_sealed
        assert history._untrained_storage[0] is history._untrained_records
        assert history._untrained_storage[1] is history._untrained_sealed
    assert life._registry._total == before["enrollments"]
    assert ledger._copy_slots >= before["slots"]
    assert ledger._minimum_copy_slots >= before["minimum_slots"]
    current, previous = ledger.accounting(), before["accounting"]
    assert current.records_created >= previous.records_created
    assert current.invocations_started >= previous.invocations_started
    assert current.metadata_bytes_charged >= previous.metadata_bytes_charged
    assert life._copy_budget._charged >= before["raw_charge"]
    assert life._admitted_bytes >= before["ingress"]
    assert runtime._budget.updates_completed >= before["work"]


def _weak_arrivals(runtime, key=FIRST):
    inbox = runtime._inbox
    references: list[ReferenceType] = []
    if key in inbox._experiences:
        references.extend((ref(inbox._experiences[key]), ref(inbox._experiences[key].features)))
    if key in inbox._labels:
        references.extend((ref(inbox._labels[key]), ref(inbox._labels[key].targets)))
    return tuple(references)


def _assert_actual_untrained_erasure(owner, runtime, controller, report, weak_arrivals):
    assert type(report) is DataCleanupReport
    assert report.reason == "deleted" and report.requested_keys == report.revoked_keys == (FIRST,)
    assert report.model_snapshots_erased == 0 and report.inboxes_cleared == 1
    assert report.checkpoints_invalidated == report.promotions_invalidated == 0
    assert owner._shared._runtime is runtime and not runtime._retired
    assert FIRST in owner._revoked_keys
    assert runtime._inbox._experiences == runtime._inbox._labels == runtime._inbox._applied == {}
    assert set(runtime._inbox._erased) == {FIRST}
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 0
    assert not runtime._candidate._model._replay_memory
    assert runtime._inbox._historical_completed_updates is None
    assert all(reference() is None for reference in weak_arrivals)
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0


def _assert_witness(owner, runtime, history, *, arrival):
    assert (
        history._records
        == history._sealed
        == history._erased_records
        == history._erased_sealed
        == {}
    )
    assert set(history._untrained_records) == set(history._untrained_sealed) == {FIRST}
    witness = history._untrained_records[FIRST]
    assert type(witness) is UntrainedInboxOrigin
    assert history._untrained_sealed[FIRST] is witness
    assert {field.name for field in fields(witness)} == {"data", "tombstone", "tombstone_stamp"}
    assert not hasattr(witness, "__dict__")
    assert type(witness.data) is UntrainedInboxOriginData
    assert {field.name for field in fields(witness.data)} == {
        "key",
        "actor_version",
        "learner_version",
        "subject_id",
        "source_id",
        "observed_at",
        "event_id",
        "arrived_at",
        "erased_at",
        "reason",
        "completed_updates",
    }
    assert all(
        not hasattr(witness, name)
        for name in ("source", "label", "declaration", "features", "targets", "receipt")
    )
    data = witness.data
    assert all(
        type(getattr(data, field.name)) in (tuple, str, int, type(None)) for field in fields(data)
    )
    provenance = owner._catalog[FIRST].provenance
    assert (data.key, data.actor_version, data.learner_version) == (FIRST, "actor-0", "candidate-0")
    assert (data.subject_id, data.source_id) == (provenance.subject_id, provenance.source_id)
    assert data.observed_at == (None if arrival == "label" else 1)
    assert data.event_id == (None if arrival == "source" else "label-s1")
    assert data.arrived_at == (None if arrival == "source" else 3)
    assert (data.erased_at, data.reason, data.completed_updates) == (3, "deleted", 0)
    assert type(witness.tombstone) is ReferenceType
    assert witness.tombstone() is runtime._inbox._erased[FIRST]
    assert type(untrained_inbox_origin_stamp(witness, history._content_limits)) is tuple
    assert ManagedInboxOrigins._untrained_state(history) == (witness,)
    return witness


def _verify(history, ledger, owner, runtime):
    return ManagedInboxOrigins.verify(history, ledger, owner, runtime, ledger._read_tick(runtime))


def _assert_tombstone_replacement_and_mutation_refuse(history, ledger, owner, runtime):
    stamp = ManagedInboxOrigins.state_stamp(history)
    tombstone = runtime._inbox._erased[FIRST]
    runtime._inbox._erased[FIRST] = replace(tombstone)
    try:
        with pytest.raises(ValueError):
            _verify(history, ledger, owner, runtime)
        with pytest.raises(ValueError):
            ledger.origins()
    finally:
        runtime._inbox._erased[FIRST] = tombstone
    assert ManagedInboxOrigins.state_stamp(history) == stamp
    _verify(history, ledger, owner, runtime)
    # A valid changed scalar tests the sealed content proof, not invalid syntax.
    object.__setattr__(tombstone, "reason", "expired")
    try:
        with pytest.raises(ValueError):
            _verify(history, ledger, owner, runtime)
        with pytest.raises(ValueError):
            ManagedInboxOrigins.state_stamp(history)
    finally:
        object.__setattr__(tombstone, "reason", "deleted")
    assert ManagedInboxOrigins.state_stamp(history) == stamp
    _verify(history, ledger, owner, runtime)


@pytest.mark.parametrize("arrival,raw_total", [("source", 88), ("label", 72), ("pair", 96)])
def test_should_observe_original_untrained_arrival_then_preserve_distinct_training(
    arrival, raw_total
):
    owner, life, runtime, ledger, controller, _ = _fresh(arrival=arrival)
    history = ledger._history()
    assert history is not None
    assert history._records == history._erased_records == history._untrained_records == {}
    weak_arrivals = _weak_arrivals(runtime)
    before = _authorities(ledger, life, runtime, history)
    report = ledger.delete((FIRST,))
    _assert_actual_untrained_erasure(owner, runtime, controller, report, weak_arrivals)
    _assert_authorities(ledger, life, runtime, history, before, transitioned=True)
    witness = _assert_witness(owner, runtime, history, arrival=arrival)
    after = ledger.accounting()
    expected_metadata = len(
        untrained_inbox_origin_metadata(witness.data, before["limits"].max_metadata_bytes)
    )
    assert after.records_created == before["accounting"].records_created + 1
    assert after.invocations_started == before["accounting"].invocations_started == 0
    assert after.metadata_bytes_charged == expected_metadata + RECORD_OVERHEAD_BYTES
    assert life._copy_budget._charged == before["raw_charge"] == raw_total - 64
    assert life._admitted_bytes == before["ingress"] == raw_total - 64
    assert ledger._copy_slots == ledger._minimum_copy_slots == 0
    assert _verify(history, ledger, owner, runtime) == () and ledger.origins() == ()
    assert ledger.accounting().live_records == 1
    erased = _authorities(ledger, life, runtime, history)
    tombstone = runtime._inbox._erased[FIRST]
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    assert life._admitted_bytes == erased["ingress"] + 32
    assert life._copy_budget._charged == erased["raw_charge"] + 32
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 1
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 1
    assert set(runtime._inbox._applied) == {SECOND}
    assert runtime._inbox._applied[SECOND] is poll.updates[0]
    assert set(runtime._inbox._erased) == {FIRST} and runtime._inbox._erased[FIRST] is tombstone
    assert history._untrained_records[FIRST] is history._untrained_sealed[FIRST] is witness
    assert witness.data.completed_updates == 0 and witness.tombstone() is tombstone
    assert FIRST in owner._revoked_keys and SECOND not in owner._revoked_keys
    assert life._copy_budget._charged == raw_total and life._admitted_bytes == raw_total - 32
    assert len(runtime._candidate._model._replay_memory) == 1
    origins = ledger.origins()
    assert len(origins) == 1 and origins[0].key == SECOND and origins[0].update_number == 1
    live = _verify(history, ledger, owner, runtime)
    assert len(live) == 1 and live[0].data.key == SECOND
    assert live[0].source() is runtime._inbox._experiences[SECOND]
    assert live[0].label() is runtime._inbox._labels[SECOND]
    assert live[0].receipt is not None and live[0].receipt() is poll.updates[0]
    assert ledger.accounting().live_records == 3
    assert all(reference() is None for reference in weak_arrivals)
    _assert_authorities(ledger, life, runtime, history, erased)
    stable = _authorities(ledger, life, runtime, history)
    _assert_tombstone_replacement_and_mutation_refuse(history, ledger, owner, runtime)
    _assert_authorities(ledger, life, runtime, history, stable)
    assert ledger.accounting() == stable["accounting"]
    assert life._copy_budget._charged == stable["raw_charge"]
    assert life._admitted_bytes == stable["ingress"]
    assert runtime._budget.updates_completed == stable["work"] == 1
    assert ledger._copy_slots == ledger._minimum_copy_slots == 0
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0


def test_should_refuse_retrospective_untrained_lineage_after_unobserved_original_delete():
    owner, life, runtime, ledger, controller, _ = _fresh()
    history = ledger._history()
    assert history is not None
    before = _authorities(ledger, life, runtime, history)
    weak_arrivals = _weak_arrivals(runtime)
    report = life.delete((FIRST,))
    _assert_actual_untrained_erasure(owner, runtime, controller, report, weak_arrivals)
    assert history._untrained_records == history._untrained_sealed == {}
    with pytest.raises(ValueError):
        ledger.origins()
    with pytest.raises(ValueError):
        ledger.delete((FIRST,))
    _assert_authorities(ledger, life, runtime, history, before)
    assert ledger.accounting() == before["accounting"]
    assert life._copy_budget._charged == before["raw_charge"] == 32
    assert life._admitted_bytes == before["ingress"] == 32
    assert history._records == history._erased_records == history._untrained_records == {}
    assert ledger._copy_slots == ledger._minimum_copy_slots == 0
    assert runtime._budget.updates_completed == 0


def test_should_refuse_default_off_untrained_capture_after_actual_delete_and_distinct_training():
    owner, life, runtime, ledger, controller, manager = _fresh(history=False)
    assert ledger._history() is None
    before = _authorities(ledger, life, runtime, None)
    weak_arrivals = _weak_arrivals(runtime)
    with pytest.raises(ValueError):
        ledger.delete((FIRST,))
    assert FIRST not in owner._revoked_keys and runtime._inbox._erased == {}
    assert all(reference() is not None for reference in weak_arrivals)
    assert ledger.accounting() == before["accounting"]
    assert life._copy_budget._charged == before["raw_charge"]
    report = life.delete((FIRST,))
    _assert_actual_untrained_erasure(owner, runtime, controller, report, weak_arrivals)
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 1
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 1
    assert set(runtime._inbox._applied) == {SECOND} and set(runtime._inbox._erased) == {FIRST}
    assert life._copy_budget._charged == 96 and life._admitted_bytes == 64
    assert ledger._history() is None and FIRST in owner._revoked_keys
    stable = _authorities(ledger, life, runtime, None)
    with pytest.raises(ValueError):
        manager.capture(controller)
    _assert_authorities(ledger, life, runtime, None, stable)
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0
    assert set(runtime._inbox._applied) == {SECOND} and set(runtime._inbox._erased) == {FIRST}
    assert runtime._budget.updates_completed == 1 and not runtime._retired
    assert ledger._history() is None and SECOND not in owner._revoked_keys


def test_should_keep_first_untrained_witness_when_original_record_allowance_exhausts():
    owner, life, runtime, ledger, controller, _ = _fresh(records=1)
    history = ledger._history()
    assert history is not None
    weak_first = _weak_arrivals(runtime)
    before = _authorities(ledger, life, runtime, history, records=1)
    report = ledger.delete((FIRST,))
    _assert_actual_untrained_erasure(owner, runtime, controller, report, weak_first)
    _assert_authorities(ledger, life, runtime, history, before, transitioned=True)
    witness = _assert_witness(owner, runtime, history, arrival="pair")
    assert ledger.accounting().records_created == ledger._admission.limits.max_records_created == 1
    tombstone = runtime._inbox._erased[FIRST]
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    weak_second = _weak_arrivals(runtime, SECOND)
    before_second = _authorities(ledger, life, runtime, history, records=1)
    experiences, labels = runtime._inbox._experiences, runtime._inbox._labels
    with pytest.raises(ValueError, match="cumulative record/invocation/metadata limit exhausted"):
        ledger.delete((SECOND,))
    assert life._failed and runtime._stopped
    assert runtime._inbox._experiences is experiences and runtime._inbox._labels is labels
    assert set(experiences) == set(labels) == {SECOND}
    assert all(reference() is not None for reference in weak_second)
    assert all(reference() is None for reference in weak_first)
    assert set(runtime._inbox._erased) == {FIRST} and runtime._inbox._erased[FIRST] is tombstone
    assert history._untrained_records[FIRST] is history._untrained_sealed[FIRST] is witness
    assert set(history._untrained_records) == {FIRST}
    assert witness.tombstone() is tombstone and witness.data.completed_updates == 0
    assert runtime._inbox._applied == {} and not runtime._candidate._model._replay_memory
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 0
    assert FIRST in owner._revoked_keys and SECOND in owner._revoked_keys
    _assert_authorities(ledger, life, runtime, history, before_second)
    assert ledger.accounting() == before_second["accounting"]
    assert ledger.accounting().records_created == 1 and ledger.accounting().invocations_started == 0
    assert life._copy_budget._charged == before_second["raw_charge"] == 64
    assert life._admitted_bytes == before_second["ingress"] == 64
    assert ledger._copy_slots == ledger._minimum_copy_slots == 0
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0


def test_should_refuse_arrival_change_during_genuine_native_erase_before_inbox_commit(monkeypatch):
    original_erase = learner_adapter._managed_erase
    fired = [0]

    def erase(model):
        result = original_erase(model)
        if model is runtime._candidate:
            assert fired[0] == 0
            fired[0] += 1
            source = runtime._inbox._experiences[FIRST]
            assert source.observed_at == 1 and runtime._inbox._labels[FIRST].arrived_at == 3
            object.__setattr__(source, "observed_at", 2)
        # Why: the port is installed at original birth, delegates each genuine
        # erase exactly once and returns its actual unchanged result.
        return result

    monkeypatch.setattr(learner_adapter, "_managed_erase", erase)
    owner, life, runtime, ledger, controller, _ = _fresh()
    history = ledger._history()
    assert history is not None
    before = _authorities(ledger, life, runtime, history)
    weak_arrivals = _weak_arrivals(runtime)
    experiences, labels = runtime._inbox._experiences, runtime._inbox._labels
    with pytest.raises(
        ValueError, match="untrained original arrival contents changed before erasure"
    ):
        ledger.delete((FIRST,))
    assert fired[0] == 1 and life._failed and runtime._stopped
    assert runtime._inbox._experiences is experiences and runtime._inbox._labels is labels
    assert set(experiences) == set(labels) == {FIRST}
    assert experiences[FIRST].observed_at == 2 and labels[FIRST].arrived_at == 3
    assert all(reference() is not None for reference in weak_arrivals)
    assert runtime._inbox._erased == runtime._inbox._applied == {}
    assert history._records == history._erased_records == history._untrained_records == {}
    assert history._untrained_sealed == {} and FIRST in owner._revoked_keys
    assert not runtime._candidate._model._replay_memory
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 0
    _assert_authorities(ledger, life, runtime, history, before)
    assert ledger.accounting() == before["accounting"]
    assert life._copy_budget._charged == before["raw_charge"] == 32
    assert life._admitted_bytes == before["ingress"] == 32
    assert ledger._copy_slots == ledger._minimum_copy_slots == 0
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0
