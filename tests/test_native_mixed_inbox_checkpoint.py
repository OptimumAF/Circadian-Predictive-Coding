"""N11: actual complete live/erased inbox capture and prepared handoff.

Why this: original erased receipts preserve work and require separate observed
lineage. Paid actual memo copies may bind that proof to copied metadata without
reviving payloads or consent, renewing quotas or dropping accounting history.
"""

from dataclasses import fields, replace
import inspect
from typing import Any
from weakref import ReferenceType

import pytest

from src.adapters.numpy_replay_origins import replay_payload_fingerprint
from src.app.erased_inbox_origins import ErasedInboxOrigin, erased_inbox_origin_stamp
from src.app.managed_inbox_origins import ManagedInboxOrigins
from test_native_checkpoint_eviction import _queue_original_distinct_sample
from test_native_erased_inbox_history import _assert_actual_erasure, _weak_payloads
from test_native_managed_replay_checkpoints import observe_native_work, setup


FIRST = ("e1", "s1")
SECOND = ("e1", "s2")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_mixed_history_work(tmp_path_factory):
    limits = dict(
        models=7,
        learners=46,
        forks=38,
        wakes=14,
        steps=28,
        stores=14,
        predicts=14,
        array_copies=70,
        captures=8,
        preparations=6,
        native_restores=6,
        handoffs=2,
        graph_events=160,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=7,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _mixed_setup(*, history=True, unobserved=False, invocations=32, fingerprint=None):
    owner, life, runtime, ledger, controller, manager = setup(
        replay_examples=1,
        retain_inbox_origins=history,
        live_records=32,
        invocations=invocations,
        fingerprint=fingerprint,
    )
    assert runtime._budget.budget.max_training_updates == 2
    assert life._copy_budget._limits.max_lifetime_owned_bytes == 4096
    assert tuple(vars(ledger._admission.limits).values()) == (
        32,
        64,
        invocations,
        128 * 1024,
        120,
        4096,
    )
    first_receipt = runtime._inbox._applied[FIRST]
    weak_payloads = _weak_payloads(runtime)
    raw_before = life._copy_budget._charged
    ingress_before = life._admitted_bytes
    report = life.delete((FIRST,)) if unobserved or not history else ledger.delete((FIRST,))
    _assert_actual_erasure(owner, runtime, controller, report, first_receipt, weak_payloads)
    assert life._copy_budget._charged == raw_before
    assert life._admitted_bytes == ingress_before
    if unobserved:
        original_history = ledger._history()
        assert history and original_history is not None
        assert original_history._erased_records == {}
        return owner, life, runtime, ledger, controller, manager, weak_payloads
    assert ledger.origins() == ()
    if history:
        original_history = ledger._history()
        assert original_history is not None
        assert original_history._records == original_history._sealed == {}
        assert set(original_history._erased_records) == {FIRST}
        assert original_history._erased_records[FIRST].receipt() is first_receipt
    else:
        assert ledger._history() is None
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    assert life._admitted_bytes == ingress_before + 32
    assert life._copy_budget._charged == raw_before + 32
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 2
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 2
    assert life._copy_budget._charged == raw_before + 64
    assert runtime._inbox._applied[FIRST] is first_receipt
    assert runtime._inbox._applied[SECOND] is poll.updates[0]
    assert set(runtime._inbox._experiences) == set(runtime._inbox._labels) == {SECOND}
    assert set(runtime._inbox._applied) == {FIRST, SECOND}
    assert set(runtime._inbox._erased) == {FIRST}
    assert FIRST in owner._revoked_keys and SECOND not in owner._revoked_keys
    assert all(reference() is None for reference in weak_payloads.values())
    assert len(runtime._candidate._model._replay_memory) == 1
    assert ledger.origins()[0].key == SECOND
    return owner, life, runtime, ledger, controller, manager, weak_payloads


def _authorities(ledger, life, runtime):
    return dict(
        admission=ledger._admission,
        limits=ledger._admission.limits,
        limit_values=tuple(vars(ledger._admission.limits).values()),
        raw=life._copy_budget,
        raw_limits=life._copy_budget._limits,
        work=runtime._budget,
        work_policy=runtime._budget.budget,
        clock=runtime._inbox._clock,
        history=ledger._history(),
        policy=life._policy,
        registry=life._registry,
        native_model=runtime._candidate._model,
        retention=runtime._candidate._model._replay_retention_budget,
        accounting=ledger.accounting(),
        charged=life._copy_budget._charged,
        ingress=life._admitted_bytes,
        slots=ledger._copy_slots,
        minimum_slots=ledger._minimum_copy_slots,
        enrollments=life._registry._total,
        updates=runtime._budget.updates_completed,
    )


def _assert_authorities(ledger, life, runtime, before, *, exact_cost=False):
    assert ledger._admission is before["admission"]
    assert ledger._admission.limits is before["limits"]
    assert tuple(vars(ledger._admission.limits).values()) == before["limit_values"]
    assert life._copy_budget is before["raw"] and life._copy_budget._limits is before["raw_limits"]
    assert (
        life._policy is before["policy"]
        and life._policy.owned_payload_copies is before["raw_limits"]
    )
    assert life._registry is before["registry"]
    assert life._copy_budget._limits.max_lifetime_owned_bytes == 4096
    assert runtime._budget is before["work"] and runtime._budget.budget is before["work_policy"]
    assert runtime._inbox._clock is before["clock"]
    assert runtime._budget.updates_completed == before["updates"]
    assert ledger._history() is before["history"]
    assert before["native_model"]._replay_retention_budget is before["retention"]
    retention = runtime._candidate._model._replay_retention_budget
    assert (retention.max_examples, retention.max_bytes) == (1, 64)
    assert life._admitted_bytes == before["ingress"]
    current, previous = ledger.accounting(), before["accounting"]
    pairs = (
        (current.records_created, previous.records_created),
        (current.invocations_started, previous.invocations_started),
        (current.metadata_bytes_charged, previous.metadata_bytes_charged),
        (life._copy_budget._charged, before["charged"]),
        (ledger._copy_slots, before["slots"]),
        (ledger._minimum_copy_slots, before["minimum_slots"]),
        (life._registry._total, before["enrollments"]),
    )
    for after, prior in pairs:
        assert after == prior if exact_cost else after >= prior


def _assert_erased_witness(record, data, receipt, tombstone, limits):
    assert type(record) is ErasedInboxOrigin
    assert {field.name for field in fields(record)} == {
        "data",
        "receipt",
        "tombstone",
        "receipt_stamp",
        "tombstone_stamp",
    }
    assert not hasattr(record, "__dict__")
    assert record.data is data
    assert type(record.receipt) is ReferenceType and type(record.tombstone) is ReferenceType
    assert record.receipt() is receipt and record.tombstone() is tombstone
    assert all(
        not hasattr(record, name)
        for name in ("source", "label", "declaration", "features", "targets")
    )
    assert type(erased_inbox_origin_stamp(record, limits)) is tuple


def _assert_original_partition(owner, runtime, ledger):
    history = ledger._history()
    assert history is not None
    assert history._storage[0] is history._records and history._storage[1] is history._sealed
    assert history._erased_storage[0] is history._erased_records
    assert history._erased_storage[1] is history._erased_sealed
    live = ManagedInboxOrigins.verify(history, ledger, owner, runtime, ledger._read_tick(runtime))
    assert len(live) == 1 and live[0].data.key == SECOND
    assert live[0].receipt is not None
    assert live[0].source() is runtime._inbox._experiences[SECOND]
    assert live[0].label() is runtime._inbox._labels[SECOND]
    assert live[0].receipt() is runtime._inbox._applied[SECOND]
    assert set(history._erased_records) == set(history._erased_sealed) == {FIRST}
    erased = history._erased_records[FIRST]
    assert history._erased_sealed[FIRST] is erased
    _assert_erased_witness(
        erased,
        erased.data,
        runtime._inbox._applied[FIRST],
        runtime._inbox._erased[FIRST],
        history._content_limits,
    )
    assert set(runtime._inbox._applied) == {FIRST, SECOND}
    assert set(runtime._inbox._experiences) == set(runtime._inbox._labels) == {SECOND}
    assert set(runtime._inbox._erased) == {FIRST}
    assert FIRST in owner._revoked_keys and SECOND not in owner._revoked_keys
    assert ledger.accounting().live_records == ledger._copy_slots + 3
    return history, live[0], erased


def _assert_no_capture_publication(owner, runtime, controller, manager):
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert controller._pending == {} and controller._attempts == 0 and controller._models == []
    assert manager._active is None


def _publication_fingerprint(probe):
    def fingerprint(snapshot, maximum):
        result = replay_payload_fingerprint(snapshot, maximum)
        if not probe["armed"]:
            return result
        frame = inspect.currentframe()
        publication = active_current = None
        try:
            while frame is not None:
                if frame.f_code.co_name == "_current":
                    active_current = frame
                elif frame.f_code.co_name == "_publication_lease":
                    publication = frame
                    break
                frame = frame.f_back
            if publication is None or active_current is None:
                return result
            op = publication.f_locals["op"]
            assert op is probe["manager"]._active and op.materialized is not None
            prepared = publication.f_locals["prepared"]
            cursor, pairs = op.materialized
            erased = op.inbox_erasures[id(cursor)]
            assert len(pairs) == 1 and len(erased) == 1
            receipt = next(
                item for item in cursor.applied if (item.episode_id, item.sample_id) == FIRST
            )
            tombstone = next(item for item in cursor.erased if item.key == FIRST)
            assert prepared._inbox._applied[FIRST] is receipt
            assert prepared._inbox._erased[FIRST] is tombstone
            _assert_erased_witness(
                erased[0], probe["data"], receipt, tombstone, op.history._content_limits
            )
            assert (
                tuple(
                    erased_inbox_origin_stamp(record, op.history._content_limits)
                    for record in erased
                )
                == op.materialized_erased_stamp
            )
            if active_current is not probe["current_frame"]:
                probe["visits"] += 1
                probe["current_frame"] = active_current
            if probe["visits"] == 3 and not probe["fired"]:
                assert prepared is not probe["runtime"] and not probe["runtime"]._retired
                assert prepared._candidate is op.controller._models[0]
                assert op.restored_model is prepared._candidate._model
                probe["materialized"] = dict(
                    witness_id=id(erased[0]),
                    receipt_id=id(receipt),
                    tombstone_id=id(tombstone),
                    prepared_id=id(prepared),
                    data_id=id(erased[0].data),
                )
                probe["fired"] = True
                if probe["mutate"]:
                    assert tombstone.reason == "deleted"
                    object.__setattr__(tombstone, "reason", "expired")
                    probe["changed_reason"] = tombstone.reason
            return result
        finally:
            del frame, publication, active_current

    return fingerprint


def _probe(*, mutate=False):
    return dict(armed=False, fired=False, visits=0, current_frame=None, mutate=mutate)


def test_should_capture_materialize_and_publish_complete_original_live_erased_history():
    probe: dict[str, Any] = _probe()
    owner, life, runtime, ledger, controller, manager, weak_payloads = _mixed_setup(
        fingerprint=_publication_fingerprint(probe)
    )
    history, live, erased = _assert_original_partition(owner, runtime, ledger)
    original = _authorities(ledger, life, runtime)
    first_receipt, tombstone = runtime._inbox._applied[FIRST], runtime._inbox._erased[FIRST]
    token = manager.capture(controller)
    pending = controller._pending[token]
    checkpoint = manager._checkpoints[id(token)]
    assert checkpoint.cursor() is pending.view.inbox
    assert len(checkpoint.inbox) == len(checkpoint.erased_inbox) == 1
    captured_receipt = next(
        item for item in pending.view.inbox.applied if (item.episode_id, item.sample_id) == FIRST
    )
    captured_tombstone = pending.view.inbox.erased[0]
    assert captured_receipt is not first_receipt and captured_receipt == first_receipt
    assert captured_tombstone is not tombstone and captured_tombstone == tombstone
    _assert_erased_witness(
        checkpoint.erased_inbox[0],
        erased.data,
        captured_receipt,
        captured_tombstone,
        history._content_limits,
    )
    assert checkpoint.erased_inbox_stamp == tuple(
        erased_inbox_origin_stamp(record, history._content_limits)
        for record in checkpoint.erased_inbox
    )
    assert checkpoint.erased_inbox[0] is not erased
    assert checkpoint.inbox[0].origin.data is live.data
    captured = _authorities(ledger, life, runtime)
    probe.update(manager=manager, runtime=runtime, data=erased.data, armed=True)
    try:
        prepared = manager.restore(controller, token)
    finally:
        probe["armed"] = False
        probe["current_frame"] = None
    assert probe["fired"] and probe["visits"] == 3
    assert owner._shared._runtime is prepared and runtime._retired
    assert runtime._inbox._historical_completed_updates == 2
    assert controller._pending == {} and controller._attempts == 1
    assert prepared._candidate is controller._models[0]
    assert id(prepared) == probe["materialized"]["prepared_id"]
    assert (
        prepared._budget is runtime._budget
        and prepared._payload_lineage is runtime._payload_lineage
    )
    assert prepared._inbox._clock is runtime._inbox._clock
    rebound_history, rebound_live, rebound_erased = _assert_original_partition(
        owner, prepared, ledger
    )
    assert rebound_history is history and rebound_live.data is live.data
    assert rebound_erased is not erased and rebound_erased is not checkpoint.erased_inbox[0]
    assert rebound_erased.data is erased.data
    assert id(rebound_erased.receipt()) == probe["materialized"]["receipt_id"]
    assert id(rebound_erased.tombstone()) == probe["materialized"]["tombstone_id"]
    assert (
        rebound_erased.receipt() is not captured_receipt
        and rebound_erased.receipt() is not first_receipt
    )
    assert (
        rebound_erased.tombstone() is not captured_tombstone
        and rebound_erased.tombstone() is not tombstone
    )
    assert rebound_erased.receipt() == first_receipt and rebound_erased.tombstone() == tombstone
    assert prepared._inbox._applied[FIRST].diagnostic == first_receipt.diagnostic
    assert prepared._inbox._erased[FIRST].reason == "deleted"
    assert all(reference() is None for reference in weak_payloads.values())
    assert len(prepared._candidate._model._replay_memory) == 1
    row = prepared._candidate._model._replay_memory[0]
    canonical = ledger._rows[id(row)]
    assert canonical.data.key == SECOND
    assert canonical.source() is prepared._inbox._experiences[SECOND]
    assert canonical.label() is prepared._inbox._labels[SECOND]
    assert canonical.receipt() is prepared._inbox._applied[SECOND]
    assert ledger.origins()[0].key == SECOND
    assert prepared._budget.updates_completed == prepared._candidate._model._epoch_count == 2
    _assert_authorities(ledger, life, prepared, original)
    _assert_authorities(ledger, life, prepared, captured)
    assert life._copy_budget._charged > captured["charged"]
    assert ledger._copy_slots > captured["slots"]
    assert ledger.accounting().records_created > captured["accounting"].records_created
    assert manager._active is None


def test_should_refuse_cloned_original_erased_receipt_during_mixed_capture():
    owner, life, runtime, ledger, controller, manager, _ = _mixed_setup()
    _, _, erased = _assert_original_partition(owner, runtime, ledger)
    original = _authorities(ledger, life, runtime)
    receipt = runtime._inbox._applied[FIRST]
    runtime._inbox._applied[FIRST] = replace(receipt)
    assert (
        runtime._inbox._applied[FIRST] == receipt and runtime._inbox._applied[FIRST] is not receipt
    )
    assert erased.receipt() is receipt
    try:
        with pytest.raises(ValueError):
            manager.capture(controller)
    finally:
        runtime._inbox._applied[FIRST] = receipt
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_original_partition(owner, runtime, ledger)
    _assert_authorities(ledger, life, runtime, original, exact_cost=True)


def test_should_refuse_changed_original_erased_tombstone_during_mixed_capture():
    owner, life, runtime, ledger, controller, manager, _ = _mixed_setup()
    _assert_original_partition(owner, runtime, ledger)
    original = _authorities(ledger, life, runtime)
    tombstone = runtime._inbox._erased[FIRST]
    object.__setattr__(tombstone, "reason", "expired")
    try:
        with pytest.raises(ValueError):
            manager.capture(controller)
    finally:
        object.__setattr__(tombstone, "reason", "deleted")
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_original_partition(owner, runtime, ledger)
    _assert_authorities(ledger, life, runtime, original, exact_cost=True)


def test_should_refuse_capture_after_unobserved_original_erasure_without_retrospective_history():
    owner, life, runtime, ledger, controller, manager, weak_payloads = _mixed_setup(unobserved=True)
    original = _authorities(ledger, life, runtime)
    history = ledger._history()
    assert history is not None
    assert history._erased_records == history._erased_sealed == {}
    with pytest.raises(ValueError):
        ManagedInboxOrigins.prune(history, runtime)
    with pytest.raises(ValueError):
        manager.capture(controller)
    assert history._erased_records == history._erased_sealed == {}
    assert runtime._budget.updates_completed == 1
    assert all(reference() is None for reference in weak_payloads.values())
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_authorities(ledger, life, runtime, original, exact_cost=True)


def test_should_refuse_mixed_capture_without_original_history_birth_enrollment():
    owner, life, runtime, ledger, controller, manager, weak_payloads = _mixed_setup(history=False)
    original = _authorities(ledger, life, runtime)
    assert ledger._history() is None
    with pytest.raises(ValueError):
        manager.capture(controller)
    assert ledger._history() is None
    assert all(reference() is None for reference in weak_payloads.values())
    assert set(runtime._inbox._applied) == {FIRST, SECOND}
    assert FIRST in owner._revoked_keys and SECOND not in owner._revoked_keys
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_authorities(ledger, life, runtime, original)


def test_should_refuse_late_actual_materialized_erased_change_before_handoff():
    probe: dict[str, Any] = _probe(mutate=True)
    owner, life, runtime, ledger, controller, manager, weak_payloads = _mixed_setup(
        fingerprint=_publication_fingerprint(probe)
    )
    _, _, erased = _assert_original_partition(owner, runtime, ledger)
    token = manager.capture(controller)
    original = _authorities(ledger, life, runtime)
    anchors = ledger._anchors
    first_receipt, tombstone = runtime._inbox._applied[FIRST], runtime._inbox._erased[FIRST]
    probe.update(manager=manager, runtime=runtime, data=erased.data, armed=True)
    try:
        with pytest.raises(ValueError):
            manager.restore(controller, token)
    finally:
        probe["armed"] = False
        probe["current_frame"] = None
    assert probe["fired"] and probe["visits"] == 3 and probe["changed_reason"] == "expired"
    assert probe["materialized"]["prepared_id"] != id(runtime)
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert controller._pending[token].owner is runtime and ledger._anchors is anchors
    assert controller._attempts == 1 and len(controller._models) == 1
    assert runtime._inbox._applied[FIRST] is first_receipt
    assert runtime._inbox._erased[FIRST] is tombstone and tombstone.reason == "deleted"
    assert manager._checkpoints[id(token)].erased_inbox[0].tombstone().reason == "deleted"
    assert all(reference() is None for reference in weak_payloads.values())
    _assert_original_partition(owner, runtime, ledger)
    _assert_authorities(ledger, life, runtime, original)
    assert life._copy_budget._charged > original["charged"]
    assert ledger._copy_slots > original["slots"]
    assert ledger.accounting().records_created > original["accounting"].records_created
    assert manager._active is None


def test_should_refuse_mixed_capture_when_original_two_invocations_are_exhausted():
    owner, life, runtime, ledger, controller, manager, _ = _mixed_setup(invocations=2)
    _assert_original_partition(owner, runtime, ledger)
    original = _authorities(ledger, life, runtime)
    assert ledger._admission.limits.max_invocations == 2
    assert original["accounting"].invocations_started == 2
    with pytest.raises(ValueError):
        manager.capture(controller)
    _assert_no_capture_publication(owner, runtime, controller, manager)
    assert ledger._admission.limits.max_invocations == 2
    assert ledger.accounting().invocations_started == 2
    _assert_original_partition(owner, runtime, ledger)
    _assert_authorities(ledger, life, runtime, original)
