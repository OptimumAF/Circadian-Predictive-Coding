"""Actual paid checkpoint lineage across live, erased and untrained history.

Why this: untrained tombstones preserve an original work boundary without a
receipt. Actual memo bindings must carry that proof beside erased receipts and
live native data through the final original publication guards.
"""

from dataclasses import fields, replace
import inspect
from math import isfinite
from typing import Any
from weakref import ReferenceType, ref

import numpy as np
import pytest

from src.adapters.numpy_replay_origins import replay_payload_fingerprint
from src.app.erased_inbox_origins import erased_inbox_origin_stamp
from src.app.managed_inbox_origins import ManagedInboxOrigins
from src.app.untrained_inbox_origins import UntrainedInboxOrigin, untrained_inbox_origin_stamp
from src.core.data_retention import DataCleanupReport
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.untrained_inbox_origin import UntrainedInboxOriginData
from test_managed_experience import declaration
from test_native_checkpoint_eviction import _queue_original_distinct_sample
from test_native_managed_replay_checkpoints import observe_native_work, setup
from test_native_mixed_inbox_checkpoint import (
    _assert_authorities,
    _assert_erased_witness,
    _assert_no_capture_publication,
    _authorities,
)


FIRST = ("e1", "s1")
SECOND = ("e1", "s2")
THIRD = ("e1", "s3")
STORES = ("_storage", "_erased_storage", "_untrained_storage")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_three_partition_work(tmp_path_factory):
    limits = dict(
        models=7,
        learners=56,
        forks=49,
        wakes=14,
        steps=28,
        stores=14,
        predicts=14,
        array_copies=70,
        captures=14,
        preparations=7,
        native_restores=7,
        handoffs=2,
        graph_events=160,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=7,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _queue_first_source(owner, clock):
    owner.declare(declaration("s1", "person-1"))
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
    clock.advance_to(1)


def _queue_third_pair(owner, clock):
    owner.declare(declaration("s3", "person-3"))
    owner.record_experience(
        Experience(
            "s3",
            "e1",
            8,
            "actor-0",
            np.array([[0.0, 0.0, 1.0]]),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(LabelArrival("label-s3", "s3", "e1", 9, "actor-0", np.array([[0.0]])))
    clock.advance_to(9)


def _train_once(owner, ledger):
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1
    return poll.updates[0]


def _weak_erased_payloads(runtime):
    inbox, row = runtime._inbox, runtime._candidate._model._replay_memory[0]
    return (
        ref(inbox._experiences[FIRST]),
        ref(inbox._experiences[FIRST].features),
        ref(inbox._experiences[SECOND]),
        ref(inbox._experiences[SECOND].features),
        ref(inbox._labels[SECOND]),
        ref(inbox._labels[SECOND].targets),
        ref(row),
        ref(row.input_batch),
        ref(row.target_batch),
    )


def _three_partition_setup(*, history=True, unobserved=False, invocations=32, fingerprint=None):
    owner, life, runtime, ledger, controller, manager = setup(
        train_first=False,
        queue_first=False,
        records_created=64,
        retain_inbox_origins=history,
        replay_examples=1,
        live_records=32,
        invocations=invocations,
        fingerprint=fingerprint,
    )
    assert tuple(vars(ledger._admission.limits).values()) == (
        32,
        64,
        invocations,
        128 * 1024,
        120,
        4096,
    )
    assert runtime._budget.budget.max_training_updates == 2
    assert life._copy_budget._limits.max_lifetime_owned_bytes == 4096
    assert runtime._budget.updates_completed == 0 and not runtime._candidate._model._replay_memory
    clock = runtime._inbox._clock
    _queue_first_source(owner, clock)
    _queue_original_distinct_sample(owner, clock)
    second_receipt = _train_once(owner, ledger)
    assert second_receipt.update_number == 1
    assert set(runtime._inbox._applied) == {SECOND} and FIRST not in runtime._inbox._labels
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 1
    assert runtime._inbox._experiences[FIRST].observed_at == 1
    assert (second_receipt.observed_at, second_receipt.arrived_at) == (3, 4)
    assert life._admitted_bytes == 56 and life._copy_budget._charged == 88
    weak_payloads = _weak_erased_payloads(runtime)
    raw_before, ingress_before = life._copy_budget._charged, life._admitted_bytes
    report = life.delete((FIRST,)) if unobserved or not history else ledger.delete((FIRST,))
    assert type(report) is DataCleanupReport
    assert report.reason == "deleted" and report.requested_keys == (FIRST,)
    assert report.revoked_keys == (FIRST, SECOND)
    assert report.model_snapshots_erased == report.inboxes_cleared == 1
    assert report.checkpoints_invalidated == report.promotions_invalidated == 0
    assert set(runtime._inbox._erased) == {FIRST, SECOND}
    assert runtime._inbox._experiences == runtime._inbox._labels == {}
    assert set(runtime._inbox._applied) == {SECOND}
    assert runtime._inbox._applied[SECOND] is second_receipt
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 1
    assert not runtime._candidate._model._replay_memory
    assert FIRST in owner._revoked_keys and SECOND in owner._revoked_keys
    assert life._copy_budget._charged == raw_before and life._admitted_bytes == ingress_before
    assert all(reference() is None for reference in weak_payloads)
    assert controller._pending == {} and controller._attempts == 0 and controller._models == []
    if unobserved:
        # Missing original observed lineage blocks further ledger training;
        # never bypass that guard to manufacture the third receipt.
        assert history and ledger._history() is not None
        return owner, life, runtime, ledger, controller, manager, weak_payloads
    assert ledger.origins() == ()
    if history:
        enrolled = ledger._history()
        assert enrolled is not None
        assert enrolled._records == enrolled._sealed == {}
        assert set(enrolled._erased_records) == {SECOND}
        assert set(enrolled._untrained_records) == {FIRST}
        assert enrolled._erased_records[SECOND].receipt() is second_receipt
        assert enrolled._untrained_records[FIRST].data.completed_updates == 1
        assert ledger.accounting().live_records == 2
    else:
        assert ledger._history() is None
    _queue_third_pair(owner, clock)
    assert life._admitted_bytes == ingress_before + 32
    assert life._copy_budget._charged == raw_before + 32
    third_receipt = _train_once(owner, ledger)
    assert third_receipt.update_number == 2
    assert set(runtime._inbox._applied) == {SECOND, THIRD}
    assert runtime._inbox._applied[SECOND] is second_receipt
    assert runtime._inbox._applied[THIRD] is third_receipt
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 2
    assert life._copy_budget._charged == 152 and life._admitted_bytes == 88
    assert FIRST not in runtime._inbox._applied and THIRD not in owner._revoked_keys
    assert all(reference() is None for reference in weak_payloads)
    assert len(runtime._candidate._model._replay_memory) == 1
    return owner, life, runtime, ledger, controller, manager, weak_payloads


def _stores(history):
    return tuple(getattr(history, name) for name in STORES)


def _assert_stores(history, previous=None, *, replaced=False):
    for name, current in zip(STORES, _stores(history)):
        assert type(current) is tuple and len(current) == 2
        assert all(type(mapping) is dict for mapping in current)
        assert current[0] is not current[1]
        if previous is not None:
            old = previous[STORES.index(name)]
            if replaced:
                assert current is not old
                assert all(mapping is not before for mapping in current for before in old)
            else:
                assert current is old
    assert history._storage[0] is history._records and history._storage[1] is history._sealed
    assert history._erased_storage[0] is history._erased_records
    assert history._erased_storage[1] is history._erased_sealed
    assert history._untrained_storage[0] is history._untrained_records
    assert history._untrained_storage[1] is history._untrained_sealed


def _assert_untrained_witness(record, data, tombstone, limits):
    assert type(record) is UntrainedInboxOrigin and type(record.data) is UntrainedInboxOriginData
    assert {field.name for field in fields(record)} == {"data", "tombstone", "tombstone_stamp"}
    assert not hasattr(record, "__dict__") and record.data is data
    assert type(record.tombstone) is ReferenceType and record.tombstone() is tombstone
    assert all(
        not hasattr(record, name)
        for name in ("receipt", "source", "label", "declaration", "features", "targets")
    )
    assert data.key == FIRST and data.completed_updates == 1
    assert (data.observed_at, data.event_id, data.arrived_at, data.erased_at, data.reason) == (
        1,
        None,
        None,
        4,
        "deleted",
    )
    assert (data.actor_version, data.learner_version, data.subject_id, data.source_id) == (
        "actor-0",
        "candidate-0",
        "person-1",
        "local",
    )
    assert type(untrained_inbox_origin_stamp(record, limits)) is tuple


def _assert_original_partition(owner, runtime, ledger):
    history = ledger._history()
    assert history is not None
    _assert_stores(history)
    live = ManagedInboxOrigins.verify(history, ledger, owner, runtime, ledger._read_tick(runtime))
    assert len(live) == 1 and live[0].data.key == THIRD
    assert live[0].source() is runtime._inbox._experiences[THIRD]
    assert live[0].label() is runtime._inbox._labels[THIRD]
    assert live[0].receipt is not None and live[0].receipt() is runtime._inbox._applied[THIRD]
    assert set(history._erased_records) == set(history._erased_sealed) == {SECOND}
    assert set(history._untrained_records) == set(history._untrained_sealed) == {FIRST}
    erased, untrained = history._erased_records[SECOND], history._untrained_records[FIRST]
    assert (
        history._erased_sealed[SECOND] is erased and history._untrained_sealed[FIRST] is untrained
    )
    _assert_erased_witness(
        erased,
        erased.data,
        runtime._inbox._applied[SECOND],
        runtime._inbox._erased[SECOND],
        history._content_limits,
    )
    _assert_untrained_witness(
        untrained,
        untrained.data,
        runtime._inbox._erased[FIRST],
        history._content_limits,
    )
    assert set(runtime._inbox._experiences) == set(runtime._inbox._labels) == {THIRD}
    assert tuple(runtime._inbox._applied) == (SECOND, THIRD)
    assert set(runtime._inbox._erased) == {FIRST, SECOND}
    assert FIRST not in runtime._inbox._applied
    assert runtime._inbox._applied[SECOND].update_number == 1
    assert runtime._inbox._applied[THIRD].update_number == 2
    assert all(isfinite(receipt.diagnostic.value) for receipt in runtime._inbox._applied.values())
    assert FIRST in owner._revoked_keys and SECOND in owner._revoked_keys
    assert THIRD not in owner._revoked_keys
    assert ledger.accounting().live_records == ledger._copy_slots + 4
    return history, live[0], erased, untrained


def _cursor_receipt(cursor, key):
    return next(item for item in cursor.applied if (item.episode_id, item.sample_id) == key)


def _cursor_tombstone(cursor, key):
    return next(item for item in cursor.erased if item.key == key)


def _probe(*, mutate=False):
    return dict(armed=False, fired=False, visits=0, current_frame=None, mutate=mutate)


def _publication_fingerprint(probe):
    def fingerprint(snapshot, maximum):
        digest = replay_payload_fingerprint(snapshot, maximum)
        if not probe["armed"] or probe["fired"]:
            return digest
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
                return digest
            op = publication.f_locals["op"]
            prepared = publication.f_locals["prepared"]
            assert op is probe["manager"]._active and op.materialized is not None
            cursor, pairs = op.materialized
            erased, untrained = op.inbox_erasures[id(cursor)], op.inbox_untrained[id(cursor)]
            assert len(pairs) == len(erased) == len(untrained) == 1
            first_tombstone = _cursor_tombstone(cursor, FIRST)
            second_tombstone = _cursor_tombstone(cursor, SECOND)
            second_receipt = _cursor_receipt(cursor, SECOND)
            third_receipt = _cursor_receipt(cursor, THIRD)
            assert prepared._inbox._erased[FIRST] is first_tombstone
            assert prepared._inbox._erased[SECOND] is second_tombstone
            assert prepared._inbox._applied[SECOND] is second_receipt
            assert prepared._inbox._applied[THIRD] is third_receipt
            assert pairs[0].source() is prepared._inbox._experiences[THIRD]
            assert pairs[0].label() is prepared._inbox._labels[THIRD]
            assert pairs[0].receipt() is third_receipt
            _assert_untrained_witness(
                untrained[0],
                probe["untrained_data"],
                first_tombstone,
                op.history._content_limits,
            )
            _assert_erased_witness(
                erased[0],
                probe["erased_data"],
                second_receipt,
                second_tombstone,
                op.history._content_limits,
            )
            assert (
                tuple(
                    untrained_inbox_origin_stamp(item, op.history._content_limits)
                    for item in untrained
                )
                == op.materialized_untrained_stamp
            )
            assert (
                tuple(
                    erased_inbox_origin_stamp(item, op.history._content_limits) for item in erased
                )
                == op.materialized_erased_stamp
            )
            if active_current is not probe["current_frame"]:
                probe["visits"] += 1
                probe["current_frame"] = active_current
            if probe["visits"] == 3:
                assert prepared is not probe["runtime"] and not probe["runtime"]._retired
                assert prepared._candidate is op.controller._models[0]
                assert op.restored_model is prepared._candidate._model
                probe["materialized"] = dict(
                    prepared_id=id(prepared),
                    cursor_id=id(cursor),
                    untrained_id=id(untrained[0]),
                    first_tombstone_id=id(first_tombstone),
                    erased_id=id(erased[0]),
                    second_tombstone_id=id(second_tombstone),
                    second_receipt_id=id(second_receipt),
                    third_receipt_id=id(third_receipt),
                )
                probe["fired"] = True
                if probe["mutate"]:
                    object.__setattr__(first_tombstone, "reason", "expired")
                    probe["changed_reason"] = first_tombstone.reason
            # The genuine fingerprint is returned even after the mutation;
            # subsequent refusal must come from production's final proof.
            return digest
        finally:
            del frame, publication, active_current

    return fingerprint


def _arm(probe, manager, runtime, erased, untrained):
    probe.update(
        manager=manager,
        runtime=runtime,
        erased_data=erased.data,
        untrained_data=untrained.data,
        armed=True,
    )


def _assert_capture(owner, runtime, ledger, controller, manager, token, erased, untrained):
    checkpoint = manager._checkpoints[id(token)]
    cursor = controller._pending[token].view.inbox
    assert checkpoint.cursor() is cursor
    assert (
        len(checkpoint.inbox)
        == len(checkpoint.erased_inbox)
        == len(checkpoint.untrained_inbox)
        == 1
    )
    first_tombstone = _cursor_tombstone(cursor, FIRST)
    second_tombstone = _cursor_tombstone(cursor, SECOND)
    second_receipt = _cursor_receipt(cursor, SECOND)
    assert first_tombstone is not runtime._inbox._erased[FIRST]
    assert first_tombstone == runtime._inbox._erased[FIRST]
    assert second_tombstone is not runtime._inbox._erased[SECOND]
    assert second_tombstone == runtime._inbox._erased[SECOND]
    assert second_receipt is not runtime._inbox._applied[SECOND]
    assert second_receipt == runtime._inbox._applied[SECOND]
    history = ledger._history()
    assert history is not None
    _assert_untrained_witness(
        checkpoint.untrained_inbox[0],
        untrained.data,
        first_tombstone,
        history._content_limits,
    )
    _assert_erased_witness(
        checkpoint.erased_inbox[0],
        erased.data,
        second_receipt,
        second_tombstone,
        history._content_limits,
    )
    assert checkpoint.untrained_inbox[0] is not untrained
    assert checkpoint.erased_inbox[0] is not erased
    assert checkpoint.untrained_inbox_stamp == tuple(
        untrained_inbox_origin_stamp(item, history._content_limits)
        for item in checkpoint.untrained_inbox
    )
    assert checkpoint.erased_inbox_stamp == tuple(
        erased_inbox_origin_stamp(item, history._content_limits) for item in checkpoint.erased_inbox
    )
    assert tuple((item.episode_id, item.sample_id) for item in cursor.applied) == (SECOND, THIRD)
    assert {item.key for item in cursor.erased} == {FIRST, SECOND}
    assert FIRST not in {(item.episode_id, item.sample_id) for item in cursor.applied}
    assert FIRST in owner._revoked_keys and SECOND in owner._revoked_keys
    return checkpoint, cursor


def test_should_publish_actual_three_partition_memo_lineage_under_original_authority():
    probe: dict[str, Any] = _probe()
    owner, life, runtime, ledger, controller, manager, weak_payloads = _three_partition_setup(
        fingerprint=_publication_fingerprint(probe),
    )
    history, live, erased, untrained = _assert_original_partition(owner, runtime, ledger)
    original, original_stores = _authorities(ledger, life, runtime), _stores(history)
    original_first, original_second = runtime._inbox._erased[FIRST], runtime._inbox._erased[SECOND]
    original_receipt = runtime._inbox._applied[SECOND]
    token = manager.capture(controller)
    checkpoint, cursor = _assert_capture(
        owner, runtime, ledger, controller, manager, token, erased, untrained
    )
    assert checkpoint.inbox[0].origin.data is live.data
    _assert_stores(history, original_stores)
    captured = _authorities(ledger, life, runtime)
    assert captured["slots"] > original["slots"]
    assert (
        captured["accounting"].records_created - original["accounting"].records_created
        == captured["slots"] - original["slots"]
    )
    _arm(probe, manager, runtime, erased, untrained)
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
    rebound_history, rebound_live, rebound_erased, rebound_untrained = _assert_original_partition(
        owner, prepared, ledger
    )
    assert rebound_history is history and rebound_live.data is live.data
    _assert_stores(history, original_stores, replaced=True)
    assert (
        rebound_untrained is not untrained
        and rebound_untrained is not checkpoint.untrained_inbox[0]
    )
    assert id(rebound_untrained) == probe["materialized"]["untrained_id"]
    assert rebound_untrained.data is untrained.data
    assert rebound_untrained.tombstone() is prepared._inbox._erased[FIRST]
    assert id(rebound_untrained.tombstone()) == probe["materialized"]["first_tombstone_id"]
    assert rebound_untrained.tombstone() is not original_first
    assert rebound_untrained.tombstone() is not _cursor_tombstone(cursor, FIRST)
    assert rebound_untrained.tombstone() == original_first
    assert rebound_erased.data is erased.data
    assert id(rebound_erased.tombstone()) == probe["materialized"]["second_tombstone_id"]
    assert id(rebound_erased.receipt()) == probe["materialized"]["second_receipt_id"]
    assert rebound_erased.tombstone() is not original_second
    assert rebound_erased.receipt() is not original_receipt
    assert rebound_erased.tombstone() is not _cursor_tombstone(cursor, SECOND)
    assert rebound_erased.receipt() is not _cursor_receipt(cursor, SECOND)
    assert (
        rebound_erased.tombstone() == original_second
        and rebound_erased.receipt() == original_receipt
    )
    assert id(prepared._inbox._applied[THIRD]) == probe["materialized"]["third_receipt_id"]
    assert prepared._inbox._applied[SECOND].diagnostic == original_receipt.diagnostic
    assert all(reference() is None for reference in weak_payloads)
    assert len(prepared._candidate._model._replay_memory) == 1
    row = prepared._candidate._model._replay_memory[0]
    canonical = ledger._rows[id(row)]
    assert canonical.data.key == THIRD and canonical.data.update_number == 2
    assert canonical.source() is prepared._inbox._experiences[THIRD]
    assert canonical.label() is prepared._inbox._labels[THIRD]
    assert canonical.receipt() is prepared._inbox._applied[THIRD]
    assert ledger.origins()[0].key == THIRD
    assert prepared._budget.updates_completed == prepared._candidate._model._epoch_count == 2
    _assert_authorities(ledger, life, prepared, original)
    _assert_authorities(ledger, life, prepared, captured)
    assert ledger._copy_slots > captured["slots"]
    assert (
        ledger.accounting().records_created - captured["accounting"].records_created
        == ledger._copy_slots - captured["slots"]
    )
    assert life._copy_budget._charged > captured["charged"] and manager._active is None
    published, published_stores = _authorities(ledger, life, prepared), _stores(history)
    _assert_original_partition(owner, prepared, ledger)
    _assert_stores(history, published_stores)
    _assert_authorities(ledger, life, prepared, published, exact_cost=True)


def test_should_refuse_three_partition_capture_without_original_history_enrollment():
    owner, life, runtime, ledger, controller, manager, weak_payloads = _three_partition_setup(
        history=False
    )
    before = _authorities(ledger, life, runtime)
    assert ledger._history() is None
    with pytest.raises(ValueError):
        manager.capture(controller)
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_authorities(ledger, life, runtime, before)
    assert ledger._history() is None and FIRST not in runtime._inbox._applied
    assert set(runtime._inbox._erased) == {FIRST, SECOND}
    assert set(runtime._inbox._applied) == {SECOND, THIRD}
    assert (
        FIRST in owner._revoked_keys
        and SECOND in owner._revoked_keys
        and THIRD not in owner._revoked_keys
    )
    assert all(reference() is None for reference in weak_payloads)


def test_should_refuse_three_partition_capture_after_unobserved_original_cleanup():
    owner, life, runtime, ledger, controller, manager, weak_payloads = _three_partition_setup(
        unobserved=True
    )
    history = ledger._history()
    assert history is not None
    before, stores = _authorities(ledger, life, runtime), _stores(history)
    assert history._erased_records == history._untrained_records == {}
    with pytest.raises(ValueError):
        ManagedInboxOrigins.prune(history, runtime)
    with pytest.raises(ValueError):
        manager.capture(controller)
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_authorities(ledger, life, runtime, before, exact_cost=True)
    _assert_stores(history, stores)
    assert history._erased_records == history._untrained_records == {}
    assert set(runtime._inbox._applied) == {SECOND} and FIRST not in runtime._inbox._applied
    assert runtime._budget.updates_completed == 1
    assert all(reference() is None for reference in weak_payloads)


def test_should_refuse_equal_valued_cloned_captured_untrained_tombstone_before_new_copy():
    owner, life, runtime, ledger, controller, manager, _ = _three_partition_setup()
    history, _, erased, untrained = _assert_original_partition(owner, runtime, ledger)
    token = manager.capture(controller)
    checkpoint, cursor = _assert_capture(
        owner, runtime, ledger, controller, manager, token, erased, untrained
    )
    before, stores, anchors = _authorities(ledger, life, runtime), _stores(history), ledger._anchors
    original_erased = cursor.erased
    original_first = _cursor_tombstone(cursor, FIRST)
    cloned = replace(original_first)
    assert cloned is not original_first and cloned == original_first
    assert checkpoint.untrained_inbox[0].tombstone() is original_first
    object.__setattr__(
        cursor, "erased", tuple(cloned if item.key == FIRST else item for item in original_erased)
    )
    try:
        with pytest.raises(ValueError):
            manager.restore(controller, token)
    finally:
        object.__setattr__(cursor, "erased", original_erased)
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert controller._pending[token].owner is runtime and ledger._anchors is anchors
    assert controller._attempts == 0 and controller._models == []
    assert checkpoint.untrained_inbox[0].tombstone() is original_first
    _assert_original_partition(owner, runtime, ledger)
    _assert_stores(history, stores)
    _assert_authorities(ledger, life, runtime, before, exact_cost=True)
    assert manager._active is None


def test_should_refuse_changed_original_untrained_content_before_capture_grant():
    owner, life, runtime, ledger, controller, manager, _ = _three_partition_setup()
    history, _, _, _ = _assert_original_partition(owner, runtime, ledger)
    before, stores = _authorities(ledger, life, runtime), _stores(history)
    tombstone = runtime._inbox._erased[FIRST]
    object.__setattr__(tombstone, "reason", "expired")
    try:
        with pytest.raises(ValueError):
            manager.capture(controller)
    finally:
        object.__setattr__(tombstone, "reason", "deleted")
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_original_partition(owner, runtime, ledger)
    _assert_stores(history, stores)
    _assert_authorities(ledger, life, runtime, before, exact_cost=True)


def test_should_refuse_final_materialized_untrained_change_after_genuine_fingerprint():
    probe: dict[str, Any] = _probe(mutate=True)
    owner, life, runtime, ledger, controller, manager, weak_payloads = _three_partition_setup(
        fingerprint=_publication_fingerprint(probe),
    )
    history, _, erased, untrained = _assert_original_partition(owner, runtime, ledger)
    token = manager.capture(controller)
    checkpoint, _ = _assert_capture(
        owner, runtime, ledger, controller, manager, token, erased, untrained
    )
    before, stores, anchors = _authorities(ledger, life, runtime), _stores(history), ledger._anchors
    first_tombstone = runtime._inbox._erased[FIRST]
    _arm(probe, manager, runtime, erased, untrained)
    try:
        with pytest.raises(
            ValueError, match="untrained inbox tombstone differs from original scalar metadata"
        ):
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
    assert runtime._inbox._erased[FIRST] is first_tombstone and first_tombstone.reason == "deleted"
    assert checkpoint.untrained_inbox[0].tombstone().reason == "deleted"
    assert FIRST not in runtime._inbox._applied
    assert all(reference() is None for reference in weak_payloads)
    _assert_original_partition(owner, runtime, ledger)
    _assert_stores(history, stores)
    _assert_authorities(ledger, life, runtime, before)
    assert ledger._copy_slots > before["slots"]
    assert ledger.accounting().records_created > before["accounting"].records_created
    assert life._copy_budget._charged > before["charged"]
    assert manager._active is None


def test_should_refuse_three_partition_capture_at_original_two_invocation_limit():
    owner, life, runtime, ledger, controller, manager, _ = _three_partition_setup(invocations=2)
    history, _, _, _ = _assert_original_partition(owner, runtime, ledger)
    before, stores = _authorities(ledger, life, runtime), _stores(history)
    assert ledger._admission.limits.max_invocations == before["accounting"].invocations_started == 2
    with pytest.raises(ValueError, match="cumulative record/invocation/metadata limit exhausted"):
        manager.capture(controller)
    _assert_no_capture_publication(owner, runtime, controller, manager)
    _assert_original_partition(owner, runtime, ledger)
    _assert_stores(history, stores)
    _assert_authorities(ledger, life, runtime, before)
    assert ledger._admission.limits is before["limits"]
    assert ledger._admission.limits.max_invocations == ledger.accounting().invocations_started == 2
    assert (
        ledger._copy_slots == before["slots"]
        and ledger._minimum_copy_slots == before["minimum_slots"]
    )
