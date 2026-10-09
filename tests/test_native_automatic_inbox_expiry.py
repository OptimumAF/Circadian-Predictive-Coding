"""Three actual original retained-target automatic expiry boundaries.

Why this: history is enabled at original birth and ordinary direct expiry must
clean the canonical and failed preparation targets before publishing lineage.
No holder release, GC, fresh allowance or additional update obtains cleanup.
"""

from weakref import ref

import numpy as np
import pytest

import src.app.expiry_inbox_observation as observation_module
from src.app.expiry_inbox_observation import ExpiryObservationError
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration
from src.core.data_retention import DataCleanupReport
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.inbox_origin import inbox_origin_metadata
from src.core.untrained_inbox_origin import untrained_inbox_origin_metadata
from test_native_checkpoint_eviction import _assert_monotone_allowances, _original_allowances
from test_native_managed_replay_checkpoints import FAIL_RESTORE_COPY, observe_native_work, setup
from test_native_retained_checkpoint_ttl import _weak_native_payloads


FIRST = ("e1", "s1")
ARRIVALS = (("e1", "s2"), ("e1", "s3"), ("e1", "s4"))


@pytest.fixture(scope="module", autouse=True)
def bounded_native_automatic_expiry_work(tmp_path_factory):
    limits = dict(
        models=3,
        learners=12,
        forks=9,
        wakes=3,
        steps=6,
        stores=3,
        predicts=3,
        array_copies=15,
        captures=3,
        preparations=3,
        native_restores=3,
        handoffs=0,
        graph_events=64,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=3,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _queue_arrivals(owner, clock):
    assert clock._time == 3 and len(owner._catalog) == 1
    for sample, source, label in (("s2", True, False), ("s3", False, True), ("s4", True, True)):
        owner.declare(
            LifecycleDeclaration(
                ("e1", sample),
                DataProvenance("local", "person-" + sample, True, False),
                DataConsent(True, True),
                "replay",
            )
        )
        if source:
            owner.record_experience(
                Experience(
                    sample,
                    "e1",
                    3,
                    "actor-0",
                    np.array([[0.0, 1.0, 0.0]]),
                    "train",
                    ExperiencePermissions(True, True),
                )
            )
        if label:
            owner.record_label(
                LabelArrival(
                    "label-" + sample,
                    sample,
                    "e1",
                    3,
                    "actor-0",
                    np.array([[1.0]]),
                )
            )
    assert all(owner._declaration_ticks[key] == 3 for key in ARRIVALS)


def _retained_original(*, live_records=16):
    owner, life, runtime, ledger, controller, manager = setup(
        replay_examples=1, retain_inbox_origins=True, live_records=live_records
    )
    history = ledger._inbox_origins
    assert history is not None and history is ledger._inbox_history_birth()
    assert life._expiry_history_on is True
    assert life._expiry_history_birth is ledger._inbox_history_birth
    assert life._expiry_authority is ledger._expiry_authority_birth
    token = manager.capture(controller)
    checkpoint = manager._checkpoints[id(token)]
    FAIL_RESTORE_COPY[0] = True
    try:
        with pytest.raises(RuntimeError, match="declared actual native restore-copy fault"):
            manager.restore(controller, token)
    finally:
        FAIL_RESTORE_COPY[0] = False
    child = controller._models[0]
    assert controller._models == [child] and controller._attempts == 1
    assert not runtime._retired and owner._shared._runtime is runtime
    assert runtime._candidate._model._epoch_count == child._model._epoch_count == 1
    assert runtime._candidate._model._replay_retention_budget.max_examples == 1
    assert runtime._candidate._model._replay_retention_budget.max_bytes == 64
    copy = manager._copies._copies[id(child._model)]
    assert (
        copy.model() is child._model and copy.rows[0].source() is runtime._inbox._experiences[FIRST]
    )
    _queue_arrivals(owner, runtime._inbox._clock)
    assert runtime._budget.updates_completed == 1
    assert set(runtime._inbox._applied) == {FIRST}
    return (
        owner,
        life,
        runtime,
        ledger,
        controller,
        manager,
        history,
        token,
        checkpoint,
        child,
        copy,
    )


def _weak_owned_payloads(runtime, child):
    native = _weak_native_payloads(runtime._candidate._model) + _weak_native_payloads(child._model)
    inbox = tuple(
        ref(value)
        for mapping, field in (
            (runtime._inbox._experiences, "features"),
            (runtime._inbox._labels, "targets"),
        )
        for record in mapping.values()
        for value in (record, getattr(record, field))
    )
    assert len(native) == 2 and len(inbox) == 12
    return native, inbox


def _assert_raw_cleanup(
    owner,
    life,
    runtime,
    ledger,
    controller,
    manager,
    token,
    checkpoint,
    child,
    copy,
    report,
    paid,
    native,
    inbox,
):
    assert observation_module._CURRENT.get() is None
    assert type(report) is DataCleanupReport and report.reason == "expired"
    assert report.requested_keys == report.revoked_keys == (FIRST, *ARRIVALS)
    assert report.model_snapshots_erased == 2 and report.inboxes_cleared == 1
    assert report.checkpoints_invalidated == 1 and report.promotions_invalidated == 0
    assert owner._shared._runtime is runtime and not runtime._retired and not runtime._stopped
    assert not life._failed and not life._retention_fault
    assert owner._revoked_keys == {FIRST, *ARRIVALS} and not owner._opted_out
    assert not runtime._inbox._experiences and not runtime._inbox._labels
    assert set(runtime._inbox._applied) == {FIRST} and set(runtime._inbox._erased) == {
        FIRST,
        *ARRIVALS,
    }
    assert all(
        value.reason == "expired" and value.erased_at == 123
        for value in runtime._inbox._erased.values()
    )
    assert not runtime._candidate._model._replay_memory and not child._model._replay_memory
    assert all(reference() is None for group in native for reference in group)
    assert all(reference() is None for reference in inbox)
    assert controller._models == [child] and controller._attempts == 1 and not controller._pending
    assert (
        checkpoint.pending()
        is checkpoint.view()
        is checkpoint.state()
        is checkpoint.cursor()
        is None
    )
    assert manager._copies._copies[id(child._model)] is copy
    assert token not in controller._pending
    _assert_monotone_allowances(ledger, life, runtime, paid, exact=False)
    assert life._copy_budget._charged == paid["charge"]
    assert life._admitted_bytes == paid["ingress"] and ledger._copy_slots == paid["slots"]
    assert ledger._minimum_copy_slots == paid["minimum_slots"]
    assert life._registry._total == paid["enrollments"] and runtime._budget.updates_completed == 1


def test_should_publish_all_four_original_histories_after_actual_retained_native_expiry():
    owner, life, runtime, ledger, controller, manager, history, token, checkpoint, child, copy = (
        # Permanent copy reservations need headroom for four prepaid histories.
        # The conservative source bound is declared before constructing this graph.
        _retained_original(live_records=23)
    )
    receipt = runtime._inbox._applied[FIRST]
    data = next(iter(history._records.values())).data
    maps = history._storage, history._erased_storage, history._untrained_storage
    paid = _original_allowances(ledger, life, runtime, expected_live_records=23)
    native, inbox = _weak_owned_payloads(runtime, child)
    runtime._inbox._clock.advance_to(123)
    with pytest.raises(ValueError, match="expired"):
        owner._require_live(FIRST)
    report = life.expire()
    _assert_raw_cleanup(
        owner,
        life,
        runtime,
        ledger,
        controller,
        manager,
        token,
        checkpoint,
        child,
        copy,
        report,
        paid,
        native,
        inbox,
    )
    assert ledger._inbox_origins is history and history is ledger._inbox_history_birth()
    assert not history._records and not history._sealed
    erased = history._erased_records[FIRST]
    assert erased.data is data and erased.receipt() is receipt
    assert erased.tombstone() is runtime._inbox._erased[FIRST]
    assert history._erased_sealed[FIRST] is erased
    assert set(history._untrained_records) == set(history._untrained_sealed) == set(ARRIVALS)
    for key, record in history._untrained_records.items():
        assert record is history._untrained_sealed[key]
        assert record.tombstone() is runtime._inbox._erased[key]
        assert record.data.key == key and record.data.completed_updates == 1
        assert not hasattr(record, "receipt") and not hasattr(record, "source")
    assert history._storage is not maps[0] and history._erased_storage is not maps[1]
    assert history._untrained_storage is not maps[2]
    charge = len(inbox_origin_metadata(data, 128 * 1024)) + 1024
    charge += sum(
        len(untrained_inbox_origin_metadata(record.data, 128 * 1024)) + 1024
        for record in history._untrained_records.values()
    )
    assert ledger._admission._progress.records_created == paid["accounting"].records_created + 4
    assert (
        ledger._admission._progress.metadata_bytes_charged
        == paid["accounting"].metadata_bytes_charged + charge
    )
    assert ledger._admission._progress.invocations_started == paid["accounting"].invocations_started
    assert history.verify(ledger, owner, runtime, 123) == ()


def test_should_clean_actual_retained_native_targets_when_original_history_bridge_is_missing():
    owner, life, runtime, ledger, controller, manager, history, token, checkpoint, child, copy = (
        _retained_original()
    )
    original_maps = history._storage, history._erased_storage, history._untrained_storage
    paid = _original_allowances(ledger, life, runtime)
    native, inbox = _weak_owned_payloads(runtime, child)
    runtime._inbox._clock.advance_to(123)
    life._expiry_history_birth = None
    with pytest.raises(ExpiryObservationError) as failure:
        life.expire()
    error = failure.value
    assert (
        type(error) is ExpiryObservationError and type(error.code) is str and len(error.code) <= 64
    )
    assert error.__cause__ is None and error.__context__ is None
    _assert_raw_cleanup(
        owner,
        life,
        runtime,
        ledger,
        controller,
        manager,
        token,
        checkpoint,
        child,
        copy,
        error.report,
        paid,
        native,
        inbox,
    )
    assert history._storage is original_maps[0] and history._erased_storage is original_maps[1]
    assert history._untrained_storage is original_maps[2]
    assert {record.data.key for record in history._records.values()} == {FIRST}
    assert not history._erased_records and not history._untrained_records
    assert ledger._admission._progress.records_created == paid["accounting"].records_created
    assert (
        ledger._admission._progress.metadata_bytes_charged
        == paid["accounting"].metadata_bytes_charged
    )
    assert ledger._admission._progress.invocations_started == paid["accounting"].invocations_started
    assert not ledger._poisoned
    with pytest.raises(ValueError):
        history.verify(ledger, owner, runtime, 123)


def test_should_clean_original_native_targets_without_publication_at_birth_record_ceiling():
    owner, life, runtime, ledger, controller, manager, history, token, checkpoint, child, copy = (
        _retained_original()
    )
    original_maps = history._storage, history._erased_storage, history._untrained_storage
    identity, original_record = next(iter(history._records.items()))
    assert original_record.data.key == FIRST and history._sealed[identity] is original_record
    paid = _original_allowances(ledger, life, runtime)
    native, inbox = _weak_owned_payloads(runtime, child)
    runtime._inbox._clock.advance_to(123)

    with pytest.raises(ExpiryObservationError) as failure:
        life.expire()

    error = failure.value
    assert (
        type(error) is ExpiryObservationError and type(error.code) is str and len(error.code) <= 64
    )
    assert error.__cause__ is None and error.__context__ is None
    _assert_raw_cleanup(
        owner,
        life,
        runtime,
        ledger,
        controller,
        manager,
        token,
        checkpoint,
        child,
        copy,
        error.report,
        paid,
        native,
        inbox,
    )
    assert history._storage is original_maps[0] and history._erased_storage is original_maps[1]
    assert history._untrained_storage is original_maps[2]
    assert history._records == {identity: original_record}
    assert history._sealed[identity] is original_record
    assert not history._erased_records and not history._untrained_records
    # Successful prepayments remain spent even though no persistent maps publish.
    progress, before = ledger._admission._progress, paid["accounting"]
    assert 0 < progress.records_created - before.records_created < 4
    assert progress.metadata_bytes_charged > before.metadata_bytes_charged
    assert progress.invocations_started == before.invocations_started
    assert not ledger._poisoned
    with pytest.raises(ValueError):
        history.verify(ledger, owner, runtime, 123)
