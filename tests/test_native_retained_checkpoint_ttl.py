"""One prospective actual failed-preparation lifecycle TTL control.

Why this: declaration TTL is tested with the original controller still owning its
failed native target. No holder release, GC, replacement budget or new work is
used to obtain cleanup. History retention stays at the original default OFF.
"""

from weakref import ref

import pytest

from test_native_checkpoint_eviction import (
    _assert_monotone_allowances,
    _original_allowances,
)
from test_native_managed_replay_checkpoints import (
    FAIL_RESTORE_COPY,
    observe_native_work,
    setup,
)


KEY = ("e1", "s1")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_retained_ttl_work(tmp_path_factory):
    limits = dict(
        models=1,
        learners=4,
        forks=3,
        wakes=1,
        steps=2,
        stores=1,
        predicts=1,
        array_copies=5,
        captures=1,
        preparations=1,
        native_restores=1,
        handoffs=0,
        graph_events=32,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=1,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _weak_native_payloads(model):
    # Only weak witnesses survive this function; the native deque owns the rows.
    return tuple(
        (ref(row), ref(row.input_batch), ref(row.target_batch)) for row in model._replay_memory
    )


def _assert_original_history(owner, runtime, ledger, controller, child, history):
    assert owner._shared is history["shared"]
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox is history["inbox"]
    assert runtime._inbox._historical_completed_updates is None
    assert runtime._inbox._applied[KEY] is history["receipt"]
    assert runtime._inbox._event_ids == history["events"]
    assert runtime._attempted_ids == history["attempted"]
    assert tuple(runtime._consolidations) == history["consolidations"]
    assert ledger._anchors is history["anchors"]
    assert ledger._inbox_origins is None and ledger._inbox_history_birth is None
    assert controller._models is history["models"]
    assert controller._models == [child]
    assert controller._attempts == 1
    assert controller._pending is history["pending"]
    assert controller._build is history["builder"]
    assert runtime._candidate._model is history["canonical"]
    assert child._model is history["target"]
    assert runtime._candidate._model._epoch_count == child._model._epoch_count == 1


def test_should_refuse_retained_target_at_original_ttl_then_clean_without_refunding():
    owner, life, runtime, ledger, controller, manager = setup()
    original = _original_allowances(ledger, life, runtime)
    clock, policy = runtime._inbox._clock, life._policy
    declaration = owner._catalog[KEY]
    assert owner._declaration_ticks[KEY] == 0
    assert owner._declaration_seconds[KEY] == 0.0
    assert policy.max_retention_ticks == 120 and policy.max_retention_seconds == 1000.0
    assert runtime._inbox._experiences[KEY].observed_at == 1
    assert runtime._budget.updates_completed == 1

    token = manager.capture(controller)
    checkpoint = manager._checkpoints[id(token)]
    FAIL_RESTORE_COPY[0] = True
    try:
        with pytest.raises(RuntimeError, match="declared actual native restore-copy fault"):
            manager.restore(controller, token)
    finally:
        FAIL_RESTORE_COPY[0] = False

    child = controller._models[0]
    copy_witness = manager._copies._copies[id(child._model)]
    history = dict(
        shared=owner._shared,
        inbox=runtime._inbox,
        receipt=runtime._inbox._applied[KEY],
        events=set(runtime._inbox._event_ids),
        attempted=set(runtime._attempted_ids),
        consolidations=tuple(runtime._consolidations),
        anchors=ledger._anchors,
        models=controller._models,
        pending=controller._pending,
        builder=controller._build,
        canonical=runtime._candidate._model,
        target=child._model,
    )
    revision = runtime._revision
    native_refs = _weak_native_payloads(runtime._candidate._model) + _weak_native_payloads(
        child._model
    )
    assert len(native_refs) == 2
    inbox_refs = (
        ref(runtime._inbox._experiences[KEY]),
        ref(runtime._inbox._experiences[KEY].features),
        ref(runtime._inbox._labels[KEY]),
        ref(runtime._inbox._labels[KEY].targets),
    )
    assert token in controller._pending and checkpoint.pending() is controller._pending[token]
    _assert_original_history(owner, runtime, ledger, controller, child, history)
    _assert_monotone_allowances(ledger, life, runtime, original, exact=False)
    assert life._copy_budget._charged > original["charge"] and ledger._copy_slots > 0

    # The access at 119 is paid from the SAME admission, after the original fault.
    before_access = _original_allowances(ledger, life, runtime)
    clock.advance_to(119)
    copied_origins = manager._copies.origins(controller, child)
    assert len(copied_origins) == 1 and copied_origins[0].key == KEY
    assert copied_origins[0] is copy_witness.rows[0].data
    paid = _original_allowances(ledger, life, runtime)
    assert paid["accounting"].invocations_started == (
        before_access["accounting"].invocations_started + 1
    )
    assert paid["accounting"].metadata_bytes_charged > (
        before_access["accounting"].metadata_bytes_charged
    )
    assert paid["charge"] == before_access["charge"]
    assert paid["slots"] == before_access["slots"]

    # At 120 lifecycle TTL has elapsed, while source age is only 119 of 120.
    clock.advance_to(120)
    assert clock.now() - copied_origins[0].observed_at < ledger._admission.limits.max_age_ticks
    with pytest.raises(ValueError, match="declared payload retention has expired"):
        manager._copies.origins(controller, child)
    with pytest.raises(ValueError, match="declared payload retention has expired"):
        manager.restore(controller, token)
    _assert_original_history(owner, runtime, ledger, controller, child, history)
    _assert_monotone_allowances(ledger, life, runtime, paid, exact=True)
    assert runtime._revision == revision and token in controller._pending
    assert manager._copies._copies[id(child._model)] is copy_witness
    assert all(reference() is not None for group in native_refs for reference in group)
    assert all(reference() is not None for reference in inbox_refs)

    report = life.expire()
    assert report.reason == "expired" and report.requested_keys == report.revoked_keys == (KEY,)
    assert report.model_snapshots_erased == 2 and report.inboxes_cleared == 1
    assert report.checkpoints_invalidated == 1 and report.promotions_invalidated == 0
    _assert_original_history(owner, runtime, ledger, controller, child, history)
    _assert_monotone_allowances(ledger, life, runtime, paid, exact=True)
    assert runtime._revision == revision + 1
    assert owner._catalog[KEY] is declaration and owner._revoked_keys == {KEY}
    assert not owner._opted_out and life._policy is policy
    assert owner._declaration_ticks[KEY] == 0 and owner._declaration_seconds[KEY] == 0.0
    assert life._last_seconds == 0.0 and not life._failed and not life._retention_fault
    assert not controller._pending
    assert checkpoint.pending() is None and checkpoint.view() is None
    assert checkpoint.state() is None and checkpoint.cursor() is None
    assert manager._copies._copies[id(child._model)] is copy_witness
    assert not runtime._candidate._model._replay_memory and not child._model._replay_memory
    assert runtime._candidate._model.replay_payload_footprint().payload_bytes == 0
    assert child._model.replay_payload_footprint().payload_bytes == 0
    assert not runtime._inbox._experiences and not runtime._inbox._labels
    tombstone = runtime._inbox._erased[KEY]
    assert (tombstone.reason, tombstone.erased_at) == ("expired", 120)
    assert (tombstone.observed_at, tombstone.arrived_at, tombstone.event_id) == (
        1,
        3,
        "label-s1",
    )
    assert all(reference() is None for group in native_refs for reference in group)
    assert all(reference() is None for reference in inbox_refs)
    footprint = life.payload_byte_snapshot()
    assert footprint.observed_retained_bytes == 0 and footprint.charged_bytes == paid["charge"]

    # Cleared originals cannot prepare another target or publish the old token.
    with pytest.raises(ValueError, match="nonempty original row origins"):
        manager.restore(controller, token)
    _assert_original_history(owner, runtime, ledger, controller, child, history)
    _assert_monotone_allowances(ledger, life, runtime, paid, exact=True)
    assert not controller._pending and runtime._revision == revision + 1
