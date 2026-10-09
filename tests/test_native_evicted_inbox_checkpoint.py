"""N6: original admitted inbox history carried across actual native eviction."""

from dataclasses import replace
import inspect
from typing import Any

import pytest

from src.adapters.numpy_replay_origins import replay_payload_fingerprint
from test_native_checkpoint_eviction import _queue_original_distinct_sample
from test_native_managed_replay_checkpoints import observe_native_work, setup


FIRST = ("e1", "s1")
SECOND = ("e1", "s2")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_inbox_history_work(tmp_path_factory):
    limits = dict(
        models=6,
        learners=24,
        forks=18,
        wakes=12,
        steps=24,
        stores=12,
        predicts=12,
        array_copies=60,
        captures=6,
        preparations=5,
        native_restores=5,
        handoffs=1,
        graph_events=128,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=0,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _evicted_setup(*, retain_inbox_origins=True, live_records=32, fingerprint=None):
    owner, life, runtime, ledger, controller, manager = setup(
        replay_examples=1,
        retain_inbox_origins=retain_inbox_origins,
        live_records=live_records,
        fingerprint=fingerprint,
    )
    first_receipt = runtime._inbox._applied[FIRST]
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 2
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 2
    assert runtime._inbox._applied[FIRST] is first_receipt
    assert runtime._inbox._applied[SECOND] is poll.updates[0]
    assert len(runtime._candidate._model._replay_memory) == 1
    assert len(ledger.origins()) == 1 and ledger.origins()[0].key == SECOND
    assert runtime._candidate._model._replay_retention_budget.max_examples == 1
    assert ledger._admission.limits.max_live_records == live_records
    return owner, life, runtime, ledger, controller, manager


def _original_authorities(ledger, life, runtime):
    raw_budget = life._copy_budget
    assert raw_budget is not None
    return dict(
        admission=ledger._admission,
        limits=ledger._admission.limits,
        raw_budget=raw_budget,
        raw_limits=raw_budget._limits,
        work_budget=runtime._budget,
        ingress=life._admitted_bytes,
        charge=raw_budget._charged,
        slots=ledger._copy_slots,
        minimum_slots=ledger._minimum_copy_slots,
        accounting=ledger.accounting(),
        history=ledger._inbox_origins,
    )


def _assert_original_monotone(ledger, life, runtime, original):
    assert ledger._admission is original["admission"]
    assert ledger._admission.limits is original["limits"]
    assert life._copy_budget is original["raw_budget"]
    assert life._copy_budget._limits is original["raw_limits"]
    assert life._policy.owned_payload_copies is original["raw_limits"]
    assert life._copy_budget._limits.max_lifetime_owned_bytes == 4096
    assert runtime._budget is original["work_budget"]
    assert runtime._budget.updates_completed == 2
    assert life._admitted_bytes == original["ingress"]
    assert life._copy_budget._charged >= original["charge"]
    assert ledger._copy_slots >= original["slots"]
    assert ledger._minimum_copy_slots >= original["minimum_slots"]
    assert ledger._inbox_origins is original["history"]
    current, before = ledger.accounting(), original["accounting"]
    assert current.records_created >= before.records_created
    assert current.invocations_started >= before.invocations_started
    assert current.metadata_bytes_charged >= before.metadata_bytes_charged
    if original["history"] is not None:
        assert current.live_records >= ledger._copy_slots + 2
    else:
        assert current.live_records >= ledger._copy_slots


def _history_pairs(ledger, runtime):
    history = ledger._history()
    assert history is ledger._inbox_origins and history is not None
    assert history._storage[0] is history._records and history._storage[1] is history._sealed
    records = {record.data.key: record for record in history._records.values()}
    assert set(records) == {FIRST, SECOND}
    assert len(history._records) == len(history._sealed) == 2
    for key, record in records.items():
        source, label, receipt = (
            runtime._inbox._experiences[key],
            runtime._inbox._labels[key],
            runtime._inbox._applied[key],
        )
        assert record.verified and record.receipt is not None
        assert record.source() is source and record.label() is label
        assert record.receipt() is receipt
        assert record.features() is source.features and record.targets() is label.targets
        assert history._sealed[id(source)] is record
        assert record.data.key == key and record.data.event_id == receipt.event_id
        assert record.data.update_number == receipt.update_number
    return records


def _assert_unpublished(owner, runtime, ledger, controller, token, anchors):
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert controller._pending[token].owner is runtime and ledger._anchors is anchors
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 2


def test_should_capture_restore_and_handoff_two_original_pairs_after_native_eviction():
    owner, life, runtime, ledger, controller, manager = _evicted_setup()
    original = _original_authorities(ledger, life, runtime)
    records = _history_pairs(ledger, runtime)
    sources = dict(runtime._inbox._experiences)
    labels = dict(runtime._inbox._labels)
    receipts = dict(runtime._inbox._applied)
    token = manager.capture(controller)
    view = controller._pending[token].view
    witness = manager._checkpoints[id(token)]
    assert witness.cursor() is view.inbox and len(witness.inbox) == 2
    for pair in witness.inbox:
        assert any(pair.source() is source for source in view.inbox.experiences)
        assert any(pair.label() is label for label in view.inbox.labels)
        assert any(pair.receipt() is receipt for receipt in view.inbox.applied)
        assert pair.features() is pair.source().features
        assert pair.targets() is pair.label().targets
    captured_cost = ledger.accounting()
    prepared = manager.restore(controller, token)
    assert owner._shared._runtime is prepared and runtime._retired
    assert controller._pending == {} and controller._attempts == 1
    assert prepared._candidate is controller._models[0]
    assert prepared._budget is runtime._budget
    assert set(prepared._inbox._experiences) == set(prepared._inbox._labels) == {FIRST, SECOND}
    assert set(prepared._inbox._applied) == {FIRST, SECOND}
    rebound = _history_pairs(ledger, prepared)
    for key, record in rebound.items():
        assert record is not records[key] and record.data is records[key].data
        source, label, receipt = record.source(), record.label(), record.receipt()
        assert (
            source is not sources[key] and label is not labels[key] and receipt is not receipts[key]
        )
        assert source.features is not sources[key].features
        assert label.targets is not labels[key].targets
        assert source.features.tolist() == sources[key].features.tolist()
        assert label.targets.tolist() == labels[key].targets.tolist()
        assert receipt.update_number == receipts[key].update_number
        assert receipt.diagnostic == receipts[key].diagnostic
    row = prepared._candidate._model._replay_memory[0]
    canonical = ledger._rows[id(row)]
    assert canonical.source() is prepared._inbox._experiences[SECOND]
    assert canonical.label() is prepared._inbox._labels[SECOND]
    assert canonical.receipt() is prepared._inbox._applied[SECOND]
    assert ledger.origins()[0].key == SECOND
    assert ledger.accounting().records_created > captured_cost.records_created
    _assert_original_monotone(ledger, life, prepared, original)


def test_should_refuse_evicted_pair_capture_without_original_birth_enrollment():
    owner, life, runtime, ledger, controller, manager = _evicted_setup(retain_inbox_origins=False)
    original = _original_authorities(ledger, life, runtime)
    assert ledger._inbox_origins is None
    with pytest.raises(ValueError):
        manager.capture(controller)
    assert ledger._inbox_origins is None and controller._pending == {}
    assert controller._attempts == 0 and controller._models == []
    assert owner._shared._runtime is runtime and not runtime._retired
    _assert_original_monotone(ledger, life, runtime, original)


def test_should_preserve_original_sixteen_live_exhaustion_without_new_allowance():
    owner, life, runtime, ledger, controller, manager = _evicted_setup(live_records=16)
    original = _original_authorities(ledger, life, runtime)
    token = manager.capture(controller)
    anchors = ledger._anchors
    with pytest.raises(ValueError, match="limit|allowance|exhaust"):
        manager.restore(controller, token)
    _assert_unpublished(owner, runtime, ledger, controller, token, anchors)
    assert ledger._admission.limits.max_live_records == 16
    assert life._copy_budget._charged > original["charge"] and ledger._copy_slots > 0
    _assert_original_monotone(ledger, life, runtime, original)


@pytest.mark.parametrize("mutation", ["evicted_payload", "evicted_consent"])
def test_should_refuse_last_opaque_evicted_pair_change_before_publication(mutation):
    armed, fired, visits = [False], [False], [0]
    current_frame: list[Any] = [None]

    def fingerprint(snapshot, maximum):
        result = replay_payload_fingerprint(snapshot, maximum)
        if not armed[0] or fired[0]:
            return result
        frame = inspect.currentframe()
        publication = active_current = None
        while frame is not None:
            if frame.f_code.co_name == "_current":
                active_current = frame
            elif frame.f_code.co_name == "_publication_lease":
                publication = frame
                break
            frame = frame.f_back
        if publication is None or active_current is None:
            return result
        if active_current is not current_frame[0]:
            visits[0] += 1
            current_frame[0] = active_current
        if visits[0] == 3:
            fired[0] = True
            if mutation == "evicted_payload":
                runtime._inbox._experiences[FIRST].features[0, 0] += 0.25
            else:
                owner._revoked_keys.add(FIRST)
        return result

    owner, life, runtime, ledger, controller, manager = _evicted_setup(fingerprint=fingerprint)
    original = _original_authorities(ledger, life, runtime)
    token = manager.capture(controller)
    anchors = ledger._anchors
    before_value = runtime._inbox._experiences[FIRST].features[0, 0]
    assert runtime._inbox._experiences[FIRST].features.flags.writeable
    armed[0] = True
    try:
        with pytest.raises(ValueError):
            manager.restore(controller, token)
    finally:
        armed[0] = False
        current_frame[0] = None
    assert fired[0] and visits[0] == 3
    if mutation == "evicted_payload":
        assert runtime._inbox._experiences[FIRST].features[0, 0] == before_value + 0.25
    else:
        assert FIRST in owner._revoked_keys
    _assert_unpublished(owner, runtime, ledger, controller, token, anchors)
    assert controller._attempts == 1 and len(controller._models) == 1
    assert life._copy_budget._charged > original["charge"]
    _assert_original_monotone(ledger, life, runtime, original)


def test_should_refuse_equal_valued_historical_receipt_replacement_after_capture():
    owner, life, runtime, ledger, controller, manager = _evicted_setup()
    original = _original_authorities(ledger, life, runtime)
    records = _history_pairs(ledger, runtime)
    token = manager.capture(controller)
    anchors = ledger._anchors
    receipt = runtime._inbox._applied[FIRST]
    runtime._inbox._applied[FIRST] = replace(receipt)
    assert runtime._inbox._applied[FIRST] == receipt
    assert runtime._inbox._applied[FIRST] is not receipt
    assert records[FIRST].receipt() is receipt
    with pytest.raises(ValueError):
        manager.restore(controller, token)
    _assert_unpublished(owner, runtime, ledger, controller, token, anchors)
    assert controller._attempts == 0 and controller._models == []
    _assert_original_monotone(ledger, life, runtime, original)
