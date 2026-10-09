"""N9: trusted original deletion records payload-free erased inbox lineage.

Why this: receipts preserve cumulative work after payload deletion. Only an
observed original commit may move a verified live witness into erased history;
matching retained receipts and tombstones cannot grant retrospective lineage.
"""

from contextvars import copy_context
from dataclasses import fields, replace
from math import isfinite
from typing import Any
from weakref import ReferenceType, ref

import pytest

from src.adapters import numpy_learners as learner_adapter
from src.app import inbox_erasure_observation
from src.app.managed_inbox_origins import ManagedInboxOrigins
from src.app.erased_inbox_origins import ErasedInboxOrigin, erased_inbox_origin_stamp
from src.core.data_retention import DataCleanupReport
from src.core.inbox_origin import inbox_origin_metadata
from src.core.replay_origin import RECORD_OVERHEAD_BYTES
from test_native_checkpoint_eviction import _queue_original_distinct_sample
from test_native_managed_replay_checkpoints import observe_native_work, setup


FIRST = ("e1", "s1")
SECOND = ("e1", "s2")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_erased_history_work(tmp_path_factory):
    limits = dict(
        models=2,
        learners=6,
        forks=4,
        wakes=4,
        steps=8,
        stores=4,
        predicts=4,
        array_copies=20,
        captures=0,
        preparations=0,
        native_restores=0,
        handoffs=0,
        graph_events=0,
        source_array_bytes=0,
        cleanup_attempts=2,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _allowances(ledger, life, runtime, history):
    limits = ledger._admission.limits
    assert tuple(vars(limits).values()) == (32, 64, 32, 128 * 1024, 120, 4096)
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
        storage=history._storage,
        erased_storage=history._erased_storage,
        accounting=ledger.accounting(),
        raw_charge=life._copy_budget._charged,
        ingress=life._admitted_bytes,
        slots=ledger._copy_slots,
        minimum_slots=ledger._minimum_copy_slots,
        enrollments=life._registry._total,
        work=runtime._budget.updates_completed,
    )


def _assert_authority(ledger, life, runtime, history, before, *, storage_transition=False):
    assert ledger._admission is before["admission"]
    assert ledger._admission.limits is before["limits"]
    assert life._copy_budget is before["raw"]
    assert life._copy_budget._limits is before["raw_limits"]
    assert life._policy is before["policy"]
    assert life._policy.owned_payload_copies is before["raw_limits"]
    assert life._registry is before["registry"]
    assert runtime._budget is before["budget"]
    assert runtime._budget.budget is before["budget_policy"]
    assert runtime._candidate._model._replay_retention_budget is before["retention"]
    assert runtime._payload_lineage is before["lineage"] and ledger._anchors is before["anchors"]
    assert ledger._history() is history is before["history"]
    for current, original in (
        (history._storage, before["storage"]),
        (history._erased_storage, before["erased_storage"]),
    ):
        assert type(current) is tuple and len(current) == 2
        assert all(type(mapping) is dict for mapping in current)
        assert current[0] is not current[1]
        if storage_transition:
            assert current is not original
            assert all(mapping is not previous for mapping in current for previous in original)
        else:
            assert current is original
    assert history._storage[0] is history._records and history._storage[1] is history._sealed
    assert history._erased_storage[0] is history._erased_records
    assert history._erased_storage[1] is history._erased_sealed
    assert ledger._copy_slots == before["slots"]
    assert ledger._minimum_copy_slots == before["minimum_slots"]
    assert life._registry._total == before["enrollments"]
    current, previous = ledger.accounting(), before["accounting"]
    assert current.records_created >= previous.records_created
    assert current.invocations_started >= previous.invocations_started
    assert current.metadata_bytes_charged >= previous.metadata_bytes_charged
    assert life._copy_budget._charged >= before["raw_charge"]
    assert life._admitted_bytes >= before["ingress"]
    assert runtime._budget.updates_completed >= before["work"]


def _weak_payloads(runtime):
    inbox, model = runtime._inbox, runtime._candidate._model
    return dict(
        native_row=ref(model._replay_memory[0]),
        native_features=ref(model._replay_memory[0].input_batch),
        native_targets=ref(model._replay_memory[0].target_batch),
        source=ref(inbox._experiences[FIRST]),
        source_features=ref(inbox._experiences[FIRST].features),
        label=ref(inbox._labels[FIRST]),
        label_targets=ref(inbox._labels[FIRST].targets),
    )


def _assert_actual_erasure(owner, runtime, controller, report, receipt, weak_payloads):
    assert type(report) is DataCleanupReport
    assert report.reason == "deleted" and report.requested_keys == (FIRST,)
    assert report.revoked_keys == (FIRST,)
    assert report.model_snapshots_erased == 1 and report.inboxes_cleared == 1
    assert report.checkpoints_invalidated == report.promotions_invalidated == 0
    assert FIRST in owner._revoked_keys
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 1
    assert runtime._inbox._experiences == {} and runtime._inbox._labels == {}
    assert runtime._inbox._applied[FIRST] is receipt
    assert type(receipt.diagnostic.value) in (int, float) and isfinite(receipt.diagnostic.value)
    assert set(runtime._inbox._erased) == {FIRST}
    assert runtime._inbox._erased[FIRST].reason == "deleted"
    assert not runtime._candidate._model._replay_memory
    assert all(reference() is None for reference in weak_payloads.values())
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0


def _verify_live_partition(history, ledger, owner, runtime):
    live = ManagedInboxOrigins.verify(history, ledger, owner, runtime, ledger._read_tick(runtime))
    assert len(live) == 1 and live[0].data.key == SECOND
    assert live[0].source() is runtime._inbox._experiences[SECOND]
    assert live[0].label() is runtime._inbox._labels[SECOND]
    assert live[0].receipt is not None
    assert live[0].receipt() is runtime._inbox._applied[SECOND]
    return ManagedInboxOrigins.state_stamp(history)


def _assert_erased_identity_and_contents_refuse(
    history, ledger, owner, runtime, receipt, tombstone
):
    stamp = _verify_live_partition(history, ledger, owner, runtime)
    for mapping, original in (
        (runtime._inbox._applied, receipt),
        (runtime._inbox._erased, tombstone),
    ):
        mapping[FIRST] = replace(original)
        try:
            with pytest.raises(ValueError):
                _verify_live_partition(history, ledger, owner, runtime)
        finally:
            mapping[FIRST] = original
        assert _verify_live_partition(history, ledger, owner, runtime) == stamp
    for original, field in ((receipt, "applied_at"), (tombstone, "erased_at")):
        value = getattr(original, field)
        object.__setattr__(original, field, value + 1)
        try:
            with pytest.raises(ValueError):
                _verify_live_partition(history, ledger, owner, runtime)
            with pytest.raises(ValueError):
                ManagedInboxOrigins.state_stamp(history)
        finally:
            object.__setattr__(original, field, value)
        assert _verify_live_partition(history, ledger, owner, runtime) == stamp


def test_should_observe_actual_original_erasure_then_preserve_distinct_live_training(monkeypatch):
    original_erase = learner_adapter._managed_erase
    scopes: list[Any] = []

    def erase(model):
        if model is runtime._candidate:
            scope = inbox_erasure_observation._CURRENT.get()
            assert scope is not None and not scopes
            assert not scope.closed and not scope.committed and not scope.used
            names = ("ledger", "owner", "runtime", "inbox", "history", "life", "before")
            borrowed = tuple(getattr(scope, name) for name in names)
            assert scope.ledger is ledger and scope.owner is owner and scope.runtime is runtime
            assert scope.inbox is runtime._inbox and scope.history is history and scope.life is life
            context_token, access_token = scope.context, scope.access
            assert context_token is not None and access_token is not None
            with pytest.raises(ValueError, match="original context"):
                copy_context().run(scope.require)
            with pytest.raises(ValueError, match="original ledger deletion"):
                copy_context().run(scope.__exit__, None, None, None)
            assert inbox_erasure_observation._CURRENT.get() is scope
            assert not scope.closed and not scope.committed and not scope.used
            assert all(getattr(scope, name) is value for name, value in zip(names, borrowed))
            assert scope.context is context_token and scope.access is access_token
            scope.require()
            scopes.append(scope)
        # Why: composition at original birth observes the genuine native erase;
        # each original actor/candidate call delegates exactly once unchanged.
        return original_erase(model)

    monkeypatch.setattr(learner_adapter, "_managed_erase", erase)
    owner, life, runtime, ledger, controller, _ = setup(
        replay_examples=1, retain_inbox_origins=True, live_records=32
    )
    history = ledger._history()
    assert history is not None and len(history._records) == 1
    data = next(iter(history._records.values())).data
    receipt = runtime._inbox._applied[FIRST]
    diagnostic = receipt.diagnostic
    diagnostic_value = diagnostic.value
    weak_payloads = _weak_payloads(runtime)
    before = _allowances(ledger, life, runtime, history)
    remaining = (
        before["limits"].max_metadata_bytes
        - before["accounting"].metadata_bytes_charged
        - RECORD_OVERHEAD_BYTES
    )
    expected_metadata = len(inbox_origin_metadata(data, remaining))
    report = ledger.delete((FIRST,))
    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.used and scope.committed and scope.closed and not scope.inflight
    assert all(
        getattr(scope, name) is None
        for name in (
            "ledger",
            "owner",
            "runtime",
            "inbox",
            "history",
            "life",
            "before",
            "context",
            "access",
        )
    )
    assert inbox_erasure_observation._CURRENT.get() is None
    _assert_actual_erasure(owner, runtime, controller, report, receipt, weak_payloads)
    _assert_authority(ledger, life, runtime, history, before, storage_transition=True)
    # The trusted commit publishes prebuilt maps. Later checks use those exact
    # maps while all original admission, budget and charge expectations persist.
    before["storage"] = history._storage
    before["erased_storage"] = history._erased_storage
    assert receipt.diagnostic is diagnostic and diagnostic.value == diagnostic_value
    assert history._records == history._sealed == {}
    assert set(history._erased_records) == set(history._erased_sealed) == {FIRST}
    witness = history._erased_records[FIRST]
    assert type(witness) is ErasedInboxOrigin
    assert history._erased_sealed[FIRST] is witness
    assert {field.name for field in fields(witness)} == {
        "data",
        "receipt",
        "tombstone",
        "receipt_stamp",
        "tombstone_stamp",
    }
    assert not hasattr(witness, "__dict__")
    assert witness.data is data
    assert type(witness.receipt) is ReferenceType and type(witness.tombstone) is ReferenceType
    tombstone = runtime._inbox._erased[FIRST]
    assert witness.receipt() is receipt and witness.tombstone() is tombstone
    assert type(erased_inbox_origin_stamp(witness, history._content_limits)) is tuple
    assert all(
        not hasattr(witness, name)
        for name in ("source", "label", "declaration", "features", "targets")
    )
    after = ledger.accounting()
    assert after.records_created == before["accounting"].records_created + 1
    assert after.invocations_started == before["accounting"].invocations_started
    assert (
        after.metadata_bytes_charged
        == before["accounting"].metadata_bytes_charged + expected_metadata + RECORD_OVERHEAD_BYTES
    )
    assert life._copy_budget._charged == before["raw_charge"]
    assert life._admitted_bytes == before["ingress"]
    assert (
        ManagedInboxOrigins.verify(history, ledger, owner, runtime, ledger._read_tick(runtime))
        == ()
    )
    erased_stamp = ManagedInboxOrigins.state_stamp(history)
    assert ledger.origins() == ()
    assert ledger.accounting().live_records == 1 + ledger._copy_slots
    erased = _allowances(ledger, life, runtime, history)
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    assert life._admitted_bytes == erased["ingress"] + 32
    assert life._copy_budget._charged == erased["raw_charge"] + 32
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 2
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 2
    assert life._copy_budget._charged == erased["raw_charge"] + 64
    assert runtime._inbox._applied[FIRST] is receipt
    assert runtime._inbox._applied[SECOND] is poll.updates[0]
    assert runtime._inbox._erased[FIRST] is tombstone
    assert history._erased_records[FIRST] is history._erased_sealed[FIRST] is witness
    assert all(reference() is None for reference in weak_payloads.values())
    assert len(runtime._candidate._model._replay_memory) == 1
    assert ledger.origins()[0].key == SECOND and SECOND not in owner._revoked_keys
    assert _verify_live_partition(history, ledger, owner, runtime) != erased_stamp
    assert ledger.accounting().live_records == 3 + ledger._copy_slots
    _assert_authority(ledger, life, runtime, history, before)
    stable = _allowances(ledger, life, runtime, history)
    _assert_erased_identity_and_contents_refuse(history, ledger, owner, runtime, receipt, tombstone)
    _assert_authority(ledger, life, runtime, history, stable)
    assert ledger.accounting() == stable["accounting"]
    assert life._copy_budget._charged == stable["raw_charge"]
    assert controller._pending == {} and controller._models == [] and controller._attempts == 0


def test_should_refuse_retrospective_lineage_after_unobserved_original_lifecycle_delete():
    owner, life, runtime, ledger, controller, _ = setup(
        replay_examples=1, retain_inbox_origins=True, live_records=32
    )
    history = ledger._history()
    assert history is not None
    receipt = runtime._inbox._applied[FIRST]
    weak_payloads = _weak_payloads(runtime)
    before = _allowances(ledger, life, runtime, history)
    report = life.delete((FIRST,))
    _assert_actual_erasure(owner, runtime, controller, report, receipt, weak_payloads)
    assert history._erased_records == history._erased_sealed == {}
    # These original metadata checks must refuse before a third cleanup attempt;
    # the module observer bounds the actual _cleanup entry at two for both cases.
    with pytest.raises(ValueError):
        ledger.origins()
    with pytest.raises(ValueError):
        ledger.delete((FIRST,))
    assert history._erased_records == history._erased_sealed == {}
    _assert_authority(ledger, life, runtime, history, before)
    assert ledger.accounting() == before["accounting"]
    assert life._copy_budget._charged == before["raw_charge"]
    assert life._admitted_bytes == before["ingress"]
    assert runtime._budget.updates_completed == 1
    assert all(reference() is None for reference in weak_payloads.values())
