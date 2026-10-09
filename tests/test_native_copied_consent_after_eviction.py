"""N8: originally admitted copied-row consent after canonical eviction.

Why this: native capacity eviction ends canonical retention, while an already
admitted copied holder still depends on the original source and consent. These
controls observe metadata only; they grant no erasure or model-operation claim.
"""

import sys
from typing import Any
from weakref import ReferenceType, ref

import numpy as np
import pytest

from src.adapters.numpy_replay_origins import replay_payload_fingerprint
from src.core.replay_origin import RECORD_OVERHEAD_BYTES
from test_native_checkpoint_eviction import (
    _assert_monotone_allowances,
    _original_allowances,
    _queue_original_distinct_sample,
)
from test_native_managed_replay_checkpoints import FAIL_RESTORE_COPY, observe_native_work, setup


FIRST = ("e1", "s1")
SECOND = ("e1", "s2")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_copied_consent_work(tmp_path_factory):
    limits = dict(
        models=2,
        learners=8,
        forks=6,
        wakes=4,
        steps=8,
        stores=4,
        predicts=4,
        array_copies=20,
        captures=2,
        preparations=2,
        native_restores=2,
        handoffs=0,
        graph_events=64,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=0,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _admit_copy_then_evict(*, fingerprint=None):
    owner, life, runtime, ledger, controller, manager = setup(
        replay_examples=1, fingerprint=fingerprint
    )
    assert ledger._history() is None
    model = runtime._candidate._model
    assert (
        model._replay_retention_budget.max_examples,
        model._replay_retention_budget.max_bytes,
    ) == (
        1,
        64,
    )
    original_row_id = id(model._replay_memory[0])
    original_row = ref(model._replay_memory[0])
    original_features = ref(model._replay_memory[0].input_batch)
    original_targets = ref(model._replay_memory[0].target_batch)
    first_receipt = runtime._inbox._applied[FIRST]
    birth = _original_allowances(ledger, life, runtime)
    token = manager.capture(controller)
    FAIL_RESTORE_COPY[0] = True
    try:
        with pytest.raises(RuntimeError, match="declared actual native restore-copy fault"):
            manager.restore(controller, token)
    finally:
        FAIL_RESTORE_COPY[0] = False
    assert controller._attempts == 1 and len(controller._models) == 1
    target = controller._models[0]
    copied = target._model._replay_memory[0]
    witness = manager._copies._copies[id(target._model)]
    assert witness.model() is target._model and len(witness.rows) == 1
    row = witness.rows[0]
    assert all(
        type(reference) is ReferenceType
        for reference in (
            witness.model,
            row.snapshot,
            row.features,
            row.targets,
            row.source,
            row.label,
            row.declaration,
            row.receipt,
        )
    )
    assert row.snapshot() is copied and copied is not original_row()
    assert row.features() is copied.input_batch and row.targets() is copied.target_batch
    assert copied.input_batch is not original_features()
    assert copied.target_batch is not original_targets()
    assert not np.shares_memory(copied.input_batch, original_features())
    assert not np.shares_memory(copied.target_batch, original_targets())
    assert row.source() is runtime._inbox._experiences[FIRST]
    assert row.label() is runtime._inbox._labels[FIRST]
    assert row.declaration() is owner._catalog[FIRST] and row.receipt() is first_receipt
    assert row.verified and row.data.key == FIRST and row.data.update_number == 1
    assert life._copy_budget._charged > birth["charge"]
    assert ledger._copy_slots > birth["slots"]
    admitted = _original_allowances(ledger, life, runtime)
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    second_payload_bytes = (
        runtime._inbox._experiences[SECOND].features.nbytes
        + runtime._inbox._labels[SECOND].targets.nbytes
    )
    assert second_payload_bytes == 32
    # Original ingress reserves 24 feature + 8 target bytes before copying.
    assert life._admitted_bytes == admitted["ingress"] + second_payload_bytes
    assert life._copy_budget._charged == admitted["charge"] + second_payload_bytes
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 2
    assert runtime._inbox._applied[FIRST] is first_receipt
    assert runtime._inbox._applied[SECOND] is poll.updates[0]
    assert original_row() is None and original_features() is None and original_targets() is None
    assert original_row_id not in ledger._rows
    assert set(ledger._rows) == {id(model._replay_memory[0])}
    assert len(model._replay_memory) == 1 and len(target._model._replay_memory) == 1
    assert target._model._replay_memory[0] is copied
    assert row.snapshot() is copied and row.features() is copied.input_batch
    assert row.targets() is copied.target_batch and row.receipt() is first_receipt
    assert manager._copies._copies[id(target._model)] is witness
    assert ledger.origins()[0].key == SECOND
    _assert_monotone_allowances(ledger, life, runtime, admitted, exact=False)
    # The original before-native-update guard reserves the same 32 bytes for
    # consumed inputs, even though capacity then evicts the FIRST native row.
    assert life._copy_budget._charged == admitted["charge"] + 2 * second_payload_bytes
    assert ledger._copy_slots == admitted["slots"]
    assert ledger._minimum_copy_slots == admitted["minimum_slots"]
    return owner, life, runtime, ledger, controller, manager, token, target, witness


def _metadata_boundary(owner, life, runtime, ledger, controller, manager, token, target):
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 2
    assert controller._attempts == 1 and controller._models == [target]
    assert controller._pending[token].owner is runtime
    return dict(
        allowances=_original_allowances(ledger, life, runtime),
        pending=controller._pending,
        entry=controller._pending[token],
        models=controller._models,
        anchors=ledger._anchors,
        rows=ledger._rows,
        copies=manager._copies,
        copy_map=manager._copies._copies,
        witness=manager._copies._copies[id(target._model)],
        copied_row=target._model._replay_memory[0],
        copied_features=target._model._replay_memory[0].input_batch,
        copied_targets=target._model._replay_memory[0].target_batch,
        retained=runtime._candidate._model._replay_memory,
        canonical=runtime._candidate._model._replay_memory[0],
        registry=life._registry,
        policy=life._policy,
        catalog=owner._catalog,
        revocations=owner._revoked_keys,
        experiences=runtime._inbox._experiences,
        labels=runtime._inbox._labels,
        applied=runtime._inbox._applied,
        first_receipt=runtime._inbox._applied[FIRST],
        second_receipt=runtime._inbox._applied[SECOND],
    )


def _assert_metadata_access_cost(
    owner, life, runtime, ledger, controller, manager, token, target, before
):
    previous = before["allowances"]
    _assert_monotone_allowances(ledger, life, runtime, previous, exact=False)
    after = ledger.accounting()
    assert after.records_created == previous["accounting"].records_created
    assert after.invocations_started == previous["accounting"].invocations_started + 1
    assert (
        after.metadata_bytes_charged
        == previous["accounting"].metadata_bytes_charged + RECORD_OVERHEAD_BYTES
    )
    assert after.live_records == previous["accounting"].live_records
    assert life._copy_budget._charged == previous["charge"]
    assert ledger._copy_slots == previous["slots"]
    assert ledger._minimum_copy_slots == previous["minimum_slots"]
    assert life._registry._total == previous["enrollments"]
    assert life._admitted_bytes == previous["ingress"]
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 2
    assert runtime._inbox._historical_completed_updates is None
    assert controller._attempts == 1 and controller._models is before["models"]
    assert controller._models == [target]
    assert (
        controller._pending is before["pending"] and controller._pending[token] is before["entry"]
    )
    assert controller._pending[token].owner is runtime
    assert ledger._anchors is before["anchors"] and ledger._rows is before["rows"]
    assert manager._copies is before["copies"] and manager._copies._copies is before["copy_map"]
    assert manager._copies._copies[id(target._model)] is before["witness"]
    assert target._model._replay_memory[0] is before["copied_row"]
    assert before["copied_row"].input_batch is before["copied_features"]
    assert before["copied_row"].target_batch is before["copied_targets"]
    assert runtime._candidate._model._replay_memory is before["retained"]
    assert runtime._candidate._model._replay_memory[0] is before["canonical"]
    assert life._registry is before["registry"] and life._policy is before["policy"]
    assert owner._catalog is before["catalog"] and owner._revoked_keys is before["revocations"]
    assert runtime._inbox._experiences is before["experiences"]
    assert (
        runtime._inbox._labels is before["labels"] and runtime._inbox._applied is before["applied"]
    )
    assert runtime._inbox._applied[FIRST] is before["first_receipt"]
    assert runtime._inbox._applied[SECOND] is before["second_receipt"]
    row = before["witness"].rows[0]
    assert row.source() is runtime._inbox._experiences[FIRST]
    assert row.label() is runtime._inbox._labels[FIRST]
    assert row.receipt() is before["first_receipt"] and row.declaration() is owner._catalog[FIRST]
    assert SECOND not in owner._revoked_keys and ledger.origins()[0].key == SECOND


def _assert_continued_metadata(
    owner, life, runtime, ledger, controller, manager, token, target, witness
):
    before = _metadata_boundary(owner, life, runtime, ledger, controller, manager, token, target)
    origins = manager._copies.origins(controller, target)
    assert len(origins) == 1 and origins[0] is witness.rows[0].data
    assert origins[0].key == FIRST and origins[0].subject_id == "person-1"
    assert origins[0].update_number == 1
    _assert_metadata_access_cost(
        owner, life, runtime, ledger, controller, manager, token, target, before
    )


def test_should_recheck_original_copied_consent_after_actual_canonical_eviction():
    owner, life, runtime, ledger, controller, manager, token, target, witness = (
        _admit_copy_then_evict()
    )
    _assert_continued_metadata(
        owner, life, runtime, ledger, controller, manager, token, target, witness
    )
    before = _metadata_boundary(owner, life, runtime, ledger, controller, manager, token, target)
    owner._revoked_keys.add(FIRST)
    with pytest.raises(ValueError, match="revoked"):
        manager._copies.origins(controller, target)
    assert FIRST in owner._revoked_keys
    _assert_metadata_access_cost(
        owner, life, runtime, ledger, controller, manager, token, target, before
    )


def test_should_refuse_consent_revoked_inside_last_actual_copied_fingerprint():
    probe: dict[str, Any] = dict(armed=False, visits=0, fired=False)

    def fingerprint(snapshot, maximum):
        result = replay_payload_fingerprint(snapshot, maximum)
        if probe["armed"] and snapshot is probe["snapshot"]:
            # The actual copied row has two _verify passes in origins(); the
            # canonical SECOND row has a different identity and never fires.
            assert sys._getframe(2).f_code.co_name == "_verify"
            assert sys._getframe(2).f_globals["__name__"] == "src.app.managed_replay_copies"
            assert sys._getframe(3).f_code.co_name == "origins"
            assert sys._getframe(3).f_locals["self"] is manager._copies
            probe["visits"] += 1
            assert probe["visits"] <= 2
            if probe["visits"] == 2:
                assert FIRST not in owner._revoked_keys
                owner._revoked_keys.add(FIRST)
                probe["fired"] = True
        return result

    owner, life, runtime, ledger, controller, manager, token, target, witness = (
        _admit_copy_then_evict(fingerprint=fingerprint)
    )
    _assert_continued_metadata(
        owner, life, runtime, ledger, controller, manager, token, target, witness
    )
    before = _metadata_boundary(owner, life, runtime, ledger, controller, manager, token, target)
    probe["snapshot"] = target._model._replay_memory[0]
    probe["armed"] = True
    try:
        with pytest.raises(ValueError, match="revoked"):
            manager._copies.origins(controller, target)
    finally:
        probe["armed"] = False
    assert probe["fired"] and probe["visits"] == 2 and FIRST in owner._revoked_keys
    _assert_metadata_access_cost(
        owner, life, runtime, ledger, controller, manager, token, target, before
    )
