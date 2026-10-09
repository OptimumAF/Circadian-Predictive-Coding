"""N5: actual capacity eviction and release of a weak copied-target holder.

Why this: retention capacity is fixed at original model birth. Holder reference
release below is an explicit simulation, not a new public API or allowance.
"""

import gc
import json
from weakref import ref

import numpy as np
import pytest

from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from test_managed_experience import declaration
from test_native_managed_replay_checkpoints import FAIL_RESTORE_COPY, observe_native_work, setup


GC_COLLECTIONS = [0]


@pytest.fixture(scope="module", autouse=True)
def bounded_native_expiry_work(tmp_path_factory):
    limits = dict(
        models=2,
        learners=8,
        forks=6,
        wakes=3,
        steps=6,
        stores=3,
        predicts=3,
        array_copies=15,
        captures=2,
        preparations=2,
        native_restores=1,
        handoffs=0,
        graph_events=64,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=0,
    )
    try:
        yield from observe_native_work(tmp_path_factory, limits)
    finally:
        assert GC_COLLECTIONS[0] <= 1
        (tmp_path_factory.mktemp("native-expiry-gc") / "gc.json").write_text(
            json.dumps({"gc_collections": GC_COLLECTIONS[0], "limit": 1}, indent=2),
            encoding="utf8",
        )


def _original_allowances(ledger, life, runtime, *, expected_live_records=16):
    raw_budget = life._copy_budget
    assert raw_budget is not None
    limits = ledger._admission.limits
    assert (
        limits.max_live_records,
        limits.max_records_created,
        limits.max_invocations,
        limits.max_metadata_bytes,
        limits.max_age_ticks,
        limits.max_payload_bytes,
    ) == (expected_live_records, 64, 32, 128 * 1024, 120, 4096)
    assert raw_budget._limits.max_lifetime_owned_bytes == 4096
    return dict(
        admission=ledger._admission,
        limits=limits,
        raw_budget=raw_budget,
        raw_limits=raw_budget._limits,
        budget=runtime._budget,
        retention=runtime._candidate._model._replay_retention_budget,
        accounting=ledger.accounting(),
        charge=raw_budget._charged,
        slots=ledger._copy_slots,
        minimum_slots=ledger._minimum_copy_slots,
        enrollments=life._registry._total,
        ingress=life._admitted_bytes,
        updates=runtime._budget.updates_completed,
    )


def _assert_monotone_allowances(ledger, life, runtime, original, *, exact):
    assert ledger._admission is original["admission"]
    assert ledger._admission.limits is original["limits"]
    assert life._copy_budget is original["raw_budget"]
    assert life._copy_budget._limits is original["raw_limits"]
    assert life._policy.owned_payload_copies is original["raw_limits"]
    assert runtime._budget is original["budget"]
    assert runtime._candidate._model._replay_retention_budget is original["retention"]
    current = ledger.accounting()
    previous = original["accounting"]
    pairs = [
        (current.records_created, previous.records_created),
        (current.invocations_started, previous.invocations_started),
        (current.metadata_bytes_charged, previous.metadata_bytes_charged),
        (life._copy_budget._charged, original["charge"]),
        (ledger._copy_slots, original["slots"]),
        (ledger._minimum_copy_slots, original["minimum_slots"]),
        (life._registry._total, original["enrollments"]),
        (life._admitted_bytes, original["ingress"]),
        (runtime._budget.updates_completed, original["updates"]),
    ]
    for after, before in pairs:
        assert after == before if exact else after >= before
    # Canonical live rows may be pruned. Permanent copy reservations never refund.
    assert current.live_records >= original["slots"]


def _queue_original_distinct_sample(owner, clock):
    owner.declare(declaration("s2", "person-2"))
    owner.record_experience(
        Experience(
            "s2",
            "e1",
            3,
            "actor-0",
            np.array([[0.0, 1.0, 0.0]]),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(LabelArrival("label-s2", "s2", "e1", 4, "actor-0", np.array([[1.0]])))
    clock.advance_to(4)


def test_should_evict_actual_canonical_row_and_refuse_old_restore_before_preparation():
    owner, life, runtime, ledger, controller, manager = setup(replay_examples=1)
    model = runtime._candidate._model
    assert model._replay_retention_budget.max_examples == 1
    assert model._replay_retention_budget.max_bytes == 64
    assert len(model._replay_memory) == 1
    original_row_id = id(model._replay_memory[0])
    row = ref(model._replay_memory[0])
    inputs = ref(model._replay_memory[0].input_batch)
    targets = ref(model._replay_memory[0].target_batch)
    token = manager.capture(controller)
    first_receipt = runtime._inbox._applied[("e1", "s1")]
    original = _original_allowances(ledger, life, runtime)
    _queue_original_distinct_sample(owner, runtime._inbox._clock)
    owner._shared._sharing.resume()
    try:
        poll = ledger.train_ready()
    finally:
        owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 2
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._budget.updates_completed == model._epoch_count == 2
    assert runtime._inbox._applied[("e1", "s1")] is first_receipt
    assert runtime._inbox._applied[("e1", "s2")] is poll.updates[0]
    assert row() is None and inputs() is None and targets() is None
    assert len(model._replay_memory) == 1
    assert original_row_id not in ledger._rows
    origins = ledger.origins()
    assert len(origins) == 1
    assert origins[0].key == ("e1", "s2") and origins[0].subject_id == "person-2"
    assert origins[0].update_number == 2
    assert set(ledger._rows) == {id(model._replay_memory[0])}
    _assert_monotone_allowances(ledger, life, runtime, original, exact=False)
    assert ledger.accounting().records_created > original["accounting"].records_created
    before_refusal = _original_allowances(ledger, life, runtime)
    assert controller._attempts == 0 and controller._models == []
    with pytest.raises(ValueError):
        manager.restore(controller, token)
    assert controller._attempts == 0 and controller._models == []
    assert controller._pending[token].owner is runtime
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert runtime._budget.updates_completed == model._epoch_count == 2
    _assert_monotone_allowances(ledger, life, runtime, before_refusal, exact=True)


def test_should_release_expired_native_copy_target_without_refunding_original_allowances():
    owner, life, runtime, ledger, controller, manager = setup()
    token = manager.capture(controller)
    charge_before_restore = life._copy_budget._charged
    FAIL_RESTORE_COPY[0] = True
    try:
        with pytest.raises(RuntimeError, match="declared actual native restore-copy fault"):
            manager.restore(controller, token)
    finally:
        FAIL_RESTORE_COPY[0] = False
    assert owner._shared._runtime is runtime and not runtime._retired
    assert controller._pending[token].owner is runtime
    assert len(controller._models) == 1 and controller._attempts == 1
    target = controller._models[0]
    assert life._copy_budget._charged > charge_before_restore
    assert manager._copies.origins(controller, target)[0].key == ("e1", "s1")
    native_model = ref(target._model)
    native_row = ref(target._model._replay_memory[0])
    inputs = ref(target._model._replay_memory[0].input_batch)
    targets = ref(target._model._replay_memory[0].target_batch)
    native_model_id = id(target._model)
    witness = manager._copies._copies[native_model_id]
    assert witness.model() is target._model
    assert len(witness.rows) == 1
    assert witness.rows[0].snapshot() is native_row()
    original = _original_allowances(ledger, life, runtime)
    assert original["slots"] > 0
    # Explicit simulated release of the original holder's sole target reference.
    # Retain only weak native references and payload-free witness metadata.
    controller._models.clear()
    del target
    GC_COLLECTIONS[0] += 1
    assert GC_COLLECTIONS[0] <= 1
    gc.collect()
    assert native_model() is None and native_row() is None
    assert inputs() is None and targets() is None
    assert manager._copies._copies[native_model_id] is witness
    assert witness.model() is None and witness.rows[0].snapshot() is None
    assert witness.rows[0].features() is None and witness.rows[0].targets() is None
    assert controller._models == [] and controller._attempts == 1
    _assert_monotone_allowances(ledger, life, runtime, original, exact=True)
    with pytest.raises(ValueError, match="original retained checkpoint holder/history"):
        manager._copies.origins(controller, runtime._candidate)
    assert ledger.origins()[0].key == ("e1", "s1")
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._budget.updates_completed == runtime._candidate._model._epoch_count == 1
    assert controller._pending[token].owner is runtime
    assert controller._models == [] and controller._attempts == 1
    _assert_monotone_allowances(ledger, life, runtime, original, exact=True)
