"""N4: original public opt-out after one handoff and one retained restore fault.

Only the already declared actual-native setup is used. Cleanup is the original
public operation; the report observer records its original return unchanged.
"""

from math import isfinite
from unittest.mock import patch

import pytest

from src.adapters.numpy_learners import CircadianLearner
from src.core.data_erasure import ErasedExperience
from src.core.data_retention import DataCleanupReport
from src.core.experience import AppliedExperience
from src.core.learner_ports import TrainingDiagnostic
from test_native_managed_replay_checkpoints import FAIL_RESTORE_COPY, observe_native_work, setup


KEY = ("e1", "s1")


@pytest.fixture(scope="module", autouse=True)
def bounded_native_cleanup_work(tmp_path_factory):
    limits = dict(
        models=2,
        learners=8,
        forks=6,
        wakes=2,
        steps=4,
        stores=2,
        predicts=2,
        array_copies=10,
        captures=2,
        preparations=2,
        native_restores=2,
        handoffs=1,
        graph_events=32,
        source_array_bytes=1024 * 1024,
        cleanup_attempts=2,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def _original_authorities(ledger, life, runtime, controller):
    admission = ledger._admission
    limits = admission.limits
    raw_budget = life._copy_budget
    assert raw_budget is not None
    assert (
        limits.max_live_records,
        limits.max_records_created,
        limits.max_invocations,
        limits.max_metadata_bytes,
        limits.max_age_ticks,
        limits.max_payload_bytes,
    ) == (16, 64, 32, 128 * 1024, 120, 4096)
    assert raw_budget._limits.max_lifetime_owned_bytes == 4096
    return dict(
        admission=admission,
        limits=limits,
        raw_budget=raw_budget,
        raw_limits=raw_budget._limits,
        work_budget=runtime._budget,
        charged=raw_budget._charged,
        copy_slots=ledger._copy_slots,
        minimum_copy_slots=ledger._minimum_copy_slots,
        accounting=ledger.accounting(),
        admitted_bytes=life._admitted_bytes,
        enrollments=life._registry._total,
        attempts=controller._attempts,
        updates=runtime._budget.updates_completed,
    )


def _assert_original_allowances(ledger, life, runtime, controller, original, *, unchanged):
    assert ledger._admission is original["admission"]
    assert ledger._admission.limits is original["limits"]
    assert life._copy_budget is original["raw_budget"]
    assert life._copy_budget._limits is original["raw_limits"]
    assert life._policy.owned_payload_copies is original["raw_limits"]
    assert runtime._budget is original["work_budget"]
    assert runtime._budget.updates_completed == original["updates"] == 1
    assert life._copy_budget._charged == original["charged"]
    assert ledger._copy_slots == original["copy_slots"] > 0
    assert ledger._minimum_copy_slots == original["minimum_copy_slots"]
    assert life._admitted_bytes == original["admitted_bytes"]
    assert life._registry._total == original["enrollments"]
    assert controller._attempts == original["attempts"] == 1
    current = ledger.accounting()
    previous = original["accounting"]
    for field in ("records_created", "invocations_started", "metadata_bytes_charged"):
        if unchanged:
            assert getattr(current, field) == getattr(previous, field)
        else:
            assert getattr(current, field) >= getattr(previous, field)
    # Pruning erased canonical rows may lower live inventory. Original copied
    # reservations remain spent and cannot create renewed capacity.
    assert current.live_records >= original["copy_slots"]


def _registered_payloads(life):
    with life._registry._lease() as groups:
        models = life._models(groups)
        inboxes = tuple(
            {id(inbox): inbox for group in groups for inbox in group.references.inboxes}.values()
        )
        checkpoints = sum(
            len(holder._pending)
            for _, kind, holder in life._registry._live()
            if kind == "checkpoint"
        )
    assert all(type(model) is CircadianLearner for model in models)
    snapshots = sum(len(model._model._replay_memory) for model in models)
    receipts = []
    for inbox in inboxes:
        assert KEY in inbox._experiences and KEY in inbox._labels
        receipt = inbox._applied[KEY]
        assert type(receipt) is AppliedExperience
        diagnostic = receipt.diagnostic
        assert type(diagnostic) is TrainingDiagnostic
        assert type(diagnostic.definition) is str
        assert type(diagnostic.value) in (int, float) and isfinite(diagnostic.value)
        receipts.append((inbox, receipt, diagnostic, diagnostic.value, inbox._completed_updates()))
    assert snapshots >= 2 and receipts
    return models, tuple(receipts), snapshots, checkpoints


def _public_opt_out_report(owner, life):
    # Why: owner.opt_out intentionally returns None. This approved observer
    # records the actual lifecycle return, calling the original exactly once.
    reports: list[DataCleanupReport] = []
    original = life.opt_out

    def observe(subject_id):
        assert subject_id == "person-1" and reports == []
        report = original(subject_id)
        assert type(report) is DataCleanupReport
        reports.append(report)
        return report

    with patch.object(life, "opt_out", observe):
        assert owner.opt_out("person-1") is None
    assert len(reports) == 1
    return reports[0]


def _assert_cleanup(owner, life, controller, report, models, receipts, snapshots, checkpoints):
    assert report.reason == "opt_out"
    assert report.requested_keys == report.revoked_keys == (KEY,)
    assert report.model_snapshots_erased == snapshots
    assert report.inboxes_cleared == len(receipts)
    assert report.checkpoints_invalidated == checkpoints
    assert report.promotions_invalidated == 0
    assert "person-1" in owner._opted_out and KEY in owner._revoked_keys
    assert not life._failed and controller._pending == {}
    for model in models:
        assert not model._model._replay_memory
    for inbox, receipt, diagnostic, value, updates in receipts:
        assert inbox._experiences == {} and inbox._labels == {}
        assert inbox._applied[KEY] is receipt
        assert receipt.diagnostic is diagnostic and diagnostic.value == value
        assert type(diagnostic.value) in (int, float) and isfinite(diagnostic.value)
        assert inbox._completed_updates() == updates == 1
        tombstone = inbox._erased[KEY]
        assert type(tombstone) is ErasedExperience
        assert tombstone.key == KEY and tombstone.reason == "opt_out"
        assert tombstone.actor_version == receipt.actor_version
        assert tombstone.observed_at == receipt.observed_at
        assert tombstone.event_id == receipt.event_id
        assert tombstone.arrived_at == receipt.arrived_at


def _assert_old_authority_refused(ledger, controller, manager, token, target):
    checkpoint_ids = tuple(manager._checkpoints)
    assert ledger.origins() == ()
    with pytest.raises(ValueError):
        controller._require(token)
    with pytest.raises(ValueError):
        manager.restore(controller, token)
    with pytest.raises(ValueError):
        manager._copies.origins(controller, target)
    # The empty original ledger cannot mint a new capture or native copy.
    with pytest.raises(ValueError):
        manager.capture(controller)
    assert ledger.origins() == ()
    assert ledger._rows == {} and not controller._pending
    assert len(controller._models) == 1
    assert tuple(manager._checkpoints) == checkpoint_ids


def test_should_publicly_opt_out_after_original_successful_handoff():
    owner, life, runtime, ledger, controller, manager = setup()
    token = manager.capture(controller)
    prepared = manager.restore(controller, token)
    assert owner._shared._runtime is prepared and runtime._retired
    assert controller._pending == {} and len(controller._models) == 1
    target = controller._models[0]
    assert prepared._candidate is target
    assert ledger.origins()[0].key == KEY
    models, receipts, snapshots, checkpoints = _registered_payloads(life)
    assert any(model is runtime._candidate for model in models)
    assert any(model is prepared._candidate for model in models)
    assert len(runtime._candidate._model._replay_memory) == 1
    assert len(target._model._replay_memory) == 1
    assert checkpoints == 0
    original = _original_authorities(ledger, life, prepared, controller)
    report = _public_opt_out_report(owner, life)
    _assert_cleanup(owner, life, controller, report, models, receipts, snapshots, checkpoints)
    _assert_original_allowances(ledger, life, prepared, controller, original, unchanged=True)
    _assert_old_authority_refused(ledger, controller, manager, token, target)
    _assert_original_allowances(ledger, life, prepared, controller, original, unchanged=False)


def test_should_publicly_opt_out_with_original_retained_failed_restore_target():
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
    assert ledger._copy_slots > 0
    assert manager._copies.origins(controller, target)[0].key == KEY
    assert ledger.origins()[0].key == KEY
    models, receipts, snapshots, checkpoints = _registered_payloads(life)
    assert any(model is runtime._candidate for model in models)
    assert any(model is target for model in models)
    assert len(runtime._candidate._model._replay_memory) == len(target._model._replay_memory) == 1
    assert checkpoints == 1
    original = _original_authorities(ledger, life, runtime, controller)
    report = _public_opt_out_report(owner, life)
    _assert_cleanup(owner, life, controller, report, models, receipts, snapshots, checkpoints)
    _assert_original_allowances(ledger, life, runtime, controller, original, unchanged=True)
    _assert_old_authority_refused(ledger, controller, manager, token, target)
    _assert_original_allowances(ledger, life, runtime, controller, original, unchanged=False)
