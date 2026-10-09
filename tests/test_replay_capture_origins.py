"""Bounded fake original managed rows; no native model or capture graph copies."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from threading import Thread
import pytest
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.replay_capture_origins import lease_replay_capture
from test_managed_replay_origins import setup, queue, LIMITS

BOUNDS = SimpleNamespace(max_nodes=64)


def inventory(groups, limits):
    return tuple((m, tuple(m.rows)) for m in groups)


@contextmanager
def leased(owner, runtime, ledger):
    with owner._operation(), runtime._exclusive():
        with lease_replay_capture(
            owner, (runtime._candidate.model,), BOUNDS, inventory, ledger
        ) as check:
            yield check


def test_should_check_original_rows_without_reentering_owner_and_spend_attempts():
    owner, ledger, runtime, clock, _, budget = setup()
    queue(owner, clock)
    ledger.train_ready()
    before = ledger.accounting()
    with leased(owner, runtime, ledger) as check:
        check()
        with pytest.raises(PayloadOwnershipBusy):
            ledger.origins()
        with pytest.raises(PayloadOwnershipBusy):
            owner.train_ready()
    after = ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert after.records_created == before.records_created and budget.updates_completed == 1
    with pytest.raises(ValueError, match="expired"):
        check()


@pytest.mark.parametrize(
    "change",
    ["missing", "uncertain", "revoked", "tombstone", "age", "payload", "metadata", "expired"],
)
def test_should_refuse_invalid_original_candidate_origins_before_downstream_work(change):
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    supplied = ledger
    if change == "missing":
        supplied = None
    elif change == "uncertain":
        ledger._poisoned = True
    elif change == "revoked":
        owner._revoked_keys.add(("e1", "s1"))
    elif change == "tombstone":
        from src.core.data_erasure import ErasedExperience

        runtime._inbox._erased[("e1", "s1")] = ErasedExperience(
            ("e1", "s1"), "actor-0", 1, "label-s1", 3, 3, "deleted"
        )
    elif change == "age":
        clock.advance_to(122)
    elif change == "payload":
        runtime._candidate.model.rows[0].features.value = "changed"
    elif change == "metadata":
        row = next(iter(ledger._rows.values()))
        row.data = replace(row.data, row_start=1)
    else:
        runtime._inbox._labels[("e1", "s1")] = replace(runtime._inbox._labels[("e1", "s1")])
    with pytest.raises(ValueError):
        with leased(owner, runtime, supplied):
            pytest.fail("downstream projection/copy")


def test_should_refuse_foreign_ledger_even_with_empty_buffer():
    owner, ledger, runtime, _, _, _ = setup()
    from copy import copy

    foreign = copy(owner)  # One metadata shell;no extra graph/model/payload constructor.
    with foreign._operation(), runtime._exclusive():
        with pytest.raises(ValueError, match="original owner"):
            with lease_replay_capture(
                foreign, (runtime._candidate.model,), BOUNDS, inventory, ledger
            ):
                pytest.fail("copy")


def test_should_refuse_unqualified_retained_model_replay_without_content_matching():
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    actor = runtime.actor._learner.model
    actor.rows = runtime._candidate.model.rows[:]  # Synthetic alias,not actual fork lineage.
    with owner._operation(), runtime._exclusive():
        with pytest.raises(ValueError, match="holder lineage"):
            with lease_replay_capture(
                owner, (runtime._candidate.model, actor), BOUNDS, inventory, ledger
            ):
                pytest.fail("copy")
    assert ledger.accounting().invocations_started == 2


def test_should_detect_changes_after_callback_and_keep_failed_copy_attempt_charge():
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    with pytest.raises(RuntimeError, match="copy failed"):
        with leased(owner, runtime, ledger):
            raise RuntimeError("copy failed")
    assert ledger.accounting().invocations_started == 2
    with leased(owner, runtime, ledger) as check:
        runtime._candidate.model.rows[0].features.value = "changed"
        with pytest.raises(ValueError, match="integrity"):
            check()
    assert ledger.accounting().invocations_started == 3


def test_should_bound_checks_and_refuse_other_thread_with_one_joined_contender():
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    with leased(owner, runtime, ledger) as check:
        errors = []

        def foreign():
            try:
                check()
            except ValueError as error:
                errors.append(error)

        worker = Thread(target=foreign)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive() and len(errors) == 1
        for _ in range(7):
            check()
        with pytest.raises(ValueError, match="allowance"):
            check()


def test_should_keep_empty_buffer_compatible_and_refuse_late_insert_without_ledger():
    owner, _, runtime, _, _, _ = setup()
    with leased(owner, runtime, None) as check:
        check()
        runtime._candidate.model.rows.append(object())
        with pytest.raises(ValueError, match="nonempty replay"):
            check()
    with pytest.raises(ValueError, match="expired"):
        check()


def test_should_preserve_invocation_ceiling_without_renewing_training_or_capture():
    owner, ledger, runtime, clock, _, _ = setup(limits=replace(LIMITS, max_invocations=1))
    queue(owner, clock)
    ledger.train_ready()
    before = ledger.accounting()
    with pytest.raises(ValueError, match="limit"):
        with leased(owner, runtime, ledger):
            pytest.fail("copy")
    assert ledger.accounting() == before


def test_should_release_original_graph_when_only_expired_check_and_ledger_survive():
    import gc
    from weakref import ref

    owner, ledger, runtime, clock, gate, budget = setup()
    queue(owner, clock)
    ledger.train_ready()
    owner_ref = ref(owner)
    payload_ref = ref(runtime._candidate.model.rows[0].features)
    with leased(owner, runtime, ledger) as check:
        check()
    del owner, runtime, clock, gate, budget
    gc.collect()
    assert owner_ref() is None and payload_ref() is None
    with pytest.raises(ValueError, match="expired"):
        check()
