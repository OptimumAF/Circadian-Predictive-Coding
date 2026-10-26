"""Successive observed DATA updates keep one latest weak proof, not a chain."""

from dataclasses import replace
import gc
import json
import pickle
from typing import Any
from weakref import ref

import numpy as np
import pytest

import test_restored_replay_updates as single_update_tests
from src.app.payload_ownership import PayloadOwnershipBusy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from test_managed_experience import declaration
from test_restored_replay_updates import (
    CURRENT_LIMITS,
    leased,
    model_reference,
    setup as original_setup,
)


REPEATED_LIMITS = replace(CURRENT_LIMITS, max_metadata_bytes=262144)


def third_data():
    """A third fixed DATA pair; no native method or learning calculation."""
    return np.array([[1.0, 1.0, 0.0]], dtype=np.float64), np.array([[1.0]], dtype=np.float64)


def corrupt_paid_native_hash(success):
    original = success.native_content
    damaged = ("1" if original[0] == "0" else "0") + original[1:]

    def changed(value):
        if type(value) is str and value == original:
            return damaged
        if type(value) is tuple:
            return tuple(changed(item) for item in value)
        return value

    before = success.metadata
    after = changed(before)
    assert after != before and len(json.dumps(after)) == len(json.dumps(before))
    object.__setattr__(success, "metadata", after)
    assert success.native_content == original


def setup(*, third=True, limits=REPEATED_LIMITS, reference=model_reference):
    graph = original_setup(limits=limits, max_updates=3, reference=reference)
    previous = graph.manager._current_success
    graph.previous_success = ref(previous)
    graph.previous_bytes = pickle.dumps(previous.metadata)
    graph.previous_charge = previous.charged_bytes
    graph.previous_native_row = ref(graph.new._candidate.model._replay_memory[0])
    graph.before_third = graph.ledger.accounting()
    graph.raw_before_third = graph.life._copy_budget._charged
    features, targets = third_data()
    graph.owner.declare(declaration("s3", "person-3"))
    graph.owner.record_experience(
        Experience("s3", "e1", 8, "actor-0", features, "train", ExperiencePermissions(True, True))
    )
    graph.owner.record_label(LabelArrival("label-s3", "s3", "e1", 10, "actor-0", targets))
    graph.clock.advance_to(10)
    graph.gate.resume()
    graph.third_poll = graph.ledger.train_ready() if third else None
    return graph


def test_should_replace_only_the_latest_paid_proof_and_release_evicted_native_history():
    graph = setup()
    assert len(graph.third_poll.updates) == 1 and graph.third_poll.updates[0].update_number == 3
    assert graph.work["trains"] == graph.budget.updates_completed == 3
    assert graph.old._retired and graph.old._inbox._historical_completed_updates == 1
    assert graph.old._candidate.model.scalar == 1 and graph.new._candidate.model.scalar == 3
    assert len(graph.new._inbox._applied) == 3 and len(graph.old._inbox._applied) == 1
    assert graph.controller._models == [graph.new._candidate]
    assert graph.builder.source is graph.old._candidate and graph.builder.calls == 1
    assert graph.manager._publications[id(graph.old._candidate.model)] is graph.publication
    assert pickle.dumps(graph.publication.metadata) == graph.publication_bytes
    latest = graph.manager._current_success
    assert latest is graph.manager._current_success_sealed
    assert graph.manager._current_success_storage == (latest, latest)
    assert latest.completed_updates == 3
    assert latest.receipt() is graph.new._inbox._applied[("e1", "s3")]
    assert len(latest.history_records) == 3 and all(
        item.verified for item in latest.history_records
    )
    assert latest.charged_bytes == 1024 + len(
        json.dumps(
            latest.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8")
    )
    gc.collect()
    assert graph.previous_success() is None and graph.previous_native_row() is None
    after = graph.ledger.accounting()
    assert after.invocations_started == graph.before_third.invocations_started + 1
    assert after.records_created == graph.before_third.records_created + 2
    assert (
        after.metadata_bytes_charged
        >= graph.before_third.metadata_bytes_charged + latest.charged_bytes
    )
    assert graph.life._copy_budget._charged > graph.raw_before_third
    assert graph.ledger._admission.limits is REPEATED_LIMITS
    assert (
        CURRENT_LIMITS.max_metadata_bytes == 131072 and REPEATED_LIMITS.max_metadata_bytes == 262144
    )
    assert graph.budget.budget.max_training_updates == 3
    work, raw, original_budget, original_clock = (
        dict(graph.work),
        graph.life._copy_budget._charged,
        graph.budget,
        graph.clock,
    )
    graph.gate.pause()
    with leased(graph) as check:
        for _ in range(7):
            check()
    assert graph.work == work and graph.life._copy_budget._charged == raw
    assert graph.old._budget is graph.new._budget is original_budget
    assert graph.old._inbox._clock is graph.new._inbox._clock is original_clock


@pytest.mark.parametrize(
    "change",
    (
        "native",
        "learner",
        "work",
        "row",
        "payload",
        "receipt",
        "history",
        "runtime_policy",
        "paid_hash",
    ),
)
def test_should_validate_the_exact_latest_success_before_any_further_fake_train(change):
    graph = setup(third=False)
    previous = graph.manager._current_success
    if change == "native":
        graph.new._candidate.model.scalar = 99
    elif change == "learner":
        graph.new._candidate._learning_rate = 0.02
    elif change == "work":
        graph.budget.updates_completed = 3
    elif change == "row":
        graph.new._candidate.model._replay_memory[0] = replace(
            graph.new._candidate.model._replay_memory[0]
        )
    elif change == "payload":
        graph.new._candidate.model._replay_memory[0].input_batch[0, 0] = 9.0
    elif change == "receipt":
        key = ("e1", "s2")
        graph.new._inbox._applied[key] = replace(graph.new._inbox._applied[key])
    elif change == "history":
        graph.ledger._inbox_origins._records = dict(graph.ledger._inbox_origins._records)
    elif change == "paid_hash":
        corrupt_paid_native_hash(previous)
    else:
        graph.new._consolidation_limit += 1
    trains = graph.work["trains"]
    with pytest.raises(ValueError):
        graph.ledger.train_ready()
    assert graph.work["trains"] == trains == 2
    assert graph.manager._current_success is previous


def test_should_refuse_rollback_to_the_actual_previous_paid_record():
    graph = setup(third=False)
    previous = graph.manager._current_success
    graph.ledger.train_ready()
    assert graph.manager._current_success is not previous
    graph.manager._current_success = graph.manager._current_success_sealed = previous
    graph.manager._current_success_storage = (previous, previous)
    work, raw = dict(graph.work), graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("an authentic superseded record certified later native work")
    assert graph.work == work and graph.life._copy_budget._charged == raw


@pytest.mark.parametrize("fault", ("native", "previous", "previous_metadata"))
def test_should_not_overwrite_a_late_changed_prior_record_or_mutated_new_outcome(fault):
    state: dict[str, Any] = {"graph": None, "fired": False}

    def reference(learner):
        graph = state["graph"]
        if (
            graph is not None
            and not state["fired"]
            and graph.budget.updates_completed == 3
            and ("e1", "s3") in graph.new._inbox._applied
        ):
            state["fired"] = True
            if fault == "native":
                learner.model.scalar += 1
            elif fault == "previous":
                graph.manager._current_success = replace(graph.manager._current_success)
            else:
                previous = graph.manager._current_success
                # Preserve the original record's identity and encoded size;
                # a paid metadata value cannot change during opaque work.
                corrupt_paid_native_hash(previous)
        return model_reference(learner)

    graph = setup(third=False, reference=reference)
    state["graph"] = graph
    previous, raw, before = (
        graph.manager._current_success,
        graph.life._copy_budget._charged,
        graph.ledger.accounting(),
    )
    if fault == "native":
        with pytest.raises(ValueError) as native_error:
            graph.ledger.train_ready()
        assert type(native_error.value) is ValueError
    else:
        with pytest.raises(ExceptionGroup) as grouped:
            graph.ledger.train_ready()
        error = grouped.value
        assert type(error) is ExceptionGroup
        assert error.message == "native update and observer failed"
        assert len(error.exceptions) == 2
        assert all(type(child) is ValueError for child in error.exceptions)
        message = (
            "original current success sealed storage changed"
            if fault == "previous"
            else "previous current success fields changed during its original poll"
        )
        # Completion and the original failed observer each preserve their own
        # refusal; neither fault may mask the other or become a new success.
        assert [str(child) for child in error.exceptions] == [message, message]
    assert state["fired"] and graph.work["trains"] == graph.budget.updates_completed == 3
    assert graph.ledger._poisoned
    assert graph.manager._current_success.completed_updates == 2
    assert graph.manager._current_success_sealed is previous
    assert graph.manager._current_success_storage == (previous, previous)
    assert graph.life._copy_budget._charged > raw
    assert graph.ledger.accounting().metadata_bytes_charged > before.metadata_bytes_charged
    assert graph.manager._publications[id(graph.old._candidate.model)] is graph.publication
    assert pickle.dumps(graph.publication.metadata) == graph.publication_bytes


def test_should_keep_original_128k_metadata_exhaustion_without_renewal_or_forced_success():
    graph = setup(third=False, limits=CURRENT_LIMITS)
    previous = graph.manager._current_success
    admission = graph.ledger._admission
    before = graph.ledger.accounting()
    # A tiny fake graph may fit more than one success in 128KiB. Spend through
    # the original admission to test exhaustion, without inventing a size claim.
    admission._charge(
        0, 0, CURRENT_LIMITS.max_metadata_bytes - before.metadata_bytes_charged, before.live_records
    )
    raw = graph.life._copy_budget._charged
    with pytest.raises(ValueError, match="limit|metadata|exhaust|allowance"):
        graph.ledger.train_ready()
    assert admission.limits is CURRENT_LIMITS and admission.limits.max_metadata_bytes == 131072
    assert graph.ledger.accounting().metadata_bytes_charged == 131072
    assert graph.work["trains"] == graph.budget.updates_completed == 2
    assert graph.manager._current_success is previous
    assert graph.life._copy_budget._charged >= raw


@pytest.mark.parametrize("authority", ("manager", "copies"))
def test_should_release_partial_leases_and_refuse_busy_prior_proof_before_work(authority):
    graph = setup(third=False)
    previous, work = graph.manager._current_success, dict(graph.work)
    holder = graph.manager if authority == "manager" else graph.manager._copies
    with holder._exclusive():
        with pytest.raises(PayloadOwnershipBusy):
            graph.ledger.train_ready()
    assert not graph.manager._gate.locked() and not graph.manager._copies._gate.locked()
    assert graph.work == work and graph.manager._current_success is previous


def test_should_pay_only_metadata_for_an_observed_deferral_then_resume_and_capture():
    graph = setup(third=False)
    previous, before, work, raw, native = (
        graph.manager._current_success,
        graph.ledger.accounting(),
        dict(graph.work),
        graph.life._copy_budget._charged,
        pickle.dumps(vars(graph.new._candidate.model)),
    )
    previous_metadata = previous.metadata
    graph.gate.pause()
    poll = graph.ledger.train_ready()
    assert not poll.updates and poll.deferred_reason == "paused" and graph.work == work
    assert graph.manager._current_success is previous
    assert previous.metadata is previous_metadata
    after = graph.ledger.accounting()
    assert after.records_created == before.records_created
    assert after.invocations_started == before.invocations_started
    observation = graph.manager._zero_work_poll
    assert observation is graph.manager._zero_work_poll_sealed
    assert graph.manager._zero_work_poll_storage == (observation, observation)
    assert observation.previous_success() is previous and observation.sharing() is graph.gate
    assert observation.deferred_reason == "paused"
    assert observation.charged_bytes == 1024 + len(
        json.dumps(
            observation.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8")
    )
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + observation.charged_bytes
    assert graph.life._copy_budget._charged == raw
    assert graph.budget.updates_completed == 2
    assert pickle.dumps(vars(graph.new._candidate.model)) == native
    assert graph.gate._deferrals == {"paused": 1}
    graph.gate.resume()
    completed = graph.ledger.train_ready()
    assert len(completed.updates) == 1 and completed.updates[0].update_number == 3
    assert graph.work["trains"] == graph.budget.updates_completed == 3
    assert graph.manager._current_success is not previous
    assert graph.manager._current_success.completed_updates == 3
    assert graph.manager._zero_work_poll is graph.manager._zero_work_poll_sealed is None
    assert graph.manager._zero_work_poll_storage == (None, None)
    assert graph.ledger.accounting().metadata_bytes_charged >= (
        after.metadata_bytes_charged + graph.manager._current_success.charged_bytes
    )
    graph.gate.pause()
    with leased(graph) as check:
        check()


def test_should_refuse_an_unobserved_deferral_increment_before_further_work():
    graph = setup(third=False)
    previous, work, raw = (
        graph.manager._current_success,
        dict(graph.work),
        graph.life._copy_budget._charged,
    )
    graph.gate._deferrals["paused"] = 1
    with pytest.raises(ValueError):
        graph.ledger.train_ready()
    assert graph.work == work and graph.budget.updates_completed == 2
    assert graph.life._copy_budget._charged == raw
    assert graph.manager._current_success is previous
    assert graph.manager._zero_work_poll is None


def test_should_capture_after_a_no_pending_poll_without_replacing_the_latest_success():
    graph = setup()
    previous, before, work, raw, revision, sharing, native = (
        graph.manager._current_success,
        graph.ledger.accounting(),
        dict(graph.work),
        graph.life._copy_budget._charged,
        graph.new._revision,
        graph.gate.snapshot(),
        pickle.dumps(vars(graph.new._candidate.model)),
    )
    poll = graph.ledger.train_ready()
    assert not poll.updates and poll.deferred_reason is None
    assert graph.work == work and graph.life._copy_budget._charged == raw
    assert graph.budget.updates_completed == 3 and graph.gate.snapshot() == sharing
    assert pickle.dumps(vars(graph.new._candidate.model)) == native
    assert graph.new._revision == revision + 1
    assert graph.manager._current_success is previous
    after = graph.ledger.accounting()
    assert after.records_created == before.records_created
    assert after.invocations_started == before.invocations_started
    observation = graph.manager._zero_work_poll
    assert observation.previous_success() is previous
    assert observation.deferred_reason is None and observation.before == observation.after
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + observation.charged_bytes
    graph.gate.pause()
    with leased(graph) as check:
        check()
    assert graph.manager._current_success is previous and graph.work == work


def test_should_refuse_native_mutation_inside_the_original_deferring_resource_probe(monkeypatch):
    state: dict[str, Any] = {"graph": None, "fired": False}
    gate_class = single_update_tests.ServingPriorityGate

    def build_gate(limits, *, resource_available):
        def resource_probe():
            graph = state["graph"]
            if graph is not None:
                state["fired"] = True
                graph.new._candidate.model.scalar += 1
                return False
            return resource_available()

        return gate_class(limits, resource_available=resource_probe)

    # Install a port at actual gate birth; the real gate/runtime/observer still
    # make the resource decision. Only the learner's work remains a fake.
    with monkeypatch.context() as patch:
        patch.setattr(single_update_tests, "ServingPriorityGate", build_gate)
        graph = setup(third=False)
    previous, work, raw = (
        graph.manager._current_success,
        dict(graph.work),
        graph.life._copy_budget._charged,
    )
    state["graph"] = graph
    with pytest.raises(ValueError):
        graph.ledger.train_ready()
    assert state["fired"] and graph.gate._deferrals == {"resource_unavailable": 1}
    assert graph.work == work and graph.budget.updates_completed == 2
    assert graph.new._candidate.model.scalar == 3
    assert graph.life._copy_budget._charged == raw
    assert graph.manager._current_success is previous
    assert graph.manager._zero_work_poll is None


@pytest.mark.parametrize("change", ("retired_consent", "latest_consent", "expiry"))
def test_should_recheck_original_retired_and_latest_current_consent_and_retention(change):
    graph = setup()
    if change == "retired_consent":
        graph.owner._opted_out.add("person-1")
    elif change == "latest_consent":
        graph.owner._revoked_keys.add(("e1", "s3"))
    else:
        graph.seconds[0] = 1000.0
    work, raw = dict(graph.work), graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("a latest record bypassed original consent or retention")
    assert graph.work == work and graph.life._copy_budget._charged == raw


def test_should_release_latest_success_and_both_runtime_roots_after_capture():
    graph = setup()
    latest = ref(graph.manager._current_success)
    old, current, owner = ref(graph.old), ref(graph.new), ref(graph.owner)
    with leased(graph) as check:
        check()
    del graph
    gc.collect()
    assert latest() is old() is current() is owner() is None
    with pytest.raises(ValueError, match="expired"):
        check()


@pytest.mark.parametrize("change", ("native", "receipt"))
def test_should_refuse_changes_after_the_latest_third_update_without_more_native_work(change):
    graph = setup()
    if change == "native":
        graph.new._candidate.model.scalar = 4
    else:
        key = ("e1", "s3")
        graph.new._inbox._applied[key] = replace(graph.new._inbox._applied[key])
    work, raw = dict(graph.work), graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("the latest paid record certified an unobserved state change")
    assert graph.work == work and graph.life._copy_budget._charged == raw
