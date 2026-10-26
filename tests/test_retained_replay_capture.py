"""Original paid fake checkpoint copies, verified inside complete capture leases.

Witnesses come only from the existing observed failed preparation. These probes
perform no native work and grant no snapshot, actor or promotion qualification.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import gc
from threading import Thread
from types import SimpleNamespace
from typing import Any
from weakref import ref

import pytest

import src.adapters.numpy_composite_capture as numpy_adapter
from src.app.candidate_checkpoint import CandidateCheckpointView
from src.app.managed_lifecycle_capture import _lease_lifecycle_sources
from src.app.managed_replay_copies import ManagedReplayCopies
from src.app.payload_copy_budget import PayloadCopyBudget
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.replay_capture_origins import lease_replay_capture
from src.core.managed_composite_state import CompositeCaptureLimits
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.native_model_copy import begin_model_copy
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import OwnedPayloadGroup, PayloadHolderMetadata, PayloadReferences
from test_managed_replay_copies import Builder, builder_source, prepare, setup
from test_managed_replay_origins import Learner, Model


BOUNDS = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 64, 16, 65536, 32)
DEFAULT = object()


def prepared(monkeypatch, *, discard=True, bound=True):
    values = setup(monkeypatch)
    owner, ledger, runtime, clock, life, seconds, controller, token, copies, copied, snapshots = (
        values
    )
    if bound:

        def fork(learner):
            window = begin_model_copy(learner.model)
            if window is None:
                return Learner(deepcopy(learner.model))
            try:
                copied.append(life._copy_budget._charged)
                memo: dict[int, Any] = {}
                model = deepcopy(learner.model, memo)
                recipient = Learner(model)
                # The original synchronous copier observes the actual receiving
                # learner, never a wrapper inferred later from a mutable list.
                window.copied(model, memo, recipient=recipient)
                return recipient
            finally:
                window.end()

        monkeypatch.setattr(Learner, "fork", fork)
    target = prepare(values)
    if discard:
        controller.discard(token)  # Unwitnessed checkpoint snapshots remain unsupported.
    return SimpleNamespace(
        owner=owner,
        ledger=ledger,
        runtime=runtime,
        clock=clock,
        life=life,
        seconds=seconds,
        controller=controller,
        token=token,
        copies=copies,
        target=target,
        copied=copied,
        snapshots=snapshots,
    )


def inventory(groups, limits):
    """Trusted fake port supports original singleton model/snapshot role scans."""
    entries = {}
    for group in groups:
        for item in group.references.models + group.references.snapshots:
            if type(item) is CandidateCheckpointView:
                model = item.state
            elif type(item) is Learner:
                model = item.model
            else:
                model = item
            if type(model) is not Model or type(model.rows) is not list:
                raise ValueError("unsupported fake original inventory item")
            entries[id(model)] = model, tuple(model.rows)
    return tuple(entries.values())


@contextmanager
def leased(graph, *, copies=DEFAULT, groups=None, limits=BOUNDS, inventory_port=inventory):
    with _lease_lifecycle_sources(graph.owner, limits=LifecycleCaptureLimits(64, 128)) as (
        _,
        enrollments,
        _,
    ):
        original = tuple(
            OwnedPayloadGroup(
                PayloadHolderMetadata(enrollment.enrollment, enrollment.kind, True),
                graph.life._registry._holders[enrollment.enrollment][1]()._payload_references(),
            )
            for enrollment in enrollments
            if enrollment.ready is True
        )
        supplied = original if groups is None else groups(original)
        options = {} if copies is DEFAULT else {"copies": copies}
        with lease_replay_capture(
            graph.owner, supplied, limits, inventory_port, graph.ledger, **options
        ) as check:
            yield check


def with_copies(graph, **kwargs):
    return leased(graph, copies=graph.copies, **kwargs)


def test_should_verify_actual_retained_copy_with_one_metadata_admission_and_no_raw_reservation(
    monkeypatch,
):
    graph = prepared(monkeypatch)
    before = graph.ledger.accounting()
    raw = graph.life._copy_budget._charged
    copied = list(graph.copied)
    with with_copies(graph) as check:
        assert graph.copies._gate.locked()
        assert all(
            lock.locked()
            for lock in (
                graph.owner._gate,
                graph.runtime._write_gate,
                graph.life._registry._gate,
                graph.life._time_gate,
                graph.life._copy_budget._gate,
                graph.controller._gate,
            )
        )
        check()
        check()
        with pytest.raises(PayloadOwnershipBusy):
            graph.copies.origins(graph.controller, graph.target)
    after = graph.ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert (
        after.records_created == before.records_created
        and after.live_records == before.live_records
    )
    assert graph.life._copy_budget._charged == raw and graph.copied == copied
    assert graph.runtime._budget.updates_completed == 1
    assert not graph.controller._pending
    assert graph.target.model.rows[0] is not graph.runtime._candidate.model.rows[0]


def test_should_verify_multiple_original_paid_models_in_one_bounded_round(monkeypatch):
    graph = prepared(monkeypatch, discard=False)
    with pytest.raises(ValueError, match="prepared native policy"):
        graph.copies.restore(graph.controller, graph.token)
    assert len(graph.controller._models) == graph.controller._attempts == 2
    graph.controller.discard(graph.token)
    before = graph.ledger.accounting()
    raw = graph.life._copy_budget._charged
    with with_copies(graph) as check:
        check()
    after = graph.ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert (
        after.records_created == before.records_created and graph.life._copy_budget._charged == raw
    )


@pytest.mark.parametrize("explicit", (False, True))
def test_should_keep_unqualified_retained_replay_refused_when_copies_are_not_enabled(
    monkeypatch, explicit
):
    graph = prepared(monkeypatch)
    with pytest.raises(ValueError):
        with leased(graph, **({"copies": None} if explicit else {})):
            pytest.fail("unqualified retained rows reached projection")


@pytest.mark.parametrize("invalid", ("object", "subclass", "foreign", "unwitnessed"))
def test_should_refuse_nonoriginal_or_unwitnessed_copies_without_raw_copy(monkeypatch, invalid):
    graph = prepared(monkeypatch)
    wrong: Any = object()
    other = None
    if invalid == "subclass":

        class Derived(ManagedReplayCopies):
            pass

        wrong = Derived(graph.ledger, builder_source=builder_source)
    elif invalid == "foreign":
        other = prepared(monkeypatch)
        wrong = other.copies
    elif invalid == "unwitnessed":
        wrong = ManagedReplayCopies(graph.ledger, builder_source=builder_source)
    raw = graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph, copies=wrong):
            pytest.fail("foreign copy witness qualified replay")
    assert graph.life._copy_budget._charged == raw
    assert other is None or other.owner is not graph.owner


def test_should_refuse_a_busy_original_copy_gate_without_downstream_work(monkeypatch):
    graph = prepared(monkeypatch)
    raw = graph.life._copy_budget._charged
    with graph.copies._exclusive():
        with pytest.raises(PayloadOwnershipBusy):
            with with_copies(graph):
                pytest.fail("busy witnesses reached projection")
    assert graph.life._copy_budget._charged == raw


def test_should_bound_same_thread_checks_and_expire_verifier_after_original_context(monkeypatch):
    graph = prepared(monkeypatch)
    errors: list[BaseException] = []
    with with_copies(graph) as check:

        def foreign():
            try:
                check()
            except BaseException as error:
                errors.append(error)

        worker = Thread(target=foreign)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive() and len(errors) == 1 and isinstance(errors[0], ValueError)
        for _ in range(7):
            check()
        with pytest.raises(ValueError, match="allowance"):
            check()
    with pytest.raises(ValueError, match="expired"):
        check()


@pytest.mark.parametrize("bound", ("groups", "items", "entries", "rows"))
def test_should_bound_every_borrowed_inventory_collection_before_qualification(monkeypatch, bound):
    graph = prepared(monkeypatch)

    def groups(original):
        if bound == "groups":
            return original * 65
        if bound == "items":
            return tuple(
                replace(group, references=replace(group.references, models=(graph.target,) * 65))
                if group.holder.kind == "checkpoint"
                else group
                for group in original
            )
        return original

    def overflowing(items, limits):
        entries = inventory(items, limits)
        if bound == "entries":
            return entries * 65
        if bound == "rows":
            return tuple((model, rows * 65 if rows else rows) for model, rows in entries)
        return entries

    with pytest.raises(ValueError):
        with with_copies(graph, groups=groups, inventory_port=overflowing):
            pytest.fail("oversized inventory reached copy")


@pytest.mark.parametrize("role", ("actor", "candidate", "promotion", "snapshot"))
def test_should_refuse_the_same_paid_model_in_any_additional_unqualified_role(monkeypatch, role):
    graph = prepared(monkeypatch)

    def aliased(original):
        cp = next(group for group in original if group.holder.kind == "checkpoint")
        # Data-only inventory overlays isolate the role boundary: the original
        # checkpoint still owns its actual target and its genuine paid witness.
        if role == "snapshot":
            return tuple(
                replace(
                    group, references=replace(group.references, snapshots=(graph.target.model,))
                )
                if group is cp
                else group
                for group in original
            )
        if role in ("actor", "candidate"):
            return tuple(
                replace(group, references=replace(group.references, models=(graph.target,)))
                if group.holder.kind == role
                else group
                for group in original
            )
        extra = OwnedPayloadGroup(
            PayloadHolderMetadata(
                max(group.holder.enrollment for group in original) + 1, "promotion", True
            ),
            PayloadReferences(models=(graph.target,)),
        )
        return (*original, extra)

    with pytest.raises(ValueError):
        with with_copies(graph, groups=aliased):
            pytest.fail("checkpoint witness escaped its original model role")
    assert graph.controller._models[0] is graph.target


@pytest.mark.parametrize(
    "change",
    (
        "target_payload",
        "target_row",
        "target_model",
        "models_list",
        "models_position",
        "learner",
        "build",
        "policy",
        "probe",
        "attempt",
        "attempt_limit",
        "enrollment",
        "ready",
        "copy_slots",
        "copy_port",
        "row_metadata",
        "source_payload",
        "source_receipt",
        "revoked",
        "optout",
        "tombstone",
        "tick_age",
        "elapsed_age",
        "paid_raw",
        "copy_limits",
        "copy_budget",
    ),
)
def test_should_refuse_changed_original_producer_target_holder_and_paid_history(
    monkeypatch, change
):
    graph = prepared(monkeypatch)
    controller, target, runtime, ledger, life = (
        graph.controller,
        graph.target,
        graph.runtime,
        graph.ledger,
        graph.life,
    )
    if change == "target_payload":
        target.model.rows[0].features.value = "changed"
    elif change == "target_row":
        target.model.rows[0] = deepcopy(target.model.rows[0])
    elif change == "target_model":
        target.model = deepcopy(target.model)
    elif change == "models_list":
        controller._models = list(controller._models)
    elif change == "models_position":
        controller._models.insert(0, Learner())
        controller._attempts = 2
    elif change == "learner":
        controller._models[0] = Learner(target.model)
    elif change == "build":
        controller._build = Builder(runtime._candidate)
    elif change == "policy":
        controller._policy = lambda _: "0" * 64
    elif change == "probe":
        controller._digest = lambda _: "0" * 64
    elif change == "attempt":
        controller._attempts = 0
    elif change == "attempt_limit":
        controller._attempt_limit += 1
    elif change == "enrollment":
        number = next(n for n, (_, weak) in life._registry._holders.items() if weak() is controller)
        _, original = life._registry._holders[number]
        life._registry._holders[number] = ("actor", original)
    elif change == "ready":
        controller._payload_ready = False
    elif change == "copy_slots":
        ledger._copy_slots = 0
    elif change == "copy_port":
        graph.copies._builder_source = lambda _: runtime._candidate
    elif change == "row_metadata":
        witness = next(iter(graph.copies._copies.values()))
        witness.rows[0].data = replace(witness.rows[0].data, row_start=1)
    elif change == "source_payload":
        runtime._candidate.model.rows[0].features.value = "changed"
    elif change == "source_receipt":
        key = ("e1", "s1")
        runtime._inbox._applied[key] = deepcopy(runtime._inbox._applied[key])
    elif change == "revoked":
        graph.owner._revoked_keys.add(("e1", "s1"))
    elif change == "optout":
        graph.owner._opted_out.add("person-1")
    elif change == "tombstone":
        runtime._inbox._erased[("e1", "s1")] = object()
    elif change == "tick_age":
        graph.clock.advance_to(120)
    elif change == "elapsed_age":
        graph.seconds[0] = 1000.0
    elif change == "paid_raw":
        life._copy_budget._charged -= 1
    elif change == "copy_limits":
        life._copy_budget._limits = PayloadCopyLimits(4096)
    else:
        life._copy_budget = PayloadCopyBudget(life._policy.owned_payload_copies)
    raw = life._copy_budget._charged
    with pytest.raises(ValueError):
        with with_copies(graph):
            pytest.fail("changed original authority reached payload copy")
    assert life._copy_budget._charged == raw and runtime._budget.updates_completed == 1


@pytest.mark.parametrize("change", ("payload", "holder"))
def test_should_reject_late_changes_and_keep_the_single_original_metadata_admission(
    monkeypatch, change
):
    graph = prepared(monkeypatch)
    before = graph.ledger.accounting()
    raw = graph.life._copy_budget._charged
    with with_copies(graph) as check:
        if change == "payload":
            graph.target.model.rows[0].features.value = "late change"
        else:
            graph.controller._models = list(graph.controller._models)
        with pytest.raises(ValueError):
            check()
    after = graph.ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert after.records_created == before.records_created
    assert graph.life._copy_budget._charged == raw


def test_should_keep_original_pending_snapshot_unqualified_without_discarding_it(monkeypatch):
    graph = prepared(monkeypatch, discard=False)
    with pytest.raises(ValueError):
        with with_copies(graph):
            pytest.fail("constructor-copy witness qualified an unobserved snapshot")
    assert graph.token in graph.controller._pending


def test_should_refuse_old_unbound_copy_witness_instead_of_learning_a_wrapper_from_current_list(
    monkeypatch,
):
    graph = prepared(monkeypatch, bound=False)
    assert graph.copies.origins(graph.controller, graph.target)
    with pytest.raises(ValueError):
        with with_copies(graph):
            pytest.fail("legacy unbound witness acquired capture qualification")


def test_should_release_original_graph_when_only_the_expired_check_survives(monkeypatch):
    with monkeypatch.context() as scoped:
        graph = prepared(scoped)
        owner, target = ref(graph.owner), ref(graph.target.model)
        with with_copies(graph) as check:
            check()
        del graph
    gc.collect()
    assert owner() is None and target() is None
    with pytest.raises(ValueError, match="expired"):
        check()


@pytest.mark.parametrize("enabled", (False, True))
def test_should_forward_optional_copies_without_replacing_original_capture_controls(
    monkeypatch, enabled
):
    owner, ledger, copies, result = object(), object(), object() if enabled else None, object()
    limits = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32)
    calls: list[dict[str, Any]] = []

    def capture(value, **kwargs):
        assert value is owner
        calls.append(kwargs)
        return result

    monkeypatch.setattr(numpy_adapter, "capture_managed_composite", capture)
    assert (
        numpy_adapter.capture_numpy_managed_composite(
            owner, limits=limits, replay_origins=ledger, replay_copies=copies
        )
        is result
    )
    assert len(calls) == 1
    assert calls[0]["limits"] is limits and calls[0]["replay_origins"] is ledger
    assert calls[0]["replay_copies"] is copies
