"""Actual scalar publication witnesses, without native ML or fabricated lineage.

The CPC shell is intentionally not a native-valid network. Known scalar records
exercise the real copier, checkpoint, inbox and publication boundaries only.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import gc
import json
import pickle
from threading import Thread
from types import SimpleNamespace
from typing import Any, cast
from weakref import ref

import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_lifecycle_capture import _lease_lifecycle_sources
from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
from src.app.managed_replay_copies import ManagedReplayCopies
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.replay_capture_origins import lease_replay_capture
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    ReplaySnapshot,
)
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.native_model_copy import begin_model_copy
from src.core.native_state_copy import copy_native_state
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import (
    OwnedPayloadGroup,
    PayloadHolderMetadata,
    PayloadOwnershipLimits,
    PayloadReferences,
)
from src.core.replay_origin import ReplayOriginPorts
from src.core.replay_write_origin import begin_replay_write
from src.core.resource_sharing import SharingLimits
from test_checkpoint_snapshot_capture import (
    BOUNDS,
    GRAPH_PORTS,
    PORTS,
    footprint,
    graph_bytes,
    payload_bytes,
    rows,
    shell,
)
from test_managed_experience import declaration
from test_managed_replay_origins import LIMITS, WINDOW


DEFAULT = object()
# Why: a genuine capture + restore includes native/inbox graph admissions that
# the older constructor-copy fixture's eight calls cannot reach. Set the same
# fixed 32-call allowance as the native protocol at birth; never renew it later.
RESTORE_LIMITS = replace(LIMITS, max_invocations=32)


def forbidden(*args, **kwargs):
    pytest.fail("capture invoked an operational learner/controller callback")


class RestoredLearner:
    def __init__(self, model=None, work=None):
        self.model = shell() if model is None else model
        self.work = dict(forks=0, snapshots=0, restores=0, trains=0) if work is None else work
        self._learning_rate = 0.01

    def fork(self):
        self.work["forks"] += 1
        window = begin_model_copy(self.model)
        try:
            memo: dict[int, Any] = {}
            recipient = RestoredLearner(deepcopy(self.model, memo), self.work)
            if window is not None:
                window.copied(recipient.model, memo, recipient=recipient)
            return recipient
        finally:
            if window is not None:
                window.end()

    def train_batch(self, features, targets):
        self.work["trains"] += 1
        window = begin_replay_write(
            self.model, features, targets, self.model._replay_memory, 1, "rows"
        )
        if window is not None:
            window.before_copy(0, 1)
        row = ReplaySnapshot(deepcopy(features), deepcopy(targets), 1.0, 0.5)
        if window is not None:
            window.copied(row)
        self.model._replay_memory.append(row)
        self.model.scalar += 1
        if window is not None:
            window.finish(self.model._replay_memory)
        return TrainingDiagnostic("restored_scalar_fixture", 1.0)

    def snapshot_state(self):
        self.work["snapshots"] += 1
        return CircadianNetworkSnapshot(
            2,
            3,
            (2,),
            2,
            2,
            CircadianConfig(),
            copy_native_state(vars(self.model), producer=self.model, kind="native_snapshot"),
        )

    def restore_state(self, state):
        self.work["restores"] += 1
        restored = copy_native_state(state.state, producer=self.model, kind="native_restore")
        self.model.__dict__.clear()
        self.model.__dict__.update(restored)

    predict = forbidden


class Builder:
    def __init__(self, source):
        self.source, self.calls = source, 0

    def __call__(self, state):
        self.calls += 1
        return self.source.fork()


def builder_source(builder):
    if type(builder) is not Builder:
        raise ValueError("exact original scalar builder required")
    return builder.source


def setup(*, retain=True, restore=True, default=False, limits=RESTORE_LIMITS, model_reference=None):
    seconds = [0.0]
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), lambda: seconds[0])
    old = ActorShadowRuntime(
        RestoredLearner(),
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
    )
    gate = ServingPriorityGate(SharingLimits(1, 8, 1), resource_available=lambda: True)
    owner = ManagedExperienceOwner(ResourceSharedRuntime(old, gate), limits=LifecycleLimits(4, 4))
    life = ManagedDataLifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096,
            120,
            PayloadOwnershipLimits(8, 12),
            owned_payload_copies=PayloadCopyLimits(4096),
            max_retention_seconds=1000.0,
        ),
        measure_payload_bytes=payload_bytes,
        native_footprint=footprint,
        native_erase=forbidden,
        measure_auxiliary_bytes=lambda _: 0,
        measure_checkpoint_bytes=lambda view: (
            graph_bytes(view.state, 4, 4096)
            + 16 * (len(view.inbox.experiences) + len(view.inbox.labels))
        ),
        native_growth_bytes=lambda *_: 32,
        # Both constructor-fork and restore payloads are genuinely copied below.
        prepare_model_bytes=lambda builder, state: (
            footprint(builder.source).payload_bytes + graph_bytes(state, 4, 4096)
        ),
        prediction_cache_bytes=forbidden,
    )
    ports = (
        PORTS
        if model_reference is None
        else ReplayOriginPorts(
            model_reference,
            PORTS.retained,
            PORTS.copy_bytes,
            PORTS.payloads,
            PORTS.fingerprint,
        )
    )
    ledger = ManagedReplayOrigins(owner, ports, limits, WINDOW)
    owner.declare(declaration("s1", "person-1"))
    owner.record_experience(
        Experience(
            "s1",
            "e1",
            1,
            "actor-0",
            ExperiencePermissions(True, True),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(
        LabelArrival(
            "label-s1",
            "s1",
            "e1",
            3,
            "actor-0",
            ExperiencePermissions(False, True),
        )
    )
    clock.advance_to(3)
    ledger.train_ready()
    gate.pause()
    builder = Builder(old._candidate)
    controller = CandidateCheckpointController(
        owner._shared,
        build_learner=builder,
        state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
        policy_digest=lambda _: "a" * 64,
    )
    kwargs: dict[str, Any] = {} if default else {"retain_capture_history": retain}
    manager = ManagedReplayCheckpoints(ledger, GRAPH_PORTS, builder_source=builder_source, **kwargs)
    token = manager.capture(controller)
    pending = controller._pending[token]
    before_restore = ledger.accounting()
    assert life._copy_budget is not None
    raw_before_restore = life._copy_budget._charged
    original_rows = tuple(ledger._rows.values())
    new = manager.restore(controller, token) if restore else old
    old_learner = cast(RestoredLearner, old._candidate)
    new_learner = cast(RestoredLearner, new._candidate)
    return SimpleNamespace(
        owner=owner,
        life=life,
        old=old,
        new=new,
        runtime=new,
        ledger=ledger,
        controller=controller,
        manager=manager,
        token=token,
        pending=pending,
        view=pending.view,
        old_model=old_learner.model,
        new_model=new_learner.model,
        builder=builder,
        clock=clock,
        gate=gate,
        seconds=seconds,
        budget=budget,
        before_restore=before_restore,
        raw_before_restore=raw_before_restore,
        original_rows=original_rows,
        work=old_learner.work,
    )


def inventory(groups, limits):
    found = {}
    for group in groups:
        for item in group.references.models + group.references.snapshots:
            root = item.model if type(item) is RestoredLearner else getattr(item, "state", item)
            found[id(root)] = root, rows(root, limits.max_nodes)
    return tuple(found.values())


@contextmanager
def leased(graph, *, checkpoints=DEFAULT, copies=DEFAULT, groups=None):
    with _lease_lifecycle_sources(graph.owner, limits=LifecycleCaptureLimits(64, 128)) as (
        _,
        enrolled,
        _,
    ):
        original = tuple(
            OwnedPayloadGroup(
                PayloadHolderMetadata(item.enrollment, item.kind, True),
                graph.life._registry._holders[item.enrollment][1]()._payload_references(),
            )
            for item in enrolled
            if item.ready is True
        )
        supplied = original if groups is None else groups(original)
        options: dict[str, Any] = {
            "checkpoints": graph.manager if checkpoints is DEFAULT else checkpoints,
        }
        if copies is not DEFAULT:
            options["copies"] = copies
        with lease_replay_capture(
            graph.owner,
            supplied,
            BOUNDS,
            inventory,
            graph.ledger,
            **options,
        ) as check:
            yield check


def assert_no_work(graph, before):
    assert graph.work == before and graph.builder.calls == 1


@pytest.mark.parametrize("explicit", (False, True))
def test_should_capture_actual_retired_and_current_roles_without_any_learner_operation(explicit):
    graph = setup()
    old_row, new_row = graph.old_model._replay_memory[0], graph.new_model._replay_memory[0]
    assert graph.old._retired and not graph.new._retired
    assert graph.owner._shared._runtime is graph.new
    assert graph.controller._models == [graph.new._candidate]
    assert graph.builder.source is graph.old._candidate
    assert graph.old._inbox._historical_completed_updates == 1
    assert graph.old._budget is graph.new._budget is graph.budget
    assert graph.old._inbox._clock is graph.new._inbox._clock is graph.clock
    assert old_row is not new_row and old_row.input_batch is not new_row.input_batch
    assert graph.original_rows[0].snapshot() is old_row
    assert graph.ledger._rows[id(new_row)].snapshot() is new_row
    assert not graph.controller._pending and graph.budget.updates_completed == 1
    before_state = pickle.dumps(
        (
            vars(graph.old_model),
            vars(graph.new_model),
            graph.old._inbox._borrow_cursor(),
            graph.new._inbox._borrow_cursor(),
        )
    )
    identities = (
        id(graph.old._actor),
        id(graph.new._actor),
        id(graph.budget),
        id(graph.clock),
        id(graph.life._copy_budget),
        id(graph.life._registry),
        id(graph.controller._models),
        id(graph.controller._pending),
        id(graph.ledger._admission),
        id(graph.builder),
    )
    before, raw, work = (
        graph.ledger.accounting(),
        graph.life._copy_budget._charged,
        dict(graph.work),
    )
    kwargs = {"copies": graph.manager._copies} if explicit else {}
    with leased(graph, **kwargs) as check:
        check()
        check()
        with pytest.raises(PayloadOwnershipBusy):
            graph.manager.capture(graph.controller)
    after = graph.ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert (
        after.records_created == before.records_created
        and after.live_records == before.live_records
    )
    assert graph.life._copy_budget._charged == raw
    assert (
        pickle.dumps(
            (
                vars(graph.old_model),
                vars(graph.new_model),
                graph.old._inbox._borrow_cursor(),
                graph.new._inbox._borrow_cursor(),
            )
        )
        == before_state
    )
    assert identities == (
        id(graph.old._actor),
        id(graph.new._actor),
        id(graph.budget),
        id(graph.clock),
        id(graph.life._copy_budget),
        id(graph.life._registry),
        id(graph.controller._models),
        id(graph.controller._pending),
        id(graph.ledger._admission),
        id(graph.builder),
    )
    assert_no_work(graph, work)


@pytest.mark.parametrize("bad", (None, 0, 1, "true", object()))
def test_should_refuse_nonboolean_birth_option_before_owner_hooks(monkeypatch, bad):
    ledger = object.__new__(ManagedReplayOrigins)
    monkeypatch.setattr(ManagedReplayOrigins, "_original_owner", forbidden)
    with pytest.raises(ValueError):
        ManagedReplayCheckpoints(
            ledger,
            GRAPH_PORTS,
            builder_source=builder_source,
            retain_capture_history=bad,
        )


def test_should_charge_bounded_publication_metadata_without_an_extra_row_or_invocation():
    graph = setup(restore=False)
    charges: list[tuple[int, int, int, int]] = []
    admission = graph.ledger._admission
    original = admission._charge

    def charge(records, payload, metadata, live):
        charges.append((records, payload, metadata, live))
        return original(records, payload, metadata, live)

    admission._charge = charge
    new = graph.manager.restore(graph.controller, graph.token)
    publication_charges = [entry for entry in charges if entry[0] == entry[1] == 0]
    assert len(publication_charges) == 1
    metadata = publication_charges[0][2]
    assert 1024 < metadata <= admission.limits.max_metadata_bytes
    assert graph.owner._shared._runtime is new and graph.old._retired
    storage = graph.manager._publication_storage
    assert (
        storage[0] is graph.manager._publications
        and storage[1] is graph.manager._publication_sealed
    )
    record = storage[0][id(graph.old_model)]
    assert storage[1][id(graph.old_model)] is record
    assert (
        record.charged_bytes
        == metadata
        == 1024
        + len(
            json.dumps(
                record.metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf8")
        )
    )
    assert all(a is b for a, b in zip(record.old_rows, graph.original_rows))
    assert len(record.old_rows) == len(graph.original_rows) == 1
    current_row = new._candidate.model._replay_memory[0]
    assert record.current_rows == (graph.ledger._rows[id(current_row)],)
    references = dict(record.references)
    assert references["old_runtime"]() is graph.old
    assert references["current_runtime"]() is new
    assert (
        graph.ledger.accounting().metadata_bytes_charged
        >= graph.before_restore.metadata_bytes_charged + metadata
    )


@pytest.mark.parametrize("default", (False, True))
def test_should_keep_optout_restore_unchanged_and_refuse_retroactive_capture(default):
    graph = setup(retain=False, default=default, restore=False)
    metadata_charges: list[int] = []
    original = graph.ledger._admission._charge

    def charge(records, invocations, size, live):
        if records == invocations == 0:
            metadata_charges.append(size)
        return original(records, invocations, size, live)

    graph.ledger._admission._charge = charge
    graph.new = graph.manager.restore(graph.controller, graph.token)
    assert graph.owner._shared._runtime is graph.new and graph.old._retired
    assert metadata_charges == []
    assert not graph.manager._publications and not graph.manager._publication_sealed
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("default restore acquired retained capture authority")


@pytest.mark.parametrize("mode", ("default", "copies_only", "foreign_copies", "foreign_manager"))
def test_should_require_the_actual_opted_in_published_pair(mode):
    graph = setup()
    kwargs: dict[str, Any] = {}
    if mode == "default":
        kwargs["checkpoints"] = None
    elif mode == "copies_only":
        kwargs.update(checkpoints=None, copies=graph.manager._copies)
    elif mode == "foreign_copies":
        kwargs["copies"] = ManagedReplayCopies(graph.ledger, builder_source=builder_source)
    elif mode == "foreign_manager":
        other = setup()
        kwargs["checkpoints"] = other.manager
    with pytest.raises(ValueError):
        with leased(graph, **kwargs):
            pytest.fail("unwitnessed publication acquired capture authority")


@pytest.mark.parametrize(
    "change",
    (
        "old_wrapper",
        "new_wrapper",
        "old_model",
        "new_model",
        "old_row",
        "new_row",
        "old_payload",
        "new_payload",
        "old_receipt",
        "new_receipt",
        "old_content",
        "new_content",
        "builder",
        "policy",
        "probe",
        "models",
        "attempt",
        "budget",
        "retired_flag",
        "revoked",
        "optout",
        "age",
        "elapsed",
        "raw_limits",
    ),
)
def test_should_refuse_changed_original_pair_content_authority_and_paid_controls(change):
    graph = setup()
    old, new, cp = graph.old, graph.new, graph.controller
    if change in ("old_wrapper", "new_wrapper"):
        runtime = old if change == "old_wrapper" else new
        runtime._candidate = RestoredLearner(runtime._candidate.model, graph.work)
    elif change in ("old_model", "new_model"):
        runtime = old if change == "old_model" else new
        runtime._candidate.model = deepcopy(runtime._candidate.model)
    elif change in ("old_row", "new_row"):
        model = graph.old_model if change == "old_row" else graph.new_model
        model._replay_memory[0] = deepcopy(model._replay_memory[0])
    elif change in ("old_payload", "new_payload"):
        model = graph.old_model if change == "old_payload" else graph.new_model
        row = model._replay_memory[0]
        object.__setattr__(row, "input_batch", replace(row.input_batch))
    elif change in ("old_receipt", "new_receipt"):
        runtime = old if change == "old_receipt" else new
        key = ("e1", "s1")
        runtime._inbox._applied[key] = replace(runtime._inbox._applied[key])
    elif change in ("old_content", "new_content"):
        (graph.old_model if change == "old_content" else graph.new_model).scalar += 1
    elif change == "builder":
        cp._build = Builder(old._candidate)
    elif change == "policy":
        cp._policy = lambda _: "a" * 64
    elif change == "probe":
        cp._digest = lambda _: "b" * 64
    elif change == "models":
        cp._models = list(cp._models)
    elif change == "attempt":
        cp._attempts += 1
    elif change == "budget":
        graph.budget.updates_completed += 1
    elif change == "retired_flag":
        old._retired = False
    elif change == "revoked":
        graph.owner._revoked_keys.add(("e1", "s1"))
    elif change == "optout":
        graph.owner._opted_out.add("person-1")
    elif change == "age":
        graph.clock.advance_to(122)
    elif change == "elapsed":
        graph.seconds[0] = 1000.0
    else:
        graph.life._copy_budget._limits = PayloadCopyLimits(4096)
    work, raw = dict(graph.work), graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("changed original publication reached capture")
    assert_no_work(graph, work)
    assert graph.life._copy_budget._charged == raw


@pytest.mark.parametrize("which", ("old", "new"))
@pytest.mark.parametrize("role", ("actor", "promotion", "snapshot", "other_checkpoint"))
def test_should_refuse_every_additional_unqualified_appearance_of_either_learner(which, role):
    graph = setup()
    learner = (graph.old if which == "old" else graph.new)._candidate

    def alias(original):
        if role == "actor":
            return tuple(
                replace(group, references=replace(group.references, models=(learner,)))
                if group.holder.kind == "actor"
                else group
                for group in original
            )
        enrollment = max(group.holder.enrollment for group in original) + 1
        holder = PayloadHolderMetadata(
            enrollment, "promotion" if role == "promotion" else "checkpoint", True
        )
        references = PayloadReferences(
            **{"snapshots" if role == "snapshot" else "models": (learner,)}
        )
        return (*original, OwnedPayloadGroup(holder, references))

    with pytest.raises(ValueError):
        with leased(graph, groups=alias):
            pytest.fail("an allowed role hid an additional foreign appearance")


@pytest.mark.parametrize("which", ("old_content", "controller"))
def test_should_recheck_late_changes_and_keep_the_existing_admission_spent(which):
    graph = setup()
    before, raw, work = (
        graph.ledger.accounting(),
        graph.life._copy_budget._charged,
        dict(graph.work),
    )
    with leased(graph) as check:
        if which == "old_content":
            graph.old_model.scalar += 1
        else:
            graph.controller._models = list(graph.controller._models)
        with pytest.raises(ValueError):
            check()
    after = graph.ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert after.records_created == before.records_created
    assert graph.life._copy_budget._charged == raw
    assert_no_work(graph, work)


def test_should_bound_all_checks_to_the_original_eight_rounds():
    graph = setup()
    with leased(graph) as check:
        for _ in range(7):
            check()
        with pytest.raises(ValueError, match="allowance|exhaust|limit"):
            check()


def test_should_refuse_a_joined_contender_without_model_work_or_lease_escape():
    graph = setup()
    errors: list[ValueError] = []
    with leased(graph) as check:

        def contend():
            try:
                check()
            except ValueError as error:
                errors.append(error)

        thread = Thread(target=contend)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive() and len(errors) == 1
        check()


def test_should_release_weak_published_roots_after_the_original_lease_expires():
    graph = setup()
    old, new, owner = ref(graph.old), ref(graph.new), ref(graph.owner)
    with leased(graph) as check:
        check()
    del graph
    gc.collect()
    assert old() is new() is owner() is None
    with pytest.raises(ValueError, match="expired"):
        check()


@pytest.mark.parametrize(
    "change",
    (
        "publications",
        "sealed",
        "storage",
        "unsealed_record",
        "extra_field",
        "flag",
        "capacity",
    ),
)
def test_should_refuse_changed_birth_controls_and_original_sealed_record_storage(change):
    graph = setup()
    manager = graph.manager
    key = id(graph.old_model)
    if change == "publications":
        manager._publications = dict(manager._publications)
    elif change == "sealed":
        manager._publication_sealed = dict(manager._publication_sealed)
    elif change == "storage":
        manager._publication_storage = (manager._publication_sealed, manager._publications)
    elif change == "unsealed_record":
        manager._publications[key] = replace(manager._publications[key])
    elif change == "extra_field":
        object.__setattr__(manager._publications[key], "foreign", True)
    elif change == "flag":
        manager._retain_capture_history = False
    else:
        manager._publication_capacity += 1
    work = dict(graph.work)
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("changed publication storage acquired capture permission")
    assert_no_work(graph, work)


def test_should_refuse_exhausted_original_metadata_without_installing_a_publication():
    graph = setup(restore=False)
    admission = graph.ledger._admission
    before = graph.ledger.accounting()
    remaining = admission.limits.max_metadata_bytes - before.metadata_bytes_charged
    # Spend the existing original allowance rather than replacing its limits or
    # injecting a fresh admission object to force a successful publication.
    admission._charge(0, 0, remaining, before.live_records)
    with pytest.raises(ValueError, match="metadata|limit|bound|exhaust"):
        graph.manager.restore(graph.controller, graph.token)
    assert graph.owner._shared._runtime is graph.old and not graph.old._retired
    assert not graph.manager._publications and not graph.manager._publication_sealed
    assert graph.ledger.accounting().metadata_bytes_charged == admission.limits.max_metadata_bytes
    assert graph.ledger.accounting().records_created == before.records_created


def test_should_keep_consumed_caller_owned_view_outside_issued_snapshot_authority():
    graph = setup()

    def retained_view(original):
        return tuple(
            replace(group, references=replace(group.references, snapshots=(graph.view,)))
            if group.holder.kind == "checkpoint"
            else group
            for group in original
        )

    with pytest.raises(ValueError):
        with leased(graph, groups=retained_view):
            pytest.fail("a consumed external view revived an issued checkpoint")
    assert not graph.controller._pending


@pytest.mark.parametrize("which", ("old", "current"))
def test_should_refuse_mutated_learner_policy_even_when_the_model_and_probe_are_unchanged(which):
    graph = setup()
    learner = (graph.old if which == "old" else graph.new)._candidate
    assert learner._learning_rate == 0.01
    learner._learning_rate = 0.02
    work, raw = dict(graph.work), graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("a fixed policy probe concealed a changed original learner field")
    assert graph.life._copy_budget._charged == raw
    assert_no_work(graph, work)


def test_should_revalidate_a_model_port_mutation_during_paid_publication_preparation(monkeypatch):
    phase = {"preparing": False, "mutations": 0}

    def model_reference(learner):
        if phase["preparing"] and phase["mutations"] == 0:
            learner.model.scalar += 1
            phase["mutations"] += 1
        return learner.model

    graph = setup(restore=False, model_reference=model_reference)
    original = ManagedReplayCheckpoints._retained_publication

    def prepare(manager, operation, prepared, records, holders):
        phase["preparing"] = True
        try:
            return original(manager, operation, prepared, records, holders)
        finally:
            phase["preparing"] = False

    monkeypatch.setattr(ManagedReplayCheckpoints, "_retained_publication", prepare)
    before, raw = graph.ledger.accounting(), graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        graph.manager.restore(graph.controller, graph.token)
    assert phase["mutations"] == 1 and graph.old_model.scalar == 2
    assert graph.owner._shared._runtime is graph.old and not graph.old._retired
    assert graph.old._inbox._historical_completed_updates is None
    assert graph.controller._pending[graph.token] is graph.pending
    assert graph.controller._attempts == 1 and len(graph.controller._models) == 1
    assert not graph.manager._publications and not graph.manager._publication_sealed
    assert graph.ledger.accounting().metadata_bytes_charged > before.metadata_bytes_charged
    assert graph.life._copy_budget._charged > raw


def test_should_preserve_the_original_eight_invocation_refusal_without_refund_or_publication():
    graph = setup(restore=False, limits=LIMITS)
    anchors, records = graph.ledger._anchors, dict(graph.ledger._rows)
    before, raw = graph.ledger.accounting(), graph.life._copy_budget._charged
    assert graph.ledger._admission.limits is LIMITS and LIMITS.max_invocations == 8
    with pytest.raises(ValueError, match="limit|allowance|exhaust"):
        graph.manager.restore(graph.controller, graph.token)
    after = graph.ledger.accounting()
    assert after.invocations_started == 8
    assert after.metadata_bytes_charged > before.metadata_bytes_charged
    assert graph.life._copy_budget._charged > raw
    assert graph.owner._shared._runtime is graph.old and not graph.old._retired
    assert graph.ledger._anchors is anchors
    assert graph.ledger._rows.keys() == records.keys()
    assert all(graph.ledger._rows[key] is record for key, record in records.items())
    assert graph.old._inbox._historical_completed_updates is None
    assert graph.controller._pending[graph.token] is graph.pending
    assert graph.controller._attempts == 1 and len(graph.controller._models) == 1
    assert not graph.manager._publications and not graph.manager._publication_sealed
