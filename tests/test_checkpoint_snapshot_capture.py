"""Issued snapshot witnesses from real app graph copies with scalar fake ML.

The exact CPC shell is deliberately not a native-valid network. Its known
records exercise the original manager's observed copy chain without any native
constructor, method, array or restoration/publication claim.
"""

from collections import deque
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256
import gc
import pickle
from threading import Lock, Thread
from types import SimpleNamespace
from typing import Any
from weakref import ref

import pytest

import src.adapters.numpy_composite_capture as numpy_adapter
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, CandidateCheckpointView
from src.app.experience_inbox import ExperienceInbox
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
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
    ReplaySnapshot,
)
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.managed_composite_state import CompositeCaptureLimits
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.native_state_copy import copy_native_state
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import (
    OwnedPayloadGroup,
    PayloadHolderMetadata,
    PayloadOwnershipLimits,
    PayloadReferences,
)
from src.core.replay_graph_origin import ReplayGraphPorts
from src.core.replay_origin import ReplayOriginPorts
from src.core.replay_write_origin import begin_replay_write
from src.core.resource_sharing import SharingLimits
from test_managed_experience import declaration
from test_managed_replay_origins import LIMITS, WINDOW


BOUNDS = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 64, 16, 65536, 32)
DEFAULT = object()
WORK = dict(forks=0, trains=0, snapshots=0)


def shell():
    model = object.__new__(CircadianPredictiveCodingNetwork)
    model.__dict__.update(_replay_memory=deque(), scalar=0)
    return model


class ScalarLearner:
    def __init__(self, model=None):
        self.model = shell() if model is None else model

    def fork(self):
        WORK["forks"] += 1
        return ScalarLearner(deepcopy(self.model))

    def train_batch(self, features, targets):
        WORK["trains"] += 1
        window = begin_replay_write(
            self.model, features, targets, self.model._replay_memory, 1, "rows"
        )
        if window is not None:
            window.before_copy(0, 1)
        row = ReplaySnapshot(deepcopy(features), deepcopy(targets), 1.0, 0.5)
        if window is not None:
            window.copied(row)
        self.model._replay_memory.append(row)
        self.model.__dict__["scalar"] += 1
        if window is not None:
            window.finish(self.model._replay_memory)
        return TrainingDiagnostic("scalar_shell_fixture", 1.0)

    def snapshot_state(self):
        WORK["snapshots"] += 1
        return CircadianNetworkSnapshot(
            2,
            3,
            (2,),
            2,
            2,
            CircadianConfig(),
            copy_native_state(vars(self.model), producer=self.model, kind="native_snapshot"),
        )

    def predict(self, payload):
        pytest.fail("prediction is outside scalar fixture scope")

    def restore_state(self, state):
        pytest.fail("native restoration is outside scalar fixture scope")


class Builder:
    def __init__(self, source):
        self.source = source

    def __call__(self, state):
        pytest.fail("snapshot capture must never prepare a learner")


def builder_source(builder):
    if type(builder) is not Builder:
        raise ValueError("exact scalar builder required")
    return builder.source


def payload_bits(value):
    if type(value) is not ExperiencePermissions or vars(value).keys() != {
        "training",
        "replay",
        "evaluation",
    }:
        raise ValueError("scalar payload requires the known exact permission record")
    result = value.training, value.replay, value.evaluation
    if any(type(flag) is not bool for flag in result):
        raise ValueError("scalar payload flags must be exact booleans")
    return result


def payload_bytes(value):
    payload_bits(value)
    return 16


def row_payloads(row):
    if type(row) is not ReplaySnapshot:
        raise ValueError("exact observed replay snapshot required")
    payload_bits(row.input_batch)
    payload_bits(row.target_batch)
    return row.input_batch, row.target_batch


def rows(root, maximum):
    if type(root) is CircadianNetworkSnapshot:
        state = root.state
    elif type(root) is CircadianPredictiveCodingNetwork:
        state = vars(root)
    elif type(root) is dict:
        state = root
    else:
        raise ValueError("unsupported scalar graph root")
    memory = state.get("_replay_memory")
    if type(memory) is not deque or len(memory) > maximum:
        raise ValueError("scalar replay row bound exceeded")
    for row in memory:
        row_payloads(row)
    return tuple(memory)


def fingerprint(row, maximum):
    features, targets = row_payloads(row)
    if maximum < 32:
        raise ValueError("scalar row exceeds modeled byte allowance")
    return 32, sha256(repr((payload_bits(features), payload_bits(targets))).encode()).hexdigest()


def copy_bytes(features, targets, start, count, maximum):
    payload_bits(features)
    payload_bits(targets)
    if (start, count) != (0, 1) or maximum < 32:
        raise ValueError("scalar copy range exceeds original scope")
    return 32


def graph_bytes(root, maximum, max_bytes):
    size = 32 * len(rows(root, maximum))
    if size > max_bytes:
        raise ValueError("scalar graph exceeds modeled byte allowance")
    return size


def footprint(learner):
    count = len(rows(learner.model, 4))
    return ReplayPayloadErasure(count, count, 32 * count)


def model_reference(learner):
    return learner.model


PORTS = ReplayOriginPorts(model_reference, rows, copy_bytes, row_payloads, fingerprint)
GRAPH_PORTS = ReplayGraphPorts(rows, graph_bytes)


def setup(*, issued=True, window=WINDOW):
    seconds = [0.0]
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), lambda: seconds[0])
    runtime = ActorShadowRuntime(
        ScalarLearner(),
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
    )
    gate = ServingPriorityGate(SharingLimits(1, 8, 1), resource_available=lambda: True)
    owner = ManagedExperienceOwner(
        ResourceSharedRuntime(runtime, gate), limits=LifecycleLimits(4, 4)
    )
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
        native_erase=lambda _: pytest.fail("cleanup outside scalar scope"),
        measure_auxiliary_bytes=lambda _: 0,
        measure_checkpoint_bytes=lambda view: (
            graph_bytes(view.state, 4, 4096)
            + 16 * (len(view.inbox.experiences) + len(view.inbox.labels))
        ),
        native_growth_bytes=lambda *_: 32,
        prepare_model_bytes=lambda *_: pytest.fail("preparation outside scalar scope"),
        prediction_cache_bytes=lambda *_: pytest.fail("cache outside scalar scope"),
    )
    ledger = ManagedReplayOrigins(owner, PORTS, LIMITS, window)
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
        LabelArrival("label-s1", "s1", "e1", 3, "actor-0", ExperiencePermissions(False, True))
    )
    clock.advance_to(3)
    ledger.train_ready()
    gate.pause()
    builder = Builder(runtime._candidate)
    controller = CandidateCheckpointController(
        owner._shared,
        build_learner=builder,
        state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
        policy_digest=lambda _: "a" * 64,
    )
    manager = ManagedReplayCheckpoints(ledger, GRAPH_PORTS, builder_source=builder_source)
    token = manager.capture(controller) if issued else controller.capture()
    pending = controller._pending[token]
    return SimpleNamespace(
        owner=owner,
        ledger=ledger,
        runtime=runtime,
        clock=clock,
        life=life,
        seconds=seconds,
        controller=controller,
        manager=manager,
        token=token,
        pending=pending,
        view=pending.view,
        state=pending.view.state,
        cursor=pending.view.inbox,
    )


def inventory(groups, limits):
    entries = {}
    for group in groups:
        for item in group.references.models + group.references.snapshots:
            if type(item) is CandidateCheckpointView:
                root = item.state
            elif type(item) is ScalarLearner:
                root = item.model
            else:
                root = item
            entries[id(root)] = root, rows(root, limits.max_nodes)
    return tuple(entries.values())


@contextmanager
def leased(graph, *, checkpoints=DEFAULT, copies=DEFAULT, groups=None, limits=BOUNDS):
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
        options = {"checkpoints": graph.manager if checkpoints is DEFAULT else checkpoints}
        if copies is not DEFAULT:
            options["copies"] = copies
        with lease_replay_capture(
            graph.owner, supplied, limits, inventory, graph.ledger, **options
        ) as check:
            yield check


@pytest.mark.parametrize("explicit_copies", (False, True))
def test_should_qualify_only_the_actual_issued_snapshot_with_one_existing_metadata_charge(
    explicit_copies,
):
    graph = setup()
    before, raw, work = graph.ledger.accounting(), graph.life._copy_budget._charged, dict(WORK)
    kwargs = {"copies": graph.manager._copies} if explicit_copies else {}
    with leased(graph, **kwargs) as check:
        assert graph.manager._gate.locked() and graph.manager._copies._gate.locked()
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
    assert graph.life._copy_budget._charged == raw and WORK == work
    assert graph.controller._pending[graph.token] is graph.pending
    assert graph.pending.view is graph.view and graph.view.state is graph.state
    original = graph.runtime._candidate.model._replay_memory[0]
    copied = graph.state.state["_replay_memory"][0]
    assert copied is not original and copied.input_batch is not original.input_batch
    assert copied.target_batch is not original.target_batch
    assert (
        graph.cursor.experiences[0].features
        is not graph.runtime._inbox._experiences[("e1", "s1")].features
    )
    assert graph.runtime._budget.updates_completed == 1 and not graph.controller._models


@pytest.mark.parametrize("mode", ("default", "constructor_only", "unissued"))
def test_should_keep_snapshot_replay_refused_without_its_actual_issuing_manager(mode):
    graph = setup(issued=mode != "unissued")
    kwargs: dict[str, Any] = {}
    if mode == "default":
        kwargs["checkpoints"] = None
    elif mode == "constructor_only":
        kwargs.update(checkpoints=None, copies=graph.manager._copies)
    with pytest.raises(ValueError):
        with leased(graph, **kwargs):
            pytest.fail("unissued snapshot acquired capture permission")


@pytest.mark.parametrize("wrong", ("object", "subclass", "foreign", "copies"))
def test_should_refuse_nonoriginal_manager_or_different_explicit_copies(wrong):
    graph = setup()
    options: dict[str, Any] = {"checkpoints": object()}
    other = None
    if wrong == "subclass":

        class Derived(ManagedReplayCheckpoints):
            pass

        options["checkpoints"] = Derived(graph.ledger, GRAPH_PORTS, builder_source=builder_source)
    elif wrong == "foreign":
        other = setup()
        options["checkpoints"] = other.manager
    elif wrong == "copies":
        options = {"copies": ManagedReplayCopies(graph.ledger, builder_source=builder_source)}
    raw = graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph, **options):
            pytest.fail("foreign manager authority reached copy")
    assert graph.life._copy_budget._charged == raw
    assert other is None or other.owner is not graph.owner


@pytest.mark.parametrize(
    "role", ("actor", "candidate", "promotion", "another_checkpoint", "direct_snapshot")
)
def test_should_refuse_every_nonoriginal_snapshot_appearance_even_if_original_still_retains_it(
    role,
):
    graph = setup()

    def aliased(original):
        cp = next(group for group in original if group.holder.kind == "checkpoint")
        if role in ("actor", "candidate"):
            return tuple(
                replace(group, references=replace(group.references, models=(graph.view,)))
                if group.holder.kind == role
                else group
                for group in original
            )
        if role == "direct_snapshot":
            return tuple(
                replace(
                    group, references=replace(group.references, snapshots=(graph.view, graph.state))
                )
                if group is cp
                else group
                for group in original
            )
        extra = OwnedPayloadGroup(
            PayloadHolderMetadata(
                max(g.holder.enrollment for g in original) + 1,
                "checkpoint" if role == "another_checkpoint" else "promotion",
                True,
            ),
            PayloadReferences(snapshots=(graph.view,)),
        )
        return (*original, extra)

    with pytest.raises(ValueError):
        with leased(graph, groups=aliased):
            pytest.fail("issued snapshot escaped its exact original snapshots role")
    assert graph.controller._pending[graph.token].view is graph.view


@pytest.mark.parametrize(
    "change",
    (
        "token",
        "pending",
        "view",
        "state",
        "cursor",
        "state_content",
        "row",
        "payload",
        "row_metadata",
        "inbox_payload",
        "inbox_receipt",
        "current_receipt",
        "builder",
        "policy",
        "probe",
        "models",
        "attempt",
        "capacity",
        "budget",
        "manager_gate",
        "copies_gate",
        "ports",
        "content_limits",
        "copies",
        "revoked",
        "optout",
        "age",
        "elapsed",
        "paid_raw",
        "raw_limits",
        "witness",
    ),
)
def test_should_refuse_changed_actual_membership_content_origin_history_and_original_controls(
    change,
):
    graph = setup()
    manager, cp, pending = graph.manager, graph.controller, graph.pending
    row = graph.state.state["_replay_memory"][0]
    witness = manager._checkpoints[id(graph.token)]
    if change == "token":
        del cp._pending[graph.token]
        cp._pending[copy(graph.token)] = pending
    elif change == "pending":
        cp._pending[graph.token] = copy(pending)
    elif change == "view":
        pending.view = replace(graph.view)
    elif change == "state":
        pending.view = replace(graph.view, state=replace(graph.state))
    elif change == "cursor":
        pending.view = replace(graph.view, inbox=replace(graph.cursor))
    elif change == "state_content":
        graph.state.state["scalar"] += 1
    elif change == "row":
        graph.state.state["_replay_memory"][0] = deepcopy(row)
    elif change == "payload":
        object.__setattr__(row, "input_batch", replace(row.input_batch))
    elif change == "row_metadata":
        witness.rows[0].data = replace(witness.rows[0].data, row_start=1)
    elif change == "inbox_payload":
        source = graph.cursor.experiences[0]
        object.__setattr__(source, "features", replace(source.features))
    elif change == "inbox_receipt":
        object.__setattr__(graph.cursor.applied[0], "update_number", 2)
    elif change == "current_receipt":
        key = ("e1", "s1")
        graph.runtime._inbox._applied[key] = deepcopy(graph.runtime._inbox._applied[key])
    elif change == "builder":
        cp._build = Builder(graph.runtime._candidate)
    elif change == "policy":
        cp._policy = lambda _: "a" * 64
    elif change == "probe":
        cp._digest = lambda _: "b" * 64
    elif change == "models":
        cp._models = list(cp._models)
    elif change == "attempt":
        cp._attempts = True
    elif change == "capacity":
        cp._limit += 1
    elif change == "budget":
        graph.view.budget_state["updates_completed"] = 2
    elif change == "manager_gate":
        manager._gate = Lock()
    elif change == "copies_gate":
        manager._copies._gate = Lock()
    elif change == "ports":
        manager._graph_ports = ReplayGraphPorts(rows, graph_bytes)
    elif change == "content_limits":
        manager._content_limits = CheckpointContentLimits(**vars(manager._content_limits))
    elif change == "copies":
        manager._copies = ManagedReplayCopies(graph.ledger, builder_source=builder_source)
    elif change == "revoked":
        graph.owner._revoked_keys.add(("e1", "s1"))
    elif change == "optout":
        graph.owner._opted_out.add("person-1")
    elif change == "age":
        graph.clock.advance_to(122)
    elif change == "elapsed":
        graph.seconds[0] = 1000.0
    elif change == "paid_raw":
        graph.life._copy_budget._charged -= 1
    elif change == "raw_limits":
        graph.life._copy_budget._limits = PayloadCopyLimits(4096)
    else:
        manager._checkpoints[id(graph.token)] = object()
    raw, work = graph.life._copy_budget._charged, dict(WORK)
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("changed issued history reached downstream copy")
    assert graph.life._copy_budget._charged == raw and WORK == work


@pytest.mark.parametrize("change", ("content", "pending"))
def test_should_recheck_late_mutation_without_refunding_the_existing_capture_admission(change):
    graph = setup()
    before, raw = graph.ledger.accounting(), graph.life._copy_budget._charged
    with leased(graph) as check:
        if change == "content":
            graph.state.state["scalar"] += 1
        else:
            graph.controller._pending[graph.token] = copy(graph.pending)
        with pytest.raises(ValueError):
            check()
    after = graph.ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert (
        after.records_created == before.records_created and graph.life._copy_budget._charged == raw
    )


@pytest.mark.parametrize("which", ("manager", "copies"))
def test_should_refuse_busy_original_nonblocking_gates_without_copy(which):
    graph = setup()
    authority = graph.manager if which == "manager" else graph.manager._copies
    with authority._exclusive():
        with pytest.raises(PayloadOwnershipBusy):
            with leased(graph):
                pytest.fail("busy manager reached copy")


def test_should_limit_checks_to_original_thread_and_live_bounded_context():
    graph = setup()
    errors: list[BaseException] = []
    with leased(graph) as check:

        def foreign():
            try:
                check()
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=foreign)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], ValueError)
        for _ in range(7):
            check()
        with pytest.raises(ValueError, match="allowance"):
            check()
    with pytest.raises(ValueError, match="expired"):
        check()


@pytest.mark.parametrize("bound", ("manager", "rows", "roles"))
def test_should_bound_original_dictionaries_rows_and_role_references(bound):
    graph = setup()
    witness = graph.manager._checkpoints[id(graph.token)]
    groups: Any = None
    if bound == "manager":
        graph.manager._checkpoints = {number: witness for number in range(65)}
    elif bound == "rows":
        object.__setattr__(witness, "rows", witness.rows * 65)
    else:

        def overflowing(original):
            return tuple(
                replace(group, references=replace(group.references, snapshots=(graph.view,) * 65))
                if group.holder.kind == "checkpoint"
                else group
                for group in original
            )

        groups = overflowing

    with pytest.raises(ValueError):
        with leased(graph, groups=groups):
            pytest.fail("unbounded witness reached copy")


def test_should_reuse_one_original_cursor_across_all_eight_checks_with_pair_capacity_two(
    monkeypatch,
):
    graph = setup(window=replace(WINDOW, max_retained_snapshots=2))
    original_borrow, original_inbox = (
        ExperienceInbox._borrow_cursor,
        ManagedReplayCheckpoints._original_inbox,
    )
    borrowed: list[Any] = []
    observations: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []

    def borrow(inbox):
        cursor = original_borrow(inbox)
        if inbox is graph.runtime._inbox:
            # Keep every fresh cursor alive so allocator ID reuse cannot hide
            # accidental growth of per-operation partition maps.
            assert all(cursor is not old for old in borrowed)
            borrowed.append(cursor)
        return cursor

    def validate(manager, operation, cursor):
        result = original_inbox(manager, operation, cursor)
        assert operation.pair_capacity == 2
        observations.append(
            (
                id(operation),
                id(cursor),
                tuple(operation.inbox_erasures),
                tuple(operation.inbox_untrained),
            )
        )
        return result

    monkeypatch.setattr(ExperienceInbox, "_borrow_cursor", borrow)
    monkeypatch.setattr(ManagedReplayCheckpoints, "_original_inbox", validate)
    with leased(graph) as check:
        for _ in range(7):
            check()
    assert len(observations) == 8 and len(borrowed) == 1
    assert len({(op, cursor) for op, cursor, _, _ in observations}) == 1
    assert all(
        erased == untrained == (id(borrowed[0]),) for _, _, erased, untrained in observations
    )


def test_should_release_original_roots_when_only_expired_check_is_retained():
    graph = setup()
    owner, state = ref(graph.owner), ref(graph.state)
    with leased(graph) as check:
        check()
    del graph
    gc.collect()
    assert owner() is None and state() is None
    with pytest.raises(ValueError, match="expired"):
        check()


@pytest.mark.parametrize("enabled", (False, True))
def test_should_forward_optional_manager_without_replacing_known_owner_and_controls(
    monkeypatch, enabled
):
    owner, ledger, result = object(), object(), object()
    manager = object() if enabled else None
    calls: list[dict[str, Any]] = []

    def capture(value, **kwargs):
        assert value is owner
        calls.append(kwargs)
        return result

    monkeypatch.setattr(numpy_adapter, "capture_managed_composite", capture)
    assert (
        numpy_adapter.capture_numpy_managed_composite(
            owner, limits=BOUNDS, replay_origins=ledger, replay_checkpoints=manager
        )
        is result
    )
    assert len(calls) == 1 and calls[0]["limits"] is BOUNDS
    assert calls[0]["replay_origins"] is ledger and calls[0]["replay_checkpoints"] is manager
