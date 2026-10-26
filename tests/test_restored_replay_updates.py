"""Observed numeric DATA updates with real managed proofs and fake learning."""

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

import numpy as np
import pytest

from src.adapters.numpy_replay_origins import (
    replay_copy_bytes,
    replay_payload_fingerprint,
    replay_payload_references,
)
from src.adapters.numpy_replay_graphs import replay_graph_rows, replay_graph_payload_bytes
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, CandidateCheckpointView
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_lifecycle_capture import _lease_lifecycle_sources
from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.replay_capture_origins import lease_replay_capture
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
    ReplaySnapshot,
)
from src.core.actor_ports import AppliedConsolidation
from src.core.data_erasure import ReplayPayloadErasure
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
from src.core.replay_graph_origin import ReplayGraphPorts
from src.core.replay_origin import ReplayOriginPorts
from src.core.replay_write_origin import begin_replay_write
from src.core.resource_sharing import SharingLimits
from test_checkpoint_snapshot_capture import BOUNDS, shell
from test_managed_experience import declaration
from test_managed_replay_origins import WINDOW
from test_restored_replay_capture import Builder, RESTORE_LIMITS, builder_source, forbidden


# Why: original copy reservations remain paid/live after eviction. Bind the
# continuation's allowance once at birth; retain live16 for its refusal control.
CURRENT_LIMITS = replace(RESTORE_LIMITS, max_live_records=32)


def data(second=False):
    """All new explicit array creation is bounded and visible to the controller."""
    features = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] if second else [[1.0, 0.0, 0.0]], dtype=np.float64
    )
    targets = np.array([[1.0], [0.0]] if second else [[0.0]], dtype=np.float64)
    assert features.nbytes + targets.nbytes <= 64
    return features, targets


def model_reference(learner):
    if type(learner) is not NumericLearner:
        raise ValueError("exact numeric fake learner required")
    return learner.model


def graph_rows(root, maximum):
    return replay_graph_rows(
        vars(root) if type(root) is CircadianPredictiveCodingNetwork else root, maximum
    )


def graph_bytes(root, maximum, bound):
    return replay_graph_payload_bytes(
        vars(root) if type(root) is CircadianPredictiveCodingNetwork else root, maximum, bound
    )


def footprint(learner):
    retained = graph_rows(learner.model, 4)
    return ReplayPayloadErasure(
        len(retained),
        sum(row.input_batch.shape[0] for row in retained),
        sum(row.input_batch.nbytes + row.target_batch.nbytes for row in retained),
    )


class NumericLearner:
    def __init__(self, model=None, work=None):
        self.model = shell() if model is None else model
        self.work = dict(trains=0, forks=0, snapshots=0, restores=0) if work is None else work
        self._learning_rate = 0.01

    def fork(self):
        self.work["forks"] += 1
        window = begin_model_copy(self.model)
        try:
            memo: dict[int, Any] = {}
            learner = NumericLearner(deepcopy(self.model, memo), self.work)
            if window is not None:
                window.copied(learner.model, memo, recipient=learner)
            return learner
        finally:
            if window is not None:
                window.end()

    def train_batch(self, features, targets):
        self.work["trains"] += 1
        # This fake copies one complete batch, so its observed range is the
        # actual batch row count rather than a single native row-mode copy.
        window = begin_replay_write(
            self.model, features, targets, self.model._replay_memory, features.shape[0], "batch"
        )
        if window is not None:
            window.before_copy(0, features.shape[0])
        row = ReplaySnapshot(deepcopy(features), deepcopy(targets), 1.0, 0.5)
        if window is not None:
            window.copied(row)
        memory = self.model._replay_memory
        memory.append(row)
        # Mirror the declared replay capacity, without a native model operation.
        while (
            sum(item.input_batch.shape[0] for item in memory) > 2
            or sum(item.input_batch.nbytes + item.target_batch.nbytes for item in memory) > 64
        ):
            memory.popleft()
        self.model.scalar += 1
        if window is not None:
            window.finish(memory)
        return TrainingDiagnostic("numeric_fake_update", 1.0)

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


def setup(
    *,
    continued=True,
    history=True,
    retain=True,
    reference=model_reference,
    limits=CURRENT_LIMITS,
    max_updates=2,
):
    seconds = [0.0]
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=max_updates, max_wall_seconds=1), lambda: seconds[0]
    )
    old = ActorShadowRuntime(
        NumericLearner(),
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
        measure_payload_bytes=lambda value: cast(np.ndarray, value).nbytes,
        native_footprint=footprint,
        native_erase=forbidden,
        measure_auxiliary_bytes=lambda _: 0,
        measure_checkpoint_bytes=lambda view: (
            graph_bytes(view.state, 4, 4096)
            + sum(item.features.nbytes for item in view.inbox.experiences)
            + sum(item.targets.nbytes for item in view.inbox.labels)
        ),
        native_growth_bytes=lambda learner, features, targets: features.nbytes + targets.nbytes,
        prepare_model_bytes=lambda builder, state: (
            footprint(builder.source).payload_bytes + graph_bytes(state, 4, 4096)
        ),
        prediction_cache_bytes=forbidden,
    )
    ledger = ManagedReplayOrigins(
        owner,
        ReplayOriginPorts(
            reference,
            lambda model, maximum: graph_rows(model, maximum),
            replay_copy_bytes,
            replay_payload_references,
            replay_payload_fingerprint,
        ),
        limits,
        WINDOW,
        retain_inbox_origins=history,
    )
    first = data()
    owner.declare(declaration("s1", "person-1"))
    owner.record_experience(
        Experience("s1", "e1", 1, "actor-0", first[0], "train", ExperiencePermissions(True, True))
    )
    owner.record_label(LabelArrival("label-s1", "s1", "e1", 3, "actor-0", first[1]))
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
    manager = ManagedReplayCheckpoints(
        ledger,
        ReplayGraphPorts(graph_rows, graph_bytes),
        builder_source=builder_source,
        retain_capture_history=retain,
    )
    token = manager.capture(controller)
    new = manager.restore(controller, token)
    learner = cast(NumericLearner, new._candidate)
    original_publication = cast(Any, manager)._publications.get(
        id(cast(NumericLearner, old._candidate).model)
    )
    first_current_row = ref(learner.model._replay_memory[0])
    before = ledger.accounting()
    assert life._copy_budget is not None
    raw_before = life._copy_budget._charged
    second = data(True)
    owner.declare(declaration("s2", "person-2"))
    owner.record_experience(
        Experience("s2", "e1", 4, "actor-0", second[0], "train", ExperiencePermissions(True, True))
    )
    owner.record_label(LabelArrival("label-s2", "s2", "e1", 6, "actor-0", second[1]))
    clock.advance_to(7)
    gate.resume()
    graph = SimpleNamespace(
        owner=owner,
        life=life,
        old=old,
        new=new,
        runtime=new,
        ledger=ledger,
        manager=manager,
        controller=controller,
        builder=builder,
        token=token,
        clock=clock,
        budget=budget,
        gate=gate,
        seconds=seconds,
        work=learner.work,
        before=before,
        raw_before=raw_before,
        publication=original_publication,
        publication_bytes=None
        if original_publication is None
        else pickle.dumps(original_publication.metadata),
        first_current_row=first_current_row,
    )
    graph.poll = ledger.train_ready() if continued else None
    return graph


def inventory(groups, limits):
    found = {}
    for group in groups:
        for item in group.references.models + group.references.snapshots:
            root = (
                item.model
                if type(item) is NumericLearner
                else item.state
                if type(item) is CandidateCheckpointView
                else item
            )
            found[id(root)] = root, graph_rows(root, limits.max_nodes)
    return tuple(found.values())


@contextmanager
def leased(graph, *, groups=None):
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
        with lease_replay_capture(
            graph.owner, supplied, BOUNDS, inventory, graph.ledger, checkpoints=graph.manager
        ) as check:
            yield check


@pytest.mark.parametrize("repeat", (False, True))
def test_should_accept_only_the_observed_current_success_and_keep_the_retired_proof(repeat):
    graph = setup()
    assert len(graph.poll.updates) == 1 and graph.poll.updates[0].update_number == 2
    assert graph.old._retired and graph.old._inbox._historical_completed_updates == 1
    assert graph.budget.updates_completed == 2 and not graph.new._retired
    assert graph.old._candidate.model.scalar == 1 and graph.new._candidate.model.scalar == 2
    assert (
        len(graph.old._candidate.model._replay_memory)
        == len(graph.new._candidate.model._replay_memory)
        == 1
    )
    assert graph.new._candidate.model._replay_memory[0].input_batch.shape == (2, 3)
    assert len(graph.new._inbox._applied) == 2 and len(graph.old._inbox._applied) == 1
    assert (
        graph.controller._models == [graph.new._candidate]
        and graph.builder.source is graph.old._candidate
    )
    assert graph.manager._publications[id(graph.old._candidate.model)] is graph.publication
    assert pickle.dumps(graph.publication.metadata) == graph.publication_bytes
    gc.collect()
    assert graph.first_current_row() is None
    before, raw, work = (
        graph.ledger.accounting(),
        graph.life._copy_budget._charged,
        dict(graph.work),
    )
    with leased(graph) as check:
        if repeat:
            for _ in range(7):
                check()
    after = graph.ledger.accounting()
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert after.records_created == before.records_created
    assert graph.work == work and graph.life._copy_budget._charged == raw
    success = graph.manager._current_success
    assert success is graph.manager._current_success_sealed
    assert graph.manager._current_success_storage == (success, success)
    assert success.completed_updates == 2
    assert success.receipt() is graph.new._inbox._applied[("e1", "s2")]
    assert len(success.history_records) == 2 and all(
        record.verified for record in success.history_records
    )
    assert before.invocations_started == graph.before.invocations_started + 1
    # Exactly one new replay snapshot and one actual original inbox record;
    # the additional success seal pays metadata rather than reissuing origins.
    assert before.records_created == graph.before.records_created + 2
    assert (
        before.metadata_bytes_charged >= graph.before.metadata_bytes_charged + success.charged_bytes
    )
    assert success.charged_bytes == 1024 + len(
        json.dumps(
            success.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8")
    )


@pytest.mark.parametrize("missing", ("history", "retention"))
def test_should_not_grant_continued_capture_without_both_original_history_flags(missing):
    graph = setup(history=missing != "history", retain=missing != "retention")
    assert graph.budget.updates_completed == 2
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("a missing birth proof acquired continuation authority")


@pytest.mark.parametrize(
    "change",
    (
        "native",
        "policy",
        "budget",
        "consolidation_limit",
        "attempted_id",
        "consolidations",
    ),
)
def test_should_refuse_unobserved_prior_native_or_work_changes_before_another_fake_train(change):
    graph = setup(continued=False)
    if change == "native":
        graph.new._candidate.model.scalar += 1
    elif change == "policy":
        graph.new._candidate._learning_rate = 0.02
    elif change == "budget":
        graph.budget.updates_completed += 1
    elif change == "consolidation_limit":
        graph.new._consolidation_limit += 1
    elif change == "attempted_id":
        graph.new._attempted_ids.add("unobserved")
    else:
        graph.new._attempted_ids.add("unobserved")
        graph.new._consolidations.append(
            AppliedConsolidation(
                "unobserved",
                graph.new._base_actor_version,
                graph.new._candidate_version,
                1,
                TrainingDiagnostic("unobserved", 1.0),
            )
        )
    trains = graph.work["trains"]
    with pytest.raises(ValueError):
        graph.ledger.train_ready()
    assert graph.work["trains"] == trains
    assert graph.manager._current_success is None


@pytest.mark.parametrize("authority", ("copies", "manager"))
def test_should_refuse_busy_original_update_gates_before_owner_or_learner_work(authority):
    graph = setup(continued=False)
    holder = graph.manager._copies if authority == "copies" else graph.manager
    before = graph.work["trains"]
    with holder._exclusive():
        with pytest.raises(PayloadOwnershipBusy):
            graph.ledger.train_ready()
    assert graph.work["trains"] == before and graph.manager._current_success is None


@pytest.mark.parametrize("change", ("ledger", "manager", "original"))
def test_should_refuse_changed_original_update_binding_before_fake_training(change):
    graph = setup(continued=False)
    if change == "ledger":
        graph.ledger._capture_update_binding = object()
    elif change == "manager":
        graph.manager._update_binding = object()
    else:
        graph.ledger._original_capture_update_binding = None
    before = graph.work["trains"]
    with pytest.raises(ValueError):
        graph.ledger.train_ready()
    assert graph.work["trains"] == before


@pytest.mark.parametrize(
    "change",
    (
        "native",
        "policy",
        "budget",
        "row",
        "payload",
        "old_payload",
        "receipt",
        "old_receipt",
        "source",
        "label",
        "history",
        "success",
        "seal",
        "storage",
        "clock",
        "limits",
    ),
)
def test_should_refuse_tampered_success_full_state_history_and_original_resources(change):
    graph = setup()
    current = graph.new
    if change == "native":
        current._candidate.model.scalar += 1
    elif change == "policy":
        current._candidate._learning_rate = 0.02
    elif change == "budget":
        graph.budget.updates_completed += 1
    elif change == "row":
        current._candidate.model._replay_memory[0] = deepcopy(
            current._candidate.model._replay_memory[0]
        )
    elif change == "payload":
        current._candidate.model._replay_memory[0].input_batch[0, 0] = 9.0
    elif change == "old_payload":
        graph.old._candidate.model._replay_memory[0].input_batch[0, 0] = 9.0
    elif change in ("receipt", "old_receipt"):
        inbox = current._inbox if change == "receipt" else graph.old._inbox
        key = ("e1", "s1")
        inbox._applied[key] = replace(inbox._applied[key])
    elif change == "source":
        key = ("e1", "s1")
        current._inbox._experiences[key] = replace(current._inbox._experiences[key])
    elif change == "label":
        key = ("e1", "s2")
        current._inbox._labels[key] = replace(current._inbox._labels[key])
    elif change == "history":
        graph.ledger._inbox_origins._records = dict(graph.ledger._inbox_origins._records)
    elif change == "success":
        graph.manager._current_success = replace(graph.manager._current_success)
    elif change == "seal":
        graph.manager._current_success_sealed = None
    elif change == "storage":
        graph.manager._current_success_storage = (None, None)
    elif change == "clock":
        graph.budget.clock = lambda: 0.0
    else:
        graph.life._copy_budget._limits = PayloadCopyLimits(4096)
    work, raw = dict(graph.work), graph.life._copy_budget._charged
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("changed full current or retired history reached capture")
    assert graph.work == work and graph.life._copy_budget._charged == raw


@pytest.mark.parametrize("subject", ("old_revoked", "new_revoked", "old_optout", "expiry"))
def test_should_recheck_both_old_and_new_consent_and_original_retention(subject):
    graph = setup()
    if subject == "old_revoked":
        graph.owner._revoked_keys.add(("e1", "s1"))
    elif subject == "new_revoked":
        graph.owner._revoked_keys.add(("e1", "s2"))
    elif subject == "old_optout":
        graph.owner._opted_out.add("person-1")
    else:
        graph.seconds[0] = 1000.0
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("original consent/retention was replaced by completed work")


@pytest.mark.parametrize("role", ("actor", "promotion", "snapshot", "foreign_checkpoint"))
def test_should_not_allow_a_success_to_hide_other_current_learner_roles(role):
    graph = setup()

    def alias(original):
        enrollment = max(group.holder.enrollment for group in original) + 1
        references = PayloadReferences(
            **{"snapshots" if role == "snapshot" else "models": (graph.new._candidate,)}
        )
        extra = OwnedPayloadGroup(
            PayloadHolderMetadata(
                enrollment,
                "checkpoint" if role in ("snapshot", "foreign_checkpoint") else role,
                True,
            ),
            references,
        )
        return (*original, extra)

    with pytest.raises(ValueError):
        with leased(graph, groups=alias):
            pytest.fail("a valid current role concealed an additional foreign role")


def test_should_bound_success_capture_checks_to_the_original_eight_rounds():
    graph = setup()
    with leased(graph) as check:
        for _ in range(7):
            check()
        with pytest.raises(ValueError, match="allowance|exhaust|limit"):
            check()


def test_should_release_both_original_pairs_when_only_an_expired_reader_is_retained():
    graph = setup()
    old, new, owner = ref(graph.old), ref(graph.new), ref(graph.owner)
    with leased(graph) as check:
        check()
    del graph
    gc.collect()
    assert old() is new() is owner() is None
    with pytest.raises(ValueError, match="expired"):
        check()


def test_should_not_issue_a_success_for_an_empty_paused_poll():
    graph = setup(continued=False)
    graph.gate.pause()
    trains, before = graph.work["trains"], graph.ledger.accounting()
    poll = graph.ledger.train_ready()
    assert not poll.updates and graph.work["trains"] == trains
    assert graph.manager._current_success is None
    after = graph.ledger.accounting()
    assert after.records_created == before.records_created
    assert after.invocations_started == before.invocations_started
    assert after.metadata_bytes_charged == before.metadata_bytes_charged


def test_should_keep_reader_thread_affinity_without_creating_another_update():
    graph = setup()
    errors: list[ValueError] = []
    with leased(graph) as check:

        def contend():
            try:
                check()
            except ValueError as error:
                errors.append(error)

        worker = Thread(target=contend)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive() and len(errors) == 1
        check()


@pytest.mark.parametrize("fault", ("mutation", "exception", "metadata", "inbox"))
def test_should_never_seal_a_provisional_result_after_a_late_opaque_port_fault(fault):
    state: dict[str, Any] = {"graph": None, "fired": False}

    def reference(learner):
        graph = state["graph"]
        if (
            graph is not None
            and not state["fired"]
            and graph.budget.updates_completed == 2
            and ("e1", "s2") in graph.new._inbox._applied
        ):
            state["fired"] = True
            if fault == "mutation":
                learner.model.scalar += 1
            elif fault == "exception":
                raise RuntimeError("declared post-commit model reference fault")
            elif fault == "inbox":
                inbox, original, ghost = graph.new._inbox, ("e1", "s1"), ("e1", "ghost")
                source = replace(
                    inbox._experiences[original],
                    sample_id="ghost",
                )
                label = replace(
                    inbox._labels[original],
                    sample_id="ghost",
                    event_id="label-ghost",
                )
                receipt = replace(
                    inbox._applied[original], sample_id="ghost", event_id="label-ghost"
                )
                assert source.key == label.key == ghost
                inbox._experiences[ghost], inbox._labels[ghost], inbox._applied[ghost] = (
                    source,
                    label,
                    receipt,
                )
                inbox._event_ids.add("label-ghost")
            else:
                admission = graph.ledger._admission
                live = graph.ledger._live_records(pending=True)
                remaining = (
                    admission.limits.max_metadata_bytes
                    - admission.accounting(live).metadata_bytes_charged
                )
                admission._charge(0, 0, remaining, live)
        return model_reference(learner)

    graph = setup(continued=False, reference=reference)
    state["graph"] = graph
    before, raw = graph.ledger.accounting(), graph.life._copy_budget._charged
    error = RuntimeError if fault == "exception" else ValueError
    with pytest.raises(error):
        graph.ledger.train_ready()
    assert state["fired"] and graph.work["trains"] == graph.budget.updates_completed == 2
    assert graph.manager._current_success is None and graph.ledger._poisoned
    assert graph.old._retired and graph.old._inbox._historical_completed_updates == 1
    assert graph.manager._publications[id(graph.old._candidate.model)] is graph.publication
    assert pickle.dumps(graph.publication.metadata) == graph.publication_bytes
    assert graph.life._copy_budget._charged > raw
    assert graph.ledger.accounting().metadata_bytes_charged > before.metadata_bytes_charged
    work = dict(graph.work)
    with pytest.raises(ValueError):
        with leased(graph):
            pytest.fail("a failed final poll supplied successful continuation authority")
    assert graph.work == work


def test_should_accept_a_monotone_original_clock_refresh_after_the_completed_observation():
    state: dict[str, Any] = {"graph": None, "fired": False}

    def reference(learner):
        graph = state["graph"]
        if (
            graph is not None
            and not state["fired"]
            and graph.budget.updates_completed == 2
            and ("e1", "s2") in graph.new._inbox._applied
        ):
            state["fired"] = True
            graph.seconds[0] = 0.125
        return model_reference(learner)

    graph = setup(continued=False, reference=reference)
    state["graph"] = graph
    original_clock = graph.budget.clock
    poll = graph.ledger.train_ready()
    assert state["fired"] and len(poll.updates) == 1 and poll.updates[0].update_number == 2
    assert graph.budget.clock is original_clock and graph.budget.last_clock == 0.125
    assert graph.budget.started_at == 0.0 and graph.budget.budget.max_wall_seconds == 1.0
    assert graph.manager._current_success.completed_updates == 2
    graph.gate.pause()
    work, raw = dict(graph.work), graph.life._copy_budget._charged
    with leased(graph) as check:
        check()
    assert graph.work == work and graph.life._copy_budget._charged == raw
    assert (
        graph.old._inbox._historical_completed_updates == 1 and graph.budget.updates_completed == 2
    )


def test_should_keep_the_original_live_sixteen_refusal_before_a_second_fake_train():
    graph = setup(continued=False, limits=RESTORE_LIMITS)
    before, raw = graph.ledger.accounting(), graph.life._copy_budget._charged
    publication = graph.manager._publications[id(graph.old._candidate.model)]
    assert graph.ledger._admission.limits is RESTORE_LIMITS
    assert RESTORE_LIMITS.max_live_records == 16
    with pytest.raises(ValueError, match="live record limit"):
        graph.ledger.train_ready()
    after = graph.ledger.accounting()
    assert graph.work["trains"] == graph.budget.updates_completed == 1
    assert graph.new._candidate.model.scalar == 1
    assert after.records_created == before.records_created
    assert after.invocations_started == before.invocations_started + 1
    assert after.metadata_bytes_charged == before.metadata_bytes_charged + 1024
    assert graph.life._copy_budget._charged >= raw
    assert graph.manager._current_success is None and graph.ledger._poisoned
    assert graph.manager._publications[id(graph.old._candidate.model)] is publication
    assert graph.old._retired and graph.old._inbox._historical_completed_updates == 1
