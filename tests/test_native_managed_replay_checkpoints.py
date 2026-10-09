"""Four prospectively bounded actual CPC checkpoint admission/handoff controls."""

from collections import deque
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
import json
import pickle
import sys
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from src.adapters.numpy_learners import (
    CircadianLearner,
    ManagedNumpyBuilder,
    make_managed_data_lifecycle,
)
from src.adapters.numpy_replay_copies import replay_builder_source
from src.adapters.numpy_replay_graphs import replay_graph_payload_bytes, replay_graph_rows
from src.adapters.numpy_replay_origins import (
    replay_copy_bytes,
    replay_model_reference,
    replay_payload_fingerprint,
    replay_payload_references,
    retained_replay_references,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.experience_inbox import ExperienceInbox
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
import src.app.managed_replay_checkpoints as checkpoint_module
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.replay_graph_origin import ReplayGraphPorts
from src.core.replay_origin import ReplayOriginLimits, ReplayOriginPorts
from src.core.resource_sharing import SharingLimits
from test_managed_replay_origins import WINDOW
from test_native_managed_replay_origins import queue
from native_work_profile import bind_original_calls, count_original_calls

FAIL_RESTORE_COPY = [False]


def source_array_bytes(root):
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if type(value) is np.ndarray:
            return value.nbytes
        if type(value) is dict:
            return sum(size(item) for item in value.values())
        if type(value) in (tuple, list, deque):
            return sum(size(item) for item in value)
        return size(vars(value)) if hasattr(value, "__dict__") else 0

    return size(root)


@pytest.fixture(scope="module", autouse=True)
def bounded_native_work(tmp_path_factory):
    limits = dict(
        models=4,
        learners=16,
        forks=12,
        wakes=4,
        steps=8,
        stores=4,
        predicts=4,
        array_copies=20,
        graph_events=64,
        source_array_bytes=1024 * 1024,
        captures=4,
        preparations=4,
        native_restores=4,
        handoffs=4,
    )
    yield from observe_native_work(tmp_path_factory, limits)


def observe_native_work(tmp_path_factory, limits):
    """Shared measurement helper; each module declares its own finite scope."""
    limits = dict(
        limits,
        memo_reads=64 * limits["graph_events"],
        max_event_reads=64,
        cleanup_attempts=limits.get("cleanup_attempts", 0),
    )
    work = dict.fromkeys(limits, 0)
    previous = sys.getprofile()
    # Preserve the public dispatch that expiry attests while measuring its entries.
    original_calls = bind_original_calls(
        (
            (CandidateCheckpointController, "capture", "captures"),
            (CandidateCheckpointController, "restore", "preparations"),
            (ExperienceInbox, "_retire_ledger", "handoffs"),
            (ManagedDataLifecycle, "_cleanup", "cleanup_attempts"),
        )
    )

    def counted(key, original):
        def call(*args, **kwargs):
            work[key] += 1
            assert work[key] <= limits[key]
            if key == "wakes":
                work["steps"] += kwargs.get("inference_steps", args[4] if len(args) > 4 else 0)
                assert work["steps"] <= limits["steps"]
            return original(*args, **kwargs)

        return call

    def profile(frame, event, function):
        count_original_calls(original_calls, work, limits, frame, event)
        if (
            event == "c_call"
            and frame.f_code.co_filename.endswith("circadian_predictive_coding.py")
            and getattr(function, "__name__", None) == "copy"
            and isinstance(getattr(function, "__self__", None), np.ndarray)
        ):
            work["array_copies"] += 1
            assert work["array_copies"] <= limits["array_copies"]
        if previous is not None:
            previous(frame, event, function)

    original_observe = checkpoint_module.observe_graph_copies

    def traced(observer, bounds):
        event_reads = [0]

        def callback(producer, kind, stage, read, lookup):
            if stage == "before_copy":
                event_reads[0] = 0

            def counted_read(*args):
                event_reads[0] += 1
                work["memo_reads"] += 1
                work["max_event_reads"] = max(work["max_event_reads"], event_reads[0])
                assert event_reads[0] <= limits["max_event_reads"]
                assert work["memo_reads"] <= limits["memo_reads"]
                return read() if not args else lookup(*args)

            if stage == "before_copy":
                work["graph_events"] += 1
                work["source_array_bytes"] += source_array_bytes(counted_read().source)
                assert work["graph_events"] <= limits["graph_events"]
                assert work["source_array_bytes"] <= limits["source_array_bytes"]
            observer(producer, kind, stage, counted_read, counted_read)
            if FAIL_RESTORE_COPY[0] and kind == "native_restore" and stage == "before_copy":
                raise RuntimeError("declared actual native restore-copy fault")

        return original_observe(callback, bounds)

    with ExitStack() as stack:
        for cls, method, key in [
            (CircadianPredictiveCodingNetwork, "__init__", "models"),
            (CircadianLearner, "__init__", "learners"),
            (CircadianLearner, "fork", "forks"),
            (CircadianPredictiveCodingNetwork, "train_epoch", "wakes"),
            (CircadianPredictiveCodingNetwork, "_store_replay_snapshot", "stores"),
            (CircadianPredictiveCodingNetwork, "predict_proba", "predicts"),
            (CircadianPredictiveCodingNetwork, "restore_state", "native_restores"),
        ]:
            stack.enter_context(patch.object(cls, method, counted(key, getattr(cls, method))))
        stack.enter_context(patch.object(checkpoint_module, "observe_graph_copies", traced))
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(previous)
            FAIL_RESTORE_COPY[0] = False
            (tmp_path_factory.mktemp("managed-handoff-work") / "work.json").write_text(
                json.dumps({"actual": work, "limits": limits}, indent=2), encoding="utf8"
            )


def setup(
    *,
    invocations=32,
    revoke_final=False,
    fingerprint=None,
    replay_examples=2,
    retain_inbox_origins=False,
    live_records=16,
    train_first=True,
    queue_first=True,
    records_created=64,
):
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: 0.0
    )
    model = CircadianPredictiveCodingNetwork(
        3, 2, seed=23, min_hidden_dim=2, circadian_config=CircadianConfig(sleep_mode="disabled")
    )
    model.configure_replay_retention(ReplayRetentionBudget(replay_examples, 64))
    learner = CircadianLearner(
        model, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    runtime = ActorShadowRuntime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
    )
    shared = ResourceSharedRuntime(
        runtime, ServingPriorityGate(SharingLimits(1, 8, 1), resource_available=lambda: True)
    )
    owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4))
    life = make_managed_data_lifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096,
            120,
            PayloadOwnershipLimits(8, 12),
            owned_payload_copies=PayloadCopyLimits(4096),
            max_retention_seconds=1000.0,
        ),
    )
    ports = ReplayOriginPorts(
        replay_model_reference,
        retained_replay_references,
        replay_copy_bytes,
        replay_payload_references,
        replay_payload_fingerprint if fingerprint is None else fingerprint,
    )
    # Limits are fixed at original birth, never enlarged after training/consumption.
    ledger = ManagedReplayOrigins(
        owner,
        ports,
        ReplayOriginLimits(live_records, records_created, invocations, 128 * 1024, 120, 4096),
        WINDOW,
        retain_inbox_origins=retain_inbox_origins,
    )
    if queue_first:
        queue(owner, clock)
    if train_first:
        ledger.train_ready()
    shared._sharing.pause()
    prepared_policy_calls = [0]

    def policy(candidate):
        if controller._models and candidate is controller._models[-1]:
            prepared_policy_calls[0] += 1
            if revoke_final and prepared_policy_calls[0] == 3:
                owner._revoked_keys.add(("e1", "s1"))
        return sha256(
            pickle.dumps(
                (
                    candidate._learning_rate,
                    candidate._inference_steps,
                    candidate._inference_learning_rate,
                )
            )
        ).hexdigest()

    controller = CandidateCheckpointController(
        shared,
        build_learner=ManagedNumpyBuilder(cast(CircadianLearner, runtime._candidate)),
        state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
        policy_digest=policy,
    )
    manager = ManagedReplayCheckpoints(
        ledger,
        ReplayGraphPorts(replay_graph_rows, replay_graph_payload_bytes),
        builder_source=replay_builder_source,
    )
    return owner, life, runtime, ledger, controller, manager


def assert_unpublished(owner, runtime, ledger, controller, token, anchors):
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    assert controller._pending[token].owner is runtime and ledger._anchors is anchors
    assert controller._attempts == 1 and len(controller._models) == 1


def test_should_publish_original_ledger_with_actual_restored_and_materialized_chains():
    owner, life, runtime, ledger, controller, manager = setup()
    source = runtime._candidate._model._replay_memory[0]
    receipt = runtime._inbox._applied[("e1", "s1")]
    admission, limits, raw_budget = ledger._admission, ledger._admission.limits, life._copy_budget
    before = ledger.accounting()
    token = manager.capture(controller)
    prepared = manager.restore(controller, token)
    assert owner._shared._runtime is prepared and runtime._retired and not controller._pending
    assert runtime._inbox._historical_completed_updates == 1
    assert (
        prepared._budget is runtime._budget
        and prepared._payload_lineage is runtime._payload_lineage
    )
    assert (
        prepared._inbox._clock is runtime._inbox._clock
        and prepared._candidate is controller._models[0]
    )
    assert (
        ledger._admission is admission
        and admission.limits is limits
        and life._copy_budget is raw_budget
    )
    row = prepared._candidate._model._replay_memory[0]
    witness = ledger._rows[id(row)]
    assert row is not source and row.input_batch is not source.input_batch
    assert witness.source() is prepared._inbox._experiences[("e1", "s1")]
    assert witness.label() is prepared._inbox._labels[("e1", "s1")]
    assert (
        witness.receipt() is prepared._inbox._applied[("e1", "s1")]
        and witness.receipt() is not receipt
    )
    assert ledger.origins()[0].key == ("e1", "s1")
    assert ledger.accounting().records_created > before.records_created
    with pytest.raises(ValueError):
        manager._copies.origins(controller, prepared._candidate)
    original = prepared._inbox._applied[("e1", "s1")]
    prepared._inbox._applied[("e1", "s1")] = replace(original)
    with pytest.raises(ValueError):
        ledger.origins()
    prepared._inbox._applied[("e1", "s1")] = original
    prepared._candidate._model._replay_memory[0] = replace(row)
    with pytest.raises(ValueError):
        ledger.origins()
    prepared._candidate._model._replay_memory[0] = row
    work = prepared._budget.updates_completed
    prepared._budget.updates_completed = 0
    with pytest.raises(ValueError):
        ledger.origins()
    prepared._budget.updates_completed = work
    # This scope exercises revocation with the actual row still present. The
    # public opt_out also performs cleanup, which has separate unfinished gates.
    owner._opted_out.add("person-1")
    assert prepared._candidate._model._replay_memory[0] is row
    with pytest.raises(ValueError):
        ledger.origins()


def test_should_refuse_revocation_from_final_policy_probe_before_publication():
    owner, life, runtime, ledger, controller, manager = setup(revoke_final=True)
    token = manager.capture(controller)
    anchors, charge = ledger._anchors, life._copy_budget._charged
    with pytest.raises(ValueError):
        manager.restore(controller, token)
    assert ("e1", "s1") in owner._revoked_keys
    assert_unpublished(owner, runtime, ledger, controller, token, anchors)
    assert life._copy_budget._charged > charge


def test_should_keep_original_eight_invocation_exhaustion_and_retained_failed_target():
    owner, life, runtime, ledger, controller, manager = setup(invocations=8)
    token = manager.capture(controller)
    anchors, charge = ledger._anchors, life._copy_budget._charged
    with pytest.raises(ValueError, match="limit|allowance|exhaust"):
        manager.restore(controller, token)
    assert_unpublished(owner, runtime, ledger, controller, token, anchors)
    assert ledger.accounting().invocations_started == 8
    assert life._copy_budget._charged > charge and ledger._copy_slots > 0


def test_should_preserve_constructor_witness_and_original_error_before_native_restore_copy():
    owner, life, runtime, ledger, controller, manager = setup()
    token = manager.capture(controller)
    anchors, charge = ledger._anchors, life._copy_budget._charged
    FAIL_RESTORE_COPY[0] = True
    try:
        with pytest.raises(RuntimeError, match="declared actual native restore-copy fault"):
            manager.restore(controller, token)
    finally:
        FAIL_RESTORE_COPY[0] = False
    assert_unpublished(owner, runtime, ledger, controller, token, anchors)
    assert life._copy_budget._charged > charge and ledger._copy_slots > 0
    child = controller._models[0]
    assert manager._copies.origins(controller, child)[0].key == ("e1", "s1")
    assert ledger.origins()[0].key == ("e1", "s1")
