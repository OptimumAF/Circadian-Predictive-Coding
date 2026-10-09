"""Tiny actual trained-row capture scope; run after initial gates/declaration."""

from copy import deepcopy
from unittest.mock import patch
import pytest
from src.adapters.numpy_learners import CircadianLearner, make_managed_data_lifecycle
from src.adapters.numpy_composite_capture import capture_numpy_managed_composite
import src.adapters.numpy_composite_capture as adapter
from src.adapters.numpy_replay_origins import (
    replay_model_reference,
    retained_replay_references,
    replay_copy_bytes,
    replay_payload_references,
    replay_payload_fingerprint,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    CircadianConfig,
    ReplayRetentionBudget,
)
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.experience import LogicalClock
from src.core.replay_origin import ReplayOriginPorts
from src.core.resource_sharing import SharingLimits
from test_managed_replay_origins import LIMITS, WINDOW
from test_managed_composite_capture import LIMITS as CAPTURE_LIMITS, values
from test_native_managed_replay_origins import queue, native_work  # noqa: F401


def setup():
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: 0.0
    )
    model = CircadianPredictiveCodingNetwork(
        3, 2, seed=23, min_hidden_dim=2, circadian_config=CircadianConfig(sleep_mode="disabled")
    )
    model.configure_replay_retention(ReplayRetentionBudget(2, 64))
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
        replay_payload_fingerprint,
    )
    ledger = ManagedReplayOrigins(owner, ports, LIMITS, WINDOW)
    queue(owner, clock)
    ledger.train_ready()
    shared._sharing.pause()
    return owner, life, runtime, ledger


def test_should_copy_actual_original_candidate_replay_after_ledger_admission():
    owner, life, runtime, ledger = setup()
    row = runtime._candidate._model._replay_memory[0]
    before = ledger.accounting()
    charged = life._copy_budget._charged
    with (
        patch.object(
            type(life), "_elapsed", side_effect=AssertionError("ordinary elapsed path reentered")
        ),
        patch.object(adapter, "deepcopy", wraps=adapter.deepcopy) as copying,
    ):
        captured = capture_numpy_managed_composite(
            owner, limits=CAPTURE_LIMITS, replay_origins=ledger
        )
        assert copying.call_count == 2
        assert copying.call_args_list[0].args[1] is copying.call_args_list[1].args[1]
    candidate = values(captured.state.holders[1][1])
    native = values(candidate["_candidate"])["_model"]
    copied = native.state["_replay_memory"][0]
    assert copied is not row and copied.input_batch is not row.input_batch
    assert native.config is native.state["config"]
    memo = copying.call_args_list[0].args[1]
    assert memo[id(row)] is copied
    assert memo[id(row.input_batch)] is copied.input_batch
    assert memo[id(row.target_batch)] is copied.target_batch
    assert life._copy_budget._charged == charged + captured.state.retained_payload_bytes
    assert ledger.accounting().invocations_started == before.invocations_started + 1
    assert ledger.origins()[0].key == ("e1", "s1") and runtime._budget.updates_completed == 1


def test_should_refuse_actual_candidate_replay_missing_ledger_before_projection():
    owner, life, runtime, ledger = setup()
    before = life._copy_budget._charged
    with (
        patch.object(adapter, "project_numpy_native", side_effect=AssertionError("projection")),
        patch.object(adapter, "deepcopy", side_effect=AssertionError("copy")),
    ):
        with pytest.raises(ValueError, match="nonempty replay"):
            capture_numpy_managed_composite(owner, limits=CAPTURE_LIMITS)
    assert life._copy_budget._charged == before and ledger.accounting().invocations_started == 1


def test_should_refuse_copied_actor_replay_without_original_retained_holder_witness():
    owner, life, runtime, ledger = setup()
    # Actual tiny detached row copy;synthetic installation is not a promotion.
    runtime.actor._learner._model._replay_memory.append(
        deepcopy(runtime._candidate._model._replay_memory[0])
    )
    before = life._copy_budget._charged
    with (
        patch.object(adapter, "project_numpy_native", side_effect=AssertionError("projection")),
        patch.object(adapter, "deepcopy", side_effect=AssertionError("copy")),
    ):
        with pytest.raises(ValueError, match="holder lineage"):
            capture_numpy_managed_composite(owner, limits=CAPTURE_LIMITS, replay_origins=ledger)
    assert life._copy_budget._charged == before and ledger.accounting().invocations_started == 2
