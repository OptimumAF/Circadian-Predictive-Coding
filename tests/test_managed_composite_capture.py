"""NEW bounded constructor graphs; complete copies, no native method work."""

from dataclasses import replace
from collections import deque
from pathlib import Path
from unittest.mock import patch
from typing import Any
import json

import numpy as np
import pytest

import src.adapters.numpy_composite_capture as module
from src.adapters.numpy_composite_capture import capture_numpy_managed_composite
from src.adapters.numpy_learners import (
    BackpropLearner,
    CircadianLearner,
    make_managed_data_lifecycle,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor, ServingPromotionController, _Slot, _Bundle
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionProgress
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    CircadianConfig,
    ReplaySnapshot,
)
from src.core.data_lifecycle import (
    LifecycleDeclaration,
    DataProvenance,
    DataConsent,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock, Experience, ExperiencePermissions, LabelArrival
from src.core.managed_composite_state import CompositeCaptureLimits
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.promotion_guard import PromotionPolicy
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration, CachedPrediction

LIMITS = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32)


def values(record):
    return dict(record.fields)


@pytest.fixture(scope="module")
def graphs(request):
    with pytest.MonkeyPatch.context() as monkey:
        for kind in (BackpropLearner, CircadianLearner):
            for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
                monkey.setattr(kind, name, lambda *a, **k: pytest.fail("native method work"))
        result = []
        for variant in range(3):
            learner: Any = (
                BackpropLearner(BackpropMLP(3, 2, 23), learning_rate=0.01)
                if variant < 2
                else CircadianLearner(
                    CircadianPredictiveCodingNetwork(
                        3, 2, 23, circadian_config=CircadianConfig(), min_hidden_dim=2
                    ),
                    learning_rate=0.01,
                    inference_steps=2,
                    inference_learning_rate=0.05,
                )
            )
            budget = ToyBudgetSession(
                ToyExecutionBudget(max_training_updates=0),
                clock=lambda: 8.0,
                progress=ToyExecutionProgress(),
            )
            actor = (
                None
                if variant == 0
                else PromotableActor(
                    learner,
                    version="actor",
                    configuration=ServingConfiguration(2, 2),
                    feature_digest=lambda x: "unused",
                    metadata={"origin": "owned"},
                )
            )
            runtime = ActorShadowRuntime(
                learner,
                actor_version="actor",
                candidate_version="candidate",
                clock=LogicalClock(0),
                budget=budget,
                max_experiences=4,
                actor=actor,
            )
            shared = ResourceSharedRuntime(
                runtime,
                ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True),
            )
            owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4))
            life = make_managed_data_lifecycle(
                owner,
                policy=DataRetentionPolicy(
                    4096,
                    20,
                    PayloadOwnershipLimits(8, 12),
                    owned_payload_copies=PayloadCopyLimits(4096),
                    max_retention_seconds=2.0,
                ),
            )
            for name in ("one", "two"):
                owner.declare(
                    LifecycleDeclaration(
                        ("episode", name),
                        DataProvenance("local", "person", True, False),
                        DataConsent(True, True),
                        "replay",
                    )
                )
            owner.record_experience(
                Experience(
                    "one",
                    "episode",
                    0,
                    "actor",
                    np.ones((1, 3)),
                    "train",
                    ExperiencePermissions(True, True),
                )
            )
            owner.record_label(
                LabelArrival("label", "one", "episode", 0, "actor", np.ones((1, 1)), "train")
            )
            shared._sharing.pause()
            controller = CandidateCheckpointController(
                shared,
                build_learner=lambda state: pytest.fail("build"),
                state_digest=lambda state: "a" * 64,
                policy_digest=lambda model: "b" * 64,
            )
            if actor is not None:
                source = runtime._inbox._experiences[("episode", "one")].features
                # Synthetic retained graph edges: no serving, promotion or training.
                current = actor._slot.bundle
                metadata: dict[str, object] = {"shared": source, "equal": source.copy()}
                current = _Bundle(
                    current.learner,
                    current.version,
                    current.configuration,
                    {"c" * 64: CachedPrediction(2, source)},
                    metadata,
                    0,
                )
                actor._slot = _Slot(current, 3, current)
                builder = lambda state: pytest.fail("promotion build")
                evaluator: PromotionGuardEvaluator[Any, Any, Any, Any] = PromotionGuardEvaluator(
                    build_learner=builder,
                    utility=lambda *a: 0.0,
                    actions=lambda *a: (),
                    state_valid=lambda *a: True,
                    prediction_valid=lambda *a: True,
                    resource_bytes=lambda *a: 0,
                    state_digest=lambda *a: "a" * 64,
                    clock=lambda: 8.0,
                )
                promotion = ServingPromotionController(
                    actor,
                    policy=PromotionPolicy("utility", "bytes", 0.0, 0.0, 0.0, 1.0, 4096, ("safe",)),
                    evaluator=evaluator,
                    build_learner=builder,
                    state_digest=lambda *a: "a" * 64,
                )
            else:
                promotion = None
            result.append((owner, life, runtime, controller, promotion))
        yield result
        for owner, life, runtime, controller, promotion in result:
            assert runtime._budget.updates_completed == 0
            assert life._admitted_bytes == 32 and life._copy_budget._charged <= 4096
        folder = Path(request.config.option.basetemp)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "composite-native-work.json").write_text(
            json.dumps(
                dict(
                    graphs=3,
                    updates=0,
                    predictions=0,
                    snapshots=0,
                    restores=0,
                    sleeps=0,
                    structural=0,
                    workers=0,
                    threads=0,
                    synthetic_rollback_replay=False,
                )
            ),
            encoding="utf8",
        )


@pytest.mark.parametrize("index", range(3))
def test_should_copy_complete_original_graph_once_after_admission(graphs, index):
    owner, life, runtime, controller, promotion = graphs[index]
    before = life._copy_budget._charged
    with patch.object(module, "deepcopy", wraps=module.deepcopy) as copying:
        result = capture_numpy_managed_composite(owner, limits=LIMITS)
        assert copying.call_count == 2
        assert copying.call_args_list[0].args[1] is copying.call_args_list[1].args[1]
    assert (
        result.state.records.lifecycle.copy.charged_bytes
        == before + result.state.retained_payload_bytes
    )
    assert len(result.state.holders) == (3 if index == 0 else 4)
    assert len({row.path for row in result.authority}) == len(result.authority)
    actor = values(result.state.holders[0][1])
    candidate = values(result.state.holders[1][1])
    inbox = values(candidate["_inbox"])
    original = runtime._inbox._experiences[("episode", "one")].features
    detached = inbox["_experiences"][("episode", "one")].features
    assert not np.shares_memory(original, detached) and detached.dtype == original.dtype
    native = values(candidate["_candidate"])["_model"]
    if index == 2:
        assert not native.state["_replay_memory"]
    if index:
        slot = values(actor["_slot"])
        assert slot["bundle"] is slot["previous"] and slot["generation"] == 3
        bundle = values(slot["bundle"])
        assert bundle["metadata"]["shared"] is detached
        assert bundle["cache"]["c" * 64].prediction is detached
        assert bundle["metadata"]["equal"] is not detached
    assert values(result.state.sharing)["_paused"] is True
    assert values(result.state.budget)["started_at"] == 8.0


def test_should_refuse_synthetic_replay_without_original_ledger_before_projection_or_copy(
    graphs, monkeypatch
):
    owner, life, runtime, *_ = graphs[2]
    inbox = runtime._inbox
    row = ReplaySnapshot(
        inbox._experiences[("episode", "one")].features,
        inbox._labels[("episode", "one")].targets,
        1.0,
        1.0,
    )
    monkeypatch.setattr(runtime._candidate._model, "_replay_memory", deque([row]))
    before = life._copy_budget._charged
    with (
        patch.object(
            module, "project_numpy_native", side_effect=AssertionError("premature projection")
        ),
        patch.object(module, "deepcopy", side_effect=AssertionError("premature copy")),
    ):
        with pytest.raises(ValueError, match="nonempty replay"):
            capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


@pytest.mark.parametrize(
    "mutation",
    [
        "not_paused",
        "serving",
        "training",
        "checkpointing",
        "revoked",
        "opted_out",
        "expired",
        "foreign_clock",
        "unknown_field",
        "bad_permission",
        "nonfinite",
        "byte_bound",
        "node_bound",
        "depth_bound",
        "wrong_shape",
    ],
)
def test_should_refuse_before_payload_copy_without_charge(graphs, monkeypatch, mutation):
    owner, life, runtime, _, _ = graphs[0]
    sharing = owner._shared._sharing
    limits = LIMITS
    if mutation == "not_paused":
        monkeypatch.setattr(sharing, "_paused", False)
    elif mutation in ("serving", "training", "checkpointing"):
        monkeypatch.setattr(sharing, "_" + mutation, 1 if mutation == "serving" else True)
    elif mutation == "revoked":
        monkeypatch.setattr(owner, "_revoked_keys", {("episode", "one")})
    elif mutation == "opted_out":
        monkeypatch.setattr(owner, "_opted_out", {"person"})
    elif mutation == "expired":
        monkeypatch.setattr(life._clock, "_time", 20)
    elif mutation == "foreign_clock":
        monkeypatch.setattr(runtime._budget, "clock", lambda: 8.0)
    elif mutation == "unknown_field":
        monkeypatch.setattr(runtime.actor, "_unknown", None, raising=False)
    elif mutation == "bad_permission":
        item = runtime._inbox._experiences[("episode", "one")]
        monkeypatch.setitem(
            runtime._inbox._experiences,
            item.key,
            replace(item, permissions=ExperiencePermissions(False, False)),
        )
    elif mutation == "nonfinite":
        monkeypatch.setattr(runtime._candidate._model, "bias_output", np.full((1, 1), np.nan))
    elif mutation == "byte_bound":
        limits = replace(LIMITS, max_array_bytes=1)
    elif mutation == "node_bound":
        limits = replace(LIMITS, max_nodes=1)
    elif mutation == "depth_bound":
        limits = replace(LIMITS, max_depth=1)
    elif mutation == "wrong_shape":
        item = runtime._inbox._experiences[("episode", "one")]
        monkeypatch.setitem(
            runtime._inbox._experiences, item.key, replace(item, features=np.ones((1, 2)))
        )
    before = life._copy_budget._charged
    old_tick = life._last_tick
    try:
        with patch.object(module, "deepcopy", side_effect=AssertionError("premature copy")):
            with pytest.raises(ValueError):
                capture_numpy_managed_composite(owner, limits=limits)
        assert life._copy_budget._charged == before
    finally:
        # Restore only synthetic clock mutation fixture state, not spent budgets.
        if mutation == "expired":
            life._last_tick = old_tick


def test_should_keep_admitted_charge_when_copy_fails(graphs):
    owner, life, *_ = graphs[0]
    before = life._copy_budget._charged
    with patch.object(module, "deepcopy", side_effect=RuntimeError("copy failed")):
        with pytest.raises(RuntimeError, match="copy failed"):
            capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before + 32


def test_should_preserve_fortran_order_and_readonly_shared_arrays(graphs, monkeypatch):
    owner, _, runtime, *_ = graphs[0]
    weights = np.asfortranarray(runtime._candidate._model.weight_input_hidden)
    weights.flags.writeable = False
    monkeypatch.setattr(runtime._candidate._model, "weight_input_hidden", weights)
    monkeypatch.setattr(runtime._candidate._model, "_hidden_weights", [weights])
    result = capture_numpy_managed_composite(owner, limits=LIMITS)
    native = values(values(result.state.holders[1][1])["_candidate"])["_model"]
    restored = native.state["weight_input_hidden"]
    assert restored is native.state["_hidden_weights"][0]
    assert restored.flags.f_contiguous and not restored.flags.c_contiguous
    assert not restored.flags.writeable and not np.shares_memory(restored, weights)


@pytest.mark.parametrize(
    "mutation",
    [
        "overlap",
        "cycle",
        "unsupported",
        "foreign_learner",
        "bad_dtype",
        "noncontiguous",
        "negative_label",
        "exhausted",
    ],
)
def test_should_refuse_unsafe_complete_graph_before_copy(graphs, monkeypatch, mutation):
    owner, life, runtime, *_ = graphs[1]
    bundle = runtime.actor._slot.bundle
    source = runtime._inbox._experiences[("episode", "one")]
    if mutation == "overlap":
        monkeypatch.setitem(bundle.metadata, "equal", source.features.view())
    elif mutation == "cycle":
        monkeypatch.setitem(bundle.metadata, "cycle", bundle.metadata)
    elif mutation == "unsupported":
        monkeypatch.setitem(bundle.metadata, "invalid", object())
    elif mutation == "foreign_learner":
        monkeypatch.setattr(runtime._inbox, "_learner", bundle.learner)
    elif mutation == "bad_dtype":
        monkeypatch.setitem(
            runtime._inbox._experiences,
            source.key,
            replace(source, features=np.ones((1, 3), dtype=complex)),
        )
    elif mutation == "noncontiguous":
        monkeypatch.setitem(
            runtime._inbox._experiences,
            source.key,
            replace(source, features=np.ones((1, 6))[:, ::2]),
        )
    elif mutation == "negative_label":
        label = runtime._inbox._labels[source.key]
        monkeypatch.setitem(
            runtime._inbox._labels, source.key, replace(label, targets=-np.ones((1, 1)))
        )
    elif mutation == "exhausted":
        # A NEW actual reservation, never refunded when this test completes.
        life._copy_budget.reserve(4096 - life._copy_budget._charged)
    before = life._copy_budget._charged
    with patch.object(module, "deepcopy", side_effect=AssertionError("premature copy")):
        with pytest.raises(ValueError):
            capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before
