"""NEW bounded issued checkpoints and labeled synthetic retained graph controls."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
import json

import numpy as np
import pytest

import src.adapters.numpy_composite_capture as adapter
from src.adapters.numpy_learners import (
    BackpropLearner,
    CircadianLearner,
    make_managed_data_lifecycle,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor, ServingPromotionController, _Pending, _Bundle
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionProgress
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork, CircadianConfig
from src.core.data_lifecycle import (
    LifecycleDeclaration,
    LifecycleLimits,
    DataProvenance,
    DataConsent,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.promotion_guard import (
    PromotionPolicy,
    PromotionGuardReport,
    PromotionEvidence,
    decide_promotion,
)
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration, PreparedPromotion, PromotionReceipt
from src.shared.process_memory import ProcessRssSampler
from test_managed_composite_capture import LIMITS, values


@pytest.fixture(scope="module")
def histories(request):
    counts: dict[str, Any] = dict(
        graphs=0,
        snapshots=0,
        checkpoint_captures=0,
        synthetic_handoffs=0,
        updates=0,
        predictions=0,
        restores=0,
        sleeps=0,
        structural=0,
        workers=0,
        threads=0,
    )
    result = []
    current_operations = 0
    counts["constructor_fork_calls_per_graph"] = []
    reader_counts = []
    with pytest.MonkeyPatch.context() as monkey:
        for kind in (
            BackpropMLP,
            CircadianPredictiveCodingNetwork,
            BackpropLearner,
            CircadianLearner,
        ):
            original_init = kind.__init__

            def initialize(self, *args, original_init=original_init, **kwargs):
                nonlocal current_operations
                current_operations += 1
                return original_init(self, *args, **kwargs)

            monkey.setattr(kind, "__init__", initialize)
        for kind in (BackpropLearner, CircadianLearner):
            original_fork = kind.fork

            def fork(self, original_fork=original_fork):
                nonlocal current_operations
                current_operations += 1
                return original_fork(self)

            monkey.setattr(kind, "fork", fork)
            original = kind.snapshot_state

            def snapshot(self, original=original):
                counts["snapshots"] += 1
                return original(self)

            monkey.setattr(kind, "snapshot_state", snapshot)
            for name in ("train_batch", "predict", "restore_state"):
                monkey.setattr(kind, name, lambda *a, **k: pytest.fail("undeclared native method"))
        for variant in range(3):
            current_operations = 0
            counts["graphs"] += 1
            learner: Any = (
                CircadianLearner(
                    CircadianPredictiveCodingNetwork(
                        3, 2, 23, circadian_config=CircadianConfig(), min_hidden_dim=2
                    ),
                    learning_rate=0.01,
                    inference_steps=2,
                    inference_learning_rate=0.05,
                )
                if variant == 1
                else BackpropLearner(BackpropMLP(3, 2, 23), learning_rate=0.01)
            )
            actor = PromotableActor(
                learner,
                version="actor",
                configuration=ServingConfiguration(2, 2),
                feature_digest=lambda *a: "unused",
                metadata={},
            )
            budget = ToyBudgetSession(
                ToyExecutionBudget(max_training_updates=0, max_process_rss_bytes=1000),
                clock=lambda: 8.0,
                progress=ToyExecutionProgress(),
            )
            reads = [0]

            def rss(counter=reads):
                counter[0] += 1
                return 100

            reader_counts.append(reads)
            sampler = ProcessRssSampler(read_rss_bytes=rss)
            sampler.sample()
            budget.attach_memory(sampler)
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
                    max_retention_seconds=1000.0,
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
            builder = lambda *a: pytest.fail("builder callback")
            checkpoint = CandidateCheckpointController(
                shared,
                build_learner=builder,
                state_digest=lambda *a: "a" * 64,
                policy_digest=lambda *a: "b" * 64,
            )
            shared._sharing.pause()
            token = checkpoint.capture()  # Actual issued bounded checkpoint, no restore/training.
            counts["checkpoint_captures"] += 1
            pending = checkpoint._pending[token]
            if variant == 2:
                # Synthetic handoff graph, NOT actual checkpoint restore/publication.
                candidate: Any = runtime._candidate
                prepared = runtime._prepare_handoff(candidate.fork(), pending.view.inbox)
                runtime._inbox._retire_ledger()
                runtime._retired = True
                shared._runtime = prepared
                counts["synthetic_handoffs"] += 1
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
            policy = PromotionPolicy("utility", "bytes", 0.0, 0.0, 0.0, 1.0, 4096, ("safe",))
            promotion = ServingPromotionController(
                actor,
                policy=policy,
                evaluator=evaluator,
                build_learner=builder,
                state_digest=lambda *a: "a" * 64,
            )
            evidence = PromotionEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0, True, True)
            report = PromotionGuardReport(
                policy,
                "actor",
                "candidate",
                (0, 0, 0),
                "a" * 64,
                "b" * 64,
                "new",
                "old",
                (("new", "one"),),
                (("old", "one"),),
                (0, 0),
                (),
                0,
                evidence,
                decide_promotion(policy, evidence),
            )
            ticket = PreparedPromotion(0, report)
            array = pending.view.inbox.experiences[0].features
            bundle = _Bundle(
                learner,
                "candidate",
                actor._slot.bundle.configuration,
                {},
                {"shared": array},
                0,
            )
            promotion._pending[ticket] = _Pending(runtime, report, bundle, 0, builder)
            receipt = PromotionReceipt(0, "previous", "actor")
            promotion._latest = (receipt, replace(receipt))
            shared._sharing.pause()
            result.append((owner, life, runtime, checkpoint, token, promotion, ticket, receipt))
            assert current_operations <= 8
            counts["constructor_fork_calls_per_graph"].append(current_operations)
        yield result
        assert (
            counts["snapshots"] == counts["checkpoint_captures"] == 3
            and counts["synthetic_handoffs"] == 1
        )
        for owner, life, *_ in result:
            assert life._budget.updates_completed == 0 and life._copy_budget._charged <= 4096
            assert life._sampler._thread is None
        assert all(row[0] <= 128 for row in reader_counts)
        counts["sampler_reads_per_graph"] = [row[0] for row in reader_counts]
        folder = Path(request.config.option.basetemp)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bindings-native-work.json").write_text(
            json.dumps(counts, indent=2), encoding="utf8"
        )


@pytest.mark.parametrize("index", range(3))
def test_should_preserve_issued_checkpoint_and_synthetic_promotion_aliases(histories, index):
    owner, life, runtime, checkpoint, token, promotion, ticket, receipt = histories[index]
    before = life._copy_budget._charged
    captured = adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    holders = {
        row.kind: values(row)
        for _, row in captured.state.holders
        if row.kind.endswith(("CandidateCheckpointController", "ServingPromotionController"))
    }
    cp = holders["src.app.candidate_checkpoint.CandidateCheckpointController"]["_pending"][0]
    pr = holders["src.app.serving_promotion.ServingPromotionController"]["_pending"][0]
    copied_view = values(cp[1])["view"]
    copied_bundle = values(values(pr[1])["bundle"])
    assert copied_bundle["metadata"]["shared"] is copied_view.inbox.experiences[0].features
    assert (
        copied_view.inbox.experiences[0].features
        is not checkpoint._pending[token].view.inbox.experiences[0].features
    )
    assert pr[2] is not ticket and pr[2].report is values(pr[1])["report"]
    assert any(ref.value is token for ref in captured.authority)
    assert any(ref.value is ticket for ref in captured.authority)
    assert any(
        ref.value is receipt and ref.path.endswith("_latest.token") for ref in captured.authority
    )
    assert life._copy_budget._charged == before + captured.state.retained_payload_bytes
    old_segment = values(cp[1])["sampler_state"][1]
    final_sampler = values(values(captured.state.budget)["process_rss_sampler"])
    assert old_segment.start_bytes == final_sampler["start_bytes"] == 100
    assert old_segment.sample_count <= final_sampler["sample_count"]
    if index == 2:
        assert runtime._retired and runtime is not owner._shared._runtime
        assert any(values(row).get("_retired") is True for _, row in captured.state.holders)


@pytest.mark.parametrize(
    "field",
    [
        "budget",
        "wall_clock",
        "event_clock",
        "sampler",
        "progress",
        "resource",
        "sharing_gate",
        "actor",
        "build",
        "policy",
        "state_probe",
    ],
)
def test_should_refuse_pending_original_reference_drift_before_copy(histories, monkeypatch, field):
    owner, life, _, checkpoint, token, *_ = histories[0]
    pending = checkpoint._pending[token]
    before = life._copy_budget._charged
    monkeypatch.setattr(pending, field, object())
    with patch.object(adapter, "deepcopy", side_effect=AssertionError("premature copy")):
        with pytest.raises(ValueError, match="pending checkpoint"):
            adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


@pytest.mark.parametrize("kind", ["checkpoint", "promotion"])
def test_should_refuse_unenrolled_pending_owner_before_projection(histories, monkeypatch, kind):
    owner, life, _, checkpoint, token, promotion, ticket, _ = histories[0]
    pending = checkpoint._pending[token] if kind == "checkpoint" else promotion._pending[ticket]
    # Another fully supported owner in another original registry is still foreign.
    object.__setattr__(pending, "owner" if kind == "checkpoint" else "runtime", histories[1][2])
    before = life._copy_budget._charged
    try:
        with (
            patch.object(adapter, "deepcopy", side_effect=AssertionError("premature copy")),
            patch.object(
                adapter, "project_numpy_native", side_effect=AssertionError("unleased projection")
            ),
        ):
            with pytest.raises(ValueError, match="enrolled and leased"):
                adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
        assert life._copy_budget._charged == before
    finally:
        object.__setattr__(pending, "owner" if kind == "checkpoint" else "runtime", histories[0][2])


@pytest.mark.parametrize("field", ["_registration_guard", "_training_guard", "_payload_copy_guard"])
def test_should_refuse_foreign_bound_consent_guard(histories, monkeypatch, field):
    owner, life, runtime, *_ = histories[0]
    monkeypatch.setattr(runtime._inbox, field, getattr(histories[1][2]._inbox, field))
    before = life._copy_budget._charged
    with pytest.raises(ValueError, match="guard differs"):
        adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


def test_should_refuse_revoked_checkpoint_payload_even_when_current_inbox_is_empty(
    histories, monkeypatch
):
    owner, life, runtime, *_ = histories[0]
    monkeypatch.setattr(runtime._inbox, "_experiences", {})
    monkeypatch.setattr(runtime._inbox, "_labels", {})
    monkeypatch.setattr(runtime._inbox, "_event_ids", set())
    monkeypatch.setattr(owner, "_revoked_keys", {("episode", "one")})
    before = life._copy_budget._charged
    with patch.object(adapter, "deepcopy", side_effect=AssertionError("premature copy")):
        with pytest.raises(ValueError, match="live consent"):
            adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


@pytest.mark.parametrize(
    "change", ["integrity", "future_revision", "snapshot_width", "budget_origin", "budget_work"]
)
def test_should_refuse_corrupt_checkpoint_before_new_copy_charge(histories, monkeypatch, change):
    owner, life, runtime, checkpoint, token, *_ = histories[0]
    pending = checkpoint._pending[token]
    before = life._copy_budget._charged
    if change == "integrity":
        monkeypatch.setattr(pending, "integrity", "0" * 64)
    elif change == "future_revision":
        monkeypatch.setattr(pending, "revision", runtime._revision + 1)
    elif change == "snapshot_width":
        monkeypatch.setattr(
            pending, "view", replace(pending.view, state=replace(pending.view.state, input_dim=4))
        )
    else:
        state = dict(pending.view.budget_state)
        state["started_at" if change == "budget_origin" else "updates_completed"] = 99
        monkeypatch.setattr(pending, "view", replace(pending.view, budget_state=state))
    with patch.object(adapter, "deepcopy", side_effect=AssertionError("premature copy")):
        with pytest.raises(ValueError):
            adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


@pytest.mark.parametrize(
    "change", ["builder", "evaluator", "token_generation", "decision", "receipt"]
)
def test_should_refuse_corrupt_promotion_or_receipt_bindings(histories, monkeypatch, change):
    owner, life, _, _, _, promotion, ticket, receipt = histories[1]
    pending = promotion._pending[ticket]
    before = life._copy_budget._charged
    original = dict(vars(pending))
    token_generation = ticket.actor_generation
    if change == "builder":
        object.__setattr__(pending, "builder", object())
    elif change == "evaluator":
        monkeypatch.setattr(promotion._evaluator, "_build", lambda *a: None)
    elif change == "token_generation":
        object.__setattr__(ticket, "actor_generation", 1)
    elif change == "decision":
        object.__setattr__(
            pending,
            "report",
            replace(pending.report, decision=replace(pending.report.decision, accepted=False)),
        )
    else:
        monkeypatch.setattr(promotion, "_latest", (receipt, replace(receipt, generation=1)))
    try:
        with patch.object(adapter, "deepcopy", side_effect=AssertionError("premature copy")):
            with pytest.raises(ValueError):
                adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
        assert life._copy_budget._charged == before
    finally:
        for name, value in original.items():
            object.__setattr__(pending, name, value)
        object.__setattr__(ticket, "actor_generation", token_generation)


def test_should_capture_stale_checkpoint_revision_without_invoking_restore_guards(
    histories, monkeypatch
):
    owner, _, runtime, checkpoint, *_ = histories[0]
    monkeypatch.setattr(runtime, "_revision", runtime._revision + 1)
    with patch.object(
        CandidateCheckpointController, "_check_current", side_effect=AssertionError("restore guard")
    ):
        adapter.capture_numpy_managed_composite(owner, limits=LIMITS)


def test_should_refuse_retired_owner_without_original_historical_ledger(histories, monkeypatch):
    owner, life, runtime, *_ = histories[2]
    monkeypatch.setattr(runtime._inbox, "_historical_completed_updates", None)
    before = life._copy_budget._charged
    with pytest.raises(ValueError):
        adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


@pytest.mark.parametrize("change", ["token", "capacity"])
def test_should_refuse_unsupported_pending_token_or_capacity(histories, monkeypatch, change):
    owner, life, _, checkpoint, token, *_ = histories[0]
    if change == "token":
        monkeypatch.setattr(checkpoint, "_pending", {object(): checkpoint._pending[token]})
    else:
        monkeypatch.setattr(checkpoint, "_limit", 0)
    before = life._copy_budget._charged
    with pytest.raises(ValueError):
        adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


def test_should_keep_charges_when_original_pending_binding_changes_during_copy(
    histories, monkeypatch
):
    owner, life, _, checkpoint, token, *_ = histories[0]
    pending = checkpoint._pending[token]
    original = adapter.copy_numpy_composite
    before = life._copy_budget._charged
    calls = 0

    def copying(state, limits, memo):
        nonlocal calls
        calls += 1
        detached = original(state, limits, memo)
        monkeypatch.setattr(pending, "build", object())
        return detached

    monkeypatch.setattr(adapter, "copy_numpy_composite", copying)
    with pytest.raises(ValueError, match="pending checkpoint"):
        adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert calls == 1 and life._copy_budget._charged > before


@pytest.mark.parametrize("change", ["reader", "count", "peak", "baseline", "pid", "type"])
def test_should_refuse_nonoriginal_archived_sampler_history(histories, monkeypatch, change):
    owner, life, _, checkpoint, token, *_ = histories[1]
    pending = checkpoint._pending[token]
    reader, segment = pending.sampler_state
    if change == "reader":
        reader = lambda: 100
    elif change == "count":
        segment = replace(segment, sample_count=100000)
    elif change == "peak":
        segment = replace(segment, peak_bytes=200)
    elif change == "baseline":
        segment = replace(segment, start_bytes=True)
    elif change == "pid":
        segment = replace(segment, pid=0)
    else:
        segment = "foreign"
    monkeypatch.setattr(pending, "sampler_state", (reader, segment))
    before = life._copy_budget._charged
    with patch.object(adapter, "deepcopy", side_effect=AssertionError("premature copy")):
        with pytest.raises(ValueError):
            adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before


def test_should_bound_checkpoint_graph_before_measurement_callbacks(histories, monkeypatch):
    owner, life, _, checkpoint, token, *_ = histories[0]
    pending = checkpoint._pending[token]
    monkeypatch.setattr(
        pending, "view", replace(pending.view, state=replace(pending.view.state, input_dim=4))
    )
    before = life._copy_budget._charged
    with patch.object(
        life, "_checkpoint_bytes", side_effect=AssertionError("premature measurement")
    ):
        with pytest.raises(ValueError):
            adapter.capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == before
