"""Paired inner guards reject an unchanged literal source without fitting."""

from dataclasses import asdict
from hashlib import sha256

import pytest

from scripts import context_head_serving as serving
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import build_context, categorical_guard, snapshot_digest
from src.app.actor_shadow import ActorShadowRuntime
from src.app.serving_promotion import PromotableActor, PromotionRejected, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import LogicalClock
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import CATALOG
from test_context_catalog_control import initial


def test_should_reject_paired_new_task_failure_without_mutating_any_owner_or_budget(monkeypatch):
    monkeypatch.setattr(
        ContextPerceptronLearner,
        "train_batch",
        lambda *a, **kw: pytest.fail("zero-fitting guard check attempted training"),
    )
    spec = DecisionFeatureSpec(serving.CONTEXT_FEATURES, CATALOG)
    _, (new, old) = serving.prepare_data(spec, head_task_transfer=True)
    source = build_context(initial())
    source.restore_state(initial())
    actor = PromotableActor(
        source,
        version="literal",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"origin": "literal-correctness-fixture"},
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="literal",
        candidate_version="untrained-paired",
        clock=LogicalClock(),
        budget=budget,
        actor=actor,
    )
    controller = ServingPromotionController(
        actor,
        policy=serving.policy(),
        evaluator=categorical_guard(spec, lambda: 0.0),
        build_learner=build_context,
        state_digest=snapshot_digest,
    )
    source_before = source.snapshot_state()
    actor_before = actor.serving_snapshot()
    candidate_before = runtime.candidate_snapshot()
    budget_before = (id(budget.budget), budget.started_at, id(budget.clock))
    with pytest.raises(PromotionRejected) as rejected:
        controller.prepare(runtime, new, old, now=0, training_ids=frozenset())
    report = rejected.value.report
    assert not report.decision.accepted
    assert "new_task_utility" in report.decision.reasons
    assert "new_task_gain" in report.decision.reasons
    assert report.evidence.candidate_new_utility == report.evidence.actor_new_utility
    assert report.evidence.candidate_old_utility == report.evidence.actor_old_utility == 1.0
    assert len(report.new_guard_keys) == 6 and len(report.old_guard_keys) == 12
    assert actor.serving_snapshot() == actor_before
    assert actor_before.generation == 0 and actor_before.actor_version == "literal"
    assert runtime.candidate_snapshot() == candidate_before
    assert source.snapshot_state() == source_before
    assert not runtime.applied_updates and budget.updates_completed == 0
    assert (id(budget.budget), budget.started_at, id(budget.clock)) == budget_before
    assert asdict(budget.budget)["max_training_updates"] == 0
