"""Causal return after measured refusal, plus ordinary paired TRAIN deployment.

Scalar fixtures test state management without model work. The two explicit
paired integrations use the existing literal source and pay their native fits.
"""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as schedule
from scripts.context_training_streams import CONTEXT_PAIRED_REHEARSAL_RECIPES
from scripts.ticket_serving_timeline import FollowupTrainingPlan
from src.app.serving_promotion import PromotableActor, PromotionRejected
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.promotion_guard import PromotionEvidence, PromotionGuardReport, decide_promotion
from src.core.serving_ports import ServingConfiguration, ServingSnapshot
from test_context_catalog_control import source as source_fixture
from test_post_publication_learning import LiteralLearner, LiteralLive, batches


def digest(state):
    return sha256(json.dumps(state).encode()).hexdigest()


def refusal(actor, owner, tick):
    candidate = owner.candidate_snapshot()
    policy = live.policy()
    evidence = PromotionEvidence(0.0, 0.0, 1.0, 1.0, 0.0, 10, True, True)
    return PromotionGuardReport(
        policy=policy,
        actor_version=actor.version,
        learner_version=candidate.learner_version,
        candidate_revision=(candidate.wake_updates, 0, 0),
        actor_snapshot_digest=digest(actor.serving_snapshot().model_state),
        candidate_snapshot_digest=digest(candidate.state),
        new_guard_task_id="new",
        old_guard_task_id="old",
        new_guard_keys=(("new", "0"),),
        old_guard_keys=(("old", "0"),),
        label_arrivals=(tick, tick),
        training_ids=(),
        assessed_at=tick,
        evidence=evidence,
        decision=decide_promotion(policy, evidence),
    )


def scalar_schedule(monkeypatch, *, optin=True, fault=None, limit=2, mutate=None, last_only=False):
    source = LiteralLearner()
    actor = PromotableActor(
        source,
        version="old",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"fixture": "scripted-state-only"},
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=limit), lambda: 0.0)
    gate = schedule.ServingPriorityGate(
        schedule.SharingLimits(1, 2, 1), resource_available=lambda: True
    )
    report: dict[str, Any] = {}
    owners = []
    original_runtime = schedule.ActorShadowRuntime

    def runtime(*a, **kw):
        result = original_runtime(*a, **kw)
        owners.append(result)
        return result

    monkeypatch.setattr(schedule, "ActorShadowRuntime", runtime)
    changed = False

    def require():
        nonlocal changed
        if mutate is not None and not last_only and "drain_publication" in report and not changed:
            changed = True
            mutate(actor, report)

    def reject(owner, tick):
        guard = refusal(actor, owner, tick)
        if fault == "none":
            return None
        if fault == "lost_return":
            raise RuntimeError("lost callback return")
        if fault == "actor_digest":
            guard = replace(guard, actor_snapshot_digest="0" * 64)
        elif fault == "candidate_digest":
            guard = replace(guard, candidate_snapshot_digest="0" * 64)
        elif fault == "revision":
            guard = replace(guard, candidate_revision=(True, 0, 0))
        elif fault == "decision":
            guard = replace(guard, decision=replace(guard.decision, reasons=("fabricated",)))
        elif fault == "tick":
            guard = replace(guard, assessed_at=float(tick))
        elif fault == "actor_mutated":
            actor._slot.bundle.learner.restore_state(99)
        elif fault == "candidate_mutated":
            owner._candidate.restore_state(99)
        elif fault == "budget_mutated":
            budget.updates_completed += 1
        raise PromotionRejected(guard)

    class WindowLive(LiteralLive):
        @contextmanager
        def opportunity(self, tick):
            with super().opportunity(tick):
                yield
            if last_only and tick == 9:
                mutate(actor, report)

    def run():
        return schedule._reference_schedule(
            source,
            actor,
            batches(),
            budget,
            "periodic",
            report,
            require,
            lambda: None,
            sharing_gate=gate,
            live=WindowLive(gate),
            on_drain=reject,
            followup_plan=FollowupTrainingPlan(1, 4),
            **(dict(rejected_guard_digest=digest) if optin else {}),
        )

    return run, report, actor, budget, owners


def test_should_continue_declared_return_after_measured_rejection_without_publication(monkeypatch):
    run, report, actor, budget, owners = scalar_schedule(monkeypatch)
    pin = (id(budget.budget), budget.started_at, id(budget.clock))
    owner, horizon = run()
    assert owners == [owner] and owner._budget is budget and horizon == 10
    assert actor.serving_snapshot().model_state == 0 and actor.version == "old"
    assert actor.serving_snapshot().generation == 0
    assert owner.candidate_snapshot().state == 2 and budget.updates_completed == 2
    assert [
        (r.actor_version, r.observed_at, r.arrived_at, r.applied_at) for r in owner.applied_updates
    ] == [("old", 0, 1, 2), ("old", 4, 5, 6)]
    assert report["followup_training"]["status"] == "complete"
    row = report["drain_publication"]
    assert row["completed"] and not row["accepted"] and not row["publication_unknown"]
    assert not row["rejected_guard"]["decision"]["accepted"]
    assert report["feedback_invalidations"] == []
    assert pin == (id(budget.budget), budget.started_at, id(budget.clock))


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "lost_return",
        "actor_digest",
        "candidate_digest",
        "revision",
        "decision",
        "tick",
        "actor_mutated",
        "candidate_mutated",
        "budget_mutated",
    ],
)
def test_should_refuse_unmeasured_or_changed_refusal_before_more_training(monkeypatch, fault):
    run, report, actor, budget, owners = scalar_schedule(monkeypatch, fault=fault)
    with pytest.raises((ValueError, RuntimeError)):
        run()
    assert budget.updates_completed == (2 if fault == "budget_mutated" else 1)
    assert len(owners[0].applied_updates) == 1
    assert report["controller_work"]["prediction_rows"] == 1


@pytest.mark.parametrize("kind", ["row_replaced", "guard_replaced", "value", "model", "metadata"])
def test_should_refuse_replaced_prior_evidence_or_changed_actor_before_new_observation(
    monkeypatch, kind
):
    def mutate(actor, report):
        row = report["drain_publication"]
        if kind == "row_replaced":
            report["drain_publication"] = deepcopy(row)
        elif kind == "guard_replaced":
            row["rejected_guard"] = deepcopy(row["rejected_guard"])
        elif kind == "value":
            row["rejected_guard"]["decision"]["reasons"] = ("changed",)
        elif kind == "model":
            actor._slot.bundle.learner.restore_state(99)
        else:
            actor._slot.bundle.metadata["changed"] = True

    run, report, actor, budget, owners = scalar_schedule(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError):
        run()
    assert budget.updates_completed == 1
    assert report["controller_work"]["prediction_rows"] == 1


@pytest.mark.parametrize("fault", ["model", "evidence"])
def test_should_refuse_mutation_during_the_final_opportunity_before_completion(monkeypatch, fault):
    def mutate(actor, report):
        if fault == "model":
            actor._slot.bundle.learner.restore_state(99)
        else:
            report["drain_publication"] = deepcopy(report["drain_publication"])

    run, report, actor, budget, owners = scalar_schedule(monkeypatch, mutate=mutate, last_only=True)
    with pytest.raises(ValueError):
        run()
    assert budget.updates_completed == 2
    assert report["followup_training"]["status"] == "incomplete"
    assert report["controller_work"]["prediction_rows"] == 2


def test_should_allow_ordinary_serving_cache_updates_after_rejection(monkeypatch):
    def serve(actor, report):
        actor.serve(np.zeros((1, 7)), now=3)

    run, report, actor, budget, owners = scalar_schedule(monkeypatch, mutate=serve)
    run()
    assert budget.updates_completed == 2 and actor.serving_snapshot().cache


@pytest.mark.parametrize("fault", ["model", "metadata", "configuration"])
def test_should_also_pin_complete_published_actor_during_accepted_followup(fault):
    current: ServingSnapshot[int, Any] = ServingSnapshot(
        1, "c0", 0, ServingConfiguration(3, 2), {}, {"fixture": True}, -1
    )
    actor = SimpleNamespace(serving_snapshot=lambda: current)
    row = dict(
        accepted=True,
        completed=True,
        publication_unknown=False,
        generation_before=0,
        actor_version_after="c0",
    )
    report = dict(drain_publication=row)
    pin = (row, deepcopy(row), None, schedule._followup_actor_values(current))
    assert schedule._require_followup_boundary(actor, report, pin) == (1, "c0")
    changes: dict[str, dict[str, Any]] = dict(
        model=dict(model_state=1),
        metadata=dict(metadata={"changed": True}),
        configuration=dict(configuration=ServingConfiguration(4, 2)),
    )
    current = replace(current, **changes[fault])
    with pytest.raises(ValueError, match="actor changed"):
        schedule._require_followup_boundary(actor, report, pin)


def test_should_preserve_default_rejection_refusal_and_original_exhausted_budget(monkeypatch):
    run, report, actor, budget, owners = scalar_schedule(monkeypatch, optin=False)
    with pytest.raises(PromotionRejected):
        run()
    assert budget.updates_completed == 1
    run, report, actor, budget, owners = scalar_schedule(monkeypatch, limit=1)
    with pytest.raises(ToyExecutionStopped):
        run()
    assert budget.updates_completed == 1 and len(owners[0].applied_updates) == 1


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_should_validate_boundary_optin_before_source_io(monkeypatch, value):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source read"))
    with pytest.raises(ValueError):
        live.run_case("unused", publish_after_learning=value)
    with pytest.raises(ValueError):
        live.run_case("unused", publish_after_learning=True)


@pytest.mark.parametrize("index", [0, 1])
def test_should_deploy_paired_learning_and_keep_causal_return_on_one_native_budget(
    monkeypatch, index
):
    source = source_fixture.__wrapped__(monkeypatch)
    rng = deepcopy(np.random.get_state())
    result = live.run_case(
        "literal-source",
        stream_recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[index],
        publish_after_learning=True,
    )
    assert result["complete"], result.get("error")
    assert result["training_work"] == dict(
        attempted_updates=6, attempted_rows=48, returned_updates=6
    )
    assert result["budget"]["updates"] == result["budget"]["limits"]["max_training_updates"] == 6
    assert result["learning_boundary_plan"] == asdict(FollowupTrainingPlan(4, 12))
    assert result["learning_boundary_tick"] < 12
    accepted = result["learning_boundary_guard"]["decision"]["accepted"]
    assert result["return_publication_attempted"] is not accepted
    assert result["publication_tick"] == (
        result["learning_boundary_tick"] if accepted else result["return_publication"]["tick"]
    )
    assert result["publication_tick"] < 24
    receipts = result["applied_updates"]
    assert [(r["observed_at"], r["arrived_at"]) for r in receipts] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (12, 13),
        (13, 14),
    ]
    assert [r["actor_version"] for r in receipts[4:]] == ["c0" if accepted else "qualified-12"] * 2
    assert result["followup_training"]["status"] == "complete"
    assert result["candidate_state"]["completed_updates"] == 18
    assert result["candidate_state"]["training_examples"] == 120
    assert result["rollback_equal_source"] and result["final_actor"]["generation"] == 2
    assert result["final_actor"]["model_state"] == asdict(source)
    assert len(result["episodes"]) == 12 and result["source_unchanged"]
    assert all(e["episode"]["success"] for e in result["episodes"][:6])
    assert schedule._same_value(np.random.get_state(), rng)
    assert result["probe_updates"] == result["replay_examples"] == result["final_examples"] == 0


def test_should_keep_early_publication_and_permanent_failed_return_charge(monkeypatch):
    source_fixture.__wrapped__(monkeypatch)
    original = live.ContextPerceptronLearner.train_batch
    entered = 0

    def lose_fifth_return(owner, features, targets):
        nonlocal entered
        entered += 1
        result = original(owner, features, targets)
        if entered == 5:
            raise RuntimeError("lost return after early publication")
        return result

    monkeypatch.setattr(live.ContextPerceptronLearner, "train_batch", lose_fifth_return)
    result = live.run_case(
        "literal-source",
        stream_recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[1],
        publish_after_learning=True,
    )
    assert not result["complete"] and "lost return after early publication" in result["error"]
    assert entered == 5
    assert result["training_work"] == dict(
        attempted_updates=5, attempted_rows=42, returned_updates=4
    )
    assert result["budget"]["updates"] == 4
    assert result["budget"]["limits"]["max_training_updates"] == 6
    assert result["promotion_receipt"]["generation"] == result["final_actor"]["generation"] == 1
    assert result["final_actor"]["model_state"] == result["native_calls"][3]["state"]
    assert result["source_unchanged"] and "rollback_generation" not in result
