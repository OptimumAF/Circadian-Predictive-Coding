"""Existing guard/serving/rollback after paid retained learning, bounded fixtures."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from time import monotonic

import pytest

from scripts import context_head_live as live
from scripts import recurring_ticket_workflow_example as entry
from scripts import run_bp_ticket_serving as schedule
from scripts.context_training_streams import WITHHELD_CONTEXT_RECIPES
from src.core.adaptation_action import AdaptationProgress
from src.core.resource_sharing import SharingSnapshot, SharingLimits
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitSnapshot
from src.shared.process_memory import ProcessRssSampler
from test_context_linear_catalog_study import literal_state
from test_context_same_context_ack import actor_for, publish_fixture, ScalarLearner, ScalarState
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def boundary():
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=9), lambda: 0.0)
    budget.updates_completed, replay.replay_examples_completed = 2, 9
    candidate = SimpleNamespace(
        wake_updates=2, consolidation_attempts=1, consolidations_completed=1
    )
    runtime = SimpleNamespace(
        applied_updates=("wake0", "wake1"),
        applied_consolidations=("joint0",),
        acknowledged_experiences=(),
        _base_actor_version="a0",
        _candidate_version="c0",
        candidate_snapshot=lambda: candidate,
    )
    paid = dict(
        attempted_updates=1,
        returned_updates=1,
        committed_updates=1,
        charged_exposures=9,
        events=[dict(status="complete")],
    )
    helper = SimpleNamespace(
        _budget=budget, _runtime=runtime, _observed=2, _replay_budget=replay, report=paid
    )
    progress = AdaptationProgress(
        wake_admissions=2,
        poll_updates_returned=2,
        native_attempts=3,
        native_completions=3,
        replay_native_attempts=1,
        replay_native_completions=1,
        replay_commits=1,
        replay_admissions=1,
        replay_examples_charged=9,
    )
    sharing = SharingSnapshot(SharingLimits(1, 4, 1), 0, False, False, 4, ())
    shared = SimpleNamespace(
        adaptation_snapshot=lambda: SimpleNamespace(progress=progress),
        _sharing=SimpleNamespace(snapshot=lambda: sharing),
    )
    return helper, shared, budget, runtime


def test_should_bind_exact_combined_history_and_copy_guard_budget_values():
    helper, shared, budget, runtime = boundary()
    before = schedule._rehearsal_drain_values(helper, shared, budget, runtime)
    object.__setattr__(budget.budget, "max_training_updates", 3)
    after = schedule._rehearsal_drain_values(helper, shared, budget, runtime)
    assert before != after
    assert before[0][1].max_training_updates == 2


@pytest.mark.parametrize(
    "fault",
    [
        "native",
        "replay",
        "unknown",
        "receipts",
        "candidate",
        "event",
        "paid",
        "reserved",
        "runtime",
        "admissions",
        "decision",
        "typed_count",
    ],
)
def test_should_refuse_inconsistent_paid_history_before_promotion(fault):
    helper, shared, budget, runtime = boundary()
    original = shared.adaptation_snapshot().progress
    if fault == "native":
        shared.adaptation_snapshot = lambda: SimpleNamespace(
            progress=replace(original, native_completions=2)
        )
    elif fault == "replay":
        shared.adaptation_snapshot = lambda: SimpleNamespace(
            progress=replace(original, replay_examples_charged=8)
        )
    elif fault == "unknown":
        shared.adaptation_snapshot = lambda: SimpleNamespace(
            progress=replace(original, unreturned_work_unknown=True)
        )
    elif fault == "receipts":
        runtime.applied_consolidations = ()
    elif fault == "candidate":
        runtime.candidate_snapshot().consolidations_completed = 0
    elif fault == "event":
        helper.report["events"][0]["status"] = "failed"
    elif fault == "paid":
        helper.report["returned_updates"] = 0
    elif fault == "reserved":
        original_sharing = shared._sharing.snapshot()
        shared._sharing.snapshot = lambda: replace(original_sharing, admitted_updates=3)
    elif fault == "admissions":
        shared.adaptation_snapshot = lambda: SimpleNamespace(
            progress=replace(original, replay_admissions=2)
        )
    elif fault == "decision":
        shared.adaptation_snapshot = lambda: SimpleNamespace(
            progress=replace(original, decisions_refused=1)
        )
    elif fault == "typed_count":
        shared.adaptation_snapshot = lambda: SimpleNamespace(
            progress=replace(original, replay_admissions=True)
        )
    else:
        helper._runtime = object()
    with pytest.raises(ValueError, match="complete original"):
        schedule._rehearsal_drain_values(helper, shared, budget, runtime)


def test_should_keep_unknown_publication_when_slot_commits_but_callback_return_is_lost():
    actor = actor_for(ScalarLearner())
    report: dict[str, Any] = {}

    def publish(runtime, tick):
        publish_fixture(actor, ScalarState(1, 13))
        raise RuntimeError("lost original publication return")

    with pytest.raises(RuntimeError, match="lost original"):
        schedule._publish_at_drain(actor, None, publish, 6, None, report, 6)
    assert report["drain_publication"]["publication_unknown"]
    assert not report["drain_publication"]["completed"]
    assert actor.serving_snapshot().generation == 1


def qualified_literal(*, new_correct=False):
    """Handwritten coefficients for guard correctness, never a native qualification."""
    state = literal_state()
    words = ("invoice", "legacy", "parcel", "password", "ready", "sealed", "staged")
    from src.core.ticket_workflow import ACTION_IDS

    heads = []
    for index, head in enumerate(state.heads):
        coefficients = [[0.0] * 6 for _ in range(8)]
        mapping = dict(
            invoice="route" if index == 0 else "inspect",
            legacy="route" if index == 0 else "defer",
            parcel="defer",
            password="inspect",
            sealed="open",
            staged="commit",
        )
        if new_correct and index == 0:
            mapping["legacy"] = "retire"
        for word, action in mapping.items():
            coefficients[words.index(word)][ACTION_IDS.index(action)] = 1.0
        model = CategoricalLinearFit(
            7, 6, 6, tuple(tuple(row) for row in coefficients), 6, (1.0,) * 6, 0.0
        )
        heads.append(
            LinearFitSnapshot(head.policy, model, head.completed_updates, head.training_examples)
        )
    return replace(state, heads=tuple(heads))


def test_should_refuse_changed_guard_budget_before_a_real_slot_commit():
    from src.adapters import context_linear_serving as binding
    from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.serving_promotion import ServingPromotionController
    from src.core.actor_ports import ConsolidatedState
    from src.core.experience import LogicalClock
    from src.core.learner_ports import TrainingDiagnostic
    from test_context_linear_serving import actor_for, SPEC

    state = qualified_literal()
    learner = ContextLinearFitLearner(state.policy)
    learner.restore_state(state)
    actor = actor_for(learner)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    runtime = ActorShadowRuntime(
        learner,
        actor_version=actor.version,
        candidate_version="c0",
        clock=LogicalClock(),
        budget=budget,
        max_consolidations=1,
        actor=actor,
    )
    # Handwritten candidate state only; this fixture performs no fit and does not
    # claim that its coefficient/rank metadata was experimentally obtained.
    runtime.consolidate(
        "literal-candidate",
        lambda _: ConsolidatedState(
            qualified_literal(new_correct=True), TrainingDiagnostic("literal_guard_candidate", 0.0)
        ),
    )
    controller = ServingPromotionController(
        actor,
        policy=live.policy(),
        evaluator=binding.categorical_guard(SPEC, monotonic),
        build_learner=binding.build_context,
        state_digest=binding.snapshot_digest,
    )
    _, (new, old) = live.prepare_data(SPEC)
    ticket = controller.prepare(runtime, new, old, now=0, training_ids=frozenset())

    def require_original_budget():
        if budget.budget.max_training_updates != 0:
            raise ValueError("guard changed original budget before commit")

    object.__setattr__(budget.budget, "max_training_updates", 1)
    with pytest.raises(ValueError, match="before commit"):
        live._commit_after_guard(controller, runtime, ticket, require_original_budget)
    assert actor.serving_snapshot().generation == 0
    assert actor.serving_snapshot().model_state == state
    controller.discard(ticket)


def test_should_guard_publish_serve_and_exactly_rollback_retained_candidate():
    state = qualified_literal()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        report = live.run_case(
            None,
            _linear_fit=True,
            _owned_source=(state, lambda: None),
            _memory=memory,
            _retained_joint=True,
            _publish_retained=True,
            stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
        )
    assert report["complete"], report.get("error")
    assert report["source_unchanged"] and report["rollback_equal_source"]
    assert not report["actor_unchanged"] and report["rollback_generation"] == 2
    publication = report["drain_publication"]
    assert (
        publication["completed"]
        and publication["accepted"]
        and not publication["publication_unknown"]
    )
    assert publication["paid_rehearsal"] == dict(
        wake_receipts=6,
        joint_receipts=5,
        native_completions=11,
        replay_examples=45,
        reserved_attempts=12,
    )
    assert report["training_work"] == dict(
        attempted_updates=11, attempted_rows=72, returned_updates=11
    )
    assert report["guard_report"]["decision"]["accepted"]
    published = [e for e in report["episodes"] if e["phase"] == "published"]
    rollback = [e for e in report["episodes"] if e["phase"] == "rollback"]
    assert len(published) == len(rollback) == 6
    assert all(e["episode"]["success"] for e in published)
    assert all(
        e["role"] == "controller_train"
        and all(
            r["actor_version"] == publication["actor_version_after"] and r["generation"] == 1
            for r in e["reads"]
        )
        for e in published
    )
    assert all(
        e["role"] == "controller_train"
        and all(
            r["actor_version"] == publication["actor_version_before"] and r["generation"] == 2
            for r in e["reads"]
        )
        for e in rollback
    )
    assert not next(e for e in rollback if (e["policy_version"], e["concept"]) == ("v1", "legacy"))[
        "episode"
    ]["success"]
    assert any(
        r["episode"]["success"]
        and r["tick"] > report["publication_tick"]
        and r["concept"] == "legacy"
        and r["policy_version"] == "v1"
        for r in report["live_serving"]["records"]
    )
    assert report["final_examples"] == report["probe_updates"] == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(_publish_retained=1),
        dict(_retained_joint=False),
        dict(acknowledge_correct=True),
        dict(reference_mode="drift"),
    ],
)
def test_should_refuse_mixed_retained_publication_before_source(change, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    kwargs = dict(
        _linear_fit=True,
        _owned_source=(object(), lambda: None),
        _retained_joint=True,
        _publish_retained=True,
        stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        live.run_case(None, **kwargs)


def test_should_route_explicit_public_mode_to_existing_workflow_error_report(monkeypatch):
    monkeypatch.setattr(
        entry,
        "prepare_training",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("fixture reached training")),
    )
    report = entry.run_example(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="retained_joint_publish",
    )
    assert (
        report["complete"] is False and report["error"] == "RuntimeError: fixture reached training"
    )
