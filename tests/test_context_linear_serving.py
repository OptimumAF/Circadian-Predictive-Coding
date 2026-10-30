"""Real LS serving/admission and atomic publication; no historical weights."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic

import pytest

from scripts import context_head_rehearsal as owned
from scripts import context_head_serving as data_source
from scripts import recurring_ticket_workflow_example as entry
from scripts.run_ticket_workflow import CONTEXT_FEATURES, CONTEXT_TASKS
from src.adapters import context_linear_serving as binding
from src.adapters import context_serving as legacy
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.adapters.context_linear_fit_decision import ContextLinearFitDecisionPredictor
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.app.coupled_attribution import AttributionStream
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.context_linear_state import ContextLinearFitPolicy
from src.core.context_perceptron import ContextPerceptronPolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import (
    ACTION_IDS,
    CATALOG,
    WorkflowCapabilities,
    WorkflowState,
    workflow_request,
)
from src.shared.process_memory import ProcessRssSampler

SPEC = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)


def request():
    return workflow_request(
        WorkflowState("test", "invoice", policy_version="v1"),
        WorkflowCapabilities(ACTION_IDS),
        "request",
        include_policy_context=True,
    )


def actor_for(source):
    return PromotableActor(
        source,
        version="source",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"source": "new-correctness-fixture"},
    )


@pytest.fixture(scope="module")
def source_case():
    # Original TRAIN producer once; no guard/probe labels converted to training.
    training, _ = data_source.prepare_data(SPEC, head_task_transfer=True)
    old = (training[1], training[3])
    packets = tuple(
        replace(old[i], context=f"old-{n}", row_ids=tuple(f"{n}-{j}" for j in range(6)))
        for n, i in enumerate((0,) * 4 + (1,) * 4 + (0,) * 4)
    )
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        budget = ToyBudgetSession(owned._limits(12), monotonic)
        budget.attach_memory(sampler)
        source = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
        initial = source.snapshot_state()
        stream = AttributionStream(
            source,
            packets,
            (),
            checkpoint_updates=(0, 4, 8, 12),
            label_delay=1,
            budget=budget,
            training_only=True,
        )
        points = [dict(source_updates=0, state=asdict(initial))]
        for n in (4, 8, 12):
            stream.run_until(n)
            points.append(dict(source_updates=n, state=asdict(stream.candidate_snapshot())))
        state = stream.candidate_snapshot()
        reader = ContextLinearFitDecisionPredictor(SPEC, state)
        episodes = []
        for version, concept in CONTEXT_TASKS:
            episode = run_workflow_episode(
                WorkflowState(f"source-{version}-{concept}", concept, policy_version=version),
                reader,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                include_policy_context=True,
            )
            assert episode["complete"] and episode["success"]
            episodes.append(
                dict(
                    source_updates=12,
                    policy_version=version,
                    concept=concept,
                    role="outer_selection",
                    episode=episode,
                )
            )
        report = dict(
            definition="separate_context_linear_ticket_workflow_v1",
            source_unchanged=True,
            actor_unchanged=True,
            serving_source_training=dict(completed_updates=12, applied_update_count=12),
            summary=dict(ordinary_successes=4),
            checkpoints=points,
            episodes=episodes,
        )

        def guard():
            assert source.snapshot_state() == initial
            assert stream._runtime.actor.snapshot_state().state == initial
            assert stream.candidate_snapshot() == state

        yield state, report, budget, guard, sampler, stream


def test_should_preserve_complete_ls_record_and_refuse_default_perceptron_copy(source_case):
    state = source_case[0]
    wire = binding.snapshot_bytes(state)
    assert json.loads(wire) == json.loads(json.dumps(asdict(state)))
    assert binding.snapshot_digest(state) == sha256(wire).hexdigest()
    assert binding.snapshot_record_bytes(state) == len(wire) > 768
    with pytest.raises(ValueError):
        legacy.snapshot_bytes(state)


def test_should_use_real_owned_source_budget_and_refuse_default_or_private_ls(source_case):
    state, report, budget, guard, sampler, _ = source_case
    copied, check = owned.prepare_owned_source(
        state,
        SPEC,
        deepcopy(report),
        budget,
        check_boundary=guard,
        training_key="serving_source_training",
        linear_fit=True,
    )
    check()
    assert copied == state and copied is not state
    assert budget.updates_completed == 12 and budget.process_rss_sampler is sampler
    with pytest.raises(ValueError):
        owned.prepare_owned_source(
            state,
            SPEC,
            report,
            budget,
            check_boundary=guard,
            training_key="serving_source_training",
        )
    with pytest.raises(ValueError):
        owned.prepare_owned_source(
            state, SPEC, report, budget, check_boundary=guard, linear_fit=True
        )


@pytest.mark.parametrize("fault", ["role", "success", "replace", "definition", "state", "budget"])
def test_should_refuse_corrupted_or_replaced_owned_evidence_before_new_work(source_case, fault):
    state, original, budget, guard, _, _ = source_case
    report = deepcopy(original)
    if fault == "budget":
        foreign = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=13, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
            ),
            monotonic,
        )
        with pytest.raises(ValueError):
            owned.prepare_owned_source(
                state,
                SPEC,
                report,
                foreign,
                check_boundary=guard,
                training_key="serving_source_training",
                linear_fit=True,
            )
        return
    _, check = owned.prepare_owned_source(
        state,
        SPEC,
        report,
        budget,
        check_boundary=guard,
        training_key="serving_source_training",
        linear_fit=True,
    )
    if fault == "replace":
        report["episodes"][0] = deepcopy(report["episodes"][0])
    elif fault == "role":
        report["episodes"][0]["role"] = "probe_test"
    elif fault == "success":
        report["episodes"][0]["episode"]["success"] = False
    elif fault == "definition":
        report["definition"] = "private_probe"
    else:
        report["checkpoints"][-1]["state"]["training_examples"] += 1
    with pytest.raises(ValueError):
        check()
    assert budget.updates_completed == 12


def test_should_publish_guarded_atomic_ls_slot_and_rollback_without_training_source(source_case):
    state, _, budget, _, _, runtime = source_case
    source = ContextLinearFitLearner(state.policy)
    actor = actor_for(source)
    predictor = binding.ContextLinearServingPredictor(actor, SPEC)
    # Existing runtime's source actor is separate; bind publication to its actor.
    runtime_actor = runtime._runtime.actor
    assert runtime_actor.snapshot_state().state.completed_updates == 0
    # A fresh promotion runtime shares the same zero actor and consolidates the
    # already-paid detached source state; this fixture performs no additional fit.
    from src.app.actor_shadow import ActorShadowRuntime
    from src.core.actor_ports import ConsolidatedState
    from src.core.learner_ports import TrainingDiagnostic
    from src.core.experience import LogicalClock

    owner = ActorShadowRuntime(
        source,
        actor_version="source",
        candidate_version="paid-train",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0),
        actor=actor,
    )
    owner.consolidate(
        "paid-state", lambda _: ConsolidatedState(state, TrainingDiagnostic("fixture", 0.0))
    )
    # Guards are original TRAIN-derived feature/targets with explicit inner role.
    from src.core.promotion_guard import GuardBatch

    train = runtime._training[0]
    new = GuardBatch(
        "new",
        tuple(("guard-new", str(i)) for i in range(6)),
        0,
        train.features.copy(),
        train.targets.copy(),
        "inner_guard",
    )
    old = replace(new, task_id="old", sample_keys=tuple(("guard-old", str(i)) for i in range(6)))
    controller = ServingPromotionController(
        actor,
        policy=data_source.policy(),
        evaluator=binding.categorical_guard(SPEC, lambda: 0.0),
        build_learner=binding.build_context,
        state_digest=binding.snapshot_digest,
        max_pending=1,
    )
    before = predictor.serve(request())
    ticket = controller.prepare(owner, new, old, now=0, training_ids=frozenset())
    assert ticket.report.decision.accepted
    receipt = controller.commit(owner, ticket)
    served = predictor.serve(request())
    assert (served.generation, served.completed_updates) == (1, 12)
    assert actor.serving_snapshot().model_state == state
    assert controller.rollback(receipt) == 2
    restored = predictor.serve(request())
    assert restored.result == before.result
    assert source.snapshot_state().completed_updates == 0 and budget.updates_completed == 12
    assert predictor.native_work.decision_returns == 3


def test_should_keep_lost_read_work_and_latch_fault(source_case, monkeypatch):
    source = ContextLinearFitLearner(source_case[0].policy)
    source.restore_state(source_case[0])
    predictor = binding.ContextLinearServingPredictor(actor_for(source), SPEC)
    native = ContextLinearFitDecisionPredictor.__call__

    def lost(reader, query):
        native(reader, query)
        raise RuntimeError("lost typed read")

    monkeypatch.setattr(ContextLinearFitDecisionPredictor, "__call__", lost)
    with pytest.raises(RuntimeError):
        predictor.serve(request())
    assert predictor.native_work.affine_returns == predictor.native_work.routing_returns == 1
    with pytest.raises(ValueError, match="stopped"):
        predictor.serve(request())
    assert predictor.work["attempts"] == 1 and predictor.work["returns"] == 0


def test_should_preserve_default_predictor_and_refuse_wrong_state_family():
    old = ContextPerceptronLearner(ContextPerceptronPolicy(9, 6, (4, 5)))
    actor = actor_for(old)
    assert legacy.ContextServingPredictor(actor, SPEC).serve(request()).completed_updates == 0
    with pytest.raises(ValueError):
        binding.ContextLinearServingPredictor(actor, SPEC).serve(request())


@pytest.mark.parametrize(
    "options",
    [
        dict(separate_context_heads=False),
        dict(private_adaptation=True),
        dict(paired_serving="invalid"),
    ],
)
def test_should_refuse_unsupported_linear_serving_before_data(monkeypatch, options):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    args = dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="joint",
    )
    args.update(options)
    with pytest.raises(ValueError):
        entry.run_example(**args)


@pytest.mark.parametrize("method", ["perceptron", "linear_fit"])
def test_should_forward_real_cli_method_and_protect_existing_output(method, tmp_path, monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", fake)
    path = tmp_path / "one.json"
    args = [
        "--output",
        str(path),
        "--context-return",
        "--separate-context-heads",
        "--paired-serving",
        "joint",
        "--method",
        method,
    ]
    assert entry.main(args) == 0
    assert calls == [
        dict(
            context_return=True,
            separate_context_heads=True,
            paired_serving="joint",
            **(dict(method="linear_fit") if method == "linear_fit" else {}),
        )
    ]
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert len(calls) == 1


def test_should_execute_full_existing_cli_owned_linear_live_route():
    destination = (
        Path("artifacts/runs/context-linear-serving-20261029") / f"{sys.platform}-integration.json"
    )
    # Actual parser/dispatch/output boundary, one full integration/platform.
    status = entry.main(
        [
            "--output",
            str(destination),
            "--method",
            "linear_fit",
            "--context-return",
            "--separate-context-heads",
            "--paired-serving",
            "joint",
        ]
    )
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert status == 0 and result["complete"], result.get("error")
    assert result["definition"] == "separate_context_linear_ticket_workflow_v1"
    child = result["paired_serving"]
    assert child["definition"] == "fresh_context_linear_paired_serving_v1"
    assert child["complete"] and child["owned_source_verified"]
    assert child["source_unchanged"] and child["rollback_equal_source"]
    assert child["typed_native_work"]["decision_returns"] > 0


def test_should_preserve_owned_failed_work_before_publication_without_inference(monkeypatch):
    from scripts import context_head_live as live
    from src.core.learner_ports import TrainingUpdateFailed

    work = dict(
        fit_attempts=1,
        fit_rows_attempted=3,
        fit_returns=0,
        candidate_commits=0,
        unreturned_work_unknown=True,
        unpublished_work=True,
    )
    error = TrainingUpdateFailed("injected lost LS return", "owned-ls-fixture", work)

    def lost(_):
        raise error

    monkeypatch.setattr(live, "_source_bytes", lost)
    result = live.run_case("unopened-source.json")
    assert not result["complete"] and "injected lost LS return" in result["error"]
    assert result["native_failure_definition"] == error.definition
    assert result["native_failure_work"] == error.work
    assert "promotion_receipt" not in result
    result["native_failure_work"]["fit_attempts"] = 99
    assert error.work["fit_attempts"] == 1
