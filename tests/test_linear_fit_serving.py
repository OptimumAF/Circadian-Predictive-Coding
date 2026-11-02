"""Shared serving ports and tiny genuine fit; not an ordinary workflow study."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json

import numpy as np
import pytest

from scripts.context_head_serving import policy
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters import linear_fit_serving as binding
from src.adapters.linear_fit_decision import LinearFitDecisionPredictor
from src.adapters.linear_fit_learner import LinearFitLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.serving_promotion import PromotableActor, ServingPromotionController, PromotionRejected
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.categorical_linear_state import LinearFitPolicy
from src.core.context_linear_state import ContextLinearFitPolicy
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.promotion_guard import GuardBatch
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import CATALOG
from src.core.typed_decision import DecisionRequest

SPEC = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
CONTEXT_COLUMNS = (SPEC.vocabulary.index("policy_v1"), SPEC.vocabulary.index("policy_v2"))


@pytest.fixture(scope="module")
def fitted():
    """Exactly one genuine two-row fit per platform; immutable fixture evidence."""
    x = np.zeros((2, 9))
    x[0, 0] = x[1, 1] = 1
    y = np.array([[1.0], [2.0]])
    learner = LinearFitLearner(LinearFitPolicy(9, 6))
    learner.train_batch(x, y)
    return learner.snapshot_state(), x, y


def actor_for(source):
    return PromotableActor(
        source,
        version="source",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"scope": "tiny fixture"},
    )


def request(index=0):
    return DecisionRequest("request", SPEC.vocabulary[index], SPEC.catalog)


def direct_case(fitted):
    state, x, y = fitted
    source = LinearFitLearner(state.policy)
    actor = actor_for(source)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="source",
        candidate_version="tiny-paid",
        clock=LogicalClock(),
        budget=budget,
        actor=actor,
    )
    runtime.consolidate(
        "fixture", lambda _: ConsolidatedState(state, TrainingDiagnostic("tiny", 0.0))
    )
    new = GuardBatch("new", (("inner-new", "0"),), 0, x[:1].copy(), y[:1].copy(), "inner_guard")
    old = GuardBatch("old", (("inner-old", "0"),), 0, x[1:].copy(), y[1:].copy(), "inner_guard")
    controller = ServingPromotionController(
        actor,
        policy=replace(policy(), resource_metric_id="complete_shared_linear_record_bytes"),
        evaluator=binding.categorical_guard(SPEC, lambda: 0.0),
        build_learner=binding.build_linear,
        state_digest=binding.snapshot_digest,
        max_pending=1,
    )
    return source, actor, budget, runtime, controller, new, old


def test_complete_shared_record_is_canonical_owned_and_conserved(fitted):
    state, _, _ = fitted
    before = deepcopy(state)
    raw = binding.snapshot_bytes(state)
    assert json.loads(raw) == json.loads(json.dumps(asdict(state)))
    assert binding.snapshot_record_bytes(state) == len(raw)
    assert binding.snapshot_digest(state) == sha256(raw).hexdigest()
    child = binding.build_linear(state)
    assert child.snapshot_state().model is None
    child.restore_state(state)
    assert child.snapshot_state() == state == before


@pytest.mark.parametrize("fitted_state", [False, True])
def test_serves_exact_frozen_slot_and_pays_fitted_or_unfitted_read(fitted, fitted_state):
    source = LinearFitLearner(LinearFitPolicy(9, 6))
    if fitted_state:
        source.restore_state(fitted[0])
    actor = actor_for(source)
    pin = actor.serving_snapshot()
    reader = binding.LinearFitServingPredictor(actor, SPEC)
    served = reader.serve(request())
    assert served.result == LinearFitDecisionPredictor(SPEC, pin.model_state.model)(request())
    assert (served.generation, served.actor_version, served.completed_updates) == (
        0,
        "source",
        int(fitted_state),
    )
    assert reader.work == dict(attempts=1, returns=1, readouts=1, typed_attempts=1)
    assert reader.native_work.decision_returns == 1
    assert reader.native_work.affine_returns == int(fitted_state)
    assert reader.native_work.unfitted_returns == int(not fitted_state)
    assert actor.serving_snapshot() == pin


def test_publishes_actual_candidate_then_rolls_back_exact_original(fitted):
    source, actor, budget, runtime, controller, new, old = direct_case(fitted)
    reader = binding.LinearFitServingPredictor(actor, SPEC)
    baseline = reader.serve(request())
    ticket = controller.prepare(runtime, new, old, now=0, training_ids=frozenset())
    assert ticket.report.decision.accepted
    assert (
        ticket.report.evidence.candidate_new_utility
        == ticket.report.evidence.candidate_old_utility
        == 1.0
    )
    receipt = controller.commit(runtime, ticket)
    served = reader.serve(request())
    assert served.result.selected_id == SPEC.catalog[1].candidate_id
    assert (served.generation, served.actor_version, served.completed_updates) == (
        1,
        "tiny-paid",
        1,
    )
    assert actor.serving_snapshot().model_state == fitted[0]
    assert controller.rollback(receipt) == 2
    rolled = reader.serve(request())
    assert rolled.result == baseline.result and rolled.generation == 2
    assert source.snapshot_state().model is None and budget.updates_completed == 0
    assert reader.native_work.affine_returns == 1 and reader.native_work.unfitted_returns == 2


def test_lost_read_retains_native_work_and_permanently_latches(fitted, monkeypatch):
    source = LinearFitLearner(fitted[0].policy)
    source.restore_state(fitted[0])
    reader = binding.LinearFitServingPredictor(actor_for(source), SPEC)
    original = LinearFitDecisionPredictor.__call__

    def lost(self, query):
        original(self, query)
        raise RuntimeError("lost typed return")

    monkeypatch.setattr(LinearFitDecisionPredictor, "__call__", lost)
    with pytest.raises(RuntimeError, match="lost"):
        reader.serve(request())
    assert reader.native_work.affine_returns == reader.native_work.decision_returns == 1
    assert reader.work["returns"] == 0
    with pytest.raises(ValueError, match="stopped"):
        reader.serve(request())
    assert reader.work["attempts"] == 1


@pytest.mark.parametrize("fault", ["context", "dimension", "classes", "model", "counts", "spec"])
def test_refuses_foreign_or_malformed_slot_before_typed_work(fitted, fault):
    state = deepcopy(fitted[0])
    source = LinearFitLearner(state.policy)
    actor = actor_for(source)
    reader = binding.LinearFitServingPredictor(actor, SPEC)
    if fault == "context":
        bad = ContextLinearFitLearner(
            ContextLinearFitPolicy(9, 6, CONTEXT_COLUMNS)
        ).snapshot_state()
    elif fault == "dimension":
        bad = LinearFitLearner(LinearFitPolicy(8, 6)).snapshot_state()
    elif fault == "classes":
        bad = LinearFitLearner(LinearFitPolicy(9, 5)).snapshot_state()
    else:
        bad = state
        if fault == "model":
            object.__setattr__(bad.model, "fit_rows", 0)
        elif fault == "counts":
            object.__setattr__(bad, "completed_updates", True)
        else:
            object.__setattr__(reader._spec, "vocabulary", tuple(reversed(SPEC.vocabulary)))
    # Deliberate invalid-serving-slot injection, not a native state admission.
    monkey = actor.serving_snapshot()
    setattr(actor, "serving_snapshot", lambda: replace(monkey, model_state=bad))
    with pytest.raises((ValueError, TypeError)):
        reader.serve(request())
    assert reader.work["typed_attempts"] == reader.native_work.affine_attempts == 0
    with pytest.raises(ValueError, match="stopped"):
        reader.serve(request())


@pytest.mark.parametrize("role", ["train", "outer_selection", "final_test"])
def test_guard_refuses_non_inner_role_before_model_prediction(fitted, role):
    _, actor, _, runtime, controller, new, old = direct_case(fitted)
    pin = actor.serving_snapshot()
    with pytest.raises(ValueError, match="inner_guard"):
        controller.prepare(runtime, replace(new, role=role), old, now=0, training_ids=frozenset())
    assert actor.serving_snapshot() == pin


def test_guard_refuses_training_identity_overlap(fitted):
    _, _, _, runtime, controller, new, old = direct_case(fitted)
    with pytest.raises(ValueError, match="disjoint"):
        controller.prepare(runtime, new, old, now=0, training_ids=frozenset(new.sample_keys))


def test_rejected_guard_preserves_actor_and_paid_candidate(fitted):
    source, actor, budget, runtime, controller, new, old = direct_case(fitted)
    pin = actor.serving_snapshot()
    candidate = runtime.candidate_snapshot()
    rejected = replace(new, targets=np.array([[0.0]]))
    # Existing prepare rejects immediately; no publication ticket is admitted.
    with pytest.raises(PromotionRejected) as caught:
        controller.prepare(runtime, rejected, old, now=0, training_ids=frozenset())
    assert not caught.value.report.decision.accepted
    assert actor.serving_snapshot() == pin and runtime.candidate_snapshot() == candidate
    assert budget.updates_completed == 0 and source.snapshot_state().model is None


def test_changed_candidate_invalidates_prepared_publication(fitted):
    _, actor, _, runtime, controller, new, old = direct_case(fitted)
    pin = actor.serving_snapshot()
    ticket = controller.prepare(runtime, new, old, now=0, training_ids=frozenset())
    runtime.consolidate(
        "changed", lambda state: ConsolidatedState(state, TrainingDiagnostic("extra", 0.0))
    )
    with pytest.raises((ValueError, PromotionRejected)):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == pin


def test_request_uses_owned_original_spec_when_caller_changes_its_copy(fitted):
    spec = deepcopy(SPEC)
    source = LinearFitLearner(fitted[0].policy)
    source.restore_state(fitted[0])
    reader = binding.LinearFitServingPredictor(actor_for(source), spec)
    object.__setattr__(spec, "vocabulary", tuple(reversed(spec.vocabulary)))
    assert reader.serve(request()).result.selected_id == SPEC.catalog[1].candidate_id


def test_shared_guard_rejects_contextual_state_without_relaxing_legacy(fitted):
    from src.adapters import context_linear_serving as contextual

    shared = LinearFitLearner(fitted[0].policy).snapshot_state()
    routed = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, CONTEXT_COLUMNS)).snapshot_state()
    assert binding.categorical_guard(SPEC, lambda: 0.0)._state_valid(shared)
    assert not binding.categorical_guard(SPEC, lambda: 0.0)._state_valid(routed)
    assert contextual.categorical_guard(SPEC, lambda: 0.0)._state_valid(routed)
    assert not contextual.categorical_guard(SPEC, lambda: 0.0)._state_valid(shared)


def test_guard_owns_original_schema_after_caller_changes(fitted):
    spec = deepcopy(SPEC)
    guard = binding.categorical_guard(spec, lambda: 0.0)
    object.__setattr__(spec, "vocabulary", tuple(reversed(spec.vocabulary)))
    assert guard._state_valid(fitted[0])
