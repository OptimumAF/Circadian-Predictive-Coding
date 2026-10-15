"""Retain measured policy rejection evidence without changing serving state.

These controls reuse the established scalar promotion fixture and real evaluator.
They cover report ownership and the existing transaction boundary, not scientific
threshold selection, native-model quality or deployment authorization.
"""

from copy import deepcopy
from dataclasses import replace
import pickle

import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.serving_promotion import (
    PromotableActor,
    PromotionRejected,
    ServingPromotionController,
)
from src.core.promotion_guard import (
    PromotionDecision,
    PromotionEvidence,
    PromotionGuardReport,
    PromotionPolicy,
    decide_promotion,
)
from test_promotion_guard_evaluation import guards, make_evaluator, policy
from test_serving_promotion import digest, prepare, setup

ScalarActor = PromotableActor[list[int], list[float], list[float], list[float]]
ScalarRuntime = ActorShadowRuntime[list[int], list[float], list[float], list[float]]
ScalarController = ServingPromotionController[list[int], list[float], list[float], list[float]]
RejectionCase = tuple[
    ScalarActor,
    ScalarRuntime,
    ScalarController,
    PromotionRejected,
    PromotionGuardReport,
    dict[str, object],
]
REASONS = ("new_task_utility", "new_task_gain", "resource", "action_safety")


def measured_report(declared: PromotionPolicy) -> PromotionGuardReport:
    """Manually derive the complete report from the established fixed scalar data."""
    return PromotionGuardReport(
        policy=declared,
        actor_version="actor-0",
        learner_version="candidate-0",
        candidate_revision=(0, 1, 1),
        actor_snapshot_digest=digest([0.5, 0.8]),
        candidate_snapshot_digest=digest([0.75, 0.8]),
        new_guard_task_id="new",
        old_guard_task_id="old",
        new_guard_keys=(("e-new", "s1"),),
        old_guard_keys=(("e-old", "s1"),),
        label_arrivals=(2, 1),
        training_ids=(),
        assessed_at=2,
        evidence=PromotionEvidence(0.5, 0.75, 0.8, 0.8, 0.0, 16, True, False),
        decision=PromotionDecision(False, REASONS),
    )


def observed_state(
    actor: ScalarActor, runtime: ScalarRuntime, controller: ScalarController
) -> dict[str, object]:
    # Why: the budget has no public snapshot; inspect the existing session without
    # replacing its clock or omitting counters from the no-work assertion.
    return {
        "serving": actor.serving_snapshot(),
        "candidate": runtime.candidate_snapshot(),
        "budget": deepcopy(vars(runtime._budget)),
        "pending": dict(controller._pending),
        "latest": controller._latest,
        "applied_updates": runtime.applied_updates,
    }


@pytest.fixture
def rejected_case() -> RejectionCase:
    # A deliberately strict transaction fixture policy, with no scientific claim.
    declared = replace(
        policy(),
        min_new_utility=0.9,
        min_new_gain=0.4,
        max_resource_bytes=0,
        allowed_actions=("other",),
    )
    actor, runtime, controller = setup(
        declared=declared, evaluator=make_evaluator(clock=lambda: 0.0)
    )
    actor.serve([0], now=1)
    actor.serve([1], now=1)
    before = observed_state(actor, runtime, controller)
    with pytest.raises(ValueError) as caught:
        prepare(controller, runtime, metadata={"route": ["rejected"]})
    error = caught.value
    assert isinstance(error, PromotionRejected)
    return actor, runtime, controller, error, measured_report(declared), before


def test_rejection_preserves_complete_measured_report_and_runtime(
    rejected_case: RejectionCase,
) -> None:
    actor, runtime, controller, error, expected, before = rejected_case
    assert type(error) is PromotionRejected and isinstance(error, ValueError)
    assert str(error) == "candidate rejected: " + ", ".join(REASONS)
    assert error.args == (str(error),)
    assert error.report == expected
    assert error.report.decision == decide_promotion(expected.policy, expected.evidence)
    assert not error.report.decision.accepted
    assert observed_state(actor, runtime, controller) == before
    assert controller._pending == {} and controller._latest is None
    snapshot = actor.serving_snapshot()
    assert snapshot.generation == 0 and snapshot.actor_version == "actor-0"
    assert snapshot.metadata == {"route": ["old"]}
    assert len(snapshot.cache) == 2


def test_each_report_read_detaches_nested_decision_policy_and_evidence(
    rejected_case: RejectionCase,
) -> None:
    actor, runtime, controller, error, expected, before = rejected_case
    returned = error.report
    another = error.report
    assert returned is not another
    assert returned.policy is not another.policy
    assert returned.evidence is not another.evidence
    assert returned.decision is not another.decision
    # Frozen dataclasses normally refuse edits. Force edits here to prove deep
    # ownership rather than relying only on that shallow immutability boundary.
    object.__setattr__(returned, "actor_version", "forged")
    object.__setattr__(returned.policy, "allowed_actions", ("forged",))
    object.__setattr__(returned.evidence, "candidate_new_utility", 999.0)
    object.__setattr__(returned.decision, "accepted", True)
    object.__setattr__(returned.decision, "reasons", ())
    assert error.report == expected and another == expected
    assert str(error) == "candidate rejected: " + ", ".join(REASONS)
    assert observed_state(actor, runtime, controller) == before


def test_exception_constructor_detaches_its_report_input(rejected_case: RejectionCase) -> None:
    actor, runtime, controller, rejected, expected, before = rejected_case
    supplied = rejected.report
    captured = PromotionRejected(supplied)
    object.__setattr__(supplied, "assessed_at", 99)
    object.__setattr__(supplied.policy, "min_new_gain", -999.0)
    object.__setattr__(supplied.evidence, "actions_safe", True)
    object.__setattr__(supplied.decision, "reasons", ("forged",))
    assert captured.report == expected and rejected.report == expected
    assert str(captured) == "candidate rejected: " + ", ".join(REASONS)
    assert observed_state(actor, runtime, controller) == before


def test_rejection_round_trips_as_a_local_exception(rejected_case: RejectionCase) -> None:
    actor, runtime, controller, error, expected, before = rejected_case
    # Trusted local exception interoperability only; no external pickle is read.
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is PromotionRejected and isinstance(restored, ValueError)
    assert str(restored) == str(error) and restored.args == error.args
    assert restored.report == expected
    returned = restored.report
    object.__setattr__(returned.evidence, "candidate_resource_bytes", 999)
    assert restored.report == expected and error.report == expected
    assert observed_state(actor, runtime, controller) == before


def test_nonmeasurement_validation_remains_an_ordinary_value_error() -> None:
    actor, runtime, controller = setup()
    new, old = guards()
    before = observed_state(actor, runtime, controller)
    with pytest.raises(ValueError, match="inner_guard") as caught:
        controller.prepare(
            runtime,
            replace(new, role="final_test"),
            old,
            now=2,
            training_ids=frozenset(),
        )
    assert not isinstance(caught.value, PromotionRejected)
    assert observed_state(actor, runtime, controller) == before


def test_successful_prepare_commit_keeps_the_existing_transaction_path() -> None:
    actor, runtime, controller = setup()
    actor.serve([0], now=1)
    candidate = runtime.candidate_snapshot()
    budget = deepcopy(vars(runtime._budget))
    ticket = prepare(controller, runtime, metadata={"route": ["new"]})
    assert ticket.report.decision.accepted
    assert ticket.report.decision == decide_promotion(ticket.report.policy, ticket.report.evidence)
    assert actor.serving_snapshot().generation == 0
    receipt = controller.commit(runtime, ticket)
    snapshot = actor.serving_snapshot()
    assert receipt.generation == snapshot.generation == 1
    assert snapshot.actor_version == "candidate-0"
    assert snapshot.model_state == [0.75, 0.8]
    assert snapshot.metadata == {"route": ["new"]} and snapshot.cache == {}
    assert actor.serve([0], now=2).prediction == [0.75]
    assert runtime.candidate_snapshot() == candidate
    assert vars(runtime._budget) == budget
    assert controller._pending == {}
