"""Real leases/two-window schedule with scalar learners and literal predictions."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import run_bp_ticket_serving as schedule
from scripts.candidate_training_feedback import read_candidate_feedback
from scripts.ticket_serving_timeline import FollowupTrainingPlan, LiveServingPlan
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.serving_ports import ServingConfiguration


@dataclass(frozen=True)
class ScalarState:
    value: int = 0
    completed_updates: int = 12


def literal_prediction(state, features):
    row = [0.0] * 6
    row[state.value] = 1.0
    return np.tile([row], (len(features), 1))


class ScalarLearner:
    def __init__(self, state=ScalarState()):
        self.state = state

    def fork(self):
        return ScalarLearner(self.state)

    def snapshot_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state

    def predict(self, features):
        return literal_prediction(self.state, features)

    def train_batch(self, features, targets):
        self.state = ScalarState(1, self.state.completed_updates + 1)
        return TrainingDiagnostic("scripted_train", 1.0)


def digest(state):
    return sha256(repr(state).encode()).hexdigest()


def actor_for(source, version="old"):
    return PromotableActor(
        source,
        version=version,
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"scope": "scalar fixture"},
    )


def publish_fixture(actor, state):
    # A scripted guard outcome; the real slot swap/receipt is exercised here.
    # Native integration separately qualifies the ordinary task guard.
    prepared = actor_for(ScalarLearner(state), "c0")
    before = actor.serving_snapshot()
    return actor._commit(
        prepared._slot.bundle,
        generation=before.generation,
        actor_digest=digest(before.model_state),
        state_digest=digest,
    )


def batch(index):
    return AttributionBatch(
        "train", "known", (f"row-{index}",), np.zeros((1, 7)), np.ones((1, 1)), num_classes=6
    )


def scalar_schedule(*, contention=False, mutate=None, corrupt_guard_history=False):
    source = ScalarLearner()
    actor = actor_for(source)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    gate = schedule.ServingPriorityGate(
        schedule.SharingLimits(1, 3, 1), resource_available=lambda: True
    )
    report: dict[str, Any] = {}
    held = None
    held_once = False

    @contextmanager
    def opportunity(tick):
        nonlocal held, held_once
        if held is not None:
            held.__exit__(None, None, None)
            held = None
        if mutate is not None and tick == 12:
            mutate(actor, report)
        if contention and budget.updates_completed == 1 and not held_once:
            # Model an asynchronous request whose lease outlives this poll.
            held = gate.serving()
            held.__enter__()
            held_once = True
            report["fixture_contention"] = [tick]
        yield

    timeline = SimpleNamespace(plan=LiveServingPlan(18, ()), gate=gate, opportunity=opportunity)

    def publish(owner, tick):
        receipt = publish_fixture(actor, owner.candidate_snapshot().state)
        if corrupt_guard_history:
            history = owner._inbox._acknowledged
            key = next(iter(history))
            history[key] = replace(history[key], wake_updates=99)
        return receipt

    runtime, _ = schedule._reference_schedule(
        source,
        actor,
        tuple(batch(i) for i in range(3)),
        budget,
        "drift",
        report,
        budget.before_final,
        lambda: None,
        sharing_gate=gate,
        live=timeline,
        on_drain=publish,
        followup_plan=FollowupTrainingPlan(2, 12),
        rejected_guard_digest=digest,
        candidate_predict=literal_prediction,
        acknowledge_correct=True,
    )
    return runtime, actor, budget, report


@pytest.mark.parametrize("contention", [False, True])
def test_should_fit_once_ack_both_windows_and_preserve_original_candidate_origin(contention):
    runtime, actor, budget, report = scalar_schedule(contention=contention)
    assert budget.updates_completed == 1 and len(runtime.applied_updates) == 1
    assert runtime._base_actor_version == "old" and actor.version == "c0"
    assert runtime.candidate_snapshot().state == ScalarState(1, 13)
    assert report["receipt_progress"] == dict(
        actual_updates=1, acknowledged=2, consumed=3, total=3, complete=True
    )
    assert report["followup_training"]["status"] == "complete"
    acknowledgements = runtime.acknowledged_experiences
    assert [(r.sample_id, r.observed_at, r.arrived_at) for r in acknowledgements] == [
        ("batch-1", 1, 2),
        ("batch-2", 12, 13),
    ]
    assert acknowledgements[-1].actor_version == "c0"
    assert all(r.wake_updates == 1 for r in acknowledgements)
    if contention:
        assert report["ack_publication_deferrals"] == report["fixture_contention"]
        assert report["drain_publication"]["tick"] == report["fixture_contention"][0] + 1
    assert report["candidate_feedback"]["records"][-1]["label_observed_at"] == 12


def test_should_refuse_guard_that_changes_paid_acknowledgement_history():
    with pytest.raises(ValueError, match="budget, receipts or sharing"):
        scalar_schedule(corrupt_guard_history=True)


@pytest.mark.parametrize("field", ["origin", "weights"])
def test_should_refuse_unobserved_candidate_change_across_return_gap(field, monkeypatch):
    owners = []
    original = schedule.ActorShadowRuntime

    def create(*args, **kwargs):
        owner = original(*args, **kwargs)
        owners.append(owner)
        return owner

    monkeypatch.setattr(schedule, "ActorShadowRuntime", create)

    def mutate(actor, report):
        if field == "origin":
            owners[0]._base_actor_version = "unobserved"
        else:
            owners[0]._candidate.restore_state(ScalarState(0, 13))

    with pytest.raises(ValueError, match="followup candidate"):
        scalar_schedule(mutate=mutate)
    assert len(owners[0].acknowledged_experiences) == 1
    assert owners[0]._budget.updates_completed == 1


@pytest.mark.parametrize("field", ["model", "metadata", "configuration", "receipt"])
def test_should_refuse_changed_verified_boundary_before_return_diagnostic(field):
    def mutate(actor, report):
        if field == "model":
            actor._slot.bundle.learner.restore_state(ScalarState(0, 99))
        elif field == "metadata":
            actor._slot.bundle.metadata["changed"] = True
        elif field == "configuration":
            object.__setattr__(actor._slot.bundle, "configuration", ServingConfiguration(4, 2))
        else:
            report["drain_publication"] = deepcopy(report["drain_publication"])

    with pytest.raises(ValueError, match="changed|replaced"):
        scalar_schedule(mutate=mutate)


def ready_return(*, published=False):
    source = ScalarLearner(ScalarState(1, 13))
    actor = actor_for(source)
    clock = LogicalClock(13)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="old",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
        actor=actor,
    )
    if published:
        publish_fixture(actor, source.state)
    b = batch(0)
    runtime.record_experience(
        Experience(
            "batch-0",
            "attribution",
            12,
            actor.version,
            b.features,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(
        LabelArrival(
            "label-0",
            "batch-0",
            "attribution",
            13,
            actor.version,
            b.targets,
        )
    )
    pin = actor.serving_snapshot()

    def require_actor():
        assert _same_value(
            schedule._followup_actor_values(actor.serving_snapshot()),
            schedule._followup_actor_values(pin),
        ), "serving actor changed"
        return pin.generation, pin.actor_version

    paid = dict(attempts=0, returns=0, rows_attempted=0, rows_returned=0, seconds=0.0, records=[])
    params = dict(
        arrived=1,
        completed=0,
        consumed=0,
        tick=13,
        label_delay=1,
        actor_version=actor.version,
        predict=literal_prediction,
        paid=paid,
        require_input=budget.before_final,
        acknowledge_correct=True,
        label_clock=(12, 13),
    )
    return runtime, actor, budget, b, require_actor, paid, params


@pytest.mark.parametrize("clock", [None, (0, 1), (12, 14), (True, 13), (13, 12)])
def test_should_refuse_early_or_replaced_clock_before_diagnostic(clock):
    runtime, actor, budget, b, guard, paid, params = ready_return()
    params["label_clock"] = clock
    with pytest.raises(ValueError):
        read_candidate_feedback(runtime, (b,), **params)
    assert not runtime.acknowledged_experiences and paid["attempts"] == 0


def test_should_require_explicit_verified_publication_without_rebasing():
    runtime, actor, budget, b, guard, paid, params = ready_return(published=True)
    with pytest.raises(ValueError, match="current unmanaged"):
        read_candidate_feedback(runtime, (b,), **params)
    result = read_candidate_feedback(runtime, (b,), require_actor=guard, **params)
    assert result.mistakes == 0 and budget.updates_completed == 0
    assert len(runtime.acknowledged_experiences) == 1 and runtime._base_actor_version == "old"
    assert paid["records"][0]["base_actor_version"] == "old"


@pytest.mark.parametrize("fault", ["actor", "candidate", "origin", "lost"])
def test_should_preserve_paid_failed_proof_without_consumption(fault):
    runtime, actor, budget, b, guard, paid, params = ready_return(published=True)

    def prediction(state, features):
        value = literal_prediction(state, features)
        if fault == "lost":
            raise RuntimeError("lost verification")
        if fault == "actor":
            actor._slot.bundle.metadata["changed"] = True
        elif fault == "origin":
            runtime._base_actor_version = "unobserved"
        else:
            runtime._candidate.restore_state(ScalarState(0, 99))
        return value

    params["predict"] = prediction
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        read_candidate_feedback(runtime, (b,), require_actor=guard, **params)
    assert paid["attempts"] == paid["rows_attempted"] == 1
    assert not runtime.acknowledged_experiences and budget.updates_completed == 0


def test_should_dispatch_explicit_same_context_ack_cli(tmp_path, monkeypatch):
    calls = []

    def example(**kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", example)
    assert (
        entry.main(
            [
                "--output",
                str(tmp_path / "out.json"),
                "--method",
                "linear_fit",
                "--context-return",
                "--separate-context-heads",
                "--paired-serving",
                "same_context_joint",
                "--adaptive-serving",
            ]
        )
        == 0
    )
    assert calls[0]["paired_serving"] == "same_context_joint" and calls[0]["adaptive_serving"]
