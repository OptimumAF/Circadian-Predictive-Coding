"""Scalar owner-retirement checks; no native model or experiment continuation."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_bp_ticket_serving as demo
from src.core.contextual_controller import choose_contextual
from src.core.reference_controller import ReferenceObservation
from test_withheld_context_orders import FIT
import json
from scripts.context_stream_comparison import restore_scalar_policy

from test_post_publication_learning import LiteralLearner, LiteralActor, LiteralLive, batches


def state():
    actor = SimpleNamespace(
        serving_snapshot=lambda: SimpleNamespace(generation=1, actor_version="c0")
    )
    report = dict(
        drain_publication=dict(
            accepted=True,
            completed=True,
            publication_unknown=False,
            generation_before=0,
            actor_version_after="c0",
        )
    )
    return actor, report, ReferenceObservation(11, 13, 0, 0, "c0", None)


def test_should_retire_chooser_without_weakening_its_stale_version_refusal():
    actor, report, observation = state()
    fitted = restore_scalar_policy(json.loads(FIT.read_bytes()))
    with pytest.raises(ValueError, match="stale learner version"):
        choose_contextual(fitted, observation)
    decision = demo._retired_contextual_wait(
        actor,
        report,
        (1, "c0"),
        observation,
        pending={},
        arrived=6,
        consumed=6,
        total=6,
    )
    assert decision.action_id == "wait"
    assert fitted.learner_version == "qualified-12"
    assert "retirement" in decision.reason


@pytest.mark.parametrize(
    "fault",
    [
        "generation",
        "version",
        "unknown",
        "rejected",
        "incomplete",
        "pending",
        "unarrived",
        "unconsumed",
        "ready",
        "feedback",
        "actor",
        "count_type",
    ],
)
def test_should_refuse_unverified_or_non_drained_retirement(fault):
    actor, report, observation = state()
    pin = (1, "c0")
    kwargs = dict(pending={}, arrived=6, consumed=6, total=6)
    row = report["drain_publication"]
    if fault == "generation":
        pin = (2, "c0")
    if fault == "version":
        pin = (1, "replacement")
    if fault == "unknown":
        row["publication_unknown"] = True
    if fault == "rejected":
        row["accepted"] = False
    if fault == "incomplete":
        row["completed"] = False
    if fault == "pending":
        kwargs["pending"] = {5: object()}
    if fault == "unarrived":
        kwargs["arrived"] = 5
    if fault == "unconsumed":
        kwargs["consumed"] = 5
    if fault == "ready":
        observation = replace(observation, ready_updates=1)
    if fault == "feedback":
        from src.core.reference_controller import ArrivedTrainingFeedback

        observation = replace(
            observation,
            feedback=ArrivedTrainingFeedback(
                "train",
                "batch-5",
                "c0",
                5,
                6,
                0,
                3,
            ),
        )
    if fault == "actor":
        observation = replace(observation, actor_version="foreign")
    if fault == "count_type":
        kwargs["consumed"] = 6.0
    with pytest.raises(ValueError):
        demo._retired_contextual_wait(actor, report, pin, observation, **kwargs)


@pytest.mark.parametrize("changed_publication", [False, True])
def test_should_finish_scripted_service_tail_or_refuse_changed_publication(
    monkeypatch, changed_publication
):
    """Exercise the actual schedule with one literal row, never native learning."""
    from scripts.ticket_serving_timeline import LiveServingPlan
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

    source = LiteralLearner()
    actor = LiteralActor(source)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    gate = demo.ServingPriorityGate(demo.SharingLimits(1, 1, 1), resource_available=lambda: True)
    live = LiteralLive(gate)
    live.plan = LiveServingPlan(24, ())
    batch = batches()[0]
    # Predicts class0; literal class1 target causes the original fitted wake arm.
    batch = replace(batch, targets=batch.targets + 1)
    fitted = replace(restore_scalar_policy(json.loads(FIT.read_bytes())), learner_version="old")
    report: dict[str, Any] = {}
    choices, owners = [], []
    choose, runtime_factory = demo.choose_contextual, demo.ActorShadowRuntime

    def select(policy, observation):
        choices.append(observation.tick)
        assert observation.actor_version == policy.learner_version == "old"
        return choose(policy, observation)

    def runtime(*args, **kwargs):
        owner = runtime_factory(*args, **kwargs)
        owners.append(owner)
        return owner

    def require():
        if changed_publication and report.get("drain_publication", {}).get("accepted"):
            actor.generation = 2

    monkeypatch.setattr(demo, "choose_contextual", select)
    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)

    def run():
        return demo._reference_schedule(
            source,
            actor,
            (batch,),
            budget,
            "drift",
            report,
            require,
            lambda: None,
            contextual_policy=fitted,
            sharing_gate=gate,
            live=live,
            on_drain=lambda owner, tick: actor.publish(owner),
        )

    if changed_publication:
        with pytest.raises(ValueError, match="publication"):
            run()
        assert live.ticks == [0, 1]
        assert "stream" not in report
    else:
        owner, horizon = run()
        assert owners == [owner] and owner._budget is budget and horizon == 24
        assert live.ticks == list(range(24))
        assert report["stream"]["pending_unapplied"] == 0
        assert report["action_snapshot"]["progress"]["decisions_attempted"] == 24
        tail = report["controller_events"][2:]
        assert len(tail) == 22
        assert all(e["action_id"] == "wait" and "retirement" in e["reason"] for e in tail)
        assert all(e["observation"]["feedback"] is None for e in tail)
    assert choices == [0, 1] and len(owners) == 1
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1
    assert source.snapshot_state() == 0 and actor.snapshot_state().state == 1
    assert fitted.learner_version == "old"
    receipts = owners[0].applied_updates
    assert len(receipts) == 1
    assert (
        receipts[0].actor_version,
        receipts[0].observed_at,
        receipts[0].arrived_at,
        receipts[0].applied_at,
        receipts[0].update_number,
    ) == ("old", 0, 1, 1, 1)
    assert report["drain_publication"]["tick"] == 1
    assert report["feedback_invalidations"][0]["feedback"]["actor_version"] == "old"
