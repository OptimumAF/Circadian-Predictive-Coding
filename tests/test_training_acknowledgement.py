"""Causal no-update consumption and existing-schedule integration, bounded fixtures."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from time import monotonic
from typing import Any

import pytest

from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as cli
from scripts.context_training_streams import CONTEXT_TRAINING_RECIPES, build_context_stream
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyExecutionBudget, ToyBudgetSession, ToyExecutionStopped
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from test_candidate_training_feedback import setup as context_setup, predict


class ScalarLearner:
    def __init__(self):
        self.value = 0

    def fork(self):
        child = ScalarLearner()
        child.value = self.value
        return child

    def snapshot_state(self):
        return self.value

    def train_batch(self, features, targets):
        self.value += 1
        return TrainingDiagnostic("fixture_v1", 0.0)

    def restore_state(self, state):
        self.value = state

    def predict(self, features):
        return [self.value]


def owner():
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    runtime = ActorShadowRuntime(
        ScalarLearner(),
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=2,
        max_consolidations=0,
    )
    for i in range(2):
        runtime.record_experience(
            Experience(
                f"s{i}",
                "e",
                i,
                "a0",
                [i],
                "train",
                ExperiencePermissions(training=True),
            )
        )
        runtime.record_label(LabelArrival(f"l{i}", f"s{i}", "e", i + 2, "a0", [i]))
    return runtime, clock, budget


def acknowledge(runtime, **kwargs):
    params = dict(
        actor_version="a0",
        learner_version="c0",
        wake_updates=0,
        verify=lambda state, source, label: True,
    )
    params.update(kwargs)
    return runtime.acknowledge_ready(("e", "s0"), **params)


def test_should_consume_once_without_update_and_train_next_ready_packet():
    runtime, clock, budget = owner()
    clock.advance_to(3)
    before = runtime.candidate_snapshot()
    receipt = acknowledge(runtime)
    assert (receipt.observed_at, receipt.arrived_at, receipt.acknowledged_at) == (0, 2, 3)
    assert receipt.wake_updates == 0 and receipt.candidate_revision > 0
    assert runtime.candidate_snapshot() == before and budget.updates_completed == 0
    assert runtime.applied_updates == () and runtime.acknowledged_experiences == (receipt,)
    with pytest.raises(ValueError, match="oldest ready"):
        acknowledge(runtime)
    applied = runtime.train_ready(max_updates=1)
    assert len(applied) == 1 and applied[0].sample_id == "s1"
    assert runtime.train_ready() == () and budget.updates_completed == 1
    assert runtime.candidate_snapshot().state == 1 and runtime.actor.snapshot_state().state == 0
    assert len(runtime._inbox._experiences) == 2  # retained, not erased or freed capacity


@pytest.mark.parametrize(
    "fault",
    ["early", "actor", "learner", "updates", "boolean", "fifo", "probe", "managed", "sleep"],
)
def test_should_refuse_stale_or_unsupported_owner_before_verifier(fault):
    runtime, clock, budget = owner()
    clock.advance_to(1 if fault == "early" else 3)
    kwargs: dict[str, Any] = dict(verify=lambda *args: pytest.fail("verifier entered"))
    if fault in ("actor", "learner"):
        kwargs[f"{fault}_version"] = "stale"
    elif fault == "updates":
        kwargs["wake_updates"] = 1
    elif fault == "boolean":
        kwargs["wake_updates"] = False
    elif fault == "fifo":
        runtime._inbox._labels[("e", "s0")] = replace(
            runtime._inbox._labels[("e", "s0")], arrived_at=5
        )
    elif fault == "probe":
        runtime._training_role = "probe_train"
    elif fault == "managed":
        runtime._actor._payload_registry._lifecycle = SimpleNamespace(_require_access=lambda: None)
    elif fault == "sleep":
        runtime._consolidation_limit = 1
    with pytest.raises(ValueError):
        acknowledge(runtime, **kwargs)
    assert budget.updates_completed == 0 and runtime.acknowledged_experiences == ()


@pytest.mark.parametrize("outcome", [False, None, "raise", "reenter", "clock"])
def test_should_not_consume_on_incorrect_invalid_failed_or_reentrant_verification(outcome):
    runtime, clock, budget = owner()
    clock.advance_to(3)

    def verify(state, source, label):
        if outcome == "raise":
            raise RuntimeError("failed proof")
        if outcome == "reenter":
            runtime.train_ready()
        if outcome == "clock":
            clock.advance_to(4)
            return True
        return outcome

    if outcome is False:
        assert acknowledge(runtime, verify=verify) is None
    else:
        with pytest.raises((ValueError, RuntimeError)):
            acknowledge(runtime, verify=verify)
    assert not runtime.acknowledged_experiences and budget.updates_completed == 0
    assert runtime.candidate_snapshot().state == 0
    assert len(runtime.train_ready()) == 2  # no fabricated receipt or retry block


def test_should_detach_payloads_and_refuse_unsupported_cursor_paths():
    runtime, clock, budget = owner()
    cursor = runtime._inbox.capture_cursor()
    clock.advance_to(3)

    def verify(state, source, label):
        source.features.append("private")
        label.targets.append("private")
        return True

    acknowledge(runtime, verify=verify)
    assert runtime._inbox._experiences[("e", "s0")].features == [0]
    assert runtime._inbox._labels[("e", "s0")].targets == [0]
    with pytest.raises(ValueError, match="acknowledgement history"):
        runtime._inbox.capture_cursor()
    with pytest.raises(ValueError, match="acknowledgement history"):
        runtime._inbox._materialize_cursor(ScalarLearner(), cursor)


def test_should_keep_default_owner_schema_unchanged_until_actual_acknowledgement():
    runtime, clock, _ = owner()
    assert "_acknowledged" not in vars(runtime._inbox)
    clock.advance_to(3)
    assert acknowledge(runtime, verify=lambda *args: False) is None
    assert "_acknowledged" not in vars(runtime._inbox)
    assert runtime._inbox.capture_cursor().completed_updates == 0


def test_should_refuse_serving_version_change_during_proof_before_consumption():
    runtime, clock, budget = owner()
    clock.advance_to(3)

    def verify(state, source, label):
        runtime.actor._version = "a1"  # simulate a concurrent serving swap
        return True

    with pytest.raises(ValueError, match="serving source changed"):
        acknowledge(runtime, verify=verify)
    assert runtime.acknowledged_experiences == () and budget.updates_completed == 0
    assert runtime.applied_updates == () and runtime.candidate_snapshot().state == 0


def test_should_refuse_expired_budget_before_verification():
    runtime, clock, budget = owner()
    clock.advance_to(3)
    budget.budget = ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1)
    budget.clock = lambda: 2.0
    with pytest.raises(ToyExecutionStopped):
        acknowledge(runtime, verify=lambda *a: pytest.fail("expired verifier entered"))
    assert not runtime.acknowledged_experiences and budget.updates_completed == 0


def test_should_settle_recurring_receipts_and_call_existing_publication_with_only_two_updates(
    monkeypatch,
):
    state, source, actor, original, _ = context_setup()
    batches, _ = build_context_stream(CONTEXT_TRAINING_RECIPES[0], *original)
    from scripts.ticket_serving_timeline import LiveServingPlan
    from src.app.resource_sharing import ServingPriorityGate
    from src.core.resource_sharing import SharingLimits

    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=6, max_wall_seconds=30), monotonic
    )
    gate = ServingPriorityGate(SharingLimits(1, 6, 1), resource_available=lambda: True)
    # Only schedule behavior here; real workflow serving is checked separately.
    timeline = SimpleNamespace(
        plan=LiveServingPlan(24, ()), gate=gate, opportunity=lambda tick: nullcontext()
    )
    reports: dict[str, Any] = {}
    published = []
    # This test asserts drain admission, not actor swap; actual guarded serving
    # is validated separately with the original PromotableActor and guards.
    monkeypatch.setattr(
        cli,
        "_publish_at_drain",
        lambda actor, runtime, callback, tick, feedback, report, completed: callback(runtime, tick),
    )
    runtime, _ = cli._reference_schedule(
        source,
        actor,
        batches,
        budget,
        "drift",
        reports,
        budget.before_final,
        lambda: None,
        label_delay=2,
        sharing_gate=gate,
        live=timeline,
        on_drain=lambda owner, tick: published.append(
            (tick, len(owner.applied_updates), len(owner.acknowledged_experiences))
        ),
        candidate_predict=predict,
        candidate_row_limit=144,
        acknowledge_correct=True,
    )
    assert published == [(7, 2, 4)]  # no reserved serving deferral in this fixture
    assert reports["receipt_progress"] == dict(
        actual_updates=2, acknowledged=4, consumed=6, total=6, complete=True
    )
    assert reports["stream"]["pending_unapplied"] == 0
    assert [r.sample_id for r in runtime.acknowledged_experiences] == [
        f"batch-{i}" for i in range(2, 6)
    ]
    assert reports["candidate_feedback"]["rows_returned"] == 27
    assert (
        budget.updates_completed == 2
        and source.snapshot_state() == actor.snapshot_state().state == state
    )


@pytest.mark.parametrize("fault", ["raise", "invalid", "mutate"])
def test_should_retain_failed_proof_without_acknowledgement_or_training(fault):
    import numpy as np

    state, source, actor, batches, budget = context_setup()
    report: dict[str, Any] = {}

    def bad(snapshot, features):
        if fault == "raise":
            raise RuntimeError("proof lost")
        if fault == "invalid":
            return np.zeros((len(features), 6))
        value = predict(snapshot, features)
        object.__setattr__(snapshot, "completed_updates", 999)
        return value

    with pytest.raises((ValueError, RuntimeError)):
        cli._reference_schedule(
            source,
            actor,
            batches,
            budget,
            "drift",
            report,
            budget.before_final,
            lambda: None,
            candidate_predict=bad,
            acknowledge_correct=True,
        )
    assert report["receipt_progress"]["consumed"] == 0
    assert report["candidate_feedback"]["attempts"] == 1
    assert report["candidate_feedback"]["rows_returned"] == 0
    assert not report["candidate_feedback"]["records"][0]["complete"]
    assert (
        budget.updates_completed == 0
        and actor.snapshot_state().state == source.snapshot_state() == state
    )


@pytest.mark.parametrize("flag", [None, 0, 1, "yes", True])
def test_should_refuse_invalid_or_implicit_optin_before_source(flag, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source entered"))
    with pytest.raises(ValueError):
        live.run_case("unused", acknowledge_correct=flag)


def test_should_require_candidate_flag_and_dispatch_cli_optin(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--source",
                "unused",
                "--context-head-serving-live",
                "--context-acknowledge-correct",
                "--output",
                str(output),
            ]
        )
    seen = []

    def run_case(path, **kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", run_case)
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--context-head-serving-live",
                "--context-live-reference",
                "drift",
                "--context-candidate-feedback",
                "--context-acknowledge-correct",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert seen == [dict(reference_mode="drift", candidate_feedback=True, acknowledge_correct=True)]
