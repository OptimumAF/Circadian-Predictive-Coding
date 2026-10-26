"""Native typed publication/rollback; reuse paid fixed snapshots, no refits."""

from copy import deepcopy

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from src.adapters import torch_serving as binding
from src.adapters.torch_bp_record import bp_snapshot_from_record
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionStream
from src.app.serving_promotion import PromotableActor, PromotionRejected, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.serving_ports import ServingConfiguration

SPEC = DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)


@pytest.fixture(scope="module", autouse=True)
def reuse_fixed_teacher_templates():
    # Construction executes teacher sandbox transitions; pay once, then copy the
    # same fixed input templates instead of regenerating them for each guard.
    original = demo.context._teacher_batches
    templates = original(SPEC)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "_teacher_batches", lambda spec: deepcopy(templates))
        yield


@pytest.fixture(scope="module")
def states():
    r = json.loads(Path("examples/ticket_workflow/bp.json").read_bytes())
    return tuple(bp_snapshot_from_record(c["state"]) for c in r["checkpoints"][:2])


def setup(states, *, changed=True, policy=None):
    source = TorchFeatureLearner(states[0].policy)
    source.restore_state(states[0])
    actor = PromotableActor(
        source,
        version="base",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: __import__("hashlib").sha256(x.tobytes()).hexdigest(),
        metadata={"context": ["old"]},
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="base",
        candidate_version="learned",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
        actor=actor,
    )
    if changed:
        runtime.consolidate(
            "paid-snapshot-fixture",
            lambda s: ConsolidatedState(states[1], TrainingDiagnostic("fixture", 0.0)),
        )
    controller = ServingPromotionController(
        actor,
        policy=policy or demo.policy(),
        evaluator=binding.categorical_guard(SPEC, lambda: 0.0),
        build_learner=binding.build_native,
        state_digest=binding.snapshot_digest,
        max_pending=1,
    )
    return actor, runtime, controller


def prepare(runtime, controller, new=None, old=None):
    n, o = demo.guards(SPEC, 0)
    return controller.prepare(
        runtime, new or n, old or o, now=0, training_ids=frozenset(), metadata={"context": ["new"]}
    )


def request():
    cases, _, _ = demo.context._teacher_batches(SPEC)
    return cases[7].request  # legacy ready: acquisition-sensitive action


def test_same_typed_adapter_follows_guarded_commit_and_complete_rollback(states):
    actor, runtime, controller = setup(states)
    p = binding.TorchServingPredictor(actor, SPEC)
    n, _ = demo.guards(SPEC, 0)
    actor.serve(n.features, now=2)
    before = actor.serving_snapshot()
    first = p.serve(request())
    assert first.result.selected_id != "defer"
    ticket = prepare(runtime, controller)
    assert ticket.report.decision.accepted
    assert ticket.report.evidence.candidate_new_utility == 1.0
    assert (
        ticket.report.evidence.candidate_old_utility
        == ticket.report.evidence.actor_old_utility
        == 1.0
    )
    assert actor.serving_snapshot().generation == 0
    receipt = controller.commit(runtime, ticket)
    second = p.serve(request())
    assert (second.generation, second.actor_version, second.native_updates) == (1, "learned", 104)
    assert second.result.selected_id == "defer"
    assert not actor.serving_snapshot().cache
    assert controller.rollback(receipt) == 2
    last = p.serve(request())
    after = actor.serving_snapshot()
    assert (first.actor_version, last.actor_version) == ("base", "base")
    assert last.generation == 2 and last.native_updates == 96
    assert first.result == last.result
    assert binding.snapshot_digest(before.model_state) == binding.snapshot_digest(after.model_state)
    assert before.cache.keys() == after.cache.keys()
    assert before.metadata == after.metadata and before.configuration == after.configuration
    assert before.last_cache_tick == after.last_cache_tick
    for key in before.cache:
        np.testing.assert_array_equal(before.cache[key].prediction, after.cache[key].prediction)
    assert p.work == dict(attempts=3, returns=3, readouts=3, native_forwards=3)
    assert not any(isinstance(v, binding.TorchDecisionPredictor) for v in vars(p).values())
    with pytest.raises(ValueError):
        controller.commit(runtime, ticket)
    with pytest.raises(ValueError):
        controller.rollback(receipt)


def test_no_new_gain_candidate_is_rejected_without_serving_changes(states):
    actor, runtime, controller = setup(states, changed=False)
    before = actor.serving_snapshot()
    with pytest.raises(PromotionRejected) as caught:
        prepare(runtime, controller)
    assert "new_task_gain" in caught.value.report.decision.reasons
    assert binding.snapshot_digest(actor.snapshot_state().state) == binding.snapshot_digest(
        before.model_state
    )
    assert actor.version == "base" and actor.serving_snapshot().generation == 0


@pytest.mark.parametrize("fault", ["old-targets", "resource", "actions", "latency"])
def test_meaningful_guard_refusals_cannot_publish(states, fault):
    policy = demo.policy()
    if fault == "resource":
        policy = replace(policy, max_resource_bytes=1)
    if fault == "actions":
        policy = replace(policy, allowed_actions=("retire",))
    actor, runtime, controller = setup(states, policy=policy)
    new, old = demo.guards(SPEC, 0)
    if fault == "old-targets":
        # A configured old contract must be retained even when acquisition fits.
        old = replace(old, targets=np.zeros_like(old.targets))
        # Isolate the actual old-drop predicate with predictions reversed below.
        controller._policy = replace(policy, max_old_drop=0.0)
        source_new = controller._evaluator._utility
        calls = []

        def measured(p, t):
            calls.append(1)
            if len(calls) == 3:
                return 1.0
            if len(calls) == 4:
                return 0.0
            return source_new(p, t)

        controller._evaluator._utility = measured
    if fault == "latency":
        times = iter(range(8))
        controller._evaluator._clock = lambda: float(next(times))
    with pytest.raises(PromotionRejected) as caught:
        prepare(runtime, controller, new, old)
    reason = {
        "old-targets": "old_task_retention",
        "resource": "resource",
        "actions": "action_safety",
        "latency": "latency",
    }[fault]
    assert reason in caught.value.report.decision.reasons
    assert actor.version == "base" and actor.serving_snapshot().generation == 0


@pytest.mark.parametrize("role", ["train", "probe_train", "final_test", "outer_selection"])
def test_wrong_guard_role_rejected_before_native_build(states, role, monkeypatch):
    _, runtime, controller = setup(states)
    new, old = demo.guards(SPEC, 0)
    if role == "probe_train":
        object.__setattr__(new, "role", role)
    else:
        new = replace(new, role=role)
    monkeypatch.setattr(
        controller._evaluator, "_state_valid", lambda s: pytest.fail("payload accessed")
    )
    with pytest.raises(ValueError, match="inner_guard"):
        prepare(runtime, controller, new, old)


def test_overlapping_and_future_guard_labels_refused(states):
    _, runtime, controller = setup(states)
    new, old = demo.guards(SPEC, 0)
    with pytest.raises(ValueError, match="disjoint"):
        controller.prepare(runtime, new, old, now=0, training_ids=frozenset(new.sample_keys))
    with pytest.raises(ValueError, match="not arrived"):
        prepare(runtime, controller, replace(new, labels_arrived_at=1), old)


def test_changed_candidate_invalidates_ticket(states):
    actor, runtime, controller = setup(states)
    ticket = prepare(runtime, controller)
    runtime.consolidate(
        "changed", lambda s: ConsolidatedState(s, TrainingDiagnostic("fixture", 0.0))
    )
    with pytest.raises(ValueError, match="revision"):
        controller.commit(runtime, ticket)
    assert actor.version == "base"
    controller.discard(ticket)


def test_typed_prediction_fault_latches_and_keeps_paid_readout(states, monkeypatch):
    actor, _, _ = setup(states, changed=False)
    p = binding.TorchServingPredictor(actor, SPEC)
    monkeypatch.setattr(
        binding.TorchDecisionPredictor,
        "__call__",
        lambda *a: (_ for _ in ()).throw(RuntimeError("native fault")),
    )
    with pytest.raises(RuntimeError, match="native fault"):
        p(request())
    assert p.work == dict(attempts=1, returns=0, readouts=1, native_forwards=0)
    with pytest.raises(ValueError, match="stopped"):
        p(request())
    assert p.work["attempts"] == 1


def test_stream_injected_actor_trains_shadow_only_with_paid_delayed_receipt(states):
    actor, _, _ = setup(states, changed=False)
    source = TorchFeatureLearner(states[0].policy)
    source.restore_state(states[0])
    _, _, batches = demo.context._teacher_batches(SPEC)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    s = AttributionStream(
        source,
        (batches[0],),
        (),
        checkpoint_updates=(0, 1),
        label_delay=1,
        budget=budget,
        training_only=True,
        actor=actor,
    )
    s.run_until(1)
    assert s.candidate_snapshot().completed_updates == 97
    assert actor.snapshot_state().state.completed_updates == 96 and source.completed_updates == 96
    assert s.applied_updates[0].observed_at < s.applied_updates[0].applied_at
    assert s.applied_updates[0].actor_version == "base"
    assert s.finish()["score_forward_calls"] == 0 and budget.updates_completed == 1


@pytest.mark.parametrize("role,training_only", [("probe_train", True), ("train", False)])
def test_injected_actor_cannot_join_probe_or_final_stream_before_payload_access(
    role, training_only, states
):
    actor, _, _ = setup(states, changed=False)
    with pytest.raises(ValueError, match="TRAIN-only"):
        AttributionStream(
            cast(Any, None),
            (),
            (),
            checkpoint_updates=(0, 1),
            label_delay=1,
            budget=cast(Any, None),
            training_only=training_only,
            training_role=role,
            actor=actor,
        )


def test_cli_preserves_existing_output_and_records_failures(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text("keep")
    monkeypatch.setattr(demo, "run_case", lambda p: pytest.fail("started work"))
    with pytest.raises(FileExistsError):
        demo.main(["--source", "unused", "--output", str(path)])
    assert path.read_text() == "keep"
    path = tmp_path / "failed.json"
    monkeypatch.setattr(demo, "run_case", lambda p: {"complete": False, "error": "paid failure"})
    assert demo.main(["--source", "unused", "--output", str(path)]) == 1
    assert json.loads(path.read_bytes())["error"] == "paid failure"


def test_tiny_full_recipe_keeps_failed_utility_without_publishing_or_retry(monkeypatch):
    monkeypatch.setattr(demo.context, "BATCHES_PER_PHASE", 1)
    result = demo.run_case(Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"))
    assert result["complete"], result.get("error")
    assert result["training_work"] == dict(
        attempted_updates=1, attempted_rows=6, returned_updates=1, unreturned_work_unknown=False
    )
    assert result["budget"]["updates_completed"] == 1
    assert not result["guard_report"]["decision"]["accepted"]
    assert [p["phase"] for p in result["publications"]] == ["before", "rejected"]
    assert all(p["generation"] == 0 for p in result["publications"])
    assert (
        result["source_unchanged"]
        and result["numpy_rng_unchanged"]
        and result["torch_rng_unchanged"]
    )
