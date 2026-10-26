"""Context bindings through existing controller: paid snapshots, no fitting."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import context_head_serving as demo
from scripts import run_bp_ticket_serving as cli
from scripts.context_head_probe import head_states
from src.adapters import context_serving as binding
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.serving_promotion import PromotableActor, PromotionRejected, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import (
    ACTION_IDS,
    CATALOG,
    WorkflowState,
    WorkflowCapabilities,
    workflow_request,
)

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")
SPEC = DecisionFeatureSpec(demo.CONTEXT_FEATURES, CATALOG)


@pytest.fixture(scope="module")
def states():
    warm, zero = head_states(json.loads(SOURCE.read_bytes()))
    return zero, warm


@pytest.fixture(scope="module")
def data():
    return demo.prepare_data(SPEC)


def setup(states, *, changed=True, policy=None):
    source = ContextPerceptronLearner(states[0].policy)
    source.restore_state(states[0])
    actor = PromotableActor(
        source,
        version="base",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"values": ["old"]},
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="base",
        candidate_version="fixture",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0),
        actor=actor,
    )
    if changed:
        runtime.consolidate(
            "paid-fixture",
            lambda s: ConsolidatedState(states[1], TrainingDiagnostic("fixture", 0.0)),
        )
    controller = ServingPromotionController(
        actor,
        policy=policy or demo.policy(),
        evaluator=binding.categorical_guard(SPEC, lambda: 0.0),
        build_learner=binding.build_context,
        state_digest=binding.snapshot_digest,
        max_pending=1,
    )
    return actor, runtime, controller


def guards(data):
    _, (_, old) = deepcopy(data)
    # Both are fixed qualified old templates. A snapshot fixture tests the
    # controller contract, not the actual study's new/old learning result.
    return (
        replace(
            old,
            task_id="fixture-new",
            sample_keys=old.sample_keys[:6],
            features=old.features[:6],
            targets=old.targets[:6],
        ),
        replace(
            old, sample_keys=old.sample_keys[6:], features=old.features[6:], targets=old.targets[6:]
        ),
    )


def prepare(runtime, controller, data):
    n, o = guards(data)
    return controller.prepare(
        runtime, n, o, now=0, training_ids=frozenset(), metadata={"values": ["new"]}
    )


def request():
    return workflow_request(
        WorkflowState("serve", "invoice", policy_version="v1"),
        WorkflowCapabilities(ACTION_IDS),
        "request",
        include_policy_context=True,
    )


def test_should_measure_complete_nonzero_record_and_counter_identity(states):
    state = states[1]
    encoded = binding.snapshot_bytes(state)
    assert json.loads(encoded) == json.loads(json.dumps(asdict(state)))
    assert binding.snapshot_record_bytes(state) == len(encoded) > 768
    assert binding.snapshot_digest(state) == sha256(encoded).hexdigest()
    advanced = replace(
        state,
        heads=(replace(state.heads[0], completed_updates=9, training_examples=49), state.heads[1]),
        completed_updates=13,
        training_examples=73,
    )
    assert binding.snapshot_digest(advanced) != binding.snapshot_digest(state)
    bad = replace(state, training_examples=73)
    with pytest.raises(ValueError):
        binding.snapshot_bytes(bad)


def test_should_follow_guarded_complete_swap_and_exact_one_step_rollback(states, data):
    actor, runtime, controller = setup(states)
    p = binding.ContextServingPredictor(actor, SPEC)
    _, old = guards(data)
    actor.serve(old.features, now=2)
    before = actor.serving_snapshot()
    first = p.serve(request())
    ticket = prepare(runtime, controller, data)
    assert ticket.report.decision.accepted
    assert ticket.report.evidence.candidate_resource_bytes == binding.snapshot_record_bytes(
        states[1]
    )
    receipt = controller.commit(runtime, ticket)
    after = p.serve(request())
    assert (after.generation, after.actor_version, after.completed_updates) == (1, "fixture", 12)
    assert actor.serving_snapshot().cache == {}
    assert controller.rollback(receipt) == 2
    last = p.serve(request())
    assert last.result == first.result and last.actor_version == first.actor_version == "base"
    restored = actor.serving_snapshot()
    assert restored.generation == 2 and restored.model_state == before.model_state
    assert restored.metadata == before.metadata and restored.configuration == before.configuration
    assert restored.last_cache_tick == before.last_cache_tick
    assert restored.cache.keys() == before.cache.keys()
    for key in before.cache:
        assert np.array_equal(restored.cache[key].prediction, before.cache[key].prediction)
    assert p.work == dict(attempts=3, returns=3, readouts=3, typed_attempts=3)
    assert not any(
        isinstance(x, binding.ContextPerceptronDecisionPredictor) for x in vars(p).values()
    )
    with pytest.raises(ValueError):
        controller.commit(runtime, ticket)
    with pytest.raises(ValueError):
        controller.rollback(receipt)


@pytest.mark.parametrize("fault", ["same", "resource", "actions", "latency", "old-drop"])
def test_should_keep_measured_refusal_without_publication(states, data, fault):
    p = demo.policy()
    if fault == "resource":
        p = replace(p, max_resource_bytes=1)
    if fault == "actions":
        p = replace(p, allowed_actions=("retire",))
    actor, runtime, controller = setup(states, changed=fault != "same", policy=p)
    if fault == "latency":
        ticks = iter(range(8))
        controller._evaluator._clock = lambda: float(next(ticks))
    if fault == "old-drop":
        values = iter((0.0, 1.0, 1.0, 0.0))
        controller._evaluator._utility = lambda *a: next(values)
    with pytest.raises(PromotionRejected) as caught:
        prepare(runtime, controller, data)
    assert {
        "same": "new_task_gain",
        "resource": "resource",
        "actions": "action_safety",
        "latency": "latency",
        "old-drop": "old_task_retention",
    }[fault] in caught.value.report.decision.reasons
    assert actor.version == "base" and actor.serving_snapshot().generation == 0


@pytest.mark.parametrize("role", ["train", "probe_train", "final_test", "outer_selection"])
def test_should_refuse_wrong_role_before_payload(states, data, role, monkeypatch):
    _, runtime, controller = setup(states)
    n, o = guards(data)
    object.__setattr__(n, "role", role)
    monkeypatch.setattr(controller._evaluator, "_state_valid", lambda *a: pytest.fail("payload"))
    with pytest.raises(ValueError, match="inner_guard"):
        controller.prepare(runtime, n, o, now=0, training_ids=frozenset())


@pytest.mark.parametrize("fault", ["overlap", "future", "candidate"])
def test_should_refuse_stale_or_contaminated_promotion(states, data, fault):
    actor, runtime, controller = setup(states)
    n, o = guards(data)
    if fault == "candidate":
        ticket = prepare(runtime, controller, data)
        runtime.consolidate(
            "changed", lambda s: ConsolidatedState(s, TrainingDiagnostic("fixture", 0.0))
        )
        with pytest.raises(ValueError, match="revision"):
            controller.commit(runtime, ticket)
        controller.discard(ticket)
    else:
        with pytest.raises(ValueError, match="disjoint" if fault == "overlap" else "not arrived"):
            controller.prepare(
                runtime,
                replace(n, labels_arrived_at=int(fault == "future")),
                o,
                now=0,
                training_ids=frozenset(n.sample_keys) if fault == "overlap" else frozenset(),
            )
    assert actor.version == "base"


def test_should_latch_typed_fault_and_account_attempts(states, monkeypatch):
    actor, _, _ = setup(states, changed=False)
    p = binding.ContextServingPredictor(actor, SPEC)

    def fail(*args):
        raise RuntimeError("typed fault")

    monkeypatch.setattr(binding.ContextPerceptronDecisionPredictor, "__call__", fail)
    with pytest.raises(RuntimeError, match="typed fault"):
        p(request())
    with pytest.raises(ValueError, match="stopped"):
        p(request())
    assert p.work == dict(attempts=1, returns=0, readouts=1, typed_attempts=1)


def test_should_refuse_busy_read_without_poisoning_idle_predictor(states):
    actor, _, _ = setup(states, changed=False)
    p = binding.ContextServingPredictor(actor, SPEC)
    p._gate.acquire()
    try:
        with pytest.raises(ValueError, match="busy"):
            p(request())
    finally:
        p._gate.release()
    assert p.serve(request()).generation == 0


def test_should_detach_train_and_guard_data_with_explicit_roles(data):
    training, (new, old) = data
    assert [len(b.features) for b in training] == [3, 6]
    assert all(b.role == "train" for b in training)
    assert len(new.features) == 6 and len(old.features) == 12
    arrays = [v for b in (*training, new, old) for v in (b.features, b.targets)]
    assert not any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    assert sum(a.nbytes for a in arrays) == 2160


def test_should_reject_wrong_source_before_teacher_or_learner(tmp_path, monkeypatch):
    p = tmp_path / "wrong.json"
    p.write_text("{}")
    monkeypatch.setattr(demo, "prepare_data", lambda *a: pytest.fail("teacher"))
    r = demo.run_case(p)
    assert not r["complete"] and "replaced" in r["error"] and r["cases"] == []


def test_should_preserve_completed_episodes_and_partial_typed_reads_on_later_fault(monkeypatch):
    calls = []
    rows: list[dict[str, Any]] = []
    predictor = SimpleNamespace(
        serve=lambda q: SimpleNamespace(
            generation=0, actor_version="base", completed_updates=12, result="typed"
        )
    )

    def episode(state, predict, **kwargs):
        calls.append(state)
        assert predict("request") == "typed"
        if len(calls) == 2:
            raise RuntimeError("later episode failed")
        return {"success": True, "complete": True}

    monkeypatch.setattr(demo, "run_workflow_episode", episode)
    with pytest.raises(RuntimeError, match="later episode"):
        demo._episodes(predictor, "initial", lambda: None, rows)
    assert len(rows) == 2 and rows[0]["complete"] and not rows[1]["complete"]
    assert rows[0]["episode"] == {"success": True, "complete": True}
    assert rows[1]["reads"] == [dict(generation=0, actor_version="base", completed_updates=12)]


def test_should_stop_after_incomplete_return_and_preserve_paid_episode(monkeypatch):
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        demo,
        "run_workflow_episode",
        lambda *a, **k: {
            "complete": False,
            "success": False,
            "error": "paid typed failure",
            "steps": [],
        },
    )
    with pytest.raises(ValueError, match="incomplete; paid prefix"):
        demo._episodes(None, "initial", lambda: None, rows)
    assert len(rows) == 1 and not rows[0]["complete"]
    assert rows[0]["episode"]["error"] == "paid typed failure"


def test_should_preserve_cli_default_exclusive_output_and_optin(tmp_path, monkeypatch):
    p = tmp_path / "out.json"
    calls = []

    def native(path):
        calls.append("native")
        return {"complete": True}

    def context(path):
        calls.append("context")
        return {"complete": False, "error": "paid"}

    monkeypatch.setattr(cli, "run_case", native)
    monkeypatch.setattr(demo, "run_case", context)
    assert cli.main(["--source", "unused", "--output", str(p)]) == 0
    with pytest.raises(FileExistsError):
        cli.main(["--source", "unused", "--output", str(p), "--context-head-serving"])
    p.unlink()
    assert cli.main(["--source", "unused", "--output", str(p), "--context-head-serving"]) == 1
    assert calls == ["native", "context"] and json.loads(p.read_bytes())["error"] == "paid"
