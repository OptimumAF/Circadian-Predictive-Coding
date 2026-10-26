"""One-update native rate effects and original causal admission/accounting."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
from typing import Any, cast

import numpy as np
import pytest

import src.adapters.torch_feature_learner as port
from src.app.learner_step import update_learner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.learner_ports import validate_learning_rate_multiplier
from src.core.resource_sharing import SharingLimits
from test_actor_shadow import PausableLearner, queue, setup
from test_torch_feature_learner import METHODS, NAMES, assert_tree, batch, native_state, policy


class RateLearner(PausableLearner):
    def __init__(self, value=None, calls=None, after=None):
        super().__init__(value)
        self.calls = [] if calls is None else calls
        self.after = after

    def fork(self):
        return RateLearner(self.value, self.calls, self.after)

    def train_batch(self, features, targets):
        self.calls.append(1.0)
        return super().train_batch(features, targets)

    def train_batch_with_rate_multiplier(self, features, targets, multiplier):
        self.calls.append(multiplier)
        result = super().train_batch(features, [v * multiplier for v in targets])
        if self.after is not None:
            self.after()
        return result


def wrapped(*, source=None, budget=None, resource=None, actions=None, maximum=1.0):
    runtime, clock, original, budget = setup(source=source or RateLearner(), budget=budget)
    gate = ServingPriorityGate(
        SharingLimits(1, 2, 1), resource_available=resource or (lambda: True)
    )
    envelope = AdaptationEnvelope(
        actions
        or (
            AdaptationAction("wait"),
            AdaptationAction("half", 1, learning_rate_multiplier=0.5),
            AdaptationAction("wake", 1),
        ),
        10,
        2,
        max_learning_rate_multiplier=maximum,
    )
    return (
        ResourceSharedRuntime(runtime, gate, action_envelope=envelope),
        runtime,
        clock,
        original,
        budget,
        gate,
    )


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, float("nan"), float("inf"), 4.01, 2**2048, "1"]
)
def test_invalid_multiplier_is_refused_before_native_access(value, monkeypatch):
    learner = object.__new__(port.TorchFeatureLearner)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("loaded torch"))
    with pytest.raises(ValueError, match="multiplier"):
        learner.train_batch_with_rate_multiplier(cast(Any, None), cast(Any, None), value)
    with pytest.raises(ValueError, match="multiplier"):
        validate_learning_rate_multiplier(value)


@pytest.mark.parametrize("base,multiplier", [(1e308, 2.0), (5e-324, 0.5)])
def test_effective_rate_overflow_or_underflow_is_refused_before_torch(
    base, multiplier, monkeypatch
):
    learner = object.__new__(port.TorchFeatureLearner)
    learner._policy = policy("bp", learning_rate=base)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("loaded torch"))
    features, targets = batch()
    with pytest.raises(ValueError, match="effective learning rate"):
        learner.train_batch_with_rate_multiplier(features, targets, multiplier)


def test_rate_choice_keeps_causal_labels_original_limits_actor_and_next_default_rate():
    w, r, c, source, b, g = wrapped()
    queue(r)
    assert w.train_action("half").training.updates == ()
    c.advance_to(2)
    assert w.train_action("half").training.updates == ()
    c.advance_to(3)
    g.pause()
    assert w.train_action("half").training.deferred_reason == "paused"
    g.resume()
    with g.serving():
        assert w.train_action("half").training.deferred_reason == "serving_active"
    assert source.calls == []
    first = w.train_action("half")
    assert first.requested_learning_rate_multiplier == 0.5
    assert first.training.updates[0].arrived_at == first.training.updates[0].applied_at == 3
    assert r.candidate_snapshot().state == [4.0, 4.0]
    queue(r, "s2", [6, 6])
    second = w.train_action("wake")
    assert second.requested_learning_rate_multiplier == 1
    assert r.candidate_snapshot().state == [6, 6] and source.calls == [0.5, 1.0]
    assert r.actor.snapshot_state().state == source.snapshot_state() == [2, 2]
    assert b.updates_completed == g.snapshot().admitted_updates == 2
    assert second.progress.native_attempts == second.progress.native_completions == 2
    with pytest.raises(ValueError, match="envelope"):
        w.train_action("half")
    assert source.calls == [0.5, 1.0]


def test_resource_deferral_then_original_budget_refusal_has_no_native_work():
    available = [False]
    b = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    w, r, c, source, _, g = wrapped(budget=b, resource=lambda: available[0])
    queue(r)
    c.advance_to(3)
    assert w.train_action("half").training.deferred_reason == "resource_unavailable"
    available[0] = True
    with pytest.raises(ToyExecutionStopped):
        w.train_action("half")
    assert source.calls == [] and b.updates_completed == 0
    assert g.snapshot().admitted_updates == 1 and r.applied_updates == ()


@pytest.mark.parametrize(
    "kind", ["missing", "rate_only", "replay_rate", "gate", "sleep", "ceiling"]
)
def test_unsupported_rate_shapes_or_owner_refuse_before_resources(kind):
    action = AdaptationAction("effect", 1, learning_rate_multiplier=0.5)
    if kind == "rate_only":
        action = replace(action, wake_updates=0)
    elif kind == "replay_rate":
        action = replace(action, replay_examples=1)
    elif kind == "gate":
        action = replace(action, gate_setting="active")
    elif kind == "sleep":
        action = replace(action, sleep_updates=1)
    elif kind == "ceiling":
        action = replace(action, learning_rate_multiplier=5.0)
    r, c, source, b = setup(source=PausableLearner() if kind == "missing" else RateLearner())
    g = ServingPriorityGate(
        SharingLimits(1, 2, 1), resource_available=lambda: pytest.fail("resource")
    )
    e = AdaptationEnvelope((action,), 2, 1, 1, 1, 5.0, ("unchanged", "active"))
    w = ResourceSharedRuntime(r, g, action_envelope=e)
    with pytest.raises(ValueError, match="unsupported|multiplier"):
        w.train_action("effect")
    assert g.snapshot().admitted_updates == b.updates_completed == 0
    assert w.adaptation_snapshot().progress.decisions_refused == 1


def test_unknown_rate_failure_stops_native_work_without_refunding_admission():
    def fail():
        raise RuntimeError("lost rate return")

    w, r, c, source, b, g = wrapped(source=RateLearner(after=fail))
    queue(r)
    c.advance_to(3)
    with pytest.raises(RuntimeError, match="lost rate return"):
        w.train_action("half")
    p = w.adaptation_snapshot().progress
    assert p.native_attempts == 1 and p.native_completions == 0 and p.unreturned_work_unknown
    assert b.updates_completed == 0 and g.snapshot().admitted_updates == 1
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")
    assert source.calls == [0.5]


def test_late_rate_stop_keeps_committed_receipt_and_original_charge():
    wall = [0.0]
    b = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: wall[0]
    )
    w, r, c, source, _, g = wrapped(
        source=RateLearner(after=lambda: wall.__setitem__(0, 2.0)), budget=b
    )
    queue(r)
    c.advance_to(3)
    with pytest.raises(ToyExecutionStopped):
        w.train_action("half")
    assert len(r.applied_updates) == b.updates_completed == g.snapshot().admitted_updates == 1
    assert source.calls == [0.5] and r.candidate_snapshot().state == [4.0, 4.0]
    assert not w.adaptation_snapshot().progress.unreturned_work_unknown


def test_nonunit_generic_port_is_optional_and_unity_uses_original_method():
    b = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    source = PausableLearner()
    with pytest.raises(ValueError, match="unsupported"):
        update_learner(source, [1], [8, 8], b, learning_rate_multiplier=0.5)
    assert b.updates_completed == 0
    update_learner(source, [1], [8, 8], b)
    assert b.updates_completed == 1 and source.value == [8, 8]


def test_rate_wake_interleaves_with_fixed_replay_using_observed_native_receipts(monkeypatch):
    from test_categorical_replay_action import setup as replay_setup, ready

    w, r, replay, source, c, wake, b, g, _ = replay_setup(
        extra_actions=(AdaptationAction("half", 1, learning_rate_multiplier=0.5),)
    )
    seen = []

    def rate(features, targets, multiplier):
        seen.append(multiplier)
        return r._candidate.train_batch(features, targets)

    monkeypatch.setattr(r._candidate, "train_batch_with_rate_multiplier", rate, raising=False)
    ready(r, c)
    w.train_action("half")
    w.train_action("replay")
    ready(r, c, "s2")
    w.train_action("wake")
    assert seen == [0.5] and not replay.stopped
    assert r.candidate_snapshot().state.completed_updates == 3
    assert wake.updates_completed == 2 and b.replay_examples_completed == 3


@pytest.mark.parametrize("method", METHODS)
def test_native_unity_matches_complete_default_continuation_and_caller_rng(method):
    torch = port.require_torch()
    rng, nrng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    left, right = port.TorchFeatureLearner(policy(method)), port.TorchFeatureLearner(policy(method))
    x, y = batch()
    original = pickle.dumps((x, y))
    for _ in range(2):
        assert left.train_batch(x, y) == right.train_batch_with_rate_multiplier(x, y, 1.0)
        assert_tree(torch, left.snapshot_state(), right.snapshot_state())
    assert original == pickle.dumps((x, y))
    assert torch.equal(rng, torch.get_rng_state()) and nrng == pickle.dumps(np.random.get_state())


def reference_rate_step(head, optimizer, features, targets, base_policy, multiplier):
    """Actual native math reference, counted as paid work by the finite meter."""
    torch = port.require_torch()
    values = torch.from_numpy(features.astype(np.float32))
    labels = torch.from_numpy(targets[:, 0].astype(np.int64))
    if optimizer is None:
        return head.train_step(
            values,
            labels,
            base_policy.learning_rate * multiplier,
            base_policy.inference_steps,
            base_policy.inference_learning_rate,
        )
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(head.forward_logits(values), labels)
    result = float(loss.item())
    loss.backward()
    optimizer.param_groups[0]["lr"] = float(base_policy.learning_rate * multiplier)
    optimizer.step()
    optimizer.param_groups[0]["lr"] = float(base_policy.learning_rate)
    return result


@pytest.mark.parametrize("method", METHODS)
def test_native_nonunit_step_and_next_default_match_native_reference_without_policy_mutation(
    method,
):
    torch = port.require_torch()
    learner = port.TorchFeatureLearner(policy(method))
    x, y = batch()
    learner.train_batch(x, y)  # Nonzero real optimizer/native history before the effect.
    saved = learner.snapshot_state()
    head, optimizer = port._restore_components(saved)
    for multiplier in (0.5, 1.0):
        expected = reference_rate_step(head, optimizer, x, y, saved.policy, multiplier)
        actual = learner.train_batch_with_rate_multiplier(x, y, multiplier)
        state = learner.snapshot_state()
        assert actual.value == expected and state.policy == saved.policy
        assert_tree(torch, state.head_state, native_state(head, method))
        if optimizer is not None:
            assert state.optimizer_state is not None
            assert_tree(torch, state.optimizer_state, optimizer.state_dict())
            assert_tree(torch, state.gradients, {name: getattr(head, name).grad for name in NAMES})
            assert state.optimizer_state["param_groups"][0]["lr"] == saved.policy.learning_rate
    assert learner.completed_updates == 3 and learner.training_examples == 6
    child = learner.fork()
    assert_tree(torch, child.snapshot_state(), learner.snapshot_state())


@pytest.mark.parametrize("method", METHODS)
def test_native_failed_scaled_step_discards_staged_mutation_and_preserves_complete_state(
    method, monkeypatch
):
    torch = port.require_torch()
    learner = port.TorchFeatureLearner(policy(method))
    x, y = batch()
    learner.train_batch(x, y)
    saved = deepcopy(learner.snapshot_state())
    cls, name = (torch.optim.SGD, "step") if method == "bp" else (type(learner._head), "train_step")
    original = getattr(cls, name)

    def mutate_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("failed staged rate step")

    monkeypatch.setattr(cls, name, mutate_then_fail)
    with pytest.raises(RuntimeError, match="staged rate"):
        learner.train_batch_with_rate_multiplier(x, y, 0.5)
    assert_tree(torch, learner.snapshot_state(), saved)


def test_native_rate_example_preserves_original_source_actor_and_policy():
    from scripts.bounded_rate_action_example import run_example

    report = run_example(Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"))
    if os.environ.get("RATE_ACTION_NATIVE_REPORT"):
        Path(os.environ["RATE_ACTION_NATIVE_REPORT"]).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf8"
        )
    assert report["complete"], report.get("error")
    assert report["source_unchanged"] and report["actor_unchanged"]
    assert report["numpy_rng_unchanged"] and report["torch_rng_unchanged"]
    assert (
        report["native_work"]["attempted_updates"] == report["native_work"]["returned_updates"] == 2
    )
    assert report["native_work"]["attempted_rows"] == 12
    assert report["candidate"]["state"]["completed_updates"] == 98
    assert report["candidate"]["state"]["training_examples"] == 588
    assert report["wake_updates"] == report["sharing"]["admitted_updates"] == 2
    assert report["base_rate_preserved"] and report["selected_multipliers"] == [0.5, 1.0]
