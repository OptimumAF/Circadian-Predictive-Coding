"""One joint native update; original effects, admission and charges stay owned."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import src.adapters.torch_feature_learner as port
from src.app.learner_step import update_learner
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.adaptation_action import AdaptationAction
from src.core.learner_ports import validate_gate_setting
from test_actor_shadow import PausableLearner, queue
from test_gate_actions import wrapped
from test_torch_feature_learner import METHODS, assert_tree, batch, policy


class JointLearner(PausableLearner):
    def __init__(self, value=None, calls=None, after=None):
        super().__init__(value)
        self.calls = [] if calls is None else calls
        self.after = after

    def fork(self):
        return JointLearner(self.value, self.calls, self.after)

    def validate_gate_setting(self, setting):
        validate_gate_setting(setting)

    def train_batch(self, features, targets):
        self.calls.append((1.0, "unchanged"))
        return super().train_batch(features, targets)

    def train_batch_with_rate_and_gate(self, features, targets, multiplier, setting):
        self.calls.append((multiplier, setting))
        result = super().train_batch(features, [v * multiplier for v in targets])
        if self.after is not None:
            self.after()
        return result


def combined(**options):
    return wrapped(
        source=options.pop("source", JointLearner()),
        action=options.pop(
            "action",
            AdaptationAction("effect", 1, learning_rate_multiplier=0.5, gate_setting="neutral"),
        ),
        **options,
    )


def test_joint_choice_waits_for_original_labels_and_counts_one_update_then_default():
    w, r, c, source, b, g = combined()
    queue(r)
    assert not w.train_action("effect").training.updates
    c.advance_to(2)
    assert not w.train_action("effect").training.updates
    c.advance_to(3)
    g.pause()
    assert w.train_action("effect").training.deferred_reason == "paused"
    g.resume()
    with g.serving():
        assert w.train_action("effect").training.deferred_reason == "serving_active"
    assert source.calls == []
    first = w.train_action("effect")
    assert (
        first.requested_learning_rate_multiplier == 0.5
        and first.requested_gate_setting == "neutral"
    )
    assert (
        len(first.training.updates) == 1
        and first.training.updates[0].arrived_at == first.training.updates[0].applied_at == 3
    )
    assert r.candidate_snapshot().state == [4.0, 4.0]
    queue(r, "s2", [6, 6])
    w.train_action("wake")
    assert source.calls == [(0.5, "neutral"), (1.0, "unchanged")]
    assert r.candidate_snapshot().state == [
        6,
        6,
    ] and source.snapshot_state() == r.actor.snapshot_state().state == [2, 2]
    assert b.updates_completed == g.snapshot().admitted_updates == 2
    progress = w.adaptation_snapshot().progress
    assert progress.native_attempts == progress.native_completions == 2
    with pytest.raises(ValueError, match="envelope"):
        w.train_action("effect")


@pytest.mark.parametrize(
    "fault", ["missing_joint", "missing_validator", "policy", "zero_wake", "replay", "sleep"]
)
def test_unsupported_joint_choices_refuse_before_resources_or_payload(fault, monkeypatch):
    action = AdaptationAction("effect", 1, learning_rate_multiplier=0.5, gate_setting="neutral")
    if fault == "zero_wake":
        action = replace(action, wake_updates=0)
    if fault == "replay":
        action = replace(action, replay_examples=1)
    if fault == "sleep":
        action = replace(action, sleep_updates=1)
    w, r, c, source, b, g = combined(action=action, resource=lambda: pytest.fail("resource"))
    if fault == "missing_joint":
        r._candidate.train_batch_with_rate_and_gate = None
    if fault == "missing_validator":
        r._candidate.validate_gate_setting = None
    if fault == "policy":

        def refuse(setting):
            raise ValueError("unsupported bound policy")

        r._candidate.validate_gate_setting = refuse
    queue(r)
    c.advance_to(3)
    with pytest.raises(ValueError, match="unsupported"):
        w.train_action("effect")
    assert source.calls == [] and b.updates_completed == g.snapshot().admitted_updates == 0


def test_direct_step_and_inbox_require_joint_capability_before_callbacks_or_copy(monkeypatch):
    from src.app import experience_inbox

    source = JointLearner()
    setattr(source, "train_batch_with_rate_and_gate", None)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    with pytest.raises(ValueError, match="unsupported"):
        update_learner(
            source,
            [1],
            [8, 8],
            budget,
            learning_rate_multiplier=0.5,
            gate_setting="neutral",
            on_started=lambda: pytest.fail("native entry"),
        )
    w, r, c, original, b, g = combined()
    r._candidate.train_batch_with_rate_and_gate = None
    queue(r)
    c.advance_to(3)
    monkeypatch.setattr(experience_inbox, "deepcopy", lambda value: pytest.fail("payload copied"))
    with pytest.raises(ValueError, match="unsupported"):
        r.train_ready(learning_rate_multiplier=0.5, gate_setting="neutral")
    assert b.updates_completed == budget.updates_completed == 0


def test_joint_resource_deferral_then_original_budget_refusal_has_no_native_work():
    available = [False]
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    w, r, c, source, b, g = combined(budget=budget, resource=lambda: available[0])
    queue(r)
    c.advance_to(3)
    assert w.train_action("effect").training.deferred_reason == "resource_unavailable"
    available[0] = True
    with pytest.raises(ToyExecutionStopped):
        w.train_action("effect")
    assert source.calls == [] and b.updates_completed == 0 and g.snapshot().admitted_updates == 1


def test_lost_joint_return_is_paid_unknown_and_stops_subsequent_work():
    def fail():
        raise RuntimeError("lost joint return")

    w, r, c, source, b, g = combined(source=JointLearner(after=fail))
    queue(r)
    c.advance_to(3)
    with pytest.raises(RuntimeError, match="lost joint"):
        w.train_action("effect")
    p = w.adaptation_snapshot().progress
    assert p.native_attempts == 1 and p.native_completions == 0 and p.unreturned_work_unknown
    assert b.updates_completed == 0 and g.snapshot().admitted_updates == 1 and not r.applied_updates
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")
    assert source.calls == [(0.5, "neutral")]


def test_post_update_stop_preserves_joint_receipt_charge_and_original_source_actor():
    wall = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: wall[0]
    )
    w, r, c, source, b, g = combined(
        source=JointLearner(after=lambda: wall.__setitem__(0, 2.0)), budget=budget
    )
    queue(r)
    c.advance_to(3)
    with pytest.raises(ToyExecutionStopped):
        w.train_action("effect")
    assert len(r.applied_updates) == b.updates_completed == g.snapshot().admitted_updates == 1
    assert not w.adaptation_snapshot().progress.unreturned_work_unknown
    assert r.actor.snapshot_state().state == source.snapshot_state() == [2, 2]


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 4.01, "1"])
def test_invalid_joint_rate_refuses_before_policy_native_or_payload(value, monkeypatch):
    learner: Any = object.__new__(port.TorchFeatureLearner)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("native import"))
    with pytest.raises(ValueError, match="multiplier"):
        learner.train_batch_with_rate_and_gate(None, None, value, "neutral")


@pytest.mark.parametrize(
    "method,setting", [("bp", "neutral"), ("pc", "active"), ("neutral_cpc", "active")]
)
def test_unsupported_bound_joint_gate_refuses_before_native_payload(method, setting, monkeypatch):
    learner: Any = object.__new__(port.TorchFeatureLearner)
    learner._policy = policy(method)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("native import"))
    with pytest.raises(ValueError, match="unsupported"):
        learner.train_batch_with_rate_and_gate(None, None, 0.5, setting)


@pytest.mark.parametrize("setting", [None, True, "NEUTRAL", []])
def test_invalid_joint_gate_refuses_before_policy_or_native_payload(setting, monkeypatch):
    learner: Any = object.__new__(port.TorchFeatureLearner)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("native import"))
    with pytest.raises(ValueError, match="gate setting"):
        learner.train_batch_with_rate_and_gate(None, None, 0.5, setting)


@pytest.mark.parametrize("base,multiplier", [(1e308, 2.0), (5e-324, 0.5)])
def test_joint_effective_rate_refuses_overflow_and_underflow_before_payload(
    base, multiplier, monkeypatch
):
    learner: Any = object.__new__(port.TorchFeatureLearner)
    learner._policy = policy("gated_cpc", learning_rate=base)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("native import"))
    with pytest.raises(ValueError, match="effective learning rate"):
        learner.train_batch_with_rate_and_gate(None, None, multiplier, "neutral")


@pytest.fixture(scope="module")
def native():
    """One finite native fixture per platform; all checking tests reuse it."""
    torch = port.require_torch()
    torch.set_num_threads(1)
    x, y = batch()
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(),
    )
    sources = {}
    for method in METHODS:
        source = port.TorchFeatureLearner(policy(method))
        before = source.snapshot_state()
        actual, expected = source.fork(), source.fork()
        assert actual.train_batch_with_rate_and_gate(
            x, y, 1.0, "unchanged"
        ) == expected.train_batch(x, y)
        assert_tree(torch, actual.snapshot_state(), expected.snapshot_state())
        assert actual.train_batch_with_rate_and_gate(
            x, y, 0.5, "unchanged"
        ) == expected.train_batch_with_rate_multiplier(x, y, 0.5)
        assert_tree(torch, actual.snapshot_state(), expected.snapshot_state())
        assert_tree(torch, source.snapshot_state(), before)
        sources[method] = source
    for method in ("neutral_cpc", "gated_cpc"):
        source = sources[method]
        actual, expected = source.fork(), source.fork()
        assert actual.train_batch_with_rate_and_gate(
            x, y, 1, "neutral"
        ) == expected.train_batch_with_gate_setting(x, y, "neutral")
        assert_tree(torch, actual.snapshot_state(), expected.snapshot_state())
        observed = actual.train_batch_with_rate_and_gate(x, y, 0.5, "neutral")
        values = port._features(x, 2)
        labels = torch.tensor(y[:, 0], dtype=torch.int64)
        energy = expected._head.train_step(values, labels, 0.015, 2, 0.04, gate_setting="neutral")
        assert observed.value == energy
        assert_tree(torch, actual.snapshot_state().head_state, expected._head.snapshot_state())
        assert (
            actual.policy == source.policy
            and actual.completed_updates == 2
            and actual.training_examples == 4
        )
    actual, expected = sources["gated_cpc"].fork(), sources["gated_cpc"].fork()
    assert actual.train_batch_with_rate_and_gate(
        x, y, 0.5, "active"
    ) == expected.train_batch_with_rate_multiplier(x, y, 0.5)
    assert_tree(torch, actual.snapshot_state(), expected.snapshot_state())
    learner = sources["gated_cpc"].fork()
    before = deepcopy(learner.snapshot_state())
    original = type(learner._head).train_step

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("failed staged joint update")

    with patch.object(type(learner._head), "train_step", fail):
        with pytest.raises(RuntimeError, match="staged joint"):
            learner.train_batch_with_rate_and_gate(x, y, 0.5, "neutral")
    assert_tree(torch, learner.snapshot_state(), before)
    from scripts.ticket_workflow_native import counted_learner

    entries, returns = [], []
    source = counted_learner(
        "gated_cpc", 2, 2, lambda v: entries.append(len(v)), lambda: returns.append(True)
    )
    learner = source.fork()
    for multiplier, setting in (
        (0.5, "neutral"),
        (1, "neutral"),
        (0.5, "unchanged"),
        (1, "unchanged"),
    ):
        learner.train_batch_with_rate_and_gate(x, y, multiplier, setting)
    assert entries == [2] * 4 and returns == [True] * 4 and source.completed_updates == 0
    assert (
        random.getstate() == python_rng
        and pickle.dumps(np.random.get_state()) == numpy_rng
        and torch.equal(torch.get_rng_state(), torch_rng)
    )
    return dict(
        default_methods=METHODS,
        gate_methods=("neutral_cpc", "gated_cpc"),
        atomic=True,
        counted_entries=entries,
        rng_equal=True,
    )


def test_combined_native_degeneracies_preserve_full_state_and_native_momentum(native):
    assert native["default_methods"] == METHODS and native["gate_methods"] == (
        "neutral_cpc",
        "gated_cpc",
    )


def test_combined_native_uses_one_direct_gradient_step_and_keeps_failed_state_atomic(native):
    assert native["atomic"] and native["counted_entries"] == [2] * 4 and native["rng_equal"]
