"""Finite ordinary TRAIN reference windows with scripted owned learner states."""

from dataclasses import replace
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.learner_ports import TrainingDiagnostic
from src.core.reference_controller import ReferencePolicy

WORK = dict(train_calls=0, train_rows=0, predict_calls=0, predict_rows=0)


class LiteralLearner:
    def __init__(self, calls=None, hooks=None, state=0):
        self.calls = [] if calls is None else calls
        self.hooks = {} if hooks is None else hooks
        self.state = state

    def fork(self):
        return LiteralLearner(self.calls, self.hooks, self.state)

    def snapshot_state(self):
        return [self.state]

    def restore_state(self, state):
        self.state = state[0]

    def train_batch(self, features, targets):
        WORK["train_calls"] += 1
        WORK["train_rows"] += len(features)
        self.calls.append((float(features[0, 0]), float(targets[0, 0])))
        self.state += 1
        if "train" in self.hooks:
            self.hooks["train"]()
        return TrainingDiagnostic("literal_update", float(self.state))

    def predict(self, features):
        WORK["predict_calls"] += 1
        WORK["predict_rows"] += len(features)
        raise AssertionError("reference window must never predict or evaluate")


def make_stream(*, delay=1, credit=4, observe=None, scope=True, checkpoints=(0, 1, 2, 3, 4)):
    source = LiteralLearner()
    now = [0.0]
    session = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=credit, max_wall_seconds=60), lambda: now[0]
    )
    training = tuple(
        AttributionBatch(
            "train", "A", (f"train-{i}",), np.array([[float(i)]]), np.array([[i % 2.0]])
        )
        for i in range(4)
    )
    heldout = (
        ()
        if scope
        else (
            AttributionBatch("final_test", "A", ("test-a",), np.array([[5.0]]), np.array([[1.0]])),
        )
    )
    owner = AttributionStream(
        source,
        training,
        heldout,
        checkpoint_updates=checkpoints,
        label_delay=delay,
        budget=session,
        training_only=scope,
        observe_checkpoint=observe,
    )
    return owner, source, session, now


@pytest.mark.parametrize("delay", [0, 1, 2, 5])
@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware"])
def test_should_preserve_causal_fifo_and_equal_window_without_forcing_training(mode, delay):
    observed = []
    owner, source, session, _ = make_stream(
        delay=delay, observe=lambda n, s, b: observed.append((n, deepcopy(s), b))
    )
    original_actor = owner._runtime.actor.snapshot_state()
    original_start, clock = session.started_at, session.clock
    result = owner.run_reference_window(ReferencePolicy(mode, period=1, reserved_updates=0))
    count = 0 if mode == "no_learning" else 4
    assert result["status"] == "complete" and result["window_complete"]
    assert result["training_complete"] is (count == 4)
    assert result["pending_unapplied"] == 4 - count
    assert result["completed_updates"] == session.updates_completed == count
    assert result["training_examples"] == count and result["score_forward_calls"] == 0
    assert len(result["decisions"]) == result["next_tick"] == 4 + delay
    assert [d["observation"]["tick"] for d in result["decisions"]] == list(range(4 + delay))
    assert all(d["returned"] and d["observation"]["feedback"] is None for d in result["decisions"])
    assert [r["arrived_at"] for r in result["applied_updates"]] == list(range(delay, delay + count))
    assert [r["applied_at"] for r in result["applied_updates"]] == list(range(delay, delay + count))
    assert source.calls == [(float(i), i % 2.0) for i in range(count)]
    assert [n for n, _, _ in observed] == list(range(count + 1))
    assert source.state == 0 and owner._runtime.actor.snapshot_state() == original_actor
    assert session.started_at == original_start and session.clock is clock
    assert not result["unreturned_work_unknown"] and result["replay_examples"] == 0


@pytest.mark.parametrize(
    "policy,credit,ticks",
    [
        (ReferencePolicy("periodic"), 4, [2, 4]),
        (ReferencePolicy("periodic", period=1), 1, [1]),
        (ReferencePolicy("periodic", period=1), 0, []),
        (ReferencePolicy("budget_aware", reserved_updates=2), 4, [3, 4]),
    ],
)
def test_should_respect_fixed_period_and_original_hard_credit(policy, credit, ticks):
    owner, source, session, _ = make_stream(credit=credit)
    result = owner.run_reference_window(policy)
    assert [r["applied_at"] for r in result["applied_updates"]] == ticks
    assert [x for x, _ in source.calls] == list(map(float, range(len(ticks))))
    assert session.updates_completed == len(ticks)
    assert result["pending_unapplied"] == 4 - len(ticks)


def test_should_seal_partial_window_and_return_only_detached_progress():
    owner, _, _, _ = make_stream()
    assert owner.reference_report is None
    result = owner.run_reference_window(ReferencePolicy("no_learning"))
    result["decisions"].clear()
    detached = owner.reference_report
    assert detached is not None and len(detached["decisions"]) == 5
    detached["policy"]["mode"] = "drift"
    assert owner.reference_report["policy"]["mode"] == "no_learning"
    for action in (
        owner.finish,
        owner.pause,
        owner.candidate_snapshot,
        lambda: owner.run_until(4),
        lambda: owner.run_reference_window(ReferencePolicy("periodic")),
    ):
        with pytest.raises(ValueError, match="failed or finished"):
            action()


@pytest.mark.parametrize("policy", [None, object(), "periodic", ReferencePolicy("drift")])
def test_should_refuse_unsupported_policy_before_any_tick_or_train(policy):
    owner, source, session, _ = make_stream()
    with pytest.raises(ValueError, match="reference window requires"):
        owner.run_reference_window(policy)
    assert owner.next_tick == 0 and source.calls == [] and session.updates_completed == 0
    assert owner.reference_report is None


@pytest.mark.parametrize(
    "scope", ["evaluation", "unbounded", "sparse", "started", "probe", "helper", "features"]
)
def test_should_refuse_incompatible_stream_before_reference_window(scope):
    owner, source, session, _ = make_stream(
        credit=None if scope == "unbounded" else 4,
        scope=scope != "evaluation",
        checkpoints=(0, 4) if scope == "sparse" else (0, 1, 2, 3, 4),
    )
    if scope == "started":
        owner.run_until(1)
    elif scope == "probe":
        owner._training_role = "probe_train"
    elif scope == "helper":
        owner._rehearsal = object()
    elif scope == "features":
        owner._feature_observer = lambda *args: None
    before = deepcopy(source.calls), owner.next_tick, session.updates_completed
    with pytest.raises(ValueError, match="reference window requires"):
        owner.run_reference_window(ReferencePolicy("periodic"))
    assert (source.calls, owner.next_tick, session.updates_completed) == before
    assert owner.reference_report is None


@pytest.mark.parametrize("mutation", ["policy", "budget", "source", "actor"])
def test_should_refuse_observed_mutation_and_preserve_paid_prefix(mutation):
    policy = ReferencePolicy("periodic", period=1)
    bound: dict[str, Any] = {}

    def observe(n, state, batch):
        if n != 1:
            return
        if mutation == "policy":
            object.__setattr__(policy, "period", 2)
        elif mutation == "budget":
            bound["session"].budget = replace(bound["session"].budget, max_training_updates=9)
        elif mutation == "source":
            bound["source"].state = 99
        else:
            bound["owner"]._runtime.actor._learner.state = 99

    owner, source, session, _ = make_stream(observe=observe)
    bound.update(owner=owner, source=source, session=session)
    with pytest.raises(ValueError, match="changed|original budget"):
        owner.run_reference_window(policy)
    result = owner.reference_report
    assert result["status"] == "failed" and not result["window_complete"]
    assert len(result["applied_updates"]) == result["budget_updates"] == 1
    assert result["pending_unapplied"] == 3 and not result["unreturned_work_unknown"]
    with pytest.raises(ValueError, match="failed or finished"):
        owner.run_until(4)


def test_should_preserve_receipt_when_checkpoint_observer_hits_original_wall_limit():
    bound: dict[str, Any] = {}

    def observe(n, state, batch):
        if n:
            bound["now"][0] = 61

    owner, _, session, now = make_stream(observe=observe)
    bound["now"] = now
    start, clock = session.started_at, session.clock
    with pytest.raises(ToyExecutionStopped):
        owner.run_reference_window(ReferencePolicy("periodic", period=1))
    result = owner.reference_report
    assert result["status"] == "failed" and result["budget_updates"] == 1
    assert len(result["applied_updates"]) == 1 and not result["unreturned_work_unknown"]
    assert session.started_at == start and session.clock is clock


def test_should_mark_lost_native_return_unknown_and_refuse_more_work():
    owner, source, session, _ = make_stream()

    def lose_return():
        raise RuntimeError("lost native return")

    source.hooks["train"] = lose_return
    with pytest.raises(RuntimeError, match="lost native return"):
        owner.run_reference_window(ReferencePolicy("periodic", period=1))
    result = owner.reference_report
    assert result["status"] == "failed" and result["attempted_wakes"] == 1
    assert result["applied_updates"] == [] and result["unreturned_work_unknown"]
    # Original ToyBudget counts committed returns; the failed attempt stays
    # separately paid/unknown and this owner cannot continue or renew it.
    assert result["budget_updates"] == session.updates_completed == 0
    assert len(source.calls) == 1
    with pytest.raises(ValueError, match="failed or finished"):
        owner.run_reference_window(ReferencePolicy("periodic"))


def test_should_keep_always_train_default_and_full_receipt_result_unchanged():
    reference, source, _, _ = make_stream()
    default, original, _, _ = make_stream()
    actual = reference.run_reference_window(ReferencePolicy("periodic", period=1))
    default.run_until(4)
    expected = default.finish()
    assert actual["applied_updates"] == expected["applied_updates"]
    assert actual["training_examples"] == expected["training_examples"]
    assert source.calls == original.calls
    assert "window_complete" not in expected and default.reference_report is None
