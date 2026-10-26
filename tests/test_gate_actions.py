"""Finite gate choices retain native chemistry and ordinary TRAIN authority."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import pickle
import random
from typing import Any, cast

import numpy as np
import pytest

import src.adapters.torch_feature_learner as port
from src.app.learner_step import update_learner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.learner_ports import validate_gate_setting
from src.core.resource_sharing import SharingLimits
from test_actor_shadow import PausableLearner, queue, setup
from test_torch_feature_learner import METHODS, NAMES, assert_tree, batch, policy


class GateLearner(PausableLearner):
    def __init__(self, value=None, calls=None, after=None):
        super().__init__(value)
        self.calls = [] if calls is None else calls
        self.after = after

    def fork(self):
        return GateLearner(self.value, self.calls, self.after)

    def validate_gate_setting(self, setting):
        validate_gate_setting(setting)

    def train_batch(self, features, targets):
        self.calls.append("unchanged")
        return super().train_batch(features, targets)

    def train_batch_with_gate_setting(self, features, targets, setting):
        self.calls.append(setting)
        result = super().train_batch(features, targets)
        if self.after is not None:
            self.after()
        return result


def wrapped(*, source=None, budget=None, resource=None, action=None):
    runtime, clock, original, budget = setup(source=source or GateLearner(), budget=budget)
    gate = ServingPriorityGate(
        SharingLimits(1, 2, 1), resource_available=resource or (lambda: True)
    )
    envelope = AdaptationEnvelope(
        (
            AdaptationAction("wait"),
            AdaptationAction("wake", 1),
            action or AdaptationAction("effect", 1, gate_setting="neutral"),
        ),
        12,
        2,
        max_replay_examples=1,
        max_sleep_updates=1,
        max_learning_rate_multiplier=2,
        gate_settings=("unchanged", "neutral", "active"),
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
    "setting", [None, True, 1, "", "NEUTRAL", "neutral ", ["active"], object()]
)
def test_invalid_gate_refuses_before_any_native_access(setting, monkeypatch):
    learner = object.__new__(port.TorchFeatureLearner)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("native import"))
    with pytest.raises(ValueError, match="gate setting"):
        learner.train_batch_with_gate_setting(cast(Any, None), cast(Any, None), setting)
    with pytest.raises(ValueError, match="gate setting"):
        validate_gate_setting(setting)


@pytest.mark.parametrize(
    "method,setting", [("bp", "neutral"), ("pc", "active"), ("neutral_cpc", "active")]
)
def test_bound_policy_refuses_unsupported_gate_without_model_state(method, setting, monkeypatch):
    learner = object.__new__(port.TorchFeatureLearner)
    learner._policy = policy(method)
    monkeypatch.setattr(port, "require_torch", lambda: pytest.fail("native import"))
    with pytest.raises(ValueError, match="unsupported"):
        learner.train_batch_with_gate_setting(cast(Any, None), cast(Any, None), setting)


def test_gate_preserves_causal_labels_deferrals_original_authorities_and_next_default():
    w, r, c, source, b, g = wrapped()
    queue(r)
    assert w.train_action("effect").training.updates == ()
    c.advance_to(2)
    assert w.train_action("effect").training.updates == ()
    c.advance_to(3)
    g.pause()
    paused = w.train_action("effect")
    assert (
        paused.requested_gate_setting == "neutral" and paused.training.deferred_reason == "paused"
    )
    g.resume()
    with g.serving():
        assert w.train_action("effect").training.deferred_reason == "serving_active"
    assert source.calls == []
    first = w.train_action("effect")
    assert (
        first.requested_gate_setting == "neutral" and first.requested_learning_rate_multiplier == 1
    )
    assert first.training.updates[0].arrived_at == first.training.updates[0].applied_at == 3
    queue(r, "s2", [6, 6])
    second = w.train_action("wake")
    assert second.requested_gate_setting == "unchanged" and source.calls == ["neutral", "unchanged"]
    assert r.candidate_snapshot().state == [
        6,
        6,
    ] and source.snapshot_state() == r.actor.snapshot_state().state == [2, 2]
    assert b.updates_completed == g.snapshot().admitted_updates == 2
    assert second.progress.native_attempts == second.progress.native_completions == 2
    with pytest.raises(ValueError, match="envelope"):
        w.train_action("effect")


@pytest.mark.parametrize(
    "kind", ["missing", "missing_validator", "zero_wake", "rate", "replay", "sleep"]
)
def test_gate_combination_or_capability_refuses_before_resources(kind):
    action = AdaptationAction("effect", 1, gate_setting="neutral")
    if kind == "zero_wake":
        action = replace(action, wake_updates=0)
    if kind == "rate":
        action = replace(action, learning_rate_multiplier=0.5)
    if kind == "replay":
        action = replace(action, replay_examples=1)
    if kind == "sleep":
        action = replace(action, sleep_updates=1)
    source = PausableLearner() if kind == "missing" else GateLearner()
    w, r, c, original, b, g = wrapped(
        source=source, action=action, resource=lambda: pytest.fail("resources")
    )
    if kind == "missing_validator":
        r._candidate.validate_gate_setting = None
    queue(r)
    c.advance_to(3)
    with pytest.raises(ValueError, match="unsupported"):
        w.train_action("effect")
    assert b.updates_completed == g.snapshot().admitted_updates == 0 and r.applied_updates == ()
    assert w.adaptation_snapshot().progress.decisions_refused == 1


def test_gate_resource_deferral_then_original_budget_refusal_never_calls_native():
    available = [False]
    b = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    w, r, c, source, _, g = wrapped(budget=b, resource=lambda: available[0])
    queue(r)
    c.advance_to(3)
    assert w.train_action("effect").training.deferred_reason == "resource_unavailable"
    available[0] = True
    with pytest.raises(ToyExecutionStopped):
        w.train_action("effect")
    assert source.calls == [] and b.updates_completed == 0 and g.snapshot().admitted_updates == 1


def test_unknown_native_failure_stops_and_never_refunds_admission():
    def fail():
        raise RuntimeError("lost return")

    w, r, c, source, b, g = wrapped(source=GateLearner(after=fail))
    queue(r)
    c.advance_to(3)
    with pytest.raises(RuntimeError, match="lost return"):
        w.train_action("effect")
    assert w.adaptation_snapshot().progress.unreturned_work_unknown
    assert (
        b.updates_completed == 0 and r.applied_updates == () and g.snapshot().admitted_updates == 1
    )
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("effect")
    assert source.calls == ["neutral"]


def test_late_gate_stop_keeps_receipt_and_charge():
    ticks = [0.0]
    b = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: ticks[0]
    )
    w, r, c, source, _, g = wrapped(
        source=GateLearner(after=lambda: ticks.__setitem__(0, 2.0)), budget=b
    )
    queue(r)
    c.advance_to(3)
    with pytest.raises(ToyExecutionStopped):
        w.train_action("effect")
    assert len(r.applied_updates) == b.updates_completed == g.snapshot().admitted_updates == 1
    assert not w.adaptation_snapshot().progress.unreturned_work_unknown


def test_generic_optional_port_and_unchanged_dispatch_preserve_old_method():
    b = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    source = PausableLearner()
    with pytest.raises(ValueError, match="unsupported"):
        update_learner(source, [1], [4, 4], b, gate_setting="neutral")
    assert b.updates_completed == 0
    update_learner(source, [1], [4, 4], b, gate_setting="unchanged")
    assert source.value == [4, 4] and b.updates_completed == 1


def test_direct_runtime_gate_refuses_unsupported_owner_before_payload_copy(monkeypatch):
    import src.app.experience_inbox as inbox_module

    runtime, clock, source, budget = setup(source=PausableLearner())
    queue(runtime)
    clock.advance_to(3)
    monkeypatch.setattr(inbox_module, "deepcopy", lambda value: pytest.fail("payload copied"))
    with pytest.raises(ValueError, match="unsupported"):
        runtime.train_ready(gate_setting="neutral")
    assert runtime.applied_updates == () and budget.updates_completed == 0


@pytest.mark.parametrize("setting", ["neutral", "active", "unchanged"])
def test_port_unchanged_delegates_public_observer_and_gate_effects_are_counted(setting):
    # Pure optional-port dispatch makes subclass/default behavior observable.
    learner = GateLearner()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    update_learner(learner, [1], [4, 4], budget, gate_setting=setting)
    assert learner.calls == [setting] and budget.updates_completed == 1


@pytest.mark.parametrize("method", METHODS)
def test_native_unchanged_is_complete_default_equivalence_and_preserves_rng(method):
    torch = port.require_torch()
    x, y = batch()
    source = port.TorchFeatureLearner(policy(method))
    normal = source.fork()
    effect = source.fork()
    numpy_rng = pickle.dumps(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    assert normal.train_batch(x, y) == effect.train_batch_with_gate_setting(x, y, "unchanged")
    assert_tree(torch, effect.snapshot_state(), normal.snapshot_state())
    assert numpy_rng == pickle.dumps(np.random.get_state()) and torch.equal(
        torch_rng, torch.get_rng_state()
    )


def test_native_active_matches_default_full_chemistry_and_neutral_matches_unit_plasticity_reference():
    torch = port.require_torch()
    x, y = batch()
    source = port.TorchFeatureLearner(policy("gated_cpc"))
    normal = source.fork()
    active = source.fork()
    neutral = source.fork()
    assert normal.train_batch(x, y) == active.train_batch_with_gate_setting(x, y, "active")
    assert_tree(torch, active.snapshot_state(), normal.snapshot_state())
    reference = source.fork()
    reference._head.config = replace(reference._head.config, min_plasticity=1.0)
    # Direct native reference intentionally differs only in its declared gate floor.
    values = port._features(x, 2)
    labels = torch.tensor(y[:, 0], dtype=torch.int64)
    reference._head.train_step(values, labels, 0.03, 2, 0.04)
    neutral.train_batch_with_gate_setting(x, y, "neutral")
    expected = reference._head.snapshot_state()
    expected["config"] = asdict(port._config(source.policy))
    assert_tree(torch, neutral.snapshot_state().head_state, expected)
    assert any(
        not torch.equal(
            neutral.snapshot_state().head_state[n], normal.snapshot_state().head_state[n]
        )
        for n in NAMES
    )
    assert neutral.snapshot_state().policy == source.policy
    before = neutral.snapshot_state()
    copy = source.fork()
    copy.restore_state(before)
    assert neutral.train_batch(x, y) == copy.train_batch(x, y)
    assert_tree(torch, neutral.snapshot_state(), copy.snapshot_state())


def test_neutral_cpc_neutral_gate_retains_original_pc_reference():
    torch = port.require_torch()
    x, y = batch()
    neutral = port.TorchFeatureLearner(policy("neutral_cpc"))
    pc = port.TorchFeatureLearner(policy("pc"))
    neutral.train_batch_with_gate_setting(x, y, "neutral")
    pc.train_batch(x, y)
    for name in NAMES:
        assert_tree(
            torch, neutral.snapshot_state().head_state[name], pc.snapshot_state().head_state[name]
        )


def test_failed_native_gate_discards_staged_weights_chemistry_and_all_clocks(monkeypatch):
    torch = port.require_torch()
    x, y = batch()
    learner = port.TorchFeatureLearner(policy("gated_cpc"))
    saved = deepcopy(learner.snapshot_state())
    original = type(learner._head).train_step

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("staged gate failure")

    monkeypatch.setattr(type(learner._head), "train_step", fail)
    with pytest.raises(RuntimeError, match="staged gate"):
        learner.train_batch_with_gate_setting(x, y, "neutral")
    assert_tree(torch, learner.snapshot_state(), saved)


def test_existing_counted_native_forks_charge_gate_once_and_unchanged_once():
    from scripts.ticket_workflow_native import counted_learner

    before = []
    after = []
    source = counted_learner(
        "gated_cpc", 2, 2, lambda x: before.append(len(x)), lambda: after.append(True)
    )
    learner = source.fork()
    x, y = batch()
    learner.train_batch_with_gate_setting(x, y, "neutral")
    learner.train_batch_with_gate_setting(x, y, "unchanged")
    assert before == [2, 2] and after == [True, True] and source.completed_updates == 0


def test_native_fixed_three_update_gate_case():
    """Run only after prerequisite gates; no warmup, teacher or selection."""
    output = os.environ.get("GATE_ACTION_NATIVE_REPORT")
    if not output:
        pytest.skip("separate prospectively budgeted native case")
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival
    from src.shared.process_memory import ProcessRssSampler
    from time import monotonic

    torch = port.require_torch()
    x, y = batch()
    python_rng = random.getstate()
    source = port.TorchFeatureLearner(policy("gated_cpc"))
    initial = source.snapshot_state()
    numpy_rng = pickle.dumps(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        b = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=3, max_wall_seconds=60, max_process_rss_bytes=512 * 1024**2
            ),
            monotonic,
        )
        b.attach_memory(sampler)
        r, c, _, _ = setup(source=source, budget=b)
        g = ServingPriorityGate(SharingLimits(1, 3, 1), resource_available=lambda: True)
        actions = tuple(
            AdaptationAction(s, 1, gate_setting=s) for s in ("neutral", "active", "unchanged")
        )
        envelope = AdaptationEnvelope(
            actions, 5, 3, gate_settings=("unchanged", "neutral", "active")
        )
        w = ResourceSharedRuntime(r, g, action_envelope=envelope)
        polls = []
        for i, setting in enumerate(("neutral", "active", "unchanged")):
            tick = 1 + 3 * i
            c.advance_to(tick)
            r.record_experience(
                Experience(
                    f"s{i}",
                    f"e{i}",
                    tick,
                    "actor-0",
                    x,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            c.advance_to(tick + 1)
            r.record_label(
                LabelArrival(f"label{i}", f"s{i}", f"e{i}", tick + 1, "actor-0", y, "train")
            )
            if i == 0:
                g.pause()
                polls.append(w.train_action(setting))
                assert polls[-1].training.deferred_reason == "paused"
                g.resume()
            polls.append(w.train_action(setting))
        current = r.candidate_snapshot().state
        assert current.completed_updates == 3 and current.training_examples == 6
        assert_tree(torch, source.snapshot_state(), initial)
        assert_tree(torch, r.actor.snapshot_state().state, initial)
        assert_tree(torch, current.policy, initial.policy)
        assert len(r.applied_updates) == b.updates_completed == g.snapshot().admitted_updates == 3
        assert numpy_rng == pickle.dumps(np.random.get_state()) and torch.equal(
            torch_rng, torch.get_rng_state()
        )
        assert random.getstate() == python_rng
        from scripts.ticket_workflow_native import snapshot_record

        report = dict(
            complete=True,
            source_unchanged=True,
            actor_unchanged=True,
            numpy_rng_unchanged=True,
            torch_rng_unchanged=True,
            python_rng_unchanged=random.getstate() == python_rng,
            candidate=snapshot_record(current),
            polls=snapshot_record(polls),
            receipts=snapshot_record(r.applied_updates),
            sharing=snapshot_record(g.snapshot()),
            budget=dict(
                limits=asdict(b.budget),
                updates_completed=b.updates_completed,
                started_at=b.started_at,
                last_clock=b.last_clock,
            ),
            sampled_process_rss=asdict(sampler.snapshot()),
        )
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
