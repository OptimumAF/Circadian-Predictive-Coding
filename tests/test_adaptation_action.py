"""Finite controller data and original native-admission authority."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits
from test_actor_shadow import PausableLearner, queue, setup


def envelope(**changes):
    return replace(
        AdaptationEnvelope(
            (AdaptationAction("wait"), AdaptationAction("wake1", 1), AdaptationAction("wake2", 2)),
            8,
            4,
        ),
        **changes,
    )


def shared(*, declared=None, limits=None, source=None, resource=None, budget=None):
    runtime, clock, original, budget = setup(source=source, budget=budget)
    gate = ServingPriorityGate(
        limits or SharingLimits(2, 8, 2), resource_available=resource or (lambda: True)
    )
    wrapped = ResourceSharedRuntime(runtime, gate, action_envelope=declared or envelope())
    return wrapped, runtime, clock, gate, original, budget


@pytest.mark.parametrize(
    "change",
    [
        {"wake_updates": True},
        {"wake_updates": -1},
        {"replay_examples": 2**31},
        {"sleep_updates": 1.0},
        {"learning_rate_multiplier": True},
        {"learning_rate_multiplier": float("nan")},
        {"learning_rate_multiplier": float("inf")},
        {"learning_rate_multiplier": 0},
        {"learning_rate_multiplier": 2**2048},
        {"gate_setting": "__import__('os')"},
        {"action_id": " wait"},
    ],
)
def test_malformed_action_fields_are_refused(change):
    with pytest.raises(ValueError):
        replace(AdaptationAction("wait"), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"max_decisions": True},
        {"max_wake_updates": -1},
        {"max_replay_examples": 1.0},
        {"max_sleep_updates": 2**31},
        {"max_learning_rate_multiplier": float("inf")},
        {"gate_settings": ("unchanged", "unchanged")},
        {"gate_settings": ("execute",)},
        {"actions": []},
        {"actions": ()},
        {"actions": tuple(AdaptationAction(str(i)) for i in range(33))},
        {"actions": (AdaptationAction("wait"), AdaptationAction("wait"))},
        {"max_wake_updates": 1},
        {"actions": (object(),)},
    ],
)
def test_malformed_or_overlimit_envelope_is_refused(change):
    with pytest.raises(ValueError):
        envelope(**change)


def test_full_finite_vocabulary_is_explicit_and_no_code_or_permissions_are_accepted():
    actions = (
        AdaptationAction("wait"),
        AdaptationAction("wake", 1),
        AdaptationAction("replay", replay_examples=2),
        AdaptationAction("rate", learning_rate_multiplier=0.5),
        AdaptationAction("gate", gate_setting="neutral"),
        AdaptationAction("sleep", sleep_updates=1),
    )
    e = AdaptationEnvelope(actions, 6, 1, 2, 1, 1.0, ("unchanged", "neutral"))
    assert all(e.select(a.action_id) == a for a in actions)
    with pytest.raises(ValueError):
        e.select("__import__('os')")
    with pytest.raises(TypeError):
        cast(Any, AdaptationAction)("code", callback=lambda: None)
    with pytest.raises(TypeError):
        cast(Any, AdaptationAction)("permission", training_permission=True)


@pytest.mark.parametrize(
    "action",
    [
        AdaptationAction("replay", replay_examples=1),
        AdaptationAction("rate", learning_rate_multiplier=0.5),
        AdaptationAction("gate", gate_setting="active"),
        AdaptationAction("sleep", sleep_updates=1),
    ],
)
def test_unsupported_choice_spends_decision_but_never_reads_data_or_probes_resources(
    action, monkeypatch
):
    e = AdaptationEnvelope((action,), 2, 0, 1, 1, 1.0, ("unchanged", "active"))
    w, r, _, g, _, b = shared(declared=e, resource=lambda: pytest.fail("resource probe"))
    monkeypatch.setattr(r, "train_ready", lambda **kw: pytest.fail("payload entry"))
    monkeypatch.setattr(r._budget, "before_final", lambda: pytest.fail("budget callback"))
    with pytest.raises(ValueError, match="unsupported"):
        w.train_action(action.action_id)
    p = w.adaptation_snapshot().progress
    assert p.decisions_attempted == p.decisions_refused == 1
    assert (
        p.native_attempts
        == p.wake_admissions
        == b.updates_completed
        == g.snapshot().admitted_updates
        == 0
    )


def test_no_update_is_explicit_zero_work_without_consuming_labels_or_sharing(monkeypatch):
    w, r, c, g, source, b = shared(resource=lambda: pytest.fail("resource probe"))
    queue(r)
    c.advance_to(3)
    before = r.candidate_snapshot()
    monkeypatch.setattr(r, "train_ready", lambda **kw: pytest.fail("payload entry"))
    p = w.train_action("wait")
    assert p.action_id == "wait" and p.requested_wake_updates == 0 and p.training.updates == ()
    assert r.candidate_snapshot() == before and r.applied_updates == ()
    assert p.progress.decisions_attempted == p.progress.decisions_returned == 1
    assert p.progress.wake_admissions == g.snapshot().admitted_updates == b.updates_completed == 0


def test_bounded_action_keeps_ready_prefix_order_and_original_actor():
    w, r, c, g, source, b = shared(limits=SharingLimits(2, 8, 1))
    queue(r)
    queue(r, "s2", [9, 9])
    c.advance_to(3)
    assert [u.sample_id for u in w.train_action("wake2").training.updates] == ["s1"]
    assert b.updates_completed == 1 and r.actor.snapshot_state().state == source.snapshot_state()
    assert [u.sample_id for u in w.train_action("wake1").training.updates] == ["s2"]
    assert [u.update_number for u in r.applied_updates] == [1, 2]
    p = w.adaptation_snapshot().progress
    assert (
        p.wake_admissions
        == p.native_attempts
        == p.native_completions
        == p.poll_updates_returned
        == 2
    )
    assert not p.unreturned_work_unknown and g.snapshot().admitted_updates == 2


def test_empty_and_deferred_opportunities_count_without_native_admission():
    w, r, c, g, _, b = shared()
    assert w.train_action("wake1").training.updates == ()
    queue(r)
    c.advance_to(1)
    assert w.train_action("wake1").training.updates == ()
    c.advance_to(3)
    with g.serving():
        assert w.train_action("wake1").training.deferred_reason == "serving_active"
    g.pause()
    assert w.train_action("wake1").training.deferred_reason == "paused"
    g.resume()
    assert w.adaptation_snapshot().progress.decisions_returned == 4
    assert b.updates_completed == g.snapshot().admitted_updates == 0
    assert len(w.train_action("wake1").training.updates) == 1


def test_false_resource_and_work_quota_remain_original_authority():
    available = [False]
    w, r, c, g, _, b = shared(limits=SharingLimits(2, 1, 2), resource=lambda: available[0])
    queue(r)
    queue(r, "s2")
    c.advance_to(3)
    assert w.train_action("wake1").training.deferred_reason == "resource_unavailable"
    available[0] = True
    p = w.train_action("wake2")
    assert len(p.training.updates) == 1 and p.training.deferred_reason == "work_quota"
    assert b.updates_completed == g.snapshot().admitted_updates == 1
    assert w.train_action("wake1").training.deferred_reason == "work_quota"


def test_envelope_and_decision_quota_refusals_never_renew_budget():
    w, r, c, g, _, b = shared(declared=envelope(max_decisions=3, max_wake_updates=2))
    queue(r)
    queue(r, "s2")
    c.advance_to(3)
    w.train_action("wake1")
    with pytest.raises(ValueError, match="cumulative"):
        w.train_action("wake2")
    assert b.updates_completed == g.snapshot().admitted_updates == 1
    w.train_action("wait")
    with pytest.raises(ValueError, match="decision quota"):
        w.train_action("wake1")
    p = w.adaptation_snapshot().progress
    assert p.decisions_attempted == 3 and p.decisions_refused == 1 and p.native_attempts == 1


def test_budget_stop_after_prefix_keeps_admissions_and_committed_receipt():
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    w, r, c, g, _, b = shared(budget=budget)
    queue(r)
    queue(r, "s2")
    c.advance_to(3)
    with pytest.raises(ToyExecutionStopped):
        w.train_action("wake2")
    p = w.adaptation_snapshot().progress
    assert p.decisions_attempted == p.decisions_refused == 1
    assert p.wake_admissions == g.snapshot().admitted_updates == 2
    assert p.native_attempts == p.native_completions == b.updates_completed == 1
    assert p.poll_updates_returned == 0 and not p.unreturned_work_unknown
    assert [u.sample_id for u in r.applied_updates] == ["s1"]


def test_native_failure_is_paid_unknown_and_actor_remains_available(monkeypatch):
    w, r, c, g, source, b = shared()
    queue(r)
    c.advance_to(3)

    def fail(*args):
        raise RuntimeError("partial native failure")

    monkeypatch.setattr(r._candidate, "train_batch", fail)
    with pytest.raises(RuntimeError, match="partial native"):
        w.train_action("wake1")
    p = w.adaptation_snapshot().progress
    assert p.wake_admissions == p.native_attempts == g.snapshot().admitted_updates == 1
    assert p.native_completions == b.updates_completed == 0 and p.unreturned_work_unknown
    assert not g.snapshot().training_active
    assert w.predict([]).actor_version == "actor-0"
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake1")
    assert w.adaptation_snapshot().progress.native_attempts == 1


def test_completed_native_work_is_preserved_when_post_update_wall_budget_stops(monkeypatch):
    elapsed = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: elapsed[0]
    )
    w, r, c, g, _, _ = shared(budget=budget)
    queue(r)
    c.advance_to(3)
    original = r._candidate.train_batch

    def train(features, targets):
        result = original(features, targets)
        elapsed[0] = 2.0
        return result

    monkeypatch.setattr(r._candidate, "train_batch", train)
    with pytest.raises(ToyExecutionStopped):
        w.train_action("wake1")
    p = w.adaptation_snapshot().progress
    assert p.native_attempts == p.native_completions == budget.updates_completed == 1
    assert p.poll_updates_returned == 0 and p.decisions_refused == 1
    assert not p.unreturned_work_unknown and g.snapshot().admitted_updates == 1
    assert [u.sample_id for u in r.applied_updates] == ["s1"]


def test_snapshots_are_detached_no_history_and_explicit_mode_cannot_be_bypassed():
    w, _, _, _, _, _ = shared()
    snapshot = w.adaptation_snapshot()
    object.__setattr__(snapshot.envelope, "max_wake_updates", 999)
    object.__setattr__(snapshot.progress, "decisions_attempted", 999)
    assert w.adaptation_snapshot().envelope.max_wake_updates == 4
    assert w.adaptation_snapshot().progress.decisions_attempted == 0
    with pytest.raises(ValueError, match="explicit adaptation"):
        w.train_ready()
    with pytest.raises(ValueError, match="finite catalog"):
        w.train_action("foreign")
    assert w.adaptation_snapshot().progress.decisions_attempted == 0
    assert not any(type(v) is list for v in vars(w).values())


@pytest.mark.parametrize("changed", ["envelope", "runtime", "gate", "budget", "limits"])
def test_replaced_authority_is_refused_before_native_work(changed):
    w, r, _, g, _, _ = shared()
    if changed == "envelope":
        w._action_envelope = replace(w._action_envelope, max_decisions=99)
    if changed == "runtime":
        w._runtime = setup()[0]
    if changed == "gate":
        w._sharing = ServingPriorityGate(SharingLimits(2, 8, 2), resource_available=lambda: True)
    if changed == "budget":
        r._budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=99), lambda: 0.0)
    if changed == "limits":
        g._limits = SharingLimits(2, 99, 2)
    with pytest.raises(ValueError, match="original"):
        w.train_action("wait")
    assert w._action_progress.decisions_attempted == 0


def test_probe_owner_cannot_enroll_controller_actions():
    r = ActorShadowRuntime(
        PausableLearner(),
        actor_version="a",
        candidate_version="c",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
        max_experiences=1,
        max_consolidations=0,
        training_role="probe_train",
    )
    gate = ServingPriorityGate(SharingLimits(1, 1, 1), resource_available=lambda: True)
    with pytest.raises(ValueError, match="ordinary TRAIN"):
        ResourceSharedRuntime(r, gate, action_envelope=envelope())


def test_nested_selection_cannot_spend_another_decision():
    wrapped: list[ResourceSharedRuntime[Any, Any, Any, Any]] = []

    def probe():
        with pytest.raises(ValueError, match="busy"):
            wrapped[0].train_action("wait")
        return True

    w, r, c, _, _, _ = shared(resource=probe)
    wrapped.append(w)
    queue(r)
    c.advance_to(3)
    assert len(w.train_action("wake1").training.updates) == 1
    assert w.adaptation_snapshot().progress.decisions_attempted == 1


def test_native_no_update_and_bounded_wake_preserve_original_authorities():
    from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
    from src.adapters.torch_decision import _same_values
    from src.shared.torch_runtime import require_torch

    torch = require_torch()
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    source = TorchFeatureLearner(TorchFeaturePolicy("bp", 2, 3, 17, num_classes=2))
    initial = source.snapshot_state()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    r = ActorShadowRuntime(
        source,
        actor_version="native-0",
        candidate_version="shadow-0",
        clock=clock,
        budget=budget,
        max_experiences=2,
        max_consolidations=0,
    )
    gate = ServingPriorityGate(SharingLimits(1, 2, 1), resource_available=lambda: True)
    w = ResourceSharedRuntime(r, gate, action_envelope=envelope(max_wake_updates=2))
    for i in range(2):
        r.record_experience(
            Experience(
                str(i),
                "native",
                0,
                "native-0",
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
                "train",
                ExperiencePermissions(training=True),
            )
        )
        r.record_label(
            LabelArrival("l" + str(i), str(i), "native", 1, "native-0", np.array([[0.0], [1.0]]))
        )
    assert w.train_action("wait").training.updates == ()
    assert w.train_action("wake1").training.updates == ()
    assert gate.snapshot().admitted_updates == 0
    clock.advance_to(1)
    with gate.serving():
        assert w.train_action("wake1").training.deferred_reason == "serving_active"
    served_before = w.predict(np.array([[0.1, 0.2]], dtype=np.float64))
    first = w.train_action("wake1")
    with pytest.raises(ValueError, match="cumulative"):
        w.train_action("wake2")
    w.train_action("wait")
    second = w.train_action("wake1")
    w.train_action("wait")
    with pytest.raises(ValueError, match="decision quota"):
        w.train_action("wait")
    assert [u.sample_id for p in (first, second) for u in p.training.updates] == ["0", "1"]
    assert all(u.applied_at >= u.arrived_at == 1 > u.observed_at == 0 for u in r.applied_updates)
    candidate = r.candidate_snapshot().state
    assert candidate.completed_updates == 2 and candidate.training_examples == 4
    assert not _same_values(candidate, initial)
    assert _same_values(initial, source.snapshot_state()) and _same_values(
        initial, r.actor.snapshot_state().state
    )
    served_after = w.predict(np.array([[0.1, 0.2]], dtype=np.float64))
    np.testing.assert_array_equal(served_before.prediction, served_after.prediction)
    assert torch.equal(torch_rng, torch.get_rng_state())
    current = np.random.get_state()
    assert type(numpy_rng) is type(current) is tuple
    assert (
        numpy_rng[0] == current[0]
        and np.array_equal(numpy_rng[1], current[1])
        and numpy_rng[2:] == current[2:]
    )
    p = w.adaptation_snapshot().progress
    assert p.decisions_attempted == 8 and p.decisions_returned == 7 and p.decisions_refused == 1
    assert (
        p.wake_admissions
        == p.native_attempts
        == p.native_completions
        == p.poll_updates_returned
        == 2
    )
    assert budget.updates_completed == gate.snapshot().admitted_updates == 2
    if os.environ.get("ADAPTATION_NATIVE_OBSERVATION"):
        Path(os.environ["ADAPTATION_NATIVE_OBSERVATION"]).write_text(
            json.dumps(
                dict(
                    complete=True,
                    seed=17,
                    updates=2,
                    rows=4,
                    source_actor_rng_unchanged=True,
                    progress=asdict(p),
                    sharing=asdict(gate.snapshot()),
                    native_clock=candidate.completed_updates,
                    training_examples=candidate.training_examples,
                    receipt_order=[asdict(u) for u in r.applied_updates],
                ),
                indent=2,
            )
            + "\n",
            encoding="utf8",
        )
