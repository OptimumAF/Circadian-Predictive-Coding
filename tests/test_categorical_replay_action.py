"""Real admission/receipt/ownership behavior with a counted scalar native port."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.categorical_replay_action import CategoricalReplayAction
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration


@dataclass(frozen=True)
class Policy:
    feature_dim: int = 7
    num_classes: int = 6


@dataclass(frozen=True)
class State:
    policy: Policy = Policy()
    completed_updates: int = 0
    training_examples: int = 0
    weight: float = 0.0
    momentum: float = 0.0


class ScalarLearner:
    def __init__(self, calls=None, faults=None):
        self.state = State()
        self.calls = [] if calls is None else calls
        self.faults = {} if faults is None else faults

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        child = ScalarLearner(self.calls, self.faults)
        child.restore_state(self.state)
        return child

    def predict(self, features):
        return np.full((len(features), 6), 1 / 6)

    def train_batch(self, features, targets):
        self.calls.append(len(features))
        if self.faults.get("raise_replay") and len(features) == 3:
            raise RuntimeError("lost native replay return")
        value = float(np.sum(features) + np.sum(targets))
        self.state = replace(
            self.state,
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + len(features),
            weight=self.state.weight + value,
            momentum=value,
        )
        return TrainingDiagnostic(
            "wrong" if self.faults.get("wrong_replay") and len(features) == 3 else "scalar-native",
            value,
        )


def episode(role="train"):
    observations = tuple(tuple(float(j == i) for j in range(7)) for i in range(3))
    return DemonstrationEpisode(
        "bank-episode",
        "invoice",
        role,
        tuple(
            DemonstrationTransition(
                i,
                observations[i],
                i,
                float(i == 2),
                observations[i + 1] if i < 2 else (0.0,) * 7,
                i == 2,
                6,
            )
            for i in range(3)
        ),
        6,
    )


def setup(*, available=None, limits=None, promotable=False, extra_actions=()):
    wall = [0.0]
    timer = lambda: wall[0]
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4, max_wall_seconds=10), timer)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=6, max_wall_seconds=10),
        timer,
    )
    source = ScalarLearner()
    clock = LogicalClock()
    actor = (
        PromotableActor(
            source,
            version="a0",
            configuration=ServingConfiguration(2, 1),
            feature_digest=lambda a: "f",
            metadata={},
        )
        if promotable
        else None
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=wake,
        max_experiences=4,
        max_consolidations=2,
        actor=actor,
    )
    replay = CategoricalReplayAction(
        runtime,
        source,
        budget,
        (episode(),),
        ReplayRetentionBudget(1, 2048),
        states_equal=lambda a, b: a == b,
        definition="scalar-native",
    )
    gate = ServingPriorityGate(
        limits or SharingLimits(2, 6, 2), resource_available=available or (lambda: True)
    )
    envelope = AdaptationEnvelope(
        (
            AdaptationAction("wait"),
            AdaptationAction("wake", 1),
            AdaptationAction("replay", replay_examples=3),
        )
        + extra_actions,
        20,
        4,
        6,
    )
    wrapped = ResourceSharedRuntime(runtime, gate, action_envelope=envelope, replay_action=replay)
    return wrapped, runtime, replay, source, clock, wake, budget, gate, wall


def ready(runtime, clock, name="s"):
    now = clock.now()
    runtime.record_experience(
        Experience(
            name, "train", now, "a0", np.ones((2, 7)), "train", ExperiencePermissions(training=True)
        )
    )
    runtime.record_label(
        LabelArrival("label-" + name, name, "train", now + 1, "a0", np.zeros((2, 1)))
    )
    clock.advance_to(now + 1)


def test_interleaved_wake_replay_wake_preserves_full_state_and_distinct_permanent_charges():
    w, r, replay, source, c, wake, b, g, _ = setup()
    original = source.snapshot_state()
    ready(r, c, "s1")
    assert len(w.train_action("wake").training.updates) == 1
    p = w.train_action("replay")
    assert p.replay_receipt.event_id == "adaptation-replay-1" and p.training.updates == ()
    ready(r, c, "s2")
    assert len(w.train_action("wake").training.updates) == 1
    w.train_action("replay")
    state = r.candidate_snapshot().state
    assert state.completed_updates == 4 and state.training_examples == 10
    assert source.snapshot_state() == r.actor.snapshot_state().state == original
    assert [x.update_number for x in r.applied_updates] == [1, 2]
    assert len(r.applied_consolidations) == 2
    assert wake.updates_completed == 2 and wake.replay_examples_completed == 0
    assert b.updates_completed == 0 and b.replay_examples_completed == 6
    p = w.adaptation_snapshot().progress
    assert p.native_attempts == p.native_completions == g.snapshot().admitted_updates == 4
    assert p.wake_admissions == p.replay_admissions == p.replay_commits == 2
    assert p.replay_native_attempts == p.replay_native_completions == 2
    assert p.replay_examples_charged == 6 and not p.unreturned_work_unknown
    with pytest.raises(ValueError, match="replay envelope"):
        w.train_action("replay")
    assert source.calls == [2, 3, 2, 3] and not replay.stopped


def test_all_resource_deferrals_preserve_bank_labels_native_and_exposure(monkeypatch):
    available = [False]
    w, r, replay, source, c, wake, b, g, _ = setup(available=lambda: available[0])
    before = r.candidate_snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(source, "fork", lambda: pytest.fail("fork on deferred replay"))
        g.pause()
        assert w.train_action("replay").training.deferred_reason == "paused"
        g.resume()
        with g.serving():
            assert w.train_action("replay").training.deferred_reason == "serving_active"
        assert w.train_action("replay").training.deferred_reason == "resource_unavailable"
    assert r.candidate_snapshot() == before and r.applied_consolidations == ()
    assert source.calls == [] and b.replay_examples_completed == g.snapshot().admitted_updates == 0
    available[0] = True
    assert w.train_action("replay").replay_receipt is not None


@pytest.mark.parametrize("promotable", [False, True])
def test_actor_snapshot_contention_is_nonblocking_known_refusal_and_can_retry(promotable):
    w, r, replay, source, c, wake, b, g, _ = setup(promotable=promotable)
    with r.actor._payload_exclusive():
        with pytest.raises(PayloadOwnershipBusy):
            r.actor.snapshot_state_if_idle()
        with pytest.raises(PayloadOwnershipBusy):
            w.train_action("replay")
    assert not replay.stopped and source.calls == [] and b.replay_examples_completed == 0
    assert g.snapshot().admitted_updates == 1
    p = w.train_action("replay")
    assert p.replay_receipt.event_id == "adaptation-replay-2"
    assert g.snapshot().admitted_updates == 2 and p.progress.replay_native_attempts == 1


@pytest.mark.parametrize(
    "fault",
    [
        "candidate",
        "source",
        "actor",
        "bank",
        "replaced_bank",
        "budget_count",
        "budget_start",
        "budget_timer",
        "budget_limits",
    ],
)
def test_corrupt_or_replaced_predecessor_evidence_is_refused_before_more_native_work(fault):
    w, r, replay, source, c, wake, b, g, _ = setup()
    if fault == "candidate":
        r._candidate.state = replace(r._candidate.state, momentum=7.0)
    if fault == "source":
        source.state = replace(source.state, momentum=7.0)
    if fault == "actor":
        r.actor._learner.state = replace(r.actor._learner.state, momentum=7.0)
    if fault == "bank":
        object.__setattr__(replay._bank[0], "role", "probe_train")
    if fault == "replaced_bank":
        replay._bank = deepcopy(replay._bank)
    if fault == "budget_count":
        b.replay_examples_completed = 1
    if fault == "budget_start":
        b.started_at = -1.0
    if fault == "budget_timer":
        b.clock = lambda: 0.0
    if fault == "budget_limits":
        b.budget = replace(b.budget, max_replay_examples=99)
    with pytest.raises(ValueError):
        w.train_action("replay")
    assert source.calls == [] and replay.stopped
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")


def test_unobserved_wake_is_not_accepted_as_a_legitimate_counter_change():
    w, r, replay, source, c, wake, b, g, _ = setup()
    ready(r, c)
    r.train_ready()
    assert source.calls == [2] and wake.updates_completed == 1
    with pytest.raises(ValueError, match="without observed"):
        w.train_action("replay")
    assert source.calls == [2] and b.replay_examples_completed == 0


def test_changed_prior_wake_or_replay_receipt_is_not_accepted():
    for kind in ("wake", "replay"):
        w, r, replay, source, c, wake, b, g, _ = setup()
        if kind == "wake":
            ready(r, c)
            w.train_action("wake")
            key = next(iter(r._inbox._applied))
            r._inbox._applied[key] = replace(r._inbox._applied[key], event_id="replaced")
        else:
            w.train_action("replay")
            r._consolidations[0] = replace(r._consolidations[0], event_id="replaced")
        before = len(source.calls)
        with pytest.raises(ValueError):
            w.train_action("replay")
        assert len(source.calls) == before and replay.stopped


@pytest.mark.parametrize("fault,returns", [("raise_replay", 0), ("wrong_replay", 1)])
def test_failed_native_work_keeps_exposure_and_returns_separate_from_commit(fault, returns):
    w, r, replay, source, c, wake, b, g, _ = setup()
    source.faults[fault] = True
    with pytest.raises((RuntimeError, ValueError)):
        w.train_action("replay")
    p = w.adaptation_snapshot().progress
    assert p.replay_examples_charged == b.replay_examples_completed == 3
    assert p.replay_native_attempts == 1 and p.replay_native_completions == returns
    assert p.replay_commits == 0 and p.unreturned_work_unknown == (returns == 0)
    assert (
        r.candidate_snapshot().state.completed_updates == 0
        and source.snapshot_state().completed_updates == 0
    )
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")
    assert w.train_action("wait").training.updates == ()


def test_late_budget_stop_preserves_published_receipt_and_all_native_work(monkeypatch):
    w, r, replay, source, c, wake, b, g, wall = setup()
    before_final = wake.before_final

    def late():
        if r._consolidations:
            wall[0] = 10.0
        before_final()

    monkeypatch.setattr(wake, "before_final", late)
    with pytest.raises(ToyExecutionStopped):
        w.train_action("replay")
    p = w.adaptation_snapshot().progress
    assert p.replay_commits == p.replay_native_attempts == p.replay_native_completions == 1
    assert b.replay_examples_completed == 3 and not p.replay_publication_unknown
    assert (
        r.candidate_snapshot().state.completed_updates == 1 and len(r.applied_consolidations) == 1
    )
    assert (
        source.snapshot_state().completed_updates
        == r.actor.snapshot_state().state.completed_updates
        == 0
    )


def test_restore_failure_records_unknown_publication_and_stops_native_work(monkeypatch):
    w, r, replay, source, c, wake, b, g, _ = setup()

    def broken(state):
        r._candidate.state = replace(state, weight=999.0)
        raise RuntimeError("restore failed")

    monkeypatch.setattr(r._candidate, "restore_state", broken)
    with pytest.raises(RuntimeError, match="restore failed"):
        w.train_action("replay")
    p = w.adaptation_snapshot().progress
    assert p.replay_native_attempts == p.replay_native_completions == 1 and p.replay_commits == 0
    assert (
        p.replay_publication_unknown
        and p.unreturned_work_unknown
        and b.replay_examples_completed == 3
    )
    assert r._stopped and source.snapshot_state() == r.actor.snapshot_state().state
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_only_complete_train_bank_may_be_bound(role):
    w, r, replay, source, c, wake, b, g, _ = setup()
    fresh = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=6), lambda: 0.0
    )
    with pytest.raises(ValueError, match="TRAIN|training"):
        CategoricalReplayAction(
            r,
            source,
            fresh,
            (episode(role),),
            ReplayRetentionBudget(1, 2048),
            states_equal=lambda a, b: a == b,
            definition="scalar-native",
        )


def test_original_sharing_quota_and_executor_binding_remain_authoritative():
    w, r, replay, source, c, wake, b, g, _ = setup(limits=SharingLimits(1, 1, 1))
    w.train_action("replay")
    ready(r, c)
    assert w.train_action("wake").training.deferred_reason == "work_quota"
    assert w.train_action("replay").training.deferred_reason == "work_quota"
    assert source.calls == [3] and b.replay_examples_completed == 3 and wake.updates_completed == 0
    with pytest.raises(ValueError, match="already bound"):
        replay.bind(r, g, object())
    with pytest.raises(ValueError, match="admission"):
        replay.execute(object(), "fake", lambda s: None)


@pytest.mark.parametrize("alias", ["source", "actor", "candidate"])
def test_replay_refuses_aliased_fork_before_native_or_exposure(alias, monkeypatch):
    w, r, replay, source, c, wake, b, g, _ = setup()
    target = {"source": source, "actor": r.actor._learner, "candidate": r._candidate}[alias]
    monkeypatch.setattr(source, "fork", lambda: target)
    with pytest.raises(ValueError, match="independently owned"):
        w.train_action("replay")
    assert source.calls == [] and b.replay_examples_completed == 0
    assert r.candidate_snapshot().state.completed_updates == 0 and replay.stopped


def test_combined_wake_replay_remains_unsupported_before_resource_or_payload(monkeypatch):
    w, r, replay, source, c, wake, b, g, _ = setup(
        extra_actions=(AdaptationAction("combined", 1, 3),)
    )
    monkeypatch.setattr(
        g, "_resource", lambda: pytest.fail("resource probe for unsupported choice")
    )
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("fork for unsupported choice"))
    with pytest.raises(ValueError, match="unsupported"):
        w.train_action("combined")
    assert source.calls == [] and b.replay_examples_completed == g.snapshot().admitted_updates == 0


def test_native_wake_replay_wake_example_preserves_full_original_state(tmp_path):
    from scripts.bounded_replay_action_example import run_example

    r = run_example(Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"))
    if os.environ.get("CATEGORICAL_REPLAY_NATIVE_REPORT"):
        Path(os.environ["CATEGORICAL_REPLAY_NATIVE_REPORT"]).write_text(
            json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    assert r["complete"], r.get("error")
    assert (
        r["source_unchanged"]
        and r["actor_unchanged"]
        and r["numpy_rng_unchanged"]
        and r["torch_rng_unchanged"]
    )
    assert r["native_work"] == dict(
        attempted_updates=3, attempted_rows=15, returned_updates=3, unreturned_work_unknown=False
    )
    assert (
        r["candidate"]["state"]["completed_updates"] == 99
        and r["candidate"]["state"]["training_examples"] == 591
    )
    assert (
        r["wake_updates"] == 2
        and r["replay_examples"] == 3
        and r["sharing"]["admitted_updates"] == 3
    )
    assert r["final_reads"] == r["probe_updates"] == r["promotions"] == 0
