"""Authentic D9 bank admission and real conventional replay/wake composition."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.catalog_contextual_controller import ContextualArmCatalog
from src.core.context_perceptron import ContextPerceptronPolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.demonstration_episode import (
    EpisodeReplaySchema,
    encode_episode,
    decode_episode,
    episode_arrays,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits
from src.core.ticket_workflow import ACTION_IDS, CATALOG


def compiler():
    from scripts.context_catalog_replay import compile_context_bank

    return compile_context_bank


def prototype(concepts=("invoice", "password")):
    rows = [
        {"policy_v1", "sealed"},
        {"policy_v1", "ready", concepts[0]},
        {"policy_v1", "staged"},
        {"policy_v1", "sealed"},
        {"policy_v1", "ready", concepts[1]},
        {"policy_v1", "staged"},
    ]
    x = np.array([[float(word in row) for word in CONTEXT_FEATURES] for row in rows])
    actions = ("open", "route", "commit", "open", "inspect", "commit")
    y = np.array([[float(ACTION_IDS.index(a))] for a in actions])
    return AttributionBatch(
        "train", "old-v1", tuple(f"update-1-{i}" for i in range(6)), x, y, num_classes=6
    )


def spec():
    return DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)


def test_should_compile_complete_detached_train_traces_with_exact_codec_roundtrip():
    old = prototype()
    bank = compiler()(spec(), old)
    assert len(bank) == 2
    assert [e.context for e in bank] == ["v1-invoice", "v1-password"]
    assert len({e.episode_id for e in bank}) == 2
    for i, episode in enumerate(bank):
        assert episode.role == "train" and episode.num_classes == 6
        x, y = episode_arrays(episode)
        assert np.array_equal(x, old.features[3 * i : 3 * (i + 1)])
        assert np.array_equal(y, old.targets[3 * i : 3 * (i + 1)])
        schema = EpisodeReplaySchema(9, 3, 6)
        encoded, dummy = encode_episode(episode, schema)
        assert decode_episode(encoded, dummy, schema) == episode
        assert episode.transitions[-1].next_observation == (0.0,) * 9
    old.features[:] = 0
    old.targets[:] = 0
    assert episode_arrays(bank[0])[0][0].sum() == 2
    assert bank[0].transitions[1].action == ACTION_IDS.index("route")


@pytest.mark.parametrize(
    "fault",
    [
        "final_test",
        "probe_train",
        "inner_guard",
        "context",
        "ids",
        "shape",
        "classes",
        "nan",
        "dtype",
        "target",
        "context_bit",
        "order",
    ],
)
def test_should_refuse_other_roles_schema_or_replaced_teacher_truth(fault):
    old = prototype()
    if fault in ("final_test", "probe_train", "inner_guard"):
        old = replace(old, role=fault)
    elif fault == "context":
        old = replace(old, context="new-v1")
    elif fault == "ids":
        old = replace(old, row_ids=tuple(f"other-{i}" for i in range(6)))
    elif fault == "shape":
        old = replace(old, features=old.features[:3], targets=old.targets[:3])
    elif fault == "classes":
        old = replace(old, num_classes=5)
    elif fault == "nan":
        old.features[0, 0] = np.nan
    elif fault == "dtype":
        old = replace(old, features=old.features.astype(np.float32))
    elif fault == "target":
        old.targets[1, 0] = ACTION_IDS.index("defer")
    elif fault == "context_bit":
        old.features[:, CONTEXT_FEATURES.index("policy_v1")] = 0
        old.features[:, CONTEXT_FEATURES.index("policy_v2")] = 1
    else:
        old.features[[1, 4]] = old.features[[4, 1]]
    with pytest.raises(ValueError):
        compiler()(spec(), old)


def test_should_refuse_a_different_schema_before_teacher_work(monkeypatch):
    import scripts.context_catalog_replay as module

    monkeypatch.setattr(module, "_ticket_trace", lambda *args: pytest.fail("teacher called"))
    with pytest.raises(ValueError):
        module.compile_context_bank(
            DecisionFeatureSpec(CONTEXT_FEATURES[:-1], CATALOG), prototype()
        )


@pytest.mark.parametrize("foreign", [True, False])
def test_should_refuse_unsupported_learner_or_head_routing_before_teacher_work(
    monkeypatch, foreign
):
    import scripts.context_catalog_replay as module

    learner = (
        object() if foreign else ContextPerceptronLearner(ContextPerceptronPolicy(9, 6, (0, 1)))
    )
    monkeypatch.setattr(module, "_ticket_trace", lambda *args: pytest.fail("teacher called"))
    with pytest.raises(ValueError):
        module.compose_context_replay(None, learner, None, spec(), prototype())


class CountedContext(ContextPerceptronLearner):
    fail_replay = False

    def fork(self):
        child = CountedContext(self.policy)
        child.restore_state(self.snapshot_state())
        child.fail_replay = self.fail_replay
        return child

    def train_batch(self, features, targets):
        result = super().train_batch(features, targets)
        if self.fail_replay and len(features) == 6:
            raise RuntimeError("lost paired-head replay return")
        return result


def setup():
    from scripts.context_catalog_replay import compose_context_replay

    old = prototype()
    source = CountedContext(ContextPerceptronPolicy(9, 6, (4, 5)))
    source.train_batch(old.features, old.targets)
    clock = LogicalClock()
    timer = lambda: 0.0
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2, max_wall_seconds=10), timer)
    replay_budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=12, max_wall_seconds=10),
        timer,
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="fixture-1",
        candidate_version="private-1",
        clock=clock,
        budget=wake,
        max_experiences=2,
        max_consolidations=2,
    )
    replay = compose_context_replay(runtime, source, replay_budget, spec(), old)
    catalog = ContextualArmCatalog(
        AdaptationEnvelope(
            (
                AdaptationAction("wait"),
                AdaptationAction("wake1", 1),
                AdaptationAction("replay6-wake1", 1, 6),
            ),
            8,
            2,
            12,
        )
    )
    gate = ServingPriorityGate(SharingLimits(1, 4, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(
        runtime, gate, action_envelope=catalog.envelope, replay_action=replay
    )
    return shared, runtime, replay, source, clock, wake, replay_budget, old


def ready(runtime, clock, name, batch):
    now = clock.now()
    runtime.record_experience(
        Experience(
            name,
            batch.context,
            now,
            "fixture-1",
            batch.features.copy(),
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(
        LabelArrival(
            "label-" + name, name, batch.context, now + 1, "fixture-1", batch.targets.copy()
        )
    )
    clock.advance_to(now + 1)


def test_should_execute_full_catalog_replay_then_wake_with_original_independent_charges():
    shared, runtime, replay, source, clock, wake, replay_budget, old = setup()
    original = source.snapshot_state()
    actor = runtime.actor.snapshot_state()
    limits = (id(wake.budget), id(replay_budget.budget), wake.started_at, replay_budget.started_at)
    shared.train_action("wait")
    assert replay_budget.replay_examples_completed == wake.updates_completed == 0
    ready(runtime, clock, "s1", old)
    result = shared.train_action("replay6-wake1")
    assert result.replay_receipt is not None and len(result.training.updates) == 1
    assert len(runtime.applied_consolidations) == 1 and len(runtime.applied_updates) == 1
    assert wake.updates_completed == 1 and replay_budget.updates_completed == 0
    assert replay_budget.replay_examples_completed == 6 and wake.replay_examples_completed == 0
    ready(runtime, clock, "s2", old)
    shared.train_action("wake1")
    state = runtime.candidate_snapshot().state
    assert state.completed_updates == original.completed_updates + 3
    assert state.training_examples == original.training_examples + 18
    assert state.heads[1] == original.heads[1]
    assert source.snapshot_state() == original and runtime.actor.snapshot_state() == actor
    assert limits == (
        id(wake.budget),
        id(replay_budget.budget),
        wake.started_at,
        replay_budget.started_at,
    )
    report = replay.report()
    assert report["retained_episodes"] == 2 and replay.rows == 6
    # Two encoded episode rows plus both zero dummy targets, not just features.
    assert report["retained_numeric_bytes"] == 2608
    assert replay._replay.session is replay_budget and replay._wake.session is wake
    assert (
        report["native_attempts"] == report["native_completions"] == report["committed_events"] == 1
    )
    assert report["replay_examples_charged"] == 6


@pytest.mark.parametrize("fault", ["bank", "receipt"])
def test_should_refuse_replaced_prior_bank_or_success_before_more_native_work(fault):
    shared, runtime, replay, source, clock, wake, replay_budget, old = setup()
    ready(runtime, clock, "s1", old)
    shared.train_action("replay6-wake1")
    if fault == "bank":
        replay._bank = deepcopy(replay._bank)
    else:
        runtime._consolidations[0] = replace(runtime._consolidations[0])
    before = runtime.candidate_snapshot()
    counts = (wake.updates_completed, replay_budget.replay_examples_completed)
    ready(runtime, clock, "s2", old)
    with pytest.raises(ValueError):
        shared.train_action("replay6-wake1")
    assert runtime.candidate_snapshot() == before
    assert counts == (wake.updates_completed, replay_budget.replay_examples_completed)


def test_should_preserve_failed_replay_exposure_and_refuse_later_native_work():
    shared, runtime, replay, source, clock, wake, replay_budget, old = setup()
    source.fail_replay = True
    before = runtime.candidate_snapshot()
    ready(runtime, clock, "s1", old)
    with pytest.raises(RuntimeError, match="lost paired-head replay return"):
        shared.train_action("replay6-wake1")
    assert replay_budget.replay_examples_completed == 6 and wake.updates_completed == 0
    after = runtime.candidate_snapshot()
    assert after.state == before.state and runtime.applied_consolidations == ()
    assert after.consolidation_attempts == before.consolidation_attempts + 1
    assert source.snapshot_state() == before.state
    assert replay.report()["stopped"] and replay.report()["native_completions"] == 0
    with pytest.raises(ValueError):
        shared.train_action("wake1")
