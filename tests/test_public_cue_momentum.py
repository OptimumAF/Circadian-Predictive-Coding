"""Literal two-head zero-loss control dispatch; no native SGD or physics."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_public_cue_reward as port
from src.app.coupled_attribution import _same_value
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from test_public_cue_recurrent_learner import ScriptedNative, configure_scripted_native, policy
from test_public_cue_reward import (
    REWARD_CALLS,
    reward_call,
    packet as reward_packet,
    trajectory,
)

ZERO_CALLS: list[Any] = []


def zero_call(self, record):
    ZERO_CALLS.append((deepcopy(record), deepcopy(self.state)))
    self.state = replace(
        self.state,
        parameters={"marker": self.state.parameters["marker"] + 2},
        optimizer_state={"momentum": self.state.optimizer_state["momentum"] + 3},
        gradients={"gradient": np.zeros(1)},
        completed_updates=self.state.completed_updates + 1,
        training_examples=self.state.training_examples + len(record.transitions),
    )
    return TrainingDiagnostic(port.TORCH_RECURRENT_ZERO_LOSS_ID, 0.0)


@pytest.fixture
def owner(monkeypatch):
    configure_scripted_native(monkeypatch)
    REWARD_CALLS.clear()
    ZERO_CALLS.clear()
    monkeypatch.setattr(ScriptedNative, "train_reward_episode", reward_call, raising=False)
    monkeypatch.setattr(ScriptedNative, "train_zero_loss_episode", zero_call, raising=False)
    return port.PublicCueRewardLearner(policy())


def control(cue="blue"):
    original = trajectory(cue)
    # Nonzero observed rewards and nonuniform recorded probabilities are kept
    # even though the zero-loss objective deliberately ignores their labels.
    return port.MomentumOnlyUpdate(
        replace(
            original,
            transitions=tuple(replace(t, reward=0.5) for t in original.transitions),
            behavior_probabilities=((0.4, 0.04, 0.4, 0.04, 0.04, 0.04, 0.04),) * 2,
        )
    )


def execute(owner, update):
    return owner.train_batch(update, update.episode.episode_id)


@pytest.mark.parametrize("cue,selected", [("blue", 0), ("red", 1)])
def test_zero_control_routes_existing_native_method_and_preserves_original_evidence(
    owner, cue, selected
):
    cold = owner.snapshot_state()
    warm = replace(
        cold,
        heads=(
            replace(cold.heads[0], completed_updates=64, training_examples=576),
            replace(cold.heads[1], completed_updates=64, training_examples=576),
        ),
    )
    owner.restore_state(warm)
    update = control(cue)
    original = deepcopy(update)
    result = execute(owner, update)
    assert result == TrainingDiagnostic(port.TORCH_RECURRENT_ZERO_LOSS_ID, 0.0)
    assert not REWARD_CALLS and len(ZERO_CALLS) == 1
    observed, native_before = ZERO_CALLS[0]
    assert observed == update.episode and observed is not update.episode
    assert _same_value(native_before, warm.heads[selected])
    state = owner.snapshot_state()
    assert (state.completed_updates, state.training_examples) == (129, 1154)
    assert _same_value(state.heads[1 - selected], warm.heads[1 - selected])
    assert state.heads[selected].optimizer_state["momentum"][0] == 3
    assert state.heads[selected].gradients["gradient"][0] == 0
    assert _same_value(update, original)


@pytest.mark.parametrize("fault", ["role", "extra", "release", "geometry"])
def test_bad_control_packet_is_refused_before_native_staging(owner, monkeypatch, fault):
    update = control()
    target = update.episode.episode_id
    if fault == "role":
        object.__setattr__(update.episode, "role", "final_test")
    elif fault == "extra":
        object.__setattr__(update, "behavior_state", owner.snapshot_state())
    elif fault == "release":
        target += "-other"
    else:
        steps = tuple(
            replace(t, observation=t.observation[:16], next_observation=t.next_observation[:16])
            for t in update.episode.transitions
        )
        object.__setattr__(update, "episode", replace(update.episode, transitions=steps))
    monkeypatch.setattr(port.public, "_stage", lambda *a: pytest.fail("native stage reached"))
    with pytest.raises(ValueError):
        owner.train_batch(update, target)
    assert not ZERO_CALLS and owner.snapshot_state().failed_work is None


@pytest.mark.parametrize("fault", ["lost", "definition", "value", "packet", "inactive"])
def test_control_failures_keep_permanent_paid_prefix_and_refuse_retry(owner, monkeypatch, fault):
    update = control()
    before = owner.snapshot_state()
    original = getattr(ScriptedNative, "train_zero_loss_episode")

    def changed(self, record):
        if fault == "lost":
            raise RuntimeError("lost zero-loss return")
        result = original(self, record)
        if fault == "definition":
            return TrainingDiagnostic(port.TORCH_RECURRENT_REWARD_ID, 0.0)
        if fault == "value":
            return TrainingDiagnostic(port.TORCH_RECURRENT_ZERO_LOSS_ID, 0.25)
        if fault == "packet":
            object.__setattr__(update, "episode", deepcopy(update.episode))
        if fault == "inactive":
            head = owner._learner._state.heads[1]
            object.__setattr__(head, "optimizer_state", deepcopy(head.optimizer_state))
        return result

    monkeypatch.setattr(ScriptedNative, "train_zero_loss_episode", changed)
    with pytest.raises(TrainingUpdateFailed):
        execute(owner, update)
    failed = owner.snapshot_state()
    assert _same_value(failed.heads, before.heads) and failed.completed_updates == 0
    assert failed.failed_work.native_calls_attempted == 1
    assert failed.failed_work.native_calls_returned == int(fault != "lost")
    with pytest.raises(TrainingUpdateFailed):
        execute(owner, control())
    with pytest.raises(ValueError):
        owner.restore_state(before)


@pytest.mark.parametrize("fault", ["live_value", "diagnostic"])
def test_final_snapshot_cannot_hide_control_state_or_diagnostic_mutation(owner, monkeypatch, fault):
    original = port.public.copy_snapshot
    native_zero = getattr(ScriptedNative, "train_zero_loss_episode")
    returned = []
    armed = fired = False

    def observe(self, record):
        result = native_zero(self, record)
        returned.append(result)
        return result

    def capture(state):
        nonlocal armed, fired
        saved = original(state)
        if state.completed_updates == 1:
            armed = True
        elif armed and not fired and state is owner._learner._state:
            fired = True
            if fault == "live_value":
                state.heads[1].parameters["marker"][0] += 1
            else:
                object.__setattr__(returned[-1], "value", 1.0)
        return saved

    monkeypatch.setattr(ScriptedNative, "train_zero_loss_episode", observe)
    monkeypatch.setattr(port.public, "copy_snapshot", capture)
    with pytest.raises(TrainingUpdateFailed):
        execute(owner, control())
    failed = owner.snapshot_state()
    assert fired and failed.failed_work.native_calls_returned == 1
    assert failed.completed_updates == 0


def test_runtime_delayed_mixed_objectives_keep_source_actor_and_budget_birth(owner):
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock

    initial = owner.snapshot_state()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    birth, progress = budget.started_at, budget.progress
    runtime = ActorShadowRuntime(
        owner,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=2,
        max_consolidations=0,
    )
    for index, update in enumerate((reward_packet(owner), control("red"))):
        identity = update.episode.episode_id
        runtime.record_experience(
            Experience(
                identity,
                identity,
                index,
                "a0",
                update,
                "train",
                ExperiencePermissions(training=True),
            )
        )
        assert not runtime.train_ready(max_updates=1)
        clock.advance_to(index + 1)
        runtime.record_label(
            LabelArrival(f"release-{index}", identity, identity, index + 1, "a0", identity, "train")
        )
        assert len(runtime.train_ready(max_updates=1)) == 1
    saved = runtime.candidate_snapshot().state
    assert (saved.completed_updates, saved.training_examples) == (2, 4)
    assert [h.completed_updates for h in saved.heads] == [1, 1]
    assert len(REWARD_CALLS) == len(ZERO_CALLS) == 1
    assert budget.updates_completed == 2 and budget.replay_examples_completed == 0
    assert budget.started_at == birth and budget.progress is progress
    assert _same_value(owner.snapshot_state(), initial)
    assert _same_value(runtime.actor.snapshot_state().state, initial)
