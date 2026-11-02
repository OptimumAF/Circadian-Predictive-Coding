"""Literal native ports; these tests do not establish learned reward usefulness."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_public_cue_reward as port
from src.app.coupled_attribution import _same_value
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.policy_gradient_episode import PolicyGradientEpisode
from test_public_cue_recurrent_learner import (
    ScriptedNative,
    configure_scripted_native,
    data,
    episode,
    policy,
)

REWARD_CALLS: list[Any] = []


def reward_call(self, trajectory, behavior, *, discount):
    assert _same_value(self.state, behavior)
    REWARD_CALLS.append((deepcopy(trajectory), deepcopy(behavior), discount))
    self.state = replace(
        self.state,
        parameters={"marker": self.state.parameters["marker"] + 3},
        optimizer_state={"momentum": self.state.optimizer_state["momentum"] + 5},
        gradients={"gradient": self.state.gradients["gradient"] + 7},
        completed_updates=self.state.completed_updates + 1,
        training_examples=self.state.training_examples + len(trajectory.transitions),
    )
    return TrainingDiagnostic(port.TORCH_RECURRENT_REWARD_ID, 0.25)


@pytest.fixture
def owner(monkeypatch):
    configure_scripted_native(monkeypatch)
    REWARD_CALLS.clear()
    monkeypatch.setattr(ScriptedNative, "train_reward_episode", reward_call, raising=False)
    return port.PublicCueRewardLearner(policy())


def trajectory(cue="blue", role="train"):
    demo = episode(cue, role)
    return PolicyGradientEpisode(
        demo.episode_id,
        demo.context,
        demo.role,
        demo.transitions,
        tuple((1 / 7,) * 7 for _ in demo.transitions),
        "terminal",
        7,
    )


def packet(owner, cue="blue", role="train"):
    return port.PublicCueRewardUpdate(trajectory(cue, role), owner.snapshot_state())


def execute(owner, update):
    return owner.train_batch(update, update.episode.episode_id)


@pytest.mark.parametrize("cue,selected", [("blue", 0), ("red", 1)])
def test_reward_preserves_full_warm_state_and_updates_only_public_cue_head(owner, cue, selected):
    cold = owner.snapshot_state()
    warm = replace(
        cold,
        heads=(
            replace(cold.heads[0], completed_updates=64, training_examples=576),
            replace(cold.heads[1], completed_updates=64, training_examples=576),
        ),
    )
    owner.restore_state(warm)
    update = packet(owner, cue)
    original = deepcopy(update)
    result = execute(owner, update)
    saved = owner.snapshot_state()
    assert result == TrainingDiagnostic(port.TORCH_RECURRENT_REWARD_ID, 0.25)
    observed, behavior, discount = REWARD_CALLS[-1]
    assert observed == update.episode and observed is not update.episode
    assert discount == 1.0 and type(discount) is float
    assert _same_value(behavior, warm.heads[selected])
    assert _same_value(saved.heads[1 - selected], warm.heads[1 - selected])
    assert (saved.completed_updates, saved.training_examples) == (129, 1154)
    assert [h.completed_updates for h in saved.heads] == ([65, 64] if selected == 0 else [64, 65])
    assert saved.heads[selected].optimizer_state["momentum"][0] == 5
    assert saved.heads[selected].gradients["gradient"][0] == 7
    assert _same_value(update, original) and _same_value(warm, update.behavior_state)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_reserved_roles_never_reach_native_reward(owner, role):
    with pytest.raises(ValueError, match="TRAIN"):
        execute(owner, packet(owner, role=role))
    assert not REWARD_CALLS and owner.snapshot_state().failed_work is None


@pytest.mark.parametrize("fault", ["target", "extra", "foreign", "inactive", "gradient", "policy"])
def test_invalid_release_or_full_behavior_is_refused_before_native_stage(owner, monkeypatch, fault):
    update = packet(owner)
    target = update.episode.episode_id
    if fault == "target":
        target += "-other"
    elif fault == "extra":
        object.__setattr__(update, "hidden", True)
    elif fault == "foreign":
        update = object()
    elif fault == "inactive":
        update.behavior_state.heads[1].optimizer_state["momentum"][0] += 1
    elif fault == "gradient":
        update.behavior_state.heads[0].gradients["gradient"][0] += 1
    else:
        object.__setattr__(update.behavior_state, "policy", port.public.PublicCueRecurrentPolicy())
    monkeypatch.setattr(port.public, "_stage", lambda *args: pytest.fail("native stage reached"))
    with pytest.raises(ValueError):
        owner.train_batch(update, target)
    assert owner.completed_updates == 0 and owner.snapshot_state().failed_work is None


@pytest.mark.parametrize("fault", ["switch", "missing", "width"])
def test_bad_public_cue_geometry_is_refused_before_staging(owner, monkeypatch, fault):
    record = trajectory()
    first, second = record.transitions
    if fault == "switch":
        changed = tuple(float(v) for v in data("red")[0][0])
        first = replace(first, next_observation=changed)
        second = replace(second, observation=changed)
    elif fault == "missing":
        first = replace(first, observation=(-1.0,) * 148)
    else:
        first = replace(
            first, observation=first.observation[:16], next_observation=first.next_observation[:16]
        )
        second = replace(
            second,
            observation=second.observation[:16],
            next_observation=second.next_observation[:16],
        )
        first = replace(first, next_observation=second.observation)
    record = replace(record, transitions=(first, second))
    update = port.PublicCueRewardUpdate(record, owner.snapshot_state())
    monkeypatch.setattr(port.public, "_stage", lambda *args: pytest.fail("native stage reached"))
    with pytest.raises(ValueError):
        execute(owner, update)


def test_fork_restore_and_existing_session_are_detached_and_do_not_reset_heads(owner):
    initial = owner.snapshot_state()
    child = owner.fork()
    execute(child, packet(child, "red"))
    saved = child.snapshot_state()
    restored = owner.fork()
    restored.restore_state(saved)
    a, b = child.start_episode(), restored.start_episode()
    assert _same_value(a.predict(data("red")[0][:1]), b.predict(data("red")[0][:1]))
    assert a.completed_steps == b.completed_steps == 1
    assert _same_value(child.snapshot_state(), saved)
    assert _same_value(restored.snapshot_state(), saved)
    assert _same_value(owner.snapshot_state(), initial)
    saved.heads[0].parameters["marker"][0] = 99
    assert child.snapshot_state().heads[0].parameters["marker"][0] == 0


@pytest.mark.parametrize("fault", ["stage", "lost", "diagnostic", "clock", "input", "owner"])
def test_paid_or_unknown_reward_failures_block_retry_and_state_erasure(owner, monkeypatch, fault):
    update = packet(owner)
    before = owner.snapshot_state()
    if fault == "stage":

        def lost_init(self, policy):
            raise RuntimeError("lost staged constructor")

        monkeypatch.setattr(ScriptedNative, "__init__", lost_init)
    else:
        original = getattr(ScriptedNative, "train_reward_episode")

        def changed(self, record, behavior, *, discount):
            if fault == "lost":
                raise RuntimeError("lost reward return")
            result = original(self, record, behavior, discount=discount)
            if fault == "diagnostic":
                return TrainingDiagnostic(port.public.native.TORCH_RECURRENT_CE_ID, 0.25)
            if fault == "clock":
                self.state = replace(self.state, training_examples=99)
            if fault == "input":
                object.__setattr__(update, "episode", deepcopy(update.episode))
            if fault == "owner":
                owner._learner = owner._learner.fork()
            return result

        monkeypatch.setattr(ScriptedNative, "train_reward_episode", changed)
    with pytest.raises(TrainingUpdateFailed):
        execute(owner, update)
    failed = owner.snapshot_state()
    assert _same_value(failed.heads, before.heads)
    assert failed.failed_work is not None
    work = failed.failed_work
    assert work.native_stages_attempted == 1
    assert work.native_stages_returned == int(fault != "stage")
    assert work.native_calls_attempted == int(fault != "stage")
    assert work.native_calls_returned == int(fault not in ("stage", "lost"))
    with pytest.raises(TrainingUpdateFailed):
        execute(owner, packet(owner))
    with pytest.raises(ValueError):
        owner.restore_state(before)


@pytest.mark.parametrize(
    "fault",
    [
        "state",
        "inactive_head",
        "packet_state",
        "parameter_leaf",
        "gradient_dict",
        "optimizer_nested",
        "packet_leaf",
    ],
)
def test_late_state_capture_refuses_equal_owner_replacement_and_preserves_paid_work(
    owner, monkeypatch, fault
):
    if fault == "optimizer_nested":
        owner._learner._state.heads[1].optimizer_state["state"] = {
            0: {"momentum_buffer": np.zeros(1)}
        }
    update = packet(owner)
    original = port.public.copy_snapshot
    fired = False

    def capture(state):
        nonlocal fired
        saved = original(state)
        if state.completed_updates == 1 and not fired:
            fired = True
            if fault == "state":
                owner._learner._state = deepcopy(owner._learner._state)
            elif fault == "inactive_head":
                current = owner._learner._state
                object.__setattr__(current, "heads", (current.heads[0], deepcopy(current.heads[1])))
            elif fault == "packet_state":
                object.__setattr__(update, "behavior_state", deepcopy(update.behavior_state))
            elif fault == "parameter_leaf":
                head = owner._learner._state.heads[1]
                head.parameters["marker"] = head.parameters["marker"].copy()
            elif fault == "gradient_dict":
                head = owner._learner._state.heads[1]
                object.__setattr__(head, "gradients", deepcopy(head.gradients))
            elif fault == "optimizer_nested":
                state = owner._learner._state.heads[1].optimizer_state["state"]
                state[0] = deepcopy(state[0])
            else:
                head = update.behavior_state.heads[1]
                head.parameters["marker"] = head.parameters["marker"].copy()
        return saved

    monkeypatch.setattr(port.public, "copy_snapshot", capture)
    with pytest.raises(TrainingUpdateFailed):
        execute(owner, update)
    work = owner.snapshot_state().failed_work
    assert fired and work.native_calls_returned == 1 and work.native_rows_returned == 2
    assert owner.completed_updates == 0


@pytest.mark.parametrize("fault", ["gradient", "diagnostic"])
def test_last_snapshot_cannot_hide_in_place_change(owner, monkeypatch, fault):
    update = packet(owner)
    original = port.public.copy_snapshot
    original_reward = getattr(ScriptedNative, "train_reward_episode")
    returned = []
    armed = fired = False

    def observe(self, record, behavior, *, discount):
        result = original_reward(self, record, behavior, discount=discount)
        returned.append(result)
        return result

    def capture(state):
        nonlocal armed, fired
        saved = original(state)
        if state.completed_updates == 1:
            armed = True
        elif armed and not fired and state is owner._learner._state:
            fired = True
            if fault == "gradient":
                state.heads[1].gradients["gradient"][0] += 99
            else:
                object.__setattr__(returned[-1], "value", 99.0)
        return saved

    monkeypatch.setattr(ScriptedNative, "train_reward_episode", observe)
    monkeypatch.setattr(port.public, "copy_snapshot", capture)
    with pytest.raises(TrainingUpdateFailed):
        execute(owner, update)
    state = owner.snapshot_state()
    assert fired and state.failed_work.native_calls_returned == 1
    assert state.completed_updates == 0 and state.heads[1].gradients["gradient"][0] == 0


def test_real_collector_and_delayed_runtime_use_two_head_reward_without_source_updates(
    owner, monkeypatch
):
    from src.adapters import minigrid_reward_collection as collection
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
    from src.core.minigrid_key_door import predefined_layouts
    import test_minigrid_reward_collection as literal

    def frame(step=0):
        return tuple(float(v) for v in data()[0][min(step, 1)])

    monkeypatch.setattr(literal, "frame", frame)
    monkeypatch.setattr(literal.Environment, "work", {}, raising=False)
    monkeypatch.setattr(collection, "MiniGridKeyDoor", literal.Environment)
    monkeypatch.setattr(collection, "_same_state", _same_value)
    initial = owner.snapshot_state()
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=1, max_wall_seconds=30), lambda: 0.0
    )
    runtime = ActorShadowRuntime(
        owner,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    report = collection.collect_minigrid_reward_episode(
        owner.fork(),
        predefined_layouts()[0],
        "blue",
        episode_id="joint-reward",
        sampler=np.random.Generator(np.random.PCG64(29)),
        check_boundary=budget.before_final,
        clock=lambda: 0.0,
    )
    assert report["status"] == "complete" and report["state_unchanged"]
    update = port.PublicCueRewardUpdate(report["episode"], report["behavior_state"])
    runtime.record_experience(
        Experience(
            "joint-reward",
            "joint-reward",
            0,
            "a0",
            update,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    assert not runtime.train_ready(max_updates=1)
    clock.advance_to(1)
    runtime.record_label(
        LabelArrival("release", "joint-reward", "joint-reward", 1, "a0", "joint-reward", "train")
    )
    receipts = runtime.train_ready(max_updates=1)
    assert len(receipts) == 1 and budget.updates_completed == 1
    state = runtime.candidate_snapshot().state
    assert (state.completed_updates, state.training_examples) == (1, 2)
    assert _same_value(state.heads[1], initial.heads[1])
    assert _same_value(owner.snapshot_state(), initial)
    assert _same_value(runtime.actor.snapshot_state().state, initial)
    assert report["environment"]["closed"] and report["work"]["sampling_draws"] == 2
