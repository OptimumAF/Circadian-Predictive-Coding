"""Actual joint stream and reward/control runner with literal native/physics ports."""

from copy import deepcopy
from dataclasses import replace
import json
import sys

import pytest

from scripts import recurrent_joint_example as example
from src.adapters import minigrid_joint_warm as handoff
from src.adapters import minigrid_reward_pilot as pilot
from src.adapters import minigrid_momentum_control as control
from src.adapters import minigrid_warm_readouts as readouts
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_reward_collection as collection
from src.adapters.minigrid_public_cue_reward import PublicCueRewardLearner
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_recurrent_joint_example import app as app, episodes
from test_public_cue_recurrent_learner import ScriptedNative, data
from test_public_cue_reward import REWARD_CALLS, reward_call, trajectory
from test_public_cue_momentum import ZERO_CALLS, zero_call
import test_minigrid_reward_collection as literal


class Environment(literal.Environment):
    def __init__(self, layout, cue, *args, **kwargs):
        # Only one literal environment runs at a time in the real collector/kernel.
        literal.frame = lambda step=0: tuple(float(v) for v in data(cue)[0][min(step, 1)])
        super().__init__(layout, cue, *args, **kwargs)

    @property
    def legal_actions(self):
        return () if self.terminal else tuple(range(7))


@pytest.fixture
def ready(app, monkeypatch):
    REWARD_CALLS.clear()
    ZERO_CALLS.clear()
    monkeypatch.setattr(ScriptedNative, "train_reward_episode", reward_call, raising=False)
    monkeypatch.setattr(ScriptedNative, "train_zero_loss_episode", zero_call, raising=False)
    monkeypatch.setattr(literal, "frame", literal.frame)
    monkeypatch.setattr(Environment, "work", {}, raising=False)
    for module in (pilot, control, readouts, collection):
        monkeypatch.setattr(module, "_same_state", _same_value)
    monkeypatch.setattr(pilot, "require_torch", lambda: None)
    monkeypatch.setattr(collection, "MiniGridKeyDoor", Environment)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", Environment)
    work = {}
    original = example.AttributionStream

    def stream(*a, **k):
        value = original(*a, **k)
        work["stream"] = value
        return value

    monkeypatch.setattr(example, "AttributionStream", stream)
    return work


def execute():
    return example.run_example(reward_training=True)


def test_live_joint_stream_handoff_reward_control_and_all_fixed_readouts(ready):
    result = execute()
    assert result["status"] == "complete", (
        result["reason"],
        [
            (r.get("status"), r.get("reason"))
            for r in result.get("adaptation", {}).get("train_diagnostics", {}).get("readouts", [])
        ],
    )
    assert result["readouts"] == []  # No duplicate original0/128 scores/games.
    assert len(result["training"]["applied_updates"]) == 128
    current = result["adaptation"]
    assert current["status"] == "complete", current["reason"]
    assert len(REWARD_CALLS) == len(ZERO_CALLS) == 8
    assert current["whole_committed_native_updates"] == 144
    assert current["whole_training_rows"] == 1152 + 32
    assert current["warm_start"]["work"]["charged_updates"] == 0
    assert current["source_unchanged"] and current["actor_unchanged"]
    assert current["wake_updates"] == current["observed_budget_updates"] == 8
    observed = current["train_diagnostics"]["readouts"]
    assert [row["updates"] for row in observed] == [128, 136, 136]
    assert all(len(row["scores"]) == len(row["episodes"]) == 6 for row in observed)
    assert all(
        e["source_role"] == "train" and e["environment"]["closed"]
        for row in observed
        for e in row["episodes"]
    )
    assert all(old[0] == zero[0] for old, zero in zip(REWARD_CALLS, ZERO_CALLS, strict=True))
    stream = ready["stream"]
    assert stream._learner.completed_updates == 0
    assert stream._runtime.actor.snapshot_state().state.completed_updates == 0
    benchmark = stream._runtime.candidate_snapshot().state
    assert (benchmark.completed_updates, benchmark.training_examples) == (128, 1152)
    assert all(h.completed_updates == 64 for h in benchmark.heads)
    assert len(Environment.work["environments"]) == 26
    assert all(e.closed for e in Environment.work["environments"])


def test_lost_third_reward_keeps_paid_prefix_and_refuses_control_or_readouts(ready, monkeypatch):
    original = getattr(ScriptedNative, "train_reward_episode")
    calls = 0

    def lost(self, episode, behavior, *, discount):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("lost third reward return")
        return original(self, episode, behavior, discount=discount)

    monkeypatch.setattr(ScriptedNative, "train_reward_episode", lost)
    result = execute()
    assert result["status"] == "incomplete"
    current = result["adaptation"]
    assert current["status"] == "incomplete" and current["wake_updates"] == 2
    assert calls == 3 and len(REWARD_CALLS) == 2 and not ZERO_CALLS
    assert current["observed_budget_updates"] >= 2
    assert current["train_diagnostics"]["status"] == "not_run"
    assert len(result["training"]["applied_updates"]) == 128
    assert ready["stream"]._runtime.candidate_snapshot().state.completed_updates == 128


def test_changed_original_joint_checkpoint_refuses_every_reward_update(ready, monkeypatch):
    original = handoff.run_joint_reward

    def changed(source, state, sources, **kwargs):
        state.heads[1].gradients["gradient"][0] += 1
        return original(source, state, sources, **kwargs)

    monkeypatch.setattr(handoff, "run_joint_reward", changed)
    result = execute()
    assert result["status"] == "incomplete" and not REWARD_CALLS and not ZERO_CALLS
    assert len(result["training"]["applied_updates"]) == 128


def test_joint_cli_intent_is_explicit_exclusive_and_metadata_only(monkeypatch, capsys):
    monkeypatch.setattr(example, "_episodes", lambda: pytest.fail("source payload read"))
    monkeypatch.setattr(sys, "argv", ["joint", "--reward-training", "--describe"])
    example.main()
    meta = json.loads(capsys.readouterr().out)
    assert meta["reward_training"] and not meta["probe_available"] and not meta["final_available"]
    assert meta["readout_points"] == meta["native_game_points"] == []
    for name in (
        "fresh_task_probe",
        "rehearsal_comparison",
        "cold_fresh_task",
        "class_balanced_joint",
    ):
        with pytest.raises(ValueError):
            example.describe(reward_training=True, **{name: True})
    with pytest.raises(ValueError):
        example.describe(reward_training=1)


@pytest.fixture
def gateway(ready):
    source = PublicCueRewardLearner()
    cold = source.snapshot_state()
    warm = replace(
        cold,
        heads=(
            replace(cold.heads[0], completed_updates=64, training_examples=576),
            replace(cold.heads[1], completed_updates=64, training_examples=576),
        ),
    )
    source.restore_state(warm)
    observed = tuple(
        replace(trajectory(cue), context=context, episode_id=f"minigrid-reward-train-{i}")
        for i, (context, cue) in enumerate(pilot.SCHEDULE)
    )
    state = replace(
        warm,
        heads=(
            replace(warm.heads[0], completed_updates=68, training_examples=584),
            replace(warm.heads[1], completed_updates=68, training_examples=584),
        ),
    )
    sources, _ = example.joint_packets(episodes())
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=210,
            max_process_rss_bytes=1024**3,
        ),
        lambda: 0.0,
    )
    return dict(
        source=source,
        state=state,
        updates=136,
        sources=sources,
        episodes=observed,
        budget=budget,
        check_boundary=lambda: None,
    )


@pytest.mark.parametrize("fault", ["rows", "updates", "role", "identity", "cue", "budget"])
def test_new_readout_gate_refuses_wrong_paid_history_before_kernel(gateway, monkeypatch, fault):
    if fault == "rows":
        first, second = gateway["state"].heads
        gateway["state"] = replace(
            gateway["state"], heads=(replace(first, training_examples=585), second)
        )
    elif fault == "updates":
        gateway["updates"] = 144
    elif fault in ("role", "identity", "cue"):
        first, *tail = gateway["episodes"]
        first = (
            replace(first, role="probe_train")
            if fault == "role"
            else replace(first, episode_id="other")
            if fault == "identity"
            else replace(trajectory("red"), episode_id=first.episode_id)
        )
        gateway["episodes"] = (first, *tail)
    else:
        gateway["budget"] = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=1), lambda: 0.0
        )
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("kernel reached")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_joint_reward_checkpoint(**gateway)


def test_readout_cohort_is_pinned_and_original_warm_admission_remains_strict(gateway, monkeypatch):
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: k)
    result = evaluation.evaluate_joint_reward_checkpoint(**gateway)
    assert len(result["live_plan"]) == 6 and result["role"] == "train"
    assert result["return_metadata"]["observed_reward_rows"] == 16
    assert result["memory_copier"](gateway["source"].snapshot_state()).completed_updates == 128
    object.__setattr__(
        gateway["episodes"][0], "transitions", deepcopy(gateway["episodes"][0].transitions)
    )
    with pytest.raises(ValueError):
        result["extra_check"]()
    with pytest.raises(ValueError):
        evaluation.evaluate_public_cue_recurrent_checkpoint(
            gateway["source"]._learner,
            gateway["state"],
            updates=136,
            sources=gateway["sources"],
            recurring_training=True,
            budget=gateway["budget"],
            check_boundary=lambda: None,
        )


@pytest.mark.parametrize("fault", ["warm_snapshot_value", "warm_leaf", "endpoint_leaf"])
def test_handoff_refuses_hidden_warm_changes_and_equal_endpoint_replacement(
    gateway, monkeypatch, fault
):
    from src.adapters.minigrid_public_cue_recurrent_learner import PublicCueRecurrentLearner

    source = PublicCueRecurrentLearner()
    endpoint = gateway["source"].snapshot_state()
    if fault == "warm_snapshot_value":
        original = PublicCueRewardLearner.snapshot_state

        def hidden(self):
            saved = original(self)
            if saved.completed_updates == 128:
                self._learner._state.heads[1].gradients["gradient"][0] += 1
            return saved

        monkeypatch.setattr(PublicCueRewardLearner, "snapshot_state", hidden)

    def pilot_stub(**kwargs):
        prepared = kwargs["joint_start"]({}, None, 1000.0, lambda: None)
        if fault == "warm_leaf":
            head = prepared["source"]._learner._state.heads[1]
            head.parameters["marker"] = head.parameters["marker"].copy()
        elif fault == "endpoint_leaf":
            head = endpoint.heads[1]
            head.parameters["marker"] = head.parameters["marker"].copy()
        prepared["guard"]()
        pytest.fail("changed handoff accepted")

    monkeypatch.setattr(handoff, "run_pilot", pilot_stub)
    with pytest.raises(ValueError):
        handoff.run_joint_reward(
            source, endpoint, gateway["sources"], check_boundary=lambda: None, memory=None
        )
    assert source.completed_updates == 0 and not REWARD_CALLS and not ZERO_CALLS


def test_joint_readouts_distinguish_both136_arms_and_refuse_changed_labels(gateway, monkeypatch):
    from src.shared.process_memory import ProcessRssSampler

    def kernel(source, state, **options):
        options["check_boundary"]()
        roster = [(c, cue) for c in ("A", "B", "C_train") for cue in ("blue", "red")]
        return dict(
            method=options["method"],
            updates=options["updates"],
            status="complete",
            scores=[
                dict(context=c, cue=cue, ce=1.0, accuracy=0.0, exact_sequence_accuracy=0.0)
                for c, cue in roster
            ],
            episodes=[
                dict(context=c, cue=cue, success=False, **{"return": 0.0}) for c, cue in roster
            ],
            work={},
        )

    monkeypatch.setattr(evaluation, "_run_checkpoint", kernel)
    report = readouts.new_train_diagnostics(momentum_control=True)
    with ProcessRssSampler() as memory:
        completed = readouts.run_train_diagnostics(
            report,
            gateway["source"],
            ((128, gateway["source"].snapshot_state()), (136, gateway["state"])),
            gateway["sources"],
            memory,
            readouts.monotonic() + 60,
            training_phase={"updates_completed": 8},
            check_boundary=lambda: None,
            control_state=deepcopy(gateway["state"]),
            extended_control_readouts=True,
            joint_episodes=gateway["episodes"],
        )
        arms = {r["comparison_arm"]: r for r in report["readouts"]}
        assert set(arms) == {"frozen_joint_bc", "reward", "momentum_only"}
        assert arms["reward"]["updates"] == arms["momentum_only"]["updates"] == 136
        assert arms["frozen_joint_bc"]["continuation_objective"] is None
        assert (
            arms["reward"]["continuation_objective"]
            != arms["momentum_only"]["continuation_objective"]
        )
        assert not REWARD_CALLS and not ZERO_CALLS
        report["readouts"][2]["comparison_arm"] = "reward"
        with pytest.raises(ValueError):
            completed["guard"]()
