"""Physical quotas use the existing actor/receipt recipe with explicit fake policies."""

from copy import deepcopy
from dataclasses import asdict
import json
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.core.key_door_navigation import scripted_actions
from test_key_door_reward_pilot import environment as base_environment
from test_legal_reward_pilot import Learner, Session
from test_torch_feature_rehearsal_pilot import forbidden


def environment(monkeypatch, *, scripted=True):
    base_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRewardUpdateLearner", Learner)
    for name in ("sessions", "updates", "identifiers", "classifications", "sources"):
        monkeypatch.setattr(Learner, name, [])
    for name, value in (
        ("committed", 0),
        ("fail_collection", False),
        ("fail_update", False),
        ("raw", (0.2,) * 5),
        ("quota_phase", 0),
    ):
        monkeypatch.setattr(Learner, name, value, raising=False)
    original_prepare = pilot._prepare_update
    original_session_predict = Session.predict
    budgets, final_collections = [], []
    active: dict[str, Any] = {}

    def prepare(row, source, state, rng, budget, index, identity, **kwargs):
        active["phase"] = row["training"][index]["quota_phase"]
        active.update(row=row, index=index, quota=kwargs["interaction_quota"])
        return original_prepare(row, source, state, rng, budget, index, identity, **kwargs)

    def session_predict(self, features):
        exposed = sum(len(u.episode.episode.transitions) for u in Learner.updates)
        if not scripted or exposed == 96:
            return original_session_predict(self, features)
        self.inputs.append(features.copy())
        phase = active["phase"]
        context = shared.TASKS[phase // 2]
        layout = next(x for x in pilot.action_remapping_layouts() if x.layout_id == context)
        cue = 1 if phase % 2 == 0 else -1
        action = scripted_actions(layout, cue)[self.completed_steps]
        self.completed_steps += 1
        result = np.zeros((1, 5), dtype=np.float64)
        result[0, action] = 1
        return result

    def predict(self, features):
        assert sum(len(u.episode.episode.transitions) for u in Learner.updates) == 96, (
            "final data arrived before physical quotas completed"
        )
        Learner.classifications.append(features.copy())
        return np.tile(np.array([Learner.raw]), (len(features), 1))

    original_collect, original_budget = shared._collect, shared._session

    def collect(*args, **kwargs):
        assert sum(len(u.episode.episode.transitions) for u in Learner.updates) == 96
        assert args[3] == "final_test"
        final_collections.append(args)
        return original_collect(*args, **kwargs)

    def budget(*args, **kwargs):
        result = original_budget(*args, **kwargs)
        budgets.append(result)
        return result

    monkeypatch.setattr(pilot, "_prepare_update", prepare)
    monkeypatch.setattr(Session, "predict", session_predict)
    monkeypatch.setattr(Learner, "predict", predict)
    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_session", budget)
    return budgets, final_collections, active


def run():
    return pilot.run_pilot(
        condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=True
    )


def test_should_fill_each_quota_across_goals_then_short_fresh_episodes(monkeypatch):
    budgets, finals, _ = environment(monkeypatch)
    report = run()
    assert report["status"] == "complete", report
    row = report["runs"][0]
    assert len(budgets) == 1 and budgets[0].updates_completed == 16
    assert len(row["training"]) == len(Learner.updates) == 16
    assert [len(s["collection"]["transitions"]) for s in row["training"]] == [8, 4] * 8
    assert [s["collection"]["termination"] for s in row["training"]] == [
        "terminal",
        "truncated",
    ] * 8
    assert [s["quota_phase"] for s in row["training"]] == [i for i in range(8) for _ in range(2)]
    assert (
        row["physical_quotas"]
        == [dict(maximum_steps=12, attempted_steps=12, completed_steps=12, failed_steps=0)] * 8
    )
    assert row["quota_checkpoints"] == [
        dict(physical_steps=p, native_updates=u)
        for p, u in zip((0, 24, 48, 72, 96), (0, 4, 8, 12, 16))
    ]
    assert [p["updates"] for p in row["stream_result"]["checkpoints"]] == [0, 4, 8, 12, 16]
    assert sorted({e["checkpoint_updates"] for e in row["episodes"]}) == [0, 4, 8, 12, 16]
    rng = np.random.Generator(np.random.PCG64(29))
    exposed = 0
    for index, slot in enumerate(row["training"]):
        collected = slot["collection"]
        assert collected["behavior_native_updates"] == index
        assert collected["behavior_training_examples"] == exposed
        assert collected["sampler_before"] == rng.bit_generator.state
        rng.random(len(collected["transitions"]))
        assert collected["sampler_after"] == rng.bit_generator.state
        assert slot["charged_updates_after"] == index + 1
        exposed += len(collected["transitions"])
    assert exposed == 96 and len(finals) == 8
    assert len(Learner.classifications) == 40 and row["live_sessions_started"] == 40
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert all(s.inputs[0][0, 6] in (-1, 1) for s in Learner.sessions)


def test_should_fill_same_physical_allowance_for_a_different_own_policy(monkeypatch):
    budgets, finals, _ = environment(monkeypatch, scripted=False)
    report = run()
    assert report["status"] == "complete", report
    row = report["runs"][0]
    assert sum(len(s["collection"]["transitions"]) for s in row["training"]) == 96
    assert 8 <= budgets[0].updates_completed <= 16
    assert len(finals) == 8 and row["quota_checkpoints"][-1]["physical_steps"] == 96


def test_should_refuse_joint_quota_and_active_evidence_mutation_during_update(monkeypatch):
    budgets, finals, active = environment(monkeypatch)
    original = Learner.train_batch

    def train(self, update, identity):
        quota = active["quota"]
        quota.begin_step()
        quota.complete_step()
        collection = active["row"]["training"][active["index"]]["collection"]
        collection["environment_steps_attempted"] += 1
        collection["environment_steps"] += 1
        collection["interaction_quota_after"] = asdict(quota.snapshot())
        return original(self, update, identity)

    monkeypatch.setattr(Learner, "train_batch", train)
    report = run()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert "evidence changed" in row["reason"]
    assert budgets[0].updates_completed == len(Learner.updates) == 1
    assert not finals and not Learner.classifications
    assert row["training"][0]["update_committed"] is True
    assert row["physical_quotas"][0]["attempted_steps"] == 9


def test_should_keep_failed_physical_attempt_and_stop_before_native_update(monkeypatch):
    budgets, finals, _ = environment(monkeypatch)

    def fail(self, action):
        raise RuntimeError("physical call failed after admission")

    import src.app.key_door_training as collection

    monkeypatch.setattr(collection.KeyDoorNavigation, "step", fail)
    report = run()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert budgets[0].updates_completed == len(Learner.updates) == 0
    assert row["physical_quotas"][0] == dict(
        maximum_steps=12, attempted_steps=1, completed_steps=0, failed_steps=1
    )
    assert row["training"][0]["collection"]["environment_steps_attempted"] == 1
    assert not finals


def test_should_refuse_changed_inactive_quota_before_another_prediction(monkeypatch):
    budgets, finals, _ = environment(monkeypatch)
    actual_quota = pilot.PhysicalInteractionQuota
    quotas = []

    def create(limit):
        quota = actual_quota(limit)
        quotas.append(quota)
        return quota

    original_predict = Session.predict
    predictions = []

    def predict(self, features):
        result = original_predict(self, features)
        predictions.append(features.copy())
        if len(predictions) == 1:
            quotas[1].begin_step()
            quotas[1].complete_step()
        return result

    monkeypatch.setattr(pilot, "PhysicalInteractionQuota", create)
    monkeypatch.setattr(Session, "predict", predict)
    report = run()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert len(predictions) == 1 and not finals
    assert budgets[0].updates_completed == 0
    assert row["training"][0]["collection"]["environment_steps"] == 1
    assert row["physical_quotas"][1]["completed_steps"] == 1


def test_should_refuse_changed_boundary_before_final_predictions(monkeypatch):
    budgets, _, active = environment(monkeypatch)
    original = shared._collect

    def collect(*args, **kwargs):
        result = original(*args, **kwargs)
        active["row"]["quota_checkpoints"][-1]["native_updates"] = 0
        return result

    monkeypatch.setattr(shared, "_collect", collect)
    report = run()
    assert report["status"] == "incomplete"
    assert budgets[0].updates_completed == 16 and not Learner.classifications


@pytest.mark.parametrize("bad", [None, 0, 1, "yes", np.bool_(True)])
def test_should_refuse_invalid_quota_mode_without_resource_work(monkeypatch, bad):
    monkeypatch.setattr(pilot, "_run_pilot", forbidden)
    with pytest.raises(ValueError):
        pilot.run_pilot(
            condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=bad
        )


def test_should_describe_fixed_physical_and_variable_native_clocks_without_models(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    old = deepcopy(pilot.describe_protocol(condition_on_legal_actions=True, action_remapping=True))
    new = pilot.describe_protocol(
        condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=True
    )
    assert new["planned_work"]["exact_training_rows"] == 96
    assert new["planned_work"]["maximum_native_train_calls"] == 16
    assert new["checkpoint_updates"] is None
    assert new["checkpoint_physical_steps"] == [0, 24, 48, 72, 96]
    assert old == pilot.describe_protocol(
        condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=False
    )


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_quota_mode_in_existing_cli(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_legal_reward_protocol", selected)
    monkeypatch.setattr(cli, "run_legal_reward_pilot", selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-legal-interaction-quota"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [
        dict(condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=True)
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
