"""Actual game collection and ordinary recipe/CLI with fake native heads."""

import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as pilot
from test_torch_feature_rehearsal_pilot import (
    Learner as FeatureStub,
    Sampler,
    TorchStub,
    environment,
    forbidden,
)


class Learner(FeatureStub):
    def predict(self, features):
        return np.tile(np.array([[0.6, 0.1, 0.1, 0.1, 0.1]], dtype=np.float64), (len(features), 1))


def fake_environment(monkeypatch):
    environment(monkeypatch)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))


def test_should_describe_without_collection_prediction_or_resource_work(monkeypatch):
    for name in (
        "build_stream",
        "KeyDoorNavigation",
        "TorchFeatureLearner",
        "require_torch",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol()
    assert protocol["planned_work"]["native_train_calls"] == 80
    assert protocol["planned_work"]["live_episodes"] == 320
    assert protocol["heldout_layouts"] == ["D"]
    assert protocol["episode_schema"] == dict(observation_dim=8, steps_per_episode=8, num_classes=5)
    assert protocol["resources"]["storage_bytes"] == 8 * 1024**2


def test_should_collect_actual_eight_step_successes_with_owned_roles_and_unseen_D(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    training, final = pilot.build_stream()
    assert [b.context for b in training] == [
        "A-red",
        "A-blue",
        "B-red",
        "B-blue",
        "C-red",
        "C-blue",
        "A-red",
        "A-blue",
    ]
    assert [b.context for b in final] == [
        "A-red",
        "A-blue",
        "B-red",
        "B-blue",
        "C-red",
        "C-blue",
        "D-red",
        "D-blue",
    ]
    ids = [name for b in (*training, *final) for name in b.row_ids]
    assert len(ids) == len(set(ids)) == 128
    for batch in (*training, *final):
        assert batch.num_classes == 5 and batch.episode is not None
        assert batch.features.shape == (8, 8) and batch.targets.shape == (8, 1)
        assert len(batch.episode.transitions) == 8
        assert sum(t.reward for t in batch.episode.transitions) == 1
        assert [t.terminal for t in batch.episode.transitions] == [False] * 7 + [True]
        assert batch.features[0, 6] in (-1, 1) and np.all(batch.features[1:, 6] == 0)
        for earlier, later in zip(
            batch.episode.transitions, batch.episode.transitions[1:], strict=False
        ):
            assert earlier.next_observation == later.observation
    for a in training:
        assert not a.context.startswith("D-")
        for b in final:
            assert not np.shares_memory(a.features, b.features)
            assert not np.shares_memory(a.targets, b.targets)


def test_should_keep_all_fake_live_negatives_without_a_quality_gate(monkeypatch):
    fake_environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete" and report["global_errors"] == []
    assert report["shared_initial_parameters_exact"] is True
    assert report["shared_replay_plans_exact"] is True and report["pc_neutral_exact"] is True
    assert len(report["demonstrations"]) == 16
    assert sum(r["wake_updates"] for r in report["runs"]) == 64
    assert sum(r["replay_charged_examples"] for r in report["runs"]) == 128
    assert sum(r["stream_result"]["score_forward_calls"] for r in report["runs"]) == 320
    episodes = [e for r in report["runs"] for e in r["episodes"]]
    assert len(episodes) == 320
    assert all(e["outcome"] == "illegal_action" and not e["success"] for e in episodes)
    assert all(e["state_unchanged"] is True and e["return"] == 0 for e in episodes)
    assert sum(e["layout_partition"] == "heldout_layout" for e in episodes) == 80
    assert all(r["source_unchanged"] and r["actor_unchanged"] for r in report["runs"])
    assert report["scientific_advantage_verified"] is False and report["full_R42_complete"] is False


def test_should_stop_changed_prediction_state_and_retain_partial_unreached_episodes(monkeypatch):
    fake_environment(monkeypatch)
    original = Learner.predict

    def mutating(self, features):
        result = original(self, features)
        # Final classification uses whole packets; corrupt only the live copy.
        if len(features) == 1:
            self.state.completed_updates += 1
        return result

    monkeypatch.setattr(Learner, "predict", mutating)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    for row in report["runs"]:
        assert row["status"] == "failed" and row["wake_updates"] == 8
        assert row["episodes"][0]["status"] == "failed"
        assert row["episodes"][0]["prediction_calls_attempted"] == 1
        assert "state changed" in row["episodes"][0]["reason"]
        assert all(e["status"] == "not_run" for e in row["episodes"][1:])
        assert row["source_unchanged"] is True and row["actor_unchanged"] is True


def test_should_preserve_primary_failure_permanent_work_and_secondary_readback(monkeypatch):
    fake_environment(monkeypatch)

    def readback():
        raise RuntimeError("secondary readback")

    def fail(row, source, states, budget):
        monkeypatch.setattr(source, "snapshot_state", readback)
        raise RuntimeError("primary live fault")

    monkeypatch.setattr(pilot, "_live", fail)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    for row in report["runs"]:
        assert row["wake_updates"] == 8 and row["replay_charged_examples"] == (
            32 if row["replay_enabled"] else 0
        )
        assert (
            "primary live fault" in row["reason"] and "secondary readback" in row["readback_error"]
        )
        assert all(e["status"] == "not_run" for e in row["episodes"])


def test_should_keep_every_planned_slot_on_resource_setup_failure(monkeypatch):
    def fail():
        raise RuntimeError("resource setup")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["global_errors"]
    assert len(report["runs"]) == 8 and sum(len(r["episodes"]) for r in report["runs"]) == 320
    assert all(r["status"] == "not_run" for r in report["runs"])


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_the_selected_key_door_cli_route(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["attribution", "--scenario", "torch-key-door"] + (["--describe"] if describe else []),
    )

    def selected():
        return dict(status="complete", selected="key-door")

    monkeypatch.setattr(
        cli, "describe_key_door_protocol" if describe else "run_key_door_pilot", selected
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected"] == "key-door"
