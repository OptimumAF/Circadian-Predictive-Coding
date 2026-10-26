"""The fixed categorical recipe composes the existing stream without native work."""

import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.categorical_trajectory_pilot as pilot
from test_torch_feature_rehearsal_pilot import (
    Learner as BinaryStub,
    Sampler,
    TorchStub,
    environment,
    forbidden,
)


class Learner(BinaryStub):
    def predict(self, features):
        return np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float64), (len(features), 1))


def fake_environment(monkeypatch):
    environment(monkeypatch)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))


def test_should_describe_the_frozen_categorical_recipe_without_work(monkeypatch):
    for name in ("require_torch", "TorchFeatureLearner", "ProcessRssSampler", "build_stream"):
        monkeypatch.setattr(pilot, name, forbidden)
    description = pilot.describe_protocol()
    assert description["num_classes"] == 3 and description["seed"] == 17
    assert description["task_sequence"] == ["A", "B", "C", "A"]
    assert description["planned_work"]["native_train_calls"] == 40
    assert description["planned_work"]["score_forwards"] == 120
    assert "NOT live" in description["transition_metadata"]
    assert description["encoded_packet_array_bytes"] == 968


def test_should_collect_only_fixed_synthetic_owned_complete_packets(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    training, final = pilot.build_stream()
    assert [b.context for b in training] == ["A", "B", "C", "A"]
    assert [b.context for b in final] == ["A", "B", "C"]
    ids = [identity for batch in (*training, *final) for identity in batch.row_ids]
    assert len(set(ids)) == len(ids) == 21
    for batch in (*training, *final):
        assert batch.num_classes == 3 and batch.episode is not None
        assert batch.episode.num_classes == 3
        assert all(t.num_classes == 3 for t in batch.episode.transitions)
        assert batch.features.shape == (3, 2) and batch.targets.shape == (3, 1)
        assert sorted(batch.targets[:, 0]) == [0, 1, 2]
        assert [t.reward for t in batch.episode.transitions] == [0, 0, 1]
        assert [t.terminal for t in batch.episode.transitions] == [False, False, True]
    for train in training:
        for heldout in final:
            assert not np.shares_memory(train.features, heldout.features)
            assert not np.shares_memory(train.targets, heldout.targets)
            assert not np.array_equal(train.features, heldout.features)


def test_should_execute_all_eight_fake_arms_and_preserve_matched_original_authority(monkeypatch):
    fake_environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete" and report["global_errors"] == []
    assert report["shared_replay_plans_exact"] is True
    assert report["shared_initial_parameters_exact"] is True
    assert report["pc_neutral_exact"] is True
    assert len(report["runs"]) == 8
    assert sum(r["wake_updates"] for r in report["runs"]) == 32
    assert sum(r["replay_charged_examples"] for r in report["runs"]) == 24
    assert sum(r["stream_result"]["score_forward_calls"] for r in report["runs"]) == 120
    for row in report["runs"]:
        assert row["source_unchanged"] is True and row["actor_unchanged"] is True
        assert row["continuation"]["original_budgets_and_charges"] is True
        for point in row["stream_result"]["checkpoints"]:
            assert tuple(point["scores"]) == ("A", "B", "C")
            assert all(set(s) == {"ce", "accuracy"} for s in point["scores"].values())
    assert report["scientific_advantage_verified"] is False
    assert report["full_R42_complete"] is False


def test_should_keep_all_failed_native_bindings_without_promoting_partial_work(monkeypatch):
    fake_environment(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic constructor fault")

    monkeypatch.setattr(pilot, "TorchFeatureLearner", fail)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 8
    assert all(row["status"] == "failed" for row in report["runs"])
    assert all("constructor fault" in row["reason"] for row in report["runs"])
    assert all(row["wake_updates"] == row["replay_charged_examples"] == 0 for row in report["runs"])
    assert all(row["stream_result"] is None for row in report["runs"])


def test_should_keep_unreached_rows_on_global_setup_failure(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)

    # Use an ordinary boundary failure; pytest.fail derives from BaseException.
    def fail():
        raise RuntimeError("RSS setup fault")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["global_errors"]
    assert all(row["status"] == "not_run" for row in report["runs"])


def test_should_preserve_primary_failure_and_permanent_wake_cost_when_readback_also_fails(
    monkeypatch,
):
    fake_environment(monkeypatch)

    def fail_readback():
        raise RuntimeError("secondary source readback")

    def fail_after_work(stream):
        stream.run_until(1)
        monkeypatch.setattr(stream._learner, "snapshot_state", fail_readback)
        raise RuntimeError("primary stream fault")

    monkeypatch.setattr(pilot, "_continue", fail_after_work)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert all(row["wake_updates"] == 1 for row in report["runs"])
    assert all("primary stream fault" in row["reason"] for row in report["runs"])
    assert all("secondary source readback" in row["readback_error"] for row in report["runs"])
    assert all(row["stream_result"] is None for row in report["runs"])


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_new_scenario_through_the_existing_cli(monkeypatch, capsys, describe):
    argv = ["attribution", "--scenario", "torch-categorical-rehearsal"]
    if describe:
        argv.append("--describe")
    calls = []

    def selected():
        calls.append(describe)
        return {"status": "complete", "selected": "categorical"}

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        cli, "describe_categorical_protocol" if describe else "run_categorical_pilot", selected
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected"] == "categorical"
    assert calls == [describe]
