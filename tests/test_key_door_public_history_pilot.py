"""Existing stream integration with fake matched native heads and real TRAIN data."""

import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import key_door_public_history_pilot as pilot
from src.adapters import key_door_pilot as shared
from test_key_door_pilot import Learner as FeatureLearner, fake_environment
from test_torch_feature_rehearsal_pilot import forbidden


def environment(monkeypatch):
    fake_environment(monkeypatch)
    evidence: dict[str, Any] = dict(
        sources=[],
        trains=[],
        predictions=[],
        collections=[],
        streams=[],
        on_train=None,
        on_predict=None,
        fail_method=None,
        fail_at=1,
    )
    collect = shared._collect

    class Learner(FeatureLearner):
        def __init__(self, policy):
            assert policy.feature_dim == 16 and policy.hidden_dim == 8 and policy.seed == 17
            assert policy.num_classes == 5
            super().__init__(policy)
            evidence["sources"].append(self)

        def train_batch(self, features, targets):
            assert features.shape == (8, 16) and targets.shape == (8, 1)
            assert np.all(features[:, :8] == features[0, :8])
            assert np.all(features[:, 6] == features[0, 6]) and features[0, 6] in (-1, 1)
            assert np.all(features[1:, 14] == 0)
            if (
                self.policy.method == evidence["fail_method"]
                and self.state.completed_updates + 1 == evidence["fail_at"]
            ):
                raise RuntimeError("declared native training failure")
            result = super().train_batch(features, targets)
            evidence["trains"].append((self.policy.method, features.copy(), targets.copy()))
            if evidence["on_train"] is not None:
                evidence["on_train"](self)
            return result

        def predict(self, features):
            assert len(evidence["trains"]) == 32, "readout preceded all original training"
            assert features.shape == (8, 16)
            result = super().predict(features)
            evidence["predictions"].append((self.policy.method, self.state.completed_updates))
            if evidence["on_predict"] is not None:
                evidence["on_predict"](self)
            return result

    def collect_train(identity, layout, cue, role):
        assert role == "train", "matched TRAIN route constructed final/probe data"
        batch = collect(identity, layout, cue, role)
        evidence["collections"].append(batch)
        return batch

    original_stream = pilot.AttributionStream

    def stream(source, training, heldout, **options):
        assert heldout == () and options["training_only"] is True
        assert options["label_delay"] == 1 and options["checkpoint_updates"] == shared.POINTS
        instance = original_stream(source, training, heldout, **options)
        evidence["streams"].append(instance)
        return instance

    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(shared, "_collect", collect_train)
    monkeypatch.setattr(shared, "_live", forbidden)
    monkeypatch.setattr(shared, "build_stream", forbidden)
    return evidence


def test_should_train_all_matched_methods_before_owned_train_readouts_without_finals(monkeypatch):
    evidence = environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["global_errors"]
    assert [row["method"] for row in report["runs"]] == list(shared.METHODS)
    assert len(evidence["sources"]) == 4 and len(evidence["trains"]) == 32
    assert len(evidence["collections"]) == 8 and len(evidence["predictions"]) == 120
    first = evidence["trains"][:8]
    for start in (8, 16, 24):
        for before, after in zip(first, evidence["trains"][start : start + 8], strict=True):
            np.testing.assert_array_equal(before[1], after[1])
            np.testing.assert_array_equal(before[2], after[2])
    for row in report["runs"]:
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert row["wake_updates"] == row["stream_result"]["completed_updates"] == 8
        assert row["stream_result"]["evaluation_scope"] == "training_only"
        assert row["stream_result"]["score_forward_calls"] == 0
        assert len(row["stream_result"]["applied_updates"]) == 8
        assert [point["updates"] for point in row["training_points"]] == [0, 2, 4, 6, 8]
        assert (
            row["work"]["score_forwards_attempted"] == row["work"]["score_forwards_completed"] == 30
        )
        assert row["work"]["score_rows_completed"] == 240
    assert report["shared_initial_parameters_exact"] is report["pc_neutral_exact"] is True
    assert report["final_data_constructed"] is False
    assert report["scientific_advantage_verified"] is report["full_R43_complete"] is False


def test_should_keep_input_roles_labels_ids_and_detach_encoded_arrays(monkeypatch):
    evidence = environment(monkeypatch)
    encoded = pilot.build_training_stream()
    for raw, transformed in zip(evidence["collections"], encoded, strict=True):
        assert raw.row_ids == transformed.row_ids and raw.role == transformed.role == "train"
        assert raw.context == transformed.context and raw.num_classes == transformed.num_classes
        np.testing.assert_array_equal(raw.targets, transformed.targets)
        np.testing.assert_array_equal(raw.features, transformed.features[:, 8:])
        assert not np.shares_memory(raw.features, transformed.features)
        assert not np.shares_memory(raw.targets, transformed.targets)


def test_should_stop_first_failed_arm_and_keep_prior_paid_training_with_unreached_methods(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    evidence["fail_method"] = "pc"
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert [row["status"] for row in report["runs"]] == ["trained", "failed", "not_run", "not_run"]
    assert [row["wake_updates"] for row in report["runs"]] == [8, 0, 0, 0]
    assert len(evidence["trains"]) == 8 and not evidence["predictions"]


@pytest.mark.parametrize("kind", ["prior_source", "prior_receipt", "prior_clock"])
def test_should_refuse_changed_completed_arm_before_admitting_another_update(monkeypatch, kind):
    evidence = environment(monkeypatch)

    def corrupt(owner):
        if owner.policy.method != "pc":
            return
        old = evidence["streams"][0]
        if kind == "prior_source":
            evidence["sources"][0].state.completed_updates += 1
        elif kind == "prior_receipt":
            old._runtime._inbox._applied.clear()
        else:
            old._budget.last_clock += 1.0

    evidence["on_train"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert len(evidence["trains"]) <= 9
    assert report["runs"][2]["status"] == "not_run" and not evidence["predictions"]


def test_should_refuse_changed_readout_checkpoint_and_preserve_attempted_work(monkeypatch):
    evidence = environment(monkeypatch)
    evidence["on_predict"] = lambda owner: setattr(owner.state, "completed_updates", 100)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    assert row["status"] == "failed" and row["wake_updates"] == 8
    assert row["work"]["score_forwards_attempted"] == 1
    assert row["work"]["score_forwards_completed"] == 0
    assert (
        row["native_score_returns"] == row["partial_native_evidence"]["score_calls_returned"] == 1
    )
    assert len(evidence["trains"]) == 32 and len(evidence["predictions"]) == 1


def test_should_keep_unknown_collection_work_on_failure(monkeypatch):
    environment(monkeypatch)
    monkeypatch.setattr(
        pilot,
        "build_training_stream",
        lambda: (_ for _ in ()).throw(ValueError("collection failed")),
    )
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["collected_episodes"] is None


def test_should_keep_partial_commits_receipts_and_candidate_when_a_later_update_fails(monkeypatch):
    evidence = environment(monkeypatch)
    evidence.update(fail_method="pc", fail_at=4)
    report = pilot.run_pilot()
    row = report["runs"][1]
    partial = row["partial_native_evidence"]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert row["wake_updates"] == partial["charged_updates"] == 3
    assert len(partial["applied_updates"]) == 3
    assert [receipt["update_number"] for receipt in partial["applied_updates"]] == [1, 2, 3]
    assert partial["preceding_success_observation"]["native_updates"] == 3
    assert partial["preceding_success_observation"]["native_training_examples"] == 24
    assert partial["current_candidate_unavailable"] is True
    assert "stopped after uncertain native mutation" in partial["readback_error"]
    assert partial["failed_work_may_be_unobserved"] is True
    assert len(evidence["trains"]) == 11 and not evidence["predictions"]
    assert all(other["status"] == "not_run" for other in report["runs"][2:])


@pytest.mark.parametrize("describe", [True, False])
def test_should_dispatch_fixed_route_without_selecting_old_game_runner(
    monkeypatch, capsys, describe
):
    calls = []

    def selected():
        calls.append(True)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli,
        "describe_public_history_protocol" if describe else "run_public_history_pilot",
        selected,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-public-history-train"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["status"] == "complete"


def test_should_describe_without_models_or_collection(monkeypatch):
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "build_training_stream", forbidden)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", forbidden)
    protocol = pilot.describe_protocol()
    assert protocol["feature_dim"] == 16 and protocol["seed"] == 17
    assert protocol["planned_work"] == dict(
        native_updates=32,
        training_rows=256,
        score_calls=120,
        score_rows=960,
        teacher_episodes=8,
        physics_steps=64,
    )
