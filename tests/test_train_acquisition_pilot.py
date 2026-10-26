"""Train-only acquisition recipe through the real stream and owned fake native ports."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
from src.app.coupled_attribution import AttributionStream
from test_recurrent_key_door_pilot import Learner as RecurrentLearner, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


@dataclass
class State:
    policy: Any
    parameters: dict[str, Any]
    optimizer_state: dict[str, Any]
    gradients: dict[str, Any]
    completed_updates: int
    training_examples: int


class Learner(RecurrentLearner):
    sources: list[Any] = []
    training_owners: list[Any] = []
    prediction_owners: list[Any] = []
    fail_third = False
    mutate_prediction = False
    mutation_update = 0

    def __init__(self, policy):
        super().__init__(policy)
        Learner.sources.append(self)

    def snapshot_state(self):
        return State(
            deepcopy(self.policy),
            deepcopy(self.state.parameters),
            deepcopy(self.state.optimizer_state),
            deepcopy(self.state.gradients),
            self.state.completed_updates,
            self.state.training_examples,
        )

    def restore_state(self, state):
        self.policy = deepcopy(state.policy)
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        Learner.training_owners.append(self)
        if Learner.fail_third and len(Learner.training_owners) == 3:
            raise RuntimeError("third train-only update failed")
        return super().train_batch(features, targets)

    def predict(self, features):
        Learner.prediction_owners.append(self)
        prediction = super().predict(features)
        if Learner.mutate_prediction and self.state.completed_updates == Learner.mutation_update:
            self.state.completed_updates += 1
        return prediction

    def start_episode(self):
        forbidden()


def environment(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", Learner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    for name in ("sources", "training_owners", "prediction_owners"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_third", False)
    monkeypatch.setattr(Learner, "mutate_prediction", False)
    monkeypatch.setattr(Learner, "mutation_update", 0)
    captured: dict[str, Any] = dict(collections=[], streams=[])
    collect = shared._collect

    def collect_train(identity, layout, cue, role):
        assert role == "train", "train-fit recipe attempted reserved/final data collection"
        batch = collect(identity, layout, cue, role)
        captured["collections"].append(deepcopy(batch))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert kwargs["checkpoint_updates"] == (0, 8, 16, 32, 64)
        assert kwargs["label_delay"] == 1
        assert kwargs.get("rehearsal") is kwargs.get("renewal") is kwargs.get("capacity") is None
        captured["streams"].append((learner, deepcopy(training), kwargs["budget"]))
        return AttributionStream(learner, training, heldout, **kwargs)

    monkeypatch.setattr(shared, "_collect", collect_train)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    return captured


def test_should_describe_frozen_train_only_diagnostic_without_native_data_or_resources(monkeypatch):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_protocol()
    assert pilot.POINTS == (0, 8, 16, 32, 64) and pilot.UPDATES == 64
    assert protocol["architecture"] == dict(input_dim=8, hidden_dim=8, num_classes=5, seed=17)
    assert protocol["checkpoint_updates"] == [0, 8, 16, 32, 64]
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 64
    assert work["training_rows"] == 512 and work["score_forwards"] == 10
    assert work["score_rows"] == 80


def test_should_repeat_two_train_traces_with_unique_ids_and_leave_original_owners_unchanged(
    monkeypatch,
):
    captured = environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert len(captured["collections"]) == 2 and len(captured["streams"]) == 1
    isolated, training, budget = captured["streams"][0]
    source = Learner.sources[0]
    assert isolated is not source
    assert budget.updates_completed == report["charged_updates"] == 64
    assert budget.replay_examples_completed == 0
    assert len(training) == len(RecurrentLearner.trains) == 64
    assert len({batch.episode.episode_id for batch in training}) == 64
    assert len({row for batch in training for row in batch.row_ids}) == 512
    for index, batch in enumerate(training):
        base = captured["collections"][index % 2]
        assert batch.role == batch.episode.role == "train"
        assert batch.context == base.context and len(batch.features) == 8
        np.testing.assert_array_equal(batch.features, base.features)
        np.testing.assert_array_equal(batch.targets, base.targets)
        assert not np.shares_memory(batch.features, base.features)
        assert not np.shares_memory(batch.targets, base.targets)
    assert [p["updates"] for p in report["checkpoints"]] == list(pilot.POINTS)
    assert len(RecurrentLearner.classifications) == len(Learner.prediction_owners) == 10
    assert all(owner is not source and owner is not isolated for owner in Learner.training_owners)
    assert all(owner is not source and owner is not isolated for owner in Learner.prediction_owners)
    assert all(
        predictor is not candidate
        for predictor in Learner.prediction_owners
        for candidate in Learner.training_owners
    )
    assert source.state.completed_updates == isolated.state.completed_updates == 0
    for key in ("source_unchanged", "isolated_source_unchanged", "actor_unchanged"):
        assert report[key] is True
    assert report["readback_error"] is None
    assert report["stream_result"]["score_forward_calls"] == 0
    assert report["stream_result"]["completed_updates"] == 64
    assert report["stream_result"]["training_examples"] == 512
    assert report["work"] == dict(
        score_forwards_attempted=10,
        score_forwards_completed=10,
        score_rows_attempted=80,
        score_rows_completed=80,
    )
    summary = report["training_summary"]
    assert summary["episodes"] == len(summary["episode_ids"]) == 64
    assert summary["unique_source_episodes"] == 2 and summary["collected_source_records"] == 16
    assert summary["training_exposures"] == 512 and summary["roles"] == ["train", "train"]
    assert summary["encoded_training_array_bytes"] == 64 * 8 * 9 * 8
    assert summary["encoded_scoring_array_bytes"] == 2 * 8 * 9 * 8
    assert report["scientific_advantage_verified"] is False


def test_should_keep_every_non_improving_train_curve_and_independent_empirical_prior_reference(
    monkeypatch,
):
    captured = environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    labels = np.concatenate([batch.targets[:, 0] for batch in captured["collections"]])
    counts = np.bincount(labels.astype(np.int64), minlength=5)
    expected_prior = counts / len(labels)
    majority = int(np.argmax(expected_prior))
    # The fake always chooses class0; completion must not imply fitting or useful adaptation.
    for point in report["checkpoints"]:
        assert set(point["scores"]) == set(point["decision_diagnostics"]) == {"A-red", "A-blue"}
        assert point["exact_sequence_accuracy"] == 0
        for context, detail in point["decision_diagnostics"].items():
            assert len(detail["decisions"]) == detail["rows"] == 8
            assert detail["predicted_class_counts"] == [8, 0, 0, 0, 0]
            assert sum(detail["class_counts"]) == 8
            assert (
                point["scores"][context]["accuracy"]
                <= report["majority_control"]["scores"][context]["accuracy"]
            )
    reference = report["majority_control"]
    assert reference["class_counts"] == counts.tolist()
    assert reference["probabilities"] == expected_prior.tolist()
    assert reference["majority_class"] == majority
    assert reference["exact_sequence_accuracy"] == 0
    assert reference["mean_accuracy"] == float(expected_prior[majority])
    supported = expected_prior[expected_prior > 0]
    assert reference["mean_ce"] == pytest.approx(float(-np.sum(supported * np.log(supported))))
    for point in report["checkpoints"]:
        assert point["ce_reduction_vs_prior"] == reference["mean_ce"] - point["mean_ce"]
        assert point["accuracy_gain_vs_majority"] == (
            point["mean_accuracy"] - reference["mean_accuracy"]
        )
        assert point["exact_sequence_gain_vs_majority"] == 0


def test_should_preserve_spent_charges_and_partial_curve_after_third_update_failure(monkeypatch):
    captured = environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_third", True)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "third train-only" in report["reason"]
    assert report["charged_updates"] == captured["streams"][0][2].updates_completed == 2
    assert len(Learner.training_owners) == 3 and len(RecurrentLearner.trains) == 2
    assert [point["updates"] for point in report["checkpoints"]] == [0]
    assert len(RecurrentLearner.classifications) == 2
    assert report["stream_result"] is None
    assert report["source_unchanged"] is report["isolated_source_unchanged"] is True
    partial = report["partial_stream"]
    assert partial["completed_wake_updates"] == partial["native_candidate_updates"] == 2
    assert partial["native_candidate_training_examples"] == 16
    assert len(partial["applied_updates"]) == 2
    assert [receipt["update_number"] for receipt in partial["applied_updates"]] == [1, 2]
    assert report["partial_readback_error"] is None


@pytest.mark.parametrize("updates", [0, 8])
def test_should_refuse_state_mutating_diagnostic_prediction_before_more_training(
    monkeypatch, updates
):
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "mutate_prediction", True)
    monkeypatch.setattr(Learner, "mutation_update", updates)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == len(Learner.training_owners) == updates
    assert len(Learner.prediction_owners) == (1 if updates == 0 else 3)
    assert [point["updates"] for point in report["checkpoints"]] == ([] if updates == 0 else [0])
    assert report["stream_result"] is None
    assert report["source_unchanged"] is True


@pytest.mark.parametrize("updates", [0, 8])
def test_should_refuse_original_source_mutation_during_diagnostics_before_more_training(
    monkeypatch, updates
):
    environment(monkeypatch)
    predict = Learner.predict
    changed: list[bool] = []

    def change_original(self, features):
        prediction = predict(self, features)
        if self.state.completed_updates == updates and not changed:
            Learner.sources[0].state.completed_updates += 1
            changed.append(True)
        return prediction

    monkeypatch.setattr(Learner, "predict", change_original)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == len(Learner.training_owners) == updates
    assert report["source_unchanged"] is False
    assert report["isolated_source_unchanged"] is True
    assert "isolation changed" in report["reason"]
    assert [point["updates"] for point in report["checkpoints"]] == ([] if updates == 0 else [0])
    assert report["stream_result"] is None
    if updates:
        assert report["partial_stream"]["completed_wake_updates"] == updates
        assert report["partial_stream"]["native_candidate_training_examples"] == 8 * updates
        assert len(report["partial_stream"]["applied_updates"]) == updates


def test_should_refuse_non_train_collected_packet_without_native_update(monkeypatch):
    captured = environment(monkeypatch)
    collect = shared._collect

    def wrong_role(*args):
        batch = collect(*args)
        assert batch.episode is not None
        return replace(batch, role="final_test", episode=replace(batch.episode, role="final_test"))

    monkeypatch.setattr(shared, "_collect", wrong_role)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == 0 and Learner.training_owners == []
    assert captured["streams"] == [] or len(Learner.prediction_owners) == 0


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_train_fit_cli_without_optional_arguments(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-train-fit"] + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="train-only")

    monkeypatch.setattr(
        cli, "describe_train_acquisition_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_train_acquisition_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_learnability_pilot", forbidden)
    monkeypatch.setattr(cli, "run_demonstration_pilot", forbidden)
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["selected"] == "train-only"


def test_should_exit_nonzero_after_printing_incomplete_train_fit_evidence(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pilot", "--scenario", "torch-key-door-train-fit"])
    monkeypatch.setattr(cli, "run_train_acquisition_pilot", lambda: dict(status="incomplete"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
