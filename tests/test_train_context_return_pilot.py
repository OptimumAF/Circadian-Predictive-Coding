"""Train-only A/B/A return recipe using the real stream and owned fake learners."""

from copy import deepcopy
from dataclasses import replace
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
from test_train_acquisition_pilot import Learner


POINTS = (0, 8, 16, 32, 64, 72, 80, 96, 128, 136, 144, 160, 192)


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
        assert role == "train", "context-return diagnostic attempted reserved/final data collection"
        batch = collect(identity, layout, cue, role)
        captured["collections"].append(deepcopy(batch))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert kwargs["checkpoint_updates"] == POINTS and kwargs["label_delay"] == 1
        assert kwargs.get("rehearsal") is kwargs.get("renewal") is kwargs.get("capacity") is None
        captured["streams"].append((learner, deepcopy(training), kwargs["budget"]))
        return AttributionStream(learner, training, heldout, **kwargs)

    monkeypatch.setattr(shared, "_collect", collect_train)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    return captured


def test_should_describe_explicit_return_protocol_without_resources_or_changing_original_default(
    monkeypatch,
):
    before = deepcopy(pilot.describe_protocol())
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_return_protocol()
    assert pilot.describe_protocol() == before
    assert pilot.UPDATES == 64 and pilot.POINTS == (0, 8, 16, 32, 64)
    assert pilot.RETURN_UPDATES == 192 and pilot.RETURN_POINTS == POINTS
    assert protocol["checkpoint_updates"] == list(POINTS)
    assert protocol["phase_schedule"] == [
        dict(context="A", start=0, end=64),
        dict(context="B", start=64, end=128),
        dict(context="A", start=128, end=192),
    ]
    assert protocol["architecture"] == dict(input_dim=8, hidden_dim=8, num_classes=5, seed=17)
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 1536
    assert work["collected_source_records"] == 32
    assert work["score_forwards"] == 52 and work["score_rows"] == 416
    assert work["recurrent_unroll_steps"] == 1952
    assert work["teacher_episodes"] == work["environment_constructions"] == 4
    assert work["environment_steps"] == 32
    for key in ("final_episodes", "probes", "replay_calls", "reward_updates", "live_episodes"):
        assert work[key] == 0


def test_should_run_original192_update_budget_through_three_phases_on_isolated_owners(monkeypatch):
    captured = environment(monkeypatch)
    report = pilot.run_return_pilot()
    assert report["status"] == "complete", report["reason"]
    assert len(captured["collections"]) == 4 and len(captured["streams"]) == 1
    isolated, training, budget = captured["streams"][0]
    source = Learner.sources[0]
    assert isolated is not source
    assert budget.updates_completed == report["charged_updates"] == 192
    assert budget.budget.max_training_updates == 192
    assert budget.replay_examples_completed == 0
    assert len(training) == len(RecurrentLearner.trains) == len(Learner.training_owners) == 192
    assert len({batch.episode.episode_id for batch in training}) == 192
    assert [batch.episode.episode_id for batch in training] == [
        f"train-fit-{i}" for i in range(192)
    ]
    assert [batch.episode.episode_id for batch in captured["collections"][:2]] == [
        "train-fit-base-1",
        "train-fit-base--1",
    ]
    assert len({row for batch in training for row in batch.row_ids}) == 1536
    assert [batch.context for batch in training] == ["A-red", "A-blue"] * 32 + [
        "B-red",
        "B-blue",
    ] * 32 + ["A-red", "A-blue"] * 32
    originals = {batch.context: batch for batch in captured["collections"]}
    for batch in training:
        original = originals[batch.context]
        assert batch.role == batch.episode.role == "train" and len(batch.features) == 8
        np.testing.assert_array_equal(batch.features, original.features)
        np.testing.assert_array_equal(batch.targets, original.targets)
        assert not np.shares_memory(batch.features, original.features)
        assert not np.shares_memory(batch.targets, original.targets)
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS)
    assert len(RecurrentLearner.classifications) == len(Learner.prediction_owners) == 52
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
    assert report["readback_error"] is report["partial_readback_error"] is None
    assert report["stream_result"]["completed_updates"] == 192
    assert report["stream_result"]["training_examples"] == 1536
    assert report["stream_result"]["score_forward_calls"] == 0
    assert report["work"] == dict(
        score_forwards_attempted=52,
        score_forwards_completed=52,
        score_rows_attempted=416,
        score_rows_completed=416,
    )
    summary = report["training_summary"]
    assert summary["episodes"] == len(summary["episode_ids"]) == 192
    assert summary["unique_source_episodes"] == 4 and summary["collected_source_records"] == 32
    assert summary["training_exposures"] == 1536 and summary["roles"] == ["train"] * 4
    assert summary["encoded_training_array_bytes"] == 110592
    assert summary["encoded_scoring_array_bytes"] == 2304
    assert summary["phase_schedule"] == report["protocol"]["phase_schedule"]
    assert report["scientific_advantage_verified"] is False


def test_should_keep_non_improving_curves_and_fixed_pooled_and_per_layout_priors(monkeypatch):
    captured = environment(monkeypatch)
    report = pilot.run_return_pilot()
    assert report["status"] == "complete", report["reason"]
    originals = captured["collections"]
    assert set(report["majority_controls"]) == {"A", "B"}
    for name, batches in (
        (None, originals),
        ("A", [batch for batch in originals if batch.context.startswith("A-")]),
        ("B", [batch for batch in originals if batch.context.startswith("B-")]),
    ):
        reference = (
            report["majority_control"] if name is None else report["majority_controls"][name]
        )
        labels = np.concatenate([batch.targets[:, 0] for batch in batches]).astype(np.int64)
        counts = np.bincount(labels, minlength=5)
        prior = counts / len(labels)
        assert reference["class_counts"] == counts.tolist()
        assert reference["probabilities"] == prior.tolist()
        assert reference["majority_class"] == int(np.argmax(prior))
        assert reference["mean_accuracy"] == pytest.approx(float(np.max(prior)))
        supported = prior[prior > 0]
        assert reference["mean_ce"] == pytest.approx(float(-np.sum(supported * np.log(supported))))
    contexts = {"A-red", "A-blue", "B-red", "B-blue"}
    for point in report["checkpoints"]:
        assert set(point["scores"]) == set(point["decision_diagnostics"]) == contexts
        assert point["exact_sequence_accuracy"] == 0
        assert point["mean_accuracy"] == sum(s["accuracy"] for s in point["scores"].values()) / 4
        assert point["mean_ce"] == sum(s["ce"] for s in point["scores"].values()) / 4
        for context, detail in point["decision_diagnostics"].items():
            assert detail["rows"] == len(detail["decisions"]) == 8
            assert detail["predicted_class_counts"] == [8, 0, 0, 0, 0]
            assert detail["role"] == "train"
    outcome = report["context_return"]
    assert outcome["incomplete_initial_A_fit"] is True
    assert outcome["useful_skill_established"] is False
    assert [endpoint["updates"] for endpoint in outcome["endpoint_values"]] == [0, 64, 128, 192]
    points = {point["updates"]: point for point in report["checkpoints"]}
    for endpoint in outcome["endpoint_values"]:
        point = points[endpoint["updates"]]
        for layout in ("A", "B"):
            context_names = (f"{layout}-red", f"{layout}-blue")
            for metric in ("ce", "accuracy"):
                assert (
                    endpoint[layout][metric]
                    == sum(point["scores"][context][metric] for context in context_names) / 2
                )
            assert endpoint[layout]["exact_sequence_accuracy"] == 0
    for values in outcome["metrics"].values():
        assert values == dict(
            initial_A_gain=0, new_B_gain=0, old_A_loss=0, return_A_gain=0, return_gap=0
        )
    assert report["scientific_advantage_verified"] is False


@pytest.mark.parametrize("failed_attempt", [65, 66])
def test_should_preserve_charges_and_receipts_when_first_new_phase_updates_fail(
    monkeypatch, failed_attempt
):
    captured = environment(monkeypatch)
    train = Learner.train_batch
    attempts: list[Any] = []

    def fail_at_boundary(self, features, targets):
        attempts.append(features.copy())
        if len(attempts) == failed_attempt:
            raise RuntimeError("new B phase update failed")
        return train(self, features, targets)

    monkeypatch.setattr(Learner, "train_batch", fail_at_boundary)
    report = pilot.run_return_pilot()
    assert report["status"] == "incomplete" and "new B phase" in report["reason"]
    committed = failed_attempt - 1
    assert report["charged_updates"] == captured["streams"][0][2].updates_completed == committed
    assert len(attempts) == failed_attempt and len(Learner.training_owners) == committed
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS[:5])
    assert len(Learner.prediction_owners) == 20
    partial = report["partial_stream"]
    assert partial["completed_wake_updates"] == partial["native_candidate_updates"] == committed
    assert partial["native_candidate_training_examples"] == committed * 8
    assert len(partial["applied_updates"]) == committed
    assert [receipt["update_number"] for receipt in partial["applied_updates"]] == list(
        range(1, committed + 1)
    )
    assert report["stream_result"] is report["context_return"] is None
    assert report["partial_readback_error"] is None


def test_should_refuse_changed_scoring_checkpoint_before_any_B_training(monkeypatch):
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "mutate_prediction", True)
    monkeypatch.setattr(Learner, "mutation_update", 64)
    report = pilot.run_return_pilot()
    assert report["status"] == "incomplete" and "scoring changed checkpoint" in report["reason"]
    assert report["charged_updates"] == len(Learner.training_owners) == 64
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS[:4])
    assert len(Learner.prediction_owners) == 17
    assert report["partial_stream"]["native_candidate_training_examples"] == 512
    assert report["stream_result"] is report["context_return"] is None
    assert report["source_unchanged"] is report["isolated_source_unchanged"] is True


def test_should_refuse_changed_original_source_during_scoring_before_any_B_training(monkeypatch):
    environment(monkeypatch)
    predict = Learner.predict
    changed: list[bool] = []

    def mutate_original(self, features):
        prediction = predict(self, features)
        if self.state.completed_updates == 64 and not changed:
            Learner.sources[0].state.completed_updates += 1
            changed.append(True)
        return prediction

    monkeypatch.setattr(Learner, "predict", mutate_original)
    report = pilot.run_return_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == len(Learner.training_owners) == 64
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS[:4])
    assert report["source_unchanged"] is False and report["isolated_source_unchanged"] is True
    assert report["stream_result"] is report["context_return"] is None


def test_should_refuse_non_train_B_packet_before_any_scoring_or_update(monkeypatch):
    captured = environment(monkeypatch)
    collect = shared._collect

    def wrong_role(identity, layout, cue, role):
        batch = collect(identity, layout, cue, role)
        if layout.layout_id == "B":
            assert batch.episode is not None
            return replace(
                batch, role="final_test", episode=replace(batch.episode, role="final_test")
            )
        return batch

    monkeypatch.setattr(shared, "_collect", wrong_role)
    report = pilot.run_return_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == 0
    assert Learner.training_owners == Learner.prediction_owners == []
    assert captured["streams"] == []


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_train_return_cli_without_options(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-train-return"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="train-context-return")

    monkeypatch.setattr(
        cli, "describe_train_context_return_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_train_context_return_pilot", forbidden if describe else selected)
    for name in (
        "describe_train_acquisition_protocol",
        "run_train_acquisition_pilot",
        "run_learnability_pilot",
        "run_demonstration_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "train-context-return"
