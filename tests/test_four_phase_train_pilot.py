"""Fixed four-phase train-only task sequence with a real stream and fake native owners."""

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


POINTS = (0, 8, 16, 32, 64, 72, 80, 96, 128, 136, 144, 160, 192, 200, 208, 224, 256)


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
    captured: dict[str, Any] = dict(collections=[])
    collect = shared._collect

    def collect_train(identity, layout, cue, role):
        assert role == "train", "four-phase train fit admitted reserved/final data"
        batch = collect(identity, layout, cue, role)
        captured["collections"].append((deepcopy(layout), deepcopy(batch)))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert kwargs["checkpoint_updates"] == POINTS and kwargs["label_delay"] == 1
        assert kwargs.get("rehearsal") is kwargs.get("renewal") is kwargs.get("capacity") is None
        captured.update(isolated=learner, training=deepcopy(training), budget=kwargs["budget"])
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured["stream"] = instance
        return instance

    monkeypatch.setattr(shared, "_collect", collect_train)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    return captured


def test_should_describe_fixed_four_phase_work_without_resources_or_default_protocol_changes(
    monkeypatch,
):
    original, returned = (
        deepcopy(pilot.describe_protocol()),
        deepcopy(pilot.describe_return_protocol()),
    )
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_four_phase_protocol()
    assert pilot.describe_protocol() == original and pilot.describe_return_protocol() == returned
    assert pilot.FOUR_PHASE_UPDATES == 256 and pilot.FOUR_PHASE_POINTS == POINTS
    assert protocol["checkpoint_updates"] == list(POINTS)
    assert protocol["architecture"] == dict(input_dim=8, hidden_dim=8, num_classes=5, seed=17)
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 256 and work["training_rows"] == 2048
    assert work["score_forwards"] == 102 and work["score_rows"] == 816
    assert work["recurrent_unroll_steps"] == 2864
    assert work["teacher_episodes"] == work["environment_constructions"] == 6
    assert work["environment_steps"] == 48
    for key in ("probes", "replay_calls", "final_episodes", "live_episodes", "reward_updates"):
        assert work[key] == 0


@pytest.mark.parametrize(
    "incompatible",
    ["return_context", "evaluate_policy_after_training", "episode_replay", "fresh_task_probes"],
)
def test_should_refuse_mixed_four_phase_modes_before_any_resource_work(monkeypatch, incompatible):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError, match="four"):
        pilot._run_pilot(four_phase=True, **{incompatible: True})


def test_should_train_four_continuous_phases_on_unique_train_copies_and_owned_scoring_forks(
    monkeypatch,
):
    captured = environment(monkeypatch)
    historical_layouts = deepcopy(shared.layouts())
    report = pilot.run_four_phase_pilot()
    assert report["status"] == "complete", report["reason"]
    assert shared.layouts() == historical_layouts
    originals = captured["collections"]
    assert len(originals) == 6
    declared = {layout.layout_id: layout for layout, _ in originals}
    assert set(declared) == {"A", "B", "C_train"}
    changed = declared["C_train"]
    assert (changed.rotation, changed.branch_row, changed.swap_keys, changed.appearance) == (
        2,
        3,
        True,
        -0.75,
    )
    training = captured["training"]
    assert len(training) == len(RecurrentLearner.trains) == len(Learner.training_owners) == 256
    assert len({batch.episode.episode_id for batch in training}) == 256
    assert len({row for batch in training for row in batch.row_ids}) == 2048
    assert [batch.context for batch in training] == (
        ["A-red", "A-blue"] * 32
        + ["B-red", "B-blue"] * 32
        + ["C_train-red", "C_train-blue"] * 32
        + ["A-red", "A-blue"] * 32
    )
    base = {batch.context: batch for _, batch in originals}
    for batch in training:
        original = base[batch.context]
        assert batch.role == batch.episode.role == "train"
        np.testing.assert_array_equal(batch.features, original.features)
        np.testing.assert_array_equal(batch.targets, original.targets)
        assert not np.shares_memory(batch.features, original.features)
        assert not np.shares_memory(batch.targets, original.targets)
    assert report["charged_updates"] == captured["budget"].updates_completed == 256
    assert captured["budget"].budget.max_training_updates == 256
    assert captured["budget"].replay_examples_completed == 0
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS)
    assert len(RecurrentLearner.classifications) == len(Learner.prediction_owners) == 102
    assert [owner.state.completed_updates for owner in Learner.prediction_owners] == [
        point for point in POINTS for _ in range(6)
    ]
    assert all(
        scorer is not trainer
        for scorer in Learner.prediction_owners
        for trainer in Learner.training_owners
    )
    assert len({id(owner) for owner in Learner.training_owners}) == 1
    assert (
        captured["isolated"].state.completed_updates
        == Learner.sources[0].state.completed_updates
        == 0
    )
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert report["stream_result"]["completed_updates"] == 256
    assert report["stream_result"]["training_examples"] == 2048
    assert report["stream_result"]["score_forward_calls"] == 0
    assert report["partial_stream"]["native_candidate_updates"] == 256
    assert report["partial_stream"]["native_candidate_training_examples"] == 2048
    summary = report["training_summary"]
    assert summary["episodes"] == len(summary["episode_ids"]) == 256
    assert summary["unique_source_episodes"] == 6 and summary["collected_source_records"] == 48
    assert summary["training_exposures"] == 2048 and summary["roles"] == ["train"] * 6
    assert summary["encoded_training_array_bytes"] == 147456
    assert summary["encoded_scoring_array_bytes"] == 3456
    assert report["protocol"]["context_aliases"] == {"C": "C_train"}
    assert set(report["majority_controls"]) == {"A", "B", "C_train"}
    for layout, reference in report["majority_controls"].items():
        labels = np.concatenate(
            [
                batch.targets[:, 0]
                for _, batch in originals
                if batch.context.startswith(f"{layout}-")
            ]
        ).astype(np.int64)
        counts = np.bincount(labels, minlength=5)
        assert reference["class_counts"] == counts.tolist()
        assert reference["probabilities"] == (counts / len(labels)).tolist()
        assert reference["majority_class"] == int(np.argmax(counts))
    outcome = report["four_phase_return"]
    assert outcome["incomplete_initial_A_fit"] is True
    assert outcome["useful_skill_established"] is False
    assert [row["updates"] for row in outcome["endpoint_values"]] == [0, 64, 128, 192, 256]
    saved = {point["updates"]: point for point in report["checkpoints"]}
    for endpoint in outcome["endpoint_values"]:
        point = saved[endpoint["updates"]]
        for alias, layout in (("A", "A"), ("B", "B"), ("C", "C_train")):
            for metric in ("ce", "accuracy"):
                assert (
                    endpoint[alias][metric]
                    == sum(point["scores"][f"{layout}-{cue}"][metric] for cue in ("red", "blue"))
                    / 2
                )
            assert endpoint[alias]["exact_sequence_accuracy"] == 0
    assert all(value == 0 for metric in outcome["metrics"].values() for value in metric.values())
    assert report["scientific_advantage_verified"] is False
    assert "policy_evaluation" not in report and "probes" not in report and "replay" not in report


@pytest.mark.parametrize("failed_attempt", [129, 193])
def test_should_keep_permanent_charges_and_receipts_when_C_or_return_training_fails(
    monkeypatch, failed_attempt
):
    captured = environment(monkeypatch)
    train = Learner.train_batch
    attempts: list[Any] = []

    def fail_boundary(self, features, targets):
        attempts.append(features.copy())
        if len(attempts) == failed_attempt:
            raise RuntimeError("four-phase boundary update failed")
        return train(self, features, targets)

    monkeypatch.setattr(Learner, "train_batch", fail_boundary)
    report = pilot.run_four_phase_pilot()
    assert report["status"] == "incomplete" and "boundary update failed" in report["reason"]
    completed = failed_attempt - 1
    assert report["charged_updates"] == captured["budget"].updates_completed == completed
    assert len(attempts) == failed_attempt and len(Learner.training_owners) == completed
    assert [point["updates"] for point in report["checkpoints"]] == [
        point for point in POINTS if point <= completed
    ]
    partial = report["partial_stream"]
    assert partial["completed_wake_updates"] == partial["native_candidate_updates"] == completed
    assert partial["native_candidate_training_examples"] == completed * 8
    assert len(partial["applied_updates"]) == completed
    assert all(batch.role == "train" for _, batch in captured["collections"])
    assert report["stream_result"] is None


def test_should_refuse_changed_current_scoring_checkpoint_before_new_C_work(monkeypatch):
    environment(monkeypatch)
    predict = Learner.predict
    changed: list[bool] = []

    def corrupt_current(self, features):
        result = predict(self, features)
        if self.state.completed_updates == 128 and not changed:
            self.state.parameters["weight_input"][0, 0] += 0.125
            changed.append(True)
        return result

    monkeypatch.setattr(Learner, "predict", corrupt_current)
    report = pilot.run_four_phase_pilot()
    assert changed == [True] and report["status"] == "incomplete"
    assert report["charged_updates"] == len(Learner.training_owners) == 128
    assert [point["updates"] for point in report["checkpoints"]] == [
        point for point in POINTS if point < 128
    ]
    assert report["stream_result"] is None


def test_should_reject_non_train_C_packet_before_any_native_work(monkeypatch):
    captured = environment(monkeypatch)
    collect = shared._collect

    def wrong_role(identity, layout, cue, role):
        batch = collect(identity, layout, cue, role)
        if layout.layout_id == "C_train":
            assert batch.episode is not None
            return replace(
                batch, role="final_test", episode=replace(batch.episode, role="final_test")
            )
        return batch

    monkeypatch.setattr(shared, "_collect", wrong_role)
    report = pilot.run_four_phase_pilot()
    assert report["status"] == "incomplete" and report["charged_updates"] == 0
    assert Learner.training_owners == Learner.prediction_owners == []
    assert "stream" not in captured


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_four_phase_cli_without_arguments(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-train-four-phase"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="four-phase")

    monkeypatch.setattr(cli, "describe_four_phase_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_four_phase_pilot", forbidden if describe else selected)
    for name in (
        "run_train_acquisition_pilot",
        "run_train_context_return_pilot",
        "run_return_policy_pilot",
        "run_return_probe_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["selected"] == "four-phase"
