"""Existing weighted native port inside the unchanged train-only live caller."""

from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
import src.adapters.torch_train_weighted_learner as weighted
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from test_recurrent_key_door_pilot import Learner as RecurrentLearner
from test_torch_feature_rehearsal_pilot import forbidden
from test_train_acquisition_pilot import Learner
from test_train_live_acquisition_pilot import environment as live_environment


def weights_from_labels(episode):
    counts = [sum(step.action == action for step in episode.transitions) for action in range(5)]
    present = sum(count > 0 for count in counts)
    return tuple(len(episode.transitions) / (present * count) if count else 1.0 for count in counts)


def environment(monkeypatch):
    captured = live_environment(monkeypatch)
    captured.update(weighted_calls=[], ordered_training=None)
    original_stream = pilot.AttributionStream

    def stream(learner, training, heldout, **kwargs):
        captured["ordered_training"] = training
        return original_stream(learner, training, heldout, **kwargs)

    def train_weighted_episode(owner, episode, class_weights):
        index = owner.state.completed_updates
        expected = captured["ordered_training"][index]
        assert episode == expected.episode and episode.role == "train"
        assert episode.episode_id == f"train-fit-{index}"
        assert class_weights == weights_from_labels(episode)
        features, targets = episode_arrays(episode)
        np.testing.assert_array_equal(features, expected.features)
        np.testing.assert_array_equal(targets, expected.targets)
        captured["weighted_calls"].append((owner, deepcopy(episode), tuple(class_weights)))
        diagnostic = Learner.train_batch(owner, features, targets)
        return TrainingDiagnostic(TORCH_RECURRENT_WEIGHTED_CE_ID, diagnostic.value)

    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(weighted, "TorchRecurrentLearner", Learner)
    monkeypatch.setattr(weighted, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    monkeypatch.setattr(Learner, "train_weighted_episode", train_weighted_episode, raising=False)
    monkeypatch.setattr(
        Learner,
        "completed_updates",
        property(lambda owner: owner.state.completed_updates),
        raising=False,
    )
    monkeypatch.setattr(
        Learner,
        "training_examples",
        property(lambda owner: owner.state.training_examples),
        raising=False,
    )
    return captured


def test_should_dispatch_all64_current_complete_train_episodes_to_existing_weighted_objective(
    monkeypatch,
):
    captured = environment(monkeypatch)
    global_rng = pickle.dumps(np.random.get_state())
    report = pilot.run_weighted_live_pilot()
    assert report["status"] == "complete", report
    assert (
        report["charged_updates"]
        == len(captured["weighted_calls"])
        == len(RecurrentLearner.trains)
        == 64
    )
    assert report["stream_result"]["completed_updates"] == 64
    assert report["stream_result"]["training_examples"] == 512
    assert len(captured["collections"]) == 2 and len(captured["finished"]) == 1
    isolated, training, budget = captured["streams"][0]
    assert type(isolated) is weighted.TorchTrainWeightedLearner
    assert budget.updates_completed == budget.budget.max_training_updates == 64
    assert budget.replay_examples_completed == 0
    assert len(training) == 64 and len({batch.episode.episode_id for batch in training}) == 64
    assert all(batch.role == batch.episode.role == "train" for batch in training)
    for index, ((owner, episode, class_weights), batch) in enumerate(
        zip(captured["weighted_calls"], training, strict=True)
    ):
        assert owner is not Learner.sources[0] and owner is not isolated._learner
        assert episode == batch.episode and class_weights == weights_from_labels(episode)
        expected_base = captured["collections"][index % 2]
        features, targets = episode_arrays(episode)
        np.testing.assert_array_equal(features, expected_base.features)
        np.testing.assert_array_equal(targets, expected_base.targets)
    receipts = report["stream_result"]["applied_updates"]
    assert len(receipts) == 64
    assert [receipt["update_number"] for receipt in receipts] == list(range(1, 65))
    assert all(
        receipt["diagnostic"]["definition"] == TORCH_RECURRENT_WEIGHTED_CE_ID
        for receipt in receipts
    )
    metadata = report["training_objective"]
    assert metadata == dict(
        objective_id=TORCH_RECURRENT_WEIGHTED_CE_ID,
        weight_rule="current_train_episode_inverse_present_class_frequency_v1",
        class_weights=[
            dict(
                episode_id=batch.episode.episode_id, values=list(weights_from_labels(batch.episode))
            )
            for batch in captured["collections"]
        ],
    )
    assert [point["updates"] for point in report["checkpoints"]] == [0, 8, 16, 32, 64]
    assert len(RecurrentLearner.classifications) == 10
    assert len(captured["sessions"]) == 4
    assert [session.state.completed_updates for session in captured["sessions"]] == [0, 0, 64, 64]
    assert all(len(session.inputs) == 12 for session in captured["sessions"])
    assert report["training_summary"]["training_exposures"] == 512
    assert report["training_summary"]["collected_source_records"] == 16
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    diagnostic = report["train_policy_diagnostics"]
    assert diagnostic["status"] == "complete"
    assert diagnostic["learner_gains"] == dict(success_rate=0.0, mean_return=0.0)
    assert all(
        slot["status"] == "complete" and slot["success"] is False for slot in diagnostic["episodes"]
    )
    assert all(
        slot["status"] == "refused" and slot["success"] is None
        for slot in diagnostic["reference"]["episodes"]
    )
    assert diagnostic["comparison"] is diagnostic["reference"]["outcomes"] is None
    assert report["scientific_advantage_verified"] is False
    assert pickle.dumps(np.random.get_state()) == global_rng


def test_should_preserve_third_weighted_update_failure_charges_and_skip_live(monkeypatch):
    captured = environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_third", True)
    report = pilot.run_weighted_live_pilot()
    assert report["status"] == "incomplete" and "third train-only" in report["reason"]
    assert len(captured["weighted_calls"]) == len(Learner.training_owners) == 3
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 2
    assert report["partial_stream"]["native_candidate_updates"] == 2
    assert report["partial_stream"]["native_candidate_training_examples"] == 16
    assert len(report["partial_stream"]["applied_updates"]) == 2
    assert all(
        receipt["diagnostic"]["definition"] == TORCH_RECURRENT_WEIGHTED_CE_ID
        for receipt in report["partial_stream"]["applied_updates"]
    )
    assert captured["finished"] == captured["sessions"] == captured["diagnostics"] == []


@pytest.mark.parametrize("target", ["schedule", "labels", "source"])
def test_should_refuse_changed_schedule_labels_or_source_before_next_weighted_sgd(
    monkeypatch, target
):
    captured = environment(monkeypatch)
    original = Learner.predict
    changed: list[bool] = []

    def corrupt(owner, features):
        result = original(owner, features)
        if owner.state.completed_updates == 8 and not changed:
            stream = captured["actual_streams"][0]
            if target == "source":
                Learner.sources[0].state.parameters["weight_input"][0, 0] += 1
            elif target == "schedule":
                object.__setattr__(
                    stream._runtime._candidate._episodes[8], "context", "changed-context"
                )
            else:
                batch = stream._training[8]
                batch.targets[0, 0] = (batch.targets[0, 0] + 1) % 5
            changed.append(True)
        return result

    monkeypatch.setattr(Learner, "predict", corrupt)
    report = pilot.run_weighted_live_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert (
        report["charged_updates"]
        == len(captured["weighted_calls"])
        == len(RecurrentLearner.trains)
        == 8
    )
    assert captured["finished"] == captured["sessions"] == captured["diagnostics"] == []
    if target == "source":
        assert report["source_unchanged"] is not True
        assert report["readback_error"] is not None
    else:
        assert report["source_unchanged"] is True


@pytest.mark.parametrize("phase", ["score", "live"])
def test_should_refuse_forged_objective_weights_even_when_target_weight_mass_is_preserved(
    monkeypatch, phase
):
    captured = environment(monkeypatch)
    metadata: list[dict[str, Any]] = []
    original_copy = pilot.deepcopy

    def copy(value, *args, **kwargs):
        result = original_copy(value, *args, **kwargs)
        if (
            not metadata
            and type(value) is dict
            and set(value) == {"objective_id", "weight_rule", "class_weights"}
            and value["objective_id"] == TORCH_RECURRENT_WEIGHTED_CE_ID
        ):
            metadata.append(result)
        return result

    monkeypatch.setattr(pilot, "deepcopy", copy)
    changed: list[bool] = []

    def corrupt():
        values = metadata[0]["class_weights"][0]["values"]
        episode = captured["collections"][0].episode
        counts = [sum(step.action == action for step in episode.transitions) for action in range(5)]
        left, right = [action for action, count in enumerate(counts) if count][:2]
        before = sum(count * weight for count, weight in zip(counts, values, strict=True))
        delta = min(0.125, values[right] * counts[right] / (2 * counts[left]))
        values[left] += delta
        values[right] -= delta * counts[left] / counts[right]
        assert min(values) > 0
        assert sum(count * weight for count, weight in zip(counts, values, strict=True)) == before
        changed.append(True)

    if phase == "score":
        original_predict = Learner.predict

        def predict(owner, features):
            result = original_predict(owner, features)
            if owner.state.completed_updates == 8 and not changed:
                corrupt()
            return result

        monkeypatch.setattr(Learner, "predict", predict)
    else:
        captured["on_prediction"] = lambda session: corrupt()
    report = pilot.run_weighted_live_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert report["training_objective"] is metadata[0]
    completed = 8 if phase == "score" else 64
    assert report["charged_updates"] == len(captured["weighted_calls"]) == completed
    assert len(RecurrentLearner.trains) == completed
    assert len(captured["sessions"]) == int(phase == "live")
    if phase == "live":
        assert len(captured["sessions"][0].inputs) == 1
        assert report["train_policy_diagnostics"]["status"] == "failed"
    else:
        assert captured["finished"] == captured["diagnostics"] == []


@pytest.mark.parametrize("entry", ["run_pilot", "run_live_pilot"])
def test_should_leave_original_mean_routes_and_metadata_unchanged(monkeypatch, entry):
    captured = environment(monkeypatch)
    report = getattr(pilot, entry)()
    assert report["status"] == "complete"
    assert captured["weighted_calls"] == [] and len(RecurrentLearner.trains) == 64
    assert "training_objective" not in report
    assert ("train_policy_diagnostics" in report) is (entry == "run_live_pilot")


def test_should_describe_separate_weighted_train_live_protocol_without_work(monkeypatch):
    for name in (
        "TorchTrainWeightedLearner",
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    old = pilot.describe_live_protocol()
    protocol = pilot.describe_weighted_live_protocol()
    assert protocol["protocol_id"] == "key_door_weighted_train_only_closed_loop_acquisition_v1"
    assert protocol["planned_work"] == old["planned_work"]
    assert protocol["architecture"] == old["architecture"]
    assert protocol["planned_work"]["native_train_calls"] == 64
    assert protocol["planned_work"]["training_rows"] == 512
    assert protocol["planned_work"]["maximum_native_predict_calls"] == 58


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_weighted_mode_bool_before_resources(monkeypatch, invalid):
    for name in (
        "TorchTrainWeightedLearner",
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(train_policy_diagnostics=True, weighted_train=invalid)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(return_context=True),
        dict(evaluate_policy_after_training=True),
        dict(episode_replay=True),
        dict(fresh_task_probes=True),
        dict(four_phase=True),
        dict(four_phase_policy=True),
        dict(four_phase_replay="coverage"),
        dict(four_phase_probes=True),
    ],
)
def test_should_refuse_weighted_without_live_or_mixed_recipes_before_work(monkeypatch, options):
    for name in (
        "TorchTrainWeightedLearner",
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    kwargs = dict(weighted_train=True, **options)
    if options:
        kwargs["train_policy_diagnostics"] = True
    with pytest.raises(ValueError):
        pilot._run_pilot(**kwargs)


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_weighted_train_live_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-weighted-train-live"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="weighted-train-live")

    monkeypatch.setattr(
        cli, "describe_weighted_train_live_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_weighted_train_live_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_train_live_pilot", forbidden)
    monkeypatch.setattr(cli, "run_train_acquisition_pilot", forbidden)
    cli.main()
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "weighted-train-live"
