"""Real bounded count memory through the existing stream and tiny live game."""

import builtins
from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import key_door_memory_pilot as pilot
from src.adapters import key_door_pilot as shared
from src.adapters import key_door_public_history_pilot as history
from src.adapters import key_door_public_history_session as history_session
from src.adapters.key_door_learnability import _bindings
from src.app.coupled_attribution import _same_value
from src.app import toy_execution_budget as budgets
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


POINTS = [0, 2, 4, 6, 8]
COUNTERS = (
    "session_construction_attempts",
    "sessions_started",
    "attempted_steps",
    "encoded_frames",
    "native_prediction_returns",
    "completed_history_steps",
)


def environment(monkeypatch):
    """Keep the production algorithm; inject only resource ports and observers."""
    evidence: dict[str, Any] = dict(
        sources=[],
        streams=[],
        budgets=[],
        trains=[],
        attempts=[],
        scores=[],
        frames=[],
        collections=[],
        helpers=[],
        training=(),
        benchmark_seal=None,
        before_live=None,
        after_live_predict=None,
        fail_train_at=None,
        fail_live_at=None,
        torch_imports=[],
    )
    native_class = pilot.ActionMemoryLearner
    original_train = native_class.train_batch
    original_predict = native_class.predict
    original_stream = pilot.AttributionStream
    original_live = shared._live
    original_collect = shared._collect
    original_session = shared._session
    original_import = builtins.__import__
    clock = lambda: 1.0

    def import_without_torch(name: str, *args: Any, **options: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            evidence["torch_imports"].append(name)
            pytest.fail("count-memory runtime attempted to import Torch")
        return original_import(name, *args, **options)

    def source(*args, **options):
        result = native_class(*args, **options)
        evidence["sources"].append(result)
        return result

    def train(self, features, targets):
        assert not evidence["scores"] and not evidence["helpers"]
        assert features.shape == (8, 16) and targets.shape == (8, 1)
        evidence["attempts"].append((features.copy(), targets.copy()))
        if evidence["fail_train_at"] == len(evidence["attempts"]):
            raise RuntimeError("third storage call failed before the atomic update")
        result = original_train(self, features, targets)
        evidence["trains"].append((features.copy(), targets.copy(), self.snapshot_state(), result))
        return result

    def predict(self, features):
        assert len(evidence["trains"]) == 8
        live = len(features) == 1
        if live:
            assert len(evidence["scores"]) == 30
            if evidence["fail_live_at"] == len(evidence["frames"]) + 1:
                raise RuntimeError("second memory prediction failed before returning")
        else:
            assert len(features) == 8 and not evidence["helpers"]
        state = self.snapshot_state()
        result = original_predict(self, features)
        record = (state.completed_updates, features.copy(), result.copy())
        evidence["frames" if live else "scores"].append(record)
        if live and evidence["after_live_predict"] is not None:
            evidence["after_live_predict"](self)
        return result

    def stream(source, training, heldout, **options):
        assert heldout == () and options["training_only"] is True
        assert options["checkpoint_updates"] == tuple(POINTS) and options["label_delay"] == 1
        evidence["training"] = training
        result = original_stream(source, training, heldout, **options)
        evidence["streams"].append(result)
        return result

    def collect(identity, layout, cue, role):
        assert role == "train" and layout.layout_id in ("A", "B", "C")
        result = original_collect(identity, layout, cue, role)
        evidence["collections"].append(result)
        return result

    def session(*args, **options):
        result = original_session(*args, **options)
        evidence["budgets"].append(result)
        return result

    def live(row, source, states, budget, **options):
        assert len(evidence["trains"]) == 8 and len(evidence["scores"]) == 30
        assert row["method"] == "observation_memory"
        assert len(row["episodes"]) == len(states) == 1
        assert options["public_history"] is True and options["same_state"] is _same_value
        assert callable(options["observe_budget"])
        if evidence["benchmark_seal"] is None:
            evidence["benchmark_seal"] = benchmark_signature(evidence)
        record = dict(row=row, source=source, states=states, budget=budget, options=options)
        evidence["helpers"].append(record)
        if evidence["before_live"] is not None:
            evidence["before_live"](record)
        return original_live(row, source, states, budget, **options)

    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(shared, "monotonic", clock)
    monkeypatch.setattr(shared, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "TorchFeatureLearner", forbidden)
    monkeypatch.setattr(shared, "_same_state", forbidden)
    monkeypatch.setattr(history_session, "_same_state", forbidden)
    monkeypatch.setattr(shared, "build_stream", forbidden)
    monkeypatch.setattr(shared, "build_probe_stream", forbidden)
    monkeypatch.setattr(pilot, "ActionMemoryLearner", source)
    monkeypatch.setattr(native_class, "train_batch", train)
    monkeypatch.setattr(native_class, "predict", predict)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_session", session)
    monkeypatch.setattr(shared, "_live", live)
    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    return evidence


def benchmark_signature(evidence):
    stream = evidence["streams"][0]
    binding = _bindings(stream._budget, None)[0]
    binding = binding[:8] + binding[9:]
    return pickle.dumps(
        (
            stream._states,
            stream._runtime.candidate_snapshot().state,
            stream._runtime.applied_updates,
            binding,
        )
    )


def test_should_learn_retain_and_recall_known_train_traces_with_real_bounded_memory_without_torch(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["global_errors"]
    assert (
        len(report["runs"])
        == len(evidence["sources"])
        == len(evidence["streams"])
        == len(evidence["budgets"])
        == 1
    )
    row, source, stream, budget = (
        report["runs"][0],
        evidence["sources"][0],
        evidence["streams"][0],
        evidence["budgets"][0],
    )
    assert row["method"] == "observation_memory"
    assert report["shared_initial_parameters_exact"] is report["pc_neutral_exact"] is None
    assert report["final_data_constructed"] is False and not evidence["torch_imports"]
    assert (
        report["scientific_advantage_verified"]
        is report["full_R42_complete"]
        is report["full_R43_complete"]
        is False
    )
    assert (
        len(evidence["trains"])
        == len(evidence["collections"])
        == len(report["demonstrations"])
        == 8
    )
    assert report["collected_episodes"] == 8
    assert report["partial_collection_work_may_be_unobserved"] is False
    assert [batch.context for batch in evidence["training"]] == [
        "A-red",
        "A-blue",
        "B-red",
        "B-blue",
        "C-red",
        "C-blue",
        "A-red",
        "A-blue",
    ]
    all_ids = [identity for batch in evidence["training"] for identity in batch.row_ids]
    assert len(all_ids) == len(set(all_ids)) == 64
    for raw, encoded, observed in zip(
        evidence["collections"], evidence["training"], evidence["trains"], strict=True
    ):
        assert raw.role == encoded.role == "train" and raw.row_ids == encoded.row_ids
        np.testing.assert_array_equal(encoded.features[:, :8], np.tile(raw.features[0], (8, 1)))
        np.testing.assert_array_equal(encoded.features[:, 8:], raw.features)
        np.testing.assert_array_equal(encoded.targets, raw.targets)
        np.testing.assert_array_equal(observed[0], encoded.features)
        np.testing.assert_array_equal(observed[1], encoded.targets)
        assert (
            observed[3].definition == "observation_action_count_increment_rows_v1"
            and observed[3].value == 8
        )
    assert (
        row["wake_updates"] == budget.updates_completed == budget.budget.max_training_updates == 8
    )
    assert budget.replay_examples_completed == 0
    assert (
        row["stream_result"]["completed_updates"] == 8
        and row["stream_result"]["training_examples"] == 64
    )
    assert (
        row["stream_result"]["evaluation_scope"] == "training_only"
        and row["stream_result"]["score_forward_calls"] == 0
    )
    assert [
        receipt["update_number"] for receipt in row["stream_result"]["applied_updates"]
    ] == list(range(1, 9))
    assert [point["wake_updates"] for point in row["observations"]] == POINTS
    assert [point["entries"] for point in row["observations"]] == [0, 16, 32, 48, 48]
    assert [point["logical_payload_bytes"] for point in row["observations"]] == [
        0,
        2688,
        5376,
        8064,
        8064,
    ]
    assert [point["conflicting_keys"] for point in row["observations"]] == [0] * 5
    assert [point["native_training_examples"] for point in row["observations"]] == [
        0,
        16,
        32,
        48,
        64,
    ]
    assert (
        source.snapshot_state().entries == ()
        and source.completed_updates == source.training_examples == 0
    )
    candidate = stream._runtime.candidate_snapshot().state
    assert (
        len(candidate.entries) == 48 and sum(sum(entry.counts) for entry in candidate.entries) == 64
    )
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert (
        len(evidence["scores"])
        == row["native_score_returns"]
        == row["work"]["score_forwards_completed"]
        == 30
    )
    assert row["work"]["score_rows_completed"] == 240
    assert [point["updates"] for point in row["training_points"]] == POINTS
    assert row["outcomes"]["new_B_gain"] > 0 and row["outcomes"]["new_C_gain"] > 0
    assert row["outcomes"]["retention_loss"] == row["outcomes"]["return_gain"] == 0
    assert (
        len(evidence["helpers"]) == len(row["live_episodes"]) == row["live_sessions_started"] == 30
    )
    assert [point["success_rate"] for point in row["live_outcomes"]] == [0, 1 / 3, 2 / 3, 1, 1]
    assert (
        row["live_outcomes"][-1]["mean_return"] == 1 and row["live_outcomes"][-1]["mean_steps"] == 8
    )
    frame_index = 0
    for episode in row["live_episodes"]:
        assert episode["source_role"] == "train" and episode["context"] in ("A", "B", "C")
        assert episode["status"] == "complete" and episode["state_unchanged"] is True
        assert episode["max_steps"] == 12 and "action_conditioning" not in episode
        initial = episode["transitions"][0]["observation"]
        for transition in episode["transitions"]:
            native, encoded, prediction = evidence["frames"][frame_index]
            frame_index += 1
            assert native == episode["checkpoint_updates"]
            np.testing.assert_array_equal(encoded, np.array([initial + transition["observation"]]))
            assert transition["probabilities"] == tuple(prediction[0])
            assert transition["action"] == int(np.argmax(prediction[0]))
        if episode["checkpoint_updates"] == 0:
            assert episode["success"] is False
            assert episode["transitions"][0]["probabilities"] == (0.2,) * 5
        if episode["checkpoint_updates"] >= 6:
            assert (
                episode["success"] is True
                and episode["return"] == 1
                and episode["steps_executed"] == 8
            )
        assert episode["history_session_work"]["sessions_started"] == 1
    assert frame_index == len(evidence["frames"])
    for name in COUNTERS:
        assert row["live_work"][name] == sum(
            slot["history_session_work"][name] for slot in row["live_episodes"]
        )
    assert row["live_work"]["native_prediction_returns"] == len(evidence["frames"])
    assert benchmark_signature(evidence) == evidence["benchmark_seal"]


@pytest.mark.parametrize(
    "fault", ("prior_output", "receipt", "malformed_receipt", "clock", "source")
)
def test_should_refuse_corrupted_completed_evidence_and_preserve_original_storage_charges(
    monkeypatch, fault
):
    evidence = environment(monkeypatch)
    changed: list[bool] = []
    if fault == "prior_output":

        def corrupt(record):
            if len(evidence["helpers"]) == 2:
                evidence["helpers"][0]["row"]["episodes"][0]["return"] = 77.0
                changed.append(True)

        evidence["before_live"] = corrupt
    else:

        def corrupt(owner):
            if changed:
                return
            stream = evidence["streams"][0]
            if fault == "receipt":
                stream._runtime._inbox._applied.clear()
            elif fault == "malformed_receipt":
                applied = stream._runtime._inbox._applied
                applied[next(iter(applied))] = object()
            elif fault == "clock":
                stream._budget.last_clock += 1.0
            else:
                evidence["sources"][0].restore_state(stream._runtime.candidate_snapshot().state)
            changed.append(True)

        evidence["after_live_predict"] = corrupt
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert changed and report["status"] == "incomplete" and row["status"] == "failed"
    assert row["wake_updates"] == evidence["budgets"][0].updates_completed == 8
    assert len(row["stream_result"]["applied_updates"]) == 8
    assert len(evidence["trains"]) == 8 and len(evidence["scores"]) == 30
    assert len(row["live_episodes"]) == 30 and row["live_outcomes"] is None
    assert row["live_work"]["native_prediction_returns"] == len(evidence["frames"]) > 0
    if fault == "prior_output":
        assert row["live_episodes"][0]["return"] == 77.0
    else:
        assert len(evidence["frames"]) == 1
    if fault == "malformed_receipt":
        assert row["partial_native_evidence"]["charged_updates"] == 8
        assert row["partial_native_evidence"]["readback_error"]
    assert all(slot["status"] == "not_run" for slot in row["live_episodes"][2:])


def test_should_preserve_two_original_receipts_and_unknown_candidate_on_third_storage_failure(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    evidence["fail_train_at"] = 3
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert len(evidence["attempts"]) == 3 and len(evidence["trains"]) == 2
    assert row["wake_updates"] == evidence["budgets"][0].updates_completed == 2
    partial = row["partial_native_evidence"]
    assert partial["charged_updates"] == 2
    assert [receipt["update_number"] for receipt in partial["applied_updates"]] == [1, 2]
    assert partial["preceding_success_observation"]["native_updates"] == 2
    assert partial["preceding_success_observation"]["native_training_examples"] == 16
    assert partial["current_candidate_unavailable"] is True
    assert not evidence["scores"] and not evidence["helpers"]
    assert all(slot["status"] == "not_run" for slot in row["live_episodes"])


def test_should_preserve_returned_native_frame_and_censored_next_frame_on_live_failure(monkeypatch):
    evidence = environment(monkeypatch)
    evidence["fail_live_at"] = 2
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["wake_updates"] == 8
    episode = row["live_episodes"][0]
    assert episode["status"] == "failed"
    assert episode["prediction_calls_attempted"] == 2 and episode["prediction_calls_returned"] == 1
    assert episode["steps_executed"] == 1
    counter = episode["history_session_work"]
    assert counter["attempted_steps"] == counter["encoded_frames"] == 2
    assert counter["native_prediction_returns"] == counter["completed_history_steps"] == 1
    assert counter["failed_work_may_be_unobserved"] is True
    assert row["live_work"]["native_prediction_returns"] == len(evidence["frames"]) == 1
    assert all(slot["status"] == "not_run" for slot in row["live_episodes"][1:])


def test_should_report_unknown_partial_teacher_collection_without_fabricating_zero_cost(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    original = shared._collect
    calls: list[bool] = []

    def collect(*args, **options):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("third teacher collection failed")
        return original(*args, **options)

    monkeypatch.setattr(shared, "_collect", collect)
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["wake_updates"] == 0
    assert report["collected_episodes"] is None
    assert report["partial_collection_work_may_be_unobserved"] is True
    assert len(evidence["collections"]) == 2 and len(calls) == 3
    assert not evidence["trains"] and not evidence["scores"] and not evidence["helpers"]
    assert all(slot["status"] == "not_run" for slot in row["live_episodes"])


def test_should_describe_same_original_data_and_clocks_with_explicit_different_control_costs(
    monkeypatch,
):
    neural = deepcopy(history.describe_live_protocol())
    monkeypatch.setattr(shared, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(history, "build_training_stream", forbidden)
    monkeypatch.setattr(pilot, "ActionMemoryLearner", forbidden)
    protocol = pilot.describe_protocol()
    assert protocol["protocol_id"] == "local_key_door_bounded_action_memory_v1"
    assert protocol["methods"] == ["observation_memory"]
    assert protocol["policy"] == dict(
        method="observation_memory",
        feature_dim=16,
        num_classes=5,
        max_entries=64,
        max_payload_bytes=10752,
    )
    assert protocol["seed"] == neural["seed"] == 17
    assert protocol["task_sequence"] == neural["task_sequence"]
    assert protocol["checkpoint_updates"] == neural["checkpoint_updates"] == POINTS
    assert protocol["stream"] == dict(training_only=True, heldout=[], label_delay=1, updates=8)
    assert protocol["update_kind"] == "supervised count storage, not SGD"
    assert protocol["planned_work"] == dict(
        storage_updates=8,
        training_rows=64,
        score_calls=30,
        score_rows=240,
        teacher_episodes=8,
        teacher_physics_steps=64,
        live_episodes=30,
        maximum_live_prediction_calls=360,
        maximum_live_physics_steps=360,
        maximum_live_native_forks=60,
    )
    assert "unequal capacity/compute" in protocol["limitations"]
    assert history.describe_live_protocol() == neural


@pytest.mark.parametrize("selected", ("describe_protocol", "run_pilot"))
def test_should_refuse_caller_override_of_frozen_control_capacity_or_data(selected):
    invalid: Any = None
    with pytest.raises(TypeError):
        getattr(pilot, selected)(max_entries=invalid)
    with pytest.raises(TypeError):
        getattr(pilot, selected)(environment_seed=0)


@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_action_memory_cli_without_selecting_neural_history(
    monkeypatch, capsys, describe
):
    calls: list[bool] = []

    def selected():
        calls.append(True)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_action_memory_protocol", selected)
    monkeypatch.setattr(cli, "run_action_memory_pilot", selected)
    monkeypatch.setattr(cli, "run_public_history_live_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "key-door-action-memory"] + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "flags",
    (
        ["--source-artifact", "historical.json"],
        ["--environment-seed", "0", "--heldout-environment-seed", "23"],
    ),
)
def test_should_refuse_artifact_or_seed_selection_before_fixed_memory_cli_dispatch(
    monkeypatch, flags
):
    monkeypatch.setattr(cli, "run_action_memory_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["run", "--scenario", "key-door-action-memory", *flags])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
