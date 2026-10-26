"""Closed-loop public history with explicitly fake TRAIN-only native memory.

The lookup head exercises causal inputs, checkpoint ancestry and success/failure
reporting. Its behavior is a test control, never evidence of native acquisition.
"""

from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import key_door_pilot as shared
from src.adapters import key_door_public_history_pilot as pilot
from src.adapters import key_door_public_history_session as history_session
from src.adapters.key_door_learnability import _bindings
from test_key_door_pilot import Learner as FeatureLearner, fake_environment
from test_torch_feature_rehearsal_pilot import forbidden


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
    fake_environment(monkeypatch)
    evidence: dict[str, Any] = dict(
        sources=[],
        streams=[],
        trains=[],
        readouts=[],
        native_frames=[],
        helpers=[],
        collections=[],
        benchmark_seal=None,
        after_live_predict=None,
        before_live=None,
        wrong_cue=False,
        fail_live_at=None,
        force_loop=False,
    )
    original_collect = shared._collect
    original_stream = pilot.AttributionStream
    original_live = shared._live

    class Learner(FeatureLearner):
        def __init__(self, policy):
            assert (policy.feature_dim, policy.hidden_dim, policy.seed, policy.num_classes) == (
                16,
                8,
                17,
                5,
            )
            super().__init__(policy)
            evidence["sources"].append(self)

        def train_batch(self, features, targets):
            assert features.shape == (8, 16) and targets.shape == (8, 1)
            assert not evidence["helpers"], "training resumed during closed-loop readout"
            result = super().train_batch(features, targets)
            memory = dict(self.state.optimizer_state or {})
            for row, target in zip(features, targets[:, 0], strict=True):
                memory[tuple(float(value) for value in row)] = int(target)
            self.state.optimizer_state = memory
            evidence["trains"].append((self.policy.method, features.copy(), targets.copy()))
            return result

        def predict(self, features):
            assert features.dtype == np.float64 and features.ndim == 2 and features.shape[1] == 16
            assert len(evidence["trains"]) == 32
            live = len(features) == 1
            if live:
                assert len(evidence["readouts"]) == 120
                if evidence["fail_live_at"] == len(evidence["native_frames"]) + 1:
                    raise RuntimeError("declared live predictor failed")
            else:
                assert len(features) == 8 and not evidence["helpers"]
            query = features.copy()
            if live and evidence["wrong_cue"]:
                # Deliberately wrong fake policy: swap only its own public cue
                # interpretation; the environment and reported input stay real.
                query[:, 6] *= -1
                query[:, 14] *= -1
            memory = self.state.optimizer_state or {}
            result = np.full((len(query), 5), 0.01, dtype=np.float64)
            for index, row in enumerate(query):
                if live and evidence["force_loop"]:
                    # A fake pure-input policy walks only floor tiles, never
                    # interacts, and loops. It receives no legal-action list.
                    action = next(action for action, tile in enumerate(row[8:12]) if tile == 0)
                else:
                    action = memory.get(tuple(float(value) for value in row), 0)
                result[index, action] = 0.96
            if live:
                evidence["native_frames"].append(
                    (self.policy.method, self.state.completed_updates, features.copy())
                )
                if evidence["after_live_predict"] is not None:
                    evidence["after_live_predict"](self)
            else:
                evidence["readouts"].append((self.policy.method, self.state.completed_updates))
            return result

    def collect(identity, layout, cue, role):
        assert role == "train", "live qualification constructed final/probe packets"
        batch = original_collect(identity, layout, cue, role)
        evidence["collections"].append(batch)
        return batch

    def stream(source, training, heldout, **options):
        assert heldout == () and options["training_only"] is True
        instance = original_stream(source, training, heldout, **options)
        evidence["streams"].append(instance)
        return instance

    def live(row, source, states, budget, **options):
        assert len(evidence["trains"]) == 32 and len(evidence["readouts"]) == 120
        assert options["public_history"] is True and callable(options["observe_budget"])
        assert len(row["episodes"]) == len(states) == 1
        if evidence["benchmark_seal"] is None:
            evidence["benchmark_seal"] = benchmark_signature(evidence, include_clock=False)
        record = dict(row=row, source=source, states=states, budget=budget, options=options)
        evidence["helpers"].append(record)
        if evidence["before_live"] is not None:
            evidence["before_live"](record)
        return original_live(row, source, states, budget, **options)

    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)
    monkeypatch.setattr(
        pilot, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    monkeypatch.setattr(
        history_session,
        "_same_state",
        lambda left, right: pickle.dumps(left) == pickle.dumps(right),
    )
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_live", live)
    monkeypatch.setattr(shared, "build_stream", forbidden)
    monkeypatch.setattr(shared, "build_probe_stream", forbidden)
    return evidence


def benchmark_signature(evidence, *, include_clock):
    values = []
    for stream in evidence["streams"]:
        binding = _bindings(stream._budget, None)[0]
        if not include_clock:
            binding = binding[:8] + binding[9:]
        values.append(
            (
                stream._states,
                stream._runtime.candidate_snapshot().state,
                stream._runtime.applied_updates,
                binding,
            )
        )
    return pickle.dumps(values)


def assert_partial_slots(report):
    assert len(report["runs"]) == 4
    assert all(len(row["live_episodes"]) == 30 for row in report["runs"])
    assert all(row["wake_updates"] == 8 for row in report["runs"])
    assert all(len(row["training_points"]) == 5 for row in report["runs"])


def test_should_preserve_benchmark_and_execute_causal_fresh_episodes_at_every_checkpoint(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    report = pilot.run_live_pilot()
    assert report["status"] == "complete", report["global_errors"]
    assert len(evidence["sources"]) == len(evidence["streams"]) == 4
    assert len(evidence["trains"]) == 32 and len(evidence["readouts"]) == 120
    assert len(evidence["collections"]) == 8 and len(evidence["helpers"]) == 120
    assert benchmark_signature(evidence, include_clock=False) == evidence["benchmark_seal"]
    assert report["final_data_constructed"] is False
    assert report["pc_neutral_exact"] is report["pc_neutral_live_exact"] is True
    assert "probe_demonstrations" not in report
    frame_index = 0
    for row, source, stream in zip(
        report["runs"], evidence["sources"], evidence["streams"], strict=True
    ):
        assert row["status"] == "complete"
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert source.state.completed_updates == source.state.training_examples == 0
        assert stream._runtime.candidate_snapshot().state.completed_updates == 8
        assert stream._runtime.candidate_snapshot().state.training_examples == 64
        assert (
            row["stream_result"]["completed_updates"] == 8
            and row["stream_result"]["training_examples"] == 64
        )
        assert len(row["stream_result"]["applied_updates"]) == stream._budget.updates_completed == 8
        assert row["work"]["score_forwards_completed"] == 30
        assert len(row["live_episodes"]) == row["live_sessions_started"] == 30
        for episode in row["live_episodes"]:
            assert episode["source_role"] == "train" and episode["context"] in ("A", "B", "C")
            assert episode["status"] == "complete" and episode["state_unchanged"] is True
            assert episode["learning_active"] is False and episode["max_steps"] == 12
            assert "action_conditioning" not in episode
            assert (
                episode["prediction_calls_attempted"]
                == episode["prediction_calls_returned"]
                == len(episode["transitions"])
            )
            initial = episode["transitions"][0]["observation"]
            assert len(initial) == 8 and initial[6] == episode["cue"]
            for transition in episode["transitions"]:
                method, native, encoded = evidence["native_frames"][frame_index]
                frame_index += 1
                assert method == row["method"] and native == episode["checkpoint_updates"]
                np.testing.assert_array_equal(
                    encoded, np.array([initial + transition["observation"]])
                )
                assert transition["action"] == int(np.argmax(transition["probabilities"]))
            counter = episode["history_session_work"]
            assert counter["session_construction_attempts"] == counter["sessions_started"] == 1
            assert (
                counter["attempted_steps"]
                == counter["encoded_frames"]
                == counter["native_prediction_returns"]
                == counter["completed_history_steps"]
                == episode["prediction_calls_returned"]
            )
        assert (
            row["live_work"]["session_construction_attempts"]
            == row["live_work"]["sessions_started"]
            == 30
        )
        for name in COUNTERS:
            assert row["live_work"][name] == sum(
                episode["history_session_work"][name] for episode in row["live_episodes"]
            )
        for point in POINTS:
            episodes = [
                episode
                for episode in row["live_episodes"]
                if episode["checkpoint_updates"] == point
            ]
            assert len(episodes) == 6
            expected = sum(episode["success"] for episode in episodes) / 6
            actual = next(
                outcome for outcome in row["live_outcomes"] if outcome["updates"] == point
            )
            assert actual["success_rate"] == expected
            assert actual["mean_return"] == sum(episode["return"] for episode in episodes) / 6
            assert (
                actual["mean_steps"] == sum(episode["steps_executed"] for episode in episodes) / 6
            )
            assert (
                actual["illegal_action_rate"]
                == sum(episode["outcome"] == "illegal_action" for episode in episodes) / 6
            )
            assert (
                actual["step_limit_rate"]
                == sum(episode["outcome"] == "step_limit" for episode in episodes) / 6
            )
        assert [point["success_rate"] for point in row["live_outcomes"]] == [0, 1 / 3, 2 / 3, 1, 1]
    assert frame_index == len(evidence["native_frames"])
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False


def test_should_keep_real_wrong_key_negative_returns_from_a_deliberately_wrong_fake_policy(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    evidence["wrong_cue"] = True
    report = pilot.run_live_pilot()
    assert report["status"] == "complete", report["global_errors"]
    for row in report["runs"]:
        endpoint = [
            episode for episode in row["live_episodes"] if episode["checkpoint_updates"] == 8
        ]
        assert len(endpoint) == 6
        assert all(
            episode["outcome"] == "wrong_key" and episode["return"] == -1 for episode in endpoint
        )
        assert all(
            episode["success"] is False and episode["terminal"] is True for episode in endpoint
        )
        assert row["live_outcomes"][-1]["mean_return"] == -1
        assert row["live_outcomes"][-1]["success_rate"] == 0


@pytest.mark.parametrize("fault", ("prior_output", "source", "clock"))
def test_should_stop_changed_prior_output_source_or_budget_clock_and_keep_paid_returns(
    monkeypatch, fault
):
    evidence = environment(monkeypatch)
    changed = []
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
            if fault == "source":
                evidence["sources"][0].state.completed_updates = 99
            else:
                evidence["streams"][0]._budget.last_clock += 1.0
            changed.append(True)

        evidence["after_live_predict"] = corrupt
    report = pilot.run_live_pilot()
    assert changed and report["status"] == "incomplete"
    assert_partial_slots(report)
    assert report["runs"][0]["status"] == "failed"
    assert len(evidence["native_frames"]) > 0
    assert all(
        episode["status"] == "not_run"
        for row in report["runs"][1:]
        for episode in row["live_episodes"]
    )
    assert report["runs"][0]["live_work"]["native_prediction_returns"] == len(
        evidence["native_frames"]
    )
    if fault == "prior_output":
        assert report["runs"][0]["live_episodes"][0]["return"] == 77.0
    else:
        assert len(evidence["native_frames"]) == 1


def test_should_preserve_returned_and_censored_work_when_second_native_prediction_raises(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    evidence["fail_live_at"] = 2
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete"
    assert_partial_slots(report)
    episode = report["runs"][0]["live_episodes"][0]
    assert episode["status"] == "failed"
    assert episode["prediction_calls_attempted"] == 2 and episode["prediction_calls_returned"] == 1
    assert episode["steps_executed"] == 1
    counters = episode["history_session_work"]
    assert counters["attempted_steps"] == counters["encoded_frames"] == 2
    assert counters["native_prediction_returns"] == counters["completed_history_steps"] == 1
    assert counters["failed_work_may_be_unobserved"] is True
    assert len(evidence["native_frames"]) == 1
    assert report["runs"][0]["live_work"]["native_prediction_returns"] == 1
    assert all(item["status"] == "not_run" for item in report["runs"][0]["live_episodes"][1:])


def test_should_keep_a_real_twelve_step_timeout_without_fabricating_aggregate_outcomes(monkeypatch):
    evidence = environment(monkeypatch)
    evidence["force_loop"] = True

    def stop_after_first_episode(record):
        if len(evidence["helpers"]) == 2:
            raise RuntimeError("bounded fixture stops after one timeout")

    evidence["before_live"] = stop_after_first_episode
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete"
    episode = report["runs"][0]["live_episodes"][0]
    assert episode["status"] == "complete" and episode["outcome"] == "step_limit"
    assert episode["steps_executed"] == episode["prediction_calls_returned"] == 12
    assert episode["return"] == 0 and episode["terminal"] is episode["success"] is False
    assert episode["history_session_work"]["completed_history_steps"] == 12
    assert len(evidence["native_frames"]) == 12
    assert report["runs"][0]["live_outcomes"] is None
    assert all(item["status"] == "not_run" for item in report["runs"][0]["live_episodes"][2:])


def test_should_preserve_failed_session_construction_without_any_native_frame(monkeypatch):
    evidence = environment(monkeypatch)

    def fail(predictor):
        raise RuntimeError("owned public session construction failed")

    monkeypatch.setattr(shared, "PublicHistoryPredictionSession", fail)
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete"
    assert_partial_slots(report)
    episode = report["runs"][0]["live_episodes"][0]
    assert episode["status"] == "failed"
    counters = episode["history_session_work"]
    assert counters["session_construction_attempts"] == 1 and counters["sessions_started"] == 0
    assert all(counters[name] == 0 for name in COUNTERS[2:])
    assert counters["failed_work_may_be_unobserved"] is True
    assert report["runs"][0]["live_sessions_started"] == 0
    assert not evidence["native_frames"]


def test_should_keep_default_train_only_route_free_of_live_or_probe_work(monkeypatch):
    evidence = environment(monkeypatch)
    monkeypatch.setattr(shared, "_live", forbidden)
    monkeypatch.setattr(pilot, "run_checkpoint_probe", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "complete"
    assert len(evidence["trains"]) == 32 and len(evidence["readouts"]) == 120
    assert not evidence["helpers"] and not evidence["native_frames"]
    assert all("live_episodes" not in row and "probes" not in row for row in report["runs"])
    assert pilot.describe_protocol() == pilot.describe_protocol(with_live=False)
    assert pilot.describe_probe_protocol() == pilot.describe_protocol(
        with_probes=True, with_live=False
    )


@pytest.mark.parametrize("invalid", (None, 0, 1, "true", np.bool_(True)))
@pytest.mark.parametrize("describe", (False, True))
def test_should_refuse_non_bool_live_mode_before_resources(monkeypatch, invalid, describe):
    monkeypatch.setattr(shared, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(with_live=invalid)


@pytest.mark.parametrize("describe", (False, True))
def test_should_refuse_joint_probe_live_mode_before_resources(monkeypatch, describe):
    monkeypatch.setattr(shared, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(with_probes=True, with_live=True)


def test_should_describe_fixed_train_only_live_matrix_and_costs_without_execution(monkeypatch):
    ordinary = deepcopy(pilot.describe_protocol())
    probes = deepcopy(pilot.describe_probe_protocol())
    for name in ("require_torch", "ProcessRssSampler", "_live", "build_probe_stream"):
        monkeypatch.setattr(shared, name, forbidden)
    monkeypatch.setattr(pilot, "build_training_stream", forbidden)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", forbidden)
    protocol = pilot.describe_live_protocol()
    assert protocol == pilot.describe_protocol(with_live=True)
    assert protocol["protocol_id"] == "local_key_door_matched_public_history_live_v1"
    assert protocol["methods"] == list(shared.METHODS)
    assert protocol["feature_dim"] == 16 and protocol["hidden_dim"] == 8
    assert protocol["num_classes"] == 5 and protocol["seed"] == 17
    assert protocol["live"]["checkpoints"] == POINTS
    assert protocol["live"]["layouts"] == ["A", "B", "C"]
    assert protocol["live"]["cues"] == [1, -1]
    assert protocol["live"]["episodes_per_method"] == 30 and protocol["live"]["max_steps"] == 12
    assert "raw argmax" in protocol["live"]["action_rule"]
    assert protocol["planned_work"] == dict(
        native_updates=32,
        training_rows=256,
        score_calls=120,
        score_rows=960,
        teacher_episodes=8,
        teacher_physics_steps=64,
        live_episodes=120,
        maximum_live_prediction_calls=1440,
        maximum_live_physics_steps=1440,
        maximum_physics_steps=1504,
        maximum_live_native_forks=240,
    )
    assert ordinary == pilot.describe_protocol()
    assert probes == pilot.describe_probe_protocol()


@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_the_public_history_live_cli_only(monkeypatch, capsys, describe):
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_public_history_live_protocol", selected)
    monkeypatch.setattr(cli, "run_public_history_live_pilot", selected)
    monkeypatch.setattr(cli, "run_public_history_probe_pilot", forbidden)
    monkeypatch.setattr(cli, "run_public_history_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-public-history-live"]
        + (["--describe"] if describe else []),
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
def test_should_refuse_caller_artifact_or_instance_selection_before_live_cli_dispatch(
    monkeypatch, flags
):
    monkeypatch.setattr(cli, "run_public_history_live_pilot", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["run", "--scenario", "torch-key-door-public-history-live", *flags]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
