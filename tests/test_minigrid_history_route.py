"""Opt-in public-history composition preserves the original recurrent control."""

import json
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from test_minigrid_learnability import ProbeEnvironment, environment
from test_minigrid_learning_pilot import Learner, METHODS, POINTS, evaluator as evaluator
from test_torch_feature_rehearsal_pilot import forbidden


def history_environment(monkeypatch, evaluator):
    case = environment(monkeypatch, evaluator)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)

    class CueFrameEnvironment(ProbeEnvironment):
        def __init__(self, layout, cue, *args, **kwargs):
            super().__init__(layout, cue, *args, **kwargs)
            # Distinct actually supplied first frames make an episode-reset mixup
            # observable without reading cue metadata in the encoder.
            self.observation = (1.0 if cue == "blue" else 0.0,) + self.observation[1:]

    monkeypatch.setattr(pilot, "MiniGridKeyDoor", CueFrameEnvironment)
    return case


def test_should_use_296_per_episode_for_all_feedforward_arms_and_keep_rnn_raw_148(
    monkeypatch, evaluator
):
    case = history_environment(monkeypatch, evaluator)
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "complete", report["reason"]
    assert report["protocol"]["feature_dim_by_method"] == {
        method: 148 if method == "recurrent_bc" else 296 for method in METHODS
    }
    assert len(case.work["trains"]) == 800 and len(case.work["predicts"]) == 230
    assert len(case.captured["finished"]) == 5 and len(case.probes) == 10
    assert report["probe_data"]["input_array_bytes"] == 193104
    assert report["probe_data"]["history_input_array_bytes"] == 384912
    assert report["probe_data"]["retained_input_array_bytes"] == 578016
    assert all(
        len(record["transitions"][0]["observation"]) == 148 for record in report["training_sources"]
    )
    assert len(report["training_sources"]) == 6
    for (_, source, packets, budget), row in zip(
        case.captured["streams"], report["methods"], strict=True
    ):
        width = 148 if row["method"] == "recurrent_bc" else 296
        assert source.policy.feature_dim == width and source.policy.seed == 17
        assert len(packets) == 128 and all(
            packet.features.shape == (9, width) for packet in packets
        )
        assert budget.updates_completed == row["charged_updates"] == 128
        assert source.state.completed_updates == 0
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert [slot["updates"] for slot in row["readouts"]] == list(POINTS)
        assert row["final_evaluation"]["status"] == "not_run"
        assert row["probe_gain_change"] == 0
        for packet in packets:
            if width == 296:
                np.testing.assert_array_equal(
                    packet.features[:, :148], np.repeat(packet.features[0:1, :148], 9, axis=0)
                )
                assert (
                    packet.episode.transitions[0].observation[:148]
                    == packet.episode.transitions[0].observation[148:]
                )
    for call in case.probes:
        width = call["source"].policy.feature_dim
        expected_bytes = 193104 if width == 148 else 384912
        assert (
            call["report"]["status"] == "complete"
            and call["report"]["reserved_data_bytes"] == expected_bytes
        )
        assert call["budget"].updates_completed == 16
        assert len(call["training"]) == 16 and len(call["evaluation"]) == 2
        assert call["inputs_before"] == call["inputs_after"]
        assert (
            call["state_before"] == call["state_after"]
            and call["source_before"] == call["source_after"]
        )
        for packet in (*call["training"], *call["evaluation"]):
            assert packet.features.shape == (9, width)
            expected_first = 1.0 if packet.context == "E-blue" else 0.0
            if width == 296:
                assert packet.features[:, 0].tolist() == [expected_first] * 9
                assert packet.features[0, 148] == expected_first
                assert packet.features[1, 148] == -1.0
            else:
                assert packet.features[0, 0] == expected_first and packet.features[1, 0] == -1.0
    assert (
        sum(call["report"]["reserved_data_bytes"] for call in case.probes)
        == 8 * 384912 + 2 * 193104
    )
    assert not case.work.get("environments") and not case.work.get("sessions")
    assert report["work"]["native_train_calls"] == 800 and report["work"]["training_rows"] == 7200
    assert report["work"]["known_raw_score_forwards_completed"] == 150
    assert report["work"]["probe_score_forwards_completed"] == 80
    assert report["scientific_advantage_verified"] is False


def test_should_refuse_changed_derived_history_packet_after_first_paid_update(
    monkeypatch, evaluator
):
    case = history_environment(monkeypatch, evaluator)
    transform = pilot._history_packets
    derived: list[Any] = []
    changed: list[bool] = []

    def capture(packets):
        value = transform(packets)
        derived.append(value)
        return value

    def mutate(owner):
        if not changed:
            assert len(derived) == 2 and len(derived[1]) == 128
            derived[1][0].features[0, 0] = 0.123
            changed.append(True)

    monkeypatch.setattr(pilot, "_history_packets", capture)
    case.work["on_train"] = mutate
    report = pilot.run_pilot(public_history=True)
    assert report["status"] == "incomplete" and changed == [True]
    assert "data changed" in report["reason"]
    first = report["methods"][0]
    assert first["charged_updates"] == 1 and first["native_training_examples"] == 9
    assert len(first["partial_stream"]["applied_updates"]) == 1
    assert first["source_unchanged"] is first["actor_unchanged"] is True
    assert len(case.work["trains"]) == 1 and not case.raw and not case.probes
    assert not case.work.get("probe_environments")
    assert report["comparisons"] is None


def test_should_describe_history_resource_free_and_preserve_explicit_default_false(monkeypatch):
    for name in (
        "require_torch",
        "ProcessRssSampler",
        "MiniGridKeyDoor",
        "_history_packets",
        "_history_learner",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    original = pilot.describe_protocol()
    assert original == pilot.describe_protocol(public_history=False)
    assert original["feature_dim"] == 148 and "public_history" not in original
    history = pilot.describe_protocol(public_history=True)
    assert json.loads(json.dumps(history, allow_nan=False)) == history
    assert history["protocol_id"] != original["protocol_id"]
    assert history["feature_dim_by_method"] == {
        method: 148 if method == "recurrent_bc" else 296 for method in METHODS
    }
    assert history["public_history"]["parameters_by_method"] == {
        method: 1319 if method == "recurrent_bc" else 2439 for method in METHODS
    }
    assert history["public_history"]["retained_probe_array_bytes"] == 578016
    assert history["planned_work"] == original["planned_work"]


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_history_mode_before_any_resources(monkeypatch, invalid):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(public_history=invalid)
    with pytest.raises(ValueError):
        pilot.run_pilot(public_history=invalid)


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("describe", [False, True])
def test_should_route_history_flag_only_for_new_explicit_cli(
    monkeypatch, capsys, history, describe
):
    scenario = "torch-minigrid-history-learnability" if history else "torch-minigrid-learnability"
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario] + (["--describe"] if describe else [])
    )
    calls: list[dict[str, Any]] = []

    def selected(**kwargs):
        calls.append(kwargs)
        return dict(status="complete", selected=scenario)

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_minigrid_pilot", forbidden)
    cli.main()
    assert calls == ([dict(public_history=True)] if history else [{}])
    assert json.loads(capsys.readouterr().out)["selected"] == scenario
