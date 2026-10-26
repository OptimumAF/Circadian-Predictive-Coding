"""Opt-in calibrated protocol/CLI boundary without native learner work."""

import json
from pathlib import Path
import sys

import pytest

import scripts.run_shift_signal_pilot as cli
import src.adapters.shift_signal_pilot as pilot


def forbidden(*args, **kwargs):
    pytest.fail("native work or unselected route reached")


def test_should_preserve_original_protocol_exactly_without_calibration():
    archived = json.loads(
        (Path(__file__).parent / "fixtures" / "shift_signal_protocol_v1.json").read_text(
            encoding="utf8"
        )
    )
    assert pilot.describe_protocol() == pilot.describe_protocol(calibrated=False) == archived


def test_should_freeze_calibration_access_and_costs_before_any_native_work(monkeypatch):
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "build_stream", forbidden)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    default = pilot.describe_protocol()
    report = pilot.describe_protocol(calibrated=True)
    assert report["protocol_id"] != default["protocol_id"]
    assert report["calibration"] == dict(prefix_batches=6, margin=1e-6)
    for key in ("seeds", "methods", "streams", "learning_rate", "inference_steps", "resources"):
        assert report[key] == default[key]
    assert report["planned_work"] == dict(
        benchmark_updates=144,
        training_row_exposures=1152,
        observation_predictions=144,
        final_score_forwards=304,
        predict_coupled_calls=448,
        detector_steps=192,
        probe_updates=0,
        svd_calls=0,
        replay_examples=0,
        calibration_fits=8,
        calibration_batches=48,
        calibration_rows_in_training_total=384,
        replayed_summary_observations=48,
        monitoring_observations=96,
    )
    assert "no final data" in report["calibration_population"]
    assert "before monitoring" in report["calibration_freeze"]
    assert "not parameter attestation" in report["calibration_identity"]
    assert "no false-positive" in report["limitations"]


@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_calibrated_mode_explicitly_without_default_run(monkeypatch, capsys, action):
    calls = []

    def selected(*, calibrated):
        calls.append(calibrated)
        return dict(status="complete", selected="calibrated")

    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_protocol" if action == "describe" else "run_pilot", selected)
    monkeypatch.setattr(sys, "argv", ["shift", f"--{action}", "--calibrated"])
    assert cli.main() == 0
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "calibrated"


def test_should_keep_all_calibration_and_monitor_slots_on_global_setup_failure(monkeypatch):
    def fail_setup():
        raise RuntimeError("declared sampler setup failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail_setup)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    report = pilot.run_pilot(calibrated=True)
    assert report["status"] == "incomplete"
    assert report["benchmark_completed_updates"] == report["completed_observations"] == 0
    assert len(report["runs"]) == 8
    for row in report["runs"]:
        observations = row["observations"]
        assert observations["calibration"]["result"] is None
        assert row["method"] in observations["calibration"]["model_id"]
        assert len(observations["rows"]) == 18
        assert all(o["status"] == "not_run" for o in observations["rows"])
        assert all(o["arrived_row_ids"] is None for o in observations["rows"])


@pytest.mark.parametrize("invalid", (None, 1, "true"))
def test_should_refuse_ambiguous_calibration_mode_before_native_work(monkeypatch, invalid):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError, match="boolean"):
        pilot.run_pilot(calibrated=invalid)
