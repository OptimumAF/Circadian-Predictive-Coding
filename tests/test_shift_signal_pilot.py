"""Fixed protocol and CLI controls; native execution is separately budgeted."""

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.run_shift_signal_pilot as cli
import src.adapters.shift_signal_pilot as pilot
from src.core.matched_backprop import MatchedBackpropSnapshot


def forbidden(*args, **kwargs):
    pytest.fail("unselected or native work was reached")


def test_should_describe_fixed_controls_without_data_or_native_construction(monkeypatch):
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "build_stream", forbidden)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    report = pilot.describe_protocol()
    assert report["seeds"] == [23]
    assert report["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc"]
    assert report["streams"] == ["stationary", "recurring"]
    assert report["checkpoint_updates"] == list(range(19))
    assert report["planned_work"] == dict(
        benchmark_updates=144,
        training_row_exposures=1152,
        observation_predictions=144,
        final_score_forwards=304,
        predict_coupled_calls=448,
        detector_steps=144,
        probe_updates=0,
        svd_calls=0,
        replay_examples=0,
    )
    assert report["policy"] == dict(
        reference_batches=3,
        window_batches=2,
        persistence_batches=2,
        uncertainty_threshold=0.05,
        feature_threshold=0.25,
        residual_threshold=0.05,
    )
    assert "AFTER" in report["prediction_timing"]


class Poison:
    def __getattribute__(self, name):
        pytest.fail("foreign snapshot fields were accessed")


@pytest.mark.parametrize("kind", ("none", "poison", "namespace"))
def test_should_refuse_foreign_snapshot_before_metadata_or_predict(monkeypatch, kind):
    # Construct hostile objects inside the test, after pytest has generated IDs.
    state = (
        None
        if kind == "none"
        else Poison()
        if kind == "poison"
        else SimpleNamespace(format_version=1)
    )
    monkeypatch.setattr(pilot, "predict_coupled", forbidden)
    invalid: Any = None
    with pytest.raises(ValueError):
        pilot._predict_snapshot(state, invalid)


@pytest.mark.parametrize("version", (True, 0, 2))
def test_should_refuse_unknown_snapshot_format_before_predict(monkeypatch, version: Any):
    monkeypatch.setattr(pilot, "predict_coupled", forbidden)
    invalid: Any = None
    state = MatchedBackpropSnapshot(version, invalid)
    with pytest.raises(ValueError):
        pilot._predict_snapshot(state, invalid)


@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_only_the_requested_action(monkeypatch, capsys, action):
    report = {"status": "complete", "scope": "stub"}
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        cli, "describe_protocol" if action == "describe" else "run_pilot", lambda: report
    )
    monkeypatch.setattr(sys, "argv", ["shift", f"--{action}"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == report


def test_should_preserve_incomplete_report_and_failure_exit_without_retry(monkeypatch, capsys):
    calls = []

    def failed():
        calls.append(True)
        return dict(status="incomplete", reason="retained failure", runs=[])

    monkeypatch.setattr(cli, "run_pilot", failed)
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(sys, "argv", ["shift", "--run"])
    assert cli.main() == 2 and len(calls) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "retained failure"


@pytest.mark.parametrize("args", ([], ["--run", "--describe"], ["--run", "--seed", "23"]))
def test_should_refuse_missing_conflicting_or_unfrozen_controls(monkeypatch, args):
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(sys, "argv", ["shift", *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("phase", ("constructor", "enter"))
def test_should_preserve_all_planned_observations_on_global_setup_failure(monkeypatch, phase):
    class FailedSampler:
        def __init__(self):
            if phase == "constructor":
                raise RuntimeError("declared constructor failure")

        def __enter__(self):
            raise RuntimeError("declared enter failure")

        def __exit__(self, *args):
            return False

        def snapshot(self):
            raise RuntimeError("no memory observation")

    monkeypatch.setattr(pilot, "ProcessRssSampler", FailedSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["benchmark_completed_updates"] == 0
    assert len(report["runs"]) == 8
    rows = [o for r in report["runs"] for o in r["observations"]["rows"]]
    assert len(rows) == 144 and all(o["status"] == "not_run" for o in rows)
    assert report["observation_predictions_attempted"] == 0
    assert report["pc_neutral_exact"] == {"stationary": None, "recurring": None}


def test_should_add_contextual_alert_annotations_without_changing_signal_rows():
    observations = {
        "rows": [
            dict(updates=n, status="complete", decision=dict(alarms=(True, False, False)))
            for n in (5, 8, 14)
        ]
    }
    stationary = pilot._signal_outcomes(observations, "stationary")
    recurring = pilot._signal_outcomes(observations, "recurring")
    assert stationary["uncertainty"]["stationary_false_alerts"] == [5, 8, 14]
    assert recurring["uncertainty"]["first_phase_false_alerts"] == [5]
    assert recurring["uncertainty"]["B_delay_updates"] == 1
    assert recurring["uncertainty"]["return_delay_updates"] == 1
    assert recurring["feature"]["B_delay_updates"] is None
    assert [o["updates"] for o in observations["rows"]] == [5, 8, 14]
