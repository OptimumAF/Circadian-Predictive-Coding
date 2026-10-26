"""Finish-time isolation and paid partial evidence with fake native updates."""

import json
from typing import Any
from copy import deepcopy

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import sensor_drift_demo
from src.app import coupled_attribution
from src.app.toy_execution_budget import ToyBudgetSession
from src.core import coupled_energy, coupled_predictive_coding, matched_backprop
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_predictive_coding import CoupledTrainResult
from src.core.matched_backprop import MatchedBackpropEvaluation


CASES = [dict(method=m, learnability=True) for m in ("bp", "pc", "neutral_cpc", "gated_cpc")]
CASES += [dict(scheduled=True), dict(replay=True)]


@pytest.fixture
def fake_math(monkeypatch):
    trace: dict[str, Any] = dict(updates=0, final_calls=0, fail_final=None, expected_updates=None)

    def bp_train(model, features, targets, rate):
        trace["updates"] += 1
        model._parameters.output_bias[:] += 0.001
        return MatchedBackpropEvaluation(0.0, model.snapshot_parameters())

    def pc_train(model, features, targets, rate, steps, inference_rate):
        trace["updates"] += 1
        model._parameters.output_bias[:] += 0.001
        if type(model) is CoupledCircadianPredictiveCodingNetwork:
            model._wake_updates += 1
        return CoupledTrainResult(0.0)

    def predict(parameters, features):
        if len(features) == 32 and np.all(features >= 0):
            trace["final_calls"] += 1
            if trace["expected_updates"] is not None:
                assert trace["updates"] == trace["expected_updates"]
            if trace["final_calls"] == trace["fail_final"]:
                raise RuntimeError("injected final inference failure")
        return np.full((len(features), 1), 0.5 + float(parameters.output_bias[0, 0]))

    monkeypatch.setattr(matched_backprop.MatchedBackpropNetwork, "train_epoch", bp_train)
    monkeypatch.setattr(
        coupled_predictive_coding.CoupledPredictiveCodingNetwork, "train_epoch", pc_train
    )
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "train_epoch", pc_train)
    for module in (coupled_energy, coupled_predictive_coding, matched_backprop):
        monkeypatch.setattr(module, "predict_coupled", predict)
    return trace


@pytest.mark.parametrize("options", CASES)
def test_should_read_final_contexts_only_after_training_and_probes_without_changing_them(
    options, fake_math
):
    baseline = sensor_drift_demo.run_demo(monitoring=True, **options)
    fake_math.update(
        updates=0,
        final_calls=0,
        expected_updates=3 if options.get("scheduled") else 22 if options.get("replay") else 28,
    )
    observed = sensor_drift_demo.run_demo(monitoring=True, heldout=True, **options)
    assert observed["status"] == baseline["status"] == "complete", observed["reason"]
    assert observed["rows"] == baseline["rows"]
    assert observed["sensor_monitoring"] == baseline["sensor_monitoring"]
    assert observed["stream"]["applied_updates"] == baseline["stream"]["applied_updates"]
    assert observed["source_unchanged"] is observed["actor_unchanged"] is True
    assert fake_math["final_calls"] == 26 and observed["stream"]["score_forward_calls"] == 26
    evidence = observed["heldout_evaluation"]
    assert evidence["progress"]["status"] == "complete"
    assert evidence["progress"]["prediction_rows_completed"] == 832
    assert len(evidence["training_receipts"]) == 12
    assert [p["updates"] for p in evidence["retention"]["endpoints"]] == [0, 4, 8, 12]
    if options.get("learnability"):
        assert observed["learnability"]["costs"]["completed_updates"] == 16
        for a, b in zip(
            baseline["learnability"]["observations"],
            observed["learnability"]["observations"],
            strict=True,
        ):
            assert a["probe"]["checkpoints"] == b["probe"]["checkpoints"]


def test_should_keep_partial_scores_receipts_and_costs_on_later_final_failure(fake_math):
    fake_math.update(fail_final=4, expected_updates=28)
    report = sensor_drift_demo.run_demo(heldout=True, learnability=True, monitoring=True)
    assert report["status"] == "incomplete" and "injected final" in report["reason"]
    evidence = report["heldout_evaluation"]
    progress = evidence["progress"]
    assert progress["status"] == "failed"
    assert (progress["prediction_calls_attempted"], progress["prediction_calls_completed"]) == (
        4,
        3,
    )
    assert (progress["prediction_rows_attempted"], progress["prediction_rows_completed"]) == (
        128,
        96,
    )
    assert [p["status"] for p in progress["checkpoints"]] == ["complete", "incomplete"]
    assert set(progress["checkpoints"][0]["scores"]) == {"A", "B"}
    assert set(progress["checkpoints"][1]["scores"]) == {"A"}
    assert evidence["retention"] is None and len(evidence["training_receipts"]) == 12
    assert report["charged_training_updates"] == 12
    assert report["learnability"]["charged_probe_updates"] == 16
    assert len(report["rows"]) == 12 and report["pending_forecasts"] == []


def test_should_refuse_final_data_if_requested_probe_report_is_incomplete(fake_math, monkeypatch):
    from src.adapters.sensor_learnability import SensorLearnability

    original = SensorLearnability.report

    def incomplete(observer):
        result = original(observer)
        result["status"] = "partial"
        return result

    monkeypatch.setattr(SensorLearnability, "report", incomplete)
    report = sensor_drift_demo.run_demo(heldout=True, learnability=True)
    assert report["status"] == "incomplete" and "probes incomplete before final" in report["reason"]
    assert fake_math["final_calls"] == 0
    assert report["heldout_evaluation"]["progress"]["status"] == "not_run"
    assert report["learnability"]["charged_probe_updates"] == 16


def test_should_mark_readout_failed_if_a_guard_fails_after_all_predictions(fake_math, monkeypatch):
    finished = [False]
    original_score = coupled_attribution._score_checkpoints
    original_boundary = ToyBudgetSession.before_final

    def score(*args, **kwargs):
        result = original_score(*args, **kwargs)
        finished[0] = True
        return result

    def boundary(budget):
        if finished[0]:
            raise RuntimeError("post-readout boundary failed")
        return original_boundary(budget)

    monkeypatch.setattr(coupled_attribution, "_score_checkpoints", score)
    monkeypatch.setattr(ToyBudgetSession, "before_final", boundary)
    report = sensor_drift_demo.run_demo(heldout=True)
    progress = report["heldout_evaluation"]["progress"]
    assert report["status"] == "incomplete" and progress["status"] == "failed"
    assert "post-readout boundary" in progress["reason"]
    assert progress["prediction_calls_completed"] == 26
    assert len(progress["checkpoints"]) == 13
    assert all(p["status"] == "complete" for p in progress["checkpoints"])


def test_should_keep_returned_score_if_verified_prediction_mutates_its_model(
    fake_math, monkeypatch
):
    original = matched_backprop.MatchedBackpropNetwork.predict_proba

    def mutate(model, features):
        result = original(model, features)
        if len(features) == 32:
            model._parameters.output_bias[:] += 0.01
        return result

    monkeypatch.setattr(matched_backprop.MatchedBackpropNetwork, "predict_proba", mutate)
    report = sensor_drift_demo.run_demo(heldout=True)
    progress = report["heldout_evaluation"]["progress"]
    assert (
        report["status"] == "incomplete" and "differs from its captured state" in report["reason"]
    )
    assert progress["prediction_calls_completed"] == 1
    assert set(progress["checkpoints"][0]["scores"]) == {"A"}


@pytest.mark.parametrize("options", CASES)
def test_should_forward_new_flag_and_freeze_costs_in_preview(options, capsys, fake_math):
    args = ["--describe", "--heldout"]
    for key, value in options.items():
        args.extend(["--" + key] + ([value] if key == "method" else []))
    assert main(args) == 0
    descriptor = json.loads(capsys.readouterr().out)
    assert descriptor["final_evaluation"] is True
    assert descriptor["heldout"]["readout_forward_calls"] == 26
    assert descriptor["heldout"]["readout_forward_rows"] == 832
    assert fake_math["updates"] == fake_math["final_calls"] == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true", []])
def test_should_refuse_non_boolean_heldout_before_work(value, fake_math):
    with pytest.raises(ValueError, match="exact bools"):
        sensor_drift_demo.run_demo(heldout=value)
    assert fake_math["updates"] == fake_math["final_calls"] == 0


@pytest.mark.parametrize(
    "progress", [{"status": "spent", "prediction_calls_completed": 2}, [], True]
)
def test_should_refuse_reused_or_foreign_progress_without_resetting_evidence(progress):
    before = deepcopy(progress)
    unavailable: Any = None
    with pytest.raises(ValueError, match="fresh empty exact dict"):
        coupled_attribution._score_checkpoints(
            unavailable, [], (), (), unavailable, score_progress=progress
        )
    assert progress == before


def test_should_keep_return_counts_without_fabricating_metric_for_invalid_predictions(
    fake_math, monkeypatch
):
    original = matched_backprop.MatchedBackpropNetwork.predict_proba

    def invalid(model, features):
        return (
            np.full((len(features), 1), np.nan)
            if len(features) == 32
            else original(model, features)
        )

    monkeypatch.setattr(matched_backprop.MatchedBackpropNetwork, "predict_proba", invalid)
    report = sensor_drift_demo.run_demo(heldout=True)
    progress = report["heldout_evaluation"]["progress"]
    assert report["status"] == "incomplete" and progress["status"] == "failed"
    assert progress["prediction_calls_attempted"] == progress["prediction_calls_completed"] == 1
    assert progress["checkpoints"][0]["scores"] == {}
    assert report["heldout_evaluation"]["retention"] is None
