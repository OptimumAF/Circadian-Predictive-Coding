"""Ordinary sensor method composition with fake native math, real stream/state ports."""

import json
from typing import Any

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import sensor_drift_demo, sensor_learners
from src.core import coupled_energy
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledTrainResult
from src.core.sensor_stream import make_sensor_stream


NEW_METHODS = ("pc", "neutral_cpc", "gated_cpc")


@pytest.fixture
def fake_coupled_math(monkeypatch):
    trace: list[Any] = []

    def train(model, features, targets, rate, steps, inference_rate):
        trace.append(("train", features.copy(), targets.copy(), rate, steps, inference_rate))
        model._parameters.output_bias[:] += 0.01
        if type(model) is CoupledCircadianPredictiveCodingNetwork:
            model._wake_updates += 1
        return CoupledTrainResult(0.0)

    def predict(parameters, features):
        count = int(round(float(parameters.output_bias[0, 0]) * 100))
        trace.append(("predict", count, features.copy()))
        return np.full((len(features), 1), 0.5 + count * 0.01)

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "train_epoch", train)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "train_epoch", train)
    monkeypatch.setattr(coupled_energy, "predict_coupled", predict)
    return trace


@pytest.mark.parametrize("method", NEW_METHODS)
@pytest.mark.parametrize("monitoring", [False, True])
def test_should_run_each_existing_method_on_real_prearrival_stream(
    method, monitoring, fake_coupled_math
):
    result = sensor_drift_demo.run_demo(method=method, monitoring=monitoring)
    assert result["status"] == "complete", result["reason"]
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert result["charged_training_updates"] == 12
    assert result["pending_forecasts"] == []
    assert result["pending_prediction_batches"] == 0
    assert result["protocol"]["comparison"] is result["protocol"]["final_evaluation"] is False
    predictions = [e for e in fake_coupled_math if e[0] == "predict"]
    training = [e for e in fake_coupled_math if e[0] == "train"]
    assert len(predictions) == len(training) == len(result["rows"]) == 12
    fixture = make_sensor_stream()
    for tick, (prediction, update, row) in enumerate(zip(predictions, training, result["rows"])):
        assert prediction[1] == row["prediction_state_updates"] == max(0, tick - 2)
        assert row["feature_tick"] == tick and row["label_tick"] == tick + 2
        assert row["arrival_update"] == tick + 1
        assert row["row_ids"] == [f"sensor-{tick}-{i}" for i in range(4)]
        assert row["probabilities"] == [[0.5 + max(0, tick - 2) * 0.01]] * 4
        np.testing.assert_array_equal(prediction[2], fixture[tick].features)
        np.testing.assert_array_equal(update[1], fixture[tick].features)
        np.testing.assert_array_equal(update[2], fixture[tick].targets)
        assert update[3:] == (0.1, 4, 0.2)
    if monitoring:
        assert result["sensor_monitoring"]["complete"] is True
        assert result["sensor_monitoring"]["native_forward_calls"] == 0
    else:
        assert "sensor_monitoring" not in result


@pytest.mark.parametrize("method", NEW_METHODS)
def test_should_preserve_paid_forecasts_and_charges_when_native_update_fails(
    method, fake_coupled_math, monkeypatch
):
    native_type = (
        CoupledPredictiveCodingNetwork
        if method == "pc"
        else CoupledCircadianPredictiveCodingNetwork
    )
    original = native_type.train_epoch
    attempts = 0

    def fail(model, *args):
        nonlocal attempts
        attempts += 1
        if attempts == 6:
            raise RuntimeError("injected coupled native failure")
        return original(model, *args)

    monkeypatch.setattr(native_type, "train_epoch", fail)
    result = sensor_drift_demo.run_demo(method=method, monitoring=True)
    assert result["status"] == "incomplete"
    assert "injected coupled native failure" in result["reason"]
    # Existing wake budget counts completed receipts; the failed native attempt
    # is separate paid physical work, retained by this trace/native wrapper.
    assert result["charged_training_updates"] == 5
    assert attempts == 6
    assert len(result["rows"]) == 5
    assert result["diagnostics"]["prediction_calls_completed"] == 8
    assert len(result["pending_forecasts"]) == result["pending_prediction_batches"] == 3
    assert [r["feature_tick"] for r in result["pending_forecasts"]] == [5, 6, 7]
    for row in result["pending_forecasts"]:
        assert "targets" not in row and "features" not in row
        assert len(row["probabilities"]) == 4


@pytest.mark.parametrize("method", NEW_METHODS)
@pytest.mark.parametrize("monitoring", [False, True])
def test_should_describe_method_objective_and_cost_without_training(method, monitoring, capsys):
    arguments = ["--describe", "--method", method] + (["--monitoring"] if monitoring else [])
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == sensor_drift_demo.describe_protocol(method=method, monitoring=monitoring)
    assert result["method"] == method
    assert result["training_updates"] == result["training_latent_initializations"] == 12
    assert result["training_forward_backward_evaluations"] == 72
    assert result["training_evaluated_rows"] == 288
    assert (result["inference_steps"], result["inference_learning_rate"]) == (4, 0.2)
    assert result["inference_policy"] is None and result["precisions"] == [1.0]
    if method != "pc":
        assert result["chemistry"]["min_plasticity"] == (1.0 if method == "neutral_cpc" else 0.2)


@pytest.mark.parametrize("method", [None, True, 0, "cpc", "PC", "", ["pc"]])
def test_should_refuse_invalid_methods_before_work(method):
    with pytest.raises(ValueError, match="method"):
        sensor_drift_demo.describe_protocol(method=method)
    with pytest.raises(ValueError, match="method"):
        sensor_drift_demo.run_demo(method=method)


@pytest.mark.parametrize("method", NEW_METHODS)
@pytest.mark.parametrize("control", ["scheduled", "replay"])
def test_should_refuse_unqualified_control_combinations_before_work(method, control):
    with pytest.raises(ValueError, match="require method bp"):
        sensor_drift_demo.run_demo(method=method, **{control: True})
    with pytest.raises(SystemExit):
        main(["--describe", "--method", method, "--" + control])


@pytest.mark.parametrize("method", NEW_METHODS)
def test_should_execute_actual_cli_dispatch_and_new_file_output(
    method, fake_coupled_math, tmp_path
):
    path = tmp_path / "result.json"
    assert main(["--method", method, "--monitoring", "--output", str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf8"))
    assert report["status"] == "complete" and report["protocol"]["method"] == method


def test_should_refuse_a_bp_snapshot_in_coupled_observation(fake_coupled_math, monkeypatch):
    original = sensor_learners.build_sensor_learner
    monkeypatch.setattr(
        sensor_learners, "build_sensor_learner", lambda method, p: original("bp", p)
    )
    result = sensor_drift_demo.run_demo(method="pc")
    assert result["status"] == "incomplete"
    assert "full native coupled snapshot" in result["reason"]
    assert result["charged_training_updates"] == 0
    assert fake_coupled_math == []
