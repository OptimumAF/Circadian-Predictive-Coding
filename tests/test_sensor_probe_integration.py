"""Sensor/probe integration through real stream/fork ports and fake native SGD."""

import json
from typing import Any

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import coupled_learners, matched_backprop_learner, sensor_drift_demo
from src.core import coupled_energy, coupled_predictive_coding, matched_backprop
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.learner_ports import TrainingDiagnostic


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


@pytest.fixture
def fake_math(monkeypatch):
    trace: list[Any] = []

    def train(learner, features, targets):
        trace.append(("train", len(features), features.copy(), targets.copy()))
        learner._model._parameters.output_bias[:] += 0.001
        if type(learner._model) is CoupledCircadianPredictiveCodingNetwork:
            learner._model._wake_updates += 1
        return TrainingDiagnostic("fake_sensor_probe_v1", 0.0)

    def predict(parameters, features):
        trace.append(("predict", len(features)))
        return np.full((len(features), 1), 0.5 + float(parameters.output_bias[0, 0]))

    def forbid(*args, **kwargs):
        raise AssertionError("native SGD forbidden in integration gate")

    for module in (coupled_energy, coupled_predictive_coding, matched_backprop):
        monkeypatch.setattr(module, "predict_coupled", predict)
    for kind in (matched_backprop_learner.MatchedBackpropLearner, coupled_learners.CoupledLearner):
        monkeypatch.setattr(kind, "train_batch", train)
    for native_kind in (
        matched_backprop.MatchedBackpropNetwork,
        coupled_predictive_coding.CoupledPredictiveCodingNetwork,
        CoupledCircadianPredictiveCodingNetwork,
    ):
        monkeypatch.setattr(native_kind, "train_epoch", forbid)
    return trace


@pytest.mark.parametrize("method", METHODS)
def test_should_add_disposable_probes_without_changing_any_benchmark_row(method, fake_math):
    baseline = sensor_drift_demo.run_demo(method=method, monitoring=True)
    fake_math.clear()
    observed = sensor_drift_demo.run_demo(method=method, monitoring=True, learnability=True)
    assert observed["status"] == baseline["status"] == "complete", observed["reason"]
    assert observed["rows"] == baseline["rows"]
    assert observed["stream"] == baseline["stream"]
    assert observed["sensor_monitoring"] == baseline["sensor_monitoring"]
    assert observed["source_unchanged"] is observed["actor_unchanged"] is True
    assert observed["charged_training_updates"] == 12
    learning = observed["learnability"]
    assert learning["status"] == "complete" and learning["charged_probe_updates"] == 16
    assert [o["updates"] for o in learning["observations"]] == [0, 4, 8, 12]
    assert all(o["status"] == "complete" for o in learning["observations"])
    assert all(
        [p["updates"] for p in o["probe"]["checkpoints"]] == [0, 1, 2, 4]
        for o in learning["observations"]
    )
    assert sum(e[:2] == ("train", 4) for e in fake_math) == 12
    assert sum(e[:2] == ("train", 8) for e in fake_math) == 16
    assert sum(e[:2] == ("predict", 4) for e in fake_math) == 12
    assert sum(e[:2] == ("predict", 32) for e in fake_math) == 16


@pytest.mark.parametrize("method", METHODS)
def test_should_retain_probe_failure_paid_forecasts_and_unreached_slots(
    method, fake_math, monkeypatch
):
    from src.adapters import sensor_learnability

    original = sensor_learnability.run_fresh_task_probe
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return dict(
                status="failed",
                reason="injected probe failure",
                checkpoints=[],
                attempted_updates=0,
                completed_updates=0,
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(sensor_learnability, "run_fresh_task_probe", fail)
    report = sensor_drift_demo.run_demo(method=method, monitoring=True, learnability=True)
    assert report["status"] == "incomplete" and "injected probe failure" in report["reason"]
    assert calls == 2 and report["charged_training_updates"] == 4
    assert len(report["rows"]) == 4
    assert report["diagnostics"]["prediction_calls_completed"] == 6
    assert [r["feature_tick"] for r in report["pending_forecasts"]] == [4, 5]
    assert all("targets" not in r and "features" not in r for r in report["pending_forecasts"])
    learning = report["learnability"]
    assert learning["charged_probe_updates"] == 4 and learning["status"] == "failed"
    assert [o["status"] for o in learning["observations"]] == [
        "complete",
        "failed",
        "not_run",
        "not_run",
    ]


@pytest.mark.parametrize("method", METHODS)
def test_should_preview_fixed_learnability_and_forward_flag_to_actual_cli(
    method, capsys, tmp_path, fake_math
):
    flags = ["--method", method, "--learnability", "--monitoring"]
    assert main(["--describe", *flags]) == 0
    descriptor = json.loads(capsys.readouterr().out)
    assert descriptor["total_training_updates"] == 28 and descriptor["total_training_rows"] == 176
    assert descriptor["learnability"]["probe_checkpoint_updates"] == [0, 1, 2, 4]
    assert fake_math == []
    path = tmp_path / "new.json"
    assert main([*flags, "--output", str(path)]) == 0
    assert json.loads(path.read_text(encoding="utf8"))["learnability"]["status"] == "complete"


@pytest.mark.parametrize("flag", ["scheduled", "replay"])
def test_should_refuse_control_combinations_before_model_work(flag, fake_math):
    options: dict[str, Any] = {flag: True}
    with pytest.raises(ValueError, match="learnability requires ordinary"):
        sensor_drift_demo.run_demo(learnability=True, **options)
    with pytest.raises(SystemExit):
        main(["--describe", "--learnability", "--" + flag])
    assert fake_math == []


@pytest.mark.parametrize("value", [None, 0, 1, "true", []])
def test_should_refuse_non_boolean_learnability_before_work(value, fake_math):
    with pytest.raises(ValueError, match="exact bools"):
        sensor_drift_demo.run_demo(learnability=value)
    assert fake_math == []
