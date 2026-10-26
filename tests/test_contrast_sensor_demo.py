"""Contrast inversion through the existing sensor stream, with fake native math."""

from copy import deepcopy
import json
from typing import Any

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import sensor_drift_demo
from src.adapters.sensor_learnability import SensorLearnability
from src.app.coupled_attribution import AttributionStream
from src.core import coupled_energy, coupled_predictive_coding, matched_backprop
import src.core.sensor_stream as stream_fixture
import src.core.sensor_retention as final_fixture
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.sensor_stream import make_sensor_stream
from src.core.sensor_retention import make_sensor_heldout
from test_sensor_heldout import fake_math as fake_math


CASES: list[dict[str, Any]] = [
    dict(method=method) for method in ("bp", "pc", "neutral_cpc", "gated_cpc")
]
CASES += [dict(scheduled=True), dict(replay=True), dict(replay=True, incident_replay=True)]


def forbidden(*args, **kwargs):
    pytest.fail("unselected or native math reached")


@pytest.fixture
def contrast_math(fake_math, monkeypatch):
    fake_math.update(training=[], predictions=[], streams=[])
    bp = matched_backprop.MatchedBackpropNetwork.train_epoch
    pc = coupled_predictive_coding.CoupledPredictiveCodingNetwork.train_epoch
    predict = coupled_energy.predict_coupled
    initialize = AttributionStream.__init__

    def bp_train(model, features, targets, *args, **kwargs):
        fake_math["training"].append((features.copy(), targets.copy()))
        return bp(model, features, targets, *args, **kwargs)

    def pc_train(model, features, targets, *args, **kwargs):
        fake_math["training"].append((features.copy(), targets.copy()))
        return pc(model, features, targets, *args, **kwargs)

    def prediction(parameters, features):
        fake_math["predictions"].append((fake_math["updates"], features.copy()))
        return predict(parameters, features)

    def stream(self, learner, training, final, *args, **kwargs):
        fake_math["streams"].append(deepcopy((training, final)))
        initialize(self, learner, training, final, *args, **kwargs)

    def health(self, state, row, guard):
        # The real probe/fork/guard composition remains; native health/SVD is
        # outside this adapter integration test's zero-native-work allowance.
        guard()
        row.update(
            feature_health=[],
            diagnostic_seconds=0.0,
            diagnostic_activation_bytes=0,
            checkpoint_snapshot_array_bytes=0,
            importance=None,
            neuron_age=None,
            unavailable_reason="fake native health boundary",
        )
        guard()

    monkeypatch.setattr(matched_backprop.MatchedBackpropNetwork, "train_epoch", bp_train)
    monkeypatch.setattr(
        coupled_predictive_coding.CoupledPredictiveCodingNetwork, "train_epoch", pc_train
    )
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "train_epoch", pc_train)
    for module in (coupled_energy, coupled_predictive_coding, matched_backprop):
        monkeypatch.setattr(module, "predict_coupled", prediction)
    monkeypatch.setattr(AttributionStream, "__init__", stream)
    monkeypatch.setattr(SensorLearnability, "_measure_health", health)
    return fake_math


@pytest.mark.parametrize("options", CASES)
def test_should_use_contrast_training_and_final_arrays_in_all_seven_existing_modes(
    options, contrast_math
):
    expected_updates = (
        3
        if options.get("scheduled")
        else 16
        if options.get("incident_replay")
        else 22
        if options.get("replay")
        else 12
    )
    contrast_math["expected_updates"] = expected_updates
    report = sensor_drift_demo.run_demo(
        monitoring=True, heldout=True, contrast_inversion=True, **options
    )
    assert report["status"] == "complete", report["reason"]
    assert contrast_math["updates"] == expected_updates
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    training, final = contrast_math["streams"][0]
    assert [batch.context for batch in training] == ["A"] * 4 + ["B"] * 4 + ["A"] * 4
    assert [batch.role for batch in training] == ["train"] * 12
    assert [batch.context for batch in final] == ["A", "B"]
    assert [batch.role for batch in final] == ["final_test"] * 2
    oracle = make_sensor_stream(contrast_inversion=True)
    for tick, batch in enumerate(training):
        np.testing.assert_array_equal(batch.features, oracle[tick].features)
        np.testing.assert_array_equal(batch.targets, oracle[tick].targets)
        expected = 1.0 - training[0].features if 4 <= tick < 8 else training[0].features
        np.testing.assert_array_equal(batch.features, expected)
        np.testing.assert_array_equal(batch.targets, training[0].targets)
    heldout = make_sensor_heldout(contrast_inversion=True)
    for actual, expected in zip(final, heldout, strict=True):
        np.testing.assert_array_equal(actual.features, expected.features)
        np.testing.assert_array_equal(actual.targets, expected.targets)
    np.testing.assert_array_equal(final[1].features, 1.0 - final[0].features)
    np.testing.assert_array_equal(final[1].targets, final[0].targets)
    assert contrast_math["final_calls"] == report["stream"]["score_forward_calls"] == 26
    assert report["heldout_evaluation"]["progress"]["prediction_rows_completed"] == 832
    assert len(report["heldout_evaluation"]["training_receipts"]) == 12
    final_predictions = [
        (updates, features)
        for updates, features in contrast_math["predictions"]
        if len(features) == 32
    ]
    assert len(final_predictions) == 26 and all(
        updates == expected_updates for updates, _ in final_predictions
    )
    for index, (_, features) in enumerate(final_predictions):
        np.testing.assert_array_equal(features, final[index % 2].features)
    if options.get("incident_replay"):
        assert report["replay_budget_charged_exposures"] == 4
        assert [event["status"] for event in report["rehearsal"]["events"]] == [
            "skipped",
            "complete",
            "skipped",
        ]


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_finish_separate_reserved_probes_before_final_scoring_without_altering_benchmark(
    method, contrast_math
):
    baseline = sensor_drift_demo.run_demo(method=method, monitoring=True, contrast_inversion=True)
    contrast_math.update(updates=0, final_calls=0, expected_updates=28)
    observed = sensor_drift_demo.run_demo(
        method=method, monitoring=True, contrast_inversion=True, heldout=True, learnability=True
    )
    assert baseline["status"] == observed["status"] == "complete", observed["reason"]
    assert baseline["rows"] == observed["rows"]
    assert baseline["sensor_monitoring"] == observed["sensor_monitoring"]
    assert baseline["stream"]["applied_updates"] == observed["stream"]["applied_updates"]
    assert observed["learnability"]["charged_probe_updates"] == 16
    assert [row["updates"] for row in observed["learnability"]["observations"]] == [0, 4, 8, 12]
    assert all(row["status"] == "complete" for row in observed["learnability"]["observations"])
    assert contrast_math["final_calls"] == 26 and contrast_math["updates"] == 28
    assert observed["source_unchanged"] is observed["actor_unchanged"] is True


def test_should_keep_paid_partial_final_scores_and_probe_charges_on_failure(contrast_math):
    contrast_math.update(fail_final=4, expected_updates=28)
    report = sensor_drift_demo.run_demo(
        monitoring=True, contrast_inversion=True, heldout=True, learnability=True
    )
    assert report["status"] == "incomplete" and "injected final" in report["reason"]
    evidence = report["heldout_evaluation"]
    progress = evidence["progress"]
    assert progress["status"] == "failed"
    assert (
        progress["prediction_calls_attempted"] == 4 and progress["prediction_calls_completed"] == 3
    )
    assert (
        progress["prediction_rows_attempted"] == 128 and progress["prediction_rows_completed"] == 96
    )
    assert set(progress["checkpoints"][0]["scores"]) == {"A", "B"}
    assert set(progress["checkpoints"][1]["scores"]) == {"A"}
    assert evidence["retention"] is None and len(evidence["training_receipts"]) == 12
    assert report["charged_training_updates"] == 12
    assert report["learnability"]["charged_probe_updates"] == 16


def test_should_refuse_finals_when_the_requested_contrast_probe_report_is_incomplete(
    contrast_math, monkeypatch
):
    original = SensorLearnability.report

    def partial(self):
        report = original(self)
        report["status"] = "partial"
        return report

    monkeypatch.setattr(SensorLearnability, "report", partial)
    report = sensor_drift_demo.run_demo(contrast_inversion=True, heldout=True, learnability=True)
    assert report["status"] == "incomplete"
    assert contrast_math["final_calls"] == 0
    assert report["learnability"]["charged_probe_updates"] == 16
    assert report["heldout_evaluation"]["progress"]["status"] == "not_run"


def test_should_isolate_each_final_prediction_input_from_saved_contrast_arrays(
    contrast_math, monkeypatch
):
    original = matched_backprop.MatchedBackpropNetwork.predict_proba

    def mutate(model, features):
        result = original(model, features)
        if len(features) == 32:
            features.fill(-999)
        return result

    monkeypatch.setattr(matched_backprop.MatchedBackpropNetwork, "predict_proba", mutate)
    report = sensor_drift_demo.run_demo(contrast_inversion=True, heldout=True)
    assert report["status"] == "complete", report["reason"]
    assert contrast_math["final_calls"] == 26
    _, final = contrast_math["streams"][0]
    oracle = make_sensor_heldout(contrast_inversion=True)
    for batch, expected in zip(final, oracle, strict=True):
        np.testing.assert_array_equal(batch.features, expected.features)
    assert report["source_unchanged"] is report["actor_unchanged"] is True


def test_should_preserve_returned_prediction_costs_without_a_fabricated_metric_for_nan(
    contrast_math, monkeypatch
):
    original = matched_backprop.MatchedBackpropNetwork.predict_proba

    def malformed(model, features):
        return np.full((32, 1), np.nan) if len(features) == 32 else original(model, features)

    monkeypatch.setattr(matched_backprop.MatchedBackpropNetwork, "predict_proba", malformed)
    report = sensor_drift_demo.run_demo(contrast_inversion=True, heldout=True)
    assert report["status"] == "incomplete"
    progress = report["heldout_evaluation"]["progress"]
    assert progress["prediction_calls_attempted"] == progress["prediction_calls_completed"] == 1
    assert progress["checkpoints"][0]["scores"] == {}
    assert report["heldout_evaluation"]["retention"] is None
    assert report["charged_training_updates"] == 12


@pytest.mark.parametrize("options", CASES)
def test_should_describe_the_new_condition_without_changing_recipes_or_readout_costs(
    options, contrast_math
):
    original = sensor_drift_demo.describe_protocol(monitoring=True, heldout=True, **options)
    contrast = sensor_drift_demo.describe_protocol(
        monitoring=True, heldout=True, contrast_inversion=True, **options
    )
    assert contrast["protocol"].startswith("sensor_contrast_inversion_")
    assert contrast["contrast_inversion"] is True
    assert contrast["phases"] == ["A", "contrast_inversion", "A_return"]
    assert contrast["heldout"]["contrast_inversion"] is True
    assert "brightness_shift" not in contrast["heldout"]
    for field in (
        "seed",
        "dimensions",
        "learning_rate",
        "training_updates",
        "training_rows",
        "diagnostic_forward_calls",
        "monitoring",
    ):
        assert contrast[field] == original[field]
    assert contrast["heldout"]["readout_forward_calls"] == 26
    assert contrast["heldout"]["readout_forward_rows"] == 832
    assert contrast_math["updates"] == contrast_math["final_calls"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"monitoring": True},
        {"scheduled": True},
        {"replay": True},
        {"monitoring": True, "stationary": True},
        {"monitoring": True, "replay": True, "incident_replay": True},
        {"method": "pc", "heldout": True, "learnability": True},
    ],
)
def test_should_preserve_old_brightness_and_stationary_descriptors_with_explicit_false(options):
    assert sensor_drift_demo.describe_protocol(**options) == sensor_drift_demo.describe_protocol(
        **options, contrast_inversion=False
    )


def test_should_preserve_old_zero_keyword_fixture_and_heldout_calls_when_false(
    contrast_math, monkeypatch
):
    calls = []
    training, final = stream_fixture.make_sensor_stream, final_fixture.make_sensor_heldout

    def stream(*args, **kwargs):
        calls.append((args, kwargs))
        return training(*args, **kwargs)

    def heldout(*args, **kwargs):
        calls.append((args, kwargs))
        return final(*args, **kwargs)

    monkeypatch.setattr(stream_fixture, "make_sensor_stream", stream)
    monkeypatch.setattr(final_fixture, "make_sensor_heldout", heldout)
    report = sensor_drift_demo.run_demo(heldout=True, contrast_inversion=False)
    assert report["status"] == "complete", report["reason"]
    assert calls == [((), {}), ((), {})]


@pytest.mark.parametrize("entry", [sensor_drift_demo.describe_protocol, sensor_drift_demo.run_demo])
@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_contrast_before_model_or_fixture_work(entry, value, monkeypatch):
    monkeypatch.setattr(stream_fixture, "make_sensor_stream", forbidden)
    monkeypatch.setattr(coupled_predictive_coding, "CoupledPredictiveCodingNetwork", forbidden)
    with pytest.raises(ValueError, match="bool"):
        entry(contrast_inversion=value)


@pytest.mark.parametrize("entry", [sensor_drift_demo.describe_protocol, sensor_drift_demo.run_demo])
def test_should_refuse_simultaneous_stationary_and_contrast_before_work(entry, monkeypatch):
    monkeypatch.setattr(stream_fixture, "make_sensor_stream", forbidden)
    with pytest.raises(ValueError):
        entry(stationary=True, monitoring=True, contrast_inversion=True)


@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_the_explicit_contrast_cli_flag_only(describe, monkeypatch, capsys):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(sensor_drift_demo, "describe_protocol", selected if describe else forbidden)
    monkeypatch.setattr(sensor_drift_demo, "run_demo", forbidden if describe else selected)
    assert (
        main(
            [
                *(["--describe"] if describe else []),
                "--contrast-inversion",
                "--monitoring",
                "--heldout",
            ]
        )
        == 0
    )
    assert calls == [dict(contrast_inversion=True, monitoring=True, heldout=True)]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


def test_should_refuse_both_sensor_conditions_in_cli_before_dispatch(monkeypatch):
    monkeypatch.setattr(sensor_drift_demo, "describe_protocol", forbidden)
    monkeypatch.setattr(sensor_drift_demo, "run_demo", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["--describe", "--stationary", "--monitoring", "--contrast-inversion"])
    assert error.value.code == 2
