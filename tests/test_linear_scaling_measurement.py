"""Independent call-order, data-isolation and bounded measurement controls.

All spy cases replace the production update kernel. The sole native control
performs exactly three schemes times two updates, with no manual training oracle.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import src.app.linear_scaling_measurement as measurement
from src.app.linear_scaling_measurement import ScalingBatch, run_linear_scaling_case
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.linear_pc_reference import evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    LinearScalingStep,
    effective_linear_weights,
    initialize_linear_scaling,
)


def batches():
    return (
        ScalingBatch(
            "train",
            ("train-0", "train-1"),
            np.array([[0.2, -0.3], [0.7, 0.1]]),
            np.array([[-0.4], [1.2]]),
        ),
        ScalingBatch(
            "final_test",
            ("test-0", "test-1"),
            np.array([[1.1, 0.6], [-0.8, 0.4]]),
            np.array([[2.3], [-1.4]]),
        ),
    )


def state(scheme="mean_field"):
    return initialize_linear_scaling(LinearScalingConfig(scheme, 2, 3, 1), 41)


def session(updates=2, clock=lambda: 0.0, wall=None):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=updates, max_wall_seconds=wall), clock
    )


def run(initial=None, training=None, heldout=None, **options):
    defaults = batches()
    return run_linear_scaling_case(
        state() if initial is None else initial,
        defaults[0] if training is None else training,
        defaults[1] if heldout is None else heldout,
        updates=options.pop("updates", 2),
        learning_rate=options.pop("learning_rate", 0.01),
        budget=options.pop("budget", session()),
        **options,
    )


class Spy:
    def __init__(self, monkeypatch, training, heldout):
        self.events = []
        self.training, self.heldout = training, heldout
        self.time = 0.0
        self.steps = 0
        self.fail_step = None
        self.fail_diagnostic = None
        self.expire_last_score = False
        self.score_calls = 0
        monkeypatch.setattr(measurement, "step_linear_pc", self.step)
        monkeypatch.setattr(measurement, "evaluate_linear_equilibrium", self.evaluate)
        monkeypatch.setattr(measurement, "predict_linear", self.predict)

    def clock(self):
        return self.time

    def step(self, initial, features, targets, learning_rate):
        self.steps += 1
        self.events.append(("step", features.copy(), targets.copy()))
        self.time += 2
        if self.steps == self.fail_step:
            raise RuntimeError("deliberate step failure")
        evaluation = evaluate_linear_equilibrium(
            effective_linear_weights(initial), features, targets
        )
        return LinearScalingStep(
            LinearScalingState(initial.config, initial.raw_weights), evaluation
        )

    def evaluate(self, weights, features, targets):
        self.events.append(("diagnostic", features.copy(), targets.copy()))
        self.time += 1
        if len([e for e in self.events if e[0] == "diagnostic"]) == self.fail_diagnostic:
            raise FloatingPointError("deliberate diagnostic failure")
        return evaluate_linear_equilibrium(weights, features, targets)

    def predict(self, weights, features):
        heldout = np.array_equal(features, self.heldout.features)
        self.events.append(("heldout_predict" if heldout else "train_predict", features.copy()))
        self.time += 3
        if heldout:
            self.score_calls += 1
            if self.expire_last_score and self.score_calls == 3:
                self.time = 1000.0
        prediction = features.copy()
        for weight in weights:
            prediction = prediction @ weight
        # Input mutation must never corrupt original batches or later scores.
        features.fill(999)
        return prediction


def test_should_defer_target_free_scores_and_report_all_checkpoint_measurements(monkeypatch):
    training, heldout = batches()
    initial = state()
    original = tuple(w.copy() for w in initial.raw_weights)
    feature_copies = training.features.copy(), heldout.features.copy()
    spy = Spy(monkeypatch, training, heldout)
    budget = session(clock=spy.clock)
    result = run(initial, training, heldout, budget=budget, clock=spy.clock)
    assert result["status"] == "complete"
    assert (
        result["attempted_updates"] == result["committed_updates"] == budget.updates_completed == 2
    )
    assert [row["updates"] for row in result["checkpoints"]] == [0, 1, 2]
    scores = [e for e in spy.events if e[0] == "heldout_predict"]
    assert len(scores) == 3
    assert max(i for i, e in enumerate(spy.events) if e[0] == "step") < min(
        i for i, e in enumerate(spy.events) if e[0] == "heldout_predict"
    )
    for event in spy.events:
        if event[0] in ("step", "diagnostic"):
            np.testing.assert_array_equal(event[1], training.features)
            np.testing.assert_array_equal(event[2], training.targets)
            assert not np.array_equal(event[2], heldout.targets)
    effective = effective_linear_weights(initial)
    expected = evaluate_linear_equilibrium(effective, training.features, training.targets)
    prediction = heldout.features.copy()
    for weight in effective:
        prediction = prediction @ weight
    heldout_loss = float(np.sum((prediction - heldout.targets) ** 2) / (2 * len(prediction)))
    np.testing.assert_allclose(result["heldout_half_mse"], [heldout_loss] * 3, atol=1e-14)
    for checkpoint in result["checkpoints"]:
        assert checkpoint["energy"] == pytest.approx(expected.energy)
        assert checkpoint["train_half_mse"] == pytest.approx(expected.mse)
        assert checkpoint["rescaling"] == pytest.approx(expected.rescaling)
        np.testing.assert_allclose(
            checkpoint["raw_weight_norms"], [np.linalg.norm(w) for w in original]
        )
        np.testing.assert_allclose(
            checkpoint["effective_weight_norms"], [np.linalg.norm(w) for w in effective]
        )
        np.testing.assert_allclose(
            checkpoint["effective_gradient_norms"],
            [np.linalg.norm(g) for g in expected.weight_gradients],
        )
        np.testing.assert_allclose(
            checkpoint["raw_gradient_norms"],
            [
                np.linalg.norm(g * scale)
                for g, scale in zip(
                    expected.weight_gradients, initial.config.forward_scales, strict=True
                )
            ],
        )
        np.testing.assert_allclose(
            checkpoint["equilibrium_layer_rms"], [np.sqrt(np.mean(z * z)) for z in expected.latents]
        )
        outputs = []
        previous = training.features
        for weight in effective:
            previous = previous @ weight
            outputs.append(float(np.sqrt(np.mean(previous * previous))))
        np.testing.assert_allclose(checkpoint["feedforward_layer_rms"], outputs)
    np.testing.assert_array_equal(training.features, feature_copies[0])
    np.testing.assert_array_equal(heldout.features, feature_copies[1])
    for actual, before in zip(initial.raw_weights, original, strict=True):
        np.testing.assert_array_equal(actual, before)
    assert result["training_step_seconds"] == 4
    assert result["diagnostic_seconds"] >= 3
    assert result["scoring_seconds"] == 9
    assert result["elapsed_seconds"] == spy.time
    assert result["elapsed_seconds"] >= sum(
        result[key] for key in ("training_step_seconds", "diagnostic_seconds", "scoring_seconds")
    )


def test_should_measure_large_finite_norms_without_squaring_overflow(monkeypatch):
    initial = LinearScalingState(
        LinearScalingConfig("fan_in", 1, 1, 1), (np.array([[1e200]]), np.zeros((1, 1)))
    )
    training = ScalingBatch("train", ("train",), np.ones((1, 1)), np.zeros((1, 1)))
    heldout = ScalingBatch("final_test", ("test",), np.ones((1, 1)), np.zeros((1, 1)))
    spy = Spy(monkeypatch, training, heldout)
    # Distinct feature values identify reserved score calls in the spy.
    heldout.features[0, 0] = 0.5
    result = run(initial, training, heldout, clock=spy.clock)
    assert result["status"] == "complete"
    for checkpoint in result["checkpoints"]:
        assert checkpoint["raw_weight_norms"][0] == pytest.approx(1e200)
        assert checkpoint["effective_weight_norms"][0] == pytest.approx(1e200)
        assert checkpoint["feedforward_layer_rms"][0] == pytest.approx(1e200)
        assert checkpoint["equilibrium_layer_rms"][0] == pytest.approx(1e200)


@pytest.mark.parametrize("phase", ("step", "diagnostic"))
def test_should_preserve_committed_work_without_scoring_on_training_failure(monkeypatch, phase):
    training, heldout = batches()
    spy = Spy(monkeypatch, training, heldout)
    if phase == "step":
        spy.fail_step = 2
    else:
        spy.fail_diagnostic = 2
    result = run(training=training, heldout=heldout, clock=spy.clock)
    assert result["status"] == "failed"
    assert "deliberate" in result["reason"]
    assert result["committed_updates"] == 1
    assert result["attempted_updates"] == (2 if phase == "step" else 1)
    assert result["heldout_half_mse"] is None
    assert [row["updates"] for row in result["checkpoints"]] == ([0, 1] if phase == "step" else [0])
    assert not any(e[0] == "heldout_predict" for e in spy.events)


def test_should_return_no_scores_if_budget_expires_after_last_prediction(monkeypatch):
    training, heldout = batches()
    spy = Spy(monkeypatch, training, heldout)
    spy.expire_last_score = True
    result = run(
        training=training,
        heldout=heldout,
        budget=session(clock=spy.clock, wall=100),
        clock=spy.clock,
    )
    assert result["status"] == "failed"
    assert result["committed_updates"] == 2
    assert result["heldout_half_mse"] is None
    assert spy.score_calls == 3
    assert len(result["checkpoints"]) == 3


def test_should_stop_update_budget_before_scoring(monkeypatch):
    training, heldout = batches()
    spy = Spy(monkeypatch, training, heldout)
    result = run(training=training, heldout=heldout, budget=session(1), clock=spy.clock)
    assert result["status"] == "failed"
    assert result["committed_updates"] == 1
    assert result["heldout_half_mse"] is None
    assert spy.steps == 1
    assert not any(e[0] == "heldout_predict" for e in spy.events)


@pytest.mark.parametrize(
    "prediction",
    ([[0.0], [0.0]], np.zeros((2,)), np.zeros((2, 1), dtype=np.float32), np.full((2, 1), np.inf)),
)
def test_should_discard_scores_when_prediction_is_malformed(monkeypatch, prediction):
    training, heldout = batches()
    spy = Spy(monkeypatch, training, heldout)
    monkeypatch.setattr(measurement, "predict_linear", lambda weights, features: prediction)
    result = run(training=training, heldout=heldout, clock=spy.clock)
    assert result["status"] == "failed"
    assert result["committed_updates"] == 2
    assert result["heldout_half_mse"] is None


def invalid_batches():
    t, h = batches()
    bad: Any = t
    return [
        (replace(t, role="final_test"), h),
        (t, replace(h, role="train")),
        (replace(t, row_ids=("train-0", "train-0")), h),
        (t, replace(h, row_ids=("train-0", "test-1"))),
        (replace(bad, row_ids=["train-0", "train-1"]), h),
        (replace(t, row_ids=(" ", "train-1")), h),
        (replace(t, row_ids=(" padded ", "train-1")), h),
        (replace(t, row_ids=("row\x00id", "train-1")), h),
        (replace(t, row_ids=("row\nid", "train-1")), h),
        (replace(t, row_ids=("train-0",)), h),
        (replace(bad, features=t.features.tolist()), h),
        (replace(t, features=t.features.astype(np.float32)), h),
        (replace(t, targets=t.targets.astype(np.int64)), h),
        (replace(t, features=np.ones((2, 3))), h),
        (replace(t, targets=np.ones((2,))), h),
        (replace(t, targets=np.full((2, 1), np.nan)), h),
        (replace(t, features=np.zeros((0, 2)), targets=np.zeros((0, 1)), row_ids=()), h),
        (t, replace(h, features=t.features)),
        (t, replace(h, targets=t.targets)),
        (t, replace(h, targets=t.features[:, :1])),
    ]


@pytest.mark.parametrize("training,heldout", invalid_batches())
def test_should_refuse_invalid_or_overlapping_batches_before_work(monkeypatch, training, heldout):
    spy = Spy(monkeypatch, training, heldout)
    with pytest.raises(ValueError):
        run(training=training, heldout=heldout, clock=spy.clock)
    assert spy.events == []


@pytest.mark.parametrize("updates", (0, 513, True, 2.0, np.int64(2)))
def test_should_refuse_invalid_update_counts_before_work(monkeypatch, updates: Any):
    training, heldout = batches()
    spy = Spy(monkeypatch, training, heldout)
    with pytest.raises(ValueError):
        run(updates=updates, clock=spy.clock)
    assert spy.events == []


@pytest.mark.parametrize("rate", (True, 0, -1, float("inf"), np.float64(0.1), 10**1000))
def test_should_refuse_invalid_rates_before_work(monkeypatch, rate: Any):
    training, heldout = batches()
    spy = Spy(monkeypatch, training, heldout)
    with pytest.raises(ValueError):
        run(learning_rate=rate, clock=spy.clock)
    assert spy.events == []


def test_should_refuse_mutated_state_foreign_budget_and_noncallable_clock_before_work(monkeypatch):
    training, heldout = batches()
    spy = Spy(monkeypatch, training, heldout)
    initial = state()
    initial.raw_weights[0][0, 0] = np.nan
    for options in ({"initial": initial}, {"budget": None}, {"clock": None}):
        with pytest.raises(ValueError):
            run(**options)
    assert spy.events == []


def test_should_run_exactly_six_native_updates_across_three_fixed_schemes():
    for scheme in ("legacy_gaussian", "fan_in", "mean_field"):
        initial = state(scheme)
        original = tuple(w.copy() for w in initial.raw_weights)
        result = run(initial)
        assert result["status"] == "complete"
        assert result["attempted_updates"] == result["committed_updates"] == 2
        assert [row["updates"] for row in result["checkpoints"]] == [0, 1, 2]
        assert len(result["heldout_half_mse"]) == 3
        assert all(np.isfinite(value) and value >= 0 for value in result["heldout_half_mse"])
        for actual, before in zip(initial.raw_weights, original, strict=True):
            np.testing.assert_array_equal(actual, before)
