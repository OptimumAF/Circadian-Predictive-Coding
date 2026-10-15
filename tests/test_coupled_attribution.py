"""Behavioral isolation controls for deferred scoring of saved stream states.

Spy traces identify actual native inputs and call order. The small native control
is eight total updates, not a comparative pilot or a seed-selection procedure.
"""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import MatchedBackpropNetwork


class SpyLearner:
    """Fork model state but share an external observation log, never training data."""

    def __init__(
        self, trace=None, state=None, prediction=None, mutate_predict=False, on_predict=None
    ):
        self.trace = [] if trace is None else trace
        self.state = np.array([0.0]) if state is None else state.copy()
        self.prediction, self.mutate_predict = prediction, mutate_predict
        self.on_predict = on_predict

    def fork(self):
        self.trace.append(("fork", float(self.state[0])))
        return SpyLearner(
            self.trace, self.state, self.prediction, self.mutate_predict, self.on_predict
        )

    def snapshot_state(self):
        self.trace.append(("snapshot", float(self.state[0])))
        return self.state.copy()

    def restore_state(self, snapshot):
        self.trace.append(("restore", float(snapshot[0])))
        self.state = snapshot.copy()

    def train_batch(self, features, targets):
        self.trace.append(("train", float(self.state[0]), features.copy(), targets.copy()))
        self.state += 1
        return TrainingDiagnostic("spy_native_v1", float(self.state[0]))

    def predict(self, features):
        self.trace.append(("predict", float(self.state[0]), features.copy()))
        if self.mutate_predict:
            features.fill(999)
        if self.on_predict is not None:
            self.on_predict()
        if self.prediction is not None:
            return self.prediction(features)
        return np.full((len(features), 1), (self.state[0] + 1) / 10)


def batches():
    training = (
        AttributionBatch("train", "A", ("train-a",), np.array([[0.1, 0.2]]), np.array([[1.0]])),
        AttributionBatch("train", "B", ("train-b",), np.array([[0.3, 0.4]]), np.array([[0.0]])),
    )
    heldout = (
        AttributionBatch("final_test", "A", ("test-a",), np.array([[0.7, 0.8]]), np.array([[0.0]])),
        AttributionBatch("final_test", "B", ("test-b",), np.array([[0.9, 1.0]]), np.array([[1.0]])),
    )
    return training, heldout


def budget(updates=2):
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=updates), lambda: 0.0)


def run(learner, training=None, heldout=None, **options):
    defaults = batches()
    return run_attribution_stream(
        learner,
        defaults[0] if training is None else training,
        defaults[1] if heldout is None else heldout,
        checkpoint_updates=options.pop("checkpoint_updates", (0, 1, 2)),
        label_delay=options.pop("label_delay", 2),
        budget=options.pop("budget", budget()),
        **options,
    )


@pytest.mark.parametrize("delay", (0, 2))
def test_should_defer_all_scoring_and_score_the_actual_saved_states(delay):
    source = SpyLearner()
    training, heldout = batches()
    session = budget()
    result = run(source, training, heldout, label_delay=delay, budget=session)
    assert result["completed_updates"] == session.updates_completed == 2
    assert result["training_examples"] == 2
    assert result["score_forward_calls"] == 6
    train_calls = [event for event in source.trace if event[0] == "train"]
    predictions = [event for event in source.trace if event[0] == "predict"]
    assert len(train_calls) == 2 and len(predictions) == 6
    assert max(i for i, e in enumerate(source.trace) if e[0] == "train") < min(
        i for i, e in enumerate(source.trace) if e[0] == "predict"
    )
    assert [e[1] for e in predictions] == [0, 0, 1, 1, 2, 2]
    for event, expected in zip(train_calls, training, strict=True):
        np.testing.assert_array_equal(event[2], expected.features)
        np.testing.assert_array_equal(event[3], expected.targets)
        for reserved in heldout:
            assert not np.array_equal(event[2], reserved.features)
    assert source.state[0] == 0
    assert [row["updates"] for row in result["checkpoints"]] == [0, 1, 2]
    assert [row["training_context"] for row in result["checkpoints"]] == [None, "A", "B"]
    for index, row in enumerate(result["checkpoints"]):
        p = (index + 1) / 10
        assert row["scores"]["A"]["bce"] == pytest.approx(-np.log(1 - p))
        assert row["scores"]["B"]["bce"] == pytest.approx(-np.log(p))
        assert row["scores"]["A"]["accuracy"] == 1.0
        assert row["scores"]["B"]["accuracy"] == 0.0
    assert len(result["applied_updates"]) == 2
    for index, receipt in enumerate(result["applied_updates"]):
        assert receipt["observed_at"] == index
        assert receipt["arrived_at"] == receipt["applied_at"] == index + delay
        assert receipt["update_number"] == index + 1


def test_should_copy_heldout_features_before_each_adversarial_native_prediction():
    source = SpyLearner(mutate_predict=True)
    training, heldout = batches()
    original = pickle.dumps((training, heldout))
    run(source, training, heldout)
    assert pickle.dumps((training, heldout)) == original
    observed = [event[2] for event in source.trace if event[0] == "predict"]
    for index, features in enumerate(observed):
        np.testing.assert_array_equal(features, heldout[index % 2].features)


def test_should_clip_common_probability_bce_and_use_inclusive_half_threshold():
    training, _ = batches()
    heldout = (
        AttributionBatch(
            "final_test",
            "A",
            ("held-a0", "held-a1", "held-a2"),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0], [0.0], [1.0]]),
        ),
        AttributionBatch(
            "final_test",
            "B",
            ("held-b0", "held-b1", "held-b2"),
            np.ones((3, 2)),
            np.array([[1.0], [0.0], [1.0]]),
        ),
    )
    source = SpyLearner(prediction=lambda _: np.array([[0.0], [1.0], [0.5]]))
    result = run(source, training, heldout, checkpoint_updates=(0, 2))
    p = np.clip(np.array([0.0, 1.0, 0.5]), 1e-12, 1 - 1e-12)
    y = np.array([1.0, 0.0, 1.0])
    expected = float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
    for checkpoint in result["checkpoints"]:
        for score in checkpoint["scores"].values():
            assert score["bce"] == pytest.approx(expected, abs=1e-13)
            assert score["accuracy"] == pytest.approx(1 / 3)


def invalid_cases():
    training, heldout = batches()
    t: Any = training[0]
    h: Any = heldout[0]
    return [
        ([], heldout),
        ((), heldout),
        (training * 257, heldout),
        (training, []),
        (training, ()),
        ((replace(t, role="final_test"), training[1]), heldout),
        (training, (replace(h, role="train"), heldout[1])),
        ((replace(t, context="C"), training[1]), heldout),
        (training, (h, replace(heldout[1], context="A"))),
        ((replace(t, row_ids=["train-a"]), training[1]), heldout),
        ((replace(t, row_ids=("train-b",)), training[1]), heldout),
        (training, (replace(h, row_ids=("train-a",)), heldout[1])),
        ((replace(t, row_ids=("",)), training[1]), heldout),
        ((replace(t, row_ids=(" padded ",)), training[1]), heldout),
        ((replace(t, row_ids=()), training[1]), heldout),
        ((replace(t, context=""), training[1]), heldout),
        ((replace(t, features=[[0.1, 0.2]]), training[1]), heldout),
        ((replace(t, features=t.features.astype(np.float32)), training[1]), heldout),
        ((replace(t, targets=t.targets.astype(np.int64)), training[1]), heldout),
        (
            (
                replace(t, features=np.ones((0, 2)), targets=np.ones((0, 1)), row_ids=()),
                training[1],
            ),
            heldout,
        ),
        (
            (
                replace(
                    t,
                    features=np.ones((1025, 2)),
                    targets=np.ones((1025, 1)),
                    row_ids=tuple(f"row-{i}" for i in range(1025)),
                ),
                training[1],
            ),
            heldout,
        ),
        ((replace(t, features=np.ones((1, 3))), training[1]), heldout),
        ((replace(t, targets=np.ones((1,))), training[1]), heldout),
        ((replace(t, features=np.full((1, 2), np.nan)), training[1]), heldout),
        ((replace(t, targets=np.array([[np.inf]])), training[1]), heldout),
        ((replace(t, targets=np.array([[1.1]])), training[1]), heldout),
        (training, (replace(h, features=training[0].features), heldout[1])),
        (training, (replace(h, targets=training[1].targets), heldout[1])),
        (training, (replace(h, targets=training[0].features[:, :1]), heldout[1])),
        (training, (replace(h, features=np.broadcast_to(training[0].targets, (1, 2))), heldout[1])),
    ]


@pytest.mark.parametrize("training,heldout", invalid_cases())
def test_should_refuse_invalid_roles_rows_arrays_or_overlap_before_any_native_work(
    training: Any, heldout: Any
):
    source = SpyLearner()
    with pytest.raises(ValueError):
        run(source, training, heldout)
    assert source.trace == []
    assert source.state[0] == 0


@pytest.mark.parametrize(
    "checkpoints",
    (
        [],
        (),
        (1, 2),
        (0, 1),
        (0, 2, 1),
        (0, 1, 1, 2),
        (0, True, 2),
        (0, 1.0, 2),
        (-1, 0, 2),
        (0, 3),
    ),
)
def test_should_refuse_invalid_checkpoints_before_any_native_work(checkpoints: Any):
    source = SpyLearner()
    with pytest.raises(ValueError):
        run(source, checkpoint_updates=checkpoints)
    assert source.trace == []


@pytest.mark.parametrize("delay", (-1, 513, True, 1.0, np.int64(1)))
def test_should_refuse_invalid_delays_before_any_native_work(delay: Any):
    source = SpyLearner()
    with pytest.raises(ValueError):
        run(source, label_delay=delay)
    assert source.trace == []


def test_should_refuse_foreign_budget_before_any_native_work():
    source = SpyLearner()
    with pytest.raises(ValueError):
        run(source, budget=None)
    assert source.trace == []


def test_should_stop_exhausted_training_budget_without_any_scoring():
    source = SpyLearner()
    session = budget(1)
    with pytest.raises(ToyExecutionStopped) as failure:
        run(source, budget=session)
    assert failure.value.stop.reason == "max_training_updates"
    assert session.updates_completed == 1
    assert len([e for e in source.trace if e[0] == "train"]) == 1
    assert not any(e[0] == "predict" for e in source.trace)
    assert source.state[0] == 0


def test_should_check_wall_budget_after_the_last_scoring_forward():
    time = [0.0]
    calls = [0]

    def after_predict():
        calls[0] += 1
        if calls[0] == 4:
            time[0] = 2.0

    source = SpyLearner(on_predict=after_predict)
    session = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1.0), lambda: time[0]
    )
    with pytest.raises(ToyExecutionStopped) as failure:
        run(source, checkpoint_updates=(0, 2), budget=session)
    assert failure.value.stop.reason == "max_wall_seconds"
    assert session.updates_completed == 2
    assert calls[0] == 4
    assert source.state[0] == 0


@pytest.mark.parametrize(
    "prediction",
    (
        lambda x: [[0.5]] * len(x),
        lambda x: np.ones((len(x),)),
        lambda x: np.ones((len(x), 1), dtype=np.float32),
        lambda x: np.full((len(x), 1), np.nan),
        lambda x: np.full((len(x), 1), -0.1),
        lambda x: np.full((len(x), 1), 1.1),
    ),
)
def test_should_refuse_invalid_native_predictions_instead_of_partial_scores(prediction):
    source = SpyLearner(prediction=prediction)
    with pytest.raises(ValueError):
        run(source)
    assert len([e for e in source.trace if e[0] == "train"]) == 2
    assert source.state[0] == 0


def test_should_run_eight_native_updates_with_matched_pc_and_neutral_scores():
    pc = CoupledPredictiveCodingNetwork(2, (3,), seed=23)
    neutral = CoupledCircadianPredictiveCodingNetwork(
        2, (3,), seed=23, config=CoupledCircadianConfig.matched_pc_control()
    )
    gated = CoupledCircadianPredictiveCodingNetwork(2, (3,), seed=23)
    bp = MatchedBackpropNetwork(pc.snapshot_parameters())
    sources: dict[str, Any] = {
        "bp": MatchedBackpropLearner(bp, learning_rate=0.03),
        "pc": CoupledLearner(
            pc, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
        ),
        "neutral": CoupledLearner(
            neutral, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
        ),
        "gated": CoupledLearner(
            gated, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
        ),
    }
    results = {}
    for name, source in sources.items():
        initial = pickle.dumps(source.snapshot_state())
        results[name] = run(source, checkpoint_updates=(0, 2))
        assert results[name]["completed_updates"] == 2
        assert results[name]["score_forward_calls"] == 4
        assert pickle.dumps(source.snapshot_state()) == initial
    assert results["pc"]["checkpoints"] == results["neutral"]["checkpoints"]
