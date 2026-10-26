"""Scheduled sensor BP ownership and admission, with fake native work only."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters.scheduled_backprop_learner import (
    BUFFERED_DIAGNOSTIC_ID,
    ScheduledBackpropLearner,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.matched_backprop import MATCHED_BACKPROP_LOSS_ID, MatchedBackpropNetwork


def _learner(*, rate: float = 0.1) -> ScheduledBackpropLearner:
    parameters = CoupledPredictiveCodingNetwork(16, (8,), seed=17).snapshot_parameters()
    return ScheduledBackpropLearner(MatchedBackpropNetwork(parameters), learning_rate=rate)


def _batch(index: int = 0) -> tuple[Any, Any]:
    return np.full((4, 16), index, dtype=np.float64), np.full((4, 1), index % 2, dtype=np.float64)


def _fake_work(monkeypatch: Any) -> list[Any]:
    calls: list[Any] = []

    def train(model: Any, features: Any, targets: Any, rate: float) -> Any:
        calls.append((features.copy(), targets.copy(), rate))
        # Make a visible fake update without evaluating any native forward/gradient.
        parameters = model.snapshot_parameters()
        parameters.output_bias[:] += 0.25
        model.restore_parameters(parameters)
        features.fill(999)
        targets.fill(999)
        return SimpleNamespace(loss_definition=MATCHED_BACKPROP_LOSS_ID, loss=0.75)

    def predict(model: Any, features: Any) -> Any:
        raise AssertionError("training admission must not call native prediction")

    monkeypatch.setattr(MatchedBackpropNetwork, "train_epoch", train)
    monkeypatch.setattr(MatchedBackpropNetwork, "predict_proba", predict)
    return calls


def _same(actual: Any, expected: Any) -> None:
    np.testing.assert_equal(asdict(actual), asdict(expected))


def test_should_buffer_three_arrivals_then_run_one_owned_ordered_mean_batch(monkeypatch):
    calls = _fake_work(monkeypatch)
    learner = _learner()
    initial = learner.snapshot_state().native
    for index in range(3):
        features, targets = _batch(index)
        diagnostic = learner.train_batch(features, targets)
        features.fill(-999)
        targets.fill(-999)
        assert diagnostic.definition == BUFFERED_DIAGNOSTIC_ID
        assert diagnostic.value == 0.0
        assert calls == []
        _same(learner.snapshot_state().native, initial)
    diagnostic = learner.train_batch(*_batch(3))
    assert diagnostic.definition == MATCHED_BACKPROP_LOSS_ID and diagnostic.value == 0.75
    assert len(calls) == 1
    features, targets, rate = calls[0]
    np.testing.assert_array_equal(features, np.concatenate([_batch(i)[0] for i in range(4)]))
    np.testing.assert_array_equal(targets, np.concatenate([_batch(i)[1] for i in range(4)]))
    assert rate == 0.1
    assert learner.report() == dict(
        arrived_batches=4,
        optimizer_steps_attempted=1,
        optimizer_steps_completed=1,
        training_rows_attempted=16,
        training_rows_completed=16,
        buffered_batches=0,
        buffer_bytes=0,
        peak_buffer_bytes=2176,
        stopped=False,
    )


def test_should_preserve_pending_windows_rate_and_native_state_across_owned_forks(monkeypatch):
    _fake_work(monkeypatch)
    learner = _learner(rate=0.03)
    for index in range(6):
        learner.train_batch(*_batch(index))
    saved = learner.snapshot_state()
    fork = learner.fork()
    restored = _learner(rate=0.03)
    restored.restore_state(saved)
    saved.buffered_batches[0][0].fill(999)
    saved.native.parameters.output_bias.fill(999)
    for owner in (learner, fork, restored):
        owner.train_batch(*_batch(6))
        owner.train_batch(*_batch(7))
    _same(learner.snapshot_state(), fork.snapshot_state())
    _same(learner.snapshot_state(), restored.snapshot_state())
    fork_state = fork.snapshot_state()
    fork_state.native.parameters.output_bias.fill(888)
    _same(learner.snapshot_state(), restored.snapshot_state())
    report = learner.report()
    report["arrived_batches"] = 999
    assert learner.report()["arrived_batches"] == 8


def test_should_preserve_failed_attempt_native_changes_and_four_batches_without_retry(monkeypatch):
    calls = _fake_work(monkeypatch)
    learner = _learner()
    for index in range(4):
        learner.train_batch(*_batch(index))
    preceding = learner.snapshot_state()

    def fail(model: Any, features: Any, targets: Any, rate: float) -> Any:
        calls.append("failed")
        parameters = model.snapshot_parameters()
        parameters.output_bias[:] += 0.5
        model.restore_parameters(parameters)
        features.fill(999)
        targets.fill(999)
        raise RuntimeError("native attempt failed")

    monkeypatch.setattr(MatchedBackpropNetwork, "train_epoch", fail)
    for index in range(4, 7):
        learner.train_batch(*_batch(index))
    with pytest.raises(RuntimeError, match="native attempt failed"):
        learner.train_batch(*_batch(7))
    state = learner.snapshot_state()
    assert state.stopped and state.arrived_batches == 8
    assert state.optimizer_steps_attempted == 2 and state.optimizer_steps_completed == 1
    assert len(state.buffered_batches) == 4
    np.testing.assert_array_equal(state.buffered_batches[0][0], _batch(4)[0])
    assert (
        state.native.parameters.output_bias[0, 0]
        == preceding.native.parameters.output_bias[0, 0] + 0.5
    )
    for owner in (learner, learner.fork()):
        with pytest.raises(RuntimeError, match="stopped"):
            owner.train_batch(*_batch(8))
        with pytest.raises(RuntimeError, match="permanently failed"):
            owner.restore_state(preceding)
        _same(owner.snapshot_state(), state)
    assert len(calls) == 2
    assert learner.report()["training_rows_attempted"] == 32
    assert learner.report()["training_rows_completed"] == 16


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", True),
        ("format_version", 2),
        ("arrived_batches", True),
        ("arrived_batches", -1),
        ("arrived_batches", 13),
        ("optimizer_steps_attempted", True),
        ("optimizer_steps_attempted", 1),
        ("optimizer_steps_completed", True),
        ("optimizer_steps_completed", 1),
        ("peak_buffer_bytes", True),
        ("peak_buffer_bytes", 1),
        ("stopped", 0),
        ("stopped", True),
        ("learning_rate", True),
        ("learning_rate", float("nan")),
        ("learning_rate", 0.2),
        ("buffered_batches", []),
        ("buffered_batches", ()),
        ("native", None),
    ],
)
def test_should_reject_inconsistent_snapshot_atomically(field, value):
    learner = _learner()
    learner.train_batch(*_batch())
    before = learner.snapshot_state()
    with pytest.raises(ValueError):
        learner.restore_state(replace(before, **{field: value}))
    _same(learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "kind", ["pending_alias", "native_alias", "shape", "dtype", "nan", "range", "pair", "topology"]
)
def test_should_reject_malformed_or_aliasing_snapshot_arrays_atomically(kind):
    learner = _learner()
    learner.train_batch(*_batch())
    learner.train_batch(*_batch(1))
    before = learner.snapshot_state()
    bad = learner.snapshot_state()
    features, targets = bad.buffered_batches[0]
    if kind == "pending_alias":
        bad = replace(bad, buffered_batches=((features, targets), (features, targets)))
    elif kind == "native_alias":
        parameters = bad.native.parameters
        # A correctly shaped view can also overlap a native parameter's allocation.
        storage = np.zeros((16, 8), dtype=np.float64)
        native = replace(
            bad.native,
            parameters=CoupledParameters(
                (storage,),
                parameters.hidden_biases,
                parameters.output_weight,
                parameters.output_bias,
            ),
        )
        bad = replace(
            bad,
            native=native,
            buffered_batches=((storage.reshape(8, 16)[:4], targets), bad.buffered_batches[1]),
        )
    elif kind == "shape":
        bad = replace(bad, buffered_batches=((features[:3], targets), bad.buffered_batches[1]))
    elif kind == "dtype":
        bad = replace(
            bad, buffered_batches=((features.astype(np.float32), targets), bad.buffered_batches[1])
        )
    elif kind == "nan":
        features[0, 0] = np.nan
    elif kind == "range":
        targets[0, 0] = 2
    elif kind == "pair":
        malformed_pair: Any = [features, targets]
        bad = replace(bad, buffered_batches=(malformed_pair, bad.buffered_batches[1]))
    else:
        other_model = MatchedBackpropNetwork(
            CoupledPredictiveCodingNetwork(16, (7,), seed=17).snapshot_parameters()
        )
        bad = replace(bad, native=other_model.snapshot_state())
    with pytest.raises(ValueError):
        learner.restore_state(bad)
    _same(learner.snapshot_state(), before)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "range"])
def test_should_refuse_bad_batch_without_admitting_arrival(kind):
    learner = _learner()
    before = learner.snapshot_state()
    features, targets = _batch()
    if kind == "shape":
        features = features[:3]
    elif kind == "dtype":
        features = features.astype(np.float32)
    elif kind == "nan":
        features[0, 0] = np.nan
    else:
        targets[0, 0] = -1
    with pytest.raises(ValueError):
        learner.train_batch(features, targets)
    _same(learner.snapshot_state(), before)


def test_should_bound_twelve_arrivals_three_optimizer_steps_and_owned_prediction(monkeypatch):
    calls = _fake_work(monkeypatch)
    learner = _learner()
    for index in range(12):
        learner.train_batch(*_batch(index))
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="allowance"):
        learner.train_batch(*_batch())
    assert len(calls) == 3
    assert learner.report()["training_rows_completed"] == 48
    _same(learner.snapshot_state(), before)
    output = np.full((4, 1), 0.5)

    def predict(model, features):
        features.fill(999)
        return output

    monkeypatch.setattr(MatchedBackpropNetwork, "predict_proba", predict)
    features, _ = _batch()
    prediction = learner.predict(features)
    prediction.fill(-1)
    assert np.all(features == 0) and np.all(output == 0.5)
    with pytest.raises(ValueError):
        learner.predict(np.zeros((1, 16)))


@pytest.mark.parametrize("rate", [True, 0.0, float("inf")])
def test_should_refuse_invalid_rate(rate):
    with pytest.raises(ValueError):
        _learner(rate=rate)


def test_should_refuse_foreign_model_and_wrong_topology():
    foreign: Any = object()
    with pytest.raises(ValueError, match="exact native"):
        ScheduledBackpropLearner(foreign, learning_rate=0.1)
    for dimensions in [(15, (8,)), (16, (7,)), (16, (8, 8))]:
        parameters = CoupledPredictiveCodingNetwork(*dimensions, seed=17).snapshot_parameters()
        with pytest.raises(ValueError, match="topology"):
            ScheduledBackpropLearner(MatchedBackpropNetwork(parameters), learning_rate=0.1)
