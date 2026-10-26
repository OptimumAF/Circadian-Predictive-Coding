"""Per-episode public history with owned fake native state and measured failures."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_feature_rehearsal_pilot as comparison
from src.adapters.key_door_public_history_session import PublicHistoryPredictionSession


class TorchStub:
    class Tensor:
        pass


@pytest.fixture(autouse=True)
def no_native_runtime(monkeypatch):
    monkeypatch.setattr(comparison, "require_torch", lambda: TorchStub)


@dataclass
class Policy:
    feature_dim: int = 16
    num_classes: int = 5
    method: str = "bp"
    learning_rate: float = 0.03


@dataclass
class State:
    policy: Policy
    parameters: dict
    gradients: dict
    optimizer_state: dict
    completed_updates: int = 8
    training_examples: int = 64


class Predictor:
    def __init__(self, policy=None):
        self.state = State(
            Policy() if policy is None else deepcopy(policy),
            {"weight": [0.2, 0.3]},
            {"weight": [0.01, -0.02]},
            {"momentum": [0.4, 0.5], "rate": 0.03},
        )
        self.children = []
        self.calls = []
        self.reads = 0
        self.hook = None
        self.output = np.full((1, 5), 0.2, dtype=np.float64)
        self.same_fork = False
        self.bad_restore = False
        self.mutate_source_during_fork = False

    @property
    def policy(self):
        return deepcopy(self.state.policy)

    def snapshot_state(self):
        self.reads += 1
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)
        if self.bad_restore:
            self.state.optimizer_state["rate"] = 0.5

    def fork(self):
        if self.same_fork:
            return self
        child = Predictor(self.policy)
        child.restore_state(self.state)
        child.bad_restore = self.bad_restore
        self.children.append(child)
        if self.mutate_source_during_fork:
            self.state.gradients["weight"][0] = 9.0
        return child

    def predict(self, values):
        self.calls.append(values.copy())
        if self.hook is not None:
            self.hook(self, values)
        return self.output


def row(cue=0.0):
    return np.array([[0.0, -1.0, 0.0, -1.0, 0.0, 0.0, cue, 0.0]], dtype=np.float64)


def counts(session):
    return (
        session.attempted_steps,
        session.encoded_frames,
        session.native_prediction_returns,
        session.completed_steps,
    )


def test_should_retain_only_first_public_cue_with_owned_inputs_and_outputs():
    source = Predictor()
    original = source.snapshot_state()
    session = PublicHistoryPredictionSession(source)
    assert counts(session) == (0, 0, 0, 0)
    first = row(1)
    current = row()
    initial = first.copy()
    session.predict(first)
    first.fill(-0.5)
    result = session.predict(current)
    child = source.children[0]
    np.testing.assert_array_equal(child.calls[0], np.concatenate((initial, initial), axis=1))
    np.testing.assert_array_equal(child.calls[1], np.concatenate((initial, current), axis=1))
    assert child.calls[1].shape == (1, 16) and child.calls[1].dtype == np.float64
    assert not np.shares_memory(result, child.output)
    result.fill(0)
    assert np.all(child.output == 0.2)
    assert source.snapshot_state() == original
    assert session.snapshot_state() == original
    assert counts(session) == (2, 2, 2, 2)


def test_should_reset_cue_and_counters_by_constructing_each_new_independent_session():
    source = Predictor()
    red = PublicHistoryPredictionSession(source)
    red.predict(row(1))
    red.predict(row())
    blue = PublicHistoryPredictionSession(source)
    assert counts(blue) == (0, 0, 0, 0)
    blue.predict(row(-1))
    blue.predict(row())
    red_values, blue_values = source.children[0].calls[-1], source.children[1].calls[-1]
    np.testing.assert_array_equal(red_values[:, 8:], blue_values[:, 8:])
    assert red_values[0, 6] == 1.0 and blue_values[0, 6] == -1.0
    assert source.children[0] is not source.children[1]


def test_should_detach_complete_native_state_and_return_independently_owned_readbacks():
    source = Predictor()
    expected = source.snapshot_state()
    session = PublicHistoryPredictionSession(source)
    readback = session.snapshot_state()
    readback.parameters["weight"][0] = 9
    readback.gradients["weight"][0] = 8
    readback.optimizer_state["momentum"][0] = 7
    source.state.parameters["weight"][0] = 6
    source.state.gradients["weight"][0] = 5
    source.state.optimizer_state["momentum"][0] = 4
    source.state.completed_updates = 100
    assert session.snapshot_state() == expected
    session.predict(row(1))
    assert session.snapshot_state() == expected
    assert not hasattr(session, "train_batch") and not hasattr(session, "reset")


def test_should_own_native_input_even_when_prediction_mutates_its_encoded_array():
    source = Predictor()
    session = PublicHistoryPredictionSession(source)
    child = source.children[0]
    child.hook = lambda learner, values: values.fill(0.75)
    first, later = row(1), row()
    session.predict(first)
    session.predict(later)
    np.testing.assert_array_equal(first, row(1))
    np.testing.assert_array_equal(later, row())
    np.testing.assert_array_equal(child.calls[-1], np.concatenate((row(1), row()), axis=1))


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_admit_all_four_existing_matched_methods(method):
    session = PublicHistoryPredictionSession(Predictor(Policy(method=method)))
    session.predict(row())
    assert counts(session) == (1, 1, 1, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_dim", 8),
        ("feature_dim", True),
        ("feature_dim", 16.0),
        ("feature_dim", np.int64(16)),
        ("num_classes", 2),
        ("num_classes", 5.0),
        ("method", "recurrent_bc"),
        ("method", True),
    ],
)
def test_should_refuse_invalid_policy_before_fork(field, value):
    source = Predictor()
    setattr(source.state.policy, field, value)
    with pytest.raises(ValueError, match="policy"):
        PublicHistoryPredictionSession(source)
    assert source.children == []


@pytest.mark.parametrize("fault", ["same_fork", "bad_restore", "mutate_source_during_fork"])
def test_should_refuse_unowned_or_changed_complete_native_state_at_construction(fault):
    source = Predictor()
    setattr(source, fault, True)
    with pytest.raises(ValueError):
        PublicHistoryPredictionSession(source)


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        True,
        [[0.0] * 8],
        np.zeros((1, 8), dtype=bool),
        np.zeros((1, 8), dtype=np.float32),
        np.zeros((2, 8)),
        np.zeros((1, 16)),
        np.zeros((8,)),
        np.zeros((1, 8)).view(ArraySubclass),
        np.full((1, 8), float("nan")),
        np.full((1, 8), float("inf")),
        np.full((1, 8), 1.01),
    ],
)
def test_should_count_invalid_admitted_input_once_then_stop_without_encoding_or_native_work(
    invalid,
):
    source = Predictor()
    session = PublicHistoryPredictionSession(source)
    with pytest.raises(ValueError, match="input"):
        session.predict(invalid)
    assert counts(session) == (1, 0, 0, 0)
    assert source.children[0].calls == []
    with pytest.raises(ValueError, match="stopped"):
        session.predict(row())
    assert counts(session) == (1, 0, 0, 0)


def test_should_refuse_thirteenth_call_before_any_validation_encoding_or_native_work():
    source = Predictor()
    session = PublicHistoryPredictionSession(source)
    for _ in range(12):
        session.predict(row())
    assert counts(session) == (12, 12, 12, 12)
    previous_reads = source.children[0].reads
    invalid: Any = object()
    with pytest.raises(ValueError, match="exhausted"):
        session.predict(invalid)
    assert counts(session) == (12, 12, 12, 12)
    assert len(source.children[0].calls) == 12
    assert source.children[0].reads == previous_reads


def test_should_preserve_successful_prior_work_when_a_later_input_validation_fails():
    source = Predictor()
    session = PublicHistoryPredictionSession(source)
    session.predict(row(1))
    with pytest.raises(ValueError, match="input"):
        session.predict(np.zeros((1, 16)))
    assert counts(session) == (2, 1, 1, 1)
    assert len(source.children[0].calls) == 1
    with pytest.raises(ValueError, match="stopped"):
        session.predict(row())
    assert counts(session) == (2, 1, 1, 1)


def test_should_preserve_encoded_attempt_when_native_prediction_raises_and_stop():
    source = Predictor()
    session = PublicHistoryPredictionSession(source)

    def fail(learner, values):
        raise RuntimeError("native prediction failed")

    source.children[0].hook = fail
    with pytest.raises(RuntimeError, match="native prediction failed"):
        session.predict(row(1))
    assert counts(session) == (1, 1, 0, 0)
    with pytest.raises(ValueError, match="stopped"):
        session.predict(row())
    assert counts(session) == (1, 1, 0, 0)
    assert session.snapshot_state() == source.snapshot_state()


@pytest.mark.parametrize(
    "field",
    [
        "parameters",
        "gradients",
        "optimizer_state",
        "completed_updates",
        "training_examples",
        "policy",
    ],
)
def test_should_count_actual_native_return_before_postguard_refuses_full_state_mutation(field):
    source = Predictor()
    session = PublicHistoryPredictionSession(source)

    def mutate(learner, values):
        if field == "parameters":
            learner.state.parameters["weight"][0] = 9
        elif field == "gradients":
            learner.state.gradients["weight"][0] = 9
        elif field == "optimizer_state":
            learner.state.optimizer_state["rate"] = 0.8
        elif field == "policy":
            learner.state.policy.learning_rate = 0.8
        else:
            setattr(learner.state, field, 9)

    source.children[0].hook = mutate
    with pytest.raises(ValueError, match="frozen native state"):
        session.predict(row(1))
    assert counts(session) == (1, 1, 1, 0)
    with pytest.raises(ValueError, match="stopped"):
        session.predict(row())
    assert counts(session) == (1, 1, 1, 0)


def test_should_refuse_prior_native_mutation_before_encoding_or_prediction():
    source = Predictor()
    session = PublicHistoryPredictionSession(source)
    source.children[0].state.optimizer_state["momentum"][0] = 3
    with pytest.raises(ValueError, match="frozen native state"):
        session.predict(row())
    assert counts(session) == (1, 0, 0, 0)
    assert source.children[0].calls == []


@pytest.mark.parametrize(
    "output",
    [
        None,
        False,
        [[0.2] * 5],
        np.zeros((1, 5)),
        np.full((2, 5), 0.2),
        np.full((1, 5), float("nan")),
        np.full((1, 5), 0.2, dtype=np.float32),
        np.full((1, 4), 0.25),
    ],
)
def test_should_keep_native_return_count_when_returned_probabilities_are_invalid(output):
    source = Predictor()
    session = PublicHistoryPredictionSession(source)
    source.children[0].output = output
    with pytest.raises(ValueError):
        session.predict(row())
    assert counts(session) == (1, 1, 1, 0)
    with pytest.raises(ValueError, match="stopped"):
        session.predict(row())


@pytest.mark.parametrize(
    "name", ["completed_steps", "attempted_steps", "encoded_frames", "native_prediction_returns"]
)
def test_should_expose_read_only_native_integer_counters_after_failure(name):
    session = PublicHistoryPredictionSession(Predictor())
    invalid: Any = False
    with pytest.raises(ValueError):
        session.predict(invalid)
    assert type(getattr(session, name)) is int
    with pytest.raises(AttributeError):
        setattr(session, name, 100)


def test_should_refuse_parent_mutation_during_terminal_child_readback_before_prediction(
    monkeypatch,
):
    source = Predictor()
    original_fork = source.fork

    def fork():
        child = original_fork()
        original_readback = child.snapshot_state

        def readback():
            source.state.optimizer_state["rate"] = 0.8
            return original_readback()

        monkeypatch.setattr(child, "snapshot_state", readback)
        return child

    monkeypatch.setattr(source, "fork", fork)
    with pytest.raises(ValueError, match="source state changed"):
        PublicHistoryPredictionSession(source)
    assert source.children[0].calls == []
