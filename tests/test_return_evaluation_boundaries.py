"""Opt-in cached/live evaluation guards using owned fake learned states."""

from copy import deepcopy
from dataclasses import asdict
from typing import Any
import numpy as np
import pytest

from src.adapters import key_door_pilot as live
from src.app import coupled_attribution as cached
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def state():
    return {
        "parameters": np.array([1.0]),
        "optimizer": {"momentum": np.array([0.5])},
        "gradients": np.array([0.25]),
        "updates": 192,
        "examples": 1536,
        "policy": {"hidden": 8},
    }


class Learner:
    def __init__(self, learned=None, trace=None, hook=None, owner=None):
        self.state = deepcopy(state() if learned is None else learned)
        self.trace = [] if trace is None else trace
        self.hook = hook
        self.owner = self if owner is None else owner

    def fire(self, event):
        self.trace.append(event)
        if self.hook is not None:
            self.hook(event, self)

    def fork(self):
        self.fire("fork")
        return Learner(self.state, self.trace, self.hook, self.owner)

    def restore_state(self, learned):
        self.restore_input = learned
        self.state = deepcopy(learned)
        self.fire("restore")

    def snapshot_state(self):
        self.fire("snapshot")
        return deepcopy(self.state)

    def predict(self, features):
        self.fire("predict")
        return np.tile(np.array([[0.6, 0.1, 0.1, 0.1, 0.1]]), (len(features), 1))

    def start_episode(self):
        session = Session(self.state, self.trace, self.hook, self.owner)
        session.completed_steps = 0
        session.fire("session")
        return session


class Session(Learner):
    completed_steps: int = 0

    def predict(self, features):
        result = super().predict(features)
        self.completed_steps += 1
        return result


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=192), lambda: 0.0)


def heldout():
    return (
        cached.AttributionBatch(
            "final_test", "A", ("final-a",), np.ones((1, 8)), np.zeros((1, 1)), num_classes=5
        ),
        cached.AttributionBatch(
            "final_test", "B", ("final-b",), np.ones((1, 8)), np.ones((1, 1)), num_classes=5
        ),
    )


def score(source, *, states=None, **options):
    return cached._score_checkpoints(
        source,
        [(0, state()), (192, state())] if states is None else states,
        (),
        heldout(),
        budget(),
        training_contexts=("A",) * 192,
        **options,
    )


@pytest.mark.parametrize("mode", [0, 1, None, np.bool_(True), "true"])
def test_should_refuse_nonbool_verification_before_any_source_call(mode):
    source = Learner()
    with pytest.raises(ValueError, match="exact bool"):
        score(source, verify_state=mode)
    assert source.trace == []


@pytest.mark.parametrize("route", ["cached", "live"])
def test_should_refuse_noncallable_boundary_before_source_calls(route, monkeypatch):
    source = Learner()
    invalid: Any = 1
    with pytest.raises(ValueError, match="callable or None"):
        if route == "cached":
            score(source, check_boundary=1)
        else:
            live._live(row(), source, [(192, state())], budget(), check_boundary=invalid)
    assert source.trace == []


def test_should_preserve_cached_default_wire_and_native_call_order_without_snapshots():
    first, second = Learner(), Learner()
    original = score(first)
    explicit = score(second, verify_state=False, check_boundary=None)
    assert original == explicit
    assert (
        first.trace
        == second.trace
        == ["fork", "restore", "predict", "predict", "restore", "predict", "predict"]
    )


def test_should_verify_full_cached_state_without_mutating_source_or_checkpoint_inputs():
    source = Learner()
    states = [(0, state()), (192, state())]
    before = deepcopy(states)
    checks = []
    result = score(
        source,
        states=states,
        verify_state=True,
        check_boundary=lambda: checks.append(len(source.trace)),
    )
    assert len(result) == 2 and len(checks) > 10
    assert source.trace.count("predict") == 4
    assert source.trace.count("snapshot") == 10
    assert cached._same_value(states, before)
    assert cached._same_value(source.state, state())


def mutate(learned, field):
    if field in ("parameters", "gradients"):
        learned[field][0] += 1
    elif field == "optimizer":
        learned[field]["momentum"][0] += 1
    elif field == "policy":
        learned[field]["hidden"] += 1
    else:
        learned[field] += 1


@pytest.mark.parametrize("event", ["restore", "predict"])
@pytest.mark.parametrize(
    "field", ["parameters", "optimizer", "gradients", "updates", "examples", "policy"]
)
def test_should_refuse_every_cached_learned_state_component_change(event, field):
    def corrupt(actual, learner):
        if actual == event:
            mutate(learner.state, field)

    source = Learner(hook=corrupt)
    states = [(0, state())]
    before = deepcopy(states)
    with pytest.raises(ValueError, match="captured state"):
        score(source, states=states, verify_state=True)
    assert source.trace.count("predict") == (1 if event == "predict" else 0)
    assert cached._same_value(states, before)


def test_should_freeze_expected_state_separately_from_adversarial_restore_input():
    def corrupt(event, learner):
        if event == "restore":
            learner.restore_input["parameters"][0] = 99
            learner.state = learner.restore_input

    states = [(0, state())]
    before = deepcopy(states)
    with pytest.raises(ValueError, match="captured state"):
        score(Learner(hook=corrupt), states=states, verify_state=True)
    assert cached._same_value(states, before)


def test_should_require_snapshot_protocol_only_when_explicitly_verifying():
    class NoSnapshot:
        def fork(self):
            return self.__class__()

        def restore_state(self, value):
            pass

        def predict(self, features):
            return np.tile(np.array([[0.6, 0.1, 0.1, 0.1, 0.1]]), (len(features), 1))

    assert len(score(NoSnapshot())) == 2
    with pytest.raises(ValueError, match="requires snapshot_state"):
        score(NoSnapshot(), verify_state=True)


@pytest.mark.parametrize("event", ["fork", "restore", "predict"])
def test_should_refuse_source_corruption_at_cached_external_boundary_before_more_predictions(event):
    def corrupt(actual, learner):
        if actual == event:
            learner.owner.state["updates"] += 1

    source = Learner(hook=corrupt)

    def guard():
        if not cached._same_value(source.state, state()):
            raise ValueError("original source changed")

    with pytest.raises(ValueError, match="original source changed"):
        score(source, verify_state=True, check_boundary=guard)
    assert source.trace.count("predict") == (1 if event == "predict" else 0)


def row():
    return {
        "method": "recurrent_bc",
        "replay_enabled": False,
        "live_sessions_started": 0,
        "episodes": [
            {
                "checkpoint_updates": 192,
                "context": context,
                "cue": 1,
                "layout_partition": "known_layout" if context in ("A", "B") else "heldout_layout",
                "status": "not_run",
            }
            for context in ("A", "B", "C", "D")
        ],
    }


def fake_episode(predict, *, identity, context, layout_partition, before_prediction, **options):
    identity.__post_init__()
    assert (context in identity.task_sequence) == (layout_partition == "known_layout")
    before_prediction()
    predict(np.ones((1, 8)))
    return {
        "identity": asdict(identity),
        "status": "complete",
        "success": False,
        "return": 0,
        "steps_executed": 0,
    }


@pytest.fixture
def live_fakes(monkeypatch):
    monkeypatch.setattr(live, "_same_state", cached._same_value)
    monkeypatch.setattr(live, "run_key_door_episode", fake_episode)


@pytest.mark.parametrize("sequence", [[], (), ("",), ("A",) * 33, (1,), (" padded ",)])
def test_should_admit_explicit_sequence_before_live_source_work(sequence):
    source = Learner()
    with pytest.raises(ValueError):
        live._live(row(), source, [(192, state())], budget(), task_sequence=sequence)
    assert source.trace == []


def test_should_use_actual_return_sequence_and_fresh_owned_sessions(live_fakes):
    source, report, checks = Learner(), row(), []
    live._live(
        report,
        source,
        [(192, state())],
        budget(),
        task_sequence=("A", "B", "A"),
        check_boundary=lambda: checks.append(len(source.trace)),
    )
    assert report["live_sessions_started"] == 4
    assert all(
        slot["state_unchanged"] is True and slot["status"] == "complete"
        for slot in report["episodes"]
    )
    assert all(slot["identity"]["task_sequence"] == ("A", "B", "A") for slot in report["episodes"])
    assert source.trace.count("session") == source.trace.count("predict") == 4
    assert len(checks) > 20 and cached._same_value(source.state, state())


def test_should_preserve_omitted_live_defaults_and_native_order(live_fakes):
    first, second = Learner(), Learner()
    first_row, second_row = row(), row()
    first_row["episodes"] = first_row["episodes"][:2]
    second_row["episodes"] = second_row["episodes"][:2]
    live._live(first_row, first, [(192, state())], budget())
    live._live(
        second_row,
        second,
        [(192, state())],
        budget(),
        task_sequence=live.TASKS,
        check_boundary=None,
    )
    assert first.trace == second.trace and first_row == second_row
    assert first_row["episodes"][0]["identity"]["task_sequence"] == live.TASKS


@pytest.mark.parametrize("event", ["fork", "restore", "session", "predict"])
def test_should_refuse_source_corruption_at_live_external_boundaries(live_fakes, event):
    def corrupt(actual, learner):
        if actual == event:
            learner.owner.state["examples"] += 1

    source, report = Learner(hook=corrupt), row()

    def guard():
        if not cached._same_value(source.state, state()):
            raise ValueError("original source changed")

    with pytest.raises(ValueError, match="original source changed"):
        live._live(
            report,
            source,
            [(192, state())],
            budget(),
            task_sequence=("A", "B", "A"),
            check_boundary=guard,
        )
    assert source.trace.count("predict") == (1 if event == "predict" else 0)
    assert report["live_sessions_started"] == (1 if event in ("session", "predict") else 0)
    assert report["episodes"][1]["status"] == "not_run"


@pytest.mark.parametrize("event", ["restore", "predict"])
def test_should_preserve_live_complete_state_refusal(live_fakes, event):
    def corrupt(actual, learner):
        if actual == event:
            mutate(learner.state, "optimizer")

    source, report = Learner(hook=corrupt), row()
    with pytest.raises(ValueError, match="captured state|during prediction"):
        live._live(report, source, [(192, state())], budget(), task_sequence=("A", "B", "A"))
    assert source.trace.count("predict") == (1 if event == "predict" else 0)


def test_should_preserve_session_birth_refusal_and_count_attempted_session(live_fakes):
    def corrupt(event, learner):
        if event == "session":
            learner.completed_steps = 1

    source, report = Learner(hook=corrupt), row()
    with pytest.raises(ValueError, match="fresh owned zero-step"):
        live._live(report, source, [(192, state())], budget(), task_sequence=("A", "B", "A"))
    assert report["live_sessions_started"] == 1
    assert report["episodes"][0]["status"] == "failed"
    assert source.trace.count("predict") == 0
