"""Bound raw/invariant count readouts on discrete engineering frames, without physics.

The environment is a two-step accounting fake. Its success is not game evidence;
the real native predictions below test only the declared representation boundary.
"""

import builtins
from dataclasses import replace
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
)
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryEntry, ActionMemoryPolicy, ActionMemorySnapshot
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)
from test_minigrid_invariant_encoding import frame
from test_minigrid_learning_pilot import Environment


KINDS = ("raw", "invariant", "invariant_fifo")


def public_frame(cue, step, *, wall_color=5, rotation=0):
    color = 2 if cue == "blue" else 0
    return frame(
        wall_color=wall_color,
        direction=(rotation + step) % 4,
        key_color=color,
        cue_color=color,
        door_state=2 if step == 0 else 0,
    )


class DiscreteEnvironment(Environment):
    def _frame(self, step):
        wall_color = {"grey": 5, "yellow": 4}[self.layout.wall_color]
        return public_frame(self.cue, step, wall_color=wall_color, rotation=self.layout.rotation)

    def __init__(self, layout, cue, *args, **options):
        super().__init__(layout, cue, *args, **options)
        self.observation = self._frame(0)

    def step(self, action):
        result = super().step(action)
        self.observation = self._frame(len(self.actions))
        result["observation"] = self.observation
        self.transitions[-1]["next_observation"] = self.observation
        return result


def state_for(kind, updates):
    native_policy: Any = (
        FifoActionMemoryPolicy if kind == "invariant_fifo" else ActionMemoryPolicy
    )(feature_dim=296, num_classes=7, max_payload_bytes=155136)
    entries = []
    if updates:
        for cue in ("blue", "red"):
            first = public_frame(cue, 0)
            for step in (0, 1):
                key = first + public_frame(cue, step)
                if kind != "raw":
                    key = canonicalize_public_history(key)
                counts = [0] * 7
                counts[6 if cue == "blue" else 4] = 9 * updates // 4
                entries.append(ActionMemoryEntry(key, tuple(counts)))
    native: Any = (
        FifoActionMemorySnapshot(native_policy, tuple(entries), updates, 9 * updates, 0, 0)
        if kind == "invariant_fifo"
        else ActionMemorySnapshot(native_policy, tuple(entries), updates, 9 * updates)
    )
    if kind == "raw":
        return native
    return InvariantMemorySnapshot(InvariantMemoryPolicy(native_policy), native)


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", DiscreteEnvironment)
    original_import = builtins.__import__

    def no_torch(name: str, *args: Any, **options: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            pytest.fail("representation readout imported Torch")
        return original_import(name, *args, **options)

    def forbidden(*args, **options):
        pytest.fail("representation readout reached training or neural comparison")

    for native in (ActionMemoryLearner, FifoActionMemoryLearner, InvariantActionMemoryLearner):
        monkeypatch.setattr(native, "train_batch", forbidden)
    monkeypatch.setattr(evaluation, "_same_state", forbidden)
    monkeypatch.setattr(builtins, "__import__", no_torch)

    def build(kind="invariant", updates=32):
        state = state_for(kind, updates)
        native_class: Any = ActionMemoryLearner if kind == "raw" else InvariantActionMemoryLearner
        source = native_class(state.policy)
        source.restore_state(state)
        original_fork, original_predict = native_class.fork, native_class.predict

        def fork(owner):
            work.setdefault("forks", []).append(owner)
            child = original_fork(owner)
            work.setdefault("scorers", []).append(child)
            return child

        def predict(owner, features):
            work.setdefault("attempts", []).append((owner, features.copy()))
            if work.get("prediction_error"):
                raise RuntimeError("representation prediction failed")
            result = original_predict(owner, features)
            work.setdefault("predictions", []).append((owner, features.copy(), result.copy()))
            callback = work.get("on_predict")
            if callback is not None:
                callback(owner)
            if work.get("bad_probability"):
                return np.ones((1, 6), dtype=np.float64)
            return result

        monkeypatch.setattr(native_class, "fork", fork)
        monkeypatch.setattr(native_class, "predict", predict)
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 10.0)
        binding = (budget.budget, budget.started_at, budget.updates_completed)

        def guard():
            work.setdefault("guard_calls", []).append(len(work.get("predictions", [])))
            if (budget.budget, budget.started_at, budget.updates_completed) != binding:
                raise ValueError("original representation readout budget changed")

        return SimpleNamespace(
            kind=kind,
            source=source,
            state=state,
            updates=updates,
            work=work,
            budget=budget,
            binding=binding,
            guard=guard,
        )

    return build


def evaluate(current, **options):
    arguments = dict(updates=current.updates, budget=current.budget, check_boundary=current.guard)
    arguments.update(options)
    return evaluation.evaluate_representation_checkpoint(current.source, current.state, **arguments)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("updates", [0, 32])
def test_should_apply_only_the_bound_projection_and_keep_raw_decision_inputs_owned(
    case, kind, updates
):
    current = case(kind, updates)
    before = pickle.dumps((current.source.snapshot_state(), current.state))
    report = evaluate(current)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["scores"] == [] and report["updates"] == updates
    assert report["model_encoding"] == (
        "original_raw_history" if kind == "raw" else INVARIANT_HISTORY_ENCODING_ID
    )
    assert report["method"] == current.state.policy.method
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
        ("E-blue", "blue", "probe_test"),
        ("E-red", "red", "probe_test"),
    ]
    assert len(current.work["forks"]) == len(current.work["scorers"]) == 1
    assert current.work["scorers"][0] is not current.source
    for index, (game, slot) in enumerate(
        zip(current.work["environments"], report["episodes"], strict=True)
    ):
        assert game.closed and game.seed == 17 and game.max_steps == 16
        first = game.transitions[0]["observation"]
        expected_actions = []
        for step, decision in enumerate(slot["predictions"]):
            raw = first + game.transitions[step]["observation"]
            assert tuple(decision["features"]) == raw
            np.testing.assert_array_equal(current.work["predictions"][index * 2 + step][1], [raw])
            recognized = updates != 0 and (game.layout.context == "A" or kind != "raw")
            action = (6 if game.cue == "blue" else 4) if recognized else 0
            expected_actions.append(action)
            assert decision["action"] == action
            probabilities = np.array(decision["probabilities"])
            if recognized:
                np.testing.assert_array_equal(probabilities, np.eye(7)[action])
            else:
                np.testing.assert_array_equal(probabilities, np.full(7, 1 / 7))
        assert game.actions == expected_actions
    work = report["work"]
    assert work["raw_score_forwards_completed"] == 0 and work["raw_score_rows_completed"] == 0
    assert work["live_prediction_calls_returned"] == 8 and work["environment_steps_returned"] == 8
    assert work["environment_resets_completed"] == 4 and work["history_live_feature_rows"] == 8
    assert pickle.dumps((current.source.snapshot_state(), current.state)) == before
    assert (
        current.budget.budget,
        current.budget.started_at,
        current.budget.updates_completed,
    ) == current.binding


@pytest.mark.parametrize(
    "fault",
    ["point_bool", "undeclared_point", "clock", "cap", "encoding", "raw_fifo", "snapshot_type"],
)
def test_should_refuse_wrong_point_clocks_cap_or_representation_before_callbacks(case, fault):
    current = case()
    options: dict[str, Any] = {}
    if fault == "point_bool":
        options["updates"] = True
    elif fault == "undeclared_point":
        options["updates"] = 16
    elif fault == "clock":
        current.state = replace(
            current.state, native_state=replace(current.state.native_state, completed_updates=31)
        )
    elif fault == "cap":
        bad_policy = InvariantMemoryPolicy(ActionMemoryPolicy(feature_dim=296, num_classes=7))
        current.state = replace(
            current.state,
            policy=bad_policy,
            native_state=replace(current.state.native_state, policy=bad_policy.native_policy),
        )
    elif fault == "encoding":
        object.__setattr__(current.state.policy, "encoding_id", "unbound")
    elif fault == "raw_fifo":
        current.state = state_for("invariant_fifo", 32).native_state
    else:
        current.state = object()
    with pytest.raises(ValueError):
        evaluate(current, **options)
    assert not current.work.get("guard_calls") and not current.work.get("forks")
    assert not current.work.get("attempts") and not current.work.get("environments")


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint", "budget"])
def test_should_refuse_same_counter_state_changes_or_budget_mutation_after_a_paid_prediction(
    case, target
):
    current = case()

    def mutate(owner):
        if target in ("source", "scorer"):
            selected = current.source if target == "source" else owner
            state = selected.snapshot_state()
            entry = state.native_state.entries[0]
            counts = tuple(entry.counts[(i + 1) % 7] for i in range(7))
            child = replace(
                state.native_state,
                entries=(
                    ActionMemoryEntry(entry.features, counts),
                    *state.native_state.entries[1:],
                ),
            )
            selected.restore_state(replace(state, native_state=child))
        elif target == "checkpoint":
            entry = current.state.native_state.entries[0]
            object.__setattr__(entry, "counts", tuple(entry.counts[(i + 1) % 7] for i in range(7)))
        else:
            current.budget.record_update()

    current.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(current)
    report = raised.value.report
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_attempted"] == 0
    assert [slot["status"] for slot in report["episodes"]] == [
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]


@pytest.mark.parametrize(
    "fault", ["prediction_error", "bad_probability", "step_error", "step_and_close"]
)
def test_should_preserve_paid_failure_prefix_and_all_unreached_slots(case, fault):
    current = case()
    current.work["step_error" if fault == "step_and_close" else fault] = True
    if fault == "step_and_close":
        current.work["close_error"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(current)
    report = raised.value.report
    assert report["scores"] == [] and report["status"] == "failed"
    assert [slot["status"] for slot in report["episodes"]] == [
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]
    assert len(current.work["environments"]) == 1 and current.work["environments"][0].closed
    work = report["work"]
    assert work["live_prediction_calls_attempted"] == 1
    assert work["live_prediction_calls_returned"] == (0 if fault == "prediction_error" else 1)
    assert work["environment_steps_attempted"] == (
        1 if fault in ("step_error", "step_and_close") else 0
    )
    assert (
        work["environment_steps_returned"] == 0 and work["failure_work_may_be_unobserved"] is True
    )
    if fault == "step_and_close":
        assert (
            "step failed" in report["reason"]
            and "close failed" in report["episodes"][0]["close_error"]
        )


@pytest.mark.parametrize("fault", ["extra", "bool_clock"])
def test_should_refuse_malformed_external_bound_snapshot_ports_before_fork(
    case, monkeypatch, fault
):
    current = case()
    original = current.source.snapshot_state

    def malformed():
        value = original()
        if fault == "extra":
            object.__setattr__(value, "extra", 1)
        else:
            object.__setattr__(value.native_state, "completed_updates", True)
        return value

    monkeypatch.setattr(current.source, "snapshot_state", malformed)
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(current)
    assert not current.work.get("forks") and not current.work.get("environments")


def test_should_keep_unknown_readback_costs_and_preserve_the_original_default_gate(
    case, monkeypatch
):
    current = case()
    with pytest.raises(ValueError, match="unsupported"):
        evaluation.evaluate_checkpoint(
            current.source,
            current.state,
            method=current.state.policy.method,
            updates=32,
            layouts=(),
            role="train",
            sources=(),
            budget=current.budget,
            check_boundary=current.guard,
            public_history=True,
        )
    descriptor: Any = vars(Environment)["report"]
    original = descriptor.fget

    def unreadable(game):
        if game.closed:
            raise RuntimeError("representation readback failed")
        return original(game)

    monkeypatch.setattr(Environment, "report", property(unreadable))
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(current)
    report = raised.value.report
    assert report["work"]["live_prediction_calls_returned"] == 2
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert "readback failed" in report["episodes"][0]["readback_error"]
    assert [slot["status"] for slot in report["episodes"]] == [
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]
