"""Read-only saved memory endpoints with real counts and a fake live environment.

Clock-valid DTOs exercise the boundary without replaying benchmark updates.
Fake game endings are harness behavior, never evidence of native MiniGrid skill.
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
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryEntry, ActionMemoryPolicy, ActionMemorySnapshot
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from src.core.minigrid_key_door import predefined_layouts, predefined_probe_layouts
from test_minigrid_learning_pilot import Environment


ENDPOINTS = (
    (False, 0, 0, "complete"),
    (True, 0, 0, "complete"),
    (False, 0, 16, "complete"),
    (True, 0, 16, "complete"),
    (False, 128, 0, "complete"),
    (True, 128, 0, "complete"),
    (False, 128, 1, "failed"),
    (True, 128, 16, "complete"),
)


def snapshot(fifo: bool, native: int):
    declared: Any = (FifoActionMemoryPolicy if fifo else ActionMemoryPolicy)(
        feature_dim=296, num_classes=7, max_payload_bytes=155136
    )
    entries = (
        ()
        if native == 0
        else (ActionMemoryEntry((0.75,) + (0.0,) * 295, (9 * native, 0, 0, 0, 0, 0, 0)),)
    )
    if fifo:
        return FifoActionMemorySnapshot(declared, entries, native, 9 * native, 0, 0)
    return ActionMemorySnapshot(declared, entries, native, 9 * native)


class ReturnEnvironment(Environment):
    """Keep existing two-step fake accounting; give each slot its own first frame."""

    def __init__(self, layout, cue, *args, **options):
        super().__init__(layout, cue, *args, **options)
        offset = {"A": -0.5, "E-blue": 0.0, "E-red": 0.25}[layout.context]
        self.observation = (offset + (0.0 if cue == "blue" else 0.125),) + self.observation[1:]


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", ReturnEnvironment)
    original_import = builtins.__import__

    def no_torch(name: str, *args: Any, **options: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            pytest.fail("read-only memory evaluation imported Torch")
        return original_import(name, *args, **options)

    def forbidden(*args, **options):
        pytest.fail("frozen memory readout reached training or neural comparison")

    for native in (ActionMemoryLearner, FifoActionMemoryLearner):
        original_predict = native.predict
        original_fork = native.fork

        def fork(self, original=original_fork):
            work.setdefault("forks", []).append(self)
            if work.get("identity_fork"):
                return self
            child = original(self)
            work.setdefault("scorers", []).append(child)
            return child

        def predict(self, features, original=original_predict):
            work.setdefault("attempts", []).append((self, features.copy()))
            if work.get("prediction_error"):
                raise RuntimeError("saved-memory prediction failed")
            result = original(self, features)
            work.setdefault("predictions", []).append((self, features.copy(), result.copy()))
            callback = work.get("on_predict")
            if callback is not None:
                callback(self)
            if work.get("bad_probability"):
                return np.ones((1, 6), dtype=np.float64)
            return result

        monkeypatch.setattr(native, "fork", fork)
        monkeypatch.setattr(native, "predict", predict)
        monkeypatch.setattr(native, "train_batch", forbidden)
    monkeypatch.setattr(evaluation, "_same_state", forbidden)
    monkeypatch.setattr(builtins, "__import__", no_torch)

    def build(fifo=False, source_updates=0, relative_updates=16, source_status="complete"):
        state = snapshot(fifo, source_updates + relative_updates)
        native_class: Any = FifoActionMemoryLearner if fifo else ActionMemoryLearner
        source = native_class(state.policy)
        source.restore_state(state)
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 10.0)
        binding = (budget.budget, budget.started_at, budget.updates_completed)

        def guard():
            work.setdefault("guard_calls", []).append(len(work.get("predictions", [])))
            if (budget.budget, budget.started_at, budget.updates_completed) != binding:
                raise ValueError("original readout-only budget changed")

        return SimpleNamespace(
            work=work,
            source=source,
            state=state,
            budget=budget,
            binding=binding,
            guard=guard,
            options=dict(
                source_updates=source_updates,
                relative_updates=relative_updates,
                source_status=source_status,
            ),
        )

    return build


def evaluate(case, **options):
    arguments = dict(case.options)
    arguments.update(options)
    return evaluation.evaluate_memory_return_checkpoint(
        case.source, case.state, budget=case.budget, check_boundary=case.guard, **arguments
    )


@pytest.mark.parametrize("fifo,source_updates,relative_updates,source_status", ENDPOINTS)
def test_should_read_only_the_eight_declared_endpoints_with_separate_history_and_no_scores(
    case, fifo, source_updates, relative_updates, source_status
):
    current = case(fifo, source_updates, relative_updates, source_status)
    before = pickle.dumps((current.source.snapshot_state(), current.state))
    report = evaluate(current)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["source_status"] == source_status
    assert (
        report["source_updates"] == source_updates
        and report["relative_updates"] == relative_updates
    )
    assert report["updates"] == source_updates + relative_updates
    assert report["method"] == ("observation_memory_fifo" if fifo else "observation_memory")
    assert report["scores"] == []
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
        ("E-blue", "blue", "probe_test"),
        ("E-red", "red", "probe_test"),
    ]
    expected_layouts = (
        predefined_layouts()[0],
        predefined_layouts()[0],
        *predefined_probe_layouts(),
    )
    assert len(current.work["scorers"]) == len(current.work["forks"]) == 1
    assert current.work["scorers"][0] is not current.source
    assert current.work["scorers"][0].snapshot_state() == current.state
    for index, (game, slot, layout) in enumerate(
        zip(current.work["environments"], report["episodes"], expected_layouts, strict=True)
    ):
        assert game.layout == layout and game.closed
        assert game.seed == 17 and game.max_steps == 16
        assert game.role == slot["source_role"] and game.actions == [0, 0]
        first = game.transitions[0]["observation"]
        for step, decision in enumerate(slot["predictions"]):
            expected = first + game.transitions[step]["observation"]
            assert tuple(decision["features"]) == expected
            owner, actual, probabilities = current.work["predictions"][index * 2 + step]
            assert owner is current.work["scorers"][0]
            np.testing.assert_array_equal(actual, [expected])
            np.testing.assert_array_equal(probabilities, np.full((1, 7), 1 / 7))
            assert decision["action"] == 0
        assert slot["status"] == "complete" and slot["state_unchanged"] is True
    initials = [game.transitions[0]["observation"] for game in current.work["environments"]]
    assert len(set(initials)) == 4
    work = report["work"]
    assert work["raw_score_forwards_attempted"] == work["raw_score_forwards_completed"] == 0
    assert work["raw_score_rows_attempted"] == work["raw_score_rows_completed"] == 0
    assert work["live_prediction_calls_attempted"] == work["live_prediction_calls_returned"] == 8
    assert work["environment_resets_completed"] == 4 and work["environment_steps_returned"] == 8
    assert work["history_episodes_started"] == 4 and work["history_live_feature_rows"] == 8
    assert work["history_live_feature_array_bytes"] == 8 * 296 * 8
    assert pickle.dumps((current.source.snapshot_state(), current.state)) == before
    assert (
        current.budget.budget,
        current.budget.started_at,
        current.budget.updates_completed,
    ) == current.binding


@pytest.mark.parametrize(
    "fault",
    [
        "source_bool",
        "relative_bool",
        "wrong_source",
        "unobserved_point",
        "exact_mature_end",
        "fifo_failed_prefix",
        "wrong_status",
        "native_clocks",
        "small_cap",
        "old_geometry",
    ],
)
def test_should_refuse_unavailable_endpoints_and_malformed_native_contracts_before_callbacks(
    case, fault
):
    current = case(False, 128, 1, "failed")
    options: dict[str, Any] = {}
    if fault == "source_bool":
        options["source_updates"] = True
    elif fault == "relative_bool":
        options["relative_updates"] = True
    elif fault == "wrong_source":
        options["source_updates"] = 64
    elif fault == "unobserved_point":
        options["relative_updates"] = 8
    elif fault == "exact_mature_end":
        current = case(False, 128, 16, "complete")
    elif fault == "fifo_failed_prefix":
        current = case(True, 128, 1, "failed")
    elif fault == "wrong_status":
        options["source_status"] = "not_run"
    elif fault == "native_clocks":
        current.state = replace(current.state, completed_updates=128)
    elif fault == "small_cap":
        current.state = replace(
            current.state, policy=ActionMemoryPolicy(feature_dim=296, num_classes=7)
        )
    else:
        current.state = ActionMemorySnapshot(
            ActionMemoryPolicy(), (ActionMemoryEntry((0.0,) * 16, (1161, 0, 0, 0, 0)),), 129, 1161
        )
    with pytest.raises(ValueError):
        evaluate(current, **options)
    assert not current.work.get("guard_calls") and not current.work.get("forks")
    assert not current.work.get("attempts") and not current.work.get("environments")
    assert current.budget.updates_completed == 0


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint", "budget", "clock"])
def test_should_refuse_same_counter_count_changes_and_budget_mutations_after_paid_prediction(
    case, target
):
    current = case()

    def alter(owner):
        if target in ("source", "scorer"):
            selected = current.source if target == "source" else owner
            state = selected.snapshot_state()
            entry = state.entries[0]
            changed = ActionMemoryEntry(entry.features, (0, sum(entry.counts), 0, 0, 0, 0, 0))
            selected.restore_state(replace(state, entries=(changed,)))
        elif target == "checkpoint":
            entry = current.state.entries[0]
            object.__setattr__(entry, "counts", (0, sum(entry.counts), 0, 0, 0, 0, 0))
        elif target == "budget":
            current.budget.record_update()
        else:
            current.budget.last_clock = 11.0

    current.work["on_predict"] = alter
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(current)
    report = raised.value.report
    assert report["status"] == "failed" and report["scores"] == []
    assert (
        report["work"]["live_prediction_calls_attempted"]
        == report["work"]["live_prediction_calls_returned"]
        == 1
    )
    assert report["work"]["environment_steps_attempted"] == 0
    assert len(report["episodes"]) == 4 and current.work["environments"][0].closed
    assert [slot["status"] for slot in report["episodes"]] == [
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]
    assert current.budget.updates_completed == (1 if target == "budget" else 0)


@pytest.mark.parametrize(
    "fault", ["prediction_error", "bad_probability", "step_error", "step_and_close"]
)
def test_should_preserve_paid_live_prefix_and_cleanup_faults_without_later_slots(case, fault):
    current = case()
    current.work["step_error" if fault == "step_and_close" else fault] = True
    if fault == "step_and_close":
        current.work["close_error"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(current)
    report = raised.value.report
    assert report["status"] == "failed" and report["scores"] == []
    assert len(report["episodes"]) == 4 and len(current.work["environments"]) == 1
    assert [slot["status"] for slot in report["episodes"]] == [
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]
    slot = report["episodes"][0]
    assert slot["status"] == "failed" and current.work["environments"][0].closed
    assert len(slot["predictions"]) == 1 and len(slot["predictions"][0]["features"]) == 296
    work = report["work"]
    assert work["live_prediction_calls_attempted"] == 1
    assert work["live_prediction_calls_returned"] == (0 if fault == "prediction_error" else 1)
    assert work["environment_steps_attempted"] == (
        1 if fault in ("step_error", "step_and_close") else 0
    )
    assert (
        work["environment_steps_returned"] == 0 and work["failure_work_may_be_unobserved"] is True
    )
    assert current.budget.updates_completed == 0
    if fault == "step_and_close":
        assert "step failed" in report["reason"] and "close failed" in slot["close_error"]


def test_should_refuse_identity_fork_before_any_environment_or_prediction(case):
    current = case()
    current.work["identity_fork"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(current)
    assert not current.work.get("attempts") and not current.work.get("environments")


@pytest.mark.parametrize("fault", ["extra", "bool_clock"])
def test_should_refuse_malformed_external_snapshot_ports_instead_of_accepting_self_consistency(
    case, monkeypatch, fault
):
    current = case()
    original_snapshot = current.source.snapshot_state

    def invalid_snapshot():
        state = original_snapshot()
        if fault == "extra":
            object.__setattr__(state, "extra", 1)
        else:
            object.__setattr__(state, "completed_updates", True)
        return state

    monkeypatch.setattr(current.source, "snapshot_state", invalid_snapshot)
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(current)
    assert not current.work.get("forks") and not current.work.get("environments")


def test_should_keep_returned_prediction_cost_when_closed_environment_readback_is_unknown(
    case, monkeypatch
):
    current = case()
    descriptor: Any = vars(Environment)["report"]
    original_report = descriptor.fget

    def unreadable(game):
        if game.closed:
            raise RuntimeError("saved return environment readback failed")
        return original_report(game)

    monkeypatch.setattr(Environment, "report", property(unreadable))
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(current)
    report = raised.value.report
    assert len(report["episodes"]) == 4
    assert [slot["status"] for slot in report["episodes"]] == [
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]
    assert "readback failed" in report["episodes"][0]["readback_error"]
    assert report["work"]["live_prediction_calls_returned"] == 2
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert len(current.work["environments"]) == 1 and current.budget.updates_completed == 0
