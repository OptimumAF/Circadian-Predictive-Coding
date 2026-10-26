"""TRAIN readout clocks after a fixed BC prefix and variable-length reward work.

Learners and environments are owned fakes. These checks grant no reward update,
final-data access, native physics, or scientific inference authority.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.core.minigrid_key_door import predefined_layouts
from test_minigrid_learning_pilot import evaluator as evaluator


METHOD = "recurrent_warm_reward"


def checkpoint(case, updates=136, rows=1173):
    policy = TorchRecurrentPolicy(148, 8, 7, 17)
    case.source.policy = case.source.state.policy = policy
    case.source.state.completed_updates, case.source.state.training_examples = 128, 1152
    state = case.source.snapshot_state()
    state.completed_updates, state.training_examples = updates, rows
    return state


def evaluate(case, state, *, updates=136, rows=1173, **options):
    arguments = dict(
        method=METHOD,
        updates=updates,
        native_training_examples=rows,
        layouts=predefined_layouts()[:3],
        role="train",
        sources=case.sources,
        budget=case.budget,
        check_boundary=case.guard,
    )
    arguments.update(options)
    return evaluation.evaluate_checkpoint(case.source, state, **arguments)


@pytest.mark.parametrize("updates,rows", [(128, 1152), (136, 1160), (136, 1173), (136, 1280)])
def test_should_use_exact_native_row_clock_and_six_owned_recurrent_train_sessions(
    evaluator, updates, rows
):
    case = evaluator
    state = checkpoint(case, updates, rows)
    before = pickle.dumps((state, case.source.snapshot_state(), case.sources))
    binding = (
        case.budget.budget,
        case.budget.started_at,
        case.budget.last_clock,
        case.budget.updates_completed,
    )
    report = evaluate(case, state, updates=updates, rows=rows)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == METHOD and report["updates"] == updates
    assert report["native_training_examples"] == rows and report["role"] == "train"
    assert "public_history" not in report
    assert len(report["scores"]) == len(report["episodes"]) == 6
    expected = [(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")]
    assert [(score["context"], score["cue"]) for score in report["scores"]] == expected
    assert all(score["ce"] == pytest.approx(float(np.log(2))) for score in report["scores"])
    assert all(
        score["accuracy"] == score["exact_sequence_accuracy"] == 0 for score in report["scores"]
    )
    work = report["work"]
    assert work["raw_score_forwards_completed"] == 6 and work["raw_score_rows_completed"] == 54
    assert work["live_sessions_started"] == work["environment_resets_completed"] == 6
    assert work["live_prediction_calls_returned"] == work["environment_steps_returned"] == 12
    sessions = case.work["sessions"]
    assert len({id(session) for session in sessions}) == 6
    assert all(session.completed_steps == 2 for session in sessions)
    for index, packet in enumerate(case.sources):
        actual = case.work["predicts"][index][1]
        np.testing.assert_array_equal(actual, packet.features)
        assert not np.shares_memory(actual, packet.features)
    for game, slot in zip(case.work["environments"], report["episodes"], strict=True):
        assert game.role == "train" and game.max_steps == 16 and game.closed
        assert game.actions == [0, 0] and slot["state_unchanged"] is True
        assert all(decision["action"] == 0 for decision in slot["predictions"])
        assert all(
            decision["probabilities"] == [0.5, 0.5, 0, 0, 0, 0, 0]
            and len(decision["observation"]) == 148
            and "features" not in decision
            for decision in slot["predictions"]
        )
    for session, features in case.work["session_inputs"]:
        assert features.shape == (1, 148)
        assert (
            session.state.completed_updates == updates and session.state.training_examples == rows
        )
    assert pickle.dumps((state, case.source.snapshot_state(), case.sources)) == before
    assert (
        case.budget.budget,
        case.budget.started_at,
        case.budget.last_clock,
        case.budget.updates_completed,
    ) == binding
    assert not case.work.get("trains") and len(case.guard_calls) > 12


@pytest.mark.parametrize(
    "updates,rows",
    [
        (128, None),
        (128, True),
        (128, 1152.0),
        (128, np.int64(1152)),
        (128, 1151),
        (128, 1153),
        (136, None),
        (136, 1159),
        (136, 1281),
        (0, 0),
        (129, 1161),
        (True, 1152),
    ],
)
def test_should_refuse_undeclared_reward_clock_before_fork_or_environment(evaluator, updates, rows):
    case = evaluator
    state = checkpoint(case)
    with pytest.raises((ValueError, evaluation.MiniGridEvaluationFailed)):
        evaluate(case, state, updates=updates, rows=rows)
    assert not case.work.get("forks") and not case.work.get("environments")
    assert not case.work.get("predicts")


@pytest.mark.parametrize(
    "fault",
    ["a_only", "final", "probe", "geometry", "order", "missing", "role", "width", "history"],
)
def test_should_keep_warm_reward_readout_tied_to_original_six_train_sources(evaluator, fault):
    case = evaluator
    state = checkpoint(case)
    arguments: dict[str, Any] = {}
    if fault == "a_only":
        arguments.update(layouts=predefined_layouts()[:1], sources=case.sources[:2])
    elif fault == "final":
        arguments.update(layouts=predefined_layouts()[-1:], sources=(), role="final_test")
    elif fault == "probe":
        arguments["role"] = "probe_train"
    elif fault == "geometry":
        layouts = predefined_layouts()[:3]
        arguments["layouts"] = (replace(layouts[0], rotation=1), *layouts[1:])
    elif fault == "order":
        arguments["sources"] = (case.sources[1], case.sources[0], *case.sources[2:])
    elif fault == "missing":
        arguments["sources"] = case.sources[:-1]
    elif fault == "role":
        first = case.sources[0]
        arguments["sources"] = (
            replace(first, role="probe_train", episode=replace(first.episode, role="probe_train")),
            *case.sources[1:],
        )
    elif fault == "width":
        object.__setattr__(case.sources[0], "features", case.sources[0].features[:, :147].copy())
    else:
        arguments["public_history"] = True
    with pytest.raises((ValueError, evaluation.MiniGridEvaluationFailed)):
        evaluate(case, state, **arguments)
    assert not case.work.get("forks") and not case.work.get("environments")
    assert not case.work.get("predicts")


@pytest.mark.parametrize("method", ["bp", "recurrent_bc", "recurrent_weighted_bc"])
def test_should_forbid_reward_row_override_on_existing_method_routes(evaluator, method):
    case = evaluator
    state = checkpoint(case, 128, 1152)
    with pytest.raises((ValueError, evaluation.MiniGridEvaluationFailed)):
        evaluate(case, state, updates=128, rows=1152, method=method)
    assert not case.work.get("forks") and not case.work.get("environments")


@pytest.mark.parametrize("fault", ["updates", "rows", "row_type"])
def test_should_compare_declared_native_clocks_to_actual_checkpoint_before_fork(evaluator, fault):
    case = evaluator
    state = checkpoint(case)
    if fault == "updates":
        state.completed_updates = 135
    elif fault == "rows":
        state.training_examples = 9 * 136
    else:
        state.training_examples = float(state.training_examples)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        evaluate(case, state)
    assert error.value.report["work"]["raw_score_forwards_attempted"] == 0
    assert not case.work.get("forks") and not case.work.get("environments")


@pytest.mark.parametrize("fault", ["source", "session", "checkpoint", "receipts"])
def test_should_keep_paid_raw_and_live_prefix_when_completed_work_guard_refuses(evaluator, fault):
    case = evaluator
    state = checkpoint(case)
    receipts = [{"update_number": index + 1, "rows": 2} for index in range(8)]
    seal = pickle.dumps(receipts)
    original_guard = case.guard

    def guard():
        original_guard()
        if pickle.dumps(receipts) != seal:
            raise ValueError("preceding reward receipt changed")

    def corrupt(session):
        if fault == "source":
            case.source.state.parameters["weight"] += 1
        elif fault == "session":
            session.state.optimizer_state["momentum"] += 1
        elif fault == "checkpoint":
            state.parameters["weight"] += 1
        else:
            receipts[-1]["rows"] = 3

    case.work["on_session_predict"] = corrupt
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        evaluate(case, state, check_boundary=guard)
    report = error.value.report
    assert report["status"] == "failed" and len(report["scores"]) == 6
    assert report["work"]["raw_score_rows_completed"] == 54
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert len(report["episodes"]) == len(case.work["environments"]) == 1
    assert case.work["environments"][0].closed
    assert len(case.work["sessions"]) == 1 and not case.work.get("trains")


def test_should_preserve_actual_step_prefix_when_next_live_prediction_fails(evaluator):
    case = evaluator
    state = checkpoint(case)
    before = deepcopy(case.source.snapshot_state())
    case.work["on_step"] = lambda game: case.work.update(prediction_error=True)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        evaluate(case, state)
    report = error.value.report
    assert "fake prediction failed" in report["reason"] and len(report["scores"]) == 6
    assert report["work"]["live_prediction_calls_attempted"] == 2
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 1
    assert len(report["episodes"][0]["environment"]["transitions"]) == 1
    assert case.work["environments"][0].closed and len(case.work["environments"]) == 1
    assert pickle.dumps(case.source.snapshot_state()) == pickle.dumps(before)
    assert not case.work.get("trains")


def test_should_require_caller_preceding_receipt_validation_even_with_valid_numeric_clocks(
    evaluator,
):
    case = evaluator
    state = checkpoint(case)

    def refused_receipt():
        raise ValueError("eight original reward receipts are no longer valid")

    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        evaluate(case, state, check_boundary=refused_receipt)
    report = error.value.report
    assert "original reward receipts" in report["reason"]
    assert report["work"]["raw_score_forwards_attempted"] == 0
    assert not case.work.get("forks") and not case.work.get("environments")
