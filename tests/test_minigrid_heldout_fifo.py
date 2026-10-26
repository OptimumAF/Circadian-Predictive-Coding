"""Scripted D-only isolation/refusal tests: no native predictions or physics.

Literal snapshots and scripted completions are boundary fixtures, not observed
held-out data or learned success. The old TRAIN fixture remains separately owned.
"""

from dataclasses import replace
import pickle

import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_key_door import predefined_layouts
from test_minigrid_recurring_fifo_evaluator import (
    fifo_case as _fifo_case,
    refusal_case as _refusal_case,
    snapshot,
)

fifo_case = _fifo_case
refusal_case = _refusal_case


@pytest.fixture
def heldout(fifo_case):
    case = fifo_case
    case.final_budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=0), lambda: 10.0
    )
    return case


def evaluate(case, *, state=None):
    return evaluation.evaluate_heldout_fifo_checkpoint(
        case.source,
        case.state if state is None else state,
        budget=case.final_budget,
        check_boundary=case.guard,
    )


def test_should_read_only_two_final_d_games_and_preserve_training_authority(heldout):
    case = heldout
    before = pickle.dumps((case.source.value, case.state, case.sources))
    original_budget = case.budget.budget
    report = evaluate(case)
    assert report["status"] == "complete" and report["state_unchanged"]
    assert report["role"] == "final_test" and report["updates"] == 128
    assert "held-out D final frame" in report["public_history"]["storage_scope"]
    assert report["scores"] == [] and report["native_training_examples"] == 1152
    assert [(s["context"], s["cue"], s["source_role"]) for s in report["episodes"]] == [
        ("D", "blue", "final_test"),
        ("D", "red", "final_test"),
    ]
    assert all(e.layout == predefined_layouts()[3] and e.closed for e in case.work["environments"])
    work = report["work"]
    assert work["raw_score_forwards_attempted"] == work["raw_score_rows_attempted"] == 0
    assert work["live_prediction_calls_attempted"] == work["live_prediction_calls_returned"] == 4
    assert work["environment_steps_attempted"] == work["environment_steps_returned"] == 4
    assert work["internal_projection_rows_returned"] == 4
    assert case.final_budget.updates_completed == case.final_budget.replay_examples_completed == 0
    assert case.budget.budget is original_budget and case.budget.updates_completed == 128
    assert pickle.dumps((case.source.value, case.state, case.sources)) == before
    assert all(row.shape == (1, 296) for row in case.work["returns"])


@pytest.mark.parametrize("point", [0, 32, 64, 96, 160])
def test_should_refuse_prefix_or_probe_snapshot_before_final_work(heldout, point):
    with pytest.raises(ValueError):
        evaluate(heldout, state=snapshot(point))
    assert heldout.work == {}


@pytest.mark.parametrize("fault", ["cap", "payload", "rows", "bool_clock", "extra"])
def test_should_refuse_foreign_or_replaced_native_state_before_final_work(heldout, fault):
    state = (
        snapshot(128, cap=64, payload=155136)
        if fault == "cap"
        else snapshot(128, payload=155136)
        if fault == "payload"
        else snapshot(128, rows=1151)
        if fault == "rows"
        else snapshot(128)
    )
    if fault == "bool_clock":
        object.__setattr__(state.native_state, "completed_updates", True)
    if fault == "extra":
        object.__setattr__(state, "unobserved", 1)
    with pytest.raises(ValueError):
        evaluate(heldout, state=state)
    assert heldout.work == {}


@pytest.mark.parametrize(
    "fault", ["training_credit", "replay_credit", "replay_none", "replay_bool", "charged"]
)
def test_should_refuse_nonreadonly_or_charged_budget_before_final_work(heldout, fault):
    budget = heldout.final_budget
    if fault == "charged":
        budget.updates_completed = 1
    else:
        name, value = {
            "training_credit": ("max_training_updates", 1),
            "replay_credit": ("max_replay_examples", 1),
            "replay_none": ("max_replay_examples", None),
            "replay_bool": ("max_replay_examples", False),
        }[fault]
        object.__setattr__(budget.budget, name, value)
    with pytest.raises(ValueError):
        evaluate(heldout)
    assert heldout.work == {}


@pytest.mark.parametrize("point", [32, 64, 96, 128])
def test_should_refuse_advanced_source_port_before_final_environment(heldout, point):
    heldout.source.value = snapshot(point)
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(heldout)
    assert not heldout.work.get("environments") and not heldout.work.get("attempts")


@pytest.mark.parametrize(
    "fault", ["prediction_error", "bad_probability", "step_error", "close_error", "readback_error"]
)
def test_should_keep_failed_paid_prefix_close_environment_and_not_admit_second_game(heldout, fault):
    heldout.work[fault] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        evaluate(heldout)
    report = error.value.report
    assert report["status"] == "failed"
    assert "held-out D final frame" in report["public_history"]["storage_scope"]
    assert len(heldout.work["environments"]) == 1 and heldout.work["environments"][0].closed
    assert report["episodes"][1]["status"] == "not_run"
    assert (
        heldout.final_budget.updates_completed
        == heldout.final_budget.replay_examples_completed
        == 0
    )
    assert heldout.budget.updates_completed == 128


@pytest.mark.parametrize("fault", ["state", "budget_identity", "clock_identity", "charge"])
def test_should_refuse_mutation_after_prediction_without_more_effects(heldout, fault):
    def mutate(port, features):
        if fault == "state":
            port.value = snapshot(128, retired=1, evictions=1)
        elif fault == "budget_identity":
            heldout.final_budget.budget = replace(heldout.final_budget.budget)
        elif fault == "clock_identity":
            heldout.final_budget.clock = lambda: 10.0
        else:
            heldout.final_budget.replay_examples_completed = 1

    heldout.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        evaluate(heldout)
    assert len(heldout.work["attempts"]) == 1
    assert error.value.report["episodes"][1]["status"] == "not_run"
    assert heldout.work["environments"][0].closed and heldout.work["environments"][0].steps == 0


def test_should_preserve_old_train_only_refusal_for_d(heldout):
    with pytest.raises(ValueError, match="predefined TRAIN layout triple"):
        evaluation.evaluate_invariant_fifo_checkpoint(
            heldout.source,
            heldout.state,
            updates=128,
            layouts=(predefined_layouts()[3],),
            role="final_test",
            sources=(),
            budget=heldout.final_budget,
            check_boundary=heldout.guard,
        )
    assert heldout.work == {}
