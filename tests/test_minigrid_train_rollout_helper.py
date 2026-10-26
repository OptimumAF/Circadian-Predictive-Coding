"""Readonly orchestration preserves partial work and original zero-update limits."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.adapters import minigrid_train_rollout as live
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed
from test_torch_feature_rehearsal_pilot import forbidden


def partial_report(status="complete"):
    return dict(
        status=status,
        reason=None,
        episodes=[dict(context="A", cue="blue")],
        work=dict(
            environment_resets_completed=1,
            environment_steps_returned=2,
            live_prediction_calls_returned=2,
        ),
    )


def arm():
    return dict(source=object(), states=[(128, SimpleNamespace(completed_updates=128))])


def test_should_keep_a_separate_original_zero_update_budget_and_paid_live_work(monkeypatch):
    calls = []

    def evaluate(source, state, **options):
        calls.append((source, state, options))
        options["check_boundary"]()
        return partial_report()

    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluate)
    owner = arm()
    record, guard = live.run_train_rollout(owner, check_boundary=lambda: None)
    assert record["status"] == "complete" and record["budget_binding_preserved"] is True
    assert record["charged_updates"] == record["replay_examples"] == 0
    assert record["elapsed_seconds"] >= 0 and record["timing_complete"] is True
    source, state, options = calls[0]
    assert source is owner["source"] and state is owner["states"][0][1]
    assert options["updates"] == 128 and options["known_a_only"] is True
    assert options["budget"].budget.max_training_updates == 0
    assert options["budget"].budget.max_replay_examples == 0
    guard()
    work = dict(total_score_forwards_completed=94, source_environment_steps_returned=90)
    live.add_work(work, record)
    assert work["learner_live_episodes"] == 1
    assert work["total_prediction_forwards_completed"] == 96
    assert work["total_environment_steps_returned"] == 92
    options["budget"].clock = lambda: 1.0
    with pytest.raises(ValueError, match="binding"):
        guard()


def test_should_retain_returned_partial_report_when_existing_evaluator_refuses(monkeypatch):
    report = partial_report("failed")

    def evaluate(*args, **kwargs):
        raise MiniGridEvaluationFailed(report)

    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluate)
    record, _ = live.run_train_rollout(arm(), check_boundary=lambda: None)
    assert record["status"] == "failed" and record["report"] == report
    report["work"]["live_prediction_calls_returned"] = 0
    assert record["report"]["work"]["live_prediction_calls_returned"] == 2


def test_should_keep_paid_result_when_final_parent_boundary_refuses(monkeypatch):
    corrupted = False

    def guard():
        if corrupted:
            raise ValueError("preceding proof changed")

    def evaluate(*args, **kwargs):
        nonlocal corrupted
        corrupted = True
        return partial_report()

    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluate)
    record, _ = live.run_train_rollout(arm(), check_boundary=guard)
    assert record["status"] == "failed" and "preceding proof" in record["reason"]
    assert record["report"]["work"]["live_prediction_calls_returned"] == 2


@pytest.mark.parametrize("fault", ["nan", "backwards", "binding", "expired"])
def test_should_refuse_bad_clock_or_final_clock_callback_without_erasing_work(monkeypatch, fault):
    clock_fault = False
    captured: list[live.ToyBudgetSession] = []

    def clock():
        if clock_fault:
            if fault == "binding":
                object.__setattr__(captured[0], "progress", SimpleNamespace())
            return {"nan": float("nan"), "backwards": -1.0, "expired": 21.0}.get(fault, 0.0)
        return 0.0

    def evaluate(*args, **options):
        nonlocal clock_fault
        captured.append(options["budget"])
        clock_fault = True
        return partial_report()

    monkeypatch.setattr(live, "monotonic", clock)
    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluate)
    record, _ = live.run_train_rollout(arm(), check_boundary=lambda: None)
    assert record["status"] == "failed" and record["timing_complete"] is False
    assert record["report"]["work"]["live_prediction_calls_returned"] == 2
    assert record["budget_binding_preserved"] is (fault != "binding")
    assert record["elapsed_seconds"] >= 0


def test_should_refuse_post_probe_clock_before_evaluator_work(monkeypatch):
    owner = arm()
    owner["states"] = [(144, deepcopy(owner["states"][0][1]))]
    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", forbidden)
    record, _ = live.run_train_rollout(owner, check_boundary=lambda: None)
    assert record["status"] == "failed" and "benchmark128" in record["reason"]
    assert record["report"] is None and record["charged_updates"] == 0


def test_should_report_unknown_live_costs_when_delegate_loses_its_paid_report(monkeypatch):
    observed = []

    def evaluate(*args, **options):
        observed.extend(["reset", "prediction"])
        raise RuntimeError("lost paid report")

    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluate)
    record, _ = live.run_train_rollout(arm(), check_boundary=lambda: None)
    assert observed == ["reset", "prediction"]
    assert record["status"] == "failed" and record["evaluator_admitted"] is True
    assert record["report"] is None
    work = dict(total_score_forwards_completed=94, source_environment_steps_returned=90)
    live.add_work(work, record)
    assert work["learner_live_episodes"] is None
    assert work["live_prediction_forwards_completed"] is None
    assert work["total_prediction_forwards_completed"] is None
    assert work["total_prediction_rows_completed"] is None
    assert work["total_environment_steps_returned"] is None
    assert work["known_prediction_forwards_lower_bound"] == 94
    assert work["known_environment_steps_returned_lower_bound"] == 90
    assert work["failure_work_may_be_unobserved"] is True


def test_should_recheck_parent_after_the_last_clock_callback(monkeypatch):
    corrupted = False
    after_evaluation = False
    clock_calls = 0

    def clock():
        nonlocal clock_calls, corrupted
        if after_evaluation:
            clock_calls += 1
            if clock_calls == 2:
                corrupted = True
        return 0.0

    def evaluate(*args, **options):
        nonlocal after_evaluation
        after_evaluation = True
        return partial_report()

    def guard():
        if corrupted:
            raise ValueError("prior checkpoint changed at terminal clock")

    monkeypatch.setattr(live, "monotonic", clock)
    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluate)
    record, _ = live.run_train_rollout(arm(), check_boundary=guard)
    assert record["status"] == "failed" and "terminal clock" in record["reason"]
    assert record["report"]["work"]["live_prediction_calls_returned"] == 2
    assert record["budget_binding_preserved"] is True
