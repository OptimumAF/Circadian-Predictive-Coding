"""A completed live acquisition cannot be forged by final caller hooks."""

import pytest

import src.adapters.train_acquisition_pilot as pilot
from test_acquired_return_pilot import environment
from test_recurrent_key_door_pilot import Learner as RecurrentLearner


@pytest.mark.parametrize("boundary", [1, 2], ids=["final_validation", "final_budget_check"])
def test_should_refuse_completed_acquisition_changed_after_helper_returns(monkeypatch, boundary):
    captured = environment(monkeypatch)
    original = pilot.acquired_tasks.evaluate_acquired_tasks
    calls: list[bool] = []

    def evaluate(report, source, states, budget, **kwargs):
        original(report, source, states, budget, **kwargs)
        assert report["status"] == "complete" and report["initial_A_acquired"] is False
        before_final = budget.before_final

        def final_boundary():
            before_final()
            calls.append(True)
            if len(calls) == boundary:
                report["initial_A_acquired"] = True

        monkeypatch.setattr(budget, "before_final", final_boundary)

    monkeypatch.setattr(pilot.acquired_tasks, "evaluate_acquired_tasks", evaluate)
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 320
    assert len(captured["sessions"]) == 12
    assert "completed acquired diagnostic changed" in str(report["reason"]) + str(
        report["readback_error"]
    )
