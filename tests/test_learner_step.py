"""The common app step keeps native payloads opaque and reuses boundary budgets."""

from dataclasses import dataclass

import pytest

from src.app.learner_step import update_learner
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.learner_ports import TrainingDiagnostic


@dataclass
class TextLearner:
    calls: int = 0

    def train_batch(self, features: dict[str, str], targets: str) -> TrainingDiagnostic:
        assert features == {"text": "local"} and targets == "label"
        self.calls += 1
        return TrainingDiagnostic("text_native_objective_v1", -2.0)

    def predict(self, features: dict[str, str]) -> list[str]:
        return list(features.values())

    def snapshot_state(self) -> int:
        return self.calls

    def restore_state(self, snapshot: int) -> None:
        self.calls = snapshot


def test_should_use_non_array_inputs_and_native_diagnostic_without_shared_loss_assumptions():
    learner = TextLearner()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    result = update_learner(learner, {"text": "local"}, "label", budget)
    assert result == TrainingDiagnostic("text_native_objective_v1", -2.0)
    assert learner.calls == budget.updates_completed == 1


def test_should_refuse_budget_before_calling_the_learner():
    learner = TextLearner()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    with pytest.raises(ToyExecutionStopped) as caught:
        update_learner(learner, {"text": "local"}, "label", budget)
    assert caught.value.stop.reason == "max_training_updates"
    assert learner.calls == budget.updates_completed == 0


def test_should_preserve_completed_work_when_post_update_wall_budget_stops():
    learner = TextLearner()
    times = iter((0.0, 0.0, 2.0))
    budget = ToyBudgetSession(ToyExecutionBudget(max_wall_seconds=1.0), lambda: next(times))
    with pytest.raises(ToyExecutionStopped) as caught:
        update_learner(learner, {"text": "local"}, "label", budget)
    assert caught.value.stop.reason == "max_wall_seconds"
    assert learner.calls == caught.value.stop.updates_completed == 1


@pytest.mark.parametrize("limit", ["width", "replay", "rss"])
def test_should_refuse_uncomposed_limits_before_calling_the_learner(limit):
    learner = TextLearner()
    limits = {
        "width": ToyExecutionBudget(max_hidden_width=4),
        "replay": ToyExecutionBudget(max_replay_examples=4),
        "rss": ToyExecutionBudget(max_process_rss_bytes=1024),
    }
    budget = ToyBudgetSession(limits[limit], lambda: 0.0)
    with pytest.raises(ValueError):
        update_learner(learner, {"text": "local"}, "label", budget)
    assert learner.calls == budget.updates_completed == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_should_reject_nonfinite_native_diagnostic(value):
    with pytest.raises(ValueError, match="finite"):
        TrainingDiagnostic("native_v1", value)
