"""Run one synchronous native wake step with existing complete-boundary budgets.

The adapter owns its loss, layout, training settings and model snapshot. This
function neither scores roles nor implements actor concurrency or promotion.
"""

from typing import Callable, TypeVar

from src.app.toy_execution_budget import ToyBudgetSession
from src.core.learner_ports import (
    NativeLearner,
    TrainingDiagnostic,
    validate_learning_rate_multiplier,
    validate_gate_setting,
)

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


def update_learner(
    learner: NativeLearner[Features, Targets, Prediction, State],
    features: Features,
    targets: Targets,
    budget: ToyBudgetSession,
    *,
    on_started: Callable[[], None] | None = None,
    on_completed: Callable[[TrainingDiagnostic], None] | None = None,
    learning_rate_multiplier: float = 1.0,
    gate_setting: str = "unchanged",
) -> TrainingDiagnostic:
    """Preserve committed work when a post-update clock/RSS check stops the run."""
    if type(budget) is not ToyBudgetSession:
        raise ValueError("learner step requires the existing ToyBudgetSession")
    validate_learning_rate_multiplier(learning_rate_multiplier)
    validate_gate_setting(gate_setting)
    gate_update = getattr(learner, "train_batch_with_gate_setting", None)
    if gate_setting != "unchanged":
        gate_validator = getattr(learner, "validate_gate_setting", None)
        if (
            learning_rate_multiplier != 1
            or not callable(gate_update)
            or not callable(gate_validator)
        ):
            raise ValueError("native gate executor or gate/rate combination is unsupported")
        gate_validator(gate_setting)
    rate_update = getattr(learner, "train_batch_with_rate_multiplier", None)
    if learning_rate_multiplier != 1 and not callable(rate_update):
        raise ValueError("native learning-rate executor is unsupported")
    if budget.budget.max_hidden_width is not None or budget.budget.max_replay_examples is not None:
        raise ValueError("learner step does not enforce width or replay limits")
    if budget.budget.max_process_rss_bytes is not None and budget.process_rss_sampler is None:
        raise ValueError("learner RSS budget requires an attached sampler")
    budget.before_update()
    if on_started is not None:
        on_started()
    if gate_setting != "unchanged":
        assert callable(gate_update)
        diagnostic = gate_update(features, targets, gate_setting)
    elif learning_rate_multiplier == 1:
        diagnostic = learner.train_batch(features, targets)
    else:
        assert callable(rate_update)  # Validated before any budget/native callbacks.
        diagnostic = rate_update(features, targets, learning_rate_multiplier)
    budget.record_update()
    if on_completed is not None:
        on_completed(diagnostic)
    # Why this: the native update is already complete. A late resource stop
    # retains its work/state rather than pretending it never ran.
    budget.before_final()
    return diagnostic
