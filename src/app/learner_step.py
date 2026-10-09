"""Run one synchronous native wake step with existing complete-boundary budgets.

The adapter owns its loss, layout, training settings and model snapshot. This
function neither scores roles nor implements actor concurrency or promotion.
"""

from typing import Callable, TypeVar

from src.app.toy_execution_budget import ToyBudgetSession
from src.core.learner_ports import NativeLearner, TrainingDiagnostic

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
) -> TrainingDiagnostic:
    """Preserve committed work when a post-update clock/RSS check stops the run."""
    if type(budget) is not ToyBudgetSession:
        raise ValueError("learner step requires the existing ToyBudgetSession")
    if budget.budget.max_hidden_width is not None or budget.budget.max_replay_examples is not None:
        raise ValueError("learner step does not enforce width or replay limits")
    if budget.budget.max_process_rss_bytes is not None and budget.process_rss_sampler is None:
        raise ValueError("learner RSS budget requires an attached sampler")
    budget.before_update()
    if on_started is not None:
        on_started()
    diagnostic = learner.train_batch(features, targets)
    budget.record_update()
    if on_completed is not None:
        on_completed(diagnostic)
    # Why this: the native update is already complete. A late resource stop
    # retains its work/state rather than pretending it never ran.
    budget.before_final()
    return diagnostic
