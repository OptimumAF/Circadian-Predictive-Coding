"""Bounded linear catalog policy and truthful work schema; no backend or IO."""

from dataclasses import dataclass

from src.core.clm_head import _integer

OBJECTIVE = "clm_union_catalog_minimum_change_squared_error_v1"
CONTROL_KIND = "untrained_uniform_union_linear"


def uniform_linear_controls():
    return {version: [[1 / 3] * 3 for _ in range(4)] for version in ("v1", "v2")}


@dataclass(frozen=True)
class CLMLinearCatalogPolicy:
    max_updates: int = 12
    max_rows: int = 4
    max_readouts: int = 8

    def __post_init__(self):
        _integer(self.max_updates, "max_updates", 1, 12)
        _integer(self.max_rows, "max_rows", 2, 16)
        _integer(self.max_readouts, "max_readouts", 1, 8)


def linear_catalog_work(updates=0, rows=0, reads=0, prediction_rows=0):
    """Expected completed work; failed attempts are retained by the learner."""
    return dict(
        train_attempts=updates,
        training_rows_attempted=rows,
        solve_attempts=updates,
        solve_returns=updates,
        commits=updates,
        prediction_attempts=reads,
        prediction_returns=reads,
        prediction_rows_attempted=prediction_rows,
        choice_prediction_attempts=reads,
        choice_prediction_returns=reads,
        choice_candidate_rows_attempted=3 * reads,
        encoder_forwards=0,
        native_head_forwards=0,
        optimizer_steps=0,
        unreturned_work_unknown=False,
    )
