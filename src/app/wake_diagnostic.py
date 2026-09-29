"""Identify genuine NumPy wake diagnostics from existing update returns.

Inputs are one successful core update result and its train-only role facts.
Outputs are typed observations and a six-trial order check. This module
does not compute a second score, train a model, or release a test role.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from src.core.backprop_mlp import BackpropTrainResult, NUMPY_BACKPROP_LOSS_ID
from src.core.circadian_predictive_coding import (
    CircadianTrainResult,
    NUMPY_CIRCADIAN_ENERGY_ID,
)
from src.core.predictive_coding import PredictiveCodingTrainResult, NUMPY_PC_ENERGY_ID

if TYPE_CHECKING:
    from src.app.continual_trigger_replay_runner import TriggerReplayTrainingResult


@dataclass(frozen=True)
class WakeDiagnostic:
    """One pre-parameter-update diagnostic from one train-only wake call."""

    seed: int
    arm: str
    phase: str
    epoch: int
    global_epoch: int
    method: str
    train_role_hash: str
    metric_name: str
    metric_definition: str
    metric_value: float
    measurement_stage: str


_METRIC_SPEC = {
    "backprop": ("loss", NUMPY_BACKPROP_LOSS_ID, "pre_parameter_update"),
    "predictive_coding": (
        "energy",
        NUMPY_PC_ENERGY_ID,
        "post_inference_pre_parameter_update",
    ),
    "circadian_predictive_coding": (
        "energy",
        NUMPY_CIRCADIAN_ENERGY_ID,
        "post_inference_pre_parameter_update",
    ),
}


def metric_spec(method: str) -> tuple[str, str, str]:
    """Return the frozen metric name, definition, and measurement stage."""
    try:
        return _METRIC_SPEC[method]
    except KeyError as exc:
        raise ValueError(f"unknown wake diagnostic method: {method}") from exc


def diagnostic_from_update(
    result: BackpropTrainResult | PredictiveCodingTrainResult | CircadianTrainResult,
    *,
    seed: int,
    arm: str,
    phase: str,
    epoch: int,
    global_epoch: int,
    method: str,
    train_role_hash: str,
) -> WakeDiagnostic:
    """Copy the diagnostic returned by the already completed update."""
    expected_type = {
        "backprop": BackpropTrainResult,
        "predictive_coding": PredictiveCodingTrainResult,
        "circadian_predictive_coding": CircadianTrainResult,
    }.get(method)
    if expected_type is None or type(result) is not expected_type:
        raise ValueError("wake diagnostic method and core result type differ")
    name, definition, stage = metric_spec(method)
    if isinstance(result, BackpropTrainResult):
        value = float(result.loss)
    else:
        if result.energy_definition != definition:
            raise ValueError("wake diagnostic energy definition differs")
        value = float(result.energy)
    if not isfinite(value):
        raise ValueError("wake diagnostic must be finite")
    return WakeDiagnostic(
        seed,
        arm,
        phase,
        epoch,
        global_epoch,
        method,
        train_role_hash,
        name,
        definition,
        value,
        stage,
    )


def validate_wake_diagnostic_trials(trials: tuple[TriggerReplayTrainingResult, ...]) -> None:
    """Check every measured update against preflighted v14 opportunity work."""
    for trial in trials:
        methods = tuple(work.method for work in trial.wake_work)
        expected = [
            (
                trial.seed,
                trial.arm,
                item.phase,
                item.epoch,
                item.global_epoch,
                method,
                item.train_role_hash,
            )
            for item in trial.opportunities
            for method in methods
        ]
        if len(trial.wake_diagnostics) != len(expected):
            raise ValueError("wake diagnostic count differs from successful update work")
        for row, identity in zip(trial.wake_diagnostics, expected, strict=True):
            if (
                (
                    row.seed,
                    row.arm,
                    row.phase,
                    row.epoch,
                    row.global_epoch,
                    row.method,
                    row.train_role_hash,
                )
                != identity
                or (row.metric_name, row.metric_definition, row.measurement_stage)
                != metric_spec(row.method)
                or type(row.metric_value) is not float
                or not isfinite(row.metric_value)
            ):
                raise ValueError("wake diagnostic identity, definition, or value differs")
        if any(work.optimizer_updates != len(trial.opportunities) for work in trial.wake_work):
            raise ValueError("wake diagnostic updates differ from matched work")
