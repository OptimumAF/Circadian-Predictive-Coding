"""Measure one bounded linear-PC scaling case without tuning its fixed controls.

Training-only diagnostics are retained at every committed update. Detached
states are scored on declared final-test features only after training finishes;
heldout labels are used exclusively in the application half-MSE metric. Failures
return partial training evidence and discard all heldout scores. This module
does not choose cases, seeds, rates, budgets, or comparative scientific claims.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from time import monotonic
from typing import Any, Callable, Generic, TypeVar

import numpy as np

from src.app.toy_execution_budget import ToyBudgetSession
from src.core.experience import require_identifier
from src.core.linear_pc_reference import Array, _matrix, evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    step_linear_pc,
)
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
    step_residual_linear_pc,
)

Result = TypeVar("Result")
MeasurementState = TypeVar("MeasurementState", LinearScalingState, LinearResidualState)


@dataclass(frozen=True)
class LinearMeasurementPorts(Generic[MeasurementState]):
    """Trusted native model operations shared by the isolated measurement loop."""

    copy_state: Callable[[MeasurementState], MeasurementState]
    effective_weights: Callable[[MeasurementState], tuple[Array, ...]]
    advance_state: Callable[[MeasurementState, Array, Array, float], MeasurementState]


def _copy_linear_state(state: LinearScalingState) -> LinearScalingState:
    effective_linear_weights(state)
    config = state.config
    detached = LinearScalingConfig(
        config.scheme, config.input_dim, config.width, config.hidden_layers, config.gamma0
    )
    return LinearScalingState(detached, state.raw_weights)


def _copy_residual_state(state: LinearResidualState) -> LinearResidualState:
    effective_residual_weights(state)
    config = state.config
    detached = LinearResidualConfig(
        config.scheme, config.input_dim, config.width, config.hidden_layers, config.gamma0
    )
    return LinearResidualState(detached, state.raw_weights)


def _linear_ports() -> LinearMeasurementPorts[LinearScalingState]:
    # Resolve the native functions when called, preserving existing integration
    # hooks without allowing callers to inject arbitrary model operations.
    return LinearMeasurementPorts(
        copy_state=lambda state: _copy_linear_state(state),
        effective_weights=lambda state: effective_linear_weights(state),
        advance_state=lambda state, features, targets, rate: (
            step_linear_pc(state, features, targets, rate).state
        ),
    )


def _residual_ports() -> LinearMeasurementPorts[LinearResidualState]:
    return LinearMeasurementPorts(
        copy_state=lambda state: _copy_residual_state(state),
        effective_weights=lambda state: effective_residual_weights(state),
        advance_state=lambda state, features, targets, rate: (
            step_residual_linear_pc(state, features, targets, rate).state
        ),
    )


@dataclass(frozen=True)
class ScalingBatch:
    role: str
    row_ids: tuple[str, ...]
    features: Array
    targets: Array


def _positive_number(value: float, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a native positive finite number")
    try:
        resolved = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


def _validate_batch(batch: ScalingBatch, role: str, width: int) -> None:
    if type(batch) is not ScalingBatch or type(batch.role) is not str or batch.role != role:
        raise ValueError(f"expected an exact ScalingBatch with role {role}")
    _matrix(batch.features, f"{role} features")
    _matrix(batch.targets, f"{role} targets")
    if batch.features.shape[1] != width or batch.targets.shape != (len(batch.features), 1):
        raise ValueError(f"{role} feature/target shapes differ from the configured input width")
    if type(batch.row_ids) is not tuple or len(batch.row_ids) != len(batch.features):
        raise ValueError(f"{role} row_ids must be a tuple matching its row count")
    for row_id in batch.row_ids:
        require_identifier(row_id, "row ID")
        if not row_id.isprintable():
            raise ValueError("scaling row IDs must contain printable characters")
    if len(set(batch.row_ids)) != len(batch.row_ids):
        raise ValueError(f"{role} row IDs must be unique")


def _copy_batch(batch: ScalingBatch) -> ScalingBatch:
    return ScalingBatch(batch.role, batch.row_ids, batch.features.copy(), batch.targets.copy())


def _prepare_inputs(
    initial: MeasurementState,
    training: ScalingBatch,
    heldout: ScalingBatch,
    updates: int,
    learning_rate: float,
    budget: ToyBudgetSession,
    clock: Callable[[], float],
    ports: LinearMeasurementPorts[MeasurementState],
) -> tuple[MeasurementState, ScalingBatch, ScalingBatch, float]:
    try:
        ports.effective_weights(initial)
    except FloatingPointError as error:
        raise ValueError("initial effective weights are numerically invalid") from error
    _validate_batch(training, "train", initial.config.input_dim)
    _validate_batch(heldout, "final_test", initial.config.input_dim)
    if set(training.row_ids).intersection(heldout.row_ids):
        raise ValueError("training and heldout row IDs must be disjoint")
    if any(
        np.shares_memory(train_array, heldout_array)
        for train_array in (training.features, training.targets)
        for heldout_array in (heldout.features, heldout.targets)
    ):
        raise ValueError("training and heldout arrays must not share memory")
    if type(updates) is not int or not 1 <= updates <= 512:
        raise ValueError("updates must be a native integer in 1..512")
    rate = _positive_number(learning_rate, "learning_rate")
    if type(budget) is not ToyBudgetSession:
        raise ValueError("scaling measurement requires an exact ToyBudgetSession")
    if not callable(clock):
        raise ValueError("measurement clock must be callable")
    return (
        ports.copy_state(initial),
        _copy_batch(training),
        _copy_batch(heldout),
        rate,
    )


class _Timings:
    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.last: float | None = None
        self.started = self.read()
        self.seconds = {
            "training_step_seconds": 0.0,
            "diagnostic_seconds": 0.0,
            "scoring_seconds": 0.0,
        }

    def read(self) -> float:
        raw = self.clock()
        if type(raw) not in (int, float):
            raise ValueError("measurement clock must return native finite seconds")
        try:
            now = float(raw)
        except OverflowError:
            raise ValueError("measurement clock must return finite seconds") from None
        if not isfinite(now):
            raise ValueError("measurement clock must return finite seconds")
        if self.last is not None and now < self.last:
            raise ValueError("measurement clock moved backwards")
        if self.last is not None and (
            not isfinite(now - self.last) or not isfinite(now - self.started)
        ):
            raise ValueError("measurement clock interval became nonfinite")
        self.last = now
        return now

    def measure(self, name: str, operation: Callable[[], Result]) -> Result:
        started = self.read()
        try:
            return operation()
        finally:
            elapsed = self.read() - started
            total = self.seconds[name] + elapsed
            if not isfinite(total):
                raise ValueError("measurement duration became nonfinite")
            self.seconds[name] = total

    def report(self) -> dict[str, float]:
        assert self.last is not None
        elapsed = self.last - self.started
        if not isfinite(elapsed):
            raise ValueError("elapsed measurement duration became nonfinite")
        return {**self.seconds, "elapsed_seconds": elapsed}


def _feedforward_layers(weights: tuple[Array, ...], features: Array) -> tuple[Array, ...]:
    if type(weights) is not tuple or len(weights) < 2:
        raise ValueError("weights must be a tuple with at least two layers")
    _matrix(features, "prediction features")
    width = features.shape[1]
    for weight in weights:
        _matrix(weight, "prediction weight")
        if weight.shape[0] != width:
            raise ValueError("prediction weights have incompatible adjacent/input widths")
        width = weight.shape[1]
    if width != 1:
        raise ValueError("prediction output width must equal 1")
    layers = []
    previous = features
    with np.errstate(over="raise", invalid="raise"):
        for weight in weights:
            previous = previous @ weight
            if not np.isfinite(previous).all():
                raise ValueError("linear prediction became nonfinite")
            layers.append(previous)
    return tuple(layers)


def predict_linear(weights: tuple[Array, ...], features: Array) -> Array:
    """Predict from validated row-oriented weights without targets or inference."""
    return _feedforward_layers(weights, features)[-1]


def _magnitude(value: Array, *, rms: bool = False) -> float:
    if not np.isfinite(value).all():
        raise ValueError("diagnostic array became nonfinite")
    maximum = float(np.max(np.abs(value)))
    if maximum == 0:
        return 0.0
    normalized = value / maximum
    squared = normalized * normalized
    reduced = float(np.mean(squared) if rms else np.sum(squared))
    result = maximum * sqrt(reduced)
    if not isfinite(result):
        raise ValueError("diagnostic magnitude became nonfinite")
    return result


def _diagnostics(
    state: MeasurementState,
    training: ScalingBatch,
    updates: int,
    ports: LinearMeasurementPorts[MeasurementState],
) -> dict[str, Any]:
    weights = ports.effective_weights(state)
    evaluation = evaluate_linear_equilibrium(weights, training.features, training.targets)
    layers = _feedforward_layers(weights, training.features)
    with np.errstate(over="raise", invalid="raise"):
        raw_gradients = tuple(
            scale * gradient
            for scale, gradient in zip(state.config.forward_scales, evaluation.weight_gradients)
        )
    if any(
        not isfinite(value) for value in (evaluation.energy, evaluation.mse, evaluation.rescaling)
    ):
        raise ValueError("equilibrium diagnostic became nonfinite")
    return {
        "updates": updates,
        "energy": evaluation.energy,
        "train_half_mse": evaluation.mse,
        "rescaling": evaluation.rescaling,
        "raw_weight_norms": [_magnitude(weight) for weight in state.raw_weights],
        "effective_weight_norms": [_magnitude(weight) for weight in weights],
        "effective_gradient_norms": [
            _magnitude(gradient) for gradient in evaluation.weight_gradients
        ],
        "raw_gradient_norms": [_magnitude(gradient) for gradient in raw_gradients],
        "feedforward_layer_rms": [_magnitude(layer, rms=True) for layer in layers],
        "equilibrium_layer_rms": [_magnitude(layer, rms=True) for layer in evaluation.latents],
    }


def _heldout_score(
    state: MeasurementState,
    heldout: ScalingBatch,
    ports: LinearMeasurementPorts[MeasurementState],
) -> float:
    prediction = predict_linear(ports.effective_weights(state), heldout.features.copy())
    _matrix(prediction, "heldout predictions")
    if prediction.shape != heldout.targets.shape:
        raise ValueError("heldout predictions must have shape (batch, 1)")
    with np.errstate(over="raise", invalid="raise"):
        error = prediction - heldout.targets
    rms = _magnitude(error, rms=True)
    result = 0.5 * rms * rms
    if not isfinite(result):
        raise ValueError("heldout half-MSE became nonfinite")
    return result


def _run_scaling_case(
    initial: MeasurementState,
    training: ScalingBatch,
    heldout: ScalingBatch,
    *,
    updates: int,
    learning_rate: float,
    budget: ToyBudgetSession,
    clock: Callable[[], float] = monotonic,
    ports: LinearMeasurementPorts[MeasurementState],
) -> dict[str, Any]:
    """Measure a fixed case; failed execution retains no heldout scores.

    Preflight errors raise ValueError. Attempts count admitted calls to the step
    kernel; committed counts record successful replacement states. Timed step
    calls include failed calls. Total time includes budget checks and orchestration
    overhead, but excludes input validation/copying and report assembly.
    """
    state, training, heldout, rate = _prepare_inputs(
        initial, training, heldout, updates, learning_rate, budget, clock, ports
    )
    timings = _Timings(clock)
    attempted, committed = 0, 0
    checkpoints: list[dict[str, Any]] = []
    states = [state]
    scores: list[float] | None = None
    status, reason = "complete", None
    try:
        budget.before_final()
        checkpoints.append(
            timings.measure("diagnostic_seconds", lambda: _diagnostics(state, training, 0, ports))
        )
        budget.before_final()
        for _ in range(updates):
            budget.before_update()
            attempted += 1
            state = timings.measure(
                "training_step_seconds",
                lambda: ports.advance_state(state, training.features, training.targets, rate),
            )
            committed += 1
            budget.record_update()
            states.append(state)
            budget.before_final()
            checkpoints.append(
                timings.measure(
                    "diagnostic_seconds", lambda: _diagnostics(state, training, committed, ports)
                )
            )
            budget.before_final()
        # All updates and training diagnostics must finish before any heldout use.
        scores = []
        for snapshot in states:
            budget.before_final()
            scores.append(
                timings.measure("scoring_seconds", lambda: _heldout_score(snapshot, heldout, ports))
            )
            budget.before_final()
        budget.before_final()
        timings.read()
    except Exception as error:
        status, reason, scores = "failed", f"{type(error).__name__}: {error}", None
        # Use the last valid timestamp if the clock itself failed. A failed case
        # never fabricates a completion time or exposes partial heldout scores.
        try:
            timings.read()
        except Exception:
            pass
    return {
        "status": status,
        "reason": reason,
        "attempted_updates": attempted,
        "committed_updates": committed,
        "checkpoints": checkpoints,
        "heldout_half_mse": scores,
        **timings.report(),
        "timing_note": "Total includes orchestration and budget overhead; excludes preflight and report assembly.",
    }


def run_linear_scaling_case(
    initial: LinearScalingState,
    training: ScalingBatch,
    heldout: ScalingBatch,
    *,
    updates: int,
    learning_rate: float,
    budget: ToyBudgetSession,
    clock: Callable[[], float] = monotonic,
) -> dict[str, Any]:
    """Measure an MLP case through the shared isolated measurement engine.

    Preflight errors raise ValueError. Execution failures retain partial training
    evidence and discard every heldout score, including failures after scoring.
    """
    return _run_scaling_case(
        initial,
        training,
        heldout,
        updates=updates,
        learning_rate=learning_rate,
        budget=budget,
        clock=clock,
        ports=_linear_ports(),
    )


def run_residual_scaling_case(
    initial: LinearResidualState,
    training: ScalingBatch,
    heldout: ScalingBatch,
    *,
    updates: int,
    learning_rate: float,
    budget: ToyBudgetSession,
    clock: Callable[[], float] = monotonic,
) -> dict[str, Any]:
    """Measure complete fixed-identity residual maps with the same isolation.

    Diagnostics and target-free predictions use complete effective matrices;
    raw gradient norms use only their trainable branch scale. This entrypoint
    advances the residual model and never invokes the ordinary MLP step.
    """
    return _run_scaling_case(
        initial,
        training,
        heldout,
        updates=updates,
        learning_rate=learning_rate,
        budget=budget,
        clock=clock,
        ports=_residual_ports(),
    )
