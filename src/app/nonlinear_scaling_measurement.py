"""Isolated bounded raw-scaling measurements on the existing delayed runtime.

Initial diagnostics use first training features without targets. Later probes
use only the last arrived training batch, after its native update. All heldout
scoring remains deferred to the existing stream. Probe gradients are ungated
objective partials, not CPC's applied update preconditioner. Total elapsed time
includes orchestration, native training, diagnostics and scoring; it is not a
training-only timing or evidence for a nonlinear scaling theorem.
"""

from copy import deepcopy
from math import isfinite
from time import monotonic
from typing import Any, Callable, cast

import numpy as np

from src.app.coupled_attribution import (
    AttributionBatch,
    _copy_batch,
    _validate_batch,
    _validate_controls,
    _validated_batches,
    run_attribution_stream,
)
from src.app.linear_scaling_measurement import _Timings, _magnitude
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.actor_ports import ForkableLearner
from src.core.coupled_energy import Array, CoupledParameters, feedforward_latents
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    _configuration,
    _raw_gradients,
    _validated_model,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
)
from src.core.coupled_predictive_coding import _parameter_arrays
from src.core.matched_backprop import evaluate_matched_backprop


def _bounded_batches(
    training: tuple[AttributionBatch, ...], heldout: tuple[AttributionBatch, ...]
) -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    if type(training) is not tuple or not 1 <= len(training) <= 8:
        raise ValueError("measurement training must contain 1..8 batches")
    if type(heldout) is not tuple or not 1 <= len(heldout) <= 4:
        raise ValueError("measurement heldout must contain 1..4 contexts")
    seen: set[str] = set()
    arrays: list[Array] = []
    for batches, role in ((training, "train"), (heldout, "final_test")):
        for batch in batches:
            if type(batch) is not AttributionBatch:
                raise ValueError("measurement batches must be exact AttributionBatch records")
            if type(batch.role) is not str or batch.role != role:
                raise ValueError(f"measurement batch role must be {role}")
            features, targets = batch.features, batch.targets
            if (
                type(features) is not np.ndarray
                or features.dtype != np.float64
                or features.ndim != 2
                or not 1 <= features.shape[0] <= 256
                or not 1 <= features.shape[1] <= 16
                or type(targets) is not np.ndarray
                or targets.dtype != np.float64
                or targets.shape != (len(features), 1)
            ):
                raise ValueError(
                    "measurement batches require float64 features <=256x16 and targets (B,1)"
                )
            _validate_batch(batch, role, seen)
            for row_id in batch.row_ids:
                if not row_id.isprintable():
                    raise ValueError("measurement row IDs must contain printable characters")
            arrays.extend((features, targets))
    for index, value in enumerate(arrays):
        if any(np.shares_memory(value, other) for other in arrays[index + 1 :]):
            raise ValueError("all measurement feature/target buffers must be disjoint")
    return _validated_batches(training, heldout)


def _bounded_state(state: CoupledScalingState, method: str, input_width: int) -> None:
    if type(state) is not CoupledScalingState or type(state.config) is not CoupledScalingConfig:
        raise ValueError("measurement requires exact coupled scaling state/config")
    config = state.config
    if (
        type(config.input_dim) is not int
        or not 1 <= config.input_dim <= 16
        or type(config.width) is not int
        or not 1 <= config.width <= 32
        or type(config.hidden_layers) is not int
        or not 1 <= config.hidden_layers <= 4
        or config.input_dim != input_width
    ):
        raise ValueError(
            "measurement geometry requires matching D<=16, width<=32, hidden layers<=4"
        )
    _configuration(config)
    if method in ("bp", "pc"):
        if config.circadian is not None or state.wake_state is not None:
            raise ValueError("bp/pc measurement methods require PC-only raw state")
    elif method == "neutral_cpc":
        if config.circadian is None or config.circadian.min_plasticity != 1:
            raise ValueError("neutral_cpc requires a circadian policy with unit minimum plasticity")
    elif config.circadian is None or config.circadian.min_plasticity >= 1:
        raise ValueError("active_cpc requires a circadian policy with minimum plasticity below 1")


def _norms(parameters: CoupledParameters) -> list[float]:
    return [_magnitude(value) for value in _parameter_arrays(parameters)]


def _nonnegative_mean(values: Array) -> float:
    maximum = float(np.max(values))
    return 0.0 if maximum == 0 else maximum * float(np.mean(values / maximum))


def _finite_diagnostic(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite_diagnostic(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_diagnostic(item)
    elif type(value) in (int, bool):
        return
    elif isinstance(value, float):
        try:
            finite = isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("measurement diagnostic contains a nonfinite number")
    elif value is not None and type(value) not in (str, bool):
        raise ValueError("measurement diagnostics must contain finite JSON values")


def _diagnostics(
    state: CoupledScalingState,
    batch: AttributionBatch | None,
    updates: int,
    method: str,
    *,
    initial_features: Array | None = None,
) -> dict[str, Any]:
    state = copy_coupled_scaling_state(state)
    config = state.config
    parameters = effective_coupled_parameters(state)
    if batch is None:
        if initial_features is None:
            raise ValueError("initial diagnostics require target-free initial_features")
        features = initial_features.copy()
    else:
        batch = _copy_batch(batch)
        features = batch.features.copy()
    latents = feedforward_latents(parameters, features)
    with np.errstate(over="raise", invalid="raise"):
        logits = latents[-1] @ parameters.output_weight + parameters.output_bias
    arrays = _parameter_arrays(parameters)
    wake = state.wake_state
    wake_summary = (
        None
        if wake is None
        else {
            "wake_updates": wake.wake_updates,
            "chemical_min": float(np.min(wake.chemical_state)),
            "chemical_max": float(np.max(wake.chemical_state)),
            "chemical_mean": _nonnegative_mean(wake.chemical_state),
            "gate_min": float(np.min(wake.plasticity_state)),
            "gate_max": float(np.max(wake.plasticity_state)),
            "gate_mean": _nonnegative_mean(wake.plasticity_state),
        }
    )
    result: dict[str, Any] = {
        "updates": updates,
        "parameter_entries": sum(value.size for value in arrays),
        "native_tensor_bytes": sum(value.nbytes for value in arrays)
        + (0 if wake is None else wake.chemical_state.nbytes + wake.plasticity_state.nbytes),
        "raw_parameter_norms": _norms(state.raw_parameters),
        "effective_parameter_norms": _norms(parameters),
        "feedforward_layer_rms": [_magnitude(value, rms=True) for value in latents],
        "logit_rms": _magnitude(logits, rms=True),
        "diagnostic_latent_rms": [_magnitude(value, rms=True) for value in latents],
        "energy_definition": None,
        "energy": None,
        "task_loss": None,
        "hidden_penalties": [],
        "raw_gradient_norms": None,
        "effective_gradient_norms": None,
        "diagnostic_inference_steps": 0,
        "diagnostic_objective_evaluations": 0,
        "training_context": None if batch is None else batch.context,
        "row_ids": None if batch is None else list(batch.row_ids),
        "initial_feature_reference": "first_training_features_without_targets"
        if batch is None
        else None,
        "forward_scales": list(config.forward_scales),
        "initial_stddevs": list(config.initial_stddevs),
        "raw_learning_rate_factor": config.raw_learning_rate_factor,
        "bias_scale": config.bias_scale,
        "effective_learning_rate_factors": list(config.effective_learning_rate_factors),
        "effective_bias_learning_rate_factor": config.effective_bias_learning_rate_factor,
        "precisions": list(config.precisions or ()),
        "wake_state_summary": wake_summary,
        "gradient_semantics": "ungated objective partials; CPC gates precondition the applied update",
        "energy_semantics": "target-free initial diagnostic"
        if batch is None
        else "postupdate probe, separate from preupdate native receipt energy",
    }
    if batch is not None:
        # Use a separate copy from the target-free diagnostic above. A probe
        # failure cannot contaminate retained checkpoint state or a later probe.
        probe = copy_coupled_scaling_state(state)
        if method == "bp":
            bp_evaluation = evaluate_matched_backprop(
                effective_coupled_parameters(probe), batch.features.copy(), batch.targets.copy()
            )
            gradients = bp_evaluation.parameter_gradients
            result.update(
                energy_definition=bp_evaluation.loss_definition,
                energy=bp_evaluation.loss,
                task_loss=bp_evaluation.loss,
                diagnostic_objective_evaluations=1,
            )
        else:
            _, _, model = _validated_model(probe)
            _, inferred, pc_evaluation, _ = model._training_with_policy(
                batch.features.copy(), batch.targets.copy(), 1.0, 2, 0.05, None
            )
            gradients = pc_evaluation.parameter_gradients
            result.update(
                energy_definition=pc_evaluation.energy_definition,
                energy=pc_evaluation.energy,
                task_loss=pc_evaluation.task_loss,
                hidden_penalties=list(pc_evaluation.hidden_penalties),
                diagnostic_latent_rms=[_magnitude(value, rms=True) for value in inferred],
                diagnostic_inference_steps=2,
                diagnostic_objective_evaluations=3,
            )
        result["effective_gradient_norms"] = _norms(gradients)
        result["raw_gradient_norms"] = _norms(_raw_gradients(config, gradients))
    _finite_diagnostic(result)
    return result


def run_nonlinear_scaling_case(
    learner: ForkableLearner[Array, Array, Array, CoupledScalingState],
    training: tuple[AttributionBatch, ...],
    heldout: tuple[AttributionBatch, ...],
    *,
    method: str,
    checkpoint_updates: tuple[int, ...],
    label_delay: int,
    budget: ToyBudgetSession,
    clock: Callable[[], float] = monotonic,
) -> dict[str, Any]:
    """Measure one fixed policy through arrived updates and deferred final scoring.

    The trusted snapshot/fork port controls its own internal allocations; the
    returned geometry is bounded before wrapper reconstruction. Public invalid
    preflight raises. Execution failures retain known checkpoint diagnostics and
    states, but never partial final scores or a successful stream result.
    """
    if type(method) is not str or method not in ("bp", "pc", "neutral_cpc", "active_cpc"):
        raise ValueError("method must be bp, pc, neutral_cpc, or active_cpc")
    training, heldout = _bounded_batches(training, heldout)
    _validate_controls(checkpoint_updates, label_delay, budget, len(training))
    if not callable(clock):
        raise ValueError("measurement clock must be callable")
    snapshot = learner.snapshot_state()
    _bounded_state(snapshot, method, training[0].features.shape[1])
    initial_state = copy_coupled_scaling_state(snapshot)
    for batch in (*training, *heldout):
        config = initial_state.config
        if (
            len(batch.features) * (config.input_dim + 1 + config.width * config.hidden_layers)
            > 1_000_000
        ):
            raise ValueError("measurement probe exceeds the activation-entry limit")
    owned = cast(ForkableLearner[Array, Array, Array, CoupledScalingState], learner.fork())
    if owned is learner:
        raise ValueError("measurement learner fork must be distinct from its source")
    owned.restore_state(copy_coupled_scaling_state(initial_state))
    timer: _Timings | None = None
    start_updates = budget.updates_completed
    diagnostics: list[dict[str, Any]] = []
    final_state: CoupledScalingState | None = None
    stream_result: dict[str, Any] | None = None
    status, reason = "complete", None

    def observe(updates: int, state: Any, batch: AttributionBatch | None) -> None:
        nonlocal initial_state, final_state
        assert timer is not None
        _bounded_state(state, method, training[0].features.shape[1])
        retained = copy_coupled_scaling_state(state)
        if updates == 0:
            initial_state = retained
        else:
            final_state = retained
        try:
            if updates == 0:
                operation = lambda: deepcopy(
                    _diagnostics(
                        copy_coupled_scaling_state(retained),
                        None,
                        updates,
                        method,
                        initial_features=training[0].features.copy(),
                    )
                )
            else:
                operation = lambda: deepcopy(
                    _diagnostics(
                        copy_coupled_scaling_state(retained),
                        None if batch is None else _copy_batch(batch),
                        updates,
                        method,
                    )
                )
            measured = timer.measure("diagnostic_seconds", operation)
            if type(measured) is not dict:
                raise ValueError("checkpoint diagnostic must return a dictionary")
            if type(measured.get("updates")) is not int or measured["updates"] != updates:
                raise ValueError("checkpoint diagnostic update count differs from its observation")
            _finite_diagnostic(measured)
            measured.setdefault("status", "complete")
            diagnostics.append(deepcopy(measured))
        except Exception as error:
            diagnostics.append(
                {
                    "updates": updates,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            raise

    try:
        timer = _Timings(clock)
        stream_result = run_attribution_stream(
            owned,
            training,
            heldout,
            checkpoint_updates=checkpoint_updates,
            label_delay=label_delay,
            budget=budget,
            observe_checkpoint=observe,
        )
        stream_result = deepcopy(stream_result)
        budget.before_final()
        timer.read()
    except Exception as error:
        status, reason, stream_result = "failed", f"{type(error).__name__}: {error}", None
        if timer is not None:
            try:
                timer.read()
            except Exception:
                pass
    timing = None if timer is None else timer.report()
    return {
        "status": status,
        "reason": reason,
        "committed_updates": budget.updates_completed - start_updates,
        "initial_state": initial_state,
        "final_state": final_state,
        "diagnostics": diagnostics,
        "stream_result": stream_result,
        "elapsed_seconds": None if timing is None else timing["elapsed_seconds"],
        "diagnostic_seconds": None if timing is None else timing["diagnostic_seconds"],
        "timing_scope": "after preflight through stream training/diagnostics/scoring; not training-only time",
        "timing_status": "unavailable"
        if timing is None
        else "complete"
        if status == "complete"
        else "partial",
    }
