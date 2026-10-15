"""One frozen small nonlinear scaling pilot with matched raw-coordinate learners.

All methods share data, initialization and rate factors within a shape/scheme/seed.
Final scores follow each case's complete training; no outcomes configure later
cases. Every planned row and failure remains in the report. No policy tuning.
"""

from dataclasses import asdict, is_dataclass
from contextlib import nullcontext
from itertools import product
import json
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.adapters.scaled_matched_backprop_learner import ScaledMatchedBackpropLearner
from src.app.coupled_attribution import AttributionBatch
from src.app.nonlinear_scaling_measurement import run_nonlinear_scaling_case
from src.app.nonlinear_scaling_summary import (
    CHECKPOINTS,
    DEPTHS,
    METHODS,
    LEGACY_SCHEMES,
    SCHEMES,
    SEEDS,
    WIDTHS,
    planned_keys,
    summarize_nonlinear_scaling,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ForkableLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import Array
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    initialize_coupled_scaling,
)
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "nonlinear_raw_scaling_fixed_pilot_v1"
RESOURCES = PilotResources(
    training_updates=288, environment_steps=0, replay_bytes=0, storage_bytes=16 * 1024**2
)
LEGACY_PROTOCOL_ID = "nonlinear_raw_legacy_extension_v1"
LEGACY_RESOURCES = PilotResources(
    training_updates=144, environment_steps=0, replay_bytes=0, storage_bytes=16 * 1024**2
)


def _selected_schemes(legacy_only: bool) -> tuple[str, ...]:
    if type(legacy_only) is not bool:
        raise ValueError("legacy_only must be an exact boolean")
    return LEGACY_SCHEMES if legacy_only else SCHEMES


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"unsupported pilot record: {type(value).__name__}")


def describe_protocol(*, legacy_only: bool = False) -> dict[str, Any]:
    schemes = _selected_schemes(legacy_only)
    resources = LEGACY_RESOURCES if legacy_only else RESOURCES
    protocol = {
        "protocol_id": LEGACY_PROTOCOL_ID if legacy_only else PROTOCOL_ID,
        "classification": "exploratory_fixed_data_pilot",
        "seeds": SEEDS,
        "widths": WIDTHS,
        "hidden_depths": DEPTHS,
        "schemes": schemes,
        "methods": METHODS,
        "input_dim": 2,
        "mean_field_gamma0": 1.5,
        "precisions": "literal1 per hidden layer; BP metadata only",
        "bias_policy": "1/sqrt(stored raw rate factor); actual factors retained",
        "learning_rate": 0.01,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "active_cpc_config": asdict(CoupledCircadianConfig()),
        "neutral_cpc_config": asdict(CoupledCircadianConfig.matched_pc_control()),
        "train_rng": 7427,
        "final_rng": 9233,
        "distribution": "uniform[-1,1]",
        "target_rule": "x0 + 0.25*x1 >= 0",
        "training_batches": 3,
        "batch_size": 8,
        "heldout_rows": 64,
        "label_delay": 2,
        "checkpoint_updates": CHECKPOINTS,
        "primary_metric": "probability_bce_clip_1e-12_lower_better",
        "secondary_metric": "accuracy_threshold_ge0.5_higher_better",
        "intervals": "nominal marginal95%t df2 critical4.303, n3 complete fixed-data seeds; no multiplicity/generalization guarantee",
        "score_release": "after ALL training in each case, output at end; no outcome configures other cases",
        "diagnostics": "target-free initial probe on first training features; later post-update probe on last arrived training batch; PC fixed-step, not equilibrium; gradients ungated",
        "planned_runs": 48 if legacy_only else 96,
        "planned_training_updates": 144 if legacy_only else 288,
        "nominal_training_objective_evaluations": 504 if legacy_only else 1008,
        "nominal_probe_objective_evaluations": 360 if legacy_only else 720,
        "resources": asdict(resources),
        "limitations": "Local nonlinear extension;3 training updates/small synthetic data do not establish convergence, width invariance, equal compute or general CPC benefit.",
    }
    if legacy_only:
        protocol["legacy_gamma0"] = 1.0
    return protocol


def build_data() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    train_rng, final_rng = np.random.default_rng(7427), np.random.default_rng(9233)
    training = []
    for index in range(3):
        features = train_rng.uniform(-1, 1, (8, 2))
        targets = (features[:, [0]] + 0.25 * features[:, [1]] >= 0).astype(np.float64)
        training.append(
            AttributionBatch(
                "train",
                "stationary",
                tuple(f"train-{index}-{row}" for row in range(8)),
                features,
                targets,
            )
        )
    features = final_rng.uniform(-1, 1, (64, 2))
    targets = (features[:, [0]] + 0.25 * features[:, [1]] >= 0).astype(np.float64)
    final = AttributionBatch(
        "final_test", "stationary", tuple(f"final-{row}" for row in range(64)), features, targets
    )
    return tuple(training), (final,)


def build_learner(
    width: int, depth: int, scheme: str, seed: int, method: str, *, legacy_only: bool = False
) -> ForkableLearner[Array, Array, Array, CoupledScalingState]:
    schemes = _selected_schemes(legacy_only)
    if (width, depth, scheme, seed, method) not in planned_keys(schemes=schemes):
        raise ValueError("learner request must match a prespecified pilot case")
    chemistry = (
        CoupledCircadianConfig.matched_pc_control()
        if method == "neutral_cpc"
        else CoupledCircadianConfig()
        if method == "active_cpc"
        else None
    )
    state = initialize_coupled_scaling(
        CoupledScalingConfig(
            scheme,
            2,
            width,
            depth,
            gamma0=1.5 if scheme == "mean_field" else 1.0,
            precisions=(1.0,) * depth,
            circadian=chemistry,
        ),
        seed,
    )
    if method == "bp":
        return ScaledMatchedBackpropLearner(state, learning_rate=0.01)
    return CoupledScalingLearner(
        state, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )


def _run_one(
    key: tuple[int, int, str, int, str],
    started: float,
    sampler: ProcessRssSampler,
    *,
    legacy_only: bool = False,
) -> dict[str, Any]:
    _selected_schemes(legacy_only)
    resources = LEGACY_RESOURCES if legacy_only else RESOURCES
    width, depth, scheme, seed, method = key
    row: dict[str, Any] = dict(width=width, depth=depth, scheme=scheme, seed=seed, method=method)
    remaining = resources.wall_seconds - (monotonic() - started)
    if remaining <= 0:
        return {
            **row,
            "status": "not_run",
            "reason": "global wall budget exhausted",
            "committed_updates": 0,
            "case_result": None,
        }
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=3,
            max_wall_seconds=remaining,
            max_process_rss_bytes=resources.process_memory_bytes,
        ),
        monotonic,
    )
    try:
        budget.attach_memory(sampler)
        budget.before_final()
        training, final = build_data()
        learner = build_learner(*key, legacy_only=True) if legacy_only else build_learner(*key)
        budget.before_final()
        result = run_nonlinear_scaling_case(
            learner,
            training,
            final,
            method=method,
            checkpoint_updates=CHECKPOINTS,
            label_delay=2,
            budget=budget,
        )
        row.update(status=result["status"], reason=result["reason"], case_result=result)
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}", case_result=None)
    row["committed_updates"] = budget.updates_completed
    row["committed_training_examples"] = 8 * budget.updates_completed
    row["nominal_completed_training_objective_evaluations"] = (
        2 if method == "bp" else 4
    ) * budget.updates_completed
    result = row["case_result"]
    row["completed_probe_objective_evaluations"] = (
        sum(d.get("diagnostic_objective_evaluations", 0) for d in result["diagnostics"])
        if type(result) is dict
        else 0
    )
    row["diagnostic_work_complete"] = type(result) is dict and all(
        "diagnostic_objective_evaluations" in d for d in result["diagnostics"]
    )
    return row


def _neutral_parity(
    rows: list[dict[str, Any]], *, schemes: tuple[str, ...] = SCHEMES
) -> list[dict[str, Any]]:
    indexed = {(r["width"], r["depth"], r["scheme"], r["seed"], r["method"]): r for r in rows}
    output = []
    for width, depth, scheme, seed in product(WIDTHS, DEPTHS, schemes, SEEDS):
        left, right = (
            indexed[width, depth, scheme, seed, method] for method in ("pc", "neutral_cpc")
        )
        equal = None
        if left["status"] == right["status"] == "complete":

            def parameters(row: dict[str, Any], stage: str) -> str:
                return json.dumps(
                    row["case_result"][stage].raw_parameters, default=json_default, sort_keys=True
                )

            equal = all(
                parameters(left, stage) == parameters(right, stage)
                for stage in ("initial_state", "final_state")
            )
        output.append(dict(width=width, depth=depth, scheme=scheme, seed=seed, exact=equal))
    return output


def _planned_rows(*, schemes: tuple[str, ...] = SCHEMES) -> list[dict[str, Any]]:
    return [
        dict(
            width=w,
            depth=d,
            scheme=s,
            seed=seed,
            method=m,
            status="not_run",
            reason="pilot stopped before execution",
            committed_updates=0,
            case_result=None,
        )
        for w, d, s, seed, m in planned_keys(schemes=schemes)
    ]


def unstarted_pilot_report(
    reason: str, started: float, *, legacy_only: bool = False
) -> dict[str, Any]:
    """Retain every planned row after CLI sampler startup fails; never retry work."""
    schemes = _selected_schemes(legacy_only)
    if legacy_only:
        return _assemble_report(
            _planned_rows(schemes=schemes), [reason], None, started, legacy_only=True
        )
    return _assemble_report(_planned_rows(), [reason], None, started)


def run_pilot(
    *,
    sampler: ProcessRssSampler | None = None,
    started: float | None = None,
    legacy_only: bool = False,
) -> dict[str, Any]:
    """Run with an owned sampler, or a caller's already active command sampler."""
    schemes = _selected_schemes(legacy_only)
    resources = LEGACY_RESOURCES if legacy_only else RESOURCES
    validate_local_pilot_request(PilotRequest(resources))
    started = monotonic() if started is None else started
    keys = planned_keys(schemes=schemes) if legacy_only else planned_keys()
    rows = _planned_rows(schemes=schemes) if legacy_only else _planned_rows()
    errors: list[str] = []
    try:
        context = ProcessRssSampler() if sampler is None else nullcontext(sampler)
        with context as active_sampler:
            for index, key in enumerate(keys):
                rows[index] = (
                    _run_one(key, started, active_sampler, legacy_only=True)
                    if legacy_only
                    else _run_one(key, started, active_sampler)
                )
            if legacy_only:
                return _assemble_report(rows, errors, active_sampler, started, legacy_only=True)
            return _assemble_report(rows, errors, active_sampler, started)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    if legacy_only:
        return _assemble_report(rows, errors, None, started, legacy_only=True)
    return _assemble_report(rows, errors, None, started)


def _assemble_report(
    rows: list[dict[str, Any]],
    errors: list[str],
    sampler: ProcessRssSampler | None,
    started: float,
    *,
    legacy_only: bool = False,
) -> dict[str, Any]:
    schemes = _selected_schemes(legacy_only)
    resources = LEGACY_RESOURCES if legacy_only else RESOURCES
    try:
        summaries = (
            summarize_nonlinear_scaling(tuple(rows), schemes=schemes)
            if legacy_only
            else summarize_nonlinear_scaling(tuple(rows))
        )
    except ValueError as error:
        summaries = []
        errors.append(f"aggregation: {error}")
    parity = _neutral_parity(rows, schemes=schemes) if legacy_only else _neutral_parity(rows)
    if any(item["exact"] is False for item in parity):
        errors.append("PC/neutral raw parameter parity failed")
    # Why: hold aggregation/parity/data allocations inside the live RSS interval.
    data = build_data()
    protocol = describe_protocol(legacy_only=True) if legacy_only else describe_protocol()
    elapsed = monotonic() - started
    try:
        if sampler is None:
            raise RuntimeError("RSS interval unavailable after pilot failure")
        sampler.sample()
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    within_resources = (
        elapsed <= resources.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= resources.process_memory_bytes
    )
    return {
        "schema_version": 1,
        "protocol": protocol,
        "runs": rows,
        "status": "complete"
        if within_resources and not errors and all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "summaries": summaries,
        "pc_neutral_parameter_parity": parity,
        "elapsed_seconds": elapsed,
        "observed_process_rss": memory,
        "within_resources": within_resources,
        "global_errors": errors,
        "data": data,
        "work_scope": "Report telemetry through aggregation/parity/data construction. CLI command receipt additionally covers imports, JSON encoding and stdout emission. Nominal kernels omit failed attempts/copies/overhead; not FLOPs/equal compute. RSS is sampled process memory, not model-only allocation or an exact allocation ceiling.",
    }
