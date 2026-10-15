"""Fixed local allocated-time comparison composition and resource observations.

The application owns training/scoring isolation and measured windows. This
adapter fixes data, case order and controls; no tuning or seed selection inputs.
"""

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from src.app.coupled_attribution import AttributionBatch
from src.app.coupled_inference_summary import (
    CASE_ORDER,
    MAX_UPDATES,
    PLANNED_SEEDS,
    WINDOW_SECONDS,
    summarize_inference_comparison,
)
from src.app.timed_coupled_inference import run_timed_inference_case
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSampler

RESOURCES = PilotResources(training_updates=480, environment_steps=0, replay_bytes=0)
GRANULARITY_WINDOW_SECONDS = 0.03
MAX_ITERATION_SECONDS = 0.003


def _validate_option(granularity_aware: bool) -> None:
    if type(granularity_aware) is not bool:
        raise ValueError("granularity_aware must be an exact native boolean")


def describe_protocol(*, granularity_aware: bool = False) -> dict[str, Any]:
    _validate_option(granularity_aware)
    description = {
        "protocol_id": "coupled_allocated_time_inference_v1",
        "classification": "exploratory_fixed_data_allocated_wall_time_pilot",
        "seeds": list(PLANNED_SEEDS),
        "case_order": [list(key) for key in CASE_ORDER],
        "geometry": {"input_dim": 2, "hidden_dims": [8, 8], "precisions": [1.0, 1.0]},
        "learning_rate": 0.03,
        "inference_steps": 6,
        "inference_learning_rate": 0.1,
        "adaptive_policy": asdict(CoupledInferencePolicy()),
        "training_seconds": WINDOW_SECONDS,
        "maximum_window_seconds": 1.1 * WINDOW_SECONDS,
        "max_updates_per_case": MAX_UPDATES,
        "training_rows": 16,
        "heldout_rows": 128,
        "training_rng": 7421,
        "heldout_rng": 9221,
        "distribution": "uniform[-1,1]",
        "target_rule": "x0 + 0.25*x1 >= 0",
        "metrics": {"bce": "probability BCE clip1e-12", "accuracy": "threshold>=0.5"},
        "score_release": "initial/final snapshots after all training in each case; no feedback",
        "timing_basis": "whole train calls plus all in-loop resource/deadline/record overhead",
        "excluded_common_work": "preflight/model/data copies, initial snapshot, deferred scoring",
        "failure_policy": "retain unscored; no warmup, calibration, retry, replacement or extension",
        "intervals": "nominal marginal95% t df2 critical4.303; initialization AND runtime variability",
        "limitations": "n3 fixeddata; 10ms windows; up to10% overshoot; not CPUtime or matched exposure; no general speed/learning claim",
        "resources": asdict(RESOURCES),
    }
    if granularity_aware:
        description.update(
            protocol_id="coupled_allocated_time_inference_granularity_v2",
            training_seconds=GRANULARITY_WINDOW_SECONDS,
            maximum_window_seconds=1.1 * GRANULARITY_WINDOW_SECONDS,
            max_iteration_seconds=MAX_ITERATION_SECONDS,
            timing_admission={
                "source": "artifacts/runs/coupled-inference-comparison-20261015/pilot.json",
                "basis": "All six timing records only: max(call) + window - sum(calls); no scores used",
                "largest_retained_iteration_envelope_seconds": 0.00232390001474414,
                "rounding": "Up to next 1ms; allocate ten times this ceiling",
                "measurement": "First interval starts at window_start, subsequent starts partition native iterations, final extends through terminal window_end",
                "qualification": "Inclusive observed iteration ceiling and 10% whole-window overshoot; excess fails unscored, no retry",
                "limitations": "Retained envelope is not a future scheduling guarantee",
            },
            limitations="n3 fixeddata; 30ms allocated windows/3ms observed iteration ceiling; up to10% overshoot; not CPUtime or matched exposure; no general speed/learning claim",
        )
    return description


def build_batches() -> tuple[AttributionBatch, AttributionBatch]:
    batches = []
    for role, seed, count in (("train", 7421, 16), ("final_test", 9221, 128)):
        features = np.random.default_rng(seed).uniform(-1, 1, (count, 2))
        targets = (features[:, [0]] + 0.25 * features[:, [1]] >= 0).astype(np.float64)
        batches.append(
            AttributionBatch(
                role, "A", tuple(f"{role}-{i}" for i in range(count)), features, targets
            )
        )
    return batches[0], batches[1]


def _initial_record(model: CoupledPredictiveCodingNetwork) -> dict[str, Any]:
    parameters = model.snapshot_parameters()
    return {
        "hidden_weights": [a.tolist() for a in parameters.hidden_weights],
        "hidden_biases": [a.tolist() for a in parameters.hidden_biases],
        "output_weight": parameters.output_weight.tolist(),
        "output_bias": parameters.output_bias.tolist(),
        "precisions": list(model.precisions),
    }


def _run_one(
    seed: int,
    method: str,
    initial: CoupledPredictiveCodingNetwork,
    started: float,
    sampler: ProcessRssSampler,
    *,
    granularity_aware: bool = False,
) -> dict[str, Any]:
    _validate_option(granularity_aware)
    case_started = perf_counter()
    row: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "status": "not_run",
        "reason": "global wall budget exhausted",
    }
    remaining = RESOURCES.wall_seconds - (perf_counter() - started)
    if remaining <= 0:
        return row
    try:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=MAX_UPDATES,
                max_wall_seconds=remaining,
                max_process_rss_bytes=RESOURCES.process_memory_bytes,
            ),
            perf_counter,
        )
        budget.attach_memory(sampler)
        training, heldout = build_batches()
        result = run_timed_inference_case(
            initial,
            training,
            heldout,
            training_seconds=GRANULARITY_WINDOW_SECONDS if granularity_aware else WINDOW_SECONDS,
            max_updates=MAX_UPDATES,
            learning_rate=0.03,
            inference_steps=6,
            inference_learning_rate=0.1,
            policy=CoupledInferencePolicy() if method == "adaptive" else None,
            budget=budget,
            max_iteration_seconds=MAX_ITERATION_SECONDS if granularity_aware else None,
        )
        row.update(status=result["status"], reason=result["reason"], result=result)
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    row["whole_case_seconds"] = perf_counter() - case_started
    return row


def run_pilot(*, granularity_aware: bool = False) -> dict[str, Any]:
    """Run the six frozen cases once, preserving every failed/unreached row."""
    _validate_option(granularity_aware)
    validate_local_pilot_request(PilotRequest(RESOURCES))
    started = perf_counter()
    rows = [
        dict(seed=seed, method=method, status="not_run", reason="pilot stopped before case")
        for seed, method in CASE_ORDER
    ]
    initial_parameters: dict[str, Any] = {}
    sources: dict[int, CoupledPredictiveCodingNetwork] = {}
    errors = []
    sampler: ProcessRssSampler | None = None
    try:
        sampler = ProcessRssSampler()
        with sampler:
            for index, (seed, method) in enumerate(CASE_ORDER):
                if seed not in sources:
                    sources[seed] = CoupledPredictiveCodingNetwork(
                        2, (8, 8), seed, precisions=(1.0, 1.0)
                    )
                    initial_parameters[str(seed)] = _initial_record(sources[seed])
                if granularity_aware:
                    rows[index] = _run_one(
                        seed, method, sources[seed], started, sampler, granularity_aware=True
                    )
                else:
                    rows[index] = _run_one(seed, method, sources[seed], started, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        summaries = (
            summarize_inference_comparison(
                tuple(rows),
                window_seconds=GRANULARITY_WINDOW_SECONDS,
                max_iteration_seconds=MAX_ITERATION_SECONDS,
            )
            if granularity_aware
            else summarize_inference_comparison(tuple(rows))
        )
    except ValueError as error:
        summaries = []
        errors.append(f"aggregation: {error}")
    memory = None
    if sampler is not None:
        try:
            memory = asdict(sampler.snapshot())
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    elapsed = perf_counter() - started
    within_resources = (
        elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return {
        "schema_version": 1,
        "protocol": describe_protocol(granularity_aware=granularity_aware),
        "status": "complete"
        if within_resources and not errors and all(row["status"] == "complete" for row in rows)
        else "incomplete",
        "initial_parameters": initial_parameters,
        "runs": rows,
        "summaries": summaries,
        "elapsed_seconds": elapsed,
        "observed_process_rss": memory,
        "within_resources": within_resources,
        "global_errors": errors,
        "work_scope": "Known completed calls only; failed internal work unknown. Counts are not FLOPs; threshold arithmetic included in window timing but not individually counted.",
    }
