"""Compose the fixed local linear scaling grid, data and resource observations.

No tuning arguments or score feedback enter case selection. The application
owns training/scoring isolation; this adapter preserves every planned row.
"""

from dataclasses import asdict
from time import monotonic
from typing import Any, Callable, TypeVar

import numpy as np

from src.app.linear_scaling_measurement import ScalingBatch, run_linear_scaling_case
from src.app.linear_scaling_summary import (
    DEPTHS,
    KEYS,
    SCHEMES,
    SEEDS,
    UPDATES,
    WIDTHS,
    summarize_scaling,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    initialize_linear_scaling,
)
from src.core.linear_pc_reference import Array
from src.core.residual_linear_pc import LinearResidualState
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSampler

RESOURCES = PilotResources(training_updates=486, environment_steps=0, replay_bytes=0)
State = TypeVar("State", LinearScalingState, LinearResidualState)


def describe_protocol() -> dict[str, Any]:
    return {
        "protocol_id": "linear_mlp_scaling_measurements_v1",
        "classification": "exploratory_fixed_regression_data",
        "seeds": list(SEEDS),
        "widths": list(WIDTHS),
        "hidden_layers": list(DEPTHS),
        "schemes": list(SCHEMES),
        "order": ["seed", "width", "hidden_layers", "scheme"],
        "updates": UPDATES,
        "learning_rate": 0.01,
        "gamma0": 1.0,
        "input_dim": 2,
        "training_rng": 7311,
        "heldout_rng": 9111,
        "training_rows": 16,
        "heldout_rows": 64,
        "distribution": "uniform[-1,1]",
        "teacher": "y=x0-0.5*x1",
        "metric": "heldout_half_mse_lower_better",
        "checkpoints": list(range(7)),
        "score_release": "after all training per case; no feedback to later cases",
        "intervals": "nominal marginal 95% t df2 critical4.303, fixed data n3; no family coverage",
        "resources": asdict(RESOURCES),
    }


def build_batches() -> tuple[ScalingBatch, ScalingBatch]:
    batches = []
    for role, prefix, seed, count in (
        ("train", "train", 7311, 16),
        ("final_test", "final", 9111, 64),
    ):
        features = np.random.default_rng(seed).uniform(-1, 1, (count, 2))
        targets = features[:, [0]] - 0.5 * features[:, [1]]
        batches.append(
            ScalingBatch(
                role, tuple(f"linear-{prefix}-{i}" for i in range(count)), features, targets
            )
        )
    return batches[0], batches[1]


def _run_one(row: dict[str, Any], started: float, sampler: ProcessRssSampler) -> None:
    _execute_case(
        row,
        started,
        sampler,
        lambda scheme, width, hidden, seed: initialize_linear_scaling(
            LinearScalingConfig(scheme, 2, width, hidden), seed
        ),
        effective_linear_weights,
        run_linear_scaling_case,
    )


def _execute_case(
    row: dict[str, Any],
    started: float,
    sampler: ProcessRssSampler,
    initialize: Callable[[str, int, int, int], State],
    effective_weights: Callable[[State], tuple[Array, ...]],
    run_case: Callable[..., dict[str, Any]],
) -> None:
    """Shared fixed-grid IO/resource boundary; model composition stays outside."""
    case_started = monotonic()
    remaining = RESOURCES.wall_seconds - (monotonic() - started)
    if remaining <= 0:
        row["reason"] = "global wall budget exhausted"
        return
    try:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=UPDATES,
                max_wall_seconds=remaining,
                max_process_rss_bytes=RESOURCES.process_memory_bytes,
            ),
            monotonic,
        )
        budget.attach_memory(sampler)
        budget.before_final()
        initialization_start = monotonic()
        initial = initialize(row["scheme"], row["width"], row["hidden_layers"], row["seed"])
        config = initial.config
        row["initialization_seconds"] = monotonic() - initialization_start
        row["initial_state"] = {
            "config": asdict(config),
            "raw_weights": [w.tolist() for w in initial.raw_weights],
            "effective_weights": [w.tolist() for w in effective_weights(initial)],
            "forward_scales": list(config.forward_scales),
            "initial_stddevs": list(config.initial_stddevs),
            "raw_learning_rate_factor": config.raw_learning_rate_factor,
            "effective_learning_rate_factors": list(config.effective_learning_rate_factors),
        }
        training, heldout = build_batches()
        row.update(
            run_case(initial, training, heldout, updates=UPDATES, learning_rate=0.01, budget=budget)
        )
    except Exception as error:
        row.update(
            status="failed", reason=f"{type(error).__name__}: {error}", heldout_half_mse=None
        )
    finally:
        row["case_elapsed_seconds"] = monotonic() - case_started
        committed = row["committed_updates"]
        row["work"] = {
            "attempted_step_calls": row["attempted_updates"],
            "committed_step_calls": committed,
            "completed_step_gradient_evaluations": 2 * committed,
            "completed_diagnostic_gradient_evaluations": len(row["checkpoints"]),
            "heldout_forwards": len(row["heldout_half_mse"])
            if row["status"] == "complete"
            else None,
            "committed_training_row_exposures": 16 * committed,
            "note": "Completed work only; failed-call internal work and discarded scoring work are unknown.",
        }


def run_pilot() -> dict[str, Any]:
    return _execute_grid(_run_one, describe_protocol)


def _execute_grid(
    run_one: Callable[[dict[str, Any], float, ProcessRssSampler], None],
    protocol: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Preserve all fixed keys and common work/resource/summary semantics."""
    validate_local_pilot_request(PilotRequest(RESOURCES))
    rows: list[dict[str, Any]] = [
        dict(
            seed=seed,
            width=width,
            hidden_layers=depth,
            scheme=scheme,
            status="not_run",
            reason="global execution did not reach this case",
            attempted_updates=0,
            committed_updates=0,
            checkpoints=[],
            heldout_half_mse=None,
        )
        for seed, width, depth, scheme in KEYS
    ]
    errors: list[str] = []
    started = monotonic()
    sampler = ProcessRssSampler()
    try:
        with sampler:
            for row in rows:
                run_one(row, started, sampler)
    except Exception as error:
        errors.append(f"execution: {type(error).__name__}: {error}")
    try:
        summaries = summarize_scaling(tuple(rows))
    except Exception as error:
        summaries = []
        errors.append(f"aggregation: {type(error).__name__}: {error}")
    try:
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"memory: {type(error).__name__}: {error}")
    elapsed = monotonic() - started
    within = (
        elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return {
        "schema_version": 1,
        "protocol": protocol(),
        "runs": rows,
        "summaries": summaries,
        "status": "complete"
        if within and not errors and all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "elapsed_seconds": elapsed,
        "work": {
            "attempted_step_calls": sum(r["attempted_updates"] for r in rows),
            "committed_step_calls": sum(r["committed_updates"] for r in rows),
            "completed_step_gradient_evaluations": 2 * sum(r["committed_updates"] for r in rows),
            "completed_diagnostic_gradient_evaluations": sum(len(r["checkpoints"]) for r in rows),
            "complete_case_heldout_forwards": sum(
                len(r["heldout_half_mse"]) for r in rows if r["status"] == "complete"
            ),
            "committed_training_row_exposures": 16 * sum(r["committed_updates"] for r in rows),
            "note": "Nominal completed work; failed-call internals and discarded scoring are unknown, not zero.",
        },
        "observed_process_rss": memory,
        "within_resources": within,
        "global_errors": errors,
        "resource_note": "After imports, before JSON encoding; safe-boundary wall/RSS checks, not native preemption or exhaustive peak measurement.",
    }
