"""Fixed synthetic pilot composition and budgeted execution; no tuning inputs.

The application owns train/heldout isolation and metrics. This adapter owns fixed
data, native model construction, resource observations and JSON-compatible output.
"""

from dataclasses import asdict
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.coupled_attribution_summary import (
    CHECKPOINTS,
    CONTEXTS,
    METHODS,
    PLANNED_SEEDS,
    STREAMS,
    summarize_attribution,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ForkableLearner
from src.core.coupled_energy import Array, CoupledParameters
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.core.matched_backprop import MatchedBackpropNetwork
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coupled_stationary_recurring_pilot_v1"
RESOURCES = PilotResources(training_updates=432, environment_steps=0, replay_bytes=0)


def describe_protocol() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "classification": "exploratory_fixed_data_pilot",
        "seeds": list(PLANNED_SEEDS),
        "streams": list(STREAMS),
        "methods": list(METHODS),
        "hidden_dims": [8, 8],
        "learning_rate": 0.03,
        "precisions": [1.0, 1.0],
        "inference_steps": 4,
        "inference_learning_rate": 0.2,
        "active_cpc_config": asdict(CoupledCircadianConfig()),
        "neutral_cpc_config": asdict(CoupledCircadianConfig.matched_pc_control()),
        "training_rng": 7301,
        "heldout_rng": 9101,
        "distribution": "uniform[-1,1]",
        "target_rules": {"A": "x0 >= 0", "B": "x1 >= 0"},
        "stream_contexts": {"stationary": ["A", "A", "A"], "recurring": ["A", "B", "A"]},
        "updates_per_phase": 6,
        "batch_size": 8,
        "heldout_rows_per_context": 64,
        "label_delay": 2,
        "checkpoint_updates": list(CHECKPOINTS),
        "primary_metric": "probability_bce_clip_1e-12_lower_better",
        "secondary_metric": "accuracy_threshold_ge_0.5_higher_better",
        "intervals": "nominal marginal 95% t, df2 critical4.303; fixeddata n3; no multiplicity claim",
        "score_release": "after all training in each learner/stream run; no feedback to other runs",
        "resources": asdict(RESOURCES),
    }


def build_stream(stream: str) -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    if stream not in STREAMS:
        raise ValueError("stream must be stationary or recurring")
    phase_contexts = ("A", "A", "A") if stream == "stationary" else ("A", "B", "A")
    train_rng, heldout_rng = np.random.default_rng(7301), np.random.default_rng(9101)
    training = []
    for index in range(18):
        context = phase_contexts[index // 6]
        features = train_rng.uniform(-1, 1, (8, 2))
        targets = (features[:, [CONTEXTS.index(context)]] >= 0).astype(np.float64)
        training.append(
            AttributionBatch(
                "train",
                context,
                tuple(f"train-{stream}-{index}-{i}" for i in range(8)),
                features,
                targets,
            )
        )
    heldout = []
    for context in CONTEXTS:
        features = heldout_rng.uniform(-1, 1, (64, 2))
        targets = (features[:, [CONTEXTS.index(context)]] >= 0).astype(np.float64)
        heldout.append(
            AttributionBatch(
                "final_test",
                context,
                tuple(f"heldout-{context}-{i}" for i in range(64)),
                features,
                targets,
            )
        )
    return tuple(training), tuple(heldout)


def build_learner(
    method: str, initial: CoupledParameters
) -> ForkableLearner[Array, Array, Array, Any]:
    if method not in METHODS:
        raise ValueError("method must be one of the four fixed pilot methods")
    if method == "bp":
        return MatchedBackpropLearner(MatchedBackpropNetwork(initial), learning_rate=0.03)
    model: CoupledPredictiveCodingNetwork
    if method == "pc":
        model = CoupledPredictiveCodingNetwork(2, (8, 8), 0)
    else:
        config = (
            CoupledCircadianConfig.matched_pc_control()
            if method == "neutral_cpc"
            else CoupledCircadianConfig()
        )
        model = CoupledCircadianPredictiveCodingNetwork(2, (8, 8), 0, config=config)
    model.restore_parameters(initial)
    return CoupledLearner(model, learning_rate=0.03, inference_steps=4, inference_learning_rate=0.2)


def _run_one(
    seed: int,
    stream: str,
    method: str,
    initial: CoupledParameters,
    started: float,
    sampler: ProcessRssSampler,
) -> dict[str, Any]:
    row: dict[str, Any] = {"seed": seed, "stream": stream, "method": method}
    remaining = RESOURCES.wall_seconds - (monotonic() - started)
    if remaining <= 0:
        return {
            **row,
            "status": "not_run",
            "reason": "global wall budget exhausted",
            "committed_updates": 0,
        }
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=18,
            max_wall_seconds=remaining,
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    try:
        budget.attach_memory(sampler)
        budget.before_final()
        training, heldout = build_stream(stream)
        learner = build_learner(method, initial)
        result = run_attribution_stream(
            learner, training, heldout, checkpoint_updates=CHECKPOINTS, label_delay=2, budget=budget
        )
        row.update(status="complete", stream_result=result)
    except Exception as error:
        # Keep every planned row and its context. No partial trajectory is scored
        # or substituted for a complete seed in statistical aggregation.
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    updates = budget.updates_completed
    row["committed_updates"] = updates
    row["committed_training_examples"] = 8 * updates
    row["nominal_completed_latent_steps"] = (0 if method == "bp" else 4) * updates
    row["nominal_completed_gradient_evaluations"] = (2 if method == "bp" else 6) * updates
    return row


def run_pilot() -> dict[str, Any]:
    """Execute the one fixed small pilot, retaining complete failure vectors."""
    validate_local_pilot_request(PilotRequest(RESOURCES))
    started = monotonic()
    keys = [
        (seed, stream, method) for seed in PLANNED_SEEDS for stream in STREAMS for method in METHODS
    ]
    rows = [
        {
            "seed": seed,
            "stream": stream,
            "method": method,
            "status": "not_run",
            "reason": "pilot stopped before execution",
            "committed_updates": 0,
        }
        for seed, stream, method in keys
    ]
    initial_parameters = {}
    global_errors = []
    sampler = ProcessRssSampler()
    try:
        with sampler:
            for index, (seed, stream, method) in enumerate(keys):
                if str(seed) not in initial_parameters:
                    initial = CoupledPredictiveCodingNetwork(2, (8, 8), seed).snapshot_parameters()
                    initial_parameters[str(seed)] = {
                        "hidden_weights": [a.tolist() for a in initial.hidden_weights],
                        "hidden_biases": [a.tolist() for a in initial.hidden_biases],
                        "output_weight": initial.output_weight.tolist(),
                        "output_bias": initial.output_bias.tolist(),
                    }
                rows[index] = _run_one(seed, stream, method, initial, started, sampler)
    except Exception as error:
        global_errors.append(f"{type(error).__name__}: {error}")
        # Records were preallocated in planned order, including unreached runs.
    try:
        summaries = summarize_attribution(tuple(rows))
    except ValueError as error:
        summaries = []
        global_errors.append(f"aggregation: {error}")
    elapsed = monotonic() - started
    try:
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        global_errors.append(f"{type(error).__name__}: {error}")
    within_resources = (
        elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return {
        "schema_version": 1,
        "protocol": describe_protocol(),
        "status": "complete"
        if within_resources and not global_errors and all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "initial_parameters": initial_parameters,
        "runs": rows,
        "summaries": summaries,
        "elapsed_seconds": elapsed,
        "observed_process_rss": memory,
        "within_resources": within_resources,
        "global_errors": global_errors,
        "work_scope": "Nominal completed kernel counts from native code; exclude failed attempts, copies, overhead; not FLOPs or matched compute.",
    }
