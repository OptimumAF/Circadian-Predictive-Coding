"""Small native capacity-accounting example on the existing stream runner.

This fixed six-arm engineering case measures a scheduled split against an
isometrically initialized final-width reference. It is not the complete R2.6
mechanism study, a time/FLOP match, or evidence of learning/efficiency benefit.
"""

from dataclasses import asdict, fields
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.numpy_capacity import measure_numpy_capacity
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.adapters.structural_scaled_matched_backprop_learner import (
    StructuralScaledMatchedBackpropLearner,
)
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.stream_capacity import StreamCapacity
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
from src.core.coupled_scaling_structure import split_coupled_scaling_width
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coherent_stream_capacity_boundaries_v1"
METHODS = ("pc", "neutral_cpc", "gated_cpc")
MODES = ("fixed_final_width", "scheduled_split")
RESOURCES = PilotResources(
    training_updates=24, environment_steps=0, replay_bytes=0, storage_bytes=2 * 1024**2
)
BP_PROTOCOL_ID = "coherent_stream_structural_bp_capacity_v1"
BP_RESOURCES = PilotResources(
    training_updates=8, environment_steps=0, replay_bytes=0, storage_bytes=2 * 1024**2
)
TRAINING_FEATURES = (
    ((0.8, -0.2), (-0.7, 0.4)),
    ((0.6, 0.9), (-0.5, -0.8)),
    ((-0.4, 0.7), (0.9, -0.3)),
    ((-0.8, 0.1), (0.3, -0.6)),
)
HELDOUT_FEATURES = ((0.9, 0.2), (-0.8, -0.3), (0.4, -0.7), (-0.2, 0.6))


def describe_protocol() -> dict[str, Any]:
    """Return declared constants without constructing models or observing scores."""
    return dict(
        protocol_id=PROTOCOL_ID,
        classification="native_capacity_accounting_engineering_example",
        methods=list(METHODS),
        modes=list(MODES),
        seed=23,
        scheme="mean_field",
        gamma0=1.5,
        input_dim=2,
        hidden_layers=1,
        initial_width=2,
        final_width=4,
        learning_rate=0.01,
        inference_steps=2,
        inference_learning_rate=0.05,
        initial_precisions=[1.0],
        split_at_update=2,
        label_delay=2,
        training_features=TRAINING_FEATURES,
        training_contexts=["A", "A", "B", "A"],
        training_targets=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        heldout_features=HELDOUT_FEATURES,
        target_A="x0>=0",
        target_B="x1>=0",
        checkpoint_updates=[0, 2, 4],
        fixed_reference="isometric split of the same untrained width2 source; duplicate units, mapped precisions and width-dependent factors preserved",
        limits=dict(max_parameters=17, max_state_array_bytes=200, max_width=4),
        planned_work=dict(
            native_updates=24,
            training_rows=48,
            final_score_forwards=36,
            scheduled_splits=3,
            fixed_initial_splits=3,
            replay=0,
            probes=0,
        ),
        expected_wake_mean_parameters=dict(fixed_final_width=17.0, scheduled_split=13.0),
        metrics=dict(bce="clip1e-12 lower", accuracy="inclusive0.5 higher"),
        initial_score_absolute_tolerance=1e-12,
        resources=asdict(RESOURCES),
        stop_policy="one invocation; retain all failures/unreached arms; no retry/tuning",
        accounting="committed native model arrays and wake-call-weighted means; completed native calls/rows, not FLOPs/temporary allocation peak; measurement wall time reported",
        limitations="original R2.6 comparison remains open: no prune/renewal/BP control cohort here; no equal compute/effective capacity, efficiency, useful adaptation or CPC advantage claim",
    )


def build_stream() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    contexts = ("A", "A", "B", "A")
    targets = ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    training = tuple(
        AttributionBatch(
            "train",
            context,
            (f"capacity-train-{i}-0", f"capacity-train-{i}-1"),
            np.array(values, dtype=np.float64),
            np.array(labels, dtype=np.float64).reshape(-1, 1),
        )
        for i, (context, values, labels) in enumerate(
            zip(contexts, TRAINING_FEATURES, targets, strict=True)
        )
    )
    heldout = tuple(
        AttributionBatch(
            "final_test",
            context,
            tuple(f"capacity-final-{context}-{i}" for i in range(4)),
            np.array(HELDOUT_FEATURES, dtype=np.float64),
            (np.array(HELDOUT_FEATURES, dtype=np.float64)[:, axis : axis + 1] >= 0).astype(
                np.float64
            ),
        )
        for axis, context in enumerate(("A", "B"))
    )
    return training, heldout


def build_learner(
    method: str, mode: str
) -> StructuralCoupledScalingLearner | StructuralScaledMatchedBackpropLearner:
    if method not in (*METHODS, "bp") or mode not in MODES:
        raise ValueError("unknown fixed capacity case")
    config = (
        None
        if method in ("pc", "bp")
        else (
            CoupledCircadianConfig.matched_pc_control()
            if method == "neutral_cpc"
            else CoupledCircadianConfig()
        )
    )
    initial = initialize_coupled_scaling(
        CoupledScalingConfig(
            "mean_field",
            2,
            2,
            1,
            gamma0=1.5,
            precisions=(1.0,),
            circadian=config,
        ),
        23,
    )
    if mode == "fixed_final_width":
        initial = split_coupled_scaling_width(initial).state
    if method == "bp":
        return StructuralScaledMatchedBackpropLearner(initial, learning_rate=0.01)
    return StructuralCoupledScalingLearner(
        initial, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )


def _checkpoint(updates: int, state: CoupledStructuralState) -> dict[str, Any]:
    raw = state.raw_state
    config = raw.config
    p = raw.raw_parameters
    parameters = {
        f.name: [a.tolist() for a in value] if type(value) is tuple else value.tolist()
        for f in fields(p)
        for value in (getattr(p, f.name),)
    }
    return dict(
        updates=updates,
        width=config.width,
        precisions=list(config.precisions or ()),
        forward_scales=list(config.forward_scales),
        raw_learning_rate_factor=config.raw_learning_rate_factor,
        effective_learning_rate_factors=list(config.effective_learning_rate_factors),
        effective_bias_learning_rate_factor=config.effective_bias_learning_rate_factor,
        native_wake_updates=None if raw.wake_state is None else raw.wake_state.wake_updates,
        capacity=asdict(measure_numpy_capacity(state)),
        raw_parameters=parameters,
        transition_kinds=[t.kind for t in state.transitions],
    )


def _run_one(row: dict[str, Any], deadline: float, sampler: ProcessRssSampler) -> None:
    started = monotonic()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=4,
            max_wall_seconds=max(0.0, deadline - started),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    helper = None
    try:
        budget.attach_memory(sampler)
        budget.before_final()
        learner = build_learner(row["method"], row["mode"])
        before = pickle.dumps(learner.snapshot_state())
        helper = StreamCapacity(
            learner,
            budget,
            total_updates=4,
            measure_state=measure_numpy_capacity,
            max_parameters=17,
            max_state_array_bytes=200,
            split_at_update=2 if row["mode"] == "scheduled_split" else None,
            max_width=4 if row["mode"] == "scheduled_split" else None,
        )
        training, heldout = build_stream()

        def observe(updates: int, state: Any, arrived: Any) -> None:
            before_observe = monotonic()
            row["native_checkpoints"].append(_checkpoint(updates, state))
            row["checkpoint_reporting_seconds"] += monotonic() - before_observe

        result = run_attribution_stream(
            learner,
            training,
            heldout,
            checkpoint_updates=(0, 2, 4),
            label_delay=2,
            budget=budget,
            capacity=helper,
            observe_checkpoint=observe,
        )
        if pickle.dumps(learner.snapshot_state()) != before:
            raise RuntimeError("capacity stream changed its source learner")
        row.update(
            status="complete", reason=None, stream_result=result, source_learner_unchanged=True
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    row.update(
        wake_updates=budget.updates_completed,
        elapsed_seconds=monotonic() - started,
        capacity=None if helper is None else helper.report,
    )


def _facts(rows: list[dict[str, Any]]) -> dict[str, bool]:
    if any(row["status"] != "complete" for row in rows):
        return dict(
            pc_neutral_exact=False,
            initial_scores_match=False,
            final_widths_exact=False,
            capacity_accounting_exact=False,
        )
    selected = {(r["method"], r["mode"]): r for r in rows}
    parity = all(
        selected["pc", mode]["stream_result"]["checkpoints"]
        == selected["neutral_cpc", mode]["stream_result"]["checkpoints"]
        and [c["raw_parameters"] for c in selected["pc", mode]["native_checkpoints"]]
        == [c["raw_parameters"] for c in selected["neutral_cpc", mode]["native_checkpoints"]]
        for mode in MODES
    )
    initial_match = True
    for method in METHODS:
        a = selected[method, "fixed_final_width"]["stream_result"]["checkpoints"][0]["scores"]
        b = selected[method, "scheduled_split"]["stream_result"]["checkpoints"][0]["scores"]
        initial_match &= all(
            abs(a[c]["bce"] - b[c]["bce"]) <= 1e-12 and a[c]["accuracy"] == b[c]["accuracy"]
            for c in ("A", "B")
        )
    accounting = True
    for row in rows:
        growing = row["mode"] == "scheduled_split"
        cpc = row["method"] != "pc"
        expected_mean_parameters = 13.0 if growing else 17.0
        expected_mean_aux = (48.0 if growing else 64.0) if cpc else 0.0
        cap = row["capacity"]
        accounting &= (
            cap["completed_native_calls"] == 4
            and cap["completed_training_rows"] == 8
            and cap["peak_parameters"] == 17
            and cap["peak_parameter_bytes"] == 136
            and cap["peak_auxiliary_bytes"] == (64 if cpc else 0)
            and cap["peak_total_array_bytes"] == (200 if cpc else 136)
            and cap["mean_parameters"] == expected_mean_parameters
            and cap["mean_parameter_bytes"] == 8 * expected_mean_parameters
            and cap["mean_auxiliary_bytes"] == expected_mean_aux
            and cap["mean_total_array_bytes"] == 8 * expected_mean_parameters + expected_mean_aux
            and cap["split_committed"] == int(growing)
        )
    return dict(
        pc_neutral_exact=parity,
        initial_scores_match=initial_match,
        capacity_accounting_exact=accounting,
        final_widths_exact=all(r["native_checkpoints"][-1]["width"] == 4 for r in rows),
    )


def run_pilot() -> dict[str, Any]:
    validate_local_pilot_request(PilotRequest(RESOURCES))
    started = monotonic()
    rows: list[dict[str, Any]] = [
        dict(
            method=m,
            mode=mode,
            status="not_run",
            reason="pilot stopped before execution",
            wake_updates=0,
            capacity=None,
            native_checkpoints=[],
            checkpoint_reporting_seconds=0.0,
        )
        for m in METHODS
        for mode in MODES
    ]
    sampler = None
    errors = []
    try:
        sampler = ProcessRssSampler()
        with sampler:
            for row in rows:
                if monotonic() - started >= RESOURCES.wall_seconds:
                    break
                _run_one(row, started + RESOURCES.wall_seconds, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    elapsed = monotonic() - started
    facts = _facts(rows)
    wake = sum(r["wake_updates"] for r in rows)
    complete = (
        not errors
        and all(r["status"] == "complete" for r in rows)
        and wake == 24
        and all(facts.values())
        and elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        wake_updates=wake,
        completed_training_rows=2 * wake,
        **facts,
        full_R26_complete=False,
        scientific_advantage_verified=False,
    )


def describe_bp_protocol() -> dict[str, Any]:
    """Describe only the two new BP arms, preserving the original six-arm recipe."""
    protocol = describe_protocol()
    for key in ("inference_steps", "inference_learning_rate"):
        del protocol[key]
    protocol.update(
        protocol_id=BP_PROTOCOL_ID,
        methods=["bp"],
        classification="structural_matched_bp_baseline_engineering_example",
        native_objective="original raw-coordinate matched feedforward BCE; no latent inference or chemical state",
        planned_work=dict(
            native_updates=8,
            training_rows=16,
            final_score_forwards=12,
            scheduled_splits=1,
            fixed_initial_splits=1,
            replay=0,
            probes=0,
        ),
        resources=asdict(BP_RESOURCES),
        limitations="original R2.6 complete mechanism study remains open; only new BP arms run; retained PC/CPC comparison is post-run; no equal-compute/effective-capacity/efficiency or learning-benefit claim",
    )
    return protocol


def _bp_facts(rows: list[dict[str, Any]]) -> dict[str, bool]:
    if len(rows) != 2 or any(r["status"] != "complete" for r in rows):
        return dict(
            initial_scores_match=False, final_widths_exact=False, capacity_accounting_exact=False
        )
    fixed, growing = rows
    a = fixed["stream_result"]["checkpoints"][0]["scores"]
    b = growing["stream_result"]["checkpoints"][0]["scores"]
    initial = all(
        abs(a[c]["bce"] - b[c]["bce"]) <= 1e-12 and a[c]["accuracy"] == b[c]["accuracy"]
        for c in ("A", "B")
    )
    accounting = all(
        r["method"] == "bp"
        and r["capacity"]["completed_native_calls"] == 4
        and r["capacity"]["completed_training_rows"] == 8
        and r["capacity"]["mean_parameters"] == (13 if r["mode"] == "scheduled_split" else 17)
        and r["capacity"]["mean_parameter_bytes"]
        == (104 if r["mode"] == "scheduled_split" else 136)
        and r["capacity"]["mean_auxiliary_bytes"] == r["capacity"]["peak_auxiliary_bytes"] == 0
        and r["capacity"]["peak_parameters"] == 17
        and r["capacity"]["peak_total_array_bytes"] == 136
        and r["capacity"]["split_committed"] == int(r["mode"] == "scheduled_split")
        for r in rows
    )
    return dict(
        initial_scores_match=initial,
        final_widths_exact=all(r["native_checkpoints"][-1]["width"] == 4 for r in rows),
        capacity_accounting_exact=accounting,
    )


def run_bp_pilot() -> dict[str, Any]:
    """Execute new BP controls through the existing per-arm stream binding."""
    validate_local_pilot_request(PilotRequest(BP_RESOURCES))
    started = monotonic()
    rows: list[dict[str, Any]] = [
        dict(
            method="bp",
            mode=mode,
            status="not_run",
            reason="pilot stopped before execution",
            wake_updates=0,
            capacity=None,
            native_checkpoints=[],
            checkpoint_reporting_seconds=0.0,
        )
        for mode in MODES
    ]
    sampler = None
    errors = []
    try:
        sampler = ProcessRssSampler()
        with sampler:
            for row in rows:
                if monotonic() - started >= BP_RESOURCES.wall_seconds:
                    break
                _run_one(row, started + BP_RESOURCES.wall_seconds, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    elapsed = monotonic() - started
    facts = _bp_facts(rows)
    wake = sum(r["wake_updates"] for r in rows)
    complete = (
        not errors
        and wake == 8
        and all(facts.values())
        and elapsed <= BP_RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= BP_RESOURCES.process_memory_bytes
    )
    return dict(
        protocol=describe_bp_protocol(),
        status="complete" if complete else "incomplete",
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        wake_updates=wake,
        completed_training_rows=2 * wake,
        **facts,
        full_R26_complete=False,
        scientific_advantage_verified=False,
    )
