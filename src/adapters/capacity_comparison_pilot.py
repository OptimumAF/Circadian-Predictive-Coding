"""Fixed R2.6 mechanism cohorts composed through the existing attribution stream.

This recipe reports all frozen outcomes and native payload/work costs. It does
not select schedules, fit controls, change objectives, promote models or pool
ordinary renewal with raw-coordinate structural methods as equal computation.
"""

from dataclasses import asdict, fields
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters import capacity_pilot
from src.adapters.coupled_attribution_pilot import build_stream
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.adapters.numpy_capacity import measure_numpy_capacity
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.adapters.structural_scaled_matched_backprop_learner import (
    StructuralScaledMatchedBackpropLearner,
)
from src.app.coupled_attribution import run_attribution_stream
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.stream_capacity import StreamCapacity
from src.app.stream_capacity_prune import StreamCapacityPrune
from src.app.stream_unit_renewal import StreamUnitRenewal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_pc_scaling import effective_coupled_parameters
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_scaling_pruning import prune_coupled_scaling_width
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.coupled_unit_renewal import RenewalPolicy
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.core.matched_backprop import MatchedBackpropNetwork
from src.core.renewable_coupled_predictive_coding import (
    RenewableCoupledCircadianPredictiveCodingNetwork,
    RenewableCoupledPredictiveCodingNetwork,
)
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coherent_capacity_mechanism_comparison_v1"
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
RAW_MODES = ("fixed_pruned_width", "fixed_wide", "scheduled_growth", "split_prune")
RESOURCES = PilotResources(
    training_updates=414, environment_steps=0, replay_bytes=0, storage_bytes=2 * 1024**2
)
CHECKPOINTS = (0, 6, 12, 18)


def describe_protocol() -> dict[str, Any]:
    """Describe the complete frozen recipe without constructing a native learner."""
    return dict(
        protocol_id=PROTOCOL_ID,
        classification="exploratory_single_seed_capacity_mechanism_comparison",
        methods=list(METHODS),
        raw_modes=list(RAW_MODES),
        ordinary_modes=dict(ordinary_fixed=list(METHODS), bounded_renewal=list(METHODS[1:])),
        seed=23,
        input_dim=2,
        hidden_layers=1,
        initial_width=2,
        split_width=4,
        pruned_width=3,
        scheme="mean_field",
        gamma0=1.5,
        initial_precisions=[1.0],
        learning_rate=0.01,
        inference_steps=2,
        inference_learning_rate=0.05,
        stream="A6/B6/A6",
        training_rng=7301,
        final_rng=9101,
        training_batches=18,
        training_batch_rows=8,
        final_rows_per_context=64,
        label_delay=2,
        checkpoint_updates=list(CHECKPOINTS),
        split_at_update=6,
        prune_at_update=12,
        removed_units=[2],
        prune_validation=dict(
            rng=307,
            rows_per_context=4,
            available_at_tick=0,
            roles=["validation", "validation"],
            target_A="x0>=0",
            target_B="x1>=0",
            limits=dict(
                max_prediction_change=0.05, max_bce_increase=0.05, max_inferred_energy_increase=0.05
            ),
            inference_steps=2,
            inference_learning_rate=0.05,
        ),
        renewal=dict(
            min_age_updates=6,
            max_importance=0.001,
            max_per_event=1,
            seed=29,
            history_decay=0.95,
            every_updates=6,
            max_replacements_per_arm=3,
        ),
        initialization="same untrained width2 seed23 source; wide is its isometric split; width3 reference is data-blind pure prune(2,) of that untrained split, hence its initial function differs; ordinary controls/renewal own detached wide effective parameters/precision",
        native_policy="raw factors preserved and reported; ordinary rate .01 uniform. Same native algorithm within each cohort, no raw/ordinary pooling or equal-compute/effective-capacity claim. BP prune gate uses shared PC energy diagnostic, not a changed BP training objective",
        comparisons="all raw modes minus both fixed references within same method; renewal minus same ordinary method; PC-BP, neutral-PC and active-neutral within supported same mode; absolute levels and initial scores retained",
        metrics=dict(
            bce="clip1e-12 lower",
            accuracy="inclusive0.5 higher",
            new_B_gain="B0-B12 higher",
            retention_loss="A12-A6 lower",
            return_gain="A12-A18 higher",
        ),
        limits=dict(max_parameters=17, max_state_array_bytes=296, max_width=4),
        planned_work=dict(
            arms=23,
            native_updates=414,
            training_rows=3312,
            final_score_forwards=184,
            initial_splits=15,
            scheduled_splits=8,
            initial_prunes=4,
            guarded_prunes=4,
            gate_score_forwards=16,
            gate_row_forwards=64,
            max_renewal_events=9,
            max_replacement_units=9,
            raw_PC_updates=216,
            ordinary_PC_updates=108,
            raw_BP_updates=72,
            ordinary_BP_updates=18,
            nominal_training_energy_evaluations=1296,
            nominal_BP_evaluations=180,
            replay=0,
            fresh_task_probes=0,
        ),
        isolation="validation available at birth only to frozen prune gate; final data only scored after each complete training stream; no score/probe feedback, promotion, tuning or external result reads",
        resources=asdict(RESOURCES),
        stop_policy="one invocation; preserve failed/unreached arms and rejected gates; rejected exact no-publication prune continues unchanged; failed/unknown gate stops; no retry/tuning",
        limitations="one fixed seed/small synthetic tasks; no intervals/equivalence/confirmation/CPC benefit. Native arrays exclude copies/temporaries/metadata/process memory; wall timing is not causal efficiency",
    )


def build_validation() -> tuple[ScalingBatch, ...]:
    """Reserved validation labels are available at birth, disjoint from final data."""
    rng = np.random.default_rng(307)
    return tuple(
        ScalingBatch(
            "validation",
            tuple(f"capacity-validation-{context}-{i}" for i in range(4)),
            features,
            (features[:, axis : axis + 1] >= 0).astype(np.float64),
        )
        for axis, context in enumerate(("A", "B"))
        for features in (rng.uniform(-1, 1, (4, 2)),)
    )


def _config(method: str) -> CoupledCircadianConfig | None:
    if method in ("bp", "pc"):
        return None
    return (
        CoupledCircadianConfig.matched_pc_control()
        if method == "neutral_cpc"
        else CoupledCircadianConfig()
    )


def build_learner(method: str, mode: str) -> Any:
    """Compose existing native ports, with no fitting or data-dependent choices."""
    if method not in METHODS or mode not in (*RAW_MODES, "ordinary_fixed", "bounded_renewal"):
        raise ValueError("unknown fixed capacity comparison cohort")
    if mode == "bounded_renewal" and method == "bp":
        raise ValueError("the existing BP port has no unit-renewal implementation")
    if mode in RAW_MODES and mode != "fixed_pruned_width":
        reference = "fixed_final_width" if mode == "fixed_wide" else "scheduled_split"
        return capacity_pilot.build_learner(method, reference)
    wide = capacity_pilot.build_learner(method, "fixed_final_width").snapshot_state().raw_state
    if mode == "fixed_pruned_width":
        initial = prune_coupled_scaling_width(wide, (2,)).state
        if method == "bp":
            return StructuralScaledMatchedBackpropLearner(initial, learning_rate=0.01)
        return StructuralCoupledScalingLearner(
            initial, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        )
    parameters = effective_coupled_parameters(wide)
    if method == "bp":
        return MatchedBackpropLearner(MatchedBackpropNetwork(parameters), learning_rate=0.01)
    options: dict[str, Any] = dict(precisions=wide.config.precisions)
    kind: Any
    if mode == "bounded_renewal":
        options["renewal_policy"] = RenewalPolicy(6, 0.001, 1, 29)
    if method != "pc":
        options["config"] = _config(method)
        kind = (
            RenewableCoupledCircadianPredictiveCodingNetwork
            if mode == "bounded_renewal"
            else CoupledCircadianPredictiveCodingNetwork
        )
    else:
        kind = (
            RenewableCoupledPredictiveCodingNetwork
            if mode == "bounded_renewal"
            else CoupledPredictiveCodingNetwork
        )
    model = kind(2, (4,), seed=23, **options)
    model.restore_parameters(parameters)
    return CoupledLearner(
        model, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )


def _checkpoint(updates: int, state: Any) -> dict[str, Any]:
    if type(state) is CoupledStructuralState:
        return dict(representation="raw", **capacity_pilot._checkpoint(updates, state))
    parameters = state.parameters
    history = getattr(state, "unit_history", None)
    wake = getattr(state, "wake_state", None)
    return dict(
        updates=updates,
        representation="ordinary",
        width=parameters.output_weight.shape[0],
        precisions=list(getattr(state, "precisions", ())),
        capacity=asdict(measure_numpy_capacity(state)),
        parameters={
            f.name: [a.tolist() for a in value] if type(value) is tuple else value.tolist()
            for f in fields(parameters)
            for value in (getattr(parameters, f.name),)
        },
        native_wake_updates=None if wake is None else wake.wake_updates,
        unit_ids=None if history is None else history.unit_ids.tolist(),
        unit_history_updates=None if history is None else history.updates,
    )


def _helpers(
    learner: Any, budget: ToyBudgetSession, mode: str
) -> tuple[StreamCapacity, StreamUnitRenewal | None]:
    pruning = (
        StreamCapacityPrune(
            at_update=12,
            removed_units=(2,),
            probes=build_validation(),
            limits=CoupledPruneLimits(0.05, 0.05, 0.05),
            budget=budget,
            inference_steps=2,
            inference_learning_rate=0.05,
        )
        if mode == "split_prune"
        else None
    )
    capacity = StreamCapacity(
        learner,
        budget,
        total_updates=18,
        measure_state=measure_numpy_capacity,
        max_parameters=17,
        max_state_array_bytes=296,
        split_at_update=6 if mode in ("scheduled_growth", "split_prune") else None,
        max_width=4,
        pruning=pruning,
    )
    renewal = (
        StreamUnitRenewal(learner, budget, every_updates=6, total_updates=18, max_replacements=3)
        if mode == "bounded_renewal"
        else None
    )
    return capacity, renewal


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    points = {point["updates"]: point["scores"] for point in result["checkpoints"]}
    return dict(
        new_B_gain=points[0]["B"]["bce"] - points[12]["B"]["bce"],
        retention_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
        return_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
    )


def _run_one(row: dict[str, Any], deadline: float, sampler: ProcessRssSampler) -> None:
    started = monotonic()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=18,
            max_wall_seconds=max(0.0, deadline - started),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    capacity: StreamCapacity | None = None
    renewal: StreamUnitRenewal | None = None
    try:
        budget.attach_memory(sampler)
        budget.before_final()
        before_initialization = monotonic()
        learner = build_learner(row["method"], row["mode"])
        row["initialization_seconds"] = monotonic() - before_initialization
        original = pickle.dumps(learner.snapshot_state())
        capacity, renewal = _helpers(learner, budget, row["mode"])
        training, heldout = build_stream("recurring")

        def observe(updates, state, arrived):
            before = monotonic()
            row["native_checkpoints"].append(_checkpoint(updates, state))
            row["checkpoint_reporting_seconds"] += monotonic() - before

        result = run_attribution_stream(
            learner,
            training,
            heldout,
            checkpoint_updates=CHECKPOINTS,
            label_delay=2,
            budget=budget,
            capacity=capacity,
            renewal=renewal,
            observe_checkpoint=observe,
        )
        if pickle.dumps(learner.snapshot_state()) != original:
            raise ValueError("capacity comparison changed the original source learner")
        row.update(
            status="complete",
            reason=None,
            source_learner_unchanged=True,
            stream_result=result,
            metrics=_metrics(result),
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        row.update(
            committed_updates=budget.updates_completed,
            committed_training_rows=8 * budget.updates_completed,
            total_seconds=monotonic() - started,
        )
        if capacity is not None:
            row["capacity"] = capacity.report
        if renewal is not None:
            row["unit_renewal"] = renewal.report


def _rows() -> list[dict[str, Any]]:
    cases = [(method, mode) for method in METHODS for mode in RAW_MODES]
    cases += [(method, "ordinary_fixed") for method in METHODS]
    cases += [(method, "bounded_renewal") for method in METHODS[1:]]
    return [
        dict(
            method=method,
            mode=mode,
            status="not_run",
            reason="not reached",
            committed_updates=0,
            committed_training_rows=0,
            source_learner_unchanged=False,
            native_checkpoints=[],
            checkpoint_reporting_seconds=0.0,
            initialization_seconds=0.0,
            initialization_work=dict(
                splits=int(
                    mode
                    in ("fixed_wide", "fixed_pruned_width", "ordinary_fixed", "bounded_renewal")
                ),
                prunes=int(mode == "fixed_pruned_width"),
            ),
        )
        for method, mode in cases
    ]


def _comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(r["method"], r["mode"]): r for r in rows}
    pairs = [
        (method, mode, method, ref)
        for method in METHODS
        for mode in RAW_MODES
        for ref in ("fixed_pruned_width", "fixed_wide")
    ]
    pairs += [(m, "bounded_renewal", m, "ordinary_fixed") for m in METHODS[1:]]
    for mode in (*RAW_MODES, "ordinary_fixed", "bounded_renewal"):
        pairs += [
            (a, mode, b, mode)
            for a, b in (("pc", "bp"), ("neutral_cpc", "pc"), ("gated_cpc", "neutral_cpc"))
            if (a, mode) in indexed and (b, mode) in indexed
        ]
    result = []
    for left, mode, right, ref in pairs:
        a, b = indexed[left, mode], indexed[right, ref]
        complete = a["status"] == b["status"] == "complete"
        result.append(
            dict(
                left=[left, mode],
                right=[right, ref],
                direction="left minus right",
                status="complete" if complete else "unavailable",
                reason=None if complete else "one or both declared runs incomplete",
                metrics={k: a["metrics"][k] - b["metrics"][k] for k in a.get("metrics", {})}
                if complete
                else None,
                checkpoints=[
                    dict(
                        updates=x["updates"],
                        context=context,
                        **{
                            metric: x["scores"][context][metric] - y["scores"][context][metric]
                            for metric in ("bce", "accuracy")
                        },
                    )
                    for x, y in zip(
                        a["stream_result"]["checkpoints"],
                        b["stream_result"]["checkpoints"],
                        strict=True,
                    )
                    for context in ("A", "B")
                ]
                if complete
                else None,
            )
        )
    return result


def _facts(rows: list[dict[str, Any]]) -> dict[str, bool]:
    complete = all(r["status"] == "complete" for r in rows)
    if not complete:
        return dict(
            completed_work_exact=False,
            initial_within_mode_match=False,
            neutral_numeric_paths_match=False,
            capacity_means_exact=False,
        )
    indexed = {(r["method"], r["mode"]): r for r in rows}
    work = all(
        r["committed_updates"] == 18
        and r["committed_training_rows"] == 144
        and [p["updates"] for p in r["native_checkpoints"]] == list(CHECKPOINTS)
        and r["source_learner_unchanged"]
        for r in rows
    )
    initial, neutral, means = True, True, True
    for mode in (*RAW_MODES, "ordinary_fixed", "bounded_renewal"):
        controls = [r for r in rows if r["mode"] == mode]
        reference = controls[0]["stream_result"]["checkpoints"][0]["scores"]
        for row in controls[1:]:
            scores = row["stream_result"]["checkpoints"][0]["scores"]
            initial &= all(
                abs(scores[c][m] - reference[c][m]) <= 1e-12
                for c in ("A", "B")
                for m in ("bce", "accuracy")
            )
        pc, control = indexed["pc", mode], indexed["neutral_cpc", mode]
        field = "raw_parameters" if mode in RAW_MODES else "parameters"
        neutral &= all(
            a[field] == b[field]
            for a, b in zip(pc["native_checkpoints"], control["native_checkpoints"], strict=True)
        )
    for row in rows:
        capacity = row["capacity"]
        prior: Any = None
        used: list[Any] = []
        for sample in capacity["samples"]:
            if sample["status"] != "complete":
                means = False
                continue
            if sample["phase"] == "post_wake":
                used.append(prior)
            prior = sample["usage"]
        means &= (
            len(used) == 18
            and all(u is not None for u in used)
            and capacity["completed_native_calls"] == 18
            and capacity["completed_training_rows"] == 144
        )
        if len(used) == 18 and all(u is not None for u in used):
            for name in ("parameters", "parameter_bytes", "auxiliary_bytes"):
                means &= capacity[f"mean_{name}"] == sum(u[name] for u in used) / 18
            means &= (
                capacity["mean_total_array_bytes"]
                == sum(u["parameter_bytes"] + u["auxiliary_bytes"] for u in used) / 18
            )
    return dict(
        completed_work_exact=work,
        initial_within_mode_match=initial,
        neutral_numeric_paths_match=neutral,
        capacity_means_exact=means,
    )


def run_pilot() -> dict[str, Any]:
    validate_local_pilot_request(PilotRequest(resources=RESOURCES))
    started = monotonic()
    rows, errors = _rows(), []
    rss: dict[str, Any] | None = None
    sampler: ProcessRssSampler | None = None
    try:
        sampler = ProcessRssSampler()
        with sampler:
            for row in rows:
                if monotonic() - started >= RESOURCES.wall_seconds:
                    errors.append("global wall allowance exhausted; later rows not run")
                    break
                _run_one(row, started + RESOURCES.wall_seconds, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup unavailable")
        rss = asdict(sampler.snapshot())
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    elapsed, facts = monotonic() - started, _facts(rows)
    complete = (
        not errors
        and all(facts.values())
        and elapsed <= RESOURCES.wall_seconds
        and rss is not None
        and rss["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        runs=rows,
        comparisons=_comparisons(rows),
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=rss,
        correctness_facts=facts,
        wake_updates=sum(r["committed_updates"] for r in rows),
        completed_training_rows=sum(r["committed_training_rows"] for r in rows),
        full_R26_complete=False,
        scientific_advantage_verified=False,
    )
