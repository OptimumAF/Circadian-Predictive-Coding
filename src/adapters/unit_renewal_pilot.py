"""Fixed renewal adaptation scenario composed from the existing stream and probes.

Only renewable PC-family arms execute here. Accepted no-renewal BP/PC/CPC controls
are named for later analysis and never opened by training or probe selection.
No tuning, replay, promotion, persistence or comparative-success gate is provided.
"""

from dataclasses import asdict, fields
import pickle
from time import monotonic
from typing import Any

from src.adapters import learnability_pilot as learnability
from src.adapters.coupled_attribution_pilot import build_stream
from src.adapters.coupled_learners import CoupledLearner
from src.app.coupled_attribution import run_attribution_stream
from src.app.coupled_attribution_summary import CHECKPOINTS, METHODS
from src.app.stream_unit_renewal import StreamUnitRenewal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_unit_renewal import RenewalPolicy
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.core.renewable_coupled_predictive_coding import (
    RenewableCoupledCircadianPredictiveCodingNetwork,
    RenewableCoupledPredictiveCodingNetwork,
)
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coherent_renewal_reserved_sixteen_v1"
EXECUTED_METHODS = METHODS[1:]
CONTROL_ARTIFACT = "artifacts/runs/r21-sixteen-probe-20261016/pilot.json"
RESOURCES = PilotResources(
    training_updates=246, environment_steps=0, replay_bytes=0, storage_bytes=2 * 1024**2
)


def renewal_policy() -> RenewalPolicy:
    """Prespecified round cutoff, without calibration on observed outcomes."""
    return RenewalPolicy(6, 0.001, 1, 29)


def describe_protocol() -> dict[str, Any]:
    original = learnability.describe_protocol(extended=True)
    return dict(
        protocol_id=PROTOCOL_ID,
        classification="exploratory_fixed_data_renewal_adaptation",
        seed=23,
        executed_methods=list(EXECUTED_METHODS),
        retained_controls=dict(
            artifact=CONTROL_ARTIFACT,
            methods=list(METHODS),
            protocol_id=original["protocol_id"],
            access="postrun analysis only; never read by this adapter or learners",
        ),
        benchmark={
            key: original[key]
            for key in (
                "benchmark_stream",
                "benchmark_updates_per_phase",
                "benchmark_checkpoint_updates",
                "benchmark_training_rng",
                "benchmark_final_rng",
                "hidden_dims",
                "learning_rate",
                "precisions",
                "active_cpc_config",
                "neutral_cpc_config",
                "benchmark_batch_rows",
                "benchmark_final_rows_per_context",
                "benchmark_target_rules",
                "label_delay",
                "inference_steps",
                "inference_learning_rate",
            )
        },
        probe={
            key: original[key]
            for key in (
                "probe_target",
                "probe_training_rng",
                "probe_evaluation_rng",
                "diagnostic_features_rng",
                "probe_training_batches",
                "probe_training_batch_rows",
                "probe_evaluation_rows",
                "probe_checkpoint_updates",
                "diagnostic_feature_rows",
                "feature_health_definition",
                "effective_rank_definition",
                "effective_rank_limit",
                "reserved_data_array_bytes",
                "diagnostic_reference_array_bytes",
            )
        },
        renewal=dict(
            policy=asdict(renewal_policy()),
            history_decay=0.95,
            every_updates=6,
            max_replacements_per_arm=3,
            eligible_selection="age >=6 AND gradient EMA <=.001; ascending EMA/stable ID/index",
            threshold_rationale="fixed round cutoff declared before new histories/outcomes; no calibration or tuning; retain zero-eligibility outcomes",
            charging="permanent before private preparation; no refund on failure/native restore",
            probes="complete renewable state copied to independent templates; no renewal helper in disposable probes",
        ),
        score_definitions=original["score_definitions"],
        comparison_directions={
            **original["comparison_directions"],
            "paired_difference": "renewal minus retained same-method no-renewal; retain absolute levels and all C points",
        },
        isolation="all probes finish before benchmark final scoring; no final/probe feedback to eligibility, training or promotion",
        planned_work=dict(
            benchmark_updates=54,
            probe_updates=192,
            total_updates=246,
            benchmark_training_row_exposures=432,
            probe_training_row_exposures=1536,
            total_training_row_exposures=1968,
            benchmark_score_forward_calls=24,
            probe_score_forward_calls=72,
            diagnostic_feedforward_passes=12,
            diagnostic_svd_calls=24,
            nominal_coupled_energy_calls=1476,
            max_replacement_units=9,
            max_consolidation_events=9,
            replay_examples=0,
            retained_control_new_native_updates=0,
        ),
        resources=asdict(RESOURCES),
        stop_policy="one invocation; preserve every failed/unreached row and permanent charge; no retry/tuning",
        limitations="one fixed seed/data; no intervals/equivalence/generalization/confirmation/CPC advantage; gradient is a proxy; BP compute differs; prior and new profiled cohort timings do not establish efficiency",
    )


def _build_learner(method: str, initial: CoupledParameters) -> CoupledLearner:
    if method not in EXECUTED_METHODS:
        raise ValueError("renewal scenario executes only PC, neutral CPC and gated CPC")
    model: CoupledPredictiveCodingNetwork
    if method == "pc":
        model = RenewableCoupledPredictiveCodingNetwork(
            2, (8, 8), 0, renewal_policy=renewal_policy()
        )
    else:
        config = (
            CoupledCircadianConfig.matched_pc_control()
            if method == "neutral_cpc"
            else CoupledCircadianConfig()
        )
        model = RenewableCoupledCircadianPredictiveCodingNetwork(
            2, (8, 8), 0, config=config, renewal_policy=renewal_policy()
        )
    model.restore_parameters(initial)
    return CoupledLearner(model, learning_rate=0.03, inference_steps=4, inference_learning_rate=0.2)


def _outcomes(result: dict[str, Any]) -> dict[str, float]:
    points = {row["updates"]: row["scores"] for row in result["checkpoints"]}
    return dict(
        retention_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
        return_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
        new_B_gain=points[0]["B"]["bce"] - points[12]["B"]["bce"],
    )


def _run_one(
    row: dict[str, Any], initial: CoupledParameters, deadline: float, sampler: ProcessRssSampler
) -> None:
    started = monotonic()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=18,
            max_wall_seconds=max(0.0, deadline - started),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    helper = None
    try:
        budget.attach_memory(sampler)
        budget.before_final()
        training, heldout = build_stream("recurring")
        learner = _build_learner(row["method"], initial)
        template = _build_learner(row["method"], initial)
        before = pickle.dumps(learner.snapshot_state())
        helper = StreamUnitRenewal(
            learner, budget, every_updates=6, total_updates=18, max_replacements=3
        )
        observer = learnability.create_sixteen_probe_observer(
            template, row["observations"], deadline, sampler
        )
        result = run_attribution_stream(
            learner,
            training,
            heldout,
            checkpoint_updates=CHECKPOINTS,
            label_delay=2,
            budget=budget,
            observe_checkpoint=observer,
            renewal=helper,
        )
        if pickle.dumps(learner.snapshot_state()) != before:
            raise RuntimeError("benchmark source changed during candidate/probe work")
        row.update(
            status="complete",
            reason=None,
            stream_result=result,
            source_unchanged=True,
            outcomes=_outcomes(result),
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}", outcomes=None)
    finally:
        row.update(
            benchmark_completed_updates=budget.updates_completed,
            renewal=None if helper is None else helper.report,
            failed_attempt_work_not_fully_counted=row["status"] != "complete",
            elapsed_seconds=monotonic() - started,
        )
        row["diagnostic_seconds"] = sum(
            o.get("diagnostic_seconds", 0.0) for o in row["observations"]
        )
        row["probe_seconds"] = sum(o.get("probe_seconds", 0.0) for o in row["observations"])
        row["benchmark_and_orchestration_seconds"] = (
            row["elapsed_seconds"] - row["diagnostic_seconds"] - row["probe_seconds"]
        )


def _parity(rows: list[dict[str, Any]]) -> bool | None:
    pc, neutral = rows[:2]
    if pc["status"] != "complete" or neutral["status"] != "complete":
        return None
    return (
        pc["stream_result"] == neutral["stream_result"]
        and [o["probe"]["checkpoints"] for o in pc["observations"]]
        == [o["probe"]["checkpoints"] for o in neutral["observations"]]
        and [o.get("native_unit_history") for o in pc["observations"]]
        == [o.get("native_unit_history") for o in neutral["observations"]]
    )


def run_pilot() -> dict[str, Any]:
    """Execute only the three new arms; final outcomes never choose later work."""
    validate_local_pilot_request(PilotRequest(RESOURCES))
    started = monotonic()
    rows: list[dict[str, Any]] = [
        dict(
            method=method,
            seed=23,
            status="not_run",
            reason="pilot not reached",
            observations=learnability.empty_learnability_observations(),
            outcomes=None,
            benchmark_completed_updates=0,
            renewal=None,
        )
        for method in EXECUTED_METHODS
    ]
    sampler = None
    errors: list[str] = []
    initial_data = None
    try:
        sampler = ProcessRssSampler()
        with sampler:
            initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
            initial_data = {
                field.name: (
                    [a.tolist() for a in value] if type(value) is tuple else value.tolist()
                )
                for field in fields(initial)
                for value in (getattr(initial, field.name),)
            }
            for row in rows:
                if monotonic() - started >= RESOURCES.wall_seconds:
                    break
                _run_one(row, initial, started + RESOURCES.wall_seconds, sampler)
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
    probes = [o["probe"] for row in rows for o in row["observations"] if o["probe"] is not None]
    wake = sum(r["benchmark_completed_updates"] for r in rows)
    probe_updates = sum(p["completed_updates"] for p in probes)
    parity = _parity(rows)
    complete = (
        not errors
        and all(r["status"] == "complete" for r in rows)
        and wake == 54
        and probe_updates == 192
        and len(probes) == 12
        and elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
        and parity is True
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        benchmark_completed_updates=wake,
        probe_completed_updates=probe_updates,
        total_completed_updates=wake + probe_updates,
        benchmark_training_row_exposures=8 * wake,
        probe_training_row_exposures=sum(p["completed_training_examples"] for p in probes),
        probe_attempted_updates=sum(p["attempted_updates"] for p in probes),
        probe_score_forward_calls_attempted=sum(p["score_forward_calls_attempted"] for p in probes),
        probe_score_forward_calls_completed=sum(p["score_forward_calls_completed"] for p in probes),
        diagnostic_feedforward_passes=sum(
            "feature_health" in o for r in rows for o in r["observations"]
        ),
        diagnostic_svd_calls=sum(
            len(o.get("feature_health", ())) for r in rows for o in r["observations"]
        ),
        charged_replacement_units=sum((r["renewal"] or {}).get("charged_units", 0) for r in rows),
        committed_replacement_units=sum(
            (r["renewal"] or {}).get("committed_units", 0) for r in rows
        ),
        pc_neutral_curves_and_histories_exact=parity,
        scientific_advantage_verified=False,
        work_scope="committed/returned work shown; failed unknown work flagged, not imputed as zero; native guard receipt counts attempted entries; no FLOP or efficiency claim",
    )
