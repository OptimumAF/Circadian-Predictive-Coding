"""Fixed bounded-inference demo using the existing delayed attribution stream.

Owns composition and resource observations, not inference mathematics, policy
selection or matched-compute comparison. Final scores cannot select settings.
"""

from dataclasses import asdict
import pickle
from time import monotonic
from typing import Any

from src.adapters.coupled_attribution_pilot import build_learner, build_stream
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSampler

METHODS = ("pc", "neutral_cpc", "gated_cpc")
RESOURCES = PilotResources(
    wall_seconds=30, training_updates=54, environment_steps=0, replay_bytes=0
)


def describe_stream_protocol() -> dict[str, Any]:
    """Describe the fixed demonstration without constructing data or learners."""
    return dict(
        protocol_id="bounded_inference_recurring_stream_v1",
        classification="engineering_demo_not_matched_compute",
        methods=list(METHODS),
        seeds=[23],
        stream="recurring",
        training_rng=7301,
        final_rng=9101,
        hidden_dims=[8, 8],
        learning_rate=0.03,
        inference_step_cap=4,
        inference_learning_rate=0.2,
        inference_policy=asdict(CoupledInferencePolicy(min_steps=2, gradient_tolerance=1.0)),
        policy_origin="existing engineering example, not selected from final scores",
        precisions=[1.0, 1.0],
        contexts=["A", "B", "A"],
        updates_per_phase=6,
        batch_rows=8,
        label_delay=2,
        checkpoint_updates=[0, 6, 12, 18],
        final_rows_per_context=64,
        primary_metric="probability_bce_clip_1e-12_lower_better",
        secondary_metric="accuracy_threshold_ge_0.5_higher_better",
        score_release="after all18 wakes in each case; no feedback",
        planned_work=dict(
            training_updates=54,
            training_rows=432,
            final_forwards=24,
            final_rows=1536,
            probes=0,
            replay_rows=0,
            health_forwards=0,
            svd_calls=0,
        ),
        resources=asdict(RESOURCES),
        inference_work="unavailable through generic TrainingDiagnostic; not fixed-step counts",
        limitations="one fixed seed/data; no matched-time/work, speed or CPC advantage claim",
    )


def _run_case(row: dict[str, Any], initial: CoupledParameters, budget: ToyBudgetSession) -> None:
    started = monotonic()
    learner = None
    before = None
    try:
        budget.before_final()
        training, final = build_stream("recurring")
        learner = build_learner(
            row["method"],
            initial,
            inference_policy=CoupledInferencePolicy(min_steps=2, gradient_tolerance=1.0),
        )
        before = pickle.dumps(learner.snapshot_state())
        budget.before_final()
        stream = AttributionStream(
            learner,
            training,
            final,
            checkpoint_updates=(0, 6, 12, 18),
            label_delay=2,
            budget=budget,
        )
        stream.run_until(18)
        row["stream_result"] = stream.finish(score_progress=row["final_score_progress"])
        budget.before_final()
        points = {p["updates"]: p["scores"] for p in row["stream_result"]["checkpoints"]}
        row["bce_outcomes"] = dict(
            initial_A_gain=points[0]["A"]["bce"] - points[6]["A"]["bce"],
            new_B_gain=points[6]["B"]["bce"] - points[12]["B"]["bce"],
            old_A_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
            return_A_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
            return_gap=points[18]["A"]["bce"] - points[6]["A"]["bce"],
        )
        row.update(status="complete", reason=None)
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        row["completed_updates"] = budget.updates_completed
        row["completed_training_rows"] = 8 * budget.updates_completed
        row["attempted_native_updates"] = None
        row["inference_work"] = None
        row["elapsed_seconds"] = monotonic() - started
        if learner is not None and before is not None:
            try:
                unchanged = pickle.dumps(learner.snapshot_state()) == before
                row["source_learner_unchanged"] = unchanged
                if not unchanged:
                    raise RuntimeError("stream changed its source learner")
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")


def run_stream_example() -> dict[str, Any]:
    """Run once with original update sessions and one shared wall/RSS ceiling."""
    validate_local_pilot_request(PilotRequest(RESOURCES))
    started = monotonic()
    deadline = started + RESOURCES.wall_seconds
    rows: list[dict[str, Any]] = [
        dict(
            method=method,
            status="not_run",
            reason="stopped before execution",
            stream_result=None,
            completed_updates=0,
            source_learner_unchanged=None,
            elapsed_seconds=0.0,
            final_score_progress={},
        )
        for method in METHODS
    ]
    errors: list[str] = []
    sampler = None
    memory = None
    try:
        sampler = ProcessRssSampler()
        with sampler:
            # Why: every case pays setup and preceding cases' time; no renewed
            # allowance when the next learner starts or final scoring begins.
            sessions = tuple(
                ToyBudgetSession(
                    ToyExecutionBudget(
                        max_training_updates=18,
                        max_wall_seconds=max(0.0, deadline - monotonic()),
                        max_process_rss_bytes=RESOURCES.process_memory_bytes,
                    ),
                    monotonic,
                )
                for _ in METHODS
            )
            for session in sessions:
                session.attach_memory(sampler)
                session.before_final()
            initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
            for row, session in zip(rows, sessions, strict=True):
                _run_case(row, initial, session)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    if sampler is not None:
        try:
            memory = asdict(sampler.snapshot())
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    elapsed = monotonic() - started
    within = (
        elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    pair = rows[:2]
    parity = (
        pair[0]["stream_result"] == pair[1]["stream_result"]
        if all(r["status"] == "complete" for r in pair)
        else None
    )
    if parity is False:
        errors.append("PC and neutral CPC stream semantics differ")
    complete = (
        within and not errors and parity is True and all(r["status"] == "complete" for r in rows)
    )
    return dict(
        protocol=describe_stream_protocol(),
        status="complete" if complete else "incomplete",
        runs=rows,
        global_errors=errors,
        elapsed_seconds=elapsed,
        observed_process_rss=memory,
        within_resources=within,
        pc_neutral_exact=parity,
        scientific_advantage_verified=False,
        full_R15b_complete=False,
    )
