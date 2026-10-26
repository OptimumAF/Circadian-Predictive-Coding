"""Fixed slow-teacher replay on the existing coherent four-learner stream.

Owns native binding and resource reporting; the existing application runner
owns training, arrived-row replay and heldout scoring. Historical results are
never read here. This is probability distillation, not DER/DER++ replication.
"""

from dataclasses import asdict, fields
import pickle
from time import monotonic
from typing import Any

from src.adapters.coupled_attribution_pilot import build_learner, build_stream
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.adapters.slow_teacher_learner import SlowTeacherLearner
from src.adapters.stream_rehearsal_pilot import CHECKPOINTS, describe_protocol as describe_replay
from src.adapters.stream_rehearsal_pilot import replay_policy
from src.app.coupled_attribution import run_attribution_stream
from src.app.coupled_attribution_summary import METHODS
from src.app.stream_rehearsal import StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.core.slow_teacher import SlowTeacherPolicy
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coherent_stream_slow_teacher_replay_v1"
RESOURCES = PilotResources(
    training_updates=120, environment_steps=0, replay_bytes=384, storage_bytes=2 * 1024**2
)


def teacher_policy(method: str) -> SlowTeacherPolicy:
    if method not in METHODS:
        raise ValueError("unknown slow-teacher pilot method")
    return SlowTeacherPolicy(f"slowteacher-{method}-seed23", "fixed_uniform_x2_v1")


def describe_protocol() -> dict[str, Any]:
    """Describe frozen work without constructing models or accessing scores."""
    protocol = describe_replay()
    protocol.update(
        protocol_id=PROTOCOL_ID,
        classification="exploratory_slow_teacher_probability_distillation",
        teachers={method: asdict(teacher_policy(method)) for method in METHODS},
        teacher_update="EMA of post-wake parameters only; no replay EMA or chemistry averaging",
        teacher_targets="fresh probability forward per selected arrived hard row; no retained q cache",
        objective="native BCE on (1-lambda)*y+lambda*q, unchanged hidden penalties; explicit diagnostic ID",
        teacher_storage_array_bytes_per_arm=840,
        comparison="post-run only against retained seed23 replay-only evidence; equal hard-row exposure, additional teacher work/memory",
        resources=asdict(RESOURCES),
        limitations="single seed/data; no equal-compute, DER/DER++, causal, confirmatory or circadian advantage claim; Torch distillation and teacher-stream continuation unsupported",
    )
    protocol["planned_work"].update(
        teacher_forward_calls=48,
        teacher_ema_updates=72,
        teacher_probability_array_bytes_generated=384,
        mixed_target_array_bytes_returned=384,
        retained_teacher_target_bytes=0,
    )
    return protocol


def _parameters(parameters: CoupledParameters) -> dict[str, Any]:
    return {
        field.name: [a.tolist() for a in value] if type(value) is tuple else value.tolist()
        for field in fields(parameters)
        for value in (getattr(parameters, field.name),)
    }


def _outcomes(result: dict[str, Any]) -> dict[str, float]:
    points = {row["updates"]: row["scores"] for row in result["checkpoints"]}
    return dict(
        new_B_gain=points[0]["B"]["bce"] - points[12]["B"]["bce"],
        retention_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
        return_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
    )


def _run_one(
    row: dict[str, Any], initial: CoupledParameters, deadline: float, sampler: ProcessRssSampler
) -> None:
    started = monotonic()
    wake = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=18,
            max_wall_seconds=max(0.0, deadline - started),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(
            max_replay_examples=12,
            max_wall_seconds=max(0.0, deadline - started),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    helper = None
    try:
        wake.attach_memory(sampler)
        replay.attach_memory(sampler)
        wake.before_final()
        replay.before_final()
        training, final = build_stream("recurring")
        policy = teacher_policy(row["method"])
        student = build_learner(row["method"], initial)
        if not isinstance(student, (CoupledLearner, MatchedBackpropLearner)):
            raise ValueError("teacher pilot requires the ordinary coherent NumPy learner")
        learner = SlowTeacherLearner(student, policy)
        before = pickle.dumps(learner.snapshot_state())
        helper = StreamRehearsal(
            learner,
            wake,
            replay_policy(),
            replay_budget=replay,
            total_updates=18,
            input_dim=2,
            distillation=policy,
        )

        def observe(updates: int, state: Any, arrived: Any) -> None:
            # Detached checkpoint data only: no update, prediction or score use.
            row["teacher_checkpoints"].append(
                dict(
                    updates=updates,
                    wake_updates=state.wake_updates,
                    replay_updates=state.replay_updates,
                    teacher_revision=state.teacher_revision,
                    student_parameters=_parameters(state.student_state.parameters),
                    teacher_parameters=_parameters(state.teacher_parameters),
                )
            )

        result = run_attribution_stream(
            learner,
            training,
            final,
            checkpoint_updates=CHECKPOINTS,
            label_delay=2,
            budget=wake,
            rehearsal=helper,
            observe_checkpoint=observe,
        )
        if pickle.dumps(learner.snapshot_state()) != before:
            raise RuntimeError("slow-teacher stream changed its source learner")
        row.update(
            status="complete",
            reason=None,
            stream_result=result,
            source_learner_unchanged=True,
            outcomes=_outcomes(result),
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    row.update(
        wake_updates=wake.updates_completed,
        replay_charged_examples=replay.replay_examples_completed,
        elapsed_seconds=monotonic() - started,
        rehearsal=None if helper is None else helper.report,
    )


def _parity(rows: list[dict[str, Any]]) -> bool:
    pc, neutral = rows[1:3]
    # Teacher IDs identify distinct learners; every numerical path must match.
    keys = ("checkpoints", "applied_updates", "applied_consolidations")
    return pc["teacher_checkpoints"] == neutral["teacher_checkpoints"] and all(
        pc["stream_result"][key] == neutral["stream_result"][key] for key in keys
    )


def run_pilot() -> dict[str, Any]:
    """Execute one fixed family, retaining failed work and unreached arms."""
    validate_local_pilot_request(PilotRequest(RESOURCES))
    started = monotonic()
    rows: list[dict[str, Any]] = [
        dict(
            method=m,
            stream="recurring",
            seed=23,
            status="not_run",
            reason="pilot stopped before execution",
            wake_updates=0,
            replay_charged_examples=0,
            rehearsal=None,
            teacher_checkpoints=[],
            planned_event_updates=list(range(3, 19, 3)),
        )
        for m in METHODS
    ]
    sampler = None
    errors = []
    initial_data = None
    try:
        sampler = ProcessRssSampler()
        with sampler:
            initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
            initial_data = _parameters(initial)
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
    wake = sum(row["wake_updates"] for row in rows)
    replay = sum(row["replay_charged_examples"] for row in rows)
    complete_rows = all(row["status"] == "complete" for row in rows)
    parity = _parity(rows) if complete_rows else None
    reports = [row["rehearsal"] for row in rows if row["rehearsal"] is not None]
    returned = sum(r["returned_updates"] for r in reports)
    committed = sum(r["committed_updates"] for r in reports)
    forwards = sum(r["distillation"]["teacher_forward_attempts"] for r in reports)
    plans = [[(e["updates"], e["selected_ids"]) for e in r["events"]] for r in reports]
    shared = len(plans) == 4 and all(p == plans[0] for p in plans)
    costs_exact = all(
        r["distillation"]["teacher_forward_attempts"]
        == r["distillation"]["teacher_forward_returns"]
        == 12
        and r["distillation"]["teacher_storage_array_bytes"] == 840
        and r["distillation"]["teacher_probability_bytes_generated"]
        == r["distillation"]["mixed_target_bytes_returned"]
        == 96
        and r["distillation"]["retained_target_array_bytes"] == 0
        for r in reports
    )
    clocks_exact = complete_rows and all(
        [
            (c["updates"], c["wake_updates"], c["replay_updates"], c["teacher_revision"])
            for c in row["teacher_checkpoints"]
        ]
        == [(0, 0, 0, 0), (6, 6, 4, 6), (12, 12, 8, 12), (18, 18, 12, 18)]
        for row in rows
    )
    complete = (
        not errors
        and complete_rows
        and wake == 72
        and replay == returned == committed == forwards == 48
        and shared
        and parity is True
        and costs_exact
        and clocks_exact
        and elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        wake_updates=wake,
        replay_charged_examples=replay,
        replay_returned_updates=returned,
        replay_committed_updates=committed,
        teacher_forward_attempts=forwards,
        row_exposures_charged=8 * wake + replay,
        shared_replay_plans_exact=shared,
        pc_neutral_exact=parity,
        teacher_costs_exact=costs_exact,
        teacher_clocks_exact=clocks_exact,
        scientific_advantage_verified=False,
        full_R25_complete=False,
    )
