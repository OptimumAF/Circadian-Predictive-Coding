"""Fixed A16/B32/A16 stream with the existing matched shared-replay runner.

Owns fixed data and bounded pilot reporting. The application owns delayed labels,
replay admission and final-score isolation. Content-to-phase classification is
report-only: no context, future labels or metrics reach the replay selector.
This is one exploratory stream, not a retention guarantee or policy comparison.
"""

from dataclasses import asdict, fields
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.coupled_attribution_pilot import build_learner, build_stream
from src.adapters.stream_rehearsal_pilot import describe_protocol as _short_protocol
from src.adapters.stream_rehearsal_pilot import replay_policy
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.coupled_attribution_summary import METHODS
from src.app.stream_rehearsal import StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import replay_sample_id
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coherent_long_stream_shared_rehearsal_v1"
CHECKPOINTS = (0, 16, 48, 64)
RESOURCES = PilotResources(
    training_updates=424, environment_steps=0, replay_bytes=384, storage_bytes=2 * 1024**2
)
EVENT_UPDATES = tuple(range(3, 64, 3))


def describe_protocol() -> dict[str, Any]:
    """Describe the frozen protocol without data, model or resource work."""
    protocol = _short_protocol()
    protocol.update(
        protocol_id=PROTOCOL_ID,
        classification="exploratory_long_stream_retention_and_exposure",
        training_batches=64,
        phase_lengths=[16, 32, 16],
        phase_contexts=["A", "B", "A"],
        checkpoint_updates=list(CHECKPOINTS),
        directions=dict(
            new_B_gain="B BCE at0 minus48",
            retention_loss="A BCE at48 minus16",
            return_gain="A BCE at48 minus64",
        ),
        planned_work=dict(
            wake_updates=256,
            replay_updates=168,
            optimizer_updates=424,
            wake_row_exposures=2048,
            replay_row_exposures=168,
            total_row_exposures=2216,
            consolidation_events=84,
            final_score_forwards=32,
            probes=0,
            svd=0,
        ),
        resources=asdict(RESOURCES),
        train_manifest="report-only content IDs and first arrival phase; never selector input",
        limitations="one fixed stream/seed; retention and actual committed replay exposure are distinct; no counterfactual short-stream comparison, intervals, universal retention, generalization or circadian advantage claim",
    )
    return protocol


def build_long_stream() -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    """Use the original final contexts and an independently fixed 64-batch train stream."""
    _, final = build_stream("recurring")
    rng = np.random.default_rng(7301)
    training = []
    for index in range(64):
        context = "B" if 16 <= index < 48 else "A"
        features = rng.uniform(-1, 1, (8, 2))
        targets = (features[:, [1 if context == "B" else 0]] >= 0).astype(np.float64)
        training.append(
            AttributionBatch(
                "train",
                context,
                tuple(f"long-train-{index}-{i}" for i in range(8)),
                features,
                targets,
            )
        )
    return tuple(training), final


def _train_manifest(training: tuple[AttributionBatch, ...]) -> list[dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for update, batch in enumerate(training, 1):
        phase = "first_A" if update <= 16 else "B" if update <= 48 else "return_A"
        for index in range(batch.features.shape[0]):
            sample_id = replay_sample_id(
                batch.features[index : index + 1], batch.targets[index : index + 1]
            )
            if sample_id in manifest:
                if manifest[sample_id]["phase"] != phase:
                    raise ValueError("training content ID reused across phases is ambiguous")
                continue
            if len(manifest) >= 512:
                raise ValueError("long-stream manifest exceeds 512 unique training rows")
            manifest[sample_id] = dict(
                sample_id=sample_id, first_arrival_update=update, phase=phase, context=batch.context
            )
    return list(manifest.values())


def _exposure(
    report: dict[str, Any] | None, manifest: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if report is None:
        return []
    old_ids = {row["sample_id"] for row in manifest if row["phase"] == "first_A"}
    result = []
    for event in report["events"]:
        if not 16 < event["updates"] <= 48:
            continue
        retained = [] if event["retention"] is None else event["retention"]["sample_ids"]
        selected = event["selected_ids"]
        # Publication commits the entire selected event; failed attempted calls
        # without a receipt are deliberately not counted as committed exposure.
        committed = (
            selected
            if event["receipt"] is not None and event["committed_updates"] == len(selected)
            else []
        )
        values = {
            "retained_first_A_ids": [item for item in retained if item in old_ids],
            "selected_first_A_ids": [item for item in selected if item in old_ids],
            "committed_first_A_ids": [item for item in committed if item in old_ids],
        }
        result.append(
            dict(
                updates=event["updates"],
                status=event["status"],
                **values,
                **{key.removesuffix("_ids"): len(ids) for key, ids in values.items()},
            )
        )
    return result


def _outcomes(result: dict[str, Any]) -> dict[str, float]:
    points = {row["updates"]: row["scores"] for row in result["checkpoints"]}
    return dict(
        new_B_gain=points[0]["B"]["bce"] - points[48]["B"]["bce"],
        retention_loss=points[48]["A"]["bce"] - points[16]["A"]["bce"],
        return_gain=points[48]["A"]["bce"] - points[64]["A"]["bce"],
    )


def _run_one(
    row: dict[str, Any],
    initial: CoupledParameters,
    training: tuple[AttributionBatch, ...],
    final: tuple[AttributionBatch, ...],
    deadline: float,
    sampler: ProcessRssSampler,
) -> None:
    wake = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=64,
            max_wall_seconds=max(0.0, deadline - monotonic()),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(
            max_replay_examples=42,
            max_wall_seconds=max(0.0, deadline - monotonic()),
            max_process_rss_bytes=RESOURCES.process_memory_bytes,
        ),
        monotonic,
    )
    rehearsal = None
    learner = None
    before = None
    started = monotonic()
    try:
        wake.attach_memory(sampler)
        replay.attach_memory(sampler)
        wake.before_final()
        replay.before_final()
        learner = build_learner(row["method"], initial)
        before = pickle.dumps(learner.snapshot_state())
        rehearsal = StreamRehearsal(
            learner, wake, replay_policy(), replay_budget=replay, total_updates=64, input_dim=2
        )
        result = run_attribution_stream(
            learner,
            training,
            final,
            checkpoint_updates=CHECKPOINTS,
            label_delay=2,
            budget=wake,
            rehearsal=rehearsal,
        )
        if pickle.dumps(learner.snapshot_state()) != before:
            raise RuntimeError("long-stream rehearsal changed the source learner")
        row.update(
            status="complete",
            reason=None,
            stream_result=result,
            source_learner_unchanged=True,
            outcomes=_outcomes(result),
        )
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
        if learner is not None and before is not None:
            try:
                row["source_learner_unchanged"] = pickle.dumps(learner.snapshot_state()) == before
            except Exception as snapshot_error:
                row["source_check_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
    rehearsal_report = None if rehearsal is None else rehearsal.report
    endpoint_complete = (
        row["status"] == "complete"
        and rehearsal_report is not None
        and rehearsal_report["wake_updates_observed"] == 64
    )
    row.update(
        wake_updates=wake.updates_completed,
        replay_charged_examples=replay.replay_examples_completed,
        elapsed_seconds=monotonic() - started,
        rehearsal=rehearsal_report,
        final_retention=(
            asdict(rehearsal.retention) if endpoint_complete and rehearsal is not None else None
        ),
        final_retention_observed_updates=64 if endpoint_complete else None,
    )


def _rows() -> list[dict[str, Any]]:
    return [
        dict(
            method=method,
            stream="long_recurring",
            seed=23,
            status="not_run",
            reason="pilot stopped before execution",
            wake_updates=0,
            replay_charged_examples=0,
            rehearsal=None,
            final_retention=None,
            final_retention_observed_updates=None,
            stream_result=None,
            outcomes=None,
            source_learner_unchanged=None,
            elapsed_seconds=None,
            planned_event_updates=list(EVENT_UPDATES),
            B_event_retention_exposure=[],
        )
        for method in METHODS
    ]


def _plan(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: event[key]
            for key in (
                "updates",
                "selected_ids",
                "retention",
                "partition_retention",
                "rotation_cursor",
            )
        }
        for event in report["events"]
    ]


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [row["rehearsal"] for row in rows if row["rehearsal"] is not None]
    wake = sum(row["wake_updates"] for row in rows)
    charged = sum(row["replay_charged_examples"] for row in rows)
    attempted = sum(report["attempted_updates"] for report in reports)
    returned = sum(report["returned_updates"] for report in reports)
    committed = sum(report["committed_updates"] for report in reports)
    scores = sum(
        row["stream_result"]["score_forward_calls"] for row in rows if row["status"] == "complete"
    )
    complete = all(row["status"] == "complete" for row in rows)
    return dict(
        wake_updates=wake,
        replay_charged_examples=charged,
        replay_attempted_updates=attempted,
        replay_returned_updates=returned,
        replay_committed_updates=committed,
        total_committed_updates=wake + committed,
        row_exposures_charged=8 * wake + charged,
        known_native_train_calls=wake + attempted,
        native_train_calls=wake + attempted if complete else None,
        known_completed_score_forwards=scores,
        final_score_forwards=scores if complete else None,
        partial_work_note="reported wake receipts and permanent replay attempts retained; failed wake/scoring calls may contain additional unknown work, never zero-imputed",
    )


def run_pilot() -> dict[str, Any]:
    """Run all planned arms once, keeping failures and unreached slots."""
    validate_local_pilot_request(PilotRequest(RESOURCES))
    rows = _rows()
    errors: list[str] = []
    sampler = None
    initial_data = None
    manifest: list[dict[str, Any]] = []
    started = monotonic()
    try:
        sampler = ProcessRssSampler()
        with sampler:
            training, final = build_long_stream()
            manifest = _train_manifest(training)
            initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
            initial_data = {
                item.name: ([a.tolist() for a in value] if type(value) is tuple else value.tolist())
                for item in fields(initial)
                for value in (getattr(initial, item.name),)
            }
            for row in rows:
                if monotonic() - started >= RESOURCES.wall_seconds:
                    break
                try:
                    _run_one(
                        row, initial, training, final, started + RESOURCES.wall_seconds, sampler
                    )
                except Exception as error:
                    row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    for row in rows:
        row["B_event_retention_exposure"] = _exposure(row["rehearsal"], manifest)
    reports = [row["rehearsal"] for row in rows if row["rehearsal"] is not None]
    plans = [_plan(report) for report in reports]
    shared_plans = len(plans) == 4 and all(plan == plans[0] for plan in plans)
    complete_rows = all(row["status"] == "complete" for row in rows)
    parity = rows[1]["stream_result"] == rows[2]["stream_result"] if complete_rows else None
    totals = _totals(rows)
    elapsed = monotonic() - started
    bounded = all(
        event["retention"] is not None
        and event["retention"]["example_count"] <= 16
        and event["retention"]["retained_bytes"] <= 384
        for report in reports
        for event in report["events"]
    ) and all(
        row["final_retention"] is not None
        and row["final_retention"]["example_count"] <= 16
        and row["final_retention"]["retained_bytes"] <= 384
        for row in rows
    )
    complete = (
        not errors
        and complete_rows
        and shared_plans
        and parity is True
        and bounded
        and totals["wake_updates"] == 256
        and totals["replay_charged_examples"]
        == totals["replay_returned_updates"]
        == totals["replay_committed_updates"]
        == 168
        and totals["native_train_calls"] == 424
        and totals["final_score_forwards"] == 32
        and sum(event["committed_updates"] > 0 for report in reports for event in report["events"])
        == 84
        and all(row["source_learner_unchanged"] is True for row in rows)
        and all(row["final_retention_observed_updates"] == 64 for row in rows)
        and elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        train_manifest=manifest,
        runs=rows,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        **totals,
        shared_replay_plans_exact=shared_plans,
        pc_neutral_exact=parity,
        retention_bounds_preserved=bounded,
        scientific_advantage_verified=False,
        full_R23_complete=False,
    )
