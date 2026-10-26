"""Fixed independent-data replay removal through the existing stream recipe.

Owns the eight-arm recipe, explicit failure vectors and descriptive paired
outcomes. Existing rehearsal composition owns native binding, clocks and budget
sessions. This module does not choose settings, tune, promote or prove benefit.
"""

from copy import deepcopy
from dataclasses import asdict, fields
from time import monotonic
from typing import Any

from src.adapters import stream_rehearsal_pilot as rehearsal
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.local_pilot_budget import PilotRequest, PilotResources, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSampler

PROTOCOL_ID = "coherent_independent_data_replay_removal_v1"
POLICIES = ("none", "rotating")
RESOURCES = PilotResources(training_updates=192, environment_steps=0, replay_bytes=384)


def describe_protocol() -> dict[str, Any]:
    """Describe the prospectively fixed family without constructing a learner."""
    protocol = rehearsal.describe_protocol()
    protocol.update(
        protocol_id=PROTOCOL_ID,
        classification="prospective_single_initialization_component_removal",
        policies=list(POLICIES),
        training_rng=rehearsal.CONFIRMATION_TRAINING_RNG,
        final_rng=rehearsal.CONFIRMATION_FINAL_RNG,
        resources=asdict(RESOURCES),
        comparison_scope="canonical rotating component versus absent component; same initial law and arrived TRAIN data, unequal work and storage",
        final_feedback=False,
        equivalence_margin=None,
        statistical_inference=None,
        limitations="one initialization and independent data pair; all outcomes retained; no significance, generalization, equivalence, minimal bundle or circadian advantage claim",
    )
    protocol["planned_work"].update(
        wake_updates=144,
        replay_updates=48,
        optimizer_updates=192,
        wake_row_exposures=1152,
        replay_row_exposures=48,
        total_row_exposures=1200,
        final_score_forwards=64,
        final_score_row_exposures=4096,
    )
    protocol["per_arm_work"] = dict(
        none=dict(
            wake_updates=18,
            replay_updates=0,
            optimizer_updates=18,
            training_rows=144,
            replay_bytes=0,
        ),
        rotating=dict(
            wake_updates=18,
            replay_updates=12,
            optimizer_updates=30,
            training_rows=156,
            replay_bytes=384,
        ),
    )
    return protocol


def _rows() -> list[dict[str, Any]]:
    return [
        dict(
            seed=23,
            policy=policy,
            method=method,
            stream="recurring",
            status="not_run",
            reason="pilot stopped before execution",
            wake_updates=0,
            replay_charged_examples=0,
            rehearsal=None,
            outcomes=None,
            planned_event_updates=[] if policy == "none" else list(range(3, 19, 3)),
        )
        for policy in POLICIES
        for method in rehearsal.METHODS
    ]


def _execute(rows, initial_data, deadline, sampler):
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
    initial_data.update(
        {
            item.name: ([a.tolist() for a in value] if type(value) is tuple else value.tolist())
            for item in fields(initial)
            for value in (getattr(initial, item.name),)
        }
    )
    for row in rows:
        if monotonic() >= deadline:
            return
        try:
            rehearsal._run_one(
                row,
                initial,
                deadline,
                sampler,
                confirmation=True,
                no_replay=row["policy"] == "none",
            )
        except Exception as error:
            row.update(status="failed", reason=f"{type(error).__name__}: {error}")


def _timing(row):
    keys = (
        "sample_id",
        "episode_id",
        "event_id",
        "actor_version",
        "learner_version",
        "observed_at",
        "arrived_at",
        "applied_at",
        "update_number",
    )
    return [
        {key: receipt[key] for key in keys} for receipt in row["stream_result"]["applied_updates"]
    ]


def _pairs(rows) -> list[dict[str, Any]]:
    indexed = {(row["policy"], row["method"]): row for row in rows}
    pairs = []
    for method in rehearsal.METHODS:
        both = {policy: indexed[policy, method] for policy in POLICIES}
        complete = all(row["status"] == "complete" for row in both.values())
        levels = (
            {p: deepcopy(row["stream_result"]["checkpoints"]) for p, row in both.items()}
            if complete
            else None
        )
        outcomes = {p: deepcopy(row["outcomes"]) for p, row in both.items()} if complete else None
        pairs.append(
            dict(
                seed=23,
                method=method,
                status="complete" if complete else "incomplete",
                reasons={p: row["reason"] for p, row in both.items()},
                primary_levels=levels,
                outcomes=outcomes,
                rotating_minus_none={
                    k: both["rotating"]["outcomes"][k] - both["none"]["outcomes"][k]
                    for k in ("new_B_gain", "retention_loss", "return_gain")
                }
                if complete
                else None,
                initial_primary_levels_exact=levels["none"][0]["scores"]
                == levels["rotating"][0]["scores"]
                if levels is not None
                else None,
                wake_receipt_timing_exact=_timing(both["none"]) == _timing(both["rotating"])
                if complete
                else None,
                native_update_difference=(
                    both["rotating"]["wake_updates"]
                    + both["rotating"]["rehearsal"]["committed_updates"]
                    - both["none"]["wake_updates"]
                )
                if complete
                else None,
                training_row_difference=(
                    8 * (both["rotating"]["wake_updates"] - both["none"]["wake_updates"])
                    + both["rotating"]["replay_charged_examples"]
                    - both["none"]["replay_charged_examples"]
                )
                if complete
                else None,
            )
        )
    return pairs


def _groups(rows) -> list[dict[str, Any]]:
    groups = []
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        complete = all(row["status"] == "complete" for row in group)
        groups.append(
            dict(
                policy=policy,
                status="complete" if complete else "unavailable",
                pc_neutral_exact=group[1]["stream_result"] == group[2]["stream_result"]
                if complete
                else None,
                shared_replay_plans_exact=all(
                    rehearsal._plan(row) == rehearsal._plan(group[0]) for row in group
                )
                if complete and policy == "rotating"
                else None,
                replay_absent=all(
                    row["rehearsal"] is None and row["replay_charged_examples"] == 0
                    for row in group
                )
                if complete and policy == "none"
                else None,
            )
        )
    return groups


def run_pilot() -> dict[str, Any]:
    """Run all declared arms once with one global deadline, never select a winner."""
    validate_local_pilot_request(PilotRequest(RESOURCES))
    rows = _rows()
    initial_data: dict[str, Any] = {}
    errors: list[str] = []
    sampler = None
    started = monotonic()
    try:
        sampler = ProcessRssSampler()
        with sampler:
            _execute(rows, initial_data, started + RESOURCES.wall_seconds, sampler)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        if sampler is None:
            raise RuntimeError("RSS sampler setup unavailable")
        memory = asdict(sampler.snapshot())
    except Exception as error:
        memory = None
        errors.append(f"{type(error).__name__}: {error}")
    pairs, groups, totals = _pairs(rows), _groups(rows), rehearsal._paired_totals(rows)
    elapsed = monotonic() - started
    complete = (
        not errors
        and all(
            pair["status"] == "complete"
            and pair["initial_primary_levels_exact"] is True
            and pair["wake_receipt_timing_exact"] is True
            for pair in pairs
        )
        and all(group["pc_neutral_exact"] is True for group in groups)
        and groups[0]["replay_absent"] is True
        and groups[1]["shared_replay_plans_exact"] is True
        and totals["wake_updates"] == 144
        and totals["replay_charged_examples"]
        == totals["replay_returned_updates"]
        == totals["replay_committed_updates"]
        == 48
        and totals["native_train_calls"] == 192
        and totals["final_score_forwards"] == 64
        and elapsed <= RESOURCES.wall_seconds
        and memory is not None
        and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    return dict(
        protocol=describe_protocol(),
        status="complete" if complete else "incomplete",
        initial_parameters=initial_data,
        runs=rows,
        pairs=pairs,
        group_facts=groups,
        global_errors=errors,
        total_seconds=elapsed,
        observed_process_rss=memory,
        **totals,
        scientific_advantage_verified=False,
        equivalence_verified=False,
        full_R28_complete=False,
    )
