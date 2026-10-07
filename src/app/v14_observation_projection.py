"""Project verified v14 result records into observed, seed-level streams.

Inputs are the existing train-only and globally scored JSON objects. Outputs
are deterministic bytes and row counts. This module neither reads files nor
trains, evaluates, selects, or estimates an unrecorded wake metric.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any

from src.app.continual_trigger_replay_outcomes import TRIGGER_REPLAY_OUTCOMES_PROTOCOL
from src.app.continual_trigger_replay_training_study import (
    TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
)


OBSERVATION_PROJECTION_ID = "v14_observation_projection_v1"
SUMMARY_COLUMNS = (
    "seed",
    "arm",
    "method",
    "a_after_a_accuracy",
    "a_after_a_bce",
    "a_after_b_accuracy",
    "a_after_b_bce",
    "b_after_b_accuracy",
    "b_after_b_bce",
    "signed_forgetting",
    "balanced_score",
    "width_after_a",
    "width_after_b",
    "parameters_after_a",
    "parameters_after_b",
    "wake_optimizer_updates",
    "replay_optimizer_updates",
)


@dataclass(frozen=True)
class ObservationProjection:
    """Exact output bytes and data-row counts for one completed v14 bundle."""

    files: dict[str, bytes]
    counts: dict[str, int]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def _require_observed_grid(training: dict[str, Any], outcomes: dict[str, Any]) -> None:
    """Reject a plausible-looking projection source with shuffled epoch cells."""
    if (
        training.get("protocol_id") != TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL
        or outcomes.get("protocol_id") != TRIGGER_REPLAY_OUTCOMES_PROTOCOL
        or training.get("manifest_digest") != outcomes.get("manifest_digest")
        or training.get("resolved_manifest") != outcomes.get("manifest")
    ):
        raise ValueError("v14 observation source protocols or manifest differ")
    try:
        manifest = training["resolved_manifest"]
        cells = [(seed, arm) for seed in manifest["seeds"] for arm in manifest["arms"]]
        train_rows = training["rows"]
        outcome_rows = outcomes["outcomes"]
        if [(row["seed"], row["arm"]) for row in train_rows] != cells or [
            (row["seed"], row["arm"]) for row in outcome_rows
        ] != cells:
            raise ValueError("v14 observation seed/arm cells are missing or reordered")
        phase_epochs = {
            "a": manifest["arrived"]["training"]["phase_a_epochs"],
            "b": manifest["arrived"]["training"]["phase_b_epochs"],
        }
        positions = [
            (phase, epoch, offset + epoch)
            for phase, offset in (("a", 0), ("b", phase_epochs["a"]))
            for epoch in range(1, phase_epochs[phase] + 1)
        ]
        methods = tuple(manifest["arrived"]["training"]["model_order"])
        for row in train_rows:
            opportunities = row["opportunities"]
            if [
                (item["phase"], item["epoch"], item["global_epoch"]) for item in opportunities
            ] != positions:
                raise ValueError("v14 observation epochs are missing or reordered")
            guards = []
            for item in opportunities:
                phase = item["phase"]
                event = item["event"]
                if (
                    item["train_role_hash"]
                    != row[f"phase_{phase}_development_role_hashes"]["train"]
                    or event["completed_epoch"] != item["global_epoch"]
                    or event["wake_batches"] != item["global_epoch"]
                    or event["final_width"] != item["width"]
                    or [work["method"] for work in item["applied_by_method"]] != list(methods)
                ):
                    raise ValueError("v14 observation epoch role, clock, or work differs")
                if event["guard"] is not None:
                    guards.append((phase, item["epoch"], event["guard"]["role_hash"]))
            if [(g["phase"], g["epoch"], g["role_hash"]) for g in row["guard_decisions"]] != guards:
                raise ValueError("v14 observation guard decisions differ from events")
            if any(
                access["role"] == "final_test"
                or (
                    access["role"] == "outer_selection"
                    and access["action"] not in {"source_release", "label_release"}
                )
                for access in row["role_accesses"]
            ):
                raise ValueError("v14 observation train-only roles opened final or outer scores")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("v14 observation source is malformed") from exc


def _summary_csv(final_rows: list[dict[str, Any]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in final_rows:
        method = row["method"]
        writer.writerow(
            {
                **{name: row[name] for name in ("seed", "arm")},
                **{name: method[name] for name in SUMMARY_COLUMNS[2:15]},
                "wake_optimizer_updates": method["wake_work"]["optimizer_updates"],
                "replay_optimizer_updates": method["replay_work"]["optimizer_updates"],
            }
        )
    return output.getvalue().encode("utf-8")


def build_v14_observation_projection(
    training: dict[str, Any], outcomes: dict[str, Any]
) -> ObservationProjection:
    """Expose all saved epoch and final facts with explicit absent wake metrics."""
    _require_observed_grid(training, outcomes)
    rows: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "wake-epochs.jsonl",
            "sleep-events.jsonl",
            "topology.jsonl",
            "replay.jsonl",
            "validation.jsonl",
            "role-access.jsonl",
            "final-results.jsonl",
        )
    }
    for trial in training["rows"]:
        identity = {"seed": trial["seed"], "arm": trial["arm"]}
        for item in trial["opportunities"]:
            epoch = {
                **identity,
                "phase": item["phase"],
                "epoch": item["epoch"],
                "global_epoch": item["global_epoch"],
            }
            rows["wake-epochs.jsonl"].append(
                {
                    **epoch,
                    "train_role_hash": item["train_role_hash"],
                    # Why this: the v14 runner discarded train_epoch return values.
                    "wake_metrics_status": "unavailable_not_recorded",
                }
            )
            rows["sleep-events.jsonl"].append({**epoch, "event": item["event"]})
            rows["topology.jsonl"].append(
                {
                    **epoch,
                    "width": item["width"],
                    "parameter_count": item["parameter_count"],
                    "cumulative_splits": item["cumulative_splits"],
                    "cumulative_prunes": item["cumulative_prunes"],
                    "changes": item["event"]["changes"],
                }
            )
            rows["replay.jsonl"].append(
                {
                    **epoch,
                    "retention": item["retention"],
                    "retained_order_ids": item["retained_order_ids"],
                    "selected_ids": item["selected_ids"],
                    "applied_by_method": item["applied_by_method"],
                    "event_replay": item["event"]["replay"],
                }
            )
        for decision in trial["guard_decisions"]:
            rows["validation.jsonl"].append({**identity, "guard_decision": decision})
        for access in trial["role_accesses"]:
            rows["role-access.jsonl"].append({**identity, "access": access})
    for trial in outcomes["outcomes"]:
        common = {key: value for key, value in trial.items() if key != "methods"}
        for method in trial["methods"]:
            rows["final-results.jsonl"].append({**common, "method": method})
    files = {name: _jsonl(records) for name, records in rows.items()}
    files["summary.csv"] = _summary_csv(rows["final-results.jsonl"])
    counts = {name: len(records) for name, records in rows.items()}
    counts["summary.csv"] = len(rows["final-results.jsonl"])
    return ObservationProjection(files, counts)
