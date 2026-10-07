"""Serialize and validate measured v14 wake rows without scoring data.

Inputs are the opt-in train-only study or a saved completed-run payload.
Outputs are canonical diagnostic JSONL and a measured projection. This
module does not train, read files, release roles, or choose an outcome.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, fields
from io import StringIO
import json
from math import isfinite
from typing import Any

from src.app.continual_trigger_replay_training_study import (
    TriggerReplayTrainingStudy,
    preflight_trigger_replay_training_study,
)
from src.app.v14_observation_projection import (
    ObservationProjection,
    build_v14_observation_projection,
)
from src.app.wake_diagnostic import WakeDiagnostic, metric_spec, validate_wake_diagnostic_trials


WAKE_DIAGNOSTIC_ID = "v14_wake_diagnostics_v1"
MEASURED_PROJECTION_ID = "v14_measured_observation_projection_v1"
WAKE_COLUMNS = tuple(item.name for item in fields(WakeDiagnostic))


def _diagnostic_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def serialize_wake_diagnostic_study(study: TriggerReplayTrainingStudy) -> bytes:
    """Capture only values returned by the successful train-only updates."""
    preflight_trigger_replay_training_study(study)
    validate_wake_diagnostic_trials(study.trials)
    return _diagnostic_jsonl(
        [asdict(row) for trial in study.trials for row in trial.wake_diagnostics]
    )


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"wake diagnostic contains nonfinite JSON token: {token}")


def validate_wake_diagnostic_bytes(
    training: dict[str, Any], payload: bytes
) -> list[dict[str, Any]]:
    """Validate saved metric rows against the raw train-only update grid."""
    try:
        rows = [json.loads(line, parse_constant=_reject_nonfinite) for line in payload.splitlines()]
        if any(not isinstance(row, dict) or set(row) != set(WAKE_COLUMNS) for row in rows):
            raise ValueError("wake diagnostic row fields differ")
        model_order = training["resolved_manifest"]["arrived"]["training"]["model_order"]
        for trial in training["rows"]:
            work = trial["wake_work"]
            if [item["method"] for item in work] != model_order or any(
                item["optimizer_updates"] != len(trial["opportunities"]) for item in work
            ):
                raise ValueError("wake diagnostic matched work differs")
        expected = [
            (
                trial["seed"],
                trial["arm"],
                item["phase"],
                item["epoch"],
                item["global_epoch"],
                method["method"],
                item["train_role_hash"],
            )
            for trial in training["rows"]
            for item in trial["opportunities"]
            for method in trial["wake_work"]
        ]
        if len(rows) != len(expected):
            raise ValueError("wake diagnostic row count differs from train-only work")
        for row, identity in zip(rows, expected, strict=True):
            if (
                tuple(row[key] for key in WAKE_COLUMNS[:7]) != identity
                or tuple(
                    row[key] for key in ("metric_name", "metric_definition", "measurement_stage")
                )
                != metric_spec(row["method"])
                or type(row["metric_value"]) is not float
                or not isfinite(row["metric_value"])
            ):
                raise ValueError("wake diagnostic identity, definition, or value differs")
        if _diagnostic_jsonl(rows) != payload:
            raise ValueError("wake diagnostic JSONL is not canonical")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("wake diagnostic JSONL is malformed") from exc
    return rows


def _diagnostic_csv(rows: list[dict[str, Any]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=WAKE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def build_measured_observation_projection(
    training: dict[str, Any], outcomes: dict[str, Any], diagnostics: bytes
) -> ObservationProjection:
    """Add genuine measured rows while retaining every v14 source view."""
    base = build_v14_observation_projection(training, outcomes)
    rows = validate_wake_diagnostic_bytes(training, diagnostics)
    files = {
        **base.files,
        "wake-metrics.jsonl": diagnostics,
        "wake-metrics.csv": _diagnostic_csv(rows),
    }
    return ObservationProjection(
        files, {**base.counts, "wake-metrics.jsonl": len(rows), "wake-metrics.csv": len(rows)}
    )
