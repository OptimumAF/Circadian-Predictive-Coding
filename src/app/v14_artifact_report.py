"""Derive a descriptive v14 table from already validated run records.

Inputs are a completed P5.1 manifest and its scored outcome object. Outputs
are deterministic JSON/CSV bytes. This module does not read files, train,
score, select a winner, or infer failures outside the published bundle.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import json
from math import isfinite
from statistics import fmean
from typing import Any

from src.app.continual_trigger_replay_outcomes import TRIGGER_REPLAY_OUTCOMES_PROTOCOL


REPORT_SCHEMA_ID = "v14_artifact_report_v1"
BENCHMARK_TRACK = "numpy_synthetic_continual_v14"
METRICS = (
    "balanced_score",
    "signed_forgetting",
    "a_after_b_accuracy",
    "b_after_b_accuracy",
)
_IDENTITY_COLUMNS = (
    "benchmark_track",
    "protocol_id",
    "source_run_id",
    "source_commit_sha",
    "source_dirty",
    "interpretation_scope",
    "arm",
    "method",
    "seed_count",
    "failed_cells_in_bundle",
    "external_attempt_failures",
    "failure_scope",
)


@dataclass(frozen=True)
class V14ArtifactReport:
    """One complete table, available as typed values and exact bytes."""

    summary: dict[str, Any]
    json_bytes: bytes
    csv_bytes: bytes


def _source_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest["source"]
    if source["commit_sha"] is None and not source["unavailable_reason"]:
        raise ValueError("v14 report source commit is unavailable without a reason")
    return {
        "commit_sha": source["commit_sha"],
        "dirty": source["dirty"],
        "workspace_sha256": source["workspace_sha256"],
        "unavailable_reason": source["unavailable_reason"],
    }


def _extract_grid(
    manifest: dict[str, Any], outcomes: dict[str, Any]
) -> tuple[list[int], list[str], list[str], dict[tuple[int, str, str], dict[str, float]]]:
    config = manifest["resolved_config"]
    seeds = config["seeds"]
    arms = config["arms"]
    methods = config["arrived"]["training"]["model_order"]
    if (
        type(seeds) is not list
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or type(arms) is not list
        or not arms
        or any(type(arm) is not str or not arm for arm in arms)
        or len(set(arms)) != len(arms)
        or type(methods) is not list
        or not methods
        or any(type(method) is not str or not method for method in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("v14 report requires distinct declared seeds, arms, and methods")
    rows = outcomes.get("outcomes")
    expected_cells = [(seed, arm) for seed in seeds for arm in arms]
    if (
        type(rows) is not list
        or [(row.get("seed"), row.get("arm")) for row in rows] != expected_cells
    ):
        raise ValueError("v14 report has missing or reordered seed/arm outcome cells")
    values: dict[tuple[int, str, str], dict[str, float]] = {}
    for row in rows:
        method_rows = row.get("methods")
        if type(method_rows) is not list or [item.get("method") for item in method_rows] != methods:
            raise ValueError("v14 report has missing or reordered methods")
        for item in method_rows:
            metrics: dict[str, float] = {}
            for metric in METRICS:
                value = item.get(metric)
                if type(value) not in {int, float} or not isfinite(value):
                    raise ValueError(f"v14 report metric {metric} must be finite")
                metrics[metric] = float(value)
            values[(row["seed"], row["arm"], item["method"])] = metrics
    return seeds, arms, methods, values


def _metric_spread(values: list[float]) -> dict[str, float]:
    low = min(values)
    high = max(values)
    return {"mean": fmean(values), "min": low, "max": high, "range": high - low}


def _render_csv(summary: dict[str, Any]) -> bytes:
    columns = (
        *_IDENTITY_COLUMNS,
        *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "min", "max", "range")),
    )
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in summary["rows"]:
        writer.writerow(
            {
                "benchmark_track": summary["benchmark_track"],
                "protocol_id": summary["protocol_id"],
                "source_run_id": summary["source_run_id"],
                "source_commit_sha": summary["source"]["commit_sha"],
                "source_dirty": summary["source"]["dirty"],
                "interpretation_scope": summary["interpretation_scope"],
                "arm": row["arm"],
                "method": row["method"],
                "seed_count": row["seed_count"],
                "failed_cells_in_bundle": summary["failed_cells_in_bundle"],
                "external_attempt_failures": summary["external_attempt_failures"],
                "failure_scope": summary["failure_scope"],
                **{
                    f"{metric}_{stat}": row[metric][stat]
                    for metric in METRICS
                    for stat in ("mean", "min", "max", "range")
                },
            }
        )
    return output.getvalue().encode("utf-8")


def build_v14_artifact_report(
    manifest: dict[str, Any], outcomes: dict[str, Any]
) -> V14ArtifactReport:
    """Aggregate every declared seed without selecting or retuning cells."""
    try:
        if (
            manifest["status"] != "completed"
            or manifest["protocol_versions"]["outcomes"] != TRIGGER_REPLAY_OUTCOMES_PROTOCOL
            or outcomes["protocol_id"] != TRIGGER_REPLAY_OUTCOMES_PROTOCOL
        ):
            raise ValueError("v14 report requires a completed matching outcome protocol")
        seeds, arms, methods, values = _extract_grid(manifest, outcomes)
        source = _source_identity(manifest)
        rows = [
            {
                "arm": arm,
                "method": method,
                "seed_count": len(seeds),
                **{
                    metric: _metric_spread([values[(seed, arm, method)][metric] for seed in seeds])
                    for metric in METRICS
                },
            }
            for arm in arms
            for method in methods
        ]
        summary = {
            "schema_id": REPORT_SCHEMA_ID,
            "benchmark_track": BENCHMARK_TRACK,
            "protocol_id": TRIGGER_REPLAY_OUTCOMES_PROTOCOL,
            "source_run_id": manifest["run_id"],
            "interpretation_scope": "descriptive_only_no_causal_attribution",
            "source": source,
            "source_file_sha256": {
                name: manifest["files"][name]["sha256"] for name in ("training", "outcomes")
            },
            "seeds": seeds,
            "seed_count": len(seeds),
            "expected_cells": len(seeds) * len(arms) * len(methods),
            "completed_cells": len(values),
            # Why this: the completed bundle has all cells, but cannot tell
            # us whether separate attempted runs failed before publication.
            "failed_cells_in_bundle": 0,
            "external_attempt_failures": "not_recorded_in_bundle",
            "failure_scope": "published_completed_bundle_only",
            "rows": rows,
        }
        return V14ArtifactReport(
            summary,
            (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
            _render_csv(summary),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("v14 report source fields are malformed") from exc
