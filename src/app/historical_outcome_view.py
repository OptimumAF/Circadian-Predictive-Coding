"""Pure historical presentation projection and unclipped plotting extents.

Input is an already accepted P6.10 body; output copies its ordered observations,
original analysis and scoped costs. This module has no IO, source admission,
scientific reader authority, new estimator, ranking or experiment execution.
The outer exporter must bind the complete original byte identity before use.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Iterable


SCHEMA_ID = "p610_complete_original_outcomes_against_scoped_costs_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("historical presentation: " + message)


def _number(value: Any) -> None:
    _require(type(value) in (int, float) and math.isfinite(value), "nonfinite/nonnumeric value")


def numeric_extent(values: Iterable[float | int | None]) -> tuple[float, float]:
    """Include every observation and interval endpoint; never clamp to [0, 1]."""
    known = [value for value in values if value is not None]
    _require(bool(known), "no measured value for numeric axis")
    for value in known:
        _number(value)
    low, high = min(known), max(known)
    # Why this: constant-zero controls need a visible axis and stable padding.
    padding = 0.06 * (high - low) if high > low else max(abs(low) * 0.06, 0.06)
    return low - padding, high + padding


def _validate_order(body: dict[str, Any]) -> None:
    families = body["original_report_records"]["analysis_contract"]["families"]
    expected = [
        (family["name"], seed, arm)
        for family in families
        for seed in family["seeds"]
        for arm in family["arms"]
    ]
    actual = [(row["family"], row["seed"], row["arm"]) for row in body["rows"]]
    _require(len(set(expected)) == len(expected), "duplicate declared cell")
    _require(actual == expected, "cell order/completeness differs from original contract")
    _require(len(actual) == body["coverage"]["cells"], "cell coverage differs")
    metrics = [m for row in body["rows"] for m in row["metrics"].values()]
    _require(len(metrics) == body["coverage"]["cell_metrics"], "metric coverage differs")
    for metric in metrics:
        if metric["value"] is None:
            _require(bool(metric["reason"]), "missing metric has no original reason")
        else:
            _number(metric["value"])


def _validate_analysis(body: dict[str, Any]) -> None:
    rows = {(r["family"], r["seed"], r["arm"]): r for r in body["rows"]}
    records = body["original_report_records"]
    families = records["analysis"]["families"]
    contracts = records["analysis_contract"]["families"]
    _require(
        [f["name"] for f in families] == [f["name"] for f in contracts],
        "analysis family order differs",
    )
    vectors, primary = 0, 0
    for family, contract in zip(families, contracts, strict=True):
        _require(
            [a["name"] for a in family["arms"]] == contract["arms"], "analysis arm order differs"
        )
        for arm in family["arms"]:
            expected_metrics = rows[(family["name"], contract["seeds"][0], arm["name"])]["metrics"]
            _require(
                {m["metric"] for m in arm["metrics"]} == set(expected_metrics),
                "analysis metrics differ",
            )
            for metric in arm["metrics"]:
                observed = metric["summary"]["observations"]
                expected = [
                    {
                        "seed": seed,
                        **rows[(family["name"], seed, arm["name"])]["metrics"][metric["metric"]],
                    }
                    for seed in contract["seeds"]
                ]
                _require(observed == expected, "analysis observation differs from original cell")
        for comparison in (*family["arms"], *family["contrasts"]):
            vectors += len(comparison["metrics"])
            for metric in comparison["metrics"]:
                summary = metric["summary"]
                _require(
                    [o["seed"] for o in summary["observations"]] == contract["seeds"],
                    "analysis seed order differs",
                )
                for name in ("marginal_interval", "simultaneous_interval"):
                    interval = summary[name]
                    if interval is not None:
                        _number(interval["lower"])
                        _number(interval["upper"])
                        _require(
                            interval["lower"] <= interval["upper"], "inverted original interval"
                        )
        primary += sum(m["primary_endpoint"] for c in family["contrasts"] for m in c["metrics"])
    _require(vectors == body["coverage"]["metric_vectors"], "analysis vector coverage differs")
    _require(
        primary == body["coverage"]["primary_contrast_statements"],
        "primary statement coverage differs",
    )


def build_historical_view(body: dict[str, Any]) -> dict[str, Any]:
    """Copy complete plotted data; large original state proofs stay in pinned input."""
    _require(body.get("schema_id") == SCHEMA_ID, "unsupported original schema")
    _validate_order(body)
    _validate_analysis(body)
    names = (
        "coverage",
        "original_report_records",
        "stage_storage_totals",
        "historical_process_segments",
        "presentation_scope",
        "original_inventory_gaps",
        "work",
    )
    view = {name: deepcopy(body[name]) for name in names}
    view["schema_id"] = "historical_outcome_cost_figure_view_v1"
    view["rows"] = [
        {
            **{
                name: deepcopy(row[name])
                for name in ("family", "seed", "arm", "outcome", "metrics", "resource_fields")
            },
            "owned_retention": {
                "checkpoints": {
                    stage: {
                        name: deepcopy(value)
                        for name, value in checkpoint.items()
                        if name in ("owned_array_bytes", "retention_status")
                    }
                    for stage, checkpoint in row["owned_retention"]["checkpoints"].items()
                }
            },
        }
        for row in body["rows"]
    ]
    view["contexts"] = [
        {
            "family": c["family"],
            "seed": c["seed"],
            "retention": {"stages": deepcopy(c["retention"]["stages"])},
        }
        for c in body["contexts"]
    ]
    return view
