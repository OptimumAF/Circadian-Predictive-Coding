"""Render an already verified exhaustive report as deterministic Markdown.

Every retained cell/vector/seed/interval/status is displayed in frozen order.
JSON carries unabridged precision/cost fields; large shared context is linked
by exact identity. This presenter owns no arithmetic, source/file validation,
model/data, ranking, selection, IO or experiment/resource measurement.
"""

from __future__ import annotations

from html import escape
import json
from typing import Any, Iterable

from src.app.continual_confirmation_report import METRICS, REPORT_SCHEMA
from src.app.continual_confirmation_json import require


def _text(value: Any) -> str:
    if value is None:
        return "null"
    raw = repr(value) if type(value) is float else str(value)
    return escape(raw).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    names = tuple(headers)
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    lines.extend("| " + " | ".join(_text(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _json(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n```\n"


def _vector(metric: dict[str, Any]) -> str:
    summary = metric["summary"]
    interval_rows = []
    for kind in ("marginal", "simultaneous"):
        interval = summary[kind + "_interval"]
        interval_rows.append(
            (
                kind,
                *(
                    interval.get(k) if interval else None
                    for k in ("lower", "upper", "confidence_level", "statement_count")
                ),
            )
        )
    return (
        f"#### {_text(metric['metric'])}\n\n"
        f"Primary: {_text(metric['primary_endpoint'])}; preferred direction: {_text(metric['preferred_direction'])}; units: {_text(metric['units'])}.\n\n"
        + _table(
            ("Seed", "Value", "Null reason"),
            ((r["seed"], r["value"], r["reason"]) for r in summary["observations"]),
        )
        + "\n"
        + _table(
            (
                "Planned N",
                "Observed N",
                "Planned mean",
                "Observed mean",
                "Observed sample SD",
                "SE",
                "Observed min",
                "Observed max",
            ),
            (
                (
                    summary[k]
                    for k in (
                        "planned_seed_count",
                        "observed_seed_count",
                        "mean",
                        "observed_mean",
                        "observed_sample_standard_deviation",
                        "standard_error",
                        "observed_minimum",
                        "observed_maximum",
                    )
                ),
            ),
        )
        + f"\nInterval status: {_text(summary['interval_status'])}.\n\n"
        + _table(
            ("Interval", "Raw lower", "Raw upper", "Confidence", "Statement count"), interval_rows
        )
    )


def _family(family: dict[str, Any]) -> str:
    parts = [f"## Family: {_text(family['name'])}\n"]
    for arm in family["arms"]:
        parts.append(f"### Arm: {_text(arm['name'])}\n")
        parts.extend(_vector(metric) for metric in arm["metrics"])
    for pair in family["contrasts"]:
        parts.append(f"### Contrast: {_text(pair['left'])} minus {_text(pair['right'])}\n")
        parts.extend(_vector(metric) for metric in pair["metrics"])
    return "\n".join(parts)


def _cell_rows(report: dict[str, Any]) -> str:
    rows = []
    for joined in report["joined_cells"]:
        outcome, values = joined["outcome"], joined["metrics"]
        rows.append(
            (
                outcome["family"],
                outcome["seed"],
                outcome["arm"],
                *(values[m]["value"] for m in METRICS),
                outcome["failure"],
                values["retention_ratio_a"]["reason"],
            )
        )
    return _table(("Family", "Seed", "Arm", *METRICS, "Failure", "Retention null reason"), rows)


def _cost_rows(report: dict[str, Any]) -> str:
    rows = []
    for joined in report["joined_cells"]:
        cost = joined["cost"]
        capacity = [f"{c['width']} / {c['parameter_count']}" for c in cost["checkpoints"]]
        rows.append(
            (
                cost["family"],
                cost["seed"],
                cost["arm"],
                cost["wake_updates"],
                cost["applied_replay_updates"],
                cost["rejected_executed_replay_updates"],
                cost["executed_optimizer_updates"],
                *capacity,
            )
        )
    return _table(
        (
            "Family",
            "Seed",
            "Arm",
            "Wake",
            "Applied replay",
            "Rejected executed replay",
            "All executed",
            "Initial width / parameters",
            "After-A width / parameters",
            "After-B width / parameters",
        ),
        rows,
    )


def render_confirmation_report(report: dict[str, Any]) -> str:
    """Render complete supplied facts; independent artifact readers prove them."""
    require(report.get("schema_id") == REPORT_SCHEMA, "report renderer schema differs")
    parts = [
        "# Complete independent confirmation: every seed, contrast and original cost\n",
        "All values use the frozen ten-seed contract. The deterministic repeat is a reproducibility check. Families are analyzed separately. Differences are left minus right: higher mean-task accuracy and lower signed forgetting are the declared directions. Lower forgetting after weak initial A is not evidence of strong retention.\n",
        "Intervals retain raw, unclipped endpoints under the predeclared model assumptions. All 116 primary contrasts form one Bonferroni family. Nulls/failures remain with their original seeds; missing and zero-dispersion vectors have no eligible interval. Retention is descriptive and is null for zero A-after-A. No comparison, seed, interval or result is ranked or selected.\n",
        "## Contract, replication and complete coverage\n",
        _json(
            {
                "contract": report["analysis_contract"],
                "replication": report["replication"],
                "coverage": report["coverage"],
                "repetition": report["analysis_repetition"],
            }
        ),
        "## All individual cells and derived metrics\n",
        _cell_rows(report),
        "## Original per-arm work and actual checkpoint capacity\n",
        "Executed work includes rolled-back replay. Width/parameter counts are observed checkpoints; raw method facts retain peak/history and each original family's cost units in the companion JSON. Shared proof/storage context is preserved in the referenced complete cost artifacts. Wall/RSS observations describe whole original runs; no per-arm time/RSS/FLOP estimate is created.\n",
        _cost_rows(report),
        "## Complete cost context and original work\n",
        _json(report["cost_reference"]),
    ]
    if "provenance" in report:
        parts.extend(
            ("## Complete input/source/artifact provenance\n", _json(report["provenance"]))
        )
    if "scored_run_facts" in report:
        parts.extend(
            (
                "## Both original scored runs and observed resources\n",
                _json(report["scored_run_facts"]),
            )
        )
    for family in report["analysis"]["families"]:
        parts.append(_family(family))
    parts.extend(
        (
            "## Unabridged outcome and cost fields\n",
            "The companion `confirmation-report.result.json` preserves every raw role/endpoint/cell, original method cost field, failure and exact numeric value. Complete shared guard/proof context is bound by the file identities and JSON pointers above.\n",
        )
    )
    return "\n".join(parts) + "\n"
