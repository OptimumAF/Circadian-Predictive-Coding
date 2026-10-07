"""Render all pure findings with complete original evidence and explicit limits.

Input is a complete re-derived findings body; output is deterministic Markdown.
No file, model, source, scoring, selection or publication belongs here. Whole
reconstruction rejects changed statements or narratives before rendering.
"""

from __future__ import annotations

import json
from typing import Any

from src.app.continual_confirmation_findings import _derive_findings
from src.app.continual_confirmation_json import require, same_json


def _text(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _interval(summary: dict[str, Any]) -> str:
    interval = summary["simultaneous_interval"]
    return "None" if interval is None else f"[{interval['lower']!r}, {interval['upper']!r}]"


def _statement_row(statement: dict[str, Any]) -> str:
    summary = statement["summary"]
    endpoint_means = [repr(m["summary"]["mean"]) for m in statement["secondary_endpoint_summaries"]]
    columns = [
        statement["statement_id"],
        statement["preferred_direction"],
        repr(summary["mean"]),
        f"{summary['observed_seed_count']}/{summary['planned_seed_count']}",
        repr(summary["observed_sample_standard_deviation"]),
        repr(summary["standard_error"]),
        _interval(summary),
        summary["interval_status"],
        statement["classification"],
        *endpoint_means,
    ]
    return "| " + " | ".join(_text(value) for value in columns) + " |"


def _hypothesis_rows(body: dict[str, Any]) -> list[str]:
    return [
        f"| {item['hypothesis_id']} | {item['status']} | {_text(item['title'])} | {_text(item['remaining_uncertainty'])} |"
        for item in body["hypothesis_conclusions"]
    ]


def render_confirmation_findings(body: dict[str, Any]) -> str:
    """Verify the whole declaration, including every interpretation, before display."""
    require(type(body) is dict and "original_report" in body, "findings presentation input differs")
    rebuilt = _derive_findings(body["original_report"])
    same_json(body, rebuilt, "complete findings declarations")
    coverage = body["coverage"]["original_report"]
    classifications = body["coverage"]["statement_classifications"]
    lines = [
        "# P6.12a complete primary confirmation evidence",
        "",
        "Independent confirmation only: all 560 cells, 626 vectors, 6,260 seed observations, 58 ordered contrasts and 116 simultaneous statements. Ten source seeds per vector; 50 distinct source seeds overall. Families are not pooled and deterministic repeats add no replications.",
        "",
        "All differences are left minus right in accuracy fractions. Higher final mean task accuracy and lower signed forgetting are the original preferred directions. The three secondary endpoint columns are the original mean differences A/A, A/B and B/B; weaker A/A can reduce forgetting without improving retention.",
        "",
        "Intervals use the original model-based Student-t df9 Bonferroni family of 116, conditional on independent source seeds and approximately normal seed outcomes/differences. Ten discrete small-role observations do not prove those assumptions. Crossing zero is unresolved, not equivalence or broad rejection. Ineligible intervals retain their original status and nulls. Marginal and secondary intervals cannot replace primary simultaneous intervals.",
        "",
        f"Primary classification counts: {json.dumps(classifications, sort_keys=True)}.",
        f"Original failed cells: {coverage['failed_cells']}; null seed observations: {coverage['null_seed_observations']}; raw negative seed observations across all vectors: {coverage['negative_seed_observations']}. Raw negative values are not all regressions: the preferred direction matters.",
        "",
        "## H1–H4 within the primary confirmation evidence",
        "",
        "These conclusions remain unresolved within this ledger. Full-system contrasts are context, not isolated mechanism proof. No hypothesis vote or model winner is computed. P6.12b must integrate complete development/tuning, activity, costs and operational failures, then establish complete current IO publication/readbacks; the original P6.12 acceptance remains unfinished.",
        "",
        "| Hypothesis | Status | Question | Remaining uncertainty |",
        "| --- | --- | --- | --- |",
        *_hypothesis_rows(body),
        "",
        "## Every original primary statement",
        "",
        "Numbers use Python's exact round-trippable representation without metric rounding. The complete original report below retains every seed, raw observation, marginal/secondary interval, eligibility reason, endpoint, role, failure and cost reference.",
        "",
        "| Statement | Preferred | Mean | n observed/planned | Sample SD | SE | Simultaneous interval | Original eligibility | Classification | A/A difference | A/B difference | B/B difference |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *[_statement_row(statement) for statement in body["primary_statements"]],
        "",
        "## Complete original report",
        "",
        "Pure reconstruction validates the entire stored declaration. It grants no fresh official-reader, current-source or original execution authority. Report identity: "
        + json.dumps(body["report_identity"], sort_keys=True)
        + ".",
        "",
        "```json",
        json.dumps(
            body["original_report"], indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        ),
        "```",
        "",
    ]
    return "\n".join(lines)
