"""Render all original outcomes alongside compute and memory, with scopes.

Input is a complete validated presentation. Output is deterministic Markdown.
Full raw cost/endpoint/array/state/history/interval records stay in companion
JSON. No IO, new arithmetic, scientific execution, selection or ranking.
"""

from __future__ import annotations

from html import escape
import json
from typing import Any

from src.app.continual_confirmation_json import require
from src.app.continual_confirmation_outcome_costs import SCHEMA_ID


def _text(value: Any) -> str:
    return escape(str(value)).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def _metric(row: dict[str, Any], name: str) -> str:
    value = row["metrics"][name]
    return (
        str(value["value"]) if value["value"] is not None else "null (" + str(value["reason"]) + ")"
    )


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(_text(v) for v in row) + " |" for row in rows],
        "",
    ]


def _identity_and_metrics(row: dict[str, Any]) -> list[Any]:
    return [
        row["seed"],
        row["arm"],
        row["outcome"]["failure"],
        *[
            _metric(row, name)
            for name in ("final_mean_task_accuracy", "signed_forgetting_a", "retention_ratio_a")
        ],
    ]


def _compute(rows: list[dict[str, Any]]) -> list[str]:
    names = [
        "wake_updates",
        "optimizer_latent_iterations",
        "applied_replay_updates",
        "rejected_replay_updates",
        "executed_optimizer_updates",
        "applied_replay_presentations",
        "distinct_applied_replay_examples",
        "sleep_attempts",
        "guard_prediction_calls",
    ]
    values = [
        _identity_and_metrics(row) + [row["resource_fields"][name]["value"] for name in names]
        for row in rows
    ]
    return _table(
        [
            "Seed",
            "Arm",
            "Failure",
            "Final mean accuracy",
            "Signed A forgetting",
            "A retention",
            "Wake updates",
            "Optimizer latent loops",
            "Applied replay updates",
            "Rejected executed replay",
            "Executed updates",
            "Applied replay presentations",
            "Distinct applied replay IDs",
            "Sleep attempts",
            "Guard prediction calls",
        ],
        values,
    )


def _memory(rows: list[dict[str, Any]]) -> list[str]:
    values = []
    for row in rows:
        fields, owned = row["resource_fields"], row["owned_retention"]["checkpoints"]
        values.append(
            _identity_and_metrics(row)
            + [
                fields[name]["value"]
                for name in (
                    "parameters_initial",
                    "parameters_after_a",
                    "parameters_after_b",
                    "parameters_peak",
                )
            ]
            + [
                f"{owned[stage]['owned_array_bytes']} ({owned[stage]['retention_status']})"
                for stage in ("initial", "after_a", "after_b")
            ]
            + [len(fields["parameter_history"]["value"])]
        )
    return _table(
        [
            "Seed",
            "Arm",
            "Failure",
            "Final mean accuracy",
            "Signed A forgetting",
            "A retention",
            "Initial parameters",
            "After-A parameters",
            "After-B parameters",
            "Peak recorded/transient parameters",
            "Initial owned array bytes / view",
            "After-A owned array bytes / view",
            "After-B owned array bytes / view",
            "History points (all in JSON)",
        ],
        values,
    )


def render_outcome_cost_presentation(body: dict[str, Any]) -> str:
    require(body.get("schema_id") == SCHEMA_ID, "outcome-cost renderer schema differs")
    require(
        len(body["rows"]) == 560 and len(body["contexts"]) == 60,
        "outcome-cost renderer incomplete scope",
    )
    lines = [
        "# All original confirmation outcomes against compute and memory",
        "",
        "Every original family/seed/arm is retained in manifest order. Accuracy and signed forgetting are shown separately against recorded/derived compute and owned array storage/capacity. Nulls retain reasons; negative forgetting and above-one retention remain. No composite winner or selected subset.",
        "",
        "Optimizer loops are formula-derived work including rolled-back replay; prediction/guard work is separately scoped. Owned arrays exclude shared FIFO, Python overhead, checkpoint copies and process RSS. Different stage snapshots are not additive live RAM. Per-arm wall time/RSS/guard duration remain unmeasured.",
        "",
        "The companion outcome-costs.result.json retains all six original cell metrics, raw outcomes/costs/endpoints, all owned array/state/role proofs, every history point, all 626 original metric vectors and 116 primary interval statements. Original paired/seed/interval rules are unchanged; repeats add no replications or work.",
        "",
        "## Coverage, original contracts and provenance",
        "",
        "```json",
    ]
    metadata = {
        name: value
        for name, value in body.items()
        if name
        not in {"rows", "contexts", "original_report_records", "historical_process_segments"}
    }
    lines.extend(
        [
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
            "## Original whole process segments",
            "",
            "These four historical train/scored segments remain distinct, including repetitions. Their wall times and sampled absolute RSS are not per-arm measurements or model-only storage.",
            "",
            "```json",
            json.dumps(
                body["historical_process_segments"], indent=2, sort_keys=True, allow_nan=False
            ),
            "```",
            "",
            "## Shared contexts and separate stage storage",
            "",
        ]
    )
    for context in body["contexts"]:
        stages = context["retention"]["stages"]
        lines.extend(
            [
                f"### {_text(context['family'])} / seed {context['seed']}",
                "",
                "One original shared FIFO; no allocation across arms. Before copies, separate stages:",
                "",
            ]
        )
        lines.extend(
            _table(
                [
                    "Stage",
                    "Owned arrays across this context",
                    "Shared FIFO arrays",
                    "Total before copies",
                    "Shared status / reason",
                ],
                [
                    [
                        stage,
                        stages[stage]["owned_array_bytes"],
                        stages[stage]["shared_fifo_array_bytes"],
                        stages[stage]["owned_plus_shared_array_bytes_before_copies"],
                        stages[stage]["shared_fifo"]["measurement_status"]
                        + ": "
                        + stages[stage]["shared_fifo"]["reason"],
                    ]
                    for stage in ("initial", "after_a", "after_b")
                ],
            )
        )
    for family in dict.fromkeys(row["family"] for row in body["rows"]):
        rows = [row for row in body["rows"] if row["family"] == family]
        lines.extend([f"## {_text(family)}: every original outcome against compute", ""])
        lines.extend(_compute(rows))
        lines.extend(
            [
                f"## {_text(family)}: every original outcome against memory and capacity",
                "",
                "Owned byte figures preserve nullable view status. Exact initial/A/B views, array fingerprints, roles, original checkpoint hashes and full histories remain in JSON.",
                "",
            ]
        )
        lines.extend(_memory(rows))
    return "\n".join(lines) + "\n"
