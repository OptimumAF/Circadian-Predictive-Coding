"""Render every inventory row and field scope without ranking outcomes.

Input is a validated inventory. Output is deterministic Markdown; complete
history points and exact pointers stay in companion JSON. No IO, arithmetic,
source proof, runtime measurement, selection, chart or model ranking.
"""

from __future__ import annotations

from html import escape
import json
from typing import Any

from src.app.continual_confirmation_json import require
from src.app.continual_confirmation_resources import RESOURCE_SCHEMA


def _text(value: Any) -> str:
    return escape(str(value)).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def render_resource_inventory(body: dict[str, Any]) -> str:
    require(body.get("schema_id") == RESOURCE_SCHEMA, "resource renderer schema differs")
    metadata = {name: value for name, value in body.items() if name not in {"rows", "contexts"}}
    lines = [
        "# Complete original confirmation resource inventory",
        "",
        "Every original arm/seed, checkpoint capacity and shared context is retained. "
        "Units and measurement scopes are explicit. Per-arm wall/RSS and isolated "
        "sleep/guard durations remain unmeasured. Formula-based loop counts are "
        "derived; rejected replay still consumes execution. No composite winner.",
        "",
        "Complete parameter history points and exact source pointers remain in "
        "resource-inventory.result.json. Shared FIFO occupancy and owned model "
        "arrays before copies are distinct from whole-process sampled RSS. "
        "Repeated runs add neither seed replication nor work to the original total.",
        "",
        "## Contract, run scopes, gaps and provenance",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
        "```",
        "",
    ]
    for context in body["contexts"]:
        lines.extend(
            [
                f"## Context {_text(context['family'])} / seed {context['seed']}",
                "",
                "```json",
                json.dumps(context, indent=2, sort_keys=True, allow_nan=False),
                "```",
                "",
            ]
        )
    for row in body["rows"]:
        lines.extend(
            [
                f"## {_text(row['family'])} / seed {row['seed']} / {_text(row['arm'])}",
                "",
                "| Field | Value | Unit | Status | Scope / reason |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for name in sorted(row["fields"]):
            field = row["fields"][name]
            value = (
                f"{len(field['value'])} recorded points; see complete JSON"
                if name == "parameter_history"
                else field["value"]
            )
            lines.append(
                "| "
                + " | ".join(
                    _text(item)
                    for item in (
                        name,
                        value,
                        field["unit"],
                        field["status"],
                        field["scope"] + ": " + field["reason"],
                    )
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
