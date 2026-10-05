"""Render every retention stage and original proof link without a winner.

Input is the complete derived retention ledger. Output is deterministic
Markdown; all original array/state/role proofs remain in companion JSON.
No IO, resource measurement, scoring or statistical interpretation.
"""

from __future__ import annotations

from typing import Any

from src.app.continual_confirmation_retention_checkpoints import STAGES


def render_retention_costs(body: dict[str, Any]) -> str:
    lines = [
        "# Original checkpoint retention costs\n",
        "Owned input/target array bytes and one shared FIFO per context are separate. "
        "Stages are separate checkpoints; do not sum them as live RAM. "
        "Checkpoint copies, Python overhead and per-arm RSS are unmeasured. "
        "Nullable original views are preserved; derived zeroes require closed state proof.\n",
        "Original sorted content IDs and ordered array fingerprints are preserved. "
        "No sample values are reopened or content IDs rehashed. Full state/role/source "
        "pointers and fingerprints are in the complete JSON.\n",
        "## Every original arm and checkpoint\n",
        "| Family | Seed | Arm | Stage | Original retention status | Owned array bytes | Examples | Original retention pointer | State identity |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in body["rows"]:
        for stage in STAGES:
            point = row["checkpoints"][stage]
            examples = 0 if point["retention"] is None else point["retention"]["example_count"]
            lines.append(
                f"| {row['family']} | {row['seed']} | {row['arm']} | {stage} | {point['retention_status']} | "
                f"{point['owned_array_bytes']} | {examples} | {point['retention_json_pointer']} | {point['state_sha256']} |"
            )
    lines.extend(
        [
            "\n## Every shared context and stage\n",
            "| Family | Seed | Stage | Owned arrays | One shared FIFO | Total before copies | Shared status/reason |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for context in body["contexts"]:
        for stage in STAGES:
            values = context["stages"][stage]
            shared = values["shared_fifo"]
            lines.append(
                f"| {context['family']} | {context['seed']} | {stage} | {values['owned_array_bytes']} | "
                f"{values['shared_fifo_array_bytes']} | {values['owned_plus_shared_array_bytes_before_copies']} | "
                f"{shared['measurement_status']}: {shared['reason']} |"
            )
    lines.extend(
        [
            "\n## Separate stage totals\n",
            "| Stage | Owned arrays | Shared FIFO arrays | Total before copies |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for stage in STAGES:
        values = body["stage_totals"][stage]
        lines.append(
            f"| {stage} | {values['owned_array_bytes']} | {values['shared_fifo_array_bytes']} | {values['owned_plus_shared_array_bytes_before_copies']} |"
        )
    lines.append("\nP6.10 outcome presentation and original parent acceptance remain unfinished.\n")
    return "\n".join(lines)
