"""Render every already validated 2x2 matrix and original endpoint identity.

Input is a complete declared matrix; output is deterministic exhaustive
Markdown with raw float precision and every unmeasured/failure reason.
No IO, source proof, arithmetic, ranking, selection or experiment here.
"""

from __future__ import annotations

from html import escape
import json
from typing import Any, Iterable

from src.app.continual_confirmation_json import require
from src.app.continual_confirmation_matrix import MATRIX_SCHEMA


def _text(value: Any) -> str:
    raw = "null" if value is None else (repr(value) if type(value) is float else str(value))
    return escape(raw).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    names = tuple(headers)
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    lines.extend("| " + " | ".join(_text(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _cell(slot: dict[str, Any]) -> str:
    return (
        repr(slot["value"])
        if slot["value"] is not None
        else f"null ({slot['status']}: {slot['reason']})"
    )


def _row_text(row: dict[str, Any]) -> str:
    matrix = row["accuracy_matrix"]
    return (
        f"## {_text(row['family'])} / seed {_text(row['seed'])} / {_text(row['arm'])}\n\n"
        f"Role: {_text(row['score_role'])}; original cell failure: {_text(row['original_cell_failure'])}.\n\n"
        + _table(
            ("Stage / task", "A", "B"),
            (
                ("after A", _cell(matrix[0][0]), _cell(matrix[0][1])),
                ("after B", _cell(matrix[1][0]), _cell(matrix[1][1])),
            ),
        )
        + "\n"
        + _table(
            ("Metric", "Value", "Reason"),
            ((name, value["value"], value["reason"]) for name, value in row["metrics"].items()),
        )
        + "\n"
        + _table(
            ("Descriptive transfer", "Value", "Direction / reason"),
            (
                (
                    "A backward: A-after-B minus A-after-A",
                    row["backward_transfer_a"]["value"],
                    row["backward_transfer_a"]["reason"] or row["backward_transfer_a"]["direction"],
                ),
                ("B forward", None, row["forward_transfer_b"]["reason"]),
            ),
        )
        + "\n"
        + _table(
            (
                "Original endpoint",
                "Checkpoint",
                "Task",
                "Correct",
                "Count",
                "Role SHA-256",
                "Failure",
                "Original JSON pointer",
            ),
            (
                (
                    item["record"]["endpoint"],
                    item["record"]["checkpoint"],
                    item["record"]["phase"],
                    item["record"]["result"]["correct_count"],
                    item["record"]["example_count"],
                    item["record"]["role_sha256"],
                    item["record"]["result"]["failure"],
                    item["pointer"],
                )
                for item in row["endpoints"]
            ),
        )
        + "\n"
    )


def render_confirmation_matrix(matrix: dict[str, Any]) -> str:
    require(matrix.get("schema_id") == MATRIX_SCHEMA, "matrix renderer schema differs")
    metadata = {name: value for name, value in matrix.items() if name != "rows"}
    introduction = (
        "# Complete stored confirmation stage/task matrices\n\n"
        "Each stage/task entry is accuracy on task j at the stored stage t. "
        "Every planned family/seed/arm is retained. B-after-A is unmeasured "
        "before task B arrival under the frozen three-endpoint protocol. "
        "No value is inferred and no new final view is opened.\n\n"
        "Signed A forgetting is A-after-A minus A-after-B; negative forgetting "
        "means positive backward transfer on A. Lower forgetting after weak "
        "initial A does not establish better retention. Retention at zero "
        "A-after-A is undefined; ratios above one remain visible. Original "
        "failed-cell metric rules are unchanged; other successful raw endpoints "
        "in that cell remain displayed. Forward transfer is unavailable.\n\n"
        "Backward transfer is a descriptive sign reversal of the original "
        "forgetting value. It adds no primary endpoint or interval family. "
        "Families are not pooled and repeats add no seed replications. "
        "The full seed/paired/interval/cost report remains the source for those "
        "unchanged analyses. Companion confirmation-matrix.result.json preserves "
        "all slot statuses, raw original endpoint records and source/input links.\n\n"
        "## Complete protocol, coverage and provenance\n\n"
        "```json\n" + json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n```\n\n"
    )
    return introduction + "".join(_row_text(row) for row in matrix["rows"])
