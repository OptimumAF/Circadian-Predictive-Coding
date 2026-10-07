"""Render every fixed finding and evidence pointer with exact original numbers.

Input is the complete pure synthesis, rebuilt before any presentation. Output
is exhaustive Markdown backed by the full original-input JSON, rather than
another raw-state dump. No filesystem, model, new statistics or selection.
"""

from __future__ import annotations

import json
from typing import Any

from src.app.continual_confirmation_findings_rendering import _statement_row, _text
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_findings_synthesis import build_complete_findings


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *[
            "| " + " | ".join(_text(repr(v) if type(v) in (float, int) else v) for v in row) + " |"
            for row in rows
        ],
        "",
    ]


def _development_tables(body: dict[str, Any]) -> list[str]:
    development = body["original_inputs"]["bodies"]["development"]
    cells = [
        [
            row["family"],
            row["seed"],
            row["arm"],
            json.dumps(row["scores"], sort_keys=True),
            row["result_pointer"],
        ]
        for row in development["cells"]
    ]
    pairs = [
        [
            row["family"],
            row["seed"],
            row["left"],
            row["right"],
            json.dumps(row["differences"], sort_keys=True),
            row["origin"],
        ]
        for row in development["paired_differences"]
    ]
    return [
        "## All fixed development cells and pairs",
        "",
        "These are outer-selection development scores, three seeds per family with shared gating/replay sources. The metric name final_mean_task_accuracy is an endpoint name, not an independent evaluation role. All fixed arms/settings remain; twelve gating/replay pairs are projections of original cells and prospective pairs, 162 are original stored contrasts. No new interval, tuning selection or replication is added. Complete request/result/audit/preflight bodies remain in original_inputs/bodies/development/original_inputs.",
        "",
        *_table(["Family", "Seed", "Arm", "Original scores", "Original pointer"], cells),
        *_table(["Family", "Seed", "Left", "Right", "Original differences", "Origin"], pairs),
    ]


def _confirmation_tables(body: dict[str, Any]) -> list[str]:
    inputs = body["original_inputs"]["bodies"]
    rows = []
    for index, cell in enumerate(body["confirmation_cells"]):
        cost = inputs["outcome_costs"]["rows"][index]
        fields = cost["resource_fields"]
        values = [
            fields[name]["value"]
            for name in (
                "wake_updates",
                "applied_replay_updates",
                "rejected_replay_updates",
                "executed_optimizer_updates",
                "sleep_attempts",
                "parameters_initial",
                "parameters_after_a",
                "parameters_after_b",
                "parameters_peak",
            )
        ]
        rows.append(
            [
                cell["family"],
                cell["seed"],
                cell["arm"],
                json.dumps(cell["metrics"], sort_keys=True),
                cell["failure"],
                *values,
                json.dumps(cell["recorded_transaction_outcomes"], sort_keys=True),
                cell["transaction_scope"],
                index,
            ]
        )
    return [
        "## All independent confirmation cells, activity and costs",
        "",
        "Every metric value/status/reason is retained. The index identifies complete outcome, matrix, activity, resource history and owned/shared retention records in original_inputs/bodies. Null unmeasured wall time, per-arm RSS and isolated guard duration remain null. Recorded/derived optimizer work includes rollback; latent iterations are not CPU or FLOPs. Capacities are named recorded points, including transient peaks, not continuous histories. Named owned-array checkpoints exclude copies and process overhead; shared FIFO bytes belong to one context. Historical whole-process segments remain separate and repeats are not summed or allocated to arms. B-after-A and forward transfer remain unmeasured.",
        "",
        *_table(
            [
                "Family",
                "Seed",
                "Arm",
                "All original metrics",
                "Failure",
                "Wake",
                "Applied replay",
                "Rejected executed replay",
                "Executed total",
                "Attempts",
                "Initial parameters",
                "A parameters",
                "B parameters",
                "Recorded peak",
                "Recorded outcomes",
                "Transaction scope",
                "Complete index",
            ],
            rows,
        ),
        "### Original complete work and named storage totals",
        "",
        "```json",
        json.dumps(
            {
                "work": inputs["outcome_costs"]["work"],
                "separate_named_stage_storage": inputs["outcome_costs"]["stage_storage_totals"],
                "separate_historical_process_segments": inputs["outcome_costs"][
                    "historical_process_segments"
                ],
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        "```",
        "",
    ]


def _activity_tables(body: dict[str, Any]) -> list[str]:
    activity = body["original_inputs"]["bodies"]["activity"]
    rows = [
        [
            row["family"],
            row["seed"],
            row["owner"],
            row["owner_scope"],
            row["phase"],
            row["epoch"],
            row["outcome"],
            row["reason"],
            row["trigger_reason"],
            row["source_pointer"],
            index,
        ]
        for index, row in enumerate(activity["decisions"])
    ]
    offers = [
        [row["family"], row["seed"], row["source_pointer"], index]
        for index, row in enumerate(activity["replay_offers"])
    ]
    return [
        "## Every original accepted, rejected and skipped decision",
        "",
        "All raw events, selector states, proposals, guard signs, rollback work, identities and reasons remain at the complete index in original_inputs/bodies/activity/decisions and the original cost pointer. A schedule decision belongs to one neutral controller with three matched appliers. Recorded attempt counters do not infer unrecorded individual replay commits. Zero attempts/empty proposals do not establish mechanism benefit. Rolled-back transaction counts and rejected executed-update counts are distinct.",
        "",
        *_table(
            [
                "Family",
                "Seed",
                "Owner",
                "Control scope",
                "Phase",
                "Epoch",
                "Outcome",
                "Reason",
                "Trigger",
                "Original pointer",
                "Complete index",
            ],
            rows,
        ),
        "### All original shared replay offers",
        "",
        "Every raw offered/retained/applied replay identity is preserved at the complete index in original_inputs/bodies/activity/replay_offers.",
        "",
        *_table(["Family", "Seed", "Original pointer", "Complete index"], offers),
    ]


def _render_validated_findings(body: dict[str, Any]) -> str:
    inputs = body["original_inputs"]
    primary = inputs["bodies"]["primary"]
    hypotheses = [
        [row["hypothesis_id"], row["status"], row["question"], row["remaining_uncertainty"]]
        for row in body["hypothesis_findings"]
    ]
    lines = [
        "# Complete fixed development and independent confirmation findings",
        "",
        "The complete fixed six-family evidence is retained without choosing a winner, favorable seed, metric or interval. Fifty distinct independent confirmation sources, ten observations per vector; deterministic repeats add no replications. Development/outer-selection findings remain separate. No family pooling, new experiment, score or final-role access occurs.",
        "",
        "All 105 available original simultaneous primary intervals include zero and eleven remain ineligible. H1–H4 are unresolved within these measured settings. This establishes neither equivalence nor broad rejection. Combined-system contrasts supply context and do not isolate mechanisms. Weaker A-after-A can reduce signed forgetting without improving retention. Raw negative signed values are not all regressions: original preferred directions and both A endpoints remain.",
        "",
        "## H1–H4 with complete activity and cost context",
        "",
        *_table(["Hypothesis", "Status", "Question", "Remaining uncertainty"], hypotheses),
        "## Every original primary simultaneous statement",
        "",
        "Original Student-t df9 Bonferroni family of 116 statements, conditional on independent seeds and approximate normality; ten small-role discrete observations do not prove those assumptions. Marginal/secondary intervals cannot replace this family. Exact round-trippable numbers and all 626 vectors/6,260 observations remain in original_inputs/bodies/primary/original_report.",
        "",
        "| Statement | Preferred | Mean | n observed/planned | Sample SD | SE | Simultaneous interval | Original eligibility | Classification | A/A difference | A/B difference | B/B difference |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *[_statement_row(row) for row in primary["primary_statements"]],
        "",
        *_development_tables(body),
        *_confirmation_tables(body),
        *_activity_tables(body),
        "## Preserved operational failures",
        "",
        "The first derivative publication exceeded its original 240-second cap and retained its claim and partial parts. A completed nonterminal audit does not establish publication success. Killed-child terminal reader/guard counters are unobserved, not zero. Later complete operations have separate records. Failed source/correctness checks and exact producer/source snapshots remain. These are operational records, not new scientific replications or scientific regression measurements.",
        "",
        "```json",
        json.dumps(body["operational_failures"], sort_keys=True, indent=2, allow_nan=False),
        "```",
        "",
        "## Complete evidence identities and preservation",
        "",
        "The companion result JSON preserves all whole input bodies and all raw UTF-8 handoffs, failed partial parts and snapshots. All original settings, seeds, metrics, roles, contrasts, nulls, failures and historical flags remain; the complete indexes/pointers above locate them. This pure reconstruction grants no fresh official-reader/current-source authority. P6.12b3 current publication and independent readbacks and the unchanged original parent audit remain required.",
        "",
        "```json",
        json.dumps(inputs["catalog"], sort_keys=True, indent=2, allow_nan=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def render_complete_findings(body: dict[str, Any]) -> str:
    """Reject any changed raw evidence or interpretation before rendering."""
    require(
        type(body) is dict and "original_inputs" in body,
        "complete findings presentation input differs",
    )
    same_json(
        body,
        build_complete_findings(body["original_inputs"]),
        "whole complete findings presentation",
    )
    return _render_validated_findings(body)
