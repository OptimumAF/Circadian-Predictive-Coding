"""Project saved memory-demo evidence; no learner, IO or metric recomputation.

Inputs are bounded public CPU/native JSON reports and optional unverified costs.
Outputs are detached source cells for the existing inspector table renderer.
"""

from copy import deepcopy
from typing import Any, cast

# Why this: keep the existing source-cell contract and bounds without a new renderer.
from src.app.coherent_stream_inspector_projection import _cell, _envelope, _fields, _label, _require

_FAMILIES = {
    "taskspace_memory_demo_v1": "offline_recorded_replay",
    "taskspace_memory_native_v1": "native_fresh_rollout",
}
_SCENES = ("pick19", "pick17", "pick20", "push17")


def _records(report: dict[str, Any], path: str, maximum: int) -> list[dict[str, Any]]:
    node: Any = report
    for key in path.split("/"):
        if type(node) is not dict or key not in node:
            return []
        node = node[key]
    _require(type(node) is list and len(node) <= maximum, f"invalid {path}")
    _require(all(type(row) is dict for row in node), f"invalid {path} records")
    return node


def _section(identity: str, title: str, note: str, columns: list[str]) -> dict[str, Any]:
    return dict(id=identity, title=title, note=note, columns=columns, rows=[])


def _rows(
    source: dict[str, Any],
    path: str,
    fields: tuple[str, ...],
    limit: int,
    identity: str,
    title: str,
    note: str,
) -> dict[str, Any]:
    section = _section(identity, title, note, list(fields))
    for index, _ in enumerate(_records(source, path, limit)):
        section["rows"].append(
            dict(cells=[_cell(source, f"/{path}/{index}/{key}") for key in fields])
        )
    return section


def _details(
    source: dict[str, Any],
    paths: tuple[str, ...],
    identity: str,
    title: str,
    note: str,
    origin: str = "report",
) -> dict[str, Any]:
    section = _section(identity, title, note, ["Recorded field", "Value"])
    for path in paths:
        section["rows"].append(dict(cells=[_label(path), _cell(source, "/" + path, origin)]))
    return section


def _typed(record: dict[str, Any], counts: tuple[str, ...], flags: tuple[str, ...]) -> None:
    _fields(record, counts, count=True)
    for key in flags:
        if key in record and record[key] is not None:
            _require(type(record[key]) is bool, f"invalid boolean {key}")


def _validate(report: dict[str, Any]) -> None:
    family = report.get("definition")
    _require(type(family) is str and family in _FAMILIES, "unsupported memory report")
    _require(report.get("mode") == _FAMILIES[cast(str, family)], "invalid memory report mode")
    _require(type(report.get("complete")) is bool, "invalid memory completion")
    _require(
        report.get("error") is None or type(report.get("error")) is str, "invalid memory error"
    )
    for key in ("stream", "work", "provenance", "outcomes"):
        if key in report:
            _require(type(report[key]) is dict, f"invalid {key}")
    for path, limit in (
        ("replays", 4),
        ("native", 4),
        ("recorded_outcomes", 4),
        ("stream/events", 128),
        ("stream/checkpoints", 128),
        ("stream/applied", 32),
    ):
        records = _records(report, path, limit)
        if path in ("native", "replays", "recorded_outcomes"):
            scenes = [row.get("scene") for row in records]
            _require(
                all(type(scene) is str and scene in _SCENES for scene in scenes), "invalid scene"
            )
            _require(len(set(scenes)) == len(scenes), "duplicate scene")
    for row in _records(report, "replays", 4):
        _typed(
            row, ("rows", "seed"), ("complete", "native_actions_exact", "selected_indices_exact")
        )
    for row in _records(report, "native", 4):
        _require(type(row.get("report")) is dict, "invalid native report")
        _typed(row["report"], (), ("complete", "any_success", "final_success"))
    for row in _records(report, "stream/events", 128):
        _typed(
            row,
            (
                "tick",
                "packet_index",
                "budget_updates",
                "candidate_prototype_rows",
                "candidate_publications",
            ),
            ("label_available",),
        )
    for row in _records(report, "stream/checkpoints", 128):
        _typed(row, ("tick", "prototype_rows", "completed_updates"), ())
    outcomes = report.get("outcomes", {})
    _require(
        all(scene in _SCENES and type(row) is dict for scene, row in outcomes.items()),
        "invalid outcomes",
    )
    for row in outcomes.values():
        _typed(
            row,
            ("controls", "control_physics_steps", "distance_candidates", "distance_feature_values"),
            ("any_success", "final_success"),
        )


def project_taskspace_memory_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Present recorded values without treating source claims as authenticated results."""
    _envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _envelope(resources, resource_label)
    source = deepcopy(report)
    native = source["definition"] == "taskspace_memory_native_v1"
    outcomes = _section(
        "outcomes",
        "Scene endpoints",
        "Native: recorded validated endpoint summaries. Offline: command-replay checks, not new physics. Unknown means no recorded value; raw partial native returns are separate below.",
        ["Scene", "Any success", "Final success", "Sparse return", "Controls", "Physics steps"]
        if native
        else [
            "Scene",
            "Replay complete",
            "Native actions exact",
            "Selected indices exact",
            "Query rows",
            "Error",
        ],
    )
    for scene in _SCENES:
        if native:
            paths = [
                f"outcomes/{scene}/{key}"
                for key in (
                    "any_success",
                    "final_success",
                    "sparse_return",
                    "controls",
                    "control_physics_steps",
                )
            ]
        else:
            index = next(
                (i for i, row in enumerate(source.get("replays", [])) if row["scene"] == scene),
                None,
            )
            paths = [
                f"replays/{index}/{key}"
                for key in (
                    "complete",
                    "native_actions_exact",
                    "selected_indices_exact",
                    "rows",
                    "error",
                )
            ]
        outcomes["rows"].append(
            dict(cells=[_label(scene)] + [_cell(source, "/" + path) for path in paths])
        )
    sections = [
        outcomes,
        _rows(
            source,
            "recorded_outcomes",
            ("scene", "seed", "outcomes"),
            4,
            "historical",
            "Historical native outcomes",
            "Copied prior evidence; offline replay does not execute these physics outcomes.",
        ),
        _details(
            source,
            tuple(f"provenance/recorded_baseline_198/{scene}" for scene in _SCENES),
            "baseline",
            "Original 198-row baseline",
            "Historical baseline including failed Pick19; no retuning or matched-compute claim.",
        ),
        _rows(
            source,
            "native",
            (
                "scene",
                "report/complete",
                "report/any_success",
                "report/final_success",
                "report/sparse_return",
                "report/work",
            ),
            4,
            "native",
            "Returned native records",
            "Raw returns may precede wrapper validation. A returned success does not imply the entire run completed.",
        ),
        _rows(
            source,
            "stream/events",
            (
                "tick",
                "packet_index",
                "label_available",
                "budget_updates",
                "candidate_prototype_rows",
                "candidate_publications",
                "applied",
            ),
            128,
            "events",
            "Delayed releases and memory growth",
            "Original event order, ticks and permanent charges. Empty applied lists are recorded waiting events.",
        ),
        _rows(
            source,
            "stream/applied",
            (
                "event_id",
                "episode_id",
                "observed_at",
                "arrived_at",
                "applied_at",
                "update_number",
                "diagnostic",
            ),
            32,
            "releases",
            "Applied TRAIN supervision",
            "Recorded consented TRAIN releases; this inspector admits no updates.",
        ),
        _rows(
            source,
            "stream/checkpoints",
            (
                "tick",
                "phase",
                "episode_id",
                "prototype_rows",
                "completed_updates",
                "known_episode_ids",
            ),
            128,
            "checkpoints",
            "Checkpoint memory composition",
            "Retained episode identifiers and numeric row counts, not a byte estimate of total process memory.",
        ),
        _details(
            source,
            (
                "stream/memory",
                "stream/model_state",
                "stream/policy",
                "stream/experiment_state",
                "stream/release_scope",
            ),
            "memory",
            "Retained memory and scope",
            "Per-owner numeric bytes exclude Python objects, snapshots and RSS unless explicitly recorded.",
        ),
        _details(
            source,
            (
                "work",
                "stream/work",
                "elapsed_seconds",
                "stream/elapsed_seconds",
                "attempted",
                "returned",
                "unreturned_stage_work_unknown",
            ),
            "work",
            "Measured work and unknown costs",
            "No missing cost becomes zero. Raw native per-scene work is linked above; no inferred totals.",
        ),
        _details(
            source,
            ("mode", "provenance", "stream/scope", "stream/input_normalization"),
            "provenance",
            "Source and limitations",
            "Recorded source claims are not independently authenticated. No circadian advantage, generalization or final-evaluation claim follows.",
        ),
    ]
    if resources is not None:
        sections.append(
            _details(
                resources,
                tuple(key.replace("~", "~0").replace("/", "~1") for key in resources),
                "resources",
                "Separately supplied resources — association UNVERIFIED",
                "Independent cost file; no source association is inferred.",
                "resources",
            )
        )
    return dict(
        schema_id="taskspace_memory_inspector_v1",
        source=source,
        source_label=source_label,
        resources=deepcopy(resources),
        resource_label=resource_label,
        protocol_id=_cell(source, "/definition"),
        status=_cell(source, "/complete"),
        reason=_cell(source, "/error"),
        sections=sections,
    )
