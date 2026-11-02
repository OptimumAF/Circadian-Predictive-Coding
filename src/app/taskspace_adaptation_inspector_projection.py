"""Present saved isolated memory adaptation; no IO, learner or metric computation.

Inputs are bounded CPU/native JSON and optional unverified cost metadata. Outputs
reuse detached source cells and the existing taskspace-memory table renderer.
"""

from copy import deepcopy
from typing import Any

from src.app.coherent_stream_inspector_projection import _cell, _envelope, _fields, _label, _require
from src.app.taskspace_memory_inspector_projection import _details, _rows, _section, _typed

CPU = "taskspace_memory_adaptation_v1"
NATIVE = "taskspace_memory_native_adaptation_v1"
_PROBE = "continuous_disposable_retained_nearest_probe_v1"
_RETURN = "continuous_retained_old_train_return_v1"
_PAIR = "continuous_retained_return_probe_v1"
_SCENES = ("pick17", "probe23", "pick19", "pick20", "push17")
_POINTS = ("before_probe", "after_probe", "after_return")
_SCORES = ("fresh", "old_0", "old_1", "old_2", "old_3", "fresh_training")


def _object(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    _require(value is None or type(value) is dict, f"invalid {key}")
    return {} if value is None else value


def _records(record: dict[str, Any], key: str, maximum: int) -> list[dict[str, Any]]:
    value = record.get(key, [])
    _require(type(value) is list and len(value) <= maximum, f"invalid {key}")
    _require(all(type(row) is dict for row in value), f"invalid {key} records")
    return value


def _header(record: dict[str, Any], definitions: tuple[str, ...]) -> None:
    _require(record.get("definition") in definitions, "unsupported adaptation definition")
    _require(type(record.get("complete")) is bool, "invalid adaptation completion")
    _require(record.get("error") is None or type(record["error"]) is str, "invalid error")


def _names(record: dict[str, Any], key: str, choices: tuple[str, ...]) -> None:
    if key not in record or record[key] is None:
        return
    names = record[key]
    _require(type(names) is list and len(names) <= len(choices), f"invalid {key}")
    _require(all(type(n) is str and n in choices for n in names), f"invalid {key} values")
    _require(len(set(names)) == len(names), f"duplicate {key}")


def _phase(record: dict[str, Any], definition: str) -> None:
    _header(record, (definition,))
    _typed(
        record, ("source_updates", "source_training_examples"), ("receipt_observation_complete",)
    )
    for key in ("plan", "progress", "source_state", "retention", "work", "metrics"):
        _object(record, key)
    plan = _object(record, "plan")
    _fields(plan, ("updates",), count=True)
    if plan.get("training_role") is not None:
        expected = "train" if definition == _RETURN else "probe_train"
        _require(plan["training_role"] == expected, "invalid private training role")
    _fields(
        _object(record, "progress"),
        ("updates_completed", "replay_examples_completed", "next_tick"),
        count=True,
    )
    metrics = _object(record, "metrics")
    _fields(
        metrics,
        ("fresh_acquisition_gain", "fresh_training_gain", "return_training_gain"),
        count=False,
    )
    old = _object(metrics, "old_retention_loss")
    _require(set(old) <= set(_SCORES[1:5]), "invalid old metric slots")
    _fields(old, tuple(old), count=False)
    for key in ("events", "applied_updates", "native_updates"):
        _records(record, key, 32)
    for point in _records(record, "checkpoints", 2):
        _typed(point, ("updates",), ("complete",))
        scores = _object(point, "scores")
        _require(set(scores) <= set(_SCORES), "invalid score slots")
        for name, value in scores.items():
            _require(type(value) is dict, "invalid score record")
            role = value.get("role")
            expected = (
                "probe_test"
                if name == "fresh"
                else "probe_train"
                if name == "fresh_training"
                else "train"
            )
            _require(role is None or role == expected, "invalid score role")
            residual = _object(value, "residual")
            _fields(_object(residual, "overall"), ("normalized_half_mse",), count=False, low=0)


def _cpu(record: dict[str, Any]) -> None:
    _header(record, (CPU,))
    _require(record.get("mode") == "isolated_retained_adaptation", "invalid adaptation mode")
    _typed(record, ("max_prototypes",), ("unreturned_private_work_unknown",))
    _require(
        record.get("return_task") is None
        or record["return_task"] in ("pick17", "pick19", "pick20", "push17"),
        "invalid return task",
    )
    for key in ("stream", "adaptation", "provenance", "retention"):
        _object(record, key)
    stream = _object(record, "stream")
    for key, maximum in (("events", 128), ("checkpoints", 128), ("applied", 32)):
        for row in _records(stream, key, maximum):
            _typed(
                row,
                ("tick", "completed_updates", "prototype_rows", "budget_updates", "update_number"),
                ("label_available",),
            )
    adaptation = _object(record, "adaptation")
    if not adaptation:
        return
    if adaptation.get("definition") == _PAIR:
        _header(adaptation, (_PAIR,))
        _object(adaptation, "work")
        for key, definition in (("probe", _PROBE), ("return_phase", _RETURN)):
            child = _object(adaptation, key)
            if child:
                _phase(child, definition)
    else:
        _phase(adaptation, _PROBE)


def _validate(report: dict[str, Any]) -> None:
    _header(report, (CPU, NATIVE))
    if report["definition"] == CPU:
        _cpu(report)
        return
    _typed(report, (), ("unreturned_native_work_unknown",))
    phase = report.get("observed_native_phase")
    _require(phase is None or phase in _POINTS[1:], "invalid observed phase")
    for key, choices in (
        ("unobserved_native_phases", _POINTS),
        ("cpu_endpoint_phases", _POINTS),
        ("attempted", _SCENES),
        ("returned", _SCENES),
    ):
        _names(report, key, choices)
    workflow = _object(report, "workflow")
    if workflow:
        _cpu(workflow)
    native = _records(report, "native", 5)
    _require(
        [row.get("scene") for row in native] == list(_SCENES[: len(native)]),
        "invalid native scene order",
    )
    for row in native:
        _require(type(row.get("report")) is dict, "invalid native report")
        _typed(row["report"], (), ("complete", "any_success", "final_success"))
    outcomes = _object(report, "outcomes")
    _require(set(outcomes) <= set(_SCENES), "invalid outcome scene")
    for row in outcomes.values():
        _require(type(row) is dict, "invalid outcome record")
        _typed(
            row,
            ("controls", "control_physics_steps", "distance_candidates", "distance_feature_values"),
            ("any_success", "final_success"),
        )
        _fields(row, ("sparse_return",), count=False)


def _phase_sections(
    source: dict[str, Any], path: str, identity: str, title: str
) -> list[dict[str, Any]]:
    # Why this: each private phase owns its own clock and charges. Do not add them
    # to source preparation or infer retention/recovery from positive fit gains.
    sections = [
        _details(
            source,
            tuple(
                f"{path}/{key}"
                for key in (
                    "complete",
                    "error",
                    "classification",
                    "plan",
                    "source_updates",
                    "source_training_examples",
                    "source_state",
                    "progress",
                    "receipt_observation_complete",
                    "retention",
                    "work",
                )
            ),
            identity,
            title,
            "Disposable learner only; source/serving/benchmark state is separate. Native outcomes are not measured by CPU fit. Per-phase clocks, exposures, current stored payload and costs are recorded, not summed.",
        )
    ]
    metrics = _details(
        source,
        tuple(
            f"{path}/metrics/{key}"
            for key in (
                "fresh_acquisition_gain",
                "fresh_training_gain",
                "return_training_gain",
                *(f"old_retention_loss/old_{i}" for i in range(4)),
            )
        ),
        identity + "-metrics",
        title + " — recorded changes",
        "Acquisition/training/return gain = before minus after loss; old retention loss = after minus before (positive is worse). A return gain without prior forgetting is not recovery. Missing remains unknown.",
    )
    sections.append(metrics)
    scores = _section(
        identity + "-scores",
        title + " — all recorded readouts",
        "Normalized half MSE: lower is better. fresh = score-only reserved probe_test; fresh_training = new probe_train; old_0..3 = original TRAIN. Neither reserved score nor good fit establishes native success. Complete is checkpoint capture, not independent acceptance.",
        ["Checkpoint", "Captured", "Slot", "Episode", "Role", "Loss (lower)", "Residual record"],
    )
    node: Any = source
    for key in path.split("/"):
        node = node.get(key) if type(node) is dict else None
    for index, point in enumerate([] if node is None else node.get("checkpoints", [])):
        base = f"/{path}/checkpoints/{index}"
        for name in _SCORES:
            if name not in (point.get("scores") or {}):
                continue
            score = f"{base}/scores/{name}"
            scores["rows"].append(
                dict(
                    cells=[
                        _cell(source, base + "/updates"),
                        _cell(source, base + "/complete"),
                        _label(name),
                        *[
                            _cell(source, score + "/" + key)
                            for key in (
                                "episode_id",
                                "role",
                                "residual/overall/normalized_half_mse",
                                "residual",
                            )
                        ],
                    ]
                )
            )
    sections.append(scores)
    for key in ("events", "applied_updates", "native_updates"):
        sections.append(
            _rows(
                source,
                f"{path}/{key}",
                (
                    "tick",
                    "event_id",
                    "episode_id",
                    "observed_at",
                    "arrived_at",
                    "applied_at",
                    "update_number",
                    "training_rows",
                    "diagnostic",
                    "work",
                ),
                32,
                identity + "-" + key,
                title + " — " + key,
                "Original receipt/event order. No work is admitted by this viewer; missing fields are unknown. Full records remain in source.",
            )
        )
    return sections


def project_taskspace_adaptation_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Return detached recorded claims; do not authenticate completion or recovery."""
    _envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _envelope(resources, resource_label)
    source = deepcopy(report)
    native = source["definition"] == NATIVE
    prefix = "workflow/" if native else ""
    sections = [
        _details(
            source,
            (
                "definition",
                "complete",
                "error",
                "classification",
                "observed_native_phase",
                "unobserved_native_phases",
                "cpu_endpoint_phases",
                "attempted",
                "returned",
                "unreturned_native_work_unknown",
                prefix + "unreturned_private_work_unknown",
                prefix + "native_outcomes",
                "retention",
                "return_task",
            ),
            "scope",
            "Capture and observation scope",
            "UNVERIFIED imported claims, not fresh execution or independent acceptance. Root failure remains failure even if a raw leaf reports success. Native final endpoint only; unobserved phases cannot establish forgetting or recovery. Private updates never deploy or alter the benchmark learner.",
        )
    ]
    if native:
        outcomes = _section(
            "outcomes",
            "Recorded final scene summaries",
            "Five fixed known TRAIN scenes, including original new task probe23. These are wrapper-recorded summaries, not authenticated by this viewer. Raw partial returns are separate; absent outcomes remain unknown.",
            [
                "Scene",
                "Any success",
                "Final success",
                "Sparse return",
                "Controls",
                "Known control physics",
                "Distance candidates",
                "Feature values",
            ],
        )
        for scene in _SCENES:
            outcomes["rows"].append(
                dict(
                    cells=[
                        _label(scene),
                        *[
                            _cell(source, f"/outcomes/{scene}/{key}")
                            for key in (
                                "any_success",
                                "final_success",
                                "sparse_return",
                                "controls",
                                "control_physics_steps",
                                "distance_candidates",
                                "distance_feature_values",
                            )
                        ],
                    ]
                )
            )
        sections.extend(
            [
                outcomes,
                _rows(
                    source,
                    "native",
                    (
                        "scene",
                        "report/complete",
                        "report/error",
                        "report/any_success",
                        "report/final_success",
                        "report/sparse_return",
                        "report/work",
                    ),
                    5,
                    "native",
                    "Raw returned native records",
                    "A raw success may precede wrapper failure or validation; do not promote it to an accepted endpoint. No native before/after change is inferred.",
                ),
            ]
        )
    sections.append(
        _details(
            source,
            tuple(
                prefix + key
                for key in (
                    "complete",
                    "error",
                    "provenance",
                    "max_prototypes",
                    "stream/complete",
                    "stream/error",
                    "stream/model_state",
                    "stream/experiment_state",
                    "stream/memory",
                    "stream/policy",
                    "stream/release_scope",
                    "stream/work",
                    "stream/elapsed_seconds",
                )
            ),
            "preparation",
            "Saved source preparation",
            "Source preparation costs/clock/exposures differ from isolated phase costs. Current retained rows and payload exclude Python objects/snapshots/RSS. Imported provenance is not authenticated.",
        )
    )
    for key, limit, fields in (
        (
            "events",
            128,
            (
                "tick",
                "packet_index",
                "label_available",
                "budget_updates",
                "candidate_prototype_rows",
                "applied",
            ),
        ),
        (
            "applied",
            32,
            (
                "event_id",
                "episode_id",
                "observed_at",
                "arrived_at",
                "applied_at",
                "update_number",
                "diagnostic",
            ),
        ),
        (
            "checkpoints",
            128,
            (
                "tick",
                "phase",
                "episode_id",
                "prototype_rows",
                "completed_updates",
                "known_episode_ids",
            ),
        ),
    ):
        sections.append(
            _rows(
                source,
                prefix + "stream/" + key,
                fields,
                limit,
                "source-" + key,
                "Source preparation — " + key,
                "Original source order, consented TRAIN releases and permanent charges. This clock is separate from private phase clocks.",
            )
        )
    cpu = source.get("workflow") if native else source
    adaptation = {} if cpu is None else cpu.get("adaptation") or {}
    path = prefix + "adaptation"
    if adaptation.get("definition") == _PAIR:
        sections.append(
            _details(
                source,
                tuple(
                    path + "/" + key
                    for key in ("definition", "complete", "error", "classification", "work")
                ),
                "adaptation",
                "Two-phase capture",
                "Aggregate attempt scope overlaps child work; do not sum these counters again.",
            )
        )
        sections.extend(_phase_sections(source, path + "/probe", "probe", "New-task acquisition"))
        sections.extend(
            _phase_sections(source, path + "/return_phase", "return", "Old TRAIN return")
        )
    else:
        sections.extend(_phase_sections(source, path, "probe", "New-task acquisition"))
    if resources is not None:
        sections.append(
            _details(
                resources,
                tuple(key.replace("~", "~0").replace("/", "~1") for key in resources),
                "resources",
                "Separate costs — association UNVERIFIED",
                "Unverified supplemental metadata; logical counts may overlap physical counters. No inferred totals, matched computation or scientific comparison.",
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
