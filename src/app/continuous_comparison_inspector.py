"""Project saved continuous replay contrasts without learning or recomputation.

Inputs are bounded JSON summaries and optional unrelated resource records.
Outputs are detached exact source cells for the existing offline table renderer.
No filesystem, model imports, winner selection or evidence authentication.
"""

from copy import deepcopy
from typing import Any, cast

from src.app.coherent_stream_inspector_projection import _cell, _envelope, _label, _require

FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
DIRECTIONS = dict(
    warm_acquisition_gain=True,
    new_acquisition_gain=True,
    warm_retention_loss=False,
    warm_return_recovery=True,
    warm_return_deficit=False,
    new_retention_loss=False,
)
VALUES = ("baseline", "replay", "replay_minus_baseline", "higher_is_better")
PROBE_METRICS = ("fresh_acquisition_gain", "fresh_training_gain", "old_retention_loss")


def _browser_envelope(value: Any, label: Any) -> None:
    _envelope(value, label)
    # Why this: JavaScript JSON.parse rounds larger integers in source dialogs.
    pending = [value]
    while pending:
        node = pending.pop()
        if type(node) is int:
            _require(abs(node) <= 2**53 - 1, "integer exceeds browser exact range")
        elif type(node) is dict:
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)


def _mapping(value: Any, allowed: tuple[str, ...], name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"invalid {name}")
    _require(all(key in allowed for key in value), f"unknown {name} field")
    return value


def _contrast(value: Any, direction: bool) -> None:
    record = _mapping(value, VALUES, "contrast")
    for key in VALUES[:3]:
        if key in record and record[key] is not None:
            _require(type(record[key]) in (float, int), f"invalid numeric {key}")
    if "higher_is_better" in record:
        _require(record["higher_is_better"] is direction, "invalid comparison direction")


def _probes(arm: dict[str, Any]) -> list[dict[str, Any]]:
    records = arm.get("probes", [])
    _require(type(records) is list and len(records) <= 4, "invalid probe records")
    wakes = []
    for record in records:
        _mapping(
            record,
            ("wake_updates", "baseline_native_updates", "replay_native_updates", "views"),
            "probe",
        )
        wake = record.get("wake_updates")
        _require(type(wake) is int and wake in (0, 1, 2, 3), "invalid probe wake")
        wakes.append(wake)
        for key in ("baseline_native_updates", "replay_native_updates"):
            if key in record and record[key] is not None:
                _require(type(record[key]) is int and record[key] >= 0, f"invalid {key}")
        views = _mapping(record.get("views", {}), ("inherited", "zero_velocity"), "probe views")
        for view in views.values():
            _mapping(view, PROBE_METRICS, "probe metrics")
            for name in PROBE_METRICS[:2]:
                if name in view:
                    _contrast(view[name], True)
            losses = _mapping(
                view.get("old_retention_loss", {}), ("old_0", "old_1"), "old contexts"
            )
            for metric in losses.values():
                _contrast(metric, False)
    _require(len(set(wakes)) == len(wakes), "duplicate probe wake")
    return records


def _validate(report: dict[str, Any]) -> None:
    _require(type(report.get("complete")) is bool, "invalid completion")
    arms = _mapping(report.get("comparisons"), FAMILIES, "comparison families")
    for arm in arms.values():
        _mapping(arm, ("benchmark", "probes"), "family")
        benchmark = _mapping(arm.get("benchmark", {}), tuple(DIRECTIONS), "benchmark")
        for metric, value in benchmark.items():
            _contrast(value, DIRECTIONS[metric])
        _probes(arm)
    work = report.get("observed_work")
    _require(type(work) is dict, "invalid observed work")
    for key, value in cast(dict[str, Any], work).items():
        _require(value is None or type(value) is int and value >= 0, f"invalid work {key}")
    _require(type(report.get("baseline_source")) is dict, "invalid baseline source")
    if "qualification" in report:
        _require(type(report["qualification"]) is str, "invalid qualification")
    if "environment" in report:
        _require(type(report["environment"]) is dict, "invalid environment")
    _require(report.get("error") is None or type(report["error"]) is str, "invalid error")


def _section(identity: str, title: str, note: str, columns: list[str]) -> dict[str, Any]:
    return dict(id=identity, title=title, note=note, columns=columns, rows=[])


def _values(source: dict[str, Any], path: str) -> list[dict[str, Any]]:
    return [_cell(source, path + "/" + key) for key in VALUES]


# Why this: each table projects one kind of evidence using the same source cells.
def _benchmark_section(source: dict[str, Any]) -> dict[str, Any]:
    columns = [
        "Method",
        "Metric",
        "No replay",
        "Replay",
        "Replay minus baseline",
        "Higher is better",
    ]
    benchmark = _section(
        "benchmark",
        "Learning, retention and return",
        "Six frozen directions per method. Signed gains/losses are not absolute errors. All controls remain visible; no aggregate winner.",
        columns,
    )
    for family in FAMILIES:
        path = "/comparisons/" + family
        for metric in DIRECTIONS:
            benchmark["rows"].append(
                dict(
                    cells=[
                        _label(family),
                        _label(metric),
                        *_values(source, path + "/benchmark/" + metric),
                    ]
                )
            )
    return benchmark


def _probe_section(source: dict[str, Any]) -> dict[str, Any]:
    probes = _section(
        "probes",
        "Isolated fresh-task probes",
        "Disposable probes do not update the benchmark or deployed learner. Fresh acquisition is TEST gain; fresh training is TRAIN gain. Negative gains and both inherited/zero-velocity views are retained. Fixed wake slots are labels, not evidence that an unrecorded probe ran.",
        [
            "Method",
            "Wake slot (not evidence)",
            "Wake",
            "Baseline native updates",
            "Replay native updates",
            "View",
            "Metric",
            "No replay",
            "Replay",
            "Replay minus baseline",
            "Higher is better",
        ],
    )
    for family in FAMILIES:
        path = "/comparisons/" + family
        records = _probes(source["comparisons"].get(family, {}))
        indices = {record["wake_updates"]: index for index, record in enumerate(records)}
        for wake in range(4):
            # Absent slots resolve to missing cells, never to another wake's evidence.
            index = str(indices[wake]) if wake in indices else f"unrecorded-wake-{wake}"
            base = path + f"/probes/{index}"
            for view in ("inherited", "zero_velocity"):
                for metric in (
                    "fresh_acquisition_gain",
                    "fresh_training_gain",
                    "old_retention_loss/old_0",
                    "old_retention_loss/old_1",
                ):
                    probes["rows"].append(
                        dict(
                            cells=[
                                _label(family),
                                _label(str(wake)),
                                *[
                                    _cell(source, base + "/" + key)
                                    for key in (
                                        "wake_updates",
                                        "baseline_native_updates",
                                        "replay_native_updates",
                                    )
                                ],
                                _label(view),
                                _label(metric),
                                *_values(source, base + "/views/" + view + "/" + metric),
                            ]
                        )
                    )
    return probes


def _work_section(source: dict[str, Any]) -> dict[str, Any]:
    work = _section(
        "work",
        "Recorded additional work",
        "Observed counters belong to this run, not zero-cost saved controls or matched total compute. Validation, copies, process peaks and overhead may be excluded; missing is unknown.",
        ["Counter", "Recorded value"],
    )
    for key in source["observed_work"]:
        token = key.replace("~", "~0").replace("/", "~1")
        work["rows"].append(dict(cells=[_label(key), _cell(source, "/observed_work/" + token)]))
    return work


def _provenance_section(source: dict[str, Any]) -> dict[str, Any]:
    provenance = _section(
        "provenance",
        "Source, environment and limits",
        "Saved source claims are not authenticated here. This synthetic continuous-task comparison is not robot task success, generalization, matched computation or circadian benefit.",
        ["Field", "Recorded value"],
    )
    for key in ("complete", "error", "qualification", "environment", "baseline_source"):
        provenance["rows"].append(dict(cells=[_label(key), _cell(source, "/" + key)]))
    return provenance


def project_continuous_comparison_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Preserve saved favorable/adverse contrasts and costs; never rank learners."""
    _browser_envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _browser_envelope(resources, resource_label)
    source = deepcopy(report)
    sections = [
        _benchmark_section(source),
        _probe_section(source),
        _work_section(source),
        _provenance_section(source),
    ]
    if resources is not None:
        section = _section(
            "resources",
            "Separate resources — association UNVERIFIED",
            "No source association is inferred from a separately supplied file.",
            ["Recorded resources"],
        )
        section["rows"].append(dict(cells=[_cell(resources, "", "resources")]))
        sections.append(section)
    return dict(
        schema_id="continuous_comparison_inspector_v1",
        source=source,
        source_label=source_label,
        resources=deepcopy(resources),
        resource_label=resource_label,
        protocol_id=_label("Saved continuous replay contrasts"),
        status=_cell(source, "/complete"),
        reason=_cell(source, "/qualification"),
        sections=sections,
    )
