"""Project saved sensor evidence into detached, source-addressable tables.

Inputs and outputs are bounded JSON records. No models, scientific metric
recomputation, filesystem, training, or resource-file authentication occurs here.
Missing evidence and recorded nulls remain distinct; partial work stays visible.
"""

from copy import deepcopy
import json
from math import isfinite
from typing import Any

_STATUSES = {"complete", "incomplete", "failed", "running", "not_run", "partial"}
_METHODS = {
    "matched_bp",
    "pc",
    "neutral_cpc",
    "gated_cpc",
    "scheduled_matched_bp",
    "recent_replay_matched_bp",
}
_MISSING = object()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, low: float | None = None, high: float | None = None) -> None:
    _require(type(value) in (int, float), f"invalid numeric {name}")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"nonfinite {name}") from error
    _require(isfinite(numeric), f"nonfinite {name}")
    _require(
        (low is None or numeric >= low) and (high is None or numeric <= high),
        f"invalid range {name}",
    )


def _integer(value: Any, name: str, high: int = 2**53 - 1) -> int:
    _require(type(value) is int and 0 <= value <= high, f"invalid integer {name}")
    return value


def _envelope(value: Any, label: Any) -> None:
    _require(
        type(label) is str and bool(label.strip()) and label == label.strip(),
        "invalid source label",
    )
    _require(len(label.encode("utf8")) <= 1024, "source label exceeds bounds")
    _require(type(value) is dict, "sensor source must be a JSON object")
    stack, count = [(value, 0)], 0
    while stack:
        node, depth = stack.pop()
        count += 1
        _require(count <= 500000 and depth <= 64, "sensor JSON structure exceeds bounds")
        if type(node) is dict:
            _require(all(type(key) is str for key in node), "JSON keys must be text")
            stack.extend((child, depth + 1) for child in node.values())
        elif type(node) is list:
            stack.extend((child, depth + 1) for child in node)
        elif type(node) in (int, float):
            _number(node, "JSON number")
        else:
            _require(type(node) in (str, bool, type(None)), "non-JSON source value")
    _require(
        len(json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf8")) <= 8 * 1024**2,
        "sensor JSON exceeds8MiB",
    )


def _token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _resolve(source: dict[str, Any], pointer: str) -> Any:
    node: Any = source
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        if type(node) is dict and key in node:
            node = node[key]
        elif type(node) is list and key.isdecimal() and int(key) < len(node):
            node = node[int(key)]
        else:
            return _MISSING
    return node


def _cell(source: dict[str, Any], pointer: str, origin: str = "report") -> dict[str, Any]:
    value = _resolve(source, pointer)
    missing = value is _MISSING
    return {
        "value": None if missing else deepcopy(value),
        "pointer": None if missing else pointer,
        "source": origin,
        "missing": missing,
    }


def _label(value: str) -> dict[str, Any]:
    return {"value": value, "pointer": None, "source": "report", "missing": False}


def _unknown(origin: str = "report") -> dict[str, Any]:
    return {"value": None, "pointer": None, "source": origin, "missing": True}


def _section(identity: str, title: str, note: str, columns: list[str]) -> dict[str, Any]:
    return {"id": identity, "title": title, "note": note, "columns": columns, "rows": []}


def _records(value: Any, maximum: int, name: str) -> list[dict[str, Any]]:
    _require(
        type(value) is list and len(value) <= maximum and all(type(row) is dict for row in value),
        f"invalid bounded {name}",
    )
    return value


def _status(record: dict[str, Any], required: bool = True) -> None:
    if required or "status" in record:
        _require(
            type(record.get("status")) is str and record["status"] in _STATUSES,
            "invalid sensor status",
        )
    if "reason" in record:
        _require(record["reason"] is None or type(record["reason"]) is str, "invalid sensor reason")


def _scores(scores: Any, contexts: set[str]) -> None:
    _require(type(scores) is dict and set(scores) <= contexts, "invalid score contexts")
    for metrics in scores.values():
        _require(
            type(metrics) is dict and set(metrics) <= {"bce", "accuracy"}, "invalid score metrics"
        )
        for name, value in metrics.items():
            _number(value, name, 0, 1 if name == "accuracy" else None)


def _prearrival(report: dict[str, Any]) -> dict[str, Any]:
    section = _section(
        "prearrival",
        "Before label arrival",
        "Recorded shadow forecasts joined only to arrived labels; no heldout feedback.",
        ["Feature tick", "State updates", "Label tick", "BCE", "Accuracy"],
    )
    previous = -1
    for index, row in enumerate(_records(report.get("rows"), 12, "prearrival rows")):
        tick = _integer(row.get("feature_tick"), "feature tick", 11)
        _require(tick > previous, "prearrival feature ticks must increase")
        previous = tick
        _integer(row.get("prediction_state_updates"), "prediction state updates", 12)
        _integer(row.get("label_tick"), "label tick", 14)
        _require(row["label_tick"] > tick, "label tick must follow feature tick")
        _scores({"scores": row.get("scores")}, {"scores"})
        base = f"/rows/{index}"
        section["rows"].append(
            {
                "cells": [
                    _cell(report, base + path)
                    for path in (
                        "/feature_tick",
                        "/prediction_state_updates",
                        "/label_tick",
                        "/scores/bce",
                        "/scores/accuracy",
                    )
                ]
            }
        )
    return section


def _heldout(report: dict[str, Any]) -> dict[str, Any]:
    section = _section(
        "heldout",
        "Heldout A and B",
        "Finish-only readouts. Missing contexts are unknown, including paid partial failures. Accuracy ceilings do not establish recovery.",
        ["Checkpoint", "Context", "BCE", "Accuracy", "Status"],
    )
    heldout: Any = report.get("heldout_evaluation")
    if heldout is None:
        section["note"] += " Heldout evaluation was not recorded."
        return section
    _require(
        type(heldout) is dict and type(heldout.get("progress")) is dict,
        "invalid heldout evaluation",
    )
    progress = heldout["progress"]
    _status(progress)
    points = _records(progress.get("checkpoints", []), 13, "heldout checkpoints")
    previous = -1
    for index, point in enumerate(points):
        update = _integer(point.get("updates"), "heldout checkpoint", 12)
        _require(update > previous, "heldout checkpoints must increase")
        previous = update
        _status(point)
        _require(
            point.get("training_context") in (None, "A", "B"), "invalid recorded training context"
        )
        _scores(point.get("scores"), {"A", "B"})
        if point["status"] == "complete":
            _require(
                set(point["scores"]) == {"A", "B"}
                and all(set(v) == {"bce", "accuracy"} for v in point["scores"].values()),
                "complete heldout point lacks scores",
            )
        base = f"/heldout_evaluation/progress/checkpoints/{index}"
        for context in ("A", "B"):
            # A/B are fixed schema labels; recorded metrics remain source-addressed.
            # Optional planned descriptor metadata must not merge distinct curves.
            cells = [_cell(report, base + "/updates"), _label(context)]
            cells.extend(
                _cell(report, base + f"/scores/{context}/{metric}")
                for metric in ("bce", "accuracy")
            )
            cells.append(_cell(report, base + "/status"))
            section["rows"].append({"cells": cells})
    if progress["status"] == "complete":
        _require(
            len(points) == 13
            and all(p["updates"] == i and p["status"] == "complete" for i, p in enumerate(points)),
            "complete heldout progress lacks checkpoints",
        )
    retention = heldout.get("retention")
    if retention is not None:
        _require(
            type(retention) is dict and type(retention.get("metrics")) is dict,
            "invalid retention summary",
        )
        for metric, values in retention["metrics"].items():
            _require(
                metric in ("bce", "accuracy") and type(values) is dict, "invalid retention metrics"
            )
            for value in values.values():
                _number(value, "signed retention metric")
    return section


def _probes(report: dict[str, Any]) -> dict[str, Any]:
    section = _section(
        "probes",
        "Isolated fresh task C",
        "Disposable probe forks are separate from the benchmark learner. These curves are recorded, not recomputed or selected.",
        ["Wake checkpoint", "Probe updates", "BCE", "Accuracy", "Status"],
    )
    learnability: Any = report.get("learnability")
    if learnability is None:
        section["note"] += " No isolated probes were recorded."
        return section
    _require(type(learnability) is dict, "invalid learnability record")
    _status(learnability)
    previous = -1
    for index, observation in enumerate(
        _records(learnability.get("observations", []), 4, "probe observations")
    ):
        wake = _integer(observation.get("updates"), "probe wake checkpoint", 12)
        _require(
            type(wake) is int and wake in (0, 4, 8, 12) and wake > previous,
            "invalid ordered probe wake checkpoint",
        )
        previous = wake
        _status(observation)
        probe = observation.get("probe")
        base = f"/learnability/observations/{index}"
        if probe is None:
            section["rows"].append(
                {
                    "cells": [
                        _cell(report, base + "/updates"),
                        _unknown(),
                        _unknown(),
                        _unknown(),
                        _cell(report, base + "/status"),
                    ]
                }
            )
            continue
        _require(type(probe) is dict, "invalid probe record")
        _status(probe)
        checkpoints = _records(probe.get("checkpoints", []), 4, "probe checkpoints")
        last = -1
        for point_index, point in enumerate(checkpoints):
            update = _integer(point.get("updates"), "probe checkpoint", 4)
            _require(
                type(update) is int and update in (0, 1, 2, 4) and update > last,
                "invalid ordered probe checkpoint",
            )
            last = update
            _scores(point.get("scores"), {"C"})
            _status(point, required=False)
            prefix = base + f"/probe/checkpoints/{point_index}"
            paths = [
                base + "/updates",
                prefix + "/updates",
                prefix + "/scores/C/bce",
                prefix + "/scores/C/accuracy",
                prefix + "/status" if "status" in point else base + "/probe/status",
            ]
            section["rows"].append({"cells": [_cell(report, path) for path in paths]})
        if probe["status"] == "complete":
            _require(
                [point["updates"] for point in checkpoints] == [0, 1, 2, 4]
                and all(
                    set(point["scores"]) == {"C"}
                    and set(point["scores"]["C"]) == {"bce", "accuracy"}
                    for point in checkpoints
                ),
                "complete probe lacks scores",
            )
        if not checkpoints:
            section["rows"].append(
                {
                    "cells": [
                        _cell(report, base + "/updates"),
                        _unknown(),
                        _unknown(),
                        _unknown(),
                        _cell(report, base + "/probe/status"),
                    ]
                }
            )
    return section


def _fields(
    section: dict[str, Any],
    source: dict[str, Any],
    pointer: str,
    label: str,
    origin: str = "report",
) -> None:
    """Flatten selected recorded subtrees; field names are UI labels, not evidence."""
    value = _resolve(source, pointer)
    if type(value) is dict and value:
        for key in value:
            _fields(section, source, pointer + "/" + _token(key), label + " / " + key, origin)
    elif type(value) is list and value:
        for index in range(len(value)):
            _fields(section, source, pointer + f"/{index}", label + f" / {index}", origin)
    else:
        _require(len(section["rows"]) < 4000, "sensor section exceeds row bounds")
        section["rows"].append({"cells": [_label(label), _cell(source, pointer, origin)]})


def _details(
    report: dict[str, Any], identity: str, title: str, note: str, paths: tuple[str, ...]
) -> dict[str, Any]:
    section = _section(identity, title, note, ["Field", "Recorded value"])
    for path in paths:
        _fields(section, report, "/" + path, path)
    return section


def _validate_measured(value: Any, key: str = "") -> None:
    """Refuse malformed displayed work without asserting experimental provenance."""
    if type(value) is dict:
        for name, child in value.items():
            _validate_measured(child, name)
    elif type(value) is list:
        for child in value:
            _validate_measured(child, key)
    elif value is not None:
        if key.endswith(
            ("_bytes", "_updates", "_examples", "_attempted", "_completed", "_exposures", "_count")
        ):
            _integer(value, key)
        elif key.endswith("seconds"):
            _number(value, key, 0)


def _validate_monitoring(record: dict[str, Any]) -> None:
    for name in ("complete", "stopped"):
        if name in record:
            _require(type(record[name]) is bool, "invalid monitoring flag")
    for row in _records(record.get("rows", []), 12, "monitoring rows"):
        for name in ("feature_tick", "label_tick"):
            _integer(row.get(name), "monitoring " + name, 14)
        decision: Any = row.get("decision")
        _require(type(decision) is dict, "invalid monitoring decision")
        alarms = decision.get("alarms")
        _require(
            type(alarms) is list and len(alarms) == 3 and all(type(v) is bool for v in alarms),
            "invalid monitoring alarms",
        )
    false_alarms = record.get("false_alarms", {})
    _require(type(false_alarms) is dict, "invalid false alarm evidence")
    for phase in false_alarms.values():
        _require(type(phase) is dict, "invalid false alarm phase")
        if "monitoring_batches" in phase:
            _integer(phase["monitoring_batches"], "monitoring batches", 12)
        for name in ("alarm_batches", "incident_onsets", "rates"):
            if name not in phase:
                continue
            values = phase[name]
            _require(type(values) is list and len(values) == 3, f"invalid {name} channels")
            for value in values:
                if name == "rates":
                    if value is not None:
                        _number(value, "false alarm rate", 0, 1)
                else:
                    _integer(value, name, 12)
    _validate_measured(record)


def _stationary_configuration(value: Any, required: bool = False) -> None:
    _require(type(value) is dict, "invalid stationary monitoring configuration")
    if required or "stationary" in value:
        _require(value.get("stationary") is True, "stationary monitoring must be explicit")
    if "phase_schedule" in value:
        _require(value["phase_schedule"] == ["A"] * 12, "contradictory stationary schedule")


def _stationary_absent_phase(value: Any, expected: dict[str, Any]) -> None:
    _require(type(value) is dict, "invalid absent stationary phase")
    # Why this: refuse contradictory evidence, but never invent missing partial fields.
    for name, actual in value.items():
        if name in expected:
            wanted = expected[name]
            _require(
                actual == wanted and type(actual) is type(wanted),
                "contradictory absent stationary phase " + name,
            )


def _validate_stationary_monitoring(record: Any) -> None:
    _require(type(record) is dict, "invalid stationary monitoring")
    if "configuration" in record:
        _stationary_configuration(record["configuration"])
    false_alarms = record.get("false_alarms", {})
    _require(type(false_alarms) is dict, "invalid stationary false alarms")
    for name in ("pre_shift_A", "post_return_A"):
        if name in false_alarms:
            _stationary_absent_phase(
                false_alarms[name],
                {
                    "monitoring_batches": 0,
                    "alarm_batches": [0] * 3,
                    "incident_onsets": [0] * 3,
                    "rates": [None] * 3,
                },
            )
    observed = false_alarms.get("stationary_A", {})
    _require(type(observed) is dict, "invalid stationary opportunities")
    if "monitoring_batches" in observed:
        _integer(observed["monitoring_batches"], "stationary opportunities", 8)
    recovery = record.get("recovery", {})
    _require(type(recovery) is dict, "invalid stationary recovery")
    for name in ("B", "A_return"):
        if name in recovery:
            _stationary_absent_phase(
                recovery[name],
                {
                    "status": "not_applicable",
                    "first_deficit_tick": None,
                    "confirmation_tick": None,
                    "labels_to_recovery": None,
                    "labels_available_before_last_prediction": 0,
                    "observed_batches": 0,
                    "planned_batches": 0,
                    "horizon_complete": False,
                },
            )


def _validate_stationary(report: dict[str, Any], protocol: dict[str, Any]) -> None:
    _require(protocol.get("stationary") is True, "stationary identity requires stationary true")
    _require(protocol.get("phases") == ["A"] * 3, "contradictory stationary phases")
    _stationary_configuration(protocol.get("monitoring"), required=True)
    _require(protocol.get("heldout") is None, "stationary heldout protocol unsupported")
    _require(
        protocol.get("final_evaluation") in (None, False), "stationary final evaluation unsupported"
    )
    for field in ("heldout_evaluation", "learnability"):
        _require(report.get(field) is None, "stationary heldout/probe extension unsupported")
        _require(protocol.get(field) is None, "stationary heldout/probe protocol unsupported")
    if report.get("sensor_monitoring") is not None:
        _validate_stationary_monitoring(report["sensor_monitoring"])


def project_sensor_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Build offline tables with exact pointers; supplying costs never authenticates them."""
    _envelope(report, source_label)
    _require(
        (resources is None) == (resource_label is None),
        "resources and resource label must be supplied together",
    )
    if resources is not None:
        _envelope(resources, resource_label)
        _validate_measured(resources)
    _status(report)
    protocol: Any = report.get("protocol")
    _require(type(protocol) is dict, "invalid sensor protocol")
    identity = protocol.get("protocol")
    _require(
        type(identity) is str
        and identity.startswith(
            (
                "sensor_brightness_prearrival_",
                "sensor_brightness_scheduled_",
                "sensor_brightness_recent_replay_",
                "sensor_stationary_prearrival_",
                "sensor_stationary_scheduled_",
                "sensor_stationary_recent_replay_",
            )
        )
        and identity.endswith("_v1"),
        "unsupported sensor protocol",
    )
    _require(
        type(protocol.get("method")) is str and protocol["method"] in _METHODS,
        "unsupported sensor method",
    )
    stationary = identity.startswith("sensor_stationary_")
    if stationary:
        _validate_stationary(report, protocol)
    sections = [_prearrival(report), _heldout(report), _probes(report)]
    if stationary:
        sections[0]["note"] += " Stationary A throughout; no physical shift."
        sections[1]["title"] = "Heldout evaluation"
        sections[1]["note"] = (
            "Not part of this stationary control; no heldout evaluation was recorded."
        )
        sections[2]["note"] = "Isolated fresh-task probes are not part of this stationary control."
    sections.append(
        _details(
            report,
            "retention",
            "Signed retention outcomes",
            "Recorded summary only. Gain positive is better; old-task loss and return gap positive is worse. Negative losses are improvements, not evidence of forgetting or recovery.",
            ("heldout_evaluation/retention",),
        )
    )
    sections.append(
        _details(
            report,
            "monitoring",
            "Alarms and recovery",
            "Descriptive training-stream observations. Preserve zero denominators and null recovery; these are not statistical false-positive or causal recovery estimates.",
            (
                "sensor_monitoring/complete",
                "sensor_monitoring/stopped",
                "sensor_monitoring/diagnostic_scope",
                "sensor_monitoring/false_alarms",
                "sensor_monitoring/recovery",
            ),
        )
    )
    monitoring: Any = report.get("sensor_monitoring")
    if stationary:
        sections[-2]["note"] = (
            "Retention and return-task outcomes are not part of this stationary control. Missing evidence remains unknown."
        )
        sections[-1]["note"] += (
            " Stationary A has at most eight dependent, noise-free post-calibration opportunities. B and A-return phases are absent; recorded not_applicable values are retained. Partial reports use observed opportunities."
        )
    if monitoring is not None:
        _require(type(monitoring) is dict, "invalid sensor monitoring")
        _validate_monitoring(monitoring)
        for index, row in enumerate(_records(monitoring.get("rows", []), 12, "monitoring rows")):
            for field in ("feature_tick", "label_tick", "decision", "incident"):
                _fields(
                    sections[-1],
                    report,
                    f"/sensor_monitoring/rows/{index}/{field}",
                    f"rows / {index} / {field}",
                )
            if "accuracy" in row:
                _number(row["accuracy"], "monitoring accuracy", 0, 1)
    measured_paths = (
        "diagnostics",
        "prearrival_observation",
        "charged_training_updates",
        "input_array_bytes",
        "parameter_count",
        "elapsed_seconds",
        "process_rss",
        "scheduled_training",
        "replay_budget_charged_exposures",
        "rehearsal/charged_exposures",
        "rehearsal/attempted_updates",
        "rehearsal/returned_updates",
        "rehearsal/committed_updates",
        "learnability/costs",
        "learnability/charged_probe_updates",
        "learnability/reserved_data_array_bytes",
        "learnability/reference_array_bytes",
        "heldout_evaluation/heldout_array_bytes",
        "heldout_evaluation/progress",
    )
    costs = _details(
        report,
        "resources",
        "Recorded work and memory",
        "Report-owned measurements below. Missing costs are unknown; planned protocol counts are not actual costs. Separately supplied resource evidence has UNVERIFIED association: no shared unique report identity exists.",
        measured_paths[:-1],
    )
    if stationary:
        costs["note"] += (
            " A separate resource file may aggregate several cases; its totals are not allocated to this learner."
        )
        for path in ("rehearsal/events", "rehearsal/incident_admission"):
            _fields(costs, report, "/" + path, path)
    for path in measured_paths:
        recorded = _resolve(report, "/" + path)
        if recorded is not _MISSING:
            _validate_measured(recorded, path.rsplit("/", 1)[-1])
    for field in (
        "status",
        "reason",
        "prediction_calls_attempted",
        "prediction_calls_completed",
        "prediction_rows_attempted",
        "prediction_rows_completed",
    ):
        _fields(
            costs, report, "/heldout_evaluation/progress/" + field, "heldout progress / " + field
        )
    learnability: Any = report.get("learnability")
    if type(learnability) is dict:
        for index, _ in enumerate(learnability.get("observations", [])):
            for field in (
                "status",
                "reason",
                "feature_health",
                "importance",
                "neuron_age",
                "unavailable_reason",
                "checkpoint_snapshot_array_bytes",
                "diagnostic_activation_bytes",
                "invocation_seconds",
                "probe/completed_updates",
                "probe/attempted_updates",
                "probe/completed_training_examples",
                "probe/attempted_training_examples",
                "probe/elapsed_seconds",
            ):
                _fields(
                    costs,
                    report,
                    f"/learnability/observations/{index}/{field}",
                    f"probe {index} / {field}",
                )
    if resources is not None:
        for field in (
            "status",
            "reason",
            "case",
            "physical_work",
            "process_rss",
            "elapsed_seconds",
            "output_bytes",
            "unknown_after_failure",
        ):
            _fields(
                costs,
                resources,
                "/" + field,
                "UNVERIFIED separate resource / " + field,
                "resources",
            )
    else:
        _fields(
            costs, {}, "/physical_work", "Separate native physical work (not supplied)", "resources"
        )
    sections.append(costs)
    sections.append(
        _details(
            report,
            "controls",
            "Scope and inactive controls",
            "Saved metadata only. Deployment, sleep, and topology work remain unobserved when absent; no scientific circadian advantage is established.",
            (
                "status",
                "reason",
                "protocol/method",
                "protocol/seed",
                "protocol/dimensions",
                "protocol/actor",
                "protocol/prediction",
                "protocol/scoring",
                "protocol/comparison",
                "source_unchanged",
                "actor_unchanged",
                "pending_prediction_batches",
                "protocol/monitoring/feedback",
                "protocol/learnability/isolation",
                "protocol/learnability/limitations",
                "deployment",
                "sleep",
                "topology",
            ),
        )
    )
    if stationary:
        for path in (
            "protocol/protocol",
            "protocol/stationary",
            "protocol/phases",
            "protocol/fixture",
            "protocol/limitations",
            "protocol/monitoring/false_alarm_scope",
        ):
            _fields(sections[-1], report, "/" + path, path)
    return {
        "schema_id": "sensor_inspector_v1",
        "source_label": source_label,
        "source": deepcopy(report),
        "resource_label": resource_label,
        "resources": deepcopy(resources),
        "protocol_id": identity,
        "status": report["status"],
        "reason": report.get("reason"),
        "method": protocol["method"],
        "sections": sections,
    }
