"""Project saved coherent-stream comparisons into detached source cells.

Owns bounded JSON validation and presentation only. No filesystem, model calls,
scientific metric recomputation, source authentication or experiment selection.
"""

from copy import deepcopy
import json
from math import isfinite
from typing import Any, cast

_PROTOCOL = "coherent_independent_data_replay_removal_v1"
_METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
_POLICIES = ("none", "rotating")
_POINTS = (0, 6, 12, 18)
_OUTCOMES = ("new_B_gain", "retention_loss", "return_gain")
_STATUSES = {"complete", "incomplete", "failed", "not_run", "running", "partial", "unavailable"}
_MISSING = object()
_TOTALS = (
    "wake_updates",
    "replay_charged_examples",
    "replay_attempted_updates",
    "replay_returned_updates",
    "replay_committed_updates",
    "total_committed_updates",
    "row_exposures_charged",
    "known_native_train_calls",
    "known_completed_score_forwards",
    "native_train_calls",
    "final_score_forwards",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, *, count: bool = False, low: float | None = None) -> None:
    if count:
        _require(type(value) is int and 0 <= value <= 2**53 - 1, f"invalid count {name}")
        return
    _require(type(value) in (int, float), f"invalid number {name}")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"nonfinite number {name}") from error
    _require(isfinite(numeric) and (low is None or numeric >= low), f"invalid number {name}")


def _envelope(value: Any, label: Any) -> None:
    _require(
        type(label) is str and bool(label.strip()) and label == label.strip(),
        "invalid source label",
    )
    _require(type(value) is dict, "source must be a JSON object")
    stack, nodes = [(value, 0)], 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        _require(nodes <= 500000 and depth <= 64, "JSON structure exceeds bounds")
        if type(node) is dict:
            _require(all(type(key) is str for key in node), "JSON keys must be text")
            stack.extend((child, depth + 1) for child in node.values())
        elif type(node) is list:
            stack.extend((child, depth + 1) for child in node)
        elif type(node) in (int, float):
            _number(node, "JSON value")
        else:
            _require(type(node) in (str, bool, type(None)), "non-JSON value")
    try:
        _require(len(label.encode("utf8")) <= 1024, "source label exceeds bounds")
        size = len(json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf8"))
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid JSON encoding") from error
    _require(size <= 8 * 1024**2, "JSON exceeds8MiB")


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
    return dict(
        value=None if missing else deepcopy(value),
        pointer=None if missing else pointer,
        source=origin,
        missing=missing,
    )


def _label(text: str) -> dict[str, Any]:
    return dict(value=text, pointer=None, source="report", missing=False)


def _records(value: Any, maximum: int, name: str) -> list[dict[str, Any]]:
    _require(type(value) is list and len(value) <= maximum, f"invalid {name}")
    _require(all(type(row) is dict for row in value), f"invalid {name} records")
    return value


def _status(record: dict[str, Any]) -> None:
    if "status" in record:
        value = record["status"]
        _require(type(value) is str and value in _STATUSES, "invalid status")
    if "reason" in record:
        _require(record["reason"] is None or type(record["reason"]) is str, "invalid reason")


def _fields(
    record: dict[str, Any],
    names: tuple[str, ...],
    *,
    count: bool,
    required: bool = False,
    low: float | None = None,
) -> None:
    for name in names:
        if required:
            _require(name in record and record[name] is not None, f"complete report lacks {name}")
        if name in record and record[name] is not None:
            _number(record[name], name, count=count, low=0 if count else low)


def _points(value: Any, *, complete: bool) -> dict[int, int]:
    points = _records(value, 4, "checkpoints")
    indexes: dict[int, int] = {}
    previous = -1
    for index, point in enumerate(points):
        update = cast(int, point.get("updates"))
        _require(
            type(update) is int and update in _POINTS and update > previous,
            "invalid checkpoint order",
        )
        previous = update
        indexes[update] = index
        expected_context = {0: None, 6: "A", 12: "B", 18: "A"}[update]
        if "training_context" in point:
            _require(point["training_context"] == expected_context, "invalid checkpoint context")
        scores = point.get("scores", {})
        _require(type(scores) is dict and set(scores) <= {"A", "B"}, "invalid score contexts")
        _require(
            list(scores) == [context for context in ("A", "B") if context in scores],
            "invalid context order",
        )
        if complete:
            _require(set(scores) == {"A", "B"}, "complete checkpoint lacks contexts")
        for metrics in scores.values():
            _require(
                type(metrics) is dict and set(metrics) <= {"bce", "accuracy"},
                "invalid score metrics",
            )
            if complete:
                _require(set(metrics) == {"bce", "accuracy"}, "complete checkpoint lacks metrics")
            for name, number in metrics.items():
                if number is None and not complete:
                    continue
                _number(number, name, low=0)
                _require(name != "accuracy" or number <= 1, "accuracy exceeds one")
    if complete:
        _require(tuple(indexes) == _POINTS, "complete arm lacks checkpoints")
    return indexes


def _replay(rehearsal: dict[str, Any], *, complete: bool) -> None:
    counts = ("charged_exposures", "attempted_updates", "returned_updates", "committed_updates")
    _fields(rehearsal, counts, count=True, required=complete)
    events = _records(rehearsal.get("events", []), 6, "replay events")
    previous = 0
    for event in events:
        _status(event)
        update = cast(int, event.get("updates"))
        _require(
            type(update) is int and update in (3, 6, 9, 12, 15, 18) and update > previous,
            "invalid replay event order",
        )
        previous = update
        _fields(event, counts, count=True, required=complete)
        if complete:
            _require(
                event.get("status") == "complete" and all(event[name] == 2 for name in counts),
                "incomplete replay event",
            )
        retention = event.get("retention")
        if retention is not None:
            _require(type(retention) is dict, "invalid replay retention")
            _fields(retention, ("example_count", "retained_bytes"), count=True)
    if complete:
        _require(
            len(events) == 6 and all(rehearsal[name] == 12 for name in counts),
            "incomplete replay work",
        )


def _arms(
    report: dict[str, Any],
) -> dict[tuple[str, str], tuple[int, dict[str, Any], dict[int, int]]]:
    indexed = {}
    previous = -1
    planned = [(policy, method) for policy in _POLICIES for method in _METHODS]
    for index, row in enumerate(_records(report.get("runs", []), 8, "runs")):
        identity = cast(tuple[str, str], (row.get("policy"), row.get("method")))
        _require(
            identity in planned and identity not in indexed, "invalid or duplicate arm identity"
        )
        position = planned.index(identity)
        _require(position > previous, "invalid arm order")
        previous = position
        _status(row)
        complete = row.get("status") == "complete"
        if complete or "seed" in row:
            _require(type(row.get("seed")) is int and row["seed"] == 23, "invalid arm seed")
        if complete or "stream" in row:
            _require(row.get("stream") == "recurring", "invalid arm stream")
        _fields(row, ("wake_updates", "replay_charged_examples"), count=True, required=complete)
        _fields(row, ("elapsed_seconds",), count=False, low=0, required=complete)
        if "source_learner_unchanged" in row:
            _require(
                type(row["source_learner_unchanged"]) is bool, "invalid source preservation fact"
            )
        outcomes = row.get("outcomes")
        if outcomes is not None:
            _require(type(outcomes) is dict and set(outcomes) <= set(_OUTCOMES), "invalid outcomes")
            _fields(outcomes, _OUTCOMES, count=False, required=complete)
        elif complete:
            raise ValueError("complete arm lacks outcomes")
        result = row.get("stream_result")
        _require(result is None or type(result) is dict, "invalid stream result")
        if complete:
            _require(type(result) is dict, "complete arm lacks stream result")
        points = {} if result is None else _points(result.get("checkpoints", []), complete=complete)
        if result is not None:
            _fields(
                result,
                ("completed_updates", "training_examples", "score_forward_calls"),
                count=True,
                required=complete,
            )
        rehearsal = row.get("rehearsal")
        _require(rehearsal is None or type(rehearsal) is dict, "invalid rehearsal")
        if rehearsal is not None:
            _replay(rehearsal, complete=complete)
        if complete:
            result = cast(dict[str, Any], result)
            _require(row["wake_updates"] == 18, "complete arm lacks wake updates")
            _require(
                row.get("source_learner_unchanged") is True,
                "complete arm lacks source preservation",
            )
            _require(
                all(
                    result[key] == value
                    for key, value in (
                        ("completed_updates", 18),
                        ("training_examples", 144),
                        ("score_forward_calls", 8),
                    )
                ),
                "contradictory completed stream counts",
            )
            _require(
                (rehearsal is None) == (identity[0] == "none"), "contradictory replay presence"
            )
            _require(
                row["replay_charged_examples"] == (0 if identity[0] == "none" else 12),
                "contradictory replay count",
            )
        indexed[identity] = (index, row, points)
    return indexed


def _pairs(report: dict[str, Any], arms: dict[Any, Any]) -> dict[str, int]:
    indexed = {}
    previous = -1
    for index, pair in enumerate(_records(report.get("pairs", []), 4, "pairs")):
        method = cast(str, pair.get("method"))
        _require(
            type(method) is str and method in _METHODS and method not in indexed,
            "invalid pair identity",
        )
        position = _METHODS.index(method)
        _require(position > previous, "invalid pair order")
        previous = position
        _status(pair)
        complete = pair.get("status") == "complete"
        delta = pair.get("rotating_minus_none")
        _require(delta is None or type(delta) is dict, "invalid paired outcomes")
        if delta is not None:
            _fields(delta, _OUTCOMES, count=False, required=complete)
        elif complete:
            raise ValueError("complete pair lacks differences")
        _fields(
            pair,
            ("native_update_difference", "training_row_difference"),
            count=True,
            required=complete,
        )
        for name in ("initial_primary_levels_exact", "wake_receipt_timing_exact"):
            if name in pair and pair[name] is not None:
                _require(type(pair[name]) is bool, "invalid pair fact")
            if complete:
                _require(pair.get(name) is True, "complete pair lacks matching facts")
        if complete:
            _require(
                all(
                    (policy, method) in arms and arms[policy, method][1].get("status") == "complete"
                    for policy in _POLICIES
                ),
                "complete pair lacks complete arms",
            )
            a, b = (
                arms[policy, method][1]["stream_result"]["checkpoints"][0]["scores"]
                for policy in _POLICIES
            )
            _require(a == b, "contradictory initial score parity")
            _require(
                pair["native_update_difference"] == pair["training_row_difference"] == 12,
                "contradictory paired work differences",
            )
        indexed[method] = index
    return indexed


def _controls(report: dict[str, Any], arms: dict[Any, Any], pairs: dict[str, int]) -> None:
    complete = report.get("status") == "complete"
    _require("status" in report, "report lacks status")
    _status(report)
    _fields(report, _TOTALS, count=True, required=complete)
    _fields(report, ("total_seconds",), count=False, required=complete, low=0)
    _memory(report.get("observed_process_rss"), required=complete)
    for name in ("scientific_advantage_verified", "equivalence_verified", "full_R28_complete"):
        if name in report:
            _require(type(report[name]) is bool, "invalid scientific flag")
        if complete:
            _require(report.get(name) is False, "unsupported scientific completion claim")
    groups = _records(report.get("group_facts", []), 2, "group facts")
    seen = set()
    for group in groups:
        policy = group.get("policy")
        _require(
            type(policy) is str and policy in _POLICIES and policy not in seen,
            "invalid group identity",
        )
        seen.add(policy)
        _status(group)
        for name in ("pc_neutral_exact", "shared_replay_plans_exact", "replay_absent"):
            if group.get(name) is not None:
                _require(type(group[name]) is bool, "invalid group fact")
        if group.get("pc_neutral_exact") is True:
            pc, neutral = arms.get((policy, "pc")), arms.get((policy, "neutral_cpc"))
            _require(pc is not None and neutral is not None, "parity claim lacks arms")
            assert pc is not None and neutral is not None
            _require(
                pc[1].get("status") == neutral[1].get("status") == "complete",
                "parity claim lacks complete arms",
            )
            _require(
                pc[1]["stream_result"]["checkpoints"] == neutral[1]["stream_result"]["checkpoints"],
                "contradictory neutral score parity",
            )
        if complete:
            _require(
                group.get("shared_replay_plans_exact" if policy == "none" else "replay_absent")
                is None,
                "inapplicable replay fact must remain null",
            )
            _require(
                group.get("status") == "complete" and group.get("pc_neutral_exact") is True,
                "complete report lacks group parity",
            )
            _require(
                group.get("replay_absent" if policy == "none" else "shared_replay_plans_exact")
                is True,
                "complete report lacks replay fact",
            )
    if complete:
        _require(
            len(arms) == 8 and len(pairs) == 4 and len(groups) == 2,
            "complete report lacks planned records",
        )
        _require(
            all(row[1].get("status") == "complete" for row in arms.values()),
            "complete report has unfinished arms",
        )
        _require(
            all(report["pairs"][i].get("status") == "complete" for i in pairs.values()),
            "complete report has unfinished pairs",
        )
        expected = dict(
            wake_updates=144,
            replay_charged_examples=48,
            replay_attempted_updates=48,
            replay_returned_updates=48,
            replay_committed_updates=48,
            total_committed_updates=192,
            row_exposures_charged=1200,
            known_native_train_calls=192,
            known_completed_score_forwards=64,
            native_train_calls=192,
            final_score_forwards=64,
        )
        _require(
            all(report[name] == value for name, value in expected.items()),
            "contradictory complete work totals",
        )


def _table(identity: str, title: str, note: str, columns: list[str]) -> dict[str, Any]:
    return dict(id=identity, title=title, note=note, columns=columns, rows=[])


def _details(
    section: dict[str, Any],
    source: dict[str, Any],
    pointer: str,
    *,
    origin: str = "report",
    label: str = "",
) -> None:
    value = _resolve(source, pointer)
    if type(value) in (dict, list) and value:
        for key in value if type(value) is dict else map(str, range(len(value))):
            token = key.replace("~", "~0").replace("/", "~1")
            _details(
                section,
                source,
                pointer + "/" + token,
                origin=origin,
                label=f"{label}.{key}" if label else key,
            )
    else:
        section["rows"].append(
            dict(cells=[_label(label or pointer.lstrip("/")), _cell(source, pointer, origin)])
        )


def _curve_rows(report: dict[str, Any], arm: Any) -> list[dict[str, Any]]:
    rows = []
    for axis, checkpoint in enumerate(_POINTS):
        point = None if arm is None else arm[2].get(checkpoint)
        base = "/runs/8" if point is None else f"/runs/{arm[0]}/stream_result/checkpoints/{point}"
        update = f"/protocol/checkpoint_updates/{axis}" if point is None else base + "/updates"
        for context in ("A", "B"):
            rows.append(
                dict(
                    checkpoint=checkpoint,
                    context=context,
                    cells=[
                        _cell(report, update),
                        _label(context),
                        _cell(report, base + f"/scores/{context}/bce"),
                        _cell(report, base + f"/scores/{context}/accuracy"),
                    ],
                )
            )
    return rows


def _resources(resources: dict[str, Any]) -> None:
    _status(resources)
    physical = resources.get("physical_work", {})
    _require(type(physical) is dict, "invalid physical work")
    for record in physical.values():
        _require(type(record) is dict, "invalid physical work record")
        _fields(
            record,
            ("attempted_calls", "returned_calls", "attempted_rows", "returned_rows"),
            count=True,
        )
    _fields(resources, ("pid", "final_threads"), count=True)
    _fields(resources, ("seconds",), count=False, low=0)


def _memory(memory: Any, *, required: bool = False) -> None:
    if memory is None:
        _require(not required, "complete report lacks RSS observations")
        return
    _require(type(memory) is dict, "invalid RSS observations")
    _fields(
        memory, ("pid", "start_bytes", "peak_bytes", "sample_count"), count=True, required=required
    )
    _fields(memory, ("interval_seconds",), count=False, low=0)


def _protocol(protocol: dict[str, Any], *, complete: bool) -> None:
    for name, expected in (
        ("methods", list(_METHODS)),
        ("policies", list(_POLICIES)),
        ("checkpoint_updates", list(_POINTS)),
        ("seeds", [23]),
        ("hidden_dims", [8, 8]),
        ("precisions", [1.0, 1.0]),
    ):
        if complete:
            _require(name in protocol, f"complete protocol lacks {name}")
        if name in protocol:
            value = protocol[name]
            _require(
                type(value) is list
                and value == expected
                and all(type(item) is type(reference) for item, reference in zip(value, expected)),
                f"invalid planned {name}",
            )
    fixed = dict(
        training_rng=7302,
        final_rng=9102,
        training_batches=18,
        batch_rows=8,
        final_rows_per_context=64,
        label_delay=2,
        learning_rate=0.03,
        inference_steps=4,
        inference_learning_rate=0.2,
    )
    for name, expected_value in fixed.items():
        if complete:
            _require(name in protocol, f"complete protocol lacks {name}")
        if name in protocol:
            _require(
                type(protocol[name]) is type(expected_value) and protocol[name] == expected_value,
                f"invalid protocol {name}",
            )
    for name in ("planned_work", "per_arm_work", "resources"):
        if complete:
            _require(type(protocol.get(name)) is dict, f"complete protocol lacks {name}")
    planned = protocol.get("planned_work", {})
    _require(type(planned) is dict, "invalid planned work")
    _fields(planned, tuple(planned), count=True)
    caps = protocol.get("resources", {})
    _require(type(caps) is dict, "invalid planned resource limits")
    _fields(caps, tuple(name for name in caps if name != "wall_seconds"), count=True)
    _fields(caps, ("wall_seconds",), count=False, low=0)
    per_arm = protocol.get("per_arm_work", {})
    _require(type(per_arm) is dict and set(per_arm) <= set(_POLICIES), "invalid planned arm work")
    for record in per_arm.values():
        _require(type(record) is dict, "invalid planned arm counts")
        _fields(record, tuple(record), count=True)


def project_coherent_stream_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Preserve recorded evidence and explicit unknowns for one fixed family."""
    _envelope(report, source_label)
    protocol = report.get("protocol")
    _require(
        type(protocol) is dict and protocol.get("protocol_id") == _PROTOCOL,
        "unsupported coherent stream protocol",
    )
    _protocol(cast(dict[str, Any], protocol), complete=report.get("status") == "complete")
    arms = _arms(report)
    pairs = _pairs(report, arms)
    _controls(report, arms, pairs)
    if resources is not None:
        _envelope(resources, resource_label)
        _resources(resources)
    else:
        _require(resource_label is None, "resource label requires a resource document")
    return _project(report, source_label, resources, resource_label, arms, pairs)


def _project(report, source_label, resources, resource_label, arms, pairs) -> dict[str, Any]:
    outcomes = _table(
        "outcomes",
        "Acquisition, forgetting and return",
        "Recorded signed BCE outcomes. B gain and return gain: higher is better. A forgetting: lower is better; negative means improvement. No winner is selected.",
        ["Policy", "Method", "Status", "B gain", "A forgetting", "A return gain"],
    )
    paired = _table(
        "pairs",
        "Rotating minus none",
        "Rotating minus no replay, with unequal work. Positive gain differences favor rotating; negative forgetting differences mean less forgetting. Every sign remains visible.",
        [
            "Method",
            "Status",
            "Δ B gain",
            "Δ forgetting",
            "Δ return gain",
            "Extra updates",
            "Extra rows",
        ],
    )
    work = _table(
        "work",
        "Measured work",
        "Reported paid work and sampled resources; unknown is not zero. Optional resource association is UNVERIFIED.",
        ["Property", "Value"],
    )
    planned = _table(
        "planned",
        "Planned work and limits",
        "Protocol budgets and planned counts are not measured work.",
        ["Property", "Value"],
    )
    controls = _table(
        "controls",
        "Controls and limitations",
        "Reported facts are evidence claims, not new acceptance or scientific benefit.",
        ["Property", "Value"],
    )
    projected_arms = []
    for policy in _POLICIES:
        for method in _METHODS:
            arm = arms.get((policy, method))
            base = "/runs/8" if arm is None else f"/runs/{arm[0]}"
            projected_arms.append(
                dict(
                    key=f"{policy}:{method}",
                    policy=policy,
                    method=method,
                    status=_cell(report, base + "/status"),
                    reason=_cell(report, base + "/reason"),
                    curves=_curve_rows(report, arm),
                )
            )
            outcomes["rows"].append(
                dict(
                    cells=[
                        _cell(report, base + path)
                        for path in (
                            "/policy",
                            "/method",
                            "/status",
                            "/outcomes/new_B_gain",
                            "/outcomes/retention_loss",
                            "/outcomes/return_gain",
                        )
                    ]
                )
            )
            _details(
                controls,
                report,
                base + "/source_learner_unchanged",
                label=f"{policy}:{method}.source_learner_unchanged",
            )
            _details(controls, report, base + "/reason", label=f"{policy}:{method}.reason")
            for path in (
                "wake_updates",
                "replay_charged_examples",
                "elapsed_seconds",
                "rehearsal/attempted_updates",
                "rehearsal/returned_updates",
                "rehearsal/committed_updates",
                "stream_result/training_examples",
                "stream_result/score_forward_calls",
            ):
                _details(work, report, base + "/" + path, label=f"{policy}:{method}.{path}")
            if arm is not None and type(arm[1].get("rehearsal")) is dict:
                for index, _event in enumerate(arm[1]["rehearsal"].get("events", [])):
                    for name in ("example_count", "retained_bytes"):
                        _details(
                            work,
                            report,
                            base + f"/rehearsal/events/{index}/retention/{name}",
                            label=f"{policy}:{method}.event{index}.retention.{name}",
                        )
    for method in _METHODS:
        base = "/pairs/4" if method not in pairs else f"/pairs/{pairs[method]}"
        paired["rows"].append(
            dict(
                method=method,
                cells=[
                    _cell(report, base + path)
                    for path in (
                        "/method",
                        "/status",
                        "/rotating_minus_none/new_B_gain",
                        "/rotating_minus_none/retention_loss",
                        "/rotating_minus_none/return_gain",
                        "/native_update_difference",
                        "/training_row_difference",
                    )
                ],
            )
        )
    for name in (*_TOTALS, "total_seconds", "observed_process_rss"):
        _details(work, report, "/" + name)
    for name in ("planned_work", "per_arm_work", "resources"):
        _details(planned, report, "/protocol/" + name, label=name)
    for name in (
        "group_facts",
        "scientific_advantage_verified",
        "equivalence_verified",
        "full_R28_complete",
        "global_errors",
        "partial_work_note",
    ):
        _details(controls, report, "/" + name)
    for name in (
        "classification",
        "training_rng",
        "final_rng",
        "seeds",
        "hidden_dims",
        "learning_rate",
        "inference_steps",
        "inference_learning_rate",
        "precisions",
        "active_cpc_config",
        "neutral_cpc_config",
        "comparison_scope",
        "limitations",
        "final_feedback",
        "equivalence_margin",
        "statistical_inference",
        "score_release",
        "directions",
    ):
        _details(controls, report, "/protocol/" + name, label=name)
    for name in ("Topology", "Lineage", "Sleep"):
        controls["rows"].append(
            dict(cells=[_label(name + " (not recorded)"), _cell({}, "/unrecorded")])
        )
    if resources is not None:
        for name in resources:
            token = name.replace("~", "~0").replace("/", "~1")
            _details(
                work,
                resources,
                "/" + token,
                origin="resources",
                label="UNVERIFIED resources." + name,
            )
    return dict(
        schema_id="coherent_stream_inspector_v1",
        source=deepcopy(report),
        source_label=source_label,
        resources=deepcopy(resources),
        resource_label=resource_label,
        protocol_id=_cell(report, "/protocol/protocol_id"),
        status=_cell(report, "/status"),
        reason=_cell(report, "/reason"),
        arms=projected_arms,
        sections=[outcomes, paired, work, planned, controls],
    )
