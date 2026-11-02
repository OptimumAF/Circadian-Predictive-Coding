"""Present saved ordinary paired linear workflows without model or metric work.

Inputs are bounded recorded JSON and optional unverified resource records.
Output uses the shared inspector's detached source cells. Producer flags remain
claims; no source authentication, inference, state restore or IO is performed.
"""

from copy import deepcopy
from typing import Any

from src.app.coherent_stream_inspector_projection import _cell, _envelope, _fields, _require
from src.app.taskspace_memory_inspector_projection import _details, _section, _typed

DEFINITION = "separate_context_linear_ticket_workflow_v1"
PAIRED_DEFINITION = "fresh_context_linear_paired_serving_v1"


def _mapping(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    _require(value is None or type(value) is dict, f"invalid {key} record")
    return {} if value is None else value


def _records(record: dict[str, Any], key: str, limit: int) -> list[dict[str, Any]]:
    value = record.get(key)
    if value is None:
        return []
    _require(type(value) is list and len(value) <= limit, f"invalid {key} records")
    _require(all(type(row) is dict for row in value), f"invalid {key} row")
    return value


def _identity(record: dict[str, Any]) -> None:
    _typed(record, ("completed_updates", "training_examples", "generation"), ("complete",))
    for key in ("phase", "policy_version", "concept", "actor_version", "error"):
        if record.get(key) is not None:
            _require(type(record[key]) is str, f"invalid {key}")


def _episode(record: dict[str, Any]) -> None:
    _typed(record, ("training_updates", "external_actions", "max_steps"), ("complete", "success"))
    for step in _records(record, "steps", 16):
        _typed(step, ("attempt",), ())
        for key in (
            "observation_state",
            "effect_state",
            "result",
            "transition",
            "capabilities_at_effect",
        ):
            _mapping(step, key)


def _validate(report: dict[str, Any]) -> None:
    _require(report.get("definition") == DEFINITION, "unsupported ordinary linear workflow")
    _require(report.get("method") == "linear_fit", "invalid ordinary linear method")
    _require(type(report.get("complete")) is bool, "invalid report completion")
    _require("paired_serving" in report, "ordinary linear paired report requires paired_serving")
    _require(report.get("reserved_probe_requested") is not True, "reserved probe uses its own view")
    _identity(report)
    _typed(
        report,
        ("final_examples", "probe_updates", "replay_examples"),
        ("source_unchanged", "actor_unchanged", "numpy_rng_unchanged", "python_rng_unchanged"),
    )
    child = _mapping(report, "paired_serving")
    if child:
        _require(child.get("definition") == PAIRED_DEFINITION, "unsupported linear paired family")
    for record in (report, child):
        _require(
            record.get("linear_train_update") in (None, "minimum_change"), "invalid linear update"
        )
        _identity(record)
        _typed(
            record,
            ("publication_tick", "learning_boundary_tick", "rollback_generation"),
            ("source_unchanged", "rollback_equal_source", "training_complete"),
        )
        for key in ("budget", "training_work", "candidate_state", "final_actor"):
            value = _mapping(record, key)
            _typed(
                value,
                (
                    "completed_updates",
                    "training_examples",
                    "updates",
                    "updates_completed",
                    "attempted_updates",
                    "attempted_rows",
                    "returned_updates",
                ),
                (),
            )
            if key == "budget":
                _fields(value, ("started_at", "last_clock"), count=False)
                _typed(
                    _mapping(value, "limits"), ("max_training_updates", "max_process_rss_bytes"), ()
                )
                _fields(_mapping(value, "limits"), ("max_wall_seconds",), count=False, low=0)
        for key in ("guard_report", "learning_boundary_guard"):
            guard = _mapping(record, key)
            _typed(_mapping(guard, "decision"), (), ("accepted",))
        for row in _records(record, "native_calls", 128):
            _typed(row, ("rows",), ("complete",))
    for row in _records(child, "episodes", 24):
        _identity(row)
        if row.get("phase") is not None:
            _require(row["phase"] in ("published", "rollback"), "invalid paired episode phase")
        _episode(_mapping(row, "episode"))
        for read in _records(row, "reads", 16):
            _identity(read)
    for row in _records(_mapping(child, "live_serving"), "records", 128):
        _identity(row)
        _typed(row, ("tick",), ())
        _episode(_mapping(row, "episode"))


def _episode_table(
    source: dict[str, Any], path: str, fields: tuple[str, ...], identity: str, title: str
) -> dict[str, Any]:
    table = _section(
        identity,
        title,
        "Recorded values only; missing entries stay unknown.",
        ["Phase", "Context", "Task", *fields],
    )
    child = _mapping(source, "paired_serving")
    for index, row in enumerate(_records(child, "episodes", 24)):
        root = f"/paired_serving/episodes/{index}"
        record = _mapping(row, "episode") if path == "episode/steps" else row
        for offset, _ in enumerate(
            _records(record, "steps" if path == "episode/steps" else path, 16)
        ):
            table["rows"].append(
                dict(
                    cells=[
                        _cell(source, root + "/" + k)
                        for k in ("phase", "policy_version", "concept")
                    ]
                    + [_cell(source, f"{root}/{path}/{offset}/{key}") for key in fields]
                )
            )
    return table


def project_context_linear_workflow_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Keep ordinary source, private candidate and serving/rollback evidence distinct."""
    _envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _envelope(resources, resource_label)
    source = deepcopy(report)
    sections = [
        _details(
            source,
            (
                "definition",
                "method",
                "linear_train_update",
                "complete",
                "error",
                "source_unchanged",
                "actor_unchanged",
                "numpy_rng_unchanged",
                "python_rng_unchanged",
                "final_examples",
                "probe_updates",
                "replay_examples",
                "paired_serving/definition",
                "paired_serving/complete",
                "paired_serving/episodes",
                "paired_serving/live_serving",
            ),
            "identity",
            "Ordinary paired linear workflow",
            "Producer claims are not authenticated. This viewer performs no learning or metric recomputation.",
        ),
        _details(
            source,
            (
                "paired_serving/prior_training",
                "paired_serving/learning_boundary_tick",
                "paired_serving/publication_tick",
                "paired_serving/guard_report/candidate_revision",
                "paired_serving/candidate_state/completed_updates",
                "paired_serving/candidate_state/training_examples",
                "paired_serving/final_actor",
                "paired_serving/action_snapshot",
                "paired_serving/rollback_generation",
                "paired_serving/rollback_equal_source",
                "paired_serving/return_publication_attempted",
            ),
            "models",
            "Source, published actor, private candidate and rollback",
            "Published and private counts differ. Read rows identify the actual serving model. No before/private-return task outcomes are invented.",
        ),
        _details(
            source,
            (
                "paired_serving/learning_boundary_guard",
                "paired_serving/guard_report",
                "paired_serving/promotion_receipt",
                "paired_serving/sharing_snapshot",
                "paired_serving/owned_source_verified",
            ),
            "guards",
            "Recorded guard and publication evidence",
            "Recorded acceptance is not independent evaluation or source authentication.",
        ),
    ]
    outcomes = _section(
        "outcomes",
        "Published and rollback task episodes",
        "All saved outcomes, including refusals/failures. A successful old task is not recovery after forgetting; no metric is recomputed.",
        [
            "Phase",
            "Context",
            "Task",
            "Complete",
            "Success",
            "Stop reason",
            "Task training",
            "External actions",
        ],
    )
    for index, _ in enumerate(_records(_mapping(source, "paired_serving"), "episodes", 24)):
        root = f"/paired_serving/episodes/{index}"
        outcomes["rows"].append(
            dict(
                cells=[
                    _cell(source, root + "/" + key)
                    for key in (
                        "phase",
                        "policy_version",
                        "concept",
                        "complete",
                        "episode/success",
                        "episode/stopped_reason",
                        "episode/training_updates",
                        "episode/external_actions",
                    )
                ]
            )
        )
    sections += [
        outcomes,
        _episode_table(
            source,
            "reads",
            ("generation", "actor_version", "completed_updates"),
            "reads",
            "Actual model used for each task read",
        ),
        _episode_table(
            source,
            "episode/steps",
            (
                "attempt",
                "observation_state/phase",
                "observation_state/revision",
                "result/selected_id",
                "transition/status",
                "effect_state/phase",
                "effect_state/revision",
                "capabilities_at_effect",
            ),
            "steps",
            "Recorded task transitions",
        ),
    ]
    live = _section(
        "live",
        "Live timeline outcomes",
        "Raw records retain revocations and incomplete work; no success denominator is changed.",
        ["Tick", "Context", "Task", "Complete", "Success", "Stop reason", "Complete record"],
    )
    for index, _ in enumerate(
        _records(_mapping(_mapping(source, "paired_serving"), "live_serving"), "records", 128)
    ):
        root = f"/paired_serving/live_serving/records/{index}"
        live["rows"].append(
            dict(
                cells=[
                    _cell(source, root + "/" + key)
                    for key in (
                        "tick",
                        "policy_version",
                        "concept",
                        "complete",
                        "episode/success",
                        "episode/stopped_reason",
                    )
                ]
                + [_cell(source, root)]
            )
        )
    sections += [
        live,
        _details(
            source,
            (
                "budget",
                "training_work",
                "stream",
                "paired_serving/budget",
                "paired_serving/training_work",
                "paired_serving/native_calls",
                "paired_serving/training_complete",
                "paired_serving/typed_work",
                "paired_serving/typed_native_work",
                "paired_serving/coefficient_bytes",
                "paired_serving/input_bytes",
                "paired_serving/source_record_bytes",
                "paired_serving/memory",
                "paired_serving/elapsed_seconds",
            ),
            "costs",
            "Costs and original clocks",
            "Reported source/child costs stay separate. Missing costs are unknown; no whole-runtime or scientific advantage is inferred.",
        ),
    ]
    if resources is not None:
        sections.append(
            _details(
                resources,
                tuple(k.replace("~", "~0").replace("/", "~1") for k in resources),
                "resources",
                "Separate resources — association UNVERIFIED",
                "No source association is inferred.",
                "resources",
            )
        )
    return dict(
        schema_id="ticket_workflow_inspector_v1",
        view_kind="ordinary_linear_paired",
        source=source,
        source_label=source_label,
        resources=deepcopy(resources),
        resource_label=resource_label,
        protocol_id=_cell(source, "/definition"),
        status=_cell(source, "/complete"),
        reason=_cell(source, "/error"),
        sections=sections,
    )
