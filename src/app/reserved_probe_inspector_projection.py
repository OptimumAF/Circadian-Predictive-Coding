"""Present saved ordinary probe records through existing detached source cells.

No IO, learner/numerical imports, source authentication or metric recomputation.
Plot geometry belongs to the existing offline renderer, not this projection.
"""

from copy import deepcopy
from typing import Any

from src.app.coherent_stream_inspector_projection import _cell, _envelope, _label, _number, _require
from src.app.taskspace_memory_inspector_projection import _details, _records, _section, _typed

DEFINITIONS = (
    "native_context_return_ticket_workflow_v1",
    "separate_context_linear_ticket_workflow_v1",
    "linear_context_return_ticket_workflow_v1",
)
SHARED = "linear_context_return_ticket_workflow_v1"
AUX_PATH = "reserved_probe/linear_rehearsal_work"
AUX_COUNTS = (
    "attempted_updates",
    "returned_updates",
    "applied_updates",
    "application_unknown_updates",
    "attempted_rows",
    "additional_rows_charged",
    "native_sgd_updates",
)
EVENT_COUNTS = (
    "fit_attempts",
    "fit_returns",
    "fit_rows",
    "additional_rows_charged",
    "retained_train_rows_used",
    "current_rows_used",
    "application_unknown",
    "unreturned_fit_work_unknown",
    "native_sgd_updates",
    "packet_array_bytes",
    "snapshot_attempts",
    "snapshot_returns",
)
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc", "linear_fit")
TASKS = (
    ("v1", "invoice"),
    ("v1", "password"),
    ("v2", "invoice"),
    ("v2", "parcel"),
    ("v1", "parcel"),
    ("v2", "password"),
)
PHASES = ("before_probe", "after_probe", "after_return")
POINTS = (0, 4, 8, 16)
CURVES = (
    ("F", "fresh", "reserved_probe/probe", ("parcel-v1", "password-v2")),
    ("F", "old", "reserved_probe/probe/training_readouts", ("batch-0", "batch-1")),
    ("Return", "fresh", "reserved_probe/probe/return_phase", ("parcel-v1", "password-v2")),
    (
        "Return",
        "old",
        "reserved_probe/probe/return_phase/training_readouts",
        ("batch-0", "batch-1"),
    ),
)
CHANGES = (
    "probe_success_change",
    "acquired_during_probe",
    "retained_through_probe",
    "lost_during_probe",
    "return_success_change",
    "recovered_on_return",
    "retained_after_return",
)
PHASE_LABELS = dict(
    zip(PHASES, ("Before learning", "After fresh learning", "After old-task return"))
)


def _mapping(record, key):
    value = record.get(key)
    _require(value is None or type(value) is dict, f"invalid {key} record")
    return {} if value is None else value


def _validate(report):
    _require(report.get("definition") in DEFINITIONS, "unsupported ordinary probe report")
    _require(report.get("reserved_probe_requested") is True, "explicit reserved probe required")
    _require(type(report.get("complete")) is bool, "invalid report completion")
    method = report.get("method")
    _require(type(method) is str and method in METHODS, "invalid probe method")
    linear = report["definition"] in (DEFINITIONS[1], SHARED)
    _require((method == "linear_fit") == linear, "method and report family differ")
    if linear:
        _require(
            report.get("reserved_probe_update") == "minimum_change", "explicit LS update required"
        )
    private = _mapping(report, "reserved_probe")
    _typed(
        _mapping(report, "reserved_probe_source_training"),
        ("completed_updates", "applied_update_count"),
        (),
    )
    _typed(report, ("probe_replay_examples",), ())
    _typed(_mapping(report, "probe_auxiliary_updates"), AUX_COUNTS, ())
    _validate_rehearsal(report, private)
    for record in (report, private):
        _typed(record, ("final_examples", "replay_examples"), ("complete",))
        _require(record.get("final_examples", 0) == 0, "final evaluation is not a reserved probe")
    episodes = _mapping(private, "episodes")
    _require(set(episodes) <= set(PHASES), "invalid probe phase")
    # Admit ALL fixed headers before reading outcomes, including later phases.
    for phase, rows in episodes.items():
        _require(type(rows) is list and len(rows) <= 6, "invalid task row count")
        for index, row in enumerate(rows):
            _require(
                type(row) is dict
                and row.get("phase") == phase
                and row.get("role") == "controller_train"
                and (row.get("policy_version"), row.get("concept")) == TASKS[index],
                "foreign task header/role/order",
            )
    for rows in episodes.values():
        for row in rows:
            _typed(row, (), ("complete",))
            episode = _mapping(row, "episode")
            _typed(episode, ("training_updates",), ("complete", "success"))
            _require(episode.get("training_updates", 0) == 0, "task readout cannot train")
            _require(episode.get("scenario", "ordinary") == "ordinary", "foreign task scenario")
            _require(len(_records(episode, "steps", 6)) <= 6, "invalid task steps")
    _validate_summary(_mapping(report, "reserved_probe_summary"))


def _validate_rehearsal(report, private):
    work = _mapping(private, "linear_rehearsal_work")
    controls = [
        record[key]
        for record, key in (
            (report, "reserved_probe_rehearsal"),
            (private, "linear_probe_rehearsal"),
            (work, "control"),
        )
        if record.get(key) is not None
    ]
    _require(
        all(type(c) is str and c in ("joint", "current_only") for c in controls),
        "invalid shared rehearsal control",
    )
    _require(len(set(controls)) <= 1, "incompatible shared rehearsal controls")
    flags = []
    for record, key in (
        (report, "reserved_probe_return_once"),
        (private, "linear_probe_return_once"),
        (work, "return_once"),
    ):
        _typed(record, (), (key,))
        if record.get(key) is not None:
            flags.append(record[key])
    _require(len(set(flags)) <= 1, "incompatible once-return intent")
    once = True in flags
    _require(not once or bool(controls), "once-return requires shared rehearsal control")
    _require(
        not (controls or once or work) or report["definition"] == SHARED,
        "shared rehearsal requires shared linear family",
    )
    _validate_auxiliary_work(work, once)


def _validate_auxiliary_work(work, once):
    expected = (
        "shared_linear_once_return_probe_update_v1"
        if once
        else "shared_linear_joint_probe_update_v1"
    )
    if work.get("definition") is not None:
        _require(work["definition"] == expected, "incompatible auxiliary recipe")
    _typed(work, AUX_COUNTS, ("return_once",))
    for event in _records(work, "events", 32):
        if event.get("definition") is not None:
            _require(event["definition"] == expected, "incompatible auxiliary receipt recipe")
        _require(event.get("phase") in ("fresh", "return"), "invalid auxiliary phase")
        _require(
            event.get("status") in ("running", "complete", "failed"), "invalid auxiliary status"
        )
        _typed(event, ("current_rows",), ("work_known", "applied"))
        _typed(_mapping(event, "work"), EVENT_COUNTS, ())


def _validate_summary(summary):
    if summary:
        _require(
            summary.get("definition") == "ordinary_reserved_task_outcomes_v1", "foreign summary"
        )
        _require(
            summary.get("status") in ("complete", "partial", "not_run", "invalid"),
            "invalid summary status",
        )
    groups = _mapping(summary, "cohorts")
    _require(set(groups) <= {"source_trained", "fresh"}, "foreign summary cohort")
    for name, group in groups.items():
        _require(type(group) is dict, "invalid cohort")
        tasks = TASKS[:4] if name == "source_trained" else TASKS[4:]
        labels = [f"{v}:{c}" for v, c in tasks]
        if "tasks" in group:
            _require(group["tasks"] == labels, "foreign summary tasks")
        successes = _mapping(group, "successes_by_phase")
        _require(set(successes) <= set(PHASES), "foreign summary phase")
        for value in successes.values():
            if value is not None:
                _number(value, "task successes", count=True)
                _require(value <= len(tasks), "task success count exceeds cohort")
        changes = _mapping(group, "task_changes")
        _require(set(changes) <= set(labels), "foreign task changes")
        for row in changes.values():
            _require(type(row) is dict, "invalid task changes")
            _typed(row, (), CHANGES[1:4] + CHANGES[5:])
            for key in (CHANGES[0], CHANGES[4]):
                if row.get(key) is not None:
                    _require(type(row[key]) is int and -1 <= row[key] <= 1, "invalid signed change")


def _outcomes(source):
    table = _section(
        "outcomes",
        "Saved task checks",
        "Planned slots remain visible when no reading was captured. These checks use a disposable learning copy, not the serving actor. Success at saved checks does not prove unsampled behavior or generalization.",
        ["Phase", "Planned task", "Captured", "Task success", "Stop reason"],
    )
    for phase in PHASES:
        for index, (version, concept) in enumerate(TASKS):
            base = f"/reserved_probe/episodes/{phase}/{index}"
            table["rows"].append(
                dict(
                    cells=[
                        _label(PHASE_LABELS[phase]),
                        _label(f"{version}:{concept}"),
                        _cell(source, base + "/complete"),
                        _cell(source, base + "/episode/success"),
                        _cell(source, base + "/episode/stopped_reason"),
                    ]
                )
            )
    return table


def _summary(source):
    table = _section(
        "task-changes",
        "Recorded acquisition and retention",
        "Recorded summary only, not recomputed or authenticated. Recovery requires observed prior success then loss. Null recovery can be inapplicable or unavailable; inspect the loss field. Initial choices can be near numerical ties: inspect raw action scores.",
        [
            "Planned task",
            "Learning success change",
            "Newly acquired",
            "Retained during learning",
            "Loss observed",
            "Return success change",
            "Recovered on return",
            "Retained after return",
        ],
    )
    totals = _section(
        "task-totals",
        "Old and fresh task totals",
        "Source-trained does not mean initially successful. Null counts remain unavailable, never zero. Cohorts are the fixed four old and two fresh tasks.",
        ["Cohort", *PHASE_LABELS.values()],
    )
    for name, tasks in (("source_trained", TASKS[:4]), ("fresh", TASKS[4:])):
        base = f"/reserved_probe_summary/cohorts/{name}"
        totals["rows"].append(
            dict(
                cells=[
                    _label("Old tasks" if name == "source_trained" else "Fresh tasks"),
                    *[_cell(source, base + "/successes_by_phase/" + p) for p in PHASES],
                ]
            )
        )
        for version, concept in tasks:
            label = f"{version}:{concept}"
            table["rows"].append(
                dict(
                    cells=[
                        _label(label),
                        *[
                            _cell(source, base + "/task_changes/" + label + "/" + key)
                            for key in CHANGES
                        ],
                    ]
                )
            )
    return totals, table


def _curves(source):
    table = _section(
        "score-readings",
        "All saved score readings",
        "Cross-entropy (CE) is prediction loss; lower is better. Keep every fixed checkpoint. Task success and class accuracy can stay unchanged while CE worsens. Missing readings are not zero.",
        ["Phase", "Cohort", "Context", "Update", "CE", "Class accuracy"],
    )
    panels = []
    for phase, cohort, path, contexts in CURVES:
        rows = _records(source, path + "/checkpoints", 4)
        lookup = {}
        previous = -1
        for index, row in enumerate(rows):
            row_update = row.get("updates")
            _require(
                type(row_update) is int and row_update in POINTS and row_update > previous,
                "invalid curve order/update",
            )
            assert type(row_update) is int
            previous = row_update
            scores = _mapping(row, "scores")
            _require(set(scores) <= set(contexts), "foreign score context")
            for score in scores.values():
                _require(type(score) is dict, "invalid scores")
                for key in ("ce", "accuracy"):
                    if score.get(key) is not None:
                        _number(score[key], key, low=0)
                        _require(key != "accuracy" or score[key] <= 1, "accuracy exceeds one")
            lookup[row_update] = index
        series = []
        for context in contexts:
            points = []
            for update in POINTS:
                saved_index = lookup.get(update)
                base = f"/{path}/checkpoints/{saved_index}"
                x = (
                    _cell(source, base + "/updates")
                    if saved_index is not None
                    else _cell({}, "/absent")
                )
                y = (
                    _cell(source, base + f"/scores/{context}/ce")
                    if saved_index is not None
                    else _cell({}, "/absent")
                )
                accuracy = (
                    _cell(source, base + f"/scores/{context}/accuracy")
                    if saved_index is not None
                    else _cell({}, "/absent")
                )
                points.append(dict(x=x, y=y))
                table["rows"].append(
                    dict(
                        cells=[
                            _label(phase),
                            _label(cohort),
                            _label(context),
                            x if not x["missing"] else _label(f"{update} (no reading)"),
                            y,
                            accuracy,
                        ]
                    )
                )
            series.append(dict(label=context, points=points))
        panels.append(
            dict(
                title=f"{'Fresh learning' if phase == 'F' else 'Old-task return'} · {cohort} tasks",
                series=series,
            )
        )
    return table, panels


def _steps(source):
    table = _section(
        "task-steps",
        "Saved actions and state transitions",
        "Original action scores and states are available from the source report. Near ties can depend on arithmetic. Nothing on this page executes an action or updates a learner.",
        [
            "Phase",
            "Task",
            "Attempt",
            "Selected action",
            "Observed phase",
            "Transition",
            "Result phase",
            "Scores",
        ],
    )
    for phase in PHASES:
        for index, row in enumerate(_records(source, "reserved_probe/episodes/" + phase, 6)):
            base = f"/reserved_probe/episodes/{phase}/{index}"
            for step, _ in enumerate(_records(row, "episode/steps", 6)):
                prefix = base + f"/episode/steps/{step}/"
                table["rows"].append(
                    dict(
                        cells=[
                            _label(phase),
                            _label(f"{row['policy_version']}:{row['concept']}"),
                            *[
                                _cell(source, prefix + key)
                                for key in (
                                    "attempt",
                                    "result/selected_id",
                                    "observation_state/phase",
                                    "transition/status",
                                    "transition/state/phase",
                                    "result/scores",
                                )
                            ],
                        ]
                    )
                )
    return table


def _costs(source):
    table = _section(
        "learning-work",
        "Separate learning costs",
        "Direct train_batch ports and logical phase work are separate scopes, not additive totals. Zero direct ports does not mean zero fitting: see auxiliary work and receipts below. Counts are recorded exposures, not independent examples or matched computation. Old TRAIN return reuses templates. Failed attempts stay paid; absent costs remain unknown. App and private elapsed times overlap.",
        ["Owner", "Attempted updates", "Returned updates", "Row exposures"],
    )
    for label, path, fields in (
        (
            "Source direct train_batch ports",
            "training_work",
            ("attempted_updates", "returned_updates", "attempted_rows"),
        ),
        (
            "Private direct train_batch ports",
            "reserved_probe/preserved_training_work",
            ("attempted_updates", "returned_updates", "attempted_rows"),
        ),
        (
            "Fresh logical phase (current rows)",
            "reserved_probe/probe",
            ("attempted_updates", "completed_updates", "attempted_training_examples"),
        ),
        (
            "Old TRAIN return logical phase (current rows)",
            "reserved_probe/probe/return_phase",
            ("attempted_updates", "completed_updates", "attempted_training_examples"),
        ),
    ):
        cells = [_cell(source, f"/{path}/{key}") for key in fields]
        for cell in cells:
            if not cell["missing"] and cell["value"] is not None:
                _number(cell["value"], "work count", count=True)
        table["rows"].append(dict(cells=[_label(label), *cells]))
    return table


def _auxiliary_costs(source):
    # Why this: callbacks bypass legacy train_batch ports. Never infer total work
    # from those zeros or call repeated-current control exposures old TRAIN replay.
    summary = _details(
        source,
        tuple(f"{AUX_PATH}/{key}" for key in AUX_COUNTS),
        "auxiliary-work",
        "Auxiliary LS applications — separate from direct ports",
        "Recorded callback admissions, returns, observed applications and unknown applications. attempted_rows counts current rows only; additional_rows_charged is extra exposure, including repeated current rows in current_only. Neither is a total fitted-row reading. Per-event fit_rows below records the actual fitting scope. No fabricated totals or SGD credits; missing categories remain unknown.",
    )
    fields = (
        "phase",
        "status",
        "work_known",
        "applied",
        "current_rows",
        "work/fit_attempts",
        "work/fit_returns",
        "work/fit_rows",
        "work/additional_rows_charged",
        "work/current_rows_used",
        "work/retained_train_rows_used",
        "work/application_unknown",
        "work/unreturned_fit_work_unknown",
        "work",
    )
    events = _section(
        "auxiliary-events",
        "Auxiliary LS receipts",
        "All captured applications, including running and failed attempts. Current rows used can include repeated current rows; retained TRAIN use is a separate reading. The work object opens original packet/snapshot/SGD and other recorded costs. No work is reconstructed; unknown application is not a known non-application.",
        [
            "Phase",
            "Status",
            "Work known",
            "Applied",
            "Current input rows",
            "Fit attempts",
            "Fit returns",
            "Fit rows",
            "Extra rows charged",
            "Current rows used",
            "Retained TRAIN rows used",
            "Application unknown",
            "Unreturned work unknown",
            "Original work",
        ],
    )
    for index, _ in enumerate(_records(source, AUX_PATH + "/events", 32)):
        events["rows"].append(
            dict(cells=[_cell(source, f"/{AUX_PATH}/events/{index}/{key}") for key in fields])
        )
    return summary, events


def project_reserved_probe_report(
    report: dict[str, Any], *, source_label: str, resources=None, resource_label=None
):
    """Build an inert, detached saved view; never reconstruct or run a model."""
    _envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _envelope(resources, resource_label)
    source = deepcopy(report)
    scores, panels = _curves(source)
    available = _section(
        "method-availability",
        "Comparison controls",
        "Only the selected report is loaded. Other controls remain visible as not provided. This is not a matched-capacity or matched-computation comparison.",
        ["Method", "Availability", "Report complete"],
    )
    for method in METHODS:
        active = source["method"] == method
        available["rows"].append(
            dict(
                cells=[
                    _label(method),
                    _label("Loaded" if active else "Not provided"),
                    _cell(source, "/complete") if active else _cell({}, "/absent"),
                ]
            )
        )
    sections = [
        _outcomes(source),
        *_summary(source),
        scores,
        _costs(source),
        *_auxiliary_costs(source),
        available,
        _steps(source),
        _details(
            source,
            (
                "reserved_probe_summary/status",
                "reserved_probe_summary/reason",
                "reserved_probe_summary/qualification",
                "scope",
                "error",
                "source_unchanged",
                "actor_unchanged",
                "numpy_rng_unchanged",
                "python_rng_unchanged",
                "reserved_probe/source_unchanged",
                "reserved_probe/error",
                "final_examples",
                "replay_examples",
                "probe_updates",
                "reserved_probe_rehearsal",
                "reserved_probe_return_once",
                "reserved_probe_source_training",
                "reserved_probe/linear_probe_rehearsal",
                "reserved_probe/linear_probe_return_once",
                "reserved_probe/linear_rehearsal_work/definition",
                "reserved_probe/linear_rehearsal_work/control",
                "reserved_probe/linear_rehearsal_work/return_once",
                "reserved_probe/linear_rehearsal_work/source_array_bytes",
                "reserved_probe/preserved_linear_rehearsal_work/budget",
                "probe_auxiliary_updates",
                "probe_replay_examples",
                "reserved_probe/probe/auxiliary_updates",
                "reserved_probe/probe/return_phase/auxiliary_updates",
                "budget",
                "reserved_probe/sessions",
                "costs",
                "reserved_probe/task_readout_work",
                "reserved_probe/probe/input_diagnostics/work",
                "reserved_probe/probe/reserved_data_bytes",
                "reserved_probe/probe/training_readouts/reserved_data_bytes",
                "reserved_probe/probe/return_phase/reserved_data_bytes",
            ),
            "scope",
            "Costs, guards and limitations",
            "Reported flags do not authenticate source preservation. Original clocks and costs stay separate. Private learning is never a serving promotion. Topology, lineage and sleep data absent from this source remain unknown; no scientific or release claim.",
        ),
    ]
    if resources is not None:
        sections.append(
            _details(
                resources,
                tuple(k.replace("~", "~0").replace("/", "~1") for k in resources),
                "resources",
                "Separate resources — association UNVERIFIED",
                "Separately supplied costs are not automatically associated with the task report.",
                "resources",
            )
        )
    return dict(
        schema_id="ticket_workflow_inspector_v1",
        view_kind="ordinary_reserved_probe",
        source=source,
        source_label=source_label,
        resources=deepcopy(resources),
        resource_label=resource_label,
        protocol_id=_cell(source, "/definition"),
        status=_cell(source, "/complete"),
        reason=_cell(source, "/error"),
        sections=sections,
        recorded_curves=panels,
    )
