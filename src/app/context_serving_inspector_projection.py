"""Project saved guarded contextual learning into existing detached source cells.

Inputs are bounded JSON and optional unverified costs. Outputs are tables for the
existing inspector. No IO, learners, metric recomputation or source authentication.
"""

from copy import deepcopy
from typing import Any

from src.app.coherent_stream_inspector_projection import _cell, _envelope, _fields, _require
from src.app.taskspace_memory_inspector_projection import _details, _records, _section, _typed
from src.app.ticket_workflow_inspector_projection import _mapping

DEFINITION = "guarded_context_serving_v1"
PHASES = ("initial", "after-new", "after-old", "rollback")
CONCEPTS = ("invoice", "password", "parcel", "legacy")


def _actor(record):
    _typed(record, ("generation",), ())
    _typed(
        _mapping(record, "model_state"),
        ("completed_updates", "training_examples", "mistake_updates"),
        (),
    )
    if record.get("actor_version") is not None:
        _require(type(record["actor_version"]) is str, "invalid actor version")


def _episode(row):
    _require(row.get("phase") in PHASES, "invalid contextual phase")
    _require(row.get("policy_version") in ("v1", "v2"), "invalid public policy context")
    _require(row.get("concept") in CONCEPTS, "invalid contextual task")
    _typed(row, (), ("complete",))
    episode = _mapping(row, "episode")
    _typed(
        episode,
        ("max_steps", "external_actions", "training_updates"),
        (
            "complete",
            "success",
            "error_recovery_demonstrated",
            "unreturned_prediction_work_unknown",
            "unvalidated_return_work_unknown",
        ),
    )
    for read in _records(row, "reads", 6):
        _actor(read)
        _typed(read, ("completed_updates",), ())
    _typed(
        _mapping(episode, "work"),
        (
            "prediction_calls_attempted",
            "prediction_calls_returned",
            "prediction_calls_validated",
            "dispatch_attempts",
            "effects_applied",
            "environment_events",
        ),
        (),
    )
    for step in _records(episode, "steps", 6):
        _typed(step, ("attempt",), ())
        for key in (
            "observation_state",
            "effect_state",
            "result",
            "transition",
            "capabilities_at_effect",
        ):
            _mapping(step, key)
        for key in ("observation_state", "effect_state"):
            _typed(_mapping(step, key), ("revision",), ())
        _typed(_mapping(step, "capabilities_at_effect"), ("version",), ())


def _validate(report):
    _require(report.get("definition") == DEFINITION, "unsupported contextual serving report")
    _require(type(report.get("complete")) is bool, "invalid contextual completion")
    _require(
        report.get("error") is None or type(report["error"]) is str, "invalid contextual error"
    )
    _typed(
        report,
        (
            "final_examples",
            "probe_updates",
            "replay_examples",
            "coefficient_bytes",
            "source_record_bytes",
            "input_bytes",
        ),
        (),
    )
    _typed(_mapping(report, "prior_training"), ("updates", "rows", "mistakes"), ())
    _fields(report, ("elapsed_seconds",), count=False, low=0)
    _typed(_mapping(report, "memory"), ("pid", "start_bytes", "peak_bytes", "sample_count"), ())
    _mapping(report, "promotion_policy")
    modes = []
    for case in _records(report, "cases", 2):
        _require(case.get("mode") in ("frozen", "guarded"), "invalid contextual arm")
        modes.append(case["mode"])
        _typed(
            case,
            ("rollback_generation",),
            ("complete", "source_unchanged", "rollback_state_equal_source"),
        )
        _actor(_mapping(case, "final_actor"))
        for key in ("training_work", "budget", "typed_work", "stream", "receipt"):
            _mapping(case, key)
        _typed(
            _mapping(case, "training_work"),
            ("attempted_updates", "attempted_rows", "returned_updates"),
            ("unreturned_work_unknown",),
        )
        _typed(_mapping(case, "budget"), ("updates",), ())
        _typed(
            _mapping(case, "typed_work"), ("attempts", "returns", "readouts", "typed_attempts"), ()
        )
        for episode in _records(case, "episodes", 24):
            _episode(episode)
        for snapshot in _records(case, "publications", 3):
            _actor(snapshot)
        for candidate in _records(case, "candidate_states", 2):
            _typed(candidate, ("wake_updates",), ())
            _typed(
                _mapping(candidate, "state"),
                ("completed_updates", "training_examples", "mistake_updates"),
                (),
            )
        for row in _records(case, "guards", 2):
            _typed(row, ("point",), ())
            guard = _mapping(row, "report")
            decision = _mapping(guard, "decision")
            _typed(decision, (), ("accepted",))
            if decision.get("reasons") is not None:
                _require(
                    type(decision["reasons"]) is list
                    and len(decision["reasons"]) <= 32
                    and all(type(v) is str for v in decision["reasons"]),
                    "invalid guard reasons",
                )
            _fields(
                _mapping(guard, "evidence"),
                (
                    "actor_new_utility",
                    "candidate_new_utility",
                    "actor_old_utility",
                    "candidate_old_utility",
                    "candidate_max_prediction_seconds",
                    "candidate_resource_bytes",
                ),
                count=False,
            )
            _typed(
                _mapping(guard, "evidence"),
                ("candidate_resource_bytes",),
                ("numerically_valid", "actions_safe"),
            )
        for receipt in _records(case, "applied_updates", 2):
            _typed(receipt, ("observed_at", "arrived_at", "applied_at", "update_number"), ())
            _fields(_mapping(receipt, "diagnostic"), ("value",), count=False)
        for key in ("before", "after"):
            _actor(_mapping(_mapping(case, "rollback"), key))
    _require(len(modes) == len(set(modes)), "duplicate contextual arm")


def _arm_rows(source, path, fields, limit, identity, title, note):
    table = _section(identity, title, note, ["Arm", *(label for label, _ in fields)])
    for i, case in enumerate(_records(source, "cases", 2)):
        for j, _ in enumerate(_records(case, path, limit)):
            prefix = f"/cases/{i}/{path}/{j}"
            table["rows"].append(
                dict(
                    cells=[_cell(source, f"/cases/{i}/mode")]
                    + [_cell(source, f"{prefix}/{key}") for _, key in fields]
                )
            )
    return table


def _episode_rows(source, path, fields, identity, title, note):
    table = _section(
        identity,
        title,
        note,
        ["Arm", "Phase", "Policy context", "Task", *(label for label, _ in fields)],
    )
    for i, case in enumerate(_records(source, "cases", 2)):
        for j, episode in enumerate(_records(case, "episodes", 24)):
            root = f"/cases/{i}/episodes/{j}"
            for k, _ in enumerate(_records(episode, path, 6)):
                table["rows"].append(
                    dict(
                        cells=[_cell(source, f"/cases/{i}/mode")]
                        + [
                            _cell(source, f"{root}/{key}")
                            for key in ("phase", "policy_version", "concept")
                        ]
                        + [_cell(source, f"{root}/{path}/{k}/{key}") for _, key in fields]
                    )
                )
    return table


def _arm_details(source, fields, identity, title, note):
    table = _section(identity, title, note, ["Arm", "Recorded field", "Value"])
    from src.app.coherent_stream_inspector_projection import _label

    for i, _ in enumerate(_records(source, "cases", 2)):
        for key in fields:
            table["rows"].append(
                dict(
                    cells=[
                        _cell(source, f"/cases/{i}/mode"),
                        _label(key),
                        _cell(source, f"/cases/{i}/{key}"),
                    ]
                )
            )
    return table


def project_context_serving_report(
    report: dict[str, Any], *, source_label: str, resources=None, resource_label=None
):
    """Retain each actual saved observation; never infer success or authenticity."""
    _envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _envelope(resources, resource_label)
    source = deepcopy(report)
    # Why this: declarative field lists reuse the existing cell/table contract;
    # observations keep original pointers rather than becoming inferred metrics.
    sections = [
        _arm_rows(
            source,
            "episodes",
            (
                ("Phase", "phase"),
                ("Policy context", "policy_version"),
                ("Task", "concept"),
                ("Captured", "episode/complete"),
                ("Task success", "episode/success"),
                ("Stop reason", "episode/stopped_reason"),
            ),
            24,
            "outcomes",
            "Phase task outcomes",
            "Saved flags only. v1legacy is the new task; v2legacy was already known. Earlier failure and restored failure after rollback remain visible. These known TRAIN templates do not establish generalization or recovery from observed old-task loss.",
        ),
        _episode_rows(
            source,
            "reads",
            (
                ("Generation", "generation"),
                ("Actor version", "actor_version"),
                ("Native updates", "completed_updates"),
            ),
            "reads",
            "Atomic actor reads",
            "Each saved typed request identifies its actual actor generation/version/native count. Candidate updates and published actor state are distinct.",
        ),
        _episode_rows(
            source,
            "episode/steps",
            (
                ("Attempt", "attempt"),
                ("Observed phase", "observation_state/phase"),
                ("Observed revision", "observation_state/revision"),
                ("Selected action", "result/selected_id"),
                ("Phase at effect", "effect_state/phase"),
                ("Revision at effect", "effect_state/revision"),
                ("Transition", "transition/status"),
                ("Result phase", "transition/state/phase"),
            ),
            "steps",
            "Complete task steps",
            "Validated action and observed/effect states stay separate. Raw pre-validation returns remain in the complete source; nothing executes from this page.",
        ),
        _arm_rows(
            source,
            "guards",
            (
                ("Update point", "point"),
                ("Accepted", "report/decision/accepted"),
                ("Reasons", "report/decision/reasons"),
                ("Actor new utility", "report/evidence/actor_new_utility"),
                ("Candidate new utility", "report/evidence/candidate_new_utility"),
                ("Actor old utility", "report/evidence/actor_old_utility"),
                ("Candidate old utility", "report/evidence/candidate_old_utility"),
                ("Prediction seconds", "report/evidence/candidate_max_prediction_seconds"),
                ("Resource bytes", "report/evidence/candidate_resource_bytes"),
            ),
            2,
            "guards",
            "Guard decisions",
            "Rejected candidates do not become the serving actor. These are recorded guard decisions on repeated TRAIN templates, not independent final evidence or authenticated safety.",
        ),
        _arm_rows(
            source,
            "candidate_states",
            (
                ("Wake updates", "wake_updates"),
                ("Native updates", "state/completed_updates"),
                ("TRAIN rows", "state/training_examples"),
                ("Mistakes", "state/mistake_updates"),
            ),
            2,
            "candidates",
            "Private candidate checkpoints",
            "Recorded counters include the original source history. No state is reconstructed or restored here.",
        ),
        _arm_rows(
            source,
            "publications",
            (
                ("Generation", "generation"),
                ("Actor version", "actor_version"),
                ("Native updates", "model_state/completed_updates"),
                ("TRAIN rows", "model_state/training_examples"),
                ("Metadata", "metadata"),
            ),
            3,
            "publications",
            "Serving actor observations",
            "Original snapshot order; a repeated generation is not a new publication. Full model state is available through the source report.",
        ),
        _arm_rows(
            source,
            "applied_updates",
            (
                ("Sample", "sample_id"),
                ("Actor version", "actor_version"),
                ("Learner version", "learner_version"),
                ("Observed", "observed_at"),
                ("Arrived", "arrived_at"),
                ("Applied", "applied_at"),
                ("Update", "update_number"),
                ("Diagnostic", "diagnostic"),
            ),
            2,
            "releases",
            "Applied TRAIN receipts",
            "Original label clocks and paid native work; source history does not grant a new budget. This viewer cannot train, select or promote.",
        ),
        _arm_details(
            source,
            (
                "receipt",
                "rollback_generation",
                "rollback_state_equal_source",
                "rollback/before/generation",
                "rollback/before/actor_version",
                "rollback/after/generation",
                "rollback/after/actor_version",
                "final_actor/actor_version",
                "final_actor/model_state/completed_updates",
                "source_unchanged",
            ),
            "rollback",
            "Publication and rollback",
            "Rollback is shown only when recorded. Full-state equality is a saved claim, not authenticated here; missing rollback evidence is unknown, not successful.",
        ),
        _arm_details(
            source,
            ("complete", "error", "training_work", "budget", "typed_work", "stream"),
            "work",
            "Arm work and incomplete evidence",
            "All attempts and unknown work remain paid. A complete capture is not task success; missing costs are not zero.",
        ),
        _details(
            source,
            (
                "prior_training",
                "elapsed_seconds",
                "memory",
                "input_bytes",
                "coefficient_bytes",
                "source_record_bytes",
                "promotion_policy",
                "source_sha256",
                "scope",
                "error",
                "final_examples",
                "probe_updates",
                "replay_examples",
                "sleep_events",
                "sleep_proposals",
            ),
            "scope",
            "History, costs and limitations",
            "Original source training is carried separately. Numeric coefficient bytes exclude snapshots/Python/RSS. Unknown mechanisms remain unknown; no CPC, efficiency, safety, generalization or release benefit is established.",
        ),
    ]
    if resources is not None:
        sections.append(
            _details(
                resources,
                tuple(key.replace("~", "~0").replace("/", "~1") for key in resources),
                "resources",
                "Separate resources — association UNVERIFIED",
                "No source association is inferred for an independently supplied cost file.",
                "resources",
            )
        )
    return dict(
        schema_id="ticket_workflow_inspector_v1",
        source=source,
        source_label=source_label,
        resources=deepcopy(resources),
        resource_label=resource_label,
        protocol_id=_cell(source, "/definition"),
        status=_cell(source, "/complete"),
        reason=_cell(source, "/error"),
        sections=sections,
    )
