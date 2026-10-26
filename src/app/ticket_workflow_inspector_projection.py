"""Present saved native ticket tasks through existing detached source cells.

Owns bounded JSON presentation only. No IO, experiment imports, model state
reconstruction, learning, metric recomputation or source authentication.
"""

from copy import deepcopy
from typing import Any, cast

from src.app.coherent_stream_inspector_projection import _cell, _envelope, _fields, _require
from src.app.taskspace_memory_inspector_projection import (
    _details,
    _records,
    _rows,
    _section,
    _typed,
)

FAMILIES = (
    "qualified_bp_new_contexts_v1",
    "matched_native_ticket_controls_v1",
    "fresh_bp_ticket_demo_v1",
)
CONCEPTS = ("invoice", "parcel", "legacy", "password")


def _mapping(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    _require(value is None or type(value) is dict, f"invalid {key} object")
    return {} if value is None else value


def _validate(report: dict[str, Any]) -> None:
    _require(report.get("definition") in FAMILIES, "unsupported ticket report")
    _require(type(report.get("complete")) is bool, "invalid ticket completion")
    _require(report.get("error") is None or type(report["error"]) is str, "invalid error")
    if report["definition"] == FAMILIES[1]:
        _require(report.get("method") in ("pc", "neutral_cpc", "gated_cpc"), "invalid peer method")
    if report["definition"] == FAMILIES[2]:
        _require(
            report.get("method") == "bp" and report.get("role") == "fresh_train_demo",
            "invalid fresh BP method or role",
        )
    _typed(
        report,
        (
            "final_examples",
            "reserved_probe_updates",
            "replay_examples",
            "external_actions",
            "warmup_updates",
            "batches_per_phase",
        ),
        ("source_seed_actor_unchanged", "numpy_rng_unchanged", "torch_rng_unchanged"),
    )
    for key in ("training_work", "prior_training_work"):
        _typed(
            _mapping(report, key),
            ("attempted_updates", "attempted_rows", "returned_updates"),
            ("unreturned_work_unknown",),
        )
    for key in ("costs", "prior_costs", "budget", "new_budget", "stream", "teacher_reads"):
        _mapping(report, key)
    summary = _mapping(report, "summary")
    for key in (
        "new_context_gain",
        "old_forgetting",
        "old_return_change",
        "new_change_during_return",
    ):
        if key in summary and summary[key] is not None:
            _require(
                type(summary[key]) is int and abs(summary[key]) <= 2**53 - 1,
                "invalid signed task count",
            )
    ordinary = _mapping(summary, "ordinary")
    _require(len(ordinary) <= 3, "too many outcome checkpoints")
    for point, outcomes in ordinary.items():
        _require(point.isdecimal() and str(int(point)) == point, "invalid outcome checkpoint")
        _require(type(outcomes) is dict and set(outcomes) <= set(CONCEPTS), "invalid task concepts")
        _typed(outcomes, (), CONCEPTS)
    previous = -1
    for checkpoint in _records(report, "checkpoints", 3):
        _typed(checkpoint, ("source_updates", "source_rows", "tensor_bytes"), ())
        update = checkpoint.get("source_updates")
        _require(type(update) is int and update > previous, "duplicate or unordered checkpoint")
        previous = cast(int, update)
        state = _mapping(checkpoint, "state")
        _typed(state, ("completed_updates", "training_examples"), ())
        _mapping(state, "policy")
        _mapping(state, "head_state")
        _mapping(checkpoint, "teacher_fit")
    for receipt in _records(report, "applied_updates", 112):
        _typed(receipt, ("observed_at", "arrived_at", "applied_at", "update_number"), ())
        _fields(_mapping(receipt, "diagnostic"), ("value",), count=False)
    for record in _records(report, "readout_work", 3):
        _typed(record, ("source_updates",), ())
        _mapping(record, "work")
    for row in _records(report, "episodes", 20):
        _require(row.get("concept") in CONCEPTS, "invalid episode concept")
        _typed(row, ("source_updates",), ())
        episode = _mapping(row, "episode")
        _typed(
            episode,
            ("max_steps", "training_updates", "external_actions"),
            (
                "complete",
                "success",
                "error_recovery_demonstrated",
                "unreturned_prediction_work_unknown",
                "unvalidated_return_work_unknown",
            ),
        )
        if "scenario" in episode:
            _require(
                episode["scenario"] in ("ordinary", "stale_once", "revoke_commit"),
                "invalid episode scenario",
            )
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
                "capabilities_at_effect",
                "result",
                "transition",
            ):
                _mapping(step, key)
            for key in ("observation_state", "effect_state"):
                _typed(_mapping(step, key), ("revision",), ())
            _typed(_mapping(step, "capabilities_at_effect"), ("version",), ())


def _outcomes(source: dict[str, Any]) -> dict[str, Any]:
    table = _section(
        "outcomes",
        "Old and new task endpoints",
        "Invoice/parcel are old; legacy/password are new input contexts. These are saved flags, not recomputed scores. Initial success can be transfer; zero forgetting without initial competence is not retention.",
        ["Local update", *CONCEPTS],
    )
    ordinary = _mapping(_mapping(source, "summary"), "ordinary")
    from src.app.coherent_stream_inspector_projection import _label

    for point in ordinary:
        table["rows"].append(
            dict(
                cells=[_label("Local " + point)]
                + [_cell(source, f"/summary/ordinary/{point}/{c}") for c in CONCEPTS]
            )
        )
    return table


def _steps(source: dict[str, Any]) -> dict[str, Any]:
    table = _section(
        "steps",
        "Full task steps",
        "Observed state precedes the effect. Selected action is the validated result; the separate raw return record is pre-validation and remains in the source report. Stale revisions and revoked capabilities are recorded at effect time.",
        [
            "Local update",
            "Concept",
            "Scenario",
            "Attempt",
            "Observed phase",
            "Observed revision",
            "Selected action",
            "Phase at effect",
            "Revision at effect",
            "Transition",
            "Result phase",
            "Result revision",
            "Capabilities at effect",
        ],
    )
    for index, row in enumerate(_records(source, "episodes", 20)):
        prefix = f"/episodes/{index}"
        for step_index, _ in enumerate(_records(_mapping(row, "episode"), "steps", 6)):
            base = f"{prefix}/episode/steps/{step_index}"
            paths = [
                prefix + "/source_updates",
                prefix + "/concept",
                prefix + "/episode/scenario",
            ] + [
                base + "/" + key
                for key in (
                    "attempt",
                    "observation_state/phase",
                    "observation_state/revision",
                    "result/selected_id",
                    "effect_state/phase",
                    "effect_state/revision",
                    "transition/status",
                    "transition/state/phase",
                    "transition/state/revision",
                    "capabilities_at_effect",
                )
            ]
            table["rows"].append(dict(cells=[_cell(source, p) for p in paths]))
    return table


def project_ticket_workflow_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Observe saved evidence, preserving unknowns, negatives and original claims."""
    _envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _envelope(resources, resource_label)
    source = deepcopy(report)
    fresh = source["definition"] == FAMILIES[2]
    # Why: the existing table/source-dialog contract already handles offline
    # publication and inert strings; no parallel viewer or runner is needed.
    sections = [
        _outcomes(source),
        _details(
            source,
            tuple(
                "summary/" + key
                for key in (
                    "new_context_gain",
                    "old_forgetting",
                    "old_return_change",
                    "new_change_during_return",
                )
            ),
            "changes",
            "Recorded task changes",
            "Signed success-count changes: new gain higher, old forgetting lower, old/new return change higher. They do not establish generalization or recovery from forgetting.",
        ),
        _rows(
            source,
            "episodes",
            (
                "source_updates",
                "concept",
                "episode/scenario",
                "episode/complete",
                "episode/success",
                "episode/stopped_reason",
                "episode/error_recovery_demonstrated",
            ),
            20,
            "episodes",
            "Complete task episodes",
            "Ordinary success, stale recovery and revoked-commit refusal are distinct outcomes. A revocation slot that never reaches its event is not verified refusal. Negative episodes remain visible.",
        ),
        _steps(source),
        _rows(
            source,
            "episodes",
            (
                "source_updates",
                "concept",
                "episode/scenario",
                "episode/work/prediction_calls_attempted",
                "episode/work/prediction_calls_returned",
                "episode/work/prediction_calls_validated",
                "episode/work/dispatch_attempts",
                "episode/work/effects_applied",
                "episode/work/environment_events",
                "episode/unreturned_prediction_work_unknown",
                "episode/unvalidated_return_work_unknown",
            ),
            20,
            "episode_work",
            "Task work and unknown returns",
            "Attempts remain paid on failure. Complete episode capture is not task success. No simulator or external effects run in this viewer.",
        ),
        _rows(
            source,
            "checkpoints",
            (
                "source_updates",
                "state/completed_updates",
                "source_rows",
                "teacher_fit/global_mean_ce",
                "teacher_fit/probability_margins/argmax_correct_rows",
                "tensor_bytes",
            ),
            3,
            "checkpoints",
            "Native checkpoints and teacher fit",
            "Local phase updates differ from native history. Teacher loss/accuracy are recorded TRAIN diagnostics, not full task success. Tensor bytes exclude other live owners/Python/RSS.",
        ),
        _rows(
            source,
            "checkpoints",
            (
                "source_updates",
                "state/policy",
                "state/head_state/neuron_ids/values",
                "state/head_state/parent_ids/values",
                "state/head_state/neuron_age/values",
                "state/head_state/sleep_events",
            ),
            3,
            "topology",
            "Native topology, lineage and sleep observations",
            "Show actual saved policy/IDs/ages when present. Missing lineage/sleep data remain unknown, including BP. No unrecorded structural changes or sleep acceptance inferred.",
        ),
        _rows(
            source,
            "applied_updates",
            ("sample_id", "observed_at", "arrived_at", "applied_at", "update_number", "diagnostic"),
            112,
            "releases",
            "Applied TRAIN receipts",
            "Original order, delayed label clocks and native diagnostics. BP CE and PC local energy have different definitions. This viewer cannot train or promote.",
        ),
        _rows(
            source,
            "readout_work",
            ("source_updates", "work"),
            3,
            "readouts",
            "Frozen readout accounting",
            "Typed readout attempts are separate from teacher forwards and training; physical/copy overhead may be recorded in a separate cost artifact.",
        ),
        _details(
            source,
            (
                "training_work",
                "prior_training_work",
                "budget",
                "new_budget",
                "costs",
                "prior_costs",
                "teacher_reads",
                "source_seed_actor_unchanged",
                "numpy_rng_unchanged",
                "torch_rng_unchanged",
            ),
            "work",
            "Training, prior charges and resources",
            (
                "Fresh BP owns all warmup/new/return training in this one stream and budget. No historical payload or budget is restored. These are recorded charges; missing costs are not zero."
                if fresh
                else "Prior BP warmup remains charged. PC/CPC have one live stream; BP adaptation restores qualified history. Equal exposure is not equal objectives, compute, optimization or whole-runtime work. Missing costs are not zero."
            ),
        ),
        _details(
            source,
            (
                "final_examples",
                "reserved_probe_updates",
                "replay_examples",
                "external_actions",
                "sleep_proposals",
                "sleep_acceptances",
                "memory_composition",
            ),
            "mechanisms",
            "Inactive and unrecorded mechanisms",
            "Zero is shown only when recorded; absent replay/probe/sleep/memory evidence stays unknown. No final evaluation, probe selection or learning is available from the page.",
        ),
        _details(
            source,
            (
                "definition",
                "method",
                "phases",
                "warmup_updates",
                "batches_per_phase",
                "source_sha256",
                "error",
            )
            + (("role",) if fresh else ()),
            "scope",
            "Source claims and limitations",
            "Saved claims are not authenticated. One synthetic seed and BP-informed dose do not establish a general method ranking or circadian advantage. Raw pre-validation returns remain in the complete source report.",
        ),
    ]
    if resources is not None:
        sections.append(
            _details(
                resources,
                tuple(key.replace("~", "~0").replace("/", "~1") for key in resources),
                "resources",
                "Separately supplied resources â€” association UNVERIFIED",
                "Independent cost file; no source association inferred.",
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
