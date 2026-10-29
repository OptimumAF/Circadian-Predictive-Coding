"""Present saved ticket score diagnoses through existing offline source cells.

Owns bounded JSON presentation, including partial failures. No IO, diagnosis,
model restoration, scoring, metric recomputation or execution authentication.
"""

from copy import deepcopy
from typing import Any

from src.app.coherent_stream_inspector_projection import (
    _cell,
    _envelope,
    _fields,
    _number,
    _require,
)
from src.app.taskspace_memory_inspector_projection import _details, _records, _section, _typed

FAMILY = "ticket_live_saved_score_diagnosis_v1"
_CONCEPTS = ("invoice", "parcel", "legacy", "password")
_ARMS = ("clean", "retained_corruption")


def _mapping(record: dict[str, Any], key: str, required: bool = False) -> dict[str, Any]:
    value = record.get(key, {})
    _require(type(value) is dict and (not required or key in record), f"invalid {key} object")
    return value


def _text(record: dict[str, Any], key: str, required: bool = False) -> None:
    if key in record or required:
        _require(type(record.get(key)) is str and bool(record[key]), f"invalid {key} text")


def _ordered(rows: list[dict[str, Any]], key: str, expected: tuple[str, ...], complete: bool):
    values = [row.get(key) for row in rows]
    _require(
        (values == list(expected) if complete else values == [v for v in expected if v in values]),
        f"invalid {key} order or duplicates",
    )


def _decisions(trace: dict[str, Any], complete: bool) -> None:
    _typed(trace, (), ("success",))
    if complete:
        _require(type(trace.get("success")) is bool, "missing task outcome")
    decisions = _records(trace, "decisions", 6)
    _require(not complete or bool(decisions), "missing decisions")
    for decision in decisions:
        _fields(decision, ("attempt",), count=True, required=complete)
        if "attempt" in decision:
            _require(
                type(decision["attempt"]) is int and 1 <= decision["attempt"] <= 6,
                "invalid attempt",
            )
        _text(decision, "selected_id", complete)
        for key in ("selected_raw_margin", "selected_probability_margin"):
            _require(not complete or key in decision, "missing recorded margin")
            if decision.get(key) is not None:
                _number(decision[key], key)
        scores = _records(decision, "scores", 8)
        _require(not complete or bool(scores), "missing saved scores")
        identifiers = []
        for score in scores:
            _text(score, "candidate_id", True)
            identifiers.append(score["candidate_id"])
            _fields(score, ("raw_score", "probability"), count=False, required=True)
            _require(type(score.get("raw_score")) in (int, float), "missing raw score")
            _require(
                type(score.get("probability")) in (int, float) and 0 <= score["probability"] <= 1,
                "invalid probability",
            )
        _require(len(set(identifiers)) == len(identifiers), "duplicate saved score")


def _arm(arm: dict[str, Any], complete: bool) -> None:
    for key in ("recipe", "corruption", "outcomes", "recorded_training"):
        _mapping(arm, key, complete)
    outcomes = _mapping(arm, "outcomes")
    _fields(
        outcomes,
        ("new_before", "new_after", "old_before", "old_after"),
        count=True,
        required=complete,
    )
    training = _mapping(arm, "recorded_training")
    _typed(
        training,
        ("attempted_updates", "returned_updates", "attempted_rows"),
        ("unreturned_work_unknown",),
    )
    traces = _records(arm, "traces", 4)
    _ordered(traces, "concept", _CONCEPTS, complete)
    for trace in traces:
        _decisions(trace, complete)


def _comparison(concept: dict[str, Any], complete: bool) -> None:
    _fields(concept, ("paired_decisions",), count=True, required=complete)
    _fields(_mapping(concept, "unpaired_steps", complete), _ARMS, count=True, required=complete)
    _require(not complete or "first_divergence" in concept, "missing first difference")
    difference = concept.get("first_divergence")
    if difference is None:
        return
    _require(type(difference) is dict, "invalid first difference")
    _fields(difference, ("attempt",), count=True, required=complete)
    _mapping(difference, "input", complete)
    for key in ("clean_selected", "retained_corruption_selected"):
        _text(difference, key, complete)
    if all(key in difference for key in ("clean_selected", "retained_corruption_selected")):
        _require(
            difference["clean_selected"] != difference["retained_corruption_selected"],
            "first difference has equal actions",
        )
    _fields(
        _mapping(difference, "noise_action_minus_clean_raw", complete),
        _ARMS,
        count=False,
        required=complete,
    )


def _validate(source: dict[str, Any]) -> None:
    _require(source.get("definition") == FAMILY, "unsupported score report")
    _require(type(source.get("complete")) is bool, "invalid diagnosis completion")
    complete = source["complete"]
    _require(source.get("error") is None or type(source["error"]) is str, "invalid error")
    if complete or "role" in source:
        _require(source.get("role") == "saved_candidate_scores_only", "invalid saved-score role")
    for key in ("qualification", "source_scope", "source_sha256"):
        if source.get(key) is not None:
            _text(source, key)
    work = _mapping(source, "work")
    _fields(work, tuple(work), count=True)
    _fields(source, ("elapsed_seconds",), count=False, low=0)
    arms = _records(source, "arms", 2)
    _ordered(arms, "name", _ARMS, complete)
    for arm in arms:
        _arm(arm, complete)
    concepts = _records(source, "concepts", 4)
    _ordered(concepts, "concept", _CONCEPTS, complete)
    for concept in concepts:
        _comparison(concept, complete)


def _arm_tables(source: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = _section(
        "outcomes",
        "Old and new task outcomes",
        "Invoice/parcel are old; legacy/password are new. Private candidate outcomes are separate from deployed actor behavior. Initial success may be transfer. All counts are copied, without rescoring.",
        ["Arm", "New before", "New after", "Old before", "Old after", "Full recorded outcomes"],
    )
    margins = _section(
        "margins",
        "Recorded selected-action gaps",
        "Every saved decision stays visible, including password separation loss. Gaps are not calibrated confidence. Null means a recorded null (for example, no competitor); unknown means absent evidence.",
        [
            "Arm",
            "Context",
            "Task success",
            "Attempt",
            "Selected action",
            "Raw gap",
            "Probability gap",
        ],
    )
    scores = _section(
        "scores",
        "All recorded candidate scores",
        "Saved scalar readouts only; no model or diagnostic computation. The page does not authenticate their execution or recompute rankings.",
        ["Arm", "Context", "Attempt", "Candidate", "Raw score", "Probability"],
    )
    feedback = _section(
        "feedback",
        "Training recipe, corruption and exposure",
        "Explicit experimental TRAIN label corruption is not verified teacher feedback. Equal exposure does not establish matched physical computation or a benefit from incorrect labels.",
        ["Arm", "Recipe", "Corruption", "Recorded training"],
    )
    for i, arm in enumerate(_records(source, "arms", 2)):
        p = f"/arms/{i}"
        name = _cell(source, p + "/name")
        outcomes["rows"].append(
            dict(
                cells=[name]
                + [
                    _cell(source, p + "/outcomes/" + k)
                    for k in ("new_before", "new_after", "old_before", "old_after")
                ]
                + [_cell(source, p + "/outcomes")]
            )
        )
        feedback["rows"].append(
            dict(
                cells=[name]
                + [
                    _cell(source, p + "/" + k)
                    for k in ("recipe", "corruption", "recorded_training")
                ]
            )
        )
        for j, trace in enumerate(_records(arm, "traces", 4)):
            t = p + f"/traces/{j}"
            for k, decision in enumerate(_records(trace, "decisions", 6)):
                d = t + f"/decisions/{k}"
                margins["rows"].append(
                    dict(
                        cells=[
                            deepcopy(name),
                            _cell(source, t + "/concept"),
                            _cell(source, t + "/success"),
                        ]
                        + [
                            _cell(source, d + "/" + f)
                            for f in (
                                "attempt",
                                "selected_id",
                                "selected_raw_margin",
                                "selected_probability_margin",
                            )
                        ]
                    )
                )
                for n, _ in enumerate(_records(decision, "scores", 8)):
                    scores["rows"].append(
                        dict(
                            cells=[
                                deepcopy(name),
                                _cell(source, t + "/concept"),
                                _cell(source, d + "/attempt"),
                            ]
                            + [
                                _cell(source, d + f"/scores/{n}/" + f)
                                for f in ("candidate_id", "raw_score", "probability")
                            ]
                        )
                    )
    return [outcomes, margins, scores, feedback]


def _differences(source: dict[str, Any]) -> dict[str, Any]:
    table = _section(
        "differences",
        "First difference on a common observation",
        "Only the identical saved request, state and capabilities prefix was paired. Pairing stopped at the first different action; later worlds are unpaired. A recorded null first difference is distinct from missing evidence. The signed gap is noisy-action minus clean-action within each arm, not a treatment effect.",
        [
            "Context",
            "Paired decisions",
            "Recorded first difference",
            "Attempt",
            "Common input",
            "Clean action",
            "Corruption action",
            "Clean signed gap",
            "Corruption signed gap",
            "Unpaired steps",
        ],
    )
    fields = (
        "concept",
        "paired_decisions",
        "first_divergence",
        "first_divergence/attempt",
        "first_divergence/input",
        "first_divergence/clean_selected",
        "first_divergence/retained_corruption_selected",
        "first_divergence/noise_action_minus_clean_raw/clean",
        "first_divergence/noise_action_minus_clean_raw/retained_corruption",
        "unpaired_steps",
    )
    for i, _ in enumerate(_records(source, "concepts", 4)):
        table["rows"].append(dict(cells=[_cell(source, f"/concepts/{i}/" + key) for key in fields]))
    return table


def project_ticket_score_report(
    report: dict[str, Any],
    *,
    source_label: str,
    resources: dict[str, Any] | None = None,
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Detach observations into the existing table/source-dialog contract."""
    _envelope(report, source_label)
    _validate(report)
    _require((resources is None) == (resource_label is None), "resources require a label")
    if resources is not None:
        _envelope(resources, resource_label)
    source = deepcopy(report)
    # Why this: reuse the existing renderer/writer; presentation cannot admit training.
    outcomes, margins, scores, feedback = _arm_tables(source)
    sections = [
        outcomes,
        _differences(source),
        margins,
        scores,
        feedback,
        _details(
            source,
            ("work", "elapsed_seconds"),
            "work",
            "Recorded diagnostic work",
            "Already-paid work, not new model work. Training exposure is separate above. Absent computation or memory costs are unknown, not zero.",
        ),
        _details(
            source,
            (
                "definition",
                "role",
                "complete",
                "error",
                "qualification",
                "source_scope",
                "source_sha256",
            ),
            "scope",
            "Source claims and limitations",
            "Captured-byte snapshot only; no postflight seal or execution authentication. No causal mechanism, beneficial corruption, robustness, generalization, circadian advantage or release claim. This page cannot train, tune, select probes or promote.",
        ),
    ]
    if resources is not None:
        sections.append(
            _details(
                resources,
                tuple(k.replace("~", "~0").replace("/", "~1") for k in resources),
                "resources",
                "Separate resources — association UNVERIFIED",
                "No source association inferred.",
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
