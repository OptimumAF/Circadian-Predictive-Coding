"""Present detached workflow32 CLM probe reviews without IO or scientific scoring.

Inputs are completed or partial saved-review JSON envelopes, never native model
state. Outputs use the existing detached source-cell table contract. Validation
checks bounded presentation types; it does not authenticate scientific claims.
"""

from copy import deepcopy
from typing import Any, cast

from src.app.coherent_stream_inspector_projection import _cell, _envelope, _number, _require
from src.app.taskspace_memory_inspector_projection import _details, _rows, _typed

REVIEW_KEYS = frozenset(
    (
        "native_complete",
        "probe",
        "probe_scores",
        "training",
        "source_work",
        "model_work",
        "additional_scoring",
        "effects",
        "read_errors",
        "execution",
    )
)
GROUPS = {"probe_train": 2, "probe_test": 2, "old_task": 12}
CATALOGS = {
    "probe_train": ("defer", "inspect", "route"),
    "probe_test": ("defer", "inspect", "route"),
    "old_task": ("commit", "defer", "inspect", "open", "route"),
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    _require(value is None or type(value) is dict, f"invalid {name} object")
    return {} if value is None else value


def _counts(value: Any, name: str) -> None:
    record = _mapping(value, name)
    for key, count in record.items():
        if key == "unreturned_work_unknown":
            _require(type(count) is bool, f"invalid {name} unknown-work flag")
        else:
            _number(count, name + "/" + key, low=0, count=True)


def _case(row: Any, group: str) -> None:
    _require(type(row) is dict, "invalid case")
    role = "train" if group == "old_task" else group
    _require(row.get("role") == role, "invalid case role")
    request = _mapping(row.get("request"), "request")
    _require(type(request.get("query")) is str, "case query required")
    values = row.get("probabilities")
    ids = CATALOGS[group]
    _require(type(values) is list and len(values) == len(ids), "invalid candidate probabilities")
    for pair, identity in zip(values, ids, strict=True):
        _require(
            type(pair) is list and len(pair) == 2 and pair[0] == identity,
            "invalid candidate order or identity",
        )
        _number(pair[1], "probability", low=0)
        _require(pair[1] <= 1, "probability exceeds one")
    _require(row.get("target_id") in ids and row.get("selected_id") in ids, "invalid action ID")
    _require(type(row.get("correct")) is bool, "invalid correctness flag")
    _require(row["correct"] == (row["target_id"] == row["selected_id"]), "contradictory case IDs")
    for key in ("target_probability", "confidence"):
        if key in row:
            _number(row[key], key, low=0)
            _require(row[key] <= 1, "invalid " + key)
    _typed(row, (), ("candidate_recalled", "obsolete_rule_error"))


def _scores(value: Any, complete: bool) -> None:
    scores = _mapping(value, "probe_scores")
    groups = _mapping(scores.get("groups"), "groups")
    _require(set(groups) <= set(GROUPS), "unknown probe group")
    if complete:
        _require(set(groups) == set(GROUPS), "complete review requires three probe groups")
    for group, record in groups.items():
        _require(type(record) is dict and set(record) <= {"before", "after"}, "invalid points")
        if complete:
            _require(set(record) == {"before", "after"}, "complete endpoints required")
        for point in record.values():
            _require(type(point) is dict, "invalid endpoint")
            cases = point.get("cases")
            _require(type(cases) is list and len(cases) <= GROUPS[group], "too many cases")
            if complete:
                _require(len(cases) == GROUPS[group], "complete cases required")
            for row in cases:
                _case(row, group)
            _metrics(point.get("metrics"))
            fit = _mapping(point.get("training_fit"), "training_fit")
            if fit:
                _require(group == "old_task", "TRAIN fit belongs to old_task")
                rows = fit.get("rows")
                _require(type(rows) is list and len(rows) == len(cases), "invalid fit rows")
                for index, row in enumerate(cast(list[Any], rows)):
                    _fit_row(row, cases[index], index)


def _metrics(value: Any) -> None:
    metrics = _mapping(value, "metrics")
    for key in ("accuracy", "candidate_recall", "brier_mean", "nll_mean", "ece_5bin"):
        if key in metrics and metrics[key] is not None:
            _number(metrics[key], key, low=0)
            if key in ("accuracy", "candidate_recall", "ece_5bin"):
                _require(metrics[key] <= 1, "invalid " + key)
    _typed(metrics, ("zero_target_probability_count",), ())
    if "nll_status" in metrics:
        _require(
            metrics["nll_status"] in ("finite", "infinite_due_to_zero_target_probability"),
            "invalid NLL status",
        )


def _fit_row(row: Any, case: dict[str, Any], index: int) -> None:
    _require(
        type(row) is dict and type(row.get("row")) is int and row["row"] == index,
        "invalid fit row index",
    )
    for key in ("phase", "concept", "target_id", "selected_id", "correct"):
        _require(row.get(key) == case.get(key), "fit/case metadata differs")
    _typed(row, (), ("correct",))
    if "probability_margin" in row:
        _number(row["probability_margin"], "probability margin", low=None)
        _require(-1 <= row["probability_margin"] <= 1, "invalid margin")


def _validate(report: dict[str, Any]) -> None:
    _require(REVIEW_KEYS <= report.keys(), "detached CLM review envelope required")
    _require(type(report.get("complete")) is bool, "review completion required")
    _require(
        report["native_complete"] is None or type(report["native_complete"]) is bool,
        "invalid native completion",
    )
    for key in ("model_work", "additional_scoring", "effects"):
        _require(type(report[key]) is int and report[key] == 0, "invalid review work")
    errors = _mapping(report["read_errors"], "read_errors")
    _require(
        len(errors) <= 8 and all(type(v) is str for v in errors.values()), "invalid read errors"
    )
    _require(report.get("prohibited_imports") == [], "invalid review imports")
    _counts(report.get("counts"), "counts")
    _counts(report["source_work"], "source_work")
    training = _mapping(report["training"], "training")
    _typed(training, ("committed_updates", "logical_clock"), ("complete",))
    if training:
        _require(
            type(training.get("requested_passes")) is int and training["requested_passes"] == 32,
            "only workflow32 reviews supported",
        )
    probe = _mapping(report["probe"], "probe")
    _typed(probe, ("source_update",), ("complete",))
    if probe:
        _require(probe.get("source_update") == 32, "invalid probe source")
    _counts(probe.get("probe_work"), "probe_work")
    scope = _mapping(probe.get("parameter_update_scope"), "parameter_update_scope")
    if scope:
        _require(scope.get("name") == "state_only", "unknown parameter scope")
        _typed(
            scope,
            (
                "optimized_parameter_elements",
                "fixed_parameter_elements",
                "total_parameter_elements",
            ),
            ("fixed_parameters_unchanged",),
        )
    execution = _mapping(report["execution"], "execution")
    _typed(
        execution,
        ("stage_bytes", "log_bytes", "native_attempt_allowance"),
        ("native_process_terminal", "source_pins_held"),
    )
    stream = _mapping(probe.get("stream"), "probe stream")
    _typed(stream, ("completed_updates", "training_pairs"), ())
    _typed(report, ("input_bytes", "peak_rss_bytes"), ("native_structure_and_cost_validation",))
    for record, keys in (
        (report, ("elapsed_seconds", "review_seconds")),
        (execution, ("seconds",)),
    ):
        for key in keys:
            if key in record and record[key] is not None:
                _number(record[key], key, low=0)
    _scores(report["probe_scores"], report["native_complete"] is True)
    if report["native_complete"] is True:
        _require(
            training.get("complete") is True and probe.get("complete") is True,
            "native completion contradicts learner records",
        )


def project_clm_probe_review(report: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    """Copy recorded endpoints and costs; never score, fit or authenticate them."""
    _envelope(report, source_label)
    _validate(report)
    source = deepcopy(report)
    sections = [
        _details(
            source,
            (
                "native_complete",
                "native_error",
                "complete",
                "review_error",
                "read_errors",
                "native_structure_and_cost_validation",
                "execution/error",
                "execution/preflight",
            ),
            "availability",
            "Native outcome and review availability",
            "Native outcome and saved-review completion are distinct. Missing or null is unknown, not zero or success. A failed preflight is not a negative learning result.",
        )
    ]
    for group in GROUPS:
        groups = (source.get("probe_scores") or {}).get("groups", {})
        if group not in groups:
            continue
        for point in ("before", "after"):
            path = f"probe_scores/groups/{group}/{point}"
            section = _rows(
                source,
                path + "/cases",
                (
                    "request/query",
                    "phase",
                    "concept",
                    "role",
                    "target_id",
                    "selected_id",
                    "correct",
                    "target_probability",
                    "confidence",
                    "probabilities",
                ),
                GROUPS[group],
                group + "_" + point,
                f"{group}: {point}",
                "Recorded cases only. Old TRAIN has repeated sealed/staged inputs; probe updates do not update the source or demonstrate deployed retention/return.",
            )
            if group == "old_task":
                section["columns"].append("probability_margin")
                for index, row in enumerate(section["rows"]):
                    row["cells"].append(
                        _cell(source, f"/{path}/training_fit/rows/{index}/probability_margin")
                    )
            sections.append(section)
            sections.append(
                _details(
                    source,
                    tuple(
                        path + "/metrics/" + key
                        for key in (
                            "accuracy",
                            "candidate_recall",
                            "nll_mean",
                            "nll_status",
                            "brier_mean",
                            "ece_5bin",
                            "zero_target_probability_count",
                            "calibration_bins",
                        )
                    ),
                    group + "_" + point + "_metrics",
                    f"{group}: {point} recorded metrics",
                    "Copied values, no recomputation. Accuracy/margins higher and NLL/Brier/ECE lower are the original directions. Tiny splits do not establish calibration or generalization.",
                )
            )
    sections.extend(
        [
            _details(
                source,
                (
                    "training",
                    "source_work",
                    "train_fit",
                    "probe/stream",
                    "probe/probe_work",
                    "probe/parameter_update_scope",
                ),
                "learning",
                "Source training and disposable probe work",
                "Source and probe own separate budgets, consent and clocks. Missing parameter-scope evidence stays unknown. No model, training or final-data access in this viewer.",
            ),
            _details(
                source,
                (
                    "counts",
                    "encoder_work",
                    "elapsed_seconds",
                    "sampled_process_rss",
                    "copy_charge_with_reserve",
                    "execution/seconds",
                    "execution/stage_bytes",
                    "execution/log_bytes",
                    "review_seconds",
                    "peak_rss_bytes",
                    "input_bytes",
                ),
                "costs",
                "Recorded preparation, probe and review costs",
                "Native elapsed/RSS and reader elapsed/RSS are different fields. Copies are not all allocations/FLOPs. Missing costs are not zero; equal exposure is not equal compute.",
            ),
            _details(
                source,
                (
                    "control_comparison",
                    "actual_sha256",
                    "model_work",
                    "additional_scoring",
                    "effects",
                    "preserved_training_prefix",
                    "preserved_train_fit",
                ),
                "scope",
                "Comparison and saved-review scope",
                "Saved claims are not authenticated. Zero review work does not mean zero native work. Source preservation is isolation, not deployed retention or context-return recovery; no CPC or release benefit follows.",
            ),
        ]
    )
    return dict(
        schema_id="clm_probe_inspector_v1",
        source=source,
        source_label=source_label,
        resources=None,
        resource_label=None,
        protocol_id=_cell(source, "/training/objective"),
        status=_cell(source, "/native_complete"),
        reason=_cell(source, "/native_error"),
        sections=sections,
    )
