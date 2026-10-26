"""Summarize fixed train-only context-return scores without choosing a run.

Inputs are four A/B score mappings at updates 0/64/128/192, or five A/B/C
mappings extending to256, or three A/C_train endpoints192/256/320, with CE,
accuracy and exact-sequence accuracy.
Outputs are detached JSON primitives
with endpoint values and signed acquisition, loss and return differences. This
module does not train, select checkpoints, evaluate generalization or establish
useful skill. Positive gains are desirable; positive loss and return gap are worse.
"""

from math import isfinite
from typing import Any


_UPDATES = (0, 64, 128, 192)
_METRICS = ("ce", "accuracy", "exact_sequence_accuracy")


def _score(value: Any, metric: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{metric} must be a finite native number excluding bool")
    try:
        score = float(value)
    except OverflowError as error:
        raise ValueError(f"{metric} must be finite") from error
    if not isfinite(score) or score < 0 or (metric != "ce" and score > 1):
        raise ValueError(f"{metric} must be finite and nonnegative, bounded by 1 for accuracy")
    return score


def _endpoints(points: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    if type(points) is not tuple or len(points) != len(_UPDATES):
        raise ValueError("context return requires an exact tuple of four endpoints")
    result = []
    for point, updates in zip(points, _UPDATES, strict=True):
        if type(point) is not dict or set(point) != {"updates", "A", "B"}:
            raise ValueError("each endpoint requires exactly updates, A and B")
        if type(point["updates"]) is not int or point["updates"] != updates:
            raise ValueError("context return endpoints must be exactly 0, 64, 128 and 192")
        row: dict[str, Any] = {"updates": updates}
        for context in ("A", "B"):
            scores = point[context]
            if type(scores) is not dict or set(scores) != set(_METRICS):
                raise ValueError(
                    "context scores require exactly ce, accuracy, exact_sequence_accuracy"
                )
            row[context] = {metric: _score(scores[metric], metric) for metric in _METRICS}
        result.append(row)
    return result


def summarize_context_return(points: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Preserve both endpoint values and signed fixed-protocol differences.

    Why this: CE improves downward, whereas accuracies improve upward. Explicit
    direction keeps negative acquisition and positive forgetting visible without
    reorienting comparisons after seeing their outcomes.
    """
    endpoints = _endpoints(points)
    initial, learned_a, learned_b, returned_a = endpoints
    metrics = {}
    for metric in _METRICS:
        direction = -1 if metric == "ce" else 1
        a0, a64 = initial["A"][metric], learned_a["A"][metric]
        a128, a192 = learned_b["A"][metric], returned_a["A"][metric]
        b64, b128 = learned_a["B"][metric], learned_b["B"][metric]
        metrics[metric] = dict(
            initial_A_gain=direction * (a64 - a0),
            new_B_gain=direction * (b128 - b64),
            old_A_loss=direction * (a64 - a128),
            return_A_gain=direction * (a192 - a128),
            return_gap=direction * (a64 - a192),
        )
    incomplete = learned_a["A"]["exact_sequence_accuracy"] < 1
    return dict(
        endpoint_values=endpoints,
        metrics=metrics,
        incomplete_initial_A_fit=incomplete,
        useful_skill_established=False,
        interpretation=(
            "incomplete_initial_acquisition_and_return_refitting"
            if incomplete
            else "train_only_context_return_refitting_not_skill_restoration"
        ),
    )


def _four_phase_endpoints(points: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    updates = (0, 64, 128, 192, 256)
    if type(points) is not tuple or len(points) != len(updates):
        raise ValueError("four-phase return requires an exact tuple of five endpoints")
    result = []
    for point, count in zip(points, updates, strict=True):
        if type(point) is not dict or set(point) != {"updates", "A", "B", "C"}:
            raise ValueError("four-phase endpoints require exactly updates, A, B and C")
        if type(point["updates"]) is not int or point["updates"] != count:
            raise ValueError("four-phase endpoints must be exactly 0, 64, 128, 192 and 256")
        row: dict[str, Any] = {"updates": count}
        for context in ("A", "B", "C"):
            scores = point[context]
            if type(scores) is not dict or set(scores) != set(_METRICS):
                raise ValueError(
                    "context scores require exactly ce, accuracy, exact_sequence_accuracy"
                )
            row[context] = {metric: _score(scores[metric], metric) for metric in _METRICS}
        result.append(row)
    return result


def summarize_four_phase_return(points: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Summarize fixed A64/B64/C64/A64 train-only acquisition and forgetting.

    C is a logical phase alias for the explicitly collected C_train layout, not
    the historical heldout C layout. The caller supplies those recorded scores;
    this function neither selects data nor verifies its physical provenance.
    """
    endpoints = _four_phase_endpoints(points)
    initial, learned_a, learned_b, learned_c, returned_a = endpoints
    metrics = {}
    for metric in _METRICS:
        direction = -1 if metric == "ce" else 1
        a0, a64 = initial["A"][metric], learned_a["A"][metric]
        a128, a192, a256 = (learned_b["A"][metric], learned_c["A"][metric], returned_a["A"][metric])
        b64, b128 = learned_a["B"][metric], learned_b["B"][metric]
        b192, b256 = learned_c["B"][metric], returned_a["B"][metric]
        c128, c192, c256 = (learned_b["C"][metric], learned_c["C"][metric], returned_a["C"][metric])
        metrics[metric] = dict(
            initial_A_gain=direction * (a64 - a0),
            new_B_gain=direction * (b128 - b64),
            new_C_gain=direction * (c192 - c128),
            old_A_loss_after_B=direction * (a64 - a128),
            old_A_loss_before_return=direction * (a64 - a192),
            old_B_loss_after_C=direction * (b128 - b192),
            old_B_loss_after_return=direction * (b128 - b256),
            old_C_loss_after_return=direction * (c192 - c256),
            return_A_gain=direction * (a256 - a192),
            return_gap=direction * (a64 - a256),
        )
    incomplete = learned_a["A"]["exact_sequence_accuracy"] < 1
    return dict(
        endpoint_values=endpoints,
        metrics=metrics,
        incomplete_initial_A_fit=incomplete,
        useful_skill_established=False,
        interpretation=(
            "incomplete_initial_acquisition_and_four_phase_return_refitting"
            if incomplete
            else "train_only_four_phase_return_refitting_not_skill_restoration"
        ),
    )


def summarize_acquired_return(points: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Preserve fixed A192/C256/A320 train scores and signed differences.

    Complete initial teacher fit is necessary evidence for this raw readout,
    not proof of operational acquisition; the separate live diagnostic checks
    that condition. Neither result selects or changes the training schedule.
    """
    updates = (192, 256, 320)
    if type(points) is not tuple or len(points) != 3:
        raise ValueError("acquired return requires an exact tuple of three endpoints")
    endpoints = []
    for point, count in zip(points, updates, strict=True):
        if type(point) is not dict or set(point) != {"updates", "A", "C_train"}:
            raise ValueError("acquired endpoints require exactly updates, A and C_train")
        if type(point["updates"]) is not int or point["updates"] != count:
            raise ValueError("acquired endpoints must be exactly192,256,320")
        row: dict[str, Any] = {"updates": count}
        for context in ("A", "C_train"):
            scores = point[context]
            if type(scores) is not dict or set(scores) != set(_METRICS):
                raise ValueError(
                    "context scores require exactly ce, accuracy, exact_sequence_accuracy"
                )
            row[context] = {metric: _score(scores[metric], metric) for metric in _METRICS}
        endpoints.append(row)
    initial, learned, returned = endpoints
    metrics = {}
    for metric in _METRICS:
        direction = -1 if metric == "ce" else 1
        metrics[metric] = dict(
            new_C_gain=direction * (learned["C_train"][metric] - initial["C_train"][metric]),
            old_A_loss=direction * (initial["A"][metric] - learned["A"][metric]),
            return_A_gain=direction * (returned["A"][metric] - learned["A"][metric]),
            return_gap=direction * (initial["A"][metric] - returned["A"][metric]),
            new_C_loss_after_return=direction
            * (learned["C_train"][metric] - returned["C_train"][metric]),
        )
    incomplete = initial["A"]["exact_sequence_accuracy"] < 1
    return dict(
        endpoint_values=endpoints,
        metrics=metrics,
        incomplete_initial_A_fit=incomplete,
        useful_skill_established=False,
        interpretation=(
            "inconclusive_incomplete_initial_A_fit_raw_train_return"
            if incomplete
            else "raw_train_only_acquired_task_return_requires_separate_live_acquisition_evidence"
        ),
    )
