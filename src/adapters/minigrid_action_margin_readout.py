"""Describe and total optional reductions of existing TRAIN score probabilities.

The checkpoint runner owns TRAIN admission, paid predictions and state guards.
This adapter only names bounded reduction costs; it never runs a learner, probe,
environment or evaluation and does not choose any action/class/checkpoint.
"""

from typing import Any

from src.core.categorical_probability_margins import DEFINITION

WORK_KEYS = (
    "action_margin_calls_attempted",
    "action_margin_calls_completed",
    "action_margin_rows_attempted",
    "action_margin_rows_completed",
    "action_margin_input_array_bytes",
    "action_margin_seconds",
)


def validate_option(flag: bool, public_history: bool, invariant_history: bool) -> None:
    if type(flag) is not bool or (
        flag and (public_history is not True or invariant_history is not True)
    ):
        raise ValueError("train_action_margins requires exact bool and public/invariant history")


def describe(protocol: dict[str, Any]) -> None:
    protocol["train_action_margins"] = dict(
        definition=DEFINITION,
        role="train",
        inputs="already-paid checkpoint probability/target rows; no probe or final inputs",
        scope="all TRAIN checkpoint scores, rows and declared classes; absent summaries null",
        tie_rule="raw zero margin; accuracy retains original first-class argmax ties",
        interpretation="descriptive confidence gaps, not causal interference, capacity or benefit",
    )
    planned = protocol["planned_work"]
    planned["train_action_margins"] = dict(
        reductions=planned["known_raw_score_forwards"],
        rows=planned["known_raw_score_rows"],
        input_array_bytes=planned["known_raw_score_rows"] * 8 * (7 + 1),
        additional_native_forwards=0,
        optimizer_steps=0,
        published_updates=0,
    )


def empty_work() -> dict[str, Any]:
    return dict.fromkeys(WORK_KEYS, 0) | dict(action_margin_seconds=0.0)


def total_work(raw_work: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: sum(work.get(key, 0) for work in raw_work) for key in WORK_KEYS} | dict(
        additional_native_forwards=0, optimizer_steps=0, published_updates=0
    )
