"""Describe the original observer's within-run causal trace positions.

Inputs are exact positive release/prediction counts already checked by the app.
Outputs order training barriers, each input/target read and successful release,
then every prediction. No IO, artifact verification, wall time, source identity,
independence claim or fresh-role authority belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecordedChronologyNode:
    ordinal: int
    kind: str
    record_index: int | None


def recorded_release_order(
    release_count: int, prediction_count: int
) -> tuple[RecordedChronologyNode, ...]:
    """Retain only order enforced by the source-bound original final observer."""
    if any(type(value) is not int or value <= 0 for value in (release_count, prediction_count)):
        raise ValueError("release chronology requires exact positive event counts")
    rows: list[tuple[str, int | None]] = [("training_before_release", None)]
    # Why this: the observer allows source reads only inside the next release,
    # and predictions only after every release and the whole live-state barrier.
    for index in range(release_count):
        rows.extend(
            (
                ("source_input_read", 2 * index),
                ("source_target_read", 2 * index + 1),
                ("role_release", index),
            )
        )
    rows.append(("training_after_release", None))
    rows.extend(("prediction", index) for index in range(prediction_count))
    rows.append(("training_after_evaluation", None))
    return tuple(
        RecordedChronologyNode(ordinal, kind, index) for ordinal, (kind, index) in enumerate(rows)
    )
