"""Bounded prequential TRAIN classification feedback for a finite stream.

Owns only pending predicted classes, causal/version bindings and scalar costs.
The stream owns label release, native prediction isolation and lifetime budgets.
No heldout data, model loading, training, promotion or new controller lives here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.actor_ports import CandidateState
from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.demonstration_episode import copy_episode, episode_arrays
from src.core.reference_controller import ArrivedTrainingFeedback


@dataclass(frozen=True)
class _Prediction:
    observed_at: int
    actor_version: str
    candidate_version: str
    wake_updates: int
    classes: tuple[int, ...]


def _matches_episode(batch) -> bool:
    if batch.episode is None:
        return True
    # Why: affine callers preserve complete TRAIN provenance even though native
    # predict consumes matrices. Keep those IDs; do not strip metadata to admit
    # feedback or claim a recurrent trajectory inference contract.
    episode = copy_episode(batch.episode)
    features, targets = episode_arrays(episode)
    return (
        episode.role == "train"
        and episode.context == batch.context
        and episode.num_classes == batch.num_classes
        and batch.row_ids
        == tuple(f"{episode.episode_id}:step:{t.step}" for t in episode.transitions)
        and np.array_equal(features, batch.features)
        and np.array_equal(targets, batch.targets)
    )


class ReferencePrequential:
    """At most31 packets/4096 rows; only arrived, current-candidate errors vote."""

    def __init__(self, training):
        if sum(len(b.features) for b in training) > 4096 or any(
            b.role != "train"
            or not _matches_episode(b)
            or (b.num_classes == 2 and not np.isin(b.targets, [0.0, 1.0]).all())
            for b in training
        ):
            raise ValueError(
                "prequential feedback requires bounded hard-class metadata-matching TRAIN"
            )
        self.pending: dict[int, _Prediction] = {}
        self.report: dict[str, Any] = dict(
            fork_attempts=0,
            fork_completed=0,
            prediction_calls_attempted=0,
            prediction_calls_completed=0,
            prediction_rows_attempted=0,
            prediction_rows_completed=0,
            events=[],
            pending_packets=0,
            peak_pending_rows=0,
        )

    def observe(self, index, tick, batch, candidate: CandidateState, predictions) -> None:
        if batch.num_classes == 2:
            if (
                type(predictions) is not np.ndarray
                or predictions.dtype != np.float64
                or predictions.shape != (len(batch.features), 1)
                or not np.isfinite(predictions).all()
                or np.any(predictions < 0)
                or np.any(predictions > 1)
            ):
                raise ValueError(
                    "prequential binary predictions require bounded float64 probabilities"
                )
            classes = predictions[:, 0] >= 0.5
        else:
            validate_class_probabilities(predictions, batch.num_classes)
            if len(predictions) != len(batch.features):
                raise ValueError("prequential prediction row count differs")
            classes = predictions.argmax(axis=1)
        self.pending[index] = _Prediction(
            tick,
            candidate.actor_version,
            candidate.learner_version,
            candidate.wake_updates,
            tuple(map(int, classes)),
        )
        self.report["peak_pending_rows"] = max(
            self.report["peak_pending_rows"], sum(len(p.classes) for p in self.pending.values())
        )
        self.report["pending_packets"] = len(self.pending)

    def arrive(self, index, tick, delay, batch, candidate: CandidateState):
        prediction = self.pending.pop(index)
        if (
            tick != prediction.observed_at + delay
            or prediction.actor_version != candidate.actor_version
        ):
            raise ValueError("prequential feedback clock or actor binding changed")
        mistakes = sum(p != int(t) for p, t in zip(prediction.classes, batch.targets[:, 0]))
        eligible = (
            prediction.candidate_version == candidate.learner_version
            and prediction.wake_updates == candidate.wake_updates
        )
        self.report["events"].append(
            dict(
                batch_index=index,
                observed_at=prediction.observed_at,
                arrived_at=tick,
                actor_version=prediction.actor_version,
                candidate_version=prediction.candidate_version,
                wake_updates=prediction.wake_updates,
                mistakes=mistakes,
                examples=len(prediction.classes),
                eligible=eligible,
                reason="current_candidate" if eligible else "stale_candidate",
            )
        )
        self.report["pending_packets"] = len(self.pending)
        if not eligible:
            return None
        return ArrivedTrainingFeedback(
            "train",
            f"batch-{index}",
            prediction.actor_version,
            prediction.observed_at,
            tick,
            mistakes,
            len(prediction.classes),
        )
