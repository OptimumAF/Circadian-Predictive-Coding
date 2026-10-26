"""Causal shadow diagnostics at the existing finite schedule's observation port.

Only already-arrived, unapplied ordinary TRAIN is read. Outputs are scalar
feedback plus paid metadata; no learner handle, promotion or final/probe access.
"""

from copy import deepcopy
from time import monotonic

import numpy as np

from src.app.coupled_attribution import _same_value
from src.core.reference_controller import ArrivedTrainingFeedback


def read_candidate_feedback(
    runtime,
    batches,
    *,
    arrived,
    completed,
    tick,
    label_delay,
    actor_version,
    predict,
    paid,
    require_input,
    consumed=None,
    acknowledge_correct=False,
):
    if type(acknowledge_correct) is not bool or (
        consumed is not None
        and (not acknowledge_correct or type(consumed) is not int or consumed < completed)
    ):
        raise ValueError("explicit acknowledgement requires a valid separate consumed count")
    index = completed if consumed is None else consumed
    if arrived <= index:
        return None
    batch = batches[index]
    if batch.role != "train" or not index + label_delay <= tick:
        raise ValueError("candidate feedback requires already-arrived ordinary TRAIN")
    if paid["rows_attempted"] + len(batch.features) > paid.get("row_limit", 84):
        raise ValueError("original candidate diagnostic row allowance exhausted")
    require_input()
    if acknowledge_correct:
        return _read_and_acknowledge(
            runtime,
            batch,
            index,
            completed,
            tick,
            label_delay,
            actor_version,
            predict,
            paid,
            require_input,
        )
    snapshot = runtime.candidate_snapshot()
    return _diagnose(
        snapshot,
        batch,
        index,
        completed,
        tick,
        label_delay,
        actor_version,
        predict,
        paid,
        require_input,
        runtime.candidate_snapshot,
    )


def _read_and_acknowledge(
    runtime,
    batch,
    index,
    completed,
    tick,
    label_delay,
    actor_version,
    predict,
    paid,
    require_input,
):
    feedback = None

    def verify(snapshot, source, label):
        nonlocal feedback
        if (
            source.key != ("attribution", f"batch-{index}")
            or source.observed_at != index
            or label.arrived_at != index + label_delay
            or not _same_value(source.features, batch.features)
            or not _same_value(label.targets, batch.targets)
        ):
            raise ValueError("acknowledgement diagnostic differs from original TRAIN receipt")
        feedback = _diagnose(
            snapshot,
            batch,
            index,
            completed,
            tick,
            label_delay,
            actor_version,
            predict,
            paid,
            require_input,
            runtime._snapshot_candidate,
        )
        return feedback.mistakes == 0

    runtime.acknowledge_ready(
        ("attribution", f"batch-{index}"),
        actor_version=actor_version,
        learner_version="c0",
        wake_updates=completed,
        verify=verify,
    )
    return feedback


def _diagnose(
    snapshot,
    batch,
    index,
    completed,
    tick,
    label_delay,
    actor_version,
    predict,
    paid,
    require_input,
    current_snapshot,
):
    if snapshot.actor_version != actor_version or snapshot.wake_updates != completed:
        raise ValueError("candidate feedback belongs to a stale source or update count")
    pin = deepcopy(snapshot)
    row = dict(
        role="train",
        index=index,
        label_observed_at=index,
        label_arrived_at=index + label_delay,
        read_at=tick,
        base_actor_version=snapshot.actor_version,
        candidate_version=snapshot.learner_version,
        wake_updates=snapshot.wake_updates,
        native_updates=snapshot.state.completed_updates,
        rows=len(batch.features),
        input_bytes=batch.features.nbytes + batch.targets.nbytes,
        complete=False,
    )
    paid["records"].append(row)
    paid["attempts"] += 1
    paid["rows_attempted"] += len(batch.features)
    started = monotonic()
    try:
        prediction = predict(snapshot.state, batch.features.copy())
        paid["returns"] += 1
        if (
            type(prediction) is not np.ndarray
            or prediction.shape != (len(batch.features), 6)
            or not np.isfinite(prediction).all()
            or not ((prediction >= 0) & (prediction <= 1)).all()
            or not np.allclose(prediction.sum(axis=1), 1, atol=1e-6)
        ):
            raise ValueError("invalid candidate TRAIN diagnostic prediction")
        require_input()
        if not _same_value(snapshot, pin) or not _same_value(current_snapshot(), pin):
            raise ValueError("candidate feedback changed its detached or original state")
        mistakes = int(np.sum(np.argmax(prediction, axis=1) != batch.targets[:, 0]))
        # This post-label diagnostic becomes visible now. Original label clocks
        # remain above; it is not a pre-label prediction from the deployed actor.
        result = ArrivedTrainingFeedback(
            "train",
            f"batch-{index}",
            actor_version,
            tick,
            tick,
            mistakes,
            len(batch.features),
        )
        row.update(complete=True, mistakes=mistakes)
        paid["rows_returned"] += len(batch.features)
        return result
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:512]
        raise
    finally:
        row["seconds"] = monotonic() - started
        paid["seconds"] += row["seconds"]
