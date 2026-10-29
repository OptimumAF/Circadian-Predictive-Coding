"""Report exact input/label constraints on probe TRAIN and old TRAIN readouts.

Reuses the core frequency bounds; owns role/bounds admission and detached row
witnesses only. No learner, evaluation labels, data selection, IO or RNG access.
"""

from dataclasses import asdict
from typing import Any

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.core.categorical_input_bounds import categorical_input_bounds

WORK_KEYS = (
    "bounds_calls_attempted",
    "bounds_calls_returned",
    "bounds_rows",
    "bounds_coordinates",
    "array_bytes_attempted",
    "array_bytes_returned",
)


def _admit(training, readouts):
    if (
        type(training) is not tuple
        or not 1 <= len(training) <= 16
        or type(readouts) is not tuple
        or len(readouts) > 2
    ):
        raise ValueError("input diagnostics require 1..16 probe TRAIN and 0..2 old TRAIN batches")
    entries = 0
    for batch in (*training, *readouts):
        if (
            type(batch) is not AttributionBatch
            or type(batch.features) is not np.ndarray
            or batch.features.ndim != 2
            or not 1 <= batch.features.shape[0] <= 128
            or not 1 <= batch.features.shape[1] <= 296
            or type(batch.targets) is not np.ndarray
            or batch.targets.ndim != 2
        ):
            raise ValueError("input diagnostics require bounded exact batch matrices")
        entries += batch.features.size + batch.targets.size
    if entries > 65536:
        raise ValueError("input diagnostic data exceeds 65536 entries")
    seen: set[str] = set()
    for batches, role in ((training, "probe_train"), (readouts, "train")):
        for batch in batches:
            _validate_batch(batch, role, seen)
            if (
                batch.features.shape[1] != training[0].features.shape[1]
                or batch.num_classes != training[0].num_classes
            ):
                raise ValueError("input diagnostic feature/class schema differs")


def _arrays(batches, field, work):
    return _concatenate([getattr(batch, field) for batch in batches], work)


def _concatenate(arrays, work):
    work["array_bytes_attempted"] += sum(a.nbytes for a in arrays)
    result = np.concatenate(arrays)
    work["array_bytes_returned"] += result.nbytes
    return result


def _bounds(features, targets, classes, work):
    work["bounds_calls_attempted"] += 1
    work["bounds_rows"] += len(features)
    work["bounds_coordinates"] += features.size
    result = categorical_input_bounds(features, targets, classes)
    work["bounds_calls_returned"] += 1
    return asdict(result)


def _witnesses(training, readouts):
    groups: dict[tuple[float, ...], dict[str, list[dict[str, Any]]]] = {}
    for batches, cohort in ((training, "probe_train"), (readouts, "old_train")):
        for index, batch in enumerate(batches):
            for row, features in enumerate(batch.features):
                key = tuple(float(v) for v in features)
                groups.setdefault(key, {"probe_train": [], "old_train": []})[cohort].append(
                    dict(
                        batch_index=index,
                        row_index=row,
                        row_id=batch.row_ids[row],
                        context=batch.context,
                        target_class=int(batch.targets[row, 0]),
                    )
                )
    count = 0
    witnesses: list[dict[str, dict[str, Any]]] = []
    for group in groups.values():
        pair = next(
            (
                (a, b)
                for a in group["probe_train"]
                for b in group["old_train"]
                if a["target_class"] != b["target_class"]
            ),
            None,
        )
        if pair is not None:
            count += 1
            if len(witnesses) < 8:
                witnesses.append(dict(probe_train=pair[0], old_train=pair[1]))
    return count, witnesses


def diagnose_probe_inputs(training, readouts, *, work=None) -> dict[str, Any]:
    """Analyze supplied equal-weight records; unsupported valid data stay unknown.

    Only these two cohorts are used, not probe_test or duplicate return data.
    Bounds describe one fixed row-only predictor on this joint record set, not
    phasewise optimizer exposure/objectives or a stateful/finite learner.
    """
    if work is None:
        work = {}
    if type(work) is not dict or not set(work) <= set(WORK_KEYS):
        raise ValueError("input diagnostics require a plain work-counter dictionary")
    for key in WORK_KEYS:
        work.setdefault(key, 0)
        if type(work[key]) is not int or work[key] != 0:
            raise ValueError("input diagnostics require fresh zero work counters")
    _admit(training, readouts)
    all_batches = (*training, *readouts)
    rows = sum(len(b.features) for b in all_batches)
    report = dict(
        definition="probe_input_constraints_v1",
        status="unavailable",
        reason=None,
        feature_dim=training[0].features.shape[1],
        num_classes=training[0].num_classes,
        input_rows=rows,
        probe_train=None,
        old_train=None,
        joint=None,
        cross_conflicting_input_keys=None,
        witnesses=[],
        witnesses_truncated=False,
        scope="supplied probe_train and old TRAIN readouts, equally weighted records; excludes probe_test/final and duplicate return templates; not optimizer exposure or phasewise training objective",
        qualification="exact equal inputs constrain one fixed row-only predictor and exactly normalized simplex CE; no finite-model attainability, convergence, stateful predictor, task success, causal plasticity or generalization claim",
    )
    reason = (
        "old TRAIN readouts absent"
        if not readouts
        else "categorical3..16only; binary loss semantics unsupported"
        if training[0].num_classes < 3
        else "joint row count exceeds128core bound"
        if rows > 128
        else "features outside core[-1,1]range"
        if any(np.any(b.features < -1) or np.any(b.features > 1) for b in all_batches)
        else None
    )
    if reason is not None:
        report.update(reason=reason, work=dict(work))
        return report
    f, y = _arrays(training, "features", work), _arrays(training, "targets", work)
    old, old_y = _arrays(readouts, "features", work), _arrays(readouts, "targets", work)
    joint = _concatenate((f, old), work)
    joint_y = _concatenate((y, old_y), work)
    report.update(
        probe_train=_bounds(f, y, training[0].num_classes, work),
        old_train=_bounds(old, old_y, training[0].num_classes, work),
        joint=_bounds(joint, joint_y, training[0].num_classes, work),
    )
    count, witnesses = _witnesses(training, readouts)
    report.update(
        status="complete",
        cross_conflicting_input_keys=count,
        witnesses=witnesses,
        witnesses_truncated=count > len(witnesses),
        work=dict(work),
    )
    return report
