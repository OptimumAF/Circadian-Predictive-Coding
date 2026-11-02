"""Nearest-specific state and work adapters for the existing continuous probe.

No runner, data loading, scheduling, selection, model release or promotion. The
caller owns reserved role admission and the original delayed-label runtime.
"""

from dataclasses import asdict
from typing import Any

import numpy as np

from src.adapters.numpy_nearest_continuous_bc import (
    FixedInputNearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
    _snapshot,
)
from src.core.nearest_continuous_bc import NearestPrediction
from src.core import continuous_bc_diagnostics as diagnostics
from src.adapters.numpy_retained_nearest_bc import (
    NumpyRetainedNearestBCLearner,
    RetainedNearestSnapshot,
    pin_retained_native,
    _native_owners,
    retained_snapshot,
    same_retained_state,
    retained_costs,
)


def is_nearest(owner):
    return type(owner) in (NumpyNearestContinuousBCLearner, NumpyRetainedNearestBCLearner)


def stored_rows(state):
    return (
        state.retained_rows if type(state) is RetainedNearestSnapshot else state.training_examples
    )


def same_state(first, second):
    if type(first) is RetainedNearestSnapshot or type(second) is RetainedNearestSnapshot:
        if type(first) is not RetainedNearestSnapshot or type(second) is not type(first):
            return False
        first = retained_snapshot(first, first.policy, first.retention)
        second = retained_snapshot(second, second.policy, second.retention)
        return same_retained_state(first, second)
    if type(first) is not NearestContinuousBCSnapshot or type(second) is not type(first):
        return False
    first, second = _snapshot(first, first.policy), _snapshot(second, second.policy)
    return (
        first.policy == second.policy
        and first.completed_updates == second.completed_updates
        and first.training_examples == second.training_examples
        and np.array_equal(first.prototype_features, second.prototype_features)
        and np.array_equal(first.prototype_actions, second.prototype_actions)
    )


def native_state(owner, expected=None):
    if type(owner) is NumpyRetainedNearestBCLearner:
        retained_state = pin_retained_native(owner).state
        if expected is not None and not same_state(expected, retained_state):
            raise RuntimeError("retained probe changed original native state")
        return retained_state
    if type(owner) is not NumpyNearestContinuousBCLearner:
        raise ValueError("nearest probe requires the exact native owner")
    policy = owner._policy
    if type(policy) is not FixedInputNearestContinuousBCPolicy or policy.training_role != "train":
        raise ValueError("nearest probe requires an explicit fixed-input TRAIN policy")
    state = _snapshot(owner._state, policy)
    if not state.training_examples:
        raise ValueError("nearest probe requires nonempty released source prototypes")
    if expected is not None and not same_state(expected, state):
        raise RuntimeError("nearest probe changed original native state")
    return state


def owner_identity(owner):
    if type(owner) is NumpyRetainedNearestBCLearner:
        return (
            *_native_owners(owner),
            owner._policy.spec,
            owner._state.policy.spec,
            owner._policy.spec.action_units,
            owner._policy.spec.lower_bounds,
            owner._policy.spec.upper_bounds,
        )
    fields: tuple[Any, ...] = (owner._state, owner._policy)
    if type(owner) is NumpyNearestContinuousBCLearner:
        fields += (
            owner._policy.spec,
            owner._state.prototype_features,
            owner._state.prototype_actions,
        )
    return fields


def append_work(before, rows, *, after=None):
    columns = before.policy.spec.observation_dim + len(before.policy.spec.action_units)
    work = dict(
        append_operations=1,
        appended_rows=rows,
        appended_numeric_bytes=rows * columns * 8,
        optimizer_updates=0,
        gradient_evaluations=0,
        dense_forward_passes=0,
        source_derived_counts=True,
    )
    if type(before) is RetainedNearestSnapshot:
        work.update(
            retention_before=retained_costs(before),
            retention_after=None if after is None else retained_costs(after),
        )
    return work


def predict(readout, state, features, work, *, selection=None):
    """Charge attempted retrieval before its delegate; validate the exact receipt."""
    rows, prototypes = len(features), stored_rows(state)
    counters = work.setdefault(
        "nearest_retrieval",
        dict(
            attempts=0,
            returns=0,
            rows_attempted=0,
            rows_returned=0,
            distance_candidates_upper_bound=0,
            distance_feature_values_upper_bound=0,
            distance_candidates_completed=0,
            distance_feature_values_completed=0,
        ),
    )
    candidates = rows * prototypes
    values = candidates * state.policy.spec.observation_dim
    counters["attempts"] += 1
    counters["rows_attempted"] += rows
    counters["distance_candidates_upper_bound"] += candidates
    counters["distance_feature_values_upper_bound"] += values
    result = readout.predict_with_work(features)
    if (
        type(result) is not NearestPrediction
        or type(result.distance_candidates) is not int
        or result.distance_candidates != candidates
        or type(result.distance_feature_values) is not int
        or result.distance_feature_values != values
        or type(result.selected_indices) is not tuple
        or len(result.selected_indices) != rows
        or any(type(i) is not int or not 0 <= i < prototypes for i in result.selected_indices)
    ):
        raise RuntimeError("nearest probe prediction work receipt differs")
    counters["returns"] += 1
    counters["rows_returned"] += rows
    counters["distance_candidates_completed"] += candidates
    counters["distance_feature_values_completed"] += values
    if selection is not None:
        selection.extend(result.selected_indices)
    return result.actions


def training_owner_identity(owner):
    """Also bind nested policy/spec containers for the training-only diagnostic."""
    policy = owner._state.policy
    return (
        *owner_identity(owner),
        policy,
        policy.spec,
        owner._policy.spec.action_units,
        owner._policy.spec.lower_bounds,
        owner._policy.spec.upper_bounds,
        policy.spec.action_units,
        policy.spec.lower_bounds,
        policy.spec.upper_bounds,
    )


def record_training_consistency(before, training, report, work, boundary, publish, seal):
    """Reuse the100-row kernel; retain every exact probe/probe and probe/source pair."""
    query = np.asarray([row.observation for row in training.transitions], dtype=np.float64)
    targets = np.asarray([row.action for row in training.transitions], dtype=np.float64)
    size = len(query)
    pair_work = diagnostics._InputTargetConsistencyWork()
    counts = work.setdefault("training_consistency", dict(attempts=0, returns=0))
    blocks: list[tuple[int | None, Any, Any]] = [(None, query, targets)]
    for start in range(0, stored_rows(before), 100 - size):
        stop = start + 100 - size
        blocks.append(
            (
                start,
                np.concatenate((query, before.prototype_features[start:stop])),
                np.concatenate((targets, before.prototype_actions[start:stop])),
            )
        )
    for block_start, features, actions in blocks:
        boundary()
        counts["attempts"] += 1
        publish()
        try:
            result = diagnostics.continuous_input_target_consistency(
                features, actions, before.policy.spec, _work=pair_work
            )
            counts["returns"] += 1
        finally:
            counts.update(asdict(pair_work))
            publish()
        pairs = []
        for pair in result.pairs:
            if (
                pair.duplicate_input
                and pair.first_row < size
                and (block_start is None or pair.second_row >= size)
            ):
                pairs.append(
                    dict(
                        probe_row=pair.first_row,
                        other_origin="probe_train" if block_start is None else "source",
                        other_row=pair.second_row
                        if block_start is None
                        else block_start + pair.second_row - size,
                        pair=asdict(pair),
                    )
                )
        report["training_consistency"].append(
            dict(
                source_start=block_start,
                rows=len(features),
                exact_pairs=pairs,
                work=dict(result.work),
            )
        )
        seal("training_consistency", len(report["training_consistency"]) - 1)
        boundary()


def summarize(report, before, owned, readouts):
    work = report["work"]
    native = report["native_updates"]
    returned_rows = sum(item["training_rows"] for item in native)
    columns = before.policy.spec.observation_dim + len(before.policy.spec.action_units)
    if type(before) is RetainedNearestSnapshot:
        # A paid but unvalidated/lost append cannot establish a resulting bank.
        after = (
            (native[-1].get("work") or {}).get("retention_after")
            if native
            else None
            if work["training_attempts"]
            else retained_costs(before)
        )
        work.update(
            native_returns=len(native),
            training_rows_native_returned=returned_rows,
            training_rows_completed=len(report["applied_updates"]) * len(owned.training.transitions)
            if report["receipt_observation_complete"]
            else None,
            unreturned_append_possible=work["training_attempts"] != len(native),
            append_attempts_permanently_spent=work["training_attempts"],
            append_slots_remaining=max(0, owned.updates - work["training_attempts"]),
            optimizer_updates=0,
            gradient_evaluations=0,
            dense_forward_passes=0,
            prototype_rows_before=before.retained_rows,
            candidate_rows_known_after_return=None if after is None else after["retained_rows"],
            prototype_numeric_bytes_per_owner_before=before.retained_rows * columns * 8,
            candidate_numeric_bytes_known_after_return=None
            if after is None
            else after["numeric_bytes"],
            returned_input_numeric_bytes=returned_rows * columns * 8,
            reserved_numeric_bytes=(
                len(owned.training.transitions)
                + (0 if owned.evaluation is None else len(owned.evaluation.transitions))
            )
            * columns
            * 8,
            old_readout_input_numeric_bytes=sum(len(e.transitions) for e in readouts) * columns * 8,
            extra_model_owners=3,
            scope="bounded nearest exposure/search/storage and returned append receipts; excludes validation, copies, hashing, Python and RSS",
        )
        report["retention"] = dict(source=retained_costs(before), candidate_after_return=after)
        report["source_state"] = dict(
            policy=asdict(before.policy),
            rows=before.retained_rows,
            training_examples=before.training_examples,
        )
        return
    work.update(
        native_returns=len(native),
        training_rows_native_returned=returned_rows,
        training_rows_completed=len(report["applied_updates"]) * len(owned.training.transitions)
        if report["receipt_observation_complete"]
        else None,
        unreturned_append_possible=work["training_attempts"] != len(native),
        append_attempts_permanently_spent=work["training_attempts"],
        append_slots_remaining=max(0, owned.updates - work["training_attempts"]),
        optimizer_updates=0,
        gradient_evaluations=0,
        dense_forward_passes=0,
        prototype_rows_before=before.training_examples,
        candidate_rows_known_after_return=before.training_examples + returned_rows,
        prototype_numeric_bytes_per_owner_before=before.training_examples * columns * 8,
        candidate_numeric_bytes_known_after_return=(before.training_examples + returned_rows)
        * columns
        * 8,
        appended_numeric_bytes_known_after_return=returned_rows * columns * 8,
        reserved_numeric_bytes=(
            len(owned.training.transitions)
            + (0 if owned.evaluation is None else len(owned.evaluation.transitions))
        )
        * columns
        * 8,
        old_readout_input_numeric_bytes=sum(len(e.transitions) for e in readouts) * columns * 8,
        extra_model_owners=3,
        scope="nearest append/retrieval returns and logical prototype payload; excludes validation/snapshots/copies/temporary arrays/Python containers/process RSS; no dense neural arithmetic",
    )
    report["source_state"] = dict(policy=asdict(before.policy), rows=before.training_examples)
