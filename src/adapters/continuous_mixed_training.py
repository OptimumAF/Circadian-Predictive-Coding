"""Explicit mixed-gripper BP copies and bounded SGD return receipts.

Reuse the existing continuous learner and native XYZ-MSE/gripper-BCE gradient.
Successful work is source-derived because train_batch returns only a diagnostic;
no extra prediction, loss, gradient, initialization, fit or head solve is run.
Optional fixed row weights are owned supervision configuration, not learner state.
"""

from dataclasses import asdict, replace
import json
import math

import numpy as np

from src.adapters.numpy_continuous_bc import (
    ContinuousBCSnapshot,
    NumpyContinuousBCLearner,
    _policy,
    _snapshot,
)
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.core import continuous_bc as bc


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _admit_source(source):
    if type(source) is not NumpyContinuousBCLearner:
        raise ValueError("mixed continuation requires an exact continuous BP learner")
    native, policy = source._state, source._policy
    before = _snapshot(native, policy)
    axis = before.policy.binary_gripper_axis
    if (
        before.policy.training_role != "train"
        or before.policy.spec.observation_dim > 64
        or before.policy.hidden_dim > 64
        or not (axis is None or (type(axis) is int and axis == 3))
    ):
        raise ValueError("mixed continuation requires TRAIN BP D/H<=64 and axis None or3")
    bc._binary_gripper_axis(3, before.policy.spec)
    return native, policy, before


def _require_unchanged(owner, native, policy, expected, statistics):
    if (
        type(owner) is not NumpyContinuousBCLearner
        or owner._state is not native
        or owner._policy is not policy
        or policy.observation_standardization is not statistics
        or not same_continuous_state(expected, _snapshot(owner._state, owner._policy))
    ):
        raise RuntimeError("original mixed continuation source changed")


def _require_disjoint(first, second):
    if any(
        np.shares_memory(a, b)
        for a in first.parameters + first.velocity
        for b in second.parameters + second.velocity
    ):
        raise RuntimeError("mixed continuation copied model arrays alias source")


def mixed_training_source(source: NumpyContinuousBCLearner) -> NumpyContinuousBCLearner:
    """Own an RNG-free private mixed copy, retaining original state and policy.

    An already mixed source preserves its objective. A common-MSE source changes
    only the private copy's explicit gripper objective; momentum is not reset.
    """
    native, original_policy, before = _admit_source(source)
    statistics = original_policy.observation_standardization
    child = source.fork()
    _require_unchanged(source, native, original_policy, before, statistics)
    if type(child) is not NumpyContinuousBCLearner or child is source:
        raise RuntimeError("mixed continuation requires a distinct native fork")
    forked = _snapshot(child._state, child._policy)
    if not same_continuous_state(before, forked):
        raise RuntimeError("mixed continuation fork changed the original state")
    _require_disjoint(native, child._state)
    candidate_policy = _policy(replace(before.policy, binary_gripper_axis=3))
    candidate = ContinuousBCSnapshot(
        candidate_policy,
        before.parameters,
        before.velocity,
        before.completed_updates,
        before.training_examples,
    )
    child._policy = candidate_policy
    child._state = _snapshot(candidate, candidate_policy)
    _require_disjoint(native, child._state)
    _require_unchanged(source, native, original_policy, before, statistics)
    return child


_ONES = (
    "gradient_evaluations_attempted",
    "gradient_evaluations_completed",
    "parameter_proposals_attempted",
    "parameter_proposals_completed",
    "published_updates",
    "normalization_attempts",
    "normalization_completed",
    "forward_passes",
    "backward_passes",
)
_ROW_COUNTS = ("normalization_rows", "forward_rows", "backward_rows", "binary_gripper_loss_rows")
_WORK_KEYS = {
    *_ONES,
    *_ROW_COUNTS,
    "normalization_values",
    "gradient_values",
    "gradient_numeric_bytes",
    "known_dense_multiply_accumulates",
    "source_derived_counts",
    "native_work_returned",
}


_WEIGHT_WORK_KEYS = {
    "objective_weight_values",
    "objective_weight_numeric_bytes",
    "native_row_weight_mass_admissions",
}


def _successful_work(policy, rows, *, weighted=False):
    dimension, hidden, actions = (
        policy.spec.observation_dim,
        policy.hidden_dim,
        len(policy.spec.action_units),
    )
    parameters = dimension * hidden + hidden + hidden * actions + actions
    work = dict(
        **{key: 1 for key in _ONES},
        **{key: rows for key in _ROW_COUNTS},
        normalization_values=rows * dimension,
        gradient_values=parameters,
        gradient_numeric_bytes=8 * parameters,
        known_dense_multiply_accumulates=rows * (2 * dimension * hidden + 3 * hidden * actions),
        source_derived_counts=True,
        native_work_returned=False,
    )
    if weighted:
        work.update(
            objective_weight_values=rows,
            objective_weight_numeric_bytes=8 * rows,
            native_row_weight_mass_admissions=2,
        )
    return work


def _admit_maximum_rows(value: int) -> int:
    """Only the existing or explicitly expanded mixed TRAIN scope is supported."""
    if type(value) is not int or value not in (148, 198):
        raise ValueError("mixed maximum_rows must be the exact integer148 or198")
    return value


def _owned_weights(value, *, maximum_rows: int = 148):
    maximum_rows = _admit_maximum_rows(maximum_rows)
    if value is None:
        return None
    if type(value) is not np.ndarray or value.ndim != 1 or not 1 <= len(value) <= maximum_rows:
        raise ValueError(
            f"mixed weights require a positive float64 vector of1..{maximum_rows} rows"
        )
    owned = bc._row_weights(value, len(value))
    bc._row_masses(owned)
    return owned


def _validate_sink(reports, limit, *, maximum_rows: int = 148):
    maximum_rows = _admit_maximum_rows(maximum_rows)
    if type(reports) is not list or len(reports) > limit:
        raise ValueError("mixed reports require a bounded exact shared list")
    mode, fixed_weights = None, None
    for item in reports:
        weighted = type(item) is dict and "objective_row_weights" in item
        if mode is not None and weighted != mode:
            raise ValueError("mixed receipts cannot combine weighted and legacy objectives")
        mode = weighted
        expected_keys = {"diagnostic", "training_rows", "work"}
        if weighted:
            expected_keys.add("objective_row_weights")
        if (
            type(item) is not dict
            or set(item) != expected_keys
            or type(item["training_rows"]) is not int
            or not 1 <= item["training_rows"] <= maximum_rows
            or type(item["diagnostic"]) is not dict
            or set(item["diagnostic"]) != {"definition", "value"}
            or item["diagnostic"]["definition"]
            != (bc.WEIGHTED_MIXED_OBJECTIVE if weighted else bc.MIXED_OBJECTIVE)
            or type(item["diagnostic"]["value"]) is not float
            or not math.isfinite(item["diagnostic"]["value"])
            or item["diagnostic"]["value"] < 0
            or type(item["work"]) is not dict
            or set(item["work"]) != (_WORK_KEYS | _WEIGHT_WORK_KEYS if weighted else _WORK_KEYS)
        ):
            raise ValueError("mixed receipt differs from one returned SGD update")
        work, rows = item["work"], item["training_rows"]
        if weighted:
            values = item["objective_row_weights"]
            if (
                type(values) is not list
                or len(values) != rows
                or any(type(value) is not float for value in values)
            ):
                raise ValueError("weighted mixed receipt requires exact raw native float values")
            if maximum_rows == 148:
                _owned_weights(np.asarray(values, dtype=np.float64))
            else:
                _owned_weights(np.asarray(values, dtype=np.float64), maximum_rows=maximum_rows)
            payload = _json(values)
            if fixed_weights is not None and payload != fixed_weights:
                raise ValueError("weighted mixed receipts require one fixed raw weight vector")
            fixed_weights = payload
            if (
                work["objective_weight_values"] != rows
                or work["objective_weight_numeric_bytes"] != 8 * rows
                or work["native_row_weight_mass_admissions"] != 2
            ):
                raise ValueError("weighted mixed native successful work differs")
        if (
            work["source_derived_counts"] is not True
            or work["native_work_returned"] is not False
            or any(type(work[key]) is not int or work[key] != 1 for key in _ONES)
            or any(type(work[key]) is not int or work[key] != rows for key in _ROW_COUNTS)
            or any(
                type(value) is not int or not 0 <= value <= 2**53 - 1
                for key, value in work.items()
                if key not in ("source_derived_counts", "native_work_returned")
            )
            or work["normalization_values"] % rows
            or not 1 <= work["normalization_values"] // rows <= 64
            or work["gradient_values"] <= 0
            or work["gradient_numeric_bytes"] != 8 * work["gradient_values"]
            or work["known_dense_multiply_accumulates"] <= 0
        ):
            raise ValueError(
                "mixed successful work requires source-derived native counters and flags"
            )


class RecordedContinuousMixed:
    """Fork-owned BP ports sharing bounded scalar receipts and one latest pin."""

    def __init__(
        self,
        learner: NumpyContinuousBCLearner,
        reports: list,
        limit: int,
        *,
        maximum_rows: int = 148,
        row_weights=None,
        _records=None,
        _latest=None,
    ):
        maximum_rows = _admit_maximum_rows(maximum_rows)
        _admit_source(learner)
        owned_weights = (
            _owned_weights(row_weights)
            if maximum_rows == 148
            else _owned_weights(row_weights, maximum_rows=maximum_rows)
        )
        if type(limit) is not int or not 1 <= limit <= 384:
            raise ValueError("mixed recording limit must be a native integer in1..384")
        if maximum_rows == 148:
            _validate_sink(reports, limit)
        else:
            _validate_sink(reports, limit, maximum_rows=maximum_rows)
        if _records is None and reports:
            raise ValueError("a new mixed recording sink must be empty")
        records, latest = (
            ([] if _records is None else _records),
            ([] if _latest is None else _latest),
        )
        if type(records) is not list or type(latest) is not list or len(latest) > 1:
            raise ValueError("mixed recording requires bounded exact shared seal and latest lists")
        self._learner, self._reports, self._limit = learner, reports, limit
        self._records, self._latest = records, latest
        self._original = (learner, reports, limit, records, latest)
        self._maximum_rows = self._maximum_rows_original = maximum_rows
        self._weight_request, self._row_weights = row_weights, owned_weights
        self._weight_owners = (row_weights, owned_weights)
        self._weight_seal = None if owned_weights is None else _json(owned_weights.tolist())
        self._policy_owner = learner._policy
        self._statistics_owner = learner._policy.observation_standardization
        self._policy_seal = _json(asdict(learner._policy))
        self.require_reports()

    def _require_weights(self):
        if (
            self._weight_request is not self._weight_owners[0]
            or self._row_weights is not self._weight_owners[1]
        ):
            raise RuntimeError("mixed original row weight owner was replaced")
        for value in (self._weight_request, self._row_weights):
            if self._weight_seal is None:
                if value is not None:
                    raise RuntimeError("legacy mixed recording acquired row weights")
            elif (
                type(value) is not np.ndarray
                or value.dtype != np.float64
                or value.ndim != 1
                or not 1 <= len(value) <= self._maximum_rows
                or _json(value.tolist()) != self._weight_seal
            ):
                raise RuntimeError("original mixed row weights changed")

    def require_reports(self, reports=None):
        if (
            type(self._maximum_rows) is not int
            or type(self._maximum_rows_original) is not int
            or self._maximum_rows_original not in (148, 198)
            or self._maximum_rows != self._maximum_rows_original
        ):
            raise RuntimeError("mixed original row scope was replaced")
        self._require_weights()
        if (
            self._learner is not self._original[0]
            or self._reports is not self._original[1]
            or type(self._limit) is not int
            or self._limit != self._original[2]
            or self._records is not self._original[3]
            or self._latest is not self._original[4]
            or self._learner._policy is not self._policy_owner
            or self._policy_owner.observation_standardization is not self._statistics_owner
            or _json(asdict(self._policy_owner)) != self._policy_seal
        ):
            raise RuntimeError("mixed recording original owner or policy was replaced")
        if reports is not None and reports is not self._reports:
            raise RuntimeError("mixed native report list was replaced")
        if len(self._reports) != len(self._records):
            raise RuntimeError("mixed native returned prefix changed")
        for row, (original, payload) in zip(self._reports, self._records, strict=True):
            if row is not original or _json(row) != payload:
                raise RuntimeError("mixed native returned prefix changed")
        if self._maximum_rows == 148:
            _validate_sink(self._reports, self._limit)
        else:
            _validate_sink(self._reports, self._limit, maximum_rows=self._maximum_rows)
        for row in self._reports:
            if self._weight_seal is None:
                if "objective_row_weights" in row:
                    raise RuntimeError("legacy mixed recording has weighted native work")
            elif _json(row.get("objective_row_weights")) != self._weight_seal:
                raise RuntimeError("weighted native receipt differs from fixed configuration")

    def require_latest(self, expected=None):
        if not self._latest:
            return
        owner, native, pinned = self._latest[0]
        if (
            type(owner) is not NumpyContinuousBCLearner
            or owner._state is not native
            or not same_continuous_state(pinned, _snapshot(owner._state, owner._policy))
            or (expected is not None and not same_continuous_state(pinned, expected))
        ):
            raise RuntimeError("mixed native returned state changed")

    def fork(self):
        self.require_reports()
        self.require_latest()
        native, policy = self._learner._state, self._learner._policy
        before = _snapshot(native, policy)
        statistics = policy.observation_standardization
        child = self._learner.fork()
        _require_unchanged(self._learner, native, policy, before, statistics)
        if type(child) is not NumpyContinuousBCLearner or child is self._learner:
            raise RuntimeError("mixed facade fork aliases its original owner")
        if not same_continuous_state(before, _snapshot(child._state, child._policy)):
            raise RuntimeError("mixed facade fork changed native state")
        _require_disjoint(native, child._state)
        self.require_reports()
        self.require_latest()
        # Default forks keep their original call shape. Expanded scopes are
        # explicit configuration and must survive every actor/candidate fork.
        if self._maximum_rows == 198:
            if self._row_weights is None:
                return RecordedContinuousMixed(
                    child,
                    self._reports,
                    self._limit,
                    maximum_rows=198,
                    _records=self._records,
                    _latest=self._latest,
                )
            return RecordedContinuousMixed(
                child,
                self._reports,
                self._limit,
                maximum_rows=198,
                row_weights=self._row_weights.copy(),
                _records=self._records,
                _latest=self._latest,
            )
        if self._row_weights is None:
            return RecordedContinuousMixed(
                child,
                self._reports,
                self._limit,
                _records=self._records,
                _latest=self._latest,
            )
        return RecordedContinuousMixed(
            child,
            self._reports,
            self._limit,
            row_weights=self._row_weights.copy(),
            _records=self._records,
            _latest=self._latest,
        )

    def snapshot_state(self):
        self.require_reports()
        native, policy = self._learner._state, self._learner._policy
        before = _snapshot(native, policy)
        result = self._learner.snapshot_state()
        _require_unchanged(self._learner, native, policy, before, self._statistics_owner)
        if not same_continuous_state(before, result):
            raise RuntimeError("mixed public snapshot differs from native state")
        self.require_reports()
        return result

    def restore_state(self, state):
        self.require_reports()
        self._learner.restore_state(state)
        self.require_reports()

    def predict(self, features):
        return self._learner.predict(features)

    def train_batch(self, features, targets):
        self.require_reports()
        self.require_latest()
        native, policy = self._learner._state, self._learner._policy
        before = _snapshot(native, policy)
        if type(policy.binary_gripper_axis) is not int or policy.binary_gripper_axis != 3:
            raise ValueError("common-MSE actor facade is read-only; only mixed axis3 may train")
        if len(self._reports) >= self._limit:
            raise ValueError("continuous mixed native report capacity exhausted")
        rows = len(features)
        if not 1 <= rows <= self._maximum_rows:
            raise ValueError(
                f"mixed continuation requires one to{self._maximum_rows} training rows"
            )
        weighted = self._row_weights is not None
        if weighted and len(self._row_weights) != rows:
            raise ValueError("mixed fixed row weights must match every training batch")
        weight_values = None
        if weighted:
            assert self._weight_seal is not None  # Already bound by require_reports.
            weight_values = json.loads(self._weight_seal)
        # Native exceptions propagate with their original .work. A successful
        # diagnostic has no returned work payload, so counts below are explicitly
        # derived from the inspected SGD source, never another model evaluation.
        if weighted:
            result = self._learner.train_batch(features, targets, row_weights=self._row_weights)
        else:
            result = self._learner.train_batch(features, targets)
        row = dict(
            diagnostic=asdict(result),
            training_rows=rows,
            work=_successful_work(before.policy, rows, weighted=True)
            if weighted
            else _successful_work(before.policy, rows),
        )
        if weighted:
            row["objective_row_weights"] = weight_values
        self._reports.append(row)
        self._records.append((row, _json(row)))
        # Publish paid scalar evidence before validating the returned state.
        observed = _snapshot(self._learner._state, self._learner._policy)
        self.require_reports()
        if (
            self._learner._state is native
            or observed.policy != before.policy
            or observed.completed_updates != before.completed_updates + 1
            or observed.training_examples != before.training_examples + rows
        ):
            raise RuntimeError("mixed returned SGD state or counters differ")
        self._latest[:] = [(self._learner, self._learner._state, observed)]
        return result


def mixed_native_work(native, *, maximum_rows: int = 148):
    """Aggregate successful native work within its original declared row scope."""
    maximum_rows = _admit_maximum_rows(maximum_rows)
    if maximum_rows == 148:
        _validate_sink(native, 384)
    else:
        _validate_sink(native, 384, maximum_rows=maximum_rows)
    if not native:
        raise ValueError("completed mixed work requires returned native updates")
    counts, flags, row_weights = {}, {}, {}
    for key in native[0]["work"]:
        values = [item["work"][key] for item in native]
        if all(type(value) is bool for value in values):
            flags[key] = any(values)
        elif all(type(value) is int for value in values):
            counts[key] = sum(values)
            row_weights[key] = sum(item["work"][key] * item["training_rows"] for item in native)
        else:
            raise ValueError("mixed source-derived counter/flag types changed")
    return counts, flags, row_weights
