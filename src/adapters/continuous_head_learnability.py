"""Bounded native head-update recording for the existing continuous stream.

Wrap the existing BP port and summarize its actual returned work separately
from read-only conditioning and residuals. No runner, initialization, SGD,
least-squares diagnostic-head installation, role conversion, simulator or IO.
"""

from dataclasses import asdict
import json
import math

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner, _snapshot
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.core import continuous_bc as bc
from src.core.continuous_head_gauss_newton import DAMPING, MAX_ROWS, MAX_STEP_NORM, RCOND
from src.core.observation_standardization import standardization_numeric_bytes


def _validate_sink(reports, limit):
    if type(reports) is not list or len(reports) > limit:
        raise ValueError("head update reports require a bounded exact shared list")
    for item in reports:
        if (
            type(item) is not dict
            or set(item) != {"diagnostic", "loss_after", "step_norm", "work", "training_rows"}
            or type(item["training_rows"]) is not int
            or not 1 <= item["training_rows"] <= MAX_ROWS
            or type(item["diagnostic"]) is not dict
            or set(item["diagnostic"]) != {"definition", "value"}
            or item["diagnostic"]["definition"] != bc.OBJECTIVE
            or any(
                type(value) is not float or not math.isfinite(value) or value < 0
                for value in (item["diagnostic"]["value"], item["loss_after"], item["step_norm"])
            )
            or item["step_norm"] > MAX_STEP_NORM
            or item["loss_after"] >= item["diagnostic"]["value"]
            or type(item["work"]) is not dict
            or not 1 <= len(item["work"]) <= 32
        ):
            raise ValueError("head report differs from an accepted original-objective update")
        for key, value in item["work"].items():
            if (
                type(key) is not str
                or not key
                or not (type(value) is bool or (type(value) is int and 0 <= value <= 2**53 - 1))
            ):
                raise ValueError("head native work requires native counters or bool flags")
        if any(
            type(item["work"].get(key)) is not int or item["work"][key] != 1
            for key in ("published_updates", "accepted_proposals")
        ):
            raise ValueError("head report did not publish one accepted proposal")


class RecordedContinuousHead:
    """Fork-owned BP models sharing one bounded returned-native scalar sink."""

    def __init__(
        self,
        learner: NumpyContinuousBCLearner,
        reports: list,
        limit: int,
        *,
        _records=None,
        _latest=None,
    ):
        if type(learner) is not NumpyContinuousBCLearner:
            raise ValueError("head recording requires an exact continuous BP learner")
        if type(limit) is not int or not 1 <= limit <= 384:
            raise ValueError("head recording limit must be a native integer in1..384")
        _validate_sink(reports, limit)
        if _records is None and reports:
            raise ValueError("a new head recording sink must be empty")
        self._learner, self._reports, self._limit = learner, reports, limit
        self._records = [] if _records is None else _records
        self._latest = [] if _latest is None else _latest
        self._original = (learner, reports, limit, self._records, self._latest)
        self.require_reports()

    def require_reports(self, reports=None):
        if (
            self._learner is not self._original[0]
            or self._reports is not self._original[1]
            or type(self._limit) is not int
            or self._limit != self._original[2]
            or self._records is not self._original[3]
            or self._latest is not self._original[4]
        ):
            raise RuntimeError("head recording owner was replaced")
        if reports is not None and reports is not self._reports:
            raise RuntimeError("head native report list was replaced")
        if len(self._reports) != len(self._records):
            raise RuntimeError("head native returned prefix changed")
        for row, (original, payload) in zip(self._reports, self._records, strict=True):
            if row is not original or json.dumps(row, sort_keys=True, allow_nan=False) != payload:
                raise RuntimeError("head native returned prefix changed")

    def require_latest(self, expected=None):
        if not self._latest:
            return
        owner, native_state, pinned = self._latest[0]
        if (
            owner._state is not native_state
            or not same_continuous_state(pinned, _snapshot(owner._state, owner._policy))
            or (expected is not None and not same_continuous_state(pinned, expected))
        ):
            raise RuntimeError("head native returned state changed")

    def fork(self):
        self.require_reports()
        return RecordedContinuousHead(
            self._learner.fork(),
            self._reports,
            self._limit,
            _records=self._records,
            _latest=self._latest,
        )

    def snapshot_state(self):
        return self._learner.snapshot_state()

    def restore_state(self, state):
        self._learner.restore_state(state)

    def predict(self, features):
        return self._learner.predict(features)

    def train_batch(self, features, targets):
        self.require_reports()
        self.require_latest()
        if len(self._reports) >= self._limit:
            raise ValueError("continuous head native report capacity exhausted")
        result = self._learner.train_head_gauss_newton_batch(features, targets)
        # Retain returned work before app/runtime receipt or ownership checks.
        row = dict(asdict(result), training_rows=len(features))
        self._reports.append(row)
        self._records.append((row, json.dumps(row, sort_keys=True, allow_nan=False)))
        # One latest model pin, replaced on success; never a model-state chain.
        # Paid return stays recorded if this private-state validation fails.
        self._latest[:] = [
            (
                self._learner,
                self._learner._state,
                _snapshot(self._learner._state, self._learner._policy),
            )
        ]
        return result.diagnostic


def head_protocol() -> dict:
    return dict(
        definition="continuous_bp_head_gauss_newton_v1",
        bp_only=True,
        objective=bc.OBJECTIVE,
        damping=DAMPING,
        max_step_norm=MAX_STEP_NORM,
        rcond=RCOND,
        hidden_parameters_frozen=True,
        hidden_momentum_preserved=True,
        head_momentum="reset_to_zero_on_accepted_update",
        acceptance="finite_strict_original_mse_decrease",
    )


def _native_work(native):
    _validate_sink(native, 384)
    if not native:
        raise ValueError("completed head work requires returned native updates")
    keys = tuple(native[0]["work"])
    if any(set(item["work"]) != set(keys) for item in native):
        raise ValueError("head native work field sets changed")
    counts, flags, row_weights = {}, {}, {}
    for key in keys:
        values = [item["work"][key] for item in native]
        if all(type(value) is bool for value in values):
            flags[key] = any(values)
        elif all(type(value) is int for value in values):
            counts[key] = sum(values)
            row_weights[key] = sum(item["work"][key] * item["training_rows"] for item in native)
        else:
            raise ValueError("head native counter/flag types changed")
    return counts, flags, row_weights


def head_work(episodes, policy: ContinuousBCPolicy, report) -> dict:
    """Successful A/B/A counts; native head solves never use SGD cost formulas.

    Logical payloads are per named owner. Actual solver arithmetic/workspace,
    gate/snapshot validation, elementwise work, copies, scalar reports and RSS
    remain additional; native row weights are exposures, not arithmetic.
    """
    warm, new = (len(episodes[name].transitions) for name in ("warm", "new"))
    native = report["native_updates"]
    counts, flags, weights = _native_work(native)
    rows = sum(item["training_rows"] for item in native)
    readouts = [item for point in report["checkpoints"] for item in point["contexts"].values()]
    conditioning_work = {
        key: sum(item["conditioning"]["conditioning"]["work"][key] for item in readouts)
        for key in (
            "forward_passes",
            "backward_passes",
            "gradient_values",
            "gradient_numeric_bytes",
            "dense_multiply_accumulates",
        )
    }
    dimension, hidden, actions = (
        policy.spec.observation_dim,
        policy.hidden_dim,
        len(policy.spec.action_units),
    )
    parameters = dimension * hidden + hidden + hidden * actions + actions
    diagnostic_rows = sum(item["residual"]["prediction_rows"] for item in readouts)
    readout_calls = sum(
        item["conditioning"]["standardization_calls"] for item in readouts
    ) + 2 * len(readouts)
    readout_rows = sum(
        item["conditioning"]["standardization_rows"] + 2 * item["residual"]["prediction_rows"]
        for item in readouts
    )
    readout_values = sum(
        item["conditioning"]["standardization_values"]
        + 2 * item["residual"]["prediction_rows"] * dimension
        for item in readouts
    )
    residual_actor_mac = 2 * diagnostic_rows * (dimension * hidden + hidden * actions)
    return dict(
        fit_calls=1,
        fit_rows=warm,
        fit_feature_values=warm * dimension,
        optimizer_updates=len(native),
        head_gauss_newton_updates=len(native),
        sgd_updates=0,
        training_gradient_evaluations=0,
        training_rows=rows,
        unique_source_rows=warm + new,
        unique_training_rows=warm + new,
        repeated_training_rows=rows - warm - new,
        replay_budget_rows=0,
        native_update_counts=counts,
        native_work_flags=flags,
        native_event_row_weights=weights,
        initializations=1,
        seeded_normal_draws=dimension * hidden + hidden * actions,
        probe_updates=0,
        native_controls=0,
        conditioning_calls=len(readouts),
        conditioning_rows=sum(item["conditioning"]["standardization_rows"] for item in readouts),
        conditioning_definition="common_task_head_bp_total_gradients_read_only",
        conditioning_work=conditioning_work,
        gradient_values=conditioning_work["gradient_values"],
        gradient_numeric_bytes=conditioning_work["gradient_numeric_bytes"],
        gradient_work_source="read_only_conditioning_only; native head uses linear solves",
        readout_normalization_calls=readout_calls,
        readout_normalization_rows=readout_rows,
        readout_normalization_values=readout_values,
        normalization_calls=counts["normalization_completed"] + readout_calls,
        normalization_rows=counts["normalization_rows"] + readout_rows,
        normalization_values=counts["normalization_values"] + readout_values,
        residual_calls=len(readouts),
        residual_rows=diagnostic_rows,
        actor_prediction_calls=len(readouts),
        actor_prediction_rows=diagnostic_rows,
        residual_actor_forward_passes=2 * len(readouts),
        residual_actor_forward_rows=2 * diagnostic_rows,
        residual_actor_dense_multiply_accumulates=residual_actor_mac,
        known_dense_multiply_accumulates=counts["known_dense_multiply_accumulates"]
        + conditioning_work["dense_multiply_accumulates"]
        + residual_actor_mac,
        solver_work_known=False,
        parameter_values_per_owner=parameters,
        model_numeric_bytes_per_owner=parameters * 16,
        statistics_numeric_bytes_per_owner=standardization_numeric_bytes(
            policy.observation_standardization
        ),
        wake_numeric_bytes_per_owner=0,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=rows * (dimension + actions) * 8,
        residual_prediction_bytes=diagnostic_rows * actions * 8,
        actor_prediction_bytes=diagnostic_rows * actions * 8,
        scope="successful fixed head-update run; known completed native forwards plus read-only "
        "conditioning/residual/actor dense work, not total solver FLOPs or process memory",
    )
