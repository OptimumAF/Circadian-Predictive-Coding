"""Family-specific native recording and common TRAIN task-head readouts.

Inputs are already released episodes and owned continuous learners. Outputs are
bounded scalar update reports and read-only physical diagnostics. This module
does not schedule updates, fit statistics, select data, promote or run probes.
"""

from dataclasses import asdict

import numpy as np

from src.adapters.numpy_continuous_bc import (
    ContinuousBCSnapshot,
    NumpyContinuousBCLearner,
    _snapshot as own_bc_snapshot,
)
from src.adapters.numpy_continuous_pc import (
    ContinuousPCSnapshot,
    NumpyContinuousPCLearner,
    _snapshot as own_pc_snapshot,
)
from src.core import continuous_bc_diagnostics as diagnostics
from src.core.continuous_demonstration import continuous_training_pairs
from src.core.observation_standardization import (
    standardize_observations,
    standardization_numeric_bytes,
)

FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
ContinuousLearner = NumpyContinuousBCLearner | NumpyContinuousPCLearner
ContinuousState = ContinuousBCSnapshot | ContinuousPCSnapshot


def validate_learner_family(value: object) -> str:
    if type(value) is not str or value not in FAMILIES:
        raise ValueError(f"learner_family must be one of {FAMILIES}")
    return value


def _same_arrays(first, second) -> bool:
    return len(first) == len(second) and all(
        a.dtype == b.dtype == np.dtype("float64") and np.array_equal(a, b)
        for a, b in zip(first, second, strict=True)
    )


def same_continuous_state(first: ContinuousState, second: ContinuousState) -> bool:
    if type(first) is not type(second) or type(first) not in (
        ContinuousBCSnapshot,
        ContinuousPCSnapshot,
    ):
        return False
    # Reuse owned snapshot admission, including native counters and complete
    # wake/config coherence. Equality never legitimizes two matching bad states.
    if isinstance(first, ContinuousPCSnapshot) and isinstance(second, ContinuousPCSnapshot):
        first, second = own_pc_snapshot(first, first.policy), own_pc_snapshot(second, second.policy)
    elif isinstance(first, ContinuousBCSnapshot) and isinstance(second, ContinuousBCSnapshot):
        first, second = own_bc_snapshot(first, first.policy), own_bc_snapshot(second, second.policy)
    if (
        type(first) is not type(second)
        or first.policy != second.policy
        or first.completed_updates != second.completed_updates
        or first.training_examples != second.training_examples
        or not _same_arrays(first.parameters + first.velocity, second.parameters + second.velocity)
    ):
        return False
    if isinstance(first, ContinuousPCSnapshot) and isinstance(second, ContinuousPCSnapshot):
        a, b = first.wake_state, second.wake_state
        if a is None or b is None:
            return a is b
        # Why: coherent replacement chemistry can change future updates even when
        # parameters/momentum match. Snapshot validation alone cannot detect it.
        fields = (
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
            "wake_updates",
        )
        return (
            type(a) is type(b)
            and all(getattr(a, key) == getattr(b, key) for key in fields)
            and _same_arrays(
                (a.chemical_state, a.plasticity_state), (b.chemical_state, b.plasticity_state)
            )
        )
    return True


class RecordedContinuousPC:
    """Native-port facade; forks own models, one bounded run sink owns scalars.

    Why: the generic runtime deliberately returns TrainingDiagnostic only. This
    adapter records the richer native return without touching runtime internals
    or making model snapshots retain success histories. The actor never trains.
    """

    def __init__(self, learner, reports, limit):
        self._learner, self._reports, self._limit = learner, reports, limit

    def fork(self):
        return RecordedContinuousPC(self._learner.fork(), self._reports, self._limit)

    def snapshot_state(self):
        return self._learner.snapshot_state()

    def restore_state(self, state):
        self._learner.restore_state(state)

    def predict(self, features):
        return self._learner.predict(features)

    def train_batch(self, features, targets):
        if len(self._reports) >= self._limit:
            raise ValueError("continuous PC native report capacity exhausted")
        result = self._learner.train_batch_with_work(features, targets)
        # Save before returning to app receipt/resource checks, so a subsequent
        # postcommit error cannot erase already returned native work.
        self._reports.append(dict(asdict(result), training_rows=len(features)))
        return result.diagnostic


def _training_view(learner: NumpyContinuousPCLearner, episode):
    state = learner.snapshot_state()
    base = state.policy.base
    if base.training_role != "train":
        raise ValueError("task-head diagnosis requires a TRAIN policy")
    features, targets = continuous_training_pairs(episode, role="train")
    if episode.spec != base.spec:
        raise ValueError("diagnostic episode units/bounds/spec differ from its policy")
    return state, np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def diagnose_conditioning(learner: ContinuousLearner, episode):
    if isinstance(learner, NumpyContinuousBCLearner):
        return learner.diagnose_training_conditioning(episode)
    state, features, targets = _training_view(learner, episode)
    base = state.policy.base
    values = standardize_observations(features, base.observation_standardization, base.spec)
    # Common task-head total BP gradients diagnose saturation/conditioning in
    # every family. They are NOT PC free-state partials and cause no update.
    result = diagnostics.continuous_bc_conditioning(values, targets, state.parameters, base.spec)
    return diagnostics.TrainingConditioningDiagnostic(
        result, 1, len(values), values.size, base.observation_standardization is not None
    )


def diagnose_residual(learner: ContinuousLearner, episode, stages):
    if isinstance(learner, NumpyContinuousBCLearner):
        return learner.diagnose_training_episode(episode, stages)
    _, features, targets = _training_view(learner, episode)
    labels = diagnostics._stage_labels(stages, len(features))
    predictions = learner.predict(features)
    result = diagnostics.continuous_bc_residuals(predictions, targets, episode.spec, labels)
    return diagnostics.TrainingResidualDiagnostic(
        result, 1, len(features), predictions.size, predictions.nbytes
    )


def pc_work(episodes, policy, report, *, a_only_control):
    """Returned native event counts plus explicitly separate diagnostic costs.

    No BP training dense/gradient formula is applied to PC/CPC. Row weights below
    are event exposures, not estimates of arithmetic or wall time.
    """
    base = policy.base
    warm, new = (len(episodes[name].transitions) for name in ("warm", "new"))
    native = report["native_updates"]
    rows = sum(item["training_rows"] for item in native)
    counts = {key: sum(item["work"][key] for item in native) for key in native[0]["work"]}
    weights = {
        key: sum(item["work"][key] * item["training_rows"] for item in native) for key in counts
    }
    readouts = [item for row in report["checkpoints"] for item in row["contexts"].values()]
    diagnostic_work = {
        key: sum(item["conditioning"]["conditioning"]["work"][key] for item in readouts)
        for key in (
            "forward_passes",
            "backward_passes",
            "gradient_values",
            "gradient_numeric_bytes",
            "dense_multiply_accumulates",
        )
    }
    parameters = (
        base.spec.observation_dim * base.hidden_dim
        + base.hidden_dim
        + base.hidden_dim * len(base.spec.action_units)
        + len(base.spec.action_units)
    )
    return dict(
        fit_calls=1,
        fit_rows=warm,
        fit_feature_values=warm * base.spec.observation_dim,
        optimizer_updates=len(native),
        training_rows=rows,
        unique_source_rows=warm + new,
        unique_training_rows=warm if a_only_control else warm + new,
        repeated_training_rows=rows - (warm if a_only_control else warm + new),
        replay_budget_rows=0,
        native_update_counts=counts,
        native_event_row_weights=weights,
        conditioning_calls=len(readouts),
        conditioning_rows=4 * (warm + new),
        conditioning_definition="common_task_head_bp_total_gradients_read_only",
        conditioning_work=diagnostic_work,
        readout_normalization_calls=sum(
            item["conditioning"]["standardization_calls"] for item in readouts
        )
        + 2 * len(readouts),
        readout_normalization_rows=sum(
            item["conditioning"]["standardization_rows"] + 2 * item["residual"]["prediction_rows"]
            for item in readouts
        ),
        readout_normalization_values=sum(
            item["conditioning"]["standardization_values"]
            + 2 * item["residual"]["prediction_rows"] * base.spec.observation_dim
            for item in readouts
        ),
        residual_calls=len(readouts),
        residual_rows=4 * (warm + new),
        actor_prediction_calls=len(readouts),
        actor_prediction_rows=4 * (warm + new),
        residual_actor_forward_passes=2 * len(readouts),
        residual_actor_forward_rows=8 * (warm + new),
        residual_actor_dense_multiply_accumulates=8
        * (warm + new)
        * (
            base.spec.observation_dim * base.hidden_dim
            + base.hidden_dim * len(base.spec.action_units)
        ),
        residual_actor_work_source="derived from complete fixed readout shapes and unchanged one-hidden prediction kernels",
        parameter_values_per_owner=parameters,
        model_numeric_bytes_per_owner=parameters * 16,
        statistics_numeric_bytes_per_owner=standardization_numeric_bytes(
            base.observation_standardization
        ),
        wake_numeric_bytes_per_owner=0 if policy.circadian is None else base.hidden_dim * 16,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=rows
        * (base.spec.observation_dim + len(base.spec.action_units))
        * 8,
        residual_prediction_bytes=4 * (warm + new) * len(base.spec.action_units) * 8,
        actor_prediction_bytes=4 * (warm + new) * len(base.spec.action_units) * 8,
        scope="successful fixed run; native event counts and diagnostic dense math are separate; excludes validation gate recomputation, copies, elementwise work, scalar report metadata, temporaries and process overhead",
    )
