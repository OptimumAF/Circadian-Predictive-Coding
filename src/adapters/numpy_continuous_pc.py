"""Owned one-hidden-layer continuous PC learner on the native learner port.

Composes matched BC initialization/state validation and continuous PC inference.
Raw arrays require caller release authority; episode updates check matching roles.
No replay, data selection, simulator, durable checkpoint, promotion or new runner.
"""

from dataclasses import asdict, dataclass, field
from typing import cast

import numpy as np

from src.adapters import numpy_continuous_bc as bc
from src.core import continuous_bc as task
from src.core import continuous_cpc as cpc
from src.core import continuous_pc as energy
from src.core import observation_standardization as observations
from src.core.continuous_bc import Array, Parameters
from src.core.continuous_demonstration import ContinuousEpisode, continuous_training_pairs
from src.core.continuous_pc_inference import ContinuousPCInferenceResult
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_predictive_coding import CoupledWakeState
from src.core.coupled_inference import (
    CoupledInferenceDiagnostics,
    CoupledInferencePolicy,
    _InferenceWork,
    _finite_number,
    _run_inference,
    _validate_policy,
)
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed


@dataclass(frozen=True)
class ContinuousPCPolicy:
    base: bc.ContinuousBCPolicy
    precision: float = 1.0
    inference_steps: int = 4
    inference_learning_rate: float = 0.05
    inference_policy: CoupledInferencePolicy = field(default_factory=CoupledInferencePolicy)
    circadian: CoupledCircadianConfig | None = None

    def __post_init__(self) -> None:
        base = bc._policy(self.base)
        if base.binary_gripper_axis is not None:
            raise ValueError("continuous PC supports only the matched default BC likelihood")
        precision = energy._own_precisions((self.precision,), 1)[0]
        if bc._count(self.inference_steps, "inference_steps", 10000) == 0:
            raise ValueError("inference_steps must be positive")
        rate = _finite_number(
            self.inference_learning_rate, "inference_learning_rate", positive=True
        )
        _validate_policy(self.inference_policy)
        stopping = CoupledInferencePolicy(
            self.inference_policy.min_steps,
            self.inference_policy.gradient_tolerance,
            self.inference_policy.relative_energy_tolerance,
        )
        if stopping.min_steps > self.inference_steps:
            raise ValueError("min_steps must not exceed inference_steps")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "inference_learning_rate", rate)
        object.__setattr__(self, "inference_policy", stopping)
        object.__setattr__(self, "circadian", cpc.copy_continuous_config(self.circadian))


def _policy(value: ContinuousPCPolicy) -> ContinuousPCPolicy:
    if type(value) is not ContinuousPCPolicy:
        raise ValueError("policy must be an exact ContinuousPCPolicy")
    return ContinuousPCPolicy(
        value.base,
        value.precision,
        value.inference_steps,
        value.inference_learning_rate,
        value.inference_policy,
        value.circadian,
    )


@dataclass(frozen=True)
class ContinuousPCSnapshot:
    policy: ContinuousPCPolicy
    parameters: Parameters
    velocity: Parameters
    completed_updates: int
    training_examples: int
    wake_state: CoupledWakeState | None = None


def _snapshot(value: ContinuousPCSnapshot, policy: ContinuousPCPolicy) -> ContinuousPCSnapshot:
    owned = _policy(policy)
    if type(value) is not ContinuousPCSnapshot or _policy(value.policy) != owned:
        raise ValueError("continuous PC snapshot policy differs from its owner")
    # Why: identical one-hidden geometry, vector biases, momentum and row-counter
    # contracts keep the robot BP comparison matched without changing its code.
    checked = bc._snapshot(
        bc.ContinuousBCSnapshot(
            owned.base,
            value.parameters,
            value.velocity,
            value.completed_updates,
            value.training_examples,
        ),
        owned.base,
    )
    return ContinuousPCSnapshot(
        owned,
        checked.parameters,
        checked.velocity,
        checked.completed_updates,
        checked.training_examples,
        cpc.copy_continuous_wake(
            value.wake_state, owned.circadian, owned.base.hidden_dim, checked.completed_updates
        ),
    )


def _coupled(parameters: Parameters) -> CoupledParameters:
    return CoupledParameters(
        (parameters[0],), (parameters[1][None, :],), parameters[2], parameters[3][None, :]
    )


def _flat(parameters: CoupledParameters) -> Parameters:
    return (
        parameters.hidden_weights[0],
        parameters.hidden_biases[0][0],
        parameters.output_weight,
        parameters.output_bias[0],
    )


@dataclass(frozen=True)
class ContinuousPCUpdateResult:
    diagnostic: TrainingDiagnostic
    task_loss: float
    inference: CoupledInferenceDiagnostics
    work: dict[str, int | bool]


def _infer_update(before, features, targets, work, progress) -> ContinuousPCInferenceResult:
    policy, parameters = before.policy, _coupled(before.parameters)
    work["initialization_attempts"] += 1
    initial = energy.feedforward_continuous_latents(parameters, features, policy.base.spec)
    work["initialization_returns"] += 1
    latents, evaluated, diagnostics = _run_inference(
        initial,
        lambda current: energy.evaluate_continuous_pc(
            parameters, features, targets, current, (policy.precision,), policy.base.spec
        ),
        policy.inference_steps,
        policy.inference_learning_rate,
        policy.inference_policy,
        _work=progress,
    )
    work["inference_returns"] += 1
    return ContinuousPCInferenceResult(latents, evaluated, diagnostics)


def _propose_update(before, gradients, rows, wake_state=None) -> ContinuousPCSnapshot:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        velocity = cast(
            Parameters,
            tuple(
                before.policy.base.momentum * previous + gradient
                for previous, gradient in zip(before.velocity, gradients, strict=True)
            ),
        )
        parameters = cast(
            Parameters,
            tuple(
                previous - before.policy.base.learning_rate * speed
                for previous, speed in zip(before.parameters, velocity, strict=True)
            ),
        )
    return _snapshot(
        ContinuousPCSnapshot(
            before.policy,
            parameters,
            velocity,
            before.completed_updates + 1,
            before.training_examples + rows,
            wake_state,
        ),
        before.policy,
    )


class NumpyContinuousPCLearner:
    def __init__(self, policy: ContinuousPCPolicy) -> None:
        self._policy = _policy(policy)
        initial = bc.NumpyContinuousBCLearner(self._policy.base).snapshot_state()
        self._state = _snapshot(
            ContinuousPCSnapshot(
                self._policy,
                initial.parameters,
                initial.velocity,
                0,
                0,
                cpc.initial_continuous_wake(self._policy.circadian, self._policy.base.hidden_dim),
            ),
            self._policy,
        )

    @property
    def policy(self) -> ContinuousPCPolicy:
        return _policy(self._policy)

    def snapshot_state(self) -> ContinuousPCSnapshot:
        return _snapshot(self._state, self._policy)

    def restore_state(self, snapshot: ContinuousPCSnapshot) -> None:
        candidate = _snapshot(snapshot, self._policy)
        self._state = candidate

    def fork(self) -> "NumpyContinuousPCLearner":
        child = object.__new__(NumpyContinuousPCLearner)
        child._policy = self.policy
        child._state = self.snapshot_state()
        return child

    @property
    def wake_updates(self) -> int:
        wake = self.snapshot_state().wake_state
        if wake is None:
            raise ValueError("wake updates require circadian mode")
        return wake.wake_updates

    def get_chemical_state(self) -> Array:
        wake = self.snapshot_state().wake_state
        if wake is None:
            raise ValueError("chemical state requires circadian mode")
        return wake.chemical_state

    def get_plasticity_state(self) -> Array:
        wake = self.snapshot_state().wake_state
        if wake is None:
            raise ValueError("plasticity state requires circadian mode")
        return wake.plasticity_state

    def predict(self, features: Array) -> Array:
        state = self.snapshot_state()
        values = observations.standardize_observations(
            features, state.policy.base.observation_standardization, state.policy.base.spec
        )
        return energy.predict_continuous_pc(
            _coupled(state.parameters), values, state.policy.base.spec
        )

    def train_episode(self, episode: ContinuousEpisode) -> TrainingDiagnostic:
        features, targets = continuous_training_pairs(episode, role=self._policy.base.training_role)
        if episode.spec != self._policy.base.spec:
            raise ValueError("continuous PC episode units/bounds/spec differ from its policy")
        return self.train_batch(
            np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        )

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        return self.train_batch_with_work(features, targets).diagnostic

    def train_batch_with_work(self, features: Array, targets: Array) -> ContinuousPCUpdateResult:
        before = self.snapshot_state()
        raw = task._features(features, before.policy.base.spec)
        actions = task._targets(targets, len(raw), before.policy.base.spec)
        bc._count(before.completed_updates + 1, "completed_updates", 2**53 - 1)
        bc._count(before.training_examples + len(raw), "training_examples", 2**53 - 1)
        work: dict[str, int | bool] = dict(
            normalization_attempts=0,
            normalization_returns=0,
            initialization_attempts=0,
            initialization_returns=0,
            inference_returns=0,
            parameter_proposals_attempted=0,
            parameter_proposals_completed=0,
            candidate_predictions_attempted=0,
            candidate_predictions_completed=0,
            published_updates=0,
        )
        progress = _InferenceWork()
        if before.policy.circadian is not None:
            work.update(
                chemistry_attempts=0,
                chemistry_returns=0,
                gate_applications_attempted=0,
                gate_applications_returned=0,
            )
        try:
            work["normalization_attempts"] += 1
            values = observations.standardize_observations(
                raw, before.policy.base.observation_standardization, before.policy.base.spec
            )
            work["normalization_returns"] += 1
            inferred = _infer_update(before, values, actions, work, progress)
            gradients = inferred.evaluation.parameter_gradients
            wake = None
            if before.policy.circadian is not None:
                assert before.wake_state is not None
                work["chemistry_attempts"] += 1
                wake = cpc.advance_continuous_wake(
                    before.wake_state, inferred.latents, before.policy.circadian
                )
                work["chemistry_returns"] += 1
                work["gate_applications_attempted"] += 1
                gradients = cpc.gate_continuous_gradients(gradients, wake.plasticity_state)
                work["gate_applications_returned"] += 1
            work["parameter_proposals_attempted"] += 1
            candidate = _propose_update(before, _flat(gradients), len(values), wake)
            work["parameter_proposals_completed"] += 1
            work["candidate_predictions_attempted"] += 1
            energy.predict_continuous_pc(
                _coupled(candidate.parameters), values, candidate.policy.base.spec
            )
            work["candidate_predictions_completed"] += 1
            diagnostic = TrainingDiagnostic(
                energy.CONTINUOUS_PC_ENERGY_ID, inferred.evaluation.energy
            )
            result = ContinuousPCUpdateResult(
                diagnostic,
                inferred.evaluation.task_loss,
                inferred.diagnostics,
                dict(work, **asdict(progress), published_updates=1),
            )
        except Exception as error:
            raise TrainingUpdateFailed(
                str(error) or type(error).__name__,
                energy.CONTINUOUS_PC_ENERGY_ID,
                dict(work, **asdict(progress)),
            ) from error
        self._state = candidate
        return result
