"""Owned CPU continuous BC learner through the existing generic native port.

Owns policy, full parameters/momentum/counters and atomic local updates. Raw
arrays require caller release authority; train_episode additionally checks roles.
No stream, replay, simulator, evaluation selection, checkpoint IO or promotion.
"""

from dataclasses import dataclass
import math
from typing import cast

import numpy as np

from src.core import continuous_bc as core
from src.core import continuous_bc_diagnostics as residual
from src.core import observation_standardization as observations
from src.core import continuous_head_gauss_newton as head_update
from src.core.continuous_bc import Array, Parameters, OBJECTIVE
from src.core.observation_standardization import ObservationStandardization
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.experience import require_training_role
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.pick_cube_rgb_features import PickCubeRGBRecipe
from src.core.pick_cube_action_history import (
    ActionHistoryRecipe,
    is_taskspace_recipe,
    validate_action_history_recipe,
    validate_action_history_training_episode,
)
from src.core.pick_cube_rgb_standardization import (
    fit_rgb_range_standardization,
    validate_rgb_range_features,
)


def _count(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a native integer in0..{maximum}")
    return value


@dataclass(frozen=True)
class ContinuousBCPolicy:
    spec: ContinuousActionSpec
    hidden_dim: int = 64
    seed: int = 17
    learning_rate: float = 0.01
    momentum: float = 0.9
    training_role: str = "train"
    observation_standardization: ObservationStandardization | None = None
    binary_gripper_axis: int | None = None

    def __post_init__(self) -> None:
        spec = core._spec(self.spec)
        core._geometry(spec)
        core._binary_gripper_axis(self.binary_gripper_axis, spec)
        if _count(self.hidden_dim, "hidden_dim", 256) == 0:
            raise ValueError("hidden_dim must be positive")
        _count(self.seed, "seed", 2**32 - 1)
        for name, value, low, high in (
            ("learning_rate", self.learning_rate, 0.0, 1.0),
            ("momentum", self.momentum, -1.0, 1.0),
        ):
            if (
                type(value) not in (int, float)
                or not low < value <= high
                or not math.isfinite(value)
            ):
                raise ValueError(f"invalid continuous BC {name}")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in[0,1)")
        require_training_role(self.training_role)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(
            self,
            "observation_standardization",
            observations._standardization(self.observation_standardization, spec.observation_dim),
        )


def _policy(value: ContinuousBCPolicy) -> ContinuousBCPolicy:
    if type(value) is not ContinuousBCPolicy:
        raise ValueError("policy must be an exact ContinuousBCPolicy")
    return ContinuousBCPolicy(
        value.spec,
        value.hidden_dim,
        value.seed,
        value.learning_rate,
        value.momentum,
        value.training_role,
        value.observation_standardization,
        value.binary_gripper_axis,
    )


def fit_training_standardization(
    episode: ContinuousEpisode,
    *,
    rgb_range_recipe: PickCubeRGBRecipe | None = None,
    scale_floors: tuple[float, ...] | None = None,
    action_history_recipe: ActionHistoryRecipe | None = None,
) -> ObservationStandardization | None:
    """Fit once from a complete TRAIN record, never probe/final observations."""
    features, _ = continuous_training_pairs(episode, role="train")
    if action_history_recipe is not None:
        if rgb_range_recipe is not None:
            raise ValueError(
                "action history requires nonvisual state data without RGB-range scaling"
            )
        recipe = validate_action_history_recipe(action_history_recipe)
        owned = validate_action_history_training_episode(episode, recipe)
        if is_taskspace_recipe(recipe):
            if scale_floors is not None:
                raise ValueError("taskspace fixed physical scaling does not fit scale floors")
            return None
        prefix_features = tuple(row.observation[:42] for row in owned.transitions)
        floors = (
            observations.validate_scale_floors(scale_floors, recipe.source_dim)
            if scale_floors is not None
            else None
        )
        prefix = (
            observations.fit_observation_standardization(
                np.asarray(prefix_features, dtype=np.float64), scale_floors=floors
            )
            if floors is not None
            else observations.fit_observation_standardization(
                np.asarray(prefix_features, dtype=np.float64)
            )
        )
        means, scales = prefix.means + (0.0,) * 5, prefix.scales + (1.0,) * 5
        if type(prefix) is observations.ScaleFlooredObservationStandardization:
            return observations.ScaleFlooredObservationStandardization(
                means, scales, prefix.fitted_rows, prefix.scale_floors + (1.0,) * 5
            )
        return ObservationStandardization(means, scales, prefix.fitted_rows)
    if rgb_range_recipe is not None:
        if scale_floors is not None:
            raise ValueError(
                "RGB-range standardization and scale floors require separate declarations"
            )
        validate_rgb_range_features(
            np.asarray([episode.transitions[-1].next_observation], dtype=np.float64),
            rgb_range_recipe,
        )
        return fit_rgb_range_standardization(
            np.asarray(features, dtype=np.float64), rgb_range_recipe
        )
    if scale_floors is not None:
        floors = observations.validate_scale_floors(scale_floors, episode.spec.observation_dim)
        return observations.fit_observation_standardization(
            np.asarray(features, dtype=np.float64), scale_floors=floors
        )
    return observations.fit_observation_standardization(np.asarray(features, dtype=np.float64))


@dataclass(frozen=True)
class ContinuousBCSnapshot:
    policy: ContinuousBCPolicy
    parameters: Parameters
    velocity: Parameters
    completed_updates: int
    training_examples: int


@dataclass(frozen=True)
class HeadGaussNewtonTrainingResult:
    diagnostic: TrainingDiagnostic
    loss_after: float
    step_norm: float
    work: dict[str, int | bool]


def _snapshot(value: ContinuousBCSnapshot, policy: ContinuousBCPolicy) -> ContinuousBCSnapshot:
    if type(value) is not ContinuousBCSnapshot or _policy(value.policy) != policy:
        raise ValueError("continuous BC snapshot policy differs from its owner")
    updates = _count(value.completed_updates, "completed_updates", 2**53 - 1)
    examples = _count(value.training_examples, "training_examples", 2**53 - 1)
    if (updates == 0 and examples != 0) or (
        updates > 0 and not updates <= examples <= 256 * updates
    ):
        raise ValueError("continuous BC snapshot update/row counters are inconsistent")
    parameters, velocity = (
        core._parameters(value.parameters, policy.spec),
        core._parameters(value.velocity, policy.spec),
    )
    if parameters[0].shape[1] != policy.hidden_dim or velocity[0].shape[1] != policy.hidden_dim:
        raise ValueError("continuous BC snapshot hidden shape differs from its policy")
    return ContinuousBCSnapshot(_policy(policy), parameters, velocity, updates, examples)


class NumpyContinuousBCLearner:
    def __init__(self, policy: ContinuousBCPolicy) -> None:
        self._policy = _policy(policy)
        rng = np.random.default_rng(self._policy.seed)
        inputs, hidden, actions = (
            self._policy.spec.observation_dim,
            self._policy.hidden_dim,
            len(self._policy.spec.action_units),
        )
        parameters = (
            rng.standard_normal((inputs, hidden)) / math.sqrt(inputs),
            np.zeros(hidden),
            rng.standard_normal((hidden, actions)) / math.sqrt(hidden),
            np.zeros(actions),
        )
        velocity = cast(Parameters, tuple(np.zeros_like(value) for value in parameters))
        self._state = _snapshot(
            ContinuousBCSnapshot(self._policy, parameters, velocity, 0, 0), self._policy
        )

    @property
    def policy(self) -> ContinuousBCPolicy:
        return _policy(self._policy)

    def snapshot_state(self) -> ContinuousBCSnapshot:
        return _snapshot(self._state, self._policy)

    def restore_state(self, snapshot: ContinuousBCSnapshot) -> None:
        candidate = _snapshot(snapshot, self._policy)
        self._state = candidate

    def fork(self) -> "NumpyContinuousBCLearner":
        child = object.__new__(NumpyContinuousBCLearner)
        child._policy = self.policy
        child._state = self.snapshot_state()
        return child

    def predict(self, features: Array) -> Array:
        state = self.snapshot_state()
        values = observations.standardize_observations(
            features, state.policy.observation_standardization, state.policy.spec
        )
        return core.continuous_bc_predict(values, state.parameters, state.policy.spec)

    def diagnose_training_episode(
        self, episode: ContinuousEpisode, stages: tuple[str, ...]
    ) -> residual.TrainingResidualDiagnostic:
        """One TRAIN-only forward; no updates, probe/final access or selection."""
        if self._policy.training_role != "train":
            raise ValueError("residual diagnosis requires a TRAIN policy")
        features, targets = continuous_training_pairs(episode, role="train")
        if episode.spec != self._policy.spec:
            raise ValueError("diagnostic episode units/bounds/spec differ from its policy")
        labels = residual._stage_labels(stages, len(features))
        predictions = self.predict(np.asarray(features, dtype=np.float64))
        report = residual.continuous_bc_residuals(
            predictions, np.asarray(targets, dtype=np.float64), episode.spec, labels
        )
        return residual.TrainingResidualDiagnostic(
            report, 1, len(features), predictions.size, predictions.nbytes
        )

    def train_episode(
        self, episode: ContinuousEpisode, *, row_weights: Array | None = None
    ) -> TrainingDiagnostic:
        features, targets = continuous_training_pairs(episode, role=self._policy.training_role)
        if episode.spec != self._policy.spec:
            raise ValueError("continuous BC episode units/bounds/spec differ from its policy")
        values = np.asarray(features, dtype=np.float64)
        actions = np.asarray(targets, dtype=np.float64)
        if row_weights is None:
            return self.train_batch(values, actions)
        return self.train_batch(values, actions, row_weights=row_weights)

    def train_head_gauss_newton_episode(
        self, episode: ContinuousEpisode
    ) -> HeadGaussNewtonTrainingResult:
        """Explicit TRAIN-only head update; the default train_episode stays SGD."""
        if type(self) is not NumpyContinuousBCLearner:
            raise ValueError("head update requires an exact continuous BP learner")
        policy = _policy(self._policy)
        if policy.training_role != "train" or policy.binary_gripper_axis is not None:
            raise ValueError("head update requires a TRAIN policy with the original MSE")
        owned = copy_continuous_episode(episode)
        features, targets = continuous_training_pairs(owned, role="train")
        if owned.spec != policy.spec:
            raise ValueError("head update episode units/bounds/spec differ from its policy")
        return self.train_head_gauss_newton_batch(
            np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        )

    def _require_head_source(self, before, native_state, native_policy) -> None:
        if self._state is not native_state or self._policy is not native_policy:
            raise RuntimeError("head update source owner was replaced")
        actual = _snapshot(self._state, _policy(self._policy))
        if (
            actual.policy != before.policy
            or actual.completed_updates != before.completed_updates
            or actual.training_examples != before.training_examples
            or any(
                not np.array_equal(a, b)
                for a, b in zip(
                    actual.parameters + actual.velocity,
                    before.parameters + before.velocity,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("head update changed its source before commit")

    def train_head_gauss_newton_batch(
        self, features: Array, targets: Array
    ) -> HeadGaussNewtonTrainingResult:
        """One bounded original-MSE head proposal; caller owns TRAIN release.

        Accepted updates preserve hidden weights AND hidden momentum. Head
        momentum is reset: a curvature step is not an SGD velocity and stale
        head momentum must not be added to this independently proposed step.
        Rejections/failures publish nothing and retain their attempted work.
        """
        if type(self) is not NumpyContinuousBCLearner:
            raise ValueError("head update requires an exact continuous BP learner")
        native_state, native_policy = self._state, self._policy
        before = _snapshot(native_state, _policy(native_policy))
        if before.policy.training_role != "train" or before.policy.binary_gripper_axis is not None:
            raise ValueError("head update requires a TRAIN policy with the original MSE")
        work: dict[str, int | bool] = dict(
            normalization_attempts=1,
            normalization_completed=0,
            normalization_rows=0,
            normalization_values=0,
            published_updates=0,
        )
        try:
            values = observations.standardize_observations(
                features, before.policy.observation_standardization, before.policy.spec
            )
            work.update(
                normalization_completed=1,
                normalization_rows=len(values),
                normalization_values=values.size,
            )
            proposal = head_update.propose_head_gauss_newton(
                values, targets, before.parameters, before.policy.spec
            )
            if type(proposal) is not head_update.HeadGaussNewtonProposal:
                raise ValueError("head update returned an invalid proposal")
            work.update(proposal.work)
            if not proposal.accepted:
                raise ValueError("head update rejected: " + proposal.reason)
            parameters = (
                before.parameters[0],
                before.parameters[1],
                proposal.head_weights,
                proposal.head_bias,
            )
            velocity = (
                before.velocity[0],
                before.velocity[1],
                np.zeros_like(before.velocity[2]),
                np.zeros_like(before.velocity[3]),
            )
            candidate = _snapshot(
                ContinuousBCSnapshot(
                    before.policy,
                    parameters,
                    velocity,
                    before.completed_updates + 1,
                    before.training_examples + len(values),
                ),
                before.policy,
            )
            diagnostic = TrainingDiagnostic(OBJECTIVE, proposal.loss_before)
            self._require_head_source(before, native_state, native_policy)
        except Exception as error:
            if isinstance(error, head_update.HeadGaussNewtonFailed):
                work.update(error.work)
            try:
                self._require_head_source(before, native_state, native_policy)
            except Exception as changed:
                raise TrainingUpdateFailed(str(changed), OBJECTIVE, work) from error
            raise TrainingUpdateFailed(
                str(error) or type(error).__name__, OBJECTIVE, work
            ) from error
        self._state = candidate
        work["published_updates"] = 1
        return HeadGaussNewtonTrainingResult(
            diagnostic, proposal.loss_after, proposal.step_norm, dict(work)
        )

    def diagnose_training_conditioning(
        self, episode: ContinuousEpisode
    ) -> residual.TrainingConditioningDiagnostic:
        """TRAIN-only owned readout; no fit, updates, RNG or budget mutation."""
        if type(self._policy.training_role) is not str or self._policy.training_role != "train":
            raise ValueError("conditioning diagnosis requires a TRAIN policy")
        features, targets = continuous_training_pairs(episode, role="train")
        state = self.snapshot_state()
        if episode.spec != state.policy.spec:
            raise ValueError("conditioning episode units/bounds/spec differ from its policy")
        values = observations.standardize_observations(
            np.asarray(features, dtype=np.float64),
            state.policy.observation_standardization,
            state.policy.spec,
        )
        report = residual.continuous_bc_conditioning(
            values,
            np.asarray(targets, dtype=np.float64),
            state.parameters,
            state.policy.spec,
            binary_gripper_axis=state.policy.binary_gripper_axis,
        )
        return residual.TrainingConditioningDiagnostic(
            report,
            1,
            len(values),
            values.size,
            state.policy.observation_standardization is not None,
        )

    def train_batch(
        self, features: Array, targets: Array, *, row_weights: Array | None = None
    ) -> TrainingDiagnostic:
        """Apply one atomic SGD step, optionally averaging positive row weights.

        Raw arrays still require caller release authority. Weights are batch
        supervision, never policy state; counters count rows, not their mass.
        """
        # Why: use the same native birth/commit guard as head updates. A public
        # snapshot callback must not replace the admitted source mid-update.
        native_state, native_policy = self._state, self._policy
        before = _snapshot(native_state, _policy(native_policy))
        values = observations.standardize_observations(
            features, before.policy.observation_standardization, before.policy.spec
        )
        actions = core._targets(targets, len(values), before.policy.spec)
        core._binary_gripper_targets(actions, before.policy.spec, before.policy.binary_gripper_axis)
        owned_weights = (
            core._row_weights(row_weights, len(values)) if row_weights is not None else None
        )
        if owned_weights is not None:
            core._row_masses(
                owned_weights
            )  # Refuse unrepresentable mass before gradient admission.
        objective = OBJECTIVE if before.policy.binary_gripper_axis is None else core.MIXED_OBJECTIVE
        if owned_weights is not None:
            objective = (
                core.WEIGHTED_OBJECTIVE
                if before.policy.binary_gripper_axis is None
                else core.WEIGHTED_MIXED_OBJECTIVE
            )
        work = dict(
            gradient_evaluations_attempted=1,
            gradient_evaluations_completed=0,
            parameter_proposals_attempted=0,
            parameter_proposals_completed=0,
            published_updates=0,
        )
        try:
            # Why: preserve the original callback signature and arithmetic when
            # no weights are supplied; explicit all-ones still records opt-in.
            if owned_weights is None:
                result = core.continuous_bc_loss_and_gradients(
                    values,
                    actions,
                    before.parameters,
                    before.policy.spec,
                    binary_gripper_axis=before.policy.binary_gripper_axis,
                )
            else:
                result = core.continuous_bc_loss_and_gradients(
                    values,
                    actions,
                    before.parameters,
                    before.policy.spec,
                    binary_gripper_axis=before.policy.binary_gripper_axis,
                    row_weights=owned_weights,
                )
            work["gradient_evaluations_completed"] = 1
            work["parameter_proposals_attempted"] = 1
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                velocity = cast(
                    Parameters,
                    tuple(
                        before.policy.momentum * old + gradient
                        for old, gradient in zip(before.velocity, result.gradients, strict=True)
                    ),
                )
                parameters = cast(
                    Parameters,
                    tuple(
                        old - before.policy.learning_rate * speed
                        for old, speed in zip(before.parameters, velocity, strict=True)
                    ),
                )
            candidate = _snapshot(
                ContinuousBCSnapshot(
                    before.policy,
                    parameters,
                    velocity,
                    before.completed_updates + 1,
                    before.training_examples + len(values),
                ),
                self._policy,
            )
            work["parameter_proposals_completed"] = 1
            diagnostic = TrainingDiagnostic(objective, result.loss)
            self._require_head_source(before, native_state, native_policy)
        except Exception as error:
            raise TrainingUpdateFailed(
                str(error) or type(error).__name__, objective, work
            ) from error
        self._state = candidate
        return diagnostic
