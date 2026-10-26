"""Bounded PC/neutral updates through separate TRAIN and reserved-probe ports.

Uses the existing weighted adapter's owned native transition and atomic donor
boundary. Returns scalar stopping diagnostics and owned work; it supplies no
schedule, budget, role release, evaluation, probe selection or promotion.
"""

from dataclasses import dataclass, replace
from typing import Any

from src.adapters import torch_pc_weighted as weighted
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_weighted import _record
from src.core.categorical_pc_inference import (
    CategoricalInferenceDiagnostics,
    _WORK_FIELDS,
    bounded_class_weighted_pc_update,
)
from src.core.categorical_pc_weighted import OBJECTIVE
from src.core.coupled_inference import CoupledInferencePolicy, _validate_policy
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.shared.torch_runtime import require_torch

TORCH_BOUNDED_WEIGHTED_PC_ID = f"torch_invariant_bounded_{OBJECTIVE}"
_ADAPTER_WORK = (
    "feature_projection_attempts",
    "native_state_restores_attempted",
    "staging_context_evaluations_attempted",
    "native_transitions_attempted",
    "native_transitions_completed",
    "candidate_validations_attempted",
    "candidate_validations_completed",
    "published_updates",
)


@dataclass(frozen=True)
class BoundedPCTrainingUpdate:
    """No model tensors escape; work is detached from subsequent native updates."""

    diagnostic: TrainingDiagnostic
    inference: CategoricalInferenceDiagnostics
    _work_items: tuple[tuple[str, int | bool], ...]

    @property
    def work(self) -> dict[str, int | bool]:
        return dict(self._work_items)


def _policy(policy: CoupledInferencePolicy) -> CoupledInferencePolicy:
    _record(policy, CoupledInferencePolicy)
    owned = replace(policy)
    _validate_policy(owned)
    return owned


def _same_policy(actual: CoupledInferencePolicy, expected: CoupledInferencePolicy) -> bool:
    # Why: numeric equality would hide an int/float or signed-zero replacement.
    validated = _policy(actual)
    for name, value in vars(validated).items():
        previous = getattr(expected, name)
        if type(value) is not type(previous):
            return False
        unchanged = value.hex() == previous.hex() if type(value) is float else value == previous
        if not unchanged:
            return False
    return True


class _ProposalFailure(RuntimeError):
    """Identify only a failure caught directly at the bounded numerical port."""

    def __init__(self, failure: TrainingUpdateFailed):
        self.failure = failure
        super().__init__(str(failure))


def _stage(head, values, labels, policy, weights, inference_policy, work):
    torch = require_torch()
    with torch.no_grad():
        try:
            result = bounded_class_weighted_pc_update(
                values,
                labels,
                tuple(getattr(head, name) for name in weighted.native._PARAMETERS),
                class_weights=weights,
                learning_rate=policy.native_policy.learning_rate,
                inference_steps=policy.native_policy.inference_steps,
                inference_learning_rate=policy.native_policy.inference_learning_rate,
                role="train",
                policy=inference_policy,
            )
        except TrainingUpdateFailed as error:
            work.update(error.work)
            raise _ProposalFailure(error) from error
        work.update(result.diagnostics.work)
        weighted._apply_proposal(head, values, labels, policy, result.update, work=work)
    return TrainingDiagnostic(
        TORCH_BOUNDED_WEIGHTED_PC_ID, result.update.energy
    ), result.diagnostics


def _train_bounded_episode(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
    *,
    policy: CoupledInferencePolicy,
    expected_role: str,
) -> BoundedPCTrainingUpdate:
    """Publish one complete native update with explicit bounded relaxation.

    Admission/mode errors precede native work. Later failures preserve observed
    work in TrainingUpdateFailed; unexpected adapter failures flag unknown work.
    Snapshot/admission/copy/closing checks are outside numerical work counters.
    Foreign mutations are refused before publication and never rolled back.
    """
    owned_policy = _policy(policy)
    prepared = weighted._admit(learner, episode, class_weights, expected_role=expected_role)
    if owned_policy.min_steps > prepared[1].native_policy.inference_steps:
        raise ValueError("min_steps must not exceed native inference_steps")
    if require_torch().is_inference_mode_enabled():
        raise ValueError("bounded PC requires non-inference mode")
    work: dict[str, Any] = dict.fromkeys((*_WORK_FIELDS, *_ADAPTER_WORK), 0)
    work.update(
        rows=len(prepared[0].transitions),
        used_fallback=False,
        failure_work_may_be_unobserved=False,
    )

    def stage(head, values, labels, native_policy, weights):
        return _stage(head, values, labels, native_policy, weights, owned_policy, work)

    try:
        diagnostic, inference = weighted._execute_episode(
            learner,
            episode,
            class_weights,
            stage,
            prepared=prepared,
            work=work,
            guard=lambda: _same_policy(policy, owned_policy),
            expected_role=expected_role,
        )
        return BoundedPCTrainingUpdate(diagnostic, inference, tuple(work.items()))
    except _ProposalFailure as error:
        raise TrainingUpdateFailed(
            str(error), TORCH_BOUNDED_WEIGHTED_PC_ID, work
        ) from error.failure
    except Exception as error:
        work["failure_work_may_be_unobserved"] = True
        raise TrainingUpdateFailed(
            str(error) or type(error).__name__, TORCH_BOUNDED_WEIGHTED_PC_ID, work
        ) from error


def train_bounded_weighted_episode(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
    *,
    policy: CoupledInferencePolicy,
) -> BoundedPCTrainingUpdate:
    """Bounded update admitted exclusively from an original train episode."""
    return _train_bounded_episode(
        learner, episode, class_weights, policy=policy, expected_role="train"
    )


def train_bounded_probe_episode(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
    *,
    policy: CoupledInferencePolicy,
) -> BoundedPCTrainingUpdate:
    """Bounded update admitted exclusively from an original probe_train episode."""
    return _train_bounded_episode(
        learner, episode, class_weights, policy=policy, expected_role="probe_train"
    )
