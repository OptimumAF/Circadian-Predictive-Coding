"""Owned bounded categorical diagnostics through the existing learner port.

The opt-in subtype leaves ordinary TrainingDiagnostic serialization unchanged.
Native candidate work and scheduled publication work have distinct scopes. No
training, data-role release, budget, IO or model payload belongs in these DTOs.
"""

from dataclasses import dataclass, fields, replace
from math import isfinite

from src.core.categorical_pc_inference import CategoricalInferenceDiagnostics, _WORK_FIELDS
from src.core.categorical_pc_weighted import OBJECTIVE
from src.core.coupled_inference import CoupledInferencePolicy, _validate_policy
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed

BOUNDED_OBJECTIVE_ID = f"torch_invariant_bounded_{OBJECTIVE}"

NATIVE_WORK_FIELDS = (
    *_WORK_FIELDS,
    "feature_projection_attempts",
    "native_state_restores_attempted",
    "staging_context_evaluations_attempted",
    "native_transitions_attempted",
    "native_transitions_completed",
    "candidate_validations_attempted",
    "candidate_validations_completed",
    "published_updates",
)
STREAM_WORK_FIELDS = (
    "candidate_forks_attempted",
    "candidate_forks_completed",
    "publication_restores_attempted",
    "publication_restores_completed",
    "publication_validations_attempted",
    "publication_validations_completed",
    "published_updates",
)


def _pairs(items, *, maximum):
    if type(items) is not tuple or len(items) > maximum:
        raise ValueError("work requires a bounded exact tuple of scalar pairs")
    owned = {}
    for pair in items:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("work requires exact key/value pairs")
        key, value = pair
        if type(key) is not str or not key or key != key.strip() or key in owned:
            raise ValueError("work requires distinct nonempty native identifiers")
        try:
            valid = len(key.encode("utf8")) <= 64
        except UnicodeEncodeError:
            valid = False
        if not valid or not (
            type(value) is bool or (type(value) is int and 0 <= value <= 2**53 - 1)
        ):
            raise ValueError("work requires bounded native scalar counters or flags")
        owned[key] = value
    return tuple((key, value) for key, value in owned.items())


def _inference(value):
    if type(value) is not CategoricalInferenceDiagnostics or set(vars(value)) != {
        field.name for field in fields(CategoricalInferenceDiagnostics)
    }:
        raise ValueError("an exact complete categorical inference diagnostic is required")
    if type(value.reason) is not str or value.reason not in {
        "max_steps",
        "gradient_tolerance",
        "relative_energy_tolerance",
        "nonfinite_candidate",
        "energy_increase",
        "numerical_failure",
    }:
        raise ValueError("categorical stopping reason is invalid")
    for name in ("initial_energy", "final_energy", "final_scaled_gradient_inf_norm"):
        scalar = getattr(value, name)
        if type(scalar) is not float or not isfinite(scalar) or scalar < 0:
            raise ValueError("categorical diagnostics require finite nonnegative native floats")
    for name in _WORK_FIELDS:
        scalar = getattr(value, name)
        flag = name in {"used_fallback", "failure_work_may_be_unobserved"}
        if type(scalar) is not (bool if flag else int) or (
            not flag and not 0 <= scalar <= 2**53 - 1
        ):
            raise ValueError("categorical inference work has invalid scalar types or bounds")
    if (
        not 1 <= value.rows <= 54
        or not 0 <= value.accepted_steps <= value.attempted_steps <= 64
        or value.rejected_steps != value.attempted_steps - value.accepted_steps
        or value.rejected_steps not in (0, 1)
        or value.used_fallback != (value.rejected_steps > 0)
        or value.attempted_latent_row_iterations != value.rows * value.attempted_steps
        or value.accepted_latent_row_iterations != value.rows * value.accepted_steps
        or value.final_energy > value.initial_energy
        or value.attempted_steps < 1
        or value.used_fallback
        != (value.reason in {"nonfinite_candidate", "energy_increase", "numerical_failure"})
    ):
        raise ValueError("categorical inference work or energy is inconsistent")
    for prefix in (
        "initial_prior_evaluations",
        "parameter_gradient_evaluations",
        "parameter_proposals",
        "final_energy_evaluations",
        "legacy_diagnostic_evaluations",
    ):
        if getattr(value, prefix + "_attempted") != 1 or getattr(value, prefix + "_completed") != 1:
            raise ValueError("successful categorical inference requires complete finalization")
    for prefix in ("energy_evaluations", "latent_gradient_evaluations"):
        if getattr(value, prefix + "_completed") > getattr(value, prefix + "_attempted"):
            raise ValueError("inference completions cannot exceed attempts")
    return replace(value)


def copy_categorical_native_work(items):
    """Validate/own the complete native namespace before merging port results."""
    native = _pairs(items, maximum=32)
    counts = dict(native)
    if set(counts) != set(NATIVE_WORK_FIELDS) or any(
        type(value)
        is not (bool if name in {"used_fallback", "failure_work_may_be_unobserved"} else int)
        for name, value in counts.items()
    ):
        raise ValueError("native work requires its exact32 typed scalar counters")
    return native


def _stream_pairs(items):
    owned = _pairs(items, maximum=8)
    counts = dict(owned)
    if set(counts) != set(STREAM_WORK_FIELDS) or any(type(v) is not int for v in counts.values()):
        raise ValueError("scheduled work requires its exact seven integer counters")
    for prefix in ("candidate_forks", "publication_restores", "publication_validations"):
        if counts[prefix + "_completed"] > counts[prefix + "_attempted"]:
            raise ValueError("scheduled completions cannot exceed attempts")
    if counts["published_updates"] > 1:
        raise ValueError("one scheduled update can publish at most once")
    return owned


@dataclass(frozen=True)
class CategoricalPCTrainingDiagnostic(TrainingDiagnostic):
    inference_policy: CoupledInferencePolicy
    inference: CategoricalInferenceDiagnostics
    native_work: tuple[tuple[str, int | bool], ...]
    stream_work: tuple[tuple[str, int | bool], ...]

    def __post_init__(self):
        super().__post_init__()
        if self.definition != BOUNDED_OBJECTIVE_ID or type(self.value) is not float:
            raise ValueError("bounded diagnostic requires its exact objective and native float")
        if type(self.inference_policy) is not CoupledInferencePolicy or set(
            vars(self.inference_policy)
        ) != {field.name for field in fields(CoupledInferencePolicy)}:
            raise ValueError("an exact complete stopping policy is required")
        policy = replace(self.inference_policy)
        _validate_policy(policy)
        inference = _inference(self.inference)
        native = copy_categorical_native_work(self.native_work)
        counts = dict(native)
        if (
            set(counts) != set(NATIVE_WORK_FIELDS)
            or any(
                type(counts[name]) is not int
                for name in NATIVE_WORK_FIELDS
                if name not in _WORK_FIELDS
            )
            or any(
                type(counts[name]) is not type(value) or counts[name] != value
                for name, value in inference.work.items()
            )
        ):
            raise ValueError("native work must preserve all categorical inference counters")
        if self.value != inference.final_energy:
            raise ValueError("training value must equal the accepted categorical energy")
        if policy.min_steps > 64 or (
            not inference.used_fallback and inference.accepted_steps < policy.min_steps
        ):
            raise ValueError("categorical stopping must respect its declared minimum and cap")
        if inference.reason == "gradient_tolerance" and (
            policy.gradient_tolerance is None
            or inference.final_scaled_gradient_inf_norm > policy.gradient_tolerance
        ):
            raise ValueError("gradient stopping requires its configured satisfied tolerance")
        if inference.reason == "relative_energy_tolerance" and (
            policy.relative_energy_tolerance is None
            or (
                policy.gradient_tolerance is not None
                and inference.final_scaled_gradient_inf_norm <= policy.gradient_tolerance
            )
        ):
            raise ValueError("relative stopping requires its configured criterion and precedence")
        if counts["failure_work_may_be_unobserved"] or any(
            counts[name] != 1 for name in NATIVE_WORK_FIELDS if name not in _WORK_FIELDS
        ):
            raise ValueError("successful native receipt requires its complete one-call work")
        object.__setattr__(self, "inference_policy", policy)
        object.__setattr__(self, "inference", inference)
        object.__setattr__(self, "native_work", native)
        stream = _stream_pairs(self.stream_work)
        if any(value != 1 for _, value in stream):
            raise ValueError("a successful scheduled diagnostic requires complete one-call work")
        object.__setattr__(self, "stream_work", stream)

    @property
    def work(self) -> dict[str, int | bool]:
        return dict(self.native_work)

    @property
    def scheduling_work(self) -> dict[str, int | bool]:
        return dict(self.stream_work)


class ScheduledCategoricalUpdateFailed(TrainingUpdateFailed):
    """Preserve private native work independently of scheduled publication."""

    def __init__(self, reason, definition, work, scheduling_work):
        super().__init__(reason, definition, work)
        if type(scheduling_work) is not dict:
            raise ValueError("scheduled failure work requires an owned scalar dictionary")
        self._scheduling_work = _stream_pairs(tuple(scheduling_work.items()))

    @property
    def scheduling_work(self) -> dict[str, int | bool]:
        return dict(self._scheduling_work)
