"""Complete coupled learner metadata, separate from native array validation.

Policy records retain the fixed update rates and optional guarded inference
settings; checkpoint records pair that policy with a complete native snapshot.
Why separate validation: outer metadata must be admitted before scanning or
copying native arrays. No array validation/copy, IO, models, execution or recovery
authority lives here; the existing native codec owns array/value validation.
"""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, cast

from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledWakeState


def _record(value: object, kind: type[Any]) -> None:
    if type(value) is not kind or vars(value).keys() != {
        item.name for item in fields(cast(Any, kind))
    }:
        raise ValueError(f"learner checkpoint requires exact complete {kind.__name__}")


def _number(value: object, name: str, *, positive: bool) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name} requires a finite native number")
    try:
        resolved = float(cast(float, value))
    except OverflowError:
        raise ValueError(f"{name} exceeds finite native range") from None
    if not isfinite(resolved) or (resolved <= 0 if positive else resolved < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")


@dataclass(frozen=True)
class CoupledUpdatePolicy:
    learning_rate: float
    inference_steps: int
    inference_learning_rate: float
    inference_policy: CoupledInferencePolicy | None = None

    def __post_init__(self) -> None:
        validate_update_policy(self)


def validate_update_policy(policy: object) -> None:
    _record(policy, CoupledUpdatePolicy)
    assert isinstance(policy, CoupledUpdatePolicy)
    _number(policy.learning_rate, "learning_rate", positive=True)
    _number(policy.inference_learning_rate, "inference_learning_rate", positive=True)
    if type(policy.inference_steps) is not int or not 1 <= policy.inference_steps <= 10000:
        raise ValueError("inference_steps must be a native integer in 1..10000")
    guarded = policy.inference_policy
    if guarded is None:
        return
    _record(guarded, CoupledInferencePolicy)
    if type(guarded.min_steps) is not int or not 1 <= guarded.min_steps <= policy.inference_steps:
        raise ValueError("guarded min_steps must be a native integer within the inference cap")
    for name in ("gradient_tolerance", "relative_energy_tolerance"):
        value = getattr(guarded, name)
        if value is not None:
            _number(value, name, positive=False)


def copy_update_policy(policy: CoupledUpdatePolicy) -> CoupledUpdatePolicy:
    """Revalidate and detach metadata, including the optional guarded record."""
    validate_update_policy(policy)
    guarded = policy.inference_policy
    return CoupledUpdatePolicy(
        policy.learning_rate,
        policy.inference_steps,
        policy.inference_learning_rate,
        None
        if guarded is None
        else CoupledInferencePolicy(
            guarded.min_steps, guarded.gradient_tolerance, guarded.relative_energy_tolerance
        ),
    )


@dataclass(frozen=True)
class CoupledLearnerCheckpoint:
    format_version: int
    policy: CoupledUpdatePolicy
    native_state: CoupledSnapshot

    def __post_init__(self) -> None:
        validate_learner_checkpoint(self)


def validate_learner_checkpoint(state: object) -> None:
    _record(state, CoupledLearnerCheckpoint)
    assert isinstance(state, CoupledLearnerCheckpoint)
    if type(state.format_version) is not int or state.format_version != 1:
        raise ValueError("learner checkpoint requires exact format_version 1")
    validate_update_policy(state.policy)
    snapshot = state.native_state
    _record(snapshot, CoupledSnapshot)
    if type(snapshot.format_version) is not int or snapshot.format_version != 1:
        raise ValueError("learner checkpoint requires exact native snapshot format 1")
    _record(snapshot.parameters, CoupledParameters)
    if type(snapshot.precisions) is not tuple:
        raise ValueError("learner checkpoint native precisions must be an exact tuple")
    if snapshot.wake_state is not None:
        _record(snapshot.wake_state, CoupledWakeState)
