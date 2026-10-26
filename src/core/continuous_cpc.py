"""Owned wake chemistry for the matched one-hidden continuous PC adapter.

Reuse the existing coupled CPC config, wake record and gate equations. Inputs
are quiescent accepted latents/gradients and owned prior wake state. No parameter
optimizer, role policy, inference loop, simulator, history, selection or IO.
"""

import numpy as np

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    _copy_wake_vector,
    _gated_gradients,
    _plasticity_gates,
)
from src.core.coupled_energy import Array, CoupledParameters
from src.core.coupled_predictive_coding import CoupledWakeState


def copy_continuous_config(value: CoupledCircadianConfig | None) -> CoupledCircadianConfig | None:
    if value is None:
        return None
    if type(value) is not CoupledCircadianConfig:
        raise ValueError("circadian must be an exact CoupledCircadianConfig or None")
    return CoupledCircadianConfig(
        value.chemical_decay,
        value.chemical_buildup_rate,
        value.plasticity_sensitivity,
        value.min_plasticity,
    )


def _record(config, chemicals, gates, updates) -> CoupledWakeState:
    return CoupledWakeState(
        config.chemical_decay,
        config.chemical_buildup_rate,
        config.plasticity_sensitivity,
        config.min_plasticity,
        chemicals,
        gates,
        updates,
    )


def copy_continuous_wake(
    value: CoupledWakeState | None,
    config: CoupledCircadianConfig | None,
    width: int,
    updates: int,
) -> CoupledWakeState | None:
    if type(width) is not int or not 1 <= width <= 256:
        raise ValueError("continuous CPC width must be native in1..256")
    if type(updates) is not int or not 0 <= updates <= 2**53 - 1:
        raise ValueError("continuous CPC updates must be a bounded native count")
    owned = copy_continuous_config(config)
    if owned is None:
        if value is not None:
            raise ValueError("PC mode requires no wake state")
        return None
    if type(value) is not CoupledWakeState:
        raise ValueError("CPC mode requires an exact complete CoupledWakeState")
    bound = CoupledCircadianConfig(
        value.chemical_decay,
        value.chemical_buildup_rate,
        value.plasticity_sensitivity,
        value.min_plasticity,
    )
    if bound != owned or type(value.wake_updates) is not int or value.wake_updates != updates:
        raise ValueError("wake policy/count differs from the continuous learner")
    chemicals = _copy_wake_vector(value.chemical_state, width, "chemical_state")
    gates = _copy_wake_vector(value.plasticity_state, width, "plasticity_state")
    if updates == 0 and (np.any(chemicals != 0) or np.any(gates != 1)):
        raise ValueError("zero updates require zero chemistry and unit gates")
    if not np.array_equal(gates, _plasticity_gates(chemicals, owned)):
        raise ValueError("wake gates disagree with chemistry and policy")
    return _record(owned, chemicals, gates, updates)


def initial_continuous_wake(
    config: CoupledCircadianConfig | None, width: int
) -> CoupledWakeState | None:
    owned = copy_continuous_config(config)
    if type(width) is not int or not 1 <= width <= 256:
        raise ValueError("continuous CPC width must be native in1..256")
    if owned is None:
        return None
    return _record(owned, np.zeros(width, dtype=np.float64), np.ones(width, dtype=np.float64), 0)


def advance_continuous_wake(
    previous: CoupledWakeState,
    latents: tuple[Array, ...],
    config: CoupledCircadianConfig,
) -> CoupledWakeState:
    if (
        type(previous) is not CoupledWakeState
        or type(latents) is not tuple
        or len(latents) != 1
        or type(latents[0]) is not np.ndarray
        or latents[0].dtype != np.float64
        or latents[0].ndim != 2
        or not 1 <= len(latents[0]) <= 256
        or not np.isfinite(latents[0]).all()
    ):
        raise ValueError("continuous CPC needs one finite float64 accepted latent batch")
    width = latents[0].shape[1]
    owned = copy_continuous_config(config)
    if owned is None:
        raise ValueError("CPC advance requires a circadian configuration")
    old = copy_continuous_wake(previous, owned, width, previous.wake_updates)
    assert old is not None
    if old.wake_updates == 2**53 - 1:
        raise ValueError("continuous CPC wake counter is exhausted")
    # Same final-activity/decay/buildup equation as the existing binary CPC.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        activity = np.mean(np.abs(latents[0]), axis=0)
        chemicals = (
            owned.chemical_decay * old.chemical_state + owned.chemical_buildup_rate * activity
        )
    if not np.isfinite(chemicals).all() or np.any(chemicals < 0):
        raise FloatingPointError("nonfinite or negative candidate continuous chemistry")
    gates = _plasticity_gates(chemicals, owned)
    return _record(owned, chemicals, gates, old.wake_updates + 1)


def gate_continuous_gradients(gradients: CoupledParameters, gates: Array) -> CoupledParameters:
    """Gate the accepted energy partials; output bias and old velocity are untouched."""
    if (
        type(gates) is not np.ndarray
        or gates.dtype != np.float64
        or gates.shape != (gradients.output_weight.shape[0],)
        or not np.isfinite(gates).all()
        or np.any(gates < 1e-6)
        or np.any(gates > 1)
    ):
        raise ValueError("continuous CPC gates require a finite matched float64 vector")
    return _gated_gradients(gradients, gates)
