"""Count validated stored arrays in the supported exact NumPy snapshot families.

Outputs separate float64 parameters from chemistry, gates and unit histories.
No model constructor, native callback, training, restore or effective-weight
materialization occurs. Counts describe stored field bytes, not allocation/RSS
peaks, metadata, temporary copies or the quality of the represented function.
"""

from dataclasses import fields
from math import isfinite
from typing import Any

import numpy as np

from src.core.capacity_cost import CapacityUsage, count_dense_parameters
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig, _plasticity_gates
from src.core.coupled_energy import CoupledParameters, _validate_parameters
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    _copy_config,
    _validate_raw,
)
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledWakeState
from src.core.coupled_scaling_topology import CoupledScalingTopologyState, _copy_history
from src.core.coupled_structural_state import (
    CoupledStructuralState,
    _admit_transitions,
    _copy_chain,
)
from src.core.coupled_topology import CoupledTopologyState, _copy_lineage
from src.core.coupled_unit_history import UnitHistory, _validate as _validate_history
from src.core.coupled_unit_renewal import (
    RenewableUnitHistory,
    RenewalPolicy,
    _validate as _validate_renewable_history,
    copy_renewal_policy,
)
from src.core.coupled_width_pruning import _geometry
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.core.renewable_coupled_predictive_coding import RenewableCoupledSnapshot
from src.core.tracked_coupled_predictive_coding import TrackedCoupledSnapshot


def _record(value: Any, kind: Any) -> None:
    if type(value) is not kind or vars(value).keys() != {item.name for item in fields(kind)}:
        raise ValueError(f"capacity requires exact complete {kind.__name__}")


def _version(value: int, expected: int) -> None:
    if type(value) is not int or value != expected:
        raise ValueError("capacity snapshot version is unsupported")


def _parameters(parameters: CoupledParameters) -> tuple[int, tuple[int, ...], int]:
    _record(parameters, CoupledParameters)
    _validate_parameters(parameters)
    input_dim = parameters.hidden_weights[0].shape[0]
    widths = tuple(weight.shape[1] for weight in parameters.hidden_weights)
    return input_dim, widths, count_dense_parameters(input_dim, widths)


def _wake(
    wake: CoupledWakeState | None, width: int, expected: CoupledCircadianConfig | None = None
) -> int:
    if wake is None:
        if expected is not None:
            raise ValueError("capacity CPC state requires complete wake arrays")
        return 0
    _record(wake, CoupledWakeState)
    config = CoupledCircadianConfig(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
    )
    if expected is not None and config != expected:
        raise ValueError("capacity wake policy differs from raw configuration")
    if type(wake.wake_updates) is not int or wake.wake_updates < 0:
        raise ValueError("capacity wake count must be a nonnegative native integer")
    for value in (wake.chemical_state, wake.plasticity_state):
        if (
            type(value) is not np.ndarray
            or value.dtype != np.float64
            or value.shape != (width,)
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise ValueError("capacity wake arrays must be aligned finite nonnegative float64")
    if wake.wake_updates == 0 and (
        np.any(wake.chemical_state != 0) or np.any(wake.plasticity_state != 1)
    ):
        raise ValueError("zero-wake capacity state requires zero chemistry and unit gates")
    if not np.array_equal(wake.plasticity_state, _plasticity_gates(wake.chemical_state, config)):
        raise ValueError("capacity wake gates differ from chemistry")
    return wake.chemical_state.nbytes + wake.plasticity_state.nbytes


def _native(state: CoupledSnapshot | MatchedBackpropSnapshot) -> CapacityUsage:
    kind = type(state)
    _record(state, kind)
    _version(
        state.format_version,
        2 if kind is TrackedCoupledSnapshot else 3 if kind is RenewableCoupledSnapshot else 1,
    )
    _, widths, parameters = _parameters(state.parameters)
    if kind is MatchedBackpropSnapshot:
        return CapacityUsage(parameters, 8 * parameters, 0)
    assert isinstance(state, CoupledSnapshot)
    precisions = state.precisions
    if (
        type(precisions) is not tuple
        or len(precisions) != len(widths)
        or any(
            type(value) not in {int, float} or not 1e-6 <= value <= 1e6 or not isfinite(value)
            for value in precisions
        )
    ):
        raise ValueError("capacity precisions must match native hidden geometry")
    auxiliary = _wake(state.wake_state, widths[-1])
    if kind in {TrackedCoupledSnapshot, RenewableCoupledSnapshot}:
        assert isinstance(state, (TrackedCoupledSnapshot, RenewableCoupledSnapshot))
        history = state.unit_history
        expected = UnitHistory if kind is TrackedCoupledSnapshot else RenewableUnitHistory
        _record(history, expected)
        if kind is TrackedCoupledSnapshot:
            assert isinstance(history, UnitHistory)
            width = _validate_history(history)
        else:
            assert isinstance(history, RenewableUnitHistory)
            width = _validate_renewable_history(history)
            assert isinstance(state, RenewableCoupledSnapshot)
            _record(state.renewal_policy, RenewalPolicy)
            assert isinstance(state.renewal_policy, RenewalPolicy)
            copy_renewal_policy(state.renewal_policy)
        if width != widths[-1] or (
            state.wake_state is not None and state.wake_state.wake_updates != history.updates
        ):
            raise ValueError("capacity history geometry or CPC clock differs")
        auxiliary += (
            history.unit_ids.nbytes + history.importance_ema.nbytes + history.update_age.nbytes
        )
    return CapacityUsage(parameters, 8 * parameters, auxiliary)


def _raw(state: CoupledScalingState) -> CapacityUsage:
    _record(state, CoupledScalingState)
    _record(state.config, CoupledScalingConfig)
    config = _copy_config(state.config)
    _record(state.raw_parameters, CoupledParameters)
    _validate_raw(config, state.raw_parameters)
    if config.circadian is None and state.wake_state is not None:
        raise ValueError("capacity raw PC state cannot contain CPC arrays")
    auxiliary = _wake(state.wake_state, config.width, config.circadian)
    parameters = count_dense_parameters(config.input_dim, (config.width,) * config.hidden_layers)
    return CapacityUsage(parameters, 8 * parameters, auxiliary)


def measure_numpy_capacity(state: object) -> CapacityUsage:
    """Read exact supported stored fields without constructing or restoring models."""
    kind = type(state)
    if kind in {
        CoupledSnapshot,
        MatchedBackpropSnapshot,
        TrackedCoupledSnapshot,
        RenewableCoupledSnapshot,
    }:
        assert isinstance(state, (CoupledSnapshot, MatchedBackpropSnapshot))
        return _native(state)
    if kind is CoupledScalingState:
        assert isinstance(state, CoupledScalingState)
        return _raw(state)
    if kind is CoupledStructuralState:
        assert isinstance(state, CoupledStructuralState)
        _record(state, CoupledStructuralState)
        _version(state.format_version, 1)
        _admit_transitions(state.transitions)
        if type(state.raw_state) is not CoupledScalingState:
            raise ValueError("capacity structural state requires exact raw state")
        origin, current = _copy_config(state.origin_config), _copy_config(state.raw_state.config)
        _copy_chain(state.transitions, origin, current)
        return _raw(state.raw_state)
    if kind is CoupledScalingTopologyState:
        assert isinstance(state, CoupledScalingTopologyState)
        _record(state, CoupledScalingTopologyState)
        _version(state.format_version, 1)
        if type(state.raw_state) is not CoupledScalingState:
            raise ValueError("capacity raw topology requires exact raw state")
        current = _copy_config(state.raw_state.config)
        _copy_history(state.prune_lineage, current)
        return _raw(state.raw_state)
    if kind is CoupledTopologyState:
        assert isinstance(state, CoupledTopologyState)
        _record(state, CoupledTopologyState)
        _version(state.format_version, 1)
        if type(state.family) is not str or state.family not in {"pc", "cpc"}:
            raise ValueError("capacity effective topology requires pc/cpc family")
        _record(state.snapshot, CoupledSnapshot)
        d, widths, precisions = _geometry(
            state.input_dim, state.hidden_dims, state.snapshot.precisions
        )
        _copy_lineage(state.prune_lineage, state.family, d, widths, precisions)
        usage = _native(state.snapshot)
        actual_d, actual_widths, _ = _parameters(state.snapshot.parameters)
        if (actual_d, actual_widths) != (d, widths) or (state.snapshot.wake_state is not None) != (
            state.family == "cpc"
        ):
            raise ValueError("capacity effective topology differs from complete native state")
        return usage
    raise ValueError("capacity measurement supports only exact native NumPy coupled/BP snapshots")
