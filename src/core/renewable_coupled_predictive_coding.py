"""Opt-in fixed-width native unit replacement with complete owned continuation.

The caller enforces replacement allowance outside this model. A reset is not a
native training update and selects no examples or evaluation outcomes. Per-ID
RNG is deterministic and local; no mutable generator or retired history grows.
Old ordinary/tracked models, formats and defaults are intentionally unchanged.
"""

from dataclasses import dataclass, replace
from math import sqrt

import numpy as np

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
    _copy_wake_vector,
    _plasticity_gates,
)
from src.core.coupled_energy import Array, CoupledParameters, _validate_parameters
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.coupled_unit_renewal import (
    MAX_NEXT_UNIT_ID,
    RenewableUnitHistory,
    RenewalPolicy,
    RenewalResult,
    advance_renewal_history,
    copy_renewal_history,
    copy_renewal_policy,
    initial_renewal_history,
    select_renewal_indices,
)
from src.core.tracked_coupled_predictive_coding import _last_width, _wake_binding


@dataclass(frozen=True)
class RenewableCoupledSnapshot(CoupledSnapshot):
    unit_history: RenewableUnitHistory | None = None
    renewal_policy: RenewalPolicy | None = None

    def __post_init__(self) -> None:
        if (
            type(self) is not RenewableCoupledSnapshot
            or type(self.format_version) is not int
            or self.format_version != 3
        ):
            raise ValueError("renewable snapshots require exact format 3")
        if (
            type(self.parameters) is not CoupledParameters
            or type(self.unit_history) is not RenewableUnitHistory
        ):
            raise ValueError("renewable snapshot requires coupled parameters and complete history")
        outgoing = self.parameters.output_weight
        if type(outgoing) is not np.ndarray or outgoing.ndim != 2 or outgoing.shape[1] != 1:
            raise ValueError("renewable snapshot requires a scalar-output matrix")
        history = copy_renewal_history(
            self.unit_history, outgoing.shape[0], self.unit_history.ema_decay
        )
        if type(self.renewal_policy) is not RenewalPolicy:
            raise ValueError("renewable snapshot requires complete RenewalPolicy")
        policy = copy_renewal_policy(self.renewal_policy)
        _wake_binding(self.wake_state, history.updates)
        object.__setattr__(self, "unit_history", history)
        object.__setattr__(self, "renewal_policy", policy)


def _project(
    snapshot: CoupledSnapshot, width: int, decay: float, policy: RenewalPolicy
) -> tuple[CoupledSnapshot, RenewableUnitHistory]:
    if (
        type(snapshot) is not RenewableCoupledSnapshot
        or type(snapshot.format_version) is not int
        or snapshot.format_version != 3
    ):
        raise ValueError("renewable models require complete format-3 RenewableCoupledSnapshot")
    if (
        type(snapshot.unit_history) is not RenewableUnitHistory
        or type(snapshot.renewal_policy) is not RenewalPolicy
    ):
        raise ValueError("renewable snapshot requires history and policy")
    supplied = copy_renewal_policy(snapshot.renewal_policy)
    if supplied != policy:
        raise ValueError("snapshot renewal policy differs from this model")
    history = copy_renewal_history(snapshot.unit_history, width, decay)
    _wake_binding(snapshot.wake_state, history.updates)
    return CoupledSnapshot(
        1, snapshot.parameters, snapshot.precisions, snapshot.wake_state
    ), history


def _snapshot(
    native: CoupledSnapshot, history: RenewableUnitHistory, policy: RenewalPolicy
) -> RenewableCoupledSnapshot:
    return RenewableCoupledSnapshot(
        3, native.parameters, native.precisions, native.wake_state, history, policy
    )


def _reset_parameters(
    parameters: CoupledParameters,
    history: RenewableUnitHistory,
    policy: RenewalPolicy,
    indices: tuple[int, ...],
) -> tuple[CoupledParameters, RenewableUnitHistory, RenewalResult]:
    _validate_parameters(parameters)
    width = parameters.output_weight.shape[0]
    history = copy_renewal_history(history, width, history.ema_decay)
    policy = copy_renewal_policy(policy)
    if (
        type(indices) is not tuple
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or not 0 <= i < width for i in indices)
    ):
        raise ValueError("renewal indices must be distinct native indices within the current width")
    if history.next_unit_id + len(indices) > MAX_NEXT_UNIT_ID:
        raise ValueError("renewal unit ID counter is exhausted")
    weights = tuple(weight.copy() for weight in parameters.hidden_weights)
    biases = tuple(bias.copy() for bias in parameters.hidden_biases)
    outgoing, output_bias = parameters.output_weight.copy(), parameters.output_bias.copy()
    ids, importance, ages = (
        history.unit_ids.copy(),
        history.importance_ema.copy(),
        history.update_age.copy(),
    )
    new_ids = tuple(range(history.next_unit_id, history.next_unit_id + len(indices)))
    old_ids = tuple(int(history.unit_ids[index]) for index in indices)
    fanin = weights[-1].shape[0]
    for index, new_id in zip(indices, new_ids):
        rng = np.random.default_rng(np.random.SeedSequence([policy.seed, new_id]))
        weights[-1][:, index] = rng.normal(0.0, 1 / sqrt(fanin), size=fanin)
        biases[-1][0, index] = 0.0
        outgoing[index, :] = 0.0
        ids[index], importance[index], ages[index] = new_id, 0.0, 0.0
    candidate = CoupledParameters(weights, biases, outgoing, output_bias)
    _validate_parameters(candidate)
    renewed = RenewableUnitHistory(
        2,
        history.ema_decay,
        ids,
        importance,
        ages,
        history.updates,
        history.next_unit_id + len(indices),
    )
    return candidate, renewed, RenewalResult(indices, old_ids, new_ids)


def _wake_vectors(
    wake: CoupledWakeState, config: CoupledCircadianConfig, width: int, updates: int
) -> tuple[Array, Array]:
    _wake_binding(wake, updates)
    if type(wake) is not CoupledWakeState:
        raise ValueError("renewable CPC requires complete wake state")
    policy = CoupledCircadianConfig(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
    )
    if policy != config:
        raise ValueError("renewable CPC chemistry policy differs")
    chemicals = _copy_wake_vector(wake.chemical_state, width, "chemical_state")
    gates = _copy_wake_vector(wake.plasticity_state, width, "plasticity_state")
    if not np.array_equal(gates, _plasticity_gates(chemicals, policy)):
        raise ValueError("renewable CPC gates disagree with chemistry")
    if updates == 0 and (np.any(chemicals != 0) or np.any(gates != 1)):
        raise ValueError("zero-update renewable CPC requires initial wake state")
    return chemicals, gates


class RenewableCoupledPredictiveCodingNetwork(CoupledPredictiveCodingNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        seed: int,
        precisions: tuple[float, ...] | None = None,
        *,
        renewal_policy: RenewalPolicy,
        history_decay: float = 0.95,
    ) -> None:
        policy = copy_renewal_policy(renewal_policy)
        history = initial_renewal_history(_last_width(hidden_dims), history_decay)
        super().__init__(input_dim, hidden_dims, seed, precisions)
        self._renewal_policy, self._unit_history = policy, history
        self._unit_history_decay = history.ema_decay

    @property
    def renewal_policy(self) -> RenewalPolicy:
        return copy_renewal_policy(self._renewal_policy)

    @property
    def unit_history_decay(self) -> float:
        return self._unit_history_decay

    def read_unit_history(self) -> RenewableUnitHistory:
        return copy_renewal_history(
            self._unit_history, self.hidden_dims[-1], self.unit_history_decay
        )

    def _publish_training_state(
        self, candidate: CoupledParameters, raw_gradients: CoupledParameters
    ) -> None:
        history = advance_renewal_history(self.read_unit_history(), raw_gradients.output_weight)
        self._parameters, self._unit_history = candidate, history

    def snapshot_state(self) -> RenewableCoupledSnapshot:
        return _snapshot(super().snapshot_state(), self.read_unit_history(), self.renewal_policy)

    def restore_state(self, snapshot: CoupledSnapshot) -> None:
        native, history = _project(
            snapshot, self.hidden_dims[-1], self.unit_history_decay, self.renewal_policy
        )
        super().restore_state(native)
        self._unit_history = history

    def renew_units(self, max_units: int) -> RenewalResult:
        history, policy = self.read_unit_history(), self.renewal_policy
        indices = select_renewal_indices(history, policy, max_units)
        if not indices:
            return RenewalResult((), (), ())
        candidate, renewed, result = _reset_parameters(
            self.snapshot_parameters(), history, policy, indices
        )
        candidate = self._copy_parameters(candidate)
        renewed = copy_renewal_history(renewed, self.hidden_dims[-1], self.unit_history_decay)
        self._parameters, self._unit_history = candidate, renewed
        return result


class RenewableCoupledCircadianPredictiveCodingNetwork(CoupledCircadianPredictiveCodingNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        seed: int,
        precisions: tuple[float, ...] | None = None,
        *,
        renewal_policy: RenewalPolicy,
        config: CoupledCircadianConfig | None = None,
        history_decay: float = 0.95,
    ) -> None:
        policy = copy_renewal_policy(renewal_policy)
        history = initial_renewal_history(_last_width(hidden_dims), history_decay)
        if config is not None and type(config) is not CoupledCircadianConfig:
            raise ValueError("config must be CoupledCircadianConfig")
        super().__init__(
            input_dim,
            hidden_dims,
            seed,
            precisions,
            config=None if config is None else replace(config),
        )
        self._renewal_policy, self._unit_history = policy, history
        self._unit_history_decay = history.ema_decay

    @property
    def renewal_policy(self) -> RenewalPolicy:
        return copy_renewal_policy(self._renewal_policy)

    @property
    def unit_history_decay(self) -> float:
        return self._unit_history_decay

    def read_unit_history(self) -> RenewableUnitHistory:
        history = copy_renewal_history(
            self._unit_history, self.hidden_dims[-1], self.unit_history_decay
        )
        if type(self.wake_updates) is not int or self.wake_updates != history.updates:
            raise ValueError("renewable CPC wake count must equal history updates")
        return history

    def _publish_wake_state(
        self,
        candidate: CoupledParameters,
        chemicals: Array,
        gates: Array,
        raw_gradients: CoupledParameters,
    ) -> None:
        history = advance_renewal_history(self.read_unit_history(), raw_gradients.output_weight)
        (
            self._parameters,
            self._chemical_state,
            self._plasticity_state,
            self._wake_updates,
            self._unit_history,
        ) = (candidate, chemicals, gates, self._wake_updates + 1, history)

    def snapshot_state(self) -> RenewableCoupledSnapshot:
        return _snapshot(super().snapshot_state(), self.read_unit_history(), self.renewal_policy)

    def restore_state(self, snapshot: CoupledSnapshot) -> None:
        native, history = _project(
            snapshot, self.hidden_dims[-1], self.unit_history_decay, self.renewal_policy
        )
        super().restore_state(native)
        self._unit_history = history

    def renew_units(self, max_units: int) -> RenewalResult:
        history, policy = self.read_unit_history(), self.renewal_policy
        indices = select_renewal_indices(history, policy, max_units)
        if not indices:
            return RenewalResult((), (), ())
        wake = super().snapshot_state().wake_state
        if type(wake) is not CoupledWakeState:
            raise ValueError("renewable CPC requires native wake state")
        chemicals, gates = _wake_vectors(wake, self.config, self.hidden_dims[-1], history.updates)
        candidate, renewed, result = _reset_parameters(
            self.snapshot_parameters(), history, policy, indices
        )
        chemicals[list(indices)], gates[list(indices)] = 0.0, 1.0
        candidate = self._copy_parameters(candidate)
        renewed = copy_renewal_history(renewed, self.hidden_dims[-1], self.unit_history_decay)
        prospective = replace(wake, chemical_state=chemicals, plasticity_state=gates)
        chemicals, gates = _wake_vectors(
            prospective, self.config, self.hidden_dims[-1], renewed.updates
        )
        self._parameters, self._chemical_state, self._plasticity_state, self._unit_history = (
            candidate,
            chemicals,
            gates,
            renewed,
        )
        return result
