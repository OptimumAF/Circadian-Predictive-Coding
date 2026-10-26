"""Opt-in coupled PC/CPC models with complete committed native unit histories.

Default models and their format-1 snapshots remain unchanged. Format-2 tracked
snapshots explicitly carry history and are incompatible with legacy exact-state
codecs. Publication/restore is failure-atomic for synchronous single-owner use;
this module provides no structural edits, persistence, budgets or authority.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import Array, CoupledParameters
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.coupled_unit_history import (
    UnitHistory,
    advance_unit_history,
    copy_unit_history,
    initial_unit_history,
)


def _last_width(hidden_dims: tuple[int, ...]) -> int:
    if type(hidden_dims) is not tuple or not hidden_dims:
        raise ValueError("tracked hidden_dims must be a nonempty tuple")
    return hidden_dims[-1]


def _wake_binding(wake: CoupledWakeState | None, updates: int) -> None:
    if wake is not None and (
        type(wake) is not CoupledWakeState
        or type(wake.wake_updates) is not int
        or wake.wake_updates != updates
    ):
        raise ValueError("tracked CPC wake count must equal its history update count")


@dataclass(frozen=True)
class TrackedCoupledSnapshot(CoupledSnapshot):
    unit_history: UnitHistory | None = None

    def __post_init__(self) -> None:
        if (
            type(self) is not TrackedCoupledSnapshot
            or type(self.format_version) is not int
            or self.format_version != 2
        ):
            raise ValueError("tracked coupled snapshot requires format 2")
        if type(self.parameters) is not CoupledParameters:
            raise ValueError("tracked snapshot requires CoupledParameters")
        outgoing = self.parameters.output_weight
        if type(outgoing) is not np.ndarray or outgoing.ndim != 2 or outgoing.shape[1] != 1:
            raise ValueError("tracked snapshot requires a scalar-output weight matrix")
        if type(self.unit_history) is not UnitHistory:
            raise ValueError("tracked snapshot requires complete UnitHistory")
        owned = copy_unit_history(self.unit_history, outgoing.shape[0], self.unit_history.ema_decay)
        _wake_binding(self.wake_state, owned.updates)
        object.__setattr__(self, "unit_history", owned)


def _project(
    snapshot: CoupledSnapshot, width: int, decay: float
) -> tuple[CoupledSnapshot, UnitHistory]:
    if (
        type(snapshot) is not TrackedCoupledSnapshot
        or type(snapshot.format_version) is not int
        or snapshot.format_version != 2
    ):
        raise ValueError("tracked models require a complete format-2 TrackedCoupledSnapshot")
    if type(snapshot.unit_history) is not UnitHistory:
        raise ValueError("tracked snapshot requires complete UnitHistory")
    history = copy_unit_history(snapshot.unit_history, width, decay)
    _wake_binding(snapshot.wake_state, history.updates)
    # This internal projection is not an interchange format and never discards
    # history: the owned history is validated before native restore can publish.
    native = CoupledSnapshot(1, snapshot.parameters, snapshot.precisions, snapshot.wake_state)
    return native, history


def _snapshot(native: CoupledSnapshot, history: UnitHistory) -> TrackedCoupledSnapshot:
    return TrackedCoupledSnapshot(
        2, native.parameters, native.precisions, native.wake_state, history
    )


class TrackedCoupledPredictiveCodingNetwork(CoupledPredictiveCodingNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        seed: int,
        precisions: tuple[float, ...] | None = None,
        *,
        history_decay: float = 0.95,
    ) -> None:
        history = initial_unit_history(_last_width(hidden_dims), history_decay)
        super().__init__(input_dim, hidden_dims, seed, precisions)
        self._unit_history_decay, self._unit_history = history.ema_decay, history

    @property
    def unit_history_decay(self) -> float:
        return self._unit_history_decay

    def read_unit_history(self) -> UnitHistory:
        return copy_unit_history(self._unit_history, self.hidden_dims[-1], self.unit_history_decay)

    def _publish_training_state(
        self, candidate: CoupledParameters, raw_gradients: CoupledParameters
    ) -> None:
        history = advance_unit_history(self.read_unit_history(), raw_gradients.output_weight)
        self._parameters, self._unit_history = candidate, history

    def snapshot_state(self) -> TrackedCoupledSnapshot:
        return _snapshot(super().snapshot_state(), self.read_unit_history())

    def restore_state(self, snapshot: CoupledSnapshot) -> None:
        native, history = _project(snapshot, self.hidden_dims[-1], self.unit_history_decay)
        super().restore_state(native)
        self._unit_history = history


class TrackedCoupledCircadianPredictiveCodingNetwork(CoupledCircadianPredictiveCodingNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        seed: int,
        precisions: tuple[float, ...] | None = None,
        *,
        config: CoupledCircadianConfig | None = None,
        history_decay: float = 0.95,
    ) -> None:
        history = initial_unit_history(_last_width(hidden_dims), history_decay)
        if config is not None and type(config) is not CoupledCircadianConfig:
            raise ValueError("config must be CoupledCircadianConfig")
        owned_config = None if config is None else replace(config)
        super().__init__(input_dim, hidden_dims, seed, precisions, config=owned_config)
        self._unit_history_decay, self._unit_history = history.ema_decay, history

    @property
    def unit_history_decay(self) -> float:
        return self._unit_history_decay

    def read_unit_history(self) -> UnitHistory:
        history = copy_unit_history(
            self._unit_history, self.hidden_dims[-1], self.unit_history_decay
        )
        if type(self.wake_updates) is not int or self.wake_updates != history.updates:
            raise ValueError("tracked CPC wake count must equal its history update count")
        return history

    def _publish_wake_state(
        self,
        candidate: CoupledParameters,
        chemicals: Array,
        gates: Array,
        raw_gradients: CoupledParameters,
    ) -> None:
        history = advance_unit_history(self.read_unit_history(), raw_gradients.output_weight)
        (
            self._parameters,
            self._chemical_state,
            self._plasticity_state,
            self._wake_updates,
            self._unit_history,
        ) = (
            candidate,
            chemicals,
            gates,
            self._wake_updates + 1,
            history,
        )

    def snapshot_state(self) -> TrackedCoupledSnapshot:
        return _snapshot(super().snapshot_state(), self.read_unit_history())

    def restore_state(self, snapshot: CoupledSnapshot) -> None:
        native, history = _project(snapshot, self.hidden_dims[-1], self.unit_history_decay)
        super().restore_state(native)
        self._unit_history = history
