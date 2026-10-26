"""Owned rooted structural snapshots around the existing raw-coordinate BP port.

The strict inner learner owns feedforward BCE training; this adapter owns the
full origin/configuration chain across reconstruction. Compatible PC-only raw
data do not attest that BP, PC or a structural operation produced those tensors.
No transform, inference, chemistry, budget renewal or durable authority is added.
"""

from dataclasses import dataclass

from src.adapters.scaled_matched_backprop_learner import ScaledMatchedBackpropLearner
from src.core.coupled_energy import Array
from src.core.coupled_pc_scaling import CoupledScalingConfig, CoupledScalingState, _copy_config
from src.core.coupled_structural_state import (
    CoupledStructuralState,
    CoupledStructuralTransition,
    copy_coupled_structural_state,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import _learning_rate


def _require_pc(state: CoupledScalingState) -> None:
    if type(state) is not CoupledScalingState or type(state.config) is not CoupledScalingConfig:
        raise ValueError("structural BP requires exact raw coupled scaling state/config")
    if state.config.circadian is not None or state.wake_state is not None:
        raise ValueError(
            "structural BP requires PC-only raw state without circadian policy/wake state"
        )


@dataclass(frozen=True)
class _Payload:
    learner: ScaledMatchedBackpropLearner
    origin_config: CoupledScalingConfig
    transitions: tuple[CoupledStructuralTransition, ...]


class StructuralScaledMatchedBackpropLearner:
    """Keep full origin/history and a fixed BP rate on independent owned ports."""

    def __init__(self, initial: CoupledScalingState, *, learning_rate: float) -> None:
        self._learning_rate = _learning_rate(learning_rate)
        _require_pc(initial)
        state = CoupledStructuralState(1, initial.config, initial)
        self._origin_config = _copy_config(state.origin_config)
        self._payload = self._prepare_payload(state)

    def _prepare_payload(self, state: CoupledStructuralState) -> _Payload:
        if type(state) is not CoupledStructuralState:
            raise ValueError("structural BP requires exact CoupledStructuralState")
        _require_pc(state.raw_state)
        origin = _copy_config(state.origin_config)
        if origin.circadian is not None:
            raise ValueError("structural BP origin must be PC-only")
        # Why the full origin comparison: supported width/precision changes must
        # join the original root; relaxing the inner fixed-config BP port is unnecessary.
        canonical = copy_coupled_structural_state(state)
        if canonical.origin_config != self._origin_config:
            raise ValueError("structural BP state differs from the fixed full origin config")
        learner = ScaledMatchedBackpropLearner(
            canonical.raw_state, learning_rate=self._learning_rate
        )
        return _Payload(learner, canonical.origin_config, canonical.transitions)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        return self._payload.learner.train_batch(features, targets)

    def predict(self, features: Array) -> Array:
        return self._payload.learner.predict(features)

    def snapshot_state(self) -> CoupledStructuralState:
        payload = self._payload
        return CoupledStructuralState(
            1, payload.origin_config, payload.learner.snapshot_state(), payload.transitions
        )

    def restore_state(self, state: CoupledStructuralState) -> None:
        """Accept valid earlier histories; publish only after complete preparation."""
        prepared = self._prepare_payload(state)
        self._payload = prepared

    def build_from_state(
        self, state: CoupledStructuralState
    ) -> "StructuralScaledMatchedBackpropLearner":
        """Construct independent state with this prototype's full origin and rate."""
        prepared = self._prepare_payload(state)
        built = object.__new__(StructuralScaledMatchedBackpropLearner)
        built._learning_rate = self._learning_rate
        built._origin_config = _copy_config(self._origin_config)
        built._payload = prepared
        return built

    def fork(self) -> "StructuralScaledMatchedBackpropLearner":
        return self.build_from_state(self.snapshot_state())
