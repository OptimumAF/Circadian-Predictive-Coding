"""Opt-in wake chemistry on the shared coupled predictive-coding update.

Chemistry follows final inferred activity in the last hidden layer. Its gates
scale that layer's incoming columns/bias and outgoing rows. No sleep, replay,
structural change, runtime adapter or legacy configuration translation is provided.
Parameter restore is inherited and parameter-only: chemistry/gates/wake count are
preserved. Full snapshots include wake state and require the same policy on restore;
they do not restore adapter rates, runtime budgets or authority.

Updates are failure-atomic under synchronous single-owner use, not thread-safe.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.core.coupled_energy import Array, CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledTrainResult,
    CoupledWakeState,
)


@dataclass(frozen=True)
class CoupledCircadianConfig:
    """Finite wake-only chemistry settings; unrelated legacy flags are refused."""

    chemical_decay: float = 0.995
    chemical_buildup_rate: float = 0.02
    plasticity_sensitivity: float = 0.7
    min_plasticity: float = 0.2

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("chemical_decay", self.chemical_decay, 0.0, 1.0),
            ("chemical_buildup_rate", self.chemical_buildup_rate, 0.0, 1e6),
            ("plasticity_sensitivity", self.plasticity_sensitivity, 0.0, 1e6),
            ("min_plasticity", self.min_plasticity, 1e-6, 1.0),
        ):
            # Range first also refuses arbitrary-size ints before float conversion.
            if (
                type(value) not in (int, float)
                or not lower <= value <= upper
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a native finite number in [{lower}, {upper}]")

    @classmethod
    def matched_pc_control(cls) -> "CoupledCircadianConfig":
        """Keep real chemistry updates while forcing every plasticity gate to one."""
        return cls(min_plasticity=1.0)


def _gated_gradients(gradients: CoupledParameters, gates: Array) -> CoupledParameters:
    return CoupledParameters(
        (*gradients.hidden_weights[:-1], gradients.hidden_weights[-1] * gates),
        (*gradients.hidden_biases[:-1], gradients.hidden_biases[-1] * gates),
        gradients.output_weight * gates[:, None],
        gradients.output_bias,
    )


def _plasticity_gates(chemicals: Array, config: CoupledCircadianConfig) -> Array:
    # Finite nonnegative chemistry can saturate exp's nonpositive exponent.
    # -inf maps to zero and then the configured floor, never invalid chemistry.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        return np.clip(
            np.exp(-config.plasticity_sensitivity * chemicals), config.min_plasticity, 1.0
        )


def _copy_wake_vector(value: Array, width: int, name: str) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.shape != (width,)
        or not np.isfinite(value).all()
        or np.any(value < 0)
    ):
        raise ValueError(f"{name} must be a finite nonnegative float64 vector of shape {(width,)}")
    return value.copy()


class CoupledCircadianPredictiveCodingNetwork(CoupledPredictiveCodingNetwork):
    """Real wake-state extension; all inference and candidate checks share PC code."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        seed: int,
        precisions: tuple[float, ...] | None = None,
        *,
        config: CoupledCircadianConfig | None = None,
    ) -> None:
        resolved = CoupledCircadianConfig() if config is None else config
        if type(resolved) is not CoupledCircadianConfig:
            raise ValueError("config must be CoupledCircadianConfig")
        super().__init__(input_dim, hidden_dims, seed, precisions)
        self._config = resolved
        self._chemical_state = np.zeros(hidden_dims[-1], dtype=np.float64)
        self._plasticity_state = np.ones(hidden_dims[-1], dtype=np.float64)
        self._wake_updates = 0

    @property
    def config(self) -> CoupledCircadianConfig:
        return self._config

    @property
    def wake_updates(self) -> int:
        return self._wake_updates

    def get_chemical_state(self) -> Array:
        """Return detached last-hidden-layer chemistry, shape (last hidden width,)."""
        return self._chemical_state.copy()

    def get_plasticity_state(self) -> Array:
        """Return detached gates from the last committed wake (initially all one)."""
        return self._plasticity_state.copy()

    def snapshot_state(self) -> CoupledSnapshot:
        """Return owned parameters and wake state for continuation under this policy."""
        return CoupledSnapshot(
            format_version=1,
            parameters=self.snapshot_parameters(),
            precisions=self.precisions,
            wake_state=CoupledWakeState(
                chemical_decay=self.config.chemical_decay,
                chemical_buildup_rate=self.config.chemical_buildup_rate,
                plasticity_sensitivity=self.config.plasticity_sensitivity,
                min_plasticity=self.config.min_plasticity,
                chemical_state=self.get_chemical_state(),
                plasticity_state=self.get_plasticity_state(),
                wake_updates=self.wake_updates,
            ),
        )

    def restore_state(self, snapshot: CoupledSnapshot) -> None:
        """Validate and copy the entire compatible state before publishing any field."""
        parameters = self._validated_snapshot_parameters(snapshot)
        wake = snapshot.wake_state
        if type(wake) is not CoupledWakeState:
            raise ValueError("CPC snapshots require CoupledWakeState")
        policy = CoupledCircadianConfig(
            wake.chemical_decay,
            wake.chemical_buildup_rate,
            wake.plasticity_sensitivity,
            wake.min_plasticity,
        )
        if policy != self.config:
            raise ValueError("snapshot chemistry policy differs from this model")
        if type(wake.wake_updates) is not int or wake.wake_updates < 0:
            raise ValueError("wake_updates must be a native nonnegative integer")
        chemicals = _copy_wake_vector(wake.chemical_state, self.hidden_dims[-1], "chemical_state")
        gates = _copy_wake_vector(wake.plasticity_state, self.hidden_dims[-1], "plasticity_state")
        if wake.wake_updates == 0 and (np.any(chemicals != 0) or np.any(gates != 1)):
            raise ValueError("zero-wake state requires zero chemistry and unit gates")
        if not np.array_equal(gates, _plasticity_gates(chemicals, policy)):
            raise ValueError("snapshot plasticity gates disagree with its chemistry and policy")
        self._parameters, self._chemical_state, self._plasticity_state, self._wake_updates = (
            parameters,
            chemicals,
            gates,
            wake.wake_updates,
        )

    def _prospective_chemistry(self, latents: tuple[Array, ...]) -> tuple[Array, Array]:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            activity = np.mean(np.abs(latents[-1]), axis=0)
            chemicals = (
                self.config.chemical_decay * self._chemical_state
                + self.config.chemical_buildup_rate * activity
            )
        if not np.isfinite(chemicals).all() or np.any(chemicals < 0):
            raise FloatingPointError("nonfinite or negative candidate chemistry")
        gates = _plasticity_gates(chemicals, self.config)
        if (
            not np.isfinite(gates).all()
            or np.any(gates < self.config.min_plasticity)
            or np.any(gates > 1)
        ):
            raise FloatingPointError("invalid candidate plasticity gates")
        return chemicals, gates

    def train_epoch(
        self,
        X: Array,
        y: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
        *,
        inference_policy: CoupledInferencePolicy | None = None,
    ) -> CoupledTrainResult:
        rate, latents, evaluation, diagnostics = self._training_with_policy(
            X, y, learning_rate, inference_steps, inference_learning_rate, inference_policy
        )
        chemicals, gates = self._prospective_chemistry(latents)
        gradients = _gated_gradients(evaluation.parameter_gradients, gates)
        candidate = self._prepare_parameters(gradients, rate, X, y, latents)
        self._publish_wake_state(candidate, chemicals, gates, evaluation.parameter_gradients)
        return CoupledTrainResult(evaluation.energy, inference=diagnostics)

    def _publish_wake_state(
        self,
        candidate: CoupledParameters,
        chemicals: Array,
        gates: Array,
        raw_gradients: CoupledParameters,
    ) -> None:
        # No committed field changes before every chemistry/gate/parameter/energy
        # check succeeds. Single-owner callers observe one completed wake state.
        self._parameters, self._chemical_state, self._plasticity_state, self._wake_updates = (
            candidate,
            chemicals,
            gates,
            self._wake_updates + 1,
        )
