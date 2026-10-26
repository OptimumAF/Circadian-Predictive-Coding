"""Read detached native histories from legacy models and explicit coupled snapshots.

The adapter reads stored fields and detached tracked/renewable snapshots only.
It does not call model methods, scan full weights, mutate state, infer missing
histories, or certify complete model validity/provenance. Arrays must be quiescent.
"""

from dataclasses import dataclass

import numpy as np

from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.learnability_diagnostics import UnitHistoryDiagnostics, summarize_unit_history
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_unit_history import UnitHistory, copy_unit_history
from src.core.coupled_predictive_coding import CoupledWakeState
from src.core.tracked_coupled_predictive_coding import TrackedCoupledSnapshot
from src.core.coupled_unit_renewal import (
    RenewableUnitHistory,
    RenewalPolicy,
    copy_renewal_history,
    copy_renewal_policy,
)
from src.core.renewable_coupled_predictive_coding import RenewableCoupledSnapshot


@dataclass(frozen=True)
class LegacyUnitDiagnostics:
    family: str
    layer: str
    history: UnitHistoryDiagnostics
    importance_definition: str
    age_definition: str


_IMPORTANCE_DEFINITION = (
    "Stored EMA of mean absolute outgoing pre-plasticity gradient times reward scale. "
    "Historical replay may update this EMA; wake-only replay preserves it. "
    "Split children inherit parent importance; prune decay attenuates pending-prune "
    "importance and removal retains surviving entries."
)
_AGE_DEFINITION = (
    "Stored native float count of successful epoch-state updates since unit birth/reset. "
    "Replay without epoch-state updates does not increment it. Split children start "
    "at zero while parent ages remain; removal retains surviving entries. "
    "This is neither wall time nor the current global wake count."
)


def read_legacy_unit_diagnostics(model: CircadianPredictiveCodingNetwork) -> LegacyUnitDiagnostics:
    """Summarize exact legacy CPC native fields at the current adaptive width.

    Why this: hidden_dims records initial geometry and can be stale after actual
    splits/prunes. The current incoming weight matrix supplies the observed width
    without invoking the hidden_dim getter or any operational model callback.
    """
    if type(model) is not CircadianPredictiveCodingNetwork:
        raise ValueError(
            "unit diagnostics require an exact legacy CircadianPredictiveCodingNetwork"
        )
    fields = vars(model)
    required = (
        "weight_input_hidden",
        "_neuron_ids",
        "_parent_ids",
        "_importance_ema",
        "_neuron_age",
        "_next_neuron_id",
    )
    if any(name not in fields for name in required):
        raise ValueError("legacy CPC is missing native unit history fields")
    incoming = fields["weight_input_hidden"]
    if (
        type(incoming) is not np.ndarray
        or incoming.dtype != np.float64
        or incoming.ndim != 2
        or incoming.shape[0] < 1
        or not 1 <= incoming.shape[1] <= 4096
    ):
        raise ValueError("current adaptive weights must be a float64 matrix with width 1..4096")
    vectors = tuple(fields[name] for name in required[1:5])
    if any(
        type(value) is not np.ndarray or value.shape != (incoming.shape[1],) for value in vectors
    ):
        raise ValueError("native unit vectors must match the current adaptive hidden width")
    history = summarize_unit_history(*vectors, next_neuron_id=fields["_next_neuron_id"])
    return LegacyUnitDiagnostics(
        "legacy_cpc", "adaptive_hidden", history, _IMPORTANCE_DEFINITION, _AGE_DEFINITION
    )


@dataclass(frozen=True)
class CoupledUnitDiagnostics:
    """Declared last-hidden history; not complete native snapshot validation."""

    family: str
    layer: str
    history: UnitHistoryDiagnostics
    successful_updates: int
    array_bytes: int
    importance_definition: str
    age_definition: str


def _tracked_history(snapshot: TrackedCoupledSnapshot) -> UnitHistory:
    """Validate only declared history geometry and its native-update clock."""
    if (
        type(snapshot) is not TrackedCoupledSnapshot
        or type(snapshot.format_version) is not int
        or snapshot.format_version != 2
    ):
        raise ValueError("coupled unit diagnostics require an exact tracked snapshot format 2")
    if type(snapshot.parameters) is not CoupledParameters:
        raise ValueError("tracked history requires exact coupled parameters")
    outgoing = snapshot.parameters.output_weight
    if (
        type(outgoing) is not np.ndarray
        or outgoing.dtype != np.float64
        or outgoing.ndim != 2
        or outgoing.shape[1] != 1
    ):
        raise ValueError("tracked history requires a float64 outgoing weight matrix")
    supplied = snapshot.unit_history
    if type(supplied) is not UnitHistory:
        raise ValueError("tracked snapshot requires native UnitHistory")
    history = copy_unit_history(supplied, outgoing.shape[0], supplied.ema_decay)
    if snapshot.wake_state is not None:
        if (
            type(snapshot.wake_state) is not CoupledWakeState
            or type(snapshot.wake_state.wake_updates) is not int
            or snapshot.wake_state.wake_updates != history.updates
        ):
            raise ValueError("tracked CPC history clock differs from native wake updates")
    return history


def read_coupled_unit_diagnostics(snapshot: TrackedCoupledSnapshot) -> CoupledUnitDiagnostics:
    """Read copied native tracked history without training or inspecting final data.

    Why this: chemistry is not measured gradient importance, and a global clock
    does not provide a per-unit history. Old snapshots remain unsupported here.
    This validates the history and its declared width, not all native parameters,
    chemistry, data provenance, or authority. Caller arrays must be quiescent.
    """
    history = _tracked_history(snapshot)
    units = summarize_unit_history(
        history.unit_ids,
        np.full(len(history.unit_ids), -1, dtype=np.int64),
        history.importance_ema,
        history.update_age,
        next_neuron_id=len(history.unit_ids),
    )
    return CoupledUnitDiagnostics(
        "tracked_coupled",
        "last_hidden",
        units,
        history.updates,
        history.unit_ids.nbytes + history.importance_ema.nbytes + history.update_age.nbytes,
        "EMA of absolute batch-aggregate outgoing coupled-energy gradient before CPC plasticity gates; one sample per successful native update, no reward scaling or causal importance claim.",
        "Successful native update count since tracked construction, including replay; failed calls and prediction do not increment age. Fixed-width original unit IDs; no replacement or retroactive history.",
    )


def _renewable_history(snapshot: RenewableCoupledSnapshot) -> RenewableUnitHistory:
    if (
        type(snapshot) is not RenewableCoupledSnapshot
        or type(snapshot.format_version) is not int
        or snapshot.format_version != 3
    ):
        raise ValueError("renewable unit diagnostics require an exact renewable snapshot format 3")
    if type(snapshot.renewal_policy) is not RenewalPolicy:
        raise ValueError("renewable snapshot requires exact RenewalPolicy")
    copy_renewal_policy(snapshot.renewal_policy)
    if type(snapshot.parameters) is not CoupledParameters:
        raise ValueError("renewable history requires exact coupled parameters")
    parameters = snapshot.parameters
    outgoing = parameters.output_weight
    if (
        type(outgoing) is not np.ndarray
        or outgoing.dtype != np.float64
        or outgoing.ndim != 2
        or not 1 <= outgoing.shape[0] <= 4096
        or outgoing.shape[1] != 1
        or type(parameters.hidden_weights) is not tuple
        or not parameters.hidden_weights
        or type(parameters.hidden_biases) is not tuple
        or len(parameters.hidden_biases) != len(parameters.hidden_weights)
    ):
        raise ValueError("renewable history requires bounded last-hidden scalar-output geometry")
    incoming, bias = parameters.hidden_weights[-1], parameters.hidden_biases[-1]
    width = outgoing.shape[0]
    if (
        type(incoming) is not np.ndarray
        or incoming.dtype != np.float64
        or incoming.ndim != 2
        or incoming.shape[0] < 1
        or incoming.shape[1] != width
        or type(bias) is not np.ndarray
        or bias.dtype != np.float64
        or bias.shape != (1, width)
    ):
        raise ValueError("renewable history geometry must align incoming, bias and outgoing widths")
    supplied = snapshot.unit_history
    if type(supplied) is not RenewableUnitHistory:
        raise ValueError("renewable snapshot requires native RenewableUnitHistory")
    history = copy_renewal_history(supplied, width, supplied.ema_decay)
    wake = snapshot.wake_state
    if wake is not None and (
        type(wake) is not CoupledWakeState
        or type(wake.wake_updates) is not int
        or wake.wake_updates != history.updates
    ):
        raise ValueError("renewable CPC history clock differs from native wake updates")
    return history


def read_renewable_coupled_unit_diagnostics(
    snapshot: RenewableCoupledSnapshot,
) -> CoupledUnitDiagnostics:
    """Read native renewable IDs and ages without inventing ancestry or importance.

    The native next-ID bound permits IDs above current width; unknown parents
    remain None. Validation covers declared last-hidden geometry, renewal policy,
    history and clock, not complete native parameters, chemistry or provenance.
    No model method is called, and borrowed snapshot arrays must be quiescent.
    """
    history = _renewable_history(snapshot)
    units = summarize_unit_history(
        history.unit_ids,
        np.full(len(history.unit_ids), -1, dtype=np.int64),
        history.importance_ema,
        history.update_age,
        next_neuron_id=history.next_unit_id,
    )
    return CoupledUnitDiagnostics(
        "renewable_coupled",
        "last_hidden",
        units,
        history.updates,
        history.unit_ids.nbytes + history.importance_ema.nbytes + history.update_age.nbytes,
        "EMA of absolute batch-aggregate outgoing coupled-energy gradient before CPC plasticity gates; reset to zero at unit replacement; no reward scaling or causal importance claim.",
        "Successful native updates since each unit's birth or last replacement, including replay; replacement resets selected ages but not the global native-update count. Current stable IDs are preserved; parents are unknown, not inferred.",
    )
