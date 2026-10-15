"""Owned raw scaling state with bounded uniform-width pruning metadata.

Full raw/native/wake validation remains the existing scaling state's boundary.
Lineage joins complete configurations, not widths alone, and grants no runtime,
data, promotion, or provenance authority. Earlier valid states and empty histories
are accepted for rollback; no external append-only history is asserted here.
No training, inference, energy evaluation, optimizer, or disk codec is provided.
"""

from dataclasses import dataclass

from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    _copy_config,
    copy_coupled_scaling_state,
)
from src.core.coupled_scaling_pruning import CoupledScalingPruneLineage

MAX_PRUNE_LINEAGE = 64


def _copy_history(
    history: tuple[CoupledScalingPruneLineage, ...], current: CoupledScalingConfig
) -> tuple[CoupledScalingPruneLineage, ...]:
    if type(history) is not tuple or len(history) > MAX_PRUNE_LINEAGE:
        raise ValueError("prune_lineage must be an exact tuple with at most 64 records")
    copied = []
    previous: CoupledScalingConfig | None = None
    for record in history:
        if type(record) is not CoupledScalingPruneLineage:
            raise ValueError(
                "raw topology lineage requires exact CoupledScalingPruneLineage records"
            )
        item = CoupledScalingPruneLineage(
            record.format_version, record.source_config, record.removed_units
        )
        if previous is not None and item.source_config != previous:
            raise ValueError(
                "raw topology lineage source differs from the previous full target config"
            )
        previous = item.target_config
        copied.append(item)
    if previous is not None and previous != current:
        raise ValueError("raw topology lineage final target differs from the current full config")
    return tuple(copied)


@dataclass(frozen=True)
class CoupledScalingTopologyState:
    """Detached raw state and copied, bounded full-configuration coordinate chain."""

    format_version: int
    raw_state: CoupledScalingState
    prune_lineage: tuple[CoupledScalingPruneLineage, ...] = ()

    def __post_init__(self) -> None:
        raw_state, history = _validated_parts(self)
        object.__setattr__(self, "raw_state", raw_state)
        object.__setattr__(self, "prune_lineage", history)


def _validated_parts(
    state: CoupledScalingTopologyState,
) -> tuple[CoupledScalingState, tuple[CoupledScalingPruneLineage, ...]]:
    if (
        type(state) is not CoupledScalingTopologyState
        or type(state.format_version) is not int
        or state.format_version != 1
    ):
        raise ValueError("unsupported raw coupled topology state format")
    # Bound history before any configuration/state copy or native allocation.
    if type(state.prune_lineage) is not tuple or len(state.prune_lineage) > MAX_PRUNE_LINEAGE:
        raise ValueError("prune_lineage must be an exact tuple with at most 64 records")
    if type(state.raw_state) is not CoupledScalingState:
        raise ValueError("raw topology requires an exact CoupledScalingState")
    config = _copy_config(state.raw_state.config)
    history = _copy_history(state.prune_lineage, config)
    return copy_coupled_scaling_state(state.raw_state), history


def copy_coupled_scaling_topology(
    state: CoupledScalingTopologyState,
) -> CoupledScalingTopologyState:
    """Revalidate mutable contents and return independently owned complete state."""
    if type(state) is not CoupledScalingTopologyState:
        raise ValueError("state must be an exact CoupledScalingTopologyState")
    return CoupledScalingTopologyState(state.format_version, state.raw_state, state.prune_lineage)
