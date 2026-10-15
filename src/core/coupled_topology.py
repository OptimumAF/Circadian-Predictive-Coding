"""Owned bounded geometry-aware state for native coupled PC/CPC reconstruction.

The full native snapshot remains the tensor/wake-state boundary. Explicit
geometry enables reconstruction after structural changes; lineage is copied
coordinate metadata, never model provenance or runtime/serving authority.
No optimizer, native training, inference, disk codec, or data-role logic lives here.
"""

from dataclasses import dataclass

from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.coupled_width_pruning import (
    CoupledPruneLineage,
    _geometry,
    _model_geometry,
    restore_coupled_model,
)

MAX_PRUNE_LINEAGE = 64


def _copy_lineage(
    lineage: tuple[CoupledPruneLineage, ...],
    family: str,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    precisions: tuple[float, ...],
) -> tuple[CoupledPruneLineage, ...]:
    if type(lineage) is not tuple or len(lineage) > MAX_PRUNE_LINEAGE:
        raise ValueError("prune_lineage must be an exact tuple containing at most 64 records")
    copied = []
    previous: tuple[int, ...] | None = None
    for record in lineage:
        if type(record) is not CoupledPruneLineage:
            raise ValueError("prune lineage records must be exact CoupledPruneLineage")
        item = CoupledPruneLineage(
            record.format_version,
            record.family,
            record.input_dim,
            record.source_hidden_dims,
            record.precisions,
            record.layer_index,
            record.unit_index,
        )
        if item.family != family or item.input_dim != input_dim or item.precisions != precisions:
            raise ValueError("prune lineage family/input/precision binding differs from topology")
        if previous is not None and item.source_hidden_dims != previous:
            raise ValueError("prune lineage source does not match the preceding target geometry")
        previous = item.target_hidden_dims
        copied.append(item)
    if previous is not None and previous != hidden_dims:
        raise ValueError("last prune lineage target differs from the current hidden geometry")
    return tuple(copied)


@dataclass(frozen=True)
class CoupledTopologyState:
    """Detached native model state with bounded explicit geometry and lineage."""

    format_version: int
    family: str
    input_dim: int
    hidden_dims: tuple[int, ...]
    snapshot: CoupledSnapshot
    prune_lineage: tuple[CoupledPruneLineage, ...] = ()

    def __post_init__(self) -> None:
        model, lineage = _validated_model(self)
        object.__setattr__(self, "input_dim", model.input_dim)
        object.__setattr__(self, "hidden_dims", tuple(width for width in model.hidden_dims))
        object.__setattr__(self, "snapshot", model.snapshot_state())
        object.__setattr__(self, "prune_lineage", lineage)


def _validated_model(
    state: CoupledTopologyState,
) -> tuple[CoupledPredictiveCodingNetwork, tuple[CoupledPruneLineage, ...]]:
    if (
        type(state) is not CoupledTopologyState
        or type(state.format_version) is not int
        or state.format_version != 1
    ):
        raise ValueError("unsupported coupled topology state format")
    if type(state.family) is not str or state.family not in ("pc", "cpc"):
        raise ValueError("topology family must be pc or cpc")
    if type(state.snapshot) is not CoupledSnapshot:
        raise ValueError("topology requires an exact complete CoupledSnapshot")
    input_dim, hidden_dims, precisions = _geometry(
        state.input_dim, state.hidden_dims, state.snapshot.precisions
    )
    lineage = _copy_lineage(state.prune_lineage, state.family, input_dim, hidden_dims, precisions)
    model = restore_coupled_model(state.family, input_dim, hidden_dims, precisions, state.snapshot)
    return model, lineage


def capture_coupled_topology(
    model: CoupledPredictiveCodingNetwork,
    *,
    prune_lineage: tuple[CoupledPruneLineage, ...] = (),
) -> CoupledTopologyState:
    """Capture validated complete state without sharing tensors or policy records."""
    family, input_dim, hidden_dims, precisions = _model_geometry(model)
    lineage = _copy_lineage(prune_lineage, family, input_dim, hidden_dims, precisions)
    return CoupledTopologyState(1, family, input_dim, hidden_dims, model.snapshot_state(), lineage)


def restore_coupled_topology(state: CoupledTopologyState) -> CoupledPredictiveCodingNetwork:
    """Revalidate mutable snapshot contents and reconstruct an independent model."""
    model, _ = _validated_model(state)
    return model
