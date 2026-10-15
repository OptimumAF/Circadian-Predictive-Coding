"""Remove a declared common hidden coordinate from owned linear PC state.

The selector acts on raw branches with stable old/new forward-scale rebinding.
Residual fixed identities remain implicit in the smaller configuration. General
predictions and equilibrated energies can change; this utility makes no damage
decision. It performs no equilibrium evaluation, SGD, optimizer/trace update,
authority transfer, RNG, or dense selector/identity allocation.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.core.linear_pc_reference import Array
from src.core.linear_width_structure import (
    Config,
    State,
    _copy_config,
    _inspect_state,
    _make_state,
    _product,
    _validate_effective,
)


def _prune_index(index: int, width: int) -> None:
    if width < 2:
        raise ValueError("linear width pruning requires a source width of at least 2")
    if type(index) is not int or not 0 <= index < width:
        raise ValueError("prune_index must be a native integer within the source width")


@dataclass(frozen=True)
class LinearPruneLineage:
    """Owned coordinate metadata only; no provenance or runtime authority."""

    format_version: int
    family: str
    source_config: Config
    prune_index: int

    def __post_init__(self) -> None:
        config = _validate_lineage(self)
        object.__setattr__(self, "source_config", config)


def _validate_lineage(lineage: LinearPruneLineage) -> Config:
    if (
        type(lineage) is not LinearPruneLineage
        or type(lineage.format_version) is not int
        or lineage.format_version != 1
    ):
        raise ValueError("unsupported linear pruning lineage format")
    if type(lineage.family) is not str or lineage.family not in ("mlp", "residual"):
        raise ValueError("pruning lineage family must be mlp or residual")
    source = _copy_config(lineage.source_config, lineage.family)
    _prune_index(lineage.prune_index, source.width)
    replace(source, width=source.width - 1)
    return source


@dataclass(frozen=True)
class LinearPruneChange:
    state: State
    lineage: LinearPruneLineage


def _pruned_weights(
    weights: tuple[Array, ...], source: Config, target: Config, index: int
) -> tuple[Array, ...]:
    coordinates = np.concatenate((np.arange(index), np.arange(index + 1, source.width)))
    selected = []
    for layer, (raw, old_scale, new_scale) in enumerate(
        zip(weights, source.forward_scales, target.forward_scales)
    ):
        value = raw
        if layer > 0:
            value = np.take(value, coordinates, axis=0)
        if layer < len(weights) - 1:
            value = np.take(value, coordinates, axis=1)
        selected.append(_product(value, old_scale, 1 / new_scale))
    return tuple(selected)


def prune_linear_width(state: State, index: int) -> LinearPruneChange:
    """Return an owned N-1 state, removing the same coordinate at every layer.

    Borrowed source arrays must be quiescent. Validation/refusal never changes
    them. General function or energy preservation is not promised; an external
    application must judge acceptable damage. Future width-dependent SGD rates
    follow the new configuration, without trajectory-preservation guarantees.
    """
    family, source = _inspect_state(state)
    _prune_index(index, source.width)
    # Validate the target configuration before allocating selector/output arrays.
    target = replace(source, width=source.width - 1)
    _validate_effective(state)
    lineage = LinearPruneLineage(1, family, source, index)
    weights = _pruned_weights(state.raw_weights, source, target, index)
    candidate = _make_state(family, target, weights)
    return LinearPruneChange(candidate, lineage)
