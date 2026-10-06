"""Scaffolding for dynamic neuron creation/pruning policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LayerTraffic:
    """Traffic summary for a single layer."""

    layer_name: str
    mean_abs_activation: Array


@dataclass(frozen=True)
class NeuronChangeProposal:
    """Planned neuron changes for a layer."""

    layer_name: str
    add_count: int = 0
    remove_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class NeuronLineageSnapshot:
    """Active adaptive-neuron IDs and stable birth-parent references."""

    neuron_ids: tuple[int, ...]
    parent_ids: tuple[int | None, ...]
    next_neuron_id: int


@dataclass(frozen=True)
class PruneOutcome:
    """Stable IDs requested, marked for delayed removal, and actually removed."""

    proposed_neuron_ids: tuple[int, ...] = ()
    scheduled_neuron_ids: tuple[int, ...] = ()
    removed_neuron_ids: tuple[int, ...] = ()


class NeuronAdaptationPolicy(Protocol):
    """Policy contract for future add/remove neuron decisions."""

    def propose(self, traffic_by_layer: list[LayerTraffic]) -> list[NeuronChangeProposal]:
        """Return explicit structural changes based on traffic."""


class NoOpNeuronAdaptationPolicy:
    """Default policy that keeps topology fixed.

    Why this: baseline experiments stay directly comparable to standard backprop.
    """

    def propose(self, traffic_by_layer: list[LayerTraffic]) -> list[NeuronChangeProposal]:
        return []

