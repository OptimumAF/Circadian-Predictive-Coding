"""Owned raw coupled state with a bounded, rooted structural configuration chain.

Why a separate envelope: split/merge changes precisions as well as width, while
the existing topology adapter deliberately pins precisions. Tagged transitions
describe full configuration reachability from an explicit origin. They do not
prove that transforms ran, recover historical parameters, or grant authority.
Earlier valid histories support rollback; merge need not reverse the last tag.
No training, inference, objective evaluation, transform execution or IO lives here.
"""

from dataclasses import dataclass

from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    _copy_config,
    copy_coupled_scaling_state,
)
from src.core.coupled_scaling_pruning import CoupledScalingPruneLineage
from src.core.coupled_scaling_structure import CoupledScalingSplitLineage

MAX_STRUCTURAL_TRANSITIONS = 64


def _validate_tag(
    kind: str, lineage: CoupledScalingPruneLineage | CoupledScalingSplitLineage
) -> None:
    if type(kind) is not str or kind not in ("prune", "split", "merge"):
        raise ValueError("structural transition kind must be prune, split or merge")
    expected = CoupledScalingPruneLineage if kind == "prune" else CoupledScalingSplitLineage
    if type(lineage) is not expected:
        raise ValueError("structural transition tag requires its exact native lineage type")


def _copy_lineage(
    kind: str, lineage: CoupledScalingPruneLineage | CoupledScalingSplitLineage
) -> CoupledScalingPruneLineage | CoupledScalingSplitLineage:
    _validate_tag(kind, lineage)
    if type(lineage) is CoupledScalingPruneLineage:
        return CoupledScalingPruneLineage(
            lineage.format_version, lineage.source_config, lineage.removed_units
        )
    return CoupledScalingSplitLineage(lineage.format_version, lineage.source_config)


@dataclass(frozen=True)
class CoupledStructuralTransition:
    """Owned operation metadata; merge reverses its declared split configuration."""

    kind: str
    lineage: CoupledScalingPruneLineage | CoupledScalingSplitLineage

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage", _copy_lineage(self.kind, self.lineage))

    @property
    def source_config(self) -> CoupledScalingConfig:
        lineage = _copy_lineage(self.kind, self.lineage)
        return (
            lineage.target_config if self.kind == "merge" else _copy_config(lineage.source_config)
        )

    @property
    def target_config(self) -> CoupledScalingConfig:
        lineage = _copy_lineage(self.kind, self.lineage)
        return (
            _copy_config(lineage.source_config) if self.kind == "merge" else lineage.target_config
        )


def _admit_transitions(transitions: tuple[CoupledStructuralTransition, ...]) -> None:
    if type(transitions) is not tuple or len(transitions) > MAX_STRUCTURAL_TRANSITIONS:
        raise ValueError("transitions must be an exact tuple with at most 64 records")
    for transition in transitions:
        if type(transition) is not CoupledStructuralTransition:
            raise ValueError("transitions require exact CoupledStructuralTransition records")
        _validate_tag(transition.kind, transition.lineage)


def _copy_chain(
    transitions: tuple[CoupledStructuralTransition, ...],
    origin: CoupledScalingConfig,
    current: CoupledScalingConfig,
) -> tuple[CoupledStructuralTransition, ...]:
    copied = []
    previous = origin
    for transition in transitions:
        item = CoupledStructuralTransition(transition.kind, transition.lineage)
        if item.source_config != previous:
            raise ValueError(
                "structural transition source differs from origin or previous full target config"
            )
        previous = item.target_config
        copied.append(item)
    if previous != current:
        raise ValueError("structural chain final target differs from the current full config")
    return tuple(copied)


@dataclass(frozen=True)
class CoupledStructuralState:
    """Complete owned raw state, original configuration and joined operation tags."""

    format_version: int
    origin_config: CoupledScalingConfig
    raw_state: CoupledScalingState
    transitions: tuple[CoupledStructuralTransition, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self) is not CoupledStructuralState
            or type(self.format_version) is not int
            or self.format_version != 1
        ):
            raise ValueError("unsupported coupled structural state format")
        # Admit all small schema/cap fields before copying raw tensors or creating
        # native validation models. Mutable snapshot contents are rechecked below.
        _admit_transitions(self.transitions)
        if type(self.raw_state) is not CoupledScalingState:
            raise ValueError("structural state requires an exact CoupledScalingState")
        origin = _copy_config(self.origin_config)
        current = _copy_config(self.raw_state.config)
        transitions = _copy_chain(self.transitions, origin, current)
        raw_state = copy_coupled_scaling_state(self.raw_state)
        object.__setattr__(self, "origin_config", origin)
        object.__setattr__(self, "raw_state", raw_state)
        object.__setattr__(self, "transitions", transitions)


def copy_coupled_structural_state(state: CoupledStructuralState) -> CoupledStructuralState:
    """Revalidate all mutable contents and return a detached complete envelope."""
    if type(state) is not CoupledStructuralState:
        raise ValueError("state must be an exact CoupledStructuralState")
    return CoupledStructuralState(
        state.format_version, state.origin_config, state.raw_state, state.transitions
    )
