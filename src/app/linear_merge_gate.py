"""Assess tolerant linear merge damage on owned training/validation probes.

Why observe the full objective: the core's branch projection tolerance is not a
prediction or energy bound; discarded components can be amplified downstream.
MergeDamageLimits and MergeProbeDiagnostic are exact aliases of the pruning
records, sharing admission/schema and inherited errors that may mention pruning.
Trusted pruning helpers provide the same role, ID, storage and workspace checks.
No final-test scoring, training, threshold selection, transform math changes,
runtime publication, authority or global damage guarantee is provided here.
"""

from dataclasses import dataclass, replace

from src.app.linear_pruning_gate import (
    PruneDamageLimits as MergeDamageLimits,
)
from src.app.linear_pruning_gate import (
    PruneProbeDiagnostic as MergeProbeDiagnostic,
)
from src.app.linear_pruning_gate import State, _copy_state, _diagnostic, _prepare, _score
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.linear_pc_scaling import LinearScalingState
from src.core.linear_width_structure import LinearWidthLineage, merge_linear_width


@dataclass(frozen=True)
class MergeDecision:
    """Owned replacement or retained source, with all completed probe observations."""

    status: str
    reason: str | None
    state: State
    lineage: LinearWidthLineage | None
    diagnostics: tuple[MergeProbeDiagnostic, ...]
    planned_probe_count: int


def _copy_lineage(lineage: LinearWidthLineage) -> LinearWidthLineage:
    if type(lineage) is not LinearWidthLineage:
        raise ValueError("merge requires an exact LinearWidthLineage")
    return LinearWidthLineage(
        lineage.format_version, lineage.family, lineage.source_config, lineage.split_index
    )


def _prepare_merge(
    initial: State,
    lineage: LinearWidthLineage,
    probes: tuple[ScalingBatch, ...],
    limits: MergeDamageLimits,
    budget: ToyBudgetSession,
) -> tuple[State, LinearWidthLineage, tuple[ScalingBatch, ...], MergeDamageLimits]:
    owned_lineage = _copy_lineage(lineage)
    source, owned_probes, owned_limits = _prepare(
        initial, owned_lineage.split_index, probes, limits, budget
    )
    family = "mlp" if type(source) is LinearScalingState else "residual"
    widened = replace(owned_lineage.source_config, width=owned_lineage.source_config.width + 1)
    if (
        family != owned_lineage.family
        or type(source.config) is not type(widened)
        or source.config != widened
    ):
        raise ValueError(
            "merge source differs from the declared full widened lineage configuration"
        )
    return source, owned_lineage, owned_probes, owned_limits


def _candidate(source: State, lineage: LinearWidthLineage) -> State:
    # Kernel callbacks cannot reach the retained source or declared lineage.
    merged = merge_linear_width(_copy_state(source), _copy_lineage(lineage))
    candidate = _copy_state(merged)
    if type(candidate) is not type(source) or candidate.config != lineage.source_config:
        raise ValueError("native merge returned an incompatible full candidate configuration")
    return candidate


def _exceeds_limits(diagnostics: list[MergeProbeDiagnostic], limits: MergeDamageLimits) -> bool:
    return any(
        item.max_prediction_change > limits.max_prediction_change
        or item.half_mse_increase > limits.max_half_mse_increase
        or item.energy_increase > limits.max_energy_increase
        for item in diagnostics
    )


def gate_linear_merge(
    initial: State,
    lineage: LinearWidthLineage,
    probes: tuple[ScalingBatch, ...],
    *,
    limits: MergeDamageLimits,
    budget: ToyBudgetSession,
) -> MergeDecision:
    """Observe every probe, then accept only when all inclusive limits pass.

    Invalid public preflight raises before budget callbacks. Candidate, numerical
    and resource/callback faults return failed with the detached full source and
    known completed diagnostic prefix. Rejection also retains the full source;
    only acceptance returns a candidate and owned accepted lineage. Observed
    limits apply only to these declared probes, not future learning or serving.
    """
    source, lineage, probes, limits = _prepare_merge(initial, lineage, probes, limits, budget)
    diagnostics: list[MergeProbeDiagnostic] = []
    try:
        budget.before_final()
        candidate = _candidate(source, lineage)
        for probe in probes:
            budget.before_final()
            before = _score(source, probe)
            budget.before_final()
            after = _score(candidate, probe)
            diagnostics.append(_diagnostic(probe, before, after))
            budget.before_final()
        budget.before_final()
        if _exceeds_limits(diagnostics, limits):
            return MergeDecision(
                "rejected",
                "observed probe damage exceeded declared limits",
                source,
                None,
                tuple(diagnostics),
                len(probes),
            )
        return MergeDecision(
            "accepted", None, candidate, _copy_lineage(lineage), tuple(diagnostics), len(probes)
        )
    except Exception as error:
        return MergeDecision(
            "failed",
            f"{type(error).__name__}: {error}",
            source,
            None,
            tuple(diagnostics),
            len(probes),
        )
