"""Analyze already verified confirmation outcomes in the complete frozen scope.

Inputs are declared final-role identities and either three accuracies or an
explicit failure for every scheduled cell. Outputs keep all raw cells and
within-seed metric/pair vectors. No data/model/IO, scoring, source proof,
experiment selection, pooling or cost invention belongs to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.app.continual_confirmation_analysis_contract import (
    ACCURACY_ENDPOINTS,
    AnalysisFamily,
    ConfirmationAnalysisContract,
    analysis_contract_digest,
    validate_analysis_contract,
)
from src.core.continual_metrics import TwoTaskAccuracy
from src.core.seed_statistics import IntervalSpec, SeedObservation, SeedSummary, summarize_seeds


@dataclass(frozen=True)
class FinalRoleIdentity:
    a_sha256: str
    b_sha256: str
    a_count: int = 40
    b_count: int = 40
    score_role: str = "independent_confirmation_final_test"


@dataclass(frozen=True)
class OutcomeCell:
    family: str
    seed: int
    arm: str
    roles: FinalRoleIdentity | None
    accuracy: TwoTaskAccuracy | None
    failure: str | None = None


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    primary_endpoint: bool
    preferred_direction: str
    units: str
    summary: SeedSummary


@dataclass(frozen=True)
class ArmAnalysis:
    name: str
    metrics: tuple[MetricSummary, ...]


@dataclass(frozen=True)
class ContrastAnalysis:
    left: str
    right: str
    metrics: tuple[MetricSummary, ...]


@dataclass(frozen=True)
class FamilyAnalysis:
    name: str
    arms: tuple[ArmAnalysis, ...]
    contrasts: tuple[ContrastAnalysis, ...]


@dataclass(frozen=True)
class ConfirmationAnalysis:
    contract_sha256: str
    cost_facts_reference_sha256: str
    evaluation_role: str
    input_validation_scope: str
    cells: tuple[OutcomeCell, ...]
    families: tuple[FamilyAnalysis, ...]
    distinct_source_seeds: int
    primary_statement_count: int
    successful_cells: int
    failed_cells: int
    complete: bool


CellKey = tuple[str, int, str]


def _verify_roles(
    roles: FinalRoleIdentity, family: AnalysisFamily, contract: ConfirmationAnalysisContract
) -> None:
    if (
        type(roles) is not FinalRoleIdentity
        or type(roles.a_count) is not int
        or type(roles.b_count) is not int
        or (roles.a_count, roles.b_count) != family.final_counts
        or roles.score_role != contract.score_role
    ):
        raise ValueError("confirmation analysis final role/count differs")
    for digest in (roles.a_sha256, roles.b_sha256):
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("confirmation analysis final role fingerprint is malformed")


def _verify_outcome(
    row: OutcomeCell, family: AnalysisFamily, contract: ConfirmationAnalysisContract
) -> None:
    if row.roles is not None:
        _verify_roles(row.roles, family, contract)
    if row.accuracy is None:
        if type(row.failure) is not str or not row.failure.strip():
            raise ValueError("missing outcome requires an explicit nonempty failure")
        return
    if type(row.accuracy) is not TwoTaskAccuracy or row.failure is not None:
        raise ValueError(
            "successful outcome cannot also declare failure or unknown accuracy fields"
        )
    if row.roles is None:
        raise ValueError("successful outcome requires the declared final roles")
    # Why this: frozen containers can still be tampered with; validate raw fields again.
    TwoTaskAccuracy(row.accuracy.a_after_a, row.accuracy.a_after_b, row.accuracy.b_after_b)


def _verified_cells(
    cells: tuple[OutcomeCell, ...], contract: ConfirmationAnalysisContract
) -> dict[CellKey, OutcomeCell]:
    families = {family.name: family for family in contract.families}
    expected = {
        (family.name, seed, arm)
        for family in contract.families
        for seed in family.seeds
        for arm in family.arms
    }
    if type(cells) is not tuple or len(cells) != len(expected):
        raise ValueError("confirmation analysis requires the complete scheduled cell scope")
    result: dict[CellKey, OutcomeCell] = {}
    shared_roles: dict[tuple[str, int], FinalRoleIdentity] = {}
    for row in cells:
        if (
            type(row) is not OutcomeCell
            or type(row.family) is not str
            or type(row.arm) is not str
            or type(row.seed) is not int
        ):
            raise ValueError(
                "confirmation analysis cell identity requires exact family/seed/arm types"
            )
        key = (row.family, row.seed, row.arm)
        if key not in expected or key in result:
            raise ValueError("confirmation analysis cell scope is unknown or duplicate")
        _verify_outcome(row, families[row.family], contract)
        if row.roles is not None:
            role_key = (row.family, row.seed)
            if role_key in shared_roles and shared_roles[role_key] != row.roles:
                raise ValueError("confirmation analysis paired final role identities differ")
            shared_roles[role_key] = row.roles
        result[key] = row
    if result.keys() != expected:
        raise ValueError("confirmation analysis cell scope is incomplete")
    return result


def _value(row: OutcomeCell, metric: str) -> SeedObservation:
    if row.accuracy is None:
        return SeedObservation(row.seed, None, row.failure)
    value = getattr(row.accuracy, metric)
    return SeedObservation(row.seed, value, "zero_a_after_a" if value is None else None)


def _difference(left: OutcomeCell, right: OutcomeCell, metric: str) -> SeedObservation:
    if left.accuracy is None or right.accuracy is None:
        reasons = [
            f"{side}_failed: {row.failure}"
            for side, row in (("left", left), ("right", right))
            if row.accuracy is None
        ]
        return SeedObservation(left.seed, None, "; ".join(reasons))
    return SeedObservation(
        left.seed, getattr(left.accuracy, metric) - getattr(right.accuracy, metric)
    )


def _metric_summary(
    metric: str,
    observations: tuple[SeedObservation, ...],
    family: AnalysisFamily,
    contract: ConfirmationAnalysisContract,
    *,
    paired: bool,
) -> MetricSummary:
    primary = metric in contract.primary_metrics
    intervals = None
    if metric != "retention_ratio_a":
        simultaneous = paired and primary
        intervals = IntervalSpec(
            contract.sample_count,
            contract.marginal_confidence_level,
            contract.marginal_critical,
            contract.primary_statement_count if simultaneous else None,
            contract.simultaneous_critical if simultaneous else None,
            contract.zero_deviation_tolerance,
        )
    return MetricSummary(
        metric,
        primary,
        "lower" if metric == "signed_forgetting_a" else "higher",
        "ratio" if metric == "retention_ratio_a" else "accuracy_fraction",
        summarize_seeds(observations, family.seeds, intervals),
    )


def _arm_analysis(
    arm: str,
    family: AnalysisFamily,
    rows: dict[CellKey, OutcomeCell],
    contract: ConfirmationAnalysisContract,
) -> ArmAnalysis:
    metrics = ACCURACY_ENDPOINTS + contract.primary_metrics
    return ArmAnalysis(
        arm,
        tuple(
            _metric_summary(
                metric,
                tuple(_value(rows[(family.name, seed, arm)], metric) for seed in family.seeds),
                family,
                contract,
                paired=False,
            )
            for metric in metrics + ("retention_ratio_a",)
        ),
    )


def _contrast_analysis(
    left: str,
    right: str,
    family: AnalysisFamily,
    rows: dict[CellKey, OutcomeCell],
    contract: ConfirmationAnalysisContract,
) -> ContrastAnalysis:
    return ContrastAnalysis(
        left,
        right,
        tuple(
            _metric_summary(
                metric,
                tuple(
                    _difference(
                        rows[(family.name, seed, left)], rows[(family.name, seed, right)], metric
                    )
                    for seed in family.seeds
                ),
                family,
                contract,
                paired=True,
            )
            for metric in ACCURACY_ENDPOINTS + contract.primary_metrics
        ),
    )


def _family_analysis(
    family: AnalysisFamily, rows: dict[CellKey, OutcomeCell], contract: ConfirmationAnalysisContract
) -> FamilyAnalysis:
    return FamilyAnalysis(
        family.name,
        tuple(_arm_analysis(arm, family, rows, contract) for arm in family.arms),
        tuple(
            _contrast_analysis(left, right, family, rows, contract)
            for left, right in family.contrasts
        ),
    )


def analyze_confirmation(
    cells: tuple[OutcomeCell, ...], contract: ConfirmationAnalysisContract
) -> ConfirmationAnalysis:
    """Validate every scheduled identity/outcome before producing any partial summary."""
    validate_analysis_contract(contract)
    rows = _verified_cells(cells, contract)
    canonical = tuple(
        rows[(family.name, seed, arm)]
        for family in contract.families
        for seed in family.seeds
        for arm in family.arms
    )
    failed = sum(row.accuracy is None for row in canonical)
    return ConfirmationAnalysis(
        analysis_contract_digest(contract),
        contract.train_result_sha256,
        contract.score_role,
        "declared_outcomes_and_pairing_only_not_source_proof",
        canonical,
        tuple(_family_analysis(family, rows, contract) for family in contract.families),
        len({seed for family in contract.families for seed in family.seeds}),
        contract.primary_statement_count,
        len(canonical) - failed,
        failed,
        failed == 0,
    )
