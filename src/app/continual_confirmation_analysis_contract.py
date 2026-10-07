"""Frozen seed-analysis scope inspected without data, model, score or IO.

Inputs are the original confirmation configuration factories. Outputs retain
all families, cells, ordered contrasts, roles and predeclared interval rules.
Actual scored provenance, evaluation and artifact publication belong to P6.7c.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID


ANALYSIS_CONTRACT_ID = "continual_confirmation_seed_analysis_v1"
TRAIN_MANIFEST_SHA256 = "8d1ed66b33bbc1bf298cc60604c3741afa22bb4b7e0f6636efe166a52672951b"
TRAIN_RESULT_SHA256 = "3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547"
PRIMARY_METRICS = ("final_mean_task_accuracy", "signed_forgetting_a")
ACCURACY_ENDPOINTS = ("a_after_a", "a_after_b", "b_after_b")


@dataclass(frozen=True)
class AnalysisFamily:
    name: str
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    contrasts: tuple[tuple[str, str], ...]
    final_counts: tuple[int, int]


@dataclass(frozen=True)
class ConfirmationAnalysisContract:
    families: tuple[AnalysisFamily, ...]
    contract_id: str = ANALYSIS_CONTRACT_ID
    train_manifest_sha256: str = TRAIN_MANIFEST_SHA256
    train_result_sha256: str = TRAIN_RESULT_SHA256
    metric_contract_id: str = TWO_TASK_METRIC_CONTRACT_ID
    score_role: str = "independent_confirmation_final_test"
    primary_metrics: tuple[str, ...] = PRIMARY_METRICS
    secondary_metrics: tuple[str, ...] = ACCURACY_ENDPOINTS + ("retention_ratio_a",)
    sample_count: int = 10
    primary_statement_count: int = 116
    family_alpha: float = 0.05
    marginal_confidence_level: float = 0.95
    # Why fixed: exactly ten observations, independently checked before final values.
    marginal_critical: float = 2.2621571627982053
    simultaneous_critical: float = 5.403490569214909
    zero_deviation_tolerance: float = 1e-12
    interval_method: str = "student_t_df9_bonferroni_116_model_based"
    comparison_sign: str = "left_minus_right"
    missing_policy: str = "retain_nulls_no_full_planned_mean_or_interval"
    constant_policy: str = "retain_raw_sd_no_interval_at_declared_numeric_tolerance"
    retention_policy: str = "descriptive_only_null_zero_denominator"
    shared_seed_policy: str = "within_family_only_no_pooling_or_repeat_replications"
    interval_assumptions: tuple[str, ...] = (
        "independent_source_seed_replications",
        "approximately_normal_seed_outcomes_or_differences",
        "complete_ten_seed_vector_with_nonzero_observed_dispersion",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fixed_analysis_contract() -> ConfirmationAnalysisContract:
    """Keep the entire original scope; no outcome participates in this declaration."""
    manifest = fixed_confirmation_manifest()
    digest = sha256(_canonical_json(asdict(manifest)).encode("utf-8")).hexdigest()
    if digest != TRAIN_MANIFEST_SHA256:
        raise ValueError("confirmation analysis source manifest changed")
    families = tuple(
        AnalysisFamily(f.name, f.seeds, f.arms, f.contrasts, f.expected_final_counts)
        for f in manifest.families
    )
    if (
        any(len(f.seeds) != 10 for f in families)
        or sum(len(f.contrasts) for f in families) * len(PRIMARY_METRICS) != 116
    ):
        raise ValueError("confirmation analysis replication or statement count changed")
    return ConfirmationAnalysisContract(families)


def _closed_container_types(contract: ConfirmationAnalysisContract) -> bool:
    if (
        type(contract.families) is not tuple
        or type(contract.primary_metrics) is not tuple
        or type(contract.secondary_metrics) is not tuple
        or type(contract.interval_assumptions) is not tuple
    ):
        return False
    return all(
        type(family) is AnalysisFamily
        and type(family.seeds) is tuple
        and type(family.arms) is tuple
        and type(family.contrasts) is tuple
        and all(type(pair) is tuple for pair in family.contrasts)
        and type(family.final_counts) is tuple
        for family in contract.families
    )


def validate_analysis_contract(contract: ConfirmationAnalysisContract) -> None:
    """JSON numeric/type identity supplements dataclass equality (116 != 116.0)."""
    if type(contract) is not ConfirmationAnalysisContract or not _closed_container_types(contract):
        raise ValueError("requires its frozen complete analysis contract")
    try:
        encoded = _canonical_json(asdict(contract))
    except (ValueError, TypeError) as error:
        raise ValueError("requires its frozen complete analysis contract") from error
    if encoded != _canonical_json(asdict(fixed_analysis_contract())):
        raise ValueError("requires its frozen complete analysis contract")


def analysis_contract_digest(contract: ConfirmationAnalysisContract) -> str:
    validate_analysis_contract(contract)
    return sha256(_canonical_json(asdict(contract)).encode("utf-8")).hexdigest()
