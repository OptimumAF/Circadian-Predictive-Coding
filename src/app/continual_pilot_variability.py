"""Retain all original pilot variability with retrospective precision forecasts.

Inputs are complete original development declarations. Outputs bind the whole
ledger and all primary paired vectors, preserving original roles and metrics.
No IO, science, experiment launch, winner or prospective count decision occurs.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.app.continual_confirmation_analysis_contract import (
    ConfirmationAnalysisContract,
    analysis_contract_digest,
    fixed_analysis_contract,
)
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_manifest import ConfirmationFamily, fixed_confirmation_manifest
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_findings_development import build_development_ledger
from src.core.pilot_precision import project_pilot_precision
from src.core.seed_statistics import SeedObservation


def _check_complete_ledger(
    ledger: dict[str, Any], families: tuple[ConfirmationFamily, ...]
) -> None:
    same_json(
        [(row["family"], row["seed"], row["arm"]) for row in ledger["cells"]],
        [(f.name, seed, arm) for f in families for seed in f.development_seeds for arm in f.arms],
        "pilot complete ordered original cells",
    )
    same_json(
        [(r["family"], r["seed"], r["left"], r["right"]) for r in ledger["paired_differences"]],
        [
            (f.name, seed, left, right)
            for f in families
            for seed in f.development_seeds
            for left, right in f.contrasts
        ],
        "pilot complete ordered original seed pairs",
    )
    require(ledger["coverage"]["cells"] == 168, "pilot complete cell coverage")
    require(ledger["coverage"]["ordered_seed_pairs"] == 174, "pilot complete paired coverage")


def _vector(
    family: ConfirmationFamily,
    pair: tuple[str, str],
    metric: str,
    rows: list[dict[str, Any]],
    contract: ConfirmationAnalysisContract,
) -> dict[str, Any]:
    field = (
        "signed_forgetting"
        if family.name == "gating" and metric == "signed_forgetting_a"
        else metric
    )
    observations = tuple(SeedObservation(row["seed"], row["differences"][field]) for row in rows)
    projection = project_pilot_precision(
        observations,
        family.development_seeds,
        contract.sample_count,
        contract.zero_deviation_tolerance,
    )
    error = projection.projected_standard_error
    return {
        "family": family.name,
        "left": pair[0],
        "right": pair[1],
        "primary_metric": metric,
        "original_metric_field": field,
        "original_pair_provenance": [
            {"seed": r["seed"], "origin": r["origin"], "result_pointer": r["result_pointer"]}
            for r in rows
        ],
        "projection": asdict(projection),
        "projected_marginal_half_width": None
        if error is None
        else error * contract.marginal_critical,
        "projected_simultaneous_half_width": None
        if error is None
        else error * contract.simultaneous_critical,
    }


def _all_vectors(
    ledger: dict[str, Any],
    families: tuple[ConfirmationFamily, ...],
    contract: ConfirmationAnalysisContract,
) -> list[dict[str, Any]]:
    vectors = []
    for family in families:
        for pair in family.contrasts:
            rows = [
                row
                for row in ledger["paired_differences"]
                if (row["family"], row["left"], row["right"]) == (family.name, *pair)
            ]
            for metric in contract.primary_metrics:
                vectors.append(_vector(family, pair, metric, rows, contract))
    require(len(vectors) == contract.primary_statement_count, "pilot complete primary vectors")
    return vectors


def _project_pilot_variability(ledger: dict[str, Any]) -> dict[str, Any]:
    """Fabricated-fixture seam; it supplies no original/current input authority."""
    try:
        require(type(ledger) is dict, "pilot ledger must be a complete object")
        manifest = fixed_confirmation_manifest()
        contract = fixed_analysis_contract()
        _check_complete_ledger(ledger, manifest.families)
        vectors = _all_vectors(ledger, manifest.families, contract)
        result = {
            "schema_id": "p67_complete_retrospective_pilot_variability_v1",
            "input_bindings": {
                "complete_development_inputs": canonical_body_identity(ledger["original_inputs"]),
                "complete_development_ledger": canonical_body_identity(ledger),
                "analysis_contract_sha256": analysis_contract_digest(contract),
            },
            "original_manifest": asdict(manifest),
            "original_analysis_contract": asdict(contract),
            "vectors": vectors,
            "coverage": {
                "families": len(manifest.families),
                "cells": 168,
                "original_seed_pairs": 174,
                "named_pairs": 58,
                "primary_vectors": len(vectors),
                "pilot_observations": 348,
                "family_pilot_seed_instances": 18,
                "distinct_pilot_source_seeds": 15,
            },
            "evidence_timing": "retrospective_after_original_confirmation",
            "prospective_sample_size_justification": False,
            "original_p67_acceptance_complete": False,
            "new_training_scoring_or_final_access": False,
            "interpretation": {
                "pilot_role": "original_outer_selection_development_not_independent_final_confirmation",
                "forecast": "pilot_SD_divided_by_sqrt_original_ten_count_conditional_on_equal_future_dispersion",
                "half_widths": "original_df9_116_statement_critical_scaling_not_measured_intervals_or_power",
                "limitations": "three_seed_variance_is_weak_role_dispersion_may_differ_constants_do_not_prove_precision",
                "replication": "within_family_only_no_pooling_shared_sources_or_repeat_replications",
                "decision": "no_precision_target_sample_count_decision_or_winner_original_justification_missing",
            },
        }
        canonical_body_identity(
            result
        )  # Reject any overflow/nonfinite derived forecast before output.
        return result
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("pilot ledger is malformed or incomplete") from error


def build_pilot_variability_report(original_development_inputs: dict[str, Any]) -> dict[str, Any]:
    """Validate the whole original input boundary before any pilot projection."""
    return _project_pilot_variability(build_development_ledger(original_development_inputs))
