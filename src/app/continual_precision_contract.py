"""Fixed prospective precision objective; no source, score, IO or execution authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ConfirmationPrecisionContract:
    contract_id: str = "mechanism_confirmation_precision_feasibility_v1"
    target_half_width: float = 0.05
    mean_family_alpha: float = 0.05
    pilot_variance_family_alpha: float = 0.05
    statement_count: int = 116
    pilot_seed_count: int = 3
    candidate_seed_count: int = 10
    mean_accuracy_difference_range: tuple[float, float] = (-1.0, 1.0)
    forgetting_difference_range: tuple[float, float] = (-2.0, 2.0)
    objective_rationale: str = "five_percentage_points_one_decision_in_twenty_two_original40_example_task_counts_same_objective_for_all_primary_pairs_independent_of_confirmation_outcomes"
    count_policy: str = (
        "candidate_ten_within_original_hard_budget_no_seed_selection_or_favorable_stopping"
    )
    normal_sensitivity_assumptions: tuple[str, ...] = (
        "exactly_normal_independent_three_seed_pilot_differences",
        "future_final_role_dispersion_equals_development_pilot_dispersion",
        "pilot_variance_and_future_mean_levels_separate_not_joint95_coverage",
        "fixed_original_df9_critical_scaling_only_not_new_historical_intervals",
    )
    bounded_mean_assumptions: tuple[str, ...] = (
        "independent_bounded_source_seed_replications_with_declared_pair_ranges",
        "simultaneous_future_mean_bound_uses_union_over_all116_no_cross_contrast_independence_required",
        "conservative_sufficient_count_not_a_necessary_count_or_optimal_method",
    )


def fixed_precision_contract() -> ConfirmationPrecisionContract:
    return ConfirmationPrecisionContract()


def validate_precision_contract(contract: ConfirmationPrecisionContract) -> None:
    """Reject partial/changed targets, numeric types, ranges and method assumptions."""
    if type(contract) is not ConfirmationPrecisionContract:
        raise ValueError("requires the complete fixed precision contract")
    for field in (
        "mean_accuracy_difference_range",
        "forgetting_difference_range",
        "normal_sensitivity_assumptions",
        "bounded_mean_assumptions",
    ):
        if type(getattr(contract, field)) is not tuple:
            raise ValueError("requires the complete fixed precision contract")
    try:
        encoded = json.dumps(asdict(contract), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("requires the complete fixed precision contract") from error
    if encoded != json.dumps(asdict(fixed_precision_contract()), sort_keys=True, allow_nan=False):
        raise ValueError("requires the complete fixed precision contract")
