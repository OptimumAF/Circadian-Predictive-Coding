"""Fabricated full-scope outcomes test analysis without final data or scores."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pytest

from src.app import continual_confirmation_analysis as analysis
from src.app.continual_confirmation_analysis_contract import (
    analysis_contract_digest,
    fixed_analysis_contract,
    validate_analysis_contract,
)
from src.core.continual_metrics import TwoTaskAccuracy


def _cells() -> tuple[analysis.OutcomeCell, ...]:
    cells = []
    for family in fixed_analysis_contract().families:
        for index, seed in enumerate(family.seeds):
            roles = analysis.FinalRoleIdentity(f"{seed * 2:064x}", f"{seed * 2 + 1:064x}")
            for position, arm in enumerate(family.arms):
                # Fabricated forty-example counts, not model/scorer observations.
                accuracy = TwoTaskAccuracy(
                    (20 + index) / 40,
                    (12 + position + (index * (position % 3 + 1)) % 10) / 40,
                    (10 + position + index) / 40,
                )
                cells.append(analysis.OutcomeCell(family.name, seed, arm, roles, accuracy))
    return tuple(cells)


def _metric(metrics: tuple[analysis.MetricSummary, ...], name: str) -> analysis.MetricSummary:
    return next(item for item in metrics if item.metric == name)


def test_should_retain_all_cells_pairs_primary_statements_and_original_seed_order() -> None:
    contract = fixed_analysis_contract()
    result = analysis.analyze_confirmation(_cells(), contract)

    assert result.complete and result.successful_cells == 560 and result.failed_cells == 0
    assert len(result.cells) == 560
    assert result.distinct_source_seeds == 50
    assert result.primary_statement_count == 116
    assert result.contract_sha256 == analysis_contract_digest(contract)
    assert result.cost_facts_reference_sha256 == contract.train_result_sha256
    assert result.input_validation_scope == "declared_outcomes_and_pairing_only_not_source_proof"
    assert [len(f.contrasts) for f in result.families] == [1, 3, 3, 9, 22, 20]
    assert sum(len(f.arms) for f in result.families) == 56
    assert sum(len(f.contrasts) for f in result.families) == 58
    for family, declared in zip(result.families, contract.families, strict=True):
        assert family.name == declared.name
        assert tuple(arm.name for arm in family.arms) == declared.arms
        assert tuple((pair.left, pair.right) for pair in family.contrasts) == declared.contrasts
        for arm in family.arms:
            assert len(arm.metrics) == 6
            for item in arm.metrics:
                assert tuple(row.seed for row in item.summary.observations) == declared.seeds
                assert item.summary.simultaneous_interval is None
        for pair in family.contrasts:
            assert len(pair.metrics) == 5
            for item in pair.metrics:
                assert tuple(row.seed for row in item.summary.observations) == declared.seeds
                if item.metric in contract.primary_metrics:
                    assert item.primary_endpoint
                    if item.summary.simultaneous_interval is not None:
                        assert item.summary.simultaneous_interval.statement_count == 116
                else:
                    assert not item.primary_endpoint
                    assert item.summary.simultaneous_interval is None
    encoded = json.dumps(asdict(result), sort_keys=True, allow_nan=False)
    assert encoded == json.dumps(
        asdict(analysis.analyze_confirmation(tuple(reversed(_cells())), contract)),
        sort_keys=True,
        allow_nan=False,
    )


def test_should_compute_differences_before_dispersion_and_keep_both_accuracy_endpoints() -> None:
    rows = list(_cells())
    for index, row in enumerate(rows):
        if row.family == "gating":
            k = fixed_analysis_contract().families[0].seeds.index(row.seed)
            base = 0.25 + k / 64
            shift = 0.25 if row.arm == "chemical_gating" else 0.0
            rows[index] = replace(
                row, accuracy=TwoTaskAccuracy(base + shift, base + shift, base + shift)
            )
    result = analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())
    pair = result.families[0].contrasts[0]
    mean = _metric(pair.metrics, "final_mean_task_accuracy").summary

    assert mean.mean == 0.25
    assert mean.observed_sample_standard_deviation == 0
    assert mean.marginal_interval is mean.simultaneous_interval is None
    assert mean.interval_status == "zero_observed_variance"
    assert _metric(pair.metrics, "a_after_a").summary.mean == 0.25
    assert _metric(pair.metrics, "a_after_b").summary.mean == 0.25
    assert _metric(pair.metrics, "signed_forgetting_a").preferred_direction == "lower"
    deviation = _metric(
        result.families[0].arms[1].metrics, "a_after_a"
    ).summary.observed_sample_standard_deviation
    assert deviation is not None and deviation > 0


def test_should_compute_marginal_and_all_116_simultaneous_intervals_for_known_paired_counts() -> (
    None
):
    report = analysis.analyze_confirmation(_cells(), fixed_analysis_contract())
    summary = _metric(report.families[0].contrasts[0].metrics, "final_mean_task_accuracy").summary

    # Independently enumerated paired count numerators, each divided by 80.
    expected = [2, 3, 4, 5, -4, 7, 8, -1, 0, 1]
    assert [row.value for row in summary.observations] == pytest.approx([x / 80 for x in expected])
    assert summary.mean == pytest.approx(1 / 32)
    assert summary.standard_error == pytest.approx(7 / 480)
    assert summary.marginal_interval is not None and summary.simultaneous_interval is not None
    assert summary.marginal_interval.lower == pytest.approx(1 / 32 - 2.2621571627982053 * 7 / 480)
    assert summary.simultaneous_interval.lower == pytest.approx(
        1 / 32 - 5.403490569214909 * 7 / 480
    )
    assert summary.simultaneous_interval.upper == pytest.approx(
        1 / 32 + 5.403490569214909 * 7 / 480
    )
    assert summary.simultaneous_interval.confidence_level == 0.95
    assert summary.simultaneous_interval.statement_count == 116
    assert summary.interval_status == "estimated"


def test_should_apply_fixed_roundoff_tolerance_to_constant_decimal_count_differences() -> None:
    rows = list(_cells())
    for index, row in enumerate(rows):
        if row.family == "gating":
            k = fixed_analysis_contract().families[0].seeds.index(row.seed)
            base = 0.30 if k % 2 else 0.15
            left = 0.35 if k % 2 else 0.20
            value = left if row.arm == "chemical_gating" else base
            rows[index] = replace(row, accuracy=TwoTaskAccuracy(value, value, value))
    report = analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())
    summary = _metric(report.families[0].contrasts[0].metrics, "final_mean_task_accuracy").summary

    assert summary.mean == pytest.approx(0.05)
    assert summary.observed_sample_standard_deviation is not None
    assert 0 < summary.observed_sample_standard_deviation < 1e-12
    assert summary.interval_status == "zero_observed_variance"
    assert summary.marginal_interval is summary.simultaneous_interval is None


def test_should_keep_failures_on_both_sides_without_reducing_the_planned_family() -> None:
    rows = list(_cells())
    rows[1] = replace(rows[1], accuracy=None, roles=None, failure="right scorer failed")
    rows[2] = replace(rows[2], accuracy=None, failure="left scorer failed")
    result = analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())
    pair = result.families[0].contrasts[0]

    assert not result.complete and result.failed_cells == 2
    assert result.primary_statement_count == 116 and len(result.cells) == 560
    metric = _metric(pair.metrics, "final_mean_task_accuracy").summary
    assert (metric.planned_seed_count, metric.observed_seed_count) == (10, 9)
    assert metric.mean is metric.standard_error is None
    assert metric.marginal_interval is metric.simultaneous_interval is None
    assert metric.interval_status == "incomplete_observations"
    assert metric.observations[0].reason is not None
    assert "left scorer failed" in metric.observations[0].reason
    assert "right scorer failed" in metric.observations[0].reason


def test_should_keep_zero_denominator_retention_null_and_positive_transfer_above_one() -> None:
    rows = list(_cells())
    rows[0] = replace(rows[0], accuracy=TwoTaskAccuracy(0.0, 0.5, 0.5))
    rows[3] = replace(rows[3], accuracy=TwoTaskAccuracy(0.25, 0.5, 0.5))
    report = analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())
    ratio = _metric(report.families[0].arms[0].metrics, "retention_ratio_a").summary

    assert ratio.observations[0].value is None
    assert ratio.observations[0].reason == "zero_a_after_a"
    assert ratio.observations[1].value == 2.0
    assert ratio.observed_seed_count == 9 and ratio.mean is None
    assert ratio.marginal_interval is ratio.simultaneous_interval is None
    assert (
        _metric(report.families[0].arms[0].metrics, "signed_forgetting_a")
        .summary.observations[0]
        .value
        == -0.5
    )


def test_should_keep_every_all_failed_parent_pair_without_inventing_an_observed_mean() -> None:
    rows = tuple(
        replace(row, accuracy=None, roles=None, failure=f"failed seed {row.seed}")
        if row.family == "parent"
        else row
        for row in _cells()
    )
    report = analysis.analyze_confirmation(rows, fixed_analysis_contract())
    assert not report.complete and report.failed_cells == 80
    assert len(report.families[-1].contrasts) == 20 and report.primary_statement_count == 116
    for pair in report.families[-1].contrasts:
        for metric in pair.metrics:
            summary = metric.summary
            assert (summary.planned_seed_count, summary.observed_seed_count) == (10, 0)
            assert (
                summary.mean
                is summary.observed_mean
                is summary.observed_sample_standard_deviation
                is None
            )
            assert summary.observed_minimum is summary.observed_maximum is None
            assert summary.marginal_interval is summary.simultaneous_interval is None
            assert all(row.value is None and row.reason for row in summary.observations)


def test_should_keep_all_defined_retention_ratios_descriptive_without_normal_model_intervals() -> (
    None
):
    report = analysis.analyze_confirmation(_cells(), fixed_analysis_contract())
    for family in report.families:
        for arm in family.arms:
            summary = _metric(arm.metrics, "retention_ratio_a").summary
            assert summary.observed_seed_count == 10 and summary.mean is not None
            assert summary.marginal_interval is summary.simultaneous_interval is None
            assert summary.interval_status == "descriptive_only"


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "unknown_arm", "unknown_family", "development_seed", "boolean_seed"],
)
def test_should_reject_partial_or_unknown_scope_before_any_summary(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = list(_cells())
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    else:
        changes: dict[str, tuple[str, Any]] = {
            "unknown_arm": ("arm", "winner"),
            "unknown_family": ("family", "other"),
            "development_seed": ("seed", 347),
            "boolean_seed": ("seed", True),
        }
        field, value = changes[change]
        rows[-1] = replace(rows[-1], **{field: value})
    monkeypatch.setattr(analysis, "summarize_seeds", _forbid)
    with pytest.raises(ValueError, match="scope|cell|seed"):
        analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())


@pytest.mark.parametrize(
    "field,value",
    [
        ("a_sha256", "f" * 64),
        ("b_sha256", "e" * 64),
        ("a_count", 39),
        ("b_count", 40.0),
        ("score_role", "outer_selection"),
        ("a_sha256", "bad"),
        ("b_sha256", "E" * 64),
    ],
)
def test_should_reject_late_mismatched_or_malformed_final_roles_before_any_summary(
    field: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = list(_cells())
    assert rows[-1].roles is not None
    rows[-1] = replace(rows[-1], roles=replace(rows[-1].roles, **{field: value}))
    monkeypatch.setattr(analysis, "summarize_seeds", _forbid)
    with pytest.raises(ValueError, match="role"):
        analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())


@pytest.mark.parametrize(
    "change",
    [
        "missing_roles",
        "missing_outcome",
        "double_outcome",
        "blank_failure",
        "false_failure",
        "forged_accuracy",
    ],
)
def test_should_reject_ambiguous_success_or_failure_and_revalidate_raw_accuracy(
    change: str,
) -> None:
    rows = list(_cells())
    if change == "missing_roles":
        rows[-1] = replace(rows[-1], roles=None)
    elif change == "missing_outcome":
        rows[-1] = replace(rows[-1], accuracy=None)
    elif change == "double_outcome":
        rows[-1] = replace(rows[-1], failure="failed")
    elif change == "blank_failure":
        rows[-1] = replace(rows[-1], accuracy=None, failure=" ")
    elif change == "false_failure":
        rows[-1] = replace(rows[-1], accuracy=None, failure=False)  # type: ignore[arg-type]
    else:
        forged = TwoTaskAccuracy(0.5, 0.5, 0.5)
        object.__setattr__(forged, "a_after_b", float("nan"))
        rows[-1] = replace(rows[-1], accuracy=forged)
    with pytest.raises(ValueError, match="role|outcome|failure|finite"):
        analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())


@pytest.mark.parametrize(
    "field,value",
    [
        ("primary_statement_count", 115),
        ("primary_statement_count", 116.0),
        ("sample_count", 9),
        ("sample_count", True),
        ("family_alpha", 0.1),
        ("marginal_critical", 1.96),
        ("simultaneous_critical", 2.262),
        ("zero_deviation_tolerance", 0.0),
        ("train_result_sha256", "f" * 64),
        ("comparison_sign", "right_minus_left"),
        ("score_role", "outer_selection"),
        ("primary_metrics", ("final_mean_task_accuracy",)),
    ],
)
def test_should_reject_changed_or_weakly_typed_predeclared_contract(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="frozen complete analysis contract"):
        validate_analysis_contract(replace(fixed_analysis_contract(), **{field: value}))


@pytest.mark.parametrize(
    "change",
    [
        "families_list",
        "seeds_list",
        "arms_list",
        "pairs_list",
        "pair_list",
        "counts_list",
        "drop_pair",
        "reverse_pair",
        "move_seed",
        "secondary_list",
        "assumptions_list",
    ],
)
def test_should_reject_nested_scope_changes_and_equal_json_but_wrong_container_types(
    change: str,
) -> None:
    contract = fixed_analysis_contract()
    family = contract.families[-1]
    replacements: dict[str, tuple[str, Any]] = {
        "seeds_list": ("seeds", list(family.seeds)),
        "arms_list": ("arms", list(family.arms)),
        "pairs_list": ("contrasts", list(family.contrasts)),
        "pair_list": ("contrasts", tuple(list(pair) for pair in family.contrasts)),
        "counts_list": ("final_counts", list(family.final_counts)),
        "drop_pair": ("contrasts", family.contrasts[:-1]),
        "reverse_pair": ("contrasts", tuple(tuple(reversed(pair)) for pair in family.contrasts)),
        "move_seed": ("seeds", tuple(reversed(family.seeds))),
    }
    if change == "families_list":
        altered = replace(contract, families=list(contract.families))  # type: ignore[arg-type]
    elif change == "secondary_list":
        altered = replace(contract, secondary_metrics=list(contract.secondary_metrics))  # type: ignore[arg-type]
    elif change == "assumptions_list":
        altered = replace(contract, interval_assumptions=list(contract.interval_assumptions))  # type: ignore[arg-type]
    else:
        field, value = replacements[change]
        changed = replace(family, **{field: value})
        altered = replace(contract, families=contract.families[:-1] + (changed,))
    with pytest.raises(ValueError, match="frozen complete analysis contract"):
        validate_analysis_contract(altered)


@pytest.mark.parametrize("field", ["a_after_a", "a_after_b", "b_after_b"])
@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_should_reject_late_tampered_raw_accuracy_before_any_summary(
    field: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = list(_cells())
    accuracy = TwoTaskAccuracy(0.5, 0.5, 0.5)
    object.__setattr__(accuracy, field, value)
    rows[-1] = replace(rows[-1], accuracy=accuracy)
    monkeypatch.setattr(analysis, "summarize_seeds", _forbid)
    with pytest.raises(ValueError, match="finite accuracy"):
        analysis.analyze_confirmation(tuple(rows), fixed_analysis_contract())


def _forbid(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(
        "analysis constructed/read/trained/scored data or emitted a partial summary"
    )


def test_should_run_full_synthetic_analysis_without_source_model_final_or_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.app import continual_arrived_benchmark as arrived
    from src.core.backprop_mlp import BackpropMLP
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
    from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
    from src.core.predictive_coding import PredictiveCodingNetwork
    from src.infra import continual_roles
    import numpy as np

    cells = _cells()
    with monkeypatch.context() as sealed:
        for model in (
            BackpropMLP,
            PredictiveCodingNetwork,
            CircadianPredictiveCodingNetwork,
            ParentControlledCircadianNetwork,
        ):
            for name in ("__init__", "train_epoch", "compute_accuracy", "predict_proba"):
                sealed.setattr(model, name, _forbid)
        sealed.setattr(arrived, "_build_phase_a_roles", _forbid)
        sealed.setattr(arrived, "_build_phase_b_roles", _forbid)
        sealed.setattr(continual_roles, "release_final_test", _forbid)
        sealed.setattr(np.random, "default_rng", _forbid)
        sealed.setattr(Path, "read_bytes", _forbid)
        sealed.setattr(Path, "read_text", _forbid)
        result = analysis.analyze_confirmation(cells, fixed_analysis_contract())
    assert len(result.cells) == 560
