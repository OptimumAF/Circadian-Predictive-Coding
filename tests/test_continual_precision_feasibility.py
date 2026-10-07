"""Full fabricated scope, fixed objective, late failures and no scientific access."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from test_continual_findings_development import fabricated_inputs
from src.app.continual_findings_development import _derive_development_ledger
from src.app.continual_pilot_variability import _project_pilot_variability
from src.app import continual_precision_feasibility as feasibility
from src.app.continual_precision_contract import fixed_precision_contract
from src.app.continual_confirmation_report_costs import canonical_body_identity


def pilot() -> dict[str, Any]:
    return _project_pilot_variability(_derive_development_ledger(fabricated_inputs()))


def test_should_keep_all_vectors_assumptions_and_whole_budget_without_authorizing_confirmation() -> (
    None
):
    source = pilot()
    before = deepcopy(source)
    result = feasibility._project_precision_feasibility(source, fixed_precision_contract())
    assert source == before
    assert len(result["vectors"]) == 116
    assert result["coverage"]["pilot_observations"] == 348
    assert result["work_budget"]["candidate_maximum_optimizer_updates"] == 15620
    assert result["work_budget"]["maximum_optimizer_updates"] == 16000
    assert result["work_budget"]["maximum_complete_replications_per_family"] == 10
    assert result["precision_objective_supported_by_all_bounds"] is False
    assert result["new_confirmation_authorized"] is False
    assert result["untouched_role_seed_binding_complete"] is False
    assert result["original_p67_acceptance_complete"] is False
    assert result["complete_original_pilot_identity"] == canonical_body_identity(source)
    for vector in result["vectors"]:
        accuracy = vector["primary_metric"] == "final_mean_task_accuracy"
        assert vector["precision"]["bounded_mean_required_seeds"] == (6754 if accuracy else 27016)
        assert vector["precision"]["pilot"]["pilot_summary"]["observations"] == next(
            r["projection"]["pilot_summary"]["observations"]
            for r in source["vectors"]
            if (r["family"], r["left"], r["right"], r["primary_metric"])
            == (vector["family"], vector["left"], vector["right"], vector["primary_metric"])
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_half_width", 0.1),
        ("mean_family_alpha", 0.1),
        ("pilot_variance_family_alpha", 0.1),
        ("candidate_seed_count", 11),
        ("statement_count", 115),
        ("statement_count", 116.0),
        ("forgetting_difference_range", [-2.0, 2.0]),
        ("normal_sensitivity_assumptions", ()),
    ],
)
def test_should_reject_changed_objective_or_type_before_reading_inputs(
    field: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def prohibited(*args: Any, **kwargs: Any) -> None:
        pytest.fail("changed objective reached original input validation")

    monkeypatch.setattr(feasibility, "build_pilot_variability_report", prohibited)
    with pytest.raises(ValueError, match="fixed precision contract"):
        feasibility.build_precision_feasibility(
            {}, replace(fixed_precision_contract(), **{field: value})
        )


@pytest.mark.parametrize(
    "change",
    [
        "last_seed",
        "last_vector",
        "duplicate",
        "last_value",
        "last_projection",
        "last_metric",
        "whole_work",
        "reorder",
    ],
)
def test_should_reject_late_changed_scope_observations_or_detached_forecasts(change: str) -> None:
    source = pilot()
    if change == "last_seed":
        source["vectors"][-1]["projection"]["pilot_summary"]["observations"][-1]["seed"] = True
    if change == "last_vector":
        source["vectors"].pop()
    if change == "duplicate":
        source["vectors"][-1] = deepcopy(source["vectors"][-2])
    if change == "last_value":
        source["vectors"][-1]["projection"]["pilot_summary"]["observations"][-1]["value"] = float(
            "nan"
        )
    if change == "last_projection":
        source["vectors"][-1]["projection"]["projected_standard_error"] = 0.1
    if change == "last_metric":
        source["vectors"][-1]["primary_metric"] = "a_after_b"
    if change == "whole_work":
        source["original_manifest"]["families"][-1]["maximum_optimizer_updates"] += 1
    if change == "reorder":
        source["vectors"].reverse()
    with pytest.raises(ValueError):
        feasibility._project_precision_feasibility(source, fixed_precision_contract())


@pytest.mark.parametrize("value", [None, [], {}, {"original_scope": {}, "input_catalog": {}}])
def test_should_reject_unbound_public_original_inputs(value: Any) -> None:
    with pytest.raises(ValueError):
        feasibility.build_precision_feasibility(value, fixed_precision_contract())


def test_should_refuse_fabricated_scope_as_original_authority() -> None:
    with pytest.raises(ValueError, match="whole original prospective scope"):
        feasibility.build_precision_feasibility(fabricated_inputs(), fixed_precision_contract())


def test_should_repeat_complete_detached_result_without_io_source_model_training_or_final_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pilot()
    before = deepcopy(source)
    from src.app import continual_arrived_benchmark as arrived
    from src.core.backprop_mlp import BackpropMLP
    from src.core.predictive_coding import PredictiveCodingNetwork
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork

    def prohibited(*args: Any, **kwargs: Any) -> None:
        pytest.fail("pure precision performed science or IO")

    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        for name in ("__init__", "train_epoch", "predict_proba", "compute_accuracy"):
            monkeypatch.setattr(model, name, prohibited)
    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, prohibited)
    for name in ("read_bytes", "read_text", "write_bytes", "write_text", "open"):
        monkeypatch.setattr(Path, name, prohibited)
    first = feasibility._project_precision_feasibility(source, fixed_precision_contract())
    second = feasibility._project_precision_feasibility(source, fixed_precision_contract())
    assert canonical_body_identity(first) == canonical_body_identity(second)
    first["vectors"][-1]["precision"]["pilot"]["pilot_summary"]["observations"][-1]["value"] = 99
    first["original_input_bindings"]["analysis_contract_sha256"] = "0" * 64
    assert source == before
    assert second == feasibility._project_precision_feasibility(source, fixed_precision_contract())
