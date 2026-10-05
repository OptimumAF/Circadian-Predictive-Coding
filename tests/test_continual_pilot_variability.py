"""Complete fabricated pilot scope and the public original-input gate."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.app import continual_pilot_variability as pilot
from src.app import continual_findings_development as development
from src.app.continual_confirmation_report_costs import canonical_body_identity
from test_continual_findings_development import fabricated_inputs


def ledger() -> dict[str, Any]:
    return development._derive_development_ledger(fabricated_inputs())


def test_should_keep_every_pair_metric_seed_and_original_scope_without_a_count_decision() -> None:
    source = ledger()
    before = deepcopy(source)
    result = pilot._project_pilot_variability(source)

    assert source == before
    assert len(result["vectors"]) == result["coverage"]["primary_vectors"] == 116
    assert result["coverage"]["pilot_observations"] == 348
    assert result["coverage"]["distinct_pilot_source_seeds"] == 15
    assert result["coverage"]["family_pilot_seed_instances"] == 18
    assert result["original_p67_acceptance_complete"] is False
    assert result["prospective_sample_size_justification"] is False
    assert result["evidence_timing"] == "retrospective_after_original_confirmation"
    assert result["vectors"][0]["primary_metric"] == "final_mean_task_accuracy"
    assert result["vectors"][1]["primary_metric"] == "signed_forgetting_a"
    assert result["vectors"][1]["original_metric_field"] == "signed_forgetting"
    assert result["vectors"][0]["projection"]["confirmation_seed_count"] == 10
    assert result["vectors"][0]["projection"]["projected_standard_error"] is None
    assert result["vectors"][0]["projection"]["status"] == "zero_observed_variance"
    for vector in result["vectors"]:
        seeds = next(
            f["development_seeds"]
            for f in source["original_inputs"]["original_scope"]["manifest"]["families"]
            if f["name"] == vector["family"]
        )
        assert [
            row["seed"] for row in vector["projection"]["pilot_summary"]["observations"]
        ] == seeds
        assert len(vector["original_pair_provenance"]) == 3
    assert result["input_bindings"]["complete_development_ledger"] == canonical_body_identity(
        source
    )
    assert result["original_manifest"]["max_optimizer_updates"] == 16000
    assert result["original_analysis_contract"]["primary_statement_count"] == 116


def test_should_scale_only_original_declared_critical_values_and_preserve_raw_signs() -> None:
    source = ledger()
    last_pair = source["paired_differences"][-1]
    last_pair["differences"]["signed_forgetting_a"] = -0.5
    result = pilot._project_pilot_variability(source)
    vector = result["vectors"][-1]
    error = vector["projection"]["projected_standard_error"]
    assert error is not None
    assert vector["projected_marginal_half_width"] == pytest.approx(2.2621571627982053 * error)
    assert vector["projected_simultaneous_half_width"] == pytest.approx(5.403490569214909 * error)
    assert vector["projection"]["pilot_summary"]["observations"][-1]["value"] == -0.5
    assert "interval" not in vector


@pytest.mark.parametrize(
    "change",
    [
        "missing_pair",
        "duplicate_pair",
        "last_seed",
        "last_family",
        "last_metric",
        "reorder_pairs",
        "missing_cell",
        "last_nan",
    ],
)
def test_should_reject_late_incomplete_malformed_or_reordered_projection_scope(change: str) -> None:
    source = ledger()
    rows = source["paired_differences"]
    if change == "missing_pair":
        rows.pop()
    elif change == "duplicate_pair":
        rows[-1] = deepcopy(rows[-2])
    elif change == "last_seed":
        rows[-1]["seed"] = True
    elif change == "last_family":
        rows[-1]["family"] = "gating"
    elif change == "last_metric":
        del rows[-1]["differences"]["signed_forgetting_a"]
    elif change == "reorder_pairs":
        rows.reverse()
    elif change == "missing_cell":
        source["cells"].pop()
    elif change == "last_nan":
        rows[-1]["differences"]["signed_forgetting_a"] = float("nan")
    with pytest.raises(ValueError):
        pilot._project_pilot_variability(source)


@pytest.mark.parametrize("value", [None, [], {}, {"original_scope": {}, "input_catalog": {}}])
def test_should_reject_incomplete_public_inputs_before_any_projection(value: Any) -> None:
    with pytest.raises(ValueError):
        pilot.build_pilot_variability_report(value)


def test_should_refuse_fabricated_original_authority_at_the_public_gate() -> None:
    with pytest.raises(ValueError, match="whole original prospective scope"):
        pilot.build_pilot_variability_report(fabricated_inputs())


def test_should_repeat_detached_complete_projection_without_source_model_or_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ledger()
    from src.core.backprop_mlp import BackpropMLP
    from src.core.predictive_coding import PredictiveCodingNetwork
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
    from src.app import continual_arrived_benchmark as arrived

    def prohibited(*args: Any, **kwargs: Any) -> None:
        pytest.fail("pilot projection performed scientific work or IO")

    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        for name in ("__init__", "train_epoch", "predict_proba", "compute_accuracy"):
            monkeypatch.setattr(model, name, prohibited)
    for name in ("read_bytes", "read_text", "write_bytes", "write_text", "open"):
        monkeypatch.setattr(Path, name, prohibited)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", prohibited)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", prohibited)
    monkeypatch.setattr(arrived, "release_final_test", prohibited)
    first = pilot._project_pilot_variability(source)
    second = pilot._project_pilot_variability(source)
    assert canonical_body_identity(first) == canonical_body_identity(second)
    first["vectors"][-1]["projection"]["pilot_summary"]["observations"][-1]["value"] = 99
    assert second == pilot._project_pilot_variability(source)
