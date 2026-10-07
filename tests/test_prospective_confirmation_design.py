"""Full fixed design and fabricated declarations with raising science/IO guards."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import prospective_confirmation_design as design_module
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot


@pytest.fixture(autouse=True)
def no_science_or_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("prospective metadata constructed scientific values or performed IO")

    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbidden)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", forbidden)
        monkeypatch.setattr(model, "compute_accuracy", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)


def test_should_retain_every_configuration_role_cell_statement_and_fixed_cap() -> None:
    design = design_module.fixed_prospective_design()
    design_module.validate_prospective_design(design)
    assert len(design["family_templates"]) == 6
    assert [len(row["arm_templates"]) for row in design["family_templates"]] == [3, 8, 9, 11, 17, 8]
    assert len(design["replica_slots"]) == 60
    assert len(design["role_requirements"]) == 480
    assert len(design["cell_plan"]) == 560
    assert len(design["ordered_primary_statements"]) == 116
    assert len(design["derived_stream_requirements"]) == 8
    assert design["resource_envelope"]["complete_count_optimizer_ceiling"] == 15620
    assert design["resource_envelope"]["maximum_optimizer_updates"] == 16000
    assert design["resource_envelope"]["wall_limit_seconds"] == 600
    assert design["resource_envelope"]["max_process_rss_bytes"] == 512 * 1024 * 1024
    assert design["stopping_contract"]["planned_complete_endpoint_count"] == 1680
    assert design["replication_policy"]["planned_shared_source_groups"] == 50
    assert design["replication_policy"]["actual_base_seed_bindings"] is None
    assert design["replication_policy"]["independent_source_replications"] is None
    assert design["actual_new_request_identity"] is design["actual_new_source_identity"] is None
    assert not design["untouched_role_binding_complete"] and not design["fresh_roles_authorized"]
    assert design["replication_policy"]["precision_status"].startswith("exploratory_")
    assert design_module.prospective_design_identity(
        design
    ) == design_module.prospective_design_identity(deepcopy(design))


def test_should_keep_original_reservations_only_in_historical_references() -> None:
    design = design_module.fixed_prospective_design()
    for family in design["family_templates"]:
        assert "seeds" not in family["configuration_without_historical_reservations"]
        assert "confirmation_seeds" not in family["configuration_without_historical_reservations"]
        assert all(
            "seed" not in row["configuration_without_seed"] for row in family["arm_templates"]
        )
    assert len(design["original_complete_manifest"]["families"][0]["seeds"]) == 10
    assert design["original_complete_analysis"]["sample_count"] == 10
    assert design["original_precision_objective"]["target_half_width"] == 0.05


def test_should_require_every_source_and_label_role_without_releasing_final_values() -> None:
    design = design_module.fixed_prospective_design()
    roles = design["role_requirements"]
    assert [row["expected_count"] for row in roles[:8]] == [72, 24, 24, 40, 36, 12, 12, 40]
    finals = [row for row in roles if row["role"] == "final_test"]
    assert len(finals) == 120
    assert all(
        row["source_available_at"] == row["labels_available_at"] == "complete_global_freeze"
        for row in finals
    )
    assert all(
        row["actual_sample_ids"] is row["actual_role_array_identity"] is None for row in roles
    )
    assert "unscored" in roles[2]["allowed_use"]
    assert "prior_usage" in design["global_order_requirements"][0]


def test_should_validate_all_fabricated_bindings_without_mutating_design_or_admitting() -> None:
    design = design_module.fixed_prospective_design()
    before = deepcopy(design)
    slots = tuple(ReplicaSlot(**row) for row in design["replica_slots"])
    groups = tuple(dict.fromkeys(row.source_group for row in slots))
    bindings = tuple(
        ReplicaSeedBinding(row, 1_000_000 + 20_000 * groups.index(row.source_group))
        for row in slots
    )
    result = design_module.declare_prospective_replica_streams(design, bindings)
    assert len(result.bindings) == 60 and len(result.source_groups) == 50
    assert result.independent_source_replications is None and not result.fresh_roles_authorized
    assert design == before


@pytest.mark.parametrize(
    "mutation",
    [
        "late_config",
        "late_arm",
        "omit_cell",
        "omit_role",
        "role_count",
        "early_final",
        "outer_score",
        "reorder_slots",
        "omit_statement",
        "change_metric",
        "float_count",
        "bool_count",
        "cap_increase",
        "wall_increase",
        "rss_increase",
        "missing_policy",
        "constant_policy",
        "favorable_stop",
        "actual_seed",
        "request_identity",
        "admission",
        "unknown_field",
        "mutable_type",
        "missing_request_proof",
    ],
)
def test_should_reject_any_late_scope_type_rule_or_authority_drift(mutation: str) -> None:
    design = design_module.fixed_prospective_design()
    if mutation == "late_config":
        design["family_templates"][-1]["arm_templates"][-1]["configuration_without_seed"][
            "initial_width"
        ] += 1
    elif mutation == "late_arm":
        design["family_templates"][-1]["arm_templates"].pop()
    elif mutation == "omit_cell":
        design["cell_plan"].pop()
    elif mutation == "omit_role":
        design["role_requirements"].pop()
    elif mutation == "role_count":
        design["role_requirements"][-1]["expected_count"] = 39
    elif mutation == "early_final":
        design["role_requirements"][-1]["labels_available_at"] = "phase_b_arrival"
    elif mutation == "outer_score":
        design["role_requirements"][2]["allowed_use"] = "setting_selection"
    elif mutation == "reorder_slots":
        design["replica_slots"].reverse()
    elif mutation == "omit_statement":
        design["ordered_primary_statements"].pop()
    elif mutation == "change_metric":
        design["ordered_primary_statements"][-1]["metric"] = "balanced_score"
    elif mutation in {"float_count", "bool_count"}:
        design["replication_policy"]["replications_per_family"] = (
            10.0 if mutation == "float_count" else True
        )
    elif mutation in {"cap_increase", "wall_increase", "rss_increase"}:
        name = {
            "cap_increase": "maximum_optimizer_updates",
            "wall_increase": "wall_limit_seconds",
            "rss_increase": "max_process_rss_bytes",
        }[mutation]
        design["resource_envelope"][name] += 1
    elif mutation in {"missing_policy", "constant_policy"}:
        design["original_complete_analysis"][mutation] = "drop_rows"
    elif mutation == "favorable_stop":
        design["stopping_contract"]["favorable_stopping_or_seed_replacement"] = True
    elif mutation == "actual_seed":
        design["replication_policy"]["actual_base_seed_bindings"] = [17001]
    elif mutation == "request_identity":
        design["actual_new_request_identity"] = "old_request"
    elif mutation == "admission":
        design["fresh_roles_authorized"] = True
    elif mutation == "unknown_field":
        design["verified_precision"] = True
    elif mutation == "mutable_type":
        design["replica_slots"] = tuple(design["replica_slots"])
    else:
        design["required_actual_request_bindings"].pop()
    with pytest.raises(ValueError):
        design_module.validate_prospective_design(design)
