"""Complete synthetic saved scope; these fixtures prove no original execution."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test_continual_precision_feasibility import pilot
from src.app.continual_precision_contract import fixed_precision_contract
from src.app.continual_precision_feasibility import _project_precision_feasibility
from src.app.prospective_confirmation_evidence import build_pending_confirmation_contract
from src.core.seed_release_chronology import recorded_release_order


def _witness(mode: str, prior: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    for item in recorded_release_order(120, 1680):
        if item.kind.startswith("training_"):
            pointer = "/result/" + item.kind
        else:
            field = (
                "source_events"
                if item.kind.startswith("source_")
                else "release_events"
                if item.kind == "role_release"
                else "prediction_events"
            )
            pointer = f"/audit/final_observation/{field}/{item.record_index}"
        nodes.append(asdict(item) | {"input_pointer": pointer, "whole_record_sha256": "e" * 64})
    links, identities = {}, {}
    for index, name in enumerate(("request", "result", "audit"), 0 if mode == "canonical" else 3):
        row = prior["content_ledger"][index]
        identities[name] = {"byte_count": row["byte_count"], "sha256": row["sha256"]}
        links[name] = {
            "path": row["aliases"][0],
            "identity": identities[name],
            "all_original_aliases": row["aliases"],
            "prior_ledger_pointer": f"/content_ledger/{index}",
            "whole_saved_witness_record_sha256": row["whole_saved_witness_record_sha256"],
        }
    chronology = {
        "causal_nodes": nodes,
        "coverage": {
            "family_seed_rows": 60,
            "cells": 560,
            "release_events": 120,
            "source_reads": 240,
            "prediction_events": 1680,
            "prediction_examples": 67200,
            "causal_nodes": 2043,
        },
        "exact_actual_release_utc": None,
        "independent_source_replications": None,
        "cross_run_release_order_verified": False,
        "complete_original_reader_verified": False,
        "fresh_roles_authorized": False,
        "complete_prior_usage_acceptance": False,
        "original_p67_acceptance_complete": False,
    }
    return {
        "schema_id": "p67_source_bound_original_release_witness_readback_v1",
        "original_bundle": mode,
        "complete_prior_ledger_identity": identity,
        "entire_prior_coverage_preserved": prior["coverage"],
        "every_prior_unknown_preserved": prior["uncertainty_counts"],
        "entire_prior_input_identity": prior["input_identity"],
        "readback": {
            "chronology": chronology,
            "complete_original_reader_verified": True,
            "file_identities": identities,
        },
        "actual_unchanged_complete_reader_dispatches": {
            "complete_scored_readers": 1,
            "complete_training_readers": {"canonical": 1, "repeat": 1},
        },
        "all_original_raw_scoring_parts_linked_to_complete_prior_ledger": links,
        "fresh_roles_authorized": False,
        "complete_prior_usage_acceptance": False,
        "original_p67_acceptance_complete": False,
        "new_current_semantic_audits": 0,
        "new_scientific_dispatches": 0,
    }


@pytest.fixture
def saved_contract_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    # Full cardinalities catch omissions at the very end without real saved artifacts.
    ledger = [
        {
            "sha256": f"{index:064x}",
            "byte_count": index + 1,
            "aliases": [f"fabricated/{index}/{alias}" for alias in range(2 if index < 264 else 3)],
            "input_pointer": f"/contents/{index}",
            "whole_saved_witness_record_sha256": "f" * 64,
            "execution_and_release_status": "unverified",
            "coverage": {"metadata_views": 1},
            "meaning_counts": {"numeric_seed_candidate": 1},
            "numeric_candidate_occurrences": [{"value": 7, "occurrences": 1}],
            "uncertainty_counts": {"unresolved": 1},
        }
        for index in range(4471)
    ]
    prior: dict[str, Any] = {
        "schema_id": "conservative_saved_seed_chronology_v1",
        "content_ledger": ledger,
        "coverage": {"contents": 4471, "alias_occurrences": 13149},
        "uncertainty_counts": {
            "unresolved": 4471,
            "actual_content_execution_and_release_unverified": 4471,
        },
        "meaning_counts": {"numeric_seed_candidate": 4471},
        "numeric_candidate_occurrences": [{"value": 7, "occurrences": 4471}],
        "input_identity": {"byte_count": 123, "sha256": "a" * 64},
        "independent_source_replication_count": None,
        "verified_execution_events": 0,
        "verified_role_release_events": 0,
        "role_screen": {"proposed_base_seeds": []},
        "complete_prior_usage_acceptance": False,
        "fresh_roles_authorized": False,
        "original_p67_acceptance_complete": False,
        "prospective_real_role_seeds_selected": False,
    }
    identities = {
        name: {"byte_count": 456, "sha256": "b" * 64}
        for name in (
            "chronology",
            "precision",
            "precision_repeat",
            "canonical_witness",
            "repeat_witness",
        )
    }
    precision = json.loads(
        json.dumps(_project_precision_feasibility(pilot(), fixed_precision_contract()))
    )
    inputs = {
        "chronology": prior,
        "precision": precision,
        "precision_repeat": deepcopy(precision),
        "canonical_witness": _witness("canonical", prior, identities["chronology"]),
        "repeat_witness": _witness("repeat", prior, identities["chronology"]),
    }
    return deepcopy(inputs), identities


def test_should_preserve_every_prior_effect_alias_precision_vector_and_causal_node_without_io(
    saved_contract_inputs: tuple[dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, identities = saved_contract_inputs
    before = deepcopy((inputs, identities))

    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure pending contract performed IO/RNG/science")

    from src.app import continual_arrived_benchmark as arrived
    from src.core.backprop_mlp import BackpropMLP
    from src.core.predictive_coding import PredictiveCodingNetwork
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork

    monkeypatch.setattr(Path, "open", forbid)
    monkeypatch.setattr(np.random, "default_rng", forbid)
    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbid)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        for name in ("__init__", "train_epoch", "predict_proba", "compute_accuracy"):
            monkeypatch.setattr(model, name, forbid)
    result = build_pending_confirmation_contract(inputs, identities)
    assert len(result["every_original_prior_effect"]) == 4471
    assert [
        row["complete_original_ledger_row"] for row in result["every_original_prior_effect"]
    ] == inputs["chronology"]["content_ledger"]
    assert (
        sum(
            len(row["complete_original_ledger_row"]["aliases"])
            for row in result["every_original_prior_effect"]
        )
        == 13149
    )
    assert result["entire_original_precision_feasibility"] == inputs["precision"]
    assert result["whole_original_reader_witness_inputs"]["repeat"] == inputs["repeat_witness"]
    assert result == build_pending_confirmation_contract(inputs, identities)
    assert result["actual_original_readers_reexecuted"] is False
    assert result["actual_current_source_proof_verified"] is False
    assert result["independent_source_replications"] is None
    assert not result["all_unresolved_prior_effects_cleared_for_fresh_roles"]
    assert not result["fresh_roles_authorized"]
    result["every_original_prior_effect"][-1]["complete_original_ledger_row"]["aliases"].pop()
    result["whole_original_reader_witness_inputs"]["repeat"]["readback"]["chronology"][
        "causal_nodes"
    ].pop()
    result["whole_declared_input_identities"]["chronology"]["sha256"] = "c" * 64
    assert (inputs, identities) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "omit_content",
        "last_alias",
        "last_witness",
        "last_unknown",
        "unknown_sum",
        "last_meaning",
        "last_numeric",
        "last_pointer",
        "executed",
        "prior_independent",
        "selected",
        "precision_repeat",
        "last_vector",
        "last_precision",
        "late_null",
        "precision_count",
        "work",
        "late_causal",
        "late_causal_pointer",
        "late_causal_hash",
        "omit_causal",
        "extra_causal",
        "raw_pointer",
        "raw_alias",
        "raw_hash",
        "raw_identity",
        "dispatch",
        "admit",
        "utc",
        "independent",
        "reader",
        "whole_identity",
        "missing_input",
    ],
)
def test_should_reject_late_omissions_corruption_false_proof_and_changed_precision(
    saved_contract_inputs: tuple[dict[str, Any], dict[str, Any]], mutation: str
) -> None:
    inputs, identities = saved_contract_inputs
    prior = inputs["chronology"]
    row = prior["content_ledger"][-1]
    witness = inputs["repeat_witness"]
    chronology = witness["readback"]["chronology"]
    raw = witness["all_original_raw_scoring_parts_linked_to_complete_prior_ledger"]["audit"]
    if mutation == "omit_content":
        prior["content_ledger"].pop()
    elif mutation == "last_alias":
        row["aliases"].pop()
    elif mutation == "last_witness":
        row["whole_saved_witness_record_sha256"] = "bad"
    elif mutation == "last_unknown":
        row["uncertainty_counts"] = {}
    elif mutation == "unknown_sum":
        prior["uncertainty_counts"]["unresolved"] -= 1
    elif mutation == "last_meaning":
        row["meaning_counts"]["numeric_seed_candidate"] = True
    elif mutation == "last_numeric":
        row["numeric_candidate_occurrences"][-1]["occurrences"] = False
    elif mutation == "last_pointer":
        row["input_pointer"] = "/contents/0"
    elif mutation == "executed":
        row["execution_and_release_status"] = "verified"
    elif mutation == "prior_independent":
        prior["independent_source_replication_count"] = 50
    elif mutation == "selected":
        prior["role_screen"]["proposed_base_seeds"] = [7]
    elif mutation == "precision_repeat":
        inputs["precision_repeat"]["vectors"].pop()
    elif mutation in ("last_vector", "last_precision", "late_null", "precision_count", "work"):
        p = inputs["precision"]
        if mutation == "last_vector":
            p["vectors"].pop()
        elif mutation == "last_precision":
            p["vectors"][-1]["precision"]["bounded_mean_half_width"] = 0.01
        elif mutation == "late_null":
            p["vectors"][-1]["precision"]["pilot"]["pilot_summary"]["observations"][-1]["value"] = (
                None
            )
        elif mutation == "precision_count":
            p["conditional_normal_sensitivity_unresolved_count"] += 1
        else:
            p["work_budget"]["families"][-1]["maximum_updates_per_replication"] += 1
        inputs["precision_repeat"] = deepcopy(p)
    elif mutation == "late_causal":
        chronology["causal_nodes"][-1]["ordinal"] = True
    elif mutation == "late_causal_pointer":
        chronology["causal_nodes"][-1]["input_pointer"] = "/result/training_before_release"
    elif mutation == "late_causal_hash":
        chronology["causal_nodes"][-1]["whole_record_sha256"] = "bad"
    elif mutation == "omit_causal":
        chronology["causal_nodes"].pop()
    elif mutation == "extra_causal":
        chronology["causal_nodes"][-1]["fresh"] = True
    elif mutation == "raw_pointer":
        raw["prior_ledger_pointer"] = "/content_ledger/4471"
    elif mutation == "raw_alias":
        raw["all_original_aliases"] = []
    elif mutation == "raw_hash":
        raw["whole_saved_witness_record_sha256"] = "c" * 64
    elif mutation == "raw_identity":
        raw["identity"]["byte_count"] += 1
    elif mutation == "dispatch":
        witness["actual_unchanged_complete_reader_dispatches"]["complete_training_readers"][
            "repeat"
        ] = 0
    elif mutation == "admit":
        witness["fresh_roles_authorized"] = True
    elif mutation == "utc":
        chronology["exact_actual_release_utc"] = "2026-10-05"
    elif mutation == "independent":
        chronology["independent_source_replications"] = 50
    elif mutation == "reader":
        witness["readback"]["complete_original_reader_verified"] = False
    elif mutation == "whole_identity":
        identities["precision"]["byte_count"] = True
    else:
        inputs.pop("precision_repeat")
    with pytest.raises(ValueError):
        build_pending_confirmation_contract(inputs, identities)
