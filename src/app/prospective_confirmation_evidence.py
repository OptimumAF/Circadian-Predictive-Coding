"""Bind every saved uncertainty and fixed precision vector to a pending design.

Inputs are the entire saved chronology, both complete original-reader witness
outputs, complete precision feasibility and declared whole input identities.
Outputs retain every ledger row/alias/unknown, all recorded links and every
precision vector, with conservative possible-use effects on future bindings.
This pure metadata consumer establishes no actual IO/current source proof,
historical reader reexecution, independence, fresh admission or resource repair.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

from src.app.continual_confirmation_json import same_json
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.continual_precision_feasibility import _work_budget
from src.app.continual_precision_contract import fixed_precision_contract
from src.core.seed_precision_budget import PrecisionBudgetSpec, plan_seed_precision_budget
from src.core.seed_release_chronology import recorded_release_order
from src.core.seed_statistics import SeedObservation
from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity


_INPUT_NAMES = {
    "chronology",
    "precision",
    "precision_repeat",
    "canonical_witness",
    "repeat_witness",
}
_LEDGER_FIELDS = {
    "aliases",
    "byte_count",
    "coverage",
    "execution_and_release_status",
    "input_pointer",
    "meaning_counts",
    "numeric_candidate_occurrences",
    "sha256",
    "uncertainty_counts",
    "whole_saved_witness_record_sha256",
}


def _identity(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"byte_count", "sha256"}:
        raise ValueError("prospective evidence requires every whole input identity")
    validate_evidence_identity(EvidenceIdentity(**value))
    return value


def _counts(value: Any) -> dict[str, int]:
    if type(value) is not dict or any(
        type(name) is not str or not name or type(count) is not int or count < 0
        for name, count in value.items()
    ):
        raise ValueError("prospective evidence requires exact named nonnegative counts")
    return value


def _occurrences(value: Any) -> Counter[int]:
    if type(value) is not list:
        raise ValueError("prospective evidence numeric witnesses must be ordered lists")
    result = Counter[int]()
    for row in value:
        if (
            type(row) is not dict
            or set(row) != {"value", "occurrences"}
            or type(row["value"]) is not int
            or type(row["occurrences"]) is not int
            or row["occurrences"] <= 0
            or row["value"] in result
        ):
            raise ValueError("prospective evidence numeric witness fields differ")
        result[row["value"]] = row["occurrences"]
    if list(result) != sorted(result):
        raise ValueError("prospective evidence numeric witness order differs")
    return result


def _prior_effects(prior: dict[str, Any]) -> list[dict[str, Any]]:
    if prior["schema_id"] != "conservative_saved_seed_chronology_v1":
        raise ValueError("prospective evidence requires the entire original chronology")
    ledger = prior["content_ledger"]
    if type(ledger) is not list or len(ledger) != 4471:
        raise ValueError("prospective evidence requires all4471 original saved contents")
    aliases, digests, uncertainty = 0, [], Counter[str]()
    meanings, numeric = Counter[str](), Counter[int]()
    result = []
    for index, row in enumerate(ledger):
        if type(row) is not dict or set(row) != _LEDGER_FIELDS:
            raise ValueError("prospective evidence complete ledger fields differ")
        _identity({"byte_count": row["byte_count"], "sha256": row["sha256"]})
        _identity({"byte_count": 0, "sha256": row["whole_saved_witness_record_sha256"]})
        if (
            row["input_pointer"] != f"/contents/{index}"
            or row["execution_and_release_status"] != "unverified"
        ):
            raise ValueError("prospective evidence original ledger order/status differs")
        if (
            type(row["aliases"]) is not list
            or not row["aliases"]
            or any(type(name) is not str or not name for name in row["aliases"])
            or len(set(row["aliases"])) != len(row["aliases"])
        ):
            raise ValueError("prospective evidence original aliases are missing/duplicated")
        aliases += len(row["aliases"])
        digests.append(row["sha256"])
        uncertainty.update(_counts(row["uncertainty_counts"]))
        meanings.update(_counts(row["meaning_counts"]))
        numeric.update(_occurrences(row["numeric_candidate_occurrences"]))
        _counts(row["coverage"])
        result.append(
            {
                "complete_original_ledger_row": deepcopy(row),
                "prior_ledger_pointer": f"/content_ledger/{index}",
                "possible_use_effect": "unknown_prior_generation_or_release_requires_proven_fresh_independent_source_provenance_before_any_admission",
                "numeric_noncollision_or_a_later_copy_clears_effect": False,
            }
        )
    if digests != sorted(set(digests)) or aliases != 13149:
        raise ValueError("prospective evidence whole content/alias membership differs")
    uncertainty["actual_content_execution_and_release_unverified"] = len(ledger)
    same_json(dict(uncertainty), prior["uncertainty_counts"], "every original unresolved effect")
    same_json(dict(meanings), prior["meaning_counts"], "every original meaning")
    same_json(
        dict(numeric),
        dict(_occurrences(prior["numeric_candidate_occurrences"])),
        "every original numeric witness",
    )
    if (
        prior["coverage"]["contents"] != len(ledger)
        or prior["coverage"]["alias_occurrences"] != aliases
    ):
        raise ValueError("prospective evidence complete coverage differs")
    for name in (
        "complete_prior_usage_acceptance",
        "fresh_roles_authorized",
        "original_p67_acceptance_complete",
        "prospective_real_role_seeds_selected",
    ):
        if prior[name] is not False:
            raise ValueError("prospective evidence cannot promote original prior acceptance")
    if (
        prior["independent_source_replication_count"] is not None
        or prior["verified_execution_events"] != 0
        or type(prior["verified_execution_events"]) is not int
        or prior["verified_role_release_events"] != 0
        or type(prior["verified_role_release_events"]) is not int
    ):
        raise ValueError("prospective evidence must retain unknown historical execution")
    same_json(
        prior["role_screen"]["proposed_base_seeds"], [], "no actual prior candidate selection"
    )
    return result


def _precision_vectors(precision: dict[str, Any], design: dict[str, Any]) -> None:
    if precision["schema_id"] != "p67_complete_prospective_precision_feasibility_v1":
        raise ValueError("prospective evidence precision schema differs")
    same_json(
        precision["contract"],
        design["original_precision_objective"],
        "whole unchanged precision objective",
    )
    vectors = precision["vectors"]
    expected = design["ordered_primary_statements"]
    if type(vectors) is not list or len(vectors) != len(expected) or len(vectors) != 116:
        raise ValueError("prospective evidence requires every116 original precision vector")
    objective, analysis = (
        design["original_precision_objective"],
        design["original_complete_analysis"],
    )
    for row, statement in zip(vectors, expected, strict=True):
        same_json(
            {key: row[key] for key in ("family", "left", "right")}
            | {"metric": row["primary_metric"]},
            statement,
            "ordered precision vector identity",
        )
        bounds = (
            objective["mean_accuracy_difference_range"]
            if row["primary_metric"] == "final_mean_task_accuracy"
            else objective["forgetting_difference_range"]
        )
        spec = PrecisionBudgetSpec(
            objective["target_half_width"],
            objective["mean_family_alpha"],
            objective["pilot_variance_family_alpha"],
            objective["statement_count"],
            bounds[0],
            bounds[1],
            objective["candidate_seed_count"],
            analysis["simultaneous_critical"],
            analysis["zero_deviation_tolerance"],
        )
        family = next(
            item
            for item in design["original_complete_manifest"]["families"]
            if item["name"] == row["family"]
        )
        observations = tuple(
            SeedObservation(item["seed"], item["value"], item["reason"])
            for item in row["precision"]["pilot"]["pilot_summary"]["observations"]
        )
        actual = plan_seed_precision_budget(observations, tuple(family["development_seeds"]), spec)
        same_json(
            json.loads(json.dumps(asdict(actual), allow_nan=False)),
            row["precision"],
            "entire original precision arithmetic and null policy",
        )
    budget = _work_budget(
        {"original_manifest": design["original_complete_manifest"]}, fixed_precision_contract()
    )
    same_json(precision["work_budget"], budget, "whole unchanged work and resource envelope")
    if precision["precision_objective_supported_by_all_bounds"] is not False:
        raise ValueError(
            "prospective evidence original negative/unresolved precision status differs"
        )
    for name, status in (
        ("conditional_normal_sensitivity_target_met_count", True),
        ("conditional_normal_sensitivity_unresolved_count", None),
    ):
        same_json(
            precision[name],
            sum(row["precision"]["conditional_normal_target_met"] is status for row in vectors),
            "all original precision status counts",
        )
    for name in (
        "new_confirmation_authorized",
        "original_p67_acceptance_complete",
        "untouched_role_seed_binding_complete",
    ):
        if precision[name] is not False:
            raise ValueError("prospective evidence cannot invent precision execution authority")


def _causal_nodes(chronology: dict[str, Any]) -> None:
    expected_nodes = recorded_release_order(120, 1680)
    nodes = chronology["causal_nodes"]
    if type(nodes) is not list or len(nodes) != len(expected_nodes):
        raise ValueError("prospective evidence requires every original causal node")
    for row, expected in zip(nodes, expected_nodes, strict=True):
        if type(row) is not dict or set(row) != {
            "ordinal",
            "kind",
            "record_index",
            "input_pointer",
            "whole_record_sha256",
        }:
            raise ValueError("prospective evidence causal fields differ")
        same_json(
            {key: row[key] for key in ("ordinal", "kind", "record_index")},
            asdict(expected),
            "original recorded causal positions",
        )
        if expected.kind.startswith("training_"):
            pointer = "/result/" + expected.kind
        else:
            field = (
                "source_events"
                if expected.kind.startswith("source_")
                else "release_events"
                if expected.kind == "role_release"
                else "prediction_events"
            )
            pointer = f"/audit/final_observation/{field}/{expected.record_index}"
        same_json(row["input_pointer"], pointer, "complete causal record pointer")
        _identity({"byte_count": 0, "sha256": row["whole_record_sha256"]})


def _raw_witness_links(witness: dict[str, Any], prior: dict[str, Any]) -> None:
    parts = witness["all_original_raw_scoring_parts_linked_to_complete_prior_ledger"]
    if type(parts) is not dict or set(parts) != {"request", "result", "audit"}:
        raise ValueError("prospective evidence requires all original raw artifact links")
    for name, link in parts.items():
        pointer = link["prior_ledger_pointer"]
        if type(pointer) is not str or not pointer.startswith("/content_ledger/"):
            raise ValueError("prospective evidence original raw witness pointer differs")
        index = int(pointer.removeprefix("/content_ledger/"))
        if (
            index < 0
            or index >= len(prior["content_ledger"])
            or pointer != f"/content_ledger/{index}"
        ):
            raise ValueError("prospective evidence original witness pointer outside complete scope")
        record = prior["content_ledger"][index]
        same_json(
            link["identity"],
            {"byte_count": record["byte_count"], "sha256": record["sha256"]},
            "whole original artifact identity",
        )
        same_json(
            link["identity"],
            witness["readback"]["file_identities"][name],
            "whole original readback part",
        )
        same_json(link["all_original_aliases"], record["aliases"], "all original witness aliases")
        if (
            link["path"] not in record["aliases"]
            or link["whole_saved_witness_record_sha256"]
            != record["whole_saved_witness_record_sha256"]
        ):
            raise ValueError("prospective evidence complete original witness link differs")


def _witness_links(
    witness: dict[str, Any], mode: str, prior: dict[str, Any], prior_identity: dict[str, Any]
) -> None:
    if witness["schema_id"] != "p67_source_bound_original_release_witness_readback_v1":
        raise ValueError("prospective evidence complete reader witness schema differs")
    if witness["original_bundle"] != mode:
        raise ValueError("prospective evidence original bundle identity/order differs")
    same_json(
        witness["complete_prior_ledger_identity"], prior_identity, "whole original ledger link"
    )
    same_json(
        witness["entire_prior_coverage_preserved"],
        prior["coverage"],
        "whole original coverage link",
    )
    same_json(
        witness["every_prior_unknown_preserved"],
        prior["uncertainty_counts"],
        "every original unknown link",
    )
    same_json(
        witness["entire_prior_input_identity"],
        prior["input_identity"],
        "entire saved raw input link",
    )
    chronology = witness["readback"]["chronology"]
    _causal_nodes(chronology)
    same_json(
        chronology["coverage"],
        {
            "family_seed_rows": 60,
            "cells": 560,
            "release_events": 120,
            "source_reads": 240,
            "prediction_events": 1680,
            "prediction_examples": 67200,
            "causal_nodes": 2043,
        },
        "entire historical witness scope",
    )
    same_json(
        witness["actual_unchanged_complete_reader_dispatches"],
        {"complete_scored_readers": 1, "complete_training_readers": {"canonical": 1, "repeat": 1}},
        "stored complete reader dispatch witnesses",
    )
    if (
        witness["readback"]["complete_original_reader_verified"] is not True
        or chronology["complete_original_reader_verified"] is not False
    ):
        raise ValueError("prospective evidence stored reader and pure declaration differ")
    for name in (
        "fresh_roles_authorized",
        "complete_prior_usage_acceptance",
        "original_p67_acceptance_complete",
    ):
        if witness[name] is not False or chronology[name] is not False:
            raise ValueError("prospective evidence cannot promote witness admission")
    same_json(witness["new_current_semantic_audits"], 0, "no witness semantic rerun")
    same_json(witness["new_scientific_dispatches"], 0, "no witness scientific execution")
    if (
        chronology["exact_actual_release_utc"] is not None
        or chronology["independent_source_replications"] is not None
        or chronology["cross_run_release_order_verified"] is not False
    ):
        raise ValueError("prospective evidence must preserve unrecorded chronology/independence")
    _raw_witness_links(witness, prior)


def build_pending_confirmation_contract(
    inputs: dict[str, dict[str, Any]], whole_input_identities: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Retain full supplied evidence without interpreting it as fresh-role proof."""
    if (
        type(inputs) is not dict
        or set(inputs) != _INPUT_NAMES
        or type(whole_input_identities) is not dict
        or set(whole_input_identities) != _INPUT_NAMES
    ):
        raise ValueError("prospective evidence requires every complete ordered input kind")
    for value in whole_input_identities.values():
        _identity(value)
    try:
        design = fixed_prospective_design()
        prior, precision = inputs["chronology"], inputs["precision"]
        effects = _prior_effects(prior)
        same_json(precision, inputs["precision_repeat"], "both entire original feasibility bodies")
        _precision_vectors(precision, design)
        for mode in ("canonical", "repeat"):
            _witness_links(
                inputs[mode + "_witness"], mode, prior, whole_input_identities["chronology"]
            )
        same_json(
            inputs["canonical_witness"]["readback"]["chronology"]["causal_nodes"],
            inputs["repeat_witness"]["readback"]["chronology"]["causal_nodes"],
            "whole repeated causal trace",
        )
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration) as error:
        raise ValueError("prospective evidence full input is malformed or incomplete") from error
    return {
        "schema_id": "p67_pending_complete_source_bound_prospective_design_v1",
        "validation_scope": "entire_supplied_saved_metadata_and_unchanged_precision_arithmetic_only",
        "prospective_design": design,
        "whole_declared_input_identities": deepcopy(whole_input_identities),
        "complete_original_chronology_header": deepcopy(
            {key: value for key, value in prior.items() if key != "content_ledger"}
        ),
        "every_original_prior_effect": effects,
        "entire_original_precision_feasibility": deepcopy(precision),
        "whole_original_reader_witness_inputs": deepcopy(
            {mode: inputs[mode + "_witness"] for mode in ("canonical", "repeat")}
        ),
        "actual_original_readers_reexecuted": False,
        "actual_current_source_proof_verified": False,
        "all_unresolved_prior_effects_cleared_for_fresh_roles": False,
        "independent_source_replications": None,
        "untouched_role_binding_complete": False,
        "actual_candidate_seeds_selected": False,
        "fresh_roles_authorized": False,
        "complete_prior_usage_acceptance": False,
        "original_p67_acceptance_complete": False,
    }
