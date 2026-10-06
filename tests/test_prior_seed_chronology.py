from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from src.app.prior_seed_chronology import build_seed_chronology
from src.core.seed_declaration_context import audit_seed_metadata
from src.core.seed_source_evidence import inspect_seed_csv, inspect_seed_text
from src.core.seed_stream_screening import EvidenceIdentity
from seed_chronology_fixtures import saved_seed_corpus


IDENTITY = EvidenceIdentity(123, "c" * 64)


def test_should_exclude_known_quantities_and_null_policies_but_keep_inventory_and_hash_unknowns() -> (
    None
):
    report = saved_seed_corpus(
        {
            "seed": 7,
            "bounded_mean_required_seeds": 99,
            "retention": {"name": "recent_fifo", "seed": None},
            "python_hash_seed": None,
            "old": {"field": "seed", "pointer": "/seed", "representation": "integer", "seed": 11},
        }
    )
    result = build_seed_chronology(report, IDENTITY, proposed_base_seeds=(99,))
    counts = {row["value"]: row["occurrences"] for row in result["numeric_candidate_occurrences"]}
    assert counts == {7: 1, 11: 1}
    assert result["meaning_counts"]["quantity"] == 1
    assert result["meaning_counts"]["nonrandom_retention_policy"] == 1
    assert result["uncertainty_counts"]["unrecorded_process_hash_seed"] == 1
    assert result["verified_execution_events"] == result["verified_role_release_events"] == 0
    assert not result["fresh_roles_authorized"]


def test_should_keep_json_csv_jsonl_and_seed_map_occurrences_with_all_origins() -> None:
    report = saved_seed_corpus({"seed_map": {"7": {"source": 201, "model": 1008}}})
    content = report["contents"][0]
    content["csv"] = json.loads(json.dumps(inspect_seed_csv("seed,seed\n11,13\n")))
    content["jsonl"] = [
        {
            "line": 1,
            "parse_status": "parsed",
            "metadata": json.loads(json.dumps(asdict(audit_seed_metadata({"seed": 17})))),
            "parse_error": None,
        },
        {"line": 2, "parse_status": "unparsed", "metadata": None, "parse_error": "invalid JSON"},
    ]
    result = build_seed_chronology(report, IDENTITY, proposed_base_seeds=(100,))
    counts = {row["value"]: row["occurrences"] for row in result["numeric_candidate_occurrences"]}
    assert set(counts) == {7, 11, 13, 17, 201, 1008}
    assert result["coverage"]["metadata_views"] == 3
    assert result["coverage"]["mapped_streams"] == 3
    assert result["coverage"]["alias_occurrences"] == 2
    assert result["uncertainty_counts"]["unparsed_jsonl"] == 1
    assert result["content_ledger"][0]["aliases"] == content["aliases"]
    assert result["role_screen"]["collisions"][0]["stream"]["name"] == "phase_b_source"


def test_should_keep_dynamic_calls_opaque_and_unparsed_chronology_unknown() -> None:
    report = saved_seed_corpus({"seed": "chosen_dynamically"})
    content = report["contents"][0]
    content["text"] = inspect_seed_text(
        "from forbidden import maker\nr = maker(7)\n", python_source=True
    )
    content["issues"] = [{"kind": "opaque_asset_or_checkpoint", "reason": "never unpickle"}]
    result = build_seed_chronology(report, IDENTITY)
    assert result["coverage"]["python_expressions"] == 2
    assert result["uncertainty_counts"]["source_expression_execution_unverified"] == 2
    assert result["uncertainty_counts"]["opaque_asset_or_checkpoint"] == 1
    assert result["uncertainty_counts"]["unresolved"] == 1
    assert not result["numeric_candidate_occurrences"]
    assert result["content_ledger"][0]["execution_and_release_status"] == "unverified"
    assert not result["role_screen"]["proposed_base_seeds"]


def test_should_not_order_actual_release_from_commit_path_copy_order_or_ast_literals() -> None:
    report = saved_seed_corpus()
    result = build_seed_chronology(report, IDENTITY)
    assert result["verified_role_release_events"] == 0
    assert result["independent_source_replication_count"] is None
    assert not result["complete_prior_usage_acceptance"]
    assert not result["fresh_roles_authorized"]
    assert build_seed_chronology(deepcopy(report), IDENTITY) == result


def test_should_preserve_blank_duplicate_csv_headers_allowed_by_the_original_reader() -> None:
    report = saved_seed_corpus()
    report["contents"][0]["csv"] = json.loads(json.dumps(inspect_seed_csv(",seed,\n1,7,2\n")))
    result = build_seed_chronology(report, IDENTITY)
    assert result["numeric_candidate_occurrences"] == [{"value": 7, "occurrences": 2}]
    assert result["coverage"]["metadata_views"] == 2
    assert not result["fresh_roles_authorized"]


def test_should_preserve_one_based_jsonl_positions_including_blank_and_unparsed_rows() -> None:
    report = saved_seed_corpus()
    report["contents"][0]["jsonl"] = [
        {"line": 1, "parse_status": "blank", "metadata": None, "parse_error": None},
        {
            "line": 2,
            "parse_status": "parsed",
            "metadata": json.loads(json.dumps(asdict(audit_seed_metadata({"seed": 17})))),
            "parse_error": None,
        },
        {"line": 3, "parse_status": "unparsed", "metadata": None, "parse_error": "invalid JSON"},
    ]
    result = build_seed_chronology(report, IDENTITY)
    assert result["coverage"]["jsonl_lines"] == 3
    assert result["coverage"]["metadata_views"] == 2
    assert result["uncertainty_counts"]["unparsed_jsonl"] == 1
    assert result["numeric_candidate_occurrences"] == [
        {"value": 7, "occurrences": 1},
        {"value": 17, "occurrences": 1},
    ]
    assert not result["fresh_roles_authorized"]


@pytest.mark.parametrize("positions", [(0,), (-1,), (True,), (2,), (1, 1), (2, 1), (1, 3)])
def test_should_reject_invalid_duplicate_reordered_or_missing_jsonl_positions(
    positions: tuple[int, ...],
) -> None:
    report = saved_seed_corpus()
    report["contents"][0]["jsonl"] = [
        {"line": line, "parse_status": "blank", "metadata": None, "parse_error": None}
        for line in positions
    ]
    with pytest.raises(ValueError, match="seed"):
        build_seed_chronology(report, IDENTITY)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_meaning",
        "wrong_related_seed",
        "invalid_map",
        "forged_text_authority",
        "bad_column",
        "invalid_hash",
        "unknown_expression_kind",
        "bad_jsonl",
    ],
)
def test_should_reject_changed_or_unknown_seed_witness_semantics(mutation: str) -> None:
    report = deepcopy(saved_seed_corpus())
    content = report["contents"][0]
    if mutation == "unknown_meaning":
        content["json"]["declarations"][0]["meaning"] = "executed_seed"
    elif mutation == "wrong_related_seed":
        content["json"]["declarations"][0]["related_seed"] = True
    elif mutation == "invalid_map":
        content["json"]["mapped_streams"] = [
            {"pointer": "/map", "base_seed": True, "stream": "source", "value": 7}
        ]
    elif mutation == "forged_text_authority":
        content["text"]["execution_or_fresh_role_authority"] = True
    elif mutation == "bad_column":
        content["text"]["seed_lines"][0]["references"][0]["column"] = -1
    elif mutation == "invalid_hash":
        content["text"]["seed_lines"][0]["line_sha256"] = "q" * 64
    elif mutation == "unknown_expression_kind":
        content["text"]["python_expressions"][0]["kind"] = "released_final"
    else:
        content["jsonl"] = [
            {"line": 1, "parse_status": "parsed", "metadata": None, "parse_error": None}
        ]
    with pytest.raises(ValueError, match="seed"):
        build_seed_chronology(report, IDENTITY)
