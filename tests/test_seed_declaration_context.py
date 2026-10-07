from dataclasses import asdict

import pytest

from src.core.seed_declaration_context import audit_seed_metadata
from src.core.seed_usage import collect_seed_usage


def test_should_preserve_every_original_declaration_and_issue_in_context() -> None:
    metadata = {
        "seed": 7,
        "seeds": [11, None, {"seed": 13}],
        "config": '{"seed":"17,19"}',
        "argv": ["tool", "--seed=23"],
    }
    original = collect_seed_usage(metadata)
    result = audit_seed_metadata(metadata)
    assert tuple(row.declaration for row in result.declarations) == original.declarations
    assert tuple(row.declaration for row in result.unresolved) == original.unresolved
    assert {row.meaning for row in result.declarations} == {"numeric_seed_candidate"}
    assert result.unresolved[0].meaning == "unresolved"
    assert result.unresolved[1].meaning == "seed_record"
    assert result.unresolved[1].related_seed == 13


@pytest.mark.parametrize("name", ["content_hash", "recent_fifo"])
def test_should_identify_declared_nonrandom_policy_without_inventing_a_seed(name: str) -> None:
    result = audit_seed_metadata({"policy": {"name": name, "seed": None}})
    assert not result.declarations
    assert result.unresolved[0].meaning == "nonrandom_retention_policy"
    assert result.unresolved[0].related_seed is None


@pytest.mark.parametrize("name", ["seeded_reservoir", "unknown"])
def test_should_keep_invalid_or_unknown_policy_seed_unresolved(name: str) -> None:
    result = audit_seed_metadata({"policy": {"name": name, "seed": None}})
    assert result.unresolved[0].meaning == "unresolved"


def test_should_distinguish_quantities_from_observed_seed_lists() -> None:
    result = audit_seed_metadata(
        {
            "bounded_mean_required_seeds": 6754,
            "distinct_source_seeds": 50,
            "persistent_labeled_array_bytes_per_seed": 960,
            "projected_feature_seconds_per_seed": 33.8,
            "observed_source_seeds": [101, 103],
        }
    )
    assert [row.meaning for row in result.declarations] == ["quantity"] * 3 + [
        "numeric_seed_candidate",
        "numeric_seed_candidate",
    ]
    assert result.unresolved[0].meaning == "quantity"


def test_should_keep_unrecorded_hash_seed_distinct_from_source_seed_null() -> None:
    result = audit_seed_metadata({"python_hash_seed": None, "seed": None})
    assert [row.meaning for row in result.unresolved] == [
        "unrecorded_process_hash_seed",
        "unresolved",
    ]


def test_should_resolve_virtual_and_escaped_locations_without_dropping_canonical_records() -> None:
    metadata = {
        "a/b~c": '{"policy":{"dict":[["name","recent_fifo"],["seed",null]]},"seeds":{"tuple":["7,11",13]}}'
    }
    result = audit_seed_metadata(metadata)
    assert [row.declaration.seed for row in result.declarations] == [7, 11, 13]  # type: ignore[union-attr]
    assert result.unresolved[0].meaning == "nonrandom_retention_policy"
    assert result.unresolved[0].declaration.pointer.startswith("/a~1b~0c/embedded_json/")


def test_should_recover_every_structured_base_and_stream_without_assuming_execution() -> None:
    metadata = {"seed_map": {"7": {"phase_a_source": 7, "phase_b_source": 108, "model_init": 1008}}}
    result = audit_seed_metadata(metadata)
    assert [(row.base_seed, row.stream, row.value) for row in result.mapped_streams] == [
        (7, "base_seed_key", 7),
        (7, "phase_a_source", 7),
        (7, "phase_b_source", 108),
        (7, "model_init", 1008),
    ]
    assert not result.map_issues


def test_should_keep_invalid_structured_seed_map_and_duplicate_canonical_keys_visible() -> None:
    metadata = {
        "seed_map": {"symbolic": {"source": 7}, "11": {"source": None}},
        "policy": {"dict": [["name", "recent_fifo"], ["name", "seeded_reservoir"], ["seed", None]]},
    }
    result = audit_seed_metadata(metadata)
    assert len(result.map_issues) == 2
    assert result.unresolved[0].meaning == "unresolved"


def test_should_retain_inventory_value_lineage_instead_of_treating_copies_as_experiments() -> None:
    result = audit_seed_metadata(
        {
            "declaration": {
                "field": "bounded_mean_required_seeds",
                "pointer": "/x",
                "representation": "integer",
                "seed": 6754,
            }
        }
    )
    assert result.declarations[0].meaning == "quantity"
    other = audit_seed_metadata(
        {"declaration": {"field": "seed", "pointer": "/x", "representation": "integer", "seed": 7}}
    )
    assert other.declarations[0].meaning == "recorded_seed_inventory_value"
    assert asdict(other.declarations[0])["declaration"]["seed"] == 7


def test_should_keep_empty_maps_and_duplicate_stream_names_unresolved() -> None:
    empty = audit_seed_metadata({"seed_map": {}})
    assert empty.map_issues[0].reason == "empty_seed_map"
    duplicate = audit_seed_metadata({"seed_map": {"7": {"dict": [["source", 7], ["source", 11]]}}})
    assert duplicate.map_issues[0].reason == "empty_or_duplicate_mapped_streams"
    assert [row.value for row in duplicate.mapped_streams] == [7, 7, 11]
