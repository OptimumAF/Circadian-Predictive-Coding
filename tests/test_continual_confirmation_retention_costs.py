"""Retained-memory costs use original unscored development proofs only."""

from copy import deepcopy
from typing import Any

import pytest

import test_continual_confirmation_resources as resource_tests
import test_continual_confirmation_work_validation as work_tests
from src.app.continual_confirmation_json import canonical_dataclass, canonical_mapping, state_digest
from src.app.continual_confirmation_resources import _derive_inventory
from src.app.continual_confirmation_retention_costs import (
    _derive_retention_costs,
    project_retention_costs,
)

development_facts = work_tests.fixtures


def development_inputs(fixture: tuple[Any, Any]) -> tuple[Any, Any, Any]:
    families, facts = fixture
    _, projection = resource_tests._projection(fixture)
    return (
        [deepcopy(facts[family.name]) for family in families],
        families,
        _derive_inventory(projection, families, []),
    )


def resolve_pointer(body: Any, pointer: str) -> Any:
    for token in pointer.split("/")[1:]:
        body = body[int(token)] if type(body) is list else body[token]
    return body


def test_should_recover_every_stage_memory_proof_without_execution(
    development_facts: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, families, inventory = development_inputs(development_facts)
    before = deepcopy(rows)
    work_tests._seal_validation(monkeypatch)
    result = _derive_retention_costs(rows, families, inventory)
    assert result["coverage"]["cells"] == 56
    assert result["coverage"]["checkpoints"] == 168
    assert result["coverage"]["contexts"] == 6
    assert result["stage_totals"]["initial"]["owned_array_bytes"] == 0
    assert result["stage_totals"]["after_a"]["owned_array_bytes"] == 3840
    assert result["stage_totals"]["after_b"] == {
        "owned_array_bytes": 3840,
        "shared_fifo_array_bytes": 768,
        "owned_plus_shared_array_bytes_before_copies": 4608,
    }
    assert result["work"] == inventory["work"]
    for row in result["rows"]:
        for stage, checkpoint in row["checkpoints"].items():
            original = resolve_pointer(
                {"seed_results": rows}, checkpoint["checkpoint_json_pointer"]
            )
            assert checkpoint["retention"] == original["retention"]
            assert checkpoint["state_sha256"] == original["state_sha256"]
            if checkpoint["retention"] is None:
                assert checkpoint["owned_array_bytes"] == 0
                assert checkpoint["retention_status"] in {"not_applicable", "disabled"}
            else:
                assert checkpoint["owned_array_bytes"] == checkpoint["retention"]["retained_bytes"]
            for proof in checkpoint["owned_state_fields"].values():
                assert (
                    resolve_pointer({"seed_results": rows}, proof["json_pointer"]) == proof["value"]
                )
            assert checkpoint["owned_array_bytes"] == sum(
                pair["input_array_bytes"] + pair["target_array_bytes"]
                for pair in checkpoint["owned_array_fingerprints"]
            )
    assert result == _derive_retention_costs(rows, families, inventory)
    assert rows == before


def test_should_resolve_missing_projection_fields_and_preserve_original_nulls(
    development_facts: tuple[Any, Any],
) -> None:
    rows, families, inventory = development_inputs(development_facts)
    result = _derive_retention_costs(rows, families, inventory)
    missing = [
        row
        for row in inventory["rows"]
        if row["fields"]["owned_replay_array_bytes"]["value"] is None
    ]
    resolved = [row for row in result["rows"] if row["original_inventory_field"]["value"] is None]
    assert len(missing) == len(resolved) == result["coverage"]["projection_gaps_resolved"] == 30
    assert all(row["original_inventory_field"]["status"] == "unmeasured" for row in resolved)
    assert all(row["checkpoints"]["after_b"]["owned_array_bytes"] >= 0 for row in resolved)
    assert result["original_P6_10_acceptance_complete"] is False
    assert result["scope"]["checkpoint_copy_bytes_measured"] is False
    assert result["scope"]["per_arm_RSS_measured"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_seed",
        "wrong_seed",
        "missing_arm",
        "missing_stage",
        "missing_retention",
        "retention_count",
        "retention_bytes",
        "retention_id",
        "array_shape",
        "state_hash",
        "role_hash",
        "null_view_with_owned_memory",
        "inventory_scope",
        "inventory_owned",
        "inventory_group",
        "inventory_work",
    ],
)
def test_should_refuse_incomplete_or_conflicting_retention_proof(
    development_facts: tuple[Any, Any],
    corruption: str,
) -> None:
    rows, families, inventory = development_inputs(development_facts)
    replay = next(row for row in rows if row["family"] == "replay")
    arm = next(name for name, point in replay["after_b"].items() if point["retention"] is not None)
    point = replay["after_b"][arm]
    if corruption == "missing_seed":
        rows.pop()
    elif corruption == "wrong_seed":
        rows[-1]["seed"] = True
    elif corruption == "missing_arm":
        rows[-1]["after_b"].pop(next(iter(rows[-1]["after_b"])))
    elif corruption == "missing_stage":
        rows[-1].pop("after_a")
    elif corruption == "missing_retention":
        point.pop("retention")
    elif corruption in {"retention_count", "retention_bytes", "retention_id"}:
        if corruption == "retention_count":
            point["retention"]["example_count"] = True
        elif corruption == "retention_bytes":
            point["retention"]["retained_bytes"] += 24
        else:
            point["retention"]["sample_ids"][0] = "0" * 64
    elif corruption in {"array_shape", "null_view_with_owned_memory"}:
        snapshot = canonical_dataclass(
            point["state"], "src.core.circadian_predictive_coding.CircadianNetworkSnapshot"
        )
        owner = canonical_mapping(snapshot["state"])
        if corruption == "array_shape":
            item = owner["_replay_memory"]["deque"][0]
            item["fields"]["input_batch"]["shape"] = [2, 1]
            point["state_sha256"] = state_digest(point["state"])
        else:
            point["retention"] = None
    elif corruption == "state_hash":
        point["state_sha256"] = "0" * 64
    elif corruption == "role_hash":
        rows[-1]["roles"][0]["hashes"]["train"] = "0" * 64
    elif corruption == "inventory_scope":
        inventory["rows"].pop()
    elif corruption == "inventory_owned":
        measured = next(
            row
            for row in inventory["rows"]
            if row["fields"]["owned_replay_array_bytes"]["value"] is not None
        )
        measured["fields"]["owned_replay_array_bytes"]["value"] += 1
    elif corruption == "inventory_group":
        inventory["contexts"][-1]["owned_retained_array_bytes_before_copies"] += 1
    else:
        inventory["work"]["totals"]["retained_array_bytes_before_copies"] += 24
    with pytest.raises(ValueError):
        _derive_retention_costs(rows, families, inventory)


@pytest.mark.parametrize("body", [{}, {"seed_results": []}, {"nonfinite": float("nan")}])
def test_should_refuse_partial_or_unbound_public_training_result(body: Any) -> None:
    with pytest.raises(ValueError):
        project_retention_costs(body, {})
