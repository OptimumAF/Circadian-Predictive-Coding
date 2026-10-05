"""Reader ports exercise complete dispatch; IO spies grant no authority."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.infra import continual_confirmation_retention_references as references


def fixture_read(root: Path, manifest: Any, reader: Any) -> dict[str, Any]:
    for name in ("canonical", "repeat"):
        reader(root / name)
    return {"bundles": [{}, {}]}


def test_should_project_both_complete_reader_bodies_and_compare_repetition(tmp_path: Path) -> None:
    calls = []
    parts = ({"request": 1}, {"whole_training": 2}, {"work": {"all": 3}})
    inventory = {"work": {"all": 3}}

    def reader(directory: Path) -> Any:
        calls.append(("reader", directory.name))
        return deepcopy(parts)

    def project(body: Any, actual_inventory: Any) -> Any:
        assert body == parts[1] and actual_inventory is inventory
        calls.append(("project", "whole"))
        return {"work": inventory["work"], "complete_rows": [1, 2, 3]}

    result = references._collect_retention_references(
        tmp_path,
        fixed_scoring_manifest(),
        reader,
        inventory,
        project,
        reference_reader=fixture_read,
    )
    assert calls == [
        ("reader", "canonical"),
        ("project", "whole"),
        ("reader", "repeat"),
        ("project", "whole"),
    ]
    assert result["projection"]["complete_rows"] == [1, 2, 3]
    assert result["deterministic_retention_repetition"] is True
    assert result["new_measurement_training_or_final_access"] is False


@pytest.mark.parametrize(
    "corruption",
    ["arity", "part_type", "audit_work", "repeated_projection", "one_reader", "three_readers"],
)
def test_should_refuse_incomplete_reader_scope_or_changed_complete_projection(
    tmp_path: Path, corruption: str
) -> None:
    count = 0

    def reader(directory: Path) -> Any:
        nonlocal count
        count += 1
        if corruption == "arity":
            return ({}, {})
        if corruption == "part_type":
            return ({}, [], {})
        return (
            {},
            {"version": count if corruption == "repeated_projection" else 1},
            {"work": {"n": 2 if corruption == "audit_work" else 1}},
        )

    def dispatch(root: Path, manifest: Any, supplied: Any) -> Any:
        amount = 1 if corruption == "one_reader" else 3 if corruption == "three_readers" else 2
        for index in range(amount):
            supplied(root / str(index))
        return {"bundles": [{}] * amount}

    with pytest.raises(ValueError):
        references._collect_retention_references(
            tmp_path,
            fixed_scoring_manifest(),
            reader,
            {},
            lambda body, inventory: {"work": {"n": 1}, "version": body["version"]},
            reference_reader=dispatch,
        )


def test_should_refuse_unbound_public_inventory_before_any_reader(tmp_path: Path) -> None:
    def forbid(directory: Path) -> Any:
        raise AssertionError("a partial inventory must not dispatch a reader")

    with pytest.raises(ValueError):
        references.read_retention_references(tmp_path, forbid, {})
