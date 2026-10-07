"""Resource inventory fixtures are unscored development facts only."""

from copy import deepcopy
from typing import Any

import pytest

import test_continual_confirmation_work_validation as work_tests
from src.app.continual_confirmation_execution import work_summary
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_report_costs import _project_seed_costs
from src.app.continual_confirmation_resources import _derive_inventory, build_resource_inventory

development_facts = work_tests.fixtures


def _projection(fixtures: tuple[Any, Any]) -> tuple[Any, dict[str, Any]]:
    families, facts = fixtures
    projected = [_project_seed_costs(facts[family.name], family) for family in families]
    work = tuple(work_tests.work._verify_seed_work(facts[f.name], f) for f in families)
    return families, {
        "cells": [cell for item in projected for cell in item["cells"]],
        "seed_contexts": [item["seed_context"] for item in projected],
        "work": work_summary(work),
    }


def test_should_inventory_every_development_cost_and_checkpoint_after_sealing_execution(
    development_facts: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    families, projection = _projection(development_facts)
    work_tests._seal_validation(monkeypatch)
    original = deepcopy(projection)
    result = _derive_inventory(projection, families, [])
    assert result["coverage"]["cells"] == 56
    assert result["coverage"]["checkpoint_capacities"] == 168
    assert result["coverage"]["shared_contexts"] == 6
    assert result["work"] == projection["work"]
    assert result["rows"][0]["fields"]["wall_time_seconds"]["status"] == "unmeasured"
    assert all(row["fields"]["peak_process_rss_bytes"]["value"] is None for row in result["rows"])
    assert result == _derive_inventory(projection, families, [])
    assert projection == original


def test_should_keep_rejected_execution_and_distinct_shared_storage(
    development_facts: tuple[Any, Any],
) -> None:
    families, projection = _projection(development_facts)
    result = _derive_inventory(projection, families, [])
    for raw, row in zip(projection["cells"], result["rows"], strict=True):
        fields = row["fields"]
        assert fields["executed_optimizer_updates"]["value"] == (
            raw["wake_updates"]
            + raw["applied_replay_updates"]
            + raw["rejected_executed_replay_updates"]
        )
        assert fields["rejected_replay_updates"]["value"] == raw["rejected_executed_replay_updates"]
    for context, work in zip(result["contexts"], projection["work"]["by_seed"], strict=True):
        assert (
            context["retained_array_bytes_before_copies"]
            == work["retained_array_bytes_before_copies"]
        )
        assert (
            context["owned_retained_array_bytes_before_copies"] + context["shared_fifo_bytes"]
            == context["retained_array_bytes_before_copies"]
        )


@pytest.mark.parametrize(
    "change", ["missing_cell", "duplicate_context", "work", "checkpoint", "negative", "bool"]
)
def test_should_refuse_incomplete_or_inconsistent_resource_facts(
    development_facts: tuple[Any, Any], change: str
) -> None:
    families, projection = _projection(development_facts)
    if change == "missing_cell":
        projection["cells"].pop()
    elif change == "duplicate_context":
        projection["seed_contexts"][-1] = deepcopy(projection["seed_contexts"][0])
    elif change == "work":
        projection["work"]["totals"]["executed_optimizer_updates"] += 1
    elif change == "checkpoint":
        projection["cells"][0]["checkpoints"][-1]["parameter_count"] += 1
    else:
        projection["cells"][0]["wake_updates"] = -1 if change == "negative" else True
    with pytest.raises(ValueError):
        _derive_inventory(projection, families, [])


@pytest.mark.parametrize("inspection", [{}, {"unfinished": True}, {"nonfinite": float("nan")}])
def test_should_refuse_unbound_public_inspection(inspection: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_resource_inventory(inspection, {}, ())


def test_should_preserve_all_560_planned_metadata_rows_without_reserved_execution(
    development_facts: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, source = _projection(development_facts)
    families = fixed_confirmation_manifest().families
    cells: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    work_rows: list[dict[str, Any]] = []
    for family in families:
        original_context = next(c for c in source["seed_contexts"] if c["family"] == family.name)
        original_work = next(c for c in source["work"]["by_seed"] if c["family"] == family.name)
        original_cells = [c for c in source["cells"] if c["family"] == family.name]
        for seed in family.seeds:
            # Explicit metadata-only scope spy: shared development proof bodies
            # are not reserved-source facts or scientific reproduction.
            contexts.append({**original_context, "seed": seed})
            work_rows.append({**original_work, "seed": seed})
            cells.extend({**cell, "seed": seed} for cell in original_cells)
    work = {
        "by_seed": work_rows,
        "totals": {name: sum(row[name] for row in work_rows) for name in source["work"]["totals"]},
        "maximum_transient_width": source["work"]["maximum_transient_width"],
    }
    work_tests._seal_validation(monkeypatch)
    result = _derive_inventory(
        {"cells": cells, "seed_contexts": contexts, "work": work}, families, []
    )
    assert result["coverage"]["cells"] == 560
    assert result["coverage"]["checkpoint_capacities"] == 1680
    assert result["coverage"]["shared_contexts"] == 60
    assert [(row["family"], row["seed"], row["arm"]) for row in result["rows"]] == [
        (f.name, seed, arm) for f in families for seed in f.seeds for arm in f.arms
    ]
    assert result["original_P6_10_acceptance_complete"] is False


@pytest.mark.parametrize(
    "change", ["guard", "storage", "late_peak", "wake_loop_bool", "history_width"]
)
def test_should_refuse_changed_late_guard_storage_history_or_loop_units(
    development_facts: tuple[Any, Any], change: str
) -> None:
    families, projection = _projection(development_facts)
    if change == "guard":
        projection["seed_contexts"][2]["supplemental_guards"].pop()
    elif change == "storage":
        projection["work"]["by_seed"][-1]["retained_array_bytes_before_copies"] += 192
    elif change == "late_peak":
        projection["cells"][-1]["method_facts"]["parameters_peak"] += 4
    elif change == "wake_loop_bool":
        projection["cells"][-1]["method_facts"]["wake_inference_loops"] = True
    else:
        projection["cells"][-1]["checkpoints"][-1]["width"] = 0
    with pytest.raises(ValueError):
        _derive_inventory(projection, families, [])
