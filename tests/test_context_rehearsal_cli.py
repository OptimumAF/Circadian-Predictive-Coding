"""Actual parser/output and pure scenario preview; delegates never train here."""

from copy import deepcopy
from dataclasses import replace, asdict
import json
from typing import Any

import pytest

from scripts import context_head_live as live
from scripts import context_stream_comparison as study
from scripts import run_bp_ticket_serving as cli
from scripts.context_training_streams import (
    CONTEXT_REHEARSAL_RECIPES,
    CONTEXT_STREAM_RECIPES,
    CONTEXT_TRAINING_RECIPES,
)


@pytest.mark.parametrize("recipe", CONTEXT_STREAM_RECIPES)
def test_should_dispatch_exact_named_recipe_through_original_live_cli(
    tmp_path, monkeypatch, recipe
):
    calls = []

    def run(path, **options):
        calls.append((path, options))
        return dict(complete=True, scripted=True)

    monkeypatch.setattr(live, "run_case", run)
    output = tmp_path / "run.json"
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(output),
                "--context-head-serving-live",
                "--context-stream-recipe",
                recipe.recipe_id,
            ]
        )
        == 0
    )
    assert len(calls) == 1 and str(calls[0][0]) == "unused"
    assert calls[0][1] == dict(stream_recipe=recipe) and calls[0][1]["stream_recipe"] is recipe
    assert json.loads(output.read_text()) == dict(complete=True, scripted=True)


@pytest.mark.parametrize("mode", ["no_learning", "drift", "budget_aware"])
def test_should_forward_explicit_reference_without_changing_recipe(tmp_path, monkeypatch, mode):
    seen = []

    def run(path, **kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", run)
    recipe = CONTEXT_REHEARSAL_RECIPES[1]
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(tmp_path / "run.json"),
                "--context-head-serving-live",
                "--context-stream-recipe",
                recipe.recipe_id,
                "--context-live-reference",
                mode,
            ]
        )
        == 0
    )
    assert seen == [dict(reference_mode=mode, stream_recipe=recipe)]


@pytest.mark.parametrize("recipe", CONTEXT_REHEARSAL_RECIPES)
def test_should_preview_all_twenty_four_exact_scenarios_and_causal_rows_before_source_io(
    tmp_path, monkeypatch, recipe
):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    monkeypatch.setattr(study, "_source_bytes", lambda *a: pytest.fail("comparison source opened"))
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: pytest.fail("runtime entered"))
    output = tmp_path / "preview.json"
    assert (
        cli.main(
            [
                "--source",
                "does-not-exist",
                "--output",
                str(output),
                "--context-stream-protocol",
                "--context-stream-recipe",
                recipe.recipe_id,
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    tasks = [
        ("v1", "invoice"),
        ("v1", "legacy"),
        ("v1", "password"),
        ("v2", "invoice"),
        ("v2", "parcel"),
        ("v1", "legacy"),
        ("v1", "invoice"),
        ("v1", "invoice"),
        ("v1", "password"),
        ("v2", "invoice"),
        ("v2", "parcel"),
        ("v1", "legacy"),
        ("v2", "legacy"),
        ("v1", "legacy"),
        ("v1", "invoice"),
        ("v2", "invoice"),
        ("v1", "password"),
        ("v1", "invoice"),
        ("v1", "password"),
        ("v2", "invoice"),
        ("v2", "parcel"),
        ("v1", "legacy"),
        ("v2", "legacy"),
        ("v1", "legacy"),
    ]
    rows = result["opportunities"]
    assert result["definition"] == "context_stream_recipe_protocol_v1" and result["complete"]
    assert [(r["policy_version"], r["concept"]) for r in rows] == tasks
    assert [r["tick"] for r in rows] == list(range(24))
    assert [r["tick"] for r in rows if r["scenario"] == "stale_once"] == [4, 12, 20]
    assert [r["tick"] for r in rows if r["scenario"] == "revoke_commit"] == [6, 14, 22]
    assert sum(r["scenario"] == "ordinary" for r in rows) == 18
    assert [r["tick"] for r in rows if r["contention"]] == [2]
    assert all(r["public_context"] for r in rows)
    assert result["recipe"]["recipe_id"] == recipe.recipe_id
    assert [r["context"] for r in result["training_schedule"]] == list(recipe.contexts)
    assert [r["observed_at"] for r in result["training_schedule"]] == [0, 1, 2]
    assert [r["label_arrival_at"] for r in result["training_schedule"]] == [1, 2, 3]
    assert [r["rows"] for r in result["training_schedule"]] == [9, 9, 6]
    joint = recipe is CONTEXT_REHEARSAL_RECIPES[1]
    assert result["training_exposures"] == dict(
        updates=3, rows=24, new_rows=6 if joint else 18, old_rows=18 if joint else 6
    )
    assert result["wake_budget"] == dict(
        max_training_updates=3, max_wall_seconds=60, max_process_rss_bytes=512 * 1024**2
    )
    assert result["reference_policy"]["mode"] == "periodic"
    assert "declaration only" in result["qualification"]


@pytest.mark.parametrize("recipe", CONTEXT_TRAINING_RECIPES)
def test_should_preview_original_recipes_with_original_delay_exposures(recipe):
    result = study.protocol_snapshot(recipe=recipe)
    assert result["recipe"] == asdict(recipe)
    assert result["training_exposures"] == dict(updates=6, rows=27, new_rows=9, old_rows=18)
    assert [r["label_arrival_at"] for r in result["training_schedule"]] == list(range(2, 8))
    assert result["wake_budget"]["max_training_updates"] == 6


def test_should_detach_preview_without_granting_or_mutating_recipe_authority():
    recipe = CONTEXT_REHEARSAL_RECIPES[0]
    pin = deepcopy(recipe)
    result = study.protocol_snapshot(recipe=recipe)
    original = deepcopy(result)
    result["recipe"]["contexts"] = ("final_test",)
    result["opportunities"][22]["scenario"] = "ordinary"
    result["training_schedule"][0]["rows"] = 999
    result["wake_budget"]["max_training_updates"] = 99
    assert recipe == pin and study.protocol_snapshot(recipe=recipe) == original


@pytest.mark.parametrize("fault", ["type", "unknown_name", "other_delay", "withheld", "boolean"])
def test_should_refuse_foreign_selected_preview_without_source_or_worker(monkeypatch, fault):
    monkeypatch.setattr(live, "recipe_plan", lambda: pytest.fail("plan entered before validation"))
    recipe: Any = CONTEXT_REHEARSAL_RECIPES[0]
    if fault == "type":
        recipe = asdict(recipe)
    elif fault == "unknown_name":
        recipe = replace(recipe, recipe_id="unknown")
    elif fault == "other_delay":
        recipe = replace(recipe, label_delay=2)
    with pytest.raises(ValueError):
        study.protocol_snapshot(
            recipe=recipe,
            withheld=True if fault == "withheld" else "yes" if fault == "boolean" else False,
        )


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--context-stream-protocol", "--context-withheld-orders"],
        ["--context-stream-protocol", "--context-live-reference", "drift"],
        ["--context-stream-protocol", "--context-candidate-feedback"],
        ["--context-stream-protocol", "--context-task-feedback"],
        ["--context-stream-protocol", "--context-arrived-shift"],
        ["--context-stream-protocol", "--context-acknowledge-correct"],
        ["--context-stream-protocol", "--context-head-serving-live"],
        ["--context-recipe-validation"],
        ["--context-stream-comparison"],
        ["--context-head-serving"],
        ["--context-stream-protocol", "--catalog-fit"],
    ],
)
def test_should_refuse_conflicting_selector_modes_before_output_source_or_runtime(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: pytest.fail("runtime entered"))
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source entered"))
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(output),
                "--context-stream-recipe",
                CONTEXT_REHEARSAL_RECIPES[0].recipe_id,
                *options,
            ]
        )
    assert not output.exists()


def test_should_refuse_unknown_name_and_example_source_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: pytest.fail("runtime entered"))
    output = tmp_path / "out.json"
    for source, recipe in (
        (["--source", "unused"], "unknown"),
        (["--example-context-source"], CONTEXT_REHEARSAL_RECIPES[0].recipe_id),
    ):
        with pytest.raises(SystemExit):
            cli.main(
                [
                    *source,
                    "--output",
                    str(output),
                    "--context-stream-protocol",
                    "--context-stream-recipe",
                    recipe,
                ]
            )
        assert not output.exists()


def test_should_preserve_failed_delegated_case_costs_and_exit_status(tmp_path, monkeypatch):
    report = dict(
        complete=False,
        error="known failed training return",
        training_work=dict(attempted_updates=1, attempted_rows=9, returned_updates=0),
    )
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: report)
    output = tmp_path / "failed.json"
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(output),
                "--context-head-serving-live",
                "--context-stream-recipe",
                CONTEXT_REHEARSAL_RECIPES[1].recipe_id,
            ]
        )
        == 1
    )
    assert json.loads(output.read_text()) == report


def test_should_protect_existing_output_before_delegating_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: pytest.fail("runtime entered"))
    output = tmp_path / "existing.json"
    output.write_text("original user data")
    with pytest.raises(FileExistsError):
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(output),
                "--context-head-serving-live",
                "--context-stream-recipe",
                CONTEXT_REHEARSAL_RECIPES[0].recipe_id,
            ]
        )
    assert output.read_text() == "original user data"


def test_should_preserve_original_default_live_dispatch_without_recipe(tmp_path, monkeypatch):
    seen = []

    def run(path, **kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", run)
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(tmp_path / "default.json"),
                "--context-head-serving-live",
            ]
        )
        == 0
    )
    assert seen == [{}]


def test_should_preserve_unselected_withheld_protocol_cli(tmp_path, monkeypatch):
    from scripts.context_training_streams import WITHHELD_CONTEXT_RECIPES

    monkeypatch.setattr(study, "_source_bytes", lambda *a: pytest.fail("source opened"))
    output = tmp_path / "withheld.json"
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(output),
                "--context-stream-protocol",
                "--context-withheld-orders",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["definition"] == "withheld_context_orders_protocol_v1"
    assert result["case_count"] == 14 and len(result["opportunities"]) == 24
    assert [r["recipe_id"] for r in result["recipes"]] == [
        r.recipe_id for r in WITHHELD_CONTEXT_RECIPES
    ]
    assert "recipe" not in result and "training_schedule" not in result
