"""Normal-command delivery and bounded output, with scalar delegates only."""

from copy import deepcopy
from dataclasses import asdict
import json
import runpy
import sys

import pytest

from scripts import context_head_live as live
from scripts import context_stream_comparison as preview
from scripts import run_bp_ticket_serving as cli
from scripts.context_training_streams import (
    CONTEXT_PAIRED_REHEARSAL_RECIPES,
    CONTEXT_REHEARSAL_RECIPES,
)


FLAG = "--context-publish-after-learning"
LIMIT = 3 * 1024**2


def args(output, recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[1], *, describe=False):
    return [
        "--source",
        "unused",
        "--output",
        str(output),
        "--context-stream-protocol" if describe else "--context-head-serving-live",
        "--context-stream-recipe",
        recipe.recipe_id,
        FLAG,
    ]


@pytest.mark.parametrize("recipe", CONTEXT_PAIRED_REHEARSAL_RECIPES)
def test_should_dispatch_exact_recipe_and_boundary_optin_through_normal_command(
    tmp_path, monkeypatch, recipe
):
    seen = []

    def run(source, **options):
        seen.append((str(source), options))
        return dict(complete=True, fixture="scalar delegate only")

    monkeypatch.setattr(live, "run_case", run)
    output = tmp_path / "boundary.json"
    assert cli.main(args(output, recipe)) == 0
    assert seen == [("unused", dict(stream_recipe=recipe, publish_after_learning=True))]
    assert seen[0][1]["stream_recipe"] is recipe
    assert json.loads(output.read_text()) == dict(complete=True, fixture="scalar delegate only")


@pytest.mark.parametrize("recipe", CONTEXT_PAIRED_REHEARSAL_RECIPES)
def test_should_preview_actual_causal_return_without_source_teacher_or_runtime(
    tmp_path, monkeypatch, recipe
):
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: pytest.fail("runtime entered"))
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source read"))
    output = tmp_path / "declaration.json"
    assert cli.main(args(output, recipe, describe=True)) == 0
    result = json.loads(output.read_text())
    assert result["definition"] == "context_learning_boundary_protocol_v1"
    assert result["learning_boundary_plan"] == dict(first_batches=4, start_tick=12)
    assert [r["observed_at"] for r in result["training_schedule"]] == [0, 1, 2, 3, 12, 13]
    assert [r["label_arrival_at"] for r in result["training_schedule"]] == [1, 2, 3, 4, 13, 14]
    assert [r["rows"] for r in result["training_schedule"]] == [9, 9, 9, 9, 6, 6]
    original = preview.protocol_snapshot(recipe=recipe)
    assert result["plan"] == json.loads(json.dumps(original["plan"]))
    assert result["opportunities"] == json.loads(json.dumps(original["opportunities"]))
    assert result["training_exposures"] == original["training_exposures"]
    assert result["wake_budget"] == dict(
        max_training_updates=6, max_wall_seconds=60, max_process_rss_bytes=512 * 1024**2
    )
    assert result["publication_policy"]["maximum_guard_attempts"] == 2
    assert result["publication_policy"]["maximum_accepted_publications"] == 1
    assert result["readout_schedule"]["maximum_episodes"] == 12
    assert len(result["readout_schedule"]["tasks"]) == 6


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--context-head-serving-live"],
        [
            "--context-head-serving-live",
            "--context-stream-recipe",
            CONTEXT_REHEARSAL_RECIPES[1].recipe_id,
        ],
        ["--context-stream-protocol", "--context-withheld-orders"],
        [
            "--context-stream-protocol",
            "--context-stream-recipe",
            CONTEXT_REHEARSAL_RECIPES[0].recipe_id,
        ],
        [
            "--context-head-serving-live",
            "--context-stream-recipe",
            CONTEXT_PAIRED_REHEARSAL_RECIPES[0].recipe_id,
            "--context-live-reference",
            "no_learning",
        ],
        [
            "--context-head-serving-live",
            "--context-stream-recipe",
            CONTEXT_PAIRED_REHEARSAL_RECIPES[0].recipe_id,
            "--context-live-reference",
            "drift",
            "--context-candidate-feedback",
        ],
    ],
)
def test_should_refuse_invalid_combinations_before_output_or_delegation(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: pytest.fail("runtime entered"))
    monkeypatch.setattr(preview, "protocol_snapshot", lambda **kw: pytest.fail("preview entered"))
    output = tmp_path / "invalid.json"
    with pytest.raises(SystemExit):
        cli.main(["--source", "unused", "--output", str(output), FLAG, *options])
    assert not output.exists()


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_should_validate_preview_boolean_and_named_recipe_before_plan(monkeypatch, value):
    monkeypatch.setattr(live, "recipe_plan", lambda **kw: pytest.fail("plan entered"))
    with pytest.raises(ValueError):
        preview.protocol_snapshot(
            recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[0], publish_after_learning=value
        )
    with pytest.raises(ValueError):
        preview.protocol_snapshot(publish_after_learning=True)


def test_should_detach_boundary_preview_and_preserve_default_preview():
    recipe = CONTEXT_PAIRED_REHEARSAL_RECIPES[1]
    original = preview.protocol_snapshot(recipe=recipe)
    changed = preview.protocol_snapshot(recipe=recipe, publish_after_learning=True)
    pin = deepcopy(changed)
    changed["training_schedule"][4]["observed_at"] = 0
    changed["learning_boundary_plan"]["start_tick"] = 0
    changed["publication_policy"]["maximum_guard_attempts"] = 99
    assert preview.protocol_snapshot(recipe=recipe, publish_after_learning=True) == pin
    assert preview.protocol_snapshot(recipe=recipe) == original
    assert original["definition"] == "context_stream_recipe_protocol_v1"
    assert [r["observed_at"] for r in original["training_schedule"]] == list(range(6))
    assert original["recipe"] == asdict(recipe)


def test_should_accept_explicit_periodic_and_execute_module_preview(tmp_path, monkeypatch):
    seen = []

    def run(source, **kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", run)
    output = tmp_path / "periodic.json"
    assert cli.main([*args(output), "--context-live-reference", "periodic"]) == 0
    assert seen == [
        dict(
            stream_recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[1],
            publish_after_learning=True,
            reference_mode="periodic",
        )
    ]
    output = tmp_path / "module-preview.json"
    monkeypatch.setattr(
        sys, "argv", ["scripts.run_bp_ticket_serving", *args(output, describe=True)]
    )
    monkeypatch.delitem(sys.modules, "scripts.run_bp_ticket_serving")
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_module("scripts.run_bp_ticket_serving", run_name="__main__")
    assert exit_status.value.code == 0
    assert json.loads(output.read_text())["learning_boundary_plan"] == dict(
        first_batches=4, start_tick=12
    )


def report():
    return dict(
        complete=False,
        error="lost original native return",
        training_work=dict(attempted_updates=5, attempted_rows=42, returned_updates=4),
        budget=dict(
            updates=4,
            started_at=12.0,
            last_clock=13.0,
            limits=dict(
                max_training_updates=6,
                max_wall_seconds=60,
                max_process_rss_bytes=512 * 1024**2,
                max_replay_examples=None,
                max_hidden_width=None,
            ),
        ),
        memory=dict(start_bytes=1024, peak_bytes=2048, sample_count=3, interval_seconds=0.01),
        elapsed_seconds=1.0,
    )


def encoded_size(value):
    return len((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


@pytest.mark.parametrize("extra", [0, 1])
def test_should_accept_exact_byte_boundary_or_preserve_failure_costs_on_overflow(
    tmp_path, monkeypatch, extra
):
    value = report()
    value.update(complete=True, padding="")
    value["padding"] = "x" * (LIMIT - encoded_size(value) + extra)
    assert encoded_size(value) == LIMIT + extra
    pin = deepcopy(value)
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: value)
    output = tmp_path / "large.json"
    assert cli.main(args(output)) == (0 if extra == 0 else 1)
    saved = json.loads(output.read_text())
    if extra == 0:
        assert output.stat().st_size == LIMIT and saved == value
    else:
        assert output.stat().st_size <= 32 * 1024
        assert not saved["complete"] and not saved["detailed_report_available"]
        assert not saved["publication_receipts_available"]
        assert saved["reported"]["training_work"] == value["training_work"]
        assert saved["reported"]["budget"] == value["budget"]
        assert saved["report_bytes"] == LIMIT + 1
    assert value == pin


@pytest.mark.parametrize("fault", ["nonfinite", "object"])
def test_should_save_original_clocks_and_known_costs_when_encoding_fails(
    tmp_path, monkeypatch, fault
):
    value = report()
    value["invalid"] = float("nan") if fault == "nonfinite" else object()
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: value)
    output = tmp_path / "encoding-failed.json"
    assert cli.main(args(output)) == 1
    saved = json.loads(output.read_text())
    assert saved["report_bytes"] is None
    assert saved["reported"]["training_work"] == value["training_work"]
    assert saved["reported"]["budget"] == value["budget"]
    assert saved["reported"]["memory"] == value["memory"]
    assert saved["reported"]["elapsed_seconds"] == 1.0
    assert not saved["publication_receipts_available"]


def test_should_preserve_runtime_failure_and_prevent_overwriting_existing_output(
    tmp_path, monkeypatch
):
    value = report()
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: value)
    output = tmp_path / "known-failure.json"
    assert cli.main(args(output)) == 1
    assert json.loads(output.read_text()) == value
    raw = output.read_bytes()
    monkeypatch.setattr(live, "run_case", lambda *a, **kw: pytest.fail("runtime entered"))
    with pytest.raises(FileExistsError):
        cli.main(args(output))
    assert output.read_bytes() == raw


def test_should_leave_old_output_limit_and_default_kwargs_unchanged(tmp_path, monkeypatch):
    seen = []

    def run(source, **kw):
        seen.append(kw)
        return dict(complete=True, padding="x" * (1024**2))

    monkeypatch.setattr(live, "run_case", run)
    output = tmp_path / "old-limit.json"
    with pytest.raises(ValueError, match="byte limit"):
        cli.main([a for a in args(output) if a != FLAG])
    assert seen == [dict(stream_recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[1])]
    assert output.read_bytes() == b"{}\n"


def test_should_keep_missing_invalid_costs_unavailable_in_failed_output():
    value = dict(
        complete=1,
        training_work=dict(attempted_updates=True, attempted_rows=-1),
        budget=dict(updates=4, started_at=float("inf")),
        memory=dict(peak_bytes="unknown"),
    )
    data, complete = cli._boundary_report_bytes(value)
    saved = json.loads(data)
    assert not complete and saved["producer_complete"] is None
    assert all(v is None for v in saved["reported"]["training_work"].values())
    assert saved["reported"]["budget"]["updates"] == 4
    assert saved["reported"]["budget"]["started_at"] is None
    assert saved["reported"]["memory"]["peak_bytes"] is None
    assert len(data) <= 32 * 1024
