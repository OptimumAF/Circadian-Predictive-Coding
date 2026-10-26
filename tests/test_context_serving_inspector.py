"""Saved JSON/scalar presentation checks; no learners or task execution."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import build_key_door_inspector as cli
from src.app.context_serving_inspector_projection import project_context_serving_report
from src.app.key_door_inspector import project_key_door_report
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector
from test_key_door_inspector_files import Page
from test_taskspace_memory_inspector import assert_sources, at, section

ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple(
    ROOT / f"artifacts/runs/{s}/actual.json"
    for s in (
        "context-head-serving-20261023",
        "portable-context-example-20261023",
    )
)


def minimal():
    return dict(definition="guarded_context_serving_v1", complete=False)


@pytest.mark.parametrize("path", SOURCES, ids=["accepted", "installed"])
def test_should_route_all_saved_outcomes_steps_reads_guards_and_rollback(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    original = deepcopy(report)
    view = project_key_door_report(report, source_label=path.name)
    assert view == project_context_serving_report(report, source_label=path.name)
    assert view["schema_id"] == "ticket_workflow_inspector_v1"
    assert view["source"] == original == report and path.read_bytes() == raw
    assert len(section(view, "outcomes")["rows"]) == 42
    assert len(section(view, "reads")["rows"]) == 126
    assert len(section(view, "steps")["rows"]) == 126
    assert len(section(view, "guards")["rows"]) == 2
    assert len(section(view, "publications")["rows"]) == 6
    assert len(section(view, "releases")["rows"]) == 2
    assert at(view, "/cases/1/guards/0/report/decision/accepted")["value"] is False
    assert at(view, "/cases/1/guards/1/report/decision/accepted")["value"] is True
    assert at(view, "/cases/1/rollback_state_equal_source")["value"] is True
    assert at(view, "/prior_training")["value"] == dict(updates=12, rows=72, mistakes=16)
    assert at(view, "/cases/1/episodes/0/reads/0/actor_version")["value"] == "qualified-12"
    assert at(view, "/cases/1/publications/2/actor_version")["value"] == "c0"
    assert at(view, "/cases/1/final_actor/actor_version")["value"] == "qualified-12"
    assert_sources(view)


def test_should_preserve_partial_null_zero_false_and_unknown_without_inferred_success():
    report = minimal()
    report.update(
        error=None,
        memory=None,
        final_examples=0,
        cases=[
            dict(
                mode="guarded",
                complete=False,
                source_unchanged=False,
                training_work=dict(
                    attempted_updates=2, returned_updates=1, unreturned_work_unknown=True
                ),
            )
        ],
    )
    view = project_key_door_report(report, source_label="partial.json")
    assert not view["status"]["value"]
    assert at(view, "/error")["value"] is None and not at(view, "/error")["missing"]
    assert at(view, "/memory")["value"] is None
    assert at(view, "/final_examples")["value"] == 0
    assert at(view, "/cases/0/source_unchanged")["value"] is False
    assert at(view, "/cases/0/training_work")["value"]["unreturned_work_unknown"] is True
    assert section(view, "outcomes")["rows"] == []
    assert all(row["cells"][-1]["missing"] for row in section(view, "rollback")["rows"][:-1])
    assert_sources(view)


def test_should_detach_cells_and_keep_separate_costs_unverified():
    report = minimal()
    report["memory"] = dict(peak_bytes=42)
    resources: dict[str, Any] = {"separate/cost~file": {"seconds": None}}
    view = project_context_serving_report(
        report, source_label="partial", resources=resources, resource_label="costs"
    )
    assert "UNVERIFIED" in section(view, "resources")["title"]
    assert at(view, "/separate~1cost~0file", "resources")["value"] == {"seconds": None}
    report["memory"]["peak_bytes"] = 999
    resources["separate/cost~file"]["seconds"] = 9
    assert at(view, "/memory")["value"] == {"peak_bytes": 42}
    view["source"]["memory"]["peak_bytes"] = 1000
    assert at(view, "/memory")["value"] == {"peak_bytes": 42}
    with pytest.raises(ValueError, match="label"):
        project_context_serving_report(minimal(), source_label="partial", resources={})


@pytest.mark.parametrize(
    "fault",
    [
        "definition",
        "completion",
        "cases",
        "mode",
        "duplicate",
        "episodes",
        "phase",
        "context",
        "concept",
        "success",
        "reads",
        "generation",
        "steps",
        "attempt",
        "guards",
        "accepted",
        "reasons",
        "utility",
        "publications",
        "count",
        "receipts",
        "budget",
        "rollback",
        "nonfinite",
        "nonjson",
        "elapsed",
        "rss",
    ],
)
def test_should_refuse_malformed_or_unbounded_saved_data(fault):
    r = minimal()
    case: dict[str, Any] = dict(mode="guarded")
    r["cases"] = [case]
    episode: dict[str, Any] = dict(
        phase="initial", policy_version="v1", concept="legacy", episode={}
    )
    case["episodes"] = [episode]
    if fault == "definition":
        r["definition"] = "unknown"
    if fault == "completion":
        r["complete"] = 1
    if fault == "cases":
        r["cases"] = [case] * 3
    if fault == "mode":
        case["mode"] = "invented"
    if fault == "duplicate":
        r["cases"] = [case] * 2
    if fault == "episodes":
        case["episodes"] = [episode] * 25
    if fault == "phase":
        episode["phase"] = "invented"
    if fault == "context":
        episode["policy_version"] = "hidden"
    if fault == "concept":
        episode["concept"] = "invented"
    if fault == "success":
        episode["episode"]["success"] = 1
    if fault == "reads":
        episode["reads"] = [{}] * 7
    if fault == "generation":
        episode["reads"] = [dict(generation=True)]
    if fault == "steps":
        episode["episode"]["steps"] = [{}] * 7
    if fault == "attempt":
        episode["episode"]["steps"] = [dict(attempt=1.0)]
    if fault == "guards":
        case["guards"] = [{}] * 3
    if fault == "accepted":
        case["guards"] = [dict(report=dict(decision=dict(accepted="true")))]
    if fault == "reasons":
        case["guards"] = [dict(report=dict(decision=dict(reasons=[1])))]
    if fault == "utility":
        case["guards"] = [dict(report=dict(evidence=dict(candidate_new_utility=True)))]
    if fault == "publications":
        case["publications"] = [{}] * 4
    if fault == "count":
        r["prior_training"] = dict(updates=True)
    if fault == "receipts":
        case["applied_updates"] = [{}] * 3
    if fault == "budget":
        case["budget"] = dict(updates=False)
    if fault == "rollback":
        case["rollback_state_equal_source"] = "true"
    if fault == "nonfinite":
        r["memory"] = dict(peak_bytes=float("nan"))
    if fault == "nonjson":
        r["memory"] = dict(tensor=object())
    if fault == "elapsed":
        r["elapsed_seconds"] = "fast"
    if fault == "rss":
        r["memory"] = dict(peak_bytes=1.5)
    with pytest.raises(ValueError):
        project_context_serving_report(r, source_label="bad.json")


def test_should_export_installed_saved_run_through_public_cli_without_recomputing(
    tmp_path, monkeypatch, capsys
):
    source = SOURCES[1]
    raw = source.read_bytes()
    output = tmp_path / "context.html"
    monkeypatch.setattr(
        sys, "argv", ["builder", "--source-artifact", str(source), "--output", str(output)]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["output"] == str(output.resolve())
    view = Page(output.read_text(encoding="utf8")).payload()
    assert view["source"] == json.loads(raw) and source.read_bytes() == raw
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert output.read_bytes() == before


def test_should_keep_partial_labels_inert_and_render_unknown_rollback(tmp_path):
    danger = '</script><img id="injected" src="https://example.invalid" onerror="alert(1)">'
    report = minimal()
    report["error"] = danger
    view = project_key_door_report(report, source_label=danger)
    html = render_key_door_inspector(view).decode("utf8")
    assert Page(html).payload()["source"]["error"] == danger
    assert not any(attrs.get("id") == "injected" for _, attrs in Page(html).tags)
    assert "Guarded contextual learning" in html and "No circadian advantage" in html
    source, target = tmp_path / "partial.json", tmp_path / "partial.html"
    source.write_text(json.dumps(report))
    write_key_door_inspector(source, target)
    assert Page(target.read_text(encoding="utf8")).payload()["status"]["value"] is False
