"""Saved JSON only: full task outcomes, source cells and exclusive HTML export."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.ticket_workflow_inspector_projection import project_ticket_workflow_report
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector
from test_key_door_inspector_files import Page
from test_taskspace_memory_inspector import assert_sources, at, section

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/runs/native-ticket-family-controls-20261023"
SOURCES = (
    ROOT / "artifacts/runs/bp-new-contexts-20261023/actual.json",
    *(STAGE / f"{m}.json" for m in ("pc", "neutral_cpc", "gated_cpc")),
)


def minimal():
    return dict(definition="matched_native_ticket_controls_v1", method="pc", complete=False)


@pytest.mark.parametrize("path", SOURCES, ids=("bp", "pc", "neutral", "active"))
def test_saved_full_tasks_route_and_resolve_every_observation_without_recomputation(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    original = deepcopy(report)
    view = project_key_door_report(report, source_label=path.name)
    assert view == project_ticket_workflow_report(report, source_label=path.name)
    assert view["schema_id"] == "ticket_workflow_inspector_v1"
    assert view["source"] == original == report and path.read_bytes() == raw
    assert len(section(view, "outcomes")["rows"]) == 3
    assert len(section(view, "episodes")["rows"]) == 20
    assert len(section(view, "steps")["rows"]) == sum(
        len(e["episode"]["steps"]) for e in report["episodes"]
    )
    assert len(section(view, "releases")["rows"]) == len(report["applied_updates"])
    assert at(view, "/summary/new_context_gain")["value"] == report["summary"]["new_context_gain"]
    assert (
        at(view, "/episodes/0/episode/steps/0/result/selected_id")["value"]
        == report["episodes"][0]["episode"]["steps"][0]["result"]["selected_id"]
    )
    assert at(view, "/checkpoints/0/state/completed_updates")["value"] == 96
    assert_sources(view)
    if path == SOURCES[0]:
        assert at(view, "/summary/ordinary/0/legacy")["value"] is False
        assert at(view, "/summary/ordinary/0/password")["value"] is True
        assert at(view, "/summary/new_context_gain")["value"] == 1
        assert at(view, "/prior_training_work")["value"]["attempted_updates"] == 96
    else:
        assert at(view, "/summary/ordinary/16/invoice")["value"] is False
        assert at(view, "/summary/new_context_gain")["value"] == 0


def test_partial_source_preserves_missing_null_zero_false_and_paid_unknown_returns():
    report = minimal()
    report.update(
        error=None,
        costs=None,
        final_examples=0,
        training_work=dict(attempted_updates=2, returned_updates=1, unreturned_work_unknown=True),
    )
    view = project_key_door_report(report, source_label="partial.json")
    assert not view["status"]["value"]
    assert at(view, "/error")["value"] is None and not at(view, "/error")["missing"]
    assert at(view, "/costs")["value"] is None and not at(view, "/costs")["missing"]
    assert at(view, "/final_examples")["value"] == 0
    assert at(view, "/training_work")["value"]["unreturned_work_unknown"]
    assert section(view, "steps")["rows"] == []
    assert all(
        c["missing"]
        for r in section(view, "mechanisms")["rows"]
        for c in r["cells"][1:]
        if r["cells"][0]["value"] != "final_examples"
    )
    assert_sources(view)


def test_detached_source_cells_and_separate_resources_stay_unverified():
    report = minimal()
    report["costs"] = {"elapsed_seconds": 0.0}
    resources = {"external/cost~file": {"seconds": None}}
    view = project_ticket_workflow_report(
        report, source_label="partial.json", resources=resources, resource_label="costs.json"
    )
    assert "UNVERIFIED" in section(view, "resources")["title"]
    assert at(view, "/external~1cost~0file", "resources")["value"] == {"seconds": None}
    report["costs"]["elapsed_seconds"] = 9
    assert at(view, "/costs")["value"] == {"elapsed_seconds": 0.0}
    view["source"]["costs"]["elapsed_seconds"] = 10
    assert at(view, "/costs")["value"] == {"elapsed_seconds": 0.0}
    assert_sources(project_key_door_report(minimal(), source_label="owned.json"))


@pytest.mark.parametrize(
    "fault",
    [
        "definition",
        "method",
        "complete",
        "count",
        "flag",
        "summary",
        "episodes",
        "steps",
        "checkpoints",
        "duplicate",
        "diagnostic",
        "nonfinite",
        "nonjson",
    ],
)
def test_refuse_malformed_or_unbounded_presentation_data(fault):
    r = minimal()
    if fault == "definition":
        r["definition"] = "unknown"
    elif fault == "method":
        r["method"] = "bp"
    elif fault == "complete":
        r["complete"] = 1
    elif fault == "count":
        r["training_work"] = {"attempted_updates": True}
    elif fault == "flag":
        r["source_seed_actor_unchanged"] = 0
    elif fault == "summary":
        r["summary"] = {"ordinary": {"0": {"invoice": "true"}}}
    elif fault == "episodes":
        r["episodes"] = [{}] * 21
    elif fault == "steps":
        r["episodes"] = [dict(concept="invoice", source_updates=0, episode=dict(steps=[{}] * 7))]
    elif fault == "checkpoints":
        r["checkpoints"] = [dict(source_updates=i) for i in range(4)]
    elif fault == "duplicate":
        r["checkpoints"] = [dict(source_updates=0)] * 2
    elif fault == "diagnostic":
        r["applied_updates"] = [dict(diagnostic={"value": False})]
    elif fault == "nonfinite":
        r["costs"] = {"seconds": float("nan")}
    else:
        r["costs"] = {"tensor": object()}
    with pytest.raises(ValueError):
        project_ticket_workflow_report(r, source_label="bad.json")


def test_exclusive_file_export_inert_labels_and_all_controls_visible(tmp_path):
    danger = '</script><img id="injected" src="https://example.invalid" onerror="alert(1)">'
    r = minimal()
    r["error"] = danger
    view = project_key_door_report(r, source_label=danger)
    html = render_key_door_inspector(view).decode("utf8")
    page = Page(html)
    assert page.payload()["source"]["error"] == danger
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert "Saved ticket workflows" in html and "ticket_workflow_inspector_v1" in html
    assert "No circadian advantage" in html
    source = tmp_path / "partial.json"
    target = tmp_path / "ticket.html"
    source.write_text(json.dumps(r), encoding="utf8")
    write_key_door_inspector(source, target)
    assert Page(target.read_text(encoding="utf8")).payload()["schema_id"] == view["schema_id"]
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, target)
    assert target.read_bytes() == before


def test_stale_revision_observation_effect_and_result_stay_distinct():
    report = json.loads(SOURCES[0].read_bytes())
    view = project_key_door_report(report, source_label="bp.json")
    for i, wrapper in enumerate(report["episodes"]):
        if wrapper["episode"]["scenario"] != "stale_once":
            continue
        for j, step in enumerate(wrapper["episode"]["steps"]):
            if step["transition"]["status"] != "stale_revision":
                continue
            prefix = f"/episodes/{i}/episode/steps/{j}"
            observed = at(view, prefix + "/observation_state/revision")["value"]
            effect = at(view, prefix + "/effect_state/revision")["value"]
            result = at(view, prefix + "/transition/state/revision")["value"]
            assert observed != effect and effect == result
            return
    pytest.fail("saved BP must contain its observed stale revision refusal")
