"""Saved metadata and offline-page behavior; no learners or numerical imports."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.reserved_probe_inspector_projection import (
    PHASES,
    POINTS,
    TASKS,
    project_reserved_probe_report,
)
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector

ROOT = Path(__file__).resolve().parents[1]
ACTUAL = ROOT / "artifacts/runs/ordinary-linear-probe-native-20261030/native.json"


def fixture(method="linear_fit"):
    report: dict[str, Any] = dict(
        definition="separate_context_linear_ticket_workflow_v1"
        if method == "linear_fit"
        else "native_context_return_ticket_workflow_v1",
        method=method,
        complete=True,
        reserved_probe_requested=True,
        final_examples=0,
        reserved_probe_update="minimum_change",
        source_unchanged=True,
        actor_unchanged=True,
        training_work=dict(attempted_updates=12, returned_updates=12, attempted_rows=72),
    )
    episodes = {
        phase: [
            dict(
                phase=phase,
                policy_version=v,
                concept=c,
                role="controller_train",
                complete=True,
                episode=dict(
                    complete=True,
                    success=(index != 5 or phase != "before_probe"),
                    scenario="ordinary",
                    training_updates=0,
                    stopped_reason="saved",
                ),
            )
            for index, (v, c) in enumerate(TASKS)
        ]
        for phase in PHASES
    }

    def points(old=False, returned=False):
        contexts = ("batch-0", "batch-1") if old else ("parcel-v1", "password-v2")
        return [
            dict(
                updates=p,
                scores={c: dict(ce=1.2 if returned else 1.0, accuracy=1.0) for c in contexts},
            )
            for p in POINTS
        ]

    probe = dict(
        checkpoints=points(),
        training_readouts=dict(checkpoints=points(True)),
        return_phase=dict(
            checkpoints=points(returned=True),
            training_readouts=dict(checkpoints=points(True, True)),
        ),
    )
    report["reserved_probe"] = dict(
        complete=True,
        final_examples=0,
        episodes=episodes,
        probe=probe,
        preserved_training_work=dict(attempted_updates=32, returned_updates=32, attempted_rows=144),
    )
    report["reserved_probe_summary"] = dict(
        definition="ordinary_reserved_task_outcomes_v1",
        status="complete",
        cohorts={
            name: dict(
                tasks=[f"{v}:{c}" for v, c in tasks],
                successes_by_phase=dict(
                    zip(PHASES, (4, 4, 4) if name == "source_trained" else (1, 2, 2))
                ),
            )
            for name, tasks in (("source_trained", TASKS[:4]), ("fresh", TASKS[4:]))
        },
    )
    return report


def section(view, identity):
    return next(s for s in view["sections"] if s["id"] == identity)


def resolve(source, pointer):
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        source = source[int(key)] if type(source) is list else source[key]
    return source


def assert_cells(view):
    def walk(node):
        if type(node) is list:
            for item in node:
                walk(item)
        elif type(node) is dict:
            if set(node) == {"value", "pointer", "source", "missing"}:
                if node["pointer"] is not None:
                    source = view["resources"] if node["source"] == "resources" else view["source"]
                    assert resolve(source, node["pointer"]) == node["value"]
            else:
                for key, value in node.items():
                    if key not in ("source", "resources"):
                        walk(value)

    walk(view)


@pytest.mark.parametrize("method", ["linear_fit", "bp", "pc", "neutral_cpc", "gated_cpc"])
def test_dispatches_all_explicit_families_without_hiding_controls_or_recomputing(method):
    source = fixture(method)
    pin = deepcopy(source)
    view = project_key_door_report(source, source_label="saved.json")
    assert view == project_reserved_probe_report(source, source_label="saved.json")
    assert view["schema_id"] == "ticket_workflow_inspector_v1"
    assert len(section(view, "outcomes")["rows"]) == 18
    assert len(section(view, "score-readings")["rows"]) == 32
    assert len(section(view, "method-availability")["rows"]) == 5
    assert len(view["recorded_curves"]) == 4
    assert (
        sum(
            row["cells"][1]["value"] == "Not provided"
            for row in section(view, "method-availability")["rows"]
        )
        == 4
    )
    assert source == pin and view["source"] == pin
    assert_cells(view)


def test_actual_saved_observation_and_adverse_return_are_read_only():
    raw = ACTUAL.read_bytes()
    source = json.loads(raw)
    view = project_key_door_report(source, source_label="actual328")
    totals = section(view, "task-totals")["rows"]
    assert [c["value"] for c in totals[0]["cells"]][1:] == [4, 4, 4]
    assert [c["value"] for c in totals[1]["cells"]][1:] == [1, 2, 2]
    panels = view["recorded_curves"]
    assert (
        panels[2]["series"][0]["points"][-1]["y"]["value"]
        > panels[0]["series"][0]["points"][-1]["y"]["value"]
    )
    assert len(section(view, "task-steps")["rows"]) == 54
    assert ACTUAL.read_bytes() == raw and source == json.loads(raw)
    assert_cells(view)


def test_preserves_unknown_null_false_zero_partial_and_missing_chart_gaps():
    source = fixture()
    source.update(complete=False, error=None, source_unchanged=False)
    source["reserved_probe"]["episodes"]["before_probe"][0]["episode"]["success"] = None
    points = source["reserved_probe"]["probe"]["checkpoints"]
    points[0]["scores"]["parcel-v1"]["ce"] = 0.0
    points[1]["scores"]["parcel-v1"]["ce"] = None
    del points[2]  # Do not bridge an absent planned checkpoint.
    view = project_key_door_report(source, source_label="partial")
    curve = view["recorded_curves"][0]["series"][0]["points"]
    assert curve[0]["y"]["value"] == 0 and not curve[0]["y"]["missing"]
    assert curve[1]["y"]["value"] is None and not curve[1]["y"]["missing"]
    assert curve[2]["y"]["missing"] and curve[2]["x"]["missing"]
    assert not view["status"]["value"]
    assert view["reason"]["value"] is None and not view["reason"]["missing"]
    assert section(view, "outcomes")["rows"][0]["cells"][3]["value"] is None
    assert_cells(view)


def test_failed_before_private_work_stays_unknown_and_uses_saved_failure():
    source = fixture("bp")
    source.update(complete=False, error="failed before fit", reserved_probe={})
    source.pop("reserved_probe_summary")
    view = project_key_door_report(source, source_label="failed")
    assert view["reason"]["value"] == "failed before fit"
    assert all(row["cells"][3]["missing"] for row in section(view, "outcomes")["rows"])
    assert all(
        p["y"]["missing"]
        for panel in view["recorded_curves"]
        for s in panel["series"]
        for p in s["points"]
    )
    assert_cells(view)


@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "family",
        "request",
        "final",
        "final_bool",
        "role",
        "task",
        "phase",
        "order",
        "row_bound",
        "success",
        "training",
        "scenario",
        "summary_count",
        "summary_bool",
        "summary_tasks",
        "curve_order",
        "curve_duplicate",
        "curve_context",
        "curve_bool",
        "curve_negative",
        "curve_accuracy",
        "work_bool",
        "work_negative",
        "nonfinite",
    ],
)
def test_refuses_foreign_or_malformed_metadata(fault):
    source = fixture()
    rows = source["reserved_probe"]["episodes"]["before_probe"]
    points = source["reserved_probe"]["probe"]["checkpoints"]
    if fault == "method":
        source["method"] = "unknown"
    elif fault == "family":
        source["definition"] = "final_test"
    elif fault == "request":
        source["reserved_probe_requested"] = False
    elif fault == "final":
        source["final_examples"] = 1
    elif fault == "final_bool":
        source["final_examples"] = False
    elif fault == "role":
        rows[0]["role"] = "final_test"
    elif fault == "task":
        rows[0]["concept"] = "parcel"
    elif fault == "phase":
        source["reserved_probe"]["episodes"]["final"] = []
    elif fault == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "row_bound":
        rows.append(deepcopy(rows[0]))
    elif fault == "success":
        rows[0]["episode"]["success"] = 1
    elif fault == "training":
        rows[0]["episode"]["training_updates"] = 1
    elif fault == "scenario":
        rows[0]["episode"]["scenario"] = "stale_once"
    elif fault == "summary_count":
        source["reserved_probe_summary"]["cohorts"]["fresh"]["successes_by_phase"][
            "before_probe"
        ] = 3
    elif fault == "summary_bool":
        source["reserved_probe_summary"]["cohorts"]["fresh"]["successes_by_phase"][
            "before_probe"
        ] = True
    elif fault == "summary_tasks":
        source["reserved_probe_summary"]["cohorts"]["fresh"]["tasks"] = ["other"]
    elif fault == "curve_order":
        points[0], points[1] = points[1], points[0]
    elif fault == "curve_duplicate":
        points[1]["updates"] = 0
    elif fault == "curve_context":
        points[0]["scores"]["final"] = {}
    elif fault == "curve_bool":
        points[0]["scores"]["parcel-v1"]["ce"] = False
    elif fault == "curve_negative":
        points[0]["scores"]["parcel-v1"]["ce"] = -1.0
    elif fault == "curve_accuracy":
        points[0]["scores"]["parcel-v1"]["accuracy"] = 1.1
    elif fault == "work_bool":
        source["training_work"]["attempted_updates"] = True
    elif fault == "work_negative":
        source["training_work"]["attempted_rows"] = -1
    else:
        source["costs"] = {"elapsed": float("inf")}
    with pytest.raises(ValueError):
        project_reserved_probe_report(source, source_label="invalid")


def test_all_headers_precede_outcome_access_even_in_later_phase():
    source = fixture()
    source["reserved_probe"]["episodes"]["before_probe"][0]["episode"] = 42
    source["reserved_probe"]["episodes"]["after_return"][0]["role"] = "final_test"
    with pytest.raises(ValueError, match="header/role/order"):
        project_reserved_probe_report(source, source_label="foreign")


def test_detaches_sources_cells_and_escapes_resource_pointers():
    source = fixture()
    resources: dict[str, Any] = {"path/with~token": {"seconds": None}}
    view = project_reserved_probe_report(
        source, source_label="saved", resources=resources, resource_label="costs"
    )
    assert "UNVERIFIED" in section(view, "resources")["title"]
    cell = section(view, "resources")["rows"][0]["cells"][1]
    assert cell["pointer"] == "/path~1with~0token"
    source["complete"] = False
    resources["path/with~token"]["seconds"] = 99
    assert view["status"]["value"] is True and cell["value"] == {"seconds": None}
    view["source"]["complete"] = False
    assert view["status"]["value"] is True
    with pytest.raises(ValueError, match="label"):
        project_reserved_probe_report(fixture(), source_label="saved", resources={})


def test_offline_writer_uses_existing_safe_template_and_exclusive_output(tmp_path):
    source = fixture()
    source["error"] = "</script><img src=x onerror=alert(1)>"
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(source), encoding="utf8")
    output = tmp_path / "saved.html"
    write_key_door_inspector(path, output)
    html = output.read_text(encoding="utf8")
    assert "recordedProbeCurves" in html and "__INSPECTOR_DATA__" not in html
    assert "</script><img" not in html and "\\u003c/script\\u003e" in html
    assert render_key_door_inspector(project_key_door_report(source, source_label=str(path)))
    with pytest.raises(FileExistsError):
        write_key_door_inspector(path, output)
