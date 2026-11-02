"""Shared-head saved projection behavior; no model or numerical imports."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.infra.key_door_inspector_files import write_key_door_inspector

ROOT = Path(__file__).resolve().parents[1]
WORK = "reserved_probe/linear_rehearsal_work"


def fixture(control="joint", once=True):
    work: dict[str, Any] = dict(
        definition="shared_linear_once_return_probe_update_v1"
        if once
        else "shared_linear_joint_probe_update_v1",
        control=control,
        attempted_updates=2,
        returned_updates=1,
        applied_updates=1,
        application_unknown_updates=1,
        attempted_rows=9,
        additional_rows_charged=18 if once else 24,
        native_sgd_updates=0,
        events=[
            dict(
                phase="return",
                status="failed",
                current_rows=6,
                work_known=True,
                applied=False,
                work=dict(
                    fit_attempts=1,
                    fit_returns=0,
                    fit_rows=12 if once else 18,
                    additional_rows_charged=6 if once else 12,
                    retained_train_rows_used=(6 if once else 12) if control == "joint" else 0,
                    current_rows_used=6 if control == "joint" else (12 if once else 18),
                    application_unknown=1,
                    unreturned_fit_work_unknown=1,
                ),
            ),
        ],
    )
    source: dict[str, Any] = dict(
        definition="linear_context_return_ticket_workflow_v1",
        method="linear_fit",
        complete=False,
        reserved_probe_requested=True,
        reserved_probe_update="minimum_change",
        reserved_probe_rehearsal=control,
        final_examples=0,
        reserved_probe=dict(
            complete=False,
            linear_probe_rehearsal=control,
            linear_rehearsal_work=work,
            preserved_training_work=dict(attempted_updates=0, returned_updates=0, attempted_rows=0),
            preserved_linear_rehearsal_work=dict(budget=dict(admissions=2, additional_rows=18)),
        ),
    )
    if once:
        source["reserved_probe_return_once"] = True
        source["reserved_probe"]["linear_probe_return_once"] = True
        work["return_once"] = True
    return source


def section(view, identity):
    return next(s for s in view["sections"] if s["id"] == identity)


def resolve(source, pointer):
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        source = source[int(key)] if type(source) is list else source[key]
    return source


def assert_cells(view):
    def walk(value):
        if type(value) is list:
            for child in value:
                walk(child)
        elif type(value) is dict:
            if set(value) == {"value", "pointer", "source", "missing"}:
                if value["pointer"] is not None:
                    assert resolve(view["source"], value["pointer"]) == value["value"]
            else:
                for key, child in value.items():
                    if key not in ("source", "resources"):
                        walk(child)

    walk(view)


@pytest.mark.parametrize("control", ["joint", "current_only"])
@pytest.mark.parametrize("once", [True, False])
def test_shared_recipe_and_actual_costs_are_saved_cells_not_inferred_totals(control, once):
    source = fixture(control, once)
    before = deepcopy(source)
    view = project_key_door_report(source, source_label="fixture")
    assert view["view_kind"] == "ordinary_reserved_probe"
    costs = section(view, "learning-work")
    assert costs["rows"][1]["cells"][0]["value"] == "Private direct train_batch ports"
    assert costs["rows"][1]["cells"][1]["value"] == 0
    aux = section(view, "auxiliary-work")
    fields = {row["cells"][0]["value"]: row["cells"][1] for row in aux["rows"]}
    assert fields[WORK + "/attempted_updates"]["value"] == 2
    assert fields[WORK + "/application_unknown_updates"]["value"] == 1
    assert fields[WORK + "/additional_rows_charged"]["value"] == (18 if once else 24)
    event = section(view, "auxiliary-events")["rows"][0]["cells"]
    by_pointer = {cell["pointer"]: cell for cell in event if cell["pointer"]}
    assert by_pointer["/" + WORK + "/events/0/work/fit_rows"]["value"] == (12 if once else 18)
    assert by_pointer["/" + WORK + "/events/0/work/retained_train_rows_used"]["value"] == (
        (6 if once else 12) if control == "joint" else 0
    )
    assert source == before
    assert_cells(view)


def test_missing_null_false_zero_and_unfinished_receipt_stay_distinct():
    source = fixture()
    work = source["reserved_probe"]["linear_rehearsal_work"]
    work["events"][0].update(work_known=False, applied=None, work=None, status="running")
    work["applied_updates"] = None
    work["application_unknown_updates"] = 0
    del work["returned_updates"]
    view = project_key_door_report(source, source_label="unfinished")
    values = {
        row["cells"][0]["value"]: row["cells"][1] for row in section(view, "auxiliary-work")["rows"]
    }
    assert values[WORK + "/applied_updates"] == dict(
        value=None, pointer="/" + WORK + "/applied_updates", source="report", missing=False
    )
    assert values[WORK + "/application_unknown_updates"]["value"] == 0
    assert values[WORK + "/returned_updates"]["missing"]
    event = section(view, "auxiliary-events")["rows"][0]["cells"]
    assert any(
        c["value"] is False and c["pointer"].endswith("/work_known") for c in event if c["pointer"]
    )
    assert any(c["missing"] for c in event)
    assert_cells(view)


@pytest.mark.parametrize("private", [{}, {"linear_rehearsal_work": None}])
def test_absent_auxiliary_costs_are_unknown_not_zero_or_a_total(private):
    source = fixture()
    source["reserved_probe"] = private
    view = project_key_door_report(source, source_label="pre-private failure")
    assert all(row["cells"][1]["missing"] for row in section(view, "auxiliary-work")["rows"])
    assert section(view, "auxiliary-events")["rows"] == []
    assert all(row["cells"][3]["missing"] for row in section(view, "outcomes")["rows"])


@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "update",
        "foreign_family",
        "control",
        "private_control",
        "once_type",
        "once_mismatch",
        "missing_control",
        "recipe",
        "count_bool",
        "count_negative",
        "event_bound",
        "event_phase",
        "event_status",
        "event_flag",
        "event_work",
        "event_count",
        "event_recipe",
        "source_count",
        "auxiliary_shape",
        "final",
        "request",
    ],
)
def test_refuses_incompatible_intent_or_malformed_costs(fault):
    source = fixture()
    private = source["reserved_probe"]
    work = private["linear_rehearsal_work"]
    event = work["events"][0]
    if fault == "method":
        source["method"] = "bp"
    elif fault == "update":
        source["reserved_probe_update"] = "sgd"
    elif fault == "foreign_family":
        source["definition"] = "separate_context_linear_ticket_workflow_v1"
    elif fault == "control":
        source["reserved_probe_rehearsal"] = "final"
    elif fault == "private_control":
        private["linear_probe_rehearsal"] = "current_only"
    elif fault == "once_type":
        source["reserved_probe_return_once"] = 1
    elif fault == "once_mismatch":
        private["linear_probe_return_once"] = False
    elif fault == "missing_control":
        source.pop("reserved_probe_rehearsal")
        private.pop("linear_probe_rehearsal")
        work.pop("control")
    elif fault == "recipe":
        work["definition"] = "shared_linear_joint_probe_update_v1"
    elif fault == "count_bool":
        work["attempted_updates"] = True
    elif fault == "count_negative":
        work["additional_rows_charged"] = -1
    elif fault == "event_bound":
        work["events"] *= 33
    elif fault == "event_phase":
        event["phase"] = "final"
    elif fault == "event_status":
        event["status"] = "accepted_science"
    elif fault == "event_flag":
        event["applied"] = 0
    elif fault == "event_work":
        event["work"] = []
    elif fault == "event_count":
        event["work"]["fit_rows"] = False
    elif fault == "event_recipe":
        event["definition"] = "final_fit"
    elif fault == "source_count":
        source["reserved_probe_source_training"] = {"completed_updates": True}
    elif fault == "auxiliary_shape":
        source["probe_auxiliary_updates"] = 32
    elif fault == "final":
        source["final_examples"] = 1
    else:
        source["reserved_probe_requested"] = False
    with pytest.raises(ValueError):
        project_key_door_report(source, source_label="invalid")


@pytest.mark.parametrize(
    "relative",
    [
        "single-linear-probe-native-20261030/native.json",
        "shared-linear-probe-comparison-20261031/joint.json",
        "shared-linear-probe-comparison-20261031/current_only.json",
        "shared-linear-once-comparison-20261031/joint.json",
        "shared-linear-once-comparison-20261031/current_only.json",
    ],
)
def test_existing_saved_shared_reports_remain_unchanged_with_exact_cells(relative):
    path = ROOT / "artifacts/runs" / relative
    if not path.is_file():
        pytest.skip("local research capture not distributed")
    raw = path.read_bytes()
    source = json.loads(raw)
    view = project_key_door_report(source, source_label=str(path))
    assert len(view["recorded_curves"]) == 4
    assert len(section(view, "outcomes")["rows"]) == 18
    assert len(section(view, "method-availability")["rows"]) == 5
    assert (
        sum(
            not p["y"]["missing"]
            for panel in view["recorded_curves"]
            for s in panel["series"]
            for p in s["points"]
        )
        == 32
    )
    if "linear_rehearsal_work" in source["reserved_probe"]:
        assert len(section(view, "auxiliary-events")["rows"]) == 32
    else:
        assert section(view, "auxiliary-events")["rows"] == []
    assert path.read_bytes() == raw and source == json.loads(raw)
    assert_cells(view)


def test_shared_report_offline_writer_keeps_existing_safe_exclusive_boundary(tmp_path):
    source = fixture()
    source["error"] = "</script><img src=x onerror=alert(1)>"
    path = tmp_path / "shared.json"
    path.write_text(json.dumps(source), encoding="utf8")
    output = tmp_path / "shared.html"
    write_key_door_inspector(path, output)
    html = output.read_text(encoding="utf8")
    assert "auxiliary-events" in html and "auxiliary-work" in html
    assert "</script><img" not in html and "\\u003c/script\\u003e" in html
    with pytest.raises(FileExistsError):
        write_key_door_inspector(path, output)
