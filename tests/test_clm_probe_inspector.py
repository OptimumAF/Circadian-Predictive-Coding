"""Literal detached metadata and offline publication; zero models or scoring."""

from copy import deepcopy
import json
import subprocess
import sys

import pytest

from src.app.clm_probe_inspector_projection import CATALOGS, project_clm_probe_review
from src.app.key_door_inspector import project_key_door_report
from src.infra.key_door_inspector_files import write_key_door_inspector
from test_key_door_inspector_files import Page


def failed():
    return dict(
        complete=False,
        native_complete=None,
        model_work=0,
        additional_scoring=0,
        effects=0,
        prohibited_imports=[],
        read_errors={"actual.json": "missing"},
        execution={"error": "RAM gate", "seconds": 0.0, "native_process_terminal": False},
        training=None,
        source_work=None,
        probe=None,
        probe_scores=None,
        counts=None,
    )


def case(group, index, after):
    ids = CATALOGS[group]
    concept = (
        ("invoice", "legacy", "parcel", "password")[index // 3]
        if group == "old_task"
        else ("tavrix", "vemlot")[index]
    )
    phase = ("sealed", "ready", "staged")[index % 3] if group == "old_task" else "after_return"
    target = "defer" if group == "old_task" else ("inspect", "route")[index]
    selected = "route" if after else target
    return dict(
        request={"query": "literal " + str(index)},
        phase=phase,
        concept=concept,
        role="train" if group == "old_task" else group,
        target_id=target,
        selected_id=selected,
        correct=selected == target,
        target_probability=0.2,
        probabilities=[[identity, 0.2] for identity in ids],
        confidence=0.8,
    )


def completed():
    report = failed()
    report.update(
        complete=True,
        native_complete=True,
        read_errors={},
        training=dict(requested_passes=32, committed_updates=32, complete=True),
        source_work=dict(commits=32, unreturned_work_unknown=False),
        probe=dict(
            complete=True,
            source_update=32,
            stream=dict(completed_updates=6),
            probe_work=dict(commits=6),
        ),
        probe_scores=dict(groups={}),
    )
    for group, count in (("probe_train", 2), ("probe_test", 2), ("old_task", 12)):
        points = report["probe_scores"]["groups"][group] = {}
        for point in ("before", "after"):
            cases = [case(group, i, point == "after") for i in range(count)]
            endpoint = dict(
                cases=cases,
                metrics=dict(
                    accuracy=0.5,
                    nll_mean=None,
                    nll_status="infinite_due_to_zero_target_probability",
                ),
            )
            if group == "old_task":
                endpoint["training_fit"] = dict(
                    rows=[
                        dict(
                            row=i,
                            phase=c["phase"],
                            concept=c["concept"],
                            target_id=c["target_id"],
                            selected_id=c["selected_id"],
                            correct=c["correct"],
                            probability_margin=-0.6,
                        )
                        for i, c in enumerate(cases)
                    ]
                )
            points[point] = endpoint
    return report


def section(view, name):
    return next(s for s in view["sections"] if s["id"] == name)


def assert_cells(view):
    for part in view["sections"]:
        for row in part["rows"]:
            for cell in row["cells"]:
                if cell["pointer"] is None:
                    continue
                value = view["source"]
                for key in cell["pointer"].split("/")[1:]:
                    value = (
                        value[int(key)]
                        if type(value) is list
                        else value[key.replace("~1", "/").replace("~0", "~")]
                    )
                assert value == cell["value"] and not cell["missing"]


@pytest.mark.parametrize("build", [failed, completed])
def test_should_route_detach_and_resolve_all_recorded_cells(build):
    source = build()
    original = deepcopy(source)
    view = project_key_door_report(source, source_label="literal.json")
    assert view == project_clm_probe_review(source, source_label="literal.json")
    assert view["schema_id"] == "clm_probe_inspector_v1"
    assert source == original == view["source"] and source is not view["source"]
    assert_cells(view)


def test_should_retain_negative_cases_and_signed_margins_without_scoring():
    view = project_key_door_report(completed(), source_label="literal.json")
    assert len(section(view, "old_task_after")["rows"]) == 12
    assert section(view, "old_task_after")["rows"][0]["cells"][-1]["value"] == -0.6
    assert section(view, "probe_train_after")["rows"][0]["cells"][6]["value"] is False
    metrics = section(view, "probe_train_after_metrics")
    assert metrics["rows"][2]["cells"][1]["value"] is None
    assert metrics["rows"][3]["cells"][1]["value"] == "infinite_due_to_zero_target_probability"


def test_should_keep_missing_null_zero_false_distinct_without_native_rows():
    view = project_key_door_report(failed(), source_label="partial.json")
    assert view["status"]["value"] is None and not view["status"]["missing"]
    assert len(view["sections"]) == 4
    assert not any(s["id"] == "probe_train_after" for s in view["sections"])
    assert_cells(view)
    costs = section(view, "costs")["rows"]
    assert costs[3]["cells"][1]["missing"]
    assert (
        next(r for r in costs if r["cells"][0]["value"] == "execution/seconds")["cells"][1]["value"]
        == 0.0
    )
    available = section(view, "availability")["rows"]
    assert available[2]["cells"][1]["value"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "shape",
        "probability",
        "bool_probability",
        "order",
        "correctness",
        "counter",
        "metric",
        "group",
        "row_count",
        "fit",
        "native_flag",
        "raw",
        "work_bool",
        "work_nonzero",
        "imports",
        "json",
        "scope",
        "cost",
        "source_updates",
    ],
)
def test_should_refuse_malformed_review_before_output(fault):
    report = completed()
    point = report["probe_scores"]["groups"]["probe_train"]["after"]
    row = point["cases"][0]
    if fault == "role":
        row["role"] = "final_test"
    elif fault == "shape":
        row["probabilities"].pop()
    elif fault == "probability":
        row["probabilities"][0][1] = 1.1
    elif fault == "bool_probability":
        row["probabilities"][0][1] = True
    elif fault == "order":
        row["probabilities"].reverse()
    elif fault == "correctness":
        row["correct"] = True
    elif fault == "counter":
        report["source_work"]["commits"] = False
    elif fault == "metric":
        point["metrics"]["accuracy"] = "0.5"
    elif fault == "group":
        report["probe_scores"]["groups"]["final_test"] = {}
    elif fault == "row_count":
        point["cases"].append(deepcopy(row))
    elif fault == "fit":
        report["probe_scores"]["groups"]["old_task"]["before"]["training_fit"]["rows"][0][
            "target_id"
        ] = "route"
    elif fault == "native_flag":
        report["native_complete"] = 1
    elif fault == "raw":
        report.pop("additional_scoring")
    elif fault == "work_bool":
        report["model_work"] = False
    elif fault == "work_nonzero":
        report["additional_scoring"] = 1
    elif fault == "imports":
        report["prohibited_imports"] = ["torch"]
    elif fault == "json":
        report["bad"] = object()
    elif fault == "scope":
        report["probe"]["parameter_update_scope"] = dict(name="all")
    elif fault == "cost":
        report["elapsed_seconds"] = True
    elif fault == "source_updates":
        report["probe"]["source_update"] = True
    with pytest.raises(ValueError):
        project_key_door_report(report, source_label="bad.json")


def test_should_refuse_extra_resource_file_for_self_contained_review():
    with pytest.raises(ValueError, match="owns"):
        project_key_door_report(
            failed(), source_label="literal.json", resources={}, resource_label="cost.json"
        )


def test_should_export_via_existing_cli_safely_and_preserve_source(tmp_path):
    report = completed()
    report["probe_scores"]["groups"]["probe_train"]["after"]["cases"][0]["request"]["query"] = (
        '</script><img src=x onerror="bad()">'
    )
    source, output = tmp_path / "review.json", tmp_path / "review.html"
    raw = json.dumps(report).encode()
    source.write_bytes(raw)
    command = [
        sys.executable,
        "-B",
        "-m",
        "scripts.build_key_door_inspector",
        "--source-artifact",
        str(source),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == raw
    page = Page(output.read_text(encoding="utf8"))
    assert page.payload()["source"] == report
    assert not any(tag == "img" for tag, _ in page.tags)
    assert "Saved CLM probe review" in output.read_text(encoding="utf8")
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, output)


def test_should_import_projection_and_cli_without_model_dependencies():
    code = "import sys; import scripts.build_key_door_inspector; import src.app.clm_probe_inspector_projection; assert not set(sys.modules) & {'torch','numpy','transformers'}"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
