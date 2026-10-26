"""Saved stationary evidence through the shared offline inspector boundary.

All inputs are existing JSON or explicitly synthetic failure/contradiction data.
No learner, prediction, environment, native diagnostic or browser is executed.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.build_key_door_inspector as cli
from src.app.key_door_inspector import project_key_door_report
from src.app.sensor_inspector_projection import project_sensor_report
from src.infra.key_door_inspector_files import write_key_door_inspector
from test_key_door_inspector_files import Page


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/runs/r46-stationary-sensor-20261017"
CASES = ("bp", "pc", "neutral_cpc", "gated_cpc", "scheduled", "periodic", "incident")


def saved(case="bp"):
    return json.loads((STAGE / f"native-{case}.json").read_text(encoding="utf8"))


def section(view, name):
    return next(item for item in view["sections"] if item["id"] == name)


def cells(view):
    return [cell for item in view["sections"] for row in item["rows"] for cell in row["cells"]]


def resolve(source, pointer):
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        source = source[int(token)] if type(source) is list else source[token]
    return source


def at(view, pointer):
    matches = [
        cell for cell in cells(view) if cell["source"] == "report" and cell["pointer"] == pointer
    ]
    assert matches, pointer
    assert all(not cell["missing"] for cell in matches)
    assert all(cell["value"] == resolve(view["source"], pointer) for cell in matches)
    return matches[0]


def assert_pointers(view):
    for item in view["sections"]:
        for row in item["rows"]:
            assert len(row["cells"]) == len(item["columns"])
            for cell in row["cells"]:
                if cell["missing"]:
                    assert cell["pointer"] is None and cell["value"] is None
                elif cell["pointer"] is not None:
                    source = view["source"] if cell["source"] == "report" else view["resources"]
                    assert cell["value"] == resolve(source, cell["pointer"])
                else:
                    assert type(cell["value"]) is str


@pytest.mark.parametrize("case", CASES)
def test_should_route_all_seven_saved_stationary_reports_with_source_exact_cells(case):
    source = saved(case)
    before = deepcopy(source)
    direct = project_sensor_report(source, source_label=f"native-{case}.json")
    routed = project_key_door_report(source, source_label=f"native-{case}.json")
    assert routed == direct
    assert routed["schema_id"] == "sensor_inspector_v1"
    assert routed["source"] == before and source == before
    assert routed["method"] == source["protocol"]["method"]
    assert [item["id"] for item in routed["sections"]] == [
        "prearrival",
        "heldout",
        "probes",
        "retention",
        "monitoring",
        "resources",
        "controls",
    ]
    assert len(section(routed, "prearrival")["rows"]) == 12
    for name in ("heldout", "probes"):
        empty = section(routed, name)
        assert empty["rows"] == []
        assert "not part" in empty["note"].lower() and "control" in empty["note"].lower()
    for pointer in (
        "/protocol/protocol",
        "/protocol/stationary",
        "/protocol/phases/0",
        "/protocol/phases/1",
        "/protocol/phases/2",
        "/protocol/fixture",
        "/protocol/limitations",
        "/protocol/monitoring/false_alarm_scope",
    ):
        at(routed, pointer)
    assert at(routed, "/protocol/stationary")["value"] is True
    assert_pointers(routed)


@pytest.mark.parametrize("case", CASES)
def test_should_keep_zero_of_eight_opportunities_and_recorded_null_nonexistent_phases(case):
    view = project_sensor_report(saved(case), source_label=f"native-{case}.json")
    prefix = "/sensor_monitoring/false_alarms/stationary_A"
    assert at(view, prefix + "/monitoring_batches")["value"] == 8
    for channel in range(3):
        assert at(view, prefix + f"/alarm_batches/{channel}")["value"] == 0
        assert at(view, prefix + f"/incident_onsets/{channel}")["value"] == 0
        assert at(view, prefix + f"/rates/{channel}")["value"] == 0.0
    for absent in ("pre_shift_A", "post_return_A"):
        assert (
            at(view, f"/sensor_monitoring/false_alarms/{absent}/monitoring_batches")["value"] == 0
        )
        for channel in range(3):
            value = at(view, f"/sensor_monitoring/false_alarms/{absent}/rates/{channel}")
            assert value["value"] is None and value["missing"] is False
    for absent in ("B", "A_return"):
        assert at(view, f"/sensor_monitoring/recovery/{absent}/status")["value"] == "not_applicable"
        assert at(view, f"/sensor_monitoring/recovery/{absent}/labels_to_recovery")["value"] is None
        assert at(view, f"/sensor_monitoring/recovery/{absent}/observed_batches")["value"] == 0
    assert_pointers(view)


@pytest.mark.parametrize(
    "case,pointer,value",
    [
        ("scheduled", "/scheduled_training/optimizer_steps_completed", 3),
        ("scheduled", "/scheduled_training/arrived_batches", 12),
        ("periodic", "/replay_budget_charged_exposures", 6),
        ("periodic", "/rehearsal/attempted_updates", 6),
        ("periodic", "/rehearsal/committed_updates", 6),
        ("incident", "/replay_budget_charged_exposures", 0),
        ("incident", "/rehearsal/attempted_updates", 0),
        ("incident", "/rehearsal/committed_updates", 0),
    ],
)
def test_should_show_actual_control_doses_without_substituting_planned_counts(case, pointer, value):
    view = project_sensor_report(saved(case), source_label=f"native-{case}.json")
    assert at(view, pointer)["value"] == value
    assert at(view, "/charged_training_updates")["value"] == 12
    for cell in [cell for row in section(view, "resources")["rows"] for cell in row["cells"]]:
        assert not cell["pointer"] or not cell["pointer"].startswith("/protocol/")
    assert_pointers(view)


def test_should_keep_incident_skips_and_zero_admissions_visible_as_recorded_evidence():
    view = project_sensor_report(saved("incident"), source_label="native-incident.json")
    for index in range(3):
        assert at(view, f"/rehearsal/events/{index}/status")["value"] == "skipped"
        assert at(view, f"/rehearsal/events/{index}/charged_exposures")["value"] == 0
    assert at(view, "/rehearsal/incident_admission/admitted_events")["value"] == 0
    assert_pointers(view)


def test_should_detach_report_metadata_and_cells_without_mutating_saved_source():
    source = saved()
    original = deepcopy(source)
    view = project_sensor_report(source, source_label="stationary.json")
    assert source == original
    source["protocol"]["phases"][0] = "B"
    source["sensor_monitoring"]["false_alarms"]["stationary_A"]["monitoring_batches"] = 999
    assert view["source"] == original
    view["source"]["protocol"]["fixture"] = "replaced copy"
    assert at_value(view, "/protocol/fixture") == original["protocol"]["fixture"]


def at_value(view, pointer):
    return next(cell["value"] for cell in cells(view) if cell["pointer"] == pointer)


@pytest.mark.parametrize("missing", [False, True])
def test_should_distinguish_missing_partial_costs_from_explicit_recorded_null(missing):
    source = saved()
    source.update(status="incomplete", reason="synthetic partial evidence")
    if missing:
        del source["charged_training_updates"]
    else:
        source["charged_training_updates"] = None
    source["sensor_monitoring"].pop("recovery")
    source["sensor_monitoring"].pop("false_alarms")
    view = project_sensor_report(source, source_label="synthetic-partial.json")
    cost = next(
        row["cells"][1]
        for row in section(view, "resources")["rows"]
        if row["cells"][0]["value"] == "charged_training_updates"
    )
    assert cost["value"] is None and cost["missing"] is missing
    assert cost["pointer"] == (None if missing else "/charged_training_updates")
    assert not any(
        cell["pointer"] and cell["pointer"].startswith("/sensor_monitoring/recovery/")
        for cell in cells(view)
    )
    assert view["source"] == source
    assert_pointers(view)


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/protocol/stationary", False),
        ("/protocol/stationary", 1),
        ("/protocol/phases", ["A", "B", "A"]),
        ("/protocol/monitoring/stationary", False),
        ("/protocol/monitoring/phase_schedule", ["A"] * 11 + ["B"]),
        ("/sensor_monitoring/configuration/stationary", False),
        ("/sensor_monitoring/configuration/phase_schedule", ["A"] * 8 + ["B"] * 4),
        ("/sensor_monitoring/recovery/B/status", "recovered"),
        ("/sensor_monitoring/recovery/A_return/status", "no_deficit"),
        ("/sensor_monitoring/recovery/B/labels_to_recovery", 8),
        ("/sensor_monitoring/recovery/B/observed_batches", 4),
        ("/sensor_monitoring/false_alarms/pre_shift_A/monitoring_batches", 1),
        ("/sensor_monitoring/false_alarms/post_return_A/monitoring_batches", 4),
        ("/protocol/final_evaluation", True),
    ],
)
def test_should_refuse_contradictory_stationary_claims_instead_of_relabelling_them(pointer, value):
    source = saved()
    parent, key = pointer.rsplit("/", 1)
    resolve(source, parent)[key] = value
    before = deepcopy(source)
    with pytest.raises(ValueError):
        project_sensor_report(source, source_label="contradiction.json")
    with pytest.raises(ValueError):
        project_key_door_report(source, source_label="contradiction.json")
    assert source == before


@pytest.mark.parametrize("field", ["stationary", "phases", "monitoring"])
def test_should_refuse_stationary_identity_without_its_required_protocol_truth(field):
    source = saved()
    del source["protocol"][field]
    with pytest.raises(ValueError):
        project_sensor_report(source, source_label="missing-identity.json")


def test_should_require_the_explicit_stationary_monitoring_configuration():
    source = saved()
    del source["protocol"]["monitoring"]["stationary"]
    with pytest.raises(ValueError):
        project_sensor_report(source, source_label="missing-monitor-mode.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("heldout_evaluation", {"progress": {"status": "not_run", "checkpoints": []}}),
        ("learnability", {"status": "not_run", "observations": []}),
    ],
)
def test_should_refuse_unqualified_stationary_probe_or_heldout_records(field, value):
    source = saved()
    source[field] = value
    with pytest.raises(ValueError):
        project_sensor_report(source, source_label="unsupported-extension.json")


def test_should_leave_optional_aggregate_resource_association_unverified_and_unallocated():
    resources: dict[str, Any] = dict(
        status="complete",
        case="aggregate-seven-controls",
        physical_work=dict(update_calls_completed=81, training_row_exposures=342),
        elapsed_seconds=0.5,
        process_rss=dict(peak_bytes=1000),
    )
    source = saved("bp")
    view = project_sensor_report(
        source,
        source_label="native-bp.json",
        resources=resources,
        resource_label="synthetic-aggregate-resources.json",
    )
    assert view["resources"] == resources
    assert "UNVERIFIED" in section(view, "resources")["note"]
    aggregate = next(
        cell
        for cell in cells(view)
        if cell["source"] == "resources"
        and cell["pointer"] == "/physical_work/update_calls_completed"
    )
    assert aggregate["value"] == 81
    assert at(view, "/charged_training_updates")["value"] == 12
    resources["physical_work"]["update_calls_completed"] = 0
    assert (
        aggregate["value"] == 81
        and view["resources"]["physical_work"]["update_calls_completed"] == 81
    )
    assert_pointers(view)


def test_should_refuse_declared_stationary_heldout_mode_without_results():
    source = saved()
    source["protocol"]["heldout"] = {"contexts": ["A", "B"]}
    with pytest.raises(ValueError, match="stationary heldout"):
        project_sensor_report(source, source_label="unsupported-heldout-mode.json")


@pytest.mark.parametrize("case", CASES)
def test_should_publish_each_saved_stationary_case_through_shared_file_boundary(tmp_path, case):
    source, output = STAGE / f"native-{case}.json", tmp_path / f"{case}.html"
    original = source.read_bytes()
    assert write_key_door_inspector(source, output) == output
    page = Page(output.read_text(encoding="utf8"))
    view = page.payload()
    assert view["schema_id"] == "sensor_inspector_v1"
    assert view["source"] == json.loads(original)
    assert source.read_bytes() == original
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, output)
    assert output.read_bytes() == before and source.read_bytes() == original
    assert_pointers(view)


@pytest.mark.parametrize("case", CASES)
def test_should_use_the_existing_cli_for_all_stationary_reports(
    tmp_path, monkeypatch, capsys, case
):
    source, output = STAGE / f"native-{case}.json", tmp_path / f"cli-{case}.html"
    before = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["builder", "--source-artifact", str(source), "--output", str(output)]
    )
    cli.main()
    response = json.loads(capsys.readouterr().out)
    assert response["output"] == str(output.resolve())
    assert response["source"] == str(source.resolve())
    assert response["bytes"] == output.stat().st_size
    assert Page(output.read_text(encoding="utf8")).payload()["source"] == json.loads(before)
    assert source.read_bytes() == before


def test_should_keep_stationary_metadata_and_failure_text_inert_when_published(tmp_path):
    source = saved()
    danger = '</script><img id="injected" src="https://example.invalid" onerror="alert(1)"> &'
    source.update(status="incomplete", reason=danger)
    source["protocol"]["fixture"] = danger
    path, output = tmp_path / "untrusted.json", tmp_path / "safe.html"
    path.write_text(json.dumps(source), encoding="utf8")
    before = path.read_bytes()
    write_key_door_inspector(path, output)
    page = Page(output.read_text(encoding="utf8"))
    assert page.payload()["source"]["reason"] == danger
    assert at_value(page.payload(), "/protocol/fixture") == danger
    assert "<" not in "".join(page.payload_parts)
    assert not any(attributes.get("id") == "injected" for _, attributes in page.tags)
    assert all(
        not (attributes.get(key) or "").lower().startswith(("https:", "http:", "//"))
        for _, attributes in page.tags
        for key in ("src", "href")
    )
    assert path.read_bytes() == before


def test_should_refuse_invalid_stationary_input_before_creating_output_parent(tmp_path):
    source = saved()
    source["protocol"]["phases"] = ["A", "B", "A"]
    path, output = tmp_path / "invalid.json", tmp_path / "new-directory" / "invalid.html"
    path.write_text(json.dumps(source), encoding="utf8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_key_door_inspector(path, output)
    assert not output.parent.exists() and path.read_bytes() == before
