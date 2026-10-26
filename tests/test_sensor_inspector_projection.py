"""Saved-evidence behavior only: no learner, probe, SVD, or native model runs."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from src.app.sensor_inspector_projection import project_sensor_report

_ROOT = Path(__file__).resolve().parents[1]
_RUN = _ROOT / "artifacts/runs/r46-sensor-heldout-20261017"
_CASES = ("bp", "pc", "neutral_cpc", "gated_cpc", "scheduled_bp", "replay_bp")


def _saved(case: str = "bp", kind: str = "result") -> dict[str, Any]:
    return json.loads((_RUN / f"{case}-{kind}.json").read_text(encoding="utf8"))


def _section(view: dict[str, Any], identity: str) -> dict[str, Any]:
    return next(section for section in view["sections"] if section["id"] == identity)


def _resolve(source: Any, pointer: str) -> Any:
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        source = source[int(token)] if type(source) is list else source[token]
    return source


def _assert_pointers(view: dict[str, Any]) -> None:
    for section in view["sections"]:
        for row in section["rows"]:
            assert len(row["cells"]) == len(section["columns"])
            for cell in row["cells"]:
                if cell["missing"]:
                    assert cell["pointer"] is None and cell["value"] is None
                elif cell["pointer"] is not None:
                    original = view["source"] if cell["source"] == "report" else view["resources"]
                    assert _resolve(original, cell["pointer"]) == cell["value"]
                else:
                    assert type(cell["value"]) is str  # Fixed UI labels only.


@pytest.mark.parametrize("case", _CASES)
def test_should_project_all_saved_methods_without_running_a_learner(case: str) -> None:
    source, resources = _saved(case), _saved(case, "resources")
    original = deepcopy(source)

    view = project_sensor_report(
        source,
        source_label=f"{case}-result.json",
        resources=resources,
        resource_label=f"{case}-resources.json",
    )

    assert view["schema_id"] == "sensor_inspector_v1"
    assert view["method"] == source["protocol"]["method"]
    assert [s["id"] for s in view["sections"]] == [
        "prearrival",
        "heldout",
        "probes",
        "retention",
        "monitoring",
        "resources",
        "controls",
    ]
    assert len(_section(view, "prearrival")["rows"]) == 12
    assert len(_section(view, "heldout")["rows"]) == 26
    assert len(_section(view, "probes")["rows"]) == (
        0 if case in ("scheduled_bp", "replay_bp") else 16
    )
    assert "UNVERIFIED" in _section(view, "resources")["note"]
    assert source == original
    _assert_pointers(view)


def test_should_keep_source_resources_and_cells_detached() -> None:
    source, resources = _saved(), _saved(kind="resources")
    view = project_sensor_report(
        source, source_label="source.json", resources=resources, resource_label="costs.json"
    )
    source["rows"][0]["scores"]["bce"] = 99
    resources["physical_work"]["update_calls_completed"] = 99

    assert view["source"]["rows"][0]["scores"]["bce"] != 99
    assert view["resources"]["physical_work"]["update_calls_completed"] == 28
    view["source"]["rows"][0]["scores"]["bce"] = 88
    assert _section(view, "prearrival")["rows"][0]["cells"][3]["value"] != 88


def test_should_show_original_default_without_inventing_optional_work() -> None:
    path = _ROOT / "artifacts/runs/r46-sensor-prearrival-20261017/native-result.json"
    view = project_sensor_report(json.loads(path.read_text()), source_label=path.name)

    assert _section(view, "heldout")["rows"] == []
    assert _section(view, "probes")["rows"] == []
    costs = _section(view, "resources")
    assert any(row["cells"][1]["missing"] for row in costs["rows"])
    assert not any(
        cell["pointer"] and cell["pointer"].startswith("/protocol/")
        for row in costs["rows"]
        for cell in row["cells"]
    )
    _assert_pointers(view)


def test_should_distinguish_recorded_null_from_unknown_and_keep_negative_loss() -> None:
    view = project_sensor_report(_saved(), source_label="source.json")
    monitoring = _section(view, "monitoring")
    cells = [cell for row in monitoring["rows"] for cell in row["cells"]]
    assert any(
        cell["pointer"] is not None and cell["value"] is None and not cell["missing"]
        for cell in cells
    )
    losses = [
        row["cells"][1]
        for row in _section(view, "retention")["rows"]
        if "old_A_loss" in row["cells"][0]["value"]
    ]
    assert losses[0]["value"] < 0
    assert any(
        cell["missing"] for row in _section(view, "controls")["rows"] for cell in row["cells"]
    )


def test_should_preserve_partial_readout_and_its_failure_costs() -> None:
    source = _saved()
    source["status"], source["reason"] = "failed", "fourth readout failed"
    progress = source["heldout_evaluation"]["progress"]
    progress.update(
        status="failed",
        reason="fourth readout failed",
        prediction_calls_attempted=4,
        prediction_calls_completed=3,
    )
    progress["checkpoints"] = progress["checkpoints"][:2]
    progress["checkpoints"][1]["status"] = "incomplete"
    del progress["checkpoints"][1]["scores"]["B"]
    source["heldout_evaluation"]["retention"] = None

    view = project_sensor_report(source, source_label="failed.json")

    rows = _section(view, "heldout")["rows"]
    assert len(rows) == 4 and rows[-1]["cells"][2]["missing"]
    assert rows[-1]["cells"][4]["value"] == "incomplete"
    assert view["status"] == "failed" and view["reason"] == "fourth readout failed"
    assert view["source"]["heldout_evaluation"]["progress"]["prediction_calls_attempted"] == 4
    _assert_pointers(view)


@pytest.mark.parametrize("status", ["incomplete", "failed", "running", "not_run"])
def test_should_preserve_probe_failure_or_not_run_without_fabricating_scores(status: str) -> None:
    source = _saved()
    source["learnability"]["status"] = status
    observation = source["learnability"]["observations"][0]
    observation["status"], observation["reason"] = status, "recorded reason"
    observation["probe"] = None
    source["learnability"]["observations"] = [observation]

    view = project_sensor_report(source, source_label="partial.json")

    cells = _section(view, "probes")["rows"][0]["cells"]
    assert all(cell["missing"] for cell in cells[1:4])
    assert cells[4]["value"] == status
    _assert_pointers(view)


def test_should_show_unbound_resources_without_relabelling_case_identity() -> None:
    resources = _saved("pc", "resources")
    resources["physical_work"]["escaped~/field"] = 7

    view = project_sensor_report(
        _saved("bp"),
        source_label="bp.json",
        resources=resources,
        resource_label="unrelated-pc.json",
    )

    assert view["method"] == "matched_bp"
    assert view["resources"]["case"] == "pc"
    cells = [cell for row in _section(view, "resources")["rows"] for cell in row["cells"]]
    assert any(
        cell["pointer"] == "/physical_work/escaped~0~1field" and cell["value"] == 7
        for cell in cells
    )
    _assert_pointers(view)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), object(), (1, 2), 10**400]
)
def test_should_refuse_non_json_or_nonfinite_values_even_in_unprojected_fields(value: Any) -> None:
    source = _saved()
    source["unprojected"] = value
    with pytest.raises(ValueError):
        project_sensor_report(source, source_label="source.json")


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/rows/0/feature_tick", True),
        ("/rows/0/scores/bce", -0.1),
        ("/rows/0/scores/accuracy", 1.1),
        ("/rows/0/label_tick", 0),
        ("/heldout_evaluation/progress/status", "invented"),
        ("/heldout_evaluation/progress/checkpoints/0/scores/A/accuracy", True),
        ("/heldout_evaluation/retention/metrics/bce/old_A_loss", "-0.1"),
        ("/learnability/observations/0/probe/checkpoints/0/updates", True),
        ("/sensor_monitoring/rows/0/decision/alarms", [0, 0, 0]),
        ("/diagnostics/prediction_calls_completed", -1),
        ("/protocol/method", []),
    ],
)
def test_should_refuse_malformed_displayed_evidence(pointer: str, value: Any) -> None:
    source = _saved()
    parent, key = pointer.rsplit("/", 1)
    target = _resolve(source, parent)
    target[int(key) if type(target) is list else key] = value
    with pytest.raises(ValueError):
        project_sensor_report(source, source_label="source.json")


def test_should_refuse_out_of_order_rows_and_false_complete_points() -> None:
    source = _saved()
    source["rows"][0], source["rows"][1] = source["rows"][1], source["rows"][0]
    with pytest.raises(ValueError, match="increase"):
        project_sensor_report(source, source_label="source.json")
    source = _saved()
    del source["heldout_evaluation"]["progress"]["checkpoints"][0]["scores"]["B"]
    with pytest.raises(ValueError, match="complete heldout"):
        project_sensor_report(source, source_label="source.json")


@pytest.mark.parametrize("source_label", ["", " a.json", "x" * 1025, None])
def test_should_refuse_invalid_labels(source_label: Any) -> None:
    with pytest.raises(ValueError):
        project_sensor_report(_saved(), source_label=source_label)


def test_should_require_paired_resource_label_and_refuse_bad_resource_counters() -> None:
    with pytest.raises(ValueError, match="together"):
        project_sensor_report(_saved(), source_label="source.json", resources={})
    resources = _saved(kind="resources")
    resources["physical_work"]["update_calls_completed"] = False
    with pytest.raises(ValueError):
        project_sensor_report(
            _saved(), source_label="source.json", resources=resources, resource_label="costs.json"
        )


def test_should_refuse_cyclic_or_excessive_bounded_source() -> None:
    source = _saved()
    source["cycle"] = source
    with pytest.raises(ValueError, match="bounds"):
        project_sensor_report(source, source_label="source.json")


def test_should_show_real_interrupted_probe_status_and_recorded_paid_costs():
    source = _saved()
    source.update(status="incomplete", reason="interrupted after first checkpoint")
    source["learnability"]["status"] = "partial"
    source["learnability"]["observations"] = source["learnability"]["observations"][:1]
    view = project_sensor_report(source, source_label="interrupted.json")
    assert len(_section(view, "probes")["rows"]) == 4
    cells = [c for row in _section(view, "resources")["rows"] for c in row["cells"]]
    assert any(c["pointer"] == "/learnability/charged_probe_updates" for c in cells)
    assert any(
        c["pointer"] == "/learnability/observations/0/probe/completed_updates" and c["value"] == 4
        for c in cells
    )
    assert any(c["pointer"] == "/learnability/observations/0/probe/elapsed_seconds" for c in cells)
    _assert_pointers(view)


def test_should_keep_contexts_separate_without_optional_planned_descriptor():
    source = _saved()
    del source["protocol"]["heldout"]
    view = project_sensor_report(source, source_label="descriptor-absent.json")
    assert [r["cells"][1]["value"] for r in _section(view, "heldout")["rows"][:2]] == ["A", "B"]
    _assert_pointers(view)


@pytest.mark.parametrize(
    "key,value",
    [
        ("rates", [-1, None, None]),
        ("rates", "bad"),
        ("rates", [None]),
        ("alarm_batches", [-1, 0, 0]),
        ("incident_onsets", [False, 0, 0]),
        ("monitoring_batches", -1),
    ],
)
def test_should_refuse_malformed_actual_monitoring_arrays(key, value):
    source = _saved()
    source["sensor_monitoring"]["false_alarms"]["pre_shift_A"][key] = value
    with pytest.raises(ValueError):
        project_sensor_report(source, source_label="bad-monitor.json")
