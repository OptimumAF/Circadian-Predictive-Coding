"""Inspect cached memory-demo metadata without learners or recomputation.

Three preserved public reports exercise routing. Tiny mutations cover source
pointers, unknown values, detached cells and inert HTML publication.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.taskspace_memory_inspector_projection import project_taskspace_memory_report
from src.infra.key_door_inspector_files import render_key_door_inspector
from test_key_door_inspector_files import Page


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("pick19", "pick17", "pick20", "push17")
REPORTS = (
    ROOT / "artifacts/runs/cpu-memory-package-20261021/installed-report.json",
    ROOT / "artifacts/runs/native-memory-demo-20261021/native-fixed.json",
    ROOT / "artifacts/runs/native-memory-demo-20261021/native-report.json",
)
SECTIONS = (
    "outcomes",
    "historical",
    "baseline",
    "native",
    "events",
    "releases",
    "checkpoints",
    "memory",
    "work",
    "provenance",
)


def minimal(*, native=False):
    return dict(
        definition="taskspace_memory_native_v1" if native else "taskspace_memory_demo_v1",
        mode="native_fresh_rollout" if native else "offline_recorded_replay",
        complete=False,
        error=None,
    )


def section(view, identity):
    return next(item for item in view["sections"] if item["id"] == identity)


def cells(view):
    result = [view[key] for key in ("protocol_id", "status", "reason")]
    for item in view["sections"]:
        for row in item["rows"]:
            assert len(row["cells"]) == len(item["columns"])
            result.extend(row["cells"])
    return result


def resolve(value, pointer):
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if type(value) is list else value[token]
    return value


def at(view, pointer, origin="report"):
    return next(
        cell for cell in cells(view) if cell["pointer"] == pointer and cell["source"] == origin
    )


def assert_sources(view):
    for cell in cells(view):
        assert set(cell) == {"value", "pointer", "source", "missing"}
        if cell["missing"]:
            assert cell["value"] is None and cell["pointer"] is None
        elif cell["pointer"] is None:
            assert type(cell["value"]) is str
        else:
            source = view["source"] if cell["source"] == "report" else view["resources"]
            assert cell["value"] == resolve(source, cell["pointer"])


@pytest.mark.parametrize(
    "path", REPORTS, ids=("installed-cpu", "native-complete", "native-preserved-failure")
)
def test_should_route_each_saved_public_report_without_changing_any_evidence(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    original = deepcopy(report)
    direct = project_taskspace_memory_report(report, source_label=path.name)
    assert direct == project_key_door_report(report, source_label=path.name)
    assert direct["schema_id"] == "taskspace_memory_inspector_v1"
    assert direct["source"] == original == report
    assert path.read_bytes() == raw
    assert [item["id"] for item in direct["sections"]] == list(SECTIONS)
    assert direct["status"]["pointer"] == "/complete"
    assert direct["status"]["value"] is report["complete"]
    assert direct["protocol_id"]["pointer"] == "/definition"
    assert [row["cells"][0]["value"] for row in section(direct, "outcomes")["rows"]] == list(SCENES)
    assert_sources(direct)


@pytest.mark.parametrize("native", [False, True])
def test_should_keep_four_unknown_scene_slots_when_optional_evidence_is_absent(native):
    view = project_taskspace_memory_report(minimal(native=native), source_label="partial.json")
    assert view["status"]["value"] is False
    assert len(section(view, "outcomes")["rows"]) == 4
    assert all(
        cell["missing"] for row in section(view, "outcomes")["rows"] for cell in row["cells"][1:]
    )
    assert section(view, "events")["rows"] == section(view, "releases")["rows"] == []
    assert all(row["cells"][1]["missing"] for row in section(view, "baseline")["rows"])
    assert_sources(view)


def test_should_preserve_present_null_and_zero_separately_from_missing_values():
    report = minimal()
    report.update(elapsed_seconds=0.0, stream={"elapsed_seconds": None}, work={"queries": 0})
    view = project_taskspace_memory_report(report, source_label="zero-null.json")
    assert at(view, "/error") == dict(value=None, pointer="/error", source="report", missing=False)
    assert at(view, "/elapsed_seconds")["value"] == 0.0
    assert at(view, "/stream/elapsed_seconds")["value"] is None
    assert at(view, "/stream/elapsed_seconds")["missing"] is False
    assert at(view, "/work")["value"] == {"queries": 0}
    assert any(cell["missing"] for cell in cells(view))
    assert_sources(view)


def test_should_detach_input_snapshot_and_object_valued_cells_independently():
    report = minimal()
    report["work"] = {"native": {"queries": 1}}
    original = deepcopy(report)
    view = project_taskspace_memory_report(report, source_label="owned.json")
    cell = at(view, "/work")
    report["work"]["native"]["queries"] = 8
    assert view["source"] == original and cell["value"] == original["work"]
    view["source"]["work"]["native"]["queries"] = 9
    assert cell["value"]["native"]["queries"] == 1
    cell["value"]["native"]["queries"] = 10
    assert view["source"]["work"]["native"]["queries"] == 9


def test_should_label_separate_resources_unverified_and_resolve_escaped_keys():
    costs: dict[str, Any] = {"path/with~characters": {"seconds": 0.0}, "unknown": None}
    expected = deepcopy(costs)
    view = project_taskspace_memory_report(
        minimal(), source_label="report.json", resources=costs, resource_label="costs.json"
    )
    resources = section(view, "resources")
    assert "UNVERIFIED" in resources["title"] or "UNVERIFIED" in resources["note"]
    assert at(view, "/path~1with~0characters", "resources")["value"] == {"seconds": 0.0}
    assert at(view, "/unknown", "resources")["missing"] is False
    costs["path/with~characters"]["seconds"] = 9
    assert view["resources"] == expected
    assert_sources(view)


def test_should_keep_replay_checks_historical_scores_and_raw_native_returns_distinct():
    report = minimal()
    report.update(
        replays=[
            dict(
                scene="pick19",
                complete=True,
                native_actions_exact=True,
                selected_indices_exact=True,
                rows=1,
            )
        ],
        recorded_outcomes=[
            dict(
                scene="pick19",
                seed=19,
                outcomes=dict(final_success=False, any_success=False, sparse_return=0.0),
            )
        ],
        provenance={"recorded_baseline_198": {"pick19": {"final_success": False}}},
    )
    view = project_taskspace_memory_report(report, source_label="replay.json")
    assert "Replay complete" in section(view, "outcomes")["columns"]
    assert "Final success" not in section(view, "outcomes")["columns"]
    assert at(view, "/replays/0/complete")["value"] is True
    assert at(view, "/recorded_outcomes/0/outcomes")["value"]["final_success"] is False
    assert at(view, "/provenance/recorded_baseline_198/pick19")["value"]["final_success"] is False
    native = minimal(native=True)
    native["native"] = [dict(scene="pick19", report=dict(complete=True, final_success=True))]
    partial = project_taskspace_memory_report(native, source_label="raw-return.json")
    assert at(partial, "/native/0/report/final_success")["value"] is True
    assert all(
        cell["missing"] for row in section(partial, "outcomes")["rows"] for cell in row["cells"][1:]
    )
    assert partial["status"]["value"] is False
    assert_sources(partial)


def test_should_preserve_paid_waiting_events_release_clocks_and_memory_identity():
    report = minimal()
    report["stream"] = dict(
        events=[
            dict(tick=0, packet_index=0, label_available=False, budget_updates=0, applied=[]),
            dict(tick=2, packet_index=0, label_available=True, budget_updates=1, applied=["paid"]),
        ],
        applied=[
            dict(
                event_id="consented-release",
                episode_id="original-B",
                observed_at=0,
                arrived_at=2,
                applied_at=2,
                update_number=1,
            )
        ],
        checkpoints=[
            dict(
                tick=2,
                phase="after_release",
                episode_id="original-B",
                prototype_rows=65,
                completed_updates=2,
                known_episode_ids=["A", "original-B"],
            )
        ],
        model_state=dict(candidate_publications=2, candidate_prototype_rows=65),
    )
    view = project_taskspace_memory_report(report, source_label="paid-partial.json")
    assert at(view, "/stream/events/0/applied")["value"] == []
    assert at(view, "/stream/events/0/budget_updates")["value"] == 0
    assert at(view, "/stream/applied/0/arrived_at")["value"] == 2
    assert at(view, "/stream/checkpoints/0/known_episode_ids")["value"] == ["A", "original-B"]
    assert at(view, "/stream/model_state")["value"]["candidate_prototype_rows"] == 65
    assert_sources(view)


def test_should_render_hostile_labels_and_recorded_strings_as_inert_data():
    danger = '</script><img id="injected" src="https://example.invalid" onerror="alert(1)"> &'
    report = minimal()
    report.update(error=danger, provenance={"notice": danger})
    view = project_taskspace_memory_report(
        report, source_label=danger, resources={"message": danger}, resource_label=danger
    )
    page = Page(render_key_door_inspector(view).decode("utf8"))
    assert page.payload()["source"]["error"] == danger
    assert page.payload()["source_label"] == danger
    assert "<" not in "".join(page.payload_parts)
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert all(
        not (attrs.get(key) or "").startswith(("http:", "https:", "//"))
        for _, attrs in page.tags
        for key in ("src", "href")
    )


@pytest.mark.parametrize(
    "fault",
    [
        "family",
        "mode",
        "mismatched-family",
        "complete-null",
        "complete-zero",
        "error-shape",
        "replay-shape",
        "native-shape",
        "stream-shape",
        "work-shape",
        "events-overflow",
        "checkpoints-overflow",
        "applied-overflow",
        "replays-overflow",
        "native-overflow",
        "history-overflow",
        "duplicate-replay",
        "duplicate-native",
        "duplicate-history",
        "nonfinite",
        "nonjson",
        "negative-count",
        "boolean-count",
    ],
)
def test_should_refuse_malformed_unbounded_or_ambiguous_metadata(fault):
    report = minimal()
    if fault == "family":
        report["definition"] = "invented"
    elif fault == "mode":
        report["mode"] = "invented"
    elif fault == "mismatched-family":
        report["definition"] = "taskspace_memory_native_v1"
    elif fault == "complete-null":
        report["complete"] = None
    elif fault == "complete-zero":
        report["complete"] = 0
    elif fault == "error-shape":
        report["error"] = {"message": "bad shape"}
    elif fault == "replay-shape":
        report["replays"] = {}
    elif fault == "native-shape":
        report["native"] = {}
    elif fault == "stream-shape":
        report["stream"] = []
    elif fault == "work-shape":
        report["work"] = []
    elif fault in ("events-overflow", "checkpoints-overflow", "applied-overflow"):
        key = fault.split("-")[0]
        report["stream"] = {key: [{}] * (33 if key == "applied" else 129)}
    elif fault in ("replays-overflow", "native-overflow", "history-overflow"):
        key = "recorded_outcomes" if fault == "history-overflow" else fault.split("-")[0]
        report[key] = [dict(scene="pick19")] * 5
    elif fault in ("duplicate-replay", "duplicate-native", "duplicate-history"):
        key = {
            "duplicate-replay": "replays",
            "duplicate-native": "native",
            "duplicate-history": "recorded_outcomes",
        }[fault]
        report[key] = [dict(scene="pick19"), dict(scene="pick19")]
    elif fault == "nonfinite":
        report["provenance"] = {"bad": float("nan")}
    elif fault == "nonjson":
        report["provenance"] = {"bad": (1, 2)}
    elif fault == "negative-count":
        report["replays"] = [dict(scene="pick19", rows=-1)]
    else:
        report["stream"] = {"checkpoints": [dict(prototype_rows=True)]}
    with pytest.raises(ValueError):
        project_taskspace_memory_report(report, source_label="refused.json")
