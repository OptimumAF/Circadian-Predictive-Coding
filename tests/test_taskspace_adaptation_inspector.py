"""Literal saved-DTO checks only: no learner, original dataset or native SDK."""

from copy import deepcopy
import json
import sys
from typing import Any

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.taskspace_adaptation_inspector_projection import (
    CPU,
    NATIVE,
    project_taskspace_adaptation_report,
)
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector
from test_key_door_inspector_files import Page


SCENES = ("pick17", "probe23", "pick19", "pick20", "push17")


def cpu():
    return dict(
        definition=CPU,
        mode="isolated_retained_adaptation",
        complete=False,
        error=None,
        stream=None,
        adaptation=None,
        native_outcomes=None,
    )


def phase(returning=False):
    scores = {
        name: dict(
            episode_id="literal-" + name,
            role="probe_test"
            if name == "fresh"
            else "probe_train"
            if name == "fresh_training"
            else "train",
            residual=dict(overall=dict(normalized_half_mse=i / 10)),
        )
        for i, name in enumerate(("fresh", "old_0", "old_1", "old_2", "old_3", "fresh_training"))
    }
    return dict(
        definition="continuous_retained_old_train_return_v1"
        if returning
        else "continuous_disposable_retained_nearest_probe_v1",
        complete=True,
        error=None,
        plan=dict(updates=1, training_id="old" if returning else "new", evaluation_id="score-only"),
        checkpoints=[
            dict(updates=7, complete=True, scores=deepcopy(scores)),
            dict(updates=8, complete=False, scores=scores),
        ],
        metrics=dict(
            fresh_acquisition_gain=-0.2,
            fresh_training_gain=0.1,
            old_retention_loss={f"old_{i}": i / 100 for i in range(4)},
        ),
        progress=dict(updates_completed=1, next_tick=2, replay_examples_completed=0),
        work=dict(
            training_attempts=1, training_rows_attempted=18, unreturned_append_possible=False
        ),
        retention=dict(
            source=dict(retained_rows=128, numeric_bytes=1024), candidate_after_return=None
        ),
        events=[dict(tick=0)],
        applied_updates=[
            dict(episode_id="new", observed_at=0, arrived_at=2, applied_at=2, update_number=1)
        ],
        native_updates=[],
    )


def report(native=False, pair=False):
    result = cpu()
    result["stream"] = dict(
        model_state=dict(candidate_training_examples=220, candidate_prototype_rows=128),
        experiment_state=dict(completed_updates=6, next_tick=18),
        events=[dict(tick=0, budget_updates=0, label_available=False)],
        applied=[dict(episode_id="source", observed_at=0, arrived_at=3, applied_at=3)],
        checkpoints=[],
    )
    result["adaptation"] = (
        dict(
            definition="continuous_retained_return_probe_v1",
            complete=False,
            error="late refusal",
            probe=phase(),
            return_phase=phase(True),
            work=dict(training_attempts=2),
        )
        if pair
        else phase()
    )
    if not native:
        return result
    return dict(
        definition=NATIVE,
        complete=False,
        error="wrapper failed",
        workflow=result,
        observed_native_phase="after_return" if pair else "after_probe",
        unobserved_native_phases=["before_probe", "after_probe"] if pair else ["before_probe"],
        cpu_endpoint_phases=["before_probe", "after_probe"],
        unreturned_native_work_unknown=True,
        native=[
            dict(
                scene=scene,
                report=dict(
                    complete=True,
                    any_success=True,
                    final_success=True,
                    sparse_return=1.0,
                    work=dict(controls=1),
                ),
            )
            for scene in SCENES
        ],
        outcomes={
            scene: dict(
                any_success=False,
                final_success=False,
                sparse_return=0.0,
                controls=0,
                control_physics_steps=None,
            )
            for scene in SCENES
        },
        attempted=list(SCENES),
        returned=list(SCENES),
    )


def cells(view):
    result = [view[key] for key in ("protocol_id", "status", "reason")]
    for section in view["sections"]:
        for row in section["rows"]:
            assert len(row["cells"]) == len(section["columns"])
            result.extend(row["cells"])
    return result


def at(view, pointer, origin="report"):
    return next(
        cell for cell in cells(view) if cell["pointer"] == pointer and cell["source"] == origin
    )


def resolve(root, pointer):
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        root = root[int(key)] if type(root) is list else root[key]
    return root


def assert_sources(view):
    for cell in cells(view):
        if cell["missing"]:
            assert cell["value"] is None and cell["pointer"] is None
        elif cell["pointer"] is not None:
            assert cell["value"] == resolve(
                view["source"] if cell["source"] == "report" else view["resources"], cell["pointer"]
            )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("pair", [False, True])
def test_should_route_every_phase_without_recomputing_or_promoting_claims(native, pair):
    saved = report(native, pair)
    original = deepcopy(saved)
    view = project_key_door_report(saved, source_label="literal.json")
    assert view == project_taskspace_adaptation_report(saved, source_label="literal.json")
    assert view["source"] == saved == original
    assert view["status"]["value"] is False
    base = "/workflow" if native else ""
    acquisition = base + "/adaptation" + ("/probe" if pair else "")
    assert at(view, acquisition + "/metrics/fresh_acquisition_gain")["value"] == -0.2
    for i in range(4):
        assert at(view, acquisition + f"/metrics/old_retention_loss/old_{i}")["value"] == i / 100
    rows = next(s for s in view["sections"] if s["id"] == "probe-scores")["rows"]
    assert len(rows) == 12
    assert {r["cells"][4]["value"] for r in rows} == {"train", "probe_train", "probe_test"}
    assert at(view, base + "/stream/experiment_state")["value"]["next_tick"] == 18
    assert at(view, acquisition + "/progress")["value"]["next_tick"] == 2
    if pair:
        assert at(view, base + "/adaptation/return_phase/complete")["value"] is True
    if native:
        assert at(view, "/native/0/report/final_success")["value"] is True
        assert at(view, "/outcomes/pick17/final_success")["value"] is False
        assert at(view, "/unobserved_native_phases")["value"] == saved["unobserved_native_phases"]
    assert_sources(view)


@pytest.mark.parametrize(
    "kind", ["missing", "null", "empty", "no-return", "partial-score", "partial-native"]
)
def test_should_preserve_incomplete_evidence_and_unknowns(kind):
    saved = cpu()
    if kind == "missing":
        del saved["adaptation"]
    elif kind == "empty":
        saved["adaptation"] = {}
    elif kind == "no-return":
        saved["adaptation"] = dict(
            definition="continuous_retained_return_probe_v1",
            complete=False,
            probe=phase(),
            return_phase=None,
        )
    elif kind == "partial-score":
        saved["adaptation"] = phase()
        saved["adaptation"]["checkpoints"] = [dict(updates=7, complete=False, scores=None)]
    elif kind == "partial-native":
        saved = dict(
            definition=NATIVE,
            complete=False,
            error=None,
            workflow=None,
            native=[dict(scene="pick17", report=dict(final_success=True))],
        )
    view = project_key_door_report(saved, source_label="partial.json")
    assert view["status"]["value"] is False
    assert any(c["missing"] for c in cells(view))
    if kind == "partial-native":
        outcomes = next(s for s in view["sections"] if s["id"] == "outcomes")
        assert len(outcomes["rows"]) == 5
        assert all(c["missing"] for row in outcomes["rows"] for c in row["cells"][1:])
    assert_sources(view)


def test_should_detach_null_zero_false_and_nested_resource_cells():
    saved = report(True)
    costs: dict[str, Any] = {"cost/with~escape": {"bytes": 0}, "unknown": None}
    view = project_key_door_report(
        saved, source_label="owned", resources=costs, resource_label="unverified"
    )
    assert at(view, "/outcomes/pick17/controls")["value"] == 0
    assert at(view, "/outcomes/pick17/final_success")["value"] is False
    assert at(view, "/outcomes/pick17/control_physics_steps")["missing"] is False
    assert at(view, "/outcomes/pick17/control_physics_steps")["value"] is None
    costcell = at(view, "/cost~1with~0escape", "resources")
    costs["cost/with~escape"]["bytes"] = 9
    assert costcell["value"]["bytes"] == 0 == view["resources"]["cost/with~escape"]["bytes"]
    view["resources"]["cost/with~escape"]["bytes"] = 8
    assert costcell["value"]["bytes"] == 0
    progress = at(view, "/workflow/adaptation/progress")
    saved["workflow"]["adaptation"]["progress"]["next_tick"] = 99
    view["source"]["workflow"]["adaptation"]["progress"]["next_tick"] = 88
    assert progress["value"]["next_tick"] == 2


@pytest.mark.parametrize(
    "path,value",
    [
        (("complete",), 1),
        (("error",), []),
        (("workflow",), []),
        (("observed_native_phase",), "before_probe"),
        (("unobserved_native_phases",), ["after_probe"] * 2),
        (("native",), [dict(scene="probe23", report={})]),
        (("native",), [dict(scene="pick17", report={})] * 6),
        (("outcomes",), {"unknown": {}}),
        (("outcomes", "pick17", "controls"), True),
        (("native", 0, "report", "final_success"), "yes"),
        (("workflow", "mode"), "wrong"),
        (("workflow", "stream", "events"), [{}] * 129),
        (("workflow", "adaptation", "checkpoints"), [{}] * 3),
        (("workflow", "adaptation", "progress", "next_tick"), -1),
        (("workflow", "adaptation", "checkpoints", 0, "scores", "fresh", "role"), "final_test"),
        (("workflow", "adaptation", "checkpoints", 0, "scores", "old_0", "role"), "probe_test"),
        (("workflow", "adaptation", "checkpoints", 0, "scores", "fresh_training", "role"), "train"),
        (("workflow", "adaptation", "metrics", "fresh_training_gain"), False),
        (("workflow", "adaptation", "definition"), "other"),
        (("attempted",), ["alien"]),
    ],
)
def test_should_refuse_malformed_headers_bounds_roles_and_counts(path, value):
    saved = report(True)
    node = saved
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        project_key_door_report(saved, source_label="malformed")


@pytest.mark.parametrize("native", [False, True])
def test_should_keep_legacy_four_scene_admission_unchanged(native):
    old: dict[str, Any] = dict(
        definition="taskspace_memory_native_v1" if native else "taskspace_memory_demo_v1",
        mode="native_fresh_rollout" if native else "offline_recorded_replay",
        complete=False,
    )
    view = project_key_door_report(old, source_label="old")
    assert len(next(s for s in view["sections"] if s["id"] == "outcomes")["rows"]) == 4
    old["native"] = [dict(scene="probe23", report={})]
    with pytest.raises(ValueError, match="scene"):
        project_key_door_report(old, source_label="old")


def test_should_build_existing_file_and_cli_routes_with_inert_labels(tmp_path, monkeypatch, capsys):
    from scripts.build_key_door_inspector import main

    danger = '</script><img id="injected" src="https://example.invalid"> &'
    saved = report(True, True)
    saved["error"] = danger
    saved["workflow"]["provenance"] = {"path": "D:/never-read", "url": "https://example.invalid"}
    source, output = tmp_path / "source.json", tmp_path / "view.html"
    costs = tmp_path / "costs.json"
    source.write_text(json.dumps(saved), encoding="utf8")
    costs.write_text('{"unknown":null}', encoding="utf8")
    original = source.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspector",
            "--source-artifact",
            str(source),
            "--resource-artifact",
            str(costs),
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["bytes"] == output.stat().st_size
    page = Page(output.read_text(encoding="utf8"))
    view = page.payload()
    assert view["source"] == saved
    assert "<" not in "".join(page.payload_parts)
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert all(
        not (attrs.get(key) or "").startswith(("http:", "https:", "//"))
        for _, attrs in page.tags
        for key in ("src", "href")
    )
    assert "UNVERIFIED" in next(s for s in view["sections"] if s["id"] == "resources")["title"]
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, output)
    with pytest.raises(ValueError):
        write_key_door_inspector(source, source)
    assert_sources(view)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_should_keep_shared_json_envelope_security(value):
    saved = cpu()
    saved["extra"] = value
    with pytest.raises(ValueError):
        project_key_door_report(saved, source_label="bad")


def test_should_render_source_labels_as_inert_text():
    danger = '</script><img id="injected">'
    view = project_key_door_report(cpu(), source_label=danger)
    page = Page(render_key_door_inspector(view).decode("utf8"))
    assert page.payload()["source_label"] == danger
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)


@pytest.mark.parametrize("native", [False, True])
def test_should_show_complete_as_a_recorded_flag_without_inventing_native_phases(native):
    saved = report(native, True)
    saved["complete"] = True
    saved["error"] = None
    view = project_key_door_report(saved, source_label="complete")
    assert view["status"]["value"] is True
    scope = next(s for s in view["sections"] if s["id"] == "scope")
    assert "UNVERIFIED" in scope["note"]
    assert "recovery" in scope["note"]
    assert not any(s["id"] == "curves" for s in view["sections"])
    assert_sources(view)


@pytest.mark.parametrize(
    "arguments", [dict(resources={}), dict(resource_label="unpaired"), dict(source_label=" ")]
)
def test_should_refuse_unpaired_resources_and_invalid_labels(arguments):
    kwargs: dict[str, Any] = dict(source_label="paired")
    kwargs.update(arguments)
    with pytest.raises(ValueError):
        project_key_door_report(cpu(), **kwargs)
