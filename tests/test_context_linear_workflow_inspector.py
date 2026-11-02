"""Pure recorded JSON; no learner construction, task execution or inference."""

from copy import deepcopy
import json
from typing import Any

import pytest

from src.app.context_linear_workflow_inspector_projection import (
    DEFINITION,
    PAIRED_DEFINITION,
    project_context_linear_workflow_report,
)
from src.app.key_door_inspector import project_key_door_report
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector
from test_taskspace_memory_inspector import assert_sources, at, section


def fixture():
    def episode(phase, count, success):
        return dict(
            phase=phase,
            policy_version="v2",
            concept="password",
            complete=True,
            reads=[dict(generation=1, actor_version=phase, completed_updates=count)],
            episode=dict(
                complete=True,
                success=success,
                stopped_reason="success" if success else "wrong_workflow",
                training_updates=0,
                external_actions=0,
                steps=[
                    dict(
                        attempt=1,
                        observation_state=dict(phase="ready", revision=0),
                        effect_state=dict(phase="committed", revision=1),
                        result=dict(selected_id="inspect"),
                        transition=dict(status="done"),
                        capabilities_at_effect=dict(version=0),
                    )
                ],
            ),
        )

    report: dict[str, Any] = dict(
        definition=DEFINITION,
        method="linear_fit",
        complete=True,
        linear_train_update="minimum_change",
        error=None,
        source_unchanged=False,
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        budget=dict(
            updates_completed=12,
            started_at=1.0,
            last_clock=4.0,
            limits=dict(max_training_updates=12, max_wall_seconds=30),
        ),
        paired_serving=dict(
            definition=PAIRED_DEFINITION,
            complete=True,
            prior_training=dict(updates=12, rows=72),
            publication_tick=10,
            rollback_generation=2,
            rollback_equal_source=True,
            candidate_state=dict(completed_updates=18, training_examples=120),
            guard_report=dict(candidate_revision=[4, 0], decision=dict(accepted=False)),
            learning_boundary_guard=dict(decision=dict(accepted=True)),
            episodes=[episode("published", 16, True), episode("rollback", 12, False)],
            live_serving=dict(
                records=[
                    dict(
                        tick=0,
                        complete=True,
                        policy_version="v1",
                        concept="invoice",
                        episode=dict(
                            complete=True,
                            success=False,
                            stopped_reason="permission_denied",
                            steps=[],
                        ),
                    )
                ]
            ),
            training_work=dict(attempted_updates=6, attempted_rows=48, returned_updates=6),
            budget=dict(updates=6, started_at=2.0, last_clock=4.0),
            native_calls=[dict(rows=9, complete=False)],
        ),
    )
    return report


def test_dispatch_preserves_full_parent_negative_rows_model_identity_and_all_pointers():
    report = fixture()
    before = deepcopy(report)
    view = project_key_door_report(report, source_label="paired.json")
    assert view == project_context_linear_workflow_report(report, source_label="paired.json")
    assert view["source"] == report == before
    assert view["view_kind"] == "ordinary_linear_paired"
    assert at(view, "/paired_serving/candidate_state/completed_updates")["value"] == 18
    assert [row["cells"][-1]["value"] for row in section(view, "reads")["rows"]] == [16, 12]
    assert [row["cells"][4]["value"] for row in section(view, "outcomes")["rows"]] == [True, False]
    assert section(view, "live")["rows"][0]["cells"][4]["value"] is False
    assert at(view, "/paired_serving/guard_report")["value"]["decision"]["accepted"] is False
    assert len(section(view, "steps")["rows"]) == 2
    assert_sources(view)


def test_partial_missing_null_zero_false_remain_distinct_without_inferred_outcomes():
    report = dict(
        definition=DEFINITION,
        method="linear_fit",
        complete=False,
        paired_serving=None,
        final_examples=0,
        error=None,
        source_unchanged=False,
    )
    view = project_key_door_report(report, source_label="partial")
    completion = next(
        row["cells"][1]
        for row in section(view, "identity")["rows"]
        if row["cells"][0]["value"] == "paired_serving/complete"
    )
    assert completion["missing"] and completion["pointer"] is None
    assert at(view, "/error")["value"] is None and not at(view, "/error")["missing"]
    assert at(view, "/source_unchanged")["value"] is False
    assert at(view, "/final_examples")["value"] == 0
    assert all(section(view, key)["rows"] == [] for key in ("outcomes", "reads", "steps", "live"))
    assert_sources(view)


def test_null_record_arrays_remain_null_and_do_not_fabricate_success_counts():
    report = fixture()
    report["paired_serving"].update(episodes=None, live_serving=None)
    view = project_key_door_report(report, source_label="partial")
    assert at(view, "/paired_serving/episodes")["value"] is None
    assert section(view, "outcomes")["rows"] == []
    assert section(view, "live")["rows"] == []


def test_cells_and_source_detach_independently_and_separate_resources_stay_unverified():
    report = fixture()
    resources: dict[str, Any] = {"cost/~": dict(seconds=None)}
    view = project_context_linear_workflow_report(
        report, source_label="paired", resources=resources, resource_label="separate"
    )
    assert "UNVERIFIED" in section(view, "resources")["title"]
    assert at(view, "/cost~1~0", "resources")["value"] == {"seconds": None}
    report["paired_serving"]["candidate_state"]["completed_updates"] = 999
    resources["cost/~"]["seconds"] = 3
    view["source"]["paired_serving"]["candidate_state"]["completed_updates"] = 1000
    assert at(view, "/paired_serving/candidate_state/completed_updates")["value"] == 18
    assert at(view, "/cost~1~0", "resources")["value"] == {"seconds": None}


@pytest.mark.parametrize(
    "path,value",
    [
        (("method",), "bp"),
        (("complete",), 1),
        (("source_unchanged",), 1),
        (("paired_serving",), []),
        (("paired_serving", "definition"), "guarded_context_serving_v1"),
        (("paired_serving", "candidate_state", "completed_updates"), True),
        (("paired_serving", "training_work", "attempted_rows"), -1),
        (("paired_serving", "episodes"), [None]),
        (("paired_serving", "episodes", 0, "phase"), "private_return"),
        (("paired_serving", "episodes", 0, "reads", 0, "completed_updates"), False),
        (("paired_serving", "episodes", 0, "episode", "success"), 1),
        (("paired_serving", "live_serving", "records", 0, "tick"), True),
        (("paired_serving", "native_calls"), "bad"),
        (("paired_serving", "linear_train_update"), "other"),
        (("budget", "updates_completed"), 1.5),
        (("budget", "last_clock"), float("nan")),
        (("budget", "last_clock"), True),
    ],
)
def test_malformed_known_fields_are_refused(path, value):
    report = fixture()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        project_key_door_report(report, source_label="bad")


@pytest.mark.parametrize("key", ["episodes", "native_calls"])
def test_record_bounds_are_enforced(key):
    report = fixture()
    report["paired_serving"][key] = [{}] * 129
    with pytest.raises(ValueError, match="records"):
        project_key_door_report(report, source_label="oversized")


def test_ordinary_nonpaired_family_does_not_silently_become_a_paired_report():
    report = fixture()
    del report["paired_serving"]
    with pytest.raises(ValueError, match="paired_serving"):
        project_key_door_report(report, source_label="ordinary")


def test_reserved_probe_dispatch_priority_is_unchanged(monkeypatch):
    from src.app import reserved_probe_inspector_projection as reserved

    marker = dict(original_reserved_path=True)
    monkeypatch.setattr(reserved, "project_reserved_probe_report", lambda *args, **kwargs: marker)
    report = fixture()
    report.update(reserved_probe_requested=True, paired_serving="not the selected family")
    assert project_key_door_report(report, source_label="reserved") is marker


def test_bad_labels_and_resource_associations_are_refused():
    with pytest.raises(ValueError, match="label"):
        project_context_linear_workflow_report(fixture(), source_label=" ")
    with pytest.raises(ValueError, match="label"):
        project_context_linear_workflow_report(fixture(), source_label="ok", resources={})


def test_shared_renderer_and_exclusive_file_export_keep_original_source_and_inert_text(tmp_path):
    report = fixture()
    report["error"] = "</script><script>unsafe()</script>"
    source, output = tmp_path / "source.json", tmp_path / "view.html"
    raw = json.dumps(report).encode()
    source.write_bytes(raw)
    write_key_door_inspector(source, output)
    page = output.read_bytes()
    assert page == render_key_door_inspector(
        project_key_door_report(report, source_label=source.name)
    )
    assert b"<script>unsafe()" not in page
    assert source.read_bytes() == raw
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, output)
    assert output.read_bytes() == page and source.read_bytes() == raw


def test_duplicate_input_keys_are_refused_before_output(tmp_path):
    source, output = tmp_path / "bad.json", tmp_path / "view.html"
    source.write_text('{"complete": false, "complete": true}')
    with pytest.raises(ValueError, match="duplicate"):
        write_key_door_inspector(source, output)
    assert not output.exists()
