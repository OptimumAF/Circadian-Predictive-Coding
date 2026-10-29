"""Saved literal evidence, no learned states, fitting, source files or physics."""

from copy import deepcopy
import json
import sys

import pytest

from scripts import build_key_door_inspector as cli
from src.app import minigrid_affine_inspector as affine
from src.app.key_door_inspector import project_key_door_report
from test_affine_workflow_inspector import complete_report, episode
from test_recurring_affine_inspector import teacher
from test_key_door_inspector_files import Page
from test_minigrid_inspector_projection import pointer


def report():
    result = complete_report()
    result["definition"] = affine.A4_DEFINITION
    result["protocol"] = dict(
        definition=affine.A4_DEFINITION,
        source_origin="current_run_A4_TRAIN",
        historical_source_reads=0,
        teacher_step_limit=32,
        training=dict(
            definition=affine.A4.training_definition,
            label_delay=1,
            checkpoint_updates=[0, 2, 4, 6],
            fit_calls=6,
            fit_rows=90,
            score_calls=16,
            score_rows=144,
            arrival_row_exposures=54,
            catalogue_replay_row_exposures=36,
            distinct_source_rows=36,
            classes=7,
            wall_seconds=90,
            peak_rss_bytes=512 * 1024**2,
        ),
    )
    result["collection"] = []
    for cue in ("blue", "red"):
        row = teacher(cue)
        row.update(
            context="A",
            episode_id=f"minigrid-source-A4-train-{cue}",
            layout=deepcopy(affine.A4.layouts["A4"]),
        )
        result["collection"].append(row)
    result["source"] = dict(
        origin="current_run_A4_TRAIN",
        teacher_steps=18,
        environment_episodes=2,
        history_rows=18,
        collection_seconds=0.1,
    )
    training = result["training"]
    training.update(
        definition=affine.A4.training_definition,
        score_calls_completed=16,
        score_calls_attempted=16,
        score_rows_completed=144,
        score_rows_attempted=144,
    )
    for i, endpoint in enumerate(training["endpoints"]):
        point = affine.A4.points[i]
        endpoint["state"].update(
            training_examples=affine.A4.total_rows[i],
            heads=[None, None] if not point else [dict(fit_rows=affine.A4.fit_rows[point - 1])] * 2,
        )
        score = next(iter(endpoint["scores"].values()))
        endpoint["scores"] = {identity: deepcopy(score) for identity, _, _ in affine.A4.sources}
    training["stream"]["training_examples"] = 90
    training["catalogue"].update(
        affine.A4.catalogue, committed_rows=dict(arrivals=54, replay=36, native=90)
    )
    previous: dict[str, tuple[str, int]] = {}
    for i, wake in enumerate(training["catalogue"]["wakes"]):
        cue = ("blue", "red")[i % 2]
        identity = "A4-train" if i in (2, 3) else "A"
        current = f"minigrid-source-{identity}-{cue}:wake:{i}"
        old = previous.get(cue)
        spans = [dict(episode_id=old[0], context="A", start=0, stop=9)] if old else []
        start = 9 if old else 0
        spans.append(dict(episode_id=current, context="A", start=start, stop=start + 9))
        wake.update(
            current_episode_id=current,
            retained_episode_id=old[0] if old else None,
            retained_arrived_at=old[1] if old else None,
            source_spans=spans,
            replay_rows=affine.A4.fit_rows[i] - 9,
            bank_rows_after=min(i + 1, 2) * 9,
        )
        previous[cue] = (current, i + 1)
    template = result["games"]
    result["games"] = []
    for point, rows in zip((2, 4, 6), (18, 54, 90), strict=True):
        games = deepcopy(template)
        games.update(
            role=affine.A4.game_role, updates=point, source_fit_calls=point, source_fit_rows=rows
        )
        games["episodes"] = []
        for label, cue in affine.A4.slots:
            row = episode("B", cue)
            layout = affine.A4.layouts[label]
            row.update(context="A")
            row["environment"].update(
                context="A",
                layout=deepcopy(layout),
                episode_id=f"live-{affine.METHOD}-{point}-A-row{layout['branch_row']}-{cue}",
            )
            if point == 2 and label == "A4":
                row.update(success=False, **{"return": -1.0})
                row["environment"].update(terminated=True, truncated=False, total_return=-1.0)
                row["environment"]["transitions"][0].update(
                    reward=-1.0, terminated=True, truncated=False
                )
            games["episodes"].append(row)
        result["games"].append(dict(updates=point, games=games))
    result["readout_budget"] = dict(
        initial_limits=deepcopy(result["readout_budget"]["limits"]),
        limits=deepcopy(result["readout_budget"]["limits"]),
        updates=0,
        replay=0,
        started_at=9.0,
        observed_started_at=9.0,
        ended_at=12.0,
        sampled_peak_rss_bytes=100000,
    )
    result["summary"] = dict(
        complete=True,
        success_by_stage=[
            dict(old_A3=1.0, new_A4=0.0),
            dict(old_A3=1.0, new_A4=1.0),
            dict(old_A3=1.0, new_A4=1.0),
        ],
        acquisition_gain=1.0,
        old_retention_delta=0.0,
        return_gain=0.0,
        learning_recovery=[False, False],
    )
    return result


def failed_summary():
    return dict(
        complete=False,
        acquisition_gain=None,
        old_retention_delta=None,
        return_gain=None,
        learning_recovery=None,
    )


def project(value=None):
    return project_key_door_report(report() if value is None else value, source_label="literal A4")


def test_should_project_all_stages_failed_goals_and_source_linked_public_frames():
    original = report()
    projection = project(original)
    condition = projection["conditions"][0]
    assert projection["recipe"] == "a4_train"
    assert len(condition["episodes"]) == 12 and len(condition["curves"]) == 16
    assert [e["updates"] for e in condition["episodes"]] == [2] * 4 + [4] * 4 + [6] * 4
    assert [e["context"] for e in condition["episodes"]] == ["A3", "A3", "A4", "A4"] * 3
    assert [e["record"]["success"] for e in condition["episodes"]] == [True, True, False, False] + [
        True
    ] * 8
    for entry in condition["episodes"]:
        assert pointer(original, entry["pointer"]) == entry["record"]
        assert entry["frames"]
        for frame in entry["frames"]:
            for name in ("pointer", "decision_pointer", "transition_pointer", "features_pointer"):
                if frame.get(name) is not None:
                    pointer(original, frame[name])
    assert len(condition["receipts"]) == len(condition["wakes"]) == 6
    for cost in condition["costs"]:
        assert pointer(original, cost["pointer"]) == cost["value"]
    original["summary"]["acquisition_gain"] = 0
    assert projection["source"]["summary"]["acquisition_gain"] == 1


@pytest.mark.parametrize("prefix", [0, 1, 2, 3])
def test_should_keep_failed_aggregate_and_missing_future_slots_unknown(prefix):
    value = report()
    value.update(status="failed", games=value["games"][:prefix], summary=failed_summary())
    result = project(value)["conditions"][0]
    assert len(result["episodes"]) == 12
    assert sum(not e["derived"] for e in result["episodes"]) == 4 * prefix
    for entry in result["episodes"][4 * prefix :]:
        assert entry["pointer"] is None and entry["record"]["success"] is None


def test_should_preserve_failed_native_phase_and_refuse_later_work():
    value = report()
    failed = deepcopy(value["games"][1])
    failed["games"].update(status="failed", state_unchanged=None)
    value.update(
        status="failed", games=value["games"][:1], failed_phase=failed, summary=failed_summary()
    )
    result = project(value)["conditions"][0]
    assert all(e["pointer"].startswith("/failed_phase") for e in result["episodes"][4:8])
    assert all(e["derived"] for e in result["episodes"][8:])


def test_should_show_failed_collection_without_inventing_training():
    value = report()
    value.update(status="failed", source=None, training=None, games=[], summary=failed_summary())
    value["collection"] = [
        dict(
            status="construction_failed",
            context="A",
            cue="blue",
            role="train",
            episode_id="minigrid-source-A4-train-blue",
            failure_work_may_be_unobserved=True,
            closed=None,
        )
    ]
    result = project(value)["conditions"][0]
    assert all(e["derived"] for e in result["episodes"])
    assert all(c["accuracy"] is None for c in result["curves"])


@pytest.mark.parametrize(
    "fault",
    [
        "definition",
        "role",
        "stage",
        "identity",
        "layout",
        "seed",
        "clock",
        "budget",
        "source",
        "teacher",
        "span",
        "rows",
        "receipt",
        "summary",
        "recovery",
        "training",
        "nativecost",
        "phasecount",
    ],
)
def test_should_refuse_mixed_corrupt_or_unobserved_evidence(fault):
    value = report()
    native = value["games"][0]["games"]
    if fault == "definition":
        value["protocol"]["definition"] = "other"
    elif fault == "role":
        native["episodes"][0]["source_role"] = "final_test"
    elif fault == "stage":
        value["games"][0]["updates"] = 4
    elif fault == "identity":
        native["episodes"][0]["environment"]["episode_id"] = "replaced"
    elif fault == "layout":
        native["episodes"][0]["environment"]["layout"]["branch_row"] = 4
    elif fault == "seed":
        native["episodes"][0]["environment"]["seed"] = 19
    elif fault == "clock":
        value["readout_budget"]["observed_started_at"] = 10
    elif fault == "budget":
        value["readout_budget"]["limits"]["max_wall_seconds"] = 151
    elif fault == "source":
        value["source"]["source_sha256"] = "historical"
    elif fault == "teacher":
        value["collection"][0]["role"] = "probe_test"
    elif fault == "span":
        value["training"]["catalogue"]["wakes"][2]["retained_arrived_at"] = 3
    elif fault == "rows":
        native["source_fit_rows"] = 90
    elif fault == "receipt":
        value["training"]["committed_receipts"][0]["arrived_at"] = 0
    elif fault == "summary":
        value["summary"]["acquisition_gain"] = 0
    elif fault == "recovery":
        value["summary"]["learning_recovery"] = [True, True]
    elif fault == "training":
        value["training"]["source_unchanged"] = False
    elif fault == "nativecost":
        native["work"]["environment_steps_returned"] = 0
    else:
        value["games"] = value["games"][:2]
    with pytest.raises(ValueError):
        project(value)


def test_should_render_shared_template_and_export_exclusively_without_source_changes(
    monkeypatch, tmp_path, capsys
):
    source, destination = tmp_path / "source.json", tmp_path / "view.html"
    value = report()
    value["reason"] = "</script><img onerror=alert(1)>"
    raw = json.dumps(value).encode()
    source.write_bytes(raw)
    monkeypatch.setattr(
        sys, "argv", ["inspect", "--source-artifact", str(source), "--output", str(destination)]
    )
    cli.main()
    page = Page(destination.read_text(encoding="utf8"))
    payload = page.payload()
    assert payload["recipe"] == "a4_train" and len(payload["conditions"][0]["episodes"]) == 12
    html = destination.read_text(encoding="utf8")
    assert "function renderA4Overview" in html and "Native learning and retention" in html
    assert "</script><img" not in html and source.read_bytes() == raw
    capsys.readouterr()
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_reject_external_resources_for_a4():
    with pytest.raises(ValueError):
        project_key_door_report(
            report(), source_label="literal", resources={}, resource_label="extra"
        )


def test_should_accept_recovery_only_after_observed_old_task_loss_and_regain():
    value = report()
    for row in value["games"][1]["games"]["episodes"][:2]:
        row.update(success=False, **{"return": -1.0})
        row["environment"]["total_return"] = -1.0
        row["environment"]["transitions"][0]["reward"] = -1.0
    value["summary"]["success_by_stage"][1]["old_A3"] = 0.0
    value["summary"].update(
        old_retention_delta=-1.0, return_gain=1.0, learning_recovery=[True, True]
    )
    assert project(value)["source"]["summary"]["learning_recovery"] == [True, True]


def test_should_preserve_negative_teacher_return_without_admitting_training():
    value = report()
    row = value["collection"][0]
    row.update(total_return=-1.0, primary_error="teacher returned a failed task")
    row["transitions"][-1]["reward"] = -1.0
    value.update(
        status="failed",
        source=None,
        training=None,
        games=[],
        collection=[row],
        summary=failed_summary(),
    )
    projected = project(value)
    assert projected["source"]["collection"][0]["total_return"] == -1.0
    assert all(e["derived"] for e in projected["conditions"][0]["episodes"])
