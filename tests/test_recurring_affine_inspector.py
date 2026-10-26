"""Recorded literal evidence only; these fixtures produce no learned outcomes."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from scripts import build_key_door_inspector as cli
from src.app.key_door_inspector import project_key_door_report
from src.infra import key_door_inspector_files as files
from test_affine_workflow_inspector import complete_report as normal_report, episode
from test_key_door_inspector_files import Page
from test_minigrid_inspector_projection import pointer

DEFINITION = "portable_affine_recurring_train_v1"
TRAIN = "minigrid_arrived_recurring_capacity3_train_stream_v1"
G = dict(context="G_train", rotation=0, branch_row=5, swap_keys=False, wall_color="grey")
A = dict(context="A", rotation=0, branch_row=3, swap_keys=False, wall_color="grey")
POINTS = (0, 2, 4, 6, 8, 10)
ROWS = (0, 18, 54, 108, 180, 252)
FITS = (9, 9, 18, 18, 27, 27, 36, 36, 36, 36)
ACTUAL = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/installed-recurring-affine-20261024/actual.json"
)


def teacher(cue):
    view = [-1.0] * 148
    return dict(
        environment_id="Circadian-MiniGrid-CueKeyDoor-v1",
        encoding_id="minigrid_cue_key_door_public_image_direction_148_v1",
        context="G_train",
        cue=cue,
        role="train",
        episode_id=f"minigrid-source-G_train-{cue}",
        observation_dim=148,
        num_classes=7,
        seed=17,
        max_steps=16,
        action_names=["left", "right", "forward", "pickup", "drop", "toggle", "done"],
        status="complete",
        closed=True,
        reset_calls=1,
        environment_steps_attempted=9,
        environment_steps=9,
        terminated=True,
        truncated=False,
        total_return=0.5,
        failure_work_may_be_unobserved=False,
        layout=deepcopy(G),
        initial_observation=dict(
            image=[[[0, 0, 0] for _ in range(7)] for _ in range(7)],
            direction=0,
            mission="remember the cue, take its key, unlock the neutral door and reach the goal",
        ),
        transitions=[
            dict(
                observation=view,
                next_observation=view,
                action=0,
                attempted=True,
                applied=True,
                reward=0.5 if i == 8 else 0.0,
                terminated=i == 8,
                truncated=False,
            )
            for i in range(9)
        ],
    )


def report():
    r = normal_report()
    r.update(
        definition=DEFINITION,
        collection=[teacher("blue"), teacher("red")],
        collection_capture_truncated=False,
        attempted_phases=["collection", "training", "games"],
        workflow_budget=dict(
            max_wall_seconds=300,
            work_deadline_seconds=290,
            max_process_rss_bytes=512 * 1024**2,
            started_at=10.0,
            ended_at=15.0,
            elapsed_seconds=5.0,
            sampled_peak_rss_bytes=100000,
            scope="workflow",
        ),
    )
    t = r["training"]
    t["definition"] = TRAIN
    t["endpoints"] = [deepcopy(t["endpoints"][0]) for _ in POINTS]
    for point, rows, e in zip(POINTS, ROWS, t["endpoints"], strict=True):
        e["updates"] = point
        e["state"].update(
            completed_updates=point,
            training_examples=rows,
            heads=[None, None] if point == 0 else [dict(fit_rows=min(point // 2, 4) * 9)] * 2,
        )
        for cue in ("blue", "red"):
            e["scores"][f"minigrid-source-G_train-{cue}"] = deepcopy(
                next(iter(e["scores"].values()))
            )
    t["committed_receipts"] = [
        dict(
            observed_at=i,
            arrived_at=i + 1,
            applied_at=i + 1,
            update_number=i + 1,
            actor_version="untrained",
            learner_version=f"c-{i}",
            diagnostic=dict(value=0.0),
        )
        for i in range(10)
    ]
    t["stream"].update(
        completed_updates=10,
        training_examples=252,
        applied_updates=deepcopy(t["committed_receipts"]),
    )
    t["budget"].update(updates_completed=10, started_at=11.0, ended_at=12.0)
    for key in ("limits", "initial_limits"):
        t["budget"][key]["max_training_updates"] = 10
    t.update(
        score_calls_completed=48,
        score_rows_completed=432,
        score_calls_attempted=48,
        score_rows_attempted=432,
    )
    bank: dict[str, list[tuple[str, int, str]]] = {cue: [] for cue in ("blue", "red")}
    wakes = []
    for i, context in enumerate(c for c in ("A", "B", "C_train", "G_train", "A") for _ in range(2)):
        cue = ("blue", "red")[i % 2]
        identity = f"minigrid-source-{context}-{cue}:wake:{i}"
        old = list(bank[cue])
        new = (identity, i + 1, context)
        bank[cue] = [*old, new][-3:]
        wakes.append(
            dict(
                observed_at=i,
                arrived_at=i + 1,
                cue=cue,
                current_rows=9,
                replay_rows=FITS[i] - 9,
                current_episode_id=identity,
                retained_episode_id=old[-1][0] if old else None,
                retained_arrived_at=old[-1][1] if old else None,
                retained_episode_ids=[v[0] for v in old],
                retained_arrival_ticks=[v[1] for v in old],
                bank_episode_ids_after=[v[0] for q in ("blue", "red") for v in bank[q]],
                bank_rows_after=min(i + 1, 6) * 9,
                source_spans=[
                    dict(episode_id=v[0], context=v[2], start=9 * j, stop=9 * (j + 1))
                    for j, v in enumerate([*old, new])
                ],
            )
        )
    t["catalogue"].update(
        arrival_row_exposures=90,
        replay_row_exposures=162,
        fit_row_exposures=252,
        peak_bank_rows=54,
        input_array_bytes=598752,
        peak_joint_template_array_bytes=85536,
        peak_bank_observation_target_float64_equivalent_bytes=256176,
        wakes=wakes,
        committed_rows=dict(arrivals=90, replay=162, native=252),
    )
    games = r["games"]
    games.update(role="knownTRAIN_GA_return", updates=10, source_fit_calls=10, source_fit_rows=252)
    games["episodes"] = []
    for context in ("G_train", "A"):
        for cue in ("blue", "red"):
            e = episode("B", cue)
            e.update(context=context)
            e["environment"].update(
                context=context,
                layout=deepcopy(G if context == "G_train" else A),
                episode_id=f"live-arrived_public_cue_affine_fit-10-{context}-{cue}",
            )
            games["episodes"].append(e)
    r["readout_budget"].update(
        started_at=12.1, observed_budget_started_at=12.1, ended_at=14.0, elapsed_seconds=1.9
    )
    r["protocol"] = dict(
        definition=DEFINITION,
        new_layout=deepcopy(G),
        returned_layout=deepcopy(A),
        new_fit_calls=10,
        new_fit_rows=252,
        historical_model_reads=0,
        readout_training_updates=0,
        readout_replay_examples=0,
        max_workflow_seconds=300,
        work_deadline_seconds=290,
        max_process_rss_bytes=512 * 1024**2,
        training=dict(
            definition=TRAIN,
            label_delay=1,
            checkpoint_updates=list(POINTS),
            fit_calls=10,
            fit_rows=252,
            score_calls=48,
            score_rows=432,
            classes=7,
            seed=None,
            wall_seconds=90,
            peak_rss_bytes=512 * 1024**2,
            arrival_row_exposures=90,
            catalogue_replay_row_exposures=162,
            distinct_source_rows=72,
        ),
    )
    return r


def test_should_project_recurring_curves_full_games_costs_and_original_pointers():
    recorded = report()
    before = deepcopy(recorded)
    view = project_key_door_report(recorded, source_label="literal.json")
    assert view["source"] == recorded == before
    assert view["recipe"] == "recurring_train"
    c = view["conditions"][0]
    assert len(c["curves"]) == 48 and len(c["receipts"]) == len(c["wakes"]) == 10
    assert [(e["context"], e["record"]["cue"], e["updates"]) for e in c["episodes"]] == [
        (context, cue, 10) for context in ("G_train", "A") for cue in ("blue", "red")
    ]
    for curve in c["curves"]:
        assert pointer(view["source"], curve["pointer"])["ce"] == curve["ce"]
    for row in c["costs"] + c["proofs"]:
        assert pointer(view["source"], row["pointer"]) == row["value"]
    for e in c["episodes"]:
        assert pointer(view["source"], e["pointer"]) == e["record"] and len(e["frames"]) == 2
    assert any(
        row["pointer"] == "/collection/0/environment_steps" and row["value"] == 9
        for row in c["costs"]
    )
    assert any(row["pointer"] == "/workflow_budget/elapsed_seconds" for row in c["costs"])
    recorded["collection"][0]["transitions"][0]["action"] = 6
    assert view["source"] == before


def test_should_keep_partial_scores_and_spent_train_charges_without_inventing_games():
    r = report()
    r.update(
        status="failed",
        games=None,
        readout_budget=None,
        attempted_phases=["collection", "training"],
    )
    t = r["training"]
    t.update(
        status="failed",
        committed_receipts=None,
        receipt_capture_error="lost",
        source_unchanged=False,
    )
    t["catalogue"]["committed_rows"] = None
    t["endpoints"] = []
    for key in (
        "score_calls_completed",
        "score_rows_completed",
        "score_calls_attempted",
        "score_rows_attempted",
    ):
        t[key] = 0
    row = project_key_door_report(r, source_label="spent.json")["conditions"][0]
    assert not row["receipts"] and all(
        c["ce"] is c["accuracy"] is c["pointer"] is None for c in row["curves"]
    )
    assert all(e["derived"] and not e["frames"] for e in row["episodes"])
    assert any(
        c["pointer"] == "/training/budget/updates_completed" and c["value"] == 10
        for c in row["costs"]
    )


@pytest.mark.parametrize("construction_failed", [False, True])
def test_should_keep_failed_collection_unknown_and_refuse_to_invent_fit_or_native_work(
    construction_failed,
):
    r = report()
    r.update(
        status="failed",
        training=None,
        games=None,
        readout_budget=None,
        attempted_phases=["collection"],
    )
    if construction_failed:
        r["collection"] = [
            dict(
                status="construction_failed",
                context="G_train",
                cue="blue",
                role="train",
                episode_id="minigrid-source-G_train-blue",
                closed=None,
                environment_steps=None,
                environment_steps_attempted=None,
                failure_work_may_be_unobserved=True,
                reason="literal",
            )
        ]
    else:
        r["collection"] = r["collection"][:1]
        r["collection"][0].update(closed=False, cleanup_error="literal failed cleanup")
    row = project_key_door_report(r, source_label="collection-failed.json")["conditions"][0]
    assert not row["receipts"] and all(e["pointer"] is None for e in row["episodes"])
    assert any(c["pointer"].startswith("/collection/0/") for c in row["costs"])


def test_should_keep_negative_games_worse_confidence_and_explicit_unrun_slots():
    r = report()
    r["training"]["endpoints"][-1]["scores"]["minigrid-source-A-blue"]["ce"] = 4.0
    first = r["games"]["episodes"][0]
    first.update(success=False, **{"return": -0.1})
    first["environment"]["total_return"] = first["environment"]["transitions"][0]["reward"] = -0.1
    c = project_key_door_report(r, source_label="negative.json")["conditions"][0]
    assert c["episodes"][0]["record"]["success"] is False
    assert any(v["context"] == "A" and v["updates"] == 10 and v["ce"] == 4.0 for v in c["curves"])
    r["status"] = r["games"]["status"] = "failed"
    r["games"]["state_unchanged"] = False
    r["games"]["episodes"] = [
        first,
        dict(
            context="G_train",
            cue="red",
            source_role="train",
            status="not_run",
            predictions=[],
            environment=None,
            prediction_calls_attempted=0,
            prediction_calls_returned=0,
        ),
    ]
    r["games"]["work"].update(
        live_prediction_calls_attempted=1,
        live_prediction_calls_returned=1,
        environment_steps_attempted=1,
        environment_steps_returned=1,
        environment_resets_attempted=1,
        environment_resets_completed=1,
    )
    c = project_key_door_report(r, source_label="partial.json")["conditions"][0]
    assert c["episodes"][1]["pointer"] == "/games/episodes/1"
    assert "success" not in c["episodes"][1]["record"] and not c["episodes"][1]["frames"]
    assert all(e["derived"] for e in c["episodes"][2:])


@pytest.mark.parametrize(
    "fault",
    [
        "definition",
        "points",
        "rows",
        "head",
        "score_source",
        "receipt",
        "charge",
        "bank",
        "future_replay",
        "current_context",
        "fit_rows",
        "protocol",
        "teacher_role",
        "teacher_order",
        "teacher_step",
        "teacher_continuity",
        "teacher_cleanup",
        "teacher_geometry",
        "before_collection",
        "workflow_limit",
        "readout_clock",
        "game_role",
        "game_geometry",
        "after_failed",
        "point_type",
        "readout_type",
        "teacher_ending",
    ],
)
def test_should_refuse_changed_or_mixed_protocol_and_unobserved_evidence(fault):
    r = report()
    t = r["training"]
    if fault == "definition":
        r["definition"] = "portable_affine_train_and_play_v1"
    elif fault == "point_type":
        t["endpoints"][0]["updates"] = False
    elif fault == "readout_type":
        r["readout_budget"] = []
    elif fault == "teacher_ending":
        r["collection"][0]["terminated"] = False
    elif fault == "points":
        t["endpoints"][-1]["updates"] = 8
    elif fault == "rows":
        t["endpoints"][-1]["state"]["training_examples"] = 108
    elif fault == "head":
        t["endpoints"][-1]["state"]["heads"][0]["fit_rows"] = 45
    elif fault == "score_source":
        t["endpoints"][0]["scores"]["final-D"] = t["endpoints"][0]["scores"].pop(
            "minigrid-source-A-blue"
        )
    elif fault == "receipt":
        t["committed_receipts"][-1]["arrived_at"] = 9
    elif fault == "charge":
        t["budget"]["updates_completed"] = 6
    elif fault == "bank":
        t["catalogue"]["wakes"][-1]["bank_rows_after"] = 63
    elif fault == "future_replay":
        t["catalogue"]["wakes"][6]["retained_arrival_ticks"][-1] = 7
    elif fault == "current_context":
        t["catalogue"]["wakes"][6]["current_episode_id"] = "final-D"
    elif fault == "fit_rows":
        t["catalogue"]["fit_row_exposures"] = 108
    elif fault == "protocol":
        r["protocol"]["training"]["distinct_source_rows"] = 90
    elif fault == "teacher_role":
        r["collection"][0]["role"] = "probe_train"
    elif fault == "teacher_order":
        r["collection"].reverse()
    elif fault == "teacher_step":
        r["collection"][0]["environment_steps"] = 8
    elif fault == "teacher_continuity":
        r["collection"][0]["transitions"][1]["observation"] = [1.0] * 148
    elif fault == "teacher_cleanup":
        r["collection"][0]["closed"] = False
    elif fault == "teacher_geometry":
        r["collection"][0]["layout"]["branch_row"] = 3
    elif fault == "before_collection":
        r["collection"] = []
    elif fault == "workflow_limit":
        r["workflow_budget"]["max_wall_seconds"] = 600
    elif fault == "readout_clock":
        r["readout_budget"]["observed_budget_started_at"] = 13.0
    elif fault == "game_role":
        r["games"]["role"] = "private_F_then_returned_A"
    elif fault == "game_geometry":
        r["games"]["episodes"][0]["environment"]["layout"]["rotation"] = 1
    else:
        r["status"] = r["games"]["status"] = "failed"
        r["games"]["episodes"][0].update(status="failed", success=None)
    with pytest.raises(ValueError):
        project_key_door_report(r, source_label="bad.json")


def test_should_export_via_existing_cli_with_inert_source_and_exclusive_output(
    tmp_path, monkeypatch, capsys
):
    r = report()
    r["reason"] = '</script><img id="injected" onerror="alert(1)"> &'
    source, output = tmp_path / "recorded.json", tmp_path / "view.html"
    source.write_text(json.dumps(r), encoding="utf8")
    raw = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["inspector", "--source-artifact", str(source), "--output", str(output)]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["output"] == str(output.resolve())
    page = Page(output.read_text(encoding="utf8"))
    assert page.payload()["recipe"] == "recurring_train"
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert source.read_bytes() == raw
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, output)
    with pytest.raises(ValueError, match="owns"):
        project_key_door_report(r, source_label="recorded", resources={})


def test_should_read_existing_installed_actual_without_any_model_or_physics():
    if not ACTUAL.is_file():
        pytest.skip("local accepted127 source is not distributed")
    raw = ACTUAL.read_bytes()
    r = json.loads(raw)
    view = project_key_door_report(r, source_label=ACTUAL.name)
    c = view["conditions"][0]
    assert len(c["curves"]) == 48 and all(
        len(e["frames"]) == 10 and e["record"]["success"] for e in c["episodes"]
    )
    assert c["curves"][-8]["ce"] > c["curves"][8]["ce"]  # Original A confidence regression.
    assert view["source"] == r and ACTUAL.read_bytes() == raw
