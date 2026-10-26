"""Saved scalar/native records only; literal traces are not learned outcomes."""

from copy import deepcopy
import json
from math import log
from pathlib import Path
import sys

import pytest

from scripts import build_key_door_inspector as cli
from src.app import minigrid_affine_inspector as affine
from src.app.key_door_inspector import project_key_door_report
from src.infra import key_door_inspector_files as files
from test_key_door_inspector_files import Page
from test_minigrid_inspector_projection import pointer

ROOT = Path(__file__).resolve().parents[1]
ACTUAL = ROOT / "artifacts/runs/portable-affine-workflow-20261024/actual.json"


def budget(updates=6, *, readout=False):
    limits = dict(
        max_training_updates=0 if readout else 6,
        max_wall_seconds=150 if readout else 90,
        max_process_rss_bytes=512 * 1024**2,
        max_replay_examples=0 if readout else None,
    )
    return dict(
        initial_limits=deepcopy(limits),
        limits=limits,
        updates_completed=0 if readout else updates,
        replay_examples_completed=0,
        started_at=10.0,
        ended_at=11.0,
        elapsed_seconds=1.0,
    )


def episode(context, cue):
    observation = [-1.0] * 148
    probabilities = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition = dict(
        observation=observation,
        action=0,
        attempted=True,
        applied=True,
        next_observation=observation,
        reward=0.5,
        terminated=True,
        truncated=False,
    )
    env = dict(
        environment_id="Circadian-MiniGrid-CueKeyDoor-v1",
        encoding_id="minigrid_cue_key_door_public_image_direction_148_v1",
        role="train",
        episode_id=f"live-{affine.METHOD}-6-{context}-{cue}",
        context=context,
        cue=cue,
        observation_dim=148,
        num_classes=7,
        seed=17,
        max_steps=16,
        action_names=["left", "right", "forward", "pickup", "drop", "toggle", "done"],
        status="complete",
        closed=True,
        reset_calls=1,
        environment_steps_attempted=1,
        environment_steps=1,
        transitions=[transition],
        terminated=True,
        truncated=False,
        total_return=0.5,
        layout=deepcopy(affine.LAYOUTS[context]),
    )
    return dict(
        context=context,
        cue=cue,
        source_role="train",
        status="complete",
        reason=None,
        predictions=[
            dict(
                observation=observation,
                features=observation + observation,
                probabilities=probabilities,
                action=0,
            )
        ],
        prediction_calls_attempted=1,
        prediction_calls_returned=1,
        environment=env,
        state_unchanged=True,
        success=True,
        **{"return": 0.5},
    )


def complete_report():
    endpoints = []
    for n, rows in zip(affine.POINTS, affine.TOTAL_ROWS, strict=True):
        endpoints.append(
            dict(
                updates=n,
                state=dict(
                    completed_updates=n,
                    training_examples=rows,
                    failed_work=None,
                    heads=[None, None] if n == 0 else [dict(fit_rows=n // 2 * 9)] * 2,
                ),
                scores={
                    identity: dict(ce=log(7), accuracy=1.0, actions=[0] * 9, expected=[0] * 9)
                    for identity, context, cue in affine.SOURCES
                },
            )
        )
    receipts = [
        dict(
            observed_at=i,
            arrived_at=i + 1,
            applied_at=i + 1,
            update_number=i + 1,
            actor_version="untrained",
            learner_version=f"candidate-{i}",
            diagnostic=dict(value=0.0),
        )
        for i in range(6)
    ]
    wakes = [
        dict(
            observed_at=i,
            arrived_at=i + 1,
            cue=("blue", "red")[i % 2],
            current_rows=9,
            replay_rows=affine.FIT_ROWS[i] - 9,
            bank_rows_after=min(i + 1, 4) * 9,
        )
        for i in range(6)
    ]
    train = dict(
        definition="minigrid_arrived_abc_capacity2_train_stream_v1",
        status="complete",
        reason=None,
        source_unchanged=True,
        actor_unchanged=True,
        endpoints=endpoints,
        committed_receipts=receipts,
        receipt_capture_error=None,
        budget=budget(),
        score_calls_completed=24,
        score_rows_completed=216,
        score_calls_attempted=24,
        score_rows_attempted=216,
        stream=dict(
            completed_updates=6,
            training_examples=108,
            score_forward_calls=0,
            evaluation_scope="training_only",
            num_classes=7,
            applied_updates=deepcopy(receipts),
        ),
        catalogue=dict(
            arrival_row_exposures=54,
            replay_row_exposures=54,
            fit_row_exposures=108,
            peak_bank_rows=36,
            input_array_bytes=256608,
            peak_joint_template_array_bytes=64152,
            peak_bank_observation_target_float64_equivalent_bytes=170784,
            wakes=wakes,
            committed_rows=dict(arrivals=54, replay=54, native=108),
        ),
        failure_work=None,
    )
    work = dict(
        raw_score_forwards_attempted=0,
        raw_score_forwards_completed=0,
        raw_score_rows_attempted=0,
        raw_score_rows_completed=0,
        live_sessions_started=0,
        live_prediction_calls_attempted=4,
        live_prediction_calls_returned=4,
        environment_steps_attempted=4,
        environment_steps_returned=4,
        environment_resets_attempted=4,
        environment_resets_completed=4,
    )
    games = dict(
        method=affine.METHOD,
        role="knownTRAIN_BC_afterC",
        updates=6,
        source_fit_calls=6,
        source_fit_rows=108,
        new_fit_calls=0,
        new_optimizer_updates=0,
        status="complete",
        reason=None,
        state_unchanged=True,
        scores=[],
        episodes=[episode(c, q) for c, q in affine.SLOTS],
        work=work,
    )
    return dict(
        definition=affine.DEFINITION,
        status="complete",
        reason=None,
        training=train,
        games=games,
        readout_budget=budget(readout=True),
    )


def test_should_project_literal_curves_receipts_and_native_frames_with_exact_sources():
    report = complete_report()
    before = deepcopy(report)
    view = project_key_door_report(report, source_label="literal.json")
    assert (
        view["schema_id"] == "minigrid_affine_inspector_v1" and view["source"] == before == report
    )
    row = view["conditions"][0]
    assert len(row["curves"]) == 24 and len(row["receipts"]) == len(row["wakes"]) == 6
    for curve in row["curves"]:
        saved = pointer(view["source"], curve["pointer"])
        assert saved["accuracy"] == curve["accuracy"] and saved["ce"] == curve["ce"]
    for cost in row["costs"] + row["proofs"]:
        assert pointer(view["source"], cost["pointer"]) == cost["value"]
    for game in row["episodes"]:
        assert pointer(view["source"], game["pointer"]) == game["record"]
        assert len(game["frames"]) == 2 and game["frames"][-1]["decision_pointer"] is None
        for frame in game["frames"]:
            assert pointer(view["source"], frame["pointer"]) == frame["observation"]
    report["games"]["episodes"][0]["predictions"][0]["action"] = 99
    assert view["source"] == before


def test_should_reuse_real_installed_result_without_any_model_or_physics():
    if not ACTUAL.is_file():
        pytest.skip("local accepted114source not distributed")
    raw = ACTUAL.read_bytes()
    report = json.loads(raw)
    view = project_key_door_report(report, source_label=ACTUAL.name)
    assert all(
        len(e["frames"]) == 10 and e["record"]["success"] for e in view["conditions"][0]["episodes"]
    )
    assert view["source"] == report and ACTUAL.read_bytes() == raw
    assert len(view["conditions"][0]["curves"]) == 24


def test_should_keep_missing_games_curves_and_costs_unknown_without_invented_pointers():
    report = dict(
        definition=affine.DEFINITION,
        status="failed",
        reason="source unavailable",
        training=None,
        games=None,
    )
    view = project_key_door_report(report, source_label="missing.json")
    row = view["conditions"][0]
    assert all(c["pointer"] is None and c["accuracy"] is c["ce"] is None for c in row["curves"])
    assert all(e["derived"] and e["pointer"] is None and not e["frames"] for e in row["episodes"])
    assert row["costs"] == row["receipts"] == row["proofs"] == []


def test_should_preserve_spent_budget_when_receipt_readback_failed():
    report = complete_report()
    report.update(status="failed", games=None, readout_budget=None)
    train = report["training"]
    train.update(
        status="failed",
        source_unchanged=False,
        actor_unchanged=None,
        committed_receipts=None,
        receipt_capture_error="unavailable",
        failure_work=dict(rows_attempted=27, unknown=True),
    )
    train["catalogue"]["committed_rows"] = None
    row = project_key_door_report(report, source_label="spent.json")["conditions"][0]
    assert row["receipts"] == []
    assert any(
        c["pointer"] == "/training/budget/updates_completed" and c["value"] == 6
        for c in row["costs"]
    )
    assert any(p["value"] is False for p in row["proofs"])
    assert any(
        c["pointer"] == "/training/failure_work/unknown" and c["value"] is True
        for c in row["costs"]
    )


def test_should_preserve_negative_native_and_training_outcomes():
    report = complete_report()
    score = next(iter(report["training"]["endpoints"][-1]["scores"].values()))
    score.update(accuracy=0.0, actions=[1] * 9)
    game = report["games"]["episodes"][0]
    game.update(success=False, **{"return": -0.1})
    game["environment"]["total_return"] = -0.1
    game["environment"]["transitions"][0]["reward"] = -0.1
    row = project_key_door_report(report, source_label="negative.json")["conditions"][0]
    assert row["episodes"][0]["record"]["success"] is False
    assert any(c["updates"] == 6 and c["accuracy"] == 0 for c in row["curves"])


def test_should_keep_a_failed_observed_game_and_unrun_slots_without_invented_results():
    report = complete_report()
    report["status"] = report["games"]["status"] = "failed"
    report["games"]["state_unchanged"] = None
    first = report["games"]["episodes"][0]
    first.update(
        status="failed",
        environment=None,
        success=None,
        state_unchanged=None,
        prediction_calls_returned=0,
        readback_error="native readback unavailable",
        **{"return": None},
    )
    first["predictions"][0].update(probabilities=None, action=None)
    report["games"]["episodes"] = [first]
    report["games"]["work"].update(
        live_prediction_calls_attempted=1,
        live_prediction_calls_returned=0,
        environment_steps_attempted=0,
        environment_steps_returned=0,
        failure_work_may_be_unobserved=True,
        environment_resets_attempted=1,
        environment_resets_completed=0,
    )
    row = project_key_door_report(report, source_label="failed.json")["conditions"][0]
    assert len(row["episodes"][0]["frames"]) == 1
    assert row["episodes"][0]["record"]["success"] is None
    assert all(
        e["derived"] and e["pointer"] is None and not e["frames"] for e in row["episodes"][1:]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "definition",
        "score_role",
        "accuracy",
        "native_rows",
        "head_rows",
        "point_order",
        "receipt_clock",
        "receipt_count",
        "catalogue_replay",
        "committed_replay",
        "source_proof",
        "budget_limit",
        "readout_charge",
        "game_role",
        "game_order",
        "native_action",
        "geometry",
        "native_work",
        "complete_missing",
        "environment_type",
        "prediction_type",
        "training_type",
        "budget_type",
    ],
)
def test_should_refuse_contradictory_or_relabelled_evidence(fault):
    report = complete_report()
    train = report["training"]
    games = report["games"]
    if fault == "definition":
        report["definition"] = "other"
    elif fault == "score_role":
        train["endpoints"][0]["scores"]["final-D"] = train["endpoints"][0]["scores"].pop(
            affine.SOURCES[0][0]
        )
    elif fault == "accuracy":
        next(iter(train["endpoints"][0]["scores"].values()))["accuracy"] = 0
    elif fault == "native_rows":
        train["endpoints"][-1]["state"]["training_examples"] = 90
    elif fault == "head_rows":
        train["endpoints"][-1]["state"]["heads"][0]["fit_rows"] = 18
    elif fault == "point_order":
        train["endpoints"][0]["updates"] = 2
    elif fault == "receipt_clock":
        train["committed_receipts"][0]["arrived_at"] = 0
    elif fault == "receipt_count":
        train["committed_receipts"].pop()
    elif fault == "catalogue_replay":
        train["catalogue"]["wakes"][-1]["replay_rows"] = 0
    elif fault == "committed_replay":
        train["catalogue"]["committed_rows"]["replay"] = 0
    elif fault == "source_proof":
        train["source_unchanged"] = False
    elif fault == "budget_limit":
        train["budget"]["limits"]["max_training_updates"] = 7
    elif fault == "readout_charge":
        report["readout_budget"]["updates_completed"] = 1
    elif fault == "game_role":
        games["episodes"][0]["source_role"] = "probe_test"
    elif fault == "game_order":
        games["episodes"].reverse()
    elif fault == "native_action":
        games["episodes"][0]["predictions"][0]["action"] = 1
    elif fault == "geometry":
        games["episodes"][0]["environment"]["layout"]["rotation"] = 0
    elif fault == "native_work":
        games["work"]["environment_steps_returned"] = 0
    elif fault == "environment_type":
        games["episodes"][0]["environment"] = []
    elif fault == "prediction_type":
        games["episodes"][0]["prediction_calls_attempted"] = "one"
    elif fault == "training_type":
        report.update(status="failed", training=None)
    elif fault == "budget_type":
        report.update(status="failed", games=None)
        train.update(status="failed", committed_receipts=None, budget=[])
    else:
        report["games"] = None
    with pytest.raises(ValueError):
        project_key_door_report(report, source_label="invalid.json")


def test_should_publish_through_public_cli_with_inert_source_and_exclusive_output(
    tmp_path, monkeypatch, capsys
):
    report = complete_report()
    report["note"] = '</script><img id="injected" src="x" onerror="alert(1)"> &'
    source = tmp_path / "saved.json"
    output = tmp_path / "inspector.html"
    source.write_text(json.dumps(report), encoding="utf8")
    raw = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["builder", "--source-artifact", str(source), "--output", str(output)]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["output"] == str(output.resolve())
    page = Page(output.read_text(encoding="utf8"))
    assert page.payload()["source"] == report
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert source.read_bytes() == raw
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, output)


def test_should_refuse_mixed_external_resources():
    with pytest.raises(ValueError, match="owns"):
        project_key_door_report(complete_report(), source_label="source", resources={})


@pytest.mark.parametrize(
    "raw",
    [
        '{"definition":"portable_affine_train_and_play_v1","status":"failed","status":"complete"}',
        '{"definition":"portable_affine_train_and_play_v1","status":"failed","x":NaN}',
    ],
    ids=["duplicate", "nonfinite"],
)
def test_should_refuse_ambiguous_json_before_publication(tmp_path, raw):
    source = tmp_path / "bad.json"
    source.write_text(raw, encoding="utf8")
    output = tmp_path / "absent" / "page.html"
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, output)
    assert not output.parent.exists()
