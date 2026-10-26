"""Saved final D playback: honest failures, paid prefixes and exclusive offline IO."""

import builtins
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

import scripts.build_key_door_inspector as cli
from src.app.key_door_inspector import project_key_door_report
from src.app.minigrid_saved_fifo_inspector import project_minigrid_heldout_fifo_report
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector
from test_minigrid_inspector_projection import pointer

ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/fifo-heldout-d-20261023/actual.json"
)
SHA = "7388f72578d25a99d0c27747669e23142ce28843ef102d93d2d9e125d98bcf18"


@pytest.fixture(scope="module")
def final_record():
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    return json.loads(raw)


def test_should_preserve_both_negative_games_and_exact_frame_decision_cost_proof_paths(
    final_record,
):
    before = deepcopy(final_record)
    projected = project_key_door_report(final_record, source_label="final D")
    assert projected["schema_id"] == "minigrid_heldout_fifo_inspector_v1"
    assert projected["source"] == before == final_record
    assert projected["source"] is not final_record
    condition = projected["conditions"][0]
    assert condition["curves"] == [] and condition["source_role"] == "final_test"
    assert condition["status"] == "complete"
    assert [e["cue"] for e in condition["episodes"]] == ["blue", "red"]
    for e in condition["episodes"]:
        assert pointer(before, e["pointer"]) == e["record"]
        assert e["context"] == "D" and e["updates"] == 128
        assert e["record"]["success"] is False and e["record"]["return"] == 0
        assert len(e["frames"]) == 17
        assert e["frames"][-1]["decision_pointer"] is None
        for f in e["frames"]:
            assert pointer(before, f["pointer"]) == f["observation"]
            for k in ("decision_pointer", "transition_pointer", "features_pointer"):
                if f[k] is not None:
                    assert pointer(before, f[k]) is not None
    for row in condition["proofs"] + condition["costs"]:
        assert pointer(before, row["pointer"]) == row["value"]
    assert projected["source"]["matched_queries"] == 0
    projected["source"]["memory"]["entries"] = 0
    assert final_record == before


def test_should_keep_failed_proofs_and_unrun_outcomes_unknown(final_record):
    r = deepcopy(final_record)
    r.update(complete=False, source_unchanged=False, error="failed before helper")
    for key in (
        "games",
        "query_matches",
        "matched_queries",
        "total_queries",
        "python_rng_unchanged",
        "memory",
        "origin",
    ):
        del r[key]
    p = project_key_door_report(r, source_label="failed")
    c = p["conditions"][0]
    assert c["status"] == "incomplete"
    assert next(x for x in c["proofs"] if x["label"] == "source_unchanged")["value"] is False
    unknown = next(x for x in c["proofs"] if x["label"] == "python_rng_unchanged")
    assert unknown["value"] is None and unknown["pointer"] is None
    for e in c["episodes"]:
        assert e["derived"] and e["pointer"] is None and e["frames"] == []
        assert e["record"]["success"] is None and e["record"]["return"] is None


def test_should_keep_paid_first_game_when_second_not_run(final_record):
    r = deepcopy(final_record)
    r.update(complete=False, error="second game admission failed")
    for k in ("query_matches", "matched_queries", "total_queries"):
        del r[k]
    g = r.pop("games")
    r["failure_report"] = g
    g.update(status="failed", state_unchanged=False)
    e = g["episodes"][1]
    e.update(
        status="not_run",
        environment=None,
        predictions=[],
        prediction_calls_attempted=0,
        prediction_calls_returned=0,
        success=None,
        state_unchanged=None,
        **{"return": None},
    )
    w = g["work"]
    w["failure_work_may_be_unobserved"] = True
    for key in (
        "live_prediction_calls_attempted",
        "live_prediction_calls_returned",
        "environment_steps_attempted",
        "environment_steps_returned",
        "internal_projection_rows_returned",
        "history_live_feature_rows",
    ):
        w[key] = 16
    for key in ("internal_projection_array_bytes_returned", "history_live_feature_array_bytes"):
        w[key] = 16 * 296 * 8
    for key in (
        "environment_resets_attempted",
        "environment_resets_completed",
        "history_episodes_started",
    ):
        w[key] = 1
    c = project_key_door_report(r, source_label="partial")["conditions"][0]
    assert c["episodes"][0]["record"]["success"] is False
    assert len(c["episodes"][0]["frames"]) == 17
    assert c["episodes"][1]["record"]["success"] is None
    assert c["episodes"][1]["pointer"] == "/failure_report/episodes/1"
    assert c["episodes"][1]["frames"] == []


@pytest.mark.parametrize(
    "path,value",
    [
        ("/role", "train"),
        ("/source_sha256", "0" * 64),
        ("/benchmark_updates", 160),
        ("/benchmark_rows", 1440),
        ("/new_training_updates", 1),
        ("/probe_updates", True),
        ("/replay_examples", 1),
        ("/teacher_episodes", 1),
        ("/promotions", 1),
        ("/source_unchanged", False),
        ("/memory/entries", 35),
        ("/readout_budget/limits/max_training_updates", 1),
        ("/readout_budget/limits/max_wall_seconds", 240),
        ("/readout_budget/last_clock", 0),
        ("/elapsed_seconds", 1),
        ("/sampled_process_rss/peak_bytes", 2**30),
        ("/games/role", "probe_test"),
        ("/games/native_training_examples", 1440),
        ("/games/state_unchanged", False),
        ("/games/work/raw_score_rows_completed", 1),
        ("/games/work/live_prediction_calls_returned", 31),
        ("/games/work/history_live_feature_rows", 31),
        ("/games/episodes/0/context", "A"),
        ("/games/episodes/0/environment/layout/rotation", 0),
        ("/games/episodes/0/environment/layout/swap_keys", 0),
        ("/games/episodes/0/predictions/0/action", 1),
        ("/games/episodes/0/predictions/0/features/0", 0.123),
        ("/games/episodes/0/environment/transitions/0/applied", False),
        ("/games/episodes/0/success", True),
        ("/games/episodes/0/return", 1),
        ("/query_matches/0/step", 1),
        ("/total_queries", 31),
        ("/matched_queries", 1),
    ],
)
def test_should_refuse_corrupt_or_mixed_final_evidence(final_record, path, value):
    r = deepcopy(final_record)
    tokens = path.split("/")[1:]
    target = r
    for token in tokens[:-1]:
        target = target[int(token)] if type(target) is list else target[token]
    key = int(tokens[-1]) if type(target) is list else tokens[-1]
    target[key] = value
    with pytest.raises(ValueError):
        project_minigrid_heldout_fifo_report(r, source_label="corrupt")


def test_should_render_inert_final_page_and_refuse_resource_mixing(final_record):
    label = "</script><script>alert('bad')</script>"
    p = project_key_door_report(final_record, source_label=label)
    html = render_key_door_inspector(p).decode("utf8")
    assert label not in html and "\\u003c/script" in html
    assert "minigrid_heldout_fifo_inspector_v1" in html
    assert "Final D outcome inspector" in html and "Recorded stored-key matches" in html
    assert "https://" not in html and "fetch(" not in html
    with pytest.raises(ValueError):
        project_key_door_report(
            final_record, source_label="x", resources={}, resource_label="extra"
        )


def test_should_build_through_cli_without_models_and_preserve_exclusive_source(
    final_record, tmp_path, monkeypatch
):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "minigrid", "gymnasium"} or name.startswith(
            "src.adapters"
        ):
            raise AssertionError("readonly projector imported native code: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    source = tmp_path / "input.json"
    raw = json.dumps(final_record).encode()
    source.write_bytes(raw)
    output = tmp_path / "final.html"
    monkeypatch.setattr(
        sys, "argv", ["build", "--source-artifact", str(source), "--output", str(output)]
    )
    cli.main()
    assert output.is_file() and source.read_bytes() == raw
    with pytest.raises((ValueError, FileExistsError)):
        write_key_door_inspector(source, output)
    with pytest.raises(ValueError):
        write_key_door_inspector(source, source)


@pytest.mark.parametrize(
    "raw", ['{"complete":true,"complete":false}', '{"complete":NaN}', '{"complete":Infinity}']
)
def test_should_refuse_nonunique_nonfinite_json_before_output(tmp_path, raw):
    source = tmp_path / "invalid.json"
    source.write_text(raw)
    output = tmp_path / "absent" / "view.html"
    with pytest.raises(ValueError):
        write_key_door_inspector(source, output)
    assert not output.parent.exists()


def test_should_preserve_partial_prediction_failure_and_all_paid_prefix_costs(final_record):
    r = deepcopy(final_record)
    r.update(complete=False, source_unchanged=None, error="native prediction failed")
    for key in ("query_matches", "matched_queries", "total_queries"):
        del r[key]
    g = r.pop("games")
    r["failure_report"] = g
    g.update(status="failed", state_unchanged=False)
    blue, red = g["episodes"]
    blue.update(
        status="failed",
        success=None,
        state_unchanged=False,
        prediction_calls_attempted=3,
        prediction_calls_returned=2,
        predictions=blue["predictions"][:3],
    )
    blue["predictions"][-1].update(probabilities=None, action=None)
    env = blue["environment"]
    env.update(
        status="failed",
        truncated=False,
        environment_steps=2,
        environment_steps_attempted=2,
        transitions=env["transitions"][:2],
    )
    red.update(
        status="not_run",
        predictions=[],
        environment=None,
        success=None,
        state_unchanged=None,
        prediction_calls_attempted=0,
        prediction_calls_returned=0,
        **{"return": None},
    )
    work = g["work"]
    work.update(
        failure_work_may_be_unobserved=True,
        live_prediction_calls_attempted=3,
        live_prediction_calls_returned=2,
        environment_resets_attempted=1,
        environment_resets_completed=1,
        environment_steps_attempted=2,
        environment_steps_returned=2,
        internal_projection_rows_returned=2,
        internal_projection_array_bytes_returned=2 * 296 * 8,
        history_episodes_started=1,
        history_live_feature_rows=3,
        history_live_feature_array_bytes=3 * 296 * 8,
    )
    c = project_key_door_report(r, source_label="partial")["conditions"][0]
    assert len(c["episodes"][0]["frames"]) == 3
    assert c["episodes"][0]["record"]["success"] is None
    assert c["episodes"][0]["frames"][-1]["decision_pointer"].endswith("/predictions/2")
    assert next(p for p in c["proofs"] if p["label"] == "source_unchanged")["value"] is None
    for row in c["costs"]:
        assert pointer(r, row["pointer"]) == row["value"]
    work["history_live_feature_array_bytes"] -= 1
    with pytest.raises(ValueError, match="row/byte"):
        project_key_door_report(r, source_label="corrupt partial bytes")


def test_should_refuse_oversized_pretty_final_source_before_output(final_record, tmp_path):
    source = tmp_path / "large.json"
    source.write_text(" " * (4 * 1024**2) + json.dumps(final_record), encoding="utf8")
    output = tmp_path / "absent" / "view.html"
    with pytest.raises(ValueError, match="4 MiB"):
        write_key_door_inspector(source, output)
    assert not output.parent.exists()
