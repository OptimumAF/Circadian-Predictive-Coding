"""Ordinary rehearsal command on real workflow loops and literal model/game ports."""

from contextlib import redirect_stdout
import io
import json
import pickle
import random
import sys
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_a4_adaptation as app
from scripts import run_minigrid_linear_fit_control as cli
from src.core.reference_controller import ReferencePolicy
from tests import test_a4_rehearsal_workflow as workflow
from tests import test_a4_fresh_probe_workflow as whole
from tests import test_retained_rehearsal_games as games

FLAG = "--context-linear-retained-rehearsal"
WORK = dict(cli_calls=0, workflows=0)


def invoke(patch, args):
    WORK["cli_calls"] += 1
    patch.setattr(sys, "argv", ["affine", *args])
    stream = io.StringIO()
    code: Any = 0
    with redirect_stdout(stream):
        try:
            cli.main()
        except SystemExit as error:
            code = error.code
    return code, json.loads(stream.getvalue()) if stream.getvalue() else None


def arguments(mode, *extra):
    return ["--example-affine-a4", FLAG, mode, *extra]


@pytest.mark.parametrize("mode", ["diagnostics", "games"])
def test_should_describe_exact_existing_recipe_without_work(monkeypatch, mode):
    def forbidden(*a, **kw):
        pytest.fail("describe performed work")

    monkeypatch.setattr(app, "run_fresh_adaptation", forbidden)
    monkeypatch.setattr(app, "_collect_fresh_a4", forbidden)
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    code, report = invoke(monkeypatch, arguments(mode, "--describe"))
    expected = app.describe_fresh_adaptation(
        reference_policy=ReferencePolicy("drift"),
        retained_rehearsal=True,
        native_games=mode == "games",
    )
    assert code == 0 and report == json.loads(json.dumps(expected))
    assert report["historical_source_reads"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "missing"],
        ["--retained-source", "missing"],
        ["--fit-updates", "2"],
        ["--fit-seconds", "90"],
        ["--ready-feedback"],
        ["--prequential-feedback"],
        ["--reference-retain-arrived"],
        ["--context-linear-reference", "drift"],
        ["--context-linear-controller", "retain_arrived"],
        ["--context-linear-rehearsal"],
        ["--context-linear-stream"],
        ["--describe", "--output", "unused"],
    ],
)
def test_should_refuse_conflicting_modes_and_paths_before_work(monkeypatch, extra):
    monkeypatch.setattr(app, "run_fresh_adaptation", lambda **kw: pytest.fail("work reached"))
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("reader reached"))
    code, report = invoke(monkeypatch, arguments("games", *extra))
    assert code == 2 and report is None


@pytest.mark.parametrize("args", [[FLAG, "games", "--describe"], arguments("other")])
def test_should_require_fresh_source_and_exact_declared_choice(monkeypatch, args):
    monkeypatch.setattr(app, "run_fresh_adaptation", lambda **kw: pytest.fail("work reached"))
    assert invoke(monkeypatch, args) == (2, None)


def bind_workflow(patch, mode):
    original = app.run_fresh_adaptation
    h = workflow.harness(patch)
    setup: Any = games.game_io
    setup.__wrapped__(patch)
    calls = []

    def run(**kw):
        assert kw == dict(
            reference_policy=ReferencePolicy("drift"),
            retained_rehearsal=True,
            native_games=mode == "games",
        )
        calls.append(kw)
        WORK["workflows"] += 1
        before_envs, before_steps = len(h.box["environments"]), h.box["teacher_calls"]
        try:
            return original(**kw)
        finally:
            whole.WORK["source_collections"] += len(h.box["environments"]) - before_envs
            whole.WORK["source_teacher_steps"] += h.box["teacher_calls"] - before_steps
            assert all(e.closed for e in h.box["environments"])

    patch.setattr(app, "run_fresh_adaptation", run)
    return calls


@pytest.mark.parametrize("mode", ["diagnostics", "games"])
def test_should_run_command_through_both_private_fits_and_capture(monkeypatch, tmp_path, mode):
    calls = bind_workflow(monkeypatch, mode)
    destination = tmp_path / "rehearsal.json"
    rng = pickle.dumps((random.getstate(), np.random.get_state()))
    code, r = invoke(monkeypatch, arguments(mode, "--output", str(destination)))
    assert code == 0 and r["status"] == "complete", r.get("reason")
    assert len(calls) == 1 and json.loads(destination.read_text()) == r
    assert destination.stat().st_size < 8 * 1024**2
    assert pickle.dumps((random.getstate(), np.random.get_state())) == rng
    q, budget = r["rehearsal"], r["readout_budget"]
    assert r["training"]["stream"]["completed_updates"] == 4
    assert r["training"]["stream"]["training_examples"] == 54
    assert r["training"]["source_unchanged"] and r["training"]["actor_unchanged"]
    assert q["comparison_work"] == dict(attempts=1, returned=1, unreturned_work_unknown=False)
    assert q["comparison"]["budget"]["updates_completed"] == 4
    assert [a["arm"] for a in q["comparison"]["arms"]] == ["current_only", "joint_A3_A4_F"]
    for arm in q["comparison"]["arms"]:
        assert arm["native_endpoint"]["completed_updates"] == 6
        assert arm["native_endpoint"]["training_examples"] == 108
        assert arm["probe"]["completed_training_examples"] == 54
    assert budget["updates"] == budget["replay"] == 0
    assert budget["started_at"] == budget["observed_started_at"]
    assert budget["initial_limits"] == budget["limits"]
    if mode == "games":
        assert q["native_game_work"] == dict(attempts=2, returned=2, unreturned_work_unknown=False)
        assert len(q["native_games"]) == 2
        assert all(
            e["environment"]["closed"] for a in q["native_games"] for e in a["games"]["episodes"]
        )
        assert all(a["games"]["state_unchanged"] for a in q["native_games"])
    else:
        assert "native_games" not in q and "native_game_work" not in q
    # This inherited summary describes source A3/A4 games, not private F retention.
    assert r["summary"]["learning_recovery"] == [None, None]


def test_should_capture_first_success_and_second_partial_without_retry(monkeypatch, tmp_path):
    calls = bind_workflow(monkeypatch, "games")
    original = games.FixedEnvironment.step

    def fail(owner, action):
        if "-joint-" in owner.episode_id and owner.cue == "red":
            games.WORK["failed_steps"] += 1
            raise RuntimeError("lost literal private game step")
        return original(owner, action)

    monkeypatch.setattr(games.FixedEnvironment, "step", fail)
    path = tmp_path / "partial.json"
    code, r = invoke(monkeypatch, arguments("games", "--output", str(path)))
    assert code == 2 and r["status"] == "failed" and len(calls) == 1
    assert json.loads(path.read_text()) == r and not r["summary"]["complete"]
    q = r["rehearsal"]
    assert q["comparison"]["budget"]["updates_completed"] == 4
    assert q["native_game_work"] == dict(attempts=2, returned=1, unreturned_work_unknown=True)
    assert q["native_games"][0]["games"]["status"] == "complete"
    partial = q["native_games"][1]["games"]["episodes"]
    assert [x["status"] for x in partial] == ["complete", "failed"] + ["not_run"] * 4
    assert all(e["environment"]["closed"] for e in partial[:2])


@pytest.mark.parametrize("mode", ["diagnostics", "games"])
@pytest.mark.parametrize("status", ["complete", "failed"])
def test_should_emit_zero_or_unknown_goals_with_original_status(
    monkeypatch, tmp_path, mode, status
):
    report = dict(
        status=status,
        goals=0 if status == "complete" else None,
        paid=dict(attempts=1, returned=0, unreturned_work_unknown=True),
    )
    calls = []

    def run(**kw):
        calls.append(kw)
        return report

    monkeypatch.setattr(app, "run_fresh_adaptation", run)
    path = tmp_path / "status.json"
    code, saved = invoke(monkeypatch, arguments(mode, "--output", str(path)))
    assert code == (0 if status == "complete" else 2)
    assert saved == json.loads(path.read_text()) == report and len(calls) == 1
    assert calls[0]["native_games"] is (mode == "games")


def test_should_refuse_existing_destination_before_work(monkeypatch, tmp_path):
    path = tmp_path / "existing.json"
    path.write_bytes(b"keep original\n")
    monkeypatch.setattr(app, "run_fresh_adaptation", lambda **kw: pytest.fail("work reached"))
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, arguments("games", "--output", str(path)))
    assert path.read_bytes() == b"keep original\n"


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_should_refuse_nonfinite_report_without_file_or_retry(monkeypatch, tmp_path, number):
    calls = []

    def run(**kw):
        calls.append(kw)
        return dict(status="failed", partial=[dict(value=number)])

    monkeypatch.setattr(app, "run_fresh_adaptation", run)
    path = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError):
        invoke(monkeypatch, arguments("games", "--output", str(path)))
    assert len(calls) == 1 and not path.exists()
