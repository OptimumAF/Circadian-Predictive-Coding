"""F execution ownership/guards; scripted games establish no native success."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import minigrid_affine_probe_games as games
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from test_after_c_linear_games import state as before_state
from test_context_linear_causal_games import PhaseEnvironment
from test_retained_context_linear_games import budget

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "artifacts/runs/catalogue-capacity-new-c-20261024/actual.json"
AFTER = ROOT / "artifacts/runs/affine-fresh-task-probe-20261024/actual.json"


def state(name):
    before = before_state()
    if name == "before_probe":
        return before
    return replace(
        before,
        completed_updates=8,
        training_examples=126,
        heads=tuple(replace(h, fit_rows=9, singular_values=(0.0,) * 9) for h in before.heads),
    )


@pytest.mark.parametrize("name", ["before_probe", "after_probe"])
def test_should_bind_fixed_f_probe_test_geometry_and_native_counter_without_authority(
    monkeypatch, name
):
    owner, saved = ContextLinearLearner(), state(name)

    def run(source, memory, **options):
        assert source is owner and memory == saved and memory is not saved
        assert options["updates"] == saved.completed_updates
        assert options["sources"] == options["layouts"] == ()
        assert options["require_initial_source"] is options["public_history"] is True
        assert options["representation_feature_dim"] == 298
        assert options["return_metadata"]["new_fit_calls"] == 0
        assert [
            (x.context, x.branch_row, x.rotation, x.wall_color, cue, role)
            for x, cue, role in options["live_plan"]
        ] == [
            ("F-blue", 2, 0, "grey", "blue", "probe_test"),
            ("F-red", 2, 0, "grey", "red", "probe_test"),
        ]
        options["extra_check"]()
        assert options["memory_copier"](owner.snapshot_state()).heads == (None, None)
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert (
        evaluation.evaluate_context_linear_probe_checkpoint(
            owner, saved, phase=name, budget=budget(), check_boundary=lambda: None
        )["status"]
        == "complete"
    )


@pytest.mark.parametrize("fault", ["phase", "count", "rows", "head", "failed", "owner"])
def test_should_refuse_wrong_f_recipe_before_the_loop(monkeypatch, fault):
    saved, name = state("after_probe"), "after_probe"
    owner: Any = ContextLinearLearner()
    if fault == "phase":
        name = "train"
    elif fault == "count":
        saved = replace(saved, completed_updates=7)
    elif fault == "rows":
        saved = replace(saved, training_examples=125)
    elif fault == "head":
        saved = replace(saved, heads=before_state().heads)
    elif fault == "failed":
        saved = replace(saved, failed_work=(1, 9, 0, 0, True))
    else:
        owner = object()
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_probe_checkpoint(
            owner, saved, phase=name, budget=budget(), check_boundary=lambda: None
        )


@pytest.mark.parametrize("fault", ["counter", "clock", "boolean_counter", "limit"])
def test_should_refuse_changed_readout_authority_inside_the_loop(monkeypatch, fault):
    session = budget()

    def run(*a, **options):
        if fault == "clock":
            session.clock = lambda: 1.0
        elif fault == "limit":
            object.__setattr__(session.budget, "max_training_updates", 1)
        else:
            session.updates_completed = False if fault == "boolean_counter" else 1
        options["extra_check"]()
        pytest.fail("budget mutation admitted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_probe_checkpoint(
            ContextLinearLearner(),
            state("before_probe"),
            phase="before_probe",
            budget=session,
            check_boundary=lambda: None,
        )


def test_should_run_both_endpoints_through_original_scripted_loop_without_fitting(monkeypatch):
    monkeypatch.setattr(PhaseEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    source = ContextLinearLearner()
    for name in ("before_probe", "after_probe"):
        result = evaluation.evaluate_context_linear_probe_checkpoint(
            source, state(name), phase=name, budget=budget(), check_boundary=lambda: None
        )
        assert result["status"] == "complete" and result["state_unchanged"] is True
        assert result["work"]["live_prediction_calls_returned"] == 4
        assert [e["context"] for e in result["episodes"]] == ["F-blue", "F-red"]
        assert all(e["environment"]["closed"] for e in result["episodes"])
        assert all(e.role == "probe_test" for e in PhaseEnvironment.work["environments"])
        assert "probe_test" in result["public_history"]["storage_scope"]
    assert source.snapshot_state().heads == (None, None)


def actual():
    if not AFTER.is_file():
        pytest.skip("accepted local113 report unavailable")
    return json.loads(games._qualified_probe_bytes(AFTER))


def test_should_restore_actual_paid_endpoints_without_loading_training_labels(monkeypatch):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN decoded"))
    before = cli._after_c_affine_state(json.loads(cli._qualified_after_c_bytes(BEFORE)))
    after = games._after_probe_state(actual())
    assert (before.completed_updates, before.training_examples) == (6, 108)
    assert (after.completed_updates, after.training_examples) == (8, 126)
    assert [h.fit_rows for h in after.heads] == [9, 9]


@pytest.mark.parametrize("fault", ["status", "fork", "source", "clock", "charge", "state"])
def test_should_refuse_incomplete_or_changed_paid_probe_evidence(fault):
    report = actual()
    if fault == "status":
        report["probe"]["status"] = "failed"
    elif fault == "fork":
        report["source_fork_calls"] = 2
    elif fault == "source":
        report["source_bindings"]["after_c_bytes"] = 0
    elif fault == "clock":
        report["budget"]["ended_at"] = report["budget"]["started_at"] - 1
    elif fault == "charge":
        report["budget"]["updates_completed"] = 0
    else:
        report["disposable_state"]["training_examples"] = 108
    with pytest.raises(ValueError):
        games._after_probe_state(report)


def setup_literal(monkeypatch):
    raw = dict(budget=dict(original_clock=11, updates_completed=2), source_fork_calls=1)
    monkeypatch.setattr(games, "_qualified_probe_bytes", lambda *a: json.dumps(raw).encode())
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: b"{}")
    monkeypatch.setattr(games, "_after_probe_state", lambda *a: state("after_probe"))
    monkeypatch.setattr(cli, "_after_c_affine_state", lambda *a: state("before_probe"))


def phase(name):
    return dict(
        status="complete",
        probe_phase=name,
        state_unchanged=True,
        episodes=[
            dict(status="complete", success=False, environment=dict(closed=True)) for _ in range(2)
        ],
    )


def test_should_accept_completed_negative_games_and_keep_original_sunk_budget(monkeypatch):
    setup_literal(monkeypatch)
    calls = []

    def run(source, saved, *, phase, budget, check_boundary):
        check_boundary()
        assert source.completed_updates == 0 and budget.updates_completed == 0
        calls.append((phase, saved.completed_updates))
        return globals()["phase"](phase)

    monkeypatch.setattr(games, "evaluate_context_linear_probe_checkpoint", run)
    result = games.run_saved_affine_probe_games("after", "before")
    assert result["status"] == "complete" and result["source_unchanged"] is True
    assert calls == [("before_probe", 6), ("after_probe", 8)]
    assert (
        result["budget"]["updates_completed"] == result["budget"]["replay_examples_completed"] == 0
    )
    assert result["original_probe_budget"] == dict(original_clock=11, updates_completed=2)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "value",
        "equal_container",
        "source",
        "budget",
        "provenance",
        "partial",
        "native_source",
        "equal_phase",
        "equal_prefix",
    ],
)
def test_should_refuse_mutation_or_incompletion_and_preserve_detached_paid_prefix(
    monkeypatch, fault
):
    setup_literal(monkeypatch)
    first = phase("before_probe")
    calls = []

    def run(source, saved, *, phase, budget, check_boundary):
        calls.append(phase)
        if phase == "before_probe":
            if fault == "incomplete":
                first["episodes"][0]["environment"]["closed"] = False
            return first
        if fault == "value":
            first["episodes"][0]["success"] = True
        elif fault == "equal_container":
            first["episodes"] = deepcopy(first["episodes"])
        elif fault == "source":
            monkeypatch.setattr(games, "_qualified_probe_bytes", lambda *a: b"replaced")
        elif fault == "budget":
            budget.replay_examples_completed = 1
            budget.started_at = -1.0
        elif fault == "native_source":
            source.restore_state(saved)
        elif fault in ("provenance", "equal_phase", "equal_prefix"):
            report = next(
                c.cell_contents
                for c in check_boundary.__closure__
                if type(c.cell_contents) is dict and "phases" in c.cell_contents
            )
            if fault == "provenance":
                report["original_probe_budget"]["updates_completed"] = 0
            elif fault == "equal_phase":
                report["phases"][0] = deepcopy(report["phases"][0])
            else:
                report["phases"] = deepcopy(report["phases"])
        elif fault == "partial":
            raise evaluation.MiniGridEvaluationFailed(
                dict(
                    status="failed",
                    reason="deliberate partial game",
                    episodes=[dict(status="failed")],
                )
            )
        check_boundary()
        pytest.fail("damaged evidence admitted")

    monkeypatch.setattr(games, "evaluate_context_linear_probe_checkpoint", run)
    result = games.run_saved_affine_probe_games("after", "before")
    assert result["status"] == "failed"
    assert len(calls) == (1 if fault == "incomplete" else 2)
    assert result["phases"][0]["episodes"][0]["success"] is False
    assert result["original_probe_budget"]["updates_completed"] == 2
    assert result["budget"]["started_at"] >= 0
    if fault == "budget":
        assert result["budget"]["observed_budget_started_at"] == -1.0
    assert len(result["phases"]) == (2 if fault == "partial" else 1)


def test_should_keep_paid_pending_phase_when_original_work_deadline_expires(monkeypatch):
    setup_literal(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(games, "monotonic", lambda: now[0])

    def run(*a, phase, **k):
        now[0] = 140.0
        return globals()["phase"](phase)

    monkeypatch.setattr(games, "evaluate_context_linear_probe_checkpoint", run)
    result = games.run_saved_affine_probe_games("after", "before")
    assert result["status"] == "failed" and "work deadline" in result["reason"]
    assert result["phases"] == [phase("before_probe")]


@pytest.mark.parametrize("option", ["describe", "conflict", "fit", "retained"])
def test_should_preserve_cli_no_io_description_and_refuse_invalid_options(
    monkeypatch, capsys, option
):
    args = ["--context-linear-probe-game"]
    args += {
        "describe": ["--describe"],
        "conflict": ["--context-linear-probe"],
        "fit": ["--fit-updates", "1"],
        "retained": ["--source", "after"],
    }[option]
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))
    monkeypatch.setattr(games, "_qualified_probe_bytes", lambda *a: pytest.fail("model opened"))
    if option == "describe":
        cli.main()
        result = json.loads(capsys.readouterr().out)
        assert result["new_fit_calls"] == 0 and result["source_native_fit_counts"] == [6, 8]
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_route_probe_game_cli_and_preserve_exclusive_failed_output(monkeypatch, tmp_path):
    path = tmp_path / "actual.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--context-linear-probe-game",
            "--source",
            "after",
            "--retained-source",
            "before",
            "--output",
            str(path),
        ],
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))

    def run(after, before):
        assert str(after) == "after" and str(before) == "before"
        return dict(status="failed", phases=[])

    monkeypatch.setattr(games, "run_saved_affine_probe_games", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert (
        caught.value.code == 2 and json.loads(path.read_text(encoding="utf8"))["status"] == "failed"
    )
    with pytest.raises(FileExistsError):
        cli.main()
