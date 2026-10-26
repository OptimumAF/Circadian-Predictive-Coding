"""Readonly returning-context bindings; scripted games aren't native outcomes."""

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
from test_after_c_linear_games import state as original_state
from test_context_linear_causal_games import PhaseEnvironment
from test_retained_context_linear_games import budget

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/affine-private-rehearsal-20261024/actual.json"


def state():
    old = original_state()
    return replace(
        old,
        completed_updates=8,
        training_examples=180,
        heads=tuple(replace(h, fit_rows=36, singular_values=(0.0,) * 36) for h in old.heads),
    )


@pytest.mark.parametrize("arm", ["current_only", "joint_ABC_F"])
def test_should_bind_native_eight_fit_state_to_fixed_f_then_a_plan(monkeypatch, arm):
    owner, saved = ContextLinearLearner(), state()

    def run(source, memory, **options):
        assert source is owner and memory == saved and memory is not saved
        assert options["updates"] == 8 and options["require_initial_source"] is True
        assert options["layouts"] == options["sources"] == ()
        assert options["return_metadata"]["rehearsal_arm"] == arm
        assert options["representation_feature_dim"] == 298
        assert [(x.context, cue, role) for x, cue, role in options["live_plan"]] == [
            ("F-blue", "blue", "probe_test"),
            ("F-red", "red", "probe_test"),
            ("A", "blue", "train"),
            ("A", "red", "train"),
        ]
        options["extra_check"]()
        assert options["memory_copier"](source.snapshot_state()).heads == (None, None)
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert (
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            owner, saved, arm=arm, budget=budget(), check_boundary=lambda: None
        )["status"]
        == "complete"
    )


@pytest.mark.parametrize("fault", ["arm", "updates", "rows", "head", "failed", "owner"])
def test_should_refuse_wrong_native_recipe_before_any_game(monkeypatch, fault):
    saved, arm = state(), "current_only"
    owner: Any = ContextLinearLearner()
    if fault == "arm":
        arm = "after_probe"
    elif fault == "updates":
        saved = replace(saved, completed_updates=7)
    elif fault == "rows":
        saved = replace(saved, training_examples=179)
    elif fault == "head":
        saved = replace(saved, heads=(original_state().heads[0], saved.heads[1]))
    elif fault == "failed":
        saved = replace(saved, failed_work=(1, 36, 0, 0, True))
    else:
        owner = object()
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            owner, saved, arm=arm, budget=budget(), check_boundary=lambda: None
        )


@pytest.mark.parametrize("fault", ["counter", "clock", "boolean"])
def test_should_preserve_original_zero_work_budget_binding(monkeypatch, fault):
    session = budget()

    def run(*a, **options):
        if fault == "clock":
            session.clock = lambda: 1.0
        else:
            session.replay_examples_completed = False if fault == "boolean" else 1
        options["extra_check"]()
        pytest.fail("changed authority admitted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            ContextLinearLearner(),
            state(),
            arm="current_only",
            budget=session,
            check_boundary=lambda: None,
        )


def test_should_run_both_models_through_original_scripted_f_then_a_loop(monkeypatch):
    monkeypatch.setattr(PhaseEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    source = ContextLinearLearner()
    for arm in ("current_only", "joint_ABC_F"):
        r = evaluation.evaluate_context_linear_rehearsal_checkpoint(
            source, state(), arm=arm, budget=budget(), check_boundary=lambda: None
        )
        assert r["status"] == "complete" and r["state_unchanged"] is True
        assert [(e["context"], e["cue"], e["source_role"]) for e in r["episodes"]] == [
            ("F-blue", "blue", "probe_test"),
            ("F-red", "red", "probe_test"),
            ("A", "blue", "train"),
            ("A", "red", "train"),
        ]
        assert r["work"]["live_prediction_calls_returned"] == 8
        assert all(e["environment"]["closed"] for e in r["episodes"])
        assert "F probe_test or A TRAIN" in r["public_history"]["storage_scope"]
    assert source.snapshot_state().heads == (None, None)


def test_should_keep_three_closed_games_and_failed_returned_a_prefix(monkeypatch):
    monkeypatch.setattr(PhaseEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    original = PhaseEnvironment.step

    def step(self, action):
        if self.layout.context == "A" and self.cue == "red":
            raise RuntimeError("synthetic returned A failure")
        return original(self, action)

    monkeypatch.setattr(PhaseEnvironment, "step", step)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            ContextLinearLearner(),
            state(),
            arm="joint_ABC_F",
            budget=budget(),
            check_boundary=lambda: None,
        )
    r = caught.value.report
    assert [e["status"] for e in r["episodes"]] == ["complete"] * 3 + ["failed"]
    assert all(e["environment"]["closed"] for e in r["episodes"])
    assert r["work"]["failure_work_may_be_unobserved"] is True


def actual():
    if not SOURCE.is_file():
        pytest.skip("accepted local118 source unavailable")
    return json.loads(games._qualified_rehearsal_bytes(SOURCE))


def test_should_restore_both_actual_states_with_original_native_counters_and_no_fits():
    result = games._rehearsal_states(actual())
    assert len(result) == 2
    assert all((s.completed_updates, s.training_examples) == (8, 180) for s in result)
    assert all([h.fit_rows for h in s.heads] == [36, 36] for s in result)


@pytest.mark.parametrize(
    "fault", ["status", "ledger", "clock", "attempt", "fork", "rows", "failed", "boolean", "order"]
)
def test_should_refuse_incomplete_or_replaced_original_evidence_before_state_conversion(
    monkeypatch, fault
):
    r = actual()
    if fault == "status":
        r["status"] = "failed"
    elif fault == "ledger":
        r["budget"]["replay_examples_completed"] = 0
    elif fault == "clock":
        r["arms"][1]["budget"]["started_at"] += 1
    elif fault == "attempt":
        r["arms"][1]["attempts"][0]["old_TRAIN_reuse_charged"] = 0
    elif fault == "fork":
        r["arms"][1]["source_fork_calls"] = 2
    elif fault == "rows":
        r["arms"][1]["native_endpoint"]["training_examples"] = 179
    elif fault == "failed":
        r["arms"][1]["native_endpoint"]["failed_work"] = [1, 36, 0, 0, True]
    elif fault == "boolean":
        r["arms"][0]["budget"]["replay_examples_completed"] = False
    else:
        r["arms"].reverse()
    monkeypatch.setattr(
        cli, "_context_linear_training_snapshot", lambda *a: pytest.fail("model reached")
    )
    with pytest.raises(ValueError):
        games._rehearsal_states(r)


@pytest.mark.parametrize("kind", ["wrong", "oversized"])
def test_should_refuse_changed_bytes_before_any_native_owner(tmp_path, monkeypatch, kind):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}" if kind == "wrong" else b"x" * (1024**2 + 1))
    monkeypatch.setattr(games, "ContextLinearLearner", lambda: pytest.fail("owner reached"))
    r = games.run_saved_affine_probe_games(path, rehearsal=True)
    assert r["status"] == "failed" and r["phases"] == [] and "qualified118" in r["reason"]


def setup_literal(monkeypatch):
    original = dict(
        budget=dict(updates_completed=4, replay_examples_completed=54),
        arms=[
            dict(budget=dict(updates_completed=2, replay_examples_completed=n), source_fork_calls=1)
            for n in (0, 54)
        ],
        original_probe_budget=dict(updates_completed=2, started_at=11),
        original_probe_source_fork_calls=1,
    )
    raw = [json.dumps(original).encode()]
    monkeypatch.setattr(games, "_qualified_rehearsal_bytes", lambda *a: raw[0])
    monkeypatch.setattr(games, "_rehearsal_states", lambda *a: (state(), state()))
    return raw


def phase(name):
    return dict(
        status="complete",
        rehearsal_arm=name,
        state_unchanged=True,
        episodes=[
            dict(status="complete", success=False, environment=dict(closed=True)) for _ in range(4)
        ],
    )


def test_should_accept_complete_negative_goals_and_preserve_all_original_charges(monkeypatch):
    setup_literal(monkeypatch)
    calls = []

    def run(source, saved, *, arm, budget, check_boundary):
        check_boundary()
        calls.append(arm)
        assert source.completed_updates == 0 and saved.training_examples == 180
        return phase(arm)

    monkeypatch.setattr(games, "evaluate_context_linear_rehearsal_checkpoint", run)
    r = games.run_saved_affine_probe_games("source", rehearsal=True)
    assert r["status"] == "complete" and calls == ["current_only", "joint_ABC_F"]
    assert r["budget"]["updates_completed"] == r["budget"]["replay_examples_completed"] == 0
    assert r["original_study_budget"]["replay_examples_completed"] == 54
    assert r["original_arm_fork_calls"] == [1, 1] and r["original_probe_source_fork_calls"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "prefix",
        "nested",
        "source",
        "counter",
        "clock",
        "provenance",
        "partial",
        "protocol",
    ],
)
def test_should_stop_after_changed_or_incomplete_prior_and_keep_paid_prefix(monkeypatch, fault):
    raw = setup_literal(monkeypatch)
    first = phase("current_only")
    calls = []

    def run(source, saved, *, arm, budget, check_boundary):
        calls.append(arm)
        if arm == "current_only":
            if fault == "incomplete":
                first["episodes"][0]["environment"]["closed"] = False
            return first
        report = next(
            c.cell_contents
            for c in check_boundary.__closure__
            if type(c.cell_contents) is dict and "phases" in c.cell_contents
        )
        if fault == "prefix":
            report["phases"] = deepcopy(report["phases"])
        elif fault == "nested":
            first["episodes"] = deepcopy(first["episodes"])
        elif fault == "source":
            raw[0] = b"replaced"
        elif fault == "counter":
            budget.replay_examples_completed = 1
        elif fault == "clock":
            budget.started_at = -1.0
        elif fault == "provenance":
            report["original_arm_budgets"][0]["updates_completed"] = 0
        elif fault == "protocol":
            report["protocol"]["phases"] = ("joint_ABC_F", "current_only")
        else:
            raise evaluation.MiniGridEvaluationFailed(
                dict(status="failed", reason="synthetic partial", episodes=[dict(status="failed")])
            )
        check_boundary()
        pytest.fail("changed prior evidence admitted")

    monkeypatch.setattr(games, "evaluate_context_linear_rehearsal_checkpoint", run)
    r = games.run_saved_affine_probe_games("source", rehearsal=True)
    assert r["status"] == "failed" and len(calls) == (1 if fault == "incomplete" else 2)
    assert r["phases"][0]["episodes"][0]["success"] is False
    assert len(r["phases"]) == (2 if fault == "partial" else 1)
    assert r["original_arm_budgets"][0]["updates_completed"] == 2
    assert r["budget"]["started_at"] >= 0


@pytest.mark.parametrize("option", ["describe", "fit", "conflict", "retained"])
def test_should_describe_without_io_and_refuse_fit_conflict_and_extra_source(
    monkeypatch, capsys, option
):
    args = {
        "describe": ["--describe"],
        "fit": ["--fit-updates", "1"],
        "conflict": ["--context-linear-probe-game"],
        "retained": ["--retained-source", "bad"],
    }[option]
    monkeypatch.setattr("sys.argv", ["control", "--context-linear-return-game", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("TRAIN read"))
    monkeypatch.setattr(games, "_qualified_rehearsal_bytes", lambda *a: pytest.fail("source read"))
    if option == "describe":
        cli.main()
        r = json.loads(capsys.readouterr().out)
        assert (
            r["new_fit_calls"] == 0
            and r["source_native_fit_counts"] == [8, 8]
            and r["games_per_phase"] == 4
        )
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_route_one_source_mode_and_preserve_exclusive_failed_output(monkeypatch, tmp_path):
    path = tmp_path / "actual.json"
    monkeypatch.setattr(
        "sys.argv",
        ["control", "--context-linear-return-game", "--source", "saved", "--output", str(path)],
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("TRAIN read"))

    def run(source, before_path=None, *, rehearsal=False):
        assert str(source) == "saved" and before_path is None and rehearsal is True
        return dict(status="failed", phases=[])

    monkeypatch.setattr(games, "run_saved_affine_probe_games", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and path.is_file()
    with pytest.raises(FileExistsError):
        cli.main()
