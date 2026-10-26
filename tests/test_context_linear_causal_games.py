"""Causal-state bindings and original game loop; literal games are not outcomes."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    ContextLinearTrainingSnapshot,
)
from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from test_minigrid_invariant_evaluation import DiscreteEnvironment, public_frame

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/context-linear-stream-20261024/actual.json"


class PhaseEnvironment(DiscreteEnvironment):
    @property
    def report(self):
        return dict(super().report, closed=self.closed)

    def _frame(self, step):
        # The old A/D fixture knew grey/yellow; this slice also exercises B.
        return public_frame(
            self.cue,
            step,
            wall_color={"grey": 5, "blue": 2}[self.layout.wall_color],
            rotation=self.layout.rotation,
        )


def state(updates=2):
    # Literal constant heads only test routing/ownership, not trained success.
    heads = []
    for action in (1, 2):
        coefficients: list[tuple[float, ...]] = [(0.0,) * 7] * 296
        coefficients.append(tuple(float(k == action) for k in range(7)))
        heads.append(CategoricalLinearFit(296, 7, 9, tuple(coefficients), 0, (0.0,) * 9, 0.0))
    return ContextLinearTrainingSnapshot(
        ContextLinearPolicy(), (heads[0], heads[1]), updates, updates * 9
    )


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=150,
            max_process_rss_bytes=512 * 1024**2,
        ),
        lambda: 0.0,
    )


@pytest.mark.parametrize("updates,context", [(2, "A"), (4, "B"), (6, "A")])
def test_should_bind_original_loop_to_causal_phase_and_real_past_counts(
    monkeypatch, updates, context
):
    source, s, session = ContextLinearLearner(), state(updates), budget()

    def run(owner, saved, **options):
        assert owner is source and saved == s and saved is not s
        assert options["updates"] == updates and not options["sources"] and not options["layouts"]
        assert [(layout.context, cue, role) for layout, cue, role in options["live_plan"]] == [
            (context, "blue", "train"),
            (context, "red", "train"),
        ]
        assert options["public_history"] is options["require_initial_source"] is True
        assert options["representation_feature_dim"] == 298
        assert options["return_metadata"]["source_fit_rows"] == updates * 9
        assert options["memory_copier"](source.snapshot_state()).heads == (None, None)
        assert options["memory_copier"](s) == s
        options["extra_check"]()
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert (
        evaluation.evaluate_context_linear_stream_checkpoint(
            source, s, budget=session, check_boundary=lambda: None
        )["status"]
        == "complete"
    )
    assert session.updates_completed == 0 and source.completed_updates == 0


@pytest.mark.parametrize("fault", ["count", "rows", "failed", "unfitted", "head_rows", "owner"])
def test_should_refuse_incompatible_causal_state_before_original_loop(monkeypatch, fault):
    source: Any = ContextLinearLearner()
    s = state()
    if fault == "count":
        s = replace(s, completed_updates=3, training_examples=27)
    elif fault == "rows":
        s = replace(s, training_examples=19)
    elif fault == "failed":
        s = replace(s, failed_work=(1, 9, 0, 0, True))
    elif fault == "unfitted":
        s = replace(s, heads=(s.heads[0], None))
    elif fault == "head_rows":
        head = replace(s.heads[0], fit_rows=8, singular_values=(0.0,) * 8)
        s = replace(s, heads=(head, s.heads[1]))
    else:
        source = object()
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            source, s, budget=budget(), check_boundary=lambda: None
        )


@pytest.mark.parametrize("fault", ["updates", "bool", "replay", "clock", "limits"])
def test_should_reject_changed_readout_budget_before_more_work(monkeypatch, fault):
    session = budget()

    def run(*a, **options):
        if fault == "updates":
            session.updates_completed = 1
        elif fault == "bool":
            session.updates_completed = False
        elif fault == "replay":
            session.replay_examples_completed = 1
        elif fault == "clock":
            session.clock = lambda: 1.0
        else:
            object.__setattr__(session.budget, "max_wall_seconds", 151)
        options["extra_check"]()
        pytest.fail("changed budget accepted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(), state(), budget=session, check_boundary=lambda: None
        )


@pytest.mark.parametrize("updates", [2, 4, 6])
def test_should_execute_original_scripted_game_pair_with_frozen_causal_state(monkeypatch, updates):
    work: dict[str, Any] = {}
    monkeypatch.setattr(DiscreteEnvironment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    source, s = ContextLinearLearner(), state(updates)
    before = source.snapshot_state()
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        source, s, budget=budget(), check_boundary=lambda: None
    )
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert result["source_fit_calls"] == updates and result["source_fit_rows"] == updates * 9
    assert all(e["status"] == "complete" and e["environment"]["closed"] for e in result["episodes"])
    assert [e["predictions"][0]["action"] for e in result["episodes"]] == [1, 2]
    assert result["work"]["live_prediction_calls_returned"] == 4
    assert source.snapshot_state() == before and s == state(updates)


def test_should_preserve_scripted_first_game_when_second_step_fails(monkeypatch):
    monkeypatch.setattr(DiscreteEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    original = DiscreteEnvironment.step

    def step(self, action):
        if self.cue == "red":
            raise RuntimeError("deliberate scripted step failure")
        return original(self, action)

    monkeypatch.setattr(DiscreteEnvironment, "step", step)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(), state(4), budget=budget(), check_boundary=lambda: None
        )
    result = caught.value.report
    assert result["episodes"][0]["status"] == "complete"
    assert result["episodes"][1]["status"] == "failed"
    assert all(e["environment"]["closed"] for e in result["episodes"])
    assert result["work"]["failure_work_may_be_unobserved"] is True


def test_should_run_source_guard_before_fork_or_reset(monkeypatch):
    source = ContextLinearLearner()
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("fork reached"))

    def guard():
        raise ValueError("replaced preceding source")

    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="replaced preceding source"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            source, state(), budget=budget(), check_boundary=guard
        )


def test_should_convert_actual_causal_states_preserving_full_values_and_ownership():
    if not SOURCE.is_file():
        pytest.skip("accepted local107 report unavailable")
    report = json.loads(SOURCE.read_text(encoding="utf8"))
    states = cli._causal_affine_states(report)
    assert [s.completed_updates for s in states] == [2, 4, 6]
    assert [s.training_examples for s in states] == [18, 36, 54]
    assert states[0].heads == states[2].heads
    assert states[0].heads[0] is not states[2].heads[0]
    assert [asdict(s) for s in states] == [
        dict(
            e["state"],
            heads=tuple(
                dict(
                    h,
                    coefficients=tuple(tuple(r) for r in h["coefficients"]),
                    singular_values=tuple(h["singular_values"]),
                )
                for h in e["state"]["heads"]
            ),
        )
        for e in report["endpoints"][1:]
    ]


def test_should_match_saved_actual_a_and_b_actions_after_conversion():
    train_path = ROOT / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    if not SOURCE.is_file() or not train_path.is_file():
        pytest.skip("accepted local TRAIN/report unavailable")
    from src.core.demonstration_episode import episode_arrays

    report = json.loads(SOURCE.read_text(encoding="utf8"))
    states = cli._causal_affine_states(report)
    episodes = cli.read_training_sources(train_path)
    reader = ContextLinearLearner()
    for index, episode in ((0, episodes[0]), (1, episodes[2])):
        reader.restore_state(states[index])
        x, _ = episode_arrays(episode)
        predicted = reader.predict(x)
        assert (
            predicted.argmax(axis=1).tolist()
            == report["endpoints"][index + 1]["scores"][episode.episode_id]["actions"]
        )


@pytest.mark.parametrize(
    "args",
    [
        ["--context-linear-causal-game", "--describe"],
        ["--context-linear-causal-game", "--context-linear-stream"],
        ["--context-linear-causal-game", "--fit-seconds", "1"],
    ],
)
def test_should_describe_without_source_io_and_refuse_conflicting_options(
    monkeypatch, capsys, args
):
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(
        cli, "_qualified_context_stream_bytes", lambda *a: pytest.fail("source reached")
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))
    if "--describe" in args:
        cli.main()
        assert json.loads(capsys.readouterr().out)["new_fit_calls"] == 0
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_refuse_unqualified_source_bytes_before_model_creation(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}", encoding="utf8")
    result = cli.run_saved_context_linear_causal_games(path)
    assert result["status"] == "failed" and result["phases"] == []
    assert "qualified107" in result["reason"] and result["budget"]["updates_completed"] == 0


def complete_phase(updates):
    return dict(
        status="complete",
        updates=updates,
        state_unchanged=True,
        episodes=[dict(status="complete", environment=dict(closed=True), success=True)] * 2,
    )


def setup_literal_source(monkeypatch):
    monkeypatch.setattr(cli, "_qualified_context_stream_bytes", lambda *a: b"{}")
    monkeypatch.setattr(cli, "_causal_affine_states", lambda *a: tuple(state(n) for n in (2, 4, 6)))


@pytest.mark.parametrize("fault", ["status", "leaf", "replace", "budget"])
def test_should_refuse_prior_result_or_original_budget_change_and_keep_known_prefix(
    monkeypatch, fault
):
    setup_literal_source(monkeypatch)
    first = complete_phase(2)
    original = deepcopy(first)
    calls = []

    def evaluate(source, saved, *, budget, check_boundary):
        calls.append(saved.completed_updates)
        if len(calls) == 1:
            return first
        if fault == "status":
            first["status"] = "failed"
        elif fault == "leaf":
            first["episodes"][0]["success"] = 1  # Strict bool-vs-int comparison.
        elif fault == "replace":
            captured = dict(
                zip(
                    check_boundary.__code__.co_freevars,
                    [c.cell_contents for c in check_boundary.__closure__],
                    strict=True,
                )
            )
            captured["report"]["phases"] = list(captured["report"]["phases"])
        else:
            budget.started_at = budget.clock()
            budget.clock = lambda: 0.0
        check_boundary()
        pytest.fail("changed prior evidence admitted")

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("literal")
    assert result["status"] == "failed" and calls == [2, 4]
    assert result["phases"] == [original] and result["source_unchanged"] is None
    assert "changed" in result["reason"] or "replaced" in result["reason"]


def test_should_keep_complete_phase_and_partial_next_phase_without_launching_return(monkeypatch):
    setup_literal_source(monkeypatch)
    partial = dict(
        status="failed",
        reason="scripted second-phase failure",
        work=dict(environment_steps_attempted=1),
    )
    calls = []

    def evaluate(source, saved, **options):
        calls.append(saved.completed_updates)
        if len(calls) == 1:
            return complete_phase(2)
        raise evaluation.MiniGridEvaluationFailed(partial)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("literal")
    assert result["status"] == "failed" and calls == [2, 4]
    assert result["phases"] == [complete_phase(2), partial]
    assert (
        result["budget"]["updates_completed"] == result["budget"]["replay_examples_completed"] == 0
    )


def test_should_route_cli_without_opening_train_and_preserve_partial_exclusive_output(
    monkeypatch, tmp_path
):
    destination = tmp_path / "actual.json"
    report = dict(status="failed", phases=[dict(status="complete"), dict(status="failed")])
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--context-linear-causal-game",
            "--source",
            "source",
            "--output",
            str(destination),
        ],
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))
    monkeypatch.setattr(cli, "run_saved_context_linear_causal_games", lambda *a: deepcopy(report))
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert json.loads(destination.read_text(encoding="utf8"))["phases"] == report["phases"]
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_stop_at_work_deadline_and_preserve_complete_pending_prefix(monkeypatch):
    setup_literal_source(monkeypatch)
    now = [0.0]
    calls = []
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])

    def evaluate(source, saved, **options):
        calls.append(saved.completed_updates)
        now[0] = 140.0
        return complete_phase(2)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("literal")
    assert result["status"] == "failed" and calls == [2]
    assert result["phases"] == [complete_phase(2)]
    assert "work deadline" in result["reason"]
    assert result["budget"]["elapsed_seconds"] == 140.0
    assert result["budget"]["updates_completed"] == 0
    assert result["initial_readout_limits"] == result["budget"]["limits"]
