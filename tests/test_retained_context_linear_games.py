"""Retained native bindings; scripted games/literal heads are not learned outcomes."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_context_linear_learner as port
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_context_linear_causal_games import PhaseEnvironment, state as replacement_state

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/arrived-cue-catalogue-20261024/actual.json"


def state():
    base = replacement_state(4)
    heads = tuple(replace(head, fit_rows=18, singular_values=(0.0,) * 18) for head in base.heads)
    return replace(base, heads=heads, training_examples=54)


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


@pytest.mark.parametrize("retained", [False, True])
def test_should_bind_original_loop_without_changing_replacement_default(monkeypatch, retained):
    source = port.ContextLinearLearner()
    saved = state() if retained else replacement_state(4)

    def run(owner, current, **options):
        assert owner is source and current == saved and current is not saved
        assert options["sources"] == options["layouts"] == () and options["updates"] == 4
        pairs = [
            (layout.context, cue) for layout, cue, role in options["live_plan"] if role == "train"
        ]
        assert pairs == (
            [("A", "blue"), ("A", "red"), ("B", "blue"), ("B", "red")]
            if retained
            else [("B", "blue"), ("B", "red")]
        )
        assert options["return_metadata"]["source_fit_rows"] == (54 if retained else 36)
        assert options["return_metadata"]["new_fit_calls"] == 0
        assert options["public_history"] is options["require_initial_source"] is True
        assert options["representation_feature_dim"] == 298
        assert options["memory_copier"](source.snapshot_state()).heads == (None, None)
        options["extra_check"]()
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        source, saved, budget=budget(), check_boundary=lambda: None, retained_catalogue=retained
    )
    assert result["status"] == "complete" and source.completed_updates == 0


@pytest.mark.parametrize("fault", ["early", "late", "rows", "head", "failed", "owner", "flag"])
def test_should_refuse_incompatible_retained_state_before_loop(monkeypatch, fault):
    source: Any = port.ContextLinearLearner()
    saved = state()
    flag: Any = True
    if fault == "early":
        saved = replace(saved, completed_updates=2, training_examples=36)
    elif fault == "late":
        saved = replace(saved, completed_updates=6, training_examples=90)
    elif fault == "rows":
        saved = replace(saved, training_examples=36)
    elif fault == "head":
        saved = replace(saved, heads=(replacement_state(4).heads[0], saved.heads[1]))
    elif fault == "failed":
        saved = replace(saved, failed_work=(1, 18, 0, 0, True))
    elif fault == "owner":
        source = object()
    else:
        flag = 1
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            source, saved, budget=budget(), check_boundary=lambda: None, retained_catalogue=flag
        )


@pytest.mark.parametrize("fault", ["updates", "bool", "replay", "clock", "limits"])
def test_should_preserve_original_zero_update_readout_budget(monkeypatch, fault):
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
        pytest.fail("budget change admitted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            port.ContextLinearLearner(),
            state(),
            budget=session,
            check_boundary=lambda: None,
            retained_catalogue=True,
        )


@pytest.mark.parametrize("retained", [False, True])
def test_should_execute_original_scripted_default_and_retained_loops(monkeypatch, retained):
    work: dict[str, Any] = {}
    monkeypatch.setattr(PhaseEnvironment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    source = port.ContextLinearLearner()
    saved = state() if retained else replacement_state(4)
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        source, saved, budget=budget(), check_boundary=lambda: None, retained_catalogue=retained
    )
    count = 4 if retained else 2
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert (
        len(result["episodes"]) == count
        and result["work"]["live_prediction_calls_returned"] == 2 * count
    )
    assert all(e["status"] == "complete" and e["environment"]["closed"] for e in result["episodes"])
    assert [e["predictions"][0]["action"] for e in result["episodes"]] == [1, 2] * (
        2 if retained else 1
    )
    assert source.snapshot_state().heads == (None, None) and saved == (
        state() if retained else replacement_state(4)
    )


def test_should_preserve_first_three_scripted_games_if_last_step_fails(monkeypatch):
    monkeypatch.setattr(PhaseEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    original = PhaseEnvironment.step

    def step(self, action):
        if self.layout.context == "B" and self.cue == "red":
            raise RuntimeError("deliberate last-game failure")
        return original(self, action)

    monkeypatch.setattr(PhaseEnvironment, "step", step)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluation.evaluate_context_linear_stream_checkpoint(
            port.ContextLinearLearner(),
            state(),
            budget=budget(),
            check_boundary=lambda: None,
            retained_catalogue=True,
        )
    result = caught.value.report
    assert [e["status"] for e in result["episodes"]] == ["complete"] * 3 + ["failed"]
    assert all(e["environment"]["closed"] for e in result["episodes"])
    assert result["work"]["failure_work_may_be_unobserved"] is True


def actual_report():
    if not SOURCE.is_file():
        pytest.skip("accepted local109 report unavailable")
    return json.loads(SOURCE.read_text(encoding="utf8"))


def test_should_convert_actual_retained_clocks_and_match_saved_old_new_actions():
    train = ROOT / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    if not train.is_file():
        pytest.skip("accepted local TRAIN source unavailable")
    from src.core.demonstration_episode import episode_arrays

    report = actual_report()
    states = cli._causal_affine_states(report, retained_catalogue=True)
    assert [(s.completed_updates, s.training_examples) for s in states] == [
        (2, 18),
        (4, 54),
        (6, 90),
    ]
    assert [h.fit_rows for h in states[1].heads] == [18, 18]
    reader = port.ContextLinearLearner()
    reader.restore_state(states[1])
    episodes = cli.read_training_sources(train)
    for episode in (episodes[0], episodes[2]):
        x, _ = episode_arrays(episode)
        assert (
            reader.predict(x).argmax(axis=1).tolist()
            == report["endpoints"][2]["scores"][episode.episode_id]["actions"]
        )


@pytest.mark.parametrize("fault", ["definition", "ledger", "unknown", "rows", "status"])
def test_should_refuse_incomplete_or_wrong_retained_report_before_models(monkeypatch, fault):
    report = actual_report()
    if fault == "definition":
        report["definition"] = "replacement"
    elif fault == "ledger":
        report["catalogue"]["committed_rows"]["replay"] = 0
    elif fault == "unknown":
        report["catalogue"]["failure_work_may_be_unobserved"] = True
    elif fault == "rows":
        report["stream"]["training_examples"] = 54
    else:
        report["status"] = "failed"
    monkeypatch.setattr(
        "src.core.categorical_linear_fit.CategoricalLinearFit",
        lambda *a, **k: pytest.fail("model reached"),
    )
    with pytest.raises(ValueError):
        cli._causal_affine_states(report, retained_catalogue=True)


@pytest.mark.parametrize("kind", ["wrong_source", "oversize"])
def test_should_refuse_unqualified_retained_bytes_before_owner(monkeypatch, tmp_path, kind):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}" if kind == "wrong_source" else b"x" * (1024**2 + 2))
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("owner reached"))
    result = cli.run_saved_context_linear_causal_games(path, retained_catalogue=True)
    assert result["status"] == "failed" and result["phases"] == []
    assert "qualified109" in result["reason"]


def setup_literal(monkeypatch):
    monkeypatch.setattr(cli, "_qualified_context_catalogue_bytes", lambda *a: b"{}")
    monkeypatch.setattr(
        cli,
        "_causal_affine_states",
        lambda *a, **k: (
            replacement_state(2),
            state(),
            replace(state(), completed_updates=6, training_examples=90),
        ),
    )


def phase():
    return dict(
        status="complete",
        updates=4,
        state_unchanged=True,
        episodes=[dict(status="complete", environment=dict(closed=True)) for _ in range(4)],
    )


def test_should_evaluate_only_after_b_state_and_keep_original_source_untrained(monkeypatch):
    setup_literal(monkeypatch)
    calls = []

    def evaluate(source, saved, *, budget, check_boundary, retained_catalogue=False):
        assert source.completed_updates == 0 and retained_catalogue is True
        assert saved.completed_updates == 4 and saved.training_examples == 54
        check_boundary()
        calls.append(saved)
        return phase()

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("literal", retained_catalogue=True)
    assert result["status"] == "complete" and result["source_unchanged"] is True
    assert len(calls) == len(result["phases"]) == 1 and result["phases"][0] == phase()
    assert (
        result["budget"]["updates_completed"] == result["budget"]["replay_examples_completed"] == 0
    )


def test_should_preserve_single_partial_phase_on_failure_without_fitting(monkeypatch):
    setup_literal(monkeypatch)
    partial = dict(
        status="failed",
        reason="partial retained game",
        episodes=[dict(status="complete"), dict(status="failed")],
    )

    def evaluate(*a, **options):
        raise evaluation.MiniGridEvaluationFailed(partial)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("literal", retained_catalogue=True)
    assert result["status"] == "failed" and result["phases"] == [partial]
    assert result["source_unchanged"] is None


def test_should_stop_at_original_work_deadline_and_keep_pending_report(monkeypatch):
    setup_literal(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])

    def evaluate(*a, **options):
        now[0] = 140.0
        return phase()

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("literal", retained_catalogue=True)
    assert result["status"] == "failed" and "work deadline" in result["reason"]
    assert (
        result["phases"] == [phase()]
        and result["initial_readout_limits"] == result["budget"]["limits"]
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--context-linear-retained-game", "--describe"],
        ["--context-linear-retained-game", "--context-linear-causal-game"],
        ["--context-linear-retained-game", "--fit-updates", "1"],
    ],
)
def test_should_describe_without_io_and_refuse_conflicting_options(monkeypatch, capsys, args):
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))
    monkeypatch.setattr(
        cli, "_qualified_context_catalogue_bytes", lambda *a: pytest.fail("saved data opened")
    )
    if "--describe" in args:
        cli.main()
        result = json.loads(capsys.readouterr().out)
        assert result["checkpoint_updates"] == [4] and result["checkpoint_fit_rows"] == 54
        assert result["max_environment_steps"] == 64 and result["new_fit_calls"] == 0
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_route_retained_cli_and_preserve_partial_exclusive_output(monkeypatch, tmp_path):
    path = tmp_path / "actual.json"
    monkeypatch.setattr(
        "sys.argv",
        ["control", "--context-linear-retained-game", "--source", "saved", "--output", str(path)],
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))

    def run(source, *, retained_catalogue=False):
        assert retained_catalogue is True
        return dict(status="failed", phases=[dict(status="failed")])

    monkeypatch.setattr(cli, "run_saved_context_linear_causal_games", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and json.loads(path.read_text(encoding="utf8"))["phases"] == [
        dict(status="failed")
    ]
    with pytest.raises(FileExistsError):
        cli.main()
