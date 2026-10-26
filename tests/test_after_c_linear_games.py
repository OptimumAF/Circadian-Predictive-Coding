"""After-C binding and original scripted game loop; no training/outcome claims."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as port
from src.adapters import minigrid_policy_evaluation as evaluation
from test_retained_context_linear_games import budget, state as retained_state
from test_context_linear_causal_games import PhaseEnvironment
from test_minigrid_invariant_evaluation import public_frame

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/catalogue-capacity-new-c-20261024/actual.json"


def state():
    saved = retained_state()
    heads = tuple(replace(head, fit_rows=27, singular_values=(0.0,) * 27) for head in saved.heads)
    return replace(saved, heads=heads, completed_updates=6, training_examples=108)


class AfterCEnvironment(PhaseEnvironment):
    def _frame(self, step):
        return public_frame(
            self.cue,
            step,
            wall_color={"blue": 2, "red": 0}[self.layout.wall_color],
            rotation=self.layout.rotation,
        )


def test_should_bind_one_after_c_model_to_original_b_c_loop_and_full_guards(monkeypatch):
    owner, saved = port.ContextLinearLearner(), state()

    def run(source, memory, **options):
        assert source is owner and memory == saved and memory is not saved
        assert options["role"] == "knownTRAIN_BC_afterC" and options["updates"] == 6
        assert options["sources"] == options["layouts"] == ()
        assert options["require_initial_source"] is options["public_history"] is True
        assert options["representation_feature_dim"] == 298
        assert options["return_metadata"]["source_fit_rows"] == 108
        assert [(layout.context, cue, role) for layout, cue, role in options["live_plan"]] == [
            (context, cue, "train") for context in ("B", "C_train") for cue in ("blue", "red")
        ]
        options["extra_check"]()
        assert options["memory_copier"](owner.snapshot_state()).heads == (None, None)
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        owner, saved, budget=budget(), check_boundary=lambda: None, after_c_capacity=True
    )
    assert result["status"] == "complete" and owner.completed_updates == 0


@pytest.mark.parametrize("fault", ["count", "rows", "head", "failed", "flag", "conflict", "owner"])
def test_should_refuse_non_after_c_geometry_or_ambiguous_modes_before_loop(monkeypatch, fault):
    saved = state()
    owner: Any = port.ContextLinearLearner()
    after: Any = True
    retained = False
    if fault == "count":
        saved = replace(saved, completed_updates=4)
    elif fault == "rows":
        saved = replace(saved, training_examples=90)
    elif fault == "head":
        saved = replace(saved, heads=(retained_state().heads[0], saved.heads[1]))
    elif fault == "failed":
        saved = replace(saved, failed_work=(1, 27, 0, 0, True))
    elif fault == "flag":
        after = 1
    elif fault == "conflict":
        retained = True
    else:
        owner = object()
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            owner,
            saved,
            budget=budget(),
            check_boundary=lambda: None,
            after_c_capacity=after,
            retained_catalogue=retained,
        )


@pytest.mark.parametrize("fault", ["counter", "clock"])
def test_should_preserve_original_zero_work_budget_binding_in_after_c_mode(monkeypatch, fault):
    session = budget()

    def run(*a, **options):
        if fault == "counter":
            session.updates_completed = 1
        else:
            session.clock = lambda: 1.0
        options["extra_check"]()
        pytest.fail("changed budget admitted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            port.ContextLinearLearner(),
            state(),
            budget=session,
            check_boundary=lambda: None,
            after_c_capacity=True,
        )


def test_should_execute_original_after_c_scripted_loop_with_same_source_and_four_closures(
    monkeypatch,
):
    monkeypatch.setattr(AfterCEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", AfterCEnvironment)
    source = port.ContextLinearLearner()
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        source, state(), budget=budget(), check_boundary=lambda: None, after_c_capacity=True
    )
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert len(result["episodes"]) == 4 and result["work"]["live_prediction_calls_returned"] == 8
    assert [e["context"] for e in result["episodes"]] == ["B", "B", "C_train", "C_train"]
    assert [e["predictions"][0]["action"] for e in result["episodes"]] == [1, 2, 1, 2]
    assert all(e["environment"]["closed"] and e["status"] == "complete" for e in result["episodes"])
    assert source.snapshot_state().heads == (None, None)


def test_should_keep_three_completed_scripted_games_if_last_c_step_fails(monkeypatch):
    monkeypatch.setattr(AfterCEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", AfterCEnvironment)
    original = AfterCEnvironment.step

    def step(self, action):
        if self.layout.context == "C_train" and self.cue == "red":
            raise RuntimeError("deliberate final C game failure")
        return original(self, action)

    monkeypatch.setattr(AfterCEnvironment, "step", step)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluation.evaluate_context_linear_stream_checkpoint(
            port.ContextLinearLearner(),
            state(),
            budget=budget(),
            check_boundary=lambda: None,
            after_c_capacity=True,
        )
    result = caught.value.report
    assert [e["status"] for e in result["episodes"]] == ["complete"] * 3 + ["failed"]
    assert all(e["environment"]["closed"] for e in result["episodes"])
    assert result["work"]["failure_work_may_be_unobserved"] is True


def actual_report():
    if not SOURCE.is_file():
        pytest.skip("accepted local111report unavailable")
    return json.loads(SOURCE.read_text(encoding="utf8"))


def test_should_convert_real_native_history_and_match_saved_b_c_actions():
    train = ROOT / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    if not train.is_file():
        pytest.skip("accepted original TRAIN source unavailable")
    from src.core.demonstration_episode import episode_arrays

    report = actual_report()
    saved = cli._after_c_affine_state(report)
    assert (saved.completed_updates, saved.training_examples) == (6, 108)
    assert [h.fit_rows for h in saved.heads] == [27, 27]
    reader = port.ContextLinearLearner()
    reader.restore_state(saved)
    episodes = cli.read_training_sources(train)
    for episode in (episodes[2], episodes[4]):
        x, _ = episode_arrays(episode)
        assert (
            reader.predict(x).argmax(axis=1).tolist()
            == report["arms"][1]["scores"][episode.episode_id]["actions"]
        )


@pytest.mark.parametrize(
    "fault", ["status", "source", "ledger", "receipt", "branch", "geometry", "clock"]
)
def test_should_refuse_incomplete_or_replaced_after_c_evidence_before_models(monkeypatch, fault):
    report = actual_report()
    if fault == "status":
        report["status"] = "failed"
    elif fault == "source":
        report["source_bindings"]["retained_bytes"] = 0
    elif fault == "ledger":
        report["arms"][1]["catalogue"]["committed_rows"]["replay"] = 0
    elif fault == "receipt":
        report["arms"][1]["committed_receipts"][0]["arrived_at"] = 0
    elif fault == "branch":
        report["arms"].reverse()
    elif fault == "clock":
        report["budget"]["ended_at"] = report["budget"]["started_at"] - 1
    else:
        report["arms"][1]["last_checkpoint"]["state"]["training_examples"] = 90
    monkeypatch.setattr(
        "src.core.categorical_linear_fit.CategoricalLinearFit",
        lambda *a, **k: pytest.fail("model reached"),
    )
    with pytest.raises(ValueError):
        cli._after_c_affine_state(report)


@pytest.mark.parametrize("kind", ["wrong", "oversize"])
def test_should_refuse_wrong_source_bytes_before_native_owner(monkeypatch, tmp_path, kind):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}" if kind == "wrong" else b"x" * (1024**2 + 2))
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("owner reached"))
    result = cli.run_saved_context_linear_causal_games(path, after_c_capacity=True)
    assert (
        result["status"] == "failed"
        and result["phases"] == []
        and "qualified111" in result["reason"]
    )


def setup_literal(monkeypatch):
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: b"{}")
    monkeypatch.setattr(cli, "_after_c_affine_state", lambda *a: state())


def phase():
    return dict(
        status="complete",
        state_unchanged=True,
        updates=6,
        episodes=[dict(status="complete", environment=dict(closed=True)) for _ in range(4)],
    )


def test_should_bind_only_capacity_two_and_leave_source_untrained(monkeypatch):
    setup_literal(monkeypatch)
    calls = []

    def evaluate(source, saved, *, budget, check_boundary, after_c_capacity=False):
        assert source.completed_updates == 0 and after_c_capacity is True
        assert saved.training_examples == 108
        check_boundary()
        calls.append(saved)
        return phase()

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("saved", after_c_capacity=True)
    assert result["status"] == "complete" and result["source_unchanged"] is True and len(calls) == 1
    assert (
        result["protocol"]["source_fit_calls"] == 10
        and result["protocol"]["source_fit_rows"] == 180
    )
    assert (
        result["budget"]["updates_completed"] == result["budget"]["replay_examples_completed"] == 0
    )


def test_should_capture_failed_partial_after_c_phase_without_refitting(monkeypatch):
    setup_literal(monkeypatch)
    partial = dict(
        status="failed",
        reason="deliberate partial phase",
        episodes=[dict(status="complete"), dict(status="failed")],
    )

    def evaluate(*a, **k):
        raise evaluation.MiniGridEvaluationFailed(partial)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("saved", after_c_capacity=True)
    assert result["status"] == "failed" and result["phases"] == [partial]


def test_should_preserve_pending_after_c_phase_if_original_work_deadline_expires(monkeypatch):
    setup_literal(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])

    def evaluate(*a, **k):
        now[0] = 140.0
        return phase()

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("saved", after_c_capacity=True)
    assert (
        result["status"] == "failed"
        and "work deadline" in result["reason"]
        and result["phases"] == [phase()]
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--context-linear-after-c-game", "--describe"],
        ["--context-linear-after-c-game", "--fit-updates", "1"],
        ["--context-linear-after-c-game", "--context-linear-retained-game"],
    ],
)
def test_should_describe_without_source_io_and_refuse_conflicting_after_c_options(
    monkeypatch, capsys, args
):
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))
    monkeypatch.setattr(
        cli, "_qualified_after_c_bytes", lambda *a: pytest.fail("saved model opened")
    )
    if "--describe" in args:
        cli.main()
        result = json.loads(capsys.readouterr().out)
        assert result["checkpoint_updates"] == [6] and result["checkpoint_fit_rows"] == 108
        assert result["new_fit_calls"] == 0 and result["max_environment_steps"] == 64
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_route_after_c_cli_and_save_failed_output_exclusively(monkeypatch, tmp_path):
    path = tmp_path / "actual.json"
    monkeypatch.setattr(
        "sys.argv",
        ["control", "--context-linear-after-c-game", "--source", "saved", "--output", str(path)],
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))

    def run(source, *, after_c_capacity=False):
        assert after_c_capacity is True
        return dict(status="failed", phases=[dict(status="failed")])

    monkeypatch.setattr(cli, "run_saved_context_linear_causal_games", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert (
        caught.value.code == 2 and json.loads(path.read_text(encoding="utf8"))["status"] == "failed"
    )
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_refuse_replaced_saved_source_before_another_native_action(monkeypatch):
    setup_literal(monkeypatch)
    raw = [b"{}"]
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: raw[0])

    def evaluate(*a, check_boundary, **k):
        check_boundary()
        raw[0] = b"replaced"
        check_boundary()
        pytest.fail("source replacement accepted")

    monkeypatch.setattr(evaluation, "evaluate_context_linear_stream_checkpoint", evaluate)
    result = cli.run_saved_context_linear_causal_games("saved", after_c_capacity=True)
    assert result["status"] == "failed" and "source changed" in result["reason"]
