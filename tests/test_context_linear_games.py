"""Frozen affine reads and existing scripted game loop; no fitting or physics."""

import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_policy as policy
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.demonstration_episode import episode_arrays
from test_minigrid_invariant_evaluation import DiscreteEnvironment

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "artifacts/runs/context-linear-train-fit-20261024/actual.json"


def state():
    # Literal constant-class models test ownership/routing, not learned success.
    heads = []
    for action in (1, 2):
        coefficients: list[tuple[float, ...]] = [(0.0,) * 7] * 296
        coefficients.append(tuple(1.0 if k == action else 0.0 for k in range(7)))
        heads.append(CategoricalLinearFit(296, 7, 27, tuple(coefficients), 0, (0.0,) * 27, 0.0))
    return policy.ContextLinearSnapshot(policy.ContextLinearPolicy(), (heads[0], heads[1]))


def frame(cue):
    values = [-0.8, -1.0, -1.0] * 49 + [-1.0]
    values[:3] = [0.2, 2.0 * cue / 5 - 1, -1.0]
    return tuple(values)


def inputs():
    return np.array(
        [frame(2) + frame(0), frame(0) + frame(2), frame(2) + frame(2)], dtype=np.float64
    )


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=90,
            max_process_rss_bytes=512 * 1024**2,
        ),
        lambda: 0.0,
    )


def test_should_route_mixed_reads_from_initial_cue_preserving_order_and_input():
    s = state()
    reader = policy.ContextLinearReader(s)
    x = inputs()
    original = x.copy()
    before = pickle.dumps(reader.snapshot_state())
    p = reader.predict(x)
    assert p.argmax(axis=1).tolist() == [1, 2, 1]
    assert p.shape == (3, 7) and np.allclose(p.sum(axis=1), 1)
    assert not np.shares_memory(x, p) and np.array_equal(x, original)
    assert pickle.dumps(reader.snapshot_state()) == before
    assert not hasattr(reader, "train_batch") and not hasattr(s, "completed_updates")


def test_should_detach_model_wrappers_and_freeze_restore_to_the_same_binding():
    s = state()
    reader = policy.ContextLinearReader(s)
    original = reader.snapshot_state()
    child = reader.fork()
    assert child is not reader and child._state.heads[0] is not reader._state.heads[0]
    assert child._state.heads[0].coefficients is reader._state.heads[0].coefficients
    object.__setattr__(s.heads[0], "squared_error", 0.25)
    assert reader.snapshot_state() == original
    reader.restore_state(original)
    foreign = state()
    object.__setattr__(foreign.heads[1], "squared_error", 1.0)
    with pytest.raises(ValueError, match="different fitted binding"):
        reader.restore_state(foreign)
    assert reader.snapshot_state() == original == child.snapshot_state()


def test_should_reproduce_all_qualified_saved_train_predictions_without_refit():
    train = ROOT / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    if not FIT.is_file() or not train.is_file():
        pytest.skip("local accepted fit/TRAIN artifacts required for54-row parity qualification")
    raw = cli._qualified_context_fit_bytes(FIT)
    report = json.loads(raw)
    reader = policy.ContextLinearReader(policy.snapshot_from_context_fit_report(report))
    episodes = cli.read_training_sources(train)
    x = np.concatenate([episode_arrays(e)[0] for e in episodes])
    actual = reader.predict(x)
    expected = np.array([p for s in report["scores"] for p in s["probabilities"]], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert actual.argmax(axis=1).tolist() == [s.action for e in episodes for s in e.transitions]


@pytest.mark.parametrize("fault", ["encoding", "rows", "classes", "coefficients", "rank"])
def test_should_refuse_corrupt_complete_snapshot_before_prediction(fault):
    s = state()
    if fault == "encoding":
        object.__setattr__(s.policy, "encoding_id", "foreign")
    elif fault == "coefficients":
        object.__setattr__(s.heads[0], "coefficients", [])
    else:
        name, value = {
            "rows": ("fit_rows", 26),
            "classes": ("num_classes", 6),
            "rank": ("rank", True),
        }[fault]
        object.__setattr__(s.heads[0], name, value)
    with pytest.raises(ValueError):
        policy.ContextLinearReader(s)


@pytest.mark.parametrize(
    "fault", ["dtype", "width", "empty", "nan", "missing", "multiple", "current"]
)
def test_should_refuse_bad_public_input_before_any_head_read(fault, monkeypatch):
    reader = policy.ContextLinearReader(state())
    x = inputs()
    if fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "width":
        x = x[:, :-1]
    elif fault == "empty":
        x = x[:0]
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "missing":
        x[0, 0] = -0.8
    elif fault == "multiple":
        x[0, 3:6] = x[0, :3]
    else:
        x[0, 148] = 0.123
    monkeypatch.setattr(policy, "predict_categorical_linear", lambda *a: pytest.fail("head read"))
    with pytest.raises(ValueError):
        reader.predict(x)


@pytest.mark.parametrize("fault", ["input", "state"])
def test_should_stop_before_second_head_when_read_callback_changes_owned_evidence(
    fault, monkeypatch
):
    reader = policy.ContextLinearReader(state())
    x = inputs()[:2].copy()
    original = policy.predict_categorical_linear
    calls = []

    def predict(*args):
        calls.append(1)
        result = original(*args)
        if fault == "input":
            x[1, 0] = -0.8
        else:
            object.__setattr__(reader._state.heads[0], "squared_error", 1.0)
        return result

    monkeypatch.setattr(policy, "predict_categorical_linear", predict)
    with pytest.raises(ValueError):
        reader.predict(x)
    assert len(calls) == 1


def test_should_refuse_invalid_probability_return_before_continuing(monkeypatch):
    reader = policy.ContextLinearReader(state())
    monkeypatch.setattr(policy, "predict_categorical_linear", lambda *a: np.ones((1, 7)))
    with pytest.raises(ValueError):
        reader.predict(inputs())


def test_should_bind_existing_known_a_plan_and_readout_only_budget(monkeypatch):
    s = state()
    source = policy.ContextLinearReader(s)
    b = budget()

    def run(owner, snapshot, **kwargs):
        assert owner is source and snapshot == s
        assert kwargs["sources"] == () and kwargs["role"] == "knownTRAIN_A_only"
        assert kwargs["updates"] == 0 and kwargs["representation_feature_dim"] == 298
        assert [(x.context, cue, role) for x, cue, role in kwargs["live_plan"]] == [
            ("A", "blue", "train"),
            ("A", "red", "train"),
        ]
        kwargs["extra_check"]()
        assert kwargs["return_metadata"]["source_fit_rows"] == 54
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert (
        evaluation.evaluate_context_linear_checkpoint(
            source, s, budget=b, check_boundary=lambda: None
        )["status"]
        == "complete"
    )


@pytest.mark.parametrize("fault", ["updates", "bool", "replay", "clock", "config"])
def test_should_refuse_original_budget_changes_before_following_work(fault, monkeypatch):
    s = state()
    b = budget()

    def run(*a, **k):
        if fault == "updates":
            b.updates_completed = 1
        elif fault == "bool":
            b.updates_completed = False
        elif fault == "replay":
            b.replay_examples_completed = 1
        elif fault == "clock":
            b.clock = lambda: 1.0
        else:
            object.__setattr__(b.budget, "max_wall_seconds", 91)
        k["extra_check"]()
        pytest.fail("changed budget admitted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_checkpoint(
            policy.ContextLinearReader(s), s, budget=b, check_boundary=lambda: None
        )


@pytest.mark.parametrize("failure", [False, True])
def test_should_execute_original_loop_and_preserve_first_episode_if_second_stops(
    failure, monkeypatch
):
    work: dict[str, Any] = {}
    monkeypatch.setattr(DiscreteEnvironment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", DiscreteEnvironment)
    original_step = DiscreteEnvironment.step

    def step(self, action):
        if failure and self.cue == "red":
            raise RuntimeError("lost red step")
        return original_step(self, action)

    monkeypatch.setattr(DiscreteEnvironment, "step", step)
    s = state()
    source = policy.ContextLinearReader(s)
    try:
        r = evaluation.evaluate_context_linear_checkpoint(
            source, s, budget=budget(), check_boundary=lambda: None
        )
    except evaluation.MiniGridEvaluationFailed as error:
        assert failure
        r = error.report
    assert r["episodes"][0]["status"] == "complete" and r["episodes"][0]["success"]
    assert r["episodes"][1]["status"] == ("failed" if failure else "complete")
    assert r["work"]["raw_score_forwards_attempted"] == 0
    assert r["work"]["live_prediction_calls_returned"] == (3 if failure else 4)
    assert r["work"]["failure_work_may_be_unobserved"] is failure
    assert source.snapshot_state() == s and all(e.closed for e in work["environments"])
    assert all(e.seed == 17 and e.max_steps == 16 for e in work["environments"])


def test_should_refuse_changed_preceding_source_before_any_fork_or_reset(monkeypatch):
    s = state()
    source = policy.ContextLinearReader(s)
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("fork"))
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", lambda *a, **k: pytest.fail("environment"))

    def changed():
        raise ValueError("paid source changed")

    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        evaluation.evaluate_context_linear_checkpoint(
            source, s, budget=budget(), check_boundary=changed
        )
    assert all(e["status"] == "not_run" for e in error.value.report["episodes"])


def test_should_describe_before_source_reader_or_game(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["fit", "--context-linear-game", "--describe"])
    monkeypatch.setattr(cli, "_qualified_context_fit_bytes", lambda *a: pytest.fail("source"))
    monkeypatch.setattr(cli, "run_saved_context_linear_games", lambda *a: pytest.fail("game"))
    cli.main()
    assert json.loads(capsys.readouterr().out)["new_fit_calls"] == 0


@pytest.mark.parametrize("option", ["--context-linear-fit", "--fit-updates", "--train-a-rollout"])
def test_should_refuse_conflicting_cli_options_before_source(option, monkeypatch):
    args = ["fit", "--context-linear-game", option] + (["16"] if option == "--fit-updates" else [])
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(cli, "_qualified_context_fit_bytes", lambda *a: pytest.fail("source"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_reject_replaced_qualified_bytes_before_reader_or_environment(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf8")
    monkeypatch.setattr(cli, "ContextLinearReader", lambda *a: pytest.fail("reader"))
    r = cli.run_saved_context_linear_games(bad)
    assert r["status"] == "failed" and r["games"] is None and r["source_unchanged"] is None


def test_should_preserve_failed_game_prefix_in_cli_composition(monkeypatch):
    if not FIT.is_file():
        pytest.skip("local accepted affine-fit artifact required for this CLI integration check")
    partial = dict(
        status="failed",
        reason="lost second game",
        episodes=[dict(status="complete"), dict(status="failed")],
        work=dict(failure_work_may_be_unobserved=True),
    )

    error = evaluation.MiniGridEvaluationFailed(partial)

    def stopped(*a, **k):
        raise error

    monkeypatch.setattr(evaluation, "evaluate_context_linear_checkpoint", stopped)
    r = cli.run_saved_context_linear_games(FIT)
    assert r["status"] == "failed" and r["games"] is error.report
    assert r["games"] == partial
    assert r["budget"]["updates_completed"] == 0 and r["source_unchanged"] is None


def test_should_dispatch_game_without_reading_train_sources_or_fitting(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "new.json"
    monkeypatch.setattr(
        "sys.argv", ["fit", "--context-linear-game", "--source", "unused", "--output", str(path)]
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN reader"))
    monkeypatch.setattr(cli, "run_saved_context_linear_games", lambda *a: dict(status="complete"))
    cli.main()
    assert json.loads(path.read_text(encoding="utf8"))["status"] == "complete"
    capsys.readouterr()
    with pytest.raises(FileExistsError):
        cli.main()
