"""Real affine updates, detached ownership and the original delayed TRAIN gate."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_context_linear_learner as port
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingUpdateFailed
from scripts import run_minigrid_linear_fit_control as cli


def inputs(cue=2):
    frame = [-0.8, -1.0, -1.0] * 49 + [-1.0]
    frame[:3] = [0.2, 2 * cue / 5 - 1, -1.0]
    return np.array([frame + frame], dtype=np.float64)


def targets(action=2):
    return np.array([[float(action)]])


def test_should_begin_unfitted_and_learn_replace_and_recover_without_replay():
    learner = port.ContextLinearLearner()
    x = inputs()
    rng = pickle.dumps(np.random.get_state())
    assert learner.snapshot_state().heads == (None, None)
    np.testing.assert_array_equal(learner.predict(x), np.full((1, 7), 1 / 7))
    results = []
    for action in (2, 5, 2):
        diagnostic = learner.train_batch(x, targets(action))
        assert diagnostic.definition == port.DIAGNOSTIC and diagnostic.value < 1e-20
        results.append(int(learner.predict(x).argmax()))
    assert results == [2, 5, 2]  # Replacing B forgets A at the same observed input.
    state = learner.snapshot_state()
    assert state.completed_updates == state.training_examples == 3
    assert state.heads[0].fit_rows == 1 and state.heads[1] is None
    assert pickle.dumps(np.random.get_state()) == rng


def test_should_keep_other_head_and_detach_full_forks_snapshots_restore():
    learner = port.ContextLinearLearner()
    learner.train_batch(inputs(), targets(2))
    blue = learner.snapshot_state()
    child = learner.fork()
    child.train_batch(inputs(0), targets(5))
    full = child.snapshot_state()
    assert learner.snapshot_state() == blue and full.heads[0] == blue.heads[0]
    assert full.heads[0] is not blue.heads[0]
    assert full.heads[0].coefficients is blue.heads[0].coefficients
    mixed = np.concatenate([inputs(0), inputs(), inputs(0)])
    assert child.predict(mixed).argmax(axis=1).tolist() == [5, 2, 5]
    child.restore_state(blue)
    assert child.snapshot_state() == blue
    np.testing.assert_array_equal(child.predict(inputs(0)), np.full((1, 7), 1 / 7))
    object.__setattr__(blue.heads[0], "squared_error", 1.0)
    assert learner.snapshot_state() != blue and child.snapshot_state() != blue


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "width",
        "empty",
        "nan",
        "offgrid",
        "mixed",
        "target_dtype",
        "fractional",
        "range",
        "target_shape",
    ],
)
def test_should_refuse_bad_batches_before_solver(monkeypatch, fault):
    learner = port.ContextLinearLearner()
    x, y = inputs(), targets()
    if fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "width":
        x = x[:, :-1]
    elif fault == "empty":
        x, y = x[:0], y[:0]
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "offgrid":
        x[0, 0] = 0.123
    elif fault == "mixed":
        x, y = np.concatenate([x, inputs(0)]), np.concatenate([y, y])
    elif fault == "target_dtype":
        y = y.astype(np.int64)
    elif fault == "fractional":
        y[0, 0] = 1.5
    elif fault == "range":
        y[0, 0] = 7
    else:
        y = y[:, 0]
    before = learner.snapshot_state()
    monkeypatch.setattr(port, "fit_categorical_linear", lambda *a: pytest.fail("solver reached"))
    with pytest.raises(ValueError):
        learner.train_batch(x, y)
    assert learner.snapshot_state() == before


@pytest.mark.parametrize("fault", ["raise", "features", "targets", "state", "geometry"])
def test_should_preserve_failed_work_preceding_fit_and_refuse_retry(monkeypatch, fault):
    learner = port.ContextLinearLearner()
    learner.train_batch(inputs(), targets())
    before = learner.snapshot_state()
    x, y = inputs(0), targets(5)
    original = port.fit_categorical_linear

    def fit(features, labels, classes):
        if fault == "raise":
            raise RuntimeError("unknown native solver failure")
        model = original(features, labels, classes)
        if fault == "features":
            features.fill(0)
        elif fault == "targets":
            y.fill(1)
        elif fault == "state":
            learner._state = replace(before, training_examples=2)
        else:
            model = replace(model, fit_rows=2, singular_values=(1.0, 0.0))
        return model

    monkeypatch.setattr(port, "fit_categorical_linear", fit)
    with pytest.raises(TrainingUpdateFailed) as caught:
        learner.train_batch(x, y)
    work = caught.value.work
    assert work["fit_calls_attempted"] == work["fit_rows_attempted"] == 1
    assert work["fit_calls_completed"] == work["fit_rows_completed"] == (fault != "raise")
    assert work["failure_work_may_be_unobserved"] is True
    failed = learner.snapshot_state()
    assert failed.heads == before.heads
    assert failed.completed_updates == failed.training_examples == 1
    work["fit_rows_attempted"] = 0
    assert caught.value.work["fit_rows_attempted"] == 1
    with pytest.raises(ValueError, match="paid work"):
        learner.restore_state(before)
    monkeypatch.setattr(port, "fit_categorical_linear", lambda *a: pytest.fail("retry reached"))
    for owner in (learner, learner.fork()):
        with pytest.raises(TrainingUpdateFailed):
            owner.train_batch(inputs(), targets())


def test_should_reject_corrupted_snapshot_counters_and_model_evidence():
    learner = port.ContextLinearLearner()
    learner.train_batch(inputs(), targets())
    valid = learner.snapshot_state()
    for state in (
        replace(valid, completed_updates=True),
        replace(valid, training_examples=0),
        replace(valid, heads=(None, None)),
        replace(valid, failed_work=(1, 1, 1, 0, True)),
    ):
        with pytest.raises(ValueError):
            learner.restore_state(state)
    object.__setattr__(valid.heads[0], "coefficients", ())
    with pytest.raises(ValueError):
        learner.restore_state(valid)
    assert learner.snapshot_state().completed_updates == 1


def test_should_use_original_delayed_stream_receipts_without_training_source_actor():
    source = port.ContextLinearLearner()
    before = source.snapshot_state()
    training = tuple(
        AttributionBatch("train", name, (f"row-{i}",), inputs(cue), targets(action), num_classes=7)
        for i, (name, cue, action) in enumerate((("A", 2, 2), ("A", 0, 5), ("B", 2, 5)))
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    observations: list[tuple[int, Any]] = []
    stream = AttributionStream(
        source,
        training,
        (),
        checkpoint_updates=(0, 2, 3),
        label_delay=2,
        budget=budget,
        training_only=True,
        observe_checkpoint=lambda n, s, b: observations.append((n, s)),
    )
    assert stream.candidate_snapshot() == before and budget.updates_completed == 0
    stream.run_until(2)
    assert source.snapshot_state() == before and budget.updates_completed == 2
    stream.run_until(3)
    state = stream.candidate_snapshot()
    result = stream.finish()
    assert state.completed_updates == state.training_examples == 3
    assert result["evaluation_scope"] == "training_only" and result["score_forward_calls"] == 0
    assert [n for n, s in observations] == [0, 2, 3]
    assert observations[1][1].heads[0] != state.heads[0]
    assert observations[1][1].heads[1] == state.heads[1]
    for i, receipt in enumerate(result["applied_updates"]):
        assert receipt["observed_at"] == i
        assert receipt["arrived_at"] == receipt["applied_at"] == i + 2
        assert receipt["update_number"] == i + 1
    assert source.snapshot_state() == before
    assert stream._runtime.actor.snapshot_state().state == before


@pytest.mark.parametrize("role", ["final_test", "controller_validation"])
def test_should_refuse_nontrain_batches_before_owner_fork(monkeypatch, role):
    source = port.ContextLinearLearner()
    batch = AttributionBatch(role, "A", ("row",), inputs(), targets(), num_classes=7)
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("fork reached"))
    with pytest.raises(ValueError):
        AttributionStream(
            source,
            (batch,),
            (),
            checkpoint_updates=(0, 1),
            label_delay=1,
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
            training_only=True,
        )


@pytest.mark.parametrize("fault", ["input", "state"])
def test_should_refuse_prediction_side_effects_before_following_head(monkeypatch, fault):
    learner = port.ContextLinearLearner()
    learner.train_batch(inputs(), targets())
    x = np.concatenate([inputs(), inputs(0)])
    before = learner.snapshot_state()
    original = port.predict_categorical_linear
    calls = []

    def predict(model, features):
        calls.append(len(features))
        value = original(model, features)
        if fault == "input":
            x.fill(0)
        else:
            learner._state = replace(before, training_examples=2)
        return value

    monkeypatch.setattr(port, "predict_categorical_linear", predict)
    with pytest.raises(ValueError, match="changed"):
        learner.predict(x)
    assert calls == [1]


def test_should_leave_failed_native_call_without_successful_stream_charge(monkeypatch):
    source = port.ContextLinearLearner()
    batch = AttributionBatch("train", "A", ("row",), inputs(), targets(), num_classes=7)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    stream = AttributionStream(
        source,
        (batch,),
        (),
        checkpoint_updates=(0, 1),
        label_delay=1,
        budget=budget,
        training_only=True,
    )

    def fail(*args):
        raise RuntimeError("native unknown")

    monkeypatch.setattr(port, "fit_categorical_linear", fail)
    with pytest.raises(TrainingUpdateFailed) as caught:
        stream.run_until(1)
    assert caught.value.work["fit_rows_attempted"] == 1
    assert budget.updates_completed == 0 and not stream._runtime.applied_updates
    assert source.snapshot_state().heads == (None, None)
    assert stream._runtime._candidate.snapshot_state().failed_work == (1, 1, 0, 0, True)
    with pytest.raises(ValueError):
        stream.run_until(1)


@pytest.mark.parametrize(
    "args",
    [
        ["--context-linear-stream", "--describe"],
        ["--context-linear-stream", "--context-linear-fit"],
        ["--context-linear-stream", "--fit-updates", "1"],
    ],
)
def test_should_describe_without_io_and_refuse_conflicting_cli_options(monkeypatch, capsys, args):
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source opened"))
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("owner created"))
    if "--describe" in args:
        cli.main()
        assert json.loads(capsys.readouterr().out)["fit_calls"] == 6
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_route_cli_and_keep_exclusive_output_and_failure_prefix(monkeypatch, tmp_path):
    destination = tmp_path / "result.json"
    report = dict(
        status="failed", endpoints=[dict(updates=0)], failure_work=dict(fit_rows_attempted=9)
    )
    monkeypatch.setattr(
        "sys.argv",
        ["control", "--context-linear-stream", "--source", "source", "--output", str(destination)],
    )
    sentinel = (object(),)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: sentinel)

    def run(episodes):
        assert episodes is sentinel
        return report.copy()

    monkeypatch.setattr(cli, "run_context_linear_stream", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert json.loads(destination.read_text(encoding="utf8"))["endpoints"] == report["endpoints"]
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a: pytest.fail("repeat reached"))
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_fit_one_original_a_episode_without_future_heads_or_labels():
    path = (
        Path(__file__).resolve().parents[1] / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    )
    if not path.is_file():
        pytest.skip("accepted local TRAIN artifact unavailable")
    from src.core.demonstration_episode import episode_arrays

    episode = cli.read_training_sources(path)[0]
    x, y = episode_arrays(episode)
    learner = port.ContextLinearLearner()
    original = pickle.dumps((x, y))
    learner.train_batch(x, y)
    assert learner.predict(x).argmax(axis=1).tolist() == y[:, 0].astype(int).tolist()
    state = learner.snapshot_state()
    assert state.completed_updates == 1 and state.training_examples == 9
    assert state.heads[0].fit_rows == 9 and state.heads[1] is None
    assert pickle.dumps((x, y)) == original


def test_should_preserve_real_paid_prefix_when_third_stream_solve_fails(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1] / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    )
    if not path.is_file():
        pytest.skip("accepted local TRAIN artifact unavailable")
    episodes = cli.read_training_sources(path)
    before = pickle.dumps(episodes)
    original = port.fit_categorical_linear
    calls = []

    def fit(x, y, classes):
        calls.append(len(x))
        if len(calls) == 3:
            raise RuntimeError("deliberate third-solve failure")
        return original(x, y, classes)

    monkeypatch.setattr(port, "fit_categorical_linear", fit)
    result = cli.run_context_linear_stream(episodes)
    assert result["status"] == "failed" and result["stream"] is None
    assert calls == [9, 9, 9]
    assert result["observed_checkpoints"] == [0, 2]
    assert [row["updates"] for row in result["unscored_checkpoints"]] == [0, 2]
    assert result["unscored_checkpoints"][1]["state"]["training_examples"] == 18
    assert len(result["committed_receipts"]) == result["budget"]["updates_completed"] == 2
    assert result["failure_work"]["work"]["fit_rows_attempted"] == 9
    assert result["failure_work"]["work"]["failure_work_may_be_unobserved"] is True
    assert result["source_unchanged"] is result["actor_unchanged"] is None
    assert result["score_calls_attempted"] == result["score_rows_attempted"] == 0
    assert result["receipt_capture_error"] is None
    assert pickle.dumps(episodes) == before


def test_should_fail_completion_if_receipt_capture_is_unavailable(monkeypatch):
    from src.app import coupled_attribution
    from types import SimpleNamespace

    # A literal wrapper fault, not a trained stream or task-quality observation.
    monkeypatch.setattr(
        "src.adapters.minigrid_train_catalogue._validate_sources",
        lambda *a: (SimpleNamespace(check=lambda *a: None), 36),
    )
    episodes = tuple(
        SimpleNamespace(
            episode_id=f"minigrid-source-{context}-{cue}",
            transitions=(SimpleNamespace(step=0),) * 9,
            context=context,
        )
        for context in ("A", "B")
        for cue in ("blue", "red")
    )
    monkeypatch.setattr(cli, "replace", lambda episode, **kw: episode)
    monkeypatch.setattr(
        "src.core.demonstration_episode.episode_arrays", lambda *a: (inputs(), targets())
    )

    class ReceiptFault:
        def __init__(self, *a, **kw):
            pass

        def run_until(self, *a):
            pass

        def finish(self):
            return dict(literal_fixture=True)

        @property
        def applied_updates(self):
            raise ValueError("preceding receipt evidence unavailable")

    monkeypatch.setattr(coupled_attribution, "AttributionStream", ReceiptFault)
    result = cli.run_context_linear_stream(episodes)
    assert result["status"] == "failed" and result["reason"] == "committed receipt capture failed"
    assert result["committed_receipts"] is None
    assert result["receipt_capture_error"].startswith("ValueError: preceding receipt")
    assert result["budget"]["updates_completed"] == result["score_calls_attempted"] == 0
