"""Causal native diagnostics on the real stream, literal heads, no solver/physics."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_shift_diagnostics as diagnostic
from src.adapters import minigrid_context_linear_learner as port
from src.app import coupled_attribution as app
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import episode_arrays
from src.core.reference_controller import ReferencePolicy
from tests.test_arrived_cue_catalogue import sources, schedule
from tests.test_minigrid_controller_bank import BankLiteral

WORK = dict(
    owners=0, snapshots=0, fits=0, fit_rows=0, predicts=0, predict_rows=0, callbacks=0, metrics=0
)
HOOK: dict[str, Any] = {}


class LiteralNative(BankLiteral):
    def __init__(self):
        WORK["owners"] += 1
        super().__init__()

    def fork(self):
        child = LiteralNative()
        child.restore_state(self.snapshot_state())
        return child

    def snapshot_state(self):
        WORK["snapshots"] += 1
        return super().snapshot_state()

    def train_batch(self, x, y):
        WORK["fits"] += 1
        WORK["fit_rows"] += len(x)
        return super().train_batch(x, y)

    def predict(self, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        result = super().predict(x)
        if "predict" in HOOK:
            HOOK["predict"](self, x)
        return result


@pytest.fixture(autouse=True)
def literal_only(monkeypatch):
    HOOK.clear()
    monkeypatch.setattr(port, "ContextLinearLearner", LiteralNative)
    monkeypatch.setattr(diagnostic, "ContextLinearLearner", LiteralNative)
    monkeypatch.setattr(BankLiteral, "hook", staticmethod(lambda *a: None))
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: pytest.fail("solver reached"))
    original_callback = app.AttributionStream._observe_reference_arrival
    original_metric = diagnostic.summarize_categorical_shift_batch

    def callback(*a, **kw):
        WORK["callbacks"] += 1
        return original_callback(*a, **kw)

    def metric(*a, **kw):
        WORK["metrics"] += 1
        return original_metric(*a, **kw)

    monkeypatch.setattr(app.AttributionStream, "_observe_reference_arrival", callback)
    monkeypatch.setattr(diagnostic, "summarize_categorical_shift_batch", metric)
    yield
    HOOK.clear()


def batches():
    result = []
    for episode in schedule():
        x, y = episode_arrays(episode)
        result.append(
            app.AttributionBatch(
                "train",
                episode.context,
                tuple(f"{episode.episode_id}:step:{i}" for i in range(9)),
                x,
                y,
                episode,
                7,
            )
        )
    return tuple(result)


def stream():
    return app.AttributionStream(
        LiteralNative(),
        batches(),
        (),
        training_only=True,
        checkpoint_updates=tuple(range(7)),
        label_delay=1,
        budget=ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=6, max_wall_seconds=90), lambda: 0.0
        ),
    )


def test_should_summarize_all_public_channels_and_directions_without_labels():
    x = np.zeros((2, 296), dtype=np.float64)
    for frame in range(2):
        for channel in range(3):
            x[:, frame * 148 + channel : frame * 148 + 147 : 3] = (channel - 1) * 0.25 + frame * 0.1
        x[:, frame * 148 + 147] = [-0.5, 0.5]
    before = x.copy()
    s = diagnostic.public_feature_summary(x)
    assert s.shape == (2, 8) and np.array_equal(x, before)
    assert np.allclose(s[0], [-0.25, 0, 0.25, -0.5, -0.15, 0.1, 0.35, -0.5])
    assert np.allclose(s[1, [3, 7]], [0.5, 0.5])
    assert not np.shares_memory(s, x)


@pytest.mark.parametrize("bad", ["width", "rows", "dtype", "nan", "range"])
def test_should_refuse_invalid_public_inputs_before_summary(bad):
    x = np.zeros((9, 296), dtype=np.float64)
    if bad == "width":
        x = x[:, :-1]
    elif bad == "rows":
        x = np.zeros((129, 296))
    elif bad == "dtype":
        x = x.astype(np.float32)
    elif bad == "nan":
        x[0, 0] = np.nan
    else:
        x[0, 0] = 1.01
    with pytest.raises(ValueError):
        diagnostic.public_feature_summary(x)


def test_should_observe_every_label_before_decision_even_without_updates():
    owner = stream()
    observed = []

    def see(tick, index, batch, candidate):
        assert tick == index + 1 and candidate.wake_updates == 0
        assert batch.row_ids == batches()[index].row_ids and batch.role == "train"
        # The delivered data are owned copies; changing them grants no live access.
        batch.features[:] = 0
        observed.append(index)
        with pytest.raises(ValueError):
            owner.run_until(1)

    r = owner.run_reference_window(ReferencePolicy("no_learning"), observe_arrived=see)
    assert observed == list(range(6)) and r["completed_updates"] == 0
    assert r["pending_unapplied"] == 6 and r["arrived_observation"]["calls_completed"] == 6
    assert np.array_equal(owner._training[0].features, batches()[0].features)


@pytest.mark.parametrize(
    "fault", ["source", "actor", "candidate", "clock", "budget", "input", "reentrant"]
)
def test_should_refuse_callback_changes_before_any_fit_or_next_read(fault):
    owner = stream()

    def corrupt(*args):
        if fault == "source":
            owner._learner.state = replace(owner._learner.state, completed_updates=1)
        elif fault == "actor":
            owner._runtime.actor._learner.state = replace(
                owner._runtime.actor._learner.state, completed_updates=1
            )
        elif fault == "candidate":
            owner._runtime._candidate.state = replace(
                owner._runtime._candidate.state, completed_updates=1
            )
        elif fault == "clock":
            owner._clock.advance_to(2)
        elif fault == "budget":
            owner._budget.clock = lambda: 0.1
        elif fault == "input":
            owner._training[0].features[0, 0] += 0.01
        else:
            owner.run_until(1)

    with pytest.raises((ValueError, RuntimeError)):
        owner.run_reference_window(ReferencePolicy("periodic"), observe_arrived=corrupt)
    r = owner.reference_report
    assert r["status"] == "failed" and r["completed_updates"] == 0
    assert r["arrived_observation"]["calls_attempted"] == 1
    assert r["arrived_observation"]["calls_completed"] == 0
    assert r["arrived_observation"]["unreturned_work_unknown"] is True


@pytest.mark.parametrize("replacement", [False, True])
def test_should_refuse_changed_preceding_arrival_charges(replacement):
    owner = stream()

    def corrupt(tick, index, *args):
        value = owner._reference_report["arrived_observation"]
        if replacement:
            owner._reference_report["arrived_observation"] = deepcopy(value)
        else:
            value["calls_completed"] = 99

    with pytest.raises(ValueError):
        owner.run_reference_window(ReferencePolicy("no_learning"), observe_arrived=corrupt)
    assert owner.reference_report["arrived_observation"]["calls_completed"] == 0


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware", "drift"])
def test_should_keep_choices_receipts_states_and_rng_while_diagnosing_all_arrivals(mode):
    options: dict[str, Any] = dict(reference_policy=ReferencePolicy(mode), retain_arrived=True)
    if mode == "drift":
        options["ready_feedback"] = True
    episodes = sources()
    before = pickle.dumps((episodes, random.getstate(), np.random.get_state()))
    control = cli.run_context_linear_stream(episodes, **options)
    run = cli.run_context_linear_stream(episodes, shift_diagnostics=True, **options)
    assert run["status"] == control["status"] == "complete", run.get("reason")
    assert pickle.dumps((episodes, random.getstate(), np.random.get_state())) == before
    assert run["committed_receipts"] == control["committed_receipts"]
    assert run["unscored_checkpoints"] == control["unscored_checkpoints"]
    assert run["stream"]["decisions"] == control["stream"]["decisions"]
    assert run["source_unchanged"] and run["actor_unchanged"]
    d = run["shift_diagnostics"]
    assert "shift_diagnostics" not in control and "arrived_observation" not in control["stream"]
    assert d["completed_arrivals"] == 6 and d["prediction_rows_returned"] == 54
    assert run["stream"]["arrived_observation"]["rows_completed"] == 90  # joint fit template
    assert all(
        r["metrics"]["rows"] == 9 and len(r["metrics"]["feature_mean"]) == 8 for r in d["rows"]
    )
    assert all(r["candidate_updates"] <= r["arrival_index"] for r in d["rows"])
    assert d["feedback"] is False and d["seconds"] >= 0
    assert d["explicit_array_bytes_created"] == 6 * 9 * (296 * 2 + 1 + 8 + 7) * 8
    if mode == "no_learning":
        assert all(r["candidate_updates"] == 0 for r in d["rows"])
        assert run["budget"]["updates_completed"] == 0 and run["stream"]["pending_unapplied"] == 6
        assert all(np.isclose(r["metrics"]["uncertainty"], 1) for r in d["rows"])


@pytest.mark.parametrize("fault", ["lost", "mutation", "invalid"])
def test_should_preserve_first_diagnostic_and_paid_failure_without_further_learning(fault):
    calls = []

    def damage(owner, x):
        calls.append(1)
        if len(calls) == 2:
            if fault == "lost":
                raise RuntimeError("lost prediction return")
            if fault == "mutation":
                x[0, 0] += 0.01
            else:
                owner.state = replace(owner.state, completed_updates=1)

    HOOK["predict"] = damage
    r = cli.run_context_linear_stream(
        sources(), reference_policy=ReferencePolicy("no_learning"), shift_diagnostics=True
    )
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 0
    d = r["shift_diagnostics"]
    assert [x["status"] for x in d["rows"]] == ["complete", "failed"] + ["not_run"] * 4
    assert d["prediction_calls_attempted"] == 2 and d["prediction_calls_returned"] == (
        1 if fault == "lost" else 2
    )
    assert d["unreturned_prediction_unknown"] is (fault == "lost")
    assert r["stream"]["arrived_observation"]["unreturned_work_unknown"]


@pytest.mark.parametrize(
    "options",
    [dict(shift_diagnostics=1), dict(shift_diagnostics=True), dict(shift_diagnostics=None)],
)
def test_should_refuse_invalid_binding_before_models(options, monkeypatch):
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("model reached"))
    with pytest.raises(ValueError, match="shift_diagnostics"):
        cli.run_context_linear_stream(sources(), **options)


def test_should_keep_legacy_shift_feature_bound_and_detached_reports():
    from src.app.arrived_shift_observer import ArrivedShiftObserver

    observer = ArrivedShiftObserver(lambda *a: pytest.fail("prediction reached"), num_classes=7)
    observer(0, None, None)
    with pytest.raises(ValueError, match="128x16"):
        observer(1, None, batches()[0])
    d = diagnostic.MiniGridShiftDiagnostics()
    r = d.report()
    r["rows"][0]["status"] = "changed"
    assert d.report()["rows"][0]["status"] == "not_run"


@pytest.mark.parametrize("fault", ["start", "backward", "end"])
def test_should_not_count_an_arrival_complete_after_failed_timing(monkeypatch, fault):
    from src.core.actor_ports import CandidateState

    d = diagnostic.MiniGridShiftDiagnostics()
    values = iter(
        [float("nan")] if fault == "start" else [2.0, 1.0 if fault == "backward" else float("inf")]
    )
    monkeypatch.setattr(diagnostic, "monotonic", lambda: next(values))
    candidate = CandidateState("actor", "learner", LiteralNative().snapshot_state(), 0, 0, 0)
    with pytest.raises(ValueError):
        d.observe(1, 0, batches()[0], candidate)
    r = d.report()
    assert r["completed_arrivals"] == 0 and r["stopped"]
    assert r["rows"][0]["status"] == "failed" and r["rows"][0]["seconds"] is None
    assert r["prediction_calls_attempted"] == (0 if fault == "start" else 1)


def test_should_charge_a_returned_invalid_prediction_and_failed_metric(monkeypatch):
    original = LiteralNative.predict

    def invalid(owner, x):
        return original(owner, x)[:, :6].copy()

    monkeypatch.setattr(LiteralNative, "predict", invalid)
    r = cli.run_context_linear_stream(
        sources(), reference_policy=ReferencePolicy("no_learning"), shift_diagnostics=True
    )
    d = r["shift_diagnostics"]
    assert r["status"] == "failed" and d["prediction_calls_returned"] == 1
    assert not d["unreturned_prediction_unknown"] and d["completed_arrivals"] == 0
    assert d["summary_calls_attempted"] == 1 and d["summary_calls_completed"] == 0


@pytest.mark.parametrize("count", [False, True, 0, -1, 65, 1.0, "12", None])
def test_should_refuse_invalid_horizons_before_allocating_a_reader(monkeypatch, count):
    monkeypatch.setattr(diagnostic, "ContextLinearLearner", lambda: pytest.fail("reader allocated"))
    with pytest.raises(ValueError, match="max_arrivals"):
        diagnostic.MiniGridShiftDiagnostics(max_arrivals=count)
    with pytest.raises(ValueError, match="max_arrivals"):
        diagnostic.describe(max_arrivals=count)


@pytest.mark.parametrize("count", [1, 12, 64])
def test_should_allocate_exactly_the_declared_bounded_scalar_slots(count):
    observer = diagnostic.MiniGridShiftDiagnostics(max_arrivals=count)
    report = observer.report()
    assert report["protocol"]["arrivals"] == count
    assert len(report["rows"]) == count and report["completed_arrivals"] == 0
    assert [row["arrival_index"] for row in report["rows"]] == list(range(count))
    assert all(row["status"] == "not_run" for row in report["rows"])
    report["rows"].append(dict(status="corrupted"))
    report["protocol"]["arrivals"] = 1000
    assert len(observer.report()["rows"]) == count
    assert observer.report()["protocol"] == diagnostic.describe(max_arrivals=count)


def test_should_preserve_the_six_arrival_default_and_require_explicit_keywords():
    assert diagnostic.describe() == diagnostic.describe(max_arrivals=6)
    assert (
        diagnostic.MiniGridShiftDiagnostics().report()
        == diagnostic.MiniGridShiftDiagnostics(max_arrivals=6).report()
    )
    with pytest.raises(TypeError):
        constructor: Any = diagnostic.MiniGridShiftDiagnostics
        constructor(12)  # Exercise an invalid runtime caller deliberately.


def long_batches():
    result = []
    for index in range(12):
        original = schedule()[index % 6]
        episode = replace(original, episode_id=f"long-shift-{index}")
        x, y = episode_arrays(episode)
        result.append(
            app.AttributionBatch(
                "train",
                episode.context,
                tuple(f"{episode.episode_id}:step:{i}" for i in range(9)),
                x,
                y,
                episode,
                7,
            )
        )
    return tuple(result)


@pytest.mark.parametrize("mode,expected_updates", [("no_learning", 0), ("periodic", 6)])
def test_should_observe_twelve_arrivals_without_changing_reference_work(mode, expected_updates):
    from src.app.probe_return_phase import budget_binding
    from src.shared.process_memory import ProcessRssSampler

    training = long_batches()
    pinned = pickle.dumps((training, random.getstate(), np.random.get_state()))
    terminal_states = []
    reports = []
    observer = diagnostic.MiniGridShiftDiagnostics(max_arrivals=12)
    with ProcessRssSampler(interval_seconds=0.01) as memory:
        for enabled in (False, True):
            source = LiteralNative()
            initial = source.snapshot_state()
            limits = ToyExecutionBudget(
                max_training_updates=12,
                max_wall_seconds=90,
                max_process_rss_bytes=512 * 1024**2,
            )
            budget = ToyBudgetSession(limits, lambda: 0.0)
            budget.attach_memory(memory)
            binding = budget_binding(budget)
            latest: list[Any] = []

            def checkpoint(n, state, batch):
                latest[:] = [state]  # Keep only the last paid callback state.

            owner = app.AttributionStream(
                source,
                training,
                (),
                training_only=True,
                checkpoint_updates=tuple(range(13)),
                label_delay=1,
                budget=budget,
                observe_checkpoint=checkpoint,
            )
            actor = owner._runtime.actor.snapshot_state()
            report = owner.run_reference_window(
                ReferencePolicy(mode),
                observe_arrived=observer.observe if enabled else None,
            )
            assert report["status"] == "complete", report
            assert report["completed_updates"] == budget.updates_completed == expected_updates
            assert budget.replay_examples_completed == 0
            assert budget.budget is limits and budget_binding(budget) == binding
            assert source.snapshot_state() == initial
            assert owner._runtime.actor.snapshot_state() == actor
            terminal_states.append(latest[0])
            reports.append(report)
    assert terminal_states[0] == terminal_states[1]
    assert reports[0]["decisions"] == reports[1]["decisions"]
    assert reports[0]["applied_updates"] == reports[1]["applied_updates"]
    assert pickle.dumps((training, random.getstate(), np.random.get_state())) == pinned
    d = observer.report()
    assert d["completed_arrivals"] == d["prediction_calls_returned"] == 12
    assert d["prediction_rows_returned"] == 108 and len(d["rows"]) == 12
    assert all(row["status"] == "complete" for row in d["rows"])
    assert all(row["candidate_updates"] <= row["arrival_index"] for row in d["rows"])
    assert d["explicit_array_bytes_created"] == 12 * 9 * (296 * 2 + 1 + 8 + 7) * 8
    assert d["feedback"] is False
    if mode == "no_learning":
        assert all(row["candidate_updates"] == 0 for row in d["rows"])
        assert reports[1]["pending_unapplied"] == 12


def test_should_refuse_an_arrival_beyond_the_bound_before_another_prediction():
    from src.core.actor_ports import CandidateState

    observer = diagnostic.MiniGridShiftDiagnostics(max_arrivals=1)
    candidate = CandidateState("actor", "learner", LiteralNative().snapshot_state(), 0, 0, 0)
    observer.observe(1, 0, batches()[0], candidate)
    paid = observer.report()
    before = dict(WORK)
    with pytest.raises(ValueError, match="next delay-one"):
        observer.observe(2, 1, batches()[1], candidate)
    assert observer.report() == paid and WORK == before


def test_should_keep_a_late_failed_arrival_and_leave_remaining_slots_unrun():
    calls = []

    def lose_eighth(owner, x):
        calls.append(1)
        if len(calls) == 8:
            raise RuntimeError("lost eighth prediction")

    HOOK["predict"] = lose_eighth
    observer = diagnostic.MiniGridShiftDiagnostics(max_arrivals=12)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=12, max_wall_seconds=90), lambda: 0.0
    )
    owner = app.AttributionStream(
        LiteralNative(),
        long_batches(),
        (),
        training_only=True,
        checkpoint_updates=tuple(range(13)),
        label_delay=1,
        budget=budget,
    )
    with pytest.raises(RuntimeError, match="lost eighth"):
        owner.run_reference_window(ReferencePolicy("no_learning"), observe_arrived=observer.observe)
    d = observer.report()
    assert [row["status"] for row in d["rows"]] == ["complete"] * 7 + ["failed"] + ["not_run"] * 4
    assert d["completed_arrivals"] == 7 and d["stopped"]
    assert d["prediction_calls_attempted"] == 8 and d["prediction_calls_returned"] == 7
    assert d["unreturned_prediction_unknown"] and budget.updates_completed == 0
    failed = owner.reference_report
    assert failed is not None
    assert failed["arrived_observation"]["calls_completed"] == 7
    assert failed["arrived_observation"]["unreturned_work_unknown"]
