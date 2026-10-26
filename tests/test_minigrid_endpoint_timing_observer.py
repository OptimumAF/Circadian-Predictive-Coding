"""Timing publication on scripted H32 ports; no native models or physics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.adapters import minigrid_probe_rollout as live
from src.adapters import minigrid_endpoint_timing as timing


@pytest.fixture
def scripted(monkeypatch):
    clock = [10.0]
    budget_clock = [10.0]
    calls = []
    monkeypatch.setattr(timing, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(live, "monotonic", lambda: budget_clock[0])
    monkeypatch.setattr(
        live, "_copy_wide_neural_probe_snapshot", lambda value, **kw: deepcopy(value)
    )
    monkeypatch.setattr(live, "_same_state", lambda left, right: left == right)
    state = SimpleNamespace(policy="scripted", completed_updates=16, training_examples=144)
    learner = SimpleNamespace(snapshot_state=lambda: deepcopy(state))

    def evaluate(owner, expected, **options):
        calls.append(options)
        timer = options["timings"]
        for phase, seconds in (
            ("preservation", 3),
            ("input_preparation", 2),
            ("prediction", 1),
            ("physics", 4),
        ):
            with timer.measure(phase):
                clock[0] += seconds
        return dict(
            status="complete",
            phase_timings=timer.report(),
            work=dict(
                environment_resets_completed=4,
                live_prediction_calls_returned=4,
                environment_steps_returned=4,
                internal_projection_rows_returned=4,
                internal_projection_array_bytes_returned=9472,
            ),
        )

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", evaluate)
    return SimpleNamespace(clock=clock, budget_clock=budget_clock, calls=calls, learner=learner)


def test_should_publish_sealed_h32_timings_without_changing_budget_or_work(scripted):
    observer = live.ProbeRollouts(0, wide_method="bp")
    observer.observe(scripted.learner, "after_probe", lambda: None)
    row = observer.records[0]
    assert row["status"] == "complete" and row["budget_binding_preserved"] is True
    assert row["phase_timings"] == row["report"]["phase_timings"]
    assert {k: v["seconds"] for k, v in row["phase_timings"]["phases"].items()} == dict(
        preservation=3, input_preparation=2, prediction=1, physics=4
    )
    assert row["elapsed_seconds"] == 0
    assert row["charged_updates"] == row["replay_examples"] == 0
    budget = scripted.calls[0]["budget"]
    assert budget.budget.max_wall_seconds == 40
    assert budget.clock() == budget.started_at == 10
    observer.check()
    row["phase_timings"]["phases"]["prediction"]["seconds"] = 0
    with pytest.raises(ValueError):
        observer.observe(scripted.learner, "after_return", lambda: None)
    assert len(scripted.calls) == 1
    observer.restore_paid()
    assert observer.records[0]["phase_timings"]["phases"]["prediction"]["seconds"] == 1


def test_should_keep_failed_elapsed_work_when_admitted_report_is_lost(scripted, monkeypatch):
    def fail(owner, expected, **options):
        with options["timings"].measure("prediction"):
            scripted.clock[0] += 2
            raise RuntimeError("scripted predictor interrupted")

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", fail)
    observer = live.ProbeRollouts(0, wide_method="bp")
    with pytest.raises(RuntimeError, match="scripted predictor interrupted"):
        observer.observe(scripted.learner, "after_probe", lambda: None)
    row = observer.records[0]
    assert row["report"] is None and row["evaluator_admitted"] is True
    phase = row["phase_timings"]["phases"]["prediction"]
    assert phase["failed"] == phase["attempted"] == 1
    assert phase["returned"] == 0 and phase["seconds"] == 2
    assert row["budget_binding_preserved"] is True
    work = dict(failure_work_may_be_unobserved=False)
    live.add_work(work, [dict(probes=[dict(probe_rollouts=observer.records)])])
    assert work["live_prediction_forwards_completed"] is None
    assert work["failure_work_may_be_unobserved"] is True


def test_should_keep_diagnostic_failure_separate_from_original_budget_expiry(scripted, monkeypatch):
    evaluate = live.evaluate_wide_neural_probe_checkpoint

    def expire(*args, **options):
        result = evaluate(*args, **options)
        scripted.budget_clock[0] = 51.0
        return result

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", expire)
    observer = live.ProbeRollouts(0, wide_method="bp")
    with pytest.raises(RuntimeError, match="max_wall_seconds"):
        observer.observe(scripted.learner, "after_probe", lambda: None)
    row = observer.records[0]
    assert row["timing_complete"] is False
    assert row["budget_binding_preserved"] is True
    assert row["phase_timings"]["status"] == "complete"
    assert row["report"]["work"]["live_prediction_calls_returned"] == 4
    assert row["charged_updates"] == row["replay_examples"] == 0


def test_should_mark_broken_diagnostic_clock_unknown_without_renewing_budget(scripted, monkeypatch):
    monkeypatch.setattr(timing, "perf_counter", lambda: float("nan"))
    observer = live.ProbeRollouts(0, wide_method="bp")
    observer.observe(scripted.learner, "after_probe", lambda: None)
    row = observer.records[0]
    assert row["status"] == "complete" and row["budget_binding_preserved"] is True
    assert row["phase_timings"]["status"] == "incomplete"
    for phase in row["phase_timings"]["phases"].values():
        assert phase["seconds"] is None and phase["known_seconds"] == 0
    assert scripted.calls[0]["budget"].budget.max_wall_seconds == 40
