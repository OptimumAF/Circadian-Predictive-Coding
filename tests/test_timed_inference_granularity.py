"""Whole-iteration qualification with synthetic clocks and four real calls.

Binary durations separate complete interval accounting from floating point
boundary noise. These engineering controls do not run the timing study.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

import src.app.timed_coupled_inference as timed
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from test_timed_coupled_inference import Clock, batches, fake_model, make_budget, run

UNIT = 1 / 1024
WINDOW = 32 * UNIT
CEILING = 2 * UNIT


def setup(monkeypatch):
    kind, initial, clock, trace = fake_model(monkeypatch)
    kind.duration = UNIT
    kind.prediction = np.full((2, 1), 0.5)
    return kind, initial, clock, trace


def case(initial, **options):
    return run(
        initial,
        training_seconds=options.pop("training_seconds", WINDOW),
        max_iteration_seconds=options.pop("max_iteration_seconds", CEILING),
        max_updates=options.pop("max_updates", 64),
        budget=options.pop("budget", make_budget(64)),
        **options,
    )


def assert_unscored(result):
    assert result["status"] == "failed" and result["reason"]
    assert result["initial_scores"] is result["final_scores"] is None


def test_should_accept_inclusive_complete_intervals_and_preserve_old_default(monkeypatch):
    kind, initial, clock, _ = setup(monkeypatch)
    kind.duration = CEILING
    result = case(initial, clock=clock)
    assert result["status"] == "complete"
    assert result["committed_updates"] == 16
    assert result["observed_max_iteration_seconds"] == result["max_iteration_seconds"] == CEILING
    assert result["training_window_seconds"] == WINDOW
    kind.total_train_calls = kind.total_predict_calls = 0
    clock.value = 0
    old = run(initial, clock=clock, training_seconds=WINDOW, max_updates=64, budget=make_budget(64))
    assert old["status"] == "complete"
    assert old.get("max_iteration_seconds") is None


@pytest.mark.parametrize("position", (0, 1, 4))
def test_should_refuse_any_excess_interval_instead_of_only_the_last(monkeypatch, position):
    kind, initial, clock, trace = setup(monkeypatch)
    native = kind.train_epoch

    def variable_train(self, *args, **kwargs):
        kind.duration = 3 * UNIT if kind.total_train_calls == position else UNIT
        return native(self, *args, **kwargs)

    monkeypatch.setattr(kind, "train_epoch", variable_train)
    result = case(initial, clock=clock)
    assert_unscored(result)
    assert result["committed_updates"] == position + 1 == len(result["records"])
    assert result["observed_max_iteration_seconds"] == 3 * UNIT
    assert all(event[0] == "train" for event in trace)
    assert initial.saved.parameters.output_bias[0, 0] == 0


def test_should_count_the_prefix_between_window_start_and_first_loop_start(monkeypatch):
    kind, initial, clock, _ = setup(monkeypatch)
    kind.duration = 0
    reads = 0

    def prefix_clock():
        nonlocal reads
        reads += 1
        # Overall-start, window-start, then the first loop-start. No native work
        # accounts for this initial prefix, which must still be qualified.
        if reads == 3:
            clock.value += 3 * UNIT
        return clock()

    result = case(initial, clock=prefix_clock)
    assert_unscored(result)
    assert kind.total_train_calls <= 1
    assert result["observed_max_iteration_seconds"] >= 3 * UNIT
    assert result["train_call_seconds"] == 0


def test_should_include_budget_and_record_bookkeeping_in_complete_intervals(monkeypatch):
    kind, initial, clock, _ = setup(monkeypatch)
    session = make_budget(64)
    before, record = session.before_update, session.record_update

    def before_update():
        before()
        clock.value += UNIT

    def record_update():
        record()
        clock.value += UNIT

    monkeypatch.setattr(session, "before_update", before_update)
    monkeypatch.setattr(session, "record_update", record_update)
    result = case(initial, clock=clock, budget=session)
    assert_unscored(result)
    assert result["committed_updates"] == 1
    assert result["train_call_seconds"] == UNIT
    assert result["observed_max_iteration_seconds"] == 3 * UNIT


def test_should_include_the_terminal_window_end_in_the_last_complete_interval(monkeypatch):
    kind, initial, clock, _ = setup(monkeypatch)
    kind.duration = CEILING
    terminal_reads = 0

    def terminal_clock():
        nonlocal terminal_reads
        if kind.total_train_calls == 16:
            terminal_reads += 1
            # Native-call end, iteration end, terminal deadline check, enclosing
            # window end. Charge overhead only to that final enclosing read.
            if terminal_reads == 4:
                clock.value += UNIT
        return clock()

    result = case(initial, clock=terminal_clock)
    assert_unscored(result)
    assert result["committed_updates"] == len(result["records"]) == 16
    assert result["observed_max_iteration_seconds"] == 3 * UNIT
    assert result["training_window_seconds"] == WINDOW + UNIT
    assert kind.total_predict_calls == 0


@pytest.mark.parametrize("commits_before_fault", (0, 1))
def test_should_preserve_known_intervals_without_fabricating_unknown_failed_timing(
    monkeypatch, commits_before_fault
):
    kind, initial, clock, _ = setup(monkeypatch)
    native = kind.train_epoch

    def fail_clock(self, *args, **kwargs):
        if kind.total_train_calls == commits_before_fault:
            kind.fail_clock_after_train = True
        return native(self, *args, **kwargs)

    monkeypatch.setattr(kind, "train_epoch", fail_clock)
    result = case(initial, clock=clock)
    assert_unscored(result)
    assert result["committed_updates"] == commits_before_fault + 1
    expected = None if commits_before_fault == 0 else UNIT
    assert result["observed_max_iteration_seconds"] == expected
    assert initial.saved.parameters.output_bias[0, 0] == 0


@pytest.mark.parametrize("failure", ("native", "cap", "scoring", "last_score_budget"))
def test_should_keep_known_work_and_clear_both_scores_on_late_failure(monkeypatch, failure):
    kind, initial, clock, _ = setup(monkeypatch)
    kind.duration = CEILING
    options: dict[str, Any] = {"clock": clock}
    expected = 16
    if failure == "native":
        kind.fail_train = 3
        expected = 2
    elif failure == "cap":
        options["max_updates"] = 2
        expected = 2
    elif failure == "scoring":
        kind.fail_predict = 2
    else:
        wall = [0.0]
        kind.after_predict = lambda: (
            wall.__setitem__(0, 2.0) if kind.total_predict_calls == 2 else None
        )
        options["budget"] = make_budget(64, clock=lambda: wall[0], wall=1)
    result = case(initial, **options)
    assert_unscored(result)
    assert result["committed_updates"] == len(result["records"]) == expected
    assert result["observed_max_iteration_seconds"] >= CEILING
    assert initial.saved.parameters.output_bias[0, 0] == 0


def test_should_copy_source_policy_and_data_before_external_budget_callbacks(monkeypatch):
    kind, initial, clock, trace = setup(monkeypatch)
    kind.duration = CEILING
    train, final = batches()
    saved = deepcopy((train, final))
    policy = CoupledInferencePolicy()
    session = make_budget(64)

    def mutate():
        initial.saved.parameters.output_bias[:] = 99
        for p in (train, final):
            p.features[:] = p.targets[:] = 99
        object.__setattr__(policy, "gradient_tolerance", 99.0)

    monkeypatch.setattr(session, "before_final", mutate)
    result = case(
        initial, training=train, heldout=final, clock=clock, budget=session, policy=policy
    )
    assert result["status"] == "complete"
    assert result["final_scores"]["bce"] == pytest.approx(np.log(2))
    for event in trace:
        np.testing.assert_array_equal(event[1], saved[0 if event[0] == "train" else 1].features)
        if event[0] == "train":
            np.testing.assert_array_equal(event[2], saved[0].targets)
            assert event[3].gradient_tolerance == 1e-6


@pytest.mark.parametrize(
    "limit", (True, np.float64(UNIT), 0, -UNIT, float("nan"), float("inf"), 0.1 * WINDOW + UNIT)
)
def test_should_refuse_invalid_ceiling_before_copy_clock_or_external_callback(monkeypatch, limit):
    kind, initial, clock, _ = setup(monkeypatch)
    touched = []
    session = make_budget(64)
    monkeypatch.setattr(session, "before_final", lambda: touched.append("budget"))
    monkeypatch.setattr(initial, "snapshot_state", lambda: touched.append("snapshot"))
    monkeypatch.setattr(timed, "_validated_batches", lambda *args: touched.append("copy"))
    with pytest.raises(ValueError):
        case(
            initial,
            clock=lambda: touched.append("clock"),
            budget=session,
            max_iteration_seconds=limit,
        )
    assert touched == [] and kind.total_train_calls == 0


def test_should_admit_the_exact_ten_percent_ceiling_before_a_separate_zero_budget_failure(
    monkeypatch,
):
    _, initial, clock, _ = setup(monkeypatch)
    result = case(initial, clock=clock, max_iteration_seconds=0.1 * WINDOW, budget=make_budget(0))
    assert_unscored(result)
    assert result["max_iteration_seconds"] == 0.1 * WINDOW
    assert result["observed_max_iteration_seconds"] is None


def test_should_record_four_actual_fixed_and_adaptive_calls_before_coarseness_refusal(monkeypatch):
    native = CoupledPredictiveCodingNetwork.train_epoch
    clock = Clock()
    calls = 0

    def controlled_train(self, *args, **kwargs):
        nonlocal calls
        result = native(self, *args, **kwargs)
        calls += 1
        clock.value += UNIT if calls % 2 else 3 * UNIT
        return result

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "train_epoch", controlled_train)
    for policy in (None, CoupledInferencePolicy()):
        source = CoupledPredictiveCodingNetwork(2, (2,), seed=23)
        saved = deepcopy(source.snapshot_state())
        clock.value = 0
        result = case(source, clock=clock, policy=policy)
        assert_unscored(result)
        assert result["committed_updates"] == len(result["records"]) == 2
        assert result["observed_max_iteration_seconds"] == 3 * UNIT
        np.testing.assert_equal(asdict(source.snapshot_state()), asdict(saved))
    assert calls == 4
