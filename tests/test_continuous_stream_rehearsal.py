"""Continuous fixed-bank replay authority with counter-only fake native ports.

No real learner, initializer, fitting, gradients, RNG or native simulation. Small
arrays are payloads, and fake updates only increment immutable scalar counters.
Per module: 31 counter-only setups/107 fake port objects, 45 fake train calls
exposing 57 payload rows, and one constant fake prediction row.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from typing import Any

import numpy as np
import pytest

from src.app import continuous_stream_rehearsal as replay
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic


DEFINITION = "fake_continuous_replay_v1"


def episode(name="warm", *, role="train", rows=2):
    declared = ContinuousActionSpec(2, ("m",), (-1.0,), (1.0,))
    return ContinuousEpisode(
        name,
        name + "-context",
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                (float(index), 0.0),
                (0.25,),
                0.0,
                (float(index + 1), 0.0),
                index == rows - 1,
                False,
            )
            for index in range(rows)
        ),
    )


@dataclass(frozen=True)
class FakeState:
    policy: str = "unchanged TRAIN policy"
    value: int = 0
    completed_updates: int = 0
    training_examples: int = 0


class CounterPort:
    def __init__(self, control=None, state=None):
        self.control: dict[str, Any] = dict(trace=[], forks=0) if control is None else control
        self.state = FakeState() if state is None else deepcopy(state)

    def fork(self):
        self.control["forks"] += 1
        return CounterPort(self.control, self.state)

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        rows = len(features)
        self.control["trace"].append(
            dict(
                rows=rows,
                features=features.copy(),
                targets=targets.copy(),
                charged=self.control["replay_budget"].replay_examples_completed,
            )
        )
        self.state = replace(
            self.state,
            value=self.state.value + 1,
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + rows,
        )
        if rows == 2 and self.control.get("fail_replay"):
            raise RuntimeError("paid fake native replay failed")
        return TrainingDiagnostic(DEFINITION, float(self.state.value))

    def predict(self, features):
        return np.zeros((len(features), 1), dtype=np.float64)


def setup(*, every=1, cap=4, total=2):
    seconds = [0.0]
    timer = lambda: seconds[0]
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=total), timer)
    replay_budget = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=cap), timer)
    source = CounterPort()
    source.control["replay_budget"] = replay_budget
    plan = replay.ContinuousReplayPlan(every, cap)
    helper = replay.ContinuousStreamRehearsal(
        source,
        wake,
        plan,
        replay_budget=replay_budget,
        total_updates=total,
        fixed_training_episodes=(episode(),),
        states_equal=lambda a, b: a == b,
        expected_definition=DEFINITION,
    )
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor",
        candidate_version="candidate",
        clock=clock,
        budget=wake,
        max_experiences=total,
        max_consolidations=total // every,
    )
    assert helper.bind(runtime) == total // every
    return source, runtime, helper, wake, replay_budget, clock, seconds


def wake_once(runtime, clock, index):
    observed, arrived = 2 * index, 2 * index + 1
    clock.advance_to(observed)
    features, targets = np.array([[10.0 + index, 0.0]]), np.array([[0.5]])
    runtime.record_experience(
        Experience(
            f"wake-{index}",
            "wake-episode",
            observed,
            "actor",
            features,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    assert runtime.train_ready(max_updates=1) == ()
    clock.advance_to(arrived)
    runtime.record_label(
        LabelArrival(
            f"wake-label-{index}", f"wake-{index}", "wake-episode", arrived, "actor", targets
        )
    )
    receipts = runtime.train_ready(max_updates=1)
    assert len(receipts) == 1
    return receipts


@pytest.mark.parametrize(
    "field,value",
    [
        ("every_updates", 0),
        ("every_updates", 385),
        ("every_updates", True),
        ("every_updates", 1.0),
        ("max_replay_examples", 0),
        ("max_replay_examples", 65537),
        ("max_replay_examples", True),
        ("max_retained_rows", 0),
        ("max_retained_rows", 129),
        ("max_retained_bytes", 0),
        ("max_retained_bytes", 1048577),
    ],
)
def test_should_refuse_invalid_exact_plan_caps(field, value):
    options = dict(every_updates=1, max_replay_examples=4)
    options[field] = value
    with pytest.raises(ValueError):
        replay.ContinuousReplayPlan(**options)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "inner_guard", "final_test"])
def test_should_never_admit_reserved_or_heldout_roles_into_fixed_bank(role):
    with pytest.raises(ValueError):
        replay.prepare_replay_bank(replay.ContinuousReplayPlan(1, 4), (episode(role=role),))


@pytest.mark.parametrize(
    "damage", ["empty", "list", "duplicates", "different_spec", "rows", "bytes"]
)
def test_should_refuse_unbounded_or_mixed_bank_before_any_training(damage):
    plan = replay.ContinuousReplayPlan(1, 4)
    bank: Any = (episode(),)
    if damage == "empty":
        bank = ()
    elif damage == "list":
        bank = [episode()]
    elif damage == "duplicates":
        bank = (episode(), episode())
    elif damage == "different_spec":
        other = episode("other")
        bank = (episode(), replace(other, spec=replace(other.spec, action_units=("other",))))
    elif damage == "rows":
        plan = replace(plan, max_retained_rows=1)
    else:
        plan = replace(plan, max_retained_bytes=95)
    with pytest.raises(ValueError):
        replay.prepare_replay_bank(plan, bank)


def test_should_own_complete_bank_and_charge_declared_numeric_payload_only():
    original = episode()
    value = replay.ContinuousReplayPlan(1, 4)
    owned, bank, rows, logical_bytes = replay.prepare_replay_bank(value, (original,))
    assert owned == value and owned is not value
    assert bank[0] is not original and rows == 2
    assert logical_bytes == 2 * (2 * 2 + 1 + 1) * 8 == 96
    object.__setattr__(original.transitions[0], "action", (-1.0,))
    assert bank[0].transitions[0].action == (0.25,)


def test_should_consolidate_entire_fixed_bank_with_separate_permanent_exposure_budget():
    source, runtime, helper, wake, paid, clock, _seconds = setup()
    initial = source.snapshot_state()
    for index in range(2):
        wake_once(runtime, clock, index)
        helper.after_update(runtime, index + 1)
    report = helper.report
    assert report["planned_events"] == 2 and report["bank_rows"] == 2
    assert report["retained_numeric_bytes"] == 96
    assert (
        report["attempted_updates"]
        == report["returned_updates"]
        == report["committed_updates"]
        == 2
    )
    assert report["charged_exposures"] == 4 and paid.replay_examples_completed == 4
    assert paid.updates_completed == 0 and wake.replay_examples_completed == 0
    assert wake.updates_completed == len(runtime.applied_updates) == 2
    assert len(runtime.applied_consolidations) == 2
    state = runtime.candidate_snapshot().state
    assert state.completed_updates == state.value == 4 and state.training_examples == 6
    assert runtime.actor.snapshot_state().state == source.snapshot_state() == initial
    trace = source.control["trace"]
    assert [row["rows"] for row in trace] == [1, 2, 1, 2]
    assert [row["charged"] for row in trace] == [0, 2, 2, 4]
    np.testing.assert_array_equal(trace[1]["features"], [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(trace[1]["targets"], [[0.25], [0.25]])
    assert [item.event_id for item in runtime.applied_consolidations] == [
        "continuous-replay-1",
        "continuous-replay-2",
    ]
    report["events"].clear()
    assert len(helper.report["events"]) == 2


def test_should_wait_for_period_and_refuse_duplicate_observed_wake_without_more_charge():
    source, runtime, helper, wake, paid, clock, _seconds = setup(every=2)
    wake_once(runtime, clock, 0)
    helper.after_update(runtime, 1)
    assert helper.report["attempted_updates"] == 0 and paid.replay_examples_completed == 0
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    assert paid.replay_examples_completed == 0 and wake.updates_completed == 1
    assert len(source.control["trace"]) == 1


def test_should_refuse_second_bind_without_resetting_original_authority():
    _source, runtime, helper, wake, paid, _clock, _seconds = setup()
    with pytest.raises((ValueError, RuntimeError)):
        helper.bind(runtime)
    assert wake.updates_completed == paid.replay_examples_completed == 0


def test_should_preserve_paid_native_failure_without_publishing_or_retrying():
    source, runtime, helper, wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    before = runtime.candidate_snapshot().state
    source.control["fail_replay"] = True
    with pytest.raises(RuntimeError, match="paid fake native replay failed"):
        helper.after_update(runtime, 1)
    assert paid.replay_examples_completed == 2 and wake.updates_completed == 1
    assert runtime.candidate_snapshot().state == before
    assert runtime.actor.snapshot_state().state == source.snapshot_state()
    assert runtime.applied_consolidations == ()
    report = helper.report
    assert (
        report["attempted_updates"] == 1
        and report["returned_updates"] == report["committed_updates"] == 0
    )
    assert report["charged_exposures"] == 2 and report["stopped"] is True
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    assert paid.replay_examples_completed == 2 and len(source.control["trace"]) == 2
    np.testing.assert_array_equal(runtime.actor.predict(np.array([[0.0, 0.0]])).prediction, [[0.0]])


def test_should_preflight_remaining_whole_bank_quota_without_overcharging():
    source, runtime, helper, wake, paid, clock, _seconds = setup(cap=3)
    wake_once(runtime, clock, 0)
    helper.after_update(runtime, 1)
    wake_once(runtime, clock, 1)
    before = runtime.candidate_snapshot().state
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 2)
    assert paid.replay_examples_completed == 2 and wake.updates_completed == 2
    assert runtime.candidate_snapshot().state == before and len(runtime.applied_consolidations) == 1
    assert [row["rows"] for row in source.control["trace"]] == [1, 2, 1]
    assert helper.report["stopped"] is True


@pytest.mark.parametrize(
    "field", ["limits", "timer", "start", "wake_float", "replay_reset", "last_clock", "session"]
)
def test_should_refuse_replaced_or_corrupted_birth_bound_budget_before_replay_charge(field):
    source, runtime, helper, wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    if field == "limits":
        paid.budget = replace(paid.budget)
    elif field == "timer":
        paid.clock = lambda: 0.0
    elif field == "start":
        paid.started_at = 0.01
    elif field == "wake_float":
        wake.updates_completed = 1.0
    elif field == "replay_reset":
        paid.replay_examples_completed = -1
    elif field == "last_clock":
        paid.last_clock = float("nan")
    else:
        runtime._budget = ToyBudgetSession(replace(wake.budget), wake.clock)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    assert [row["rows"] for row in source.control["trace"]] == [1]
    assert helper.report["returned_updates"] == helper.report["committed_updates"] == 0


def test_should_keep_observed_postcommit_receipt_and_charge_even_when_runtime_throws(monkeypatch):
    _source, runtime, helper, wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    consolidate = runtime.consolidate

    def fail_after_commit(*args, **options):
        consolidate(*args, **options)
        raise RuntimeError("postcommit observation failed")

    monkeypatch.setattr(runtime, "consolidate", fail_after_commit)
    with pytest.raises(RuntimeError, match="postcommit"):
        helper.after_update(runtime, 1)
    assert paid.replay_examples_completed == 2 and wake.updates_completed == 1
    assert runtime.candidate_snapshot().state.completed_updates == 2
    assert len(runtime.applied_consolidations) == 1
    assert helper.report["committed_updates"] == 1
    assert helper.report["charged_exposures"] == 2


@pytest.mark.parametrize("damage", ["alias", "restore_argument", "source"])
def test_should_refuse_invalid_temporary_fork_without_publishing_or_leaking_source(
    monkeypatch, damage
):
    source, runtime, helper, _wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    before = runtime.candidate_snapshot().state
    if damage == "alias":
        monkeypatch.setattr(source, "fork", lambda: source)
    else:
        original = source.fork

        def fork():
            child = original()
            restore = child.restore_state

            def mutate(state):
                if damage == "restore_argument":
                    object.__setattr__(state, "value", 999)
                else:
                    source.state = replace(source.state, value=999)
                restore(state)

            child.restore_state = mutate
            return child

        monkeypatch.setattr(source, "fork", fork)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    assert runtime.candidate_snapshot().state == before and runtime.applied_consolidations == ()
    assert helper.report["committed_updates"] == 0
    assert paid.replay_examples_completed == 0


def test_should_refuse_replaced_preceding_consolidation_receipt_before_next_replay():
    source, runtime, helper, wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    helper.after_update(runtime, 1)
    runtime._consolidations[0] = replace(
        runtime._consolidations[0], event_id="replaced-prior-success"
    )
    wake_once(runtime, clock, 1)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 2)
    assert wake.updates_completed == 2 and paid.replay_examples_completed == 2
    assert [row["rows"] for row in source.control["trace"]] == [1, 2, 1]
    assert helper.report["committed_updates"] == 1


@pytest.mark.parametrize("target", ["bank", "plan"])
def test_should_refuse_changed_owned_bank_or_equal_looking_plan_before_charge(target):
    source, runtime, helper, _wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    if target == "bank":
        object.__setattr__(helper._bank[0].transitions[0], "action", (-0.5,))
    else:
        helper._plan = replace(helper._plan)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    assert paid.replay_examples_completed == 0
    assert [row["rows"] for row in source.control["trace"]] == [1]


def test_should_retain_invalid_returned_diagnostic_after_paid_training_without_publish(monkeypatch):
    source, runtime, helper, wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    before, original = runtime.candidate_snapshot().state, source.fork

    def fork():
        child = original()
        train = child.train_batch

        def invalid(features, targets):
            train(features, targets)
            return TrainingDiagnostic("foreign native objective", 0.5)

        child.train_batch = invalid
        return child

    monkeypatch.setattr(source, "fork", fork)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    report = helper.report
    assert wake.updates_completed == 1 and paid.replay_examples_completed == 2
    assert report["returned_updates"] == 1 and report["committed_updates"] == 0
    assert report["events"][0]["diagnostics"] == [
        dict(definition="foreign native objective", value=0.5)
    ]
    assert runtime.candidate_snapshot().state == before


@pytest.mark.parametrize(
    "damage",
    [
        "inbox_budget",
        "inbox_clock",
        "logical_clock",
        "logical_advance",
        "candidate",
        "attempts_reset",
        "attempts_replaced",
    ],
)
def test_should_revalidate_preceding_authority_and_state_before_another_wake(damage):
    attempts = damage.startswith("attempts_")
    source, runtime, helper, wake, paid, clock, _seconds = setup(every=1 if attempts else 2)
    helper.before_update(runtime, 0)
    wake_once(runtime, clock, 0)
    helper.after_update(runtime, 1)
    if damage == "inbox_budget":
        runtime._inbox._budget = ToyBudgetSession(replace(wake.budget), wake.clock)
    elif damage == "inbox_clock":
        runtime._inbox._clock = LogicalClock()
    elif damage == "logical_clock":
        clock._time = 0
    elif damage == "logical_advance":
        clock.advance_to(2)
    elif damage == "candidate":
        runtime._candidate.state = replace(runtime._candidate.state, value=999)
    elif damage == "attempts_reset":
        runtime._attempted_ids.clear()
    else:
        runtime._attempted_ids = set(runtime._attempted_ids)
    with pytest.raises((ValueError, RuntimeError)):
        helper.before_update(runtime, 1)
    assert wake.updates_completed == 1 and paid.replay_examples_completed == (2 if attempts else 0)
    assert [row["rows"] for row in source.control["trace"]] == ([1, 2] if attempts else [1])


def test_should_refuse_coherent_candidate_change_inside_held_consolidation_before_charge(
    monkeypatch,
):
    source, runtime, helper, _wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    original_consolidate, original_snapshot, active = (
        runtime.consolidate,
        source.snapshot_state,
        [False],
    )

    def snapshot():
        returned = original_snapshot()
        if active[0]:
            runtime._candidate.state = replace(runtime._candidate.state, value=999)
        return returned

    def consolidate(event_id, operation):
        def wrapped(state):
            active[0] = True
            try:
                return operation(state)
            finally:
                active[0] = False

        return original_consolidate(event_id, wrapped)

    monkeypatch.setattr(source, "snapshot_state", snapshot)
    monkeypatch.setattr(runtime, "consolidate", consolidate)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    assert paid.replay_examples_completed == 0 and runtime.applied_consolidations == ()
    assert [row["rows"] for row in source.control["trace"]] == [1]
    assert helper.report["stopped"] is True


def test_should_detect_lying_publication_with_correct_counters_and_preserve_charge(monkeypatch):
    _source, runtime, helper, wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    restore = runtime._candidate.restore_state

    def lie(state):
        restore(replace(state, value=999))

    monkeypatch.setattr(runtime._candidate, "restore_state", lie)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    assert wake.updates_completed == 1 and paid.replay_examples_completed == 2
    assert runtime.candidate_snapshot().state.completed_updates == 2
    assert runtime.candidate_snapshot().state.value == 999
    assert len(runtime.applied_consolidations) == helper.report["committed_updates"] == 1
    assert helper.report["charged_exposures"] == 2 and helper.report["stopped"] is True


def test_should_not_accept_corrupt_postcommit_receipt_as_valid_publication(monkeypatch):
    _source, runtime, helper, wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    original = runtime.consolidate

    def corrupt(*args, **options):
        receipt = original(*args, **options)
        replaced = replace(receipt, diagnostic=TrainingDiagnostic("foreign receipt objective", 0.0))
        runtime._consolidations[0] = replaced
        return replaced

    monkeypatch.setattr(runtime, "consolidate", corrupt)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    report = helper.report
    assert wake.updates_completed == 1 and paid.replay_examples_completed == 2
    assert runtime.candidate_snapshot().state.completed_updates == 2
    assert report["returned_updates"] == 1 and report["committed_updates"] == 0
    assert report["events"][0].get("receipt_observation_error")
    assert report["charged_exposures"] == 2


def test_should_retain_nonfinite_returned_diagnostic_as_json_safe_paid_failure(monkeypatch):
    source, runtime, helper, _wake, paid, clock, _seconds = setup()
    wake_once(runtime, clock, 0)
    original = source.fork

    def fork():
        child = original()
        train = child.train_batch

        def bad(features, targets):
            diagnostic = train(features, targets)
            object.__setattr__(diagnostic, "value", float("nan"))
            return diagnostic

        child.train_batch = bad
        return child

    monkeypatch.setattr(source, "fork", fork)
    with pytest.raises((ValueError, RuntimeError)):
        helper.after_update(runtime, 1)
    report = helper.report
    assert paid.replay_examples_completed == 2
    assert report["returned_updates"] == 1 and report["committed_updates"] == 0
    assert report["events"][0]["diagnostics"] == [dict(definition=DEFINITION, value=None)]
    assert report["events"][0].get("returned_diagnostic_value_error")
    json.dumps(report, allow_nan=False)
