"""Feature arrival ordering, ownership and paid failure with fake native learners."""

from dataclasses import fields, replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.app import coupled_attribution as app
from src.app.toy_execution_budget import (
    ToyBudgetSession,
    ToyExecutionBudget,
    ToyExecutionStopped,
)
from src.core.actor_ports import CandidateState
from tests.test_coupled_attribution import SpyLearner, batches, budget


def training_batches():
    first, second = batches()[0]
    third = replace(first, row_ids=("train-c",), features=np.array([[0.5, 0.6]]))
    return first, second, third


def stream(source=None, **options):
    return app.AttributionStream(
        SpyLearner() if source is None else source,
        training_batches(),
        (),
        checkpoint_updates=(0, 1, 3),
        label_delay=options.pop("label_delay", 0),
        budget=options.pop("budget", budget(3)),
        training_only=True,
        **options,
    )


def observation_report(owner) -> dict[str, Any]:
    result = owner.prearrival_report
    assert result is not None
    return result


@pytest.mark.parametrize("delay", [0, 2])
def test_should_observe_once_before_same_tick_record_and_update_with_current_shadow(
    monkeypatch, delay
):
    events = []
    original_record = app._record_tick

    def record(runtime, training, tick, label_delay):
        events.append(("record", tick))
        original_record(runtime, training, tick, label_delay)

    def observe(observation, candidate):
        events.append(("observe", observation.tick))
        expected_updates = max(0, observation.tick - delay)
        assert type(observation) is app.PrearrivalObservation
        assert [field.name for field in fields(observation)] == ["tick", "row_ids", "features"]
        assert type(candidate) is CandidateState
        assert candidate.actor_version == "a0" and candidate.learner_version == "c0"
        assert candidate.wake_updates == expected_updates
        assert candidate.consolidation_attempts == candidate.consolidations_completed == 0
        np.testing.assert_array_equal(candidate.state, [expected_updates])
        batch = training_batches()[observation.tick]
        assert observation.row_ids == batch.row_ids
        np.testing.assert_array_equal(observation.features, batch.features)

    monkeypatch.setattr(app, "_record_tick", record)
    owner = stream(label_delay=delay, observe_features=observe)
    owner.run_until(3)
    result = owner.finish()
    assert events == [
        event
        for tick in range(3 + delay)
        for event in ([("observe", tick), ("record", tick)] if tick < 3 else [("record", tick)])
    ]
    assert result["prearrival_observation"] == {
        "calls_attempted": 3,
        "calls_completed": 3,
        "feature_rows_attempted": 3,
        "feature_rows_completed": 3,
        "prediction_subject": "current_shadow_candidate",
        "label_order": "before_same_tick_label_arrival",
    }
    assert [receipt["arrived_at"] for receipt in result["applied_updates"]] == [
        delay,
        delay + 1,
        delay + 2,
    ]


def test_should_discard_detached_mutations_without_changing_sources_actor_or_training():
    source = SpyLearner()
    training = training_batches()
    original = pickle.dumps(training)

    def observe(observation, candidate):
        observation.features.fill(999)
        candidate.state.fill(999)

    owner = app.AttributionStream(
        source,
        training,
        (),
        checkpoint_updates=(0, 3),
        label_delay=0,
        budget=budget(3),
        training_only=True,
        observe_features=observe,
    )
    actor = owner._runtime.actor.snapshot_state()
    owner.run_until(3)
    owner.finish()
    assert pickle.dumps(training) == original
    assert source.state[0] == 0
    assert app._same_value(actor, owner._runtime.actor.snapshot_state())
    for event, batch in zip(
        [event for event in source.trace if event[0] == "train"], training, strict=True
    ):
        np.testing.assert_array_equal(event[2], batch.features)
        np.testing.assert_array_equal(event[3], batch.targets)


def test_should_preserve_paid_callback_failure_and_refuse_further_work():
    def observe(observation, candidate):
        raise RuntimeError("prediction failed after paid work")

    session = budget(3)
    owner = stream(budget=session, observe_features=observe)
    with pytest.raises(RuntimeError, match="prediction failed"):
        owner.run_until(3)
    report = observation_report(owner)
    assert report["calls_attempted"] == report["feature_rows_attempted"] == 1
    assert report["calls_completed"] == report["feature_rows_completed"] == 0
    report["calls_attempted"] = 999
    assert observation_report(owner)["calls_attempted"] == 1
    assert session.updates_completed == owner.completed_updates == 0
    assert owner._runtime.applied_updates == ()
    with pytest.raises(ValueError, match="failed or finished"):
        owner.run_until(1)


@pytest.mark.parametrize("subject", ["candidate", "source", "actor", "features"])
def test_should_refuse_actual_state_corruption_without_repair_or_training(subject):
    source = SpyLearner()

    def observe(observation, candidate):
        value = {
            "candidate": owner._runtime._candidate.state,
            "source": source.state,
            "actor": owner._runtime.actor._learner.state,
            "features": owner._training[observation.tick].features,
        }[subject]
        value.fill(777)

    owner = stream(source, observe_features=observe)
    with pytest.raises(ValueError, match="changed its source, actor or shadow state"):
        owner.run_until(3)
    assert owner.completed_updates == 0
    assert observation_report(owner)["calls_completed"] == 1
    value = {
        "candidate": owner._runtime._candidate.state,
        "source": source.state,
        "actor": owner._runtime.actor._learner.state,
        "features": owner._training[0].features,
    }[subject]
    assert np.all(value == 777)
    assert not any(event[0] == "train" for event in source.trace)


def test_should_refuse_nested_runtime_training_during_feature_callback():
    def observe(observation, candidate):
        owner._runtime.train_ready(max_updates=1)

    owner = stream(observe_features=observe)
    with pytest.raises(ValueError, match="candidate is busy"):
        owner.run_until(3)
    assert observation_report(owner)["calls_attempted"] == 1
    assert observation_report(owner)["calls_completed"] == 0
    assert owner.completed_updates == 0


def test_should_refuse_replaced_candidate_even_when_its_complete_state_is_equal():
    def observe(observation, candidate):
        owner._runtime._candidate = owner._learner.fork()

    owner = stream(observe_features=observe)
    with pytest.raises(ValueError, match="changed its source, actor or shadow state"):
        owner.run_until(3)
    assert observation_report(owner)["calls_completed"] == 1
    assert owner.completed_updates == 0


def test_should_keep_original_budget_clock_including_callback_work():
    now = [0.0]
    session = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=3, max_wall_seconds=1), lambda: now[0]
    )

    def observe(observation, candidate):
        now[0] = 2.0

    owner = stream(budget=session, observe_features=observe)
    with pytest.raises(ToyExecutionStopped) as error:
        owner.run_until(3)
    assert error.value.stop.reason == "max_wall_seconds"
    assert observation_report(owner)["calls_completed"] == 1
    assert session.started_at == 0.0 and session.last_clock == 2.0
    assert owner.completed_updates == session.updates_completed == 0


@pytest.mark.parametrize("delay", [0, 2])
def test_should_continue_original_counts_without_duplicate_features_after_pause(delay):
    ticks = []
    owner = stream(
        label_delay=delay,
        observe_features=lambda observation, _: ticks.append(observation.tick),
    )
    owner.run_until(1)
    before = owner.prearrival_report
    token = owner.pause()
    assert token.snapshot_state()["prearrival_observation"] == before
    owner.resume(token)
    owner.run_until(3)
    result = owner.finish()
    assert ticks == [0, 1, 2]
    assert result["prearrival_observation"]["calls_completed"] == 3


def test_should_refuse_changed_pause_observation_without_more_work():
    owner = stream(observe_features=lambda *_: None)
    owner.run_until(1)
    token = owner.pause()
    token._snapshot["prearrival_observation"]["calls_completed"] = 0
    with pytest.raises(ValueError, match="corrupted or replaced"):
        owner.resume(token)
    assert owner.completed_updates == 1
    assert observation_report(owner)["calls_completed"] == 1


def test_should_refuse_replaced_feature_callback_before_more_work():
    owner = stream(observe_features=lambda *_: None)
    owner._feature_observer = lambda *_: None
    with pytest.raises(ValueError, match="original feature observer"):
        owner.run_until(3)
    assert observation_report(owner)["calls_attempted"] == 0


@pytest.mark.parametrize("authority", ["budget", "limits", "wall_clock", "event_clock", "helper"])
def test_should_refuse_authority_replacement_in_callback_before_recording_labels(authority):
    source = SpyLearner()
    session = budget(3)

    def observe(*_):
        if authority == "budget":
            owner._budget = budget(3)
        elif authority == "limits":
            session.budget = ToyExecutionBudget(max_training_updates=4)
        elif authority == "wall_clock":
            session.clock = lambda: 0.0
        elif authority == "event_clock":
            from src.core.experience import LogicalClock

            owner._clock = LogicalClock()
        else:
            owner._original_helper = object()

    owner = stream(source, budget=session, observe_features=observe)
    with pytest.raises(ValueError, match="original budget, clock and helper authority"):
        owner.run_until(3)
    assert owner.completed_updates == session.updates_completed == 0
    assert owner._runtime.applied_updates == ()
    assert not any(event[0] == "train" for event in source.trace)
    assert observation_report(owner)["calls_completed"] == 1


@pytest.mark.parametrize("invalid", [False, 1, (), "observer"])
def test_should_validate_callback_before_forking(invalid):
    source = SpyLearner()
    with pytest.raises(ValueError, match="observe_features must be callable"):
        stream(source, observe_features=invalid)
    assert source.trace == []


def test_should_preserve_default_wire_and_expose_option_through_existing_runner():
    ordinary = stream()
    assert ordinary.prearrival_report is None
    ordinary.run_until(3)
    assert "prearrival_observation" not in ordinary.finish()
    ticks = []
    result = app.run_attribution_stream(
        SpyLearner(),
        training_batches(),
        (),
        checkpoint_updates=(0, 3),
        label_delay=2,
        budget=budget(3),
        training_only=True,
        observe_features=lambda observation, _: ticks.append(observation.tick),
    )
    assert ticks == [0, 1, 2]
    assert result["prearrival_observation"]["calls_completed"] == 3
