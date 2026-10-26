"""Released complete episodes, bounded context coverage and owned stream continuation."""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import CoverageStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import EpisodeReplaySchema, episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from test_demonstration_episode import episode


SCHEMA = EpisodeReplaySchema(2, 2)
ROW_BYTES = (SCHEMA.encoded_dim + 1) * 8


class Model:
    def __init__(self, control=None, state=None):
        self.control: dict[str, Any] = {"calls": [], "forks": 0} if control is None else control
        self.state = np.array([0.0]) if state is None else state.copy()

    def fork(self):
        self.control["forks"] += 1
        return Model(self.control, self.state)

    def snapshot_state(self):
        return self.state.copy()

    def restore_state(self, state):
        self.state = state.copy()

    def train_batch(self, features, targets):
        self.control["calls"].append((features.copy(), targets.copy()))
        if self.control.get("fail_call") == len(self.control["calls"]):
            raise RuntimeError("coverage replay native failure")
        self.state += 1
        if self.control.get("mutate_inputs"):
            features.fill(999)
            targets.fill(0.5)
        return TrainingDiagnostic("fake_complete_episode_v1", float(self.state[0]))

    def predict(self, features):
        pytest.fail("training-only stream attempted final scoring")


def policy(**changes):
    values: dict[str, Any] = dict(
        every_updates=2,
        rows_per_event=2,
        retention=ReplayRetentionBudget(6, ROW_BYTES * 3),
        contexts=("A", "B", "C"),
    )
    values.update(changes)
    return CoverageStreamReplayPolicy(**values)


def batches(contexts=("B", "A", "C", "A", "B", "C")):
    output = []
    for index, context in enumerate(contexts):
        packet = episode(dim=2, steps=2, episode_id=f"train-{index}", context=context)
        features, targets = episode_arrays(packet)
        output.append(
            AttributionBatch(
                "train",
                context,
                tuple(f"{packet.episode_id}:step:{step.step}" for step in packet.transitions),
                features,
                targets,
                packet,
            )
        )
    return tuple(output)


def setup(**faults):
    source = Model()
    source.control.update(faults)
    wake = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=6, max_wall_seconds=10), lambda: 0.0
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=6, max_wall_seconds=10), lambda: 0.0
    )
    helper = StreamRehearsal(
        source,
        wake,
        policy(),
        replay_budget=replay,
        total_updates=6,
        input_dim=2,
        episode_schema=SCHEMA,
    )
    return source, wake, replay, helper


def stream(source, wake, helper, training):
    return AttributionStream(
        source,
        training,
        (),
        checkpoint_updates=(0, 2, 4, 6),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )


@pytest.mark.parametrize(
    "contexts", [[], (), ("A", "A"), tuple("ABCDEFG"), ("",), (" A",), ("x" * 33,), ("é" * 17,)]
)
def test_should_reject_nonexact_duplicate_oversized_or_invalid_declared_context_keys(contexts):
    with pytest.raises(ValueError):
        policy(contexts=contexts)


@pytest.mark.parametrize(
    "fault", ["missing_schema", "insufficient_rows", "insufficient_bytes", "distillation"]
)
def test_should_require_complete_episode_geometry_and_capacity_for_every_declared_key_before_fork(
    fault,
):
    source = Model()
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=6), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=6), lambda: 0.0)
    kwargs: dict[str, Any] = dict(episode_schema=SCHEMA)
    selected = policy()
    if fault == "missing_schema":
        kwargs = {}
    elif fault == "insufficient_rows":
        selected = policy(retention=ReplayRetentionBudget(5, ROW_BYTES * 3))
    elif fault == "insufficient_bytes":
        selected = policy(retention=ReplayRetentionBudget(6, ROW_BYTES * 3 - 1))
    else:
        kwargs["distillation"] = object()
    with pytest.raises(ValueError):
        StreamRehearsal(
            source, wake, selected, replay_budget=replay, total_updates=6, input_dim=2, **kwargs
        )
    assert source.control["forks"] == 0


def test_should_cycle_declared_eligible_keys_using_only_latest_released_complete_episodes():
    source, wake, replay, helper = setup(mutate_inputs=True)
    training = batches()
    original = pickle.dumps(training)
    run = stream(source, wake, helper, training)
    run.run_until(6)
    report = run.finish()["rehearsal"]
    assert pickle.dumps(training) == original and source.state[0] == 0
    assert wake.updates_completed == 6 and wake.replay_examples_completed == 0
    assert replay.updates_completed == 0 and replay.replay_examples_completed == 6
    assert len(source.control["calls"]) == 9
    assert run._runtime._candidate.snapshot_state()[0] == 9
    assert run._runtime.actor.snapshot_state().state[0] == 0
    events = report["events"]
    assert report["control"] == "context_coverage_v1"
    assert report["coverage_keys"] == ["A", "B", "C"] and report["coverage_cursor"] == "C"
    assert [row["coverage_keys"] for row in events] == [
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B", "C"],
    ]
    assert [row["coverage_cursor"] for row in events] == ["A", "B", "C"]
    expected = [training[index].episode for index in (1, 0, 5)]
    for row, packet in zip(events, expected, strict=True):
        assert row["selected_episodes"] == [asdict(packet)]
        assert row["charged_exposures"] == 2
        assert row["attempted_updates"] == row["returned_updates"] == row["committed_updates"] == 1
        assert row["retention"]["example_count"] <= 6
        assert row["retention"]["retained_bytes"] <= ROW_BYTES * 3
    assert len(run._runtime.applied_consolidations) == 3
    detached = helper.report
    detached["events"][0]["selected_episodes"][0]["role"] = "final_test"
    detached["coverage_keys"].clear()
    assert helper.report["events"][0]["selected_episodes"][0]["role"] == "train"
    assert helper.report["coverage_keys"] == ["A", "B", "C"]


def test_should_resume_owned_coverage_cursor_without_repeating_replay_or_renewing_budgets():
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper, batches())
    run.run_until(2)
    before = deepcopy(helper.report["events"][0])
    pause = run.pause()
    assert wake.updates_completed == replay.replay_examples_completed == 2
    run.resume(pause)
    assert helper.report["events"][0] == before
    run.run_until(6)
    result = run.finish()
    assert wake.updates_completed == replay.replay_examples_completed == 6
    assert result["rehearsal"]["events"][0] == before
    assert [row["coverage_cursor"] for row in result["rehearsal"]["events"]] == ["A", "B", "C"]
    assert len(run._runtime.applied_consolidations) == 3


def test_should_refuse_changed_pause_cursor_before_further_replay_and_keep_original_charges():
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper, batches())
    run.run_until(2)
    pause = run.pause()
    pause._snapshot["rehearsal_state"].events[0]["coverage_cursor"] = "C"
    with pytest.raises(ValueError):
        run.resume(pause)
    assert wake.updates_completed == replay.replay_examples_completed == 2
    assert len(source.control["calls"]) == 3
    assert helper.report["coverage_cursor"] == "A"


def test_should_keep_failed_attempt_exposure_charged_without_publishing_failed_replay():
    source, wake, replay, helper = setup(fail_call=3)
    run = stream(source, wake, helper, batches())
    with pytest.raises(RuntimeError, match="coverage replay native failure"):
        run.run_until(6)
    assert wake.updates_completed == replay.replay_examples_completed == 2
    event = helper.report["events"][0]
    assert event["status"] == "failed" and event["charged_exposures"] == 2
    assert (
        event["attempted_updates"] == 1
        and event["returned_updates"] == event["committed_updates"] == 0
    )
    assert run._runtime._candidate.snapshot_state()[0] == 2
    assert run._runtime.applied_consolidations == ()


def test_should_detach_coverage_policy_and_report_from_caller_mutation():
    source, wake, replay, _ = setup()
    retained = ReplayRetentionBudget(6, ROW_BYTES * 3)
    selected = policy(retention=retained)
    object.__setattr__(retained, "max_examples", 1)
    assert selected.retention.max_examples == 6
    helper = StreamRehearsal(
        source,
        wake,
        selected,
        replay_budget=replay,
        total_updates=6,
        input_dim=2,
        episode_schema=SCHEMA,
    )
    object.__setattr__(selected, "contexts", ("unknown",))
    object.__setattr__(selected.retention, "max_bytes", 1)
    assert helper.report["policy"]["contexts"] == ("A", "B", "C")
    assert helper.report["policy"]["retention"]["max_bytes"] == ROW_BYTES * 3


def test_should_refuse_undeclared_key_without_retaining_it_or_running_replay():
    source, wake, replay, helper = setup()
    with pytest.raises(ValueError):
        run = stream(source, wake, helper, batches(("Z", "A", "C", "A", "B", "C")))
        run.run_until(6)
    assert wake.updates_completed == len(source.control["calls"])
    assert wake.updates_completed in (0, 1) and replay.replay_examples_completed == 0
    assert helper.report["coverage_keys"] == []
    assert helper.retention.example_count == 0


def test_should_reject_non_train_episode_metadata_before_any_native_or_replay_work():
    source, wake, replay, helper = setup()
    training = list(batches())
    training[0] = replace(training[0], episode=replace(training[0].episode, role="final_test"))
    with pytest.raises(ValueError):
        stream(source, wake, helper, tuple(training))
    assert wake.updates_completed == replay.replay_examples_completed == 0
    assert source.control["calls"] == []
