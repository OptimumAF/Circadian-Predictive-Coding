"""Complete demonstrations cross existing stream, replay and continuation ports."""

from copy import deepcopy
from dataclasses import replace
import json
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.torch_feature_rehearsal_pilot as pilot
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal, StreamReplayPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.line_navigation import LineNavigation
from src.core.shared_replay_schedule import ReplayAllocation
from test_torch_feature_rehearsal_pilot import environment, forbidden


def batch(identity, context="A", role="train"):
    direction = 1.0 if context == "A" else -1.0
    start = 0.0 if role == "train" else -0.5 * direction
    goal = direction if role == "train" else 0.5 * direction
    packet = DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                step,
                (start + step * 0.5 * direction, goal),
                int(direction > 0),
                float(step),
                (start + (step + 1) * 0.5 * direction, goal),
                step == 1,
            )
            for step in range(2)
        ),
    )
    features, targets = episode_arrays(packet)
    return AttributionBatch(
        role,
        context,
        tuple(f"{identity}:step:{step}" for step in range(2)),
        features,
        targets,
        packet,
    )


def data():
    return (
        tuple(
            batch(f"train-{index}", context) for index, context in enumerate(("A", "A", "B", "B"))
        ),
        (batch("final-A", "A", "final_test"), batch("final-B", "B", "final_test")),
    )


def policy(**changes):
    values: dict[str, Any] = dict(
        every_updates=2,
        rows_per_event=2,
        retention=ReplayRetentionBudget(4, 1808),
        allocation=ReplayAllocation(ReplayRetentionBudget(2, 904), ReplayRetentionBudget(2, 904)),
        seed=53,
    )
    values.update(changes)
    return StreamReplayPolicy(**values)


class Learner:
    """Trace attempted fake calls and preserve ordinary owned-fork behavior."""

    def __init__(self, control=None, state=0.0):
        self.control: dict[str, Any] = (
            dict(forks=0, trace=[], last_charge=0, fail_replay=False, mutate_inputs=False)
            if control is None
            else control
        )
        self.state = np.array([state], dtype=np.float64)

    def fork(self):
        self.control["forks"] += 1
        return Learner(self.control, self.state[0])

    def snapshot_state(self):
        return self.state.copy()

    def restore_state(self, state):
        self.state = state.copy()

    def train_batch(self, features, targets):
        charge = self.control["replay"].replay_examples_completed
        replay = charge > self.control["last_charge"]
        self.control["last_charge"] = charge
        self.control["trace"].append((replay, charge, features.copy(), targets.copy()))
        self.state += 1.0 + float(np.mean(features)) + float(np.mean(targets))
        if replay and self.control["fail_replay"]:
            raise RuntimeError("whole episode replay fault")
        if self.control["mutate_inputs"]:
            features.fill(99.0)
            targets.fill(0.5)
        return TrainingDiagnostic("fake_trajectory_v1", float(self.state[0]))

    def predict(self, features):
        return np.full((len(features), 1), 0.5, dtype=np.float64)


def setup(*, replay_limit=4, schema=True, selected_policy=None, **faults):
    clock = lambda: 0.0
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4, max_wall_seconds=10), clock)
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=replay_limit, max_wall_seconds=10), clock
    )
    source = Learner()
    source.control.update(replay=replay, **faults)
    helper = StreamRehearsal(
        source,
        wake,
        policy() if selected_policy is None else selected_policy,
        replay_budget=replay,
        total_updates=4,
        input_dim=2,
        episode_schema=EpisodeReplaySchema(2, 2) if schema else None,
    )
    return source, wake, replay, helper


def stream(source, wake, helper, *, training=None, heldout=None, delay=1, observer=None):
    defaults = data()
    return AttributionStream(
        source,
        defaults[0] if training is None else training,
        defaults[1] if heldout is None else heldout,
        checkpoint_updates=(0, 2, 4),
        label_delay=delay,
        budget=wake,
        rehearsal=helper,
        observe_checkpoint=observer,
    )


@pytest.mark.parametrize("delay", [0, 1])
def test_should_replay_one_complete_ordered_packet_per_event_and_charge_transitions(delay):
    source, wake, replay, helper = setup()
    owner = stream(source, wake, helper, delay=delay)
    owner.run_until(4)
    result = owner.finish()

    assert result["training_examples"] == 8
    assert result["completed_updates"] == wake.updates_completed == 4
    assert replay.replay_examples_completed == 4
    trace = source.control["trace"]
    assert [item[0] for item in trace] == [False, False, True, False, False, True]
    assert [item[1] for item in trace if item[0]] == [2, 4]
    assert all(item[2].shape == (2, 2) and item[3].shape == (2, 1) for item in trace)
    assert helper.retention.example_count <= 4
    assert helper.retention.retained_bytes <= 1808
    assert helper.report["episode_schema"] == dict(observation_dim=2, steps_per_episode=2)
    for event, actual in zip(
        helper.report["events"], [row for row in trace if row[0]], strict=True
    ):
        assert event["status"] == "complete"
        assert event["charged_exposures"] == 2
        assert (
            event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 1
        )
        assert len(event["selected_episodes"]) == 1
        chosen = event["selected_episodes"][0]
        assert chosen["role"] == "train"
        np.testing.assert_array_equal(
            actual[2], [step["observation"] for step in chosen["transitions"]]
        )
        np.testing.assert_array_equal(
            actual[3], [[step["action"]] for step in chosen["transitions"]]
        )
        assert event["retention"]["retained_bytes"] in (904, 1808)
    assert source.state[0] == 0.0
    assert [row["episode_id"] for row in result["applied_updates"]] == [
        f"train-{index}" for index in range(4)
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "features",
        "targets",
        "context",
        "role",
        "row_ids",
        "missing_train",
        "missing_final",
        "duplicate_id",
    ],
)
def test_should_refuse_disagreeing_or_mixed_metadata_before_binding_or_forks(fault):
    source, wake, _, helper = setup()
    training, heldout = data()
    first = training[0]
    if fault in ("features", "targets"):
        changed = getattr(first, fault).copy()
        changed[0, 0] = 0.25 if fault == "features" else 0.0
        first = replace(first, **{fault: changed})
    elif fault == "context":
        first = replace(first, context="B")
    elif fault == "role":
        first = replace(first, episode=replace(first.episode, role="final_test"))
    elif fault == "row_ids":
        first = replace(first, row_ids=("wrong-0", "wrong-1"))
    elif fault == "missing_train":
        first = replace(first, episode=None)
    elif fault == "missing_final":
        heldout = (replace(heldout[0], episode=None), heldout[1])
    else:
        heldout = (batch("train-0", "A", "final_test"), heldout[1])

    with pytest.raises(ValueError):
        stream(source, wake, helper, training=(first, *training[1:]), heldout=heldout)
    assert source.control["forks"] == 0 and source.control["trace"] == []
    assert helper.report["bound"] is False


@pytest.mark.parametrize("mismatch", ["episode_with_row_helper", "row_with_episode_helper"])
def test_should_require_the_helper_geometry_to_match_all_ingress_metadata(mismatch):
    source, wake, _, helper = setup(schema=mismatch != "episode_with_row_helper")
    training, heldout = data()
    if mismatch == "row_with_episode_helper":
        training = tuple(replace(row, episode=None) for row in training)
        heldout = tuple(replace(row, episode=None) for row in heldout)
    with pytest.raises(ValueError):
        stream(source, wake, helper, training=training, heldout=heldout)
    assert helper.report["bound"] is False
    assert source.control["forks"] == 0


def test_should_copy_ingress_and_observer_packets_before_caller_or_observer_mutation():
    source, wake, _, helper = setup(mutate_inputs=True)
    training, heldout = data()
    observed = []

    def observer(updates, state, arrived):
        if arrived is not None:
            observed.append(deepcopy(arrived.episode))
            object.__setattr__(arrived.episode, "episode_id", "mutated-observer")
            arrived.features.fill(77.0)

    owner = stream(source, wake, helper, training=training, heldout=heldout, observer=observer)
    training[0].features.fill(88.0)
    object.__setattr__(training[0].episode, "episode_id", "mutated-caller")
    owner.run_until(4)
    owner.finish()
    assert [packet.episode_id for packet in observed] == ["train-1", "train-3"]
    assert all(
        packet["episode_id"].startswith("train-")
        for event in helper.report["events"]
        for packet in event["selected_episodes"]
    )
    assert all(np.max(np.abs(call[2])) <= 1.0 for call in source.control["trace"])


def test_should_refuse_a_partial_episode_budget_before_replay_fork_or_charge():
    source, wake, replay, helper = setup(replay_limit=1)
    owner = stream(source, wake, helper)
    owner.run_until(1)
    forks_before = source.control["forks"]
    with pytest.raises(ToyExecutionStopped):
        owner.run_until(2)
    assert replay.replay_examples_completed == 0
    assert source.control["forks"] == forks_before
    assert len(source.control["trace"]) == wake.updates_completed == 2
    event = helper.report["events"][0]
    assert event["status"] == "failed" and event["receipt"] is None
    assert event["attempted_updates"] == event["charged_exposures"] == 0


def test_should_preserve_both_charges_when_a_whole_episode_native_call_fails():
    source, wake, replay, helper = setup(fail_replay=True)
    owner = stream(source, wake, helper)
    with pytest.raises(RuntimeError, match="whole episode replay fault"):
        owner.run_until(2)
    event = helper.report["events"][0]
    assert replay.replay_examples_completed == event["charged_exposures"] == 2
    assert event["attempted_updates"] == 1
    assert event["returned_updates"] == event["committed_updates"] == 0
    assert event["receipt"] is None and wake.updates_completed == 2
    assert source.state[0] == 0.0


def test_should_retain_numerically_identical_demonstrations_as_distinct_episode_ids():
    # Why FIFO here: partition membership can overlap, so two stored slots are
    # not guaranteed by two partition quotas. Isolate identity from selection.
    source, wake, _, helper = setup(
        selected_policy=RecentStreamReplayPolicy(2, 2, ReplayRetentionBudget(4, 1808))
    )
    owner = stream(source, wake, helper)
    owner.run_until(2)
    assert helper.retention.example_count == 4
    assert helper.retention.retained_bytes == 1808
    assert len(helper.retention.sample_ids) == 2
    assert len(set(helper.retention.sample_ids)) == 2


@pytest.mark.parametrize("fault", ["tamper", "replace"])
def test_should_refuse_changed_original_episode_schema_without_native_work(fault):
    source, wake, _, helper = setup()
    if fault == "tamper":
        object.__setattr__(helper._episode_schema, "steps_per_episode", 1)
    else:
        helper._episode_schema = EpisodeReplaySchema(2, 2)
    with pytest.raises(ValueError):
        stream(source, wake, helper)
    assert source.control["forks"] == 0


def test_should_expose_detached_episode_schema_without_granting_mutation_authority():
    _, _, _, helper = setup()
    detached = helper.episode_schema
    object.__setattr__(detached, "steps_per_episode", 1)
    assert helper.episode_schema == EpisodeReplaySchema(2, 2)


@pytest.mark.parametrize("fault", ["bytes", "transitions", "event_rows"])
def test_should_refuse_quotas_that_cannot_hold_one_complete_episode(fault):
    changes: dict[str, Any] = {}
    if fault == "bytes":
        changes["allocation"] = ReplayAllocation(
            ReplayRetentionBudget(2, 903), ReplayRetentionBudget(2, 904)
        )
    elif fault == "transitions":
        changes["allocation"] = ReplayAllocation(
            ReplayRetentionBudget(1, 904), ReplayRetentionBudget(2, 904)
        )
    else:
        changes["rows_per_event"] = 1
    with pytest.raises(ValueError):
        setup(selected_policy=policy(**changes))


def test_should_resume_the_original_stream_with_identical_later_episode_selection_and_cost():
    first, first_wake, first_replay, first_helper = setup()
    second, second_wake, second_replay, second_helper = setup()
    reference = stream(first, first_wake, first_helper)
    resumed = stream(second, second_wake, second_helper)
    reference.run_until(4)
    expected_state = reference.candidate_snapshot()
    expected = reference.finish()
    resumed.run_until(2)
    runtime, actor, original_limits = resumed._runtime, resumed._runtime.actor, second_wake.budget
    token = resumed.pause()
    snapshot = token.snapshot_state()
    assert [item.episode_id for item in snapshot["inbox_cursor"].experiences] == [
        "train-0",
        "train-1",
        "train-2",
    ]
    assert [item.episode_id for item in snapshot["inbox_cursor"].labels] == ["train-0", "train-1"]
    resumed.resume(token)
    assert resumed._runtime is runtime and runtime.actor is actor
    assert second_wake.budget is original_limits
    assert second_replay.replay_examples_completed == 2
    resumed.run_until(4)
    np.testing.assert_array_equal(resumed.candidate_snapshot(), expected_state)
    assert resumed.finish() == expected
    assert second_wake.updates_completed == first_wake.updates_completed == 4
    assert second_replay.replay_examples_completed == first_replay.replay_examples_completed == 4


def test_should_build_actual_legal_complete_navigation_demonstrations_without_torch(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    training, heldout = pilot.build_trajectory_stream()
    assert [row.context for row in training] == ["A", "A", "B", "B", "A", "A"]
    identifiers = []
    for row in (*training, *heldout):
        packet = row.episode
        assert packet is not None
        identifiers.append(packet.episode_id)
        assert packet.role == row.role and packet.context == row.context
        assert row.row_ids == tuple(f"{packet.episode_id}:step:{step}" for step in range(2))
        features, targets = episode_arrays(packet)
        np.testing.assert_array_equal(row.features, features)
        np.testing.assert_array_equal(row.targets, targets)
        for step in packet.transitions:
            expected_move = 0.5 if step.action else -0.5
            assert step.next_observation[0] - step.observation[0] == expected_move
            assert step.next_observation[1] == step.observation[1]
            assert step.reward == float(step.terminal)
        assert packet.transitions[-1].next_observation[0] == packet.transitions[-1].observation[1]
    assert len(set(identifiers)) == len(identifiers) == 8
    assert all(abs(row.features[0, 1]) == 0.5 for row in heldout)


@pytest.mark.parametrize("goal,action", [(-2, 0), (2, 1)])
def test_should_observe_two_actual_environment_steps_and_terminate_at_the_goal(goal, action):
    environment = LineNavigation(0, goal)
    first = environment.step(action)
    last = environment.step(action)
    assert first.observation == (goal / 4, goal / 2)
    assert first.reward == 0.0 and first.terminal is False
    assert last.observation == (goal / 2, goal / 2)
    assert last.reward == 1.0 and last.terminal is True


@pytest.mark.parametrize("fault", ["boolean", "float", "invalid", "boundary", "terminal"])
def test_should_refuse_invalid_environment_actions_without_advancing_state(fault):
    environment = LineNavigation(-2 if fault == "boundary" else 0, 2)
    if fault == "terminal":
        environment.step(1)
        environment.step(1)
    action: Any = {"boolean": True, "float": 1.0, "invalid": 2, "boundary": 0, "terminal": 0}[fault]
    before = environment.observation
    with pytest.raises(ValueError):
        environment.step(action)
    assert environment.observation == before


def test_should_describe_frozen_transition_costs_without_constructing_data_or_native_resources(
    monkeypatch,
):
    for name in (
        "require_torch",
        "TorchFeatureLearner",
        "ProcessRssSampler",
        "build_trajectory_stream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    description = pilot.describe_trajectory_protocol()
    assert description["trajectory_enabled"] is True
    assert description["planned_work"]["native_train_calls"] == 60
    assert description["planned_work"]["training_row_exposures"] == 120
    assert description["replay"]["retention"] == dict(max_examples=4, max_bytes=1808)


def test_should_execute_all_eight_fake_trajectory_arms_with_complete_transition_costs(monkeypatch):
    environment(monkeypatch)
    report = pilot.run_trajectory_pilot()
    assert report["status"] == "complete", report["runs"]
    assert report["native_train_calls"] == 60
    assert report["training_row_exposures"] == 120
    assert report["final_score_forwards"] == 64
    assert report["completed_resumes"] == 16
    assert report["replay_charged_examples"] == 24
    assert report["replay_committed_updates"] == 12
    for arm in report["runs"]:
        assert arm["source_learner_unchanged"] is True
        if arm["replay_enabled"]:
            assert arm["final_retention"]["example_count"] <= 4
            assert arm["final_retention"]["retained_bytes"] <= 1808
            assert all(len(event["selected_episodes"]) == 1 for event in arm["rehearsal"]["events"])


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_the_trajectory_cli_only_to_the_requested_callback(
    monkeypatch, capsys, describe
):
    selected = "describe_trajectory_protocol" if describe else "run_trajectory_pilot"
    for name in ("describe_trajectory_protocol", "run_trajectory_pilot"):
        monkeypatch.setattr(
            cli, name, (lambda: {"status": "complete"}) if name == selected else forbidden
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-trajectory-rehearsal"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
