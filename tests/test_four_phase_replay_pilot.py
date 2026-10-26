"""Matched existing replay controls on the real four-phase stream with fake native ports."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID
from src.app.coupled_attribution import AttributionStream
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal, StreamReplayPolicy
from src.core.learner_ports import TrainingDiagnostic
from test_recurrent_key_door_pilot import Learner as RecurrentLearner, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden
from test_train_acquisition_pilot import Learner


class DerivedControl(str):
    pass


class Session:
    def __init__(self, source):
        self.state = source.snapshot_state()
        self.completed_steps = 0

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert ReplayLearner.final_started and len(Learner.training_owners) == 384
        assert features.shape == (1, 8)
        self.completed_steps += 1
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)


class ReplayLearner(Learner):
    final_started = False
    cached_native_updates: list[int] = []
    live_sessions: list[Session] = []
    corrupt: Any = None

    def train_batch(self, features, targets):
        assert not ReplayLearner.final_started, "final data entered wake or replay updates"
        super().train_batch(features, targets)
        return TrainingDiagnostic(TORCH_RECURRENT_CE_ID, 0.5)

    def predict(self, features):
        if ReplayLearner.final_started:
            assert len(Learner.training_owners) == 384
            ReplayLearner.cached_native_updates.append(self.state.completed_updates)
        result = super().predict(features)
        if ReplayLearner.final_started and ReplayLearner.corrupt is not None:
            ReplayLearner.corrupt()
        return result

    def start_episode(self):
        assert ReplayLearner.final_started and len(Learner.training_owners) == 384
        session = Session(self)
        ReplayLearner.live_sessions.append(session)
        return session


def environment(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", ReplayLearner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    for name in ("sources", "training_owners", "prediction_owners"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_third", False)
    monkeypatch.setattr(Learner, "mutate_prediction", False)
    monkeypatch.setattr(Learner, "mutation_update", 0)
    monkeypatch.setattr(ReplayLearner, "final_started", False)
    monkeypatch.setattr(ReplayLearner, "cached_native_updates", [])
    monkeypatch.setattr(ReplayLearner, "live_sessions", [])
    monkeypatch.setattr(ReplayLearner, "corrupt", None)
    captured: dict[str, Any] = dict(collections=[], finished=False)
    collect, finish = shared._collect, AttributionStream.finish

    def collect_packet(identity, layout, cue, role):
        assert role in ("train", "final_test")
        if role == "final_test":
            assert captured["finished"] is True and len(Learner.training_owners) == 384
            ReplayLearner.final_started = True
        else:
            assert not ReplayLearner.final_started
        batch = collect(identity, layout, cue, role)
        captured["collections"].append(deepcopy(batch))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert len(training) == 256 and heldout == () and kwargs["training_only"] is True
        assert kwargs["checkpoint_updates"] == pilot.FOUR_PHASE_POINTS
        helper = kwargs["rehearsal"]
        assert type(helper) is StreamRehearsal
        captured.update(
            training=deepcopy(training),
            helper=helper,
            wake=kwargs["budget"],
            replay_budget=helper._replay_budget,
        )
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured.update(stream=instance, isolated=learner)
        return instance

    def finished(instance):
        result = finish(instance)
        assert instance.completed_updates == 256
        captured["finished"] = True
        return result

    monkeypatch.setattr(shared, "_collect", collect_packet)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(AttributionStream, "finish", finished)
    return captured


@pytest.mark.parametrize("control", ["recent", "partitioned"])
def test_should_describe_equal_work_controls_without_resource_work_or_changing_no_replay(
    monkeypatch, control
):
    before = deepcopy(pilot.describe_four_phase_evaluation_protocol())
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
        "StreamRehearsal",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_four_phase_replay_protocol(control=control)
    assert protocol["replay_control"] == control
    assert protocol["replay"]["native_checkpoint_updates"] == [0, 96, 192, 288, 384]
    assert pilot.describe_four_phase_evaluation_protocol() == before
    work = protocol["planned_work"]
    assert work["wake_train_calls"] == 256 and work["native_train_calls"] == 384
    assert work["wake_training_rows"] == 2048 and work["training_rows"] == 3072
    assert work["replay_calls"] == 128 and work["replay_rows"] == 1024
    assert work["final_cached_forwards"] == 40 and work["live_episodes"] == 40
    assert work["maximum_recurrent_unroll_steps"] == 4688


@pytest.mark.parametrize(
    "entry", ["describe_four_phase_replay_protocol", "run_four_phase_replay_pilot"]
)
@pytest.mark.parametrize("invalid", [None, 0, "unknown", DerivedControl("recent")])
def test_should_refuse_unrecognized_or_non_string_control_before_resources(
    monkeypatch, entry, invalid
):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError, match="control"):
        getattr(pilot, entry)(control=invalid)


@pytest.mark.parametrize(
    "flag",
    [
        "four_phase",
        "four_phase_policy",
        "return_context",
        "evaluate_policy_after_training",
        "episode_replay",
        "fresh_task_probes",
    ],
)
def test_should_refuse_mixed_replay_modes_before_resources(monkeypatch, flag):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(four_phase_replay="recent", **{flag: True})


@pytest.mark.parametrize("control", ["recent", "partitioned"])
def test_should_match_original_budgets_exposures_and_retention_while_using_distinct_existing_policies(
    monkeypatch, control
):
    captured = environment(monkeypatch)
    report = pilot.run_four_phase_replay_pilot(control=control)
    assert report["status"] == "complete", report["reason"]
    assert report["replay_control"] == control
    assert report["policy_evaluation"]["replay_control"] == control
    helper, wake, replay_budget = captured["helper"], captured["wake"], captured["replay_budget"]
    assert wake is not replay_budget
    assert (
        wake.budget.max_training_updates
        == wake.updates_completed
        == report["charged_updates"]
        == 256
    )
    assert wake.replay_examples_completed == 0
    assert (
        replay_budget.budget.max_replay_examples == replay_budget.replay_examples_completed == 1024
    )
    assert replay_budget.updates_completed == 0
    assert len(Learner.training_owners) == len(RecurrentLearner.trains) == 384
    policy = helper._policy
    assert policy.every_updates == 2 and policy.rows_per_event == 8
    assert policy.retention.max_examples == 16 and policy.retention.max_bytes == 4032
    if control == "recent":
        assert type(policy) is RecentStreamReplayPolicy and not hasattr(policy, "seed")
    else:
        assert type(policy) is StreamReplayPolicy and policy.seed == 53
        assert (
            policy.allocation.recent.max_examples == policy.allocation.reservoir.max_examples == 8
        )
        assert policy.allocation.recent.max_bytes == policy.allocation.reservoir.max_bytes == 2016
    replay = report["replay"]
    assert replay["charged_examples"] == 1024
    assert replay["final_retention"]["example_count"] <= 16
    assert replay["final_retention"]["retained_bytes"] <= 4032
    events, receipts = replay["rehearsal"]["events"], replay["applied_consolidations"]
    assert len(events) == len(receipts) == 128
    released = {
        batch.episode.episode_id: (index + 1, asdict(batch.episode))
        for index, batch in enumerate(captured["training"])
    }
    for event, receipt in zip(events, receipts, strict=True):
        assert event["status"] == "complete" and event["receipt"] == receipt
        assert event["charged_exposures"] == 8
        assert (
            event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 1
        )
        assert event["diagnostics"] == [receipt["diagnostic"]]
        assert len(event["selected_episodes"]) == 1
        selected = event["selected_episodes"][0]
        admitted_at, original = released[selected["episode_id"]]
        assert (
            admitted_at <= event["updates"] and selected == original and selected["role"] == "train"
        )
        if control == "recent":
            assert admitted_at == event["updates"]
    assert report["partial_stream"]["native_candidate_updates"] == 384
    assert report["partial_stream"]["native_candidate_training_examples"] == 3072
    for point in report["checkpoints"]:
        wake_count = point["updates"]
        assert point["clocks"] == dict(
            wake_updates=wake_count,
            replay_committed_updates=wake_count // 2,
            replay_charged_examples=4 * wake_count,
            native_updates=3 * wake_count // 2,
            native_training_examples=12 * wake_count,
        )
    assert ReplayLearner.cached_native_updates == [
        n for n in (0, 96, 192, 288, 384) for _ in range(8)
    ]
    evaluation = report["policy_evaluation"]
    assert evaluation["replay_enabled"] is True and evaluation["live_sessions_started"] == 40
    assert len(evaluation["episodes"]) == len(ReplayLearner.live_sessions) == 40
    assert len(RecurrentLearner.classifications) == 142
    for episode in evaluation["episodes"]:
        assert episode["native_checkpoint_updates"] == 3 * episode["checkpoint_updates"] // 2
        assert episode["outcome"] == "illegal_action" and episode["state_unchanged"] is True
        assert not episode["success"] and episode["return"] == 0
    assert [batch.role for batch in captured["collections"]] == ["train"] * 6 + ["final_test"] * 8
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert report["scientific_advantage_verified"] is False


@pytest.mark.parametrize("control", ["recent", "partitioned"])
def test_should_keep_first_failed_replay_charge_and_two_committed_wakes_without_final_data(
    monkeypatch, control
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_third", True)
    report = pilot.run_four_phase_replay_pilot(control=control)
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["wake"].updates_completed == 2
    assert (
        report["replay"]["charged_examples"]
        == captured["replay_budget"].replay_examples_completed
        == 8
    )
    assert report["partial_stream"]["native_candidate_updates"] == 2
    event = report["replay"]["rehearsal"]["events"][0]
    assert event["status"] == "failed" and event["charged_exposures"] == 8
    assert (
        event["attempted_updates"] == 1
        and event["returned_updates"] == event["committed_updates"] == 0
    )
    assert event["receipt"] is None
    assert all(batch.role == "train" for batch in captured["collections"])
    assert ReplayLearner.cached_native_updates == ReplayLearner.live_sessions == []


@pytest.mark.parametrize(
    "fault", ["native_counter", "selected_episode", "receipt", "replay_budget"]
)
def test_should_refuse_corrupted_complete_replay_evidence_before_a_second_final_prediction(
    monkeypatch, fault
):
    captured = environment(monkeypatch)
    changed: list[bool] = []

    def corrupt():
        assert not changed, "corrupted replay evidence admitted more final work"
        changed.append(True)
        helper, stream = captured["helper"], captured["stream"]
        if fault == "native_counter":
            stream._runtime._candidate.state.completed_updates += 1
        elif fault == "selected_episode":
            transition = helper._events[0]["selected_episodes"][0]["transitions"][0]
            transition["action"] = (transition["action"] + 1) % 5
        elif fault == "receipt":
            receipt = stream._runtime._consolidations[0]
            stream._runtime._consolidations[0] = replace(
                receipt,
                diagnostic=replace(receipt.diagnostic, value=receipt.diagnostic.value + 0.125),
            )
        else:
            captured["replay_budget"].replay_examples_completed -= 8

    monkeypatch.setattr(ReplayLearner, "corrupt", staticmethod(corrupt))
    report = pilot.run_four_phase_replay_pilot(control="partitioned")
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == captured["wake"].updates_completed == 256
    assert len(Learner.training_owners) == 384 and len(ReplayLearner.cached_native_updates) == 1
    assert ReplayLearner.live_sessions == []
    assert report["policy_evaluation"]["status"] == "failed"
    assert report["policy_evaluation"]["checkpoints"] == []
    assert all(slot["status"] == "not_run" for slot in report["policy_evaluation"]["episodes"])
    assert report["stream_result"]["rehearsal"]["charged_exposures"] == 1024
    assert len(report["stream_result"]["applied_consolidations"]) == 128


@pytest.mark.parametrize("control", ["recent", "partitioned"])
def test_should_dispatch_explicit_control_cli_with_fixed_choice(monkeypatch, capsys, control):
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", f"torch-key-door-four-phase-{control}"]
    )
    calls = []

    def selected(*, control):
        calls.append(control)
        return dict(status="complete", selected=control)

    monkeypatch.setattr(cli, "run_four_phase_replay_pilot", selected)
    monkeypatch.setattr(cli, "run_four_phase_evaluation_pilot", forbidden)
    cli.main()
    assert calls == [control] and json.loads(capsys.readouterr().out)["selected"] == control
