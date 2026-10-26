"""Real bounded episode rehearsal and return evaluation with owned fake learners."""

from copy import deepcopy
from dataclasses import replace
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
from src.app.stream_rehearsal import StreamRehearsal
from src.core.demonstration_episode import EpisodeReplaySchema
from src.core.learner_ports import TrainingDiagnostic
from test_recurrent_key_door_pilot import Learner as RecurrentLearner, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden
from test_train_acquisition_pilot import Learner


class Session:
    def __init__(self, source):
        self.state = source.snapshot_state()
        self.completed_steps = 0
        self.inputs: list[Any] = []

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert ReplayLearner.final_started and len(Learner.training_owners) == 288
        assert features.shape == (1, 8)
        self.inputs.append(features.copy())
        self.completed_steps += 1
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)


class ReplayLearner(Learner):
    final_started = False
    fit_native_updates: list[int] = []
    cached_native_updates: list[int] = []
    live_sessions: list[Session] = []
    corrupt: Any = None

    def train_batch(self, features, targets):
        assert not ReplayLearner.final_started, "final data reached training/rehearsal"
        super().train_batch(features, targets)
        return TrainingDiagnostic(TORCH_RECURRENT_CE_ID, 0.5)

    def predict(self, features):
        if ReplayLearner.final_started:
            assert len(Learner.training_owners) == 288
            ReplayLearner.cached_native_updates.append(self.state.completed_updates)
        else:
            ReplayLearner.fit_native_updates.append(self.state.completed_updates)
        result = super().predict(features)
        if ReplayLearner.final_started and ReplayLearner.corrupt is not None:
            ReplayLearner.corrupt()
        return result

    def start_episode(self):
        assert ReplayLearner.final_started and len(Learner.training_owners) == 288
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
    for name in ("fit_native_updates", "cached_native_updates", "live_sessions"):
        monkeypatch.setattr(ReplayLearner, name, [])
    monkeypatch.setattr(ReplayLearner, "final_started", False)
    monkeypatch.setattr(ReplayLearner, "corrupt", None)
    captured: dict[str, Any] = dict(collections=[], finished=False)
    collect, finish = shared._collect, AttributionStream.finish

    def collect_packet(identity, layout, cue, role):
        assert role in ("train", "final_test")
        if role == "final_test":
            assert captured["finished"] is True and len(Learner.training_owners) == 288
            ReplayLearner.final_started = True
        else:
            assert not ReplayLearner.final_started
        batch = collect(identity, layout, cue, role)
        captured["collections"].append(deepcopy(batch))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert len(training) == 192 and kwargs["checkpoint_updates"] == pilot.RETURN_POINTS
        helper = kwargs["rehearsal"]
        assert type(helper) is StreamRehearsal
        assert helper.episode_schema == EpisodeReplaySchema(8, 8, 5)
        captured.update(helper=helper, wake=kwargs["budget"], replay_budget=helper._replay_budget)
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured.update(stream=instance, isolated=learner)
        return instance

    def finished(instance):
        result = finish(instance)
        assert instance.completed_updates == 192
        captured["finished"] = True
        return result

    monkeypatch.setattr(shared, "_collect", collect_packet)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(AttributionStream, "finish", finished)
    return captured


def test_should_describe_fixed_replay_costs_without_resources_or_changing_replay_off_protocol(
    monkeypatch,
):
    before = deepcopy(pilot.describe_return_evaluation_protocol())
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
        "StreamRehearsal",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_return_replay_protocol()
    assert pilot.describe_return_evaluation_protocol() == before
    replay = protocol["replay"]
    assert replay["max_replay_examples"] == 768
    assert replay["episode_schema"] == dict(observation_dim=8, steps_per_episode=8, num_classes=5)
    assert replay["policy"]["every_updates"] == 2 and replay["policy"]["rows_per_event"] == 8
    assert replay["policy"]["seed"] == 53
    assert replay["policy"]["retention"] == dict(max_examples=16, max_bytes=4032)
    work = protocol["planned_work"]
    assert work["wake_train_calls"] == 192 and work["native_train_calls"] == 288
    assert work["wake_training_rows"] == 1536 and work["training_rows"] == 2304
    assert work["replay_calls"] == 96 and work["replay_rows"] == 768
    assert work["maximum_recurrent_unroll_steps"] == 3360


def test_should_replay_complete_train_episodes_with_bounded_retention_and_original_separate_budgets(
    monkeypatch,
):
    captured = environment(monkeypatch)
    report = pilot.run_return_replay_pilot()
    assert report["status"] == "complete", report["reason"]
    assert report["charged_updates"] == captured["wake"].updates_completed == 192
    assert captured["wake"].budget.max_training_updates == 192
    assert captured["wake"].replay_examples_completed == 0
    assert captured["replay_budget"].budget.max_replay_examples == 768
    assert captured["replay_budget"].updates_completed == 0
    assert captured["replay_budget"].replay_examples_completed == 768
    assert len(Learner.training_owners) == len(RecurrentLearner.trains) == 288
    assert ReplayLearner.fit_native_updates == [
        point + point // 2 for point in pilot.RETURN_POINTS for _ in range(4)
    ]
    assert ReplayLearner.cached_native_updates == [n for n in (0, 96, 192, 288) for _ in range(8)]
    assert len(ReplayLearner.live_sessions) == 32
    assert [s.state.completed_updates for s in ReplayLearner.live_sessions] == [
        n for n in (0, 96, 192, 288) for _ in range(8)
    ]
    replay = report["replay"]
    assert replay["charged_examples"] == replay["rehearsal"]["charged_exposures"] == 768
    assert replay["rehearsal"]["committed_updates"] == 96
    events = replay["rehearsal"]["events"]
    receipts = replay["applied_consolidations"]
    assert len(events) == len(receipts) == 96
    assert [event["updates"] for event in events] == list(range(2, 193, 2))
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
        episode = event["selected_episodes"][0]
        assert episode["role"] == "train" and episode["num_classes"] == 5
        assert len(episode["transitions"]) == 8
        assert event["retention"]["example_count"] <= 16
        assert event["retention"]["retained_bytes"] <= 4032
    assert replay["final_retention"]["example_count"] <= 16
    assert replay["final_retention"]["retained_bytes"] <= 4032
    assert report["partial_stream"]["native_candidate_updates"] == 288
    assert report["partial_stream"]["native_candidate_training_examples"] == 2304
    for point in report["checkpoints"]:
        wake = point["updates"]
        assert point["clocks"] == dict(
            wake_updates=wake,
            replay_committed_updates=wake // 2,
            replay_charged_examples=4 * wake,
            native_updates=wake + wake // 2,
            native_training_examples=8 * (wake + wake // 2),
        )
    for point in report["policy_evaluation"]["checkpoints"]:
        assert point["clocks"]["wake_updates"] == point["updates"]
        assert point["clocks"]["native_updates"] == point["updates"] * 3 // 2
    for episode in report["policy_evaluation"]["episodes"]:
        assert episode["native_checkpoint_updates"] == episode["checkpoint_updates"] * 3 // 2
        assert episode["outcome"] == "illegal_action" and episode["state_unchanged"] is True
        assert not episode["success"] and episode["return"] == 0
    assert [batch.role for batch in captured["collections"]] == ["train"] * 4 + ["final_test"] * 8
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert report["scientific_advantage_verified"] is False


def test_should_charge_failed_first_replay_exposure_and_keep_two_wakes_without_final_data(
    monkeypatch,
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_third", True)
    report = pilot.run_return_replay_pilot()
    assert report["status"] == "incomplete" and "third train-only" in report["reason"]
    assert report["charged_updates"] == captured["wake"].updates_completed == 2
    assert (
        captured["replay_budget"].replay_examples_completed
        == report["replay"]["charged_examples"]
        == 8
    )
    assert len(Learner.training_owners) == 3 and len(RecurrentLearner.trains) == 2
    event = report["replay"]["rehearsal"]["events"][0]
    assert event["status"] == "failed" and event["charged_exposures"] == 8
    assert event["attempted_updates"] == 1
    assert event["returned_updates"] == event["committed_updates"] == 0
    assert event["receipt"] is None and report["replay"]["applied_consolidations"] == []
    assert report["partial_stream"]["native_candidate_updates"] == 2
    assert report["partial_stream"]["native_candidate_training_examples"] == 16
    assert [point["updates"] for point in report["checkpoints"]] == [0]
    assert all(batch.role == "train" for batch in captured["collections"])
    assert report["policy_evaluation"]["status"] == "not_run"
    assert ReplayLearner.cached_native_updates == ReplayLearner.live_sessions == []


@pytest.mark.parametrize(
    "fault",
    ["native_counter", "receipt", "event", "selected_episode", "replay_budget", "budget_binding"],
)
def test_should_refuse_changed_replay_evidence_after_one_final_prediction_without_more_work(
    monkeypatch, fault
):
    captured = environment(monkeypatch)
    changed: list[bool] = []

    def corrupt():
        assert not changed, "corrupted replay evidence admitted a second final prediction"
        changed.append(True)
        stream, helper = captured["stream"], captured["helper"]
        if fault == "native_counter":
            stream._runtime._candidate.state.completed_updates += 1
        elif fault == "receipt":
            receipt = stream._runtime._consolidations[0]
            stream._runtime._consolidations[0] = replace(
                receipt,
                diagnostic=replace(receipt.diagnostic, value=receipt.diagnostic.value + 0.125),
            )
        elif fault == "event":
            helper._events[0]["diagnostics"][0]["value"] += 0.125
        elif fault == "selected_episode":
            transition = helper._events[0]["selected_episodes"][0]["transitions"][0]
            transition["action"] = (transition["action"] + 1) % 5
        elif fault == "replay_budget":
            captured["replay_budget"].replay_examples_completed -= 8
        else:
            helper._replay_budget = deepcopy(captured["replay_budget"])

    monkeypatch.setattr(ReplayLearner, "corrupt", staticmethod(corrupt))
    report = pilot.run_return_replay_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == captured["wake"].updates_completed == 192
    assert len(Learner.training_owners) == 288
    assert len(ReplayLearner.cached_native_updates) == 1 and ReplayLearner.live_sessions == []
    assert report["policy_evaluation"]["status"] == "failed"
    assert report["policy_evaluation"]["checkpoints"] == []
    assert all(slot["status"] == "not_run" for slot in report["policy_evaluation"]["episodes"])
    assert report["stream_result"]["rehearsal"]["charged_exposures"] == 768
    assert len(report["stream_result"]["applied_consolidations"]) == 96
    if fault != "replay_budget":
        assert captured["replay_budget"].replay_examples_completed == 768
    if fault == "budget_binding":
        assert report["replay_readback_error"] is not None
        assert report["reason"] != report["replay_readback_error"]
        assert not report["reason"].startswith("replay readback")


def test_should_refuse_non_train_packet_before_replay_or_scoring(monkeypatch):
    captured = environment(monkeypatch)
    collect = shared._collect

    def wrong_role(identity, layout, cue, role):
        batch = collect(identity, layout, cue, role)
        assert batch.episode is not None
        return replace(
            batch, role="probe_train", episode=replace(batch.episode, role="probe_train")
        )

    monkeypatch.setattr(shared, "_collect", wrong_role)
    report = pilot.run_return_replay_pilot()
    assert report["status"] == "incomplete" and report["charged_updates"] == 0
    assert Learner.training_owners == Learner.prediction_owners == []
    assert "helper" not in captured


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_episode_replay_cli_without_arguments(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-return-replay"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="episode-replay")

    monkeypatch.setattr(cli, "describe_return_replay_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_return_replay_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_return_policy_pilot", forbidden)
    monkeypatch.setattr(cli, "run_train_context_return_pilot", forbidden)
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["selected"] == "episode-replay"
