"""Post-training cached/live policy evaluation with real streams and owned fake ports."""

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
from src.app.coupled_attribution import AttributionStream
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
        assert PolicyLearner.final_started and len(Learner.training_owners) == 192
        assert features.shape == (1, 8)
        self.inputs.append(features.copy())
        if PolicyLearner.failure == "live_raise":
            raise RuntimeError("first live policy callback failed")
        self.completed_steps += 1
        if PolicyLearner.failure == "live_state":
            self.state.completed_updates += 1
        if PolicyLearner.failure == "live_source":
            Learner.sources[0].state.completed_updates += 1
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)


class PolicyLearner(Learner):
    final_started = False
    cached_updates: list[int] = []
    live_sessions: list[Session] = []
    failure: str | None = None
    corrupt_receipt: Any = None
    corrupt_checkpoint: Any = None

    def train_batch(self, features, targets):
        assert not PolicyLearner.final_started, "final data became available before training ended"
        return super().train_batch(features, targets)

    def predict(self, features):
        if PolicyLearner.final_started:
            assert len(Learner.training_owners) == 192
            assert self.state.completed_updates in (0, 64, 128, 192)
            PolicyLearner.cached_updates.append(self.state.completed_updates)
            if PolicyLearner.failure == "cached_raise":
                raise RuntimeError("first cached policy prediction failed")
        prediction = super().predict(features)
        if PolicyLearner.final_started:
            if PolicyLearner.failure == "cached_state":
                self.state.completed_updates += 1
            elif PolicyLearner.failure == "cached_source":
                Learner.sources[0].state.completed_updates += 1
            elif PolicyLearner.failure == "receipt":
                PolicyLearner.corrupt_receipt()
            elif PolicyLearner.failure == "saved_state":
                PolicyLearner.corrupt_checkpoint()
        return prediction

    def start_episode(self):
        assert PolicyLearner.final_started and len(Learner.training_owners) == 192
        session = Session(self)
        PolicyLearner.live_sessions.append(session)
        return session


def environment(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", PolicyLearner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    for name in ("sources", "training_owners", "prediction_owners"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_third", False)
    monkeypatch.setattr(Learner, "mutate_prediction", False)
    monkeypatch.setattr(Learner, "mutation_update", 0)
    monkeypatch.setattr(PolicyLearner, "final_started", False)
    monkeypatch.setattr(PolicyLearner, "cached_updates", [])
    monkeypatch.setattr(PolicyLearner, "live_sessions", [])
    monkeypatch.setattr(PolicyLearner, "failure", None)
    captured: dict[str, Any] = dict(collections=[], streams=[], finished=False)
    collect, finish = shared._collect, AttributionStream.finish

    def collect_packet(identity, layout, cue, role):
        assert role in ("train", "final_test")
        if role == "final_test":
            assert captured["finished"] is True and len(Learner.training_owners) == 192
            PolicyLearner.final_started = True
        else:
            assert not PolicyLearner.final_started
        batch = collect(identity, layout, cue, role)
        captured["collections"].append(deepcopy(batch))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert kwargs["checkpoint_updates"] == pilot.RETURN_POINTS
        assert kwargs["label_delay"] == 1 and len(training) == 192
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured["streams"].append(instance)
        captured["budget"] = kwargs["budget"]
        return instance

    def finished(instance):
        result = finish(instance)
        assert instance.completed_updates == 192
        captured["finished"] = True
        return result

    def corrupt_receipt():
        inbox = captured["streams"][0]._runtime._inbox
        key = next(iter(inbox._applied))
        receipt = inbox._applied[key]
        inbox._applied[key] = replace(
            receipt, diagnostic=replace(receipt.diagnostic, value=receipt.diagnostic.value + 0.125)
        )

    def corrupt_checkpoint():
        state = captured["streams"][0]._states[0][1]
        state.parameters["weight_input"][0, 0] += 1.0

    monkeypatch.setattr(shared, "_collect", collect_packet)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(AttributionStream, "finish", finished)
    monkeypatch.setattr(PolicyLearner, "corrupt_receipt", staticmethod(corrupt_receipt))
    monkeypatch.setattr(PolicyLearner, "corrupt_checkpoint", staticmethod(corrupt_checkpoint))
    return captured


def test_should_describe_explicit_policy_protocol_without_native_or_data_work(monkeypatch):
    original = deepcopy(pilot.describe_return_protocol())
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_return_evaluation_protocol()
    assert pilot.describe_return_protocol() == original
    assert protocol["architecture"] == dict(input_dim=8, hidden_dim=8, num_classes=5, seed=17)
    assert protocol["checkpoint_updates"] == list(pilot.RETURN_POINTS)
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 192
    assert work["train_fit_score_forwards"] == 52 and work["final_cached_forwards"] == 32
    assert work["final_cached_rows"] == 256 and work["live_episodes"] == 32
    assert work["maximum_live_forwards"] == 384
    assert protocol["final_evaluation"]["checkpoint_updates"] == [0, 64, 128, 192]
    assert protocol["final_evaluation"]["task_sequence"] == ["A", "B", "A"]


def test_should_finish_all_training_before_collecting_disjoint_finals_and_evaluating_owned_states(
    monkeypatch,
):
    captured = environment(monkeypatch)
    report = pilot.run_return_evaluation_pilot()
    assert report["status"] == "complete", report["reason"]
    assert captured["finished"] is True
    assert report["charged_updates"] == captured["budget"].updates_completed == 192
    assert captured["budget"].budget.max_training_updates == 192
    assert len(Learner.training_owners) == 192
    assert [batch.role for batch in captured["collections"]] == ["train"] * 4 + ["final_test"] * 8
    training, final = captured["collections"][:4], captured["collections"][4:]
    assert {b.episode.episode_id for b in training}.isdisjoint(b.episode.episode_id for b in final)
    assert {row for batch in training for row in batch.row_ids}.isdisjoint(
        row for batch in final for row in batch.row_ids
    )
    assert all(not np.shares_memory(a.features, b.features) for a in training for b in final)
    assert all(not np.shares_memory(a.targets, b.targets) for a in training for b in final)
    assert PolicyLearner.cached_updates == [point for point in (0, 64, 128, 192) for _ in range(8)]
    assert len(RecurrentLearner.classifications) == 84
    assert (
        len(PolicyLearner.live_sessions) == len({id(s) for s in PolicyLearner.live_sessions}) == 32
    )
    assert [session.state.completed_updates for session in PolicyLearner.live_sessions] == [
        point for point in (0, 64, 128, 192) for _ in range(8)
    ]
    evaluation = report["policy_evaluation"]
    assert evaluation["status"] == "complete" and evaluation["method"] == "recurrent_bc"
    assert evaluation["replay_enabled"] is False
    assert evaluation["live_sessions_started"] == len(evaluation["episodes"]) == 32
    assert [point["updates"] for point in evaluation["checkpoints"]] == [0, 64, 128, 192]
    assert [point["training_context"] for point in evaluation["checkpoints"]] == [
        None,
        "A-blue",
        "B-blue",
        "A-blue",
    ]
    assert all(len(point["scores"]) == 8 for point in evaluation["checkpoints"])
    assert len(evaluation["final_demonstrations"]) == 8
    assert all(episode["role"] == "final_test" for episode in evaluation["final_demonstrations"])
    assert evaluation["work"]["final_collections_attempted"] == 8
    assert evaluation["work"]["final_collections_completed"] == 8
    assert evaluation["work"]["cached_forwards_attempted"] == 32
    assert evaluation["work"]["cached_forwards_completed"] == 32
    assert evaluation["work"]["cached_rows_attempted"] == 256
    assert evaluation["work"]["cached_rows_completed"] == 256
    assert evaluation["work"]["failure_work_may_be_unobserved"] is False
    for key in ("source_unchanged", "isolated_source_unchanged", "actor_unchanged"):
        assert report[key] is True
    assert report["partial_stream"]["native_candidate_updates"] == 192
    assert len(report["partial_stream"]["applied_updates"]) == 192
    assert report["scientific_advantage_verified"] is False


def test_should_keep_all_failed_live_outcomes_and_real_decisions_without_repair_or_retry(
    monkeypatch,
):
    environment(monkeypatch)
    report = pilot.run_return_evaluation_pilot()
    assert report["status"] == "complete", report["reason"]
    evaluation = report["policy_evaluation"]
    for episode, session in zip(evaluation["episodes"], PolicyLearner.live_sessions, strict=True):
        assert episode["status"] == "complete" and episode["outcome"] == "illegal_action"
        assert not episode["success"] and episode["return"] == 0
        assert episode["state_unchanged"] is True
        assert episode["prediction_calls_attempted"] == episode["prediction_calls_returned"]
        assert episode["prediction_calls_returned"] == len(session.inputs)
        assert 1 <= len(session.inputs) <= 12
        assert episode["layout_partition"] == (
            "known_layout" if episode["context"] in ("A", "B") else "heldout_layout"
        )
        assert tuple(episode["identity"]["task_sequence"]) == ("A", "B", "A")
        for transition in episode["transitions"]:
            assert transition["action"] == 0
            assert tuple(transition["probabilities"]) == (1.0, 0.0, 0.0, 0.0, 0.0)
        assert (
            episode["transitions"][-1]["action"] not in episode["transitions"][-1]["legal_actions"]
        )
    assert [point["updates"] for point in evaluation["outcomes"]] == [0, 64, 128, 192]
    for point, cached in zip(evaluation["outcomes"], evaluation["checkpoints"], strict=True):
        assert set(point["by_layout"]) == set("ABCD")
        for layout, summary in point["by_layout"].items():
            assert summary["success_rate"] == summary["mean_return"] == 0
            assert (
                summary["mean_ce"]
                == sum(cached["scores"][f"{layout}-{color}"]["ce"] for color in ("red", "blue")) / 2
            )
            assert (
                summary["mean_accuracy"]
                == sum(
                    cached["scores"][f"{layout}-{color}"]["accuracy"] for color in ("red", "blue")
                )
                / 2
            )
    assert report["context_return"]["incomplete_initial_A_fit"] is True
    assert report["context_return"]["useful_skill_established"] is False


@pytest.mark.parametrize(
    "failure", ["cached_raise", "cached_state", "cached_source", "receipt", "saved_state"]
)
def test_should_keep192_charges_and_skip_later_final_work_after_first_cached_failure(
    monkeypatch, failure
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(PolicyLearner, "failure", failure)
    report = pilot.run_return_evaluation_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["budget"].updates_completed == 192
    assert len(Learner.training_owners) == 192
    assert len(PolicyLearner.cached_updates) == 1
    assert PolicyLearner.live_sessions == []
    evaluation = report["policy_evaluation"]
    assert evaluation["status"] == "failed"
    assert evaluation["checkpoints"] == []
    assert evaluation["work"]["cached_forwards_attempted"] == 1
    assert evaluation["work"]["cached_forwards_completed"] == int(failure != "cached_raise")
    assert evaluation["work"]["cached_rows_attempted"] == 8
    assert evaluation["work"]["cached_rows_completed"] == 8 * int(failure != "cached_raise")
    assert evaluation["work"]["failure_work_may_be_unobserved"] is True
    assert all(slot["status"] == "not_run" for slot in evaluation["episodes"])
    assert evaluation["outcomes"] is None
    assert len(report["checkpoints"]) == 13 and report["stream_result"]["completed_updates"] == 192
    if failure == "cached_source":
        assert report["source_unchanged"] is False


@pytest.mark.parametrize("failure", ["live_raise", "live_state", "live_source"])
def test_should_keep_partial_live_slot_and_unreached_slots_after_first_callback_failure(
    monkeypatch, failure
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(PolicyLearner, "failure", failure)
    report = pilot.run_return_evaluation_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["budget"].updates_completed == 192
    assert len(PolicyLearner.cached_updates) == 32
    assert len(PolicyLearner.live_sessions) == 1 and len(PolicyLearner.live_sessions[0].inputs) == 1
    evaluation = report["policy_evaluation"]
    assert evaluation["status"] == "failed" and evaluation["outcomes"] is None
    assert len(evaluation["checkpoints"]) == 4
    assert evaluation["work"]["live_callback_attempts"] == 1
    assert evaluation["work"]["live_callback_returns"] == 0
    assert evaluation["episodes"][0]["status"] == "failed"
    assert all(slot["status"] == "not_run" for slot in evaluation["episodes"][1:])
    if failure == "live_source":
        assert report["source_unchanged"] is False


def test_should_keep_original_train_return_route_without_final_fields_or_work(monkeypatch):
    captured = environment(monkeypatch)
    report = pilot.run_return_pilot()
    assert report["status"] == "complete", report["reason"]
    assert "policy_evaluation" not in report
    assert all(batch.role == "train" for batch in captured["collections"])
    assert len(RecurrentLearner.classifications) == 52
    assert PolicyLearner.cached_updates == PolicyLearner.live_sessions == []


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_return_policy_cli_without_arguments(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-return-policy"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="return-policy")

    monkeypatch.setattr(cli, "describe_return_policy_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_return_policy_pilot", forbidden if describe else selected)
    for name in (
        "run_train_context_return_pilot",
        "run_train_acquisition_pilot",
        "run_learnability_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["selected"] == "return-policy"
