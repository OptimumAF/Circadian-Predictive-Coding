"""Five post-training key/door policy checkpoints on owned fake native learners."""

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


FINAL_POINTS = (0, 64, 128, 192, 256)


class Session:
    def __init__(self, source):
        self.state = source.snapshot_state()
        self.completed_steps = 0
        self.inputs: list[Any] = []

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert PolicyLearner.final_started and len(Learner.training_owners) == 256
        assert features.shape == (1, 8)
        self.inputs.append(features.copy())
        if PolicyLearner.failure == "live_raise":
            raise RuntimeError("first four-phase live prediction failed")
        self.completed_steps += 1
        if PolicyLearner.failure == "live_state":
            self.state.parameters["weight_input"][0, 0] += 0.125
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)


class PolicyLearner(Learner):
    final_started = False
    cached_native_updates: list[int] = []
    live_sessions: list[Session] = []
    failure: str | None = None
    corrupt: Any = None

    def train_batch(self, features, targets):
        assert not PolicyLearner.final_started, "final packets reached training"
        return super().train_batch(features, targets)

    def predict(self, features):
        if PolicyLearner.final_started:
            assert len(Learner.training_owners) == 256
            assert self.state.completed_updates in FINAL_POINTS
            PolicyLearner.cached_native_updates.append(self.state.completed_updates)
            if PolicyLearner.failure == "cached_raise":
                raise RuntimeError("first four-phase cached prediction failed")
        result = super().predict(features)
        if PolicyLearner.final_started:
            if PolicyLearner.failure == "cached_state":
                self.state.gradients["weight_input"] = np.full_like(
                    self.state.parameters["weight_input"], 0.125
                )
            elif PolicyLearner.corrupt is not None:
                PolicyLearner.corrupt()
        return result

    def start_episode(self):
        assert PolicyLearner.final_started and len(Learner.training_owners) == 256
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
    monkeypatch.setattr(PolicyLearner, "cached_native_updates", [])
    monkeypatch.setattr(PolicyLearner, "live_sessions", [])
    monkeypatch.setattr(PolicyLearner, "failure", None)
    monkeypatch.setattr(PolicyLearner, "corrupt", None)
    captured: dict[str, Any] = dict(collections=[], finished=False)
    collect, finish = shared._collect, AttributionStream.finish

    def collect_packet(identity, layout, cue, role):
        assert role in ("train", "final_test")
        if role == "final_test":
            assert captured["finished"] is True and len(Learner.training_owners) == 256
            PolicyLearner.final_started = True
        else:
            assert not PolicyLearner.final_started
        batch = collect(identity, layout, cue, role)
        captured["collections"].append((deepcopy(layout), deepcopy(batch)))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True and len(training) == 256
        assert kwargs["checkpoint_updates"] == pilot.FOUR_PHASE_POINTS
        assert kwargs.get("rehearsal") is None
        captured.update(budget=kwargs["budget"], isolated=learner, training=deepcopy(training))
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured["stream"] = instance
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


def test_should_describe_five_fixed_policy_checkpoints_without_resources_or_changing_train_protocol(
    monkeypatch,
):
    before = deepcopy(pilot.describe_four_phase_protocol())
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_four_phase_evaluation_protocol()
    assert pilot.describe_four_phase_protocol() == before
    assert protocol["architecture"] == before["architecture"]
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 256 and work["training_rows"] == 2048
    assert work["final_cached_forwards"] == 40 and work["final_cached_rows"] == 320
    assert work["live_episodes"] == 40 and work["maximum_live_forwards"] == 480


@pytest.mark.parametrize(
    "flag",
    [
        "four_phase",
        "return_context",
        "evaluate_policy_after_training",
        "episode_replay",
        "fresh_task_probes",
    ],
)
def test_should_refuse_four_phase_policy_mixed_modes_before_resources(monkeypatch, flag):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError, match="four"):
        pilot._run_pilot(four_phase_policy=True, **{flag: True})


def test_should_finish256_train_commits_before_disjoint_final_data_and_all_five_owned_policy_states(
    monkeypatch,
):
    captured = environment(monkeypatch)
    historical_layouts = deepcopy(shared.layouts())
    report = pilot.run_four_phase_evaluation_pilot()
    assert report["status"] == "complete", report["reason"]
    assert shared.layouts() == historical_layouts
    assert captured["finished"] is True
    assert [batch.role for _, batch in captured["collections"]] == ["train"] * 6 + [
        "final_test"
    ] * 8
    train = captured["training"]
    final = captured["collections"][6:]
    assert {batch.episode.episode_id for batch in train}.isdisjoint(
        batch.episode.episode_id for _, batch in final
    )
    assert all(not np.shares_memory(a.features, b.features) for a in train for _, b in final)
    assert all(not np.shares_memory(a.targets, b.targets) for a in train for _, b in final)
    assert {layout.layout_id for layout, _ in final} == {"A", "B", "C_train", "D"}
    assert report["charged_updates"] == captured["budget"].updates_completed == 256
    assert captured["budget"].budget.max_training_updates == 256
    assert len(Learner.training_owners) == 256 and len(RecurrentLearner.classifications) == 142
    assert PolicyLearner.cached_native_updates == [
        point for point in FINAL_POINTS for _ in range(8)
    ]
    assert (
        len(PolicyLearner.live_sessions) == len({id(s) for s in PolicyLearner.live_sessions}) == 40
    )
    assert [s.state.completed_updates for s in PolicyLearner.live_sessions] == [
        point for point in FINAL_POINTS for _ in range(8)
    ]
    assert [p["updates"] for p in report["checkpoints"]] == list(pilot.FOUR_PHASE_POINTS)
    assert report["stream_result"]["completed_updates"] == 256
    assert report["stream_result"]["training_examples"] == 2048
    assert (
        "context_return" not in report
        and report["four_phase_return"]["useful_skill_established"] is False
    )
    evaluation = report["policy_evaluation"]
    assert evaluation["status"] == "complete" and evaluation["live_sessions_started"] == 40
    assert [p["updates"] for p in evaluation["checkpoints"]] == list(FINAL_POINTS)
    assert len(evaluation["final_demonstrations"]) == 8
    assert all(len(p["scores"]) == 8 for p in evaluation["checkpoints"])
    assert [p["updates"] for p in evaluation["outcomes"]] == list(FINAL_POINTS)
    for point, cached in zip(evaluation["outcomes"], evaluation["checkpoints"], strict=True):
        assert set(point["by_layout"]) == {"A", "B", "C_train", "D"}
        for layout, outcome in point["by_layout"].items():
            assert outcome["success_rate"] == outcome["mean_return"] == 0
            assert (
                outcome["mean_ce"]
                == sum(cached["scores"][f"{layout}-{cue}"]["ce"] for cue in ("red", "blue")) / 2
            )
    assert evaluation["work"]["final_collections_attempted"] == 8
    assert evaluation["work"]["cached_forwards_completed"] == 40
    assert evaluation["work"]["cached_rows_completed"] == 320
    for episode, session in zip(evaluation["episodes"], PolicyLearner.live_sessions, strict=True):
        assert episode["status"] == "complete" and episode["outcome"] == "illegal_action"
        assert not episode["success"] and episode["return"] == 0
        assert episode["state_unchanged"] is True
        assert tuple(episode["identity"]["task_sequence"]) == ("A", "B", "C_train", "A")
        assert episode["layout_partition"] == (
            "heldout_layout" if episode["context"] == "D" else "known_layout"
        )
        assert (
            episode["prediction_calls_attempted"]
            == episode["prediction_calls_returned"]
            == len(session.inputs)
        )
        assert all(transition["action"] == 0 for transition in episode["transitions"])
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert report["scientific_advantage_verified"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "cached_raise",
        "cached_state",
        "source",
        "candidate_optimizer",
        "receipt",
        "saved_checkpoint",
    ],
)
def test_should_preserve256_charges_and_refuse_further_final_work_after_cached_failure(
    monkeypatch, fault
):
    captured = environment(monkeypatch)
    changed: list[bool] = []

    def corrupt():
        assert not changed
        changed.append(True)
        stream = captured["stream"]
        if fault == "source":
            Learner.sources[0].state.parameters["weight_input"][0, 0] += 0.125
        elif fault == "candidate_optimizer":
            stream._runtime._candidate.state.optimizer_state["weight_input"][0, 0] += 0.125
        elif fault == "saved_checkpoint":
            stream._states[0][1].parameters["weight_input"][0, 0] += 0.125
        else:
            inbox = stream._runtime._inbox
            key = next(iter(inbox._applied))
            receipt = inbox._applied[key]
            inbox._applied[key] = replace(
                receipt,
                diagnostic=replace(receipt.diagnostic, value=receipt.diagnostic.value + 0.125),
            )

    if fault in ("cached_raise", "cached_state"):
        monkeypatch.setattr(PolicyLearner, "failure", fault)
    else:
        monkeypatch.setattr(PolicyLearner, "corrupt", staticmethod(corrupt))
    report = pilot.run_four_phase_evaluation_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["budget"].updates_completed == 256
    assert len(Learner.training_owners) == 256
    assert len(PolicyLearner.cached_native_updates) == 1 and PolicyLearner.live_sessions == []
    evaluation = report["policy_evaluation"]
    assert evaluation["status"] == "failed" and evaluation["checkpoints"] == []
    assert evaluation["outcomes"] is None
    assert all(slot["status"] == "not_run" for slot in evaluation["episodes"])
    assert evaluation["work"]["cached_forwards_attempted"] == 1
    assert evaluation["work"]["cached_forwards_completed"] == int(fault != "cached_raise")
    assert evaluation["work"]["cached_rows_completed"] == 8 * int(fault != "cached_raise")
    assert len(report["checkpoints"]) == 17 and report["stream_result"]["completed_updates"] == 256


@pytest.mark.parametrize("fault", ["live_raise", "live_state"])
def test_should_keep_failed_first_live_slot_and_unreached_later_slots_without_retry(
    monkeypatch, fault
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(PolicyLearner, "failure", fault)
    report = pilot.run_four_phase_evaluation_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["budget"].updates_completed == 256
    assert len(PolicyLearner.cached_native_updates) == 40
    assert len(PolicyLearner.live_sessions) == 1 and len(PolicyLearner.live_sessions[0].inputs) == 1
    evaluation = report["policy_evaluation"]
    assert len(evaluation["checkpoints"]) == 5 and evaluation["outcomes"] is None
    assert evaluation["episodes"][0]["status"] == "failed"
    assert all(slot["status"] == "not_run" for slot in evaluation["episodes"][1:])
    assert evaluation["work"]["live_callback_attempts"] == 1
    assert evaluation["work"]["live_callback_returns"] == 0


def test_should_preserve_default_four_phase_train_route_without_final_data_or_policy_field(
    monkeypatch,
):
    captured = environment(monkeypatch)
    report = pilot.run_four_phase_pilot()
    assert report["status"] == "complete", report["reason"]
    assert "policy_evaluation" not in report
    assert len(RecurrentLearner.classifications) == 102
    assert all(batch.role == "train" for _, batch in captured["collections"])
    assert PolicyLearner.cached_native_updates == PolicyLearner.live_sessions == []


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_four_phase_policy_cli_without_arguments(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-four-phase-policy"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="four-phase-policy")

    monkeypatch.setattr(
        cli, "describe_four_phase_evaluation_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_four_phase_evaluation_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_four_phase_pilot", forbidden)
    monkeypatch.setattr(cli, "run_return_policy_pilot", forbidden)
    cli.main()
    assert (
        calls == [True] and json.loads(capsys.readouterr().out)["selected"] == "four-phase-policy"
    )
