"""Acquired-task replay callers use the real bounded stream and legal live helper."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.acquired_task_return as acquired
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID
from src.app.coupled_attribution import AttributionStream
from src.app.stream_rehearsal import (
    CoverageStreamReplayPolicy,
    RecentStreamReplayPolicy,
    StreamRehearsal,
)
from src.core.demonstration_episode import EpisodeReplaySchema
from src.core.learner_ports import TrainingDiagnostic
from test_acquired_return_pilot import POINTS
from test_recurrent_key_door_pilot import Learner as RecurrentLearner, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden
from test_train_acquisition_pilot import Learner
from test_train_live_acquisition_pilot import Session


KEYS = ("A-red", "A-blue", "B-red", "B-blue", "C_train-red", "C_train-blue")


class DerivedControl(str):
    pass


class ReplayLearner(Learner):
    work: dict[str, Any] = {}

    def train_batch(self, features, targets):
        assert self.work["finished"] is False, "native training entered post-finish diagnostic"
        diagnostic = super().train_batch(features, targets)
        return TrainingDiagnostic(TORCH_RECURRENT_CE_ID, diagnostic.value)

    def predict(self, features):
        self.work["fit_native_updates"].append(self.state.completed_updates)
        return super().predict(features)

    def start_episode(self):
        assert self.work["finished"] is True and len(RecurrentLearner.trains) == 480
        session = Session(self)
        self.work["sessions"].append(session)
        return session


def environment(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", ReplayLearner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    monkeypatch.setattr(acquired, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    for name in ("sources", "training_owners", "prediction_owners"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_third", False)
    monkeypatch.setattr(Learner, "mutate_prediction", False)
    monkeypatch.setattr(Learner, "mutation_update", 0)
    captured: dict[str, Any] = dict(
        collections=[],
        sessions=[],
        finished=False,
        fit_native_updates=[],
        acquired_calls=[],
        before_diagnostic=None,
        on_prediction=None,
    )
    monkeypatch.setattr(ReplayLearner, "work", captured)
    monkeypatch.setattr(Session, "work", captured)
    original_collect, original_diagnostic = shared._collect, acquired.evaluate_acquired_tasks

    def collect(identity, layout, cue, role):
        assert role == "train" and captured["finished"] is False
        batch = original_collect(identity, layout, cue, role)
        captured["collections"].append(batch)
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert len(training) == 320 and kwargs["checkpoint_updates"] == POINTS
        assert kwargs["label_delay"] == 1
        helper = kwargs["rehearsal"]
        assert type(helper) is StreamRehearsal and helper.episode_schema == EpisodeReplaySchema(
            8, 8, 5
        )
        captured.update(
            training=training,
            isolated=learner,
            helper=helper,
            wake=kwargs["budget"],
            replay_budget=helper._replay_budget,
        )
        captured["budget_bindings"] = [
            (id(budget), id(budget.budget), id(budget.clock), budget.started_at)
            for budget in (captured["wake"], captured["replay_budget"])
        ]
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured["stream"] = instance
        original_finish = instance.finish

        def finish():
            result = original_finish()
            assert instance.completed_updates == 320 and len(RecurrentLearner.trains) == 480
            captured["finished"] = True
            return result

        monkeypatch.setattr(instance, "finish", finish)
        return instance

    def evaluate(report, source, states, budget, **kwargs):
        assert captured["finished"] is True and len(RecurrentLearner.trains) == 480
        assert [point for point, _ in states] == [192, 256, 320]
        assert kwargs["replay_control"] in ("recent48", "coverage")
        captured["acquired_calls"].append((report, source, states, budget))
        if captured["before_diagnostic"] is not None:
            captured["before_diagnostic"]()
        return original_diagnostic(report, source, states, budget, **kwargs)

    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(acquired, "evaluate_acquired_tasks", evaluate)
    return captured


def clocks(wake):
    return dict(
        wake_updates=wake,
        replay_committed_updates=wake // 2,
        replay_charged_examples=4 * wake,
        native_updates=wake + wake // 2,
        native_training_examples=8 * (wake + wake // 2),
    )


@pytest.mark.parametrize("control", ["recent48", "coverage"])
def test_should_keep_equal_replay_costs_release_provenance_and_owned_live_clocks(
    monkeypatch, control
):
    captured = environment(monkeypatch)
    global_rng = pickle.dumps(np.random.get_state())
    report = pilot.run_acquired_replay_pilot(control=control)
    assert report["status"] == "complete", report
    helper, wake, replay_budget = captured["helper"], captured["wake"], captured["replay_budget"]
    assert report["replay_control"] == control
    assert wake is not replay_budget and helper._replay_budget is replay_budget
    assert (
        wake.updates_completed
        == wake.budget.max_training_updates
        == report["charged_updates"]
        == 320
    )
    assert wake.replay_examples_completed == replay_budget.updates_completed == 0
    assert (
        replay_budget.replay_examples_completed == replay_budget.budget.max_replay_examples == 1280
    )
    assert captured["budget_bindings"] == [
        (id(budget), id(budget.budget), id(budget.clock), budget.started_at)
        for budget in (wake, replay_budget)
    ]
    assert len(RecurrentLearner.trains) == len(Learner.training_owners) == 480
    assert len(captured["collections"]) == 6 and all(
        batch.role == "train" for batch in captured["collections"]
    )
    training = captured["training"]
    assert len(training) == len({batch.episode.episode_id for batch in training}) == 320
    assert [batch.context for batch in training] == [
        context
        for layout in ("A", "B", "A", "C_train", "A")
        for context in [f"{layout}-red", f"{layout}-blue"] * 32
    ]
    assert all(batch.role == batch.episode.role == "train" for batch in training)
    policy = helper._policy
    assert policy.every_updates == 2 and policy.rows_per_event == 8
    assert policy.retention.max_examples == 48 and policy.retention.max_bytes == 12096
    assert type(policy) is (
        CoverageStreamReplayPolicy if control == "coverage" else RecentStreamReplayPolicy
    )
    if control == "coverage":
        assert isinstance(policy, CoverageStreamReplayPolicy)
        assert policy.contexts == KEYS
    replay = report["replay"]
    assert replay["charged_examples"] == replay["rehearsal"]["charged_exposures"] == 1280
    assert replay["final_retention"]["example_count"] <= 48
    assert replay["final_retention"]["retained_bytes"] <= 12096
    events, receipts = replay["rehearsal"]["events"], replay["applied_consolidations"]
    assert len(events) == len(receipts) == replay["rehearsal"]["committed_updates"] == 160
    assert [event["updates"] for event in events] == list(range(2, 321, 2))
    sources = {
        batch.episode.episode_id: (index + 1, asdict(batch.episode))
        for index, batch in enumerate(training)
    }
    for number, (event, receipt) in enumerate(zip(events, receipts, strict=True), 1):
        assert event["status"] == "complete" and event["receipt"] == receipt
        assert event["charged_exposures"] == 8
        assert (
            event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 1
        )
        assert (
            receipt["attempt_number"] == number
            and receipt["diagnostic"]["definition"] == TORCH_RECURRENT_CE_ID
        )
        assert len(event["selected_episodes"]) == 1
        episode = event["selected_episodes"][0]
        release, original = sources[episode["episode_id"]]
        assert release <= event["updates"] and episode == original
        assert episode["role"] == "train" and len(episode["transitions"]) == 8
        assert (
            event["retention"]["example_count"] <= 48
            and event["retention"]["retained_bytes"] <= 12096
        )
        if control == "coverage":
            assert ("C_train-red" in event["coverage_keys"]) is (event["updates"] >= 194)
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS)
    assert [point["clocks"] for point in report["checkpoints"]] == [
        clocks(point) for point in POINTS
    ]
    assert captured["fit_native_updates"] == [
        point + point // 2 for point in POINTS for _ in range(6)
    ]
    assert report["stream_result"]["completed_updates"] == 320
    assert report["stream_result"]["training_examples"] == 2560
    assert len(report["stream_result"]["applied_updates"]) == 320
    assert report["partial_stream"]["native_candidate_updates"] == 480
    assert report["partial_stream"]["native_candidate_training_examples"] == 3840
    assert len(captured["sessions"]) == 12 and len(captured["acquired_calls"]) == 1
    assert [session.state.completed_updates for session in captured["sessions"]] == [288] * 4 + [
        384
    ] * 4 + [480] * 4
    diagnostic = report["acquired_return_diagnostics"]
    assert diagnostic["status"] == "complete" and diagnostic["replay_enabled"] is True
    assert diagnostic["replay_control"] == control
    assert diagnostic["checkpoint_clocks"] == [clocks(point) for point in (192, 256, 320)]
    assert all(
        slot["status"] == "complete" and slot["success"] is False for slot in diagnostic["episodes"]
    )
    assert (
        report["acquired_initial_condition_met"]
        is report["acquired_new_task_condition_met"]
        is False
    )
    assert all(
        field not in report
        for field in (
            "policy_evaluation",
            "probes",
            "training_objective",
            "train_policy_diagnostics",
        )
    )
    assert "reference" not in diagnostic and report["scientific_advantage_verified"] is False
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert pickle.dumps(np.random.get_state()) == global_rng
    # A returned episode record must not retain writable aliases into the live helper.
    saved = deepcopy(helper.report)
    episode = events[0]["selected_episodes"][0]
    episode["transitions"][0]["action"] = (episode["transitions"][0]["action"] + 1) % 5
    assert helper.report == saved


@pytest.mark.parametrize("control", ["recent48", "coverage"])
def test_should_charge_failed_first_replay_and_keep_two_wake_commits_without_live(
    monkeypatch, control
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_third", True)
    report = pilot.run_acquired_replay_pilot(control=control)
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
    assert event["receipt"] is None and event["attempted_updates"] == 1
    assert event["returned_updates"] == event["committed_updates"] == 0
    assert report["replay"]["applied_consolidations"] == []
    assert report["partial_stream"]["native_candidate_updates"] == 2
    assert (
        captured["sessions"] == captured["acquired_calls"] == [] and captured["finished"] is False
    )


@pytest.mark.parametrize(
    "fault", ["consolidation", "wake_receipt", "selected_episode", "replay_budget", "checkpoint"]
)
def test_should_refuse_changed_completed_replay_authority_before_any_live_prediction(
    monkeypatch, fault
):
    captured = environment(monkeypatch)

    def corrupt():
        stream, helper = captured["stream"], captured["helper"]
        if fault == "consolidation":
            original = stream._runtime._consolidations[0]
            stream._runtime._consolidations[0] = replace(
                original,
                diagnostic=replace(original.diagnostic, value=original.diagnostic.value + 0.125),
            )
        elif fault == "wake_receipt":
            diagnostic = stream._runtime.applied_updates[0].diagnostic
            object.__setattr__(diagnostic, "value", diagnostic.value + 0.125)
        elif fault == "selected_episode":
            transition = helper._events[0]["selected_episodes"][0]["transitions"][0]
            transition["action"] = (transition["action"] + 1) % 5
        elif fault == "replay_budget":
            captured["replay_budget"].replay_examples_completed -= 8
        else:
            state = next(state for count, state in stream._states if count == 192)
            state.parameters["weight_input"][0, 0] += 1

    captured["before_diagnostic"] = corrupt
    report = pilot.run_acquired_replay_pilot(control="coverage")
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["wake"].updates_completed == 320
    assert len(RecurrentLearner.trains) == 480 and captured["finished"] is True
    assert captured["sessions"] == []
    assert report["acquired_return_diagnostics"]["status"] == "failed"
    assert all(
        slot["status"] == "not_run" for slot in report["acquired_return_diagnostics"]["episodes"]
    )
    if fault != "replay_budget":
        assert captured["replay_budget"].replay_examples_completed == 1280


@pytest.mark.parametrize(
    "phase,target", [("score", "control"), ("live", "control"), ("live", "pending_claim")]
)
def test_should_bind_outer_control_and_pending_claim_before_more_work(monkeypatch, phase, target):
    captured = environment(monkeypatch)
    outer = []

    def construct(*args, **kwargs):
        result = dict(*args, **kwargs)
        if {"protocol", "charged_updates", "stream_result", "checkpoints"} <= result.keys():
            outer.append(result)
            # Capture its ordinary construction, then restore builtins before
            # any data, learner or runtime is created.
            monkeypatch.delattr(pilot, "dict")
        return result

    monkeypatch.setattr(pilot, "dict", construct, raising=False)
    changed: list[bool] = []

    def corrupt():
        assert not changed
        if target == "control":
            outer[0]["replay_control"] = "recent48"
        else:
            assert outer[0]["acquired_new_task_condition_met"] is None
            outer[0]["acquired_new_task_condition_met"] = True
        changed.append(True)

    if phase == "score":
        original = ReplayLearner.predict

        def predict(owner, features):
            result = original(owner, features)
            if owner.state.completed_updates == 12 and not changed:
                corrupt()
            return result

        monkeypatch.setattr(ReplayLearner, "predict", predict)
    else:
        captured["on_prediction"] = lambda session: corrupt()
    report = pilot.run_acquired_replay_pilot(control="coverage")
    assert report is outer[0] and report["status"] == "incomplete" and changed == [True]
    wake = 8 if phase == "score" else 320
    assert report["charged_updates"] == captured["wake"].updates_completed == wake
    assert len(RecurrentLearner.trains) == wake + wake // 2
    assert captured["replay_budget"].replay_examples_completed == 4 * wake
    assert len(captured["sessions"]) == int(phase == "live")
    if phase == "live":
        assert len(captured["sessions"][0].inputs) == 1
        assert report["acquired_return_diagnostics"]["status"] == "failed"
    else:
        assert captured["finished"] is False and captured["acquired_calls"] == []


def test_should_keep_default_acquired_protocol_and_helper_schema_without_replay(monkeypatch):
    old, diagnostic = deepcopy(pilot.describe_acquired_return_protocol()), acquired.new_diagnostic()
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
        "StreamRehearsal",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    for control in ("recent48", "coverage"):
        protocol = pilot.describe_acquired_replay_protocol(control=control)
        assert protocol["replay_control"] == control
        policy = protocol["replay"]["policy"]
        assert policy["retention"] == dict(max_examples=48, max_bytes=12096)
        assert policy["every_updates"] == 2 and policy["rows_per_event"] == 8
        assert protocol["planned_work"]["native_train_calls"] == 480
        assert protocol["planned_work"]["training_rows"] == 3840
        assert protocol["planned_work"]["score_forwards"] == 126
    assert pilot.describe_acquired_return_protocol() == old
    assert acquired.new_diagnostic() == diagnostic and diagnostic["replay_enabled"] is False
    assert "replay_control" not in diagnostic and "checkpoint_clocks" not in diagnostic


@pytest.mark.parametrize(
    "entry", ["describe_acquired_replay_protocol", "run_acquired_replay_pilot"]
)
@pytest.mark.parametrize(
    "invalid", [None, 0, "unknown", "recent", "partitioned48", DerivedControl("coverage")]
)
def test_should_refuse_every_undeclared_or_non_exact_control_before_resources(
    monkeypatch, entry, invalid
):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, entry)(control=invalid)


@pytest.mark.parametrize(
    "mix",
    [
        dict(acquired_return=True),
        dict(return_context=True),
        dict(evaluate_policy_after_training=True),
        dict(episode_replay=True),
        dict(fresh_task_probes=True),
        dict(four_phase=True),
        dict(four_phase_policy=True),
        dict(four_phase_replay="recent48"),
        dict(four_phase_probes=True),
        dict(train_policy_diagnostics=True),
        dict(weighted_train=True),
        dict(fixed_a_exposure=True),
    ],
)
def test_should_refuse_public_mixed_modes_before_normalizing_internal_replay_options(
    monkeypatch, mix
):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(acquired_replay="coverage", **mix)


@pytest.mark.parametrize("control", ["recent48", "coverage"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_acquired_replay_cli(monkeypatch, capsys, control, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", f"torch-key-door-acquired-{control}-replay"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected(*, control):
        calls.append(control)
        return dict(status="complete", selected=control)

    monkeypatch.setattr(
        cli, "describe_acquired_replay_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_acquired_replay_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_acquired_task_return_pilot", forbidden)
    cli.main()
    assert calls == [control] and json.loads(capsys.readouterr().out)["selected"] == control
