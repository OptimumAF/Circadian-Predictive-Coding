"""Fixed A128 exposure through the real delayed stream and train-only live helper."""

from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
import src.adapters.train_policy_diagnostics as train_live
from src.app.coupled_attribution import AttributionStream
from test_recurrent_key_door_pilot import Learner as RecurrentLearner, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden
from test_train_acquisition_pilot import Learner
from test_train_live_acquisition_pilot import Session


POINTS = (0, 8, 16, 32, 64, 72, 80, 96, 128)


def environment(monkeypatch, *, updates=128, points=POINTS):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", Learner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    monkeypatch.setattr(train_live, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    for name in ("sources", "training_owners", "prediction_owners"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_third", False)
    monkeypatch.setattr(Learner, "mutate_prediction", False)
    monkeypatch.setattr(Learner, "mutation_update", 0)
    captured: dict[str, Any] = dict(
        collections=[], streams=[], sessions=[], finished=[], diagnostics=[], on_prediction=None
    )
    original_collect, original_diagnostic = shared._collect, train_live.evaluate_train_policy

    def collect(identity, layout, cue, role):
        assert role == "train", "fixed exposure constructed reserved/final data"
        result = original_collect(identity, layout, cue, role)
        captured["collections"].append(result)
        return result

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert kwargs["checkpoint_updates"] == points and kwargs["label_delay"] == 1
        assert kwargs.get("rehearsal") is kwargs.get("renewal") is kwargs.get("capacity") is None
        assert kwargs["budget"].budget.max_training_updates == updates
        result = AttributionStream(learner, training, heldout, **kwargs)
        captured["streams"].append((result, learner, training, kwargs["budget"]))
        original_finish = result.finish

        def finish():
            returned = original_finish()
            captured["finished"].append(deepcopy(returned))
            return returned

        monkeypatch.setattr(result, "finish", finish)
        return result

    def start_episode(owner):
        assert len(RecurrentLearner.trains) == updates
        assert len(captured["finished"]) == 1, "live readout began before fixed stream.finish"
        session = Session(owner)
        captured["sessions"].append(session)
        return session

    def diagnostic(report, source, states, reference, budget, **kwargs):
        assert len(captured["finished"]) == 1 and len(RecurrentLearner.trains) == updates
        assert [point for point, _ in states] == [0, updates]
        assert kwargs.get("endpoint", 64) == updates
        captured["diagnostics"].append((report, source, states, reference, budget, kwargs))
        return original_diagnostic(report, source, states, reference, budget, **kwargs)

    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(Learner, "start_episode", start_episode)
    monkeypatch.setattr(Session, "work", captured)
    monkeypatch.setattr(train_live, "evaluate_train_policy", diagnostic)
    return captured


def test_should_keep_all128_releases_A_only_and_run_owned_live_endpoints_after_finish(monkeypatch):
    captured = environment(monkeypatch)
    global_rng = pickle.dumps(np.random.get_state())
    report = pilot.run_fixed_exposure_pilot()
    assert report["status"] == "complete", report
    assert len(captured["collections"]) == 2 and len(captured["streams"]) == 1
    stream, isolated, training, budget = captured["streams"][0]
    assert (
        report["charged_updates"] == budget.updates_completed == len(RecurrentLearner.trains) == 128
    )
    assert budget.budget.max_training_updates == 128 and budget.replay_examples_completed == 0
    assert len(training) == 128
    assert [batch.context for batch in training] == ["A-red", "A-blue"] * 64
    assert [batch.episode.episode_id for batch in training] == [
        f"train-fit-{index}" for index in range(128)
    ]
    assert len({row for batch in training for row in batch.row_ids}) == 1024
    for index, batch in enumerate(training):
        original = captured["collections"][index % 2]
        assert batch.role == batch.episode.role == "train" and len(batch.features) == 8
        np.testing.assert_array_equal(batch.features, original.features)
        np.testing.assert_array_equal(batch.targets, original.targets)
        assert not np.shares_memory(batch.features, original.features)
        assert not np.shares_memory(batch.targets, original.targets)
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS)
    assert len(RecurrentLearner.classifications) == 18
    assert report["work"] == dict(
        score_forwards_attempted=18,
        score_forwards_completed=18,
        score_rows_attempted=144,
        score_rows_completed=144,
    )
    result = report["stream_result"]
    assert result["completed_updates"] == 128 and result["training_examples"] == 1024
    assert result["score_forward_calls"] == 0 and len(result["applied_updates"]) == 128
    assert [receipt["update_number"] for receipt in result["applied_updates"]] == list(
        range(1, 129)
    )
    summary = report["training_summary"]
    assert summary["episodes"] == len(summary["episode_ids"]) == 128
    assert summary["unique_source_episodes"] == 2 and summary["collected_source_records"] == 16
    assert summary["training_exposures"] == 1024 and summary["roles"] == ["train", "train"]
    assert summary["encoded_training_array_bytes"] == 128 * 8 * 9 * 8
    assert len(captured["finished"]) == len(captured["diagnostics"]) == 1
    assert captured["diagnostics"][0][4] is budget
    assert captured["diagnostics"][0][5]["endpoint"] == 128
    assert len(captured["sessions"]) == 4
    assert [session.state.completed_updates for session in captured["sessions"]] == [0, 0, 128, 128]
    for session in captured["sessions"]:
        assert session.owner is not Learner.sources[0] and session.owner is not isolated
        assert session.owner is not stream._runtime._candidate
        assert len(session.inputs) == session.completed_steps == 12
    diagnostic = report["train_policy_diagnostics"]
    assert [slot["checkpoint_updates"] for slot in diagnostic["episodes"]] == [0, 0, 128, 128]
    assert [slot["cue"] for slot in diagnostic["episodes"]] == [1, -1, 1, -1]
    assert all(
        slot["status"] == "complete" and slot["success"] is False for slot in diagnostic["episodes"]
    )
    assert diagnostic["outcomes"] == [
        dict(updates=0, success_rate=0.0, mean_return=0.0),
        dict(updates=128, success_rate=0.0, mean_return=0.0),
    ]
    assert diagnostic["learner_gains"] == dict(success_rate=0.0, mean_return=0.0)
    prior = diagnostic["reference"]
    assert (
        prior["probabilities"]
        == report["majority_control"]["probabilities"]
        == [0.5, 0.125, 0, 0.125, 0.25]
    )
    assert len(prior["episodes"]) == 2 and all(
        slot["status"] == "refused" for slot in prior["episodes"]
    )
    assert all(slot["success"] is None for slot in prior["episodes"])
    assert prior["outcomes"] is diagnostic["comparison"] is None
    assert diagnostic["work"]["learner_callback_attempts"] == 48
    assert diagnostic["work"]["prior_callback_attempts"] == 8
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert "policy_evaluation" not in report and "probes" not in report and "replay" not in report
    assert "training_objective" not in report and "context_return" not in report
    assert report["scientific_advantage_verified"] is False
    assert pickle.dumps(np.random.get_state()) == global_rng


def test_should_keep_original64_charges_when_first_added_update_fails(monkeypatch):
    captured = environment(monkeypatch)
    original, attempts = Learner.train_batch, []

    def train(owner, features, targets):
        attempts.append(owner.state.completed_updates)
        if owner.state.completed_updates == 64:
            raise RuntimeError("first added A-exposure update failed")
        return original(owner, features, targets)

    monkeypatch.setattr(Learner, "train_batch", train)
    report = pilot.run_fixed_exposure_pilot()
    assert report["status"] == "incomplete" and "first added A-exposure" in report["reason"]
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 64
    assert attempts == list(range(65))
    assert captured["streams"][0][3].budget.max_training_updates == 128
    assert report["partial_stream"]["native_candidate_updates"] == 64
    assert report["partial_stream"]["native_candidate_training_examples"] == 512
    assert len(report["partial_stream"]["applied_updates"]) == 64
    assert [point["updates"] for point in report["checkpoints"]] == [0, 8, 16, 32, 64]
    assert captured["finished"] == captured["sessions"] == captured["diagnostics"] == []


@pytest.mark.parametrize("target", ["source", "scorer"])
def test_should_validate_preceding64_state_before_more_A_training(monkeypatch, target):
    captured = environment(monkeypatch)
    original = Learner.predict
    changed: list[bool] = []

    def predict(owner, features):
        result = original(owner, features)
        if owner.state.completed_updates == 64 and not changed:
            selected = Learner.sources[0] if target == "source" else owner
            selected.state.parameters["weight_input"][0, 0] += 1
            changed.append(True)
        return result

    monkeypatch.setattr(Learner, "predict", predict)
    report = pilot.run_fixed_exposure_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 64
    assert report["partial_stream"]["native_candidate_updates"] == 64
    assert captured["finished"] == captured["sessions"] == captured["diagnostics"] == []


@pytest.mark.parametrize("target", ["source", "receipt", "checkpoint", "prior"])
def test_should_bind_completed128_authority_during_live_diagnostic(monkeypatch, target):
    captured = environment(monkeypatch)

    def corrupt(session):
        stream = captured["streams"][0][0]
        if target == "source":
            Learner.sources[0].state.parameters["weight_input"][0, 0] += 1
        elif target == "receipt":
            diagnostic = stream._runtime.applied_updates[-1].diagnostic
            object.__setattr__(diagnostic, "value", diagnostic.value + 1)
        elif target == "checkpoint":
            stream._states[-1][1].parameters["weight_input"][0, 0] += 1
        else:
            captured["diagnostics"][0][3]["probabilities"][0] -= 0.125
            captured["diagnostics"][0][3]["probabilities"][2] += 0.125

    captured["on_prediction"] = corrupt
    report = pilot.run_fixed_exposure_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 128
    assert len(report["stream_result"]["applied_updates"]) == 128
    assert len(captured["sessions"]) == 1 and len(captured["sessions"][0].inputs) == 1
    diagnostic = report["train_policy_diagnostics"]
    assert diagnostic["status"] == "failed"
    assert all(slot["status"] == "not_run" for slot in diagnostic["episodes"][1:])
    assert all(slot["status"] == "not_run" for slot in diagnostic["reference"]["episodes"])


@pytest.mark.parametrize("route", ["mean64", "return192"])
def test_should_preserve_original64_and_return192_recipe_endpoints(monkeypatch, route):
    updates = 64 if route == "mean64" else 192
    points = (0, 8, 16, 32, 64) if route == "mean64" else pilot.RETURN_POINTS
    captured = environment(monkeypatch, updates=updates, points=points)
    report = pilot.run_live_pilot() if route == "mean64" else pilot.run_return_pilot()
    assert report["status"] == "complete", report["reason"]
    assert report["charged_updates"] == len(RecurrentLearner.trains) == updates
    assert [point["updates"] for point in report["checkpoints"]] == list(points)
    if route == "mean64":
        assert [
            slot["checkpoint_updates"] for slot in report["train_policy_diagnostics"]["episodes"]
        ] == [0, 0, 64, 64]
        assert "endpoint" not in captured["diagnostics"][0][5]
    else:
        assert len(captured["collections"]) == 4 and captured["sessions"] == []
        assert "context_return" in report and "train_policy_diagnostics" not in report
        assert [batch.context for batch in captured["streams"][0][2]] == [
            "A-red",
            "A-blue",
        ] * 32 + ["B-red", "B-blue"] * 32 + ["A-red", "A-blue"] * 32


def test_should_describe_fixed128_cost_and_keep_old_descriptors_without_resources(monkeypatch):
    old, returned = (
        deepcopy(pilot.describe_live_protocol()),
        deepcopy(pilot.describe_return_protocol()),
    )
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_fixed_exposure_protocol()
    assert protocol["protocol_id"] == "key_door_train_only_fixed_a_exposure_v1"
    assert pilot.FIXED_A_UPDATES == 128 and pilot.FIXED_A_POINTS == POINTS
    assert protocol["checkpoint_updates"] == list(POINTS)
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 128 and work["training_rows"] == 1024
    assert work["score_forwards"] == 18 and work["score_rows"] == 144
    assert work["recurrent_unroll_steps"] == 1216
    assert work["maximum_native_predict_calls"] == 66
    assert work["environment_constructions"] == 8 and work["maximum_environment_steps"] == 88
    assert pilot.describe_live_protocol() == old and pilot.describe_return_protocol() == returned


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_fixed_exposure_bool_before_work(monkeypatch, invalid):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(train_policy_diagnostics=True, fixed_a_exposure=invalid)
    with pytest.raises(ValueError):
        pilot._packets(fixed_a_exposure=invalid)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(weighted_train=True),
        dict(return_context=True),
        dict(evaluate_policy_after_training=True),
        dict(episode_replay=True),
        dict(fresh_task_probes=True),
        dict(four_phase=True),
        dict(four_phase_policy=True),
        dict(four_phase_replay="coverage"),
        dict(four_phase_probes=True),
    ],
)
def test_should_require_plain_live_diagnostics_and_refuse_mixed_modes_before_work(
    monkeypatch, options
):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    kwargs = dict(fixed_a_exposure=True, **options)
    if options:
        kwargs["train_policy_diagnostics"] = True
    with pytest.raises(ValueError):
        pilot._run_pilot(**kwargs)


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_fixed_A_exposure_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-fixed-a-exposure"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="fixed-A128")

    monkeypatch.setattr(
        cli, "describe_fixed_a_exposure_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_fixed_a_exposure_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_train_live_pilot", forbidden)
    monkeypatch.setattr(cli, "run_weighted_train_live_pilot", forbidden)
    cli.main()
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "fixed-A128"
