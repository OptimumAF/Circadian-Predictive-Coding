"""Acquired-task continuation preserves its old prefix and isolates live readout."""

from copy import deepcopy
import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.acquired_task_return as acquired
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
from test_fixed_a_exposure_pilot import environment as exposure_environment
from test_recurrent_key_door_pilot import Learner as RecurrentLearner
from test_torch_feature_rehearsal_pilot import forbidden
from test_train_acquisition_pilot import Learner


POINTS = (
    0,
    8,
    16,
    32,
    64,
    72,
    80,
    96,
    128,
    136,
    144,
    160,
    192,
    200,
    208,
    224,
    256,
    264,
    272,
    288,
    320,
)


def environment(monkeypatch):
    captured = exposure_environment(monkeypatch, updates=320, points=POINTS)
    captured["acquired_calls"] = []
    original = acquired.evaluate_acquired_tasks

    def evaluate(report, source, states, budget, **kwargs):
        assert len(captured["finished"]) == 1 and len(RecurrentLearner.trains) == 320
        assert [point for point, _ in states] == [192, 256, 320]
        captured["acquired_calls"].append((report, source, states, budget))
        return original(report, source, states, budget, **kwargs)

    monkeypatch.setattr(acquired, "evaluate_acquired_tasks", evaluate)
    monkeypatch.setattr(acquired, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    return captured


def test_should_keep_old192_prefix_and_all320_train_releases_before_twelve_live_episodes(
    monkeypatch,
):
    captured = environment(monkeypatch)
    global_rng = pickle.dumps(np.random.get_state())
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "complete", report
    assert len(captured["collections"]) == 6 and len(captured["streams"]) == 1
    stream, isolated, training, budget = captured["streams"][0]
    assert (
        report["charged_updates"] == budget.updates_completed == len(RecurrentLearner.trains) == 320
    )
    assert budget.budget.max_training_updates == 320 and budget.replay_examples_completed == 0
    assert len(training) == len({batch.episode.episode_id for batch in training}) == 320
    assert [batch.episode.episode_id for batch in training] == [
        f"train-fit-{index}" for index in range(320)
    ]
    assert [batch.context for batch in training] == sum(
        ([f"{layout}-red", f"{layout}-blue"] * 32 for layout in ("A", "B", "A", "C_train", "A")), []
    )
    assert len({row for batch in training for row in batch.row_ids}) == 2560
    assert [batch.episode.episode_id for batch in captured["collections"][:4]] == [
        "train-fit-base-1",
        "train-fit-base--1",
        "train-return-base-B-1",
        "train-return-base-B--1",
    ]
    originals = {batch.context: batch for batch in captured["collections"]}
    assert set(originals) == {
        f"{layout}-{cue}" for layout in ("A", "B", "C_train") for cue in ("red", "blue")
    }
    for batch in training:
        base = originals[batch.context]
        assert batch.role == batch.episode.role == "train" and len(batch.features) == 8
        np.testing.assert_array_equal(batch.features, base.features)
        np.testing.assert_array_equal(batch.targets, base.targets)
        assert not np.shares_memory(batch.features, base.features)
        assert not np.shares_memory(batch.targets, base.targets)
    assert [point["updates"] for point in report["checkpoints"]] == list(POINTS)
    assert len(RecurrentLearner.classifications) == 126
    assert all(set(point["scores"]) == set(originals) for point in report["checkpoints"])
    assert report["work"] == dict(
        score_forwards_attempted=126,
        score_forwards_completed=126,
        score_rows_attempted=1008,
        score_rows_completed=1008,
    )
    result = report["stream_result"]
    assert result["completed_updates"] == 320 and result["training_examples"] == 2560
    assert result["score_forward_calls"] == 0 and len(result["applied_updates"]) == 320
    assert [receipt["update_number"] for receipt in result["applied_updates"]] == list(
        range(1, 321)
    )
    summary = report["training_summary"]
    assert summary["episodes"] == len(summary["episode_ids"]) == 320
    assert summary["unique_source_episodes"] == 6 and summary["collected_source_records"] == 48
    assert summary["training_exposures"] == 2560 and summary["roles"] == ["train"] * 6
    assert len(captured["finished"]) == len(captured["acquired_calls"]) == 1
    assert captured["acquired_calls"][0][3] is budget
    assert captured["diagnostics"] == []  # No pooled-prior rollout is added by this route.
    assert len(captured["sessions"]) == 12
    assert [session.state.completed_updates for session in captured["sessions"]] == [192] * 4 + [
        256
    ] * 4 + [320] * 4
    for session in captured["sessions"]:
        assert session.owner is not Learner.sources[0] and session.owner is not isolated
        assert session.owner is not stream._runtime._candidate
        assert len(session.inputs) == session.completed_steps == 12
    diagnostic = report["acquired_return_diagnostics"]
    assert diagnostic["status"] == "complete" and diagnostic["live_sessions_started"] == 12
    assert [slot["checkpoint_updates"] for slot in diagnostic["episodes"]] == [192] * 4 + [
        256
    ] * 4 + [320] * 4
    assert [slot["context"] for slot in diagnostic["episodes"]] == [
        "A",
        "A",
        "C_train",
        "C_train",
    ] * 3
    assert [slot["cue"] for slot in diagnostic["episodes"]] == [1, -1] * 6
    for slot in diagnostic["episodes"]:
        assert slot["status"] == "complete" and slot["state_unchanged"] is True
        assert slot["action_conditioning"] == "legal_actions_v1"
        assert slot["outcome"] == "step_limit" and slot["success"] is False and slot["return"] == 0
        assert all(step["action"] in step["legal_actions"] for step in slot["transitions"])
    assert "reference" not in diagnostic
    assert diagnostic["initial_A_acquired"] is False
    assert diagnostic["interpretation"] == "inconclusive_initial_A_not_acquired"
    assert all(value == 0 for metric in diagnostic["metrics"].values() for value in metric.values())
    assert diagnostic["work"] == dict(
        live_callback_attempts=144,
        live_callback_returns=144,
        live_applied_steps=144,
        live_sessions_started=12,
        failure_work_may_be_unobserved=False,
    )
    pure = report["acquired_return"]
    assert [point["updates"] for point in pure["endpoint_values"]] == [192, 256, 320]
    assert all(set(point) == {"updates", "A", "C_train"} for point in pure["endpoint_values"])
    assert pure["incomplete_initial_A_fit"] is True and pure["useful_skill_established"] is False
    assert all(value == 0 for metric in pure["metrics"].values() for value in metric.values())
    assert report["acquired_initial_condition_met"] is False
    assert report["acquired_interpretation"] == "inconclusive_initial_A_not_acquired"
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert all(
        field not in report
        for field in (
            "policy_evaluation",
            "probes",
            "replay",
            "training_objective",
            "train_policy_diagnostics",
        )
    )
    assert report["scientific_advantage_verified"] is False
    assert pickle.dumps(np.random.get_state()) == global_rng


def test_should_keep192_commits_and_complete_old_prefix_when_first_C_update_fails(monkeypatch):
    captured = environment(monkeypatch)
    original, attempts = Learner.train_batch, []

    def train(owner, features, targets):
        attempts.append(owner.state.completed_updates)
        if owner.state.completed_updates == 192:
            raise RuntimeError("first acquired-continuation C update failed")
        return original(owner, features, targets)

    monkeypatch.setattr(Learner, "train_batch", train)
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "incomplete" and "first acquired-continuation" in report["reason"]
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 192
    assert attempts == list(range(193))
    assert captured["streams"][0][3].budget.max_training_updates == 320
    assert report["partial_stream"]["native_candidate_updates"] == 192
    assert report["partial_stream"]["native_candidate_training_examples"] == 1536
    assert len(report["partial_stream"]["applied_updates"]) == 192
    assert [point["updates"] for point in report["checkpoints"]] == list(pilot.RETURN_POINTS)
    assert len(RecurrentLearner.classifications) == 78
    assert captured["finished"] == captured["sessions"] == captured["acquired_calls"] == []


@pytest.mark.parametrize("target", ["source", "scorer"])
def test_should_refuse_changed_preceding192_state_before_admitting_new_context_updates(
    monkeypatch, target
):
    captured = environment(monkeypatch)
    original = Learner.predict
    changed: list[bool] = []

    def predict(owner, features):
        result = original(owner, features)
        if owner.state.completed_updates == 192 and not changed:
            selected = Learner.sources[0] if target == "source" else owner
            selected.state.parameters["weight_input"][0, 0] += 1
            changed.append(True)
        return result

    monkeypatch.setattr(Learner, "predict", predict)
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 192
    assert report["partial_stream"]["native_candidate_updates"] == 192
    assert captured["finished"] == captured["sessions"] == captured["acquired_calls"] == []


@pytest.mark.parametrize("target", ["source", "actor", "receipt", "checkpoint", "teacher"])
def test_should_guard_all320_commits_and_selected_states_during_first_live_callback(
    monkeypatch, target
):
    captured = environment(monkeypatch)

    def corrupt(session):
        stream = captured["streams"][0][0]
        if target == "source":
            Learner.sources[0].state.parameters["weight_input"][0, 0] += 1
        elif target == "actor":
            stream._runtime.actor._learner.state.parameters["weight_input"][0, 0] += 1
        elif target == "receipt":
            diagnostic = stream._runtime.applied_updates[-1].diagnostic
            object.__setattr__(diagnostic, "value", diagnostic.value + 1)
        elif target == "checkpoint":
            state = next(state for count, state in stream._states if count == 192)
            state.parameters["weight_input"][0, 0] += 1
        else:
            captured["collections"][0].features[0, 0] += 1

    captured["on_prediction"] = corrupt
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 320
    assert len(report["stream_result"]["applied_updates"]) == 320
    assert len(captured["sessions"]) == 1 and len(captured["sessions"][0].inputs) == 1
    diagnostic = report["acquired_return_diagnostics"]
    assert diagnostic["status"] == "failed"
    assert all(slot["status"] == "not_run" for slot in diagnostic["episodes"][1:])


def test_should_refuse_changed_selected192_copy_before_starting_live_work(monkeypatch):
    captured = environment(monkeypatch)
    original = acquired.evaluate_acquired_tasks

    def replace_state(report, source, states, budget, **kwargs):
        states[0][1].parameters["weight_input"][0, 0] += 1
        return original(report, source, states, budget, **kwargs)

    monkeypatch.setattr(acquired, "evaluate_acquired_tasks", replace_state)
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 320
    assert len(captured["finished"]) == 1 and captured["sessions"] == []
    assert report["acquired_return_diagnostics"]["status"] == "failed"


def test_should_keep_live_fault_prefix_and_all_training_receipts_without_more_episodes(monkeypatch):
    captured = environment(monkeypatch)

    def fail(session):
        if len(session.inputs) == 2:
            raise RuntimeError("second acquired-return live prediction failed")

    captured["on_prediction"] = fail
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "incomplete" and report["charged_updates"] == 320
    assert len(RecurrentLearner.trains) == len(report["stream_result"]["applied_updates"]) == 320
    assert len(captured["sessions"]) == 1
    first = report["acquired_return_diagnostics"]["episodes"][0]
    assert first["prediction_calls_attempted"] == 2 and first["prediction_calls_returned"] == 1
    assert first["steps_executed"] == 1
    assert all(
        slot["status"] == "not_run"
        for slot in report["acquired_return_diagnostics"]["episodes"][1:]
    )


def test_should_refuse_changed_preceding_live_record_before_finishing_the_next_episode(monkeypatch):
    captured = environment(monkeypatch)

    def corrupt(session):
        if len(captured["sessions"]) == 2:
            diagnostic = captured["acquired_calls"][0][0]
            assert diagnostic["episodes"][0]["status"] == "complete"
            diagnostic["episodes"][0]["return"] += 1

    captured["on_prediction"] = corrupt
    report = pilot.run_acquired_return_pilot()
    assert report["status"] == "incomplete" and report["charged_updates"] == 320
    assert len(RecurrentLearner.trains) == 320 and len(captured["sessions"]) == 2
    assert [len(session.inputs) for session in captured["sessions"]] == [12, 1]
    diagnostic = report["acquired_return_diagnostics"]
    assert diagnostic["status"] == "failed"
    assert diagnostic["work"]["live_callback_attempts"] == 13
    assert all(slot["status"] == "not_run" for slot in diagnostic["episodes"][2:])


def test_should_describe_fixed320_work_without_resources_and_preserve_old_protocols(monkeypatch):
    old = [
        deepcopy(function())
        for function in (
            pilot.describe_protocol,
            pilot.describe_return_protocol,
            pilot.describe_fixed_exposure_protocol,
        )
    ]
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_acquired_return_protocol()
    assert protocol["protocol_id"] == "key_door_train_only_acquired_task_return_v1"
    assert pilot.ACQUIRED_RETURN_UPDATES == 320 and pilot.ACQUIRED_RETURN_POINTS == POINTS
    assert protocol["checkpoint_updates"] == list(POINTS)
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 320 and work["training_rows"] == 2560
    assert work["score_forwards"] == 126 and work["score_rows"] == 1008
    assert work["recurrent_unroll_steps"] == 3712
    assert work["maximum_native_predict_calls"] == 270
    assert work["environment_constructions"] == 18 and work["maximum_environment_steps"] == 192
    assert [
        function()
        for function in (
            pilot.describe_protocol,
            pilot.describe_return_protocol,
            pilot.describe_fixed_exposure_protocol,
        )
    ] == old


@pytest.mark.parametrize(
    "entry,expected",
    [
        ("run_pilot", {}),
        ("run_return_pilot", dict(return_context=True)),
        ("run_fixed_exposure_pilot", dict(train_policy_diagnostics=True, fixed_a_exposure=True)),
    ],
)
def test_should_leave_old_public_default_calls_free_of_acquired_return_flag(
    monkeypatch, entry, expected
):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(status="unchanged-default")

    monkeypatch.setattr(pilot, "_run_pilot", run)
    assert getattr(pilot, entry)() == dict(status="unchanged-default")
    assert calls == [expected]


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_acquired_return_bool_before_resources(monkeypatch, invalid):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(acquired_return=invalid)
    with pytest.raises(ValueError):
        pilot._packets(acquired_return=invalid)


@pytest.mark.parametrize(
    "mix",
    [
        dict(return_context=True),
        dict(evaluate_policy_after_training=True),
        dict(episode_replay=True),
        dict(fresh_task_probes=True),
        dict(four_phase=True),
        dict(four_phase_policy=True),
        dict(four_phase_replay="coverage"),
        dict(four_phase_probes=True),
        dict(train_policy_diagnostics=True),
        dict(weighted_train=True),
        dict(fixed_a_exposure=True),
    ],
)
def test_should_refuse_all_other_modes_before_collecting_or_allocating(monkeypatch, mix):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(acquired_return=True, **mix)


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_acquired_return_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-acquired-return"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="acquired-return")

    monkeypatch.setattr(
        cli, "describe_acquired_task_return_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_acquired_task_return_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_fixed_a_exposure_pilot", forbidden)
    monkeypatch.setattr(cli, "run_train_context_return_pilot", forbidden)
    cli.main()
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "acquired-return"
