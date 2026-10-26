"""Warm native state, exact prefix parity and local delayed reward accounting."""

from copy import deepcopy
import json
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_minigrid_reward_smoke as cli
from src.adapters import minigrid_reward_pilot as pilot
from src.adapters import minigrid_warm_start as warm
from src.adapters import torch_demonstration_reward_learner as typed
from src.adapters import torch_reward_update_learner as reward_port
from src.adapters.minigrid_reward_collection import MiniGridRewardCollectionFailed
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.actor_shadow import ActorShadowRuntime
import src.app.toy_execution_budget as budgets
from src.core.demonstration_episode import DemonstrationTransition
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode
import src.shared.torch_runtime as torch_runtime
from test_minigrid_learning_pilot import State, source_batch
from test_minigrid_reward_collection import frame
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


def snapshot(policy, updates=0, rows=0):
    shapes = dict(
        weight_input=(148, 8),
        weight_recurrent=(8, 8),
        bias_hidden=(1, 8),
        weight_output=(8, 7),
        bias_output=(1, 7),
    )
    parameters = {
        key: np.full(shape, 0.1 + index * 0.05, dtype=np.float32)
        for index, (key, shape) in enumerate(shapes.items())
    }
    gradients = {
        key: np.full(shape, 0.01 * (index + 1), dtype=np.float32) if updates else None
        for index, (key, shape) in enumerate(shapes.items())
    }
    momentum = (
        {
            index: {"momentum_buffer": np.full(shape, 0.2 * (index + 1), dtype=np.float32)}
            for index, shape in enumerate(shapes.values())
        }
        if updates
        else {}
    )
    return State(
        policy,
        parameters,
        {"state": momentum, "param_groups": [{"lr": 0.03, "momentum": 0.9}]},
        gradients,
        updates,
        rows,
    )


class Native:
    work: dict[str, Any] = {}

    def __init__(self, policy):
        self.policy, self.state = policy, snapshot(policy)
        self.work.setdefault("owners", []).append(self)

    @property
    def completed_updates(self):
        return self.state.completed_updates

    @property
    def training_examples(self):
        return self.state.training_examples

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.work.setdefault("restores", []).append((self, deepcopy(state)))
        self.state, self.policy = deepcopy(state), deepcopy(state.policy)

    def fork(self):
        child = Native(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_reward_episode(self, episode, behavior_state, *, discount):
        assert type(episode) is PolicyGradientEpisode and episode.role == "train"
        assert discount == 1 and episode.num_classes == 7
        assert pickle.dumps(behavior_state) == pickle.dumps(self.state)
        before = self.snapshot_state()
        calls = self.work.setdefault("reward_calls", [])
        calls.append(dict(owner=self, episode=deepcopy(episode), before=before))
        if self.work.get("native_fail") == len(calls):
            raise RuntimeError("atomic reward update refused")
        for index, (key, parameter) in enumerate(self.state.parameters.items()):
            old = (
                self.state.optimizer_state["state"]
                .get(index, {})
                .get("momentum_buffer", np.zeros_like(parameter))
            )
            carried = 0.9 * old
            self.state.optimizer_state["state"][index] = {"momentum_buffer": carried.copy()}
            self.state.parameters[key] = parameter - 0.03 * carried
            self.state.gradients[key] = np.zeros_like(parameter)
        self.state.completed_updates += 1
        self.state.training_examples += len(episode.transitions)
        calls[-1]["after"] = self.snapshot_state()
        callback = self.work.get("after_native")
        if callback:
            callback(self)
        return TrainingDiagnostic("torch_recurrent_reward_logprob_sum_v1", 0.0)

    def train_batch(self, *args):
        pytest.fail("warm reconstruction is separately stubbed; reward uses typed packets")

    def predict(self, *args):
        pytest.fail("readouts are separately stubbed; no new evaluation in reward phase")

    def start_episode(self):
        pytest.fail("collection is separately stubbed with observed raw-law records")


def collected(source, layout, cue, episode_id, sampler, count=2, failed=False):
    state = source.snapshot_state()
    before = deepcopy(sampler.bit_generator.state)
    probabilities = (0.0,) * 6 + (1.0,)
    predictions = [
        dict(
            observation=frame(step),
            probabilities=probabilities,
            uniform=float(sampler.random()),
            action=6,
        )
        for step in range(count)
    ]
    transitions = tuple(
        DemonstrationTransition(step, frame(step), 6, 0.0, frame(step + 1), False, 7)
        for step in range(count)
    )
    episode = PolicyGradientEpisode(
        episode_id,
        layout.context,
        "train",
        transitions,
        tuple(probabilities for _ in transitions),
        "truncated",
        7,
    )
    return dict(
        status="failed" if failed else "complete",
        reason="partial collection failed" if failed else None,
        episode=None if failed else episode,
        behavior_state=state,
        sampler_before=before,
        sampler_after=deepcopy(sampler.bit_generator.state),
        sampler_binding_preserved=True,
        state_unchanged=True,
        elapsed_seconds=0.0,
        predictions=predictions,
        environment=dict(
            status="failed" if failed else "complete",
            closed=True,
            role="train",
            context=layout.context,
            cue=cue,
            total_return=0.0,
            environment_steps_attempted=count,
            environment_steps=count,
            transitions=[
                dict(
                    observation=step.observation,
                    next_observation=step.next_observation,
                    action=step.action,
                    reward=0.0,
                    terminated=False,
                    truncated=not failed and i == count - 1,
                    applied=True,
                )
                for i, step in enumerate(transitions)
            ],
            terminated=False,
            truncated=not failed,
            failure_work_may_be_unobserved=False,
        ),
        work=dict(
            snapshots_attempted=2,
            snapshots_returned=2,
            sessions_attempted=1,
            sessions_started=1,
            environment_constructions_attempted=1,
            environment_constructions_returned=1,
            prediction_calls_attempted=count,
            prediction_calls_returned=count,
            sampling_draws=count,
            environment_steps_attempted=count,
            environment_steps_returned=count,
            close_calls_attempted=1,
            close_calls_returned=1,
            failure_work_may_be_unobserved=failed,
        ),
        close_error=None,
        readback_error=None,
    )


@pytest.fixture
def ports(monkeypatch):
    work: dict[str, Any] = dict(collections=[], runtimes=[], reports=[], polls=[])
    monkeypatch.setattr(Native, "work", work)
    monkeypatch.setattr(typed, "TorchRecurrentLearner", Native)
    monkeypatch.setattr(reward_port, "TorchRecurrentLearner", Native)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        pilot, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    monkeypatch.setattr(
        warm, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    factory = pilot._new_report

    def report_factory(*args, **kwargs):
        value = factory(*args, **kwargs)
        work["reports"].append(value)
        return value

    def runtime(source, **kwargs):
        actual = ActorShadowRuntime(source, **kwargs)
        work["runtimes"].append(actual)
        work["budget"] = kwargs["budget"]
        poll = actual.train_ready

        def ready(**options):
            before = kwargs["budget"].updates_completed
            receipts = poll(**options)
            work["polls"].append((before, kwargs["budget"].updates_completed, receipts))
            return receipts

        monkeypatch.setattr(actual, "train_ready", ready)
        return actual

    def collect(source, layout, cue, *, episode_id, sampler, check_boundary, **options):
        index = len(work["collections"])
        check_boundary()
        state = source.snapshot_state()
        offset = work.get("offset", 0)
        assert state.completed_updates == offset + index
        assert state.training_examples == (1152 if offset else 0) + 2 * index
        call = dict(source=source, state=state, sampler=sampler, layout=layout, cue=cue)
        work["collections"].append(call)
        callback = work.get("during_collection")
        if callback:
            callback(index, call)
        check_boundary()
        failed = work.get("collection_fail") == index
        report = collected(source, layout, cue, episode_id, sampler, 1 if failed else 2, failed)
        call["report"] = report
        check_boundary()
        if failed:
            raise MiniGridRewardCollectionFailed(report)
        return report

    monkeypatch.setattr(pilot, "_new_report", report_factory)
    monkeypatch.setattr(pilot, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(pilot, "collect_minigrid_reward_episode", collect)
    return SimpleNamespace(work=work)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_warm_mode_before_resources(monkeypatch, invalid):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(warm_start=invalid)
    with pytest.raises(ValueError):
        pilot.run_pilot(warm_start=invalid)


def test_should_keep_cold_descriptor_and_explicit_false_identical_and_resource_free(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(warm, "prepare_warm_start", forbidden)
    cold = pilot.describe_protocol()
    assert cold == pilot.describe_protocol(warm_start=False)
    assert cold["protocol_id"] == "minigrid_cue_key_door_recurrent_reward_smoke_v1"
    assert cold["limits"]["body_seconds"] == 45 and cold["planned_work"]["native_train_calls"] == 8
    assert json.loads(json.dumps(pilot.describe_protocol(warm_start=True), allow_nan=False))


def readout(point):
    probabilities = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.015625]
    scores = [
        dict(
            context="A",
            cue=cue,
            ce=float(np.log(4)),
            accuracy=0.0,
            exact_sequence_accuracy=0.0,
            probabilities=[list(probabilities) for _ in range(9)],
            predicted_actions=[0] * 9,
            target_actions=[1] * 9,
        )
        for cue in ("blue", "red")
    ]
    episodes = []
    if point in (0, 128):
        for cue in ("blue", "red"):
            predictions = [
                dict(observation=frame(step), probabilities=list(probabilities), action=0)
                for step in range(2)
            ]
            environment = dict(
                episode_id=f"recreated-{point}-{cue}",
                context="A",
                cue=cue,
                role="train",
                closed=True,
                status="complete",
                terminated=False,
                truncated=True,
                environment_steps_attempted=2,
                environment_steps=2,
                total_return=0.0,
                transitions=[
                    dict(
                        observation=frame(step),
                        next_observation=frame(step + 1),
                        action=0,
                        reward=0.0,
                        terminated=False,
                        truncated=step == 1,
                        attempted=True,
                        applied=True,
                    )
                    for step in range(2)
                ],
            )
            episodes.append(
                dict(
                    context="A",
                    cue=cue,
                    source_role="train",
                    status="complete",
                    predictions=predictions,
                    environment=environment,
                    success=False,
                    **{"return": 0.0},
                )
            )
    live = point in (0, 128)
    return dict(
        method="recurrent_bc",
        updates=point,
        role="train",
        status="complete",
        reason=None,
        scores=scores,
        episodes=episodes,
        work=dict(
            raw_score_forwards_attempted=2,
            raw_score_forwards_completed=2,
            raw_score_rows_attempted=18,
            raw_score_rows_completed=18,
            live_prediction_calls_attempted=4 if live else 0,
            live_prediction_calls_returned=4 if live else 0,
            live_sessions_started=2 if live else 0,
            environment_resets_completed=2 if live else 0,
            environment_steps_returned=4 if live else 0,
            failure_work_may_be_unobserved=False,
        ),
    )


def training_row(updates=128):
    row = warm.shared._row("recurrent_bc")
    row.update(
        status="trained",
        charged_updates=updates,
        native_training_examples=updates * 9,
        source_unchanged=True,
        actor_unchanged=True,
        wake_calls_attempted=updates,
        wake_calls_returned=updates,
        checkpoint_updates=list(warm.POINTS),
        stream_result=dict(
            completed_updates=updates,
            training_examples=updates * 9,
            applied_updates=[
                dict(
                    update_number=index + 1,
                    diagnostic=dict(
                        definition="torch_recurrent_episode_cross_entropy_mean_v1", value=0.5
                    ),
                )
                for index in range(updates)
            ],
        ),
    )
    return row


def prepare_fixture(monkeypatch, ports, *, partial=False):
    work = ports.work

    def prepare(report, memory, deadline, check_boundary):
        assert report == {} and deadline == 190.0
        check_boundary()
        updates = 2 if partial else 128
        row = training_row(updates)
        evaluations = [readout(point) for point in warm.POINTS] if not partial else []
        report.update(
            status="incomplete" if partial else "complete",
            reason="partial warm prefix" if partial else None,
            training=row,
            evaluations=evaluations,
            parity_checks=[None] * 5 if partial else [True] * 5,
            parity_matched=not partial,
            full_state_restored=not partial,
            reference={"fixed": "accepted_unweighted_A128"},
            historical_work={"charged_updates": 128, "teacher_environment_steps_observed": 54},
            work=warm._work([row], evaluations, [], partial),
            readback_error=None,
        )
        work["warm_report"] = report
        if partial:
            raise RuntimeError("partial warm prefix preserved two paid updates")
        source = typed.TorchDemonstrationRewardLearner(TorchRecurrentPolicy(148, 8, 7, 17))
        state = snapshot(source.policy, 128, 1152)
        source.restore_state(state)
        work.update(offset=128, warm_state=deepcopy(state), warm_source=source)
        payload = pickle.dumps(report)

        def guard():
            if pickle.dumps(report) != payload or pickle.dumps(
                source.snapshot_state()
            ) != pickle.dumps(state):
                raise ValueError("original warm evidence/full state changed")
            callback = work.get("warm_guard_callback")
            if callback:
                callback()
            # Match the actual helper's final common(): its boundary callbacks
            # finish before the complete saved warm record is checked again.
            check_boundary()
            if pickle.dumps(report) != payload:
                raise ValueError("original warm evidence changed after boundary")

        return dict(source=source, guard=guard, updates=128, rows=1152)

    monkeypatch.setattr(pilot, "prepare_warm_start", prepare)


def test_should_restore_full_warm_state_then_use_local_reward_receipts_and_original_momentum(
    monkeypatch, ports
):
    prepare_fixture(monkeypatch, ports)
    report = pilot.run_pilot(warm_start=True)
    assert report["status"] == "complete", report["reason"]
    work = ports.work
    assert report["warm_start"]["parity_checks"] == [True] * 5
    assert report["warm_start"]["full_state_restored"] is True
    assert report["wake_updates"] == work["budget"].updates_completed == 8
    assert work["budget"].budget.max_training_updates == 8
    assert work["budget"].budget.max_wall_seconds == 45
    assert work["budget"].started_at == work["budget"].last_clock == 10.0
    assert report["training_examples"] == 16
    assert report["total_new_native_updates"] == 136 and report["total_new_training_rows"] == 1168
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert pickle.dumps(work["warm_source"].snapshot_state()) == pickle.dumps(work["warm_state"])
    runtime = work["runtimes"][0]
    assert runtime.actor.snapshot_state().state.completed_updates == 128
    assert runtime.candidate_snapshot().state.completed_updates == 136
    assert len(work["reward_calls"]) == len(work["collections"]) == 8
    for index, (slot, call) in enumerate(
        zip(report["training"], work["reward_calls"], strict=True)
    ):
        assert slot["receipt"]["update_number"] == index + 1
        assert (
            slot["charged_updates_before"] == index and slot["charged_updates_after"] == index + 1
        )
        assert (
            slot["native_updates_before"] == 128 + index
            and slot["native_updates_after"] == 129 + index
        )
        assert slot["native_training_examples_before"] == 1152 + 2 * index
        assert slot["native_training_examples_after"] == 1154 + 2 * index
        assert slot["early_release_receipts"] == [] and slot["commit_validated"] is True
        assert call["before"].completed_updates == 128 + index
        if index:
            assert pickle.dumps(call["before"]) == pickle.dumps(
                work["reward_calls"][index - 1]["after"]
            )
        for key, value in call["before"].gradients.items():
            if not index:
                np.testing.assert_array_equal(value, work["warm_state"].gradients[key])
        for parameter_index, key in enumerate(call["before"].parameters):
            prior_momentum = call["before"].optimizer_state["state"][parameter_index][
                "momentum_buffer"
            ]
            after_momentum = call["after"].optimizer_state["state"][parameter_index][
                "momentum_buffer"
            ]
            np.testing.assert_allclose(after_momentum, 0.9 * prior_momentum, rtol=0, atol=0)
            assert not np.array_equal(call["before"].parameters[key], call["after"].parameters[key])
    assert report["total_new_work"]["raw_score_forwards"] == 10
    assert report["total_new_work"]["raw_score_rows"] == 90
    assert (
        report["total_new_work"]["teacher_episodes"]
        == report["total_new_work"]["final_episodes"]
        == 0
    )
    assert report["scientific_metrics"] is None and report["scientific_advantage_verified"] is False


def test_should_keep_explicit_cold_execution_free_of_warm_helper_and_extra_schema(
    monkeypatch, ports
):
    monkeypatch.setattr(pilot, "prepare_warm_start", forbidden)
    report = pilot.run_pilot(warm_start=False)
    assert report["status"] == "complete", report["reason"]
    assert report["wake_updates"] == 8 and report["training_examples"] == 16
    assert report["training"][0]["native_updates_before"] == 0
    assert "warm_start" not in report and "total_new_work" not in report
    assert ports.work["runtimes"][0].actor.snapshot_state().state.completed_updates == 0


def test_should_give_reward_only_the_original_whole_deadline_remainder(monkeypatch, ports):
    prepare_fixture(monkeypatch, ports)
    prepare = pilot.prepare_warm_start
    clock = [10.0]
    monkeypatch.setattr(pilot, "monotonic", lambda: clock[0])

    def finish_prefix(*args):
        prepared = prepare(*args)
        clock[0] = 170.0
        return prepared

    monkeypatch.setattr(pilot, "prepare_warm_start", finish_prefix)
    report = pilot.run_pilot(warm_start=True)
    assert report["status"] == "complete", report["reason"]
    assert ports.work["budget"].started_at == 170.0
    assert ports.work["budget"].budget.max_wall_seconds == 20.0
    assert report["total_seconds"] == 160.0
    assert report["total_new_native_updates"] == 136


def test_should_preserve_partial_warm_charges_and_admit_no_reward_runtime(monkeypatch, ports):
    prepare_fixture(monkeypatch, ports, partial=True)
    report = pilot.run_pilot(warm_start=True)
    assert report["status"] == "incomplete" and "two paid updates" in report["reason"]
    assert report["warm_start"]["training"]["charged_updates"] == 2
    assert report["total_new_native_updates"] == 2 and report["total_new_training_rows"] == 18
    assert report["wake_updates"] == 0 and ports.work["runtimes"] == []
    assert not ports.work.get("reward_calls") and ports.work["collections"] == []
    assert all(slot["status"] == "not_run" for slot in report["training"])


@pytest.mark.parametrize("phase", ["collection", "native"])
def test_should_preserve_warm_and_two_local_commits_when_third_reward_work_fails(
    monkeypatch, ports, phase
):
    prepare_fixture(monkeypatch, ports)
    ports.work["collection_fail" if phase == "collection" else "native_fail"] = (
        2 if phase == "collection" else 3
    )
    report = pilot.run_pilot(warm_start=True)
    assert report["status"] == "incomplete" and report["wake_updates"] == 2
    assert report["warm_start"]["status"] == "complete"
    assert report["total_new_native_updates"] == 130 and report["total_new_training_rows"] == 1156
    assert len(ports.work["runtimes"][0].applied_updates) == 2
    assert report["training"][2]["charged_updates_after"] == 2
    assert all(slot["status"] == "not_run" for slot in report["training"][3:])
    expected = np.random.Generator(np.random.PCG64(29))
    expected.random(5 if phase == "collection" else 6)
    assert report["training"][2]["sampler_after"] == expected.bit_generator.state


@pytest.mark.parametrize("target", ["warm_record", "candidate", "budget"])
def test_should_refuse_changed_preceding_warm_or_reward_state_before_second_collection_spends(
    monkeypatch, ports, target
):
    prepare_fixture(monkeypatch, ports)

    def corrupt(index, call):
        if index != 1:
            return
        if target == "warm_record":
            ports.work["warm_report"]["training"]["stream_result"]["applied_updates"][0][
                "update_number"
            ] = 99
        elif target == "candidate":
            ports.work["runtimes"][0]._candidate._learner.state.parameters["bias_output"] += 1
        else:
            ports.work["budget"].updates_completed += 1

    ports.work["during_collection"] = corrupt
    report = pilot.run_pilot(warm_start=True)
    assert report["status"] == "incomplete" and report["wake_updates"] == 1
    assert report["total_new_native_updates"] == 129
    assert len(ports.work["reward_calls"]) == 1
    assert (
        report["warm_start"]["training"]["stream_result"]["applied_updates"][0]["update_number"]
        == 1
    )
    assert "report" not in ports.work["collections"][1]


@pytest.mark.parametrize("target", ["candidate", "sampler", "budget"])
def test_should_refuse_late_warm_guard_callback_mutation_before_publication(
    monkeypatch, ports, target
):
    prepare_fixture(monkeypatch, ports)
    complete = budgets.ToyBudgetSession.complete_memory
    final_phase: list[bool] = []
    changed: list[bool] = []

    def memory_complete(owner):
        value = complete(owner)
        final_phase.append(True)
        return value

    def corrupt():
        if not final_phase or changed:
            return
        changed.append(True)
        if target == "candidate":
            ports.work["runtimes"][0]._candidate._learner.state.parameters["bias_output"] += 1
        elif target == "sampler":
            ports.work["collections"][-1]["sampler"].random()
        else:
            ports.work["budget"].updates_completed += 1

    monkeypatch.setattr(budgets.ToyBudgetSession, "complete_memory", memory_complete)
    ports.work["warm_guard_callback"] = corrupt
    report = pilot.run_pilot(warm_start=True)
    assert changed == [True] and report["status"] == "incomplete"
    assert report["wake_updates"] == 8 and report["total_new_native_updates"] == 136
    assert len(ports.work["reward_calls"]) == len(ports.work["collections"]) == 8
    assert report["work"]["sampling_draws"] == 16
    assert report["warm_start"]["parity_checks"] == [True] * 5


@pytest.mark.parametrize("target", ["warm_evidence", "source"])
def test_should_retain_first_committed_receipt_when_warm_binding_changes_after_native_return(
    monkeypatch, ports, target
):
    prepare_fixture(monkeypatch, ports)

    def corrupt(owner):
        if target == "warm_evidence":
            ports.work["warm_report"]["evaluations"][0]["scores"][0]["accuracy"] = 1.0
        else:
            ports.work["warm_source"]._learner.state.gradients["bias_output"] += 1

    ports.work["after_native"] = corrupt
    report = pilot.run_pilot(warm_start=True)
    first = report["training"][0]
    assert report["status"] == "incomplete" and report["wake_updates"] == 1
    assert report["total_new_native_updates"] == 129
    assert first["update_committed"] is True and first["commit_validated"] is False
    assert first["receipt"]["update_number"] == 1 and first["charged_updates_after"] == 1
    assert report["warm_start"]["evaluations"][0]["scores"][0]["accuracy"] == 0.0
    assert len(ports.work["collections"]) == len(ports.work["reward_calls"]) == 1
    assert all(slot["status"] == "not_run" for slot in report["training"][1:])


def test_should_describe_warm_cli_without_loading_optional_native_resources(monkeypatch, capsys):
    calls = []

    def describe(**options):
        calls.append(options)
        return {"protocol_id": "warm"}

    monkeypatch.setattr(pilot, "describe_protocol", describe)
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    assert cli.main(["--warm-start", "--describe"]) == 0
    assert calls == [{"warm_start": True}]
    assert json.loads(capsys.readouterr().out) == {"protocol_id": "warm"}


@pytest.mark.parametrize(
    "flag,status,code", [(True, "complete", 0), (True, "incomplete", 1), (False, "complete", 0)]
)
def test_should_forward_only_explicit_warm_cli_option_and_preserve_incomplete_exit(
    monkeypatch, capsys, flag, status, code
):
    calls, threads = [], []
    stub = SimpleNamespace(
        set_num_threads=lambda count: threads.append(("intra", count)),
        set_num_interop_threads=lambda count: threads.append(("inter", count)),
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: stub)

    def run(**options):
        calls.append(options)
        return {"status": status}

    monkeypatch.setattr(pilot, "run_pilot", run)
    assert cli.main(["--warm-start"] if flag else []) == code
    assert calls == ([{"warm_start": True}] if flag else [{}])
    assert threads == [("intra", 1), ("inter", 1)]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_should_refuse_existing_warm_output_before_preparing_any_prefix(monkeypatch, tmp_path):
    destination = tmp_path / "original.json"
    destination.write_bytes(b"original evidence")
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--warm-start", "--output", str(destination)])
    assert error.value.code == 2 and destination.read_bytes() == b"original evidence"


def test_should_refuse_warm_record_mutation_during_internal_memory_boundary(monkeypatch, ports):
    prepare_fixture(monkeypatch, ports)
    original = Sampler.snapshot
    inside_boundary: list[bool] = []
    changed: list[bool] = []

    def warm_guard_callback():
        # The first actual collection is already entered, before its next draw.
        if ports.work["collections"] and not changed:
            inside_boundary.append(True)

    def snapshot(owner):
        value = original(owner)
        if inside_boundary and not changed:
            changed.append(True)
            ports.work["warm_report"]["training"]["charged_updates"] = 0
        return value

    ports.work["warm_guard_callback"] = warm_guard_callback
    monkeypatch.setattr(Sampler, "snapshot", snapshot)
    report = pilot.run_pilot(warm_start=True)
    assert changed == [True] and report["status"] == "incomplete"
    assert report["warm_start"]["training"]["charged_updates"] == 128
    assert report["wake_updates"] == 0 and report["total_new_native_updates"] == 128
    assert not ports.work.get("reward_calls")
    assert "report" not in ports.work["collections"][0]


@pytest.mark.parametrize("partial", [False, True])
def test_should_prepare_full_native_state_only_after_all_five_exact_readouts_or_keep_partial_charges(
    monkeypatch, ports, partial
):
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    recorded = json.loads(json.dumps([readout(point) for point in warm.POINTS]))
    admitted = dict(
        sources=sources,
        row=training_row(),
        evaluations=recorded,
        provenance={"source_reports": [], "reference_path": "fixed-accepted-reference"},
    )
    monkeypatch.setattr(warm, "_load_reference", lambda: {"accepted": True})
    monkeypatch.setattr(warm, "_admit_reference", lambda raw: admitted)
    monkeypatch.setattr(warm, "monotonic", lambda: 10.0)
    monkeypatch.setattr(warm.shared, "_collect_sources", forbidden)
    frozen_sources = pickle.dumps(sources)
    calls: list[int] = []
    arms: list[dict[str, Any]] = []
    boundaries: list[bool] = []

    def train(row, packets, memory, deadline, check_boundary, *, checkpoint_updates):
        assert deadline == 190.0 and checkpoint_updates == warm.POINTS
        assert len(packets) == 128 and all(packet.context == "A" for packet in packets)
        assert len({packet.episode.episode_id for packet in packets}) == 128
        assert all(
            packet.episode.role == "train" and packet.features.shape == (9, 148)
            for packet in packets
        )
        check_boundary()
        row.update(training_row(2 if partial else 128))
        if partial:
            row["status"] = "incomplete"
            raise RuntimeError("two original prefix commits then native refusal")
        source = Native(TorchRecurrentPolicy(148, 8, 7, 17))
        states = [(point, snapshot(source.policy, point, 9 * point)) for point in warm.POINTS]
        budget = budgets.ToyBudgetSession(
            budgets.ToyExecutionBudget(max_training_updates=128), lambda: 10.0
        )
        budget.updates_completed = 128
        initial, state_payload = pickle.dumps(source.snapshot_state()), pickle.dumps(states)

        def guard():
            assert pickle.dumps(source.snapshot_state()) == initial
            assert pickle.dumps(states) == state_payload

        arm = dict(source=source, states=states, budget=budget, guard=guard)
        arms.append(arm)
        return arm

    def score(source, state, *, method, updates, sources, budget, check_boundary, **options):
        assert method == "recurrent_bc" and state.completed_updates == updates
        assert state.training_examples == 9 * updates
        assert len(sources) == 2 and [packet.context for packet in sources] == ["A", "A"]
        assert budget is arms[0]["budget"] and budget.updates_completed == 128
        assert bool(options) == (updates in (0, 128))
        if options:
            assert options["role"] == "train" and len(options["layouts"]) == 1
        check_boundary()
        budget.before_final()
        calls.append(updates)
        return readout(updates)

    monkeypatch.setattr(warm.shared, "_train_arm", train)
    monkeypatch.setattr(warm, "_score_checkpoint", score)
    monkeypatch.setattr(warm, "evaluate_checkpoint", score)
    report: dict[str, Any] = {}
    if partial:
        with pytest.raises(RuntimeError, match="two original prefix commits"):
            warm.prepare_warm_start(report, Sampler(), 190.0, lambda: boundaries.append(True))
        assert report["status"] == "incomplete" and report["full_state_restored"] is False
        assert report["work"]["charged_updates"] == 2
        assert report["work"]["observed_native_training_rows"] == 18
        assert report["parity_checks"] == [None] * 5 and calls == []
        assert all(record["status"] == "not_run" for record in report["evaluations"])
        assert not ports.work.get("restores")
    else:
        prepared = warm.prepare_warm_start(
            report, Sampler(), 190.0, lambda: boundaries.append(True)
        )
        assert calls == list(warm.POINTS) and report["parity_checks"] == [True] * 5
        assert report["status"] == "complete" and report["full_state_restored"] is True
        assert type(prepared["source"]) is typed.TorchDemonstrationRewardLearner
        assert prepared["updates"] == 128 and prepared["rows"] == 1152
        assert pickle.dumps(prepared["source"].snapshot_state()) == pickle.dumps(
            arms[0]["states"][-1][1]
        )
        assert report["work"]["charged_updates"] == 128
        assert report["work"]["raw_score_forwards_completed"] == 10
        assert report["work"]["raw_score_rows_completed"] == 90
        assert report["work"]["live_prediction_calls_returned"] == 8
        prepared["guard"]()
        report["evaluations"][0]["scores"][0]["accuracy"] = 1.0
        with pytest.raises(ValueError, match="preceding evidence"):
            prepared["guard"]()
    assert boundaries and pickle.dumps(sources) == frozen_sources
    assert not ports.work.get("reward_calls")
