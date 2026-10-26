"""Independent delayed-release momentum control on unchanged observed episodes.

The generic runtime and budgets are real. Native SGD and original observations
are synthetic, owned fakes; no learner, environment or experiment runs here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_momentum_control as control
from src.adapters import torch_demonstration_reward_learner as typed
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.adapters.torch_recurrent_zero_loss import TORCH_RECURRENT_ZERO_LOSS_ID
from src.app.actor_shadow import ActorShadowRuntime
import src.app.toy_execution_budget as budgets
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_key_door import predefined_layouts
from src.core.policy_gradient_episode import PolicyGradientEpisode
from test_minigrid_warm_reward import Native, collected, snapshot
from test_torch_feature_rehearsal_pilot import Sampler


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {"calls": [], "runtimes": [], "polls": [], "boundaries": []}
    monkeypatch.setattr(Native, "work", work)
    monkeypatch.setattr(typed, "TorchRecurrentLearner", Native)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    clock = [10.0]
    monkeypatch.setattr(control, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        control, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )

    def zero_loss(owner, episode):
        assert type(episode) is PolicyGradientEpisode and episode.role == "train"
        call = dict(owner=owner, episode=deepcopy(episode), before=owner.snapshot_state())
        work["calls"].append(call)
        if len(work["calls"]) == work.get("fail_native"):
            raise RuntimeError("atomic momentum-only update refused")
        for index, key in enumerate(owner.state.parameters):
            carried = 0.9 * owner.state.optimizer_state["state"][index]["momentum_buffer"]
            owner.state.optimizer_state["state"][index]["momentum_buffer"] = carried.copy()
            owner.state.parameters[key] -= 0.03 * carried
            owner.state.gradients[key] = np.zeros_like(owner.state.parameters[key])
        owner.state.completed_updates += 1
        owner.state.training_examples += len(episode.transitions)
        call["after"] = owner.snapshot_state()
        callback = work.get("after_native")
        if callback:
            callback(owner)
        return TrainingDiagnostic(TORCH_RECURRENT_ZERO_LOSS_ID, 0.0)

    monkeypatch.setattr(Native, "train_zero_loss_episode", zero_loss, raising=False)
    source = typed.TorchDemonstrationRewardLearner(TorchRecurrentPolicy(148, 8, 7, 17))
    source.restore_state(snapshot(source.policy, 128, 1152))
    layouts = {layout.context: layout for layout in predefined_layouts()[:3]}
    contexts = ("A", "A", "B", "B", "C_train", "C_train", "A", "A")
    sampler = np.random.Generator(np.random.PCG64(29))
    episodes = []
    for index, context in enumerate(contexts):
        episode = collected(
            source,
            layouts[context],
            "blue" if index % 2 == 0 else "red",
            f"minigrid-reward-train-{index}",
            sampler,
            count=index + 1,
        )["episode"]
        episodes.append(
            replace(
                episode,
                transitions=tuple(
                    replace(step, reward=float(step.step + 1) / 10) for step in episode.transitions
                ),
            )
        )
    report = control.new_momentum_control()
    work["report"] = report

    def runtime(learner, **options):
        actual = ActorShadowRuntime(learner, **options)
        work["runtimes"].append(actual)
        work["budget"] = options["budget"]
        poll = actual.train_ready

        def ready(**arguments):
            before = options["budget"].updates_completed
            receipts = poll(**arguments)
            work["polls"].append((before, options["budget"].updates_completed, receipts))
            callback = work.get("after_poll")
            if callback:
                callback(before, receipts)
            return receipts

        monkeypatch.setattr(actual, "train_ready", ready)
        return actual

    def boundary():
        work["boundaries"].append(True)
        callback = work.get("on_boundary")
        if callback:
            callback()

    monkeypatch.setattr(control, "ActorShadowRuntime", runtime)
    return SimpleNamespace(
        work=work,
        source=source,
        episodes=tuple(episodes),
        report=report,
        memory=Sampler(),
        clock=clock,
        boundary=boundary,
    )


def run(case, *, episodes=None, deadline=190.0):
    return control.run_momentum_control(
        case.report,
        case.source,
        case.episodes if episodes is None else episodes,
        case.memory,
        deadline,
        check_boundary=case.boundary,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "missing",
        "extra",
        "wrong_type",
        "role",
        "context",
        "identifier",
        "width",
        "classes",
        "overlong",
    ],
)
def test_should_refuse_unbound_original_episode_contract_before_runtime_or_control_updates(
    case, fault
):
    episodes: Any = case.episodes
    if fault == "list":
        episodes = list(episodes)
    elif fault == "missing":
        episodes = episodes[:-1]
    elif fault == "extra":
        episodes = (*episodes, episodes[-1])
    elif fault == "wrong_type":
        episodes = ({"episode": episodes[0]}, *episodes[1:])
    else:
        first = deepcopy(episodes[0])
        if fault == "role":
            first = replace(first, role="final_test")
        elif fault == "context":
            first = replace(first, context="D")
        elif fault == "identifier":
            first = replace(first, episode_id="different-collection")
        elif fault == "width":
            step = first.transitions[0]
            first = replace(
                first,
                transitions=(replace(step, observation=(0.0, 0.0), next_observation=(0.25, 0.0)),),
            )
        elif fault == "classes":
            object.__setattr__(first, "num_classes", 3)
        else:
            object.__setattr__(first, "transitions", first.transitions * 17)
        episodes = (first, *episodes[1:])
    with pytest.raises((ValueError, RuntimeError)):
        run(case, episodes=episodes)
    assert case.work["runtimes"] == [] and case.work["calls"] == []


@pytest.mark.parametrize("field,value", [("completed_updates", 127), ("training_examples", 1153)])
def test_should_require_original_full_warm_native_clocks_before_runtime_admission(
    case, field, value
):
    setattr(case.source._learner.state, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        run(case)
    assert case.work["runtimes"] == [] and case.work["calls"] == []


def test_should_require_caller_binding_to_original_collections_before_admission(case):
    def invalid_original_receipt():
        raise ValueError("caller no longer owns eight validated original reward collections")

    case.work["on_boundary"] = invalid_original_receipt
    with pytest.raises((ValueError, RuntimeError), match="original reward collections"):
        run(case)
    assert case.work["runtimes"] == [] and case.work["calls"] == []


def test_should_reuse_original_eight_episodes_on_an_independent_full_warm_fork(case):
    source_before = pickle.dumps(case.source.snapshot_state())
    original_episodes = pickle.dumps(case.episodes)
    prepared = run(case)
    state = prepared["state"]
    assert state.completed_updates == 136 and state.training_examples == 1188
    assert case.report["status"] == "complete"
    assert case.report["source_unchanged"] is case.report["actor_unchanged"] is True
    assert case.report["work"]["native_updates_committed"] == 8
    assert case.report["work"]["committed_training_rows"] == 36
    assert case.report["work"]["attempted_training_rows"] == 36
    assert case.report["work"]["repeated_observation_exposure_rows"] == 36
    assert case.work["budget"].updates_completed == 8
    assert case.work["budget"].budget.max_training_updates == 8
    assert case.work["budget"].budget.max_wall_seconds == 45.0
    assert len(case.work["calls"]) == 8 and len(case.work["runtimes"]) == 1
    runtime = case.work["runtimes"][0]
    assert runtime.actor.snapshot_state().state.completed_updates == 128
    assert runtime.candidate_snapshot().state.completed_updates == 136
    assert pickle.dumps(case.source.snapshot_state()) == source_before
    assert pickle.dumps(case.episodes) == original_episodes
    assert len(case.work["polls"]) == 16
    for index, (call, episode) in enumerate(zip(case.work["calls"], case.episodes, strict=True)):
        assert call["episode"] == episode and call["episode"] is not episode
        assert call["episode"].behavior_probabilities == episode.behavior_probabilities
        slot = case.report["training"][index]
        assert slot["episode"] == asdict(episode)
        assert slot["early_release_receipts"] == [] and slot["commit_validated"] is True
        assert slot["receipt"]["diagnostic"]["definition"] == TORCH_RECURRENT_ZERO_LOSS_ID
        assert all(step.reward > 0 for step in call["episode"].transitions)
        assert call["before"].completed_updates == 128 + index
        assert call["after"].completed_updates == 129 + index
        assert call["before"].training_examples == 1152 + index * (index + 1) // 2
        assert call["after"].training_examples == 1152 + (index + 1) * (index + 2) // 2
        assert call["owner"] is not case.source._learner
        assert call["owner"] is not runtime.actor._learner._learner
        early, committed = case.work["polls"][2 * index : 2 * index + 2]
        assert early[0] == early[1] == index and early[2] == ()
        assert committed[0] == index and committed[1] == index + 1
        assert len(committed[2]) == 1 and committed[2][0].update_number == index + 1
        assert committed[2][0].diagnostic.definition == TORCH_RECURRENT_ZERO_LOSS_ID
        assert committed[2][0].sample_id == committed[2][0].episode_id == episode.episode_id
        assert committed[2][0].event_id == f"momentum-release-{index}"
    assert sum(len(call["episode"].transitions) for call in case.work["calls"]) == 36
    prepared["guard"]()
    case.clock[0] = 60.0
    prepared["guard"]()
    assert case.work["budget"].started_at == case.work["budget"].last_clock == 10.0


def test_should_bound_control_by_remaining_original_deadline_without_renewal(case):
    prepared = run(case, deadline=30.0)
    budget = case.work["budget"]
    assert budget.budget.max_wall_seconds == 20.0
    assert budget.started_at == budget.last_clock == 10.0 and budget.updates_completed == 8
    prepared["guard"]()
    assert budget.started_at == budget.last_clock == 10.0


def test_should_keep_two_commits_and_original_source_when_third_native_stage_fails(case):
    case.work["fail_native"] = 3
    source_before = pickle.dumps(case.source.snapshot_state())
    with pytest.raises((ValueError, RuntimeError), match="atomic momentum-only update refused"):
        run(case)
    assert case.report["status"] == "incomplete"
    assert case.work["budget"].updates_completed == 2
    assert case.report["work"]["native_updates_committed"] == 2
    assert case.report["work"]["committed_training_rows"] == 3
    assert case.report["work"]["attempted_training_rows"] == 6
    assert case.report["work"]["failure_work_may_be_unobserved"] is True
    assert len(case.work["calls"]) == 3
    assert case.work["calls"][1]["after"].completed_updates == 130
    assert case.work["calls"][1]["after"].training_examples == 1155
    assert "after" not in case.work["calls"][2]
    assert len(case.work["runtimes"][0].applied_updates) == 2
    assert pickle.dumps(case.source.snapshot_state()) == source_before


@pytest.mark.parametrize("target", ["source", "input", "candidate", "budget", "receipt", "record"])
def test_should_refuse_changed_prior_authority_before_second_matching_release(case, target):
    changed: list[bool] = []

    def corrupt(before, receipts):
        if before != 1 or receipts or changed:
            return
        changed.append(True)
        runtime = case.work["runtimes"][0]
        if target == "source":
            case.source._learner.state.gradients["bias_output"] += 1
        elif target == "input":
            object.__setattr__(case.episodes[0].transitions[0], "reward", -99.0)
        elif target == "candidate":
            runtime._candidate._learner.state.parameters["bias_output"] += 1
        elif target == "budget":
            case.work["budget"].started_at += 1
        elif target == "receipt":
            object.__setattr__(runtime.applied_updates[0], "update_number", 99)
        else:
            case.report["training"][0]["episode"]["transitions"][0]["reward"] = 99.0

    case.work["after_poll"] = corrupt
    with pytest.raises((ValueError, RuntimeError)):
        run(case)
    assert changed == [True] and case.report["status"] == "incomplete"
    assert case.work["budget"].updates_completed == 1
    assert len(case.work["calls"]) == 1
    assert case.report["work"]["native_updates_committed"] == 1
    assert case.report["training"][0]["receipt"]["update_number"] == 1
    assert case.report["training"][0]["episode"]["transitions"][0]["reward"] == 0.1


@pytest.mark.parametrize("target", ["clock", "limits", "input"])
def test_should_keep_completed_control_guard_bound_to_original_clock_budget_and_inputs(
    case, target
):
    prepared = run(case)
    budget = case.work["budget"]
    if target == "clock":
        budget.last_clock += 1
    elif target == "limits":
        budget.budget = replace(budget.budget, max_training_updates=9)
    else:
        object.__setattr__(case.episodes[-1].transitions[0], "reward", -99.0)
    with pytest.raises((ValueError, RuntimeError)):
        prepared["guard"]()
    assert len(case.work["calls"]) == 8 and budget.updates_completed == 8


def test_should_keep_first_native_commit_when_caller_proof_rejects_after_poll(case):
    changed: list[bool] = []

    def reject_proof():
        raise ValueError("original caller reward proof changed after control commit")

    def invalidate_after_commit(before, receipts):
        if receipts and not changed:
            changed.append(True)
            case.work["on_boundary"] = reject_proof

    case.work["after_poll"] = invalidate_after_commit
    with pytest.raises((ValueError, RuntimeError), match="original caller reward proof"):
        run(case)
    assert changed == [True] and case.report["status"] == "incomplete"
    assert case.work["budget"].updates_completed == 1 and len(case.work["calls"]) == 1
    assert len(case.work["runtimes"][0].applied_updates) == 1
    assert case.report["training"][0]["update_committed"] is True
    assert case.report["training"][0]["receipt"]["update_number"] == 1
    assert case.report["work"]["native_updates_committed"] == 1


def test_should_refuse_prior_receipt_corruption_from_last_source_snapshot_callback(
    monkeypatch, case
):
    snapshot_state = Native.snapshot_state
    changed: list[bool] = []

    def snapshot_callback(owner):
        state = snapshot_state(owner)
        if (
            owner is case.source._learner
            and case.work.get("budget") is not None
            and case.work["budget"].updates_completed == 8
            and not changed
        ):
            changed.append(True)
            object.__setattr__(case.work["runtimes"][0].applied_updates[0], "update_number", 99)
        return state

    monkeypatch.setattr(Native, "snapshot_state", snapshot_callback)
    with pytest.raises((ValueError, RuntimeError)):
        run(case)
    assert changed == [True] and case.report["status"] == "incomplete"
    assert case.work["budget"].updates_completed == 8 and len(case.work["calls"]) == 8
    assert case.report["work"]["native_updates_committed"] == 8
    assert case.report["training"][0]["receipt"]["update_number"] == 1
