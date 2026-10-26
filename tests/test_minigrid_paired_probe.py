"""Paired MiniGrid probe transport, disposable updates and observer evidence.

These ports exercise the real probe runner without Torch or environments. Native
math is qualified separately; the observer must retain its original data/clock
and distinguish a local eight-update probe from its warm absolute counters.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import torch_paired_probe as adapter
from src.adapters import torch_recurrent_episode_batch as native
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
import src.app.toy_execution_budget as budgets
from test_minigrid_learning_pilot import Learner, same_state
from test_minigrid_probe_admission import packet
from test_paired_episode_stream import FakeNative, fake_snapshot
from test_torch_feature_rehearsal_pilot import Sampler


def original_data():
    originals = []
    for role in ("probe_train", "probe_test"):
        for cue in ("blue", "red"):
            episode = packet(cue, role).episode
            assert episode is not None
            originals.append(replace(episode, episode_id=f"minigrid-probe-source-{role}-{cue}"))
    training = tuple(
        pilot._packet(
            replace(
                originals[index % 2],
                episode_id=f"{originals[index % 2].episode_id}:copy:{index}",
            )
        )
        for index in range(16)
    )
    return training, tuple(pilot._packet(episode) for episode in originals[2:])


def probe_slot(point=1):
    return dict(
        checkpoint_updates=point,
        status="not_run",
        reason="not reached",
        report=None,
        charged_updates=None,
        source_unchanged=None,
        state_unchanged=None,
        budget_binding_preserved=None,
        outcomes=None,
        work_unknown=None,
        total_seconds=None,
    )


def forbidden(*args, **kwargs):
    raise AssertionError("preadmission reached native allocation, clock or copying")


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = dict(owners=[], trains=[], predictions=[])
    monkeypatch.setattr(FakeNative, "work", work)
    monkeypatch.setattr(
        FakeNative, "train_probe_episode_pair", FakeNative.train_episode_pair, raising=False
    )
    monkeypatch.setattr(adapter, "TorchRecurrentEpisodeBatchLearner", FakeNative)
    monkeypatch.setattr(adapter, "_same_state", same_state)
    monkeypatch.setattr(native, "_snapshot", fake_snapshot)
    monkeypatch.setattr(pilot, "_same_state", same_state)
    monkeypatch.setattr(pilot, "monotonic", lambda: 10.0)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    source = FakeNative(TorchRecurrentPolicy(148, 8, 7, 17))
    captured: dict[str, Any] = dict(work=work, source=source, budgets=[], probes=[])
    make_budget = pilot.ToyBudgetSession
    make_probe = adapter.TorchPairedProbeLearner

    def budget(*args, **kwargs):
        result = make_budget(*args, **kwargs)
        captured["budgets"].append(result)
        return result

    def probe(*args, **kwargs):
        result = make_probe(*args, **kwargs)
        captured["probes"].append(result)
        return result

    monkeypatch.setattr(pilot, "ToyBudgetSession", budget)
    monkeypatch.setattr(pilot, "TorchPairedProbeLearner", probe)
    checkpoint = replace(
        source.snapshot_state(),
        completed_updates=1,
        training_episodes=2,
        training_examples=18,
        optimizer_state={"momentum": 0.5},
        gradients={"weight": np.ones((1, 1), dtype=np.float32)},
    )
    captured.update(data=original_data(), slot=probe_slot(), state=checkpoint)
    return captured


def run(case, *, check_boundary=lambda: None, paired=True):
    return pilot._run_probe(
        case["slot"],
        {"source": case["source"]},
        case["state"],
        case["data"],
        sampler=Sampler(),
        deadline=100.0,
        check_boundary=check_boundary,
        paired=paired,
    )


def test_should_join_transport_without_concatenating_recurrent_episode_metadata():
    data = original_data()
    before = pickle.dumps(data)
    (training, evaluation), schedule = pilot._paired_probe_data(data)
    assert len(training) == 8 and len(evaluation) == 2 and len(schedule) == 16
    assert all(item.episode is None for item in (*training, *evaluation))
    assert all(
        len(episode.transitions) == 9 and episode.role == "probe_train" for episode in schedule
    )
    assert [episode.context for episode in schedule] == ["E-blue", "E-red"] * 8
    assert (
        sum(item.features.nbytes + item.targets.nbytes for item in (*training, *evaluation))
        == 193104
    )
    for index, joined in enumerate(training):
        pair = data[0][2 * index : 2 * index + 2]
        np.testing.assert_array_equal(
            joined.features, np.concatenate([item.features for item in pair])
        )
        np.testing.assert_array_equal(
            joined.targets, np.concatenate([item.targets for item in pair])
        )
        assert joined.row_ids == pair[0].row_ids + pair[1].row_ids
        assert not any(np.shares_memory(joined.features, item.features) for item in pair)
    for original, owned in zip(data[0], schedule, strict=True):
        assert owned == original.episode and owned is not original.episode
    assert pickle.dumps(data) == before


@pytest.mark.parametrize(
    "fault",
    [
        "training_count",
        "evaluation_count",
        "list",
        "metadata_none",
        "wrong_role",
        "reordered_cues",
        "reordered_evaluation",
        "short_episode",
        "width",
        "classes",
        "targets",
        "features",
        "duplicate_ids",
        "changed_repeated_source",
        "extra_episode",
    ],
)
def test_should_refuse_changed_original_recipe_before_native_allocation(monkeypatch, fault):
    training, evaluation = original_data()
    train, test = list(training), list(evaluation)
    if fault == "training_count":
        train.pop()
    elif fault == "evaluation_count":
        test.pop()
    elif fault == "metadata_none":
        train[0] = replace(train[0], episode=None)
    elif fault == "wrong_role":
        episode = train[0].episode
        assert episode is not None
        train[0] = pilot._packet(replace(episode, role="train"))
    elif fault == "reordered_cues":
        train[0], train[1] = train[1], train[0]
    elif fault == "reordered_evaluation":
        test.reverse()
    elif fault == "short_episode":
        episode = packet("blue", "probe_train").episode
        assert episode is not None
        transitions = episode.transitions[:8]
        transitions = (*transitions[:-1], replace(transitions[-1], terminal=True))
        train[0] = pilot._packet(replace(episode, transitions=transitions))
    elif fault == "width":
        train[0] = replace(train[0], features=train[0].features[:, :147].copy())
    elif fault == "classes":
        train[0] = replace(train[0], num_classes=6)
    elif fault == "targets":
        train[0].targets[0, 0] = (train[0].targets[0, 0] + 1) % 7
    elif fault == "features":
        train[0].features[0, 0] = 0.25
    elif fault == "duplicate_ids":
        train[2] = deepcopy(train[0])
    elif fault == "changed_repeated_source":
        episode = train[2].episode
        assert episode is not None
        first = replace(episode.transitions[0], action=(episode.transitions[0].action + 1) % 7)
        train[2] = pilot._packet(replace(episode, transitions=(first, *episode.transitions[1:])))
    elif fault == "extra_episode":
        assert train[0].episode is not None
        object.__setattr__(train[0].episode, "unexpected", True)
    data: Any = (tuple(train), tuple(test))
    if fault == "list":
        data = [data[0], data[1]]
    monkeypatch.setattr(pilot, "TorchPairedProbeLearner", forbidden)
    with pytest.raises(ValueError):
        pilot._paired_probe_data(data)


def test_should_run_eight_disposable_pairs_with_absolute_native_and_local_probe_clocks(case):
    before = pickle.dumps((case["source"].snapshot_state(), case["state"], case["data"]))
    complete = run(case)
    complete()
    slot, work = case["slot"], case["work"]
    result = slot["report"]
    assert slot["status"] == "complete" and slot["charged_updates"] == 8
    assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert slot["budget_binding_preserved"] is True
    assert result["completed_updates"] == 8 and result["completed_training_examples"] == 144
    assert result["score_forward_calls_completed"] == 8
    assert result["reserved_data_bytes"] == 193104 and result["num_classes"] == 7
    assert [point["updates"] for point in result["checkpoints"]] == [0, 2, 4, 8]
    assert "episode_scope" not in result
    assert len(work["trains"]) == 8 and all(
        record["owner"] is not case["source"] for record in work["trains"]
    )
    trained = [episode for record in work["trains"] for episode in record["episodes"]]
    assert [episode.episode_id for episode in trained] == [
        item.episode.episode_id for item in case["data"][0]
    ]
    assert all(
        episode.role == "probe_train" and len(episode.transitions) == 9 for episode in trained
    )
    final = work["trains"][-1]["owner"].snapshot_state()
    assert (final.completed_updates, final.training_episodes, final.training_examples) == (
        9,
        18,
        162,
    )
    assert case["source"].events == []
    assert before == pickle.dumps((case["source"].snapshot_state(), case["state"], case["data"]))
    scope = slot["paired_probe_scope"]
    assert scope["probe_updates"] == 8 and scope["probe_episode_exposures"] == 16
    assert scope["native_training_episodes_before"] == 2
    assert case["budgets"][0].budget.max_training_updates == 8


def test_should_preserve_default_single_episode_probe_recipe(case, monkeypatch):
    monkeypatch.setattr(Learner, "work", {})
    source = Learner(TorchRecurrentPolicy(148, 8, 7, 17))
    case.update(source=source, state=source.snapshot_state(), slot=probe_slot(0))
    pilot._run_probe(
        case["slot"],
        {"source": source},
        case["state"],
        case["data"],
        sampler=Sampler(),
        deadline=100.0,
        check_boundary=lambda: None,
    )
    result = case["slot"]["report"]
    assert result["completed_updates"] == 16 and result["completed_training_examples"] == 144
    assert [point["updates"] for point in result["checkpoints"]] == [0, 4, 8, 16]
    assert "episode_scope" in result and "paired_probe_scope" not in case["slot"]
    assert len(Learner.work["trains"]) == 16
    assert all(len(features) == 9 for _, features, _ in Learner.work["trains"])
    assert not case["probes"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_reject_non_bool_before_clock_and_copy(monkeypatch, value):
    monkeypatch.setattr(pilot, "monotonic", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    with pytest.raises(ValueError, match="bool"):
        pilot._run_probe(
            {},
            {},
            object(),
            ((), ()),
            sampler=None,
            deadline=100.0,
            check_boundary=forbidden,
            paired=value,
        )


@pytest.mark.parametrize("fault", ["point", "policy", "steps", "episodes", "rows", "legacy"])
def test_should_refuse_wrong_paired_header_before_probe_model(case, monkeypatch, fault):
    state: Any = case["state"]
    if fault == "point":
        case["slot"]["checkpoint_updates"] = True
    elif fault == "policy":
        state = replace(state, policy=replace(state.policy, seed=18))
    elif fault == "steps":
        state = replace(state, completed_updates=2)
    elif fault == "episodes":
        state = replace(state, training_episodes=3)
    elif fault == "rows":
        state = replace(state, training_examples=17)
    else:
        state = Learner(TorchRecurrentPolicy(148, 8, 7, 17)).snapshot_state()
    case["state"] = state
    monkeypatch.setattr(pilot, "TorchPairedProbeLearner", forbidden)
    with pytest.raises(ValueError):
        run(case)
    assert not case["work"]["trains"] and not case["budgets"]


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "state",
        "packet",
        "scope",
        "budget",
        "checkpoint_label",
        "native_rows_label",
    ],
)
def test_should_refuse_callback_corruption_before_another_native_pair(case, fault):
    fired = False

    def after_train(owner):
        nonlocal fired
        if fired:
            return
        fired = True
        if fault == "source":
            case["source"].state.parameters["weight"][0, 0] = 12
        elif fault == "state":
            case["state"].parameters["weight"][0, 0] = 12
        elif fault == "packet":
            case["data"][0][0].features[0, 0] = 0.5
        elif fault == "scope":
            case["slot"]["paired_probe_scope"]["probe_updates"] = 99
        elif fault == "checkpoint_label":
            case["slot"]["checkpoint_updates"] = 0
        elif fault == "native_rows_label":
            case["slot"]["native_training_examples_before"] = 0
        else:
            case["budgets"][0].budget = replace(case["budgets"][0].budget, max_training_updates=80)

    case["work"]["after_train"] = after_train
    with pytest.raises(ValueError):
        run(case)
    assert case["slot"]["status"] == "failed"
    assert len(case["work"]["trains"]) == 1
    result = case["slot"]["report"]
    assert result["completed_updates"] == 1
    assert result["completed_training_examples"] == 18
    assert "failed_native_work" not in result
    assert case["slot"]["charged_updates"] == 1
    assert case["slot"]["outcomes"] is None


def test_should_preserve_paid_unpublished_native_failure_without_local_charge(case):
    case["work"]["native_failure"] = True
    with pytest.raises(ValueError):
        run(case)
    slot = case["slot"]
    assert slot["charged_updates"] == 0
    assert slot["report"]["completed_updates"] == 0
    assert slot["report"]["score_forward_calls_completed"] == 2
    work = slot["report"]["failed_native_work"]["work"]
    assert work["optimizer_steps"] == 1 and work["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is True
    assert len(case["work"]["trains"]) == 1


def test_should_restore_paid_report_after_post_return_public_evidence_corruption(case, monkeypatch):
    real = pilot.run_fresh_task_probe
    returned = False
    original: dict[str, Any] = {}

    def probe(*args, **kwargs):
        nonlocal returned
        result = real(*args, **kwargs)
        original.update(deepcopy(result))
        returned = True
        return result

    def boundary():
        nonlocal returned
        if returned and case["slot"]["report"] is not None:
            returned = False
            case["slot"]["report"]["completed_training_examples"] = 0

    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    with pytest.raises(ValueError):
        run(case, check_boundary=boundary)
    assert case["slot"]["status"] == "failed" and case["slot"]["charged_updates"] == 8
    assert case["slot"]["report"] == original
    assert case["slot"]["outcomes"] is None
    assert len(case["work"]["trains"]) == 8


def test_should_refuse_late_source_snapshot_callback_mutating_paired_scope(case, monkeypatch):
    snapshot = case["source"].snapshot_state
    fired = False

    def readback():
        nonlocal fired
        result = snapshot()
        if not fired and case["slot"]["report"] is not None:
            fired = True
            case["slot"]["paired_probe_scope"]["probe_training_rows"] = 0
        return result

    monkeypatch.setattr(case["source"], "snapshot_state", readback)
    with pytest.raises(ValueError):
        run(case)
    assert fired and case["slot"]["status"] == "failed"
    assert case["slot"]["charged_updates"] == 8
    assert case["slot"]["report"]["completed_training_examples"] == 144
    assert case["slot"]["paired_probe_scope"]["probe_training_rows"] == 144
    assert case["slot"]["outcomes"] is None
    assert len(case["work"]["trains"]) == 8


@pytest.mark.parametrize("fault", ["charge", "point", "flag", "report", "outcomes"])
def test_should_recheck_preceding_paired_success_before_admitting_more_work(case, fault):
    complete = run(case)
    slot = case["slot"]
    if fault == "charge":
        slot["charged_updates"] = 0
    elif fault == "point":
        slot["checkpoint_updates"] = 0
    elif fault == "flag":
        slot["state_unchanged"] = False
    elif fault == "report":
        slot["report"]["completed_training_examples"] = 0
    else:
        slot["outcomes"]["ce_gain"] = 100.0
    with pytest.raises(ValueError):
        complete()
    assert len(case["work"]["trains"]) == 8
    assert case["budgets"][0].updates_completed == 8
