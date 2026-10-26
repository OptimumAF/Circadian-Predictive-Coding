"""Explicit paired updates through the ordinary delayed TRAIN array stream.

Native paired loss/state mathematics are already qualified. These deterministic
ports test schedule binding, release/accounting, ownership and failed prefixes.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_learnability as scoring
from src.adapters import torch_recurrent_episode_batch as native
from src.adapters import torch_train_episode_batch as paired
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy, TorchRecurrentSnapshot
from src.app.coupled_attribution import AttributionBatch
import src.app.toy_execution_budget as budgets
from src.core.demonstration_episode import episode_arrays
from test_minigrid_learning_pilot import same_state, source_batch
from test_torch_feature_rehearsal_pilot import Sampler, forbidden

METHOD = "recurrent_paired_bc"


def schedule(pairs=4):
    blue, red = source_batch("A", "blue").episode, source_batch("A", "red").episode
    assert blue is not None and red is not None
    bases = (blue, red)
    return tuple(
        replace(bases[index % 2], episode_id=f"paired-{index}-{('blue', 'red')[index % 2]}")
        for index in range(2 * pairs)
    )


def arrays(episodes, index=0):
    values = [episode_arrays(episode) for episode in episodes[2 * index : 2 * index + 2]]
    return np.concatenate([value[0] for value in values]), np.concatenate(
        [value[1] for value in values]
    )


def packets(episodes):
    result = []
    for index in range(len(episodes) // 2):
        features, targets = arrays(episodes, index)
        result.append(
            AttributionBatch(
                "train",
                "A",
                tuple(f"paired-{index}:row:{row}" for row in range(18)),
                features,
                targets,
                None,
                7,
            )
        )
    return tuple(result)


def fake_snapshot(state):
    if type(state) is not native.TorchRecurrentEpisodeBatchSnapshot:
        raise ValueError("fake native boundary requires the new paired snapshot type")
    return deepcopy(state)


class FakeNative:
    work: dict[str, Any] = {}

    def __init__(self, policy):
        self.policy = deepcopy(policy)
        self.state = native.TorchRecurrentEpisodeBatchSnapshot(
            1,
            deepcopy(policy),
            {"weight": np.zeros((1, 1), dtype=np.float32)},
            {"momentum": 0.0},
            {"weight": None},
            0,
            0,
            0,
        )
        self.calls = []
        self.events = []
        self.work["owners"].append(self)

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.events.append("restore")
        self.state = fake_snapshot(state)

    def fork(self):
        self.events.append("fork")
        if self.work.get("identity_fork"):
            return self
        child = FakeNative(self.policy)
        child.state = deepcopy(self.state)
        return child

    def train_episode_pair(self, episodes):
        record = dict(owner=self, episodes=deepcopy(episodes), before=self.snapshot_state())
        self.calls.append(record)
        self.work["trains"].append(record)
        work = dict(
            forward_calls_attempted=2,
            forward_calls_completed=2,
            forward_rows_attempted=18,
            forward_rows_completed=18,
            backward_calls_attempted=1,
            backward_calls_completed=1,
            backward_rows_attempted=18,
            backward_rows_completed=18,
            optimizer_steps_attempted=1,
            optimizer_steps=1,
            published_updates=1,
            failure_work_may_be_unobserved=False,
        )
        if self.work.get("native_failure"):
            work.update(published_updates=0, failure_work_may_be_unobserved=True)
            raise native.EpisodePairUpdateFailed("paid unpublished native pair refused", work)
        self.state = replace(
            self.state,
            parameters={"weight": self.state.parameters["weight"] + 0.1},
            optimizer_state={"momentum": self.state.optimizer_state["momentum"] + 0.5},
            gradients={"weight": np.full((1, 1), 0.25, dtype=np.float32)},
            completed_updates=self.state.completed_updates + 1,
            training_episodes=self.state.training_episodes + 2,
            training_examples=self.state.training_examples + 18,
        )
        if self.work.get("bad_counter"):
            self.state = replace(self.state, training_examples=self.state.training_examples + 1)
        callback = self.work.get("after_train")
        if callback:
            callback(self)
        diagnostic = native.EpisodePairDiagnostic(
            native.TORCH_RECURRENT_EPISODE_PAIR_CE_ID,
            0.5,
            tuple(episode.episode_id for episode in episodes),
            (0.4, 0.6),
            work,
        )
        if self.work.get("bad_objective"):
            object.__setattr__(diagnostic, "definition", "wrong-objective")
        return diagnostic

    def predict(self, features):
        self.work["predictions"].append((self, features.copy()))
        callback = self.work.get("after_predict")
        if callback:
            callback(self)
        return np.full(
            (len(features), self.policy.num_classes), 1 / self.policy.num_classes, dtype=np.float64
        )


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = dict(owners=[], trains=[], predictions=[])
    monkeypatch.setattr(FakeNative, "work", work)
    monkeypatch.setattr(paired, "TorchRecurrentEpisodeBatchLearner", FakeNative)
    monkeypatch.setattr(paired, "_same_state", same_state)
    monkeypatch.setattr(native, "_snapshot", fake_snapshot)
    monkeypatch.setattr(shared, "_same_state", same_state)
    monkeypatch.setattr(scoring, "_same_state", same_state)
    monkeypatch.setattr(shared, "monotonic", lambda: 10.0)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    policy = TorchRecurrentPolicy(148, 8, 7, 17)
    episodes = schedule()
    learner = paired.TorchTrainEpisodeBatchLearner(policy, episodes)
    return SimpleNamespace(work=work, policy=policy, episodes=episodes, learner=learner)


def test_should_release_the_exact_next_two_original_episodes_once_per_array_update(case):
    original = pickle.dumps(case.episodes)
    for index in range(4):
        features, targets = arrays(case.episodes, index)
        diagnostic = case.learner.train_batch(features, targets)
        assert diagnostic.definition == native.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
        call = case.work["trains"][-1]
        assert call["episodes"] == case.episodes[2 * index : 2 * index + 2]
        assert all(
            owned is not supplied
            for owned, supplied in zip(
                call["episodes"], case.episodes[2 * index : 2 * index + 2], strict=True
            )
        )
        state = case.learner.snapshot_state()
        assert (state.completed_updates, state.training_episodes, state.training_examples) == (
            index + 1,
            2 * (index + 1),
            18 * (index + 1),
        )
        features.fill(99)
        targets.fill(99)
    assert pickle.dumps(case.episodes) == original
    with pytest.raises(ValueError):
        case.learner.train_batch(*arrays(case.episodes))
    assert len(case.work["trains"]) == 4


@pytest.mark.parametrize("fault", ["feature", "target", "dtype", "shape", "nonfinite"])
def test_should_refuse_wrong_array_binding_before_native_training(case, fault):
    features, targets = arrays(case.episodes)
    if fault == "feature":
        features[0, 0] += 0.1
    elif fault == "target":
        targets[0, 0] = 2
    elif fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "shape":
        features = features[:9]
    else:
        features[0, 0] = float("nan")
    with pytest.raises(ValueError):
        case.learner.train_batch(features, targets)
    assert case.work["trains"] == []


@pytest.mark.parametrize("fault", ["role", "odd", "list", "classes", "width", "extra_field"])
def test_should_refuse_invalid_schedule_before_creating_native_resource(case, fault):
    episodes: Any = deepcopy(case.episodes)
    if fault == "role":
        episodes = (replace(episodes[0], role="probe_train"), *episodes[1:])
    elif fault == "odd":
        episodes = episodes[:-1]
    elif fault == "list":
        episodes = list(episodes)
    elif fault == "classes":
        object.__setattr__(episodes[0], "num_classes", 3)
    elif fault == "width":
        transition = episodes[0].transitions[0]
        object.__setattr__(transition, "observation", transition.observation[:-1])
    else:
        object.__setattr__(episodes[0], "unrecorded", True)
    count = len(case.work["owners"])
    with pytest.raises(ValueError):
        paired.TorchTrainEpisodeBatchLearner(case.policy, episodes)
    assert len(case.work["owners"]) == count


@pytest.mark.parametrize("fault", ["schedule", "seal", "native", "counter", "objective"])
def test_should_refuse_changed_schedule_or_full_native_owner_before_next_update(case, fault):
    if fault == "schedule":
        object.__setattr__(case.learner._episodes[0], "role", "final_test")
    elif fault == "seal":
        case.learner._schedule_binding = tuple(list(case.learner._schedule_binding))
    elif fault == "native":
        case.learner._learner.state.parameters["weight"][0, 0] += 1
    elif fault == "counter":
        case.learner._learner.state = replace(case.learner._learner.state, training_episodes=1)
    else:
        case.learner._objective_id = "wrong-objective"
    with pytest.raises(ValueError):
        case.learner.train_batch(*arrays(case.episodes))
    assert case.work["trains"] == []


def test_should_fork_and_restore_only_new_snapshot_without_schedule_or_state_alias(case):
    case.learner.train_batch(*arrays(case.episodes))
    state = case.learner.snapshot_state()
    child = case.learner.fork()
    child.train_batch(*arrays(case.episodes, 1))
    assert case.learner.completed_updates == 1 and child.completed_updates == 2
    assert child._episodes is not case.learner._episodes
    assert same_state(case.learner.snapshot_state(), state)
    child.restore_state(state)
    assert same_state(child.snapshot_state(), state)
    legacy_state = TorchRecurrentSnapshot(
        1,
        state.policy,
        state.parameters,
        state.optimizer_state,
        state.gradients,
        state.completed_updates,
        state.training_examples,
    )
    with pytest.raises(ValueError):
        child.restore_state(legacy_state)
    state.parameters["weight"][0, 0] += 100
    assert not same_state(child.snapshot_state(), state)


@pytest.mark.parametrize("fault", ["native_failure", "bad_counter", "bad_objective"])
def test_should_refuse_paid_native_failure_without_hiding_it_with_restore(case, fault):
    case.work[fault] = True
    native_owner = case.learner._learner
    before = native_owner.snapshot_state()
    with pytest.raises((ValueError, native.EpisodePairUpdateFailed)) as error:
        case.learner.train_batch(*arrays(case.episodes))
    assert "restore" not in native_owner.events
    assert len(case.work["trains"]) == 1
    if fault == "native_failure":
        assert isinstance(error.value, native.EpisodePairUpdateFailed)
        assert error.value.work["optimizer_steps"] == 1
        assert error.value.work["published_updates"] == 0
        assert same_state(native_owner.snapshot_state(), before)
    else:
        assert native_owner.snapshot_state().completed_updates == 1


def test_should_run_four_delayed_pairs_on_existing_training_only_stream_with_original_receipts(
    case,
):
    row = shared._row(METHOD)
    initial = case.learner.snapshot_state()
    training = packets(case.episodes)
    result = shared._train_arm(
        row,
        training,
        Sampler(),
        100.0,
        lambda: None,
        learner_factory=lambda method: case.learner,
        checkpoint_updates=(0, 4),
    )
    assert row["status"] == "trained" and row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["charged_updates"] == 4 and row["native_training_examples"] == 72
    assert row["native_training_episodes"] == 8
    assert [point for point, _ in result["states"]] == [0, 4]
    assert result["states"][-1][1].training_episodes == 8
    assert row["stream_result"]["evaluation_scope"] == "training_only"
    assert row["stream_result"]["score_forward_calls"] == 0
    assert len(row["stream_result"]["applied_updates"]) == 4
    assert all(
        receipt["diagnostic"]["definition"] == native.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
        for receipt in row["stream_result"]["applied_updates"]
    )
    assert all(
        packet.episode is None and packet.role == "train" and packet.features.shape == (18, 148)
        for packet in training
    )
    assert same_state(case.learner.snapshot_state(), initial)
    result["guard"]()
    row["stream_result"]["applied_updates"][0]["diagnostic"]["value"] = 999.0
    with pytest.raises(ValueError):
        result["guard"]()


def test_should_score_original_six_train_episodes_with_new_pair_counters_without_training(case):
    case.learner.train_batch(*arrays(case.episodes))
    checkpoint = case.learner.snapshot_state()
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    budget = budgets.ToyBudgetSession(
        budgets.ToyExecutionBudget(max_training_updates=0, max_wall_seconds=30), lambda: 10.0
    )
    before = pickle.dumps(case.learner.snapshot_state())
    report = scoring._score_checkpoint(
        case.learner,
        checkpoint,
        method=METHOD,
        updates=1,
        sources=sources,
        budget=budget,
        check_boundary=lambda: None,
    )
    assert report["status"] == "complete", report["reason"]
    assert len(report["scores"]) == 6
    assert report["work"]["score_forward_calls_completed"] == 6
    assert report["work"]["score_rows_completed"] == 54
    assert len(case.work["trains"]) == 1
    assert pickle.dumps(case.learner.snapshot_state()) == before
    assert budget.updates_completed == 0


def test_should_keep_owned_schedule_when_callers_replace_original_episode_metadata(case):
    expected = deepcopy(case.episodes[:2])
    features, targets = arrays(case.episodes)
    object.__setattr__(case.episodes[0], "role", "final_test")
    object.__setattr__(case.episodes[1], "episode_id", "replaced-outside-owner")
    case.learner.train_batch(features, targets)
    assert case.work["trains"][0]["episodes"] == expected
    assert all(episode.role == "train" for episode in case.work["trains"][0]["episodes"])


def test_should_refuse_identity_fork_before_returning_shared_trainable_owner(case):
    case.work["identity_fork"] = True
    with pytest.raises(ValueError):
        case.learner.fork()
    assert case.work["trains"] == []


@pytest.mark.parametrize("fault", ["legacy_type", "episodes", "rows"])
def test_should_refuse_wrong_pair_checkpoint_clocks_before_raw_prediction(case, fault):
    case.learner.train_batch(*arrays(case.episodes))
    checkpoint = case.learner.snapshot_state()
    if fault == "legacy_type":
        checkpoint = TorchRecurrentSnapshot(
            1,
            checkpoint.policy,
            checkpoint.parameters,
            checkpoint.optimizer_state,
            checkpoint.gradients,
            1,
            18,
        )
    elif fault == "episodes":
        checkpoint = replace(checkpoint, training_episodes=1)
    else:
        checkpoint = replace(checkpoint, training_examples=9)
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    budget = budgets.ToyBudgetSession(
        budgets.ToyExecutionBudget(max_training_updates=0), lambda: 10.0
    )
    # Admission may fail directly or be represented by the scorer's failed
    # result; either form must precede native forks, prediction and more SGD.
    owners = len(case.work["owners"])
    try:
        report = scoring._score_checkpoint(
            case.learner,
            checkpoint,
            method=METHOD,
            updates=1,
            sources=sources,
            budget=budget,
            check_boundary=lambda: None,
        )
    except ValueError:
        pass
    else:
        assert report["status"] == "failed"
        assert report["work"]["score_forward_calls_attempted"] == 0
    assert len(case.work["owners"]) == owners
    assert case.work["predictions"] == [] and len(case.work["trains"]) == 1


def test_should_keep_paid_raw_prefix_when_a_prediction_changes_full_paired_state(case):
    case.learner.train_batch(*arrays(case.episodes))
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    budget = budgets.ToyBudgetSession(
        budgets.ToyExecutionBudget(max_training_updates=0), lambda: 10.0
    )

    def corrupt(owner):
        owner.state.parameters["weight"][0, 0] += 1

    case.work["after_predict"] = corrupt
    report = scoring._score_checkpoint(
        case.learner,
        case.learner.snapshot_state(),
        method=METHOD,
        updates=1,
        sources=sources,
        budget=budget,
        check_boundary=lambda: None,
    )
    assert report["status"] == "failed"
    assert report["work"]["score_forward_calls_attempted"] == 1
    assert len(case.work["predictions"]) == 1 and len(case.work["trains"]) == 1
    assert budget.updates_completed == 0


@pytest.mark.parametrize("fault", ["work", "preceding_score"])
def test_should_refuse_tampered_raw_evidence_and_keep_original_observed_costs(
    monkeypatch, case, fault
):
    case.learner.train_batch(*arrays(case.episodes))
    public: list[dict[str, Any]] = []

    def capture_report(*args, **kwargs):
        record = dict(*args, **kwargs)
        if kwargs.get("method") == METHOD and "scores" in kwargs:
            public.append(record)
            # The construction spy exists only long enough to capture the real
            # public report; all subsequent type and value checks use builtins.
            monkeypatch.delattr(scoring, "dict")
        return record

    def corrupt(owner):
        if len(case.work["predictions"]) == 2:
            if fault == "work":
                public[0]["work"]["score_rows_completed"] = 999
            else:
                public[0]["scores"][0]["ce"] = 999.0

    monkeypatch.setattr(scoring, "dict", capture_report, raising=False)
    case.work["after_predict"] = corrupt
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    budget = budgets.ToyBudgetSession(
        budgets.ToyExecutionBudget(max_training_updates=0), lambda: 10.0
    )
    report = scoring._score_checkpoint(
        case.learner,
        case.learner.snapshot_state(),
        method=METHOD,
        updates=1,
        sources=sources,
        budget=budget,
        check_boundary=lambda: None,
    )
    assert report["status"] == "failed" and len(case.work["predictions"]) == 2
    assert report["work"]["score_forward_calls_completed"] == 2
    assert report["work"]["score_rows_completed"] == 18
    assert len(report["scores"]) == 1 and report["scores"][0]["ce"] != 999.0
    assert len(case.work["trains"]) == 1 and budget.updates_completed == 0


def test_should_bound_schedule_length_before_traversing_copying_or_creating_native_resources(
    monkeypatch, case
):
    class Opaque:
        def __getattribute__(self, name):
            raise AssertionError("overlong schedule entry was traversed")

    def forbidden(*args, **kwargs):
        pytest.fail("overlong schedule reached copy or native construction")

    monkeypatch.setattr(paired, "copy_episode", forbidden)
    monkeypatch.setattr(paired, "TorchRecurrentEpisodeBatchLearner", forbidden)
    count = len(case.work["owners"])
    with pytest.raises(ValueError):
        paired.TorchTrainEpisodeBatchLearner(case.policy, cast(Any, (Opaque(),) * 130))
    assert len(case.work["owners"]) == count and case.work["trains"] == []


def test_should_refuse_final_training_readback_charge_tamper_and_retain_all_paid_pairs(
    monkeypatch, case
):
    row = shared._row(METHOD)
    readback = shared._readback
    calls = [0]

    def final_corruption(*args, **kwargs):
        readback(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            row["charged_updates"] = 0

    monkeypatch.setattr(shared, "_readback", final_corruption)
    with pytest.raises(ValueError):
        shared._train_arm(
            row,
            packets(case.episodes),
            Sampler(),
            100.0,
            lambda: None,
            learner_factory=lambda method: case.learner,
            checkpoint_updates=(0, 4),
        )
    assert calls[0] >= 2
    assert row["status"] == "incomplete"
    assert row["charged_updates"] == 4 and row["native_training_examples"] == 72
    assert row["native_training_episodes"] == 8
    assert len(case.work["trains"]) == 4
    assert len(row["stream_result"]["applied_updates"]) == 4
    assert len(row["partial_stream"]["applied_updates"]) == 4


def test_should_refuse_overlong_transition_tuple_before_copy_or_traversal(monkeypatch, case):
    class Opaque:
        def __getattribute__(self, name):
            raise AssertionError("overlong transitions were traversed")

    oversized = deepcopy(case.episodes[0])
    object.__setattr__(oversized, "transitions", (Opaque(),) * 17)
    monkeypatch.setattr(paired, "copy_episode", forbidden)
    monkeypatch.setattr(paired, "TorchRecurrentEpisodeBatchLearner", forbidden)
    with pytest.raises(ValueError):
        paired.TorchTrainEpisodeBatchLearner(case.policy, (oversized, case.episodes[1]))
