"""ONE prospective native observer case: nine physical SGD / 162 rows.

One TRAIN warm pair precedes eight disposable E probe pairs. No environment,
teacher, baseline rerun or scientific comparison is part of this qualification.
The parent controller, not this source-only test author, authorizes execution.
"""

from copy import deepcopy
import json
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import torch_paired_probe as adapter
from src.adapters import torch_recurrent_episode_batch as native
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch
from test_minigrid_learning_pilot import source_batch
from test_minigrid_paired_probe import original_data, probe_slot
from test_torch_feature_learner import assert_tree


@pytest.fixture
def physical_cost(monkeypatch, record_property):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    counts: dict[str, Any] = dict(steps=0, rows=0, active=False, active_rows=None, episodes=[])
    step = torch.optim.SGD.step
    warm_train = native.TorchRecurrentEpisodeBatchLearner.train_episode_pair
    probe_train = native.TorchRecurrentEpisodeBatchLearner.train_probe_episode_pair

    def measured_step(owner, *args, **kwargs):
        assert counts["active"] is True, "unbudgeted physical SGD"
        assert type(counts["active_rows"]) is int and counts["active_rows"] == 18
        counts["steps"] += 1
        counts["rows"] += counts["active_rows"]
        return step(owner, *args, **kwargs)

    def measured_train(operation, role, owner, episodes):
        assert counts["active_rows"] is None
        assert len(episodes) == 2 and all(episode.role == role for episode in episodes)
        counts["active_rows"] = sum(len(episode.transitions) for episode in episodes)
        counts["episodes"].append(deepcopy(episodes))
        try:
            return operation(owner, episodes)
        finally:
            counts["active_rows"] = None

    def warm(owner, episodes):
        return measured_train(warm_train, "train", owner, episodes)

    def probe(owner, episodes):
        return measured_train(probe_train, "probe_train", owner, episodes)

    monkeypatch.setattr(torch.optim.SGD, "step", measured_step)
    monkeypatch.setattr(native.TorchRecurrentEpisodeBatchLearner, "train_episode_pair", warm)
    monkeypatch.setattr(native.TorchRecurrentEpisodeBatchLearner, "train_probe_episode_pair", probe)
    yield torch, counts
    record_property("physical_sgd_steps", counts["steps"])
    record_property("physical_training_rows", counts["rows"])
    assert (counts["steps"], counts["rows"]) == (9, 162)


def test_should_run_actual_paired_minigrid_probe_observer_at_warm_checkpoint(
    physical_cost,
    monkeypatch,
    record_property,
):
    torch, counts = physical_cost
    source = native.TorchRecurrentEpisodeBatchLearner(TorchRecurrentPolicy(148, 8, 7, 17))
    source_before = source.snapshot_state()
    candidate = source.fork()
    blue = source_batch("A", "blue").episode
    red = source_batch("A", "red").episode
    assert blue is not None and red is not None
    episodes = (blue, red)
    assert all(len(episode.transitions) == 9 for episode in episodes)
    counts["active"] = True
    try:
        candidate.train_episode_pair(episodes)
    finally:
        counts["active"] = False
    state = candidate.snapshot_state()
    state_before = deepcopy(state)
    input_data = original_data()
    input_before = deepcopy(input_data)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = cast(tuple[Any, ...], deepcopy(np.random.get_state()))
    owned: list[Any] = []
    probe_factory = adapter.TorchPairedProbeLearner
    fork = adapter.TorchPairedProbeLearner.fork

    def create(*args, **kwargs):
        result = probe_factory(*args, **kwargs)
        owned.append(result)
        return result

    def own_fork(owner):
        result = fork(owner)
        owned.append(result)
        return result

    monkeypatch.setattr(pilot, "TorchPairedProbeLearner", create)
    monkeypatch.setattr(adapter.TorchPairedProbeLearner, "fork", own_fork)
    slot = probe_slot(1)
    with ProcessRssSampler() as sampler:
        counts["active"] = True
        try:
            complete = pilot._run_probe(
                slot,
                {"source": source},
                state,
                input_data,
                sampler=sampler,
                deadline=pilot.monotonic() + 35.0,
                check_boundary=lambda: None,
                paired=True,
            )
        finally:
            counts["active"] = False
        complete()
    assert slot["status"] == "complete" and slot["charged_updates"] == 8
    result = slot["report"]
    assert (result["completed_updates"], result["completed_training_examples"]) == (8, 144)
    assert result["score_forward_calls_completed"] == 8
    assert result["reserved_data_bytes"] == 193104
    assert [point["updates"] for point in result["checkpoints"]] == [0, 2, 4, 8]
    assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert slot["budget_binding_preserved"] is True
    assert len(counts["episodes"]) == 9
    assert [episode.episode_id for pair in counts["episodes"][1:] for episode in pair] == [
        packet.episode.episode_id for packet in input_data[0]
    ]
    final = owned[-1].snapshot_state()
    assert (final.completed_updates, final.training_episodes, final.training_examples) == (
        9,
        18,
        162,
    )
    assert (source.completed_updates, source.training_episodes, source.training_examples) == (
        0,
        0,
        0,
    )
    assert_tree(torch, source.snapshot_state(), source_before)
    assert_tree(torch, candidate.snapshot_state(), state_before)
    assert_tree(torch, state, state_before)
    for original, before in zip(
        (*input_data[0], *input_data[1]), (*input_before[0], *input_before[1]), strict=True
    ):
        np.testing.assert_array_equal(original.features, before.features)
        np.testing.assert_array_equal(original.targets, before.targets)
        assert original.episode == before.episode
    assert torch.equal(torch.get_rng_state(), torch_rng)
    after_numpy = cast(tuple[Any, ...], np.random.get_state())
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
    record_property("paired_probe_slot", json.dumps(slot, allow_nan=False))
