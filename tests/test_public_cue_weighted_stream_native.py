"""One native delayed weighted schedule case; literal rows, no task experiment."""

import pickle
from time import monotonic

import numpy as np

from src.adapters import torch_train_weighted_learner as weighted
from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters.minigrid_learning_pilot import _batch
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import episode_arrays
from src.shared.torch_runtime import require_torch
from test_public_cue_recurrent_learner import policy
from test_public_cue_weighted_stream import schedule
from test_torch_feature_learner import assert_tree, tensors


def test_should_match_direct_native_weighted_heads_through_delayed_stream_restore_and_sessions():
    torch = require_torch()
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    episodes = schedule()
    source = weighted.TorchTrainWeightedLearner(policy(), episodes)
    initial = source.snapshot_state()
    refs = [public.native.TorchRecurrentLearner(policy().native_policy) for _ in range(2)]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=3, max_wall_seconds=90), monotonic
    )
    checkpoints = {}

    def observe(point, state, batch):
        checkpoints[point] = state

    stream = AttributionStream(
        source,
        tuple(_batch(p) for p in episodes),
        (),
        training_only=True,
        label_delay=1,
        checkpoint_updates=(0, 2, 3),
        budget=budget,
        observe_checkpoint=observe,
    )
    stream.run_until(3)
    actual = stream.candidate_snapshot()
    result = stream.finish()
    for episode in episodes:
        index = int(episode.transitions[0].observation[1] == -1.0)
        refs[index].train_weighted_episode(episode, episode_class_weights(episode))
    assert [h.completed_updates for h in actual.heads] == [2, 1]
    assert [h.training_examples for h in actual.heads] == [8, 4]
    for i in range(2):
        assert_tree(torch, actual.heads[i], refs[i].snapshot_state())
    resumed = source.fork()
    resumed.restore_state(checkpoints[2])
    resumed.train_batch(*episode_arrays(episodes[2]))
    assert_tree(torch, resumed.snapshot_state(), actual)
    for i, episode in enumerate(episodes[:2]):
        rows, _ = episode_arrays(episode)
        np.testing.assert_array_equal(resumed.predict(rows), refs[i].predict(rows))
    session, reference = resumed.start_episode(), refs[0].start_episode()
    for row in episode_arrays(episodes[0])[0]:
        np.testing.assert_array_equal(
            session.predict(row[None, :]), reference.predict(row[None, :])
        )
    assert session.completed_steps == reference.completed_steps == 4
    assert_tree(torch, session.snapshot_state(), actual)
    forked = resumed.fork().snapshot_state()
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(tensors(torch, forked), tensors(torch, actual), strict=True)
    )
    assert _same_value(source.snapshot_state(), initial)
    assert _same_value(stream._runtime.actor.snapshot_state().state, initial)
    assert budget.updates_completed == 3 and budget.replay_examples_completed == 0
    assert len(result["applied_updates"]) == 3 and result["score_forward_calls"] == 0
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1]
