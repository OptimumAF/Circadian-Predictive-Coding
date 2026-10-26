"""One bounded native objective-continuation parity case, not task acquisition."""

import pickle
from time import monotonic

import numpy as np

from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.adapters.minigrid_learning_pilot import _batch
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import episode_arrays
from src.shared.torch_runtime import require_torch
from test_public_cue_recurrent_learner import policy
from test_public_cue_weighted_train import packet
from test_torch_feature_learner import assert_tree, tensors


def test_should_match_existing_native_weighted_updates_with_mean_ce_momentum_and_frozen_actor():
    torch = require_torch()
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    source = port.PublicCueRecurrentLearner(policy())
    initial = source.snapshot_state()
    refs = [port.native.TorchRecurrentLearner(policy().native_policy) for _ in range(2)]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=1, max_wall_seconds=40),
        monotonic,
    )
    stream = AttributionStream(
        source,
        (_batch(packet()),),
        (),
        training_only=True,
        label_delay=1,
        checkpoint_updates=(0, 1),
        budget=budget,
    )
    stream.run_until(1)
    owner = source.fork()
    owner.restore_state(stream.candidate_snapshot())
    result = stream.finish()
    refs[0].train_batch(*episode_arrays(packet()))
    for cue in ("red", "blue", "blue"):
        episode = packet(cue)
        selected = int(cue == "red")
        other = owner.snapshot_state().heads[1 - selected]
        actual = owner.train_class_balanced_episode(episode)
        expected = refs[selected].train_weighted_episode(episode, episode_class_weights(episode))
        assert actual.definition == TORCH_RECURRENT_WEIGHTED_CE_ID
        assert actual == expected
        state = owner.snapshot_state()
        assert_tree(torch, state.heads[selected], refs[selected].snapshot_state())
        assert_tree(torch, state.heads[1 - selected], other)
    frozen = owner.snapshot_state()
    assert frozen.completed_updates == 4 and frozen.training_examples == 16
    assert [h.completed_updates for h in frozen.heads] == [3, 1]
    for index, cue in enumerate(("blue", "red")):
        features, _ = episode_arrays(packet(cue))
        np.testing.assert_array_equal(owner.predict(features), refs[index].predict(features))
    own_session, ref_session = owner.start_episode(), refs[0].start_episode()
    for row in episode_arrays(packet())[0]:
        np.testing.assert_array_equal(
            own_session.predict(row[None, :]), ref_session.predict(row[None, :])
        )
    assert own_session.completed_steps == ref_session.completed_steps == 4
    assert_tree(torch, own_session.snapshot_state(), frozen)
    detached = owner.fork().snapshot_state()
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(
            tensors(torch, detached),
            tensors(torch, frozen),
            strict=True,
        )
    )
    assert _same_value(source.snapshot_state(), initial)
    assert _same_value(stream._runtime.actor.snapshot_state().state, initial)
    assert len(result["applied_updates"]) == 1
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1]
