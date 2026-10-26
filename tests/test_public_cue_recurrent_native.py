"""One separately budgeted current-native parity fixture, not a task experiment."""

from copy import deepcopy
import pickle
from time import monotonic

import numpy as np

from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.adapters.minigrid_learning_pilot import _batch
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_public_cue_recurrent_learner import data, episode, policy
from test_torch_feature_learner import assert_tree, tensors


def test_should_match_direct_native_heads_full_continuation_and_real_episode_activation():
    torch = require_torch()
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = pickle.dumps(np.random.get_state())
    p = policy()
    source = port.PublicCueRecurrentLearner(p)
    initial = source.snapshot_state()
    refs = [TorchRecurrentLearner(p.native_policy) for _ in range(2)]
    stream = AttributionStream(
        source,
        tuple(_batch(episode(c)) for c in ("blue", "red")),
        (),
        training_only=True,
        label_delay=1,
        checkpoint_updates=(0, 2),
        budget=ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=2, max_wall_seconds=30), monotonic
        ),
    )
    stream.run_until(2)
    saved = stream.candidate_snapshot()
    result = stream.finish()
    for index, cue in enumerate(("blue", "red")):
        refs[index].train_batch(*data(cue))
        assert_tree(torch, saved.heads[index], refs[index].snapshot_state())
    owner = source.fork()
    owner.restore_state(saved)
    child = owner.fork()
    child.train_batch(*data())
    refs[0].train_batch(*data())
    assert_tree(torch, child.snapshot_state().heads[0], refs[0].snapshot_state())
    restored = port.PublicCueRecurrentLearner(p)
    restored.restore_state(child.snapshot_state())
    assert_tree(torch, restored.snapshot_state(), child.snapshot_state())
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(
            tensors(torch, restored.snapshot_state()),
            tensors(torch, child.snapshot_state()),
            strict=True,
        )
    )
    for index, cue in enumerate(("blue", "red")):
        x, _ = data(cue)
        np.testing.assert_array_equal(child.predict(x), refs[index].predict(x))
        np.testing.assert_array_equal(restored.predict(x), refs[index].predict(x))
    a, b = child.start_episode(), refs[0].start_episode()
    before = deepcopy(child.snapshot_state())
    for row in data()[0]:
        np.testing.assert_array_equal(a.predict(row[None, :]), b.predict(row[None, :]))
    assert a.completed_steps == b.completed_steps == 2
    assert_tree(torch, a.snapshot_state(), before)
    assert _same_value(source.snapshot_state(), initial) and _same_value(
        stream._runtime.actor.snapshot_state().state, initial
    )
    assert len(result["applied_updates"]) == 2
    assert (
        torch.equal(torch.get_rng_state(), torch_rng)
        and pickle.dumps(np.random.get_state()) == numpy_rng
    )
