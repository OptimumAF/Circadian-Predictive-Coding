"""One bounded native integration fixture; two LS fits/four TRAIN rows only."""

from copy import deepcopy
from time import monotonic
import pickle
import random

import numpy as np
import pytest

from src.adapters import minigrid_recurrent_readout as readout
from src.adapters.minigrid_learning_pilot import _batch
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_public_cue_recurrent_learner import data, episode


def test_actual_stream_fit_and_prefix_sessions_preserve_source_actor_and_rng():
    torch = require_torch()
    torch_rng = torch.get_rng_state().clone()
    numpy_rng, python_rng = pickle.dumps(np.random.get_state()), random.getstate()
    source = readout.RecurrentReadoutLearner()
    original = source.snapshot_state()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=90), monotonic
    )
    stream = AttributionStream(
        source,
        tuple(_batch(episode(cue)) for cue in ("blue", "red")),
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    actor = deepcopy(stream._runtime.actor.snapshot_state())
    stream.run_until(2)
    saved = stream.candidate_snapshot()
    report = stream.finish()
    assert report["completed_updates"] == budget.updates_completed == 2
    assert report["training_examples"] == 4 and len(report["applied_updates"]) == 2
    assert report["score_forward_calls"] == 0
    assert saved.fit_updates == (1, 1) and saved.fit_rows == (2, 2)
    assert saved.native.completed_updates == saved.native.training_examples == 0
    reader = source.fork()
    reader.restore_state(saved)
    sibling = reader.fork()
    for cue in ("blue", "red"):
        features, targets = data(cue)
        expected = reader.predict(features)
        assert expected.dtype == np.float64 and expected.shape == (2, 7)
        assert np.array_equal(np.argmax(expected, axis=1), targets[:, 0].astype(np.int64))
        session = sibling.start_episode()
        for i in range(2):
            actual = session.predict(features[i : i + 1])
            np.testing.assert_allclose(actual, expected[i : i + 1], rtol=0, atol=1e-12)
        assert session.completed_steps == 2
        assert session.feature_rows_attempted == session.feature_rows_returned == 3
    assert _same_state(reader.snapshot_state(), saved)
    assert _same_state(sibling.snapshot_state(), saved)
    assert _same_state(source.snapshot_state(), original)
    assert _same_state(stream._runtime.actor.snapshot_state(), actor)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert numpy_rng == pickle.dumps(np.random.get_state()) and python_rng == random.getstate()
    # Equal replacement of a real tensor is still an ownership violation before inference.
    parameters = sibling._state.native.heads[0].parameters
    parameters["weight_input"] = parameters["weight_input"].clone()
    with pytest.raises(ValueError, match="replaced"):
        sibling.snapshot_state()
