"""Joint54-row wakes through four real heads; no scientific or game outcome."""

from time import monotonic

import numpy as np
import pytest

from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.adapters.torch_feature_learner import TORCH_CE_ID, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.resnet50_variants import TORCH_PC_ENERGY_ID
from test_minigrid_train_catalogue import episode


@pytest.mark.parametrize("method", ("bp", "pc", "neutral_cpc", "gated_cpc"))
def test_should_train_all_catalogue_rows_after_labels_without_touching_source_or_actor(method):
    sources = tuple(
        episode(f"{context}-{cue}", context=context, cue=cue, actions=(2, 2, 0, 3, 1, 2, 5, 2, 2))
        for context in ("A", "B", "C_train")
        for cue in (0, 2)
    )
    plan = compose_catalogue_training(sources, 2)
    learner = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))
    )
    source_before = learner.snapshot_state()
    expected_inputs = tuple(
        (batch.features.copy(), batch.targets.copy()) for batch in plan.training
    )
    observed = []

    def observe(updates, state, arrived):
        assert state.completed_updates == updates
        assert state.training_examples == updates * 54
        observed.append(updates)
        if updates == 0:
            assert arrived is None
        else:
            assert arrived is not None and arrived.episode is None
            assert arrived.features.shape == (54, 296)
            # Observer receives owned transport arrays, not a future update's rows.
            arrived.features.fill(999)

    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=30), monotonic
    )
    stream = AttributionStream(
        learner,
        plan.training,
        (),
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=budget,
        observe_checkpoint=observe,
        training_only=True,
    )
    actor_before = stream._runtime.actor.snapshot_state().state
    stream.run_until(2)
    endpoint = stream._runtime.candidate_snapshot().state
    result = stream.finish()
    assert observed == [0, 1, 2]
    assert result["completed_updates"] == budget.updates_completed == 2
    assert result["training_examples"] == plan.row_exposures == 108
    assert result["num_classes"] == 7
    assert result["evaluation_scope"] == "training_only"
    assert result["checkpoints"] == [] and result["score_forward_calls"] == 0
    assert [r["arrived_at"] for r in result["applied_updates"]] == [1, 2]
    definition = TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID
    assert all(r["diagnostic"]["definition"] == definition for r in result["applied_updates"])
    assert endpoint.completed_updates == 2 and endpoint.training_examples == 108
    assert _same_state(learner.snapshot_state(), source_before)
    assert _same_state(stream._runtime.actor.snapshot_state().state, actor_before)
    for batch, (features, targets) in zip(plan.training, expected_inputs, strict=True):
        np.testing.assert_array_equal(batch.features, features)
        np.testing.assert_array_equal(batch.targets, targets)
    reader = learner.fork()
    reader.restore_state(endpoint)
    before_predict = reader.snapshot_state()
    probabilities = reader.predict(plan.training[0].features)
    assert probabilities.shape == (54, 7) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(54), rtol=0, atol=1e-6)
    assert _same_state(reader.snapshot_state(), before_predict)
    assert _same_state(learner.snapshot_state(), source_before)
