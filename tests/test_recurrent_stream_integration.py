"""Native recurrent packets use the ordinary stream and actual live evaluator.

These tiny bounded integration checks establish composition/isolation, not
acquisition, retention, generalization or comparative learning benefit.
"""

from typing import Any

import numpy as np
import pytest

from src.adapters.key_door_pilot import _collect, layouts, replay_policy
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.app.coupled_attribution import AttributionStream
from src.app.key_door_policy_evaluation import run_key_door_episode
from src.app.line_policy_evaluation import EvaluationIdentity
from src.app.stream_rehearsal import StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import EpisodeReplaySchema
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree


@pytest.mark.parametrize("replay_enabled", [False, True])
def test_should_rehearse_native_recurrent_episodes_then_score_and_act_without_final_feedback(
    monkeypatch, replay_enabled
):
    torch = require_torch()
    known, _, _, heldout = layouts()
    training = tuple(_collect(f"rnn-train-{cue}", known, cue, "train") for cue in (1, -1))
    final = tuple(
        _collect(f"rnn-final-{layout.layout_id}-{cue}", layout, cue, "final_test")
        for layout in (known, heldout)
        for cue in (1, -1)
    )
    source = TorchRecurrentLearner(TorchRecurrentPolicy(8, 4, 5, 17))
    initial = source.snapshot_state()
    clock = lambda: 0.0
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), clock)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=8), clock)
    helper = (
        StreamRehearsal(
            source,
            wake,
            replay_policy(),
            replay_budget=replay,
            total_updates=2,
            input_dim=8,
            episode_schema=EpisodeReplaySchema(8, 8, 5),
        )
        if replay_enabled
        else None
    )
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    scored: list[np.ndarray] = []
    original_train = TorchRecurrentLearner.train_batch
    original_predict = TorchRecurrentLearner.predict

    def train(owner, features, targets):
        assert not scored, "no training after any final scoring"
        assert wake.updates_completed < 2 or replay_enabled
        assert any(
            np.array_equal(features, b.features) and np.array_equal(targets, b.targets)
            for b in training
        ), "only complete original training packets may update"
        calls.append((features.copy(), targets.copy()))
        return original_train(owner, features, targets)

    def predict(owner, features):
        assert len(calls) == (3 if replay_enabled else 2)
        assert wake.updates_completed == 2
        assert any(np.array_equal(features, b.features) for b in final)
        scored.append(features.copy())
        return original_predict(owner, features)

    monkeypatch.setattr(TorchRecurrentLearner, "train_batch", train)
    monkeypatch.setattr(TorchRecurrentLearner, "predict", predict)
    states: list[tuple[int, Any]] = []
    stream = AttributionStream(
        source,
        training,
        final,
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        observe_checkpoint=lambda count, state, batch: states.append((count, state)),
    )
    stream.run_until(2)
    trained = stream.candidate_snapshot()
    assert trained.completed_updates == (3 if replay_enabled else 2)
    assert trained.training_examples == (24 if replay_enabled else 16)
    assert not scored
    result = stream.finish()
    assert result["completed_updates"] == 2 and result["training_examples"] == 16
    assert len(scored) == 8 and len(states) == 2
    assert wake.updates_completed == 2
    assert replay.replay_examples_completed == (8 if replay_enabled else 0)
    if helper is not None:
        event = helper.report["events"][0]
        assert event["charged_exposures"] == 8
        assert len(event["selected_episodes"]) == 1
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, stream._runtime.actor.snapshot_state().state, initial)

    # The evaluator callback keeps ordinary full learned-state checks. Only the
    # new session's episode activation changes; no existing guard is disabled.
    evaluator = source.fork()
    evaluator.restore_state(trained)
    session = evaluator.start_episode()
    frozen = session.snapshot_state()

    def live(features):
        assert_tree(torch, session.snapshot_state(), frozen)
        probabilities = session.predict(features)
        assert_tree(torch, session.snapshot_state(), frozen)
        return probabilities

    report = run_key_door_episode(
        live,
        identity=EvaluationIdentity(
            "recurrent_bc", 17, "eval:rnn:updates2", ("A",), 2, replay_enabled
        ),
        context="D",
        layout=heldout,
        cue=1,
        layout_partition="heldout_layout",
        max_steps=12,
    )
    assert report["status"] == "complete"
    assert report["identity"]["method"] == "recurrent_bc"
    assert not report["learning_active"] and report["phase"] == "evaluation"
    assert report["outcome"] in {"success", "wrong_key", "illegal_action", "step_limit"}
    assert session.completed_steps == report["prediction_calls_returned"]
    assert 1 <= session.completed_steps <= 12
    assert len(calls) == (3 if replay_enabled else 2)
    assert_tree(torch, session.snapshot_state(), frozen)
    assert_tree(torch, evaluator.snapshot_state(), trained)
    assert_tree(torch, source.snapshot_state(), initial)
