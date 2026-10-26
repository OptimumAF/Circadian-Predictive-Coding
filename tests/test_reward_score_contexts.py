"""Explicit sampled-task context labels never require fabricated teacher training."""

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, _score_checkpoints
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


class Scorer:
    def fork(self):
        return Scorer()

    def restore_state(self, state):
        self.state = state

    def predict(self, features):
        return np.tile(np.array([[0.6]]), (len(features), 1))


def batches():
    return (
        AttributionBatch(
            "final_test", "query", ("q0", "q1"), np.zeros((2, 1)), np.array([[0.0], [1.0]])
        ),
    )


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)


def test_should_preserve_default_scores_and_report_explicit_sampled_contexts():
    source = Scorer()
    heldout = batches()
    old = _score_checkpoints(source, [(0, 0), (1, 1)], heldout, heldout, budget())
    new = _score_checkpoints(
        source, [(0, 0), (1, 1)], (), heldout, budget(), training_contexts=("sampled-A",)
    )
    assert old[0] == new[0]
    assert old[1]["scores"] == new[1]["scores"]
    assert old[1]["training_context"] == "query"
    assert new[1]["training_context"] == "sampled-A"
    np.testing.assert_array_equal(heldout[0].features, np.zeros((2, 1)))


@pytest.mark.parametrize("contexts", [[], (), (True,), ("",), (" A",), ("x" * 33,), ("x",) * 513])
def test_should_refuse_invalid_explicit_contexts_before_forking(contexts):
    class RefuseFork(Scorer):
        def fork(self):
            raise AssertionError("invalid metadata reached scoring")

    with pytest.raises(ValueError, match="explicit scoring contexts"):
        _score_checkpoints(
            RefuseFork(), [(0, 0), (1, 1)], (), batches(), budget(), training_contexts=contexts
        )


def test_should_refuse_mixed_teacher_contexts_or_missing_sampled_checkpoint_before_prediction():
    with pytest.raises(ValueError):
        _score_checkpoints(
            Scorer(), [(0, 0), (1, 1)], batches(), batches(), budget(), training_contexts=("A",)
        )
    with pytest.raises(ValueError):
        _score_checkpoints(
            Scorer(), [(0, 0), (2, 2)], (), batches(), budget(), training_contexts=("A",)
        )


def test_should_admit_explicit_reward_method_in_common_evaluation_identity():
    from src.app.line_policy_evaluation import EvaluationIdentity

    identity = EvaluationIdentity("recurrent_rl", 17, "rl-final-0", ("A", "B", "C", "A"), 0, False)
    assert identity.method == "recurrent_rl" and identity.replay_enabled is False
