"""Compile ordinary contextual TRAIN traces for the existing replay executor.

Inputs are the fixed D9 schema and old-v1 TRAIN prototype. Outputs are owned
complete episodes or the existing bounded executor. No IO, session creation,
controller fit/selection, source restore, model training or evaluation here.
"""

import numpy as np

from scripts.run_ticket_workflow import CONTEXT_FEATURES
from scripts.ticket_workflow_rehearsal import _ticket_trace
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.app.categorical_replay_action import CategoricalReplayAction
from src.app.coupled_attribution import AttributionBatch, _same_value, _validate_batch
from src.core.categorical_perceptron import PERCEPTRON_MISTAKES_DIAGNOSTIC
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.context_perceptron import ContextPerceptronPolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.ticket_workflow import ACTION_IDS, CATALOG


def compile_context_bank(
    spec: DecisionFeatureSpec, old_train: AttributionBatch
) -> tuple[DemonstrationEpisode, ...]:
    """Validate the exact six old TRAIN rows against authentic sandbox traces."""
    if type(spec) is not DecisionFeatureSpec:
        raise ValueError("context replay requires exact D9 feature/catalog schema")
    spec.__post_init__()
    if spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG):
        raise ValueError("context replay requires exact D9 feature/catalog schema")
    if (
        type(old_train) is not AttributionBatch
        or type(old_train.features) is not np.ndarray
        or old_train.features.shape != (6, 9)
        or type(old_train.targets) is not np.ndarray
        or old_train.targets.shape != (6, 1)
        or old_train.context != "old-v1"
        or old_train.row_ids != tuple(f"update-1-{i}" for i in range(6))
        or old_train.num_classes != 6
        or old_train.episode is not None
    ):
        raise ValueError("context replay requires the original old-v1 six-row TRAIN prototype")
    _validate_batch(old_train, "train", set())
    episodes = []
    for index, concept in enumerate(("invoice", "password")):
        observations, actions = _ticket_trace(spec, "v1", concept)
        labels = np.array([[float(ACTION_IDS.index(a))] for a in actions])
        part = slice(3 * index, 3 * (index + 1))
        if not np.array_equal(
            old_train.features[part], np.array(observations)
        ) or not np.array_equal(old_train.targets[part], labels):
            raise ValueError("context replay TRAIN rows differ from authentic public-context trace")
        # Why: retain the existing codec's terminal-zero convention. It is a
        # placeholder after commit, not a claim about the environment's final frame.
        transitions = tuple(
            DemonstrationTransition(
                step,
                observations[step],
                ACTION_IDS.index(action),
                float(step == 2),
                observations[step + 1] if step < 2 else (0.0,) * 9,
                step == 2,
                6,
            )
            for step, action in enumerate(actions)
        )
        episodes.append(
            DemonstrationEpisode(
                f"context-bank-v1-{concept}", f"v1-{concept}", "train", transitions, 6
            )
        )
    return tuple(episodes)


def compose_context_replay(runtime, learner, replay_budget, spec, old_train):
    """Bind caller-owned ordinary TRAIN runtime/budget; no new authority or fit."""
    if not isinstance(learner, ContextPerceptronLearner):
        raise ValueError("context replay requires the existing paired-head learner")
    expected = ContextPerceptronPolicy(
        9, 6, (CONTEXT_FEATURES.index("policy_v1"), CONTEXT_FEATURES.index("policy_v2"))
    )
    if learner.policy != expected:
        raise ValueError("context replay requires the original public-context head policy")
    episodes = compile_context_bank(spec, old_train)
    return CategoricalReplayAction(
        runtime,
        learner,
        replay_budget,
        episodes,
        ReplayRetentionBudget(2, 4096),
        states_equal=_same_value,
        definition=PERCEPTRON_MISTAKES_DIAGNOSTIC,
    )
