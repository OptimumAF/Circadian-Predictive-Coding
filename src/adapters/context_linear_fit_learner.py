"""One homogeneous current-packet fit, atomically replacing one of two heads.

Existing stream owns roles/consent/label release/budgets/exclusive access. The
inactive head persists without retaining observations. No new optimizer or IO.
"""

from dataclasses import replace

from src.adapters.linear_fit_learner import LinearFitLearner, _rng_signature, _same_array
from src.core.categorical_linear_state import MAX_COUNT
from src.core.categorical_probability_scores import validate_class_targets
from src.core.context_linear_state import (
    ContextLinearFitPolicy,
    ContextLinearFitSnapshot,
    copy_context_linear_state,
    create_context_linear_state,
    predict_context_linear,
    routing_policy,
)
from src.core.context_perceptron import route_context_features
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed

CONTEXT_LINEAR_RESIDUAL_ID = (
    "public_context_current_packet_one_hot_least_squares_sum_squared_residual_v1"
)
CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID = (
    "public_context_minimum_change_one_hot_least_squares_sum_squared_residual_v1"
)


class ContextLinearFitLearner:
    def __init__(self, policy: ContextLinearFitPolicy):
        self._state = create_context_linear_state(policy)
        self._policy = replace(self._state.policy)

    @property
    def policy(self):
        return replace(self._policy)

    @property
    def completed_updates(self):
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self):
        return self.snapshot_state().training_examples

    def snapshot_state(self):
        state = copy_context_linear_state(self._state)
        if state.policy != self.policy:
            raise ValueError("context linear source differs from bound policy")
        return state

    def restore_state(self, state):
        owned = copy_context_linear_state(state)
        if owned.policy != self.policy:
            raise ValueError("context linear snapshot differs from bound policy")
        self._state = owned

    def fork(self):
        child = ContextLinearFitLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features, targets):
        return self._train_batch(features, targets, minimum_change=False)

    def train_batch_minimum_change(self, features, targets):
        """Explicit nonsticky current-head operation; caller owns roles/lease/budget."""
        return self._train_batch(features, targets, minimum_change=True)

    def _owners(self):
        # Why: equal-valued replacement is still an unobserved owner change.
        # Hold references through the operation so object IDs cannot be recycled.
        return (
            self._policy,
            self._state,
            self._state.policy,
            *(owner for head in self._state.heads for owner in (head, head.policy, head.model)),
        )

    def _train_batch(self, features, targets, *, minimum_change):
        before = self.snapshot_state()
        index, projected = route_context_features(features, routing_policy(before.policy))
        validate_class_targets(targets, len(features), before.policy.num_classes)
        rows = len(features)
        if before.completed_updates == MAX_COUNT or before.training_examples + rows > MAX_COUNT:
            raise ValueError("context linear counters would overflow")
        owners = self._owners()
        original_features, original_targets = features.copy(), targets.copy()
        rng = _rng_signature()
        private = LinearFitLearner(before.heads[index].policy)
        private.restore_state(before.heads[index])
        definition = (
            CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID if minimum_change else CONTEXT_LINEAR_RESIDUAL_ID
        )
        private_commits = 0
        try:
            diagnostic = (
                private.train_batch_minimum_change(projected, targets)
                if minimum_change
                else private.train_batch(projected, targets)
            )
            private_commits = 1
            heads = list(before.heads)
            heads[index] = private.snapshot_state()
            after = copy_context_linear_state(
                ContextLinearFitSnapshot(
                    before.policy,
                    (heads[0], heads[1]),
                    before.completed_updates + 1,
                    before.training_examples + rows,
                )
            )
            if (
                any(a is not b for a, b in zip(owners, self._owners(), strict=True))
                or self.snapshot_state() != before
                or not _same_array(features, original_features)
                or not _same_array(targets, original_targets)
                or _rng_signature() != rng
            ):
                raise ValueError("context linear donor, raw inputs or RNG changed")
            result = TrainingDiagnostic(definition, diagnostic.value)
        except Exception as error:
            work = (
                error.work
                if isinstance(error, TrainingUpdateFailed)
                else dict(
                    fit_attempts=1,
                    fit_rows_attempted=rows,
                    fit_returns=private_commits,
                    fit_rows_returned=rows * private_commits,
                    unreturned_work_unknown=private_commits == 0,
                )
            )
            work.update(
                candidate_commits=0,
                private_candidate_commits=private_commits,
                context_index=index,
                unpublished_work=True,
            )
            raise TrainingUpdateFailed(
                f"{type(error).__name__}: {error}"[:512], definition, work
            ) from error
        self._state = after
        return result

    def predict(self, features):
        return predict_context_linear(self.snapshot_state(), features)
