"""Explicit reserved-role binding for the existing owned full-network update.

Inputs: one owned probe_train packet, then the runtime's identical arrays.
Output: paid native update result on a disposable exact BP owner. No source
selection, role conversion, replay, runtime, publication, RNG or IO here.
"""

from dataclasses import dataclass

import numpy as np

from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_demonstration import ContinuousEpisode, continuous_training_pairs


@dataclass(frozen=True)
class _FullProbeBinding:
    features: tuple[tuple[float, ...], ...]
    targets: tuple[tuple[float, ...], ...]

    @classmethod
    def from_training(cls, training: ContinuousEpisode):
        # Why: a private TRAIN-policy clone does not authorize reserved labels.
        # Bind the actual released probe_train packet independently of policy.
        features, targets = continuous_training_pairs(training, role="probe_train")
        return cls(tuple(tuple(row) for row in features), tuple(tuple(row) for row in targets))

    def update(self, learner, features, targets):
        if type(learner) is not NumpyContinuousBCLearner:
            raise ValueError("full-network probe requires exact BP")
        for actual, expected in ((features, self.features), (targets, self.targets)):
            if (
                type(actual) is not np.ndarray
                or actual.dtype != np.float64
                or actual.shape != (len(expected), len(expected[0]))
                or not np.array_equal(actual, expected)
            ):
                raise ValueError("full-network probe arrays differ from bound probe_train packet")
        return learner._apply_full_gauss_newton_batch(features, targets)
