"""Disposable TRAIN head fitting on an existing owned stream readout.

Inputs are a BP/PC-family learner and complete released TRAIN episodes. Outputs
are detached diagnostic arrays and costs. No learner construction, updates,
statistics fitting, actor access, promotion, probe selection, simulator or IO.
"""

from dataclasses import dataclass

import numpy as np

from src.adapters.continuous_learnability_modes import ContinuousLearner, same_continuous_state
from src.adapters.numpy_continuous_bc import (
    ContinuousBCSnapshot,
    NumpyContinuousBCLearner,
    _snapshot as own_bc_snapshot,
)
from src.adapters.numpy_continuous_pc import (
    NumpyContinuousPCLearner,
    _snapshot as own_pc_snapshot,
)
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.continuous_readout_fit import ContinuousReadoutFit, fit_continuous_readout
from src.core.observation_standardization import standardize_observations


@dataclass(frozen=True)
class TrainingReadoutFit:
    fit: ContinuousReadoutFit
    standardization_calls: int
    standardization_rows: int
    standardization_values: int
    standardization_applied: bool


def _owned_state(learner: ContinuousLearner):
    # Existing native validators inspect the actual owner, without relying on a
    # replaceable public snapshot callback to attest to its own isolation.
    if isinstance(learner, NumpyContinuousBCLearner):
        return own_bc_snapshot(learner._state, learner._policy)
    return own_pc_snapshot(learner._state, learner._policy)


def _episodes(episodes: tuple[ContinuousEpisode, ...], spec):
    if type(episodes) is not tuple or not 1 <= len(episodes) <= 16:
        raise ValueError("readout fit requires an exact tuple of1..16 TRAIN episodes")
    # Why: refuse every non-TRAIN role before extracting any labels, including
    # a later forbidden episode in an otherwise valid prefix.
    if any(
        type(e) is not ContinuousEpisode or type(e.role) is not str or e.role != "train"
        for e in episodes
    ):
        raise ValueError("readout fit accepts only complete TRAIN episodes")
    episodes = tuple(copy_continuous_episode(e) for e in episodes)
    if any(e.spec != spec for e in episodes):
        raise ValueError("readout fit episode units/bounds/spec differ from its policy")
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("readout fit episode IDs must be distinct")
    if sum(len(e.transitions) for e in episodes) > 128:
        raise ValueError("readout fit allows at most128 total TRAIN rows")
    pairs = [continuous_training_pairs(e, role="train") for e in episodes]
    features = np.asarray([row for x, _ in pairs for row in x], dtype=np.float64)
    targets = np.asarray([row for _, y in pairs for row in y], dtype=np.float64)
    return features, targets


def diagnose_readout_fit(
    learner: ContinuousLearner, episodes: tuple[ContinuousEpisode, ...]
) -> TrainingReadoutFit:
    """Fit one disposable head; never install it in the supplied readout.

    The existing stream's detached candidate can be supplied directly. Caller
    owns release authority; episode roles are not converted or inferred.
    """
    if type(learner) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner):
        raise ValueError("readout fit requires an exact owned BP/PC-family learner")
    state = _owned_state(learner)
    base = state.policy if isinstance(state, ContinuousBCSnapshot) else state.policy.base
    if base.training_role != "train":
        raise ValueError("readout fit requires a TRAIN policy")
    try:
        features, targets = _episodes(episodes, base.spec)
        values = standardize_observations(features, base.observation_standardization, base.spec)
        fit = fit_continuous_readout(values, targets, state.parameters, base.spec)
    finally:
        # Catch accidental diagnostic writes even when the solve itself fails.
        # No restore: hiding a changed source would erase evidence of a defect.
        if not same_continuous_state(state, _owned_state(learner)):
            raise RuntimeError("readout fitting changed its source learner")
    return TrainingReadoutFit(
        fit, 1, len(values), values.size, base.observation_standardization is not None
    )
