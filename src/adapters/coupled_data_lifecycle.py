"""Typed raw-payload cleanup ports for the fixed native coupled learner.

Inputs are trusted, quiescent learners and exact real numeric NumPy arrays.
Outputs describe raw replay ownership, never learned information. The fixed
CoupledLearner owns parameters and chemistry, but no raw replay buffer; therefore
its erasure port drops nothing and never resets or snapshots the native model.
Extra owned-copy and elapsed-retention policies need additional ports and are
explicitly unsupported here. No scheduling, IO, consent authority or unlearning.
"""

from dataclasses import replace

import numpy as np

from src.adapters.coupled_learners import CoupledLearner
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_ownership import PayloadOwnershipLimits


def measure_coupled_payload_bytes(value: object) -> int:
    if type(value) is not np.ndarray or value.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("coupled managed payloads require exact real numeric NumPy arrays")
    return value.nbytes


def coupled_replay_footprint(learner: object) -> ReplayPayloadErasure:
    if type(learner) is not CoupledLearner:
        raise ValueError("coupled replay footprint requires an exact CoupledLearner")
    return ReplayPayloadErasure(0, 0, 0)


def erase_coupled_replay_payloads(learner: object) -> ReplayPayloadErasure:
    if type(learner) is not CoupledLearner:
        raise ValueError("coupled replay erasure requires an exact CoupledLearner")
    return ReplayPayloadErasure(0, 0, 0)


def make_coupled_data_lifecycle(
    owner: ManagedExperienceOwner, *, policy: DataRetentionPolicy
) -> ManagedDataLifecycle:
    if type(policy) is not DataRetentionPolicy:
        raise ValueError("coupled lifecycle requires an exact DataRetentionPolicy")
    if policy.owned_payload_copies is not None or policy.max_retention_seconds is not None:
        raise ValueError(
            "coupled lifecycle does not support owned-copy or elapsed-retention policy"
        )
    if type(policy.holders) is not PayloadOwnershipLimits:
        raise ValueError("coupled lifecycle requires exact payload holder limits")
    # Revalidate nested frozen records too: callers can have tampered with them.
    replace(policy, holders=replace(policy.holders))
    return ManagedDataLifecycle(
        owner,
        policy=policy,
        measure_payload_bytes=measure_coupled_payload_bytes,
        native_footprint=coupled_replay_footprint,
        native_erase=erase_coupled_replay_payloads,
    )
