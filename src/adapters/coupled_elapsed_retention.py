"""Explicit coupled elapsed-retention composition, without a worker factory.

An exact positive elapsed policy and optional raw-copy allowance are validated
before consulting the managed owner. The existing lifecycle owns admission,
cleanup and its original clock; the caller explicitly constructs/starts/stops
RetentionExpiryDriver. No scheduling, native state reset, IO or unlearning here.
"""

from dataclasses import replace
from typing import Callable

from src.adapters.coupled_data_lifecycle import (
    measure_coupled_payload_bytes,
    coupled_replay_footprint,
    erase_coupled_replay_payloads,
)
from src.adapters.coupled_payload_copies import (
    measure_coupled_auxiliary_bytes,
    measure_coupled_checkpoint_bytes,
    coupled_native_growth_bytes,
    coupled_preparation_bytes,
    coupled_prediction_cache_bytes,
)
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.core.data_retention import DataRetentionPolicy
from src.core.data_erasure import ReplayPayloadErasure
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits


def make_coupled_elapsed_lifecycle(
    owner: ManagedExperienceOwner, *, policy: DataRetentionPolicy
) -> ManagedDataLifecycle:
    return _make_coupled_elapsed_lifecycle(
        owner, policy=policy, native_erase=erase_coupled_replay_payloads
    )


def _make_coupled_elapsed_lifecycle(
    owner: ManagedExperienceOwner,
    *,
    policy: DataRetentionPolicy,
    native_erase: Callable[[object], ReplayPayloadErasure],
) -> ManagedDataLifecycle:
    """Internal birth composition; public defaults always use the normal port."""
    if type(policy) is not DataRetentionPolicy:
        raise ValueError("coupled elapsed lifecycle requires an exact DataRetentionPolicy")
    if policy.max_retention_seconds is None:
        raise ValueError("coupled elapsed lifecycle requires positive elapsed seconds")
    if type(policy.holders) is not PayloadOwnershipLimits:
        raise ValueError("coupled elapsed lifecycle requires exact holder limits")
    copies = policy.owned_payload_copies
    if copies is not None and type(copies) is not PayloadCopyLimits:
        raise ValueError("coupled elapsed lifecycle requires exact optional copy limits")
    replace(
        policy,
        holders=replace(policy.holders),
        owned_payload_copies=None if copies is None else replace(copies),
    )
    if not callable(native_erase):
        raise ValueError("coupled elapsed lifecycle requires a typed native erasure port")
    return ManagedDataLifecycle(
        owner,
        policy=policy,
        measure_payload_bytes=measure_coupled_payload_bytes,
        native_footprint=coupled_replay_footprint,
        native_erase=native_erase,
        measure_auxiliary_bytes=measure_coupled_auxiliary_bytes,
        measure_checkpoint_bytes=measure_coupled_checkpoint_bytes,
        native_growth_bytes=coupled_native_growth_bytes,
        prepare_model_bytes=coupled_preparation_bytes,
        prediction_cache_bytes=coupled_prediction_cache_bytes,
    )
