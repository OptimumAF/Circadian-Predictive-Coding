"""Opt-in bounded TRAIN memory with cumulative paid exposure accounting.

Owns current fixed-input prototypes, bounded IDs/counters and full-state copies.
Uses existing nearest predictions and retention rules. No runner, data loading,
probe/final scope, promotion, normalization fit, optimizer or model RNG.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from src.adapters.numpy_nearest_continuous_bc import (
    FixedInputNearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
    _policy,
)
from src.core import continuous_bc as bc
from src.core.learner_ports import TrainingDiagnostic
from src.core.nearest_retention import (
    NearestRetentionPolicy,
    content_ids,
    retention_policy,
    select_retained_rows,
    rank_evaluations_per_discard,
)

OBJECTIVE = "continuous_nearest_retained_pairs_v1"
MAX_COUNTER = 2**63 - 1
_PAIR_DOMAIN = b"nearest_retained_pair_v1"


def _count(value: int) -> int:
    if type(value) is not int or not 0 <= value <= MAX_COUNTER:
        raise ValueError("retained counters must be native nonnegative63-bit integers")
    return value


def _ids(features, actions):
    return tuple(
        sha256(_PAIR_DOMAIN + x.astype("<f8").tobytes() + y.astype("<f8").tobytes()).hexdigest()
        for x, y in zip(features, actions, strict=True)
    )


@dataclass(frozen=True)
class RetainedNearestSnapshot(NearestContinuousBCSnapshot):
    retention: NearestRetentionPolicy
    prototype_ids: tuple[str, ...]
    duplicate_rows: int
    discarded_rows: int
    selection_candidates: int
    reservoir_rank_evaluations: int

    @property
    def retained_rows(self) -> int:
        return len(self.prototype_ids)


def retained_snapshot(value, policy, retention) -> RetainedNearestSnapshot:
    if (
        type(value) is not RetainedNearestSnapshot
        or type(policy) is not FixedInputNearestContinuousBCPolicy
        or policy.training_role != "train"
        or _policy(value.policy) != _policy(policy)
        or retention_policy(value.retention) != retention_policy(retention)
    ):
        raise ValueError("retained snapshot owner, fixed-input TRAIN policy or retention differs")
    ids = content_ids(value.prototype_ids, policy.max_prototypes)
    if len(set(ids)) != len(ids):
        raise ValueError("retained snapshot contains duplicate IDs")
    exposures, updates, duplicates, discarded, candidates, ranks = (
        _count(item)
        for item in (
            value.training_examples,
            value.completed_updates,
            value.duplicate_rows,
            value.discarded_rows,
            value.selection_candidates,
            value.reservoir_rank_evaluations,
        )
    )
    if (
        exposures != len(ids) + duplicates + discarded
        or not (updates == 0 if exposures == 0 else 1 <= updates <= exposures)
        or bool(ids) != bool(exposures)
        or not exposures <= candidates <= exposures * (policy.max_prototypes + 1)
        or ranks != discarded * rank_evaluations_per_discard(policy.max_prototypes, retention)
    ):
        raise ValueError("retained snapshot exposure, selection or update counters differ")
    features = bc._array(
        value.prototype_features, (len(ids), policy.spec.observation_dim), "prototypes"
    )
    actions = bc._targets(value.prototype_actions, len(ids), policy.spec)
    if _ids(features, actions) != ids:
        raise ValueError("retained IDs differ from owned feature/target content")
    return RetainedNearestSnapshot(
        _policy(policy),
        features,
        actions,
        updates,
        exposures,
        retention_policy(retention),
        ids,
        duplicates,
        discarded,
        candidates,
        ranks,
    )


def advance_retained(before: RetainedNearestSnapshot, features, targets) -> RetainedNearestSnapshot:
    """Prepare an owned atomic replacement; validation work is additional cost."""
    before = retained_snapshot(before, before.policy, before.retention)
    x, y = (
        bc._features(features, before.policy.spec),
        bc._targets(targets, len(features), before.policy.spec),
    )
    chosen = select_retained_rows(
        before.prototype_ids, _ids(x, y), before.policy.max_prototypes, before.retention
    )
    indices = list(chosen.indices)
    result = RetainedNearestSnapshot(
        before.policy,
        np.concatenate((before.prototype_features, x))[indices],
        np.concatenate((before.prototype_actions, y))[indices],
        before.completed_updates + 1,
        before.training_examples + len(x),
        before.retention,
        chosen.ids,
        before.duplicate_rows + chosen.duplicate_rows,
        before.discarded_rows + chosen.discarded_rows,
        before.selection_candidates + chosen.selection_candidates,
        before.reservoir_rank_evaluations + chosen.reservoir_rank_evaluations,
    )
    return retained_snapshot(result, before.policy, before.retention)


def retained_costs(state: RetainedNearestSnapshot) -> dict:
    state = retained_snapshot(state, state.policy, state.retention)
    row_bytes = 8 * (state.policy.spec.observation_dim + len(state.policy.spec.action_units))
    return dict(
        training_examples=state.training_examples,
        retained_rows=state.retained_rows,
        duplicate_rows=state.duplicate_rows,
        discarded_rows=state.discarded_rows,
        numeric_bytes=state.retained_rows * row_bytes,
        content_id_bytes=state.retained_rows * 64,
        digest_payload_bytes=state.retained_rows * 32,
        incoming_pair_hash_bytes=state.training_examples * (row_bytes + len(_PAIR_DOMAIN)),
        discarded_pair_bytes=state.discarded_rows * row_bytes,
        duplicate_pair_bytes=state.duplicate_rows * row_bytes,
        selection_candidates=state.selection_candidates,
        reservoir_rank_evaluations=state.reservoir_rank_evaluations,
        scope="current numeric/hex payload and selected hash work; excludes validation hashing/copies/Python/RSS",
    )


class NumpyRetainedNearestBCLearner(NumpyNearestContinuousBCLearner):
    """Distinct full-state type; original exact native/probe admissions refuse it."""

    def __init__(
        self, policy: FixedInputNearestContinuousBCPolicy, retention: NearestRetentionPolicy
    ):
        self._policy = _policy(policy)
        self._retention = retention_policy(retention)
        rank_evaluations_per_discard(policy.max_prototypes, self._retention)
        self._state = retained_snapshot(
            RetainedNearestSnapshot(
                self._policy,
                np.empty((0, policy.spec.observation_dim), dtype=np.float64),
                np.empty((0, len(policy.spec.action_units)), dtype=np.float64),
                0,
                0,
                self._retention,
                (),
                0,
                0,
                0,
                0,
            ),
            self._policy,
            self._retention,
        )

    def snapshot_state(self) -> RetainedNearestSnapshot:
        return retained_snapshot(self._state, self._policy, self._retention)

    def restore_state(self, snapshot: NearestContinuousBCSnapshot) -> None:
        self._state = retained_snapshot(snapshot, self._policy, self._retention)

    def fork(self) -> "NumpyRetainedNearestBCLearner":
        child = object.__new__(NumpyRetainedNearestBCLearner)
        child._policy, child._retention = self.policy, retention_policy(self._retention)
        child._state = self.snapshot_state()
        return child

    def train_batch(self, features: bc.Array, targets: bc.Array) -> TrainingDiagnostic:
        candidate = advance_retained(self.snapshot_state(), features, targets)
        diagnostic = TrainingDiagnostic(OBJECTIVE, len(features))
        self._state = candidate
        return diagnostic


def same_retained_state(first, second) -> bool:
    if type(first) is not RetainedNearestSnapshot or type(second) is not RetainedNearestSnapshot:
        return False
    return (
        all(
            getattr(first, name) == getattr(second, name)
            for name in (
                "policy",
                "retention",
                "prototype_ids",
                "completed_updates",
                "training_examples",
                "duplicate_rows",
                "discarded_rows",
                "selection_candidates",
                "reservoir_rank_evaluations",
            )
        )
        and np.array_equal(first.prototype_features, second.prototype_features)
        and np.array_equal(first.prototype_actions, second.prototype_actions)
    )


def _native_owners(owner):
    if (
        type(owner) is not NumpyRetainedNearestBCLearner
        or type(owner._state) is not RetainedNearestSnapshot
    ):
        raise ValueError("expected exact retained nearest native owner")
    state = owner._state
    return (
        owner,
        state,
        owner._policy,
        owner._retention,
        state.policy,
        state.retention,
        state.prototype_features,
        state.prototype_actions,
        state.prototype_ids,
    )


@dataclass(frozen=True)
class RetainedNativePin:
    owners: tuple
    state: RetainedNearestSnapshot


def pin_retained_native(owner) -> RetainedNativePin:
    """Pin declared native owners before any public observation callback."""
    identities = _native_owners(owner)
    state = retained_snapshot(owner._state, owner._policy, owner._retention)
    if not state.retained_rows:
        raise ValueError("retained native serving needs released prototypes")
    if np.shares_memory(owner._state.prototype_features, owner._state.prototype_actions):
        raise ValueError("retained feature/target owners must not alias")
    return RetainedNativePin(identities, state)


def require_retained_native(owner, pin: RetainedNativePin, *, observe=True) -> None:
    def native():
        if any(a is not b for a, b in zip(_native_owners(owner), pin.owners, strict=True)):
            raise ValueError("retained native owner identity changed")
        if not same_retained_state(
            pin.state, retained_snapshot(owner._state, owner._policy, owner._retention)
        ):
            raise ValueError("retained native state changed")

    native()
    if observe:
        public, policy = owner.snapshot_state(), owner.policy
        if not same_retained_state(pin.state, public) or policy != pin.state.policy:
            raise ValueError("retained public/native state differs")
    native()
