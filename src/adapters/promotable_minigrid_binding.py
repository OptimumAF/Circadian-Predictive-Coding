"""Pin one actual analytical serving actor for a target-free MiniGrid episode.

Inputs are an exact actor and caller episode ID. Outputs are public actor
probabilities and observed work. No fork, restore, training, publication, cache
write, teacher, environment construction or arbitrary-graph certification.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    ContextLinearTrainingSnapshot,
    copy_context_linear_training_snapshot,
)
from src.app.coupled_attribution import _same_value
from src.adapters.minigrid_shared_prediction import bind_shared_prediction
from src.app.serving_promotion import PromotableActor, _Slot, _Bundle
from src.core.actor_ports import VersionedPrediction
from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.experience import require_identifier, require_tick
from src.core.serving_ports import ServingConfiguration, ServingSnapshot


def _auxiliary(value):
    config = value.configuration
    if type(config) is not ServingConfiguration:
        raise ValueError("MiniGrid serving requires an exact cache configuration")
    ServingConfiguration(config.cache_ttl_ticks, config.max_cache_entries)
    if config.cache_ttl_ticks > 1024 or config.max_cache_entries > 16:
        raise ValueError("MiniGrid serving cache configuration exceeds1024ticks/16entries")
    if type(value.cache) is not dict or value.cache:
        raise ValueError("MiniGrid episode binding requires an empty serving cache")
    require_tick(value.last_cache_tick, "last cache tick")
    if value.last_cache_tick > 2**53 - 1:
        raise ValueError("MiniGrid cache tick exceeds its native scalar bound")
    metadata = value.metadata
    if type(metadata) is not dict or len(metadata) > 16:
        raise ValueError("MiniGrid serving metadata must have at most16scalar entries")
    for key, item in metadata.items():
        if type(key) is not str or not 1 <= len(key) <= 64:
            raise ValueError("MiniGrid metadata keys require1..64native characters")
        kind = type(item)
        if kind not in (str, int, float, bool, type(None)):
            raise ValueError("MiniGrid metadata values must be native scalars")
        if (
            (kind is str and len(item) > 256)
            or (kind is int and abs(item) > 2**53 - 1)
            or (kind is float and not isfinite(item))
        ):
            raise ValueError("MiniGrid metadata scalar exceeds its finite bound")


def _backend(actor):
    if type(actor) is not PromotableActor or type(actor._slot) is not _Slot:
        raise ValueError("MiniGrid requires an exact PromotableActor slot")
    bundle = actor._slot.bundle
    if type(bundle) is not _Bundle or type(bundle.learner) is not ContextLinearLearner:
        raise ValueError("MiniGrid requires an exact analytical serving backend")
    if actor._payload_registry._lifecycle is not None:
        raise ValueError("MiniGrid binding does not admit managed lifecycle owners")
    require_identifier(bundle.version, "serving actor version")
    require_tick(actor._slot.generation, "serving generation")
    if len(bundle.version) > 64 or actor._slot.generation > 2**53 - 1:
        raise ValueError("MiniGrid serving identity exceeds its finite bound")
    _auxiliary(bundle)
    # Validate bounded native data before invoking any public snapshot port.
    state = copy_context_linear_training_snapshot(bundle.learner._state)
    if state.failed_work is not None:
        raise ValueError("failed analytical work cannot serve MiniGrid")
    return bundle.learner


def _snapshot(value):
    if type(value) is not ServingSnapshot:
        raise ValueError("MiniGrid requires an exact complete serving snapshot")
    require_identifier(value.actor_version, "serving actor version")
    require_tick(value.generation, "serving generation")
    if len(value.actor_version) > 64 or value.generation > 2**53 - 1:
        raise ValueError("MiniGrid snapshot identity exceeds its finite bound")
    _auxiliary(value)
    state = copy_context_linear_training_snapshot(value.model_state)
    if state.failed_work is not None:
        raise ValueError("failed analytical work cannot serve MiniGrid")
    return state


@dataclass
class PromotableMiniGridBinding:
    actor: Any
    backend: ContextLinearLearner
    slot: Any
    snapshot: Any
    episode_id: str
    stopped: bool = False
    predictions_attempted: int = 0
    predictions_returned: int = 0
    actions_attempted: int = 0
    actions_returned: int = 0
    sharing: Any = None

    @property
    def state(self) -> ContextLinearTrainingSnapshot:
        return copy_context_linear_training_snapshot(self.snapshot.model_state)

    def check(self):
        if self.stopped:
            raise ValueError("MiniGrid serving binding is stopped")
        try:
            self._check_identity()
            if self.sharing is not None:
                self.sharing.check()
            observed = self.actor.serving_snapshot_if_idle()
            _snapshot(observed)
            self._check_identity()
            if not _same_value(observed, self.snapshot):
                raise ValueError("pinned MiniGrid serving bundle changed")
            # Reconcile after the public callback under the original gate.
            with self.actor._payload_exclusive():
                self._check_locked()
        except BaseException:
            self.stopped = True
            raise

    def _check_identity(self):
        if _backend(self.actor) is not self.backend or self.actor._slot is not self.slot:
            raise ValueError("pinned MiniGrid serving slot or backend changed")

    def _check_locked(self):
        # The physical action owns the existing actor gate. Calling the actor's
        # locking public snapshot here would reenter that non-reentrant gate.
        self._check_identity()
        if self.sharing is not None:
            self.sharing.check()
        bundle = self.slot.bundle
        if not _same_value(self.backend._state, self.snapshot.model_state) or not (
            self.slot.generation == self.snapshot.generation
            and bundle.version == self.snapshot.actor_version
            and bundle.configuration == self.snapshot.configuration
            and bundle.metadata == self.snapshot.metadata
            and bundle.last_cache_tick == self.snapshot.last_cache_tick
        ):
            raise ValueError("pinned MiniGrid serving values changed during action")

    def snapshot_state(self):
        self.check()
        return self.state

    def predict(self, features):
        try:
            self.check()
            if (
                type(features) is not np.ndarray
                or features.dtype != np.dtype(np.float64)
                or features.shape != (1, 296)
                or not np.isfinite(features).all()
                or np.any(np.abs(features) > 1)
                or self.predictions_attempted >= 16
            ):
                raise ValueError(
                    "MiniGrid serving requires one finite public296 row and <=16predictions"
                )
            inputs = features.copy()
            self.predictions_attempted += 1
            result = (
                self.actor.predict(inputs)
                if self.sharing is None
                else self.sharing.shared.predict(inputs)
            )
            self.predictions_returned += 1
            self.check()
            if (
                type(result) is not VersionedPrediction
                or type(result.actor_version) is not str
                or result.actor_version != self.snapshot.actor_version
            ):
                raise ValueError("MiniGrid prediction belongs to a different actor version")
            values = result.prediction
            validate_class_probabilities(values, 7)
            if values.shape != (1, 7) or not np.array_equal(inputs, features):
                raise ValueError("MiniGrid prediction shape or public input changed")
            return values.copy()
        except BaseException:
            self.stopped = True
            raise

    def step(self, environment, action):
        try:
            if type(action) is not int or action not in range(7):
                raise ValueError("MiniGrid serving action must be a native integer in0..6")
            self.check()
            if self.actions_attempted >= 16:
                raise ValueError("MiniGrid serving action allowance exhausted")
            # The same gate serializes ordinary commit/rollback and physics.
            # Native environment.step must not reenter actor/controller ports.
            with self.actor._payload_exclusive():
                self._check_locked()
                if environment.legal_actions != tuple(range(7)):
                    raise ValueError("MiniGrid must expose all seven valid native inputs")
                self._check_locked()
                self.actions_attempted += 1
                result = environment.step(action)
                self.actions_returned += 1
                self._check_locked()
                return result
        except BaseException:
            self.stopped = True
            raise

    def report(self):
        result = dict(
            actor_version=self.snapshot.actor_version,
            generation=self.snapshot.generation,
            native_updates=self.snapshot.model_state.completed_updates,
            native_rows=self.snapshot.model_state.training_examples,
            predictions_attempted=self.predictions_attempted,
            predictions_returned=self.predictions_returned,
            actions_attempted=self.actions_attempted,
            actions_returned=self.actions_returned,
            stopped=self.stopped,
            unreturned_work_unknown=(
                self.predictions_attempted > self.predictions_returned
                or self.actions_attempted > self.actions_returned
            ),
        )
        if self.sharing is not None:
            result["sharing"] = self.sharing.report()
        return result


def bind_promotable_minigrid(actor, *, episode_id, shared_runtime=None):
    require_identifier(episode_id, "MiniGrid serving episode ID")
    if len(episode_id) > 64:
        raise ValueError("MiniGrid serving episode ID exceeds64characters")
    backend = _backend(actor)
    slot = actor._slot
    sharing = None if shared_runtime is None else bind_shared_prediction(shared_runtime, actor)
    if sharing is not None and (actor._slot is not slot or _backend(actor) is not backend):
        raise ValueError("MiniGrid actor changed during shared admission")
    snapshot = actor.serving_snapshot_if_idle()
    _snapshot(snapshot)
    reader = PromotableMiniGridBinding(actor, backend, slot, deepcopy(snapshot), episode_id)
    reader.sharing = sharing
    reader.check()
    return reader
