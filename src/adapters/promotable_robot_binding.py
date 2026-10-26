"""Bind one exact nearest-BC serving bundle for read-only robot controls.

Inputs: an actual PromotableActor and its owned public snapshots. Output: a
frozen binding with explicit checks. No controls, training, promotion, cache
updates, serialization or arbitrary Python-graph certification happens here.
"""

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
    _snapshot as validate_nearest_snapshot,
)
from src.app.serving_promotion import PromotableActor, _Bundle, _Slot
from src.core.continuous_bc import Array
from src.core.experience import require_identifier, require_tick
from src.core.serving_ports import CachedPrediction, ServingConfiguration, ServingSnapshot

NearestPromotableActor = PromotableActor[Array, Array, Array, NearestContinuousBCSnapshot]
NearestServingSnapshot = ServingSnapshot[NearestContinuousBCSnapshot, Array]
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")
_MAX_NODES = 4096
_MAX_BYTES = 1024**2


def _payload_size(value: Any, remaining: int) -> int:
    kind = type(value)
    if kind is str:
        if len(value) > remaining:
            raise ValueError("serving auxiliary values exceed1MiB payload")
        size = len(value.encode("utf8"))
    elif kind is bytes:
        size = len(value)
    elif kind is int:
        size = max(1, (value.bit_length() + 7) // 8)
    else:
        size = 8 if kind is float else 1
    if size > remaining:
        raise ValueError("serving auxiliary values exceed1MiB payload")
    return size


def _backend(actor: NearestPromotableActor) -> NumpyNearestContinuousBCLearner:
    # Why: inspect only declared native storage before public snapshot callbacks.
    if type(actor) is not PromotableActor or type(actor._slot) is not _Slot:
        raise ValueError("expected an exact promotable actor slot")
    bundle = actor._slot.bundle
    if type(bundle) is not _Bundle or type(bundle.learner) is not NumpyNearestContinuousBCLearner:
        raise ValueError("expected an exact nearest BC serving bundle")
    return bundle.learner


def _owned_values_equal(first: Any, second: Any) -> bool:
    """Compare bounded plain owned values; never call custom equality methods."""
    pending = [(first, second)]
    nodes, sizes = 0, [0, 0]
    while pending:
        left, right = pending.pop()
        nodes += 1
        if nodes > _MAX_NODES:
            raise ValueError("serving auxiliary values exceed4096 nodes or contain a cycle")
        if type(left) is not type(right):
            return False
        kind = type(left)
        if kind is dict:
            if max(len(left), len(right)) + nodes + len(pending) > _MAX_NODES:
                raise ValueError("serving auxiliary dictionary exceeds node limit")
            for side, value in enumerate((left, right)):
                for key in value:
                    if type(key) is not str:
                        raise ValueError("serving auxiliary dictionary keys must be native strings")
                    sizes[side] += _payload_size(key, _MAX_BYTES - sizes[side])
            if left.keys() != right.keys():
                return False
            pending.extend((left[key], right[key]) for key in left)
        elif kind in (list, tuple):
            if max(len(left), len(right)) + nodes + len(pending) > _MAX_NODES:
                raise ValueError("serving auxiliary sequence exceeds node limit")
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right, strict=True))
        elif kind is np.ndarray:
            if left.dtype != right.dtype or left.shape != right.shape:
                return False
            if left.dtype.kind not in "biufc":
                raise ValueError("serving auxiliary arrays must have numeric native dtypes")
            for side, value in enumerate((left, right)):
                sizes[side] += value.nbytes
                if sizes[side] > _MAX_BYTES:
                    raise ValueError("serving auxiliary values exceed1MiB payload")
                if not np.isfinite(value).all():
                    raise ValueError("serving auxiliary arrays must be finite")
            if not np.array_equal(left, right):
                return False
        elif kind in (type(None), bool, int, float, str, bytes):
            for side, value in enumerate((left, right)):
                if kind is float and not math.isfinite(value):
                    raise ValueError("serving auxiliary floats must be finite")
                sizes[side] += _payload_size(value, _MAX_BYTES - sizes[side])
            if left != right:
                return False
        else:
            raise ValueError("unsupported serving auxiliary value type")
        if any(size > _MAX_BYTES for size in sizes):
            raise ValueError("serving auxiliary values exceed1MiB payload")
    return True


def _auxiliary(snapshot: NearestServingSnapshot) -> dict[str, Any]:
    """Validate declared serving fields; return borrowed values from owned copy."""
    if type(snapshot) is not ServingSnapshot:
        raise ValueError("expected an exact serving snapshot")
    require_tick(snapshot.generation, "serving generation")
    require_identifier(snapshot.actor_version, "serving actor version")
    require_tick(snapshot.last_cache_tick, "last cache tick")
    if type(snapshot.model_state) is not NearestContinuousBCSnapshot:
        raise ValueError("serving requires an exact nearest BC snapshot")
    validate_nearest_snapshot(snapshot.model_state, snapshot.model_state.policy)
    config = snapshot.configuration
    if type(config) is not ServingConfiguration:
        raise ValueError("expected an exact serving configuration")
    ServingConfiguration(config.cache_ttl_ticks, config.max_cache_entries)
    if type(snapshot.cache) is not dict or len(snapshot.cache) > min(
        config.max_cache_entries, _MAX_NODES
    ):
        raise ValueError("serving cache differs from declared capacity")
    if (
        type(snapshot.metadata) is not dict
        or len(snapshot.metadata) > _MAX_NODES
        or any(type(k) is not str for k in snapshot.metadata)
    ):
        raise ValueError("serving metadata must be a native string-keyed dictionary")
    cache = {}
    cache_bytes = 0
    for key, entry in snapshot.cache.items():
        if type(key) is not str or _CACHE_KEY.fullmatch(key) is None:
            raise ValueError("serving cache requires full feature SHA256 keys")
        if type(entry) is not CachedPrediction:
            raise ValueError("expected an exact cached prediction")
        require_tick(entry.expires_at, "cache expiry")
        if (
            not snapshot.last_cache_tick
            < entry.expires_at
            <= snapshot.last_cache_tick + config.cache_ttl_ticks
        ):
            raise ValueError("cache expiry must lie within the last serving TTL window")
        value = entry.prediction
        spec = snapshot.model_state.policy.spec
        if type(value) is not np.ndarray:
            raise ValueError("cache prediction must be a native array")
        cache_bytes += 64 + _payload_size(entry.expires_at, _MAX_BYTES) + value.nbytes
        if cache_bytes > _MAX_BYTES:
            raise ValueError("serving cache exceeds1MiB aggregate payload")
        if (
            value.dtype != np.dtype("float64")
            or value.ndim != 2
            or value.shape[0] < 1
            or value.shape[1] != 4
            or value.nbytes > _MAX_BYTES
            or not np.isfinite(value).all()
            or np.any(value < spec.lower_bounds)
            or np.any(value > spec.upper_bounds)
        ):
            raise ValueError("cache prediction must be finite bounded float64 Nx4 actions")
        cache[key] = (entry.expires_at, value)
    return dict(
        configuration=(config.cache_ttl_ticks, config.max_cache_entries),
        cache=cache,
        metadata=snapshot.metadata,
        last_cache_tick=snapshot.last_cache_tick,
    )


@dataclass(frozen=True)
class PromotableRobotBinding:
    actor: NearestPromotableActor
    backend: NumpyNearestContinuousBCLearner
    snapshot: NearestServingSnapshot

    @property
    def state(self) -> NearestContinuousBCSnapshot:
        return self.snapshot.model_state

    @property
    def version(self) -> str:
        return self.snapshot.actor_version

    def check(self) -> None:
        actor = self.actor
        if _backend(actor) is not self.backend:
            raise ValueError("admitted promotable actor backend was replaced")
        observed = actor.serving_snapshot()
        auxiliary = _auxiliary(observed)
        if (
            _backend(actor) is not self.backend
            or actor.version != self.version
            or observed.actor_version != self.version
            or observed.generation != self.snapshot.generation
        ):
            raise ValueError("admitted promotable generation, version or backend changed")
        before, after = self.state, observed.model_state
        if not (
            before.policy == after.policy
            and before.completed_updates == after.completed_updates
            and before.training_examples == after.training_examples
            and np.array_equal(before.prototype_features, after.prototype_features)
            and np.array_equal(before.prototype_actions, after.prototype_actions)
            and _owned_values_equal(_auxiliary(self.snapshot), auxiliary)
        ):
            raise ValueError("frozen promotable serving bundle changed")


def bind_promotable_robot(actor: NearestPromotableActor) -> PromotableRobotBinding:
    if type(actor) is not PromotableActor:
        raise ValueError("expected an exact PromotableActor")
    backend, version = _backend(actor), actor.version
    snapshot = actor.serving_snapshot()
    auxiliary = _auxiliary(snapshot)
    _owned_values_equal(auxiliary, auxiliary)
    if actor.version != version or snapshot.actor_version != version:
        raise ValueError("promotable version changed during admission")
    if snapshot.model_state.training_examples == 0:
        raise ValueError("promotable robot actor requires released prototypes")
    binding = PromotableRobotBinding(actor, backend, snapshot)
    binding.check()
    return binding
