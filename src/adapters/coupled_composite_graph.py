"""Fixed data graph projection for the coupled composite byte codec.

Only capture's explicit data records are allowed. Structural set keys precede
node IDs; distinct members with equal keys are refused because outside aliases
cannot be ordered locally without inventing identity. No live authority, model
operations, IO, class loading or payload copying is performed here.
"""

from collections import deque
from dataclasses import fields
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Any

import numpy as np

from src.adapters.coupled_composite_capture import (
    _CoupledGraphPreflight,
    preflight_coupled_composite,
)
from src.adapters.numpy_checkpoint_frames import canonical_json, native_order
from src.core.coupled_energy import CoupledParameters, _validate_parameters
from src.core.managed_composite_state import (
    CompositeCaptureLimits,
    ManagedCompositeState,
    SourceRecord,
)
from src.core.managed_lifecycle_validation import validate_lifecycle_metadata
from src.core.managed_record_state import _aggregate, _observation, _relationships

RECORD_TYPES = MappingProxyType(
    {kind.__module__ + "." + kind.__name__: kind for kind in _CoupledGraphPreflight._record_types}
)
SCALAR_TYPES = (type(None), bool, int, float, str)


def scalar(value: Any, limits: CompositeCaptureLimits) -> None:
    if type(value) not in SCALAR_TYPES:
        raise ValueError("composite token requires an exact supported scalar")
    if type(value) is int and value.bit_length() > 1024:
        raise ValueError("composite integer exceeds supported size")
    if type(value) is float and not isfinite(value):
        raise ValueError("composite scalar must be finite")
    if type(value) is str and len(value.encode("utf8")) > max(
        1024, limits.records.max_identifier_bytes
    ):
        raise ValueError("composite string exceeds original UTF8 bound")


def record_fields(value: Any) -> tuple[str, ...]:
    kind = type(value)
    name = kind.__module__ + "." + kind.__name__
    if RECORD_TYPES.get(name) is not kind:
        raise ValueError("unsupported composite data record")
    names = tuple(field.name for field in fields(kind))
    if vars(value).keys() != set(names):
        raise ValueError("composite record fields differ from exact schema")
    return names


def validate_data(
    state: ManagedCompositeState, limits: CompositeCaptureLimits, max_layers: int
) -> None:
    """Reuse capture validation and pure complete-metadata relationship rules."""
    preflight_coupled_composite(state, limits)
    seen: set[int] = set()
    retained: list[Any] = []

    def visit(value: Any) -> None:
        if type(value) in SCALAR_TYPES or id(value) in seen:
            return
        seen.add(id(value))
        if type(value) is np.ndarray:
            return
        if type(value) is np.random.Generator:
            snapshot = value.bit_generator.state
            retained.append(snapshot)  # Pin nested temporary state IDs too.
            visit(snapshot)
            return
        if type(value) is dict:
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if type(value) in (list, tuple, set, frozenset, deque):
            for item in value:
                visit(item)
            return
        names = record_fields(value)
        if type(value) is CoupledParameters:
            if (
                type(value.hidden_weights) is not tuple
                or not 1 <= len(value.hidden_weights) <= max_layers
            ):
                raise ValueError("coupled native depth exceeds codec max_layers")
            _validate_parameters(value)
        for name in names:
            visit(getattr(value, name))
        # Constructors are fixed trusted data types, never wire-selected code.
        type(value)(**{name: getattr(value, name) for name in names})

    visit(state)
    records = state.records
    if type(records.format_version) is not int or records.format_version != 1:
        raise ValueError("unsupported complete paired metadata version")
    _observation(records.owner, limits.records)
    _aggregate(records, limits.records)
    validate_lifecycle_metadata(records.lifecycle, limits.records)
    _relationships(records)
    _duplicate_observations(state)
    if (
        type(state.retained_payload_bytes) is not int
        or not 0 <= state.retained_payload_bytes <= limits.max_array_bytes
    ):
        raise ValueError("invalid complete retained payload byte observation")


def _source_fields(value: Any, kind: str) -> dict[str, Any]:
    if type(value) is not SourceRecord or value.kind != kind:
        raise ValueError("complete composite source role differs from its stored kind")
    return dict(value.fields)


def _same_observation(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _duplicate_observations(state: ManagedCompositeState) -> None:
    """Bind duplicated observations from one quiescent capture, without authority.

    Historical retired inbox counts and lifecycle ticks may legitimately lag;
    only the identified current holder's exact duplicated fields are compared.
    """
    roles = {
        "actor": {"src.app.actor_shadow.StableActor", "src.app.serving_promotion.PromotableActor"},
        "candidate": {"src.app.actor_shadow.ActorShadowRuntime"},
        "checkpoint": {"src.app.candidate_checkpoint.CandidateCheckpointController"},
        "promotion": {"src.app.serving_promotion.ServingPromotionController"},
    }
    live = [item for item in state.records.lifecycle.registry.holders if item.ready is not None]
    if type(state.holders) is not tuple or len(state.holders) != len(live):
        raise ValueError("complete composite live holder inventory differs from registry")
    holders: dict[int, SourceRecord] = {}
    for pair, enrollment in zip(state.holders, live):
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not int
            or pair[0] != enrollment.enrollment
            or pair[0] in holders
            or type(pair[1]) is not SourceRecord
            or pair[1].kind not in roles[enrollment.kind]
        ):
            raise ValueError("complete composite live holder identity/kind differs from registry")
        if not _same_observation(enrollment.ready, dict(pair[1].fields)["_payload_ready"]):
            raise ValueError("complete composite holder readiness observations disagree")
        holders[pair[0]] = pair[1]
    owner = state.records.owner
    budget = _source_fields(state.budget, "src.app.toy_execution_budget.ToyBudgetSession")
    if not _same_observation(owner.budget_updates, budget["updates_completed"]):
        raise ValueError("complete composite budget update observations disagree")
    current = _source_fields(holders[owner.enrollment], "src.app.actor_shadow.ActorShadowRuntime")
    for field, projected in (
        ("base_actor_version", "_base_actor_version"),
        ("learner_version", "_candidate_version"),
        ("revision", "_revision"),
        ("consolidation_limit", "_consolidation_limit"),
        ("stopped", "_stopped"),
        ("retired", "_retired"),
        ("payload_ready", "_payload_ready"),
    ):
        if not _same_observation(getattr(owner, field), current[projected]):
            raise ValueError("complete composite current owner observations disagree")
    cursor = state.records.consolidation
    if (
        type(current["_attempted_ids"]) is not set
        or type(current["_consolidations"]) is not list
        or tuple(sorted(current["_attempted_ids"])) != cursor.attempted_ids
        or tuple(current["_consolidations"]) != cursor.consolidations
    ):
        raise ValueError("complete composite current consolidation histories disagree")
    inbox = _source_fields(current["_inbox"], "src.app.experience_inbox.ExperienceInbox")
    completed = (
        budget["updates_completed"]
        if inbox["_historical_completed_updates"] is None
        else inbox["_historical_completed_updates"]
    )
    for left, right in (
        (owner.inbox_completed_updates, completed),
        (owner.inbox_last_tick, inbox["_last_time"]),
        (owner.inbox_stopped, inbox["_stopped"]),
        (owner.learner_version, inbox["_learner_version"]),
    ):
        if not _same_observation(left, right):
            raise ValueError("complete composite current inbox observations disagree")
    actors = [holders[item.enrollment] for item in live if item.kind == "actor"]
    if len(actors) != 1:
        raise ValueError("complete composite requires its original single serving holder")
    actor = dict(actors[0].fields)
    if actors[0].kind == "src.app.serving_promotion.PromotableActor":
        slot = _source_fields(actor["_slot"], "src.app.serving_promotion._Slot")
        version = _source_fields(slot["bundle"], "src.app.serving_promotion._Bundle")["version"]
    else:
        version = actor["_version"]
    if not _same_observation(version, owner.serving_actor_version):
        raise ValueError("complete composite serving version observations disagree")


def structural_key(
    value: Any,
    memo: dict[int, tuple[Any, bytes]] | None = None,
    rng_states: dict[int, Any] | None = None,
) -> bytes:
    """Memoized value-only ordering; digests bound aliased-DAG key expansion.

    This is local ordering, never provenance or authority. A key tie is refused,
    including a hypothetical digest collision, rather than resolving by identity.
    """
    known = {} if memo is None else memo
    snapshots = {} if rng_states is None else rng_states

    def project(item: Any) -> bytes:
        kind = type(item)
        if kind in SCALAR_TYPES:
            return sha256(
                canonical_json([kind.__name__, item.hex() if kind is float else item])
            ).digest()
        if id(item) in known:
            return known[id(item)][1]
        if kind is np.ndarray:
            digest = sha256(
                canonical_json(
                    [
                        "array",
                        item.dtype.str,
                        list(item.shape),
                        native_order(item),
                        bool(item.flags.writeable),
                    ]
                )
            )
            # A C-contiguous memoryview (transpose view for F) reads physical
            # bytes without ndarray.tobytes or a payload/frame copy. This also
            # retains signed zero, endian and noncanonical native bool bytes.
            if item.nbytes:
                digest.update(memoryview(item if item.flags.c_contiguous else item.T).cast("B"))
            result = digest.digest()
            known[id(item)] = (item, result)
            return result
        description: list[Any]
        if kind is np.random.Generator:
            if id(item) not in snapshots:
                snapshots[id(item)] = item.bit_generator.state
            description = ["pcg64", project(snapshots[id(item)]).hex()]
        elif kind is dict:
            description = [
                "dict",
                [[project(key).hex(), project(child).hex()] for key, child in item.items()],
            ]
        elif kind in (set, frozenset):
            children = sorted(project(child) for child in item)
            if len(children) != len(set(children)):
                raise ValueError("ambiguous unordered structural keys are unsupported")
            description = [kind.__name__, [child.hex() for child in children]]
        elif kind in (tuple, list, deque):
            description = [
                kind.__name__,
                item.maxlen if kind is deque else None,
                [project(child).hex() for child in item],
            ]
        else:
            description = [
                "record",
                kind.__module__ + "." + kind.__name__,
                [[name, project(getattr(item, name)).hex()] for name in record_fields(item)],
            ]
        result = sha256(canonical_json(description)).digest()
        known[id(item)] = (item, result)
        return result

    return project(value)


def ordered_members(value: set[Any] | frozenset[Any], rng_states: dict[int, Any]) -> list[Any]:
    memo: dict[int, tuple[Any, bytes]] = {}
    keyed = sorted(
        ((structural_key(item, memo, rng_states), item) for item in value), key=lambda pair: pair[0]
    )
    if any(left[0] == right[0] for left, right in zip(keyed, keyed[1:])):
        raise ValueError("ambiguous unordered structural keys are unsupported")
    return [item for _, item in keyed]


class GraphEncoder:
    """Project one bounded first-visit DAG with empty array payload fields."""

    def __init__(self, limits: CompositeCaptureLimits) -> None:
        self.limits = limits
        self.nodes: list[dict[str, Any]] = []
        self.arrays: list[tuple[np.ndarray, dict[str, Any]]] = []
        self.memo: dict[int, int] = {}
        self.references: list[Any] = []
        self.rng_states: dict[int, Any] = {}
        self.edges = 0

    def token(self, value: Any, depth: int = 0) -> dict[str, Any]:
        self.edges += 1
        if self.edges > self.limits.max_nodes or depth > self.limits.max_depth:
            raise ValueError("composite graph exceeds node/depth bound")
        if type(value) in SCALAR_TYPES:
            scalar(value, self.limits)
            return {"value": value}
        if id(value) in self.memo:
            return {"ref": self.memo[id(value)]}
        index = len(self.nodes)
        self.memo[id(value)] = index
        self.references.append(value)  # Keep temporary PCG64 state objects alive.
        node: dict[str, Any] = {}
        self.nodes.append(node)
        kind = type(value)
        child = lambda item: self.token(item, depth + 1)
        if kind is np.ndarray:
            node.update(
                tag="array",
                dtype=value.dtype.str,
                shape=list(value.shape),
                order=native_order(value),
                writeable=bool(value.flags.writeable),
                data="",
            )
            self.arrays.append((value, node))
        elif kind is np.random.Generator:
            if id(value) not in self.rng_states:
                self.rng_states[id(value)] = value.bit_generator.state
            node.update(tag="pcg64", state=child(self.rng_states[id(value)]))
        elif kind is dict:
            node.update(
                tag="dict", items=[[child(key), child(item)] for key, item in value.items()]
            )
        elif kind in (tuple, list, set, frozenset, deque):
            items = ordered_members(value, self.rng_states) if kind in (set, frozenset) else value
            node.update(tag=kind.__name__, items=[child(item) for item in items])
            if kind is deque:
                node["maxlen"] = value.maxlen
        else:
            node.update(
                tag="record",
                type=kind.__module__ + "." + kind.__name__,
                fields=[[name, child(getattr(value, name))] for name in record_fields(value)],
            )
        return {"ref": index}
