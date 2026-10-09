"""Bounded weak/scalar proof of original cleanup ownership, without ports.

The original coordinator supplies leases and performs removal. These helpers
check exact known holder fields and supported native replay containers; they do
not erase, traverse arbitrary graphs, call adapters or retain raw containers.
"""

from collections import deque
from threading import Lock
from types import GetSetDescriptorType
from typing import Any, NamedTuple
from weakref import ReferenceType, ref

from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.erased_inbox_origins import _counter, _schema
from src.app.serving_promotion import PromotableActor, ServingPromotionController, _Bundle, _Slot
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.payload_ownership import OwnedPayloadGroup, PayloadHolderMetadata, PayloadReferences


def _plain(value: Any) -> dict:
    """Reject custom attribute/dict descriptors before borrowing instance data."""
    kind = type(value)
    if (
        type(kind) is not type
        or type.__getattribute__(kind, "__getattribute__") is not object.__getattribute__
    ):
        raise ValueError("expiry cleanup requires plain supported instance attributes")
    descriptor = next(
        (base.__dict__["__dict__"] for base in kind.__mro__ if "__dict__" in base.__dict__), None
    )
    if type(descriptor) is not GetSetDescriptorType:
        raise ValueError("expiry cleanup requires original instance dictionary")
    result = object.__getattribute__(value, "__dict__")
    if (
        type(result) is not dict
        or len(result) > 512
        or any(type(key) is not str or len(key) > 128 for key in result)
    ):
        raise ValueError("expiry cleanup instance fields exceed supported bounds")
    return result


def _native(learner: Any) -> tuple:
    fields = _plain(learner)
    name = "_model" if "_model" in fields else "model"
    model = fields[name]
    native = _plain(model)
    kind: type
    if type(model) is CircadianPredictiveCodingNetwork:
        field, kind = "_replay_memory", deque
    elif native.keys() == {"rows"}:
        field, kind = "rows", list
    else:
        raise ValueError("expiry cleanup native family lacks a pure replay container proof")
    container = native[field]
    if type(container) is not kind:
        raise ValueError("expiry cleanup replay container is unsupported")
    return ref(learner), name, ref(model), field, kind, id(container)


def _bundle(bundle: Any) -> tuple:
    _schema(bundle, _Bundle)
    if type(bundle.cache) is not dict or type(bundle.metadata) is not dict:
        raise ValueError("expiry cleanup requires exact auxiliary dictionaries")
    return ref(bundle), id(bundle.cache), id(bundle.metadata)


def _inventory(life: Any) -> tuple:
    registry = life._registry
    maximum = min(life._policy.holders.max_live_holders, 4096)
    if type(registry._holders) is not dict or len(registry._holders) > maximum:
        raise ValueError("expiry cleanup holder inventory exceeds original bound")
    result = []
    for number, entry in registry._holders.items():
        _counter(number)
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not ReferenceType
        ):
            raise ValueError("expiry cleanup requires exact weak holder entries")
        holder = entry[1]()
        if holder is not None:
            result.append((number, entry[0], entry[1], holder))
    return tuple(result)


class _ProjectionBudget:
    """Conservative metadata grammar, including simultaneous reproof indexes.

    These are logical metadata units, not a measurement of Python heap bytes.
    A model occurrence includes its six-field weak proof and the temporary
    identity/reference indexes used during reproof. Aliases still cost each
    occurrence; no growing projection or weak witness is allocated here.
    """

    def __init__(self, limits: CheckpointContentLimits) -> None:
        _schema(limits, CheckpointContentLimits)
        CheckpointContentLimits.__post_init__(limits)
        self.limits = limits
        self.nodes = self.metadata = 0
        # Why: original due results, the lexical scalar timing token and final
        # scratch coexist with the weak graph. Reserve before any projection.
        self.add(24, 1024)

    def add(self, nodes: int, metadata: int) -> None:
        self.nodes += nodes
        self.metadata += metadata
        if (
            self.nodes > self.limits.max_nodes
            or self.metadata > self.limits.max_metadata_bytes
            or self.limits.max_depth < 6
        ):
            raise ValueError("expiry cleanup projection exceeds aggregate original limits")


def _preflight_projection(life: Any, budget: _ProjectionBudget) -> None:
    """Count exact supported fields before building the first graph projection."""
    registry = life._registry
    _counter(registry._total)
    holders = registry._holders
    maximum = min(life._policy.holders.max_lifetime_enrollments, 4096)
    if type(holders) is not dict or len(holders) > min(life._policy.holders.max_live_holders, 4096):
        raise ValueError("expiry cleanup holder inventory exceeds original bound")
    for number, entry in holders.items():
        _counter(number)
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or entry[0] not in ("actor", "candidate", "checkpoint", "promotion")
            or type(entry[1]) is not ReferenceType
        ):
            raise ValueError("expiry cleanup requires closed original weak holder entries")
        holder = entry[1]()
        if holder is None:
            continue
        budget.add(24, 2048)
        _preflight_holder(budget, entry[0], holder, maximum)


def _preflight_models(budget: _ProjectionBudget, models: Any, maximum: int) -> None:
    if type(models) not in (tuple, list) or len(models) > maximum:
        raise ValueError("expiry cleanup retained model inventory exceeds original bound")
    for learner in models:
        # Charge before inspecting the occurrence; no _native weak tuple yet.
        budget.add(12, 1024)
        fields = _plain(learner)
        model = fields["_model" if "_model" in fields else "model"]
        native = _plain(model)
        kind: type
        if type(model) is CircadianPredictiveCodingNetwork:
            field, kind = "_replay_memory", deque
        elif native.keys() == {"rows"}:
            field, kind = "rows", list
        else:
            raise ValueError("expiry cleanup native family lacks a pure replay container proof")
        if type(native[field]) is not kind:
            raise ValueError("expiry cleanup replay container is unsupported")


def _preflight_bundle(budget: _ProjectionBudget, bundle: Any, maximum: int) -> None:
    budget.add(8, 512)
    _schema(bundle, _Bundle)
    if type(bundle.cache) is not dict or type(bundle.metadata) is not dict:
        raise ValueError("expiry cleanup requires exact auxiliary dictionaries")
    _preflight_models(budget, (bundle.learner,), maximum)


def _preflight_holder(budget: _ProjectionBudget, kind: str, holder: Any, maximum: int) -> None:
    fields = _plain(holder)
    pending = None
    if kind == "actor" and type(holder) is StableActor:
        gate = fields["_read_gate"]
        _preflight_models(budget, (fields["_learner"],), maximum)
    elif kind == "actor" and type(holder) is PromotableActor:
        gate, slot = fields["_read_gate"], fields["_slot"]
        _schema(slot, _Slot)
        _counter(slot.generation)
        budget.add(6, 256)
        _preflight_bundle(budget, slot.bundle, maximum)
        if slot.previous is not None:
            _preflight_bundle(budget, slot.previous, maximum)
    elif kind == "candidate" and type(holder) is ActorShadowRuntime:
        gate = fields["_write_gate"]
        _counter(fields["_revision"])
        budget.add(4, 256)
        _preflight_models(budget, (fields["_candidate"],), maximum)
    elif kind == "checkpoint" and type(holder) is CandidateCheckpointController:
        gate, pending = fields["_gate"], fields["_pending"]
        if type(fields["_models"]) is not list:
            raise ValueError("expiry cleanup requires original retained model list")
        _preflight_models(budget, fields["_models"], maximum)
    elif kind == "promotion" and type(holder) is ServingPromotionController:
        gate, pending = fields["_gate"], fields["_pending"]
        if type(pending) is not dict or len(pending) > maximum:
            raise ValueError("expiry cleanup promotion inventory exceeds original bound")
        for item in pending.values():
            _preflight_bundle(budget, _plain(item)["bundle"], maximum)
    else:
        raise ValueError("expiry cleanup holder family is unsupported")
    if type(gate) is not type(Lock()):
        raise ValueError("expiry cleanup requires original plain gates")
    if pending is not None:
        if type(pending) is not dict or len(pending) > maximum:
            raise ValueError("expiry cleanup requires bounded original pending maps")
        budget.add(4, 256)


def _saved_models(budget: _ProjectionBudget, models: Any, maximum: int) -> None:
    if type(models) is not tuple or len(models) > maximum:
        raise ValueError("expiry cleanup saved models exceed original bound")
    budget.add(len(models) * 12, len(models) * 1024)
    for row in models:
        if (
            type(row) is not tuple
            or len(row) != 6
            or type(row[0]) is not ReferenceType
            or type(row[1]) is not str
            or row[1] not in ("_model", "model")
            or type(row[2]) is not ReferenceType
            or type(row[3]) is not str
            or row[3] not in ("_replay_memory", "rows")
            or (row[4] is not list and row[4] is not deque)
        ):
            raise ValueError("expiry cleanup saved native proof requires closed exact fields")
        _counter(row[5])


def _saved_holder(budget: _ProjectionBudget, row: Any, maximum: int) -> None:
    budget.add(24, 2048)
    if (
        type(row) is not tuple
        or len(row) != 9
        or type(row[0]) is not ReferenceType
        or type(row[1]) is not str
        or row[1] not in ("actor", "candidate", "checkpoint", "promotion")
    ):
        raise ValueError("expiry cleanup saved holder requires closed exact fields")
    _counter(row[2])
    _saved_models(budget, row[3], maximum)
    inboxes, bundles, revision, pending, slot = row[4:]
    if type(inboxes) is not tuple or len(inboxes) > 1:
        raise ValueError("expiry cleanup saved inbox proof exceeds original bound")
    budget.add(len(inboxes) * 4, len(inboxes) * 256)
    if any(type(value) is not ReferenceType for value in inboxes):
        raise ValueError("expiry cleanup saved inbox requires original weak fields")
    if type(bundles) is not tuple or len(bundles) > maximum:
        raise ValueError("expiry cleanup saved bundles exceed original bound")
    budget.add(len(bundles) * 8, len(bundles) * 512)
    for bundle in bundles:
        if type(bundle) is not tuple or len(bundle) != 3 or type(bundle[0]) is not ReferenceType:
            raise ValueError("expiry cleanup saved bundle requires exact weak/scalar fields")
        _counter(bundle[1])
        _counter(bundle[2])
    if revision is not None:
        _counter(revision)
    if pending is not None:
        budget.add(4, 256)
        if type(pending) is not tuple or len(pending) != 2:
            raise ValueError("expiry cleanup saved pending schema changed")
        _counter(pending[0])
        _counter(pending[1])
        if pending[1] > maximum:
            raise ValueError("expiry cleanup saved pending exceeds original bound")
    if slot is not None:
        budget.add(6, 256)
        if (
            type(slot) is not tuple
            or len(slot) != 4
            or type(slot[1]) is not ReferenceType
            or (slot[2] is not None and type(slot[2]) is not ReferenceType)
        ):
            raise ValueError("expiry cleanup saved slot schema changed")
        _counter(slot[0])
        _counter(slot[3])


def _preflight_saved(proof: Any, life: Any, budget: _ProjectionBudget) -> None:
    """Bound retained proof even if actual promotion/previous state has shrunk."""
    for value in (proof.registry_id, proof.holder_map_id, proof.total):
        _counter(value)
    maximum = min(life._policy.holders.max_lifetime_enrollments, 4096)
    if (
        type(proof.entries) is not tuple
        or type(proof.holders) is not tuple
        or len(proof.entries) != len(proof.holders)
        or len(proof.entries) > min(life._policy.holders.max_live_holders, 4096)
    ):
        raise ValueError("expiry cleanup saved holder inventory exceeds original bound")
    for entry, holder in zip(proof.entries, proof.holders):
        if (
            type(entry) is not tuple
            or len(entry) != 4
            or type(entry[1]) is not str
            or entry[1] not in ("actor", "candidate", "checkpoint", "promotion")
        ):
            raise ValueError("expiry cleanup saved enrollment schema changed")
        for value in (entry[0], entry[2], entry[3]):
            _counter(value)
        _saved_holder(budget, holder, maximum)
        if entry[1] != holder[1]:
            raise ValueError("expiry cleanup saved holder kind differs from enrollment")


def _holder(kind: str, holder: Any, maximum: int) -> tuple:
    fields = _plain(holder)
    bundles: tuple = ()
    inboxes: tuple = ()
    revision, pending, slot = None, None, None
    if kind == "actor" and type(holder) is StableActor:
        models, gate = (fields["_learner"],), fields["_read_gate"]
    elif kind == "actor" and type(holder) is PromotableActor:
        slot = fields["_slot"]
        _schema(slot, _Slot)
        _counter(slot.generation)
        originals = (slot.bundle,) if slot.previous is None else (slot.bundle, slot.previous)
        bundles = tuple(_bundle(bundle) for bundle in originals)
        models, gate = tuple(bundle.learner for bundle in originals), fields["_read_gate"]
        slot = (
            id(slot),
            ref(slot.bundle),
            None if slot.previous is None else ref(slot.previous),
            slot.generation,
        )
    elif kind == "candidate" and type(holder) is ActorShadowRuntime:
        models, gate = (fields["_candidate"],), fields["_write_gate"]
        inboxes = (ref(fields["_inbox"]),)
        revision = fields["_revision"]
        _counter(revision)
    elif kind == "checkpoint" and type(holder) is CandidateCheckpointController:
        models, gate, pending = fields["_models"], fields["_gate"], fields["_pending"]
        if type(models) is not list or len(models) > maximum:
            raise ValueError("expiry cleanup retained model inventory exceeds original bound")
        models = tuple(models)
    elif kind == "promotion" and type(holder) is ServingPromotionController:
        gate, pending = fields["_gate"], fields["_pending"]
        if type(pending) is not dict or len(pending) > maximum:
            raise ValueError("expiry cleanup promotion inventory exceeds original bound")
        originals = tuple(_plain(item)["bundle"] for item in pending.values())
        bundles = tuple(_bundle(bundle) for bundle in originals)
        models = tuple(bundle.learner for bundle in originals)
    else:
        raise ValueError("expiry cleanup holder family is unsupported")
    if type(gate) is not type(Lock()) or len(models) > maximum:
        raise ValueError("expiry cleanup requires bounded models and original plain gates")
    if pending is not None and (type(pending) is not dict or len(pending) > maximum):
        raise ValueError("expiry cleanup requires bounded original pending maps")
    return (
        ref(holder),
        kind,
        id(gate),
        tuple(_native(model) for model in models),
        inboxes,
        bundles,
        revision,
        None if pending is None else (id(pending), len(pending)),
        slot,
    )


class CleanupGraphProof(NamedTuple):
    registry_id: int
    holder_map_id: int
    total: int
    entries: tuple
    holders: tuple
    limits: Any
    limit_values: tuple


def observe_cleanup_graph(life: Any, limits: CheckpointContentLimits) -> CleanupGraphProof:
    """Borrow exact fields before opaque enumeration; retain weak/scalar pins."""
    _preflight_projection(life, _ProjectionBudget(limits))
    entries = _inventory(life)
    maximum = min(life._policy.holders.max_lifetime_enrollments, 4096)
    holders = tuple(_holder(kind, holder, maximum) for _, kind, _, holder in entries)
    _counter(life._registry._total)
    return CleanupGraphProof(
        id(life._registry),
        id(life._registry._holders),
        life._registry._total,
        tuple(
            (number, kind, id(reference), id(holder)) for number, kind, reference, holder in entries
        ),
        holders,
        ref(limits),
        (limits.max_nodes, limits.max_metadata_bytes, limits.max_array_bytes, limits.max_depth),
    )


def require_cleanup_graph(
    proof: Any, life: Any, limits: CheckpointContentLimits, *, final: bool, groups: Any
) -> None:
    """Allow only actual invalidation/replacement, then require empty raw state."""
    if type(proof) is not CleanupGraphProof:
        raise ValueError("expiry cleanup lost its original graph proof")
    _schema(limits, CheckpointContentLimits)
    CheckpointContentLimits.__post_init__(limits)
    if (
        type(proof.limits) is not ReferenceType
        or proof.limits() is not limits
        or type(proof.limit_values) is not tuple
        or len(proof.limit_values) != 4
        or any(type(value) is not int or not 0 < value < 2**63 for value in proof.limit_values)
        or proof.limit_values
        != (limits.max_nodes, limits.max_metadata_bytes, limits.max_array_bytes, limits.max_depth)
    ):
        raise ValueError("expiry cleanup original projection limits changed")
    budget = _ProjectionBudget(limits)
    _preflight_saved(proof, life, budget)
    _preflight_projection(life, budget)
    entries = _inventory(life)
    current = tuple(
        (number, kind, id(reference), id(holder)) for number, kind, reference, holder in entries
    )
    if (
        id(life._registry) != proof.registry_id
        or id(life._registry._holders) != proof.holder_map_id
        or life._registry._total != proof.total
        or current != proof.entries
    ):
        raise ValueError("expiry cleanup original holder authority changed")
    if type(groups) is not tuple or len(groups) != len(proof.holders):
        raise ValueError("expiry cleanup original opaque group inventory changed")
    for group, entry, saved in zip(groups, proof.entries, proof.holders):
        _schema(group, OwnedPayloadGroup)
        _schema(group.holder, PayloadHolderMetadata)
        _schema(group.references, PayloadReferences)
        if (
            type(group.holder.kind) is not str
            or type(group.holder.enrollment) is not int
            or group.holder.ready is not True
        ):
            raise ValueError("expiry cleanup opaque group metadata is unsupported")
        if (group.holder.enrollment, group.holder.kind) != entry[:2]:
            raise ValueError("expiry cleanup opaque group authority differs from original fields")
        expected = (
            tuple(id(model[0]()) for model in saved[3]),
            tuple(id(inbox()) for inbox in saved[4]),
            tuple(identity for bundle in saved[5] for identity in (bundle[2], bundle[1])),
        )
        for values, identities in zip(
            (group.references.models, group.references.inboxes, group.references.auxiliary),
            expected,
        ):
            if (
                type(values) is not tuple
                or len(values) > 4096
                or len(values) != len(identities)
                or tuple(id(value) for value in values) != identities
            ):
                raise ValueError("expiry cleanup opaque ownership differs from original fields")
        if final and any(type(value) is not dict or value for value in group.references.auxiliary):
            raise ValueError("expiry cleanup original owned auxiliary removal is incomplete")
    for (
        reference,
        kind,
        gate_id,
        models,
        inboxes,
        bundles,
        revision,
        pending,
        slot,
    ) in proof.holders:
        holder = reference()
        if holder is None:
            raise ValueError("expiry cleanup original holder expired")
        fields = _plain(holder)
        gate = fields[
            "_read_gate" if kind == "actor" else "_write_gate" if kind == "candidate" else "_gate"
        ]
        if id(gate) != gate_id or type(gate) is not type(Lock()) or not gate.locked():
            raise ValueError("expiry cleanup original holder gate changed")
        if kind == "candidate":
            _counter(fields["_revision"])
            if fields["_revision"] != revision + int(final) or fields["_stopped"] is not False:
                raise ValueError("expiry cleanup actual candidate invalidation changed")
        if pending is not None and (
            type(fields["_pending"]) is not dict
            or id(fields["_pending"]) != pending[0]
            or len(fields["_pending"]) != (0 if final else pending[1])
        ):
            raise ValueError("expiry cleanup actual pending invalidation changed")
        if final and kind == "promotion" and fields["_latest"] is not None:
            raise ValueError("expiry cleanup promotion history was not invalidated")
        if slot is not None:
            current_slot = fields["_slot"]
            _schema(current_slot, _Slot)
            _counter(current_slot.generation)
            if (
                current_slot.bundle is not slot[1]()
                or current_slot.generation != slot[3] + int(final)
                or (
                    current_slot.previous is not None
                    if final
                    else current_slot.previous is not (None if slot[2] is None else slot[2]())
                )
                or (not final and id(current_slot) != slot[0])
            ):
                raise ValueError("expiry cleanup serving slot differs from original replacement")
        # Stored model references are weak; compare actual holder fields too.
        if kind == "candidate":
            actual = (fields["_candidate"],)
        elif kind == "actor":
            actual = (
                (fields["_learner"],)
                if slot is None
                else tuple(
                    bundle[0]().learner for bundle in bundles[: 1 if final else len(bundles)]
                )
            )
        elif kind == "checkpoint":
            actual = fields["_models"]
            if type(actual) is not list or len(actual) > 4096:
                raise ValueError("expiry cleanup final retained model inventory changed")
        else:
            actual = (
                ()
                if final
                else tuple(_plain(item)["bundle"].learner for item in fields["_pending"].values())
            )
        expected = tuple(model[0]() for model in models)
        if tuple(id(model) for model in actual) != tuple(
            id(model)
            for model in (
                expected[:1]
                if final and slot is not None
                else ()
                if final and kind == "promotion"
                else expected
            )
        ):
            raise ValueError("expiry cleanup original holder model inventory changed")
        for model_reference, name, native_reference, field, container_kind, container_id in models:
            model, native = model_reference(), native_reference()
            if model is None or native is None or _plain(model).get(name) is not native:
                raise ValueError("expiry cleanup original native model changed or expired")
            container = _plain(native).get(field)
            if (
                type(container) is not container_kind
                or id(container) != container_id
                or (final and container)
            ):
                raise ValueError("expiry cleanup final native replay removal is incomplete")
        for inbox_reference in inboxes:
            inbox = inbox_reference()
            if inbox is None or fields["_inbox"] is not inbox:
                raise ValueError("expiry cleanup original inbox changed")
            for mapping in (inbox._experiences, inbox._labels):
                if type(mapping) is not dict or len(mapping) > 4096 or (final and mapping):
                    raise ValueError("expiry cleanup final raw inbox removal is incomplete")
        for bundle_reference, cache_id, metadata_id in bundles:
            bundle = bundle_reference()
            if bundle is None:
                if final and (
                    kind == "promotion"
                    or (slot is not None and bundle_reference is not bundles[0][0])
                ):
                    continue
                raise ValueError("expiry cleanup original auxiliary owner expired")
            _schema(bundle, _Bundle)
            for value, identity in ((bundle.cache, cache_id), (bundle.metadata, metadata_id)):
                if type(value) is not dict or id(value) != identity or (final and value):
                    raise ValueError("expiry cleanup final auxiliary removal is incomplete")
