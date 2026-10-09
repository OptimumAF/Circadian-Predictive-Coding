"""Exact NumPy complete source projection and bounded graph preflight.

Reads native state without payload copies or native snapshot methods. The app
admits original raw payload bytes before copying the full graph once. No codecs,
disk IO, reconstruction, training, publication or scientific authority.
"""

from collections import deque
from copy import deepcopy
from dataclasses import fields
from math import isfinite

import numpy as np

from src.adapters.numpy_learners import BackpropLearner, CircadianLearner, BackpropSnapshot
from src.adapters.backprop_checkpoint_codec import _FIELDS, _shapes, _native_arrays
from src.adapters.circadian_checkpoint_codec import _native_body, _specifications, _preflight
from src.adapters.circadian_checkpoint_schema import validate_complete_native
from src.adapters.managed_record_checkpoint_schema import PAIR_SCHEMAS
from src.app.candidate_checkpoint import CandidateCheckpointView, _validate_view
from src.app.circadian_checkpoint import CircadianResumePosition
from src.app.managed_composite_sources import SCHEMAS
from src.app.managed_composite_capture import capture_managed_composite
from src.adapters.numpy_replay_capture import numpy_replay_inventory
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianNetworkSnapshot,
    CircadianConfig,
    ReplaySnapshot,
    ReplayRetentionBudget,
    CircadianPredictiveCodingNetwork,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.resource_sharing import SharingLimits, SharingSnapshot
from src.core.inbox_cursor import InboxCursor, validate_inbox_cursor
from src.core.inbox_codec_policy import NUMERIC_DTYPES
from src.core.inbox_codec_policy import InboxCodecPolicy
from src.adapters.inbox_checkpoint_schema import native_data, metadata_cursor
from src.adapters.inbox_checkpoint_payloads import specifications
from src.core.experience import Experience, LabelArrival, AppliedExperience, ExperiencePermissions
from src.core.data_erasure import ErasedExperience
from src.core.serving_ports import (
    ServingConfiguration,
    CachedPrediction,
    PromotionReceipt,
    PreparedPromotion,
)
from src.core.promotion_guard import (
    PromotionPolicy,
    PromotionEvidence,
    PromotionDecision,
    PromotionGuardReport,
)
from src.core.checkpoint_codec import CodecLimits
from src.core.managed_composite_state import (
    AuthorityPath,
    SourceRecord,
    NativeProjection,
    ManagedCompositeState,
)
from src.core.managed_lifecycle_state import AuthorityReference
from src.app.toy_execution_budget import ToyExecutionBudget
from src.shared.process_memory import ProcessRssSegment

NATIVE_FIELDS = {
    BackpropLearner: {"_model", "_learning_rate"},
    CircadianLearner: {"_model", "_learning_rate", "_inference_steps", "_inference_learning_rate"},
}
RECORDS = set(PAIR_SCHEMAS) | {
    AuthorityPath,
    SourceRecord,
    ManagedCompositeState,
    BackpropSnapshot,
    CircadianNetworkSnapshot,
    CircadianConfig,
    ReplaySnapshot,
    ReplayRetentionBudget,
    ReplayRetentionPolicy,
    CandidateCheckpointView,
    CircadianResumePosition,
    InboxCursor,
    Experience,
    LabelArrival,
    AppliedExperience,
    ExperiencePermissions,
    ErasedExperience,
    ServingConfiguration,
    CachedPrediction,
    PromotionReceipt,
    PreparedPromotion,
    PromotionPolicy,
    PromotionEvidence,
    PromotionDecision,
    PromotionGuardReport,
    ToyExecutionBudget,
    ProcessRssSegment,
    SharingLimits,
    SharingSnapshot,
}
SOURCE_FIELDS = {kind.__module__ + "." + kind.__name__: set(spec) for kind, spec in SCHEMAS.items()}
SOURCE_FIELDS.update(
    {kind.__module__ + "." + kind.__name__: spec for kind, spec in NATIVE_FIELDS.items()}
)


def _native_limits(limits):
    return CodecLimits(
        max(4096, limits.max_nodes * 256 + limits.max_array_bytes * 4),
        limits.max_array_bytes,
        limits.max_depth,
        limits.max_dimension,
    )


def project_numpy_native(learner, path, limits):
    spec = NATIVE_FIELDS.get(type(learner))
    if spec is None or vars(learner).keys() != spec:
        raise ValueError("unsupported complete original NumPy learner schema")
    model = learner._model
    snapshot: BackpropSnapshot | CircadianNetworkSnapshot
    if type(learner) is BackpropLearner:
        if type(model) is not BackpropMLP or vars(model).keys() != _FIELDS:
            raise ValueError("unsupported complete original Backprop model")
        snapshot = BackpropSnapshot(1, model.input_dim, model.hidden_dims, vars(model))
    else:
        if type(model) is not CircadianPredictiveCodingNetwork:
            raise ValueError("unsupported complete original Circadian model")
        snapshot = CircadianNetworkSnapshot(
            2,
            model.input_dim,
            model.hidden_dims,
            model._min_hidden_dim,
            model.max_hidden_dim,
            model.config,
            vars(model),
        )
    _validate_native(snapshot, limits)
    policy = tuple(
        (name, snapshot if name == "_model" else getattr(learner, name)) for name in sorted(spec)
    )
    for name, value in policy:
        if name != "_model" and (
            type(value) not in (int, float)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError("original native training policy invalid")
    if type(learner) is CircadianLearner and type(learner._inference_steps) is not int:
        raise ValueError("original inference steps require exact integer")
    return NativeProjection(
        SourceRecord(type(learner).__module__ + "." + type(learner).__name__, policy),
        (
            AuthorityReference(path + ".original", learner),
            AuthorityReference(path + "._model.original", model),
        ),
    )


def _validate_native(snapshot, limits):
    bound = _native_limits(limits)
    if type(snapshot) is BackpropSnapshot:
        if (
            type(snapshot.state) is not dict
            or snapshot.state.keys() != _FIELDS
            or type(snapshot.format_version) is not int
            or snapshot.format_version != 1
        ):
            raise ValueError("complete Backprop fields changed")
        if (
            snapshot.state["input_dim"] != snapshot.input_dim
            or snapshot.state["hidden_dims"] != snapshot.hidden_dims
        ):
            raise ValueError("Backprop snapshot topology differs")
        shapes = _shapes(
            snapshot.input_dim, snapshot.hidden_dims, snapshot.state["_traffic_steps"], bound
        )
        _native_arrays(snapshot.state, shapes)
    elif type(snapshot) is CircadianNetworkSnapshot:
        body = _native_body(snapshot, bound)
        _preflight(body, _specifications(body, False, bound), bound, False)
        validate_complete_native(snapshot)
    else:
        raise ValueError("unsupported complete native snapshot")


class GraphPreflight:
    def __init__(self, limits):
        self.limits = limits
        self.nodes = 0
        self.seen: set[int] = set()
        self.active: set[int] = set()
        self.arrays: list[np.ndarray] = []
        self.array_bytes = 0
        self.budget_updates = 0

    def visit(self, value, depth=0):
        self.nodes += 1
        if self.nodes > self.limits.max_nodes or depth > self.limits.max_depth:
            raise ValueError("complete composite graph exceeds node/depth bound")
        if value is None or type(value) in (bool, int, float, str):
            self.scalar(value)
            return
        if id(value) in self.active:
            raise ValueError("unsupported cyclic payload graph")
        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        self.active.add(id(value))
        if type(value) is np.ndarray:
            self.array(value)
        elif type(value) is np.random.Generator:
            if type(value.bit_generator) is not np.random.PCG64:
                raise ValueError("unsupported native generator")
            self.visit(value.bit_generator.state, depth + 1)
        elif type(value) in (list, tuple, set, frozenset, deque, dict):
            if len(value) > self.limits.max_nodes:
                raise ValueError("complete composite collection exceeds bound")
            if type(value) is dict:
                for key, item in value.items():
                    if type(key) not in (str, int, tuple):
                        raise ValueError("unsupported composite dictionary key")
                    self.visit(key, depth + 1)
                    self.visit(item, depth + 1)
            else:
                for item in value:
                    self.visit(item, depth + 1)
        elif type(value) in RECORDS:
            if vars(value).keys() != {field.name for field in fields(value)}:
                raise ValueError("complete composite record fields changed")
            if type(value) is SourceRecord:
                names = tuple(name for name, _ in value.fields)
                if value.kind not in SOURCE_FIELDS or names != tuple(
                    sorted(SOURCE_FIELDS[value.kind])
                ):
                    raise ValueError("unknown or incomplete projected source fields")
                self.source(value)
            if type(value) in (BackpropSnapshot, CircadianNetworkSnapshot):
                _validate_native(value, self.limits)
            for field in fields(value):
                self.visit(getattr(value, field.name), depth + 1)
            if type(value) is InboxCursor:
                validate_inbox_cursor(value)
            if type(value) is CandidateCheckpointView:
                _preflight_checkpoint_inbox(value, self.limits)
                _validate_view(value)
        else:
            raise ValueError("unsupported complete composite payload value")
        self.active.remove(id(value))

    def source(self, value):
        data = dict(value.fields)
        if value.kind == "src.app.actor_shadow.ActorShadowRuntime":
            inbox = dict(data["_inbox"].fields)
            native = dict(data["_candidate"].fields)["_model"]
            if any(
                type(data[name]) is not bool for name in ("_stopped", "_retired", "_payload_ready")
            ):
                raise ValueError("retained candidate flags require exact booleans")
            cursor = InboxCursor(
                2 if inbox["_erased"] else 1,
                inbox["_learner_version"],
                inbox["_capacity"],
                tuple(inbox["_experiences"].values()),
                tuple(inbox["_labels"].values()),
                tuple(inbox["_applied"].values()),
                inbox["_last_time"],
                inbox["_stopped"],
                inbox["_historical_completed_updates"]
                if inbox["_historical_completed_updates"] is not None
                else self.budget_updates,
                tuple(inbox["_erased"].values()),
            )
            policy = InboxCodecPolicy(
                native.input_dim,
                self.limits.max_dimension,
                self.limits.records.max_records,
                self.limits.records.max_identifier_bytes,
                self.limits.records.max_records,
            )
            raw = native_data(cursor, policy)
            metadata_cursor(raw, policy)
            specifications(raw, policy, _native_limits(self.limits), False)

    def scalar(self, value):
        if type(value) is float and not isfinite(value):
            raise ValueError("nonfinite complete composite scalar")
        if type(value) is int and value.bit_length() > 1024:
            raise ValueError("oversized complete composite integer")
        if type(value) is str and len(value.encode("utf8")) > max(
            1024, self.limits.records.max_identifier_bytes
        ):
            raise ValueError("oversized complete composite string")

    def array(self, value):
        if value.dtype.str not in (*NUMERIC_DTYPES, "|b1") or not 0 <= value.ndim <= 2:
            raise ValueError("unsupported complete composite array dtype/rank")
        if any(size > self.limits.max_dimension for size in value.shape):
            raise ValueError("complete composite array dimension exceeds bound")
        if not (value.flags.c_contiguous or value.flags.f_contiguous):
            raise ValueError("unsupported complete composite array layout")
        self.array_bytes += value.nbytes
        if self.array_bytes > self.limits.max_array_bytes:
            raise ValueError("complete composite arrays exceed independent byte bound")
        if not np.all(np.isfinite(value)):
            raise ValueError("nonfinite complete composite array")
        if any(np.shares_memory(value, other) for other in self.arrays):
            raise ValueError("unsupported distinct overlapping composite arrays")
        self.arrays.append(value)


def preflight_numpy_composite(state, limits):
    if (
        type(state) is not ManagedCompositeState
        or type(state.format_version) is not int
        or state.format_version != 1
    ):
        raise ValueError("unsupported complete composite format")
    graph = GraphPreflight(limits)
    graph.budget_updates = _budget_updates(state)
    graph.visit(state)


def copy_numpy_composite(state, limits, memo):
    graph = GraphPreflight(limits)
    graph.budget_updates = _budget_updates(state)
    graph.visit(state)
    detached = deepcopy(state, memo)
    for original in graph.arrays:
        copied = memo[id(original)]
        if type(copied) is not np.ndarray:
            raise ValueError("complete array copy changed native type")
        copied.flags.writeable = original.flags.writeable
    return detached


def _budget_updates(state) -> int:
    value = dict(state.budget.fields)["updates_completed"]
    if type(value) is not int or value < 0:
        raise ValueError("original complete budget requires nonnegative committed updates")
    assert isinstance(value, int)
    return value


def _preflight_checkpoint_inbox(view, limits):
    snapshot = view.state
    if type(snapshot) not in (BackpropSnapshot, CircadianNetworkSnapshot):
        raise ValueError("retained checkpoint requires complete supported native snapshot")
    policy = InboxCodecPolicy(
        snapshot.input_dim,
        limits.max_dimension,
        limits.records.max_records,
        limits.records.max_identifier_bytes,
        limits.records.max_records,
    )
    raw = native_data(view.inbox, policy)
    metadata_cursor(raw, policy)
    specifications(raw, policy, _native_limits(limits), False)


def capture_numpy_managed_composite(owner, *, limits, replay_origins=None):
    """Compose supported NumPy ports with the inward original-owner coordinator."""
    return capture_managed_composite(
        owner,
        limits=limits,
        project_native=project_numpy_native,
        preflight=preflight_numpy_composite,
        copy_state=copy_numpy_composite,
        replay_inventory=numpy_replay_inventory,
        replay_origins=replay_origins,
    )
