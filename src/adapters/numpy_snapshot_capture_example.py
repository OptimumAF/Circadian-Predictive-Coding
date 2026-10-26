"""Fixed issued-checkpoint capture example using original managed replay proofs.

Setup performs one authorized native wake and issues one genuine checkpoint.
Capture/report then reads full state through borrowed projections and existing
component codecs, with no native methods. Results are bounded same-process DATA;
they grant no restoration, publication, portable authority or recovery claim.
"""

from collections import deque
from dataclasses import asdict, fields, is_dataclass
from hashlib import sha256
import os
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from src.adapters.circadian_checkpoint_codec import CircadianCheckpointCodec
from src.adapters.numpy_capture_example import (
    BINDING,
    CAPTURE_LIMITS,
    NATIVE_LIMITS,
    _callbacks,
    _plain,
    _policy,
    _require,
    _same_budget,
)
from src.adapters.numpy_checkpoint_frames import canonical_json
from src.adapters.numpy_composite_capture import (
    capture_numpy_managed_composite,
    project_numpy_native,
)
from src.adapters.numpy_inbox_checkpoint_codec import NumpyInboxCheckpointCodec
from src.adapters.numpy_learners import CircadianLearner, make_managed_data_lifecycle
from src.adapters.numpy_replay_copies import replay_builder_source
from src.adapters.numpy_replay_graphs import replay_graph_payload_bytes, replay_graph_rows
from src.adapters.numpy_replay_origins import (
    replay_copy_bytes,
    replay_model_reference,
    replay_payload_fingerprint,
    replay_payload_references,
    retained_replay_references,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import (
    CandidateCheckpointController,
    CandidateCheckpointView,
    _budget_state,
)
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.inbox_codec_policy import InboxCodecPolicy
from src.core.inbox_cursor import InboxCursor
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.replay_graph_origin import ReplayGraphPorts
from src.core.replay_origin import ReplayOriginLimits, ReplayOriginPorts
from src.core.replay_write_origin import ReplayWriteLimits
from src.core.resource_sharing import SharingLimits

_GRAPH_KEYS = frozenset(
    (
        "owner",
        "life",
        "runtime",
        "ledger",
        "controller",
        "manager",
        "token",
        "pending",
        "view",
        "snapshot",
        "source",
        "shared",
        "actor",
        "budget",
        "clock",
        "gate",
        "callbacks",
        "codec",
        "features",
        "targets",
        "wall_clock",
        "builder",
    )
)


def _retention() -> DataRetentionPolicy:
    return DataRetentionPolicy(
        4096,
        120,
        PayloadOwnershipLimits(8, 12),
        owned_payload_copies=PayloadCopyLimits(4096),
        max_retention_seconds=1000.0,
    )


def _origin_limits() -> ReplayOriginLimits:
    return ReplayOriginLimits(16, 64, 32, 131072, 120, 4096)


def _inbox_policy() -> InboxCodecPolicy:
    return InboxCodecPolicy(3, 32, 64, 128, 64)


def _setup_snapshot_case(
    *,
    retain_capture_history: bool = False,
    retain_inbox_origins: bool = False,
    max_updates: int = 2,
    origin_limits: ReplayOriginLimits | None = None,
) -> dict[str, Any]:
    """Return trusted original handles; this is the sole native-work boundary."""
    if type(retain_capture_history) is not bool:
        raise ValueError("retain_capture_history requires an exact boolean")
    if type(retain_inbox_origins) is not bool:
        raise ValueError("retain_inbox_origins requires an exact boolean")
    if type(max_updates) is not int or not 0 < max_updates < 2**63:
        raise ValueError("snapshot setup requires a positive bounded native update limit")
    if origin_limits is not None:
        if type(origin_limits) is not ReplayOriginLimits:
            raise ValueError("snapshot setup requires exact original replay origin limits")
        ReplayOriginLimits.__post_init__(origin_limits)
    clock = LogicalClock()
    wall_clock = lambda: 0.0
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=max_updates, max_wall_seconds=1), wall_clock
    )
    model = CircadianPredictiveCodingNetwork(
        3,
        2,
        seed=23,
        min_hidden_dim=2,
        circadian_config=CircadianConfig(sleep_mode="disabled"),
    )
    model.configure_replay_retention(ReplayRetentionBudget(2, 64))
    source = CircadianLearner(
        model, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
    )
    gate = ServingPriorityGate(SharingLimits(1, 8, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4))
    life = make_managed_data_lifecycle(owner, policy=_retention())
    ledger = ManagedReplayOrigins(
        owner,
        ReplayOriginPorts(
            replay_model_reference,
            retained_replay_references,
            replay_copy_bytes,
            replay_payload_references,
            replay_payload_fingerprint,
        ),
        _origin_limits() if origin_limits is None else origin_limits,
        ReplayWriteLimits(16, 4, 64),
        retain_inbox_origins=retain_inbox_origins,
    )
    features = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    targets = np.array([[0.0]], dtype=np.float64)
    owner.declare(
        LifecycleDeclaration(
            ("e1", "s1"),
            DataProvenance("local", "person-1", True, False),
            DataConsent(True, True),
            "replay",
        )
    )
    owner.record_experience(
        Experience(
            "s1",
            "e1",
            1,
            "actor-0",
            features,
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(LabelArrival("label-s1", "s1", "e1", 3, "actor-0", targets))
    clock.advance_to(3)
    poll = ledger.train_ready()
    _require(
        len(poll.updates) == budget.updates_completed == 1,
        "snapshot setup did not commit its one authorized wake",
    )
    gate.pause()
    codec = CircadianCheckpointCodec()
    callbacks = _callbacks(runtime._candidate, "circadian", codec)
    builder = callbacks["build"]
    controller = CandidateCheckpointController(
        shared,
        build_learner=builder,
        state_digest=callbacks["state_digest"],
        policy_digest=callbacks["policy_digest"],
    )
    manager = ManagedReplayCheckpoints(
        ledger,
        ReplayGraphPorts(replay_graph_rows, replay_graph_payload_bytes),
        builder_source=replay_builder_source,
        retain_capture_history=retain_capture_history,
    )
    token = manager.capture(controller)
    pending = controller._pending[token]
    charge = life.payload_byte_snapshot()
    _require(
        charge.charged_bytes == 224 and charge.observed_retained_bytes == 128,
        "original issued snapshot setup accounting changed",
    )
    return dict(
        owner=owner,
        life=life,
        runtime=runtime,
        ledger=ledger,
        controller=controller,
        manager=manager,
        token=token,
        pending=pending,
        view=pending.view,
        snapshot=pending.view.state,
        source=source,
        shared=shared,
        actor=runtime.actor,
        budget=budget,
        clock=clock,
        gate=gate,
        callbacks=callbacks,
        codec=codec,
        features=features,
        targets=targets,
        wall_clock=wall_clock,
        builder=builder,
    )


def _identity(graph: dict[str, Any]) -> dict[str, int]:
    result = {
        name: id(graph[name])
        for name in (
            "owner",
            "life",
            "runtime",
            "ledger",
            "controller",
            "manager",
            "token",
            "pending",
            "view",
            "snapshot",
            "actor",
            "budget",
            "clock",
        )
    }
    result.update(
        copies=id(graph["manager"]._copies),
        cursor=id(graph["view"].inbox),
        candidate=id(graph["runtime"]._candidate),
        copy_budget=id(graph["life"]._copy_budget),
    )
    return result


def _native(learner: Any) -> Any:
    return dict(project_numpy_native(learner, "snapshot-demo", CAPTURE_LIMITS).state.fields)[
        "_model"
    ]


def _view_metadata(view: Any) -> dict[str, Any]:
    return _plain(
        {
            field.name: getattr(view, field.name)
            for field in fields(CandidateCheckpointView)
            if field.name not in ("state", "inbox")
        }
    )


def _components(graph: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, Any]]:
    runtime, view = graph["runtime"], graph["view"]
    native = dict(
        actor=_native(runtime.actor._learner),
        candidate=_native(runtime._candidate),
        issued=view.state,
    )
    native_frames = {
        name: graph["codec"].encode(value, binding=BINDING, limits=NATIVE_LIMITS)
        for name, value in native.items()
    }
    inbox_codec = NumpyInboxCheckpointCodec(_inbox_policy())
    inbox_frames = {
        name: inbox_codec.encode(value, binding=BINDING, limits=NATIVE_LIMITS)
        for name, value in dict(current=runtime._inbox._borrow_cursor(), issued=view.inbox).items()
    }
    return native_frames, inbox_frames, _view_metadata(view)


def _holder(graph: dict[str, Any], captured: Any, original: Any) -> dict[str, Any]:
    number = next(
        number
        for number, (_, reference) in graph["life"]._registry._holders.items()
        if reference() is original
    )
    return dict(dict(captured.state.holders)[number].fields)


def _detached(graph: dict[str, Any], captured: Any) -> tuple[dict[str, Any], Any, Any]:
    runtime = _holder(graph, captured, graph["runtime"])
    actor = _holder(graph, captured, graph["actor"])
    checkpoint = _holder(graph, captured, graph["controller"])
    _require(
        len(checkpoint["_pending"]) == 1 and not checkpoint["_models"],
        "captured checkpoint inventory changed",
    )
    view = dict(checkpoint["_pending"][0][1].fields)["view"]
    native = dict(
        actor=dict(actor["_learner"].fields)["_model"],
        candidate=dict(runtime["_candidate"].fields)["_model"],
        issued=view.state,
    )
    return native, dict(runtime["_inbox"].fields), view


def _array_detachment(original: Any, detached: Any, memo: dict[int, Any]) -> int:
    """Compare actual joint graph identities; component hashes cannot prove aliases."""
    if id(original) in memo:
        _require(memo[id(original)] is detached, "joint capture lost an original graph alias")
        return 0
    if type(original) is np.ndarray:
        _require(
            type(detached) is np.ndarray
            and not np.shares_memory(original, detached)
            and original.dtype == detached.dtype
            and original.shape == detached.shape
            and np.array_equal(original, detached),
            "captured array changed or remained borrowed",
        )
        _require(
            not any(detached is previous for previous in memo.values()),
            "joint capture merged distinct original arrays",
        )
        memo[id(original)] = detached
        return 1
    memo[id(original)] = detached
    if type(original) is dict:
        _require(
            type(detached) is dict and original.keys() == detached.keys(),
            "captured native dictionary changed",
        )
        return sum(_array_detachment(value, detached[key], memo) for key, value in original.items())
    if type(original) in (tuple, list, deque):
        _require(
            type(detached) is type(original) and len(original) == len(detached),
            "captured native collection changed",
        )
        return sum(_array_detachment(a, b, memo) for a, b in zip(original, detached))
    if is_dataclass(original) and not isinstance(original, type):
        _require(type(detached) is type(original), "captured native record changed")
        return sum(
            _array_detachment(getattr(original, field.name), getattr(detached, field.name), memo)
            for field in fields(original)
        )
    return 0


def _captured_cursor(
    captured: Any, data: dict[str, Any], *, completed_updates: int | None = None
) -> Any:
    """Represent detached inbox DATA for its codec, never construct a live inbox."""
    # A retired inbox must receive its historical count before cursor validation;
    # the current candidate continues to observe the original global budget.
    observed_updates = (
        dict(captured.state.budget.fields)["updates_completed"]
        if completed_updates is None
        else completed_updates
    )
    return InboxCursor(
        2 if data["_erased"] else 1,
        data["_learner_version"],
        data["_capacity"],
        tuple(data["_experiences"][key] for key in sorted(data["_experiences"])),
        tuple(data["_labels"][key] for key in sorted(data["_labels"])),
        tuple(data["_applied"].values()),
        data["_last_time"],
        data["_stopped"],
        observed_updates,
        tuple(data["_erased"][key] for key in sorted(data["_erased"])),
    )


def _capture_snapshot_case(graph: dict[str, Any]) -> dict[str, Any]:
    """Capture/report only; native methods must remain unavailable after setup."""
    if type(graph) is not dict or graph.keys() != _GRAPH_KEYS:
        raise ValueError("issued snapshot report requires the complete original setup graph")
    before_native, before_inbox, before_view = _components(graph)
    identities = _identity(graph)
    budget_before = _plain(_budget_state(graph["budget"]))
    charge_before = graph["life"].payload_byte_snapshot()
    origins_before = graph["ledger"].accounting()
    try:
        capture_numpy_managed_composite(
            graph["owner"], limits=CAPTURE_LIMITS, replay_origins=graph["ledger"]
        )
    except ValueError as error:
        refusal = str(error)
    else:
        raise AssertionError("issued snapshot acquired capture permission without its manager")
    _require(
        graph["life"]._copy_budget._charged == charge_before.charged_bytes,
        "default-off refusal charged a raw copy",
    )
    captured = capture_numpy_managed_composite(
        graph["owner"],
        limits=CAPTURE_LIMITS,
        replay_origins=graph["ledger"],
        replay_checkpoints=graph["manager"],
    )
    detached_native, detached_inbox, detached_view = _detached(graph, captured)
    detached_native_frames = {
        name: graph["codec"].encode(value, binding=BINDING, limits=NATIVE_LIMITS)
        for name, value in detached_native.items()
    }
    inbox_codec = NumpyInboxCheckpointCodec(_inbox_policy())
    detached_inbox_frames = {
        name: inbox_codec.encode(value, binding=BINDING, limits=NATIVE_LIMITS)
        for name, value in dict(
            current=_captured_cursor(captured, detached_inbox), issued=detached_view.inbox
        ).items()
    }
    _require(
        detached_native_frames == before_native
        and detached_inbox_frames == before_inbox
        and _view_metadata(detached_view) == before_view,
        "complete issued/native component data changed during capture",
    )
    memo: dict[int, Any] = {}
    arrays = 0
    runtime = graph["runtime"]
    for name, learner in (("actor", runtime.actor._learner), ("candidate", runtime._candidate)):
        arrays += _array_detachment(vars(learner._model), detached_native[name].state, memo)
        _require(
            detached_native[name].config is detached_native[name].state["config"],
            "captured native config alias changed",
        )
    arrays += _array_detachment(graph["view"], detached_view, memo)
    for name in ("_experiences", "_labels", "_applied", "_erased"):
        arrays += _array_detachment(getattr(runtime._inbox, name), detached_inbox[name], memo)
    after_native, after_inbox, after_view = _components(graph)
    charge_after = graph["life"].payload_byte_snapshot()
    origins_after = graph["ledger"].accounting()
    budget_after = _plain(_budget_state(graph["budget"]))
    _require(
        (before_native, before_inbox, before_view) == (after_native, after_inbox, after_view)
        and identities == _identity(graph),
        "capture mutated native/view data or original authority",
    )
    _require(
        charge_before.charged_bytes == 224
        and charge_after.charged_bytes == 352
        and charge_before.observed_retained_bytes
        == charge_after.observed_retained_bytes
        == captured.state.retained_payload_bytes
        == 128,
        "issued capture did not retain its exact original raw accounting",
    )
    _require(
        origins_after.invocations_started == origins_before.invocations_started + 2
        and origins_after.metadata_bytes_charged == origins_before.metadata_bytes_charged + 2048
        and origins_after.records_created == origins_before.records_created,
        "complete capture renewed or duplicated original origin admission",
    )
    _require(
        _same_budget(_plain(budget_before), _plain(budget_after)),
        "capture changed original cumulative native budget",
    )
    authority = {item.path: item.value for item in captured.authority}
    for path, key in (
        ("root.owner", "owner"),
        ("root.lifecycle", "life"),
        ("runtime.root", "runtime"),
        ("runtime.actor", "actor"),
        ("runtime.budget", "budget"),
        ("inbox.clock", "clock"),
    ):
        _require(
            authority[path] is graph[key], "capture replaced an original live authority reference"
        )
    _require(
        graph["controller"]._pending.get(graph["token"]) is graph["pending"]
        and graph["pending"].view is graph["view"]
        and graph["manager"]._checkpoints[id(graph["token"])].view() is graph["view"]
        and not graph["controller"]._models
        and graph["controller"]._attempts == 0
        and runtime._budget.updates_completed == 1
        and not runtime._retired,
        "capture changed issued membership/preparation/work ownership",
    )
    payloads = {
        sha256(raw).hexdigest(): raw.decode("utf8")
        for raw in (*before_native.values(), *before_inbox.values())
    }
    digest = lambda values: {name: sha256(raw).hexdigest() for name, raw in values.items()}
    result = dict(
        family="circadian",
        policy=_policy(runtime._candidate, "circadian"),
        native_before=digest(before_native),
        native_after=digest(after_native),
        native_captured=digest(detached_native_frames),
        inbox_before=digest(before_inbox),
        inbox_after=digest(after_inbox),
        inbox_captured=digest(detached_inbox_frames),
        component_payloads=payloads,
        issued_view_fields=[field.name for field in fields(CandidateCheckpointView)],
        issued_view_before=before_view,
        issued_view_after=after_view,
        issued_view_captured=_view_metadata(detached_view),
        copy_before=asdict(charge_before),
        copy_after=asdict(charge_after),
        origins_before=asdict(origins_before),
        origins_after=asdict(origins_after),
        budget_before=budget_before,
        budget_after=budget_after,
        identity=dict(original=identities, current=_identity(graph)),
        default_off_refusal=refusal,
        capture_calls=2,
        graph_copy_passes=2,
        pending_checkpoints=1,
        preparation_attempts=0,
        native_updates=1,
        detached_array_count=arrays,
        joint_aliases_preserved=True,
        original_authority_preserved=True,
        native_unchanged=True,
        original_token_still_pending=True,
        data_only=True,
    )
    _require(len(canonical_json(result)) <= 131072, "issued snapshot report exceeds bounded output")
    return result


def describe_snapshot_example() -> dict[str, Any]:
    return dict(
        protocol_id="numpy_issued_snapshot_capture_engineering_v1",
        family="circadian",
        seed=23,
        geometry=dict(input_dim=3, hidden_dim=2),
        features=[[1.0, 0.0, 0.0]],
        targets=[[0.0]],
        learning_rate=0.01,
        inference_steps=2,
        inference_learning_rate=0.05,
        circadian_config=asdict(CircadianConfig(sleep_mode="disabled")),
        replay_retention=asdict(ReplayRetentionBudget(2, 64)),
        retention_policy=asdict(_retention()),
        origin_limits=asdict(_origin_limits()),
        write_limits=asdict(ReplayWriteLimits(16, 4, 64)),
        budget=asdict(ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1)),
        capture_limits=asdict(CAPTURE_LIMITS),
        native_limits=asdict(NATIVE_LIMITS),
        inbox_policy=asdict(_inbox_policy()),
        binding=asdict(BINDING),
        native_training_calls=1,
        capture_calls=2,
        graph_copy_passes=2,
        expected_setup_charge=224,
        expected_retained_bytes=128,
        expected_capture_charge=352,
        telemetry=dict(source="injected", wall_seconds=0.0, physical_rss=False, workers=0),
        data_only=True,
        limitations="Original issued snapshot capture DATA with same-process references; not native restoration/publication, later-current history, other-holder lineage, erasure, portable authority, bytes of the full managed graph, durable/live recovery or scientific benefit.",
    )


def run_snapshot_example() -> dict[str, Any]:
    return dict(
        protocol=describe_snapshot_example(),
        case=_capture_snapshot_case(_setup_snapshot_case()),
        issued_snapshot_capture_verified=True,
        full_managed_process_recovery_verified=False,
    )
