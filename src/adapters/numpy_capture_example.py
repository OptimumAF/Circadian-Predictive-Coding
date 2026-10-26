"""Genuine populated legacy capture with retired and pending owned holders.

The fixed zero-training composition uses existing native learners, builders and
controllers. It captures DATA under the original managed leases/copy allowance;
no lineage, replay exemption, storage, live restore or recovery authority is
manufactured. Replay is naturally empty because no training occurs. Nonempty
retained replay still needs qualified original lineage and remains unsupported.
"""

from copy import copy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from src.adapters.backprop_checkpoint_codec import BackpropCheckpointCodec
from src.adapters.circadian_checkpoint_codec import CircadianCheckpointCodec
from src.adapters.numpy_checkpoint_frames import canonical_json
from src.adapters.numpy_composite_capture import NATIVE_FIELDS, capture_numpy_managed_composite
from src.adapters.numpy_learners import (
    BackpropLearner,
    CircadianLearner,
    ManagedNumpyBuilder,
    make_managed_data_lifecycle,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import (
    CandidateCheckpoint,
    CandidateCheckpointController,
    _budget_state,
)
from src.app.managed_experience import ManagedExperienceOwner
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionProgress
from src.core.backprop_mlp import BackpropMLP
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.managed_composite_state import CompositeCaptureLimits
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import PreparedPromotion, ServingConfiguration
from src.shared.process_memory import ProcessRssSampler

FAMILIES = ("backprop", "circadian")
NATIVE_LIMITS = CodecLimits(131072, 65536, 8, 32)
CAPTURE_LIMITS = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32)
BINDING = CodecBinding(
    sha256(b"legacy populated retained capture source v1").hexdigest(),
    sha256(b"legacy populated retained capture policy v1").hexdigest(),
)


def _require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def _plain(value: Any) -> Any:
    def record(item: Any) -> Any:
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        raise TypeError("unsupported legacy example observation")

    return json.loads(json.dumps(value, default=record, allow_nan=False))


def _policy(learner: Any, family: str) -> dict[str, Any]:
    expected = BackpropLearner if family == "backprop" else CircadianLearner
    native = BackpropMLP if family == "backprop" else CircadianPredictiveCodingNetwork
    if (
        type(learner) is not expected
        or vars(learner).keys() != NATIVE_FIELDS[expected]
        or type(learner._model) is not native
    ):
        raise ValueError("legacy callback requires its exact configured learner/model")
    result = {
        name: getattr(learner, name) for name in sorted(NATIVE_FIELDS[expected]) if name != "_model"
    }
    result.update(
        family=family, input_dim=learner._model.input_dim, hidden_dims=learner._model.hidden_dims
    )
    if family == "circadian":
        model = learner._model
        assert isinstance(model, CircadianPredictiveCodingNetwork)
        result.update(
            config=asdict(model.config),
            min_hidden_dim=model._min_hidden_dim,
            max_hidden_dim=model.max_hidden_dim,
        )
    return result


def _callbacks(source: Any, family: str, codec: Any) -> dict[str, Any]:
    policy_bytes = canonical_json(_policy(source, family))

    def native_bytes(state: Any) -> bytes:
        return codec.encode(state, binding=BINDING, limits=NATIVE_LIMITS)

    def state_digest(state: Any) -> str:
        return sha256(native_bytes(state)).hexdigest()

    def policy_digest(learner: Any) -> str:
        actual = canonical_json(_policy(learner, family))
        if len(actual) > NATIVE_LIMITS.max_encoded_bytes or actual != policy_bytes:
            raise ValueError("legacy learner fixed policy changed")
        return sha256(actual).hexdigest()

    return dict(
        build=ManagedNumpyBuilder(source),
        native_bytes=native_bytes,
        state_digest=state_digest,
        policy_digest=policy_digest,
    )


def _feature_digest(features: np.ndarray) -> str:
    return sha256(
        canonical_json([features.dtype.str, list(features.shape)]) + features.tobytes()
    ).hexdigest()


def _retention_policy() -> DataRetentionPolicy:
    return DataRetentionPolicy(
        128,
        100,
        PayloadOwnershipLimits(8, 12),
        max_checkpoint_pending=2,
        max_checkpoint_preparations=2,
        max_promotion_pending=1,
        owned_payload_copies=PayloadCopyLimits(4096),
    )


def _evaluator(callbacks: dict[str, Any], clock: Any) -> PromotionGuardEvaluator:
    def valid(state: Any) -> bool:
        try:
            callbacks["native_bytes"](state)
            return True
        except (ValueError, TypeError, OverflowError, FloatingPointError):
            return False

    # Ordinary constructors keep __orig_class__ out of exact capture schemas.
    return PromotionGuardEvaluator(
        build_learner=callbacks["build"],
        utility=lambda prediction, targets: 1.0 - float(np.mean((prediction - targets) ** 2)),
        actions=lambda prediction: tuple(
            "1" if value >= 0.5 else "0" for value in prediction[:, 0]
        ),
        state_valid=valid,
        prediction_valid=lambda value: (
            type(value) is np.ndarray
            and value.dtype == np.float64
            and value.shape == (1, 1)
            and bool(np.isfinite(value).all())
            and bool(np.all((value >= 0) & (value <= 1)))
        ),
        resource_bytes=lambda state: len(callbacks["native_bytes"](state)),
        state_digest=callbacks["state_digest"],
        clock=clock,
    )


def _setup_case(family: str) -> dict[str, Any]:
    """Create one actual graph; tests can reuse it with native methods forbidden.

    Original handles are returned only to trusted local callers, not encoded in
    the demo report. No work is trained and no replay buffer is modified.
    """
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("legacy capture family must be backprop or circadian")
    wall_clock = lambda: 8.0
    rss_reader = lambda: 100
    sampler = ProcessRssSampler(read_rss_bytes=rss_reader)
    sampler.sample()  # Explicit injected telemetry, never host RSS or a worker.
    progress = ToyExecutionProgress()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_process_rss_bytes=1000), wall_clock, progress
    )
    budget.attach_memory(sampler)
    source: Any = (
        BackpropLearner(BackpropMLP(3, 2, 23), learning_rate=0.01)
        if family == "backprop"
        else CircadianLearner(
            CircadianPredictiveCodingNetwork(
                3, 2, 23, circadian_config=CircadianConfig(), min_hidden_dim=2
            ),
            learning_rate=0.01,
            inference_steps=2,
            inference_learning_rate=0.05,
        )
    )
    codec: Any = BackpropCheckpointCodec() if family == "backprop" else CircadianCheckpointCodec()
    callbacks = _callbacks(source, family, codec)
    builder = callbacks["build"]
    features = np.array([[0.2, -0.3, 0.4]], dtype=np.float64)
    targets = np.array([[1.0]], dtype=np.float64)
    actor = PromotableActor(
        source,
        version="actor",
        configuration=ServingConfiguration(20, 2),
        feature_digest=_feature_digest,
        metadata={"raw": np.array([0.25, 0.5, 0.75], dtype=np.float64)},
    )
    clock = LogicalClock()
    old = ActorShadowRuntime(
        source,
        actor_version="actor",
        candidate_version="candidate",
        clock=clock,
        budget=budget,
        max_experiences=4,
        max_consolidations=0,
        actor=actor,
    )
    gate = ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(old, gate)
    owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4))
    life = make_managed_data_lifecycle(owner, policy=_retention_policy())
    for number in range(2):
        owner.declare(
            LifecycleDeclaration(
                ("episode", f"s{number}"),
                DataProvenance("local-fixed", "person", True, False),
                DataConsent(True, True),
                "replay",
            )
        )
    owner.record_experience(
        Experience(
            "s0", "episode", 0, "actor", features, "train", ExperiencePermissions(True, True)
        )
    )
    owner.record_label(LabelArrival("label0", "s0", "episode", 0, "actor", targets))
    served = shared.serve(features.copy(), now=0)
    _require(not served.cache_hit, "legacy setup did not create its real actor cache")
    gate.pause()
    checkpoint = CandidateCheckpointController(
        shared,
        build_learner=builder,
        state_digest=callbacks["state_digest"],
        policy_digest=callbacks["policy_digest"],
        max_pending=2,
        max_prepare_attempts=2,
    )
    restored_token = checkpoint.capture()
    new = checkpoint.restore(restored_token)
    pending_checkpoint = checkpoint.capture()
    promotion = ServingPromotionController(
        actor,
        policy=PromotionPolicy(
            "one_minus_brier_v1",
            "native_bytes_v1",
            0.0,
            -1.0,
            1.0,
            1.0,
            NATIVE_LIMITS.max_encoded_bytes,
            ("0", "1"),
        ),
        evaluator=_evaluator(callbacks, wall_clock),
        build_learner=builder,
        state_digest=callbacks["state_digest"],
        max_pending=1,
    )
    new_guard = GuardBatch(
        "new",
        (("guard-new", "g0"),),
        0,
        np.array([[0.7, -0.4, 0.2]], dtype=np.float64),
        np.array([[1.0]], dtype=np.float64),
        "inner_guard",
    )
    old_guard = GuardBatch(
        "old",
        (("guard-old", "g0"),),
        0,
        np.array([[-0.6, 0.5, -0.1]], dtype=np.float64),
        np.array([[0.0]], dtype=np.float64),
        "inner_guard",
    )
    ticket = promotion.prepare(
        new,
        new_guard,
        old_guard,
        now=0,
        training_ids=frozenset(("episode", f"s{number}") for number in range(2)),
    )
    _require(
        old._retired and shared._runtime is new and ticket.report.decision.accepted,
        "legacy setup did not genuinely retire/prepare its original candidate",
    )
    snapshot = life.payload_byte_snapshot()
    _require(
        snapshot.charged_bytes == 184 and snapshot.observed_retained_bytes == 152,
        "legacy retained setup accounting changed",
    )
    return dict(
        family=family,
        source=source,
        owner=owner,
        life=life,
        old=old,
        new=new,
        shared=shared,
        actor=actor,
        checkpoint=checkpoint,
        pending_checkpoint=pending_checkpoint,
        pending_view=checkpoint._pending[pending_checkpoint].view,
        promotion=promotion,
        ticket=ticket,
        pending_bundle=promotion._pending[ticket].bundle,
        builder=builder,
        sampler=sampler,
        progress=progress,
        budget=budget,
        clock=clock,
        gate=gate,
        features=features,
        targets=targets,
        callbacks=callbacks,
        codec=codec,
        wall_clock=wall_clock,
        rss_reader=rss_reader,
    )


def _native_frames(graph: dict[str, Any]) -> dict[str, bytes]:
    checkpoint = graph["checkpoint"]
    learners = dict(
        source=graph["source"],
        actor=graph["actor"]._slot.bundle.learner,
        retired=graph["old"]._candidate,
        current=graph["new"]._candidate,
        restored_model=checkpoint._models[0],
        pending_promotion=graph["pending_bundle"].learner,
    )
    result = {
        name: graph["callbacks"]["native_bytes"](learner.snapshot_state())
        for name, learner in learners.items()
    }
    result["pending_checkpoint"] = graph["callbacks"]["native_bytes"](graph["pending_view"].state)
    return result


def _identity(graph: dict[str, Any]) -> dict[str, int]:
    owner, shared = graph["owner"], graph["owner"]._shared
    runtime, budget = shared._runtime, shared._runtime._budget
    return dict(
        owner=id(owner),
        life=id(owner._lifecycle),
        old=id(graph["old"]),
        new=id(runtime),
        shared=id(shared),
        actor=id(runtime._actor),
        checkpoint=id(graph["checkpoint"]),
        pending_checkpoint=id(graph["pending_checkpoint"]),
        promotion=id(graph["promotion"]),
        ticket=id(graph["ticket"]),
        builder=id(graph["checkpoint"]._build),
        sampler=id(budget.process_rss_sampler),
        progress=id(budget.progress),
        budget=id(budget),
        clock=id(runtime._inbox._clock),
        gate=id(shared._sharing),
        wall_clock=id(budget.clock),
        rss_reader=id(budget.process_rss_sampler.read_rss_bytes),
    )


def _same_budget(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Only genuine cumulative RSS observations may advance in zero-work capture."""
    for value in (before, after):
        value.pop("process_rss_segment")
        if value["progress_state"] is not None:
            value["progress_state"].pop("process_rss_segment")
    return before == after


def _refusals(graph: dict[str, Any]) -> list[dict[str, str]]:
    actions = (
        (
            "copied_checkpoint",
            lambda: graph["checkpoint"].restore(copy(graph["pending_checkpoint"])),
        ),
        ("foreign_checkpoint", lambda: graph["checkpoint"].restore(CandidateCheckpoint())),
        (
            "copied_promotion",
            lambda: graph["promotion"].commit(graph["new"], copy(graph["ticket"])),
        ),
        (
            "foreign_promotion",
            lambda: graph["promotion"].commit(
                graph["new"],
                PreparedPromotion(graph["ticket"].actor_generation, graph["ticket"].report),
            ),
        ),
    )
    result = []
    for phase, action in actions:
        try:
            action()
        except ValueError as error:
            result.append(dict(phase=phase, message=str(error)))
        else:
            raise AssertionError("copied/foreign token acquired legacy live authority")
    return result


def _native_aliases(snapshot: Any) -> bool:
    state = snapshot.state
    if "_hidden_weights" in state:
        return (
            state["weight_input_hidden"] is state["_hidden_weights"][0]
            and state["bias_hidden"] is state["_hidden_biases"][0]
        )
    return snapshot.config is state["config"]


def _summary(state: Any) -> dict[str, Any]:
    holders = []
    for enrollment, record in state.holders:
        data = dict(record.fields)
        row: dict[str, Any] = dict(
            enrollment=enrollment, kind=record.kind, fields=[name for name, _ in record.fields]
        )
        if record.kind == "src.app.actor_shadow.ActorShadowRuntime":
            inbox = dict(data["_inbox"].fields)
            native = dict(data["_candidate"].fields)["_model"]
            row.update(
                retired=data["_retired"],
                experiences=len(inbox["_experiences"]),
                labels=len(inbox["_labels"]),
                applied=len(inbox["_applied"]),
                historical_completed_updates=inbox["_historical_completed_updates"],
                native_aliases=_native_aliases(native),
            )
        holders.append(row)
    return dict(
        holders=holders,
        retained_payload_bytes=state.retained_payload_bytes,
        sharing=dict(state.sharing.fields)["_paused"],
        budget_started_at=dict(state.budget.fields)["started_at"],
    )


def _case(family: str) -> dict[str, Any]:
    graph = _setup_case(family)
    before = _native_frames(graph)
    identities = _identity(graph)
    budget_before = _plain(_budget_state(graph["budget"]))
    original_charge = graph["life"].payload_byte_snapshot()
    captured = capture_numpy_managed_composite(graph["owner"], limits=CAPTURE_LIMITS)
    after = _native_frames(graph)
    charge = graph["life"].payload_byte_snapshot()
    _require(
        before == after and identities == _identity(graph),
        "complete legacy capture changed native state or original authority",
    )
    _require(
        charge.charged_bytes
        == original_charge.charged_bytes + captured.state.retained_payload_bytes
        and charge.observed_retained_bytes == original_charge.observed_retained_bytes == 152
        and captured.state.records.lifecycle.copy.charged_bytes == charge.charged_bytes,
        "legacy capture did not retain its exact original RAW charge",
    )
    refs = {item.path: item.value for item in captured.authority}
    for path, key in (
        ("root.owner", "owner"),
        ("root.lifecycle", "life"),
        ("runtime.root", "new"),
        ("runtime.actor", "actor"),
        ("runtime.budget", "budget"),
        ("inbox.clock", "clock"),
        ("lifecycle._sampler", "sampler"),
    ):
        _require(refs[path] is graph[key], "legacy capture replaced original live authority")
    runtime_records = [
        dict(record.fields)
        for _, record in captured.state.holders
        if record.kind == "src.app.actor_shadow.ActorShadowRuntime"
    ]
    _require(
        len(runtime_records) == 2
        and sorted(value["_retired"] for value in runtime_records) == [False, True],
        "legacy capture omitted retired/current candidate",
    )
    for data in runtime_records:
        inbox = dict(data["_inbox"].fields)
        _require(
            len(inbox["_experiences"]) == len(inbox["_labels"]) == 1 and not inbox["_applied"],
            "legacy capture changed its populated untrained inbox",
        )
        original = graph["old"] if data["_retired"] else graph["new"]
        detached = inbox["_experiences"][("episode", "s0")].features
        _require(
            not np.shares_memory(
                detached, original._inbox._experiences[("episode", "s0")].features
            ),
            "legacy captured raw payload remained borrowed",
        )
    refusals = _refusals(graph)
    _require(
        identities == _identity(graph)
        and len(graph["checkpoint"]._pending) == len(graph["promotion"]._pending) == 1
        and graph["pending_checkpoint"] in graph["checkpoint"]._pending
        and graph["ticket"] in graph["promotion"]._pending
        and graph["budget"].updates_completed == 0
        and graph["sampler"]._thread is None,
        "legacy refusal changed pending ownership or performed work",
    )
    budget_after = _plain(_budget_state(graph["budget"]))
    _require(
        _same_budget(_plain(budget_before), _plain(budget_after)),
        "legacy capture renewed original budget/progress accounting",
    )
    _require(
        graph["sampler"].read_rss_bytes is graph["rss_reader"]
        and graph["budget"].started_at == 8.0
        and graph["sampler"].start_bytes == graph["sampler"].peak_bytes == 100,
        "legacy capture changed original injected telemetry",
    )
    native_payloads = {sha256(raw).hexdigest(): raw.decode("utf8") for raw in before.values()}
    row = dict(
        family=family,
        policy=_policy(graph["source"], family),
        native_before={name: sha256(raw).hexdigest() for name, raw in before.items()},
        native_after={name: sha256(raw).hexdigest() for name, raw in after.items()},
        native_payloads=native_payloads,
        detached_graph=_summary(captured.state),
        copy_before=asdict(original_charge),
        copy_after=asdict(charge),
        identity=dict(original=identities, current=_identity(graph)),
        budget_before=budget_before,
        budget_after=budget_after,
        sampler=asdict(graph["sampler"].snapshot()),
        pending_checkpoints=1,
        pending_promotions=1,
        preparation_attempts=graph["checkpoint"]._attempts,
        guard_report=asdict(graph["ticket"].report),
        refusals=refusals,
        native_unchanged=True,
        original_authority_preserved=True,
        no_training=True,
        replay_scope="Naturally empty; nonempty retained replay lineage remains unsupported",
    )
    _require(len(canonical_json(row)) <= 131072, "legacy capture report exceeds bounded row output")
    return row


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="numpy_populated_retired_capture_engineering_v1",
        families=list(FAMILIES),
        seed=23,
        geometry=dict(input_dim=3, hidden_dim=2),
        learning_rate=0.01,
        inference_steps=2,
        inference_learning_rate=0.05,
        declarations=2,
        features=[[0.2, -0.3, 0.4]],
        targets=[[1.0]],
        native_training_calls=0,
        energy_evaluations=0,
        inference_calls=0,
        capture_calls=2,
        retention_policy=asdict(_retention_policy()),
        capture_limits=asdict(CAPTURE_LIMITS),
        native_limits=asdict(NATIVE_LIMITS),
        binding=asdict(BINDING),
        expected_setup_charge=184,
        expected_retained_bytes=152,
        expected_capture_charge=336,
        telemetry=dict(
            source="injected", rss_bytes=100, rss_budget=1000, wall_seconds=8.0, workers=0
        ),
        limitations="Complete same-process capture DATA plus original live references, not portable authority, bytes, restoration or recovery. Replay is empty because no training occurred; nonempty retained fork/snapshot replay lineage and broader parent work remain open. Native objectives differ; no scientific comparison.",
    )


def run_example() -> dict[str, Any]:
    return dict(
        protocol=describe_example(),
        cases=[_case(family) for family in FAMILIES],
        populated_retired_capture_verified=True,
        full_managed_process_recovery_verified=False,
    )
