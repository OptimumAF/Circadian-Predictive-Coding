"""Fixed managed cache, handoff and pending/committed promotion expiry controls.

The existing actor/controllers/lifecycle/driver own all transactions and cleanup.
This bounded local example observes original raw-copy charges, cache age and
complete native continuation. Injected time demonstrates admission, not physical
latency. Guard arrays, reports and caller copies are outside the owned ledger.
No new scheduler, learning rule, unlearning, attestation or process recovery.
"""

from dataclasses import asdict
import json
import os
from typing import Any, Callable, cast

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from src.adapters.coupled_candidate_handoff import (
    BINDING,
    LIMITS,
    FAMILIES,
    MODES,
    _model,
    _payload,
    _plain,
    _canonical,
    _require,
    _refused,
)
from src.adapters.coupled_cache_transactions import _guards
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_cleanup_failure import (
    OneShotCoupledEraseFailure,
    make_failing_coupled_elapsed_lifecycle,
    _poll_cleanup_failure,
    _candidate_fences,
)
from src.adapters.coupled_elapsed_retention import make_coupled_elapsed_lifecycle
from src.adapters.coupled_growth_limit import _feature_digest
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_managed_cleanup import _record
from src.adapters.coupled_managed_consent import _declaration, _managed, _PoisonPayload
from src.adapters.coupled_payload_copies import (
    ManagedCoupledBuilder,
    measure_coupled_auxiliary_bytes,
)
from src.adapters.coupled_retention_example import (
    _LockedElapsedClock,
    _driver_limits,
    _stop,
    _inbox_metadata,
    describe_elapsed_retention_example,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, _budget_state
from src.app.managed_experience import ManagedExperienceOwner
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.retention_expiry import RetentionExpiryDriver
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledSnapshot
from src.core.coupled_learner_state import CoupledUpdatePolicy
from src.core.checkpoint_codec import require_codec_digest
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.promotion_guard import PromotionPolicy
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import PromotionReceipt, ServingConfiguration, ServingSnapshot

Runtime = ActorShadowRuntime[Array, Array, Array, CoupledSnapshot]
Actor = PromotableActor[Array, Array, Array, CoupledSnapshot]


def _retention_policy(
    *, committed: bool = False, complete_checkpoints: bool = False
) -> DataRetentionPolicy:
    if type(complete_checkpoints) is not bool or (complete_checkpoints and committed is not True):
        raise ValueError("complete checkpoints require an exact boolean and committed serving mode")
    return DataRetentionPolicy(
        192,
        100,
        PayloadOwnershipLimits(5, 5),
        max_checkpoint_pending=2,
        max_checkpoint_preparations=2,
        max_promotion_pending=1,
        owned_payload_copies=PayloadCopyLimits(
            1496 if complete_checkpoints else 656 if committed else 640
        ),
        max_retention_seconds=2.0,
    )


def _promotion_policy() -> PromotionPolicy:
    # Broad fixed thresholds exercise the transaction; they select no benefit.
    return PromotionPolicy(
        "one_minus_brier_v1",
        "coupled_learner_envelope_bytes_v1",
        0.0,
        -1.0,
        1.0,
        60.0,
        65536,
        ("0", "1"),
    )


def describe_managed_serving_example() -> dict[str, Any]:
    description = describe_elapsed_retention_example()
    description.update(
        protocol_id="coupled_managed_serving_expiry_engineering_v1",
        retention_policy=asdict(_retention_policy()),
        serving_configuration=asdict(ServingConfiguration(20, 2)),
        serving_metadata={"raw": [0.25, 0.5, 0.75], "source": "synthetic-local"},
        promotion_policy=asdict(_promotion_policy()),
        guard_batches=dict(
            new=dict(
                task_id="new",
                sample_keys=[["guard-new", "g0"]],
                features=[[0.7, -0.4]],
                targets=[[1.0]],
                labels_arrived_at=7,
                role="inner_guard",
            ),
            old=dict(
                task_id="old",
                sample_keys=[["guard-old", "g0"]],
                features=[[-0.6, 0.5]],
                targets=[[0.0]],
                labels_arrived_at=7,
                role="inner_guard",
            ),
        ),
        native_constructor_cap=84,
        native_constructors=72,
        prediction_calls=54,
        prediction_cap=96,
        serving_requests=42,
        guard_evaluations=6,
        measured_guard_models=12,
        build_calls_per_case=4,
        copy_charges=[24, 168, 184, 328, 472, 616, 640, 640],
        retained_raw_bytes=[24, 168, 184, 328, 328, 472, 496, 0],
        limitations="Injected guard latency is exactly0, not physical timing. Conservative raw-array accounting excludes learned parameters, Python/RSS, guards/reports/caller copies. Expiry drops owned references/tickets and preserves learned state; no RAM overwrite, unlearning, provenance, recovery or scientific claim.",
    )
    return description


def describe_committed_serving_example() -> dict[str, Any]:
    description = describe_managed_serving_example()
    description.update(
        protocol_id="coupled_committed_serving_expiry_engineering_v1",
        retention_policy=asdict(_retention_policy(committed=True)),
        prediction_calls=60,
        serving_requests=54,
        copy_charges=[24, 168, 184, 328, 472, 616, 640, 640, 656, 656, 656],
        retained_raw_bytes=[24, 168, 184, 328, 328, 472, 496, 496, 512, 512, 0],
        promotion_committed=True,
        previous_cache_expires_at=20,
        current_cache_expires_at=28,
    )
    return description


def describe_busy_committed_serving_example() -> dict[str, Any]:
    description = describe_committed_serving_example()
    description.update(
        protocol_id="coupled_busy_committed_serving_cleanup_engineering_v1",
        serving_requests=66,
        copy_charges=[24, 168, 184, 328, 472, 616, 640, 640, 656, 656, 656, 656],
        retained_raw_bytes=[24, 168, 184, 328, 328, 472, 496, 496, 512, 512, 512, 0],
        automatic_purge=False,
        terminal_cleanup="original joined driver finish_cleanup after holder release",
    )
    return description


def describe_cleanup_failure_example() -> dict[str, Any]:
    description = describe_committed_serving_example()
    description.update(
        protocol_id="coupled_cleanup_failure_engineering_v1",
        serving_requests=66,
        retention_workers=0,
        automatic_purge=False,
        copy_charges=[24, 168, 184, 328, 472, 616, 640, 640, 656, 656, 656, 656],
        retained_raw_bytes=[24, 168, 184, 328, 328, 472, 496, 496, 512, 512, 368, 0],
        expected_erasure_failures=6,
        terminal_cleanup="original failed driver finish_cleanup; error and candidate stop fences persist",
    )
    return description


def _busy_stop(driver: RetentionExpiryDriver, runtime: Runtime) -> dict[str, Any]:
    """Join the original worker while its candidate holder prevents erasure.

    Why hold the real lease: this demonstrates genuine unfinished cleanup rather
    than simulating a driver result. Release it before sizing or public controls.
    Scheduling may consume multiple busy attempts within the original allowance.
    """
    with runtime._exclusive():
        poll = driver.poll_once()
        _require(poll.outcome == "busy", "held candidate did not refuse ready cleanup")
        driver.start()
        driver.wake()
        stopped = driver.stop()
        observed = driver.snapshot()
        _require(
            stopped is False
            and observed.state == "stopped"
            and not observed.alive
            and observed.pending_cleanup
            and observed.purges == 0
            and observed.error_type is None
            and 1 <= observed.polls <= observed.limits.max_polls
            and observed.cleanup_attempts < observed.limits.max_polls + 2,
            "busy stop did not join with unfinished cleanup and a reserved finish attempt",
        )
        return dict(
            busy_poll=_plain(asdict(poll)),
            driver_busy_stopped=_plain(asdict(observed)),
            stop_returned=stopped,
        )


def _forensic_serving(actor: Actor) -> dict[str, Any]:
    """Trusted post-stop observation; no public raw/native admission is reopened."""
    slot, bundle = actor._slot, actor._slot.bundle
    return _bundle_snapshot(bundle, slot.generation)


def _bundle_snapshot(bundle: Any, generation: int) -> dict[str, Any]:
    """Observe a trusted owned current/previous bundle without publishing it."""
    return _plain(
        asdict(
            ServingSnapshot(
                generation,
                bundle.version,
                bundle.learner.snapshot_state(),
                bundle.configuration,
                bundle.cache,
                bundle.metadata,
                bundle.last_cache_tick,
            )
        )
    )


def _evaluator(
    builder: ManagedCoupledBuilder,
    callbacks: CoupledCheckpointCallbacks,
    policy: CoupledUpdatePolicy,
    elapsed: _LockedElapsedClock,
    counts: dict[str, int],
) -> PromotionGuardEvaluator[Array, Array, Array, CoupledSnapshot]:
    def state_valid(state: CoupledSnapshot) -> bool:
        counts["state_valid"] += 1
        try:
            callbacks.state_digest(state)
        except (ValueError, TypeError, OverflowError, FloatingPointError):
            return False
        return True

    def prediction_valid(prediction: Array) -> bool:
        counts["prediction_valid"] += 1
        return bool(
            type(prediction) is np.ndarray
            and prediction.dtype == np.float64
            and prediction.ndim == 2
            and prediction.shape[1] == 1
            and np.isfinite(prediction).all()
            and np.all((prediction >= 0) & (prediction <= 1))
        )

    def utility(prediction: Array, expected: Array) -> float:
        counts["utility"] += 1
        return 1.0 - float(np.mean((prediction - expected) ** 2))

    def actions(prediction: Array) -> tuple[str, ...]:
        counts["actions"] += 1
        return tuple("1" if value >= 0.5 else "0" for value in prediction[:, 0])

    def resource(state: CoupledSnapshot) -> int:
        counts["resource"] += 1
        return len(_payload(policy, state))

    # Runtime generic subscription adds __orig_class__ to instances. The return
    # annotation supplies typing without changing the exact retained source schema.
    return PromotionGuardEvaluator(
        build_learner=builder,
        utility=utility,
        actions=actions,
        state_valid=state_valid,
        prediction_valid=prediction_valid,
        resource_bytes=resource,
        state_digest=callbacks.state_digest,
        clock=elapsed,
    )


def _case(
    family: str,
    mode: str,
    *,
    committed: bool = False,
    busy_stop: bool = False,
    cleanup_failure: bool = False,
    checkpoint_observer: Callable[[ManagedExperienceOwner, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Keep this single ordered lifecycle visible; shared helpers own setup,
    # measurement, cache observations and refusal evidence rather than a new engine.
    if type(committed) is not bool:
        raise ValueError("committed serving mode must be an exact boolean")
    if type(busy_stop) is not bool or (busy_stop and not committed):
        raise ValueError("busy stop requires an exact boolean and committed serving mode")
    if type(cleanup_failure) is not bool or (cleanup_failure and (not committed or busy_stop)):
        raise ValueError(
            "cleanup failure requires an exact boolean, committed mode and no busy stop"
        )
    if checkpoint_observer is not None and (
        not callable(checkpoint_observer) or not committed or busy_stop or cleanup_failure
    ):
        raise ValueError(
            "checkpoint observer requires a callable, committed mode and no busy/failure mode"
        )
    features = np.array([[0.2, -0.3], [-0.4, 0.6]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    inference = CoupledInferencePolicy(1, 1e6, None) if mode == "guarded" else None
    reference = CoupledLearner(
        _model(family),
        learning_rate=0.025,
        inference_steps=3,
        inference_learning_rate=0.07,
        inference_policy=inference,
    )
    for _ in range(2):
        reference.train_batch(features.copy(), targets.copy())
    policy = reference.checkpoint_policy()
    callbacks = CoupledCheckpointCallbacks(policy, binding=BINDING, limits=LIMITS)
    counts = dict(
        feature_digest=0, state_valid=0, prediction_valid=0, utility=0, actions=0, resource=0
    )

    def feature_digest(value: Array) -> str:
        counts["feature_digest"] += 1
        return _feature_digest(value)

    metadata = {"raw": np.array([0.25, 0.5, 0.75], dtype=np.float64), "source": "synthetic-local"}
    # Generic aliases belong in annotations: calling them adds __orig_class__
    # and changes the exact retained source schema used by complete capture.
    actor: Actor = PromotableActor(
        reference,
        version="a0",
        configuration=ServingConfiguration(20, 2),
        feature_digest=feature_digest,
        metadata=metadata,
    )
    elapsed, clock = _LockedElapsedClock(), LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), elapsed)
    old: Runtime = ActorShadowRuntime(
        reference,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        actor=actor,
        max_experiences=3,
        max_consolidations=0,
    )
    gate = ServingPriorityGate(SharingLimits(2, 3, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(old, gate)
    owner = ManagedExperienceOwner(
        shared, limits=LifecycleLimits(4, 4, allow_synthetic=True, allow_unverified=True)
    )
    erase_failure = OneShotCoupledEraseFailure() if cleanup_failure else None
    lifecycle = (
        make_failing_coupled_elapsed_lifecycle(
            owner, policy=_retention_policy(committed=True), erase_failure=erase_failure
        )
        if erase_failure is not None
        else make_coupled_elapsed_lifecycle(
            owner,
            policy=_retention_policy(
                committed=committed, complete_checkpoints=checkpoint_observer is not None
            ),
        )
    )
    driver = RetentionExpiryDriver(lifecycle, limits=_driver_limits())
    primary: BaseException | None = None
    try:
        driver_ready = _plain(asdict(driver.snapshot()))
        created_at, original_limits = driver._created_at, driver._limits
        registry, copy_budget = actor._payload_registry, lifecycle._copy_budget

        def identity() -> dict[str, int]:
            current = shared._runtime
            return dict(
                owner=id(owner),
                lifecycle=id(owner._lifecycle),
                driver=id(lifecycle._retention_driver),
                driver_limits=id(driver._limits),
                elapsed_clock=id(budget.clock),
                lifecycle_clock=id(lifecycle._wall_clock),
                clock=id(current._inbox._clock),
                budget=id(current._budget),
                gate=id(shared._sharing),
                actor=id(current.actor),
                registry=id(current.actor._payload_registry),
                copy_budget=id(lifecycle._copy_budget),
                catalog=id(owner._catalog),
                opted_out=id(owner._opted_out),
                revoked_keys=id(owner._revoked_keys),
                limits=id(owner._limits),
                registration_guard=id(current._inbox._registration_guard),
                eligibility_guard=id(current._inbox._training_guard),
            )

        original_identity = identity()
        accounting: list[dict[str, Any]] = []
        checkpoint_receipts: list[dict[str, Any]] = []
        checkpoint_offset = 0

        def account(phase: str, charged: int, retained: int) -> None:
            value = lifecycle.payload_byte_snapshot()
            _require(
                value.charged_bytes == charged + checkpoint_offset
                and value.observed_retained_bytes == retained,
                f"managed serving copy accounting differs at {phase}",
            )
            accounting.append(dict(phase=phase, **_plain(asdict(value))))

        def observe_checkpoint(phase: str, base_charge: int) -> None:
            nonlocal checkpoint_offset
            if checkpoint_observer is None:
                return
            if copy_budget is None:
                raise AssertionError("complete checkpoint mode lost its original copy budget")
            before = lifecycle.payload_byte_snapshot()
            before_identity, original_copy_limits = identity(), copy_budget._limits
            original_cap = original_copy_limits.max_lifetime_owned_bytes
            receipt = checkpoint_observer(owner, phase)
            after = lifecycle.payload_byte_snapshot()
            _require(
                after.charged_bytes - before.charged_bytes == before.observed_retained_bytes
                and after.observed_retained_bytes == before.observed_retained_bytes
                and identity() == before_identity
                and copy_budget._limits is original_copy_limits
                and original_copy_limits.max_lifetime_owned_bytes == original_cap == 1496,
                "complete checkpoint observer changed original ownership or exact capture charge",
            )
            if (
                type(receipt) is not dict
                or receipt.keys() != {"path", "digest", "data_only", "retained_payload_bytes"}
                or type(receipt["path"]) is not str
                or not receipt["path"]
                or receipt["data_only"] is not True
                or type(receipt["retained_payload_bytes"]) is not int
                or receipt["retained_payload_bytes"] != before.observed_retained_bytes
            ):
                raise ValueError(
                    "checkpoint observer must return only a complete small data receipt"
                )
            require_codec_digest(receipt["digest"])
            _require(
                len(_canonical(receipt).encode()) <= 65536,
                "complete checkpoint receipt exceeds output bound",
            )
            checkpoint_offset += before.observed_retained_bytes
            checkpoint_receipts.append(dict(phase=phase, **receipt))
            account("complete_checkpoint_" + phase, base_charge, before.observed_retained_bytes)

        account("initial", 24, 24)
        for number, subject in enumerate(("subject-A", "subject-B", "subject-A")):
            owner.declare(_declaration(number, subject))
        _record(owner, features, targets)
        account("ingress", 168, 168)
        actor_a = _payload(policy, actor.snapshot_state().state)
        cache_observations: list[dict[str, Any]] = []

        def serve(phase: str, now: int, hit: bool, *, promoted: bool = False):
            result = shared.serve(features.copy(), now=now)
            _require(
                result.cache_hit is hit
                and result.actor_version == ("c0" if promoted else "a0")
                and result.generation == (1 if promoted else 0),
                f"managed serving cache phase {phase} changed actor/hit",
            )
            cache_observations.append(dict(phase=phase, result=_plain(asdict(result))))
            return result.prediction.tolist()

        prediction_a = serve("prime", 0, False)
        account("cache", 184, 184)

        def update(number: int):
            poll = owner.train_ready()
            direct = reference.train_batch(features.copy(), targets.copy())
            current = shared._runtime
            payload = _payload(policy, current.candidate_snapshot().state)
            prediction = current._candidate.predict(features.copy()).tolist()
            _require(
                [v.sample_id for v in poll.updates] == [f"s{number}"]
                and poll.updates[0].diagnostic == direct
                and payload == _payload(policy, reference.snapshot_state())
                and prediction == reference.predict(features.copy()).tolist()
                and callbacks.policy_digest(current._candidate)
                == callbacks.policy_digest(reference),
                "managed serving native continuation differs from independent reference",
            )
            return poll, payload, prediction

        elapsed.advance_to(0.5)
        clock.advance_to(3)
        first, payload_b, _ = update(0)
        gate.pause()
        builder = ManagedCoupledBuilder(callbacks)
        checkpoint = CandidateCheckpointController(
            shared,
            build_learner=builder,
            state_digest=callbacks.state_digest,
            policy_digest=callbacks.policy_digest,
            max_pending=2,
            max_prepare_attempts=2,
        )
        token = checkpoint.capture()
        account("first_capture", 328, 328)
        new = checkpoint.restore(token)
        account("restore", 472, 328)
        _require(
            old._retired
            and shared._runtime is new
            and serve("handoff_hit", 3, True) == prediction_a,
            "managed serving handoff changed original cached actorA",
        )
        observe_checkpoint("B", 472)
        gate.resume()
        clock.advance_to(5)
        waiting = owner.train_ready()
        _require(
            waiting.updates == () and waiting.deferred_reason is None,
            "future serving label trained at5",
        )
        elapsed.advance_to(1.0)
        clock.advance_to(7)
        second, payload_c, prediction_c = update(1)
        clock.advance_to(8)
        gate.pause()
        pending_checkpoint = checkpoint.capture()
        account("second_capture", 616, 472)
        evaluator = _evaluator(builder, callbacks, policy, elapsed, counts)
        promotion = ServingPromotionController(
            actor,
            policy=_promotion_policy(),
            evaluator=evaluator,
            build_learner=builder,
            state_digest=callbacks.state_digest,
            max_pending=1,
        )
        new_guard, old_guard = _guards()
        training_ids = frozenset((f"e{number}", f"s{number}") for number in range(3))

        def prepare(context):
            return promotion.prepare(
                new, new_guard, old_guard, now=8, training_ids=training_ids, metadata=context
            )

        ticket = prepare(metadata)
        pending_bundle = promotion._pending[ticket].bundle
        account("promotion", 640, 496)
        _require(
            ticket.report.decision.accepted
            and ticket.report.evidence.candidate_max_prediction_seconds == 0.0
            and builder.build_calls == 4
            and serve("prepared_hit", 8, True) == prediction_a,
            "managed serving guard/preparation changed actor/cache or fixed controls",
        )
        _require(
            lifecycle._auxiliary_started_at == 0.0
            and lifecycle._last_seconds == 1.0
            and next(iter(actor._slot.bundle.cache.values())).expires_at == 20,
            "promotion renewed original cache/metadata retention age",
        )
        receipt: PromotionReceipt | None = None
        previous_bundle = None
        commit_evidence: dict[str, Any] | None = None
        if committed:
            # Commit reuses the prepared model/metadata; only the subsequent
            # retained prediction needs16 additional bytes under the opt-in cap.
            previous_bundle = actor._slot.bundle
            receipt = promotion.commit(new, ticket)
            _require(
                receipt.generation == actor._slot.generation == 1
                and receipt.previous_version == "a0"
                and receipt.actor_version == "c0"
                and actor._slot.previous is previous_bundle
                and promotion._latest is not None
                and promotion._latest[0] is receipt
                and not promotion._pending,
                "committed promotion lost its original receipt or previous serving bundle",
            )
            account("commit", 640, 496)
            _require(
                serve("committed_miss", 8, False, promoted=True) == prediction_c,
                "committed cache prediction differs from matched candidateC",
            )
            account("committed_cache", 656, 512)
            _require(
                serve("committed_hit", 8, True, promoted=True) == prediction_c,
                "committed cache hit changed candidateC prediction",
            )
            account("committed_hit", 656, 512)
            _require(
                next(iter(previous_bundle.cache.values())).expires_at == 20
                and next(iter(actor._slot.bundle.cache.values())).expires_at == 28
                and next(iter(previous_bundle.cache.values())).prediction.tolist() == prediction_a
                and previous_bundle.metadata is not actor._slot.bundle.metadata
                and previous_bundle.cache is not actor._slot.bundle.cache
                and not np.shares_memory(
                    previous_bundle.metadata["raw"], actor._slot.bundle.metadata["raw"]
                )
                and not np.shares_memory(
                    next(iter(previous_bundle.cache.values())).prediction,
                    next(iter(actor._slot.bundle.cache.values())).prediction,
                )
                and lifecycle._auxiliary_started_at == 0.0,
                "committed promotion changed previous/current cache age or retention anchor",
            )
            commit_evidence = dict(
                generation_before=0,
                generation_after=1,
                pending_before=1,
                pending_after=0,
                original_receipt_live=True,
                previous_bundle_retained=True,
                independent_metadata=True,
                independent_cache=True,
            )
            observe_checkpoint("C", 656)

        def rollback() -> int:
            if receipt is None:
                raise AssertionError("committed fixture lost its original rollback receipt")
            return promotion.rollback(receipt)

        def capture(*, terminal: bool = False) -> dict[str, Any]:
            observed = dict(
                old_cursor=_plain(asdict(old._inbox._borrow_cursor())),
                new_cursor=_plain(asdict(new._inbox._borrow_cursor())),
                old_revision=old._revision,
                new_revision=new._revision,
                serving=_forensic_serving(actor)
                if terminal
                else _plain(asdict(actor.serving_snapshot())),
                pending_promotion_metadata=_plain(pending_bundle.metadata),
                pending_promotion_cache=_plain(pending_bundle.cache),
                budget=_plain(_budget_state(budget)),
                sharing=_plain(asdict(gate.snapshot())),
                registry=_plain(asdict(registry.snapshot())),
                managed=_managed(owner),
                owned_copies=_plain(asdict(lifecycle.payload_byte_snapshot())),
                admitted_payload_bytes=lifecycle.admitted_payload_bytes,
                auxiliary_started_at=lifecycle._auxiliary_started_at,
                preparation_attempts=checkpoint._attempts,
                prepared_models=len(checkpoint._models),
                pending_checkpoints=len(checkpoint._pending),
                pending_promotions=len(promotion._pending),
                build_calls=builder.build_calls,
            )
            if committed:
                if previous_bundle is None:
                    raise AssertionError("committed fixture lost its retained previous bundle")
                observed["prepared_bundle_metadata"] = observed.pop("pending_promotion_metadata")
                observed["prepared_bundle_cache"] = observed.pop("pending_promotion_cache")
                observed["previous_serving"] = (
                    None
                    if actor._slot.previous is None
                    else _bundle_snapshot(actor._slot.previous, 0)
                )
                observed["previous_owned_metadata"] = _plain(previous_bundle.metadata)
                observed["previous_owned_cache"] = _plain(previous_bundle.cache)
                observed["rollback_live"] = (
                    promotion._latest is not None and promotion._latest[0] is receipt
                )
            return observed

        before_expiry = capture()
        _require(
            len(before_expiry["registry"]["holders"])
            == before_expiry["registry"]["total_enrollments"]
            == 5,
            "managed serving must retain the original five holders",
        )
        poison = _PoisonPayload()

        def refusal_state() -> dict[str, Any]:
            # Compare metadata/ownership without copying expired raw arrays or
            # invoking native snapshots/hash callbacks just to prove a refusal.
            slot, bundle = actor._slot, actor._slot.bundle
            return dict(
                old=_inbox_metadata(old),
                new=_inbox_metadata(new),
                slot=id(slot),
                bundle=id(bundle),
                generation=slot.generation,
                metadata=id(bundle.metadata),
                cache=id(bundle.cache),
                cached=[
                    (key, id(value), value.expires_at, id(value.prediction))
                    for key, value in bundle.cache.items()
                ],
                pending_metadata=id(pending_bundle.metadata),
                pending_cache=id(pending_bundle.cache),
                previous=id(slot.previous),
                latest=id(promotion._latest),
                attempts=checkpoint._attempts,
                models=len(checkpoint._models),
                checkpoints=tuple(id(v) for v in checkpoint._pending),
                promotions=tuple(id(v) for v in promotion._pending),
                builds=builder.build_calls,
                counts=dict(counts),
                budget=_plain(_budget_state(budget)),
                sharing=_plain(asdict(gate.snapshot())),
                charged=copy_budget._charged if copy_budget is not None else None,
            )

        def refusals(
            *, terminal: bool, cleanup_pending: bool = False, cleanup_failed: bool = False
        ) -> list[dict[str, Any]]:
            closed = (
                "payload cleanup failed; retry cleanup before access"
                if cleanup_failed
                else (
                    "retention driver terminal; raw admission/access is closed"
                    if terminal
                    else "declared payload retention has expired; cleanup required"
                )
            )
            actions: tuple[tuple[str, Callable[[], Any], str], ...] = (
                ("cached_hit", lambda: shared.serve(features, now=8), closed),
                ("poison_cache", lambda: shared.serve(cast(Array, poison), now=8), closed),
                ("serving_snapshot", actor.serving_snapshot, closed),
                (
                    "promotion_commit",
                    lambda: promotion.commit(new, ticket),
                    "promotion ticket is foreign, forged, discarded or consumed"
                    if terminal or committed
                    else closed,
                ),
                (
                    "checkpoint_restore",
                    lambda: checkpoint.restore(pending_checkpoint),
                    "checkpoint is foreign, copied, discarded or already used"
                    if terminal and not cleanup_pending
                    else closed,
                ),
                # The live full ticket capacity legitimately refuses before its
                # deadline check; after purge, terminal lifecycle admission does.
                (
                    "fresh_prepare",
                    lambda: prepare({"raw": poison}),
                    "candidate is stopped after uncertain native mutation"
                    if cleanup_failure and terminal
                    else closed
                    if terminal or committed
                    else "pending promotion ticket quota exhausted; discard obsolete tickets",
                ),
            )
            if committed:
                actions += (
                    (
                        "rollback",
                        rollback,
                        "rollback receipt is foreign, stale or consumed"
                        if terminal and not cleanup_pending
                        else closed,
                    ),
                )
            records = []
            for phase, action, message in actions:
                before = refusal_state()
                result = _refused(action, message)
                _require(
                    _canonical(before) == _canonical(refusal_state()) and poison.copy_attempts == 0,
                    "managed serving refusal changed owned slots/counters or copied poisoned data",
                )
                records.append(
                    dict(
                        phase=phase,
                        message=result,
                        poison_copy_attempts=poison.copy_attempts,
                        build_calls=builder.build_calls,
                        feature_digest_calls=counts["feature_digest"],
                        complete_state_preserved=True,
                        quota_precedes_deadline=(
                            phase == "fresh_prepare" and not terminal and not committed
                        ),
                    )
                )
                if committed:
                    records[-1]["consumed_ticket_precedes_deadline"] = (
                        phase == "promotion_commit" and not terminal
                    )
                if cleanup_pending:
                    records[-1]["cleanup_pending"] = True
                if cleanup_failed:
                    records[-1]["after_erasure_failure"] = True
            return records

        elapsed.advance_to(2.0)
        deadline_refusals = refusals(terminal=False)
        _require(
            driver.snapshot().state == "ready" and driver.snapshot().polls == 0,
            "managed serving worker started before deadline refusals",
        )
        busy_evidence: dict[str, Any] | None = None
        pending_cleanup: dict[str, Any] | None = None
        pending_refusals: list[dict[str, Any]] = []
        finished_cleanup: dict[str, Any] | None = None

        def retained_observation() -> dict[str, Any]:
            if previous_bundle is None:
                raise AssertionError("committed fixture lost its previous serving bundle")
            # No expired raw publication merely to prove failed/busy behavior.
            return dict(
                driver=_plain(asdict(driver.snapshot())),
                owned_copies=_plain(asdict(lifecycle.payload_byte_snapshot())),
                old=_inbox_metadata(old),
                new=_inbox_metadata(new),
                generation=actor._slot.generation,
                previous_present=actor._slot.previous is previous_bundle,
                rollback_live=promotion._latest is not None and promotion._latest[0] is receipt,
                pending_checkpoints=len(checkpoint._pending),
                pending_promotions=len(promotion._pending),
                raw_counts=dict(
                    old_sources=len(old._inbox._experiences),
                    old_labels=len(old._inbox._labels),
                    new_sources=len(new._inbox._experiences),
                    new_labels=len(new._inbox._labels),
                ),
                cache_entries=dict(
                    current=len(actor._slot.bundle.cache), previous=len(previous_bundle.cache)
                ),
                metadata_bytes=dict(
                    current=measure_coupled_auxiliary_bytes(actor._slot.bundle.metadata),
                    previous=measure_coupled_auxiliary_bytes(previous_bundle.metadata),
                ),
                budget=_plain(_budget_state(budget)),
                sharing=_plain(asdict(gate.snapshot())),
                registry=_plain(asdict(registry.snapshot())),
                managed=_managed(owner),
                identity=identity(),
                preparation_attempts=checkpoint._attempts,
                prepared_models=len(checkpoint._models),
                build_calls=builder.build_calls,
            )

        failure_evidence: dict[str, Any] | None = None
        after_failure: dict[str, Any] | None = None
        postfailure_refusals: list[dict[str, Any]] = []
        failed_fences: list[dict[str, Any]] = []
        erase_after_failure: dict[str, int] | None = None
        if cleanup_failure:
            if erase_failure is None:
                raise AssertionError("failure fixture lost its birth-bound erasure callback")
            failure_evidence = _poll_cleanup_failure(driver, erase_failure)
            # Token invalidation already relinquished the pending144-byte view.
            # Its lifetime charge stays spent; the remaining owned raw is368.
            account("failed_cleanup", 656, 368)
            after_failure = retained_observation()
            after_failure.update(
                candidate_stopped=dict(old=old._stopped, new=new._stopped),
                lifecycle_failed=lifecycle._failed,
            )
            _require(
                after_failure["generation"] == 1
                and after_failure["previous_present"]
                and not after_failure["rollback_live"]
                and after_failure["pending_checkpoints"] == 0
                and after_failure["pending_promotions"] == 0
                and after_failure["raw_counts"]
                == dict(old_sources=3, old_labels=3, new_sources=3, new_labels=3)
                and after_failure["cache_entries"] == dict(current=1, previous=1)
                and after_failure["metadata_bytes"] == dict(current=24, previous=24)
                and after_failure["old"]["revision"] == before_expiry["old_revision"] + 1
                and after_failure["new"]["revision"] == before_expiry["new_revision"] + 1
                and after_failure["candidate_stopped"] == dict(old=True, new=True)
                and after_failure["lifecycle_failed"]
                and after_failure["old"]["applied"] == before_expiry["old_cursor"]["applied"]
                and after_failure["new"]["applied"] == before_expiry["new_cursor"]["applied"]
                and after_failure["managed"]["revoked_keys"]
                == [["e0", "s0"], ["e1", "s1"], ["e2", "s2"]]
                and _canonical(after_failure["budget"]) == _canonical(before_expiry["budget"])
                and _canonical(after_failure["sharing"]) == _canonical(before_expiry["sharing"])
                and _canonical(after_failure["registry"]) == _canonical(before_expiry["registry"])
                and after_failure["identity"] == original_identity,
                "one-shot erasure failure did not retain truthful partial invalidation and raw ownership",
            )
            erase_after_failure = dict(
                calls=erase_failure.calls,
                failures=erase_failure.failures,
                delegated_calls=erase_failure.delegated_calls,
            )
            postfailure_refusals = refusals(terminal=True, cleanup_failed=True)
            failed_fences = _candidate_fences(old, new)
            finished = driver.finish_cleanup()
            _require(
                finished.outcome == "purged", "original failed driver did not finish raw cleanup"
            )
            finished_cleanup = _plain(asdict(finished))
            _require(
                finished.report is not None
                and finished.report.requested_keys == ()
                and finished.report.revoked_keys == (("e0", "s0"), ("e1", "s1"), ("e2", "s2"))
                and finished.report.checkpoints_invalidated
                == finished.report.promotions_invalidated
                == 0,
                "terminal retry fabricated newly requested identities or invalidated tickets twice",
            )
        elif busy_stop:
            busy_evidence = _busy_stop(driver, new)
            account("busy_stop", 656, 512)
            pending_cleanup = retained_observation()
            _require(
                pending_cleanup["generation"] == 1
                and pending_cleanup["previous_present"]
                and pending_cleanup["rollback_live"]
                and pending_cleanup["pending_checkpoints"] == 1
                and pending_cleanup["pending_promotions"] == 0
                and pending_cleanup["raw_counts"]
                == dict(old_sources=3, old_labels=3, new_sources=3, new_labels=3)
                and pending_cleanup["cache_entries"] == dict(current=1, previous=1)
                and pending_cleanup["metadata_bytes"] == dict(current=24, previous=24)
                and pending_cleanup["old"]["revision"] == before_expiry["old_revision"]
                and pending_cleanup["new"]["revision"] == before_expiry["new_revision"]
                and pending_cleanup["old"]["applied"] == before_expiry["old_cursor"]["applied"]
                and pending_cleanup["new"]["applied"] == before_expiry["new_cursor"]["applied"]
                and not pending_cleanup["old"]["erased_keys"]
                and not pending_cleanup["new"]["erased_keys"]
                and _canonical(pending_cleanup["budget"]) == _canonical(before_expiry["budget"])
                and _canonical(pending_cleanup["sharing"]) == _canonical(before_expiry["sharing"])
                and _canonical(pending_cleanup["registry"]) == _canonical(before_expiry["registry"])
                and _canonical(pending_cleanup["managed"]) == _canonical(before_expiry["managed"])
                and pending_cleanup["identity"] == original_identity,
                "busy stop mutated retained serving/history/resources before cleanup",
            )
            pending_refusals = refusals(terminal=True, cleanup_pending=True)
            finished = driver.finish_cleanup()
            _require(
                finished.outcome == "purged", "released busy holder did not permit terminal cleanup"
            )
            finished_cleanup = _plain(asdict(finished))
        else:
            driver.start()
            # start holds the operation gate across Thread.start; wake explicitly
            # retries a first busy poll without changing the original long interval.
            driver.wake()
            _require(
                driver.wait_for_purge(2.0), "managed serving automatic purge exceeded bounded wait"
            )
        after_purge = _plain(asdict(driver.snapshot()))
    except BaseException as error:
        primary = error
        raise
    finally:
        _stop(driver, primary)

    final_driver = _plain(asdict(driver.snapshot()))
    _require(
        final_driver["state"] == ("failed" if cleanup_failure else "stopped")
        and not final_driver["alive"]
        and not final_driver["pending_cleanup"]
        and final_driver["error_type"] == ("RuntimeError" if cleanup_failure else None)
        and 1 <= final_driver["polls"] <= 4
        and 1
        <= final_driver["purges"]
        <= final_driver["cleanup_attempts"]
        <= original_limits.max_polls + 2,
        "managed serving worker did not stop/join with complete cleanup",
    )
    account("cleanup", 656 if committed else 640, 0)
    if busy_stop or cleanup_failure:
        _require(final_driver["purges"] == 1, "busy path performed another successful cleanup")
    if cleanup_failure:
        _require(
            final_driver["polls"] == 1 and final_driver["cleanup_attempts"] == 2,
            "failure/retry renewed original driver work allowance",
        )
    after_stop = capture(terminal=True)
    if cleanup_failure:
        after_stop.update(
            candidate_stopped=dict(old=old._stopped, new=new._stopped),
            lifecycle_failed=lifecycle._failed,
        )
    for name, completed in (("old_cursor", 1), ("new_cursor", 2)):
        before, after = before_expiry[name], after_stop[name]
        _require(
            after["experiences"] == after["labels"] == []
            and len(after["erased"]) == 3
            and after["applied"] == before["applied"]
            and after["completed_updates"] == completed
            and not after["stopped"]
            and after["last_tick"] == 8,
            "managed serving expiry lost consumed receipts or retained raw inboxes",
        )
    purges = final_driver["purges"]
    _require(
        after_stop["serving"]["metadata"] == after_stop["serving"]["cache"] == {}
        and after_stop["prepared_bundle_metadata" if committed else "pending_promotion_metadata"]
        == after_stop["prepared_bundle_cache" if committed else "pending_promotion_cache"]
        == {}
        and after_stop["pending_checkpoints"] == after_stop["pending_promotions"] == 0
        and after_stop["serving"]["generation"] == purges + (1 if committed else 0)
        and after_stop["old_revision"]
        == before_expiry["old_revision"] + purges + (1 if cleanup_failure else 0)
        and after_stop["new_revision"]
        == before_expiry["new_revision"] + purges + (1 if cleanup_failure else 0)
        and after_stop["serving"]["actor_version"] == ("c0" if committed else "a0")
        and actor._slot.previous is None
        and promotion._latest is None,
        "managed serving cleanup failed to clear owned cache/tickets or changed transaction generation",
    )
    terminal_refusals = refusals(terminal=True)
    renewal_refusals = [
        dict(
            phase="restart",
            message=_refused(driver.start, "retention driver cannot restart or renew allowance"),
        ),
        dict(
            phase="replacement",
            message=_refused(
                lambda: RetentionExpiryDriver(lifecycle, limits=_driver_limits()),
                "original retention driver allowance cannot be renewed",
            ),
        ),
    ]
    if busy_stop or cleanup_failure:
        renewal_refusals.append(
            dict(
                phase="second_finish",
                message=_refused(driver.finish_cleanup, "no unfinished retention cleanup"),
            )
        )
    for name in ("budget", "sharing", "registry"):
        _require(
            _canonical(before_expiry[name]) == _canonical(after_stop[name]),
            f"managed serving cleanup changed original {name}",
        )
    _require(
        identity() == original_identity
        and driver._created_at == created_at
        and driver._limits is original_limits
        and _plain(asdict(driver.snapshot())) == final_driver
        and lifecycle._retention_driver is driver
        and lifecycle._copy_budget is copy_budget
        and budget.clock is elapsed,
        "managed serving changed original driver/clock/resources",
    )
    _require(
        _payload(policy, actor._slot.bundle.learner.snapshot_state())
        == (payload_c if committed else actor_a)
        and _payload(policy, old._candidate.snapshot_state()) == payload_b
        and _payload(policy, new._candidate.snapshot_state()) == payload_c
        and _payload(policy, pending_bundle.learner.snapshot_state()) == payload_c
        and callbacks.policy_digest(new._candidate) == callbacks.policy_digest(reference)
        and budget.updates_completed == gate.snapshot().admitted_updates == 2
        and old._inbox._historical_completed_updates == 1
        and lifecycle.admitted_payload_bytes == 144
        and checkpoint._attempts == len(checkpoint._models) == 1
        and builder.build_calls == 4,
        "managed serving expiry changed learned state, policy or original accounting",
    )
    if committed:
        _require(
            previous_bundle is not None
            and _payload(policy, previous_bundle.learner.snapshot_state()) == actor_a
            and after_stop["previous_serving"] is None
            and after_stop["previous_owned_metadata"] == after_stop["previous_owned_cache"] == {}
            and before_expiry["rollback_live"]
            and not after_stop["rollback_live"],
            "committed expiry changed previous learnedA or retained raw/rollback authority",
        )
    for count, payload in ((2, actor_a), (3, payload_b), (4, payload_c)):
        wake = json.loads(payload)["native"]["wake_state"]
        _require(
            wake is None if family == "pc" else wake["wake_updates"] == count,
            "managed serving lost genuine CPC wake history",
        )
    _require(
        counts
        == dict(
            feature_digest=5 if committed else 3,
            state_valid=2,
            prediction_valid=4,
            utility=4,
            actions=4,
            resource=1,
        ),
        "managed serving reached unexpected hash/guard callbacks",
    )
    row = dict(
        family=family,
        mode=mode,
        policy=asdict(policy),
        actor_a_utf8=actor_a.decode(),
        captured_b_utf8=payload_b.decode(),
        continued_c_utf8=payload_c.decode(),
        prediction_a=prediction_a,
        prediction_c=prediction_c,
        first_updates=[asdict(v) for v in first.updates],
        second_updates=[asdict(v) for v in second.updates],
        waiting_at5=_plain(asdict(waiting)),
        copy_accounting=accounting,
        cache_observations=cache_observations,
        guard_report=_plain(asdict(ticket.report)),
        guard_callback_counts=counts,
        driver_ready=driver_ready,
        driver_after_purge=after_purge,
        driver_final=final_driver,
        driver_origin=dict(created_at=created_at, limits_id=id(original_limits), preserved=True),
        deadline_refusals=deadline_refusals,
        terminal_refusals=terminal_refusals,
        renewal_refusals=renewal_refusals,
        before_expiry=before_expiry,
        after_stop=after_stop,
        identity=dict(original=original_identity, current=identity()),
        build_calls=builder.build_calls,
        automatic_purge_verified=True,
        thread_joined=True,
        complete_continuation_exact=True,
        actor_unchanged=True,
        original_auxiliary_anchor=0.0,
        same_managed_authority=True,
        same_actor_budget_clock_gate=True,
        pending_promotion_committed=False,
    )
    if committed:
        if receipt is None:
            raise AssertionError("committed fixture lost its original receipt")
        row.pop("actor_unchanged")
        row.pop("pending_promotion_committed")
        row.update(
            commit_receipt=asdict(receipt),
            commit_evidence=commit_evidence,
            rollback_live_before_expiry=True,
            rollback_revoked=True,
            serving_c_unchanged=True,
            previous_a_preserved=True,
            promotion_committed=True,
        )
    if checkpoint_observer is not None:
        _require(
            checkpoint_offset == 840
            and [receipt["phase"] for receipt in checkpoint_receipts] == ["B", "C"],
            "complete checkpoints lost their exact original B/C capture allowance",
        )
        row["complete_checkpoints"] = checkpoint_receipts
    if busy_stop:
        if busy_evidence is None:
            raise AssertionError("busy fixture lost its actual driver evidence")
        row.update(
            automatic_purge_verified=False,
            busy_cleanup_verified=True,
            manual_cleanup_verified=True,
            busy_poll=busy_evidence["busy_poll"],
            driver_busy_stopped=busy_evidence["driver_busy_stopped"],
            busy_stop_returned=busy_evidence["stop_returned"],
            pending_cleanup=pending_cleanup,
            pending_refusals=pending_refusals,
            finish_cleanup=finished_cleanup,
        )
    if cleanup_failure:
        if erase_failure is None or failure_evidence is None:
            raise AssertionError("failure fixture lost its original callback/poll evidence")
        _require(
            erase_failure.calls == 5
            and erase_failure.failures == 1
            and erase_failure.delegated_calls == 4
            and lifecycle._erase is erase_failure
            and old._stopped
            and new._stopped
            and not lifecycle._failed,
            "terminal retry changed erase callback, fault count or stopped candidate fences",
        )
        row.update(
            automatic_purge_verified=False,
            thread_joined=False,
            worker_started=False,
            manual_cleanup_verified=True,
            cleanup_failure_verified=True,
            terminal_retry_verified=True,
            failure_poll=failure_evidence["poll"],
            driver_after_failure=failure_evidence["driver"],
            after_failure=after_failure,
            postfailure_refusals=postfailure_refusals,
            candidate_fences_after_failure=failed_fences,
            candidate_fences_final=_candidate_fences(old, new),
            finish_cleanup=finished_cleanup,
            erase_callback=dict(
                after_failure=erase_after_failure,
                final=dict(
                    calls=erase_failure.calls,
                    failures=erase_failure.failures,
                    delegated_calls=erase_failure.delegated_calls,
                ),
                original=id(erase_failure),
                current=id(lifecycle._erase),
            ),
        )
    _require(
        len(_canonical(row).encode()) <= 65536,
        "managed serving case exceeds bounded JSON allowance",
    )
    return row


def run_managed_serving_example() -> dict[str, Any]:
    rows = [_case(family, mode) for mode in MODES for family in FAMILIES]
    return _report(rows, committed=False)


def run_committed_serving_example() -> dict[str, Any]:
    rows = [_case(family, mode, committed=True) for mode in MODES for family in FAMILIES]
    return _report(rows, committed=True)


def run_busy_committed_serving_example() -> dict[str, Any]:
    rows = [
        _case(family, mode, committed=True, busy_stop=True) for mode in MODES for family in FAMILIES
    ]
    return _report(rows, committed=True, busy_stop=True)


def run_cleanup_failure_example() -> dict[str, Any]:
    rows = [
        _case(family, mode, committed=True, cleanup_failure=True)
        for mode in MODES
        for family in FAMILIES
    ]
    return _report(rows, committed=True, cleanup_failure=True)


def _report(
    rows: list[dict[str, Any]],
    *,
    committed: bool,
    busy_stop: bool = False,
    cleanup_failure: bool = False,
) -> dict[str, Any]:
    for offset in (0, 3):
        for stage in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            pc, neutral = (
                json.loads(rows[offset + number][stage])["native"]["arrays"] for number in (0, 1)
            )
            _require(
                pc == neutral, "PC-neutral complete native frames differ under managed serving"
            )
    result = dict(
        protocol=describe_cleanup_failure_example()
        if cleanup_failure
        else describe_busy_committed_serving_example()
        if busy_stop
        else describe_committed_serving_example()
        if committed
        else describe_managed_serving_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        managed_serving_expiry_verified=True,
        full_managed_process_recovery_verified=False,
    )
    if committed:
        result["committed_serving_expiry_verified"] = True
    if busy_stop:
        result["busy_cleanup_verified"] = True
    if cleanup_failure:
        result["cleanup_failure_verified"] = True
    return result
