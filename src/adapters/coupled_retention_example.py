"""Fixed elapsed-retention control with one original bounded worker per case.

An injected locked clock exposes elapsed deadline admission separately from real
thread scheduling. The existing driver automatically clears retained raw copies
after deadline refusals; stop always joins it within its declared bound. Learned
state and consumed receipts persist. Caller reports/copies are outside deletion;
no RAM overwrite, unlearning, physical latency, process recovery or science claim.
"""

from dataclasses import asdict
from math import isfinite
import json
import os
from threading import Lock
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
    describe_example,
)
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_elapsed_retention import make_coupled_elapsed_lifecycle
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_managed_cleanup import _record
from src.adapters.coupled_managed_consent import _declaration, _managed, _PoisonPayload
from src.adapters.coupled_payload_copies import ManagedCoupledBuilder
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, _budget_state
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.retention_expiry import RetentionExpiryDriver
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits
from src.core.retention_driver import RetentionDriverLimits


class _LockedElapsedClock:
    """One explicit monotonic injected clock shared with the retention worker."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value = 0.0

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance_to(self, value: float) -> None:
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("elapsed example clock requires finite nonnegative time")
        with self._lock:
            if value < self._value:
                raise ValueError("elapsed example clock cannot move backwards")
            self._value = float(value)


def _policy() -> DataRetentionPolicy:
    return DataRetentionPolicy(
        192,
        100,
        PayloadOwnershipLimits(4, 4),
        max_checkpoint_pending=2,
        max_checkpoint_preparations=2,
        max_promotion_pending=1,
        owned_payload_copies=PayloadCopyLimits(576),
        max_retention_seconds=2.0,
    )


def _driver_limits() -> RetentionDriverLimits:
    return RetentionDriverLimits(30.0, 4, 60.0, 2.0)


def describe_elapsed_retention_example() -> dict[str, Any]:
    description = describe_example()
    description.update(
        protocol_id="coupled_elapsed_retention_engineering_v1",
        retention_policy=asdict(_policy()),
        driver_limits=asdict(_driver_limits()),
        lifecycle_limits=asdict(LifecycleLimits(4, 4, allow_synthetic=True, allow_unverified=True)),
        sharing_limits=asdict(SharingLimits(2, 3, 1)),
        runtime_update_limit=3,
        injected_elapsed_times=[0.0, 0.5, 1.0, 2.0],
        elapsed_deadline=2.0,
        label_arrivals=[3, 7, 9],
        preparation_attempt_limit=2,
        native_training_calls=36,
        native_energy_evaluation_cap=180,
        native_constructor_cap=48,
        prediction_calls=36,
        prediction_cap=64,
        guarded_inference_cap=18,
        retention_workers=6,
        limitations="Injected elapsed time demonstrates admission, not physical scheduling latency. One actual bounded worker per case drops owned raw references, not learned state or caller copies. No RAM overwrite, unlearning, process recovery, attestation or scientific claim.",
    )
    return description


def _inbox_metadata(runtime: ActorShadowRuntime[Any, Any, Any, Any]) -> dict[str, Any]:
    inbox = runtime._inbox
    return dict(
        source_keys=_plain(sorted(inbox._experiences)),
        label_keys=_plain(sorted(inbox._labels)),
        erased_keys=_plain(sorted(inbox._erased)),
        applied=_plain([asdict(v) for v in inbox._applied.values()]),
        last_tick=inbox._last_time,
        stopped=inbox.stopped,
        revision=runtime._revision,
    )


def _stop(driver: RetentionExpiryDriver, primary: BaseException | None) -> None:
    try:
        if not driver.stop():
            raise RuntimeError("elapsed retention worker did not join with cleanup complete")
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note(f"Retention stop failed: {type(error).__name__}: {error}")


def _case(family: str, mode: str) -> dict[str, Any]:
    # One ordered transaction keeps the original driver/clock/resource lifetime
    # visible; observation, deadline refusal and stop responsibilities are helpers.
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
    elapsed, clock = _LockedElapsedClock(), LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), elapsed)
    old = ActorShadowRuntime(
        reference,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=3,
        max_consolidations=0,
    )
    actor = old.actor
    gate = ServingPriorityGate(SharingLimits(2, 3, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(old, gate)
    owner = ManagedExperienceOwner(
        shared, limits=LifecycleLimits(4, 4, allow_synthetic=True, allow_unverified=True)
    )
    lifecycle = make_coupled_elapsed_lifecycle(owner, policy=_policy())
    driver = RetentionExpiryDriver(lifecycle, limits=_driver_limits())
    primary: BaseException | None = None
    try:
        ready = _plain(asdict(driver.snapshot()))
        created_at, original_limits = driver._created_at, driver._limits
        registry, copy_budget = actor._payload_registry, lifecycle._copy_budget
        registration, eligibility = old._inbox._registration_guard, old._inbox._training_guard

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
        for number, subject in enumerate(("subject-A", "subject-B", "subject-A")):
            owner.declare(_declaration(number, subject))
        _record(owner, features, targets)
        accounting: list[dict[str, Any]] = []

        def account(phase: str, charged: int, retained: int) -> None:
            observed = lifecycle.payload_byte_snapshot()
            _require(
                observed.charged_bytes == charged and observed.observed_retained_bytes == retained,
                f"elapsed copy accounting differs at {phase}",
            )
            accounting.append(dict(phase=phase, **_plain(asdict(observed))))

        account("ingress", 144, 144)
        actor_a = _payload(policy, actor.snapshot_state().state)
        prediction_a = shared.predict(features.copy()).prediction.tolist()

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
                "elapsed-retention continuation differs from complete independent reference",
            )
            return poll, payload, prediction

        elapsed.advance_to(0.5)
        clock.advance_to(3)
        first, payload_b, _ = update(0)
        gate.pause()
        builder = ManagedCoupledBuilder(callbacks)
        control = CandidateCheckpointController(
            shared,
            build_learner=builder,
            state_digest=callbacks.state_digest,
            policy_digest=callbacks.policy_digest,
            max_pending=2,
            max_prepare_attempts=2,
        )
        first_token = control.capture()
        account("first_capture", 288, 288)
        new = control.restore(first_token)
        account("restore", 432, 288)
        _require(
            shared.predict(features.copy()).prediction.tolist() == prediction_a
            and old._retired
            and shared._runtime is new
            and builder.build_calls == 1,
            "elapsed handoff changed actorA or original ownership",
        )
        gate.resume()
        clock.advance_to(5)
        waiting = owner.train_ready()
        _require(
            waiting.updates == () and waiting.deferred_reason is None,
            "future elapsed label trained at5",
        )
        elapsed.advance_to(1.0)
        clock.advance_to(7)
        second, payload_c, prediction_c = update(1)
        clock.advance_to(8)
        gate.pause()
        pending = control.capture()
        account("second_capture", 576, 432)

        def capture() -> dict[str, Any]:
            return dict(
                old_cursor=_plain(asdict(old._inbox._borrow_cursor())),
                new_cursor=_plain(asdict(new._inbox._borrow_cursor())),
                old_revision=old._revision,
                new_revision=new._revision,
                budget=_plain(_budget_state(budget)),
                sharing=_plain(asdict(gate.snapshot())),
                registry=_plain(asdict(registry.snapshot())),
                managed=_managed(owner),
                owned_copies=_plain(asdict(lifecycle.payload_byte_snapshot())),
                admitted_payload_bytes=lifecycle.admitted_payload_bytes,
                preparation_attempts=control._attempts,
                prepared_models=len(control._models),
                pending_checkpoints=len(control._pending),
                build_calls=builder.build_calls,
            )

        before_expiry = capture()
        _require(
            len(before_expiry["registry"]["holders"])
            == before_expiry["registry"]["total_enrollments"]
            == 4,
            "elapsed fixture lost its four original retained holders",
        )
        poison = _PoisonPayload()

        def refusal_state() -> dict[str, Any]:
            # No expired payload is copied or serialized just to prove refusal.
            return dict(
                old=_inbox_metadata(old),
                new=_inbox_metadata(new),
                attempts=control._attempts,
                models=len(control._models),
                pending=len(control._pending),
                builds=builder.build_calls,
                budget=_plain(_budget_state(budget)),
                sharing=_plain(asdict(gate.snapshot())),
            )

        def refusals(*, terminal: bool) -> list[dict[str, Any]]:
            closed = (
                "retention driver terminal; raw admission/access is closed"
                if terminal
                else "declared payload retention has expired; cleanup required"
            )
            actions: tuple[tuple[str, Callable[[], Any], str], ...] = (
                ("predict", lambda: shared.predict(cast(Array, poison)), closed),
                ("snapshot", new.candidate_snapshot, closed),
                (
                    "restore",
                    lambda: control.restore(pending),
                    "checkpoint is foreign, copied, discarded or already used"
                    if terminal
                    else closed,
                ),
                (
                    "ingress_source",
                    lambda: owner.record_experience(
                        Experience(
                            "s3",
                            "e3",
                            8,
                            "a0",
                            cast(Array, poison),
                            "train",
                            ExperiencePermissions(True, True),
                        )
                    ),
                    closed,
                ),
                (
                    "ingress_label",
                    lambda: owner.record_label(
                        LabelArrival("l3", "s3", "e3", 9, "a0", cast(Array, poison))
                    ),
                    closed,
                ),
            )
            records = []
            for phase, action, message in actions:
                before = refusal_state()
                result = _refused(action, message)
                _require(
                    _canonical(before) == _canonical(refusal_state()) and poison.copy_attempts == 0,
                    "elapsed refusal changed resource/history or copied poisoned raw data",
                )
                records.append(
                    dict(
                        phase=phase,
                        message=result,
                        poison_copy_attempts=poison.copy_attempts,
                        preparation_attempts=control._attempts,
                        build_calls=builder.build_calls,
                        complete_state_preserved=True,
                    )
                )
            return records

        elapsed.advance_to(2.0)
        deadline_refusals = refusals(terminal=False)
        _require(
            driver.snapshot().state == "ready" and driver.snapshot().polls == 0,
            "retention worker started before deadline refusal evidence",
        )
        driver.start()
        # start holds the operation lease while creating its worker. Wake the
        # first retry if that worker observes startup contention before polling.
        driver.wake()
        _require(
            driver.wait_for_purge(2.0), "automatic elapsed cleanup did not finish within wait bound"
        )
        after_purge = _plain(asdict(driver.snapshot()))
    except BaseException as error:
        primary = error
        raise
    finally:
        _stop(driver, primary)

    final_driver = _plain(asdict(driver.snapshot()))
    _require(
        final_driver["state"] == "stopped"
        and not final_driver["alive"]
        and not final_driver["pending_cleanup"]
        and final_driver["error_type"] is None
        and 1 <= final_driver["polls"] <= 4
        and 1
        <= final_driver["purges"]
        <= final_driver["cleanup_attempts"]
        <= original_limits.max_polls + 2,
        "elapsed worker did not terminate with complete bounded cleanup",
    )
    account("cleanup", 576, 0)
    after_stop = capture()
    for name, completed in (("old_cursor", 1), ("new_cursor", 2)):
        before, after = before_expiry[name], after_stop[name]
        _require(
            after["experiences"] == after["labels"] == []
            and len(after["erased"]) == 3
            and after["applied"] == before["applied"]
            and after["completed_updates"] == completed
            and not after["stopped"]
            and after["last_tick"] == 8,
            "elapsed cleanup lost consumed history or retained owned raw data",
        )
    _require(
        after_stop["pending_checkpoints"] == 0
        and old._inbox._historical_completed_updates == 1
        and after_stop["admitted_payload_bytes"] == 144
        and budget.updates_completed == 2
        and gate.snapshot().admitted_updates == 2
        and builder.build_calls == control._attempts == 1,
        "elapsed cleanup reset original quotas/old ledger or reached another builder",
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
    _require(
        identity() == original_identity
        and budget.clock is elapsed
        and lifecycle._wall_clock is elapsed
        and lifecycle._retention_driver is driver
        and lifecycle._copy_budget is copy_budget
        and driver._created_at == created_at
        and driver._limits is original_limits
        and _plain(asdict(driver.snapshot())) == final_driver
        and new._inbox._registration_guard is registration
        and new._inbox._training_guard is eligibility,
        "elapsed handoff/cleanup replaced original clock/driver/resource/managed ownership",
    )
    # Public raw/native access is terminal. Only trusted learner snapshots check
    # preservation; this is a local forensic observation, never resumed admission.
    _require(
        _payload(policy, actor._learner.snapshot_state()) == actor_a
        and _payload(policy, old._candidate.snapshot_state()) == payload_b
        and _payload(policy, new._candidate.snapshot_state()) == payload_c
        and callbacks.policy_digest(new._candidate) == callbacks.policy_digest(reference),
        "elapsed purge changed learned full state or update policy",
    )
    for count, payload in ((2, actor_a), (3, payload_b), (4, payload_c)):
        wake = json.loads(payload)["native"]["wake_state"]
        _require(
            wake is None if family == "pc" else wake["wake_updates"] == count,
            "elapsed cleanup lost genuine CPC wake history",
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
        driver_ready=ready,
        driver_after_purge=after_purge,
        driver_final=final_driver,
        deadline_refusals=deadline_refusals,
        terminal_refusals=terminal_refusals,
        renewal_refusals=renewal_refusals,
        driver_origin=dict(created_at=created_at, limits_id=id(original_limits), preserved=True),
        before_expiry=before_expiry,
        after_stop=after_stop,
        identity=dict(original=original_identity, current=identity()),
        build_calls=builder.build_calls,
        automatic_purge_verified=True,
        thread_joined=True,
        complete_continuation_exact=True,
        actor_unchanged=True,
        same_managed_authority=True,
        same_actor_budget_clock_gate=True,
    )
    _require(len(_canonical(row).encode()) <= 65536, "elapsed case exceeds bounded JSON allowance")
    return row


def run_elapsed_retention_example() -> dict[str, Any]:
    rows = [_case(family, mode) for mode in MODES for family in FAMILIES]
    for offset in (0, 3):
        for stage in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            pc, neutral = (
                json.loads(rows[offset + number][stage])["native"]["arrays"] for number in (0, 1)
            )
            _require(
                pc == neutral, "PC-neutral complete native frames differ under elapsed retention"
            )
    return dict(
        protocol=describe_elapsed_retention_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        elapsed_retention_verified=True,
        full_managed_process_recovery_verified=False,
    )
