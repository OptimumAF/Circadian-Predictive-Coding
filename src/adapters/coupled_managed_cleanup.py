"""Bounded native handoff and conservative managed raw-payload cleanup example.

The six fixed local cases keep one managed owner, lifecycle, registry and resource
lineage across a candidate handoff. Opt-out erases delivered raw inbox copies and
pending checkpoints while preserving learned state and consumed receipts. An
inspected caller copy remains outside that ownership guarantee. These synthetic,
unverified declarations establish no provenance, human consent, RAM overwrite,
unlearning, process recovery or scientific benefit. No new cleanup algorithm.
"""

from dataclasses import asdict
import json
import os
from typing import Any, Callable

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
from src.adapters.coupled_data_lifecycle import (
    coupled_replay_footprint,
    make_coupled_data_lifecycle,
)
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_payload_copies import (
    ManagedCoupledBuilder,
    make_coupled_copy_lifecycle,
)
from src.adapters.coupled_managed_consent import _declaration, _managed
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, _budget_state
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataCleanupReport, DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.resource_sharing import SharingLimits

_ARRIVALS = (3, 7, 9)
_SUBJECTS = ("subject-A", "subject-B", "subject-A")


def describe_managed_cleanup_example() -> dict[str, Any]:
    description = describe_example()
    description.update(
        protocol_id="coupled_managed_cleanup_engineering_v1",
        runtime_update_limit=3,
        sharing_limits=asdict(SharingLimits(2, 3, 1)),
        preparation_attempt_limit=1,
        label_arrivals=list(_ARRIVALS),
        cleanup_at=8,
        lifecycle_limits=asdict(LifecycleLimits(4, 4, allow_synthetic=True, allow_unverified=True)),
        retention_policy=asdict(_retention_policy()),
        native_training_calls=36,
        native_energy_evaluation_cap=180,
        native_constructor_cap=48,
        prediction_calls=42,
        prediction_cap=64,
        guarded_inference_cap=18,
        limitations="Local synthetic/unverified declarations only. Conservative raw-payload cleanup preserves learned parameters/chemistry and consumed receipts; caller copies remain. No RAM overwrite, unlearning, quota refund, provenance attestation, scientific claim or managed process recovery.",
    )
    return description


def describe_owned_copy_example() -> dict[str, Any]:
    description = describe_managed_cleanup_example()
    description.update(
        protocol_id="coupled_owned_copies_engineering_v1",
        retention_policy=asdict(_retention_policy(owned_copies=True)),
        preparation_attempt_limit=2,
        prediction_calls=36,
        limitations="Conservative original lifetime raw-copy accounting, not learned parameter/Python/RSS/caller-copy storage. Cleanup releases owned raw references without refunding charges or unlearning. No provenance attestation, physical-memory bound, process recovery or scientific claim.",
    )
    return description


def _retention_policy(*, owned_copies: bool = False) -> DataRetentionPolicy:
    return DataRetentionPolicy(
        192 if owned_copies else 144,
        100,
        PayloadOwnershipLimits(4, 4),
        max_checkpoint_pending=2 if owned_copies else 1,
        max_checkpoint_preparations=2 if owned_copies else 1,
        max_promotion_pending=1,
        owned_payload_copies=PayloadCopyLimits(576) if owned_copies else None,
    )


def _record(owner, features, targets) -> None:
    for number, arrival in enumerate(_ARRIVALS):
        owner.record_experience(
            Experience(
                f"s{number}",
                f"e{number}",
                0,
                "a0",
                features.copy(),
                "train",
                ExperiencePermissions(True, True),
            )
        )
        owner.record_label(
            LabelArrival(f"l{number}", f"s{number}", f"e{number}", arrival, "a0", targets.copy())
        )


def _lifecycle_state(lifecycle) -> dict[str, Any]:
    result = dict(
        policy=asdict(lifecycle._policy),
        admitted_payload_bytes=lifecycle.admitted_payload_bytes,
        last_tick=lifecycle._last_tick,
        failed=lifecycle._failed,
    )
    if lifecycle._copy_budget is not None:
        result["owned_copies"] = _plain(asdict(lifecycle.payload_byte_snapshot()))
    return result


def _cleanup_cursors(before: dict[str, Any], after: dict[str, Any]) -> None:
    for name, completed in (("old_cursor", 1), ("new_cursor", 2)):
        old, new = before[name], after[name]
        _require(
            len(old["experiences"]) == len(old["labels"]) == 3
            and new["experiences"] == new["labels"] == []
            and len(new["erased"]) == 3
            and new["applied"] == old["applied"]
            and new["completed_updates"] == old["completed_updates"] == completed
            and new["last_tick"] == 8
            and not new["stopped"],
            "cleanup changed consumed history or failed to clear an owned raw inbox",
        )
    for name in ("old_revision", "new_revision"):
        _require(after[name] == before[name] + 1, "cleanup must invalidate each candidate revision")
    for name in ("budget", "sharing", "registry", "native_digests", "policy_digests"):
        _require(_canonical(before[name]) == _canonical(after[name]), f"cleanup changed {name}")
    _require(
        after["lifecycle"]["admitted_payload_bytes"] == 144
        and before["lifecycle"]["policy"] == after["lifecycle"]["policy"],
        "cleanup refunded lifetime ingress or changed retention limits",
    )


def _case(family: str, mode: str, *, owned_copies: bool = False) -> dict[str, Any]:
    # Keep the ordered single-owner transaction visible. Recording, cleanup
    # assertions and observations are separate helpers; no parallel fixture graph.
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
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
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
    limits = LifecycleLimits(4, 4, allow_synthetic=True, allow_unverified=True)
    owner = ManagedExperienceOwner(shared, limits=limits)
    lifecycle = (
        make_coupled_copy_lifecycle(owner, policy=_retention_policy(owned_copies=True))
        if owned_copies
        else make_coupled_data_lifecycle(owner, policy=_retention_policy())
    )
    copy_budget = lifecycle._copy_budget
    copy_accounting: list[dict[str, Any]] = []

    def account(phase: str, charged: int, observed: int) -> None:
        if owned_copies:
            value = lifecycle.payload_byte_snapshot()
            _require(
                value.charged_bytes == charged and value.observed_retained_bytes == observed,
                f"copy accounting differs at {phase}",
            )
            copy_accounting.append(dict(phase=phase, **_plain(asdict(value))))

    registry = actor._payload_registry
    catalog, opted_out, revoked = owner._catalog, owner._opted_out, owner._revoked_keys
    registration, eligibility = old._inbox._registration_guard, old._inbox._training_guard

    def identities() -> dict[str, int]:
        current = shared._runtime
        return dict(
            owner=id(owner),
            lifecycle=id(owner._lifecycle),
            registry=id(current.actor._payload_registry),
            catalog=id(owner._catalog),
            opted_out=id(owner._opted_out),
            revoked_keys=id(owner._revoked_keys),
            limits=id(owner._limits),
            registration_guard=id(current._inbox._registration_guard),
            eligibility_guard=id(current._inbox._training_guard),
            actor=id(current.actor),
            budget=id(current._budget),
            clock=id(current._inbox._clock),
            gate=id(shared._sharing),
        )

    original_identity = identities()
    for number, subject in enumerate(_SUBJECTS):
        owner.declare(_declaration(number, subject))
    _record(owner, features, targets)
    account("ingress", 144, 144)
    _require(lifecycle.admitted_payload_bytes == 144, "fixed ingress must consume original144B")
    actor_a = _payload(policy, actor.snapshot_state().state)
    prediction_a = shared.predict(features.copy()).prediction.tolist()
    builds: list[str] = []

    def exact_update(number: int):
        poll = owner.train_ready()
        direct = reference.train_batch(features.copy(), targets.copy())
        current = shared._runtime
        payload = _payload(policy, current.candidate_snapshot().state)
        prediction = current._candidate.predict(features.copy()).tolist()
        _require(
            [item.sample_id for item in poll.updates] == [f"s{number}"]
            and poll.updates[0].diagnostic == direct
            and payload == _payload(policy, reference.snapshot_state())
            and prediction == reference.predict(features.copy()).tolist()
            and callbacks.policy_digest(current._candidate) == callbacks.policy_digest(reference),
            "native cleanup continuation differs from complete independent reference",
        )
        return poll, payload, prediction

    clock.advance_to(3)
    first, payload_b, _ = exact_update(0)
    gate.pause()

    def build(state):
        _require(
            shared.predict(features.copy()).prediction.tolist() == prediction_a
            and _payload(policy, actor.snapshot_state().state) == actor_a,
            "actorA unavailable or changed inside cleanup handoff builder",
        )
        builds.append("correct_policy")
        return callbacks.build(state)

    managed_builder = ManagedCoupledBuilder(callbacks) if owned_copies else None
    control = CandidateCheckpointController(
        shared,
        build_learner=managed_builder if managed_builder is not None else build,
        state_digest=callbacks.state_digest,
        policy_digest=callbacks.policy_digest,
        max_pending=2 if owned_copies else 1,
        max_prepare_attempts=2 if owned_copies else 1,
    )
    first_token = control.capture()
    account("first_capture", 288, 288)
    new = control.restore(first_token)
    account("restore", 432, 288)
    if owned_copies:
        _require(
            shared.predict(features.copy()).prediction.tolist() == prediction_a
            and _payload(policy, actor.snapshot_state().state) == actor_a,
            "actorA changed or unavailable across copy-aware handoff",
        )
        builds.append("correct_policy")
    _require(old._retired and shared._runtime is new, "handoff did not retire original owner")
    gate.resume()
    clock.advance_to(5)
    waiting = owner.train_ready()
    _require(waiting.updates == () and waiting.deferred_reason is None, "future label trained at5")
    clock.advance_to(7)
    second, payload_c, prediction_c = exact_update(1)
    clock.advance_to(8)
    gate.pause()
    pending = control.capture()
    account("second_capture", 576, 432)
    caller_view = None if owned_copies else control.inspect(pending)

    def capture() -> dict[str, Any]:
        return dict(
            old_cursor=_plain(asdict(old._inbox.capture_cursor())),
            new_cursor=_plain(asdict(new._inbox.capture_cursor())),
            old_revision=old._revision,
            new_revision=new._revision,
            budget=_plain(_budget_state(budget)),
            sharing=_plain(asdict(gate.snapshot())),
            registry=_plain(asdict(registry.snapshot())),
            lifecycle=_lifecycle_state(lifecycle),
            managed=_managed(owner),
            native_digests=dict(
                actor=callbacks.state_digest(actor.snapshot_state().state),
                old=callbacks.state_digest(old._candidate.snapshot_state()),
                new=callbacks.state_digest(new.candidate_snapshot().state),
            ),
            policy_digests=dict(
                actor=callbacks.policy_digest(actor._learner),
                old=callbacks.policy_digest(old._candidate),
                new=callbacks.policy_digest(new._candidate),
            ),
        )

    copy_refusals: list[dict[str, Any]] = []
    if owned_copies:
        before = capture()
        message = _refused(
            lambda: control.restore(pending),
            "original lifetime owned payload byte allowance exhausted",
        )
        after = capture()
        _require(_canonical(before) == _canonical(after), "copy refusal changed original candidate")
        _require(
            managed_builder is not None
            and managed_builder.build_calls == 1
            and control._attempts == 2
            and len(control._models) == 1
            and len(control._pending) == 1,
            "copy restore refusal did not preserve spent attempt before builder",
        )
        copy_refusals.append(
            dict(
                phase="copy_restore",
                message=message,
                preparation_attempts=2,
                prepared_models=1,
                builder_calls=1,
                complete_state_preserved=True,
            )
        )
        message = _refused(
            control.capture, "original lifetime owned payload byte allowance exhausted"
        )
        _require(
            _canonical(after) == _canonical(capture()), "copy capture refusal changed candidate"
        )
        _require(len(control._pending) == 1, "copy capture refusal retained another checkpoint")
        copy_refusals.append(
            dict(
                phase="copy_capture",
                message=message,
                preparation_attempts=2,
                prepared_models=1,
                builder_calls=1,
                complete_state_preserved=True,
            )
        )
    before_cleanup = capture()
    _require(
        len(before_cleanup["registry"]["holders"])
        == before_cleanup["registry"]["total_enrollments"]
        == 4
        and control._attempts == (2 if owned_copies else 1)
        and len(control._models) == len(builds) == len(control._pending) == 1,
        "cleanup fixture must retain original four holders and one pending checkpoint",
    )
    report = lifecycle.opt_out("subject-A")
    expected = DataCleanupReport(
        "opt_out",
        (("e0", "s0"), ("e2", "s2")),
        (("e0", "s0"), ("e1", "s1"), ("e2", "s2")),
        0,
        2,
        1,
        0,
    )
    _require(type(report) is DataCleanupReport and report == expected, "unexpected cleanup report")
    after_cleanup = capture()
    account("cleanup", 576, 0)
    _cleanup_cursors(before_cleanup, after_cleanup)
    _require(
        old._inbox._historical_completed_updates == 1
        and old._retired
        and owner._opted_out == {"subject-A"}
        and owner._revoked_keys == set(expected.revoked_keys)
        and not control._pending
        and all(
            coupled_replay_footprint(item).payload_bytes == 0
            for item in (actor._learner, old._candidate, new._candidate)
        ),
        "cleanup lost old ledger, permanent revocation or fixed no-replay model contract",
    )
    refusals: list[dict[str, Any]] = copy_refusals

    def refuse(phase: str, action: Callable[[], Any], message: str, delta: int) -> None:
        before = capture()
        result = _refused(action, message)
        after = capture()
        expected_after = _plain(before)
        expected_after["new_revision"] += delta
        _require(
            _canonical(expected_after) == _canonical(after), "cleanup refusal changed owned state"
        )
        _require(
            control._attempts == (2 if owned_copies else 1)
            and len(control._models) == len(builds) == 1,
            "refusal reached builder",
        )
        refusals.append(
            dict(
                phase=phase,
                message=result,
                revision_before=before["new_revision"],
                revision_after=after["new_revision"],
                revision_delta=delta,
                complete_state_preserved=True,
                preparation_attempts=control._attempts,
                prepared_models=len(control._models),
                builder_calls=managed_builder.build_calls
                if managed_builder is not None
                else len(builds),
            )
        )

    refuse(
        "invalidated_checkpoint",
        lambda: control.restore(pending),
        "checkpoint is foreign, copied, discarded or already used",
        0,
    )
    owner.declare(_declaration(3, "subject-C"))
    refuse(
        "ingress_source",
        lambda: owner.record_experience(
            Experience("s3", "e3", 8, "a0", features, "train", ExperiencePermissions(True, True))
        ),
        "original lifetime owned payload byte allowance exhausted"
        if owned_copies
        else "lifetime ingress byte allowance exhausted",
        1,
    )
    refuse(
        "ingress_label",
        lambda: owner.record_label(LabelArrival("l3", "s3", "e3", 9, "a0", targets)),
        "original lifetime owned payload byte allowance exhausted"
        if owned_copies
        else "lifetime ingress byte allowance exhausted",
        1,
    )
    refuse(
        "regrant_subject_a",
        lambda: owner.declare(_declaration(4, "subject-A")),
        "subject has opted out",
        0,
    )
    refuse(
        "fifth_declaration",
        lambda: owner.declare(_declaration(4, "subject-D")),
        "lifetime approved or replay record quota exhausted",
        0,
    )
    caller_copy = (
        dict(inspected=False, outside_cleanup_guarantee=True)
        if caller_view is None
        else dict(
            experiences=len(caller_view.inbox.experiences),
            labels=len(caller_view.inbox.labels),
            payload_bytes=sum(s.features.nbytes for s in caller_view.inbox.experiences)
            + sum(t.targets.nbytes for t in caller_view.inbox.labels),
            outside_cleanup_guarantee=True,
        )
    )
    _require(
        caller_view is None
        or (
            caller_copy["experiences"] == caller_copy["labels"] == 3
            and caller_copy["payload_bytes"] == 144
        ),
        "caller inspected copy was incorrectly claimed erased",
    )
    gate.resume()
    clock.advance_to(9)
    empty = owner.train_ready()
    final_cursor = new._inbox.capture_cursor()
    _require(
        empty.updates == ()
        and empty.deferred_reason is None
        and not new._stopped
        and not final_cursor.stopped
        and final_cursor.experiences == final_cursor.labels == ()
        and final_cursor.completed_updates
        == budget.updates_completed
        == gate.snapshot().admitted_updates
        == 2
        and budget.budget.max_training_updates == gate.snapshot().limits.max_admitted_updates == 3
        and final_cursor.last_tick == 9
        and len(final_cursor.erased) == 3
        and [item.applied_at for item in final_cursor.applied] == [3, 7],
        "post-cleanup empty poll changed quota, consumed history or stopped the candidate",
    )
    _require(
        identities() == original_identity
        and owner._lifecycle is lifecycle
        and owner._catalog is catalog
        and owner._opted_out is opted_out
        and owner._revoked_keys is revoked
        and new._inbox._registration_guard is registration
        and new._inbox._training_guard is eligibility
        and lifecycle._registry is registry
        and registry._lifecycle is lifecycle
        and lifecycle._copy_budget is copy_budget,
        "cleanup handoff replaced original managed/resource ownership",
    )
    _require(
        _payload(policy, new.candidate_snapshot().state) == payload_c
        and _payload(policy, actor.snapshot_state().state) == actor_a
        and (owned_copies or shared.predict(features.copy()).prediction.tolist() == prediction_a),
        "cleanup changed learned candidateC or serving actorA",
    )
    for count, payload in ((2, actor_a), (3, payload_b), (4, payload_c)):
        wake = json.loads(payload)["native"]["wake_state"]
        _require(
            wake is None if family == "pc" else wake["wake_updates"] == count,
            "cleanup lost full CPC wake state",
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
        first_updates=[asdict(item) for item in first.updates],
        second_updates=[asdict(item) for item in second.updates],
        before_cleanup=before_cleanup,
        after_cleanup=after_cleanup,
        cleanup_report=asdict(report),
        caller_copy=caller_copy,
        identity=dict(original=original_identity, current=identities()),
        refusals=refusals,
        final_cursor=_plain(asdict(final_cursor)),
        final_budget=_plain(_budget_state(budget)),
        final_sharing=_plain(asdict(gate.snapshot())),
        final_registry=_plain(asdict(registry.snapshot())),
        final_lifecycle=_lifecycle_state(lifecycle),
        managed_after=_managed(owner),
        builds=builds,
        preparation_attempts=control._attempts,
        prepared_models=len(control._models),
        pending_checkpoints=len(control._pending),
        waiting_at5=_plain(asdict(waiting)),
        empty_at9=_plain(asdict(empty)),
        actor_unchanged=True,
        complete_continuation_exact=True,
        same_managed_authority=True,
        same_actor_budget_clock_gate=True,
        owned_payloads_cleared=True,
    )
    if owned_copies:
        if managed_builder is None:
            raise AssertionError("copy-aware fixture lost its inspectable builder")
        _require(
            managed_builder.build_calls == 1,
            "copy-aware builder call count changed after refusal/cleanup",
        )
        row.update(
            copy_accounting=copy_accounting,
            copy_budget_identity=dict(original=id(copy_budget), current=id(lifecycle._copy_budget)),
            build_calls=managed_builder.build_calls,
            owned_copy_limits_verified=True,
        )
    _require(
        len(_canonical(row).encode("utf-8")) <= 65536,
        "managed cleanup case exceeds bounded JSON allowance",
    )
    return row


def run_managed_cleanup_example() -> dict[str, Any]:
    rows = [_case(family, mode) for mode in MODES for family in FAMILIES]
    return _report(rows, owned_copies=False)


def run_owned_copy_example() -> dict[str, Any]:
    rows = [_case(family, mode, owned_copies=True) for mode in MODES for family in FAMILIES]
    return _report(rows, owned_copies=True)


def _report(rows: list[dict[str, Any]], *, owned_copies: bool) -> dict[str, Any]:
    for offset in (0, 3):
        for stage in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            pc, neutral = (
                json.loads(rows[offset + number][stage])["native"]["arrays"] for number in (0, 1)
            )
            _require(pc == neutral, "PC-neutral complete frames differ under managed cleanup")
    result = dict(
        protocol=describe_owned_copy_example()
        if owned_copies
        else describe_managed_cleanup_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        managed_cleanup_handoff_verified=True,
        full_managed_process_recovery_verified=False,
    )
    if owned_copies:
        result["owned_copy_limits_verified"] = True
    return result
