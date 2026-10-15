"""Live managed consent across a complete-policy coupled candidate handoff.

The fixed synthetic/unverified declarations are local engineering inputs, not
attested provenance or human consent. Existing managed admission owns registration
and eligibility; the existing controller retains those live guards through one
handoff. Why preserve raw opted-out records: this control demonstrates future-use
blocking, not deletion, unlearning or replay-cache erasure. No new authority,
lifecycle/process restoration, algorithm or scientific claim is introduced.
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
    describe_example,
)
from src.adapters.coupled_repeated_handoff import _observation, _metadata
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, _budget_state
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits

_SOURCE_ID = "local-fixed-engineering-fixture"
_SUBJECTS = ("subject-A", "subject-B", "subject-A")
_ARRIVALS = (3, 7, 9)


class _PoisonPayload:
    def __init__(self) -> None:
        self.copy_attempts = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        self.copy_attempts += 1
        raise AssertionError("refused managed payload must never be copied")


def _declaration(number: int, subject: str) -> LifecycleDeclaration:
    return LifecycleDeclaration(
        (f"e{number}", f"s{number}"),
        DataProvenance(_SOURCE_ID, subject, verified=False, synthetic=True),
        DataConsent(training=True, replay=True),
        "replay",
    )


def describe_managed_consent_example() -> dict[str, Any]:
    description = describe_example()
    description.update(
        protocol_id="coupled_managed_consent_engineering_v1",
        runtime_update_limit=3,
        sharing_limits=asdict(SharingLimits(2, 3, 1)),
        preparation_attempt_limit=1,
        label_arrivals=list(_ARRIVALS),
        opt_out_at=5,
        declarations=[
            asdict(_declaration(number, subject)) for number, subject in enumerate(_SUBJECTS)
        ],
        lifecycle_limits=asdict(LifecycleLimits(3, 3, allow_synthetic=True, allow_unverified=True)),
        initial_poison_probes=8,
        post_optout_poison_probes=2,
        native_training_calls=36,
        native_energy_evaluation_cap=180,
        native_constructor_cap=48,
        prediction_cap=56,
        guarded_inference_cap=18,
        limitations="Synthetic/unverified local declarations and live same-process consent guards, not provenance attestation or human consent. Opt-out blocks future use; raw records remain. No deletion/unlearning/replay erasure, quota renewal, managed lifecycle/process recovery or scientific benefit claim.",
    )
    return description


def _managed(owner) -> dict[str, Any]:
    return dict(
        declarations=_plain(owner.declarations),
        limits=asdict(owner._limits),
        opted_out=sorted(owner._opted_out),
        revoked_keys=_plain(sorted(owner._revoked_keys)),
        declaration_ticks=_plain(list(owner._declaration_ticks.items())),
        declaration_seconds=_plain(list(owner._declaration_seconds.items())),
        lifecycle_installed=owner._lifecycle is not None,
    )


def _preserved(policy, before, after, *, revision_delta: int = 0, opted_out: bool = False) -> None:
    _require(
        _payload(policy, before["native"]) == _payload(policy, after["native"]),
        "managed consent operation changed complete native state",
    )
    expected, actual = _plain(_metadata(before)), _plain(_metadata(after))
    expected["consolidation_cursor"]["revision"] += revision_delta
    if opted_out:
        expected["managed"]["opted_out"] = ["subject-A"]
    _require(
        _canonical(expected) == _canonical(actual),
        "managed consent changed fields beyond the declared opt-out/revision transition",
    )


def _case(family: str, mode: str) -> dict[str, Any]:
    # Keep the one live consent owner and its guards visible alongside the
    # stale-token and replacement-owner transactions; helpers handle observations.
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
    initial = ActorShadowRuntime(
        reference,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=3,
        max_consolidations=0,
    )
    actor = initial.actor
    gate = ServingPriorityGate(SharingLimits(2, 3, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(initial, gate)
    limits = LifecycleLimits(3, 3, allow_synthetic=True, allow_unverified=True)
    owner = ManagedExperienceOwner(shared, limits=limits)
    catalog, opted_out, revoked_keys = owner._catalog, owner._opted_out, owner._revoked_keys
    registration, eligibility = initial._inbox._registration_guard, initial._inbox._training_guard
    original = dict(
        owner=id(owner),
        catalog=id(owner._catalog),
        opted_out=id(owner._opted_out),
        revoked_keys=id(owner._revoked_keys),
        limits=id(limits),
        registration_guard=id(registration),
        eligibility_guard=id(eligibility),
        actor=id(actor),
        budget=id(budget),
        clock=id(clock),
        gate=id(gate),
    )
    for number, subject in enumerate(_SUBJECTS):
        owner.declare(_declaration(number, subject))
    managed_before = _managed(owner)
    actor_a = _payload(policy, actor.snapshot_state().state)
    prediction_a = shared.predict(features.copy()).prediction.tolist()
    poison = _PoisonPayload()
    refusals: list[dict[str, Any]] = []
    builds: list[str] = []

    def capture() -> dict[str, Any]:
        observed = _observation(shared._runtime, gate)
        observed.update(managed=_managed(owner), event_tick=clock.now())
        return observed

    def refuse(phase: str, action: Callable[[], Any], message: str, *, delta: int) -> None:
        build_count = len(builds)
        before = capture()
        result = _refused(action, message)
        after = capture()
        _preserved(policy, before, after, revision_delta=delta)
        _require(
            poison.copy_attempts == 0 and len(builds) == build_count,
            "managed refusal copied an opaque payload or reached a builder",
        )
        refusals.append(
            dict(
                phase=phase,
                message=result,
                revision_before=before["consolidation_cursor"]["revision"],
                revision_after=after["consolidation_cursor"]["revision"],
                revision_delta=delta,
                poison_copy_attempts=poison.copy_attempts,
                complete_state_preserved=True,
            )
        )

    def source(number=0, role="train"):
        permissions = (
            ExperiencePermissions(True, True)
            if role == "train"
            else ExperiencePermissions(evaluation=True)
        )
        return Experience(
            f"s{number}", f"e{number}", 0, "a0", cast(Array, poison), role, permissions
        )

    def label(number=0, role="train"):
        return LabelArrival(
            f"poison-l{number}",
            f"s{number}",
            f"e{number}",
            _ARRIVALS[number],
            "a0",
            cast(Array, poison),
            role,
        )

    for role in ("inner_guard", "outer_selection", "final_test"):
        refuse(
            role + "_source",
            lambda: owner.record_experience(source(role=role)),
            "source requires training and replay permission",
            delta=1,
        )
        refuse(
            role + "_label",
            lambda: owner.record_label(label(role=role)),
            "label requires train role",
            delta=1,
        )
    refuse(
        "unmanaged_source",
        lambda: initial.record_experience(source()),
        "registration requires the original managed admission owner",
        delta=1,
    )
    refuse(
        "unmanaged_label",
        lambda: initial.record_label(label()),
        "registration requires the original managed admission owner",
        delta=1,
    )
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
            "consent-eligible native continuation differs from independent complete reference",
        )
        return poll, payload, prediction

    clock.advance_to(3)
    first, payload_b, _ = exact_update(0)
    gate.pause()

    def build(state):
        _require(
            shared.predict(features.copy()).prediction.tolist() == prediction_a
            and _payload(policy, actor.snapshot_state().state) == actor_a,
            "original actor changed or unavailable during managed handoff builder",
        )
        builds.append("correct_policy")
        return callbacks.build(state)

    control = CandidateCheckpointController(
        shared,
        build_learner=build,
        state_digest=callbacks.state_digest,
        policy_digest=callbacks.policy_digest,
        max_pending=1,
        max_prepare_attempts=1,
    )
    stale = control.capture()
    clock.advance_to(5)
    before_opt_out = capture()
    owner.opt_out("subject-A")
    after_opt_out = capture()
    _preserved(policy, before_opt_out, after_opt_out, revision_delta=1, opted_out=True)
    stale_refusal = _refused(
        lambda: control.restore(stale),
        "checkpoint owner, work, builder or resource identity changed; stale checkpoint",
    )
    after_stale_refusal = capture()
    _preserved(policy, after_opt_out, after_stale_refusal)
    _require(
        shared._runtime is initial
        and not initial._retired
        and not builds
        and control._attempts == 0
        and not control._models
        and len(control._pending) == 1,
        "stale consent token reached preparation or changed its original live owner",
    )
    control.discard(stale)
    fresh = control.capture()
    before_handoff = capture()
    new = control.restore(fresh)
    after_handoff = capture()
    _preserved(policy, before_handoff, after_handoff)
    _require(
        shared._runtime is new
        and new is not initial
        and initial._retired
        and control._attempts == len(control._models) == len(builds) == 1
        and not control._pending,
        "fresh managed handoff lost original preparation accounting/owner retirement",
    )

    def identity() -> dict[str, Any]:
        current = shared._runtime
        _require(
            owner._shared is shared
            and owner._catalog is catalog
            and owner._opted_out is opted_out
            and owner._revoked_keys is revoked_keys
            and owner._limits is limits
            and owner._lifecycle is None
            and owner._issuing is None
            and current.actor is actor
            and current._budget is budget
            and current._inbox._clock is clock
            and shared._sharing is gate
            and current._inbox._registration_guard is registration
            and current._inbox._training_guard is eligibility
            and getattr(registration, "__self__", None) is owner
            and getattr(eligibility, "__self__", None) is owner,
            "managed handoff replaced original authority/catalog/sets/limits/guards/resources",
        )
        return dict(
            original=original,
            current=dict(
                owner=id(owner),
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
            ),
        )

    identities = identity()
    retired_refusal = _refused(
        initial.train_ready, "candidate owner is retired after checkpoint handoff"
    )
    used_refusal = _refused(
        lambda: control.restore(fresh), "checkpoint is foreign, copied, discarded or already used"
    )
    _require(
        control._attempts == len(control._models) == len(builds) == 1,
        "retired owner/used managed token reached another builder",
    )
    gate.resume()
    waiting = owner.train_ready()
    _require(
        waiting.updates == () and waiting.deferred_reason is None and budget.updates_completed == 1,
        "managed handoff trained future labels at5",
    )
    clock.advance_to(7)
    second, payload_c, prediction_c = exact_update(1)
    clock.advance_to(9)
    ineligible = owner.train_ready()
    pending_source = new._inbox._experiences[("e2", "s2")]
    pending_label = new._inbox._labels[("e2", "s2")]
    if eligibility is None:
        raise AssertionError("original managed eligibility guard was lost")
    eligibility_s2 = eligibility(pending_source, pending_label)
    _require(
        ineligible.updates == ()
        and ineligible.deferred_reason is None
        and not eligibility_s2
        and budget.updates_completed == gate.snapshot().admitted_updates == 2
        and budget.budget.max_training_updates == gate.snapshot().limits.max_admitted_updates == 3
        and not new._stopped
        and not new._inbox.stopped,
        "opt-out filtering was replaced with quota exhaustion or uncertain mutation",
    )
    refuse(
        "post_optout_source",
        lambda: owner.record_experience(source(2)),
        "subject has opted out",
        delta=1,
    )
    refuse(
        "post_optout_label", lambda: owner.record_label(label(2)), "subject has opted out", delta=1
    )
    refuse(
        "regrant_subject_a",
        lambda: owner.declare(_declaration(3, "subject-A")),
        "subject has opted out",
        delta=0,
    )
    refuse(
        "fourth_declaration",
        lambda: owner.declare(_declaration(3, "subject-C")),
        "lifetime approved or replay record quota exhausted",
        delta=0,
    )
    final_cursor = new._inbox.capture_cursor()
    _require(
        [item.sample_id for item in final_cursor.experiences] == ["s0", "s1", "s2"]
        and len(final_cursor.labels) == 3
        and [item.sample_id for item in final_cursor.applied] == ["s0", "s1"]
        and [item.update_number for item in final_cursor.applied] == [1, 2]
        and [item.applied_at for item in final_cursor.applied] == [3, 7]
        and final_cursor.completed_updates == 2
        and final_cursor.last_tick == 9
        and not final_cursor.stopped
        and len(owner.declarations) == 3
        and owner._opted_out == {"subject-A"}
        and poison.copy_attempts == 0
        and builds == ["correct_policy"],
        "managed handoff lost raw history, original catalog or permanent opt-out",
    )
    _require(
        _payload(policy, new.candidate_snapshot().state) == payload_c
        and _payload(policy, actor.snapshot_state().state) == actor_a
        and shared.predict(features.copy()).prediction.tolist() == prediction_a,
        "managed consent refusals changed complete candidate or original actor",
    )
    for count, payload in ((2, actor_a), (3, payload_b), (4, payload_c)):
        wake = json.loads(payload)["native"]["wake_state"]
        _require(
            wake is None if family == "pc" else wake["wake_updates"] == count,
            "managed handoff lost genuine CPC chemistry/gates/wake history",
        )
    _require(
        identity() == identities, "managed authority identity changed after continuation/refusals"
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
        declarations=managed_before["declarations"],
        managed_before=managed_before,
        managed_after=_managed(owner),
        before_opt_out=_metadata(before_opt_out),
        after_opt_out=_metadata(after_opt_out),
        after_stale_refusal=_metadata(after_stale_refusal),
        before_handoff=_metadata(before_handoff),
        after_handoff=_metadata(after_handoff),
        final_cursor=_plain(asdict(final_cursor)),
        final_budget=_plain(_budget_state(budget)),
        final_sharing=_plain(asdict(gate.snapshot())),
        first_updates=[asdict(item) for item in first.updates],
        second_updates=[asdict(item) for item in second.updates],
        waiting_at5=_plain(asdict(waiting)),
        ineligible_at9=_plain(asdict(ineligible)),
        refusals=refusals,
        stale_refusal=stale_refusal,
        retired_owner_refusal=retired_refusal,
        used_token_refusal=used_refusal,
        builds=builds,
        preparation_attempts=control._attempts,
        prepared_models=len(control._models),
        pending_checkpoints=len(control._pending),
        poison_copy_attempts=poison.copy_attempts,
        identity=identities,
        eligibility_s2=eligibility_s2,
        pending_payloads_retained=True,
        actor_unchanged=True,
        complete_continuation_exact=True,
        same_managed_authority=True,
        same_actor_budget_clock_gate=True,
    )
    _require(
        len(_canonical(row).encode("utf-8")) <= 65536,
        "managed consent case exceeds bounded JSON allowance",
    )
    return row


def run_managed_consent_example() -> dict[str, Any]:
    rows = [_case(family, mode) for mode in MODES for family in FAMILIES]
    for offset in (0, 3):
        for stage in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            pc, neutral = (
                json.loads(rows[offset + number][stage])["native"]["arrays"] for number in (0, 1)
            )
            _require(
                pc == neutral, "PC-neutral complete parameter frames differ under managed consent"
            )
    return dict(
        protocol=describe_managed_consent_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        managed_consent_handoff_verified=True,
        full_managed_process_recovery_verified=False,
    )
