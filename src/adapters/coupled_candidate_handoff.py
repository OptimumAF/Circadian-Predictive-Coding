"""Fixed policy-complete candidate handoff under original runtime resources.

Inputs are fixed train-only data and existing controller/native ports. Outputs
retain full native envelopes and managed cursor/budget/actor observations. The
controller performs same-process preparation/publication; no process recovery,
disk authority, new algorithm or comparative learning claim is introduced.
"""

from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json
import os
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_checkpoint_interruption import _model
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, _budget_state
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_learner_state import CoupledLearnerCheckpoint
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits

BINDING = CodecBinding(
    sha256(b"coupled candidate handoff engineering source v1").hexdigest(),
    sha256(b"coupled candidate handoff engineering policy v1").hexdigest(),
)
LIMITS = CodecLimits(65536, 8192, 4, 16)
FAMILIES = ("pc", "neutral_cpc", "active_cpc")
MODES = ("fixed", "guarded")


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_candidate_handoff_engineering_v1",
        families=list(FAMILIES),
        inference_modes=list(MODES),
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        training=dict(
            features=[[0.2, -0.3], [-0.4, 0.6]],
            targets=[[1.0], [0.0]],
            role="train",
            learning_rate=0.025,
            inference_steps=3,
            inference_learning_rate=0.07,
            guarded_policy=dict(
                min_steps=1, gradient_tolerance=1e6, relative_energy_tolerance=None
            ),
        ),
        label_arrivals=[3, 7, 9],
        runtime_update_limit=2,
        sharing_limits=asdict(SharingLimits(2, 2, 1)),
        preparation_attempt_limit=2,
        native_training_calls=36,
        native_energy_evaluation_cap=180,
        native_constructor_cap=80,
        prediction_cap=64,
        process_launches=0,
        binding=asdict(BINDING),
        limits=asdict(LIMITS),
        limitations="Same-process candidate handoff under a surviving original actor, budget, clock and sharing gate. No process restart, lifecycle/erasure authority restoration, quota renewal, performance or scientific benefit claim.",
    )


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _plain(value):
    def project(item):
        if type(item) is np.ndarray:
            return item.tolist()
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        raise TypeError("unsupported trusted handoff observation")

    return json.loads(json.dumps(value, default=project, allow_nan=False))


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _refused(action, text):
    try:
        action()
    except ValueError as error:
        _require(text == str(error), "candidate handoff refusal reason differs")
        return str(error)
    raise AssertionError("candidate handoff accepted an unsupported operation")


def _payload(policy, state):
    return CoupledLearnerCheckpointCodec().encode(
        CoupledLearnerCheckpoint(1, policy, state), binding=BINDING, limits=LIMITS
    )


def _queue(runtime, x, y):
    for number, arrival in enumerate((3, 7, 9)):
        sample, episode = f"s{number}", f"e{number}"
        runtime.record_experience(
            Experience(
                sample, episode, 0, "a0", x.copy(), "train", ExperiencePermissions(training=True)
            )
        )
        runtime.record_label(LabelArrival(f"l{number}", sample, episode, arrival, "a0", y.copy()))


def _observe(runtime, gate):
    return dict(
        native=runtime.candidate_snapshot().state,
        cursor=_plain(asdict(runtime._inbox.capture_cursor())),
        budget=_plain(_budget_state(runtime._budget)),
        sharing=_plain(asdict(gate.snapshot())),
    )


def _case(family, mode):
    # Keep the original resource objects and the single publication transaction
    # visible together; queueing, byte encoding and observations use helpers.
    x = np.array([[0.2, -0.3], [-0.4, 0.6]], dtype=np.float64)
    y = np.array([[1.0], [0.0]], dtype=np.float64)
    inference = CoupledInferencePolicy(1, 1e6, None) if mode == "guarded" else None
    reference = CoupledLearner(
        _model(family),
        learning_rate=0.025,
        inference_steps=3,
        inference_learning_rate=0.07,
        inference_policy=inference,
    )
    for _ in range(2):
        reference.train_batch(x.copy(), y.copy())
    policy = reference.checkpoint_policy()
    callbacks = CoupledCheckpointCallbacks(policy, binding=BINDING, limits=LIMITS)
    wrong = CoupledCheckpointCallbacks(
        replace(policy, learning_rate=0.05), binding=BINDING, limits=LIMITS
    )
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    old = ActorShadowRuntime(
        reference,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=3,
        max_consolidations=0,
    )
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(old, gate)
    actor_a = _payload(policy, old.actor.snapshot_state().state)
    prediction_a = shared.predict(x.copy()).prediction.tolist()
    _queue(old, x, y)
    clock.advance_to(3)
    first = shared.train_ready()
    direct_b = reference.train_batch(x.copy(), y.copy())
    before = _observe(old, gate)
    payload_b = _payload(policy, before["native"])
    _require(
        [u.sample_id for u in first.updates] == ["s0"]
        and first.updates[0].diagnostic == direct_b
        and payload_b == _payload(policy, reference.snapshot_state()),
        "original candidate B differs from independent reference",
    )
    gate.pause()
    before = _observe(old, gate)
    builds: list[str] = []

    def build(state):
        _require(
            shared.predict(x.copy()).prediction.tolist() == prediction_a
            and _payload(policy, old.actor.snapshot_state().state) == actor_a,
            "actor unavailable or changed during candidate preparation",
        )
        selected = wrong if not builds else callbacks
        builds.append("wrong_policy" if not builds else "correct_policy")
        return selected.build(state)

    control = CandidateCheckpointController(
        shared,
        build_learner=build,
        state_digest=callbacks.state_digest,
        policy_digest=callbacks.policy_digest,
        max_pending=1,
        max_prepare_attempts=2,
    )
    token = control.capture()
    view = control.inspect(token)
    wrong_refusal = _refused(
        lambda: control.restore(token),
        "coupled learner policy differs from the original callback policy",
    )
    after_refusal = _observe(old, gate)
    _require(
        shared._runtime is old
        and not old._retired
        and control._attempts == 1
        and _payload(policy, after_refusal["native"]) == payload_b
        and _canonical({k: v for k, v in after_refusal.items() if k != "native"})
        == _canonical({k: v for k, v in before.items() if k != "native"}),
        "wrong policy preparation changed current owner/state/resources",
    )
    new = control.restore(token)
    after_handoff = _observe(new, gate)
    _require(
        shared._runtime is new
        and new is not old
        and old._retired
        and new.actor is old.actor
        and new._budget is budget
        and new._inbox._clock is clock
        and shared._sharing is gate
        and gate.snapshot().paused
        and control._attempts == 2
        and builds == ["wrong_policy", "correct_policy"]
        and _payload(policy, after_handoff["native"]) == payload_b
        and _canonical({k: v for k, v in after_handoff.items() if k != "native"})
        == _canonical({k: v for k, v in before.items() if k != "native"}),
        "candidate handoff lost original owner bindings/cursor/budget/sharing",
    )
    retired_refusal = _refused(
        old.train_ready, "candidate owner is retired after checkpoint handoff"
    )
    token_refusal = _refused(
        lambda: control.restore(token), "checkpoint is foreign, copied, discarded or already used"
    )
    _require(len(builds) == 2, "retired/used checkpoint executed another native builder")
    gate.resume()
    _require(
        shared.train_ready().updates == () and budget.updates_completed == 1,
        "future label trained before arrival or renewed resources",
    )
    clock.advance_to(7)
    second = shared.train_ready()
    direct_c = reference.train_batch(x.copy(), y.copy())
    payload_c = _payload(policy, new.candidate_snapshot().state)
    _require(
        [u.sample_id for u in second.updates] == ["s1"]
        and second.updates[0].diagnostic == direct_c
        and payload_c == _payload(policy, reference.snapshot_state())
        and [u.update_number for u in new.applied_updates] == [1, 2],
        "replacement candidate C/history differs from original reference",
    )
    prediction_c = new._candidate.predict(x.copy()).tolist()
    _require(prediction_c == reference.predict(x.copy()).tolist(), "continued prediction differs")
    clock.advance_to(9)
    quota = shared.train_ready()
    _require(
        quota.updates == ()
        and quota.deferred_reason == "work_quota"
        and gate.snapshot().admitted_updates == budget.updates_completed == 2,
        "handoff renewed original sharing or update allowance",
    )
    try:
        budget.before_update()
    except ToyExecutionStopped as error:
        _require(error.stop.reason == "max_training_updates", "original budget refusal differs")
        budget_refusal = error.stop.reason
    else:
        raise AssertionError("original runtime update quota was renewed")
    _require(
        _payload(policy, new.actor.snapshot_state().state) == actor_a
        and shared.predict(x.copy()).prediction.tolist() == prediction_a
        and _payload(policy, new.candidate_snapshot().state) == payload_c,
        "quota refusal changed candidate or serving actor",
    )
    for expected_wakes, payload in ((2, actor_a), (3, payload_b), (4, payload_c)):
        wake = json.loads(payload)["native"]["wake_state"]
        _require(
            wake is None if family == "pc" else wake["wake_updates"] == expected_wakes,
            "handoff lost complete CPC wake history",
        )
    row = dict(
        family=family,
        mode=mode,
        policy=asdict(policy),
        actor_a_utf8=actor_a.decode(),
        captured_b_utf8=payload_b.decode(),
        continued_c_utf8=payload_c.decode(),
        captured_cursor=_plain(asdict(view.inbox)),
        before={k: v for k, v in before.items() if k != "native"},
        after_refusal={k: v for k, v in after_refusal.items() if k != "native"},
        after_handoff={k: v for k, v in after_handoff.items() if k != "native"},
        final_cursor=_plain(asdict(new._inbox.capture_cursor())),
        final_budget=_plain(_budget_state(budget)),
        final_sharing=asdict(gate.snapshot()),
        first_updates=[asdict(u) for u in first.updates],
        second_updates=[asdict(u) for u in second.updates],
        prediction_a=prediction_a,
        prediction_c=prediction_c,
        refusals=dict(
            wrong_policy=wrong_refusal,
            retired_owner=retired_refusal,
            used_token=token_refusal,
            sharing=quota.deferred_reason,
            budget=budget_refusal,
        ),
        builds=builds,
        preparation_attempts=control._attempts,
        same_actor_budget_clock_gate=True,
        actor_unchanged=True,
        complete_continuation_exact=True,
    )
    _require(
        len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 32768,
        "candidate handoff summary exceeds fixed case byte cap",
    )
    return row


def run_example() -> dict[str, Any]:
    rows = [_case(family, mode) for mode in MODES for family in FAMILIES]
    fields = ("hidden_weights", "hidden_biases", "output_weight", "output_bias")
    for offset in (0, 3):
        for stage in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            pc, neutral = (json.loads(rows[offset + n][stage])["native"]["arrays"] for n in (0, 1))
            _require(
                all(pc[field] == neutral[field] for field in fields),
                "PC-neutral full parameter frames differ under matched handoff",
            )
    return dict(
        protocol=describe_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        candidate_handoff_verified=True,
        full_managed_process_recovery_verified=False,
    )
