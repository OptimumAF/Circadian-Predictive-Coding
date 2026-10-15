"""Original elapsed/RSS refusal across two real coupled candidate handoffs.

Elapsed seconds and RSS bytes are explicit injected ports, not host measurements
or model-memory attribution. Full native/reference observations and cumulative
sampler/budget records expose work and refusal boundaries. Why allow telemetry
advancement: existing checkpoint checks legitimately sample resources; all other
configuration, state and work must remain exact. No new controller, algorithm,
physical resource guarantee, process recovery or scientific benefit is provided.
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
    _queue,
    describe_example,
)
from src.adapters.coupled_repeated_handoff import _observation, _metadata
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, _budget_state
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.experience import LogicalClock
from src.core.resource_sharing import SharingLimits
from src.shared.process_memory import ProcessRssSampler

_STOPS = ("wall", "rss")


def describe_resource_stop_example() -> dict[str, Any]:
    description = describe_example()
    description.update(
        protocol_id="coupled_handoff_resources_engineering_v1",
        stop_modes=list(_STOPS),
        injected_clock=True,
        injected_rss=True,
        resource_sources=dict(clock="injected_logical_clock_seconds", rss="injected_reader_bytes"),
        label_arrivals=[3, 7, 9],
        runtime_update_limit=3,
        sharing_limits=asdict(SharingLimits(2, 3, 1)),
        preparation_attempt_limit=3,
        handoff_cycles=2,
        simulated_wall_limits=dict(wall=8.0, rss=100.0),
        injected_rss_start_bytes=1024,
        injected_rss_pressure_bytes=4096,
        injected_rss_limit_bytes=2048,
        native_training_calls=72,
        native_energy_evaluation_cap=360,
        native_constructor_cap=128,
        prediction_cap=128,
        guarded_inference_cap=36,
        limitations="Original resource-port continuity through two same-process transfers. Clock and RSS are injected controls, not host timing, continuous memory measurements or model attribution. No quota renewal, process recovery, long-run or scientific benefit claim.",
    )
    return description


def _observe_resources(runtime, gate, sampler: ProcessRssSampler) -> dict[str, Any]:
    observed = _observation(runtime, gate)
    observed["sampler"] = _plain(asdict(sampler.snapshot()))
    return observed


def _rss_progress(before: dict[str, Any], after: dict[str, Any]) -> None:
    _require(
        all(after[name] == before[name] for name in ("pid", "start_bytes", "interval_seconds"))
        and after["peak_bytes"] >= before["peak_bytes"]
        and after["sample_count"] >= before["sample_count"],
        "handoff reset or replaced original sampler lineage/high water",
    )


def _preserved(policy, payload: bytes, before, after, *, delivery_tick: int | None = None) -> None:
    _require(
        _payload(policy, after["native"]) == payload,
        "resource refusal/handoff changed complete native candidate",
    )
    old, new = _plain(_metadata(before)), _plain(_metadata(after))
    _rss_progress(old.pop("sampler"), new.pop("sampler"))
    before_budget, after_budget = old["budget"], new["budget"]
    _require(
        after_budget.pop("last_clock") >= before_budget.pop("last_clock"),
        "handoff original wall-clock observation regressed",
    )
    _rss_progress(before_budget.pop("process_rss_segment"), after_budget.pop("process_rss_segment"))
    if delivery_tick is not None:
        # A real refused delivery advances chronology and spends its sharing
        # admission, even though before_update prevents any native mutation.
        old["cursor"]["last_tick"] = delivery_tick
        old["consolidation_cursor"]["revision"] += 1
        old["sharing"]["admitted_updates"] += 1
    _require(
        _canonical(old) == _canonical(new),
        "handoff/resource refusal changed fields beyond cumulative telemetry/admission",
    )


def _stop(action: Callable[[], Any], reason: str) -> dict[str, Any]:
    try:
        action()
    except ToyExecutionStopped as error:
        _require(
            error.stop.reason == reason
            and error.stop.updates_completed == 2
            and error.stop.elapsed_seconds == 9.0,
            "resource stop reason/work/elapsed differs",
        )
        return _plain(asdict(error.stop))
    raise AssertionError("original handoff resource ceiling did not stop work")


def _case(stop_mode: str, family: str, mode: str) -> dict[str, Any]:
    # Retain the single original resource/controller lifetimes beside their
    # publication and refusal boundaries; helpers isolate comparison/telemetry.
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
    wall = lambda: float(clock.now())
    pressure = [1024]
    reader = lambda: pressure[0]
    sampler = ProcessRssSampler(read_rss_bytes=reader)
    sampler.sample()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=3,
            max_wall_seconds=8.0 if stop_mode == "wall" else 100.0,
            max_process_rss_bytes=2048,
        ),
        wall,
    )
    budget.attach_memory(sampler)
    initial_sampler = _plain(asdict(sampler.snapshot()))
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
    actor_a = _payload(policy, actor.snapshot_state().state)
    prediction_a = shared.predict(features.copy()).prediction.tolist()
    _queue(initial, features, targets)
    builds: list[str] = []

    def build(state):
        _require(
            shared.predict(features.copy()).prediction.tolist() == prediction_a
            and _payload(policy, actor.snapshot_state().state) == actor_a,
            "original actor changed or unavailable during resource handoff builder",
        )
        builds.append("correct_policy")
        return callbacks.build(state)

    control = CandidateCheckpointController(
        shared,
        build_learner=build,
        state_digest=callbacks.state_digest,
        policy_digest=callbacks.policy_digest,
        max_pending=1,
        max_prepare_attempts=3,
    )

    def identity() -> dict[str, Any]:
        current = shared._runtime
        _require(
            current.actor is actor
            and current._budget is budget
            and current._inbox._clock is clock
            and budget.clock is wall
            and budget.process_rss_sampler is sampler
            and sampler.read_rss_bytes is reader
            and shared._sharing is gate
            and budget.started_at == 0.0,
            "resource handoff replaced an original resource object or absolute start",
        )
        return dict(
            original=dict(
                actor=id(actor),
                budget=id(budget),
                clock=id(clock),
                wall_clock=id(wall),
                sampler=id(sampler),
                rss_reader=id(reader),
                gate=id(gate),
            ),
            current=dict(
                actor=id(current.actor),
                budget=id(current._budget),
                clock=id(current._inbox._clock),
                wall_clock=id(budget.clock),
                sampler=id(budget.process_rss_sampler),
                rss_reader=id(sampler.read_rss_bytes),
                gate=id(shared._sharing),
            ),
        )

    cycles: list[dict[str, Any]] = []
    payload_c = b""
    prediction_c: list[list[float]] = []
    for number, tick in enumerate((3, 7), 1):
        clock.advance_to(tick)
        old = shared._runtime
        poll = shared.train_ready()
        diagnostic = reference.train_batch(features.copy(), targets.copy())
        payload = _payload(policy, old.candidate_snapshot().state)
        prediction_c = old._candidate.predict(features.copy()).tolist()
        _require(
            [item.sample_id for item in poll.updates] == [f"s{number - 1}"]
            and poll.updates[0].diagnostic == diagnostic
            and payload == _payload(policy, reference.snapshot_state())
            and prediction_c == reference.predict(features.copy()).tolist()
            and callbacks.policy_digest(old._candidate) == callbacks.policy_digest(reference),
            "resource handoff native continuation differs from independent complete reference",
        )
        gate.pause()
        before = _observe_resources(old, gate, sampler)
        token = control.capture()
        view = control.inspect(token)
        new = control.restore(token)
        after = _observe_resources(new, gate, sampler)
        _preserved(policy, payload, before, after)
        identity()
        _require(
            shared._runtime is new
            and new is not old
            and old._retired
            and gate.snapshot().paused
            and control._attempts == len(control._models) == len(builds) == number
            and len(control._pending) == 0,
            "resource handoff reset owner/attempt/model accounting",
        )
        retired = _refused(old.train_ready, "candidate owner is retired after checkpoint handoff")
        used = _refused(
            lambda: control.restore(token),
            "checkpoint is foreign, copied, discarded or already used",
        )
        _require(
            control._attempts == len(control._models) == len(builds) == number,
            "retired/used resource checkpoint reached another builder",
        )
        gate.resume()
        empty = shared.train_ready()
        _require(
            empty.updates == () and budget.updates_completed == number,
            "resource handoff trained future labels at the same tick",
        )
        cycles.append(
            dict(
                cycle=number,
                tick=tick,
                updates=[asdict(item) for item in poll.updates],
                direct_diagnostic=asdict(diagnostic),
                payload_utf8=payload.decode(),
                captured_cursor=_plain(asdict(view.inbox)),
                before=_metadata(before),
                after_handoff=_metadata(after),
                preparation_attempts=control._attempts,
                prepared_models=len(control._models),
                pending_checkpoints=len(control._pending),
                retired_owner_refusal=retired,
                used_token_refusal=used,
                same_tick_updates=list(empty.updates),
            )
        )
        payload_c = payload

    current = shared._runtime
    gate.pause()
    token = control.capture()
    before_prepare = _observe_resources(current, gate, sampler)
    clock.advance_to(9)
    if stop_mode == "rss":
        pressure[0] = 4096
    reason = "max_wall_seconds" if stop_mode == "wall" else "max_process_rss_bytes"
    preparation_error = _stop(lambda: control.restore(token), reason)
    after_prepare = _observe_resources(current, gate, sampler)
    _preserved(policy, payload_c, before_prepare, after_prepare)
    _require(
        shared._runtime is current
        and not current._retired
        and not current._stopped
        and control._attempts == 3
        and len(control._models) == len(builds) == 2
        and len(control._pending) == 1,
        "resource stop did not spend preparation attempt before builder",
    )
    control.discard(token)
    _require(len(control._pending) == 0, "resource-stopped checkpoint was not discarded")
    preparation_stop = dict(
        before=_metadata(before_prepare),
        after=_metadata(after_prepare),
        stop=preparation_error,
        attempts=control._attempts,
        models=len(control._models),
        pending_before_discard=1,
        pending_after_discard=len(control._pending),
    )
    gate.resume()
    pressure[0] = 1024
    before_update = _observe_resources(current, gate, sampler)
    update_error = _stop(shared.train_ready, reason)
    after_update = _observe_resources(current, gate, sampler)
    _preserved(policy, payload_c, before_update, after_update, delivery_tick=9)
    quota = shared.train_ready()
    _require(
        quota.updates == ()
        and quota.deferred_reason == "work_quota"
        and gate.snapshot().admitted_updates == 3
        and budget.updates_completed == 2
        and not current._stopped
        and not current._inbox.stopped,
        "resource-refused update was retried/refunded or mislabeled uncertain mutation",
    )
    final_cursor = current._inbox.capture_cursor()
    _require(
        [item.sample_id for item in final_cursor.experiences] == ["s0", "s1", "s2"]
        and [item.sample_id for item in final_cursor.applied] == ["s0", "s1"]
        and [item.update_number for item in final_cursor.applied] == [1, 2]
        and [item.applied_at for item in final_cursor.applied] == [3, 7]
        and final_cursor.last_tick == 9
        and not final_cursor.stopped,
        "resource stop lost original pending pair or causal receipt history",
    )
    _require(
        _payload(policy, current.candidate_snapshot().state) == payload_c
        and _payload(policy, actor.snapshot_state().state) == actor_a
        and shared.predict(features.copy()).prediction.tolist() == prediction_a,
        "resource stops changed complete candidate or original actor",
    )
    final_sampler = _plain(asdict(sampler.snapshot()))
    _rss_progress(initial_sampler, final_sampler)
    _require(
        final_sampler["start_bytes"] == 1024
        and final_sampler["peak_bytes"] == (4096 if stop_mode == "rss" else 1024)
        and pressure[0] == 1024,
        "original resource peak was reset after pressure fell",
    )
    for expected_wakes, payload in enumerate(
        (actor_a, *(item["payload_utf8"].encode() for item in cycles)), 2
    ):
        wake = json.loads(payload)["native"]["wake_state"]
        _require(
            wake is None if family == "pc" else wake["wake_updates"] == expected_wakes,
            "resource handoff lost complete genuine CPC wake history",
        )
    row = dict(
        stop_mode=stop_mode,
        family=family,
        mode=mode,
        policy=asdict(policy),
        injected_clock=True,
        injected_rss=True,
        actor_a_utf8=actor_a.decode(),
        final_c_utf8=payload_c.decode(),
        prediction_a=prediction_a,
        prediction_c=prediction_c,
        cycles=cycles,
        preparation_stop=preparation_stop,
        update_stop=dict(
            before=_metadata(before_update), after=_metadata(after_update), stop=update_error
        ),
        final_cursor=_plain(asdict(final_cursor)),
        final_budget=_plain(_budget_state(budget)),
        final_sharing=_plain(asdict(gate.snapshot())),
        resource_identity=identity(),
        sampler_lineage=dict(initial=initial_sampler, final=final_sampler),
        final_reader_bytes=pressure[0],
        builds=builds,
        preparation_attempts=control._attempts,
        prepared_models=len(control._models),
        pending_checkpoints=len(control._pending),
        sharing_refusal=quota.deferred_reason,
        final_quota=_plain(asdict(quota)),
        actor_unchanged=True,
        same_actor_budget_clock_gate=True,
        same_original_resources=True,
        complete_continuation_exact=True,
    )
    _require(
        len(_canonical(row).encode("utf-8")) <= 65536,
        "resource handoff summary exceeds case byte cap",
    )
    return row


def run_resource_stop_example() -> dict[str, Any]:
    rows = [_case(stop, family, mode) for stop in _STOPS for mode in MODES for family in FAMILIES]
    for offset in (0, 3, 6, 9):
        for stage in range(3):
            states = []
            for number in (0, 1):
                row = rows[offset + number]
                payload = (
                    row["actor_a_utf8"] if stage == 0 else row["cycles"][stage - 1]["payload_utf8"]
                )
                states.append(json.loads(payload)["native"]["arrays"])
            _require(
                states[0] == states[1], "PC-neutral complete frames differ under resource controls"
            )
    return dict(
        protocol=describe_resource_stop_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        resource_stops_verified=True,
        full_managed_process_recovery_verified=False,
    )
