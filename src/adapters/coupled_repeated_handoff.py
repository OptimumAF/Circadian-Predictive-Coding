"""Fixed candidate transfers using one controller and original runtime resources.

The fixed train-only fixture compares each real continuation with an independent
native reference. Full envelopes and cursor/resource observations expose each
transfer and bounded preparation history. Why reuse one controller: rebuilding
it would renew preparation attempts and conceal retained model growth. This short
control provides no new learner, process recovery, scientific or long-run claim.
"""

from dataclasses import asdict, replace
import json
import os
from time import monotonic
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
    _observe,
    describe_example,
)
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController, _budget_state
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits
from src.shared.process_memory import ProcessRssSampler, read_process_rss_bytes

_ARRIVALS = (3, 7, 9, 11)
_NATIVE_MONOTONIC = monotonic
_NATIVE_RSS_READER = read_process_rss_bytes


def describe_repeated_example() -> dict[str, Any]:
    description = describe_example()
    description.update(
        protocol_id="coupled_repeated_handoff_engineering_v1",
        label_arrivals=list(_ARRIVALS),
        runtime_update_limit=3,
        sharing_limits=asdict(SharingLimits(2, 3, 1)),
        preparation_attempt_limit=4,
        handoff_cycles=3,
        native_training_calls=48,
        native_energy_evaluation_cap=240,
        native_constructor_cap=96,
        prediction_cap=96,
        guarded_inference_cap=24,
        limitations="Three same-process transfers under one controller and original actor/budget/clock/sharing resources. No process restart, quota renewal, long-run memory, lifecycle/erasure recovery or scientific benefit claim.",
    )
    return description


def describe_endurance_example() -> dict[str, Any]:
    description = describe_repeated_example()
    description.update(
        protocol_id="coupled_handoff_endurance_engineering_v1",
        horizon=16,
        handoff_cycles=16,
        label_arrivals=[3, *(7 + 2 * number for number in range(16))],
        runtime_update_limit=16,
        sharing_limits=asdict(SharingLimits(2, 16, 1)),
        preparation_attempt_limit=17,
        native_training_calls=204,
        native_energy_evaluation_cap=1020,
        native_constructor_cap=256,
        prediction_cap=340,
        guarded_inference_cap=102,
        expected_native_energy_evaluations=816,
        expected_native_constructors=228,
        expected_native_predictions=306,
        resource_limits=dict(
            max_training_updates=16, max_wall_seconds=45.0, max_process_rss_bytes=512 * 1024**2
        ),
        resource_sources=_resource_sources(),
        limitations="Sixteen same-process transfers with original controller and sampled real monotonic/OS RSS resources. Samples are boundary observations, not continuous/hard memory enforcement or model attribution. No indefinite stability, full managed/process/lifecycle recovery or scientific benefit claim.",
    )
    return description


def _queue(runtime, features, targets, *, arrivals=_ARRIVALS) -> None:
    for number, arrival in enumerate(arrivals):
        sample, episode = f"s{number}", f"e{number}"
        runtime.record_experience(
            Experience(
                sample,
                episode,
                0,
                "a0",
                features.copy(),
                "train",
                ExperiencePermissions(training=True),
            )
        )
        runtime.record_label(
            LabelArrival(f"l{number}", sample, episode, arrival, "a0", targets.copy())
        )


def _observation(runtime, gate, *, sampler: ProcessRssSampler | None = None) -> dict[str, Any]:
    observed = _observe(runtime, gate)
    observed["consolidation_cursor"] = _plain(asdict(runtime.capture_consolidation_cursor()))
    if sampler is not None:
        observed["sampler"] = _plain(asdict(sampler.snapshot()))
    return observed


def _metadata(observed: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in observed.items() if key != "native"}


def _unchanged(policy, expected_payload: bytes, before, after, message: str) -> None:
    previous, current = _plain(_metadata(before)), _plain(_metadata(after))
    if "sampler" in previous or "sampler" in current:
        _sample_progress(previous.pop("sampler"), current.pop("sampler"))
        before_budget, after_budget = previous["budget"], current["budget"]
        before_clock, after_clock = before_budget.pop("last_clock"), after_budget.pop("last_clock")
        _require(
            type(after_clock) is type(before_clock) and after_clock >= before_clock,
            "handoff wall-clock high water regressed",
        )
        _sample_progress(
            before_budget.pop("process_rss_segment"), after_budget.pop("process_rss_segment")
        )
    _require(
        _payload(policy, after["native"]) == expected_payload
        and _canonical(current) == _canonical(previous),
        message,
    )


def _sample_progress(before, after) -> None:
    fields = {"pid", "start_bytes", "peak_bytes", "sample_count", "interval_seconds"}
    _require(
        type(before) is dict and type(after) is dict and before.keys() == after.keys() == fields,
        "handoff sampler observation differs from its complete schema",
    )
    _require(
        all(
            type(after[name]) is type(before[name]) and after[name] == before[name]
            for name in ("pid", "start_bytes", "interval_seconds")
        )
        and type(after["peak_bytes"]) is int
        and type(before["peak_bytes"]) is int
        and type(after["sample_count"]) is int
        and type(before["sample_count"]) is int
        and after["peak_bytes"] >= before["peak_bytes"]
        and after["sample_count"] >= before["sample_count"],
        "handoff original sampler lineage/high water regressed",
    )


def _read_live_rss(source: Callable[[], int | None]) -> int:
    value = source()
    if type(value) is not int or not 0 < value < 2**63:
        raise RuntimeError("handoff endurance requires a positive native OS RSS observation")
    return value


def _resource_setup(
    handoff_cycles: int,
    live: bool,
    *,
    wall_source: Callable[[], float] | None = None,
    rss_source: Callable[[], int | None] | None = None,
) -> tuple[LogicalClock, ToyBudgetSession, ProcessRssSampler | None]:
    if (
        type(handoff_cycles) is not int
        or type(live) is not bool
        or (handoff_cycles, live) not in ((3, False), (16, True))
    ):
        raise ValueError("handoff resources require ordinary3 or live16 control")
    if (wall_source is not None and not callable(wall_source)) or (
        rss_source is not None and not callable(rss_source)
    ):
        raise ValueError("handoff resource sources must be trusted callables")
    clock = LogicalClock()
    if not live:
        return (
            clock,
            ToyBudgetSession(ToyExecutionBudget(max_training_updates=handoff_cycles), lambda: 0.0),
            None,
        )
    raw_wall = monotonic if wall_source is None else wall_source
    raw_reader = read_process_rss_bytes if rss_source is None else rss_source

    def reader() -> int:
        return _read_live_rss(raw_reader)

    sampler = ProcessRssSampler(read_rss_bytes=reader)
    sampler.sample()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=handoff_cycles,
            max_wall_seconds=45.0,
            max_process_rss_bytes=512 * 1024**2,
        ),
        raw_wall,
    )
    budget.attach_memory(sampler)
    budget.before_final()
    return clock, budget, sampler


def _resource_sources(wall_source=None, rss_source=None) -> dict[str, Any]:
    injected_clock = (monotonic if wall_source is None else wall_source) is not _NATIVE_MONOTONIC
    injected_rss = (
        read_process_rss_bytes if rss_source is None else rss_source
    ) is not _NATIVE_RSS_READER
    # This identifies local test injection, not authenticity of OS observations.
    return dict(
        clock="test_injected_clock" if injected_clock else "time.monotonic",
        rss="test_injected_rss" if injected_rss else "read_process_rss_bytes",
        injected_clock=injected_clock,
        injected_rss=injected_rss,
    )


def _resource_record(observed) -> dict[str, Any]:
    return {name: observed[name] for name in ("budget", "sampler")}


def _compact_cycle(record: dict[str, Any]) -> dict[str, Any]:
    compact = {
        name: value
        for name, value in record.items()
        if name not in ("captured_cursor", "before", "after_refusal", "after_handoff")
    }
    compact["resources"] = dict(
        before=_resource_record(record["before"]),
        after_refusal=None
        if record["after_refusal"] is None
        else _resource_record(record["after_refusal"]),
        after_handoff=_resource_record(record["after_handoff"]),
    )
    compact["source_assertions"] = dict(
        complete_native_reference_exact=True,
        complete_cursor_resources_preserved=True,
        same_original_resources=True,
        old_owner_retired=True,
        token_consumed=True,
    )
    return compact


def _case(
    family: str,
    mode: str,
    *,
    handoff_cycles: int = 3,
    live_resources: bool = False,
    compact: bool = False,
) -> dict[str, Any]:
    if (
        type(handoff_cycles) is not int
        or handoff_cycles not in (3, 16)
        or type(live_resources) is not bool
        or type(compact) is not bool
    ):
        raise ValueError(
            "handoff control requires fixed horizon3/16 and native resource/compact flags"
        )
    if (handoff_cycles, live_resources, compact) not in ((3, False, False), (16, True, True)):
        raise ValueError("handoff control supports only ordinary3 or live compact16 cycles")
    if (
        type(family) is not str
        or family not in FAMILIES
        or type(mode) is not str
        or mode not in MODES
    ):
        raise ValueError("handoff control requires a supported fixed family and inference mode")
    arrivals = (3, *(7 + 2 * number for number in range(handoff_cycles)))
    raw_wall, raw_reader = monotonic, read_process_rss_bytes
    sources = _resource_sources(raw_wall, raw_reader)
    clock, budget, sampler = _resource_setup(
        handoff_cycles, live_resources, wall_source=raw_wall, rss_source=raw_reader
    )
    wall, reader = budget.clock, None if sampler is None else sampler.read_rss_bytes
    initial_resources = (
        None
        if sampler is None
        else dict(budget=_plain(_budget_state(budget)), sampler=_plain(asdict(sampler.snapshot())))
    )
    # Keep the original controller/resource lifetimes and three publication
    # transactions visible in one fixed narrative. Factoring each cycle into
    # independent setup would obscure the no-reset assertion being exercised.
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
    wrong = CoupledCheckpointCallbacks(
        replace(policy, learning_rate=0.05), binding=BINDING, limits=LIMITS
    )
    initial = ActorShadowRuntime(
        reference,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=handoff_cycles + 1,
        max_consolidations=0,
    )
    actor = initial.actor
    gate = ServingPriorityGate(SharingLimits(2, handoff_cycles, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(initial, gate)
    actor_a = _payload(policy, actor.snapshot_state().state)
    prediction_a = shared.predict(features.copy()).prediction.tolist()
    _queue(initial, features, targets, arrivals=arrivals)
    builds: list[str] = []

    def build(state):
        _require(
            shared.predict(features.copy()).prediction.tolist() == prediction_a
            and _payload(policy, actor.snapshot_state().state) == actor_a,
            "actor unavailable or changed during repeated candidate preparation",
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
        max_prepare_attempts=handoff_cycles + 1,
    )
    cycles: list[dict[str, Any]] = []
    wrong_refusal = None
    prediction_d: list[list[float]] = []
    payload_d = b""
    for number, tick in enumerate(arrivals[:handoff_cycles], 1):
        clock.advance_to(tick)
        old = shared._runtime
        poll = shared.train_ready()
        diagnostic = reference.train_batch(features.copy(), targets.copy())
        observed = _observation(old, gate, sampler=sampler)
        payload = _payload(policy, observed["native"])
        prediction_d = old._candidate.predict(features.copy()).tolist()
        _require(
            [item.sample_id for item in poll.updates] == [f"s{number - 1}"]
            and poll.updates[0].diagnostic == diagnostic
            and payload == _payload(policy, reference.snapshot_state())
            and prediction_d == reference.predict(features.copy()).tolist()
            and callbacks.policy_digest(old._candidate) == callbacks.policy_digest(reference),
            "repeated native continuation differs from the independent complete reference",
        )
        gate.pause()
        before = _observation(old, gate, sampler=sampler)
        token = control.capture()
        view = control.inspect(token)
        after_refusal = None
        if number == 1:
            wrong_refusal = _refused(
                lambda: control.restore(token),
                "coupled learner policy differs from the original callback policy",
            )
            after_refusal = _observation(old, gate, sampler=sampler)
            _unchanged(
                policy,
                payload,
                before,
                after_refusal,
                "wrong first preparation changed complete owner/native/resources",
            )
            _require(
                shared._runtime is old
                and not old._retired
                and control._attempts == len(control._models) == len(builds) == 1,
                "wrong first preparation did not spend exactly one bounded attempt",
            )
        new = control.restore(token)
        after_handoff = _observation(new, gate, sampler=sampler)
        _unchanged(
            policy,
            payload,
            before,
            after_handoff,
            "repeated handoff changed complete native/cursor/resource state",
        )
        _require(
            shared._runtime is new
            and new is not old
            and old._retired
            and new.actor is actor
            and new._budget is budget
            and new._inbox._clock is clock
            and shared._sharing is gate
            and budget.clock is wall
            and budget.process_rss_sampler is sampler
            and (sampler is None or sampler.read_rss_bytes is reader)
            and gate.snapshot().paused
            and control._attempts == len(control._models) == len(builds) == number + 1
            and len(control._pending) == 0,
            "repeated handoff renewed original identities or preparation accounting",
        )
        retired_refusal = _refused(
            old.train_ready, "candidate owner is retired after checkpoint handoff"
        )
        token_refusal = _refused(
            lambda: control.restore(token),
            "checkpoint is foreign, copied, discarded or already used",
        )
        _require(
            control._attempts == len(control._models) == len(builds) == number + 1,
            "retired owner or used token reached an extra native builder",
        )
        gate.resume()
        empty = shared.train_ready()
        _require(
            empty.updates == () and budget.updates_completed == number,
            "same-tick repeated handoff poll trained an unarrived label",
        )
        cycle_record = dict(
            cycle=number,
            tick=tick,
            updates=[asdict(item) for item in poll.updates],
            direct_diagnostic=asdict(diagnostic),
            payload_utf8=payload.decode(),
            captured_cursor=_plain(asdict(view.inbox)),
            before=_metadata(before),
            after_refusal=None if after_refusal is None else _metadata(after_refusal),
            after_handoff=_metadata(after_handoff),
            preparation_attempts=control._attempts,
            prepared_models=len(control._models),
            pending_checkpoints=len(control._pending),
            retired_owner_refusal=retired_refusal,
            used_token_refusal=token_refusal,
            same_tick_updates=list(empty.updates),
        )
        cycles.append(_compact_cycle(cycle_record) if compact else cycle_record)
        payload_d = payload

    current = shared._runtime
    gate.pause()
    quota_before = _observation(current, gate, sampler=sampler)
    token = control.capture()
    quota_refusal = _refused(
        lambda: control.restore(token), "checkpoint preparation attempt quota exhausted"
    )
    quota_after = _observation(current, gate, sampler=sampler)
    _unchanged(
        policy,
        payload_d,
        quota_before,
        quota_after,
        "exhausted preparation changed complete current native/cursor/resources",
    )
    _require(
        shared._runtime is current
        and not current._retired
        and control._attempts == len(control._models) == len(builds) == handoff_cycles + 1
        and len(control._pending) == 1,
        "exhausted preparation invoked builder or consumed/reset original preparation quota",
    )
    control.discard(token)
    _require(len(control._pending) == 0, "obsolete exhausted checkpoint remained pending")
    preparation_quota = dict(
        before=_metadata(quota_before),
        after_refusal=_metadata(quota_after),
        reason=quota_refusal,
        preparation_attempts=control._attempts,
        prepared_models=len(control._models),
        pending_before_discard=1,
        pending_after_discard=len(control._pending),
    )
    if compact:
        preparation_quota["before"] = _resource_record(_metadata(quota_before))
        preparation_quota["after_refusal"] = _resource_record(_metadata(quota_after))
        preparation_quota["source_assertions"] = dict(
            complete_cursor_resources_preserved=True,
            same_original_resources=True,
            builder_not_called=True,
            token_discarded=True,
        )
    gate.resume()
    clock.advance_to(arrivals[-1])
    quota = shared.train_ready()
    _require(
        quota.updates == ()
        and quota.deferred_reason == "work_quota"
        and gate.snapshot().admitted_updates == budget.updates_completed == handoff_cycles,
        "repeated handoff renewed original sharing/update work allowance",
    )
    try:
        budget.before_update()
    except ToyExecutionStopped as error:
        _require(
            error.stop.reason == "max_training_updates", "original repeated budget refusal differs"
        )
        budget_refusal = error.stop.reason
    else:
        raise AssertionError("repeated handoff renewed original native update quota")
    _require(
        _payload(policy, current.actor.snapshot_state().state) == actor_a
        and shared.predict(features.copy()).prediction.tolist() == prediction_a
        and _payload(policy, current.candidate_snapshot().state) == payload_d,
        "final repeated quota refusals changed candidate or original actor",
    )
    receipts = current.applied_updates
    _require(
        [item.sample_id for item in receipts] == [f"s{number}" for number in range(handoff_cycles)]
        and [item.update_number for item in receipts] == list(range(1, handoff_cycles + 1))
        and [item.arrived_at for item in receipts] == list(arrivals[:handoff_cycles])
        and [item.applied_at for item in receipts] == list(arrivals[:handoff_cycles]),
        "repeated handoff lost cumulative causal receipt history",
    )
    for expected_wakes, payload in enumerate(
        (actor_a, *(item["payload_utf8"].encode() for item in cycles)), 2
    ):
        wake = json.loads(payload)["native"]["wake_state"]
        _require(
            wake is None if family == "pc" else wake["wake_updates"] == expected_wakes,
            "repeated handoff lost genuine complete CPC wake history",
        )
    final_cursor = current._inbox.capture_cursor()
    canonical_order = sorted(
        range(handoff_cycles + 1), key=lambda number: (f"e{number}", f"s{number}")
    )
    _require(
        [(item.episode_id, item.sample_id) for item in final_cursor.experiences]
        == [(f"e{number}", f"s{number}") for number in canonical_order]
        and [(item.episode_id, item.sample_id, item.arrived_at) for item in final_cursor.labels]
        == [(f"e{number}", f"s{number}", arrivals[number]) for number in canonical_order]
        and len(final_cursor.applied) == handoff_cycles
        and final_cursor.completed_updates == handoff_cycles
        and final_cursor.last_tick == arrivals[-1]
        and not final_cursor.stopped,
        "repeated handoff lost the fourth pending pair or misreported a shared quota deferral",
    )
    row = dict(
        family=family,
        mode=mode,
        policy=asdict(policy),
        actor_a_utf8=actor_a.decode(),
        final_d_utf8=payload_d.decode(),
        prediction_a=prediction_a,
        prediction_d=prediction_d,
        cycles=cycles,
        preparation_quota=preparation_quota,
        final_cursor=_plain(asdict(final_cursor)),
        final_budget=_plain(_budget_state(budget)),
        final_sharing=_plain(asdict(gate.snapshot())),
        refusals=dict(
            wrong_policy=wrong_refusal,
            preparation_quota=quota_refusal,
            sharing=quota.deferred_reason,
            budget=budget_refusal,
        ),
        builds=builds,
        preparation_attempts=control._attempts,
        prepared_models=len(control._models),
        pending_checkpoints=len(control._pending),
        same_actor_budget_clock_gate=True,
        actor_unchanged=True,
        complete_continuation_exact=True,
    )
    if sampler is not None:
        # Final native/metadata work remains under the same absolute resource
        # budget. Read-only reporting never renews its start, reader or limits.
        budget.before_final()
        final_resources = dict(
            budget=_plain(_budget_state(budget)), sampler=_plain(asdict(sampler.snapshot()))
        )
        assert initial_resources is not None
        _sample_progress(initial_resources["sampler"], final_resources["sampler"])
        _require(
            current.actor is actor
            and current._budget is budget
            and current._inbox._clock is clock
            and shared._sharing is gate
            and budget.clock is wall
            and budget.process_rss_sampler is sampler
            and sampler.read_rss_bytes is reader
            and budget.started_at == initial_resources["budget"]["started_at"],
            "endurance changed original resource identities or absolute start",
        )
        row.update(
            horizon=handoff_cycles,
            resource_sources=sources,
            resources=dict(initial=initial_resources, final=final_resources),
            resource_identity=dict(
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
            ),
            final_budget=final_resources["budget"],
            same_original_resources=True,
        )
    if compact:
        row["final_payload_utf8"] = row.pop("final_d_utf8")
        row["final_prediction"] = row.pop("prediction_d")
    _require(
        len(_canonical(row).encode("utf-8")) <= (131072 if compact else 65536),
        "repeated candidate handoff summary exceeds fixed case byte cap",
    )
    return row


def run_repeated_example() -> dict[str, Any]:
    rows = [_case(family, mode) for mode in MODES for family in FAMILIES]
    for offset in (0, 3):
        for stage in range(4):
            states = []
            for number in (0, 1):
                row = rows[offset + number]
                payload = (
                    row["actor_a_utf8"] if stage == 0 else row["cycles"][stage - 1]["payload_utf8"]
                )
                states.append(json.loads(payload)["native"]["arrays"])
            _require(
                states[0] == states[1],
                "PC-neutral complete parameter frames differ across repeated handoffs",
            )
    return dict(
        protocol=describe_repeated_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        repeated_handoff_verified=True,
        full_managed_process_recovery_verified=False,
    )


def run_endurance_example() -> dict[str, Any]:
    rows = [
        _case(family, mode, handoff_cycles=16, live_resources=True, compact=True)
        for mode in MODES
        for family in FAMILIES
    ]
    for offset in (0, 3):
        for stage in range(17):
            states = []
            for number in (0, 1):
                row = rows[offset + number]
                payload = (
                    row["actor_a_utf8"] if stage == 0 else row["cycles"][stage - 1]["payload_utf8"]
                )
                states.append(json.loads(payload)["native"]["arrays"])
            _require(
                states[0] == states[1],
                "PC-neutral complete frames differ across endurance handoffs",
            )
    report = dict(
        protocol=describe_endurance_example(),
        cases=rows,
        horizon=16,
        pc_neutral_parameters_exact=True,
        handoff_endurance_verified=True,
        full_managed_process_recovery_verified=False,
    )
    _require(
        len(_canonical(report).encode("utf-8")) <= 1048576,
        "complete handoff endurance report exceeds bounded JSON allowance",
    )
    return report
