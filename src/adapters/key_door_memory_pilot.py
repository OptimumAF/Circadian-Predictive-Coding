"""Bounded supervised count-memory control on the existing TRAIN stream/live runner.

Inputs are the frozen original eight public-history TRAIN episodes. Outputs are
five checkpoint score/live curves and charged storage/physics/memory work. This
recipe does not select data, tune other learners or access final data. An opt-in
assay runs the existing isolated E16 helpers instead of known-TRAIN live readouts.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any

from src.adapters import key_door_pilot as shared
from src.adapters import key_door_public_history_pilot as history
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.key_door_learnability import (
    _bindings as _original_bindings,
    probe_slots,
    run_checkpoint_probe,
    valid_probe,
    valid_probes,
)
from src.adapters.train_acquisition_pilot import _point
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.fresh_task_probe import run_fresh_task_probe
from src.core.action_memory import ActionMemoryPolicy

WALL_SECONDS = 90.0


def describe_protocol(*, with_probes: bool = False) -> dict[str, Any]:
    if type(with_probes) is not bool:
        raise ValueError("with_probes must be an exact bool")
    if with_probes:
        protocol = describe_protocol()
        protocol.update(
            protocol_id="local_key_door_bounded_action_memory_probe_v1",
            classification="bounded_memory_reserved_E16_capacity_not_generalization",
            readout="all8TRAINupdates then30TRAINscores then disposable initial0/end8 E16 probes; no live/final/replay",
            probes=dict(
                benchmark_checkpoints=[0, 8],
                relative_updates=[0, 4, 8, 16],
                updates_per_cohort=16,
                training_packets=16,
                evaluation_packets=2,
                roles=["probe_train", "probe_test"],
                decision_diagnostics=True,
                isolation="full owned copies with separate original16-update budgets; no promotion",
            ),
            reserved_probe_array_bytes=19584,
            shared_encoded_array_bytes=28288,
            probe_table_payload_limit_bytes=10752,
            planned_work=dict(
                benchmark_storage_updates=8,
                probe_storage_updates=32,
                storage_updates=40,
                training_rows=320,
                score_calls=46,
                score_rows=368,
                teacher_episodes=26,
                teacher_physics_steps=208,
                live_episodes=0,
            ),
            limitations="fixed deterministic E16 reserved resubstitution; bounded memory may refuse; unequal capacity/compute to neural controls; no generalization/CPC advantage/recovery-deficit/full R4.2/R4.3 claim",
        )
        return protocol
    return dict(
        protocol_id="local_key_door_bounded_action_memory_v1",
        classification="bounded_supervised_known_train_memory_control",
        methods=["observation_memory"],
        policy=asdict(ActionMemoryPolicy()),
        seed=17,
        task_sequence=list(shared.TASKS),
        checkpoint_updates=list(shared.POINTS),
        stream=dict(training_only=True, heldout=[], label_delay=1, updates=8),
        history="public initial8 + current8; fresh session per episode",
        known_key="five empirical action counts normalized; conflicts preserved",
        unseen_key="uniform five actions; raw first-tie argmax",
        update_kind="supervised count storage, not SGD",
        readout="all updates then30 original TRAIN scores then30 known TRAIN live episodes; no final/probe/replay",
        metric_direction="CE lower; accuracy/exact_sequence/success/return higher; positive retention loss worse",
        planned_work=dict(
            storage_updates=8,
            training_rows=64,
            score_calls=30,
            score_rows=240,
            teacher_episodes=8,
            teacher_physics_steps=64,
            live_episodes=30,
            maximum_live_prediction_calls=360,
            maximum_live_physics_steps=360,
            maximum_live_native_forks=60,
        ),
        maximum_wall_seconds=WALL_SECONDS,
        maximum_process_rss_bytes=shared.MAX_RSS,
        limitations="known TRAIN recall only; unequal capacity/compute to neural controls; no fresh-task/generalization/CPC advantage/full R4.2/R4.3 claim",
    )


def describe_probe_protocol() -> dict[str, Any]:
    return describe_protocol(with_probes=True)


def run_probe_pilot() -> dict[str, Any]:
    return run_pilot(with_probes=True)


def _budget_binding(wake: Any) -> tuple[Any, ...]:
    # _same_value intentionally refuses arbitrary objects, including type records.
    # Preserve the original counter/clock type identities as process-local IDs;
    # all values, budget/progress/clock identities and fields stay exact.
    return tuple(
        (*binding[:9], *(id(kind) for kind in binding[9:]))
        for binding in _original_bindings(wake, None)
    )


def _observation(updates: int, state: Any) -> dict[str, Any]:
    return dict(
        wake_updates=updates,
        native_updates=state.completed_updates,
        native_training_examples=state.training_examples,
        entries=len(state.entries),
        logical_payload_bytes=168 * len(state.entries),
        conflicting_keys=sum(
            sum(count > 0 for count in entry.counts) > 1 for entry in state.entries
        ),
        snapshot=asdict(state),
    )


def run_pilot(*, with_probes: bool = False) -> dict[str, Any]:
    # Why a concrete recipe: compose the existing stream and live runner without
    # adding a backend framework or changing the four accepted neural recipes.
    protocol = describe_protocol(with_probes=with_probes)
    started = monotonic()
    deadline = started + WALL_SECONDS
    row: dict[str, Any] = dict(
        method="observation_memory",
        status="not_run",
        reason="not reached",
        observations=[],
        training_points=[],
        stream_result=None,
        outcomes=None,
        wake_updates=0,
        source_unchanged=None,
        actor_unchanged=None,
        partial_native_evidence=None,
        native_score_returns=0,
        work=dict(
            score_forwards_attempted=0,
            score_forwards_completed=0,
            score_rows_attempted=0,
            score_rows_completed=0,
        ),
        live_episodes=history._live_slots(),
        live_work={},
        live_sessions_started=0,
        live_outcomes=None,
    )
    if with_probes:
        for field in ("live_episodes", "live_work", "live_sessions_started", "live_outcomes"):
            del row[field]
        row["probes"] = probe_slots()
    expected = deepcopy(row)
    training: tuple[Any, ...] = ()
    data_seal: Any = None
    data_owners: Any = None
    source: Any = None
    stream: Any = None
    wake: Any = None
    sampler: Any = None
    memory: Any = None
    initial: Any = None
    sealed: Any = None
    preceding: Any = None
    collection_started = False
    published: list[tuple[int, int]] = []
    errors: list[str] = []
    probe_data: tuple[Any, ...] = ()
    probe_seal: Any = None
    probe_owners: Any = None
    probe_collection_started = False
    published_probes: list[tuple[int, int, int]] = []

    def publish(**changes: Any) -> None:
        row.update(changes)
        expected.update(deepcopy(changes))

    def check() -> None:
        if monotonic() > deadline:
            raise ValueError("original action-memory invocation deadline exhausted")
        if not _same_value(expected, row) or not _same_value(
            data_seal, history._signature(training)
        ):
            raise ValueError("original TRAIN data or preceding evidence changed")
        if training and data_owners != tuple(
            (id(b), id(b.features), id(b.targets), id(b.episode)) for b in training
        ):
            raise ValueError("original TRAIN data was replaced")
        if not _same_value(
            probe_seal, history._signature((*probe_data[0], *probe_data[1])) if probe_data else None
        ):
            raise ValueError("original reserved probe data changed")
        if probe_data and probe_owners != tuple(
            (id(b), id(b.features), id(b.targets), id(b.episode))
            for group in probe_data
            for b in group
        ):
            raise ValueError("original reserved probe data was replaced")
        for position, slot_id, report_id in published_probes:
            slot = row["probes"][position]
            if id(slot) != slot_id or id(slot["report"]) != report_id:
                raise ValueError("preceding probe evidence was replaced")
        for position, identity in published:
            if id(row["live_episodes"][position]) != identity:
                raise ValueError("preceding live evidence was replaced")
        if stream is not None:
            stream._check_original()
            if not _same_value(initial, source.snapshot_state()) or not _same_value(
                initial, stream._runtime.actor.snapshot_state().state
            ):
                raise ValueError("memory source or serving actor changed")
            if sealed is not None and not _same_value(
                sealed,
                (
                    stream._states,
                    stream._runtime.candidate_snapshot().state,
                    stream._runtime.applied_updates,
                    _budget_binding(wake),
                ),
            ):
                raise ValueError("completed state, receipt, budget or clock changed")

    def observe_budget() -> None:
        nonlocal sealed
        check()
        before = _budget_binding(wake)
        wake.before_final()
        after = _budget_binding(wake)
        if (
            not _same_value(before[0][:8] + before[0][9:], after[0][:8] + after[0][9:])
            or after[0][8] < before[0][8]
        ):
            raise ValueError("readout changed original budget binding or progress")
        sealed = (*sealed[:3], after)
        check()

    try:
        sampler = shared.ProcessRssSampler()
        with sampler:
            collection_started = True
            training = history.build_training_stream()
            data_seal = history._signature(training)
            data_owners = tuple(
                (id(b), id(b.features), id(b.targets), id(b.episode)) for b in training
            )
            check()
            wake = shared._session(deadline, sampler)
            source = ActionMemoryLearner()
            initial = source.snapshot_state()

            def observe(updates: int, state: Any, batch: Any) -> None:
                check()
                publish(observations=expected["observations"] + [_observation(updates, state)])
                check()

            stream = AttributionStream(
                source,
                training,
                (),
                checkpoint_updates=shared.POINTS,
                label_delay=1,
                budget=wake,
                observe_checkpoint=observe,
                training_only=True,
            )
            for point in range(1, 9):
                check()
                stream.run_until(point)
                check()
                preceding = _observation(point, stream._runtime.candidate_snapshot().state)
            result = stream.finish()
            check()
            publish(
                stream_result=result,
                wake_updates=wake.updates_completed,
                status="trained",
                reason="TRAIN readout follows all storage updates",
            )
            sealed = deepcopy(
                (
                    stream._states,
                    stream._runtime.candidate_snapshot().state,
                    stream._runtime.applied_updates,
                    _budget_binding(wake),
                )
            )
            check()
            for point, state in stream._states:
                check()
                evaluator = source.fork()
                check()
                if evaluator is source:
                    raise ValueError("TRAIN scorer requires an owned fork")
                evaluator.restore_state(deepcopy(state))
                check()
                if not _same_value(state, evaluator.snapshot_state()):
                    raise ValueError("TRAIN scorer did not restore its checkpoint")
                predictions = []
                for batch in training[:6]:
                    observe_budget()
                    work = deepcopy(expected["work"])
                    work["score_forwards_attempted"] += 1
                    work["score_rows_attempted"] += len(batch.features)
                    publish(work=work)
                    prediction = evaluator.predict(batch.features.copy())
                    publish(native_score_returns=expected["native_score_returns"] + 1)
                    check()
                    if not _same_value(state, evaluator.snapshot_state()):
                        raise ValueError("TRAIN prediction changed its checkpoint")
                    work["score_forwards_completed"] += 1
                    work["score_rows_completed"] += len(batch.features)
                    publish(work=work)
                    predictions.append(prediction)
                point_result = _point(point, tuple(predictions), training[:6])
                check()
                publish(training_points=expected["training_points"] + [point_result])
            publish(
                status="benchmark_complete",
                reason=None,
                source_unchanged=True,
                actor_unchanged=True,
                outcomes=shared._outcomes({"checkpoints": row["training_points"]}),
            )
            check()
            if not with_probes:
                for position, planned in enumerate(row["live_episodes"]):
                    check()
                    point = planned["checkpoint_updates"]
                    state = next(state for at, state in stream._states if at == point)
                    private: dict[str, Any] = dict(
                        method="observation_memory",
                        replay_enabled=False,
                        episodes=[deepcopy(planned)],
                        live_sessions_started=0,
                    )
                    try:
                        shared._live(
                            private,
                            source,
                            [(point, state)],
                            wake,
                            declared_layouts=shared.layouts()[:3],
                            public_history=True,
                            check_boundary=check,
                            observe_budget=observe_budget,
                            same_state=_same_value,
                        )
                    except Exception as error:
                        slot = private["episodes"][0]
                        if slot["status"] != "failed":
                            slot.update(status="failed", reason=f"{type(error).__name__}: {error}")
                        raise
                    finally:
                        slot = private["episodes"][0]
                        row["live_episodes"][position] = slot
                        expected["live_episodes"][position] = deepcopy(slot)
                        published.append((position, id(slot)))
                        publish(
                            live_sessions_started=expected["live_sessions_started"]
                            + private["live_sessions_started"],
                            live_work=history._live_work(row["live_episodes"]),
                        )
                    check()
                    if slot["status"] != "complete" or slot["state_unchanged"] is not True:
                        raise ValueError("live episode did not preserve its memory checkpoint")
                publish(
                    status="complete", live_outcomes=history._live_outcomes(row["live_episodes"])
                )
                check()
            else:
                # Why: reuse accepted live evidence and spend this case on fresh
                # capacity, after authentic benchmark training and TRAIN readouts.
                probe_collection_started = True
                probe_data = history.build_probe_stream()
                probe_seal = history._signature((*probe_data[0], *probe_data[1]))
                probe_owners = tuple(
                    (id(b), id(b.features), id(b.targets), id(b.episode))
                    for group in probe_data
                    for b in group
                )
                check()
                for position, point in enumerate((0, 8)):
                    check()
                    state = next(state for at, state in stream._states if at == point)
                    private = dict(probes=probe_slots())
                    try:
                        run_checkpoint_probe(
                            private,
                            source,
                            state,
                            probe_data,
                            checkpoint=point,
                            wake=wake,
                            replay=None,
                            deadline=deadline,
                            sampler=sampler,
                            same_state=_same_value,
                            probe_runner=run_fresh_task_probe,
                            decision_diagnostics=True,
                            check_boundary=check,
                        )
                    finally:
                        slot = private["probes"][position]
                        row["probes"][position] = slot
                        expected["probes"][position] = deepcopy(slot)
                        published_probes.append((position, id(slot), id(slot["report"])))
                    check()
                    if not valid_probe(slot):
                        raise ValueError(
                            "memory probe did not satisfy its original cohort contract"
                        )
                publish(status="complete")
                check()

    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        partial: dict[str, Any] = dict(
            failed_work_may_be_unobserved=True, preceding_success_observation=deepcopy(preceding)
        )
        if wake is not None:
            partial["charged_updates"] = wake.updates_completed
        if stream is not None:
            try:
                partial["applied_updates"] = [asdict(r) for r in stream._runtime.applied_updates]
                state = stream._runtime.candidate_snapshot().state
                partial["candidate_observation"] = _observation(state.completed_updates, state)
            except Exception as readback_error:
                partial["readback_error"] = f"{type(readback_error).__name__}: {readback_error}"
                partial["current_candidate_unavailable"] = True
        publish(
            status="failed",
            reason=errors[-1],
            partial_native_evidence=partial,
            wake_updates=0 if wake is None else wake.updates_completed,
        )
    finally:
        try:
            if sampler is not None:
                memory = asdict(sampler.snapshot())
            check()
        except Exception as error:
            errors.append(f"readback {type(error).__name__}: {error}")
    demonstrations = []
    try:
        demonstrations = [asdict(b.episode) for b in training]
    except Exception as error:
        errors.append(f"demonstration readback {type(error).__name__}: {error}")
    probe_demonstrations = []
    reserved_probe_array_bytes = None
    if with_probes:
        try:
            probe_demonstrations = [asdict(b.episode) for group in probe_data for b in group]
            if probe_data:
                reserved_probe_array_bytes = sum(
                    b.features.nbytes + b.targets.nbytes for group in probe_data for b in group
                )
        except Exception as error:
            errors.append(f"probe demonstration readback {type(error).__name__}: {error}")
    complete = (
        not errors
        and row["status"] == "complete"
        and memory is not None
        and memory["peak_bytes"] <= shared.MAX_RSS
        and (not with_probes or valid_probes(row))
    )
    report = dict(
        protocol=protocol,
        status="complete" if complete else "incomplete",
        runs=[row],
        demonstrations=demonstrations,
        global_errors=errors,
        total_seconds=monotonic() - started,
        observed_process_rss=memory,
        shared_initial_parameters_exact=None,
        pc_neutral_exact=None,
        collected_episodes=len(training) if training else (None if collection_started else 0),
        partial_collection_work_may_be_unobserved=collection_started and not training,
        final_data_constructed=False,
        scientific_advantage_verified=False,
        full_R42_complete=False,
        full_R43_complete=False,
    )
    if with_probes:
        report.update(
            probe_demonstrations=probe_demonstrations,
            collected_probe_episodes=sum(len(group) for group in probe_data)
            if probe_data
            else (None if probe_collection_started else 0),
            partial_probe_collection_work_may_be_unobserved=probe_collection_started
            and not probe_data,
            reserved_probe_array_bytes=reserved_probe_array_bytes,
        )
    return report
