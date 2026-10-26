"""Matched public-history TRAIN streams on the existing four native learners.

The adapter collects only TRAIN episodes, applies the pure local history encoder,
and composes AttributionStream and existing TRAIN metric projection. An opt-in
mode composes existing disposable E16 probes after every benchmark readout.
An alternative opt-in mode composes existing known TRAIN-layout live readouts.
No final, replay, selection or new training objective lives here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from time import monotonic
from typing import Any

from src.adapters import key_door_pilot as shared
from src.adapters.key_door_learnability import (
    _bindings,
    probe_slots,
    run_checkpoint_probe,
    valid_probe,
    valid_probes,
)
from src.adapters.train_acquisition_pilot import _point
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy, _config
from src.adapters.torch_feature_rehearsal_pilot import _observation, _same_state
from src.app.coupled_attribution import AttributionStream, _validate_batch
from src.app.fresh_task_probe import (
    run_fresh_task_probe,
    _validate_batches as validate_probe_batches,
)
from src.core.demonstration_episode import episode_arrays
from src.core.key_door_public_history import encode_public_history_episode

WALL_SECONDS = 90.0


def describe_protocol(*, with_probes: bool = False, with_live: bool = False) -> dict[str, Any]:
    if type(with_probes) is not bool:
        raise ValueError("with_probes must be an exact bool")
    if type(with_live) is not bool or (with_live and with_probes):
        raise ValueError("with_live must be an exact bool and cannot combine with probes")
    if with_live:
        protocol = describe_protocol()
        protocol.update(
            protocol_id="local_key_door_matched_public_history_live_v1",
            classification="matched_history_known_train_layout_closed_loop",
            live_metrics="pooled and per-layout success/return/illegal-action/step-limit rates at all five points; step/session/encoding work is cost, not skill",
            readout="original TRAIN scores after all four streams finish; then owned causal-history sessions on known TRAIN layouts only; no final/probe/replay",
            live=dict(
                checkpoints=list(shared.POINTS),
                layouts=["A", "B", "C"],
                cues=[1, -1],
                episodes_per_method=30,
                max_steps=12,
                action_rule="raw argmax/first tie; no mask, repair or retry",
                reset="fresh owned full native fork and initial public frame per episode",
            ),
            planned_work=dict(
                native_updates=32,
                training_rows=256,
                score_calls=120,
                score_rows=960,
                teacher_episodes=8,
                teacher_physics_steps=64,
                live_episodes=120,
                maximum_live_prediction_calls=1440,
                maximum_live_physics_steps=1440,
                maximum_physics_steps=1504,
                maximum_live_native_forks=240,
            ),
            limitations="known TRAIN layouts/fixed seed/raw greedy closed loop; policies may fail. Equal inputs/exposure/initialization, unequal BP/PC computation; no acquired-task premise/generalization/CPC advantage/useful retention/recovery/fullR42/fullR43 claim",
        )
        return protocol
    if with_probes:
        protocol = describe_protocol()
        protocol.update(
            protocol_id="local_key_door_matched_public_history_probe_v1",
            classification="matched_history_reserved_E16_capacity_not_generalization",
            readout="original TRAIN scores after all four streams finish; then disposable initial0/end8 reserved E16 probes; no final/live/replay",
            probes=dict(
                benchmark_checkpoints=[0, 8],
                relative_updates=[0, 4, 8, 16],
                per_probe_updates=16,
                training_packets=16,
                evaluation_packets=2,
                roles=["probe_train", "probe_test"],
                decision_diagnostics=True,
                execution="all four benchmark streams and TRAIN readouts finish first; existing helpers on disposable full-state forks; no benchmark mutation",
            ),
            reserved_encoded_array_bytes=19584,
            shared_encoded_array_bytes=28288,
            planned_work=dict(
                benchmark_updates=32,
                probe_updates=128,
                native_updates=160,
                training_rows=1280,
                score_calls=184,
                score_rows=1472,
                teacher_episodes=26,
                physics_steps=208,
            ),
            limitations="deterministic reserved E16 fit/class1absent/fixed seed; unequal BP/PC compute; initial-to-end includes all benchmark history, no causal attribution/live/generalization/CPC benefit/useful retention/recovery/fullR43 claim",
        )
        return protocol
    return dict(
        protocol_id="local_key_door_matched_public_history_train_v1",
        classification="matched_train_fit_observability_qualification",
        methods=list(shared.METHODS),
        feature_dim=16,
        hidden_dim=8,
        num_classes=5,
        seed=17,
        task_sequence=list(shared.TASKS),
        checkpoint_updates=list(shared.POINTS),
        history="own public initial8 + current8; one initial frame/reset per episode; no labels/context/future",
        learners={
            method: asdict(TorchFeaturePolicy(method, 16, 8, 17, num_classes=5))
            for method in shared.METHODS
        },
        cpc_configs={
            method: asdict(_config(TorchFeaturePolicy(method, 16, 8, 17, num_classes=5)))
            for method in ("neutral_cpc", "gated_cpc")
        },
        stream=dict(training_only=True, heldout=[], label_delay=1, updates_per_method=8),
        readout="all four streams finish before six original TRAIN episodes scored at all five checkpoints; no final/probe/live/replay",
        metric_direction="CE lower/accuracy/exact_sequence higher; positive new_B/C/return gain better, positive retention loss worse",
        parameter_count_per_model=181,
        logical_initial_frame_bytes=64,
        raw_array_bytes=4608,
        encoded_array_bytes=8704,
        planned_work=dict(
            native_updates=32,
            training_rows=256,
            score_calls=120,
            score_rows=960,
            teacher_episodes=8,
            physics_steps=64,
        ),
        maximum_wall_seconds=WALL_SECONDS,
        limitations="eight TRAIN episodes/fixed seed and teacher-forced resubstitution; history resolves demonstrated A cue alias only, not hidden B/context identity; unequal BP/PC native computation, no acquired/live/generalization/CPC advantage/fullR43 claim",
    )


def _signature(batches: Any) -> Any:
    return [
        (
            batch.role,
            batch.context,
            batch.row_ids,
            batch.num_classes,
            batch.features.dtype.str,
            batch.features.tolist(),
            batch.targets.dtype.str,
            batch.targets.tolist(),
            asdict(batch.episode),
        )
        for batch in batches
    ]


def build_training_stream() -> tuple[Any, ...]:
    encoded = []
    for batch in shared.build_training_stream():
        if batch.episode is None or batch.role != "train":
            raise ValueError("public history requires complete TRAIN episodes")
        episode = encode_public_history_episode(batch.episode)
        features, targets = episode_arrays(episode)
        encoded.append(replace(batch, features=features, targets=targets, episode=episode))
    seen: set[str] = set()
    for batch in encoded:
        _validate_batch(batch, "train", seen)
    return tuple(encoded)


def build_probe_stream() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Encode each original reserved packet independently, preserving roles."""
    encoded = []
    for batches in shared.build_probe_stream():
        group = []
        for batch in batches:
            if batch.episode is None:
                raise ValueError("public-history probes require complete episode records")
            episode = encode_public_history_episode(batch.episode)
            features, targets = episode_arrays(episode)
            group.append(replace(batch, features=features, targets=targets, episode=episode))
        encoded.append(tuple(group))
    training, evaluation = encoded
    validate_probe_batches(training, evaluation)
    return training, evaluation


def describe_probe_protocol() -> dict[str, Any]:
    return describe_protocol(with_probes=True)


def run_probe_pilot() -> dict[str, Any]:
    return run_pilot(with_probes=True)


def describe_live_protocol() -> dict[str, Any]:
    return describe_protocol(with_live=True)


def run_live_pilot() -> dict[str, Any]:
    return run_pilot(with_live=True)


def _live_slots() -> list[dict[str, Any]]:
    return [
        dict(
            checkpoint_updates=point,
            context=context,
            cue=cue,
            layout_partition="known_layout",
            source_role="train",
            status="not_run",
            reason="not reached",
            state_unchanged=None,
        )
        for point in shared.POINTS
        for context in ("A", "B", "C")
        for cue in (1, -1)
    ]


def _live_work(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    work = {
        key: sum(slot.get("history_session_work", {}).get(key, 0) for slot in episodes)
        for key in (
            "session_construction_attempts",
            "sessions_started",
            "attempted_steps",
            "encoded_frames",
            "native_prediction_returns",
            "completed_history_steps",
        )
    }
    work.update(
        callback_attempts=sum(slot.get("prediction_calls_attempted", 0) for slot in episodes),
        callback_returns=sum(slot.get("prediction_calls_returned", 0) for slot in episodes),
        applied_steps=sum(slot.get("steps_executed", 0) for slot in episodes),
        failed_work_may_be_unobserved=any(slot["status"] == "failed" for slot in episodes),
    )
    return work


def _live_outcomes(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, float]:
        count = len(group)
        return dict(
            success_rate=sum(slot["success"] for slot in group) / count,
            mean_return=sum(slot["return"] for slot in group) / count,
            mean_steps=sum(slot["steps_executed"] for slot in group) / count,
            illegal_action_rate=sum(slot["outcome"] == "illegal_action" for slot in group) / count,
            step_limit_rate=sum(slot["outcome"] == "step_limit" for slot in group) / count,
        )

    results = []
    for point in shared.POINTS:
        group = [slot for slot in episodes if slot["checkpoint_updates"] == point]
        if len(group) != 6 or any(slot["status"] != "complete" for slot in group):
            raise ValueError("live outcomes require all six original completed TRAIN slots")
        contexts = {
            context: [slot for slot in group if slot["context"] == context]
            for context in ("A", "B", "C")
        }
        if any(len(slots) != 2 for slots in contexts.values()):
            raise ValueError("live outcomes require both original cues per TRAIN context")
        results.append(
            dict(
                updates=point,
                **summarize(group),
                contexts={context: summarize(slots) for context, slots in contexts.items()},
            )
        )
    return results


def _live_projection(slot: dict[str, Any]) -> dict[str, Any]:
    result = dict(slot)
    # Method/version names differ by design; all other identity fields still match.
    result["identity"] = {
        key: value
        for key, value in slot["identity"].items()
        if key not in ("method", "actor_version")
    }
    return result


def run_pilot(*, with_probes: bool = False, with_live: bool = False) -> dict[str, Any]:
    protocol = describe_protocol(with_probes=with_probes, with_live=with_live)
    started = monotonic()
    deadline = started + WALL_SECONDS
    rows: list[dict[str, Any]] = [
        dict(
            method=method,
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
        )
        for method in shared.METHODS
    ]
    if with_probes:
        for row in rows:
            row["probes"] = probe_slots()
    if with_live:
        for row in rows:
            row.update(
                live_episodes=_live_slots(),
                live_sessions_started=0,
                live_work={},
                live_outcomes=None,
            )
    expected = deepcopy(rows)
    owned: list[Any] = []
    training: tuple[Any, ...] = ()
    training_seal: Any = None
    errors: list[str] = []
    sampler = memory = None
    active = None
    initial_equal = neutral_equal = False
    collection_started = False
    probe_data: tuple[Any, ...] = ()
    probe_seal: Any = None
    probe_owners: Any = None
    published_probe_ids: list[tuple[int, int, int, int]] = []
    probe_collection_started = False
    probe_equal = False
    live_equal = False
    published_live_ids: list[tuple[int, int, int]] = []

    def publish(index: int, **changes: Any) -> None:
        rows[index].update(changes)
        expected[index].update(deepcopy(changes))

    def check() -> None:
        if monotonic() > deadline:
            raise ValueError("original public-history invocation deadline exhausted")
        if not _same_state(expected, rows) or not _same_state(training_seal, _signature(training)):
            raise ValueError("original matched TRAIN data or prior/unreached evidence changed")
        if not _same_state(
            probe_seal, _signature((*probe_data[0], *probe_data[1])) if probe_data else None
        ):
            raise ValueError("original reserved history probe data changed")
        if probe_data and probe_owners != tuple(
            (id(batch), id(batch.features), id(batch.targets), id(batch.episode))
            for group in probe_data
            for batch in group
        ):
            raise ValueError("original reserved history probe data was replaced")
        for index, point, slot_id, report_id in published_probe_ids:
            slot = rows[index]["probes"][point]
            if id(slot) != slot_id or id(slot["report"]) != report_id:
                raise ValueError("preceding probe evidence was replaced")
        for index, position, slot_id in published_live_ids:
            if id(rows[index]["live_episodes"][position]) != slot_id:
                raise ValueError("preceding live episode evidence was replaced")
        for record in owned:
            source, before, stream = record["source"], record["before"], record["stream"]
            stream._check_original()
            if not _same_state(before, source.snapshot_state()) or not _same_state(
                before, stream._runtime.actor.snapshot_state().state
            ):
                raise ValueError("matched source or serving actor changed")
            if record["sealed"] is not None and not _same_state(
                record["sealed"],
                (
                    stream._states,
                    stream._runtime.candidate_snapshot().state,
                    stream._runtime.applied_updates,
                    _bindings(record["budget"], None),
                ),
            ):
                raise ValueError("completed matched state, receipt, budget or clock changed")

    def observe_budget(record: Any) -> None:
        check()
        before = _bindings(record["budget"], None)
        record["budget"].before_final()
        after = _bindings(record["budget"], None)
        # Only this owned observation may advance last_clock (binding index8).
        if (
            not _same_state(before[0][:8] + before[0][9:], after[0][:8] + after[0][9:])
            or after[0][8] < before[0][8]
        ):
            raise ValueError("TRAIN readout changed original budget binding or progress")
        record["sealed"] = (*record["sealed"][:3], after)
        check()

    try:
        sampler = shared.ProcessRssSampler()
        with sampler:
            torch = shared.require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            collection_started = True
            training = build_training_stream()
            training_seal = _signature(training)
            check()
            for index, row in enumerate(rows):
                active = index
                check()
                wake = shared._session(deadline, sampler)
                source = TorchFeatureLearner(
                    TorchFeaturePolicy(row["method"], 16, 8, 17, num_classes=5)
                )
                check()
                before = source.snapshot_state()

                def observe(updates: int, state: Any, batch: Any, index: int = index) -> None:
                    check()
                    observation = _observation(updates, state)
                    publish(index, observations=expected[index]["observations"] + [observation])
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
                record = dict(
                    source=source,
                    before=before,
                    stream=stream,
                    budget=wake,
                    sealed=None,
                    score_returns=0,
                )
                owned.append(record)
                for point in range(1, 9):
                    check()
                    stream.run_until(point)
                    check()
                    record["preceding_success_observation"] = _observation(
                        point, stream._runtime.candidate_snapshot().state
                    )
                result = stream.finish()
                check()
                publish(
                    index,
                    stream_result=result,
                    wake_updates=wake.updates_completed,
                    status="trained",
                    reason="TRAIN fit deferred until all methods finish",
                )
                record["sealed"] = deepcopy(
                    (
                        stream._states,
                        stream._runtime.candidate_snapshot().state,
                        stream._runtime.applied_updates,
                        _bindings(wake, None),
                    )
                )
                check()
            initial_equal = all(
                row["observations"][0]["parameters"] == rows[0]["observations"][0]["parameters"]
                for row in rows
            )
            neutral_equal = all(
                a["parameters"] == b["parameters"]
                for a, b in zip(rows[1]["observations"], rows[2]["observations"], strict=True)
            )
            if not initial_equal or not neutral_equal:
                raise ValueError(
                    "matched initial weights or PC-neutral checkpoint parameters differ"
                )
            base = training[:6]
            for index, record in enumerate(owned):
                active = index
                for point, state in record["stream"]._states:
                    check()
                    evaluator = record["source"].fork()
                    check()
                    if evaluator is record["source"]:
                        raise ValueError("TRAIN scorer requires an owned fork")
                    evaluator.restore_state(deepcopy(state))
                    check()
                    predictions = []
                    for batch in base:
                        observe_budget(record)
                        work = deepcopy(expected[index]["work"])
                        work["score_forwards_attempted"] += 1
                        work["score_rows_attempted"] += 8
                        publish(index, work=work)
                        prediction = evaluator.predict(batch.features.copy())
                        record["score_returns"] += 1
                        publish(index, native_score_returns=record["score_returns"])
                        check()
                        if not _same_state(state, evaluator.snapshot_state()):
                            raise ValueError("TRAIN readout changed its checkpoint")
                        work["score_forwards_completed"] += 1
                        work["score_rows_completed"] += 8
                        publish(index, work=work)
                        predictions.append(prediction)
                    point_result = _point(point, tuple(predictions), base)
                    check()
                    publish(
                        index, training_points=expected[index]["training_points"] + [point_result]
                    )
                publish(
                    index,
                    status="benchmark_complete" if (with_probes or with_live) else "complete",
                    reason=None,
                    source_unchanged=True,
                    actor_unchanged=True,
                    outcomes=shared._outcomes({"checkpoints": rows[index]["training_points"]}),
                )
                check()
            neutral_equal = (
                neutral_equal and rows[1]["training_points"] == rows[2]["training_points"]
            )
            if not neutral_equal:
                raise ValueError("PC-neutral TRAIN readouts differ")
            check()
            if with_live:
                declared_layouts = shared.layouts()[:3]
                for index, record in enumerate(owned):
                    active = index
                    for position, planned in enumerate(rows[index]["live_episodes"]):
                        check()
                        point = planned["checkpoint_updates"]
                        state = next(state for at, state in record["stream"]._states if at == point)
                        private: dict[str, Any] = dict(
                            method=rows[index]["method"],
                            replay_enabled=False,
                            episodes=[deepcopy(planned)],
                            live_sessions_started=0,
                        )
                        try:
                            shared._live(
                                private,
                                record["source"],
                                [(point, state)],
                                record["budget"],
                                declared_layouts=declared_layouts,
                                public_history=True,
                                check_boundary=check,
                                observe_budget=lambda: observe_budget(record),
                            )
                        except Exception as error:
                            slot = private["episodes"][0]
                            if slot["status"] != "failed":
                                slot.update(
                                    status="failed", reason=f"{type(error).__name__}: {error}"
                                )
                            raise
                        finally:
                            slot = private["episodes"][0]
                            rows[index]["live_episodes"][position] = slot
                            expected[index]["live_episodes"][position] = deepcopy(slot)
                            published_live_ids.append((index, position, id(slot)))
                            publish(
                                index,
                                live_sessions_started=expected[index]["live_sessions_started"]
                                + private["live_sessions_started"],
                                live_work=_live_work(rows[index]["live_episodes"]),
                            )
                        check()
                        if slot["status"] != "complete" or slot["state_unchanged"] is not True:
                            raise ValueError(
                                "TRAIN live episode did not preserve its native checkpoint"
                            )
                    publish(
                        index,
                        status="complete",
                        live_outcomes=_live_outcomes(rows[index]["live_episodes"]),
                    )
                    check()
                live_equal = all(
                    _live_projection(a) == _live_projection(b)
                    for a, b in zip(rows[1]["live_episodes"], rows[2]["live_episodes"], strict=True)
                )
                if not live_equal:
                    raise ValueError("PC-neutral known TRAIN-layout live readouts differ")
                check()
            if with_probes:
                probe_collection_started = True
                probe_data = build_probe_stream()
                probe_seal = _signature((*probe_data[0], *probe_data[1]))
                probe_owners = tuple(
                    (id(batch), id(batch.features), id(batch.targets), id(batch.episode))
                    for group in probe_data
                    for batch in group
                )
                check()
                for index, record in enumerate(owned):
                    active = index
                    for position, point in enumerate((0, 8)):
                        check()
                        # Keep the active helper's output private. Previously published
                        # successes/unreached slots remain sealed throughout its calls.
                        private = dict(probes=probe_slots())
                        state = next(state for at, state in record["stream"]._states if at == point)
                        try:
                            run_checkpoint_probe(
                                private,
                                record["source"],
                                state,
                                probe_data,
                                checkpoint=point,
                                wake=record["budget"],
                                replay=None,
                                deadline=deadline,
                                sampler=sampler,
                                same_state=_same_state,
                                probe_runner=run_fresh_task_probe,
                                decision_diagnostics=True,
                                check_boundary=check,
                            )
                        finally:
                            # Publish only this attempt, never reauthorize prior rows.
                            slot = private["probes"][position]
                            rows[index]["probes"][position] = slot
                            expected[index]["probes"][position] = deepcopy(slot)
                            published_probe_ids.append(
                                (index, position, id(slot), id(slot["report"]))
                            )
                        check()
                        if not valid_probe(rows[index]["probes"][position]):
                            raise ValueError(
                                "matched history probe did not satisfy its original contract"
                            )
                    publish(index, status="complete")
                    check()
                # Runtime durations differ; compare every other deterministic report field.
                probe_equal = all(
                    {key: value for key, value in a["report"].items() if key != "elapsed_seconds"}
                    == {
                        key: value for key, value in b["report"].items() if key != "elapsed_seconds"
                    }
                    for a, b in zip(rows[1]["probes"], rows[2]["probes"], strict=True)
                )
                if not probe_equal or not all(valid_probes(row) for row in rows):
                    raise ValueError(
                        "matched history probe completion or PC-neutral parity differs"
                    )
                check()
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        if active is not None:
            partial: dict[str, Any] = dict(failed_work_may_be_unobserved=True)
            if active < len(owned):
                record = owned[active]
                partial["charged_updates"] = record["budget"].updates_completed
                partial["score_calls_returned"] = record["score_returns"]
                partial["preceding_success_observation"] = deepcopy(
                    record.get("preceding_success_observation")
                )
                try:
                    partial["applied_updates"] = [
                        asdict(receipt) for receipt in record["stream"]._runtime.applied_updates
                    ]
                    state = record["stream"]._runtime.candidate_snapshot().state
                    partial["candidate_observation"] = _observation(state.completed_updates, state)
                except Exception as readback_error:
                    partial["readback_error"] = f"{type(readback_error).__name__}: {readback_error}"
                    partial["current_candidate_unavailable"] = True
            publish(
                active,
                status="failed",
                reason=errors[-1],
                partial_native_evidence=partial,
                wake_updates=owned[active]["budget"].updates_completed
                if active < len(owned)
                else 0,
            )
    finally:
        try:
            if sampler is not None:
                memory = asdict(sampler.snapshot())
            if owned:
                check()
        except Exception as error:
            errors.append(f"readback {type(error).__name__}: {error}")
    complete = (
        not errors
        and all(row["status"] == "complete" for row in rows)
        and initial_equal
        and neutral_equal
        and memory is not None
        and memory["peak_bytes"] <= shared.MAX_RSS
        and (not with_probes or (probe_equal and all(valid_probes(row) for row in rows)))
        and (not with_live or live_equal)
    )
    report = dict(
        protocol=protocol,
        status="complete" if complete else "incomplete",
        runs=rows,
        demonstrations=[asdict(batch.episode) for batch in training],
        global_errors=errors,
        total_seconds=monotonic() - started,
        observed_process_rss=memory,
        shared_initial_parameters_exact=initial_equal,
        pc_neutral_exact=neutral_equal,
        final_data_constructed=False,
        collected_episodes=(None if collection_started else 0) if not training else len(training),
        scientific_advantage_verified=False,
        full_R42_complete=False,
        full_R43_complete=False,
    )
    if with_probes:
        report.update(
            probe_demonstrations=[asdict(batch.episode) for group in probe_data for batch in group],
            collected_probe_episodes=(None if probe_collection_started else 0)
            if not probe_data
            else sum(len(group) for group in probe_data),
            pc_neutral_probe_exact=probe_equal,
        )
    if with_live:
        report["pc_neutral_live_exact"] = live_equal
    return report
