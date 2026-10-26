"""One bounded count-memory control through the existing MiniGrid TRAIN runner.

Inputs are the original six public teacher traces and frozen A/B/C/A schedule.
Outputs preserve native checkpoints, raw TRAIN/live scores and observed costs.
No final/probe data, tuning, neural updates, replay, promotion or file IO occurs.
"""

from copy import deepcopy
from dataclasses import asdict
from math import isfinite
import pickle
from time import monotonic
from typing import Any

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    copy_invariant_memory_snapshot,
)
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_checkpoint,
    evaluate_invariant_memory_checkpoint,
    evaluate_invariant_fifo_checkpoint,
)
from src.app.coupled_attribution import AttributionBatch
from src.core.action_memory import ActionMemoryPolicy
from src.core.fifo_action_memory import (
    FIFO_RETIREMENT_METADATA_BYTES,
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
)
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_key_door import predefined_layouts
from src.core.minigrid_public_history import encode_public_history_episode
from src.shared.process_memory import ProcessRssSampler

BODY_SECONDS = 180.0
METHOD = "observation_memory"
INVARIANT_METHOD = "public_invariant_observation_memory"
INVARIANT_FIFO_METHOD = "public_invariant_observation_memory_fifo"


def memory_policy() -> ActionMemoryPolicy:
    return ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)


def fifo_memory_policy() -> FifoActionMemoryPolicy:
    # Two original18-key teacher tasks: a fixed capacity control, not a sweep.
    return FifoActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
    )


def _validate_invariant_option(value) -> None:
    if type(value) is not bool:
        raise ValueError("invariant_memory must be an exact bool")


def _validate_memory_options(invariant_memory, invariant_fifo_memory) -> None:
    _validate_invariant_option(invariant_memory)
    if type(invariant_fifo_memory) is not bool:
        raise ValueError("invariant_fifo_memory must be an exact bool")
    if invariant_memory and invariant_fifo_memory:
        raise ValueError("invariant_memory and invariant_fifo_memory are mutually exclusive")


def describe_protocol(
    *, invariant_memory: bool = False, invariant_fifo_memory: bool = False
) -> dict[str, Any]:
    """Declare one resource-free case, never inheriting a previous run allowance."""
    _validate_memory_options(invariant_memory, invariant_fifo_memory)
    protocol = dict(
        protocol_id="minigrid_bounded_public_history_action_memory_v1",
        classification="exploratory_known_train_supervised_count_control",
        methods=[METHOD],
        policy=asdict(memory_policy()),
        entry_payload_bytes=2424,
        seed=17,
        layouts=[asdict(layout) for layout in predefined_layouts()[:3]],
        task_sequence=list(shared.CONTEXTS),
        phase_updates=32,
        checkpoint_updates=list(shared.POINTS),
        stream=dict(training_only=True, heldout=[], label_delay=1, updates=128),
        encoding="public initial148 + current148; own reset per complete episode",
        update_kind="supervised observation/action count increments, not SGD",
        conflicts="empirical distribution; no overwrite, eviction or disambiguation",
        unseen="uniform seven, raw first-tie argmax; no mask/repair/retry",
        evaluation="all128updates finish before30 original TRAIN scores and30 known TRAIN live episodes; no final/probe/replay/promotion",
        metric_direction="CE lower; accuracy/exact_sequence/success/return higher; positive retention loss worse",
        differences=shared.describe_protocol()["differences"],
        acquisition_premise="both cues exact_sequence=1 and live success at A32/B64/C96; otherwise forgetting/recovery inconclusive",
        trainable_parameters=None,
        planned_work=dict(
            storage_updates=128,
            training_rows=1152,
            teacher_episodes=6,
            expected_teacher_steps=54,
            maximum_teacher_steps=96,
            raw_score_forwards=30,
            raw_score_rows=270,
            live_episodes=30,
            maximum_live_prediction_calls=480,
            maximum_live_steps=480,
        ),
        limits=dict(
            body_seconds=BODY_SECONDS,
            parent_seconds=210,
            process_rss_bytes=shared.RSS_LIMIT,
            output_bytes=32 * 1024**2,
            cpu_threads=1,
            gpu=0,
        ),
        limitations="known deterministic TRAIN recall only; unequal capacity/memory/compute to neural controls; no generalization, recurrent RL or CPC advantage established",
    )
    if invariant_memory or invariant_fifo_memory:
        protocol.update(
            protocol_id="minigrid_bounded_invariant_history_action_memory_v1",
            methods=[INVARIANT_METHOD],
            policy=asdict(InvariantMemoryPolicy(memory_policy())),
            model_encoding=INVARIANT_HISTORY_ENCODING_ID,
            encoding="raw public initial148+current148 admitted by the fixed invariant wrapper; no raw-key relabeling",
            computation="existing public canonical projection plus count increments/lookups; no Torch/SGD; validation/copies/RSS additional",
        )
        protocol["planned_work"].update(
            canonical_training_rows=1152, canonical_score_rows=270, maximum_canonical_live_rows=480
        )
    if invariant_fifo_memory:
        protocol.update(
            protocol_id="minigrid_bounded_invariant_fifo36_history_action_memory_v1",
            methods=[INVARIANT_FIFO_METHOD],
            policy=asdict(InvariantMemoryPolicy(fifo_memory_policy())),
            conflicts="empirical counts; insertion-order FIFO retirement; hits do not refresh order",
            memory_retirement=dict(
                retained_entries=36,
                retained_logical_payload_bytes=87264,
                retirement_metadata_bytes=FIFO_RETIREMENT_METADATA_BYTES,
                accounting="retained exposure plus retired examples equals lifetime training rows; logical table excludes metadata, copies and Python/RSS",
            ),
        )
    return protocol


def _history_packets(packets: tuple[AttributionBatch, ...]) -> tuple[AttributionBatch, ...]:
    encoded = []
    for packet in packets:
        episode = encode_public_history_episode(shared._episode(packet))
        features, targets = episode_arrays(episode)
        encoded.append(
            AttributionBatch("train", packet.context, packet.row_ids, features, targets, episode, 7)
        )
    return tuple(encoded)


def _learner(method: str) -> ActionMemoryLearner:
    if method != METHOD:
        raise ValueError("this fixed control admits only observation_memory")
    return ActionMemoryLearner(memory_policy())


def _invariant_learner(method: str) -> InvariantActionMemoryLearner:
    if type(method) is not str or method != INVARIANT_METHOD:
        raise ValueError("invariant control requires its explicit bound method")
    return InvariantActionMemoryLearner(InvariantMemoryPolicy(memory_policy()))


def _invariant_fifo_learner(method: str) -> InvariantActionMemoryLearner:
    if type(method) is not str or method != INVARIANT_FIFO_METHOD:
        raise ValueError("invariant FIFO control requires its explicit bound method")
    return InvariantActionMemoryLearner(InvariantMemoryPolicy(fifo_memory_policy()))


def _observe_checkpoint(point: int, state, *, invariant_memory: bool) -> dict[str, Any]:
    owned = copy_invariant_memory_snapshot(state) if invariant_memory else state
    native: Any = owned.native_state if invariant_memory else owned
    result = dict(
        updates=point,
        native_training_examples=owned.training_examples,
        entries=len(native.entries),
        logical_payload_bytes=len(native.entries) * 2424,
        snapshot=asdict(owned),
    )
    if invariant_memory:
        result["model_encoding"] = INVARIANT_HISTORY_ENCODING_ID
        if type(native) is FifoActionMemorySnapshot:
            result.update(
                evictions=native.evictions,
                retired_examples=native.retired_examples,
                retained_examples=sum(sum(entry.counts) for entry in native.entries),
                retirement_metadata_bytes=FIFO_RETIREMENT_METADATA_BYTES,
            )
    return result


def run_pilot(
    *, invariant_memory: bool = False, invariant_fifo_memory: bool = False
) -> dict[str, Any]:
    """Run one frozen case and retain a failed prefix without renewing its budget."""
    _validate_memory_options(invariant_memory, invariant_fifo_memory)
    bound_memory = invariant_memory or invariant_fifo_memory
    started = monotonic()
    method = (
        INVARIANT_FIFO_METHOD
        if invariant_fifo_memory
        else INVARIANT_METHOD
        if invariant_memory
        else METHOD
    )
    row = shared._row(method)
    row["final_evaluation"] = dict(
        status="not_applicable", role="final_test", reason="no final data in this control"
    )
    report: dict[str, Any] = dict(
        protocol=describe_protocol(invariant_fifo_memory=True)
        if invariant_fifo_memory
        else describe_protocol(invariant_memory=True)
        if invariant_memory
        else describe_protocol(),
        status="running",
        reason=None,
        training_sources=[],
        source_reports=[],
        methods=[row],
        observations=[],
        total_seconds=None,
        observed_process_rss=None,
        scientific_advantage_verified=False,
    )
    protocol = report["protocol"]
    protocol_bytes = pickle.dumps(protocol, protocol=5)
    methods = report["methods"]
    readouts, final_slot = row["readouts"], row["final_evaluation"]
    slots = list(readouts)
    saved = [pickle.dumps(slot, protocol=5) for slot in slots]
    final_bytes = pickle.dumps(final_slot, protocol=5)
    original_sources, original_reports = report["training_sources"], report["source_reports"]
    source_bytes = source_report_bytes = None
    cohorts: list[tuple[tuple[AttributionBatch, ...], Any]] = []
    sampler = arm = None
    observations: list[dict[str, Any]] = []
    paid_training: bytes | None = None
    pending_readout: tuple[int, bytes] | None = None
    rendered_result: dict[str, Any] | None = None
    active_readout: int | None = None
    lost_readout = False

    def common_guard() -> None:
        if monotonic() - started > BODY_SECONDS:
            raise ValueError("fixed memory whole-case wall allowance exhausted")
        if sampler is not None:
            sampler.sample()
            if sampler.snapshot().peak_bytes > shared.RSS_LIMIT:
                raise ValueError("fixed memory sampled RSS allowance exhausted")
        if (
            report["protocol"] is not protocol
            or pickle.dumps(protocol, protocol=5) != protocol_bytes
            or report["methods"] is not methods
            or len(methods) != 1
            or methods[0] is not row
            or row["method"] != method
            or row["readouts"] is not readouts
            or len(readouts) != len(shared.POINTS)
            or row["final_evaluation"] is not final_slot
            or pickle.dumps(final_slot, protocol=5) != final_bytes
            or report["training_sources"] is not original_sources
            or report["source_reports"] is not original_reports
        ):
            raise ValueError("memory protocol, original sources or publication slots changed")
        for actual, expected, payload in zip(readouts, slots, saved, strict=True):
            if actual is not expected or (
                type(actual) is not bytes and pickle.dumps(actual, protocol=5) != payload
            ):
                raise ValueError("preceding memory readout was replaced or corrupted")
        if source_bytes is not None and (
            pickle.dumps(original_sources, protocol=5) != source_bytes
            or pickle.dumps(original_reports, protocol=5) != source_report_bytes
            or any(not shared._check_signature(data, signature) for data, signature in cohorts)
        ):
            raise ValueError("original memory TRAIN evidence changed")

    def completed_guard() -> None:
        common_guard()
        if arm is not None:
            arm["guard"]()
        common_guard()

    try:
        with ProcessRssSampler() as active_sampler:
            sampler = active_sampler
            common_guard()
            sources = shared._collect_sources(common_guard, original_reports)
            packets = shared._packets(sources)
            history = _history_packets(packets)
            original_sources.extend(asdict(shared._episode(packet)) for packet in sources)
            source_bytes, source_report_bytes = (
                pickle.dumps(original_sources, protocol=5),
                pickle.dumps(original_reports, protocol=5),
            )
            cohorts = [(data, shared._signature(data)) for data in (sources, packets, history)]
            training_options: dict[str, Any] = (
                dict(invariant_fifo36=True) if invariant_fifo_memory else {}
            )
            arm = shared._train_arm(
                row,
                history,
                sampler,
                started + BODY_SECONDS,
                common_guard,
                learner_factory=(
                    _invariant_fifo_learner
                    if invariant_fifo_memory
                    else _invariant_learner
                    if invariant_memory
                    else _learner
                ),
                **training_options,
            )
            completed_guard()
            if bound_memory:
                paid_training = pickle.dumps(row, protocol=5)
            observations = [
                _observe_checkpoint(point, state, invariant_memory=bound_memory)
                for point, state in arm["states"]
            ]
            for index, (point, state) in enumerate(arm["states"]):
                completed_guard()
                try:
                    if bound_memory:
                        active_readout = index
                        evaluator = (
                            evaluate_invariant_fifo_checkpoint
                            if invariant_fifo_memory
                            else evaluate_invariant_memory_checkpoint
                        )
                        readout = evaluator(
                            arm["source"],
                            state,
                            updates=point,
                            layouts=predefined_layouts()[:3],
                            role="train",
                            sources=sources,
                            budget=arm["budget"],
                            check_boundary=completed_guard,
                        )
                        # Returned work is paid before any subsequent callback.
                        pending_readout = (index, pickle.dumps(readout, protocol=5))
                        active_readout = None
                    else:
                        readout = evaluate_checkpoint(
                            arm["source"],
                            state,
                            method=METHOD,
                            updates=point,
                            layouts=predefined_layouts()[:3],
                            role="train",
                            sources=sources,
                            budget=arm["budget"],
                            check_boundary=completed_guard,
                            public_history=True,
                        )
                except MiniGridEvaluationFailed as error:
                    payload = pickle.dumps(error.report, protocol=5)
                    active_readout = None
                    readouts[index] = slots[index] = payload
                    saved[index] = payload
                    raise
                completed_guard()
                payload = (
                    pending_readout[1]
                    if pending_readout is not None
                    else pickle.dumps(readout, protocol=5)
                )
                # Why: retain exactly five immutable private reports, not an
                # ever-growing chain or mutable public copies between callbacks.
                readouts[index] = slots[index] = payload
                saved[index] = payload
                pending_readout = None
                completed_guard()
            completed_guard()
            rendered = shared._restore_readouts(row, saved, final_bytes)
            outcomes, metrics, acquired = shared._outcomes(rendered["readouts"])
            rendered.update(
                status="complete",
                outcomes=outcomes,
                signed_metrics=metrics,
                acquisition_conditions=acquired,
            )
            rendered["work"] = shared._work(rendered)
            if bound_memory:
                rendered_result = rendered
            else:
                report.update(status="complete", methods=[rendered], observations=observations)
    except Exception as error:
        report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        if bound_memory and active_readout is not None:
            lost_readout = True
            marker = dict(
                updates=shared.POINTS[active_readout],
                method=method,
                role="train",
                status="failed",
                reason=report["reason"],
                evaluator_admitted=True,
                receipt_unavailable=True,
                work_unknown=True,
                scores=[],
                episodes=[],
            )
            pending_readout = (active_readout, pickle.dumps(marker, protocol=5))
        trusted = list(saved)
        if pending_readout is not None:
            trusted[pending_readout[0]] = pending_readout[1]
        restored = shared._restore_readouts(
            pickle.loads(paid_training) if paid_training is not None else row, trusted, final_bytes
        )
        restored["status"] = "incomplete"
        restored["work"] = shared._work(restored)
        if bound_memory:
            rendered_result = restored
        else:
            report["methods"] = [restored]
            report["observations"] = deepcopy(observations)
    finally:
        if bound_memory:
            # Why: final sampler ports cannot replace a proved source or erase
            # already-returned paid readouts before structured publication.
            closing_error: Exception | None = None
            elapsed: float | None = None
            try:
                completed_guard()
            except Exception as error:
                closing_error = error
            try:
                if sampler is not None:
                    report["observed_process_rss"] = asdict(sampler.snapshot())
                ended = monotonic()
                if type(ended) not in (int, float) or not isfinite(ended) or ended < started:
                    raise ValueError("final count-control clock must be finite and monotonic")
                elapsed = float(ended - started)
            except Exception as error:
                closing_error = closing_error or error
            finally:
                try:
                    completed_guard()
                except Exception as error:
                    closing_error = closing_error or error
            if closing_error is not None:
                report.update(
                    status="incomplete",
                    reason=report["reason"]
                    or f"closure {type(closing_error).__name__}: {closing_error}",
                )
                trusted = list(saved)
                if pending_readout is not None:
                    trusted[pending_readout[0]] = pending_readout[1]
                rendered_result = shared._restore_readouts(
                    pickle.loads(paid_training) if paid_training is not None else row,
                    trusted,
                    final_bytes,
                )
                rendered_result["status"] = "incomplete"
                rendered_result["work"] = shared._work(rendered_result)
                rendered_result["work"]["failure_work_may_be_unobserved"] = True
            if rendered_result is not None:
                report["methods"] = [rendered_result]
                report["observations"] = deepcopy(observations)
                if report["status"] == "running":
                    report["status"] = "complete"
            report["total_seconds"] = elapsed
            if rendered_result is not None:
                work = rendered_result["work"]
                work.update(
                    count_update_calls=work["native_train_calls"],
                    count_training_rows=work["training_rows"],
                    neural_sgd_updates=0,
                    committed_training_projection_rows=work["training_rows"],
                    projection_scope="readout arrays are returned projections; training rows cover committed updates only; failed-attempt allocations/copies/Python/RSS additional or unknown",
                )
                if invariant_fifo_memory:
                    last = observations[-1] if observations else {}
                    partial = rendered_result.get("partial_stream") or {}
                    work.update(
                        count_update_calls=partial.get("native_candidate_updates"),
                        charged_count_updates=work["native_train_calls"],
                        committed_training_projection_rows=partial.get(
                            "committed_training_examples"
                        ),
                        observed_native_training_projection_rows=partial.get(
                            "native_candidate_training_examples"
                        ),
                        evictions=last.get("evictions"),
                        retired_examples=last.get("retired_examples"),
                        retained_examples=last.get("retained_examples"),
                        maximum_retained_entries=max(
                            (item["entries"] for item in observations), default=None
                        ),
                        maximum_retained_logical_payload_bytes=max(
                            (item["logical_payload_bytes"] for item in observations), default=None
                        ),
                        retirement_metadata_bytes=FIFO_RETIREMENT_METADATA_BYTES,
                        retirement_observation_updates=last.get("updates"),
                        retirement_observation_scope="last complete typed captured checkpoint and checkpoint occupancy maxima; partial-training retirement unknown without a captured checkpoint; metadata/copies/Python/RSS are additional to retained table bytes",
                    )
                if lost_readout:
                    lower: dict[str, int] = {}
                    for readout in rendered_result["readouts"]:
                        for key, value in readout.get("work", {}).items():
                            if type(value) is int and value >= 0:
                                lower[key] = lower.get(key, 0) + value
                    work["known_readout_work_lower_bounds"] = lower
                    for key in set(lower) | {
                        "raw_score_forwards_attempted",
                        "raw_score_forwards_completed",
                        "raw_score_rows_attempted",
                        "raw_score_rows_completed",
                        "live_prediction_calls_attempted",
                        "live_prediction_calls_returned",
                        "environment_resets_attempted",
                        "environment_resets_completed",
                        "environment_steps_attempted",
                        "environment_steps_returned",
                        "internal_projection_rows_returned",
                        "internal_projection_array_bytes_returned",
                    }:
                        work[key] = None
                    work["failure_work_may_be_unobserved"] = True
        else:
            report["total_seconds"] = monotonic() - started
            if sampler is not None:
                report["observed_process_rss"] = asdict(sampler.snapshot())
    return report
