"""Finite growth/refusal endurance with bounded streaming engineering evidence.

Existing growth helpers and native ports perform the mathematics. Every request
and cached read is checked; evidence retains counters, eight blocks and six tiny
samples rather than per-request payloads. Why a new clock boundary: this frozen
workload has an explicit 240-second allowance, independent of the earlier demo.
Observed boundary RSS does not establish continuous preemption, hours of uptime,
scientific benefit, full recovery or actor promotion.
"""

from dataclasses import asdict
from math import isfinite
import os
from time import monotonic
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_growth_limit import (
    Actor,
    Runtime,
    _cursor,
    _encoded,
    _feature_digest,
    _native_bytes,
    _same,
    _source,
    describe_example as _growth_description,
)
from src.adapters.coupled_checkpoint_interruption import FEATURES, FAMILIES, _plain
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler, read_process_rss_bytes

_WALL_CAP = 240.0
_RSS_CAP = 512 * 1024**2
_SUMMARY_CAP = 16384
_NATIVE_CAP = 968


def _controls(attempt_limit: int, reads_per_request: int) -> None:
    if type(attempt_limit) is not int or not 4 <= attempt_limit <= 128:
        raise ValueError("attempt_limit must be an exact integer in4..128")
    if type(reads_per_request) is not int or not 2 <= reads_per_request <= 64:
        raise ValueError("reads_per_request must be an exact integer in2..64")


def _description(attempt_limit: int, reads_per_request: int) -> dict[str, Any]:
    report = _growth_description()
    report.update(
        protocol_id="coupled_growth_endurance_engineering_v1",
        run_kind="full_endurance"
        if (attempt_limit, reads_per_request) == (128, 64)
        else "engineering_control",
        attempt_limit=attempt_limit,
        reads_per_request=reads_per_request,
        consolidation_requests=3 * (attempt_limit + 2),
        split_callbacks=3 * attempt_limit,
        serve_calls=3 * ((attempt_limit + 2) * reads_per_request + 2),
        cache_hits=3 * ((attempt_limit + 2) * reads_per_request + 1),
    )
    report["limits"].update(
        max_consolidations=attempt_limit,
        max_wall_seconds=_WALL_CAP,
        max_summary_bytes=_SUMMARY_CAP,
        max_native_array_bytes=_NATIVE_CAP,
        max_blocks=8,
        max_samples=6,
    )
    report["limitations"] = (
        "Finite fixed engineering endurance, not scientific quality or timing evidence. Streaming evidence covers every request/read, with point-sampled RSS rather than continuous preemption or a long-run/recovery guarantee. No actor promotion."
    )
    return report


def describe_example() -> dict[str, Any]:
    return _description(128, 64)


def _rss() -> int:
    value = read_process_rss_bytes()
    if type(value) is not int or value <= 0:
        raise RuntimeError("endurance requires valid positive process RSS")
    return value


def _boundary(sampler: ProcessRssSampler, started: float) -> dict[str, Any]:
    elapsed = monotonic() - started
    if not isfinite(elapsed) or not 0 <= elapsed < _WALL_CAP:
        raise RuntimeError("endurance exceeded its global240second boundary")
    sampler.sample()
    segment = sampler.snapshot()
    if segment.peak_bytes > _RSS_CAP:
        raise RuntimeError("endurance exceeded its observed512MiB RSS boundary")
    return dict(elapsed_seconds=elapsed, process_rss_segment=asdict(segment))


def _bounded_summary(value: Any) -> int:
    size = len(_encoded(value))
    if size > _SUMMARY_CAP:
        raise ValueError("aggregate endurance summaries exceeded16384bytes")
    return size


def _case(
    family: str,
    sampler: ProcessRssSampler,
    started: float,
    attempt_limit: int,
    reads_per_request: int,
) -> dict[str, Any]:
    start = _boundary(sampler, started)
    features = np.asarray(FEATURES, dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    source = _source(family, features, targets)
    source_record = asdict(source)
    _boundary(sampler, started)
    learner = StructuralCoupledScalingLearner(
        source, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    actor = Actor(
        learner,
        version="actor-0",
        configuration=ServingConfiguration(10, 2),
        feature_digest=_feature_digest,
        metadata={"route": ["original"]},
    )
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=1, max_wall_seconds=_WALL_CAP, max_process_rss_bytes=_RSS_CAP
        ),
        monotonic,
    )
    budget.attach_memory(sampler)
    runtime = Runtime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=attempt_limit,
        actor=actor,
    )
    experience = Experience(
        "train-batch-0",
        "growth-endurance",
        0,
        "actor-0",
        features.copy(),
        "train",
        ExperiencePermissions(training=True),
    )
    runtime.record_experience(experience)
    clock.advance_to(2)
    prime = actor.serve(features.copy(), now=2)
    if prime.cache_hit:
        raise AssertionError("endurance cache prime unexpectedly hit")
    actor_before = asdict(actor.serving_snapshot())
    initial = runtime.candidate_snapshot()
    _same(
        actor_before["model_state"]["raw_state"],
        source_record,
        "endurance actor differs from warm source",
    )
    _same(
        asdict(initial.state.raw_state),
        source_record,
        "endurance candidate differs from warm source",
    )
    initial_cursor, initial_size = _cursor(runtime)
    counters = dict(
        requests=0,
        unique_attempts=0,
        accepted=0,
        width_refusals=0,
        duplicates=0,
        exhausted=0,
        callbacks=0,
        serve_calls=1,
        cache_hits=0,
        callback_cache_hits=0,
    )
    blocks: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    max_cursor_bytes = initial_size
    max_native_bytes = max(_native_bytes(source), _native_bytes(initial.state.raw_state))
    capped = None
    splitter = CoupledScalingSplitConsolidator(max_width=8)

    def read_cache(*, callback: bool = False) -> None:
        result = actor.serve(features.copy(), now=2)
        counters["serve_calls"] += 1
        if (
            not result.cache_hit
            or result.actor_version != prime.actor_version
            or result.generation != prime.generation
        ):
            raise AssertionError("endurance cached serving identity/hit changed")
        _same(result.prediction, prime.prediction, "endurance cached prediction changed")
        _same(result.metadata, prime.metadata, "endurance cached metadata changed")
        counters["cache_hits"] += 1
        counters["callback_cache_hits"] += int(callback)

    def split(state: CoupledStructuralState) -> ConsolidatedState[CoupledStructuralState]:
        counters["callbacks"] += 1
        try:
            return splitter(state)
        finally:
            # Every admitted callback serves once, including refused width16.
            read_cache(callback=True)

    def request(kind: str, event_id: str, unique_number: int) -> None:
        nonlocal capped, max_cursor_bytes, max_native_bytes
        _boundary(sampler, started)
        before_callbacks = counters["callbacks"]
        reason = None
        receipt = None
        try:
            receipt = runtime.consolidate(event_id, split)
        except ValueError as error:
            reason = str(error)
        expected = (
            None
            if kind == "accepted"
            else "split target width exceeds configured ceiling"
            if kind == "width_refused"
            else "duplicate consolidation event_id"
            if kind == "duplicate"
            else "consolidation attempt quota exhausted"
        )
        admitted = kind in ("accepted", "width_refused")
        if reason != expected or counters["callbacks"] != before_callbacks + admitted:
            raise AssertionError("endurance admission/refusal ordering changed")
        if (receipt is not None) != (kind == "accepted"):
            raise AssertionError("endurance receipt acceptance changed")
        for _ in range(reads_per_request - int(admitted)):
            read_cache()
        counters["requests"] += 1
        counters["unique_attempts"] += int(admitted)
        if kind == "accepted":
            counters["accepted"] += 1
        elif kind == "width_refused":
            counters["width_refusals"] += 1
        else:
            counters["duplicates" if kind == "duplicate" else "exhausted"] += 1
        current = runtime.candidate_snapshot()
        if kind == "accepted" and unique_number == 2:
            capped = current
        if capped is not None:
            _same(
                asdict(current.state),
                asdict(capped.state),
                "endurance refusal changed capped raw/history",
            )
        if current.state.raw_state.config.width != min(2 ** (counters["accepted"] + 1), 8):
            raise AssertionError("endurance current width differs from accepted growth")
        _same(asdict(actor.serving_snapshot()), actor_before, "endurance full actor/cache changed")
        cursor, size = _cursor(runtime)
        max_cursor_bytes = max(max_cursor_bytes, size)
        max_native_bytes = max(max_native_bytes, _native_bytes(current.state.raw_state))
        if max_native_bytes > _NATIVE_CAP or len(actor_before["cache"]) != 1:
            raise AssertionError("endurance native byte or cache-entry bound exceeded")
        expected_ids = tuple(
            f"split-{index:03d}" for index in range(1, counters["unique_attempts"] + 1)
        )
        if (
            cursor["attempted_ids"] != expected_ids
            or len(cursor["consolidations"]) != counters["accepted"]
            or cursor["stopped"]
            or cursor["retired"]
            or not cursor["payload_ready"]
            or cursor["revision"] != initial_cursor["revision"] + counters["requests"]
            or current.wake_updates != 0
            or current.consolidation_attempts != counters["unique_attempts"]
            or current.consolidations_completed != counters["accepted"]
        ):
            raise AssertionError("endurance cursor/committed-work counters changed")
        boundary = _boundary(sampler, started)
        if admitted and (unique_number % 16 == 0 or unique_number == attempt_limit):
            blocks.append(
                dict(
                    unique_through=unique_number,
                    counters=dict(counters),
                    rss_peak_bytes=boundary["process_rss_segment"]["peak_bytes"],
                )
            )
        if kind in ("duplicate", "exhausted") or unique_number in (1, 2, 3, attempt_limit):
            samples.append(
                dict(
                    kind=kind,
                    event_id=event_id,
                    requested_width=4 if unique_number == 1 else 8 if unique_number == 2 else 16,
                    accepted_width=current.state.raw_state.config.width
                    if kind == "accepted"
                    else None,
                    refusal=reason,
                    callbacks=counters["callbacks"],
                    serve_calls=counters["serve_calls"],
                )
            )
        if len(blocks) > 8 or len(samples) > 6:
            raise AssertionError("endurance retained too many block/sample records")
        budget.before_final()

    for unique in range(1, attempt_limit + 1):
        request("accepted" if unique <= 2 else "width_refused", f"split-{unique:03d}", unique)
        if unique == 3:
            request("duplicate", "split-003", unique)
    request("exhausted", f"split-{attempt_limit + 1:03d}", attempt_limit + 1)
    assert capped is not None
    # Keep the original width-eight payload witness above for every refusal;
    # report the actual pre-continuation snapshot with all attempts already spent.
    capped_after_stream = runtime.candidate_snapshot()
    label = LabelArrival(
        "label-0", experience.sample_id, experience.episode_id, 2, "actor-0", targets.copy()
    )
    runtime.record_label(label)
    applied = runtime.train_ready(max_updates=1)
    final = runtime.candidate_snapshot()
    if (
        len(applied) != 1
        or final.wake_updates != 1
        or final.state.raw_state.config != capped.state.raw_state.config
    ):
        raise AssertionError("endurance lost original delayed training allowance/config")
    _same(
        asdict(final.state.origin_config),
        asdict(capped.state.origin_config),
        "endurance continuation lost origin",
    )
    _same(
        [asdict(item) for item in final.state.transitions],
        [asdict(item) for item in capped.state.transitions],
        "endurance continuation lost tags",
    )
    wake = final.state.raw_state.wake_state
    if wake is not None and (
        capped.state.raw_state.wake_state is None
        or capped.state.raw_state.wake_state.wake_updates != 2
        or wake.wake_updates != 3
    ):
        raise AssertionError("endurance lost complete CPC count")
    read_cache()
    actor_after = actor.serving_snapshot()
    _same(asdict(actor_after), actor_before, "endurance continuation changed actor/cache")
    _same(asdict(source), source_record, "endurance changed source")
    final_cursor, size = _cursor(runtime)
    max_cursor_bytes = max(max_cursor_bytes, size)
    max_native_bytes = max(max_native_bytes, _native_bytes(final.state.raw_state))
    if (
        counters
        != dict(
            requests=attempt_limit + 2,
            unique_attempts=attempt_limit,
            accepted=2,
            width_refusals=attempt_limit - 2,
            duplicates=1,
            exhausted=1,
            callbacks=attempt_limit,
            serve_calls=(attempt_limit + 2) * reads_per_request + 2,
            cache_hits=(attempt_limit + 2) * reads_per_request + 1,
            callback_cache_hits=attempt_limit,
        )
        or max_native_bytes > _NATIVE_CAP
        or final_cursor["stopped"]
        or final_cursor["retired"]
        or not final_cursor["payload_ready"]
        or final.consolidation_attempts != attempt_limit
        or final.consolidations_completed != 2
        or final_cursor["attempted_ids"]
        != tuple(f"split-{index:03d}" for index in range(1, attempt_limit + 1))
        or len(final_cursor["consolidations"]) != 2
        or final_cursor["revision"] != initial_cursor["revision"] + counters["requests"] + 2
    ):
        raise AssertionError("endurance final streaming counts or resource bound changed")
    budget.before_final()
    end = _boundary(sampler, started)
    summary = dict(
        family=family,
        counters=counters,
        blocks=blocks,
        samples=samples,
        max_cursor_bytes=max_cursor_bytes,
        max_native_array_bytes=max_native_bytes,
        cache_entries=len(actor_after.cache),
        process_rss_segment=end["process_rss_segment"],
        start_elapsed_seconds=start["elapsed_seconds"],
        end_elapsed_seconds=end["elapsed_seconds"],
    )
    _bounded_summary(summary)
    return dict(
        family=family,
        source_state=source_record,
        actor_before=actor_before,
        actor_after=asdict(actor_after),
        candidate_initial=asdict(initial),
        candidate_capped=asdict(capped_after_stream),
        candidate_final=asdict(final),
        final_cursor=final_cursor,
        experience=asdict(experience),
        label=asdict(label),
        applied_updates=[asdict(item) for item in applied],
        summary=summary,
    )


def _run(*, attempt_limit: int = 128, reads_per_request: int = 64) -> dict[str, Any]:
    _controls(attempt_limit, reads_per_request)
    started = monotonic()
    sampler = ProcessRssSampler(read_rss_bytes=_rss)
    _boundary(sampler, started)
    rows = [
        _case(family, sampler, started, attempt_limit, reads_per_request) for family in FAMILIES
    ]
    for phase in ("candidate_initial", "candidate_capped", "candidate_final"):
        _same(
            rows[0][phase]["state"]["raw_state"]["raw_parameters"],
            rows[1][phase]["state"]["raw_state"]["raw_parameters"],
            "PC/neutral endurance raw parameters differ",
        )
    summaries = [row["summary"] for row in rows]
    totals = {
        key: sum(row["summary"]["counters"][key] for row in rows)
        for key in rows[0]["summary"]["counters"]
    }
    summary_bytes = _bounded_summary(summaries)
    end = _boundary(sampler, started)
    return _plain(
        dict(
            **_description(attempt_limit, reads_per_request),
            rows=rows,
            totals=totals,
            summary_bytes=summary_bytes,
            elapsed_seconds=end["elapsed_seconds"],
            process_rss_segment=end["process_rss_segment"],
        )
    )


def run_example() -> dict[str, Any]:
    return _run(attempt_limit=128, reads_per_request=64)
