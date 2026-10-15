"""Fixed bounded rejected consolidations with independent cached native serving.

Transforms destroy only detached copies, serve the primed actor, then refuse.
Why compare native state separately from the cursor: failed attempts consume
identities and revisions even though candidate parameters/chemistry are intact.
Boundary RSS samples are observations, not continuous preemption or a long-run
memory guarantee. No promotion, final data, tuning, new runtime or IO is added.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
import os
from time import monotonic
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_interruption import FEATURES, FAMILIES, _model, _plain
from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_energy import Array
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.experience import (
    AppliedExperience,
    Experience,
    ExperiencePermissions,
    LabelArrival,
    LogicalClock,
)
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler, read_process_rss_bytes

_RSS_CAP = 512 * 1024**2
_WALL_CAP = 60.0
_CURSOR_CAP = 16384
_TICK = 2
_REFUSAL = "injected detached consolidation refusal"
Runtime = ActorShadowRuntime[Array, Array, Array, CoupledSnapshot]
Actor = PromotableActor[Array, Array, Array, CoupledSnapshot]


def describe_example() -> dict[str, Any]:
    """Return fixed metadata without operational/model/RSS/thread work."""
    return dict(
        protocol_id="coupled_rejected_consolidations_engineering_v1",
        families=list(FAMILIES),
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        training=dict(
            role="train",
            features=[list(row) for row in FEATURES],
            targets=[[1.0], [0.0]],
            observed_at=0,
            arrived_at=_TICK,
        ),
        external_training_settings=dict(
            learning_rate=0.02, inference_steps=2, inference_learning_rate=0.05
        ),
        native_model_constructions=12,
        native_training_calls=9,
        native_energy_evaluations=36,
        cold_actor_predictions=3,
        owned_child_launches=0,
        guarded_inference_calls=0,
        transform_callbacks=24,
        consolidation_requests=30,
        limits=dict(
            max_consolidations=8,
            max_experiences=1,
            max_training_updates=1,
            max_cursor_bytes=_CURSOR_CAP,
            max_process_rss_bytes=_RSS_CAP,
            max_wall_seconds=_WALL_CAP,
            cache_ttl_ticks=10,
            max_cache_entries=2,
        ),
        limitations="Fixed engineering stream, no final data/promotion/science. RSS is sampled at explicit boundaries, not continuous hard preemption or a long-run memory bound. Failed attempt IDs/revisions remain spent.",
    )


def _rss() -> int:
    value = read_process_rss_bytes()
    if type(value) is not int or value <= 0:
        raise RuntimeError("rejected consolidation example requires valid positive process RSS")
    return value


def _boundary(sampler: ProcessRssSampler, started: float, phase: str) -> dict[str, Any]:
    elapsed = monotonic() - started
    if not isfinite(elapsed) or elapsed < 0 or elapsed >= _WALL_CAP:
        raise RuntimeError("rejected consolidation example exceeded its monotonic 60-second bound")
    sampler.sample()
    segment = sampler.snapshot()
    if segment.peak_bytes > _RSS_CAP:
        raise RuntimeError("rejected consolidation example exceeded its sampled process RSS bound")
    return dict(phase=phase, elapsed_seconds=elapsed, rss_segment=asdict(segment))


def _encoded(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _same(left: Any, right: Any, message: str) -> None:
    if _encoded(left) != _encoded(right):
        raise AssertionError(message)


def _arrays(state: CoupledSnapshot) -> tuple[Array, ...]:
    p = state.parameters
    arrays = (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)
    if state.wake_state is not None:
        arrays += (state.wake_state.chemical_state, state.wake_state.plasticity_state)
    return arrays


def _native_bytes(state: CoupledSnapshot) -> int:
    return sum(array.nbytes for array in _arrays(state))


def _feature_digest(features: Array) -> str:
    return sha256(
        str((features.dtype.str, features.shape)).encode() + features.tobytes()
    ).hexdigest()


def _cursor(runtime: Runtime) -> tuple[dict[str, Any], int]:
    cursor = asdict(runtime.capture_consolidation_cursor())
    size = len(_encoded(cursor))
    if size > _CURSOR_CAP:
        raise AssertionError("serialized consolidation cursor exceeded16384bytes")
    return cursor, size


def _train_runtime(
    model: CoupledPredictiveCodingNetwork,
    sampler: ProcessRssSampler,
    started: float,
    memory: list[dict[str, Any]],
) -> tuple[
    Actor,
    Runtime,
    Array,
    Experience[Array],
    LabelArrival[Array],
    tuple[AppliedExperience, ...],
    ToyBudgetSession,
]:
    x = np.asarray(FEATURES, dtype=np.float64)
    y = np.array([[1.0], [0.0]], dtype=np.float64)
    for _ in range(2):
        model.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05)
    memory.append(_boundary(sampler, started, "source_warmed"))
    learner = CoupledLearner(
        model, learning_rate=0.02, inference_steps=2, inference_learning_rate=0.05
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
        max_consolidations=8,
        actor=actor,
    )
    memory.append(_boundary(sampler, started, "runtime_constructed"))
    experience = Experience(
        "train-batch-0",
        "rejected-consolidations",
        0,
        "actor-0",
        x.copy(),
        "train",
        ExperiencePermissions(training=True),
    )
    runtime.record_experience(experience)
    clock.advance_to(_TICK)
    label = LabelArrival(
        "label-0", experience.sample_id, experience.episode_id, _TICK, "actor-0", y.copy()
    )
    runtime.record_label(label)
    updates = runtime.train_ready(max_updates=1)
    if len(updates) != 1 or updates[0].applied_at != _TICK:
        raise AssertionError("delayed train stream did not commit exactly one arrived update")
    memory.append(_boundary(sampler, started, "arrived_update"))
    return actor, runtime, x, experience, label, updates, budget


def _case(family: str, sampler: ProcessRssSampler, started: float) -> dict[str, Any]:
    memory = [_boundary(sampler, started, "before_source")]
    model = _model(family)
    actor, runtime, features, experience, label, updates, budget = _train_runtime(
        model, sampler, started, memory
    )
    memory.append(_boundary(sampler, started, "runtime_ready"))
    source = model.snapshot_state()
    first = actor.serve(features.copy(), now=_TICK)
    if first.cache_hit:
        raise AssertionError("actor cache prime unexpectedly hit")
    actor_before = asdict(actor.serving_snapshot())
    if len(actor_before["cache"]) != 1:
        raise AssertionError(
            "primed actor must retain exactly one entry within its configured capacity"
        )
    candidate_before = runtime.candidate_snapshot()
    cursor_before, _ = _cursor(runtime)
    _same(actor_before["model_state"], asdict(source), "actor differs from warm source A")
    actor_bytes = _native_bytes(source)
    candidate_bytes = _native_bytes(candidate_before.state)
    callback_records: list[dict[str, Any]] = []

    def reject(state: CoupledSnapshot) -> ConsolidatedState[CoupledSnapshot]:
        weight_before = float(state.parameters.hidden_weights[0][0, 0])
        chemistry_before = (
            None if state.wake_state is None else float(state.wake_state.chemical_state[0])
        )
        size = _native_bytes(state)
        for array in _arrays(state):
            array.fill(777.0)
        if state.wake_state is not None:
            state.wake_state.chemical_state.fill(123.0)
            state.wake_state.plasticity_state.fill(0.0)
        served = actor.serve(features.copy(), now=_TICK)
        callback_records.append(
            dict(
                serving=asdict(served),
                mutation=dict(
                    weight_before=weight_before,
                    weight_after=float(state.parameters.hidden_weights[0][0, 0]),
                    chemistry_before=chemistry_before,
                    chemistry_after=None
                    if state.wake_state is None
                    else float(state.wake_state.chemical_state[0]),
                    native_array_bytes_before=size,
                    native_array_bytes_after=_native_bytes(state),
                ),
            )
        )
        raise ValueError(_REFUSAL)

    requests = [("rejected", "reject-00"), ("duplicate", "reject-00")]
    requests += [("rejected", f"reject-{index:02d}") for index in range(1, 8)]
    requests += [("exhausted", "reject-08")]
    steps = []
    for kind, event_id in requests:
        memory.append(_boundary(sampler, started, f"before_{kind}_{event_id}"))
        before_count = len(callback_records)
        try:
            runtime.consolidate(event_id, reject)
        except ValueError as error:
            refusal = str(error)
        else:
            raise AssertionError("injected consolidation request unexpectedly succeeded")
        expected = (
            _REFUSAL
            if kind == "rejected"
            else "duplicate consolidation event_id"
            if kind == "duplicate"
            else "consolidation attempt quota exhausted"
        )
        if refusal != expected or len(callback_records) != before_count + (kind == "rejected"):
            raise AssertionError("consolidation admission or detached refusal ordering changed")
        observed: dict[str, Any] = (
            callback_records[-1]
            if kind == "rejected"
            else dict(serving=asdict(actor.serve(features.copy(), now=_TICK)), mutation=None)
        )
        served = observed["serving"]
        if (
            not served["cache_hit"]
            or served["actor_version"] != first.actor_version
            or served["generation"] != first.generation
        ):
            raise AssertionError("rejected consolidation missed or changed the actor cache")
        _same(served["prediction"], first.prediction, "cached actor prediction changed")
        _same(asdict(actor.serving_snapshot()), actor_before, "full actor/cache bundle changed")
        current = runtime.candidate_snapshot()
        _same(
            asdict(current.state),
            asdict(candidate_before.state),
            "rejected transform changed full native candidate state",
        )
        if (
            current.wake_updates != 1
            or current.consolidations_completed != 0
            or _native_bytes(current.state) != candidate_bytes
        ):
            raise AssertionError("candidate wake/completion count or native array bytes changed")
        cursor, cursor_bytes = _cursor(runtime)
        boundary = _boundary(sampler, started, f"after_{kind}_{event_id}")
        memory.append(boundary)
        steps.append(
            dict(
                kind=kind,
                event_id=event_id,
                refusal=refusal,
                callback_count_before=before_count,
                callback_count_after=len(callback_records),
                cursor=cursor,
                cursor_bytes=cursor_bytes,
                serving=served,
                mutation=observed["mutation"],
                rss_segment=boundary["rss_segment"],
            )
        )
        budget.before_final()
    candidate_after = runtime.candidate_snapshot()
    actor_after = actor.serving_snapshot()
    final_cursor, _ = _cursor(runtime)
    if (
        final_cursor["attempted_ids"] != tuple(f"reject-{index:02d}" for index in range(8))
        or final_cursor["consolidations"]
        or final_cursor["stopped"]
        or final_cursor["retired"]
        or not final_cursor["payload_ready"]
        or len(callback_records) != 8
        or final_cursor["revision"] != cursor_before["revision"] + 10
    ):
        raise AssertionError("final consumed consolidation cursor differs from the bounded stream")
    _same(asdict(model.snapshot_state()), asdict(source), "source model changed")
    if source.wake_state is not None and (
        source.wake_state.wake_updates != 2
        or candidate_after.state.wake_state is None
        or candidate_after.state.wake_state.wake_updates != 3
    ):
        raise AssertionError("complete CPC wake count was lost")
    return dict(
        family=family,
        source_state=asdict(source),
        experience=asdict(experience),
        label=asdict(label),
        applied_updates=[asdict(update) for update in updates],
        actor_before=actor_before,
        actor_after=asdict(actor_after),
        candidate_before=asdict(candidate_before),
        candidate_after=asdict(candidate_after),
        cursor_before=cursor_before,
        cursor_after=final_cursor,
        steps=steps,
        measurements=dict(
            max_cursor_bytes=max(step["cursor_bytes"] for step in steps),
            native_array_bytes=dict(
                actor_before=actor_bytes,
                actor_after=_native_bytes(actor_after.model_state),
                candidate_before=candidate_bytes,
                candidate_after=_native_bytes(candidate_after.state),
            ),
            cache_entries_before=len(actor_before["cache"]),
            cache_entries_after=len(actor_after.cache),
            memory_boundaries=memory,
        ),
    )


def run_example() -> dict[str, Any]:
    started = monotonic()
    sampler = ProcessRssSampler(read_rss_bytes=_rss)
    initial = _boundary(sampler, started, "invocation_start")
    rows = [_case(family, sampler, started) for family in FAMILIES]
    for key in ("source_state",):
        _same(
            rows[0][key]["parameters"],
            rows[1][key]["parameters"],
            "PC/neutral warm parameters differ",
        )
    _same(
        rows[0]["candidate_after"]["state"]["parameters"],
        rows[1]["candidate_after"]["state"]["parameters"],
        "PC/neutral delayed candidate parameters differ",
    )
    final = _boundary(sampler, started, "invocation_end")
    return _plain(
        dict(
            **describe_example(),
            rows=rows,
            elapsed_seconds=final["elapsed_seconds"],
            initial_memory=initial,
            process_rss_segment=final["rss_segment"],
        )
    )
