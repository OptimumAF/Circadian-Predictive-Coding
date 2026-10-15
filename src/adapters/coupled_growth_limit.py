"""Fixed raw structural growth, ceiling refusal and delayed native continuation.

The existing trusted split consolidator enforces the configured width ceiling;
the original runtime retains spent IDs and the original training allowance.
Why separate state from cursor: a refused split changes attempt/revision metadata
while preserving the full current width/precision/raw/CPC/history state. Cached
actor serving remains independent. No promotion, scientific result, new math,
continuous memory preemption or long-run resource guarantee is provided.
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

from src.adapters.coupled_checkpoint_interruption import FEATURES, FAMILIES, _plain
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import Array
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    initialize_coupled_scaling,
    step_coupled_scaling,
)
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler, read_process_rss_bytes

_RSS_CAP = 512 * 1024**2
_WALL_CAP = 60.0
_CURSOR_CAP = 16384
_WIDTH_CAP = 8
_TICK = 2
Runtime = ActorShadowRuntime[Array, Array, Array, CoupledStructuralState]
Actor = PromotableActor[Array, Array, Array, CoupledStructuralState]


def describe_example() -> dict[str, Any]:
    """Describe fixed controls without models, IO, RSS reads, threads or processes."""
    return dict(
        protocol_id="coupled_growth_limit_engineering_v1",
        families=list(FAMILIES),
        seed=23,
        geometry=dict(input_dim=2, width=2, hidden_layers=2, precisions=[1.0, 1.0]),
        scheme="mean_field",
        gamma0=1.5,
        training=dict(
            role="train",
            features=[list(row) for row in FEATURES],
            targets=[[1.0], [0.0]],
            observed_at=0,
            arrived_at=_TICK,
        ),
        external_training_settings=dict(
            learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        ),
        native_training_calls=9,
        raw_sgd_steps=9,
        native_energy_evaluations=36,
        cold_actor_predictions=3,
        owned_child_launches=0,
        guarded_inference_calls=0,
        consolidation_requests=15,
        split_callbacks=9,
        successful_splits=6,
        limits=dict(
            max_width=_WIDTH_CAP,
            max_consolidations=3,
            max_training_updates=1,
            max_experiences=1,
            max_cursor_bytes=_CURSOR_CAP,
            max_wall_seconds=_WALL_CAP,
            max_process_rss_bytes=_RSS_CAP,
            cache_ttl_ticks=10,
            max_cache_entries=2,
        ),
        limitations="Fixed engineering integration. Width-dependent raw SGD factors change; no trajectory, learning-benefit or actor-promotion claim. RSS is observed at boundaries, not continuous hard preemption or a long-run guarantee.",
    )


def _rss() -> int:
    value = read_process_rss_bytes()
    if type(value) is not int or value <= 0:
        raise RuntimeError("growth example requires valid positive process RSS")
    return value


def _boundary(sampler: ProcessRssSampler, started: float, phase: str) -> dict[str, Any]:
    elapsed = monotonic() - started
    if not isfinite(elapsed) or not 0 <= elapsed < _WALL_CAP:
        raise RuntimeError("growth example exceeded its monotonic 60-second bound")
    sampler.sample()
    segment = sampler.snapshot()
    if segment.peak_bytes > _RSS_CAP:
        raise RuntimeError("growth example exceeded its sampled process RSS bound")
    return dict(phase=phase, elapsed_seconds=elapsed, rss_segment=asdict(segment))


def _encoded(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _same(left: Any, right: Any, message: str) -> None:
    if _encoded(left) != _encoded(right):
        raise AssertionError(message)


def _native_bytes(state: CoupledScalingState) -> int:
    p = state.raw_parameters
    arrays = (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)
    if state.wake_state is not None:
        arrays += (state.wake_state.chemical_state, state.wake_state.plasticity_state)
    return sum(array.nbytes for array in arrays)


def _scaling(state: CoupledScalingState) -> dict[str, Any]:
    config = state.config
    return dict(
        width=config.width,
        precisions=config.precisions,
        forward_scales=config.forward_scales,
        raw_learning_rate_factor=config.raw_learning_rate_factor,
        bias_scale=config.bias_scale,
        effective_learning_rate_factors=config.effective_learning_rate_factors,
        effective_bias_learning_rate_factor=config.effective_bias_learning_rate_factor,
    )


def _feature_digest(features: Array) -> str:
    return sha256(
        str((features.dtype.str, features.shape)).encode() + features.tobytes()
    ).hexdigest()


def _cursor(runtime: Runtime) -> tuple[dict[str, Any], int]:
    cursor = asdict(runtime.capture_consolidation_cursor())
    size = len(_encoded(cursor))
    if size > _CURSOR_CAP:
        raise AssertionError("growth cursor exceeded16384serializedbytes")
    return cursor, size


def _source(family: str, features: Array, targets: Array) -> CoupledScalingState:
    policy = (
        CoupledCircadianConfig.matched_pc_control()
        if family == "neutral_cpc"
        else CoupledCircadianConfig()
        if family == "active_cpc"
        else None
    )
    state = initialize_coupled_scaling(
        CoupledScalingConfig(
            "mean_field", 2, 2, 2, gamma0=1.5, precisions=(1.0, 1.0), circadian=policy
        ),
        23,
    )
    for _ in range(2):
        state = step_coupled_scaling(state, features.copy(), targets.copy(), 0.01, 2, 0.05).state
    return state


def _case(family: str, sampler: ProcessRssSampler, started: float) -> dict[str, Any]:
    memory = [_boundary(sampler, started, "before_source")]
    features = np.asarray(FEATURES, dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    source = _source(family, features, targets)
    source_before = asdict(source)
    memory.append(_boundary(sampler, started, "source_warmed"))
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
        max_consolidations=3,
        actor=actor,
    )
    memory.append(_boundary(sampler, started, "runtime_constructed"))
    experience = Experience(
        "train-batch-0",
        "growth-limit",
        0,
        "actor-0",
        features.copy(),
        "train",
        ExperiencePermissions(training=True),
    )
    runtime.record_experience(experience)
    clock.advance_to(_TICK)
    prime = actor.serve(features.copy(), now=_TICK)
    if prime.cache_hit:
        raise AssertionError("growth actor cache prime unexpectedly hit")
    actor_before = asdict(actor.serving_snapshot())
    if len(actor_before["cache"]) != 1:
        raise AssertionError("growth actor cache must contain one entry within capacity2")
    candidate_before = runtime.candidate_snapshot()
    cursor_before, _ = _cursor(runtime)
    splitter = CoupledScalingSplitConsolidator(max_width=_WIDTH_CAP)
    callbacks = 0

    def split(state: CoupledStructuralState) -> ConsolidatedState[CoupledStructuralState]:
        nonlocal callbacks
        callbacks += 1
        return splitter(state)

    steps: list[dict[str, Any]] = []
    for kind, event_id in (
        ("accepted", "split-1"),
        ("accepted", "split-2"),
        ("width_refused", "split-3"),
        ("duplicate", "split-3"),
        ("exhausted", "split-4"),
    ):
        memory.append(_boundary(sampler, started, f"before_{kind}_{event_id}"))
        before = runtime.candidate_snapshot()
        count_before = callbacks
        receipt, refusal = None, None
        try:
            receipt = runtime.consolidate(event_id, split)
        except ValueError as error:
            refusal = str(error)
        expected = (
            None
            if kind == "accepted"
            else "split target width exceeds configured ceiling"
            if kind == "width_refused"
            else "duplicate consolidation event_id"
            if kind == "duplicate"
            else "consolidation attempt quota exhausted"
        )
        if refusal != expected or callbacks != count_before + (
            kind in ("accepted", "width_refused")
        ):
            raise AssertionError("growth ceiling or original runtime admission ordering changed")
        after = runtime.candidate_snapshot()
        requested = 2 * before.state.raw_state.config.width
        if kind == "accepted":
            if receipt is None or after.state.raw_state.config.width != requested:
                raise AssertionError("accepted growth lost its exact requested width/receipt")
        else:
            if receipt is not None:
                raise AssertionError("refused growth returned a receipt")
            _same(
                asdict(before.state),
                asdict(after.state),
                "growth refusal changed complete candidate raw/history",
            )
        serving = actor.serve(features.copy(), now=_TICK)
        if not serving.cache_hit:
            raise AssertionError("growth actor missed its primed cache")
        _same(
            asdict(actor.serving_snapshot()), actor_before, "growth changed full actor/cache bundle"
        )
        _same(serving.prediction, prime.prediction, "growth changed cached actor prediction")
        cursor, size = _cursor(runtime)
        memory.append(_boundary(sampler, started, f"after_{kind}_{event_id}"))
        steps.append(
            dict(
                kind=kind,
                event_id=event_id,
                requested_width=requested,
                accepted_width=requested if kind == "accepted" else None,
                refusal=refusal,
                callback_count_before=count_before,
                callback_count_after=callbacks,
                receipt=None if receipt is None else asdict(receipt),
                candidate_before=asdict(before),
                candidate_after=asdict(after),
                cursor=cursor,
                cursor_bytes=size,
                serving=asdict(serving),
                native_array_bytes_before=_native_bytes(before.state.raw_state),
                native_array_bytes_after=_native_bytes(after.state.raw_state),
                scaling_before=_scaling(before.state.raw_state),
                scaling_after=_scaling(after.state.raw_state),
            )
        )
        budget.before_final()
    grown = runtime.candidate_snapshot()
    cursor_growth, _ = _cursor(runtime)
    if (
        callbacks != 3
        or grown.state.raw_state.config.width != _WIDTH_CAP
        or grown.consolidation_attempts != 3
        or grown.consolidations_completed != 2
        or cursor_growth["attempted_ids"] != ("split-1", "split-2", "split-3")
        or cursor_growth["stopped"]
        or cursor_growth["retired"]
        or not cursor_growth["payload_ready"]
    ):
        raise AssertionError("growth did not retain bounded attempts and two committed splits")
    label = LabelArrival(
        "label-0", experience.sample_id, experience.episode_id, _TICK, "actor-0", targets.copy()
    )
    runtime.record_label(label)
    applied = runtime.train_ready(max_updates=1)
    if len(applied) != 1 or applied[0].applied_at != _TICK:
        raise AssertionError("width refusal prevented the original delayed native allowance")
    continued = runtime.candidate_snapshot()
    _same(
        asdict(continued.state.origin_config),
        asdict(grown.state.origin_config),
        "continued growth lost origin",
    )
    _same(
        [asdict(item) for item in continued.state.transitions],
        [asdict(item) for item in grown.state.transitions],
        "continued growth lost transition tags",
    )
    if (
        continued.state.raw_state.config != grown.state.raw_state.config
        or continued.wake_updates != 1
    ):
        raise AssertionError("continued growth changed width/precision/policy or lost wake count")
    if continued.state.raw_state.wake_state is not None:
        if (
            grown.state.raw_state.wake_state is None
            or grown.state.raw_state.wake_state.wake_updates != 2
            or continued.state.raw_state.wake_state.wake_updates != 3
        ):
            raise AssertionError("continued growth lost complete CPC wake history")
    serving = actor.serve(features.copy(), now=_TICK)
    if not serving.cache_hit:
        raise AssertionError("continued growth missed actor cache")
    actor_after = actor.serving_snapshot()
    _same(asdict(actor_after), actor_before, "continued growth changed actor A/cache")
    _same(asdict(source), source_before, "growth changed original source raw state")
    cursor_after, final_size = _cursor(runtime)
    if (
        cursor_after["attempted_ids"] != cursor_growth["attempted_ids"]
        or cursor_after["consolidations"] != cursor_growth["consolidations"]
        or cursor_after["stopped"]
    ):
        raise AssertionError("delayed wake changed spent growth history or stopped the candidate")
    budget.before_final()
    memory.append(_boundary(sampler, started, "after_continuation"))
    return dict(
        family=family,
        source_state=source_before,
        actor_before=actor_before,
        actor_after=asdict(actor_after),
        candidate_before=asdict(candidate_before),
        candidate_before_continuation=asdict(grown),
        candidate_after_continuation=asdict(continued),
        cursor_before=cursor_before,
        cursor_after_growth=cursor_growth,
        cursor_after=cursor_after,
        steps=steps,
        experience=asdict(experience),
        label=asdict(label),
        applied_updates=[asdict(item) for item in applied],
        serving_after_continuation=asdict(serving),
        measurements=dict(
            max_cursor_bytes=max(final_size, max(step["cursor_bytes"] for step in steps)),
            native_array_bytes=dict(
                actor_before=_native_bytes(source),
                actor_after=_native_bytes(actor_after.model_state.raw_state),
                candidate_before=_native_bytes(candidate_before.state.raw_state),
                candidate_grown=_native_bytes(grown.state.raw_state),
                candidate_after_continuation=_native_bytes(continued.state.raw_state),
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
    _same(
        rows[0]["source_state"]["raw_parameters"],
        rows[1]["source_state"]["raw_parameters"],
        "PC/neutral growth source parameters differ",
    )
    for phase in (
        "candidate_before",
        "candidate_before_continuation",
        "candidate_after_continuation",
    ):
        _same(
            rows[0][phase]["state"]["raw_state"]["raw_parameters"],
            rows[1][phase]["state"]["raw_state"]["raw_parameters"],
            "PC/neutral growth parameters differ",
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
