"""Fixed native cache eviction/expiry and guarded promotion/rollback controls.

Inputs are frozen engineering features/settings; output is complete JSON-native
state and transaction evidence. Existing structural learners/controller own
native reconstruction and publication. Why broad guard thresholds: exercise
binary transaction plumbing without selecting a learning benefit. No final-data
scoring, tuning, new transaction framework or durable recovery claim is provided.
"""

from dataclasses import asdict, replace
from hashlib import sha256
from math import isfinite
import os
from time import monotonic
from typing import Any, Callable

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_interruption import FEATURES, FAMILIES, _plain
from src.adapters.coupled_growth_limit import (
    Actor,
    Runtime,
    _source,
    _encoded,
    _same,
    _feature_digest,
)
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array
from src.core.coupled_structural_state import CoupledStructuralState, copy_coupled_structural_state
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler, read_process_rss_bytes

_RSS_CAP = 512 * 1024**2
_WALL_CAP = 60.0
_PHASES = (
    "prime",
    "hit",
    "second",
    "evict",
    "revisit",
    "expire",
    "prepared",
    "committed",
    "promoted_miss",
    "promoted_hit",
    "rolled_back",
    "rollback_hit",
    "rollback_expire",
)


def _policy() -> PromotionPolicy:
    return PromotionPolicy(
        "one_minus_brier_v1",
        "canonical_structural_json_bytes_v1",
        0.0,
        -1.0,
        1.0,
        60.0,
        65536,
        ("0", "1"),
    )


def describe_example() -> dict[str, Any]:
    """Return frozen metadata with no operational IO/models/RSS/thread work."""
    return dict(
        protocol_id="coupled_cache_transactions_engineering_v1",
        families=list(FAMILIES),
        seed=23,
        geometry=dict(input_dim=2, width=2, hidden_layers=2, precisions=[1.0, 1.0]),
        scheme="mean_field",
        gamma0=1.5,
        training=dict(
            role="train",
            features=[list(row) for row in FEATURES],
            targets=[[1.0], [0.0]],
            sample_key=["cache-transactions", "train-batch-0"],
            observed_at=0,
            arrived_at=2,
        ),
        external_training_settings=dict(
            learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        ),
        serving=dict(
            cache_ttl_ticks=3,
            max_cache_entries=2,
            eviction="FIFO insertion order",
            metadata={"route": ["original"]},
            promoted_metadata={"route": ["promoted"]},
            x1=[list(row) for row in FEATURES],
            x2_offset=0.125,
            x3_offset=-0.25,
        ),
        guards=dict(
            new=dict(
                task_id="new",
                sample_keys=[["guard-new", "g0"]],
                features=[[0.7, -0.4]],
                targets=[[1.0]],
                labels_arrived_at=7,
                role="inner_guard",
            ),
            old=dict(
                task_id="old",
                sample_keys=[["guard-old", "g0"]],
                features=[[-0.6, 0.5]],
                targets=[[0.0]],
                labels_arrived_at=7,
                role="inner_guard",
            ),
            now=7,
        ),
        policy=asdict(_policy()),
        observation_phases=list(_PHASES),
        native_training_calls=9,
        native_energy_evaluations=36,
        split_kernel_calls=3,
        guard_prediction_calls=12,
        native_prediction_calls=33,
        serving_requests=33,
        successful_serves=30,
        cache_hits=9,
        cache_misses=21,
        limits=dict(
            max_experiences=1,
            max_training_updates=1,
            max_consolidations=1,
            max_wall_seconds=_WALL_CAP,
            max_process_rss_bytes=_RSS_CAP,
        ),
        limitations="Broad guards exercise transactions, not learning benefit. Point-sampled RSS is not continuous preemption or a long-run bound. No full recovery or scientific qualification.",
    )


def _rss() -> int:
    value = read_process_rss_bytes()
    if type(value) is not int or value <= 0:
        raise RuntimeError("cache transactions require valid positive process RSS")
    return value


def _boundary(sampler: ProcessRssSampler, started: float, phase: str) -> dict[str, Any]:
    elapsed = monotonic() - started
    if not isfinite(elapsed) or not 0 <= elapsed < _WALL_CAP:
        raise RuntimeError("cache transactions exceeded60second boundary")
    sampler.sample()
    segment = sampler.snapshot()
    if segment.peak_bytes > _RSS_CAP:
        raise RuntimeError("cache transactions exceeded sampled512MiB RSS")
    return dict(phase=phase, elapsed_seconds=elapsed, rss=asdict(segment))


def _digest(state: CoupledStructuralState) -> str:
    return sha256(_encoded(asdict(state))).hexdigest()


def _guards() -> tuple[GuardBatch[Array, Array], GuardBatch[Array, Array]]:
    return (
        GuardBatch(
            "new",
            (("guard-new", "g0"),),
            7,
            np.array([[0.7, -0.4]]),
            np.array([[1.0]]),
            "inner_guard",
        ),
        GuardBatch(
            "old",
            (("guard-old", "g0"),),
            7,
            np.array([[-0.6, 0.5]]),
            np.array([[0.0]]),
            "inner_guard",
        ),
    )


def _case(family: str, sampler: ProcessRssSampler, started: float) -> dict[str, Any]:
    boundaries = [_boundary(sampler, started, "before_source")]
    x1 = np.asarray(FEATURES, dtype=np.float64)
    x2, x3 = x1 + 0.125, x1 - 0.25
    targets = np.array([[1.0], [0.0]])
    source = _source(family, x1, targets)
    source_record = asdict(source)
    learner = StructuralCoupledScalingLearner(
        source, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    actor = Actor(
        learner,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
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
        max_consolidations=1,
        actor=actor,
    )
    experience = Experience(
        "train-batch-0",
        "cache-transactions",
        0,
        "actor-0",
        x1.copy(),
        "train",
        ExperiencePermissions(training=True),
    )
    runtime.record_experience(experience)
    split_receipt = runtime.consolidate("split-1", CoupledScalingSplitConsolidator())
    clock.advance_to(2)
    label = LabelArrival(
        "label-0", experience.sample_id, experience.episode_id, 2, "actor-0", targets.copy()
    )
    runtime.record_label(label)
    updates = runtime.train_ready(max_updates=1)
    if len(updates) != 1:
        raise AssertionError("cache transaction candidate did not receive one arrived update")
    candidate_before = asdict(runtime.candidate_snapshot())
    cursor_before = asdict(runtime.capture_consolidation_cursor())
    counts = dict(state_valid=0, prediction_valid=0, utility=0, actions=0, resource=0)
    observations: list[dict[str, Any]] = []
    refusals: list[dict[str, str]] = []

    def observe(phase: str, features: Array | None = None, now: int = 7, hit: bool = False) -> None:
        result = None if features is None else actor.serve(features.copy(), now=now)
        if result is not None and result.cache_hit != hit:
            raise AssertionError(f"cache phase {phase} had unexpected hit/miss")
        observations.append(
            dict(
                phase=phase,
                serving=asdict(actor.serving_snapshot()),
                result=None if result is None else asdict(result),
            )
        )
        boundaries.append(_boundary(sampler, started, phase))
        budget.before_final()

    def refuse(phase: str, operation: Callable[[], Any], expected: str) -> None:
        before = (
            asdict(actor.serving_snapshot()),
            asdict(runtime.candidate_snapshot()),
            asdict(runtime.capture_consolidation_cursor()),
            dict(counts),
        )
        try:
            operation()
        except ValueError as error:
            if str(error) != expected:
                raise
            refusals.append(dict(phase=phase, message=str(error)))
        else:
            raise AssertionError(f"expected refusal did not occur: {phase}")
        after = (
            asdict(actor.serving_snapshot()),
            asdict(runtime.candidate_snapshot()),
            asdict(runtime.capture_consolidation_cursor()),
            dict(counts),
        )
        _same(before, after, f"refused {phase} changed complete state/cursor/probes")
        boundaries.append(_boundary(sampler, started, phase))

    observe("prime", x1, 2)
    observe("hit", x1, 3, True)
    observe("second", x2, 3)
    observe("evict", x3, 4)
    observe("revisit", x1, 4)
    observe("expire", x1, 7)
    refuse(
        "backward_tick", lambda: actor.serve(x1.copy(), now=6), "serving cache tick moved backwards"
    )

    def state_valid(state: CoupledStructuralState) -> bool:
        counts["state_valid"] += 1
        try:
            copy_coupled_structural_state(state)
        except (ValueError, TypeError, OverflowError, FloatingPointError):
            return False
        return True

    def prediction_valid(prediction: Array) -> bool:
        counts["prediction_valid"] += 1
        return bool(
            type(prediction) is np.ndarray
            and prediction.dtype == np.float64
            and prediction.ndim == 2
            and prediction.shape[1] == 1
            and np.isfinite(prediction).all()
            and np.all((prediction >= 0) & (prediction <= 1))
        )

    def utility(prediction: Array, expected: Array) -> float:
        counts["utility"] += 1
        return 1.0 - float(np.mean((prediction - expected) ** 2))

    def actions(prediction: Array) -> tuple[str, ...]:
        counts["actions"] += 1
        return tuple("1" if value >= 0.5 else "0" for value in prediction[:, 0])

    def resource(state: CoupledStructuralState) -> int:
        counts["resource"] += 1
        return len(_encoded(asdict(state)))

    builder = learner.build_from_state
    evaluator = PromotionGuardEvaluator[Array, Array, Array, CoupledStructuralState](
        build_learner=builder,
        utility=utility,
        actions=actions,
        state_valid=state_valid,
        prediction_valid=prediction_valid,
        resource_bytes=resource,
        state_digest=_digest,
        clock=monotonic,
    )
    controller = ServingPromotionController(
        actor, policy=_policy(), evaluator=evaluator, build_learner=builder, state_digest=_digest
    )
    new, old = _guards()
    training_ids = frozenset((experience.key,))

    def prepare(new_guard):
        return controller.prepare(
            runtime,
            new_guard,
            old,
            now=7,
            training_ids=training_ids,
            metadata={"route": ["promoted"]},
        )

    refuse(
        "final_test_guard",
        lambda: prepare(replace(new, role="final_test")),
        "promotion measurements require inner_guard role before payload access",
    )
    refuse(
        "future_guard",
        lambda: prepare(replace(new, labels_arrived_at=8)),
        "guard labels have not arrived",
    )
    refuse(
        "training_overlap",
        lambda: prepare(replace(new, sample_keys=(experience.key,))),
        "new/old guards and declared training IDs must be disjoint",
    )
    saved = actor.serving_snapshot()
    ticket = prepare(new)
    _same(
        asdict(actor.serving_snapshot()), asdict(saved), "preparation changed complete actor/cache"
    )
    observe("prepared")
    receipt = controller.commit(runtime, ticket)
    committed = actor.serving_snapshot()
    if (
        committed.generation != 1
        or committed.actor_version != "candidate-0"
        or committed.cache
        or committed.last_cache_tick != 0
        or committed.configuration != saved.configuration
        or committed.metadata != {"route": ["promoted"]}
    ):
        raise AssertionError("promotion did not atomically publish the complete configured bundle")
    _same(
        asdict(committed.model_state),
        candidate_before["state"],
        "promoted full candidate state differs",
    )
    observe("committed")
    refuse(
        "consumed_ticket",
        lambda: controller.commit(runtime, ticket),
        "promotion ticket is foreign, forged, discarded or consumed",
    )
    observe("promoted_miss", x1, 7)
    observe("promoted_hit", x1, 8, True)
    if controller.rollback(receipt) != 2:
        raise AssertionError("rollback did not advance generation exactly once")
    _same(
        asdict(actor.serving_snapshot()),
        asdict(replace(saved, generation=2)),
        "rollback lost full precommit serving bundle",
    )
    observe("rolled_back")
    refuse(
        "repeated_rollback",
        lambda: controller.rollback(receipt),
        "rollback receipt is foreign, stale or consumed",
    )
    observe("rollback_hit", x1, 8, True)
    observe("rollback_expire", x1, 10)
    candidate_after = asdict(runtime.candidate_snapshot())
    cursor_after = asdict(runtime.capture_consolidation_cursor())
    _same(candidate_before, candidate_after, "cache transactions changed complete candidate state")
    _same(cursor_before, cursor_after, "cache transactions changed complete candidate cursor")
    _same(asdict(source), source_record, "cache transactions changed source raw state")
    if tuple(item["phase"] for item in observations) != _PHASES:
        raise AssertionError("cache transaction observation phases changed")
    if counts != dict(state_valid=2, prediction_valid=4, utility=4, actions=4, resource=1):
        raise AssertionError("guard metadata refusal invoked native measurement probes")
    return dict(
        family=family,
        source_state=source_record,
        candidate_before=candidate_before,
        candidate_after=candidate_after,
        cursor_before=cursor_before,
        cursor_after=cursor_after,
        guard_report=asdict(ticket.report),
        receipt=asdict(receipt),
        observations=observations,
        refusals=refusals,
        guard_callback_counts=counts,
        rss_boundaries=boundaries,
        training=dict(
            experience=asdict(experience),
            label=asdict(label),
            applied_updates=[asdict(item) for item in updates],
            split_receipt=asdict(split_receipt),
        ),
        guards=dict(new=asdict(new), old=asdict(old), training_ids=sorted(training_ids)),
        policy=asdict(_policy()),
    )


def run_example() -> dict[str, Any]:
    started = monotonic()
    sampler = ProcessRssSampler(read_rss_bytes=_rss)
    _boundary(sampler, started, "invocation_start")
    cases = [_case(family, sampler, started) for family in FAMILIES]
    _same(
        cases[0]["source_state"]["raw_parameters"],
        cases[1]["source_state"]["raw_parameters"],
        "PC/neutral cache source parameters differ",
    )
    _same(
        cases[0]["candidate_after"]["state"]["raw_state"]["raw_parameters"],
        cases[1]["candidate_after"]["state"]["raw_state"]["raw_parameters"],
        "PC/neutral cache candidate parameters differ",
    )
    for phase in ("committed", "rolled_back"):
        left = next(item for item in cases[0]["observations"] if item["phase"] == phase)
        right = next(item for item in cases[1]["observations"] if item["phase"] == phase)
        _same(
            left["serving"]["model_state"]["raw_state"]["raw_parameters"],
            right["serving"]["model_state"]["raw_state"]["raw_parameters"],
            "PC/neutral cache transaction parameters differ",
        )
    end = _boundary(sampler, started, "invocation_end")
    return _plain(
        dict(
            protocol_id="coupled_cache_transactions_engineering_v1",
            description=describe_example(),
            cases=cases,
            elapsed_seconds=end["elapsed_seconds"],
            rss=end["rss"],
        )
    )
