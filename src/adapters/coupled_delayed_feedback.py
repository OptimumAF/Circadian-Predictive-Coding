"""Fixed delayed producing-version feedback across native serving transitions.

The existing inbox owns pair chronology/identity and the existing controller owns
promotion/rollback. Why record revisions separately: refused labels consume one
runtime revision while leaving payloads, models and training budget unchanged.
Outputs retain full phase snapshots; poisoned invalid targets are never copied
or serialized. No feedback framework, final-data scoring, tuning, durable recovery
or equality claim between different model predictions is provided.
"""

from dataclasses import asdict, replace
import os
from time import monotonic
from typing import Any, cast

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
from src.adapters.coupled_cache_transactions import _policy, _guards, _digest, _rss, _boundary
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array
from src.core.coupled_structural_state import CoupledStructuralState, copy_coupled_structural_state
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler

_PHASES = (
    "prime",
    "role_refused",
    "version_refused",
    "chronology_refused",
    "waiting_without_label",
    "source_hit",
    "waiting_at_promotion",
    "promoted",
    "promoted_prime",
    "current_version_refused",
    "buffered",
    "event_reused_refused",
    "waiting_buffered",
    "waiting_before_arrival",
    "applied_at_arrival",
    "promoted_hit",
    "repeated_drain",
    "duplicate_label_refused",
    "rolled_back",
    "rollback_hit",
    "post_rollback_drain",
)
_REVISIONS = (2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 15, 15, 16)


class _PoisonTargets:
    """Engineering sentinel: reaching any invalid target copy is a failed control."""

    def __init__(self) -> None:
        self.copy_attempts = 0

    def __deepcopy__(self, memo):
        self.copy_attempts += 1
        raise AssertionError("refused label targets must not be copied")


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_delayed_feedback_engineering_v1",
        families=list(FAMILIES),
        seed=23,
        geometry=dict(input_dim=2, width=2, hidden_layers=2, precisions=[1.0, 1.0]),
        scheme="mean_field",
        gamma0=1.5,
        training=dict(
            role="train",
            key=["delayed-feedback", "s0"],
            observed_at=2,
            labels_arrived_at=10,
            model_version="actor-0",
            features=[list(row) for row in FEATURES],
            targets=[[1.0], [0.0]],
        ),
        external_training_settings=dict(
            learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        ),
        serving=dict(
            cache_ttl_ticks=20,
            max_cache_entries=2,
            original_metadata={"route": ["original"]},
            promoted_metadata={"route": ["promoted"]},
        ),
        guards=dict(
            new=dict(
                task_id="new", key=["guard-new", "g0"], features=[[0.7, -0.4]], targets=[[1.0]]
            ),
            old=dict(
                task_id="old", key=["guard-old", "g0"], features=[[-0.6, 0.5]], targets=[[0.0]]
            ),
            role="inner_guard",
            labels_arrived_at=7,
            now=7,
        ),
        policy=asdict(_policy()),
        observation_phases=list(_PHASES),
        expected_revisions=list(_REVISIONS),
        native_training_calls=9,
        native_energy_evaluations=36,
        native_prediction_calls=18,
        serving_requests=18,
        guard_prediction_calls=12,
        split_kernel_calls=3,
        cache_misses=6,
        cache_hits=12,
        owned_child_launches=0,
        guarded_inference_calls=0,
        budget_limits=dict(
            max_training_updates=1,
            max_wall_seconds=60.0,
            max_replay_examples=None,
            max_hidden_width=None,
            max_process_rss_bytes=512 * 1024**2,
        ),
        runtime_limits=dict(max_experiences=1, max_consolidations=1),
        limitations="Producing-version/chronology and serving transaction control, not scientific benefit or full recovery. Refusals spend revisions. RSS is point-sampled, not continuously enforced or a long-run guarantee.",
    )


def _case(family: str, sampler: ProcessRssSampler, started: float) -> dict[str, Any]:
    boundaries = [_boundary(sampler, started, "before_source")]
    features = np.asarray(FEATURES, dtype=np.float64)
    targets = np.array([[1.0], [0.0]])
    source = _source(family, features, targets)
    source_record = asdict(source)
    learner = StructuralCoupledScalingLearner(
        source, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    actor = Actor(
        learner,
        version="actor-0",
        configuration=ServingConfiguration(20, 2),
        feature_digest=_feature_digest,
        metadata={"route": ["original"]},
    )
    clock = LogicalClock(2)
    limits = ToyExecutionBudget(
        max_training_updates=1, max_wall_seconds=60.0, max_process_rss_bytes=512 * 1024**2
    )
    limits_record = asdict(limits)
    budget = ToyBudgetSession(limits, monotonic)
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
        "s0",
        "delayed-feedback",
        2,
        "actor-0",
        features.copy(),
        "train",
        ExperiencePermissions(training=True),
    )
    runtime.record_experience(experience)
    runtime.consolidate("split-1", CoupledScalingSplitConsolidator())
    poison = _PoisonTargets()
    observations: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    counts = dict(state_valid=0, prediction_valid=0, utility=0, actions=0, resource=0)

    def budget_identity() -> dict[str, int]:
        if (
            runtime._budget is not budget
            or runtime._inbox._budget is not budget
            or budget.budget is not limits
        ):
            raise AssertionError("delayed feedback replaced original budget identity")
        _same(asdict(budget.budget), limits_record, "delayed feedback changed original limits")
        return dict(
            original=id(budget),
            runtime=id(runtime._budget),
            inbox=id(runtime._inbox._budget),
            original_limits=id(limits),
            current_limits=id(budget.budget),
        )

    def capture() -> dict[str, Any]:
        budget_identity()
        return dict(
            serving=asdict(actor.serving_snapshot()),
            candidate=asdict(runtime.candidate_snapshot()),
            cursor=asdict(runtime.capture_consolidation_cursor()),
            inbox=asdict(runtime._inbox.capture_cursor()),
            updates_completed=budget.updates_completed,
        )

    def observe(phase: str, *, now: int | None = None, hit: bool = False) -> None:
        result = None if now is None else actor.serve(features.copy(), now=now)
        if result is not None and result.cache_hit != hit:
            raise AssertionError(f"delayed serving phase {phase} had wrong hit/miss")
        snapshot = capture()
        index = len(observations)
        if phase != _PHASES[index] or snapshot["cursor"]["revision"] != _REVISIONS[index]:
            raise AssertionError("delayed feedback phase or truthful revision differs")
        expected_updates = int(index >= 14)
        if (
            snapshot["updates_completed"] != expected_updates
            or snapshot["candidate"]["wake_updates"] != expected_updates
        ):
            raise AssertionError("delayed feedback updated outside the actual arrival boundary")
        observations.append(
            dict(phase=phase, **snapshot, result=None if result is None else asdict(result))
        )
        boundaries.append(_boundary(sampler, started, phase))
        budget.before_final()

    def invalid_label(
        phase: str,
        event: str,
        *,
        version: str = "actor-0",
        arrival: int = 10,
        sample: str = "s0",
        role: Any = "train",
        message: str,
    ) -> None:
        before = capture()
        before_budget = dict(vars(budget))
        before_counts = dict(counts)
        # Targets are deliberately non-array poison: metadata must refuse before
        # touching the otherwise opaque payload at the native Array port.
        label = LabelArrival(
            event, sample, "delayed-feedback", arrival, version, cast(Array, poison), role
        )
        try:
            runtime.record_label(label)
        except ValueError as error:
            if str(error) != message:
                raise
        else:
            raise AssertionError("invalid delayed label was accepted")
        after = capture()
        revision_before, revision_after = before["cursor"]["revision"], after["cursor"]["revision"]
        if revision_after != revision_before + 1:
            raise AssertionError("refused label lost its actual runtime revision")
        normalized = dict(after["cursor"], revision=revision_before)
        _same(before["cursor"], normalized, "refused label changed cursor beyond revision")
        for key in ("serving", "candidate", "inbox", "updates_completed"):
            _same(before[key], after[key], f"refused label changed full {key}")
        if before_budget != dict(vars(budget)) or before_counts != counts or poison.copy_attempts:
            raise AssertionError("refused label changed budget/probes or copied poison")
        refusals.append(
            dict(
                phase=phase,
                message=message,
                revision_before=revision_before,
                revision_after=revision_after,
            )
        )
        observe(phase)

    def empty_drain(phase: str) -> None:
        before = runtime.candidate_snapshot().state
        if runtime.train_ready(max_updates=1):
            raise AssertionError("delayed pair trained before arrival or more than once")
        _same(
            asdict(before),
            asdict(runtime.candidate_snapshot().state),
            "empty drain mutated native state",
        )
        observe(phase)

    observe("prime", now=2)
    invalid_label(
        "role_refused",
        "bad-role",
        role="final_test",
        message="label must have the train role before payload access",
    )
    invalid_label(
        "version_refused",
        "bad-version",
        version="candidate-0",
        message="label role or actor version differs from its experience",
    )
    invalid_label(
        "chronology_refused",
        "bad-time",
        arrival=1,
        message="label cannot arrive before source observation",
    )
    empty_drain("waiting_without_label")
    clock.advance_to(7)
    observe("source_hit", now=7, hit=True)
    empty_drain("waiting_at_promotion")

    def state_valid(state: CoupledStructuralState) -> bool:
        counts["state_valid"] += 1
        try:
            copy_coupled_structural_state(state)
        except (ValueError, TypeError, FloatingPointError, OverflowError):
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
    saved = actor.serving_snapshot()
    ticket = controller.prepare(
        runtime,
        new,
        old,
        now=7,
        training_ids=frozenset((experience.key,)),
        metadata={"route": ["promoted"]},
    )
    _same(
        asdict(actor.serving_snapshot()), asdict(saved), "delayed preparation changed full actor A"
    )
    receipt = controller.commit(runtime, ticket)
    promoted = actor.serving_snapshot()
    _same(
        asdict(promoted.model_state),
        asdict(runtime.candidate_snapshot().state),
        "delayed promotion lost full B",
    )
    if (
        promoted.generation != 1
        or promoted.actor_version != "candidate-0"
        or promoted.cache
        or promoted.last_cache_tick != 0
        or promoted.configuration != saved.configuration
        or promoted.metadata != {"route": ["promoted"]}
    ):
        raise AssertionError("delayed promotion lost complete bundle transaction")
    observe("promoted")
    observe("promoted_prime", now=7)
    invalid_label(
        "current_version_refused",
        "bad-current",
        version=actor.version,
        message="label role or actor version differs from its experience",
    )
    correct = LabelArrival("label-0", "s0", "delayed-feedback", 10, "actor-0", targets.copy())
    runtime.record_label(correct)
    observe("buffered")
    invalid_label(
        "event_reused_refused",
        "label-0",
        sample="s1",
        message="duplicate label or event identity: ('delayed-feedback', 's1'), label-0",
    )
    empty_drain("waiting_buffered")
    clock.advance_to(9)
    if runtime.train_ready(max_updates=1):
        raise AssertionError("delayed label trained before tick10")
    observe("waiting_before_arrival", now=9, hit=True)
    before_apply = actor.serving_snapshot()
    preupdate = runtime.candidate_snapshot().state
    clock.advance_to(10)
    applied = runtime.train_ready(max_updates=1)
    if len(applied) != 1 or (
        applied[0].actor_version,
        applied[0].observed_at,
        applied[0].arrived_at,
        applied[0].applied_at,
        applied[0].update_number,
    ) != ("actor-0", 2, 10, 10, 1):
        raise AssertionError("delayed update lost producing-version chronology or original quota")
    _same(
        asdict(actor.serving_snapshot()),
        asdict(before_apply),
        "candidate C update changed independently serving B",
    )
    updated = runtime.candidate_snapshot().state
    if updated.raw_state.config != preupdate.raw_state.config:
        raise AssertionError("delayed native update changed structural configuration")
    _same(
        asdict(updated.origin_config),
        asdict(preupdate.origin_config),
        "delayed native update lost root origin",
    )
    _same(
        [asdict(item) for item in updated.transitions],
        [asdict(item) for item in preupdate.transitions],
        "delayed native update lost split history",
    )
    if updated.raw_state.wake_state is not None and (
        preupdate.raw_state.wake_state is None
        or preupdate.raw_state.wake_state.wake_updates != 2
        or updated.raw_state.wake_state.wake_updates != 3
    ):
        raise AssertionError("delayed native update lost complete CPC count")
    observe("applied_at_arrival")
    observe("promoted_hit", now=10, hit=True)
    empty_drain("repeated_drain")
    invalid_label(
        "duplicate_label_refused",
        "label-again",
        message="duplicate label or event identity: ('delayed-feedback', 's0'), label-again",
    )
    if controller.rollback(receipt) != 2:
        raise AssertionError("delayed rollback generation differs")
    _same(
        asdict(actor.serving_snapshot()),
        asdict(replace(saved, generation=2)),
        "delayed rollback lost complete A bundle",
    )
    _same(
        asdict(runtime.candidate_snapshot().state),
        asdict(updated),
        "actor rollback changed native candidate C",
    )
    observe("rolled_back")
    observe("rollback_hit", now=10, hit=True)
    empty_drain("post_rollback_drain")
    final = observations[-1]
    if (
        len(observations) != 21
        or len(refusals) != 6
        or poison.copy_attempts
        or len(final["inbox"]["experiences"]) != 1
        or len(final["inbox"]["labels"]) != 1
        or len(final["inbox"]["applied"]) != 1
    ):
        raise AssertionError("delayed feedback final identity/payload ledger differs")
    if (
        final["cursor"]["attempted_ids"] != ("split-1",)
        or len(final["cursor"]["consolidations"]) != 1
        or final["cursor"]["stopped"]
        or final["cursor"]["retired"]
        or not final["cursor"]["payload_ready"]
    ):
        raise AssertionError("delayed feedback lost original consolidation ownership/history")
    _same(asdict(source), source_record, "delayed feedback changed source")
    if counts != dict(state_valid=2, prediction_valid=4, utility=4, actions=4, resource=1):
        raise AssertionError("delayed refusal or update ran extra guard probes")
    return dict(
        family=family,
        source_state=source_record,
        guard_report=asdict(ticket.report),
        receipt=asdict(receipt),
        observations=observations,
        refusals=refusals,
        budget_limits=limits_record,
        budget_identity=budget_identity(),
        poison_copy_attempts=poison.copy_attempts,
        guard_callback_counts=counts,
        rss_boundaries=boundaries,
    )


def run_example() -> dict[str, Any]:
    started = monotonic()
    sampler = ProcessRssSampler(read_rss_bytes=_rss)
    _boundary(sampler, started, "invocation_start")
    cases = [_case(family, sampler, started) for family in FAMILIES]
    _same(
        cases[0]["source_state"]["raw_parameters"],
        cases[1]["source_state"]["raw_parameters"],
        "PC/neutral delayed source differs",
    )
    for phase in ("prime", "promoted", "applied_at_arrival", "rolled_back"):
        left = next(item for item in cases[0]["observations"] if item["phase"] == phase)
        right = next(item for item in cases[1]["observations"] if item["phase"] == phase)
        for key in ("candidate", "serving"):
            state_key = "state" if key == "candidate" else "model_state"
            _same(
                left[key][state_key]["raw_state"]["raw_parameters"],
                right[key][state_key]["raw_state"]["raw_parameters"],
                "PC/neutral delayed transaction raw parameters differ",
            )
    end = _boundary(sampler, started, "invocation_end")
    return _plain(
        dict(
            protocol_id="coupled_delayed_feedback_engineering_v1",
            description=describe_example(),
            cases=cases,
            elapsed_seconds=end["elapsed_seconds"],
            rss=end["rss"],
        )
    )
