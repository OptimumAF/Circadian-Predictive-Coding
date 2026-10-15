"""Fixed mixed raw structural runtime and independent serving transactions.

Broad guard thresholds and zero clocks test transactions, not quality/timing.
The detached negative branch preserves its prune tag after invalid inverse merge.
"""

from dataclasses import replace
from typing import Any, cast

import numpy as np

from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.adapters.coupled_scaling_topology_demo import source_state
from src.adapters.coupled_topology_demo import (
    encode_record as encode_record,
    state_digest,
    probes,
    guards,
    selection_ids,
    continue_candidate as _continue_candidate,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.coupled_scaling_prune_consolidation import CoupledScalingPruneRefused
from src.app.coupled_structural_consolidation import (
    CoupledScalingSplitConsolidator,
    CoupledScalingMergeConsolidator,
    CoupledScalingStructuralPruneConsolidator,
)
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor, PromotionRejected, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array
from src.core.coupled_pc_scaling import CoupledScalingState
from src.core.coupled_scaling_structure import CoupledScalingSplitLineage
from src.core.coupled_structural_state import CoupledStructuralState, copy_coupled_structural_state
from src.core.experience import LogicalClock
from src.core.promotion_guard import PromotionPolicy
from src.core.serving_ports import ServingConfiguration

StructuralRuntime = ActorShadowRuntime[Array, Array, Array, CoupledStructuralState]
StructuralActor = PromotableActor[Array, Array, Array, CoupledStructuralState]
StructuralController = ServingPromotionController[Array, Array, Array, CoupledStructuralState]


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_mixed_structural_runtime_engineering_v1",
        scope="local_engineering_integration_not_scientific_comparison",
        variants=["pc", "neutral_cpc", "active_cpc"],
        seed=23,
        input_dim=2,
        source_width=3,
        hidden_layers=2,
        scheme="mean_field",
        gamma0=1.5,
        precisions=[1.0, 1.0],
        learning_rate=0.01,
        inference_steps=2,
        inference_learning_rate=0.05,
        main_operations=["split", "merge", "prune"],
        branch_operations=["split", "prune", "refused_original_inverse_merge"],
        train_calls=9,
        guarded_infer_calls=36,
        coupled_energy_evaluations=144,
        limitations="Fixed adapter settings; no source/split SGD or inference identity. Lineage is metadata, guards use declared selection IDs. Native tensor bytes exclude heap/history. No scientific quality or timing claim.",
    )


def _runtime(
    learner: StructuralCoupledScalingLearner,
) -> tuple[StructuralActor, StructuralRuntime, LogicalClock]:
    actor = StructuralActor(
        learner,
        version="actor-0",
        configuration=ServingConfiguration(20, 4),
        feature_digest=state_digest,
        metadata={"route": ["original"]},
    )
    clock = LogicalClock()
    runtime = StructuralRuntime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
        max_experiences=1,
        max_consolidations=8,
        actor=actor,
    )
    return actor, runtime, clock


def setup_runtime(
    variant: str,
) -> tuple[StructuralActor, StructuralRuntime, LogicalClock, StructuralCoupledScalingLearner]:
    learner = StructuralCoupledScalingLearner(
        source_state(variant), learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    actor, runtime, clock = _runtime(learner)
    return actor, runtime, clock, learner


def _state_valid(state: CoupledStructuralState) -> bool:
    try:
        copy_coupled_structural_state(state)
        return True
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return False


def resource_bytes(state: CoupledStructuralState) -> int:
    raw: CoupledScalingState = copy_coupled_structural_state(state).raw_state
    p = raw.raw_parameters
    arrays = [*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias]
    if raw.wake_state is not None:
        arrays.extend((raw.wake_state.chemical_state, raw.wake_state.plasticity_state))
    return sum(a.nbytes for a in arrays)


def make_controller(
    actor: StructuralActor,
    prototype: StructuralCoupledScalingLearner,
    *,
    max_resource_bytes: int = 1_000_000,
) -> StructuralController:
    builder = prototype.build_from_state
    evaluator = PromotionGuardEvaluator[Array, Array, Array, CoupledStructuralState](
        build_learner=builder,
        utility=lambda prediction, targets: float(1 - np.mean(np.abs(prediction - targets))),
        actions=lambda prediction: ("binary",),
        state_valid=_state_valid,
        prediction_valid=lambda prediction: bool(
            np.isfinite(prediction).all() and np.all((prediction >= 0) & (prediction <= 1))
        ),
        resource_bytes=resource_bytes,
        state_digest=state_digest,
        clock=lambda: 0.0,
    )
    return StructuralController(
        actor,
        policy=PromotionPolicy(
            "one_minus_mean_absolute_probability_error_v1",
            "raw_native_tensor_bytes_v1",
            0.0,
            -1.0,
            1.0,
            1.0,
            max_resource_bytes,
            ("binary",),
        ),
        evaluator=evaluator,
        build_learner=builder,
        state_digest=state_digest,
    )


def prune_consolidator(
    removed_units: tuple[int, ...] = (0, 1), *, failed_budget: bool = False
) -> CoupledScalingStructuralPruneConsolidator:
    ticks = iter((0.0, 1.0))
    return CoupledScalingStructuralPruneConsolidator(
        removed_units,
        probes(),
        limits=CoupledPruneLimits(0, 0, 0)
        if removed_units == (1, 0)
        else CoupledPruneLimits(1e-12, 1e-12, 1e-12),
        budget=ToyBudgetSession(
            ToyExecutionBudget(max_wall_seconds=0 if failed_budget else 5),
            (lambda: next(ticks, 1.0)) if failed_budget else (lambda: 0.0),
        ),
        inference_steps=2,
        inference_learning_rate=0.05,
    )


def _negative_branch(
    prototype: StructuralCoupledScalingLearner,
    split: CoupledStructuralState,
    lineage: CoupledScalingSplitLineage,
) -> dict[str, Any]:
    actor, runtime, _ = _runtime(prototype.build_from_state(split))
    actor.serve(guards()[0].features, now=1)
    before_actor = actor.serving_snapshot()
    receipt = runtime.consolidate("prune-after-split", prune_consolidator((0, 2)))
    pruned = runtime.candidate_snapshot()
    try:
        runtime.consolidate("invalid-original-inverse", CoupledScalingMergeConsolidator(lineage))
    except ValueError as error:
        reason = str(error)
    else:
        raise AssertionError("original inverse merge survived a single-coordinate prune")
    after = runtime.candidate_snapshot()
    if state_digest(after.state) != state_digest(pruned.state) or state_digest(
        actor.serving_snapshot()
    ) != state_digest(before_actor):
        raise AssertionError("invalid inverse changed mixed state/history or serving bundle")
    return dict(
        prune_receipt=receipt,
        pruned_candidate=pruned,
        refused_candidate=after,
        invalid_inverse_reason=reason,
        actor_before=before_actor,
        actor_after=actor.serving_snapshot(),
    )


def run_variant(variant: str) -> dict[str, Any]:
    actor, runtime, clock, prototype = setup_runtime(variant)
    actor.serve(guards()[0].features, now=1)
    initial = actor.serving_snapshot()
    lineage = CoupledScalingSplitLineage(1, runtime.candidate_snapshot().state.raw_state.config)
    split_receipt = runtime.consolidate("split", CoupledScalingSplitConsolidator())
    split = runtime.candidate_snapshot()
    negative = _negative_branch(prototype, split.state, lineage)
    merge_receipt = runtime.consolidate("merge", CoupledScalingMergeConsolidator(lineage))
    merged = runtime.candidate_snapshot()
    refusals = []
    for name, operation in (
        ("damage", prune_consolidator((1, 0))),
        ("budget", prune_consolidator(failed_budget=True)),
    ):
        try:
            runtime.consolidate(name, operation)
        except CoupledScalingPruneRefused as error:
            decision = error.decision
            if decision.status != ("rejected" if name == "damage" else "failed"):
                raise AssertionError("unexpected prune refusal")
            if state_digest(runtime.candidate_snapshot().state) != state_digest(merged.state):
                raise AssertionError("refused prune lost full mixed candidate state")
            refusals.append(
                dict(
                    scenario=name,
                    status=decision.status,
                    reason=decision.reason,
                    diagnostics=decision.diagnostics,
                )
            )
        else:
            raise AssertionError("expected damage or budget prune refusal")
    prune_receipt = runtime.consolidate("accepted-prune", prune_consolidator())
    pruned = runtime.candidate_snapshot()
    # Existing helper uses only public generic runtime ports and the same data IDs.
    _continue_candidate(cast(Any, runtime), clock)
    continued = runtime.candidate_snapshot()
    if state_digest(actor.serving_snapshot()) != state_digest(initial):
        raise AssertionError("candidate work changed original serving bundle")
    try:
        make_controller(actor, prototype, max_resource_bytes=0).prepare(
            runtime, *guards(), now=2, training_ids=selection_ids()
        )
    except PromotionRejected as error:
        rejected = error.report
    else:
        raise AssertionError("expected independent resource guard refusal")
    controller = make_controller(actor, prototype)
    ticket = controller.prepare(
        runtime, *guards(), now=2, training_ids=selection_ids(), metadata={"route": ["mixed"]}
    )
    promotion = controller.commit(runtime, ticket)
    promoted = actor.serving_snapshot()
    served = actor.serve(guards()[0].features, now=2)
    controller.rollback(promotion)
    rolled_back = actor.serving_snapshot()
    if state_digest(rolled_back) != state_digest(replace(initial, generation=2)):
        raise AssertionError("rollback did not restore complete original serving bundle")
    if state_digest(runtime.candidate_snapshot()) != state_digest(continued):
        raise AssertionError("serving transaction changed candidate")
    return dict(
        variant=variant,
        initial=initial,
        split_candidate=split,
        merged_candidate=merged,
        pruned_candidate=pruned,
        continued_candidate=continued,
        split_receipt=split_receipt,
        merge_receipt=merge_receipt,
        prune_receipt=prune_receipt,
        refusals=refusals,
        negative_branch=negative,
        rejected_promotion_report=rejected,
        accepted_promotion_report=ticket.report,
        promotion_receipt=promotion,
        promoted=promoted,
        served=served,
        rolled_back=rolled_back,
    )


def run_example() -> dict[str, Any]:
    records = [run_variant(variant) for variant in describe_example()["variants"]]
    parity = {}
    for stage in ("split_candidate", "merged_candidate", "pruned_candidate", "continued_candidate"):
        pc = cast(Any, records[0][stage]).state.raw_state.raw_parameters
        neutral = cast(Any, records[1][stage]).state.raw_state.raw_parameters
        parity[stage] = state_digest(pc) == state_digest(neutral)
    if not all(parity.values()):
        raise AssertionError("PCneutral raw mixed continuation differs")
    return dict(
        **describe_example(),
        probes=probes(),
        guards=guards(),
        excluded_selection_ids=sorted(selection_ids()),
        pc_neutral_parameter_parity=parity,
        records=records,
    )
