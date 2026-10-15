"""Fixed local raw coupled pruning, consolidation, serving and rollback example.

Synthetic train/validation/inner-guard data exercise existing runtime boundaries.
Utility thresholds cover the full probability domain for a transaction check;
they do not establish learning benefit, acceptable deployment quality or latency.
No final-test data, checkpoint codec, external access or scientific study.
"""

from dataclasses import replace
from typing import Any, cast

import numpy as np

from src.adapters.topology_coupled_scaling_learner import TopologyCoupledScalingLearner
from src.adapters.coupled_topology_demo import (
    encode_record as encode_record,
    state_digest,
    probes,
    guards,
    selection_ids,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_scaling_prune_consolidation import (
    CoupledScalingPruneConsolidator,
    CoupledScalingPruneRefused,
)
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import (
    PromotableActor,
    PromotionRejected,
    ServingPromotionController,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import Array
from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    initialize_coupled_scaling,
)
from src.core.coupled_scaling_topology import (
    CoupledScalingTopologyState,
    copy_coupled_scaling_topology,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.promotion_guard import PromotionPolicy
from src.core.serving_ports import ServingConfiguration

TopologyRuntime = ActorShadowRuntime[Array, Array, Array, CoupledScalingTopologyState]
TopologyActor = PromotableActor[Array, Array, Array, CoupledScalingTopologyState]
TopologyController = ServingPromotionController[Array, Array, Array, CoupledScalingTopologyState]


def describe_example() -> dict[str, Any]:
    return {
        "protocol_id": "coupled_raw_topology_runtime_engineering_v1",
        "scope": "local_engineering_integration_not_scientific_comparison",
        "variants": ["pc", "neutral_cpc", "active_cpc"],
        "seed": 23,
        "input_dim": 2,
        "hidden_dims": [3, 3],
        "scheme": "mean_field",
        "gamma0": 1.5,
        "removed_units": [0, 1],
        "learning_rate": 0.01,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "train_calls": 9,
        "guarded_infer_calls": 24,
        "coupled_energy_evaluations": 108,
        "limitations": "Fixed-step relaxed energy; no global damage, future-step identity, quality or timing claim. Raw factors change with width. Arbitrary consolidation remains refused with managed retention.",
    }


def source_state(variant: str) -> CoupledScalingState:
    if variant not in ("pc", "neutral_cpc", "active_cpc"):
        raise ValueError("unknown fixed raw example variant")
    chemistry = (
        CoupledCircadianConfig.matched_pc_control()
        if variant == "neutral_cpc"
        else CoupledCircadianConfig()
        if variant == "active_cpc"
        else None
    )
    state = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, gamma0=1.5, circadian=chemistry), 23
    )
    learner = CoupledScalingLearner(
        state, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    training = probes()[0]
    for _ in range(2):
        learner.train_batch(training.features.copy(), training.targets.copy())
    state = learner.snapshot_state()
    parameters = state.raw_parameters
    # Declared disconnected fixture after genuine raw history, preserving traces.
    for layer, unit in enumerate((0, 1)):
        parameters.hidden_weights[layer][:, unit] = 0
        parameters.hidden_biases[layer][:, unit] = 0
        if layer + 1 < 2:
            parameters.hidden_weights[layer + 1][unit, :] = 0
        else:
            parameters.output_weight[unit, :] = 0
    return CoupledScalingState(state.config, parameters, state.wake_state)


def setup_runtime(
    variant: str,
) -> tuple[TopologyActor, TopologyRuntime, LogicalClock, TopologyCoupledScalingLearner]:
    learner = TopologyCoupledScalingLearner(
        source_state(variant),
        learning_rate=0.01,
        inference_steps=2,
        inference_learning_rate=0.05,
    )
    actor = TopologyActor(
        learner,
        version="actor-0",
        configuration=ServingConfiguration(20, 4),
        feature_digest=state_digest,
        metadata={"route": ["original"]},
    )
    clock = LogicalClock()
    runtime = TopologyRuntime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
        max_experiences=1,
        max_consolidations=4,
        actor=actor,
    )
    return actor, runtime, clock, learner


def _state_valid(state: CoupledScalingTopologyState) -> bool:
    try:
        copy_coupled_scaling_topology(state)
        return True
    except (ValueError, TypeError, OverflowError, FloatingPointError):
        return False


def resource_bytes(state: CoupledScalingTopologyState) -> int:
    raw = copy_coupled_scaling_topology(state).raw_state
    parameters = raw.raw_parameters
    arrays = [
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    ]
    if raw.wake_state is not None:
        arrays.extend((raw.wake_state.chemical_state, raw.wake_state.plasticity_state))
    # Native arrays only; wrapper metadata, Python heap and copies are not counted.
    return sum(array.nbytes for array in arrays)


def make_controller(
    actor: TopologyActor,
    prototype: TopologyCoupledScalingLearner,
    *,
    max_resource_bytes: int = 1_000_000,
) -> TopologyController:
    builder = prototype.build_from_state  # Bind ONCE: guard/preparation check identity.
    evaluator = PromotionGuardEvaluator[Array, Array, Array, CoupledScalingTopologyState](
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
    policy = PromotionPolicy(
        "one_minus_mean_absolute_probability_error_v1",
        "raw_native_tensor_bytes_v1",
        0.0,
        -1.0,
        1.0,
        1.0,
        max_resource_bytes,
        ("binary",),
    )
    return TopologyController(
        actor,
        policy=policy,
        evaluator=evaluator,
        build_learner=builder,
        state_digest=state_digest,
    )


def consolidator(
    removed_units: tuple[int, ...] = (0, 1), *, failed_budget: bool = False
) -> CoupledScalingPruneConsolidator:
    # Deterministic clocks exercise accounting; neither latency nor wall throughput
    # is measured by this integration example.
    ticks = iter((0.0, 1.0))
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_wall_seconds=0.0 if failed_budget else 5.0),
        (lambda: next(ticks, 1.0)) if failed_budget else (lambda: 0.0),
    )
    return CoupledScalingPruneConsolidator(
        removed_units,
        probes(),
        limits=CoupledPruneLimits(1e-12, 1e-12, 1e-12)
        if removed_units == (0, 1)
        else CoupledPruneLimits(0, 0, 0),
        budget=budget,
        inference_steps=2,
        inference_learning_rate=0.05,
    )


def continue_candidate(runtime: TopologyRuntime, clock: LogicalClock) -> None:
    training = probes()[0]
    runtime.record_experience(
        Experience(
            "train-0",
            "selection",
            0,
            "actor-0",
            training.features,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(
        LabelArrival(
            "label-0",
            "train-0",
            "selection",
            2,
            "actor-0",
            training.targets,
        )
    )
    clock.advance_to(1)
    if runtime.train_ready():
        raise AssertionError("training occurred before label arrival")
    clock.advance_to(2)
    if len(runtime.train_ready()) != 1:
        raise AssertionError("expected exactly one arrived training update")


def run_variant(variant: str) -> dict[str, Any]:
    actor, runtime, clock, prototype = setup_runtime(variant)
    actor.serve(guards()[0].features, now=1)
    initial = actor.serving_snapshot()
    before_candidate = runtime.candidate_snapshot()
    refusals = []
    for name, operation in (
        ("damage", consolidator((1, 0))),
        ("budget", consolidator((0, 1), failed_budget=True)),
    ):
        try:
            runtime.consolidate(name, operation)
        except CoupledScalingPruneRefused as error:
            decision = error.decision
            expected = "rejected" if name == "damage" else "failed"
            if decision.status != expected:
                raise AssertionError(f"unexpected refusal {name}: {decision.reason}")
            if state_digest(runtime.candidate_snapshot().state) != state_digest(
                before_candidate.state
            ):
                raise AssertionError("refused consolidation changed candidate payload")
            if state_digest(actor.serving_snapshot()) != state_digest(initial):
                raise AssertionError("refused consolidation changed serving bundle")
            refusals.append(
                {
                    "scenario": name,
                    "status": decision.status,
                    "reason": decision.reason,
                    "diagnostics": decision.diagnostics,
                }
            )
        else:
            raise AssertionError("expected refused consolidation")
    receipt = runtime.consolidate("accepted", consolidator((0, 1)))
    pruned = runtime.candidate_snapshot()
    continue_candidate(runtime, clock)
    continued = runtime.candidate_snapshot()
    if state_digest(actor.serving_snapshot()) != state_digest(initial):
        raise AssertionError("candidate work changed serving bundle")
    rejected_controller = make_controller(actor, prototype, max_resource_bytes=0)
    try:
        rejected_controller.prepare(runtime, *guards(), now=2, training_ids=selection_ids())
    except PromotionRejected as error:
        rejected_report = error.report
    else:
        raise AssertionError("expected independent resource promotion refusal")
    if state_digest(actor.serving_snapshot()) != state_digest(initial):
        raise AssertionError("guard refusal changed serving bundle")
    controller = make_controller(actor, prototype)
    ticket = controller.prepare(
        runtime, *guards(), now=2, training_ids=selection_ids(), metadata={"route": ["pruned"]}
    )
    promoted_receipt = controller.commit(runtime, ticket)
    promoted = actor.serving_snapshot()
    served = actor.serve(guards()[0].features, now=2)
    controller.rollback(promoted_receipt)
    rolled_back = actor.serving_snapshot()
    if state_digest(rolled_back) != state_digest(replace(initial, generation=2)):
        raise AssertionError("rollback did not restore exact previous serving bundle")
    if state_digest(continued) != state_digest(runtime.candidate_snapshot()):
        raise AssertionError("promotion or rollback changed candidate")
    return {
        "variant": variant,
        "initial": initial,
        "refusals": refusals,
        "consolidation_receipt": receipt,
        "pruned_candidate": pruned,
        "continued_candidate": continued,
        "rejected_promotion_report": rejected_report,
        "accepted_promotion_report": ticket.report,
        "promotion_receipt": promoted_receipt,
        "promoted": promoted,
        "served": served,
        "rolled_back": rolled_back,
    }


def run_example() -> dict[str, Any]:
    records = [run_variant(variant) for variant in describe_example()["variants"]]
    parity = {}
    for stage in ("pruned_candidate", "continued_candidate"):
        pc = cast(Any, records[0][stage]).state.raw_state.raw_parameters
        neutral = cast(Any, records[1][stage]).state.raw_state.raw_parameters
        parity[stage] = state_digest(pc) == state_digest(neutral)
    if not all(parity.values()):
        raise AssertionError("PC/neutral parameter parity lost")
    return {
        **describe_example(),
        "probes": probes(),
        "guards": guards(),
        "excluded_selection_ids": sorted(selection_ids()),
        "pc_neutral_parameter_parity": parity,
        "records": records,
    }
