"""Fixed local coupled pruning, consolidation, serving and rollback example.

Synthetic train/validation/inner-guard data exercise existing runtime boundaries.
Utility thresholds cover the full probability domain for a transaction check;
they do not establish learning benefit, acceptable deployment quality or latency.
No final-test data, checkpoint codec, external access or scientific study.
"""

from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json
from typing import Any, cast

import numpy as np

from src.adapters.topology_coupled_learner import TopologyCoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_prune_consolidation import CoupledPruneConsolidator, CoupledPruneRefused
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import (
    PromotableActor,
    PromotionRejected,
    ServingPromotionController,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import Array
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_topology import CoupledTopologyState, restore_coupled_topology
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import ServingConfiguration

TopologyRuntime = ActorShadowRuntime[Array, Array, Array, CoupledTopologyState]
TopologyActor = PromotableActor[Array, Array, Array, CoupledTopologyState]
TopologyController = ServingPromotionController[Array, Array, Array, CoupledTopologyState]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"unsupported engineering record: {type(value).__name__}")


def encode_record(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False, sort_keys=True, indent=indent)


def state_digest(value: Any) -> str:
    # Existing promotion ports require a digest. Canonical value encoding includes
    # all wrapper geometry, native wake/policy state and ordered coordinate records.
    return sha256(encode_record(value).encode("utf-8")).hexdigest()


def describe_example() -> dict[str, Any]:
    return {
        "protocol_id": "coupled_topology_runtime_engineering_v1",
        "scope": "local_engineering_integration_not_scientific_comparison",
        "variants": ["pc", "neutral_cpc", "active_cpc"],
        "seed": 23,
        "input_dim": 2,
        "hidden_dims": [3, 3],
        "learning_rate": 0.03,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "train_calls": 9,
        "guarded_infer_calls": 24,
        "coupled_energy_evaluations": 108,
        "limitations": "Fixed-step relaxed energy; no global damage, nonlinear scaling, quality or timing claim. Arbitrary consolidation remains refused with managed retention.",
    }


def probes() -> tuple[ScalingBatch, ...]:
    return (
        ScalingBatch(
            "train",
            ("train-0", "train-1"),
            np.array([[0.2, -0.1], [-0.3, 0.4]]),
            np.array([[1.0], [0.0]]),
        ),
        ScalingBatch(
            "validation",
            ("validation-0", "validation-1"),
            np.array([[0.7, 0.2], [-0.5, 0.1]]),
            np.array([[1.0], [0.0]]),
        ),
    )


def guards() -> tuple[GuardBatch[Array, Array], GuardBatch[Array, Array]]:
    return (
        GuardBatch(
            "new",
            (("guard-new", "n0"),),
            2,
            np.array([[0.6, -0.3]]),
            np.array([[1.0]]),
            "inner_guard",
        ),
        GuardBatch(
            "old",
            (("guard-old", "o0"),),
            1,
            np.array([[-0.4, 0.2]]),
            np.array([[0.0]]),
            "inner_guard",
        ),
    )


def selection_ids() -> frozenset[tuple[str, str]]:
    return frozenset(("selection", row) for probe in probes() for row in probe.row_ids)


def source_model(variant: str) -> CoupledPredictiveCodingNetwork:
    model: CoupledPredictiveCodingNetwork
    if variant == "pc":
        model = CoupledPredictiveCodingNetwork(2, (3, 3), 23)
    elif variant in ("neutral_cpc", "active_cpc"):
        config = (
            CoupledCircadianConfig.matched_pc_control()
            if variant == "neutral_cpc"
            else CoupledCircadianConfig()
        )
        model = CoupledCircadianPredictiveCodingNetwork(2, (3, 3), 23, config=config)
    else:
        raise ValueError("unknown fixed example variant")
    training = probes()[0]
    for _ in range(2):
        model.train_epoch(training.features, training.targets, 0.03, 2, 0.05)
    # Declared fixture surgery after genuine history, retaining CPC traces/count.
    parameters = model.snapshot_parameters()
    parameters.hidden_weights[-1][:, 2] = 0
    parameters.hidden_biases[-1][:, 2] = 0
    parameters.output_weight[2, :] = 0
    model.restore_parameters(parameters)
    return model


def setup_runtime(
    variant: str,
) -> tuple[TopologyActor, TopologyRuntime, LogicalClock, TopologyCoupledLearner]:
    learner = TopologyCoupledLearner(
        source_model(variant),
        learning_rate=0.03,
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


def _state_valid(state: CoupledTopologyState) -> bool:
    try:
        restore_coupled_topology(state)
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def resource_bytes(state: CoupledTopologyState) -> int:
    snapshot = state.snapshot
    parameters = snapshot.parameters
    arrays = [
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    ]
    if snapshot.wake_state is not None:
        arrays.extend((snapshot.wake_state.chemical_state, snapshot.wake_state.plasticity_state))
    return sum(array.nbytes for array in arrays)


def make_controller(
    actor: TopologyActor, prototype: TopologyCoupledLearner, *, max_resource_bytes: int = 1_000_000
) -> TopologyController:
    builder = prototype.build_from_state  # Bind ONCE: guard/preparation check identity.
    evaluator = PromotionGuardEvaluator[Array, Array, Array, CoupledTopologyState](
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
        "native_tensor_bytes_v1",
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


def consolidator(unit: int, *, failed_budget: bool = False) -> CoupledPruneConsolidator:
    # Deterministic clocks exercise accounting; neither latency nor wall throughput
    # is measured by this integration example.
    ticks = iter((0.0, 1.0))
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_wall_seconds=0.0 if failed_budget else 5.0),
        (lambda: next(ticks, 1.0)) if failed_budget else (lambda: 0.0),
    )
    return CoupledPruneConsolidator(
        1,
        unit,
        probes(),
        limits=CoupledPruneLimits(0.0, 0.0, 0.0),
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
        ("damage", consolidator(0)),
        ("budget", consolidator(2, failed_budget=True)),
    ):
        try:
            runtime.consolidate(name, operation)
        except CoupledPruneRefused as error:
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
    receipt = runtime.consolidate("accepted", consolidator(2))
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
        pc = cast(Any, records[0][stage]).state.snapshot.parameters
        neutral = cast(Any, records[1][stage]).state.snapshot.parameters
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
