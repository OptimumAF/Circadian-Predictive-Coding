"""One native gradient update followed by an inspectable local guard refusal.

Input: an explicit seed. Output: JSON observations from real native/serving ports.
A zero serialized-state budget deliberately exercises rejection. No final data,
training comparison, checkpoint decoding, policy tuning or promotion is performed.
"""

from dataclasses import asdict
from hashlib import sha256
import pickle
from time import perf_counter

import numpy as np

from src.adapters.numpy_learners import Array, BackpropLearner, BackpropSnapshot
from src.app.actor_shadow import ActorShadowRuntime
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor, PromotionRejected, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.backprop_mlp import BackpropMLP
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import ServingConfiguration


def _serialized(value: object) -> bytes:
    # Trusted in-memory state only. This example never unpickles external data.
    return pickle.dumps(value, protocol=5)


def _digest(value: object) -> str:
    return sha256(_serialized(value)).hexdigest()


def _controller(
    actor: PromotableActor[Array, Array, Array, BackpropSnapshot], source: BackpropLearner
) -> ServingPromotionController[Array, Array, Array, BackpropSnapshot]:
    def build(_state: BackpropSnapshot) -> BackpropLearner:
        return source.fork()

    def state_valid(state: BackpropSnapshot) -> bool:
        try:
            source.fork().restore_state(state)
        except ValueError:
            return False
        return True

    evaluator = PromotionGuardEvaluator(
        build_learner=build,
        utility=lambda prediction, target: float(1 - np.mean((prediction - target) ** 2)),
        actions=lambda prediction: tuple("1" if p >= 0.5 else "0" for p in prediction[:, 0]),
        state_valid=state_valid,
        prediction_valid=lambda prediction: bool(np.isfinite(prediction).all()),
        resource_bytes=lambda state: len(_serialized(state)),
        state_digest=_digest,
        clock=perf_counter,
    )
    # Why: zero bytes is an explicit fault injection, fixed before any run. Other
    # bounds allow all valid binary probabilities; no threshold is fitted to data.
    policy = PromotionPolicy(
        "binary_one_minus_brier_v1",
        "trusted_local_pickle_bytes_v1",
        0.0,
        -1.0,
        1.0,
        60.0,
        0,
        ("0", "1"),
    )
    return ServingPromotionController(
        actor,
        policy=policy,
        evaluator=evaluator,
        build_learner=build,
        state_digest=_digest,
    )


def run_rejection_demo(seed: int = 23) -> dict[str, object]:
    """Expose original rejection measurements while keeping serving unchanged."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    source = BackpropLearner(BackpropMLP(2, 4, seed=seed), learning_rate=0.03)
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=_digest,
        metadata={"route": "original"},
    )
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), perf_counter),
        max_experiences=1,
        max_consolidations=0,
        actor=actor,
    )
    features = np.array([[0.3, -0.2]])
    runtime.record_experience(
        Experience(
            "s0",
            "train",
            0,
            "actor-0",
            features,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(LabelArrival("label-0", "s0", "train", 2, "actor-0", np.array([[1.0]])))
    clock.advance_to(2)
    runtime.train_ready()
    served_before = actor.serve(features, now=2)
    before = actor.serving_snapshot()
    controller = _controller(actor, source)
    new = GuardBatch(
        "new", (("guard-new", "g0"),), 2, np.array([[0.7, -0.4]]), np.array([[1.0]]), "inner_guard"
    )
    old = GuardBatch(
        "old", (("guard-old", "g0"),), 2, np.array([[-0.6, 0.5]]), np.array([[0.0]]), "inner_guard"
    )
    try:
        controller.prepare(runtime, new, old, now=2, training_ids=frozenset({("train", "s0")}))
    except PromotionRejected as error:
        report = error.report
    else:
        raise RuntimeError("zero-resource fault injection unexpectedly accepted a candidate")
    after = actor.serving_snapshot()
    served_after = actor.serve(features, now=2)
    return {
        "schema_version": 1,
        "protocol_id": "native_promotion_rejection_v1",
        "scope": "integration_fault_injection",
        "seed": seed,
        "method": "backprop",
        "completed_updates": runtime.candidate_snapshot().wake_updates,
        "guard_roles": [new.role, old.role],
        "guard_report": asdict(report),
        "serving_unchanged": _serialized(before) == _serialized(after),
        "actor_version_before": served_before.actor_version,
        "actor_version_after": served_after.actor_version,
        "actor_predictions_before": served_before.prediction.tolist(),
        "actor_predictions_after": served_after.prediction.tolist(),
        "cache_hit_after": served_after.cache_hit,
    }
