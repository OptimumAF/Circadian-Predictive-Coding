"""Compose existing native adapters and delayed-label runtime for a CPU example.

Inputs: an explicit seed. Outputs: JSON-compatible events and version metadata.
This fixed training-only stream does not score, tune, promote, sleep or resume a
checkpoint. Native model snapshots stay in memory; the report records their
format versions separately from the experiment cursor format.
"""

from dataclasses import asdict
from time import monotonic
from typing import TypeVar

import numpy as np

from src.adapters.numpy_learners import Array, BackpropLearner, BackpropSnapshot, CircadianLearner
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ForkableLearner
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.matched_backprop import MatchedBackpropNetwork, MatchedBackpropSnapshot

State = TypeVar(
    "State",
    bound=BackpropSnapshot | CircadianNetworkSnapshot | CoupledSnapshot | MatchedBackpropSnapshot,
)


def _record_arrivals(
    runtime: ActorShadowRuntime[Array, Array, Array, State],
    features: Array,
    targets: Array,
    tick: int,
) -> None:
    if tick < 4:
        runtime.record_experience(
            Experience(
                f"s{tick}",
                "demo",
                tick,
                "actor-0",
                features[tick : tick + 1],
                "train",
                ExperiencePermissions(training=True),
            )
        )
    if tick >= 2:
        index = tick - 2
        runtime.record_label(
            LabelArrival(
                f"label-{index}",
                f"s{index}",
                "demo",
                tick,
                "actor-0",
                targets[index : index + 1],
            )
        )


def _run_stream(
    method: str, learner: ForkableLearner[Array, Array, Array, State]
) -> dict[str, object]:
    features = np.array([[-0.8, -0.2], [-0.2, 0.7], [0.4, -0.5], [0.8, 0.3]])
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), monotonic),
        max_experiences=4,
        max_consolidations=0,
    )
    initial_predictions = learner.predict(features).tolist()
    events: list[dict[str, object]] = []
    for tick in range(6):
        clock.advance_to(tick)
        # Why: serving is observed before this tick's label/update work. No
        # future label enters the inbox until its declared arrival tick.
        served = runtime.actor.predict(features)
        _record_arrivals(runtime, features, targets, tick)
        applied = runtime.train_ready(max_updates=1)
        events.append(
            {
                "tick": tick,
                "observed_sample_id": f"s{tick}" if tick < 4 else None,
                "applied_sample_ids": [item.sample_id for item in applied],
                "actor_version": served.actor_version,
                "actor_predictions": served.prediction.tolist(),
            }
        )

    candidate = runtime.candidate_snapshot()
    # Restore into a detached adapter to inspect the trained candidate without
    # exposing or mutating the runtime's private learner.
    learner.restore_state(candidate.state)
    report: dict[str, object] = {
        "method": method,
        "experiment_state": {
            "format_version": 1,
            "completed_updates": candidate.wake_updates,
            "next_tick": clock.now() + 1,
        },
        "model_state": {
            "actor_version": candidate.actor_version,
            "candidate_version": candidate.learner_version,
            "snapshot_format_version": candidate.state.format_version,
        },
        "events": events,
        "initial_candidate_predictions": initial_predictions,
        "final_candidate_predictions": learner.predict(features).tolist(),
        "diagnostics": [asdict(item.diagnostic) for item in runtime.applied_updates],
    }
    if isinstance(candidate.state, CoupledSnapshot):
        wake = candidate.state.wake_state
        report["native_adaptation"] = (
            None
            if wake is None
            else {
                "wake_updates": wake.wake_updates,
                "chemistry": wake.chemical_state.tolist(),
                "gates": wake.plasticity_state.tolist(),
            }
        )
    return report


def run_demo(seed: int = 23) -> dict[str, object]:
    """Run the same six ticks and four native updates for both owned adapters."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    return {
        "schema_version": 1,
        "protocol_id": "native_adapter_stream_v1",
        "seed": seed,
        "scope": "integration_smoke",
        "settings": {
            "samples": 4,
            "label_delay_ticks": 2,
            "hidden_width": 4,
            "learning_rate": 0.03,
            "circadian_inference_steps": 2,
            "circadian_inference_learning_rate": 0.2,
            "sleep_calls": 0,
            "promotion_calls": 0,
        },
        "methods": [
            _run_stream(
                "backprop", BackpropLearner(BackpropMLP(2, 4, seed=seed), learning_rate=0.03)
            ),
            _run_stream(
                "circadian",
                CircadianLearner(
                    CircadianPredictiveCodingNetwork(2, 4, seed=seed),
                    learning_rate=0.03,
                    inference_steps=2,
                    inference_learning_rate=0.2,
                ),
            ),
        ],
    }


def run_coupled_demo(seed: int = 23) -> dict[str, object]:
    """Run the same delayed stream with coupled PC and its genuine neutral CPC."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    pc = CoupledPredictiveCodingNetwork(2, (4,), seed)
    cpc = CoupledCircadianPredictiveCodingNetwork(
        2,
        (4,),
        seed,
        config=CoupledCircadianConfig.matched_pc_control(),
    )
    return {
        "schema_version": 1,
        "protocol_id": "coupled_native_adapter_stream_v1",
        "seed": seed,
        "scope": "integration_smoke",
        "settings": {
            "samples": 4,
            "label_delay_ticks": 2,
            "hidden_width": 4,
            "learning_rate": 0.03,
            "inference_steps": 2,
            "inference_learning_rate": 0.2,
            "circadian_control": "matched_pc_control",
            "sleep_calls": 0,
            "promotion_calls": 0,
        },
        "methods": [
            _run_stream(
                name,
                CoupledLearner(
                    model, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
                ),
            )
            for name, model in (("coupled_pc", pc), ("coupled_cpc", cpc))
        ],
    }


def run_matched_demo(seed: int = 23) -> dict[str, object]:
    """Match initialized tensors/likelihood on the same delayed training stream.

    PC's relaxed total energy differs from BP's feedforward likelihood, and PC
    performs additional latent steps. This remains an integration example.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    pc = CoupledPredictiveCodingNetwork(2, (3, 2), seed)
    initial = pc.snapshot_parameters()
    cpc = CoupledCircadianPredictiveCodingNetwork(
        2, (3, 2), seed, config=CoupledCircadianConfig.matched_pc_control()
    )
    # Explicit tensor assignment, rather than relying on matching seed semantics.
    cpc.restore_parameters(initial)
    bp = MatchedBackpropNetwork(initial)
    return {
        "schema_version": 1,
        "protocol_id": "matched_binary_adapter_stream_v1",
        "seed": seed,
        "scope": "integration_smoke",
        "initial_parameters": {
            "hidden_weights": [w.tolist() for w in initial.hidden_weights],
            "hidden_biases": [b.tolist() for b in initial.hidden_biases],
            "output_weight": initial.output_weight.tolist(),
            "output_bias": initial.output_bias.tolist(),
        },
        "settings": {
            "samples": 4,
            "label_delay_ticks": 2,
            "hidden_dims": [3, 2],
            "learning_rate": 0.03,
            "latent_steps_per_update": {"bp": 0, "pc": 2, "cpc": 2},
            "inference_learning_rate": 0.2,
            "circadian_control": "matched_pc_control",
            "sleep_calls": 0,
            "promotion_calls": 0,
        },
        "methods": [
            _run_stream("matched_bp", MatchedBackpropLearner(bp, learning_rate=0.03)),
            *[
                _run_stream(
                    name,
                    CoupledLearner(
                        model, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
                    ),
                )
                for name, model in (("coupled_pc", pc), ("coupled_cpc", cpc))
            ],
        ],
    }
