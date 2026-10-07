"""Print a fixed, small v6 role-arrival and inner-guard audit report."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json

from src.app.continual_arrived_benchmark import (
    ContinualArrivedRolesConfig,
    run_continual_arrived_benchmark,
)
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig


def main() -> None:
    config = ContinualArrivedRolesConfig(
        training=ContinualGlobalSealConfig(
            sample_count_phase_a=40,
            sample_count_phase_b=40,
            hidden_dim=4,
            phase_a_epochs=2,
            phase_b_epochs=2,
            pc_inference_steps=2,
            circadian_inference_steps=2,
            circadian_sleep_interval_phase_a=1,
            circadian_sleep_interval_phase_b=1,
            circadian_config=CircadianConfig(
                sleep_mode="components",
                max_split_per_sleep=0,
                max_prune_per_sleep=0,
                replay_steps=1,
                replay_memory_size=1,
            ),
            replay_max_examples=4,
            replay_max_bytes=96,
        ),
        inner_guard_fraction=0.2,
        outer_selection_fraction=0.2,
    )
    result = run_continual_arrived_benchmark(config, [17, 19])
    report = {
        "protocol_id": result.protocol_id,
        "seeds": list(result.seeds),
        "seed_results": [
            {
                "seed": item.seed,
                "role_hashes": item.role_hashes,
                "guard_decisions": [asdict(decision) for decision in item.guard_decisions],
                "sleep_events": [
                    asdict(event) for event in item.metrics.circadian_predictive_coding.sleep_events
                ],
                "method_task_information": [
                    {
                        "method": information.method,
                        "phase": information.phase,
                        "arrived_phases": information.arrived_phases,
                    }
                    for information in item.method_task_information
                ],
                "role_access_counts": dict(
                    sorted(Counter(event.action for event in item.role_accesses).items())
                ),
                "final_release_events": sum(
                    event.role == "final_test" and event.event == "global_freeze"
                    for event in item.role_accesses
                ),
                "balanced_scores": {
                    "backprop": item.metrics.backprop.balanced_score,
                    "predictive_coding": item.metrics.predictive_coding.balanced_score,
                    "circadian_predictive_coding": (
                        item.metrics.circadian_predictive_coding.balanced_score
                    ),
                },
            }
            for item in result.seed_results
        ],
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
