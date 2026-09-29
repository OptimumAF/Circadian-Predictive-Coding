"""Print a fixed, tiny outer-selected continual comparison and full trial ledger."""

from __future__ import annotations

from dataclasses import asdict, replace
import json

from src.app.continual_arrived_benchmark import ContinualArrivedRolesConfig
from src.app.continual_arrived_selection import (
    ArrivedSelectionCandidate,
    run_arrived_outer_selection,
)
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig


def main() -> None:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=1,
        phase_b_epochs=1,
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
    )
    first = ContinualArrivedRolesConfig(training, 0.2, 0.2)
    second = replace(
        first,
        training=replace(
            training,
            backprop_learning_rate=training.backprop_learning_rate * 0.7,
            pc_learning_rate=training.pc_learning_rate * 0.7,
            circadian_learning_rate=training.circadian_learning_rate * 0.7,
        ),
    )
    result = run_arrived_outer_selection(
        (
            ArrivedSelectionCandidate("default", first),
            ArrivedSelectionCandidate("lower_rate", second),
        ),
        [17, 19],
    )
    report = {
        "protocol_id": result.protocol_id,
        "seeds": result.seeds,
        "candidate_rates": {
            item.candidate_id: {
                "backprop": item.config.training.backprop_learning_rate,
                "predictive_coding": item.config.training.pc_learning_rate,
                "circadian_predictive_coding": item.config.training.circadian_learning_rate,
            }
            for item in result.candidates
        },
        "trials": [asdict(item) for item in result.trials],
        "candidate_sleep_histories": [asdict(item) for item in result.candidate_sleep_histories],
        "selections": [asdict(item) for item in result.selections],
        "freeze": asdict(result.freeze),
        "final_seed_scores": [
            {
                "seed": item.seed,
                "backprop": item.metrics.backprop.balanced_score,
                "predictive_coding": item.metrics.predictive_coding.balanced_score,
                "circadian_predictive_coding": item.metrics.circadian_predictive_coding.balanced_score,
                "selected_sleep_events": [
                    asdict(event) for event in item.metrics.circadian_predictive_coding.sleep_events
                ],
                "final_role_hashes": {
                    key: digest
                    for key, digest in item.role_hashes.items()
                    if key.endswith("final_test")
                },
            }
            for item in result.final_seed_results
        ],
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
