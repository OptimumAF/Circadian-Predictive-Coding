"""Run one fixed, small local v8 FIFO/reservoir comparison and save every row."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from src.app.continual_arrived_benchmark import ContinualArrivedRolesConfig
from src.app.continual_replay_policy_comparison import (
    ReplayPolicyComparisonManifest,
    run_replay_policy_comparison,
)
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.replay_retention import ReplayRetentionPolicy
from src.infra.local_result_json import write_local_result_json
from src.infra.circadian_checkpoint_files import TrustedLocalReplayPolicyCheckpointStore


def _fixed_manifest() -> ReplayPolicyComparisonManifest:
    training = ContinualGlobalSealConfig(
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
    )
    return ReplayPolicyComparisonManifest(
        arrived=ContinualArrivedRolesConfig(training, 0.2, 0.2),
        seeds=(17, 19),
        policies=(
            ReplayRetentionPolicy("recent_fifo"),
            ReplayRetentionPolicy("seeded_reservoir", seed=53),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")
    if args.checkpoint is not None and args.result.resolve() == args.checkpoint.resolve():
        parser.error("--result and --checkpoint must be different paths")
    # Why this: the checkpoint store replaces its file during training, so a
    # fresh CLI run must protect an existing checkpoint before source access.
    if args.result.exists():
        raise FileExistsError(f"v8 policy result already exists: {args.result}")
    if args.checkpoint is not None and not args.resume and args.checkpoint.exists():
        raise FileExistsError(f"v8 policy checkpoint already exists: {args.checkpoint}")
    store = (
        TrustedLocalReplayPolicyCheckpointStore(args.checkpoint)
        if args.checkpoint is not None
        else None
    )
    result = run_replay_policy_comparison(
        _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=args.resume
    )
    write_local_result_json(result, args.result)
    print(
        json.dumps(
            {
                "result": str(args.result),
                "sha256": sha256(args.result.read_bytes()).hexdigest(),
                "manifest_digest": result.manifest_digest,
                "scores": [
                    {
                        "policy": item.policy.name,
                        "seed": seed.arrived.seed,
                        "backprop": seed.arrived.metrics.backprop.balanced_score,
                        "predictive_coding": seed.arrived.metrics.predictive_coding.balanced_score,
                        "circadian": seed.arrived.metrics.circadian_predictive_coding.balanced_score,
                        "replay_updates": seed.exposure.after_b.replay_updates,
                    }
                    for item in result.policies
                    for seed in item.seeds
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
