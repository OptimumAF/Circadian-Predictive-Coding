"""Save one fixed, train-only FIFO/reservoir schedule without model scores."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from src.app.continual_arrived_benchmark import ContinualArrivedRolesConfig
from src.app.continual_matched_replay_schedule import (
    MATCHED_REPLAY_SCHEDULE_PROTOCOL,
    PARTITIONED_REPLAY_SAMPLER,
    PARTITIONED_REPLAY_SCHEDULE_PROTOCOL,
    ROTATING_REPLAY_SAMPLER,
    ROTATING_REPLAY_SCHEDULE_PROTOCOL,
    MatchedReplayScheduleManifest,
    MatchedReplayScheduleSession,
)
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig, ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import ReplayAllocation


def _manifest(policy: ReplayRetentionPolicy) -> MatchedReplayScheduleManifest:
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
            replay_steps=2,
            replay_memory_size=1,
            replay_prioritized=False,
            replay_class_balanced=False,
            replay_inference_steps=3,
        ),
        replay_max_examples=4,
        replay_max_bytes=96,
    )
    return MatchedReplayScheduleManifest(
        arrived=ContinualArrivedRolesConfig(training, 0.2, 0.2),
        seeds=(17, 19),
        policy=policy,
        replay_updates_per_sleep=2,
        pc_replay_inference_steps=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--partitioned", action="store_true", help="use combined recent/reservoir budgets"
    )
    parser.add_argument(
        "--rotate-reservoir",
        action="store_true",
        help="rotate reservoir proposals across boundaries",
    )
    args = parser.parse_args()
    if args.rotate_reservoir and not args.partitioned:
        parser.error("--rotate-reservoir requires --partitioned")
    result_path = args.result
    # Why this: a fixed smoke should not spend its budget before refusing
    # an occupied output; the later exclusive write still protects races.
    if result_path.exists():
        raise FileExistsError(f"matched replay schedule result already exists: {result_path}")
    rows = []
    policies = (
        (ReplayRetentionPolicy("seeded_reservoir", 53),)
        if args.partitioned
        else (ReplayRetentionPolicy("recent_fifo"), ReplayRetentionPolicy("seeded_reservoir", 53))
    )
    allocation = (
        ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48))
        if args.partitioned
        else None
    )
    for policy in policies:
        manifest = _manifest(policy)
        for seed in manifest.seeds:
            if args.rotate_reservoir:
                session = MatchedReplayScheduleSession(
                    manifest, seed=seed, allocation=allocation, rotate_reservoir=True
                )
            elif allocation is None:
                session = MatchedReplayScheduleSession(manifest, seed=seed)
            else:
                session = MatchedReplayScheduleSession(manifest, seed=seed, allocation=allocation)
            boundaries = []
            for phase in ("a", "b"):
                if phase == "b":
                    session.arrive_phase_b(manifest)
                count = (
                    manifest.arrived.training.phase_a_epochs
                    if phase == "a"
                    else manifest.arrived.training.phase_b_epochs
                )
                for _ in range(count):
                    boundary = session.complete_wake_epoch(manifest, source_role="train")
                    if boundary is not None:
                        boundaries.append(
                            {
                                "phase": boundary.phase,
                                "epoch": boundary.epoch,
                                "train_role_hash": boundary.train_role_hash,
                                "retention": asdict(boundary.retention),
                                "retained_order_ids": boundary.retained_order_ids,
                                "selected_ids": boundary.selected_ids,
                                "method_work": [asdict(work) for work in boundary.method_work],
                            }
                        )
                        if allocation is not None:
                            partitions = session.partition_retention
                            assert partitions is not None
                            boundaries[-1]["partitions"] = asdict(partitions)
                        if args.rotate_reservoir:
                            boundaries[-1]["rotation_cursor"] = asdict(session.rotation_cursor)
            rows.append(
                {
                    "policy": asdict(policy),
                    "seed": seed,
                    "manifest_digest": session.manifest_digest,
                    "resolved_manifest": asdict(manifest),
                    "boundaries": boundaries,
                }
            )
            if allocation is not None:
                rows[-1]["allocation"] = asdict(allocation)
                rows[-1]["sampling_policy"] = (
                    ROTATING_REPLAY_SAMPLER if args.rotate_reservoir else PARTITIONED_REPLAY_SAMPLER
                )
                rows[-1]["protocol_id"] = session.protocol_id
    payload = (
        json.dumps(
            {
                "protocol_id": ROTATING_REPLAY_SCHEDULE_PROTOCOL
                if args.rotate_reservoir
                else (
                    PARTITIONED_REPLAY_SCHEDULE_PROTOCOL
                    if args.partitioned
                    else MATCHED_REPLAY_SCHEDULE_PROTOCOL
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with result_path.open("x", encoding="utf-8") as output:
        output.write(payload)
    # Why this: text-mode newline translation can change on-disk bytes on
    # Windows; the reported digest must identify the file actually written.
    print(
        json.dumps(
            {"result": str(result_path), "sha256": sha256(result_path.read_bytes()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
