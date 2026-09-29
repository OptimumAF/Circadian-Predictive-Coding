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
    MatchedReplayScheduleManifest,
    MatchedReplayScheduleSession,
)
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.replay_retention import ReplayRetentionPolicy


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
    result_path = parser.parse_args().result
    rows = []
    for policy in (
        ReplayRetentionPolicy("recent_fifo"),
        ReplayRetentionPolicy("seeded_reservoir", 53),
    ):
        manifest = _manifest(policy)
        for seed in manifest.seeds:
            session = MatchedReplayScheduleSession(manifest, seed=seed)
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
            rows.append(
                {
                    "policy": asdict(policy),
                    "seed": seed,
                    "manifest_digest": session.manifest_digest,
                    "resolved_manifest": asdict(manifest),
                    "boundaries": boundaries,
                }
            )
    payload = (
        json.dumps(
            {"protocol_id": MATCHED_REPLAY_SCHEDULE_PROTOCOL, "rows": rows},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with result_path.open("x", encoding="utf-8") as output:
        output.write(payload)
    print(json.dumps({"result": str(result_path), "sha256": sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
