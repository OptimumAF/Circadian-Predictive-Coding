"""Save four fixed, unscored matched-replay training traces locally."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.run_continual_matched_replay_schedule_smoke import _manifest
from src.app.continual_matched_replay_runner import (
    MATCHED_REPLAY_TRAINING_PROTOCOL,
    run_matched_replay_training,
)
from src.core.replay_retention import ReplayRetentionPolicy


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
            result = run_matched_replay_training(manifest, seed=seed)
            clocks = result.pending.state.circadian_model.get_sleep_clocks()
            rows.append(
                {
                    "policy": asdict(policy),
                    "seed": seed,
                    "manifest_digest": result.manifest_digest,
                    "resolved_manifest": asdict(manifest),
                    "train_role_hashes": {
                        "a": result.pending.phase_a.split_hashes["train"],
                        "b": result.pending.phase_b.split_hashes["train"],
                    },
                    "guard_role_hashes": {
                        "a": result.pending.phase_a.split_hashes["inner_guard"],
                        "b": result.pending.phase_b.split_hashes["inner_guard"],
                    },
                    "boundaries": [asdict(item) for item in result.boundaries],
                    "circadian_clocks": asdict(clocks),
                }
            )
    payload = (
        json.dumps(
            {"protocol_id": MATCHED_REPLAY_TRAINING_PROTOCOL, "rows": rows},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with result_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    print(json.dumps({"result": str(result_path), "sha256": sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
