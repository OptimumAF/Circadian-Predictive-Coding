"""Save every fixed matched-replay policy/seed outcome without choosing a winner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.run_continual_matched_replay_schedule_smoke import _manifest
from src.app.continual_matched_replay_outcomes import (
    MatchedReplayOutcomeManifest,
    MatchedReplaySeedOutcome,
    run_matched_replay_outcomes,
)
from src.core.replay_retention import ReplayRetentionPolicy


def _fixed_manifest() -> MatchedReplayOutcomeManifest:
    fifo = ReplayRetentionPolicy("recent_fifo")
    planned = _manifest(fifo)
    return MatchedReplayOutcomeManifest(
        arrived=planned.arrived,
        seeds=planned.seeds,
        policies=(fifo, ReplayRetentionPolicy("seeded_reservoir", 53)),
        replay_updates_per_sleep=planned.replay_updates_per_sleep,
        pc_replay_inference_steps=planned.pc_replay_inference_steps,
    )


def _seed_payload(seed: MatchedReplaySeedOutcome) -> dict[str, object]:
    metrics = seed.arrived.metrics
    circadian = asdict(metrics.circadian_predictive_coding)
    # Why this: execution durations vary without changing training or
    # outcomes; retain all typed sleep facts except those wall-clock fields.
    sleep_events = circadian.pop("sleep_events")
    assert isinstance(sleep_events, (list, tuple))
    for event in sleep_events:
        event.pop("durations")
    return {
        "seed": seed.arrived.seed,
        "schedule_manifest_digest": seed.schedule_manifest_digest,
        "role_ids": seed.arrived.role_ids,
        "role_hashes": seed.arrived.role_hashes,
        "metrics": {
            "backprop": asdict(metrics.backprop),
            "predictive_coding": asdict(metrics.predictive_coding),
            "circadian_predictive_coding": circadian,
            "replay_retention": asdict(metrics.replay_retention),
        },
        "sleep_events_without_durations": sleep_events,
        "boundaries": [asdict(boundary) for boundary in seed.boundaries],
        "applied_work": [asdict(work) for work in seed.applied_work],
        "replay_exposure": asdict(seed.replay_exposure),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    path = parser.parse_args().result
    result = run_matched_replay_outcomes(_fixed_manifest())
    payload = (
        json.dumps(
            {
                "protocol_id": result.protocol_id,
                "manifest": asdict(result.manifest),
                "manifest_digest": result.manifest_digest,
                "policies": [
                    {
                        "policy": asdict(policy.policy),
                        "seeds": [_seed_payload(seed) for seed in policy.seeds],
                        "aggregate": asdict(policy.aggregate),
                    }
                    for policy in result.policies
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    print(json.dumps({"result": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
