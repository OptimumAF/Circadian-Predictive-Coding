"""Write the fixed v14 train-only replay opportunities to new local JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.continual_trigger_replay_schedule import (
    TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL,
    TriggerReplayScheduleSession,
    fixed_trigger_replay_manifest,
)


def build_payload() -> str:
    """Serialize only prospective train-role facts, never decision/final values."""
    manifest = fixed_trigger_replay_manifest()
    rows: list[dict[str, Any]] = []
    for seed in manifest.seeds:
        session = TriggerReplayScheduleSession(manifest, seed=seed)
        opportunities: list[dict[str, Any]] = []
        for phase in ("a", "b"):
            if phase == "b":
                session.arrive_phase_b(manifest)
            epochs = (
                manifest.arrived.training.phase_a_epochs
                if phase == "a"
                else manifest.arrived.training.phase_b_epochs
            )
            for _ in range(epochs):
                item = session.complete_wake_epoch(manifest, source_role="train")
                opportunities.append(
                    {
                        "phase": item.phase,
                        "epoch": item.epoch,
                        "global_epoch": item.global_epoch,
                        "train_role_hash": item.train_role_hash,
                        "train_role_ids": item.train_role_ids,
                        "retention": asdict(item.retention),
                        "retained_order_ids": item.retained_order_ids,
                        "selected_ids": item.selected_ids,
                        "method_work": [asdict(work) for work in item.method_work],
                    }
                )
        rows.append(
            {
                "seed": seed,
                "manifest_digest": session.manifest_digest,
                "opportunities": opportunities,
            }
        )
    return (
        json.dumps(
            {
                "protocol_id": TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL,
                "resolved_manifest": asdict(manifest),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    result_path = parser.parse_args().result
    payload = build_payload()
    with result_path.open("x", encoding="utf-8") as output:
        output.write(payload)
    print(json.dumps({"result": str(result_path), "sha256": sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
