"""Save the fixed eight-trial matched replay side-effect ablation locally."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.run_continual_matched_replay_outcomes import _fixed_manifest, _seed_payload
from src.app.continual_replay_side_effect_ablation import (
    ReplaySideEffectAblationManifest,
    run_replay_side_effect_ablation,
)


def _payload() -> str:
    result = run_replay_side_effect_ablation(ReplaySideEffectAblationManifest(_fixed_manifest()))
    return (
        json.dumps(
            {
                "protocol_id": result.protocol_id,
                "manifest": asdict(result.manifest),
                "manifest_digest": result.manifest_digest,
                "outcomes": [
                    {
                        "side_effect_policy": group.side_effect_policy,
                        "retention": [
                            {
                                "policy": asdict(retention.policy),
                                "seeds": [_seed_payload(seed) for seed in retention.seeds],
                                "aggregate": asdict(retention.aggregate),
                            }
                            for retention in group.retention
                        ],
                    }
                    for group in result.outcomes
                ],
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
    path = parser.parse_args().result
    if path.exists():
        raise FileExistsError(f"result already exists: {path}")
    payload = _payload()
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    print(json.dumps({"result": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
