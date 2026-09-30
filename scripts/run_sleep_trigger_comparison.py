"""Write the fixed v13 train-only trigger gate or globally sealed outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from src.app.sleep_trigger_comparison import (
    manifest_digest,
    run_trigger_comparison,
    train_unscored_trigger_study,
)
from src.app.sleep_trigger_trial import PROTOCOL_ID, fixed_trigger_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    # Why this: an occupied destination should not rerun training or open final roles.
    if args.result.exists():
        raise FileExistsError(f"v13 trigger result already exists: {args.result}")
    manifest = fixed_trigger_manifest()
    if args.train_only:
        study = train_unscored_trigger_study(manifest)
        report = {
            "protocol_id": PROTOCOL_ID,
            "manifest": asdict(manifest),
            "manifest_digest": manifest_digest(manifest),
            "trials": [asdict(trial.facts) for trial in study.trials],
            "final_released": False,
        }
    else:
        report = asdict(run_trigger_comparison(manifest))
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with args.result.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    print(
        json.dumps(
            {"result": str(args.result), "sha256": sha256(args.result.read_bytes()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
