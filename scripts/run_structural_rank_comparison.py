"""Save fixed v12 train-only facts or globally sealed structural outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from src.app.structural_rank_comparison import (
    _manifest_digest,
    run_structural_rank_comparison,
    train_unscored_structural_rank_study,
)
from src.app.structural_rank_trial import PROTOCOL_ID, fixed_structural_rank_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    manifest = fixed_structural_rank_manifest()
    if args.train_only:
        study = train_unscored_structural_rank_study(manifest)
        report = {
            "protocol_id": PROTOCOL_ID,
            "manifest": asdict(manifest),
            "manifest_digest": _manifest_digest(manifest),
            "trials": [asdict(trial.facts) for trial in study.trials],
            "final_released": False,
        }
    else:
        report = asdict(run_structural_rank_comparison(manifest))
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
