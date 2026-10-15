"""Run with python -m scripts.run_native_stream_demo; write one JSON report."""

import argparse
import json

from src.adapters.native_stream_demo import run_demo, run_coupled_demo, run_matched_demo
from src.adapters.promotion_rejection_demo import run_rejection_demo


def main() -> None:
    scenarios = {
        "stream": run_demo,
        "promotion-rejection": run_rejection_demo,
        "coupled-neutral": run_coupled_demo,
        "matched-baselines": run_matched_demo,
    }
    parser = argparse.ArgumentParser(description="Local delayed-label native adapter stream")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--scenario", choices=tuple(scenarios), default="stream")
    args = parser.parse_args()
    try:
        report = scenarios[args.scenario](args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
