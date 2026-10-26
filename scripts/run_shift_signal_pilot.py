"""Describe or run one fixed local arrived-only shift diagnostic protocol."""

import argparse
import json
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

from src.adapters.shift_signal_pilot import describe_protocol, run_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--describe", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--calibrated", action="store_true", help="use frozen training-prefix calibration"
    )
    parser.add_argument(
        "--incidents", action="store_true", help="also report channel incident transitions"
    )
    parser.add_argument(
        "--incident-replay",
        action="store_true",
        help="admit bounded replay from prior feature incidents",
    )
    parser.add_argument(
        "--learnability",
        action="store_true",
        help="run isolated C probes on recurring replay snapshots",
    )
    args = parser.parse_args()
    if args.incident_replay and not (args.calibrated and args.incidents):
        parser.error("--incident-replay requires --calibrated --incidents")
    if args.learnability and not (args.calibrated and args.incidents and args.incident_replay):
        parser.error("--learnability requires --calibrated --incidents --incident-replay")
    if args.learnability:
        report = (
            describe_protocol(
                calibrated=True, incidents=True, incident_replay=True, learnability=True
            )
            if args.describe
            else run_pilot(calibrated=True, incidents=True, incident_replay=True, learnability=True)
        )
    elif args.incident_replay:
        report = (
            describe_protocol(calibrated=True, incidents=True, incident_replay=True)
            if args.describe
            else run_pilot(calibrated=True, incidents=True, incident_replay=True)
        )
    elif args.incidents:
        report = (
            describe_protocol(calibrated=args.calibrated, incidents=True)
            if args.describe
            else run_pilot(calibrated=args.calibrated, incidents=True)
        )
    elif args.calibrated:
        report = describe_protocol(calibrated=True) if args.describe else run_pilot(calibrated=True)
    else:
        report = describe_protocol() if args.describe else run_pilot()
    raw = json.dumps(report, allow_nan=False, indent=2)
    if len(raw.encode("utf8")) > 32 * 1024**2:
        raise ValueError("shift report exceeds fixed output allowance")
    print(raw)
    return 0 if args.describe or report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
