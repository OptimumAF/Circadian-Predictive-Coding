"""Describe or execute fixed local rehearsal and capacity demonstrations."""

import argparse
import json
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

from src.adapters.stream_rehearsal_pilot import (
    describe_paired_protocol,
    describe_protocol,
    describe_tracked_protocol,
    run_paired_pilot,
    run_pilot,
    run_tracked_pilot,
)
from src.adapters.long_stream_rehearsal_pilot import (
    describe_protocol as describe_long_protocol,
    run_pilot as run_long_pilot,
)
from src.adapters.slow_teacher_pilot import (
    describe_protocol as describe_teacher_protocol,
    run_pilot as run_teacher_pilot,
)
from src.adapters.capacity_pilot import (
    describe_bp_protocol,
    describe_protocol as describe_capacity_protocol,
    run_bp_pilot,
    run_pilot as run_capacity_pilot,
)
from src.adapters.capacity_comparison_pilot import (
    describe_protocol as describe_capacity_comparison_protocol,
    run_pilot as run_capacity_comparison_pilot,
)
from src.adapters.stream_removal_confirmation import (
    describe_protocol as describe_removal_confirmation,
    run_pilot as run_removal_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--describe", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=(
            "single",
            "paired-retention",
            "tracked-learnability",
            "long-stream",
            "slow-teacher",
            "capacity",
            "capacity-bp",
            "capacity-comparison",
            "removal-confirmation",
        ),
        default="single",
        help="select a fixed rehearsal scenario",
    )
    parser.add_argument(
        "--diagnose-learnability",
        action="store_true",
        help="run isolated reserved sixteen-step probes on copied checkpoints",
    )
    replay_control = parser.add_mutually_exclusive_group()
    replay_control.add_argument(
        "--recent-only", action="store_true", help="use the fixed recent FIFO rehearsal control"
    )
    replay_control.add_argument(
        "--no-replay", action="store_true", help="remove rehearsal entirely from the single case"
    )
    args = parser.parse_args()
    if args.no_replay and args.scenario != "single":
        parser.error("--no-replay requires --scenario single")
    if args.scenario == "removal-confirmation":
        if args.recent_only or args.diagnose_learnability:
            parser.error(
                "removal-confirmation fixes both arms without probes; omit single-case flags"
            )
        selected = describe_removal_confirmation if args.describe else run_removal_confirmation
        report = selected()
    elif args.scenario == "capacity-comparison":
        if args.recent_only or args.diagnose_learnability:
            parser.error("capacity-comparison fixes all cohorts; omit single-case flags")
        selected = (
            describe_capacity_comparison_protocol
            if args.describe
            else run_capacity_comparison_pilot
        )
        report = selected()
    elif args.scenario == "capacity-bp":
        if args.recent_only or args.diagnose_learnability:
            parser.error(
                "capacity-bp fixes the BP schedule without replay/probes; omit single-case flags"
            )
        selected = describe_bp_protocol if args.describe else run_bp_pilot
        report = selected()
    elif args.scenario == "capacity":
        if args.recent_only or args.diagnose_learnability:
            parser.error(
                "capacity fixes its schedule without replay/probes; omit single-case flags"
            )
        selected = describe_capacity_protocol if args.describe else run_capacity_pilot
        report = selected()
    elif args.scenario == "slow-teacher":
        if args.recent_only or args.diagnose_learnability:
            parser.error(
                "slow-teacher fixes rotating replay without probes; omit single-case flags"
            )
        selected = describe_teacher_protocol if args.describe else run_teacher_pilot
        report = selected()
    elif args.scenario == "long-stream":
        if args.recent_only or args.diagnose_learnability:
            parser.error("long-stream fixes rotating replay without probes; omit single-case flags")
        selected = describe_long_protocol if args.describe else run_long_pilot
        report = selected()
    elif args.scenario == "tracked-learnability":
        if args.recent_only or args.diagnose_learnability:
            parser.error(
                "tracked-learnability fixes rotating replay and probes; omit single-case flags"
            )
        selected = describe_tracked_protocol if args.describe else run_tracked_pilot
        report = selected()
    elif args.scenario == "paired-retention":
        if args.recent_only or args.diagnose_learnability:
            parser.error(
                "paired-retention fixes both policies without probes; omit single-case flags"
            )
        selected = describe_paired_protocol if args.describe else run_paired_pilot
        report = selected()
    else:
        selected = describe_protocol if args.describe else run_pilot
        if args.no_replay:
            report = (
                selected(no_replay=True, diagnose_learnability=True)
                if args.diagnose_learnability
                else selected(no_replay=True)
            )
        elif args.recent_only:
            report = selected(recent_only=True, diagnose_learnability=args.diagnose_learnability)
        else:
            report = (
                selected(diagnose_learnability=True) if args.diagnose_learnability else selected()
            )
    raw = json.dumps(report, allow_nan=False, indent=2)
    storage_limit = (
        2
        if args.scenario in ("slow-teacher", "capacity", "capacity-bp", "capacity-comparison")
        else 32
    ) * 1024**2
    if len(raw.encode("utf8")) > storage_limit:
        raise ValueError("rehearsal report exceeds fixed output allowance")
    print(raw)
    return 0 if args.describe or report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
