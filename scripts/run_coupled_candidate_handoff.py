"""Describe or run fixed coupled handoff and resource controls."""

import argparse
import json
from pathlib import Path

from src.adapters.coupled_live_checkpoints import (
    describe_complete_checkpoint_example,
    run_complete_checkpoint_example,
)

from src.adapters.coupled_candidate_handoff import describe_example, run_example
from src.adapters.coupled_managed_serving import (
    describe_busy_committed_serving_example,
    describe_cleanup_failure_example,
    describe_committed_serving_example,
    describe_managed_serving_example,
    run_busy_committed_serving_example,
    run_cleanup_failure_example,
    run_committed_serving_example,
    run_managed_serving_example,
)
from src.adapters.coupled_retention_example import (
    describe_elapsed_retention_example,
    run_elapsed_retention_example,
)
from src.adapters.coupled_managed_consent import (
    describe_managed_consent_example,
    run_managed_consent_example,
)
from src.adapters.coupled_managed_cleanup import (
    describe_managed_cleanup_example,
    describe_owned_copy_example,
    run_managed_cleanup_example,
    run_owned_copy_example,
)
from src.adapters.coupled_handoff_resources import (
    describe_resource_stop_example,
    run_resource_stop_example,
)
from src.adapters.coupled_repeated_handoff import (
    describe_endurance_example,
    describe_repeated_example,
    run_endurance_example,
    run_repeated_example,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument(
        "--repeat-handoffs",
        action="store_true",
        help="use the fixed three-cycle cumulative control",
    )
    controls.add_argument(
        "--resource-stops", action="store_true", help="use the fixed original elapsed/RSS controls"
    )
    controls.add_argument(
        "--endurance", action="store_true", help="use the fixed16-cycle physical-resource control"
    )
    controls.add_argument(
        "--managed-consent",
        action="store_true",
        help="use the fixed managed consent/opt-out control",
    )
    controls.add_argument(
        "--managed-cleanup",
        action="store_true",
        help="use the fixed managed raw-payload cleanup control",
    )
    controls.add_argument(
        "--managed-copies",
        action="store_true",
        help="use the fixed retained raw-copy byte control",
    )
    controls.add_argument(
        "--elapsed-retention",
        action="store_true",
        help="use the fixed original-clock expiry and bounded worker control",
    )
    controls.add_argument(
        "--managed-serving-expiry",
        action="store_true",
        help="use the fixed managed cache and pending-promotion expiry control",
    )
    controls.add_argument(
        "--committed-serving-expiry",
        action="store_true",
        help="use the fixed current/previous bundle expiry and rollback control",
    )
    controls.add_argument(
        "--busy-serving-cleanup",
        action="store_true",
        help="use the fixed busy-holder stop and terminal cleanup control",
    )
    controls.add_argument(
        "--cleanup-failure",
        action="store_true",
        help="use the fixed erasure failure and terminal cleanup retry control",
    )
    controls.add_argument(
        "--complete-checkpoints",
        action="store_true",
        help="save complete data checkpoints during live committed-serving continuation and cleanup",
    )
    parser.add_argument(
        "--directory", type=Path, help="New directory for complete checkpoint files"
    )
    args = parser.parse_args()
    if args.directory is not None and (not args.complete_checkpoints or args.describe):
        parser.error("--directory is only valid with --run --complete-checkpoints")
    if args.complete_checkpoints and args.run and args.directory is None:
        parser.error("--directory is required with --run --complete-checkpoints")
    handlers = {
        "complete_checkpoints": (
            describe_complete_checkpoint_example,
            lambda: run_complete_checkpoint_example(args.directory),
        ),
        "repeat_handoffs": (describe_repeated_example, run_repeated_example),
        "resource_stops": (describe_resource_stop_example, run_resource_stop_example),
        "endurance": (describe_endurance_example, run_endurance_example),
        "managed_consent": (describe_managed_consent_example, run_managed_consent_example),
        "managed_cleanup": (describe_managed_cleanup_example, run_managed_cleanup_example),
        "managed_copies": (describe_owned_copy_example, run_owned_copy_example),
        "elapsed_retention": (describe_elapsed_retention_example, run_elapsed_retention_example),
        "managed_serving_expiry": (describe_managed_serving_example, run_managed_serving_example),
        "cleanup_failure": (describe_cleanup_failure_example, run_cleanup_failure_example),
        "busy_serving_cleanup": (
            describe_busy_committed_serving_example,
            run_busy_committed_serving_example,
        ),
        "committed_serving_expiry": (
            describe_committed_serving_example,
            run_committed_serving_example,
        ),
    }
    describe, run = describe_example, run_example
    for flag, pair in handlers.items():
        if getattr(args, flag):
            describe, run = pair
    value = describe() if args.describe else run()
    print(
        json.dumps(
            value,
            indent=2 if args.describe else None,
            separators=None if args.describe else (",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
