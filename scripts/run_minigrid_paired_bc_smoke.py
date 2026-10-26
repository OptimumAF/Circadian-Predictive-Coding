"""One fixed row-exposure paired-episode MiniGrid acquisition case."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

MAX_OUTPUT_BYTES = 16 * 1024**2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    condition = parser.add_mutually_exclusive_group()
    condition.add_argument(
        "--endpoint-first",
        action="store_true",
        help="observe fixed endpoints before descriptive middle checkpoints",
    )
    condition.add_argument(
        "--public-cue",
        action="store_true",
        help="run the separately described two-head raw148 recurrent TRAIN-A control",
    )
    condition.add_argument(
        "--public-history",
        action="store_true",
        help="run the separately described296-input fixed-endpoint condition",
    )
    parser.add_argument("--output", type=Path, help="new JSON destination; never overwritten")
    parser.add_argument(
        "--recurring-training",
        action="store_true",
        help="public-cue original packaged TRAIN A/B/C/A recipe",
    )
    parser.add_argument(
        "--class-balanced-training",
        action="store_true",
        help="explicit weighted public-cue recurring TRAIN recipe",
    )
    parser.add_argument(
        "--activation-health",
        action="store_true",
        help="read existing TRAIN hidden-health diagnostics before each public-cue recurring readout",
    )
    args = parser.parse_args(argv)
    health_options: dict[str, Any] = {"activation_health": True} if args.activation_health else {}
    if args.recurring_training and not args.public_cue:
        parser.error("--recurring-training requires --public-cue")
    if args.class_balanced_training and not args.recurring_training:
        parser.error("--class-balanced-training requires --public-cue --recurring-training")
    if args.activation_health and not args.recurring_training:
        parser.error("--activation-health requires --public-cue --recurring-training")
    if args.describe and args.output is not None:
        parser.error("--describe does not publish a run artifact")
    if args.output is not None:
        if args.output.exists():
            parser.error("output already exists; preserve the original attempt")
        if not args.output.parent.is_dir():
            parser.error("output parent directory must exist")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from src.adapters.minigrid_paired_bc_pilot import describe_protocol, run_pilot

    if args.public_cue:
        from src.adapters.minigrid_public_cue_recurrent_pilot import (
            describe_protocol as cue_protocol,
            run_pilot as cue_run,
        )

    if args.describe:
        protocol = (
            cue_protocol(
                typed_input=True,
                recurring_training=True,
                class_balanced_training=args.class_balanced_training,
                **health_options,
            )
            if args.public_cue and args.recurring_training
            else cue_protocol()
            if args.public_cue
            else describe_protocol(public_history=True)
            if args.public_history
            else (
                describe_protocol(endpoint_first=True)
                if args.endpoint_first
                else describe_protocol()
            )
        )
        print(json.dumps(protocol, indent=2, allow_nan=False))
        return 0
    episodes = None
    if args.recurring_training:
        from src.infra.minigrid_affine_example import example_affine_training
        from scripts.run_minigrid_linear_fit_control import read_training_sources

        with example_affine_training() as path:
            episodes = read_training_sources(path, raw_frames=True)
    from src.shared.torch_runtime import require_torch

    torch = require_torch()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = (
        cue_run(
            episodes,
            recurring_training=True,
            class_balanced_training=args.class_balanced_training,
            **health_options,
        )
        if args.public_cue and args.recurring_training
        else cue_run()
        if args.public_cue
        else run_pilot(public_history=True)
        if args.public_history
        else (run_pilot(endpoint_first=True) if args.endpoint_first else run_pilot())
    )
    from src.infra.local_result_json import encode_bounded_local_json

    output_limit = 8 * 1024**2 if args.public_history or args.public_cue else MAX_OUTPUT_BYTES
    payload = encode_bounded_local_json(report, max_bytes=output_limit)
    if args.output is None:
        sys.stdout.write(payload.decode("utf8"))
    else:
        with args.output.open("xb") as handle:
            handle.write(payload)
        print(
            json.dumps(dict(status=report["status"], output=str(args.output), bytes=len(payload)))
        )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
