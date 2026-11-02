"""Rebuild bundled TRAIN memory; replay on CPU or explicitly run native scenes.

Default replay verifies recorded inputs/actions without SDKs or research files.
--native needs the optional simulator and a fresh --output for partial evidence.
"""

import argparse
from importlib.resources import as_file, files
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.adapters.taskspace_memory_demo import (  # noqa: E402
    load_bundle,
    run_demo,
    training_probe_outputs,
    _retention_options,
    _RetentionOptions,
)
from src.core.replay_retention import ReplayRetentionPolicy  # noqa: E402
from src.core.nearest_retention import NearestRetentionPolicy, RecentPriorityRetentionPolicy  # noqa: E402

DEFAULT_BUNDLE = ROOT / "examples/taskspace_memory/episodes.json"


def _load_default_bundle():
    # Why: keep checkout/direct-file usage while wheels own their bundled data.
    # as_file also supports zipped package resources; admission remains bounded.
    if DEFAULT_BUNDLE.is_file():
        return load_bundle(DEFAULT_BUNDLE)
    resource = files("src.infra.taskspace_demo_data").joinpath("episodes.json")
    with as_file(resource) as path:
        return load_bundle(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--retention", choices=("recent_fifo", "seeded_reservoir", "recent_priority")
    )
    parser.add_argument("--max-prototypes", type=int)
    parser.add_argument("--retention-seed", type=int)
    parser.add_argument(
        "--training-probe",
        type=Path,
        metavar="ORIGINAL23_JSON",
        help="private one-update original probe_train23 fit; requires its fixed-digest export and --output; optional --native observes before/after23, no transfer claim",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="run frozen TRAIN scenes (default four; adaptation five at its final endpoint); requires simulator and --output",
    )
    parser.add_argument(
        "--adaptation-probe",
        nargs=2,
        type=Path,
        metavar=("TRAIN_JSON", "TEST_JSON"),
        help="isolated bounded-memory acquisition; original23/24 fixed exports, retention/cap/output required; --native observes final scenes",
    )
    parser.add_argument(
        "--return-task",
        choices=("pick17", "pick19", "pick20", "push17"),
        help="one original old TRAIN packet after --adaptation-probe; no learner adoption",
    )
    parser.add_argument(
        "--development-probe",
        action="store_true",
        help="score fixed Push TRAIN resets19/20 absent from bundled memory; requires --native/--output",
    )
    args = parser.parse_args(argv)
    if args.return_task is not None and args.adaptation_probe is None:
        parser.error("--return-task requires --adaptation-probe")
    if args.adaptation_probe is not None and (
        args.training_probe is not None
        or args.development_probe
        or args.retention is None
        or args.max_prototypes is None
        or args.output is None
    ):
        parser.error(
            "--adaptation-probe requires explicit --retention/--max-prototypes/--output; excludes other probes"
        )
    retained_options: _RetentionOptions = {}
    try:
        if args.retention is None:
            if args.max_prototypes is not None or args.retention_seed is not None:
                raise ValueError("--max-prototypes/--retention-seed require --retention")
        else:
            if args.training_probe is not None or args.development_probe:
                raise ValueError("--retention is incompatible with training/development probes")
            rule: NearestRetentionPolicy = (
                RecentPriorityRetentionPolicy(args.retention_seed)
                if args.retention == "recent_priority"
                else ReplayRetentionPolicy(args.retention, args.retention_seed)
            )
            maximum = 256 if args.max_prototypes is None else args.max_prototypes
            _retention_options(rule, maximum)
            retained_options = dict(retention=rule, max_prototypes=maximum)
    except ValueError as error:
        parser.error(str(error))
    if args.training_probe is not None and args.development_probe:
        parser.error("--training-probe and --development-probe are mutually exclusive")
    if args.training_probe is not None and args.output is None:
        parser.error("--training-probe requires --output to retain paid work")
    if args.development_probe and not args.native:
        parser.error("--development-probe requires --native and --output")
    if args.native and args.output is None:
        parser.error("--native requires --output to retain partial simulator work")
    if args.output is not None and (
        not args.output.parent.is_dir() or args.output.exists() or args.output.is_symlink()
    ):
        raise ValueError("demo output must be fresh in an existing directory")
    if retained_options and args.output is not None and not args.native:
        # Both retained CPU modes preserve paid work in this one fallback.
        failure = args.output.with_suffix(".failure.json")
        if failure.exists() or failure.is_symlink():
            raise ValueError("retained demo failure output must be fresh")
    if args.training_probe is not None:
        training_probe_outputs(args.output)
        from scripts.push_cube_reserved_probe_inputs import load_reserved_training_input

        training, receipt = load_reserved_training_input(args.training_probe)
    if args.adaptation_probe is not None:
        if args.native:
            from scripts.taskspace_memory_native_adaptation import native_adaptation_outputs

            native_adaptation_outputs(args.output)
        from scripts.push_cube_reserved_probe_inputs import load_reserved_probe_inputs
        from src.adapters.continuous_probe import ContinuousProbePlan

        loaded, adaptation_receipts = load_reserved_probe_inputs(*args.adaptation_probe)
        adaptation = ContinuousProbePlan(loaded.training, loaded.evaluation, updates=1)
    payload = load_bundle(args.bundle) if args.bundle is not None else _load_default_bundle()
    if args.adaptation_probe is not None:
        if args.native:
            from scripts.taskspace_memory_native_adaptation import run_native_adaptation

            report = run_native_adaptation(
                payload,
                adaptation,
                args.output,
                return_task=args.return_task,
                input_receipts=adaptation_receipts,
                **retained_options,
            )
        else:
            report = run_demo(
                payload,
                adaptation_probe=adaptation,
                return_task=args.return_task,
                **retained_options,
            )
            report["adaptation_input_receipts"] = adaptation_receipts
    elif args.training_probe is not None:
        if args.native:
            from scripts.taskspace_memory_native import run_native_training_probe

            report = run_native_training_probe(
                payload, training, args.output, training_input_receipt=receipt
            )
        else:
            report = run_demo(payload, training_probe=training)
            report["training_input_receipt"] = receipt
    elif args.native:
        from scripts.taskspace_memory_native import run_native_demo

        if args.development_probe:
            from scripts.taskspace_memory_native import NativeScene

            report = run_native_demo(
                payload,
                args.output,
                development_scenes=(NativeScene("push", 19), NativeScene("push", 20)),
            )
        else:
            report = run_native_demo(payload, args.output, **retained_options)
    else:
        report = run_demo(payload, **retained_options)
    encoded = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    report_limit = 3 * 1024**2 if args.native else 512 * 1024
    if len(encoded.encode("utf8")) > report_limit:
        raise ValueError("demo report exceeds its bounded output size")
    if args.output is not None and not args.native:
        if args.training_probe is None and not retained_options:
            with args.output.open("x", encoding="utf8") as handle:
                handle.write(encoded)
        else:
            try:
                with args.output.open("x", encoding="utf8") as handle:
                    handle.write(encoded)
            except Exception as error:
                if args.adaptation_probe is not None:
                    # Opaque IO must not redefine the paid private result on fallback.
                    report = json.loads(encoded)
                # The returned probe is already paid. Preserve it on late IO
                # failure; stdout remains available if both captures refuse.
                report["complete"] = False
                report["capture_error"] = f"{type(error).__name__}: {error}"
                encoded = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
                if len(encoded.encode("utf8")) > 512 * 1024:
                    raise ValueError("training probe failure report exceeds512KiB") from error
                try:
                    with args.output.with_suffix(".failure.json").open(
                        "x", encoding="utf8"
                    ) as handle:
                        handle.write(encoded)
                except Exception as failure_error:
                    report["failure_capture_error"] = (
                        f"{type(failure_error).__name__}: {failure_error}"
                    )
                    encoded = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    print(encoded, end="")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
