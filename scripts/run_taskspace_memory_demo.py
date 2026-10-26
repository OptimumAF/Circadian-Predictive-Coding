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

from src.adapters.taskspace_memory_demo import load_bundle, run_demo  # noqa: E402

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
        "--native",
        action="store_true",
        help="run four fresh frozen TRAIN scenes; requires the optional simulator and --output",
    )
    parser.add_argument(
        "--development-probe",
        action="store_true",
        help="score fixed Push TRAIN resets19/20 absent from bundled memory; requires --native/--output",
    )
    args = parser.parse_args(argv)
    if args.development_probe and not args.native:
        parser.error("--development-probe requires --native and --output")
    if args.native and args.output is None:
        parser.error("--native requires --output to retain partial simulator work")
    if args.output is not None and (
        not args.output.parent.is_dir() or args.output.exists() or args.output.is_symlink()
    ):
        raise ValueError("demo output must be fresh in an existing directory")
    payload = load_bundle(args.bundle) if args.bundle is not None else _load_default_bundle()
    if args.native:
        from scripts.taskspace_memory_native import run_native_demo

        if args.development_probe:
            from scripts.taskspace_memory_native import NativeScene

            report = run_native_demo(
                payload,
                args.output,
                development_scenes=(NativeScene("push", 19), NativeScene("push", 20)),
            )
        else:
            report = run_native_demo(payload, args.output)
    else:
        report = run_demo(payload)
    encoded = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    if len(encoded.encode("utf8")) > (3 * 1024**2 if args.native else 512 * 1024):
        raise ValueError("demo report exceeds512KiB")
    if args.output is not None and not args.native:
        with args.output.open("x", encoding="utf8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
