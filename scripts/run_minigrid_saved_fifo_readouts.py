"""Read only the missing fixed TRAIN cells from the saved FIFO36 report."""

import argparse
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 16 * 1024**2


def describe_protocol() -> dict[str, Any]:
    # Pure preview deliberately avoids importing the evaluator and its native ports.
    return dict(
        protocol_id="minigrid_saved_fifo36_pending_train_readouts_v1",
        new_training_updates=0,
        new_training_rows=0,
        replay_examples=0,
        teacher_episodes=0,
        planned_work=dict(
            raw_score_forwards=12,
            raw_score_rows=108,
            live_episodes=16,
            maximum_live_predictions=256,
            maximum_environment_steps=256,
            maximum_total_prediction_calls=268,
            maximum_total_prediction_rows=364,
        ),
        limits=dict(
            body_seconds=120.0,
            process_rss_bytes=768 * 1024**2,
            input_bytes=MAX_INPUT_BYTES,
            output_bytes=MAX_OUTPUT_BYTES,
            cpu_threads=1,
            gpu=0,
        ),
        limitations="Known TRAIN saved FIFO36 control only; old failure/charges remain. No training, final data or CPC advantage claim.",
    )


def _finite_number(text: str) -> float:
    value = float(text)
    if not isfinite(value):
        raise ValueError("source artifact contains a nonfinite JSON number")
    return value


def _reject_constant(text: str) -> Any:
    raise ValueError(f"source artifact contains nonfinite JSON {text}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("source artifact contains duplicate JSON fields")
        result[key] = value
    return result


def load_inputs(path: Path) -> Any:
    from src.adapters.minigrid_saved_fifo_inputs import decode_saved_fifo_inputs

    with path.open("rb") as handle:
        payload = handle.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("source artifact exceeds 16 MiB")
    value = json.loads(
        payload.decode("utf-8"),
        parse_constant=_reject_constant,
        parse_float=_finite_number,
        object_pairs_hook=_unique_object,
    )
    if type(value) is not dict:
        raise ValueError("source artifact must be a JSON object")
    return decode_saved_fifo_inputs(value)


def run_pilot(inputs: Any, *, max_seconds: float = 120.0) -> dict[str, Any]:
    from src.adapters.minigrid_saved_fifo_readouts import run_pilot as run

    return run(inputs, max_seconds=max_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    args = parser.parse_args(argv)
    if not isfinite(args.max_seconds) or not 0 < args.max_seconds <= 120:
        parser.error("--max-seconds must be finite in (0, 120]")
    if args.describe:
        if args.source is not None or args.output is not None:
            parser.error("--describe cannot read or publish artifacts")
        print(json.dumps(describe_protocol(), indent=2, allow_nan=False))
        return 0
    if args.source is None or args.output is None:
        parser.error("running requires --source and --output")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    inputs = load_inputs(args.source)
    report = run_pilot(inputs, max_seconds=args.max_seconds)
    report["source_artifact"] = str(args.source)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("saved FIFO output exceeds 16 MiB")
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(
        json.dumps(
            dict(status=report["status"], output=str(args.output), output_bytes=len(payload))
        )
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
