"""One frozen real Choice smoke using upstream loading and scoring unchanged.

The pure run_case accepts a recording fake. Optional inference imports occur only
in _load; this script neither trains nor selects examples or model parameters.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
from math import isclose
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any, Callable

from src.adapters.open_jev_choice import (
    MAX_REPLY_BYTES,
    build_open_jev_choice,
    decode_open_jev_choice,
)
from src.core.typed_decision import DecisionCandidate, DecisionRequest

BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
TEMPERATURE = 1.518796342858676


def _validate_metadata(response: dict[str, Any], *, candidate_count: int = 2) -> None:
    expected = dict(
        base_revision=BASE_REVISION,
        method="lora_decision_head",
        temperature=TEMPERATURE,
        max_length=256,
        backbone_parameter_dtypes=["torch.float32"],
        candidate_sequences=candidate_count,
        prefix_cache={"enabled": False, "mode": "independent_candidates"},
    )
    if response.get("model") != "Qwen/Qwen3.5-2B":
        raise ValueError("unexpected native model identity")
    metadata = response.get("metadata", {})
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("unexpected native metadata")
    usage = response.get("usage", {})
    tokens = usage.get("input_tokens")
    if type(tokens) is not int or not 0 <= tokens <= 256 * candidate_count:
        raise ValueError("invalid native input token count")
    if type(usage.get("output_tokens")) is not int or usage["output_tokens"] != 0:
        raise ValueError("unexpected generated output tokens")


def run_case(predict: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Capture returned evidence before validation; never retry a paid failure."""
    report: dict[str, Any] = dict(
        definition="open-jev-native-choice-smoke-v1",
        complete=False,
        requests=[],
        work=dict(
            prediction_calls_attempted=0,
            prediction_calls_returned=0,
            prediction_calls_validated=0,
            unreturned_call_cost_unknown=False,
            returned_unvalidated_cost=False,
            counter_boundary="run_case_callable_return",
        ),
    )
    candidates = (
        DecisionCandidate("password", "Password resets and login problems"),
        DecisionCandidate("billing", "Charges, invoices and refunds"),
    )
    work = report["work"]
    try:
        for order in (candidates, tuple(reversed(candidates))):
            request = DecisionRequest("smoke-1", "I forgot my password and cannot sign in.", order)
            wire = build_open_jev_choice(request, "Choose the customer's support intent.")
            row: dict[str, Any] = {"request": deepcopy(wire)}
            report["requests"].append(row)
            work["prediction_calls_attempted"] += 1
            response = predict(wire)
            work["prediction_calls_returned"] += 1
            try:
                raw = json.dumps(response, allow_nan=True, separators=(",", ":"))
                if len(raw.encode()) > MAX_REPLY_BYTES:
                    raise ValueError("native response exceeds capture bound")
                row["response_json"] = raw
                body = json.dumps(response, allow_nan=False, separators=(",", ":")).encode()
                row["response"] = json.loads(body)
            except (TypeError, ValueError, OverflowError, RecursionError) as error:
                row["response_unavailable"] = type(error).__name__
                raise ValueError("native return is not bounded strict JSON") from error
            if wire != row["request"]:
                raise ValueError("predictor changed frozen request")
            decoded = decode_open_jev_choice(request, body)
            _validate_metadata(response)
            row["decoded"] = {
                key: value for key, value in asdict(decoded).items() if key != "raw_response"
            }
            work["prediction_calls_validated"] += 1
        first, second = [row["decoded"] for row in report["requests"]]
        a, b = dict(first["probabilities"]), dict(second["probabilities"])
        difference = max(abs(a[key] - b[key]) for key in a)
        probabilities_agree = all(isclose(a[key], b[key], abs_tol=1e-6, rel_tol=0) for key in a)
        choice_agrees = first["choice"] == second["choice"]
        report["order_control"] = dict(
            probabilities_agree=probabilities_agree,
            choice_agrees=choice_agrees,
            agreement=probabilities_agree and choice_agrees,
            max_absolute_probability_difference=difference,
        )
        # Why this: a wrong or order-sensitive answer is a result, not a reason
        # to retry the frozen example or silently change the success criterion.
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    work["unreturned_call_cost_unknown"] = (
        work["prediction_calls_attempted"] > work["prediction_calls_returned"]
    )
    work["returned_unvalidated_cost"] = (
        work["prediction_calls_returned"] > work["prediction_calls_validated"]
    )
    return report


def _load(
    source: Path,
    checkpoint: Path,
    progress: Callable[[dict[str, Any]], None],
    *,
    prediction_limit: int = 2,
    forward_limit: int = 4,
    token_limit: int = 1024,
    model_observer: Callable[[Any], None] | None = None,
    work_reporter: Callable[[Callable[[], None]], None] | None = None,
):
    """Load the original pinned package in an offline CPU-only process."""
    # Limits belong to the prospective caller protocol. Closed smoke defaults
    # remain unchanged; a distinct cohort must declare its own finite allowance.
    for value, maximum in ((prediction_limit, 128), (forward_limit, 1024), (token_limit, 262144)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("native work limits must be bounded positive integers")
    import resource

    for key in tuple(os.environ):
        if key.startswith("JEV_"):
            raise ValueError(f"refuse inherited inference override {key}")
    os.environ.update(
        JEV_TORCH_DTYPE="float32",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    sys.path.insert(0, str(source.resolve()))
    import torch

    torch.set_num_threads(8)
    torch.manual_seed(17)
    started = monotonic()
    counts: dict[str, Any] = dict(
        scorer_calls=0,
        decision_forward_calls=0,
        initializer_cpu_elements=0,
        initializer_meta_elements=0,
        native_prediction_calls_attempted=0,
        native_prediction_calls_returned=0,
        candidate_sequence_lengths=[],
    )

    def state():
        return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()

    def observe(event, **extra):
        peak = getattr(resource, "getrusage")(getattr(resource, "RUSAGE_SELF")).ru_maxrss * 1024
        progress(
            dict(
                event=event,
                seconds=monotonic() - started,
                peak_rss_bytes=peak,
                observed_work=deepcopy(counts),
                **extra,
            )
        )
        if peak > 20 * 1024**3 or monotonic() - started > 600:
            raise RuntimeError("model RSS/time allowance exceeded")

    def profile(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        if event == "call":
            if module == "jev.serving" and name == "score":
                counts["scorer_calls"] += 1
                if counts["scorer_calls"] > forward_limit:
                    raise RuntimeError("scorer call allowance exceeded")
            elif module == "jev.model" and name == "forward":
                counts["decision_forward_calls"] += 1
                if counts["decision_forward_calls"] > forward_limit:
                    raise RuntimeError("model forward allowance exceeded")
            elif module == "torch.nn.init" and name in ("_no_grad_uniform_", "_no_grad_normal_"):
                tensor = frame.f_locals["tensor"]
                key = "initializer_meta_elements" if tensor.is_meta else "initializer_cpu_elements"
                counts[key] += tensor.numel()
                if counts["initializer_cpu_elements"] > 5_000_000_000:
                    raise RuntimeError("initializer allowance exceeded")
        elif event == "return" and module == "jev.model" and name == "forward":
            lengths = frame.f_locals.get("lengths")
            if lengths is not None:
                counts["candidate_sequence_lengths"].extend(lengths.tolist())
                if (
                    any(size > 256 for size in counts["candidate_sequence_lengths"])
                    or sum(counts["candidate_sequence_lengths"]) > token_limit
                ):
                    raise RuntimeError("candidate token allowance exceeded")

    before = state()
    sys.setprofile(profile)
    observe("load_attempt", rng_state_sha256=before)
    try:
        serving = importlib.import_module("jev.serving")
        predictor = serving.load_predictor(
            checkpoint=str(checkpoint),
            device="cpu",
            max_length=256,
            batch_size=1,
            prefix_cache=False,
            backend="torch",
        )
        model = predictor.scorer.model
        if model.training or any(p.is_meta or p.device.type != "cpu" for p in model.parameters()):
            raise RuntimeError("model is not materialized CPU evaluation state")
        origins = {}
        for name in (
            "jev",
            "jev.api",
            "jev.metrics",
            "jev.model",
            "jev.serving",
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "safetensors",
            "huggingface_hub",
        ):
            module = importlib.import_module(name)
            path = Path(str(module.__file__)).resolve()
            if name.startswith("jev") and not path.is_relative_to(source.resolve()):
                raise RuntimeError("upstream source origin mismatch")
            origins[name] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
                "huggingface-hub",
            )
        }
        observe(
            "loaded",
            rng_state_sha256=state(),
            origins=origins,
            versions=versions,
            total_parameters=sum(p.numel() for p in model.parameters()),
            uninstrumented_initialization_random_draws="unknown",
        )
        if model_observer is not None:
            model_observer(model)
    except BaseException:
        observe("load_failed", rng_state_sha256=state())
        sys.setprofile(None)
        raise

    def predict(wire):
        before = state()
        observe("prediction_attempt", rng_state_sha256=before)
        try:
            counts["native_prediction_calls_attempted"] += 1
            if counts["native_prediction_calls_attempted"] > prediction_limit:
                raise RuntimeError("prediction allowance exceeded")
            with torch.inference_mode():
                reply = predictor.predict(wire)
            counts["native_prediction_calls_returned"] += 1
            # Retain the native return even if a subsequent resource/RNG check fails.
            observe("native_return", raw_response=reply, rng_state_sha256=state())
            if state() != before:
                raise RuntimeError("prediction changed Torch CPU RNG state")
            return reply
        finally:
            observe("prediction_finished", rng_state_sha256=state())

    # Why: TRAIN-only callers need final profiler counts without buying an
    # inference merely to trigger the usual prediction_finished observation.
    if work_reporter is not None:
        work_reporter(lambda: observe("work_observed", rng_state_sha256=state()))
    return predict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; preserve previous evidence")
    # Reserve output before even importing inference libraries; preserve failures.
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    report: dict[str, Any] = dict(complete=False, model_load_attempts=0, events=[])

    def progress(event):
        report["events"].append(event)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )

    try:
        if (
            not (args.source / "jev" / "serving.py").is_file()
            or not (args.checkpoint / "head.pt").is_file()
        ):
            raise ValueError("missing native source or trained checkpoint")
        report["model_load_attempts"] = 1
        predict = _load(args.source, args.checkpoint, progress)
        report.update(run_case(predict))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        sys.setprofile(None)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
