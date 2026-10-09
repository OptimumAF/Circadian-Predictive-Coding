"""Execute one reserved, frozen local engineering latency protocol; retain every result.

Run as a module from the repository root. NumPy/BLAS thread variables must be
set before interpreter import. This is a finite synthetic fixture, not a study.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
from time import perf_counter, perf_counter_ns

import numpy as np

from src.adapters.numpy_learners import BackpropLearner, CircadianLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.live_serving_measurement import (
    MeasurementIncomplete,
    MeasurementLimits,
    NativeTimingObserver,
    ObservedLearner,
    measure_shared_serving,
)
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits
from src.core.serving_latency import select_native_overlap, summarize_latencies
from src.shared.process_memory import ProcessRssSampler


def digest(state):
    return hashlib.sha256(pickle.dumps(state)).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_binding(path, protocol):
    binding = json.loads(path.read_bytes())
    root = Path.cwd()
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        )
        .decode()
        .split("\0")
    )
    actual = {n for n in names if n}
    checkout = binding["checkout"]
    if actual != set(checkout["files"]):
        raise ValueError("measurement source inventory changed")
    for name, identity in checkout["files"].items():
        raw = (root / name).read_bytes()
        if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
            raise ValueError("measurement source changed: " + name)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    packages = dict(
        sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    )
    if head != checkout["head"] or packages != checkout["packages"]:
        raise ValueError("measurement reviewed HEAD or installed packages changed")
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != binding["protocol_sha256"]:
        raise ValueError("measurement frozen protocol changed")
    for name, identity in binding["owned_sources"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("measurement owned instrumentation source changed")


def _source(method, p):
    if method == "backprop":
        return BackpropLearner(
            BackpropMLP(p["input_dim"], p["hidden_dim"], p["seed"]),
            learning_rate=p["learning_rate"],
        )
    if method == "circadian":
        model = CircadianPredictiveCodingNetwork(
            p["input_dim"],
            p["hidden_dim"],
            p["seed"],
            min_hidden_dim=p["hidden_dim"],
            max_hidden_dim=p["hidden_dim"],
        )
        return CircadianLearner(
            model,
            learning_rate=p["learning_rate"],
            inference_steps=p["inference_steps"],
            inference_learning_rate=p["inference_learning_rate"],
        )
    raise ValueError("protocol permits only the two declared native methods")


def measure_method(method, p):
    source = _source(method, p)
    source_before = digest(source.snapshot_state())
    observer = NativeTimingObserver()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=p["native_wakes_per_method"],
            max_wall_seconds=p["measurement_wall_seconds"],
            max_process_rss_bytes=p["max_process_rss_bytes"],
        ),
        perf_counter,
    )
    with ProcessRssSampler(interval_seconds=p["memory_sample_seconds"]) as sampler:
        budget.attach_memory(sampler)
        runtime = ActorShadowRuntime(
            ObservedLearner(source, observer),
            actor_version="actor-0",
            candidate_version="candidate-0",
            clock=LogicalClock(p["logical_tick"]),
            budget=budget,
            max_experiences=p["training_blocks"],
        )
        gate = ServingPriorityGate(
            SharingLimits(
                p["max_serving_requests"], p["native_wakes_per_method"], p["max_updates_per_poll"]
            ),
            resource_available=lambda: True,
        )
        shared = ResourceSharedRuntime(runtime, gate)
        rng = np.random.default_rng(p["data_seed"])
        x = rng.normal(size=(p["training_batch"], p["input_dim"])).astype(np.float64)
        targets = (x[:, :1] > 0).astype(np.float64)
        serving = rng.normal(size=(p["serving_batch"], p["input_dim"])).astype(np.float64)
        for block in range(p["training_blocks"]):
            sample = "sample-" + str(block)
            runtime.record_experience(
                Experience(
                    sample,
                    "episode",
                    1,
                    "actor-0",
                    x,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            runtime.record_label(
                LabelArrival(
                    "label-" + str(block), sample, "episode", p["logical_tick"], "actor-0", targets
                )
            )
        actor_before = digest(runtime.actor.snapshot_state())
        warmups = []
        for index in range(p["warmup_requests"]):
            start = perf_counter_ns()
            prediction = shared.predict(serving)
            end = perf_counter_ns()
            warmups.append(
                {
                    "index": index,
                    "start_ns": start,
                    "end_ns": end,
                    "actor_version": prediction.actor_version,
                }
            )
        limits = MeasurementLimits(
            p["idle_requests"],
            p["training_blocks"],
            p["requests_per_block"],
            p["worker_join_seconds"],
            p["measurement_wall_seconds"],
        )
        try:
            measured = measure_shared_serving(shared, serving, observer, limits)
        except MeasurementIncomplete as error:
            return {
                "method": method,
                "PASS": False,
                "status": "incomplete",
                "error": str(error),
                "warmups": warmups,
                "requests": [asdict(r) for r in error.requests],
                "native_calls": [asdict(r) for r in error.native_calls],
                "polls": [asdict(r) for r in error.polls],
                "live_owned_worker": error.live_worker,
                "active_native": error.active_native,
                "resource": asdict(sampler.snapshot()),
            }
        budget.complete_memory()
        idle = tuple(r for r in measured.requests if r.phase == "idle")
        shared_rows = tuple(r for r in measured.requests if r.phase == "shared")
        overlap = select_native_overlap(shared_rows, measured.native_calls)
        contained = select_native_overlap(shared_rows, measured.native_calls, fully_contained=True)
        populations = {
            "idle_all": idle,
            "shared_phase_all": shared_rows,
            "overlapping_native": overlap,
            "fully_contained_native": contained,
        }
        summaries = {name: asdict(summarize_latencies(rows)) for name, rows in populations.items()}
        actor_after = digest(runtime.actor.snapshot_state())
        source_after = digest(source.snapshot_state())
        checks = {
            "counts": len(idle) == p["idle_requests"]
            and len(shared_rows) == p["training_blocks"] * p["requests_per_block"],
            "actual_native_updates": len(measured.native_calls)
            == budget.updates_completed
            == gate.snapshot().admitted_updates
            == p["native_wakes_per_method"],
            "native_completed": all(c.completed for c in measured.native_calls),
            "actor_unchanged": actor_before == actor_after,
            "source_unchanged": source_before == source_after,
            "stable_versions": all(
                r.actor_version == "actor-0" and r.completed for r in measured.requests
            ),
            "coverage": len(contained) >= 20,
            "rss": sampler.snapshot().peak_bytes <= p["max_process_rss_bytes"],
            "quiescent": not gate.snapshot().training_active
            and gate.snapshot().active_serving_requests == 0,
        }
        return {
            "method": method,
            "PASS": all(checks.values()),
            "status": "complete",
            "checks": checks,
            "summaries": summaries,
            "population_indices": {
                n: [[r.block, r.index] for r in rows] for n, rows in populations.items()
            },
            "warmups": warmups,
            "requests": [asdict(r) for r in measured.requests],
            "native_calls": [asdict(c) for c in measured.native_calls],
            "polls": [asdict(r) for r in measured.polls],
            "sharing": asdict(gate.snapshot()),
            "budget": {
                "updates_completed": budget.updates_completed,
                "started_at": budget.started_at,
                "last_clock": budget.last_clock,
            },
            "resource": asdict(sampler.snapshot()),
            "actor_before": actor_before,
            "actor_after": actor_after,
            "source_before": source_before,
            "source_after": source_after,
            "candidate_after": digest(runtime.candidate_snapshot().state),
            "data_digest": digest((x, targets, serving)),
            "scientific_claim": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_bytes())
    if (
        p["methods"] != ["backprop", "circadian"]
        or p["method_order"] != p["methods"]
        or p["reruns_allowed"] != 0
    ):
        raise ValueError("require one frozen two-method protocol in declared order without reruns")
    if any(
        os.environ.get(n) != "1"
        for n in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("set all four declared BLAS thread variables to1 before importing NumPy")
    if not (args.output_dir.parent / "source-binding.json").is_file():
        raise ValueError("source/static correctness binding required before native measurement")
    verify_binding(args.output_dir.parent / "source-binding.json", args.protocol)
    args.output_dir.mkdir(exist_ok=True)
    with (args.output_dir / "experiment.reserved").open("xb") as stream:
        stream.write(args.protocol.read_bytes())
    results = []
    try:
        for method in p["method_order"]:
            result = measure_method(method, p)
            save(args.output_dir / (method + ".json"), result)
            results.append(result)
            print(
                method
                + ": "
                + json.dumps(
                    {
                        "PASS": result["PASS"],
                        "status": result["status"],
                        "summaries": result.get("summaries"),
                    }
                )
            )
            if result.get("live_owned_worker", False):
                break  # Preserve the live owned handle outcome; start no next method.
        final = {
            "PASS": len(results) == len(p["method_order"]) and all(r["PASS"] for r in results),
            "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "blas_environment": {
                n: os.environ.get(n)
                for n in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "methods": [r["method"] for r in results],
            "planned_methods": p["method_order"],
            "native_wakes": sum(len(r["native_calls"]) for r in results),
            "native_wake_attempts": sum(
                len(r["native_calls"]) + int(r.get("active_native") is not None) for r in results
            ),
            "actor_predictions": sum(len(r["requests"]) + len(r["warmups"]) for r in results),
            "native_sleeps": 0,
            "reruns": 0,
            "scientific_claim": False,
        }
        save(args.output_dir / "result.json", final)
        return 0 if final["PASS"] else 1
    except BaseException as error:
        save(
            args.output_dir / "failure.json",
            {
                "PASS": False,
                "error": repr(error),
                "completed_methods": [r["method"] for r in results],
                "reruns": 0,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
