"""Read saved E8/A8 memory endpoints through the existing target-free game runner.

Inputs are the accepted original memory-live and return-comparison JSON reports.
Outputs are frozen known-game decisions/costs; no training authority is restored.
Evidence admission checks consistency, not independent origin authentication.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import sys
from time import monotonic
from typing import Any

from scripts.run_minigrid_memory_probe import (
    MAX_OUTPUT_BYTES,
    _object,
    _read_source,
    _require,
    _return_episodes,
    _snapshot,
    _snapshots_from_report,
)


def describe_protocol() -> dict[str, Any]:
    """Describe the fixed readout without reading sources or constructing resources."""
    return dict(
        protocol_id="minigrid_saved_memory_return_live_v1",
        source_protocol_id="minigrid_saved_memory_E8_A8_return_v1",
        classification="exploratory_known_game_frozen_endpoint_readout",
        cohort_order=[
            [0, "observation_memory"],
            [0, "observation_memory_fifo"],
            [128, "observation_memory"],
            [128, "observation_memory_fifo"],
        ],
        endpoint_order="each cohort initial then saved end; no checkpoint8 reconstruction",
        slots=[
            ["A", "blue", "train"],
            ["A", "red", "train"],
            ["E-blue", "blue", "probe_test"],
            ["E-red", "red", "probe_test"],
        ],
        seed=17,
        max_steps=16,
        policy="raw seven-action first argmax; public initial/current148+148 history",
        training_updates=0,
        score_forwards=0,
        maximum_readouts=8,
        maximum_environment_resets=32,
        maximum_predictions=512,
        maximum_environment_steps=512,
        metrics="per-slot success/return/steps; per-cohort initial/end A/E success counts and end-initial differences",
        limits=dict(
            body_seconds=90,
            parent_work_seconds=110,
            shared_cleanup_seconds=120,
            process_rss_bytes=768 * 1024**2,
            stage_output_bytes=MAX_OUTPUT_BYTES,
            cpu_threads=1,
            gpu=0,
        ),
        limitations="known A and inspected E only; failed source preserved; no intermediate live deficit/recovery, generalization, equal compute, neural or CPC advantage",
    )


def _saved_endpoint(value: Any, method: str, point: int) -> Any:
    if method == "observation_memory":
        return _snapshot(value, point)
    from src.core.action_memory import ActionMemoryEntry
    from src.core.fifo_action_memory import (
        FifoActionMemoryPolicy,
        FifoActionMemorySnapshot,
        copy_fifo_action_memory_snapshot,
    )

    value = _object(
        value,
        {
            "policy",
            "entries",
            "completed_updates",
            "training_examples",
            "evictions",
            "retired_examples",
        },
        "FIFO endpoint",
    )
    policy = _object(
        value["policy"],
        {"method", "feature_dim", "num_classes", "max_entries", "max_payload_bytes"},
        "FIFO policy",
    )
    records = value["entries"]
    _require(type(records) is list and len(records) <= 64, "invalid FIFO entries")
    entries = []
    for record in records:
        record = _object(record, {"features", "counts"}, "FIFO entry")
        _require(
            type(record["features"]) is list and type(record["counts"]) is list,
            "FIFO entries require recorded lists",
        )
        entries.append(ActionMemoryEntry(tuple(record["features"]), tuple(record["counts"])))
    state = copy_fifo_action_memory_snapshot(
        FifoActionMemorySnapshot(
            FifoActionMemoryPolicy(**policy),
            tuple(entries),
            value["completed_updates"],
            value["training_examples"],
            value["evictions"],
            value["retired_examples"],
        )
    )
    _require(
        state.policy
        == FifoActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136),
        "FIFO endpoint geometry/cap differs",
    )
    _require(
        state.completed_updates == point and state.training_examples == 9 * point,
        "FIFO endpoint clocks differ",
    )
    return state


def load_inputs(original_path: Path, return_path: Path) -> list[dict[str, Any]]:
    """Read each report once and bind typed endpoints to its original saved source."""
    from src.adapters.minigrid_action_memory_probes import describe_return_protocol
    from src.app.coupled_attribution import _same_value
    from src.core.fifo_action_memory import convert_action_memory_to_fifo

    original, returned = _read_source(original_path), _read_source(return_path)
    initial_states = _snapshots_from_report(original)
    original_a = _return_episodes(original)
    _require(
        _same_value(returned.get("protocol"), describe_return_protocol()),
        "return source protocol differs",
    )
    _require(returned.get("status") == "incomplete", "preserve original incomplete return case")
    _require(
        type(returned.get("source_artifact")) is str
        and Path(returned["source_artifact"]).resolve() == original_path.resolve(),
        "return source must bind the provided original artifact path",
    )
    _require(
        _same_value(
            returned.get("return_sources"),
            json.loads(json.dumps([asdict(episode) for episode in original_a])),
        ),
        "original TRAIN A provenance differs",
    )
    cohorts: Any = returned.get("cohorts")
    _require(type(cohorts) is list and len(cohorts) == 4, "four return cohorts required")
    inputs = []
    for row, (point, method) in zip(cohorts, describe_protocol()["cohort_order"], strict=True):
        _require(
            type(row) is dict
            and type(row.get("source_updates")) is int
            and row["source_updates"] == point
            and row.get("method") == method,
            "fixed source cohort order differs",
        )
        failed = point == 128 and method == "observation_memory"
        status, relative = ("failed", 1) if failed else ("complete", 16)
        raw = row.get("report")
        _require(
            row.get("status") == status
            and type(raw) is dict
            and raw.get("status") == status
            and row.get("source_unchanged") is True
            and row.get("budget_binding_preserved") is True
            and row.get("charge_readback_unknown", False) is False,
            "source status/isolation/accounting evidence differs",
        )
        for field in ("charged_updates", "budget_updates_observed"):
            _require(
                type(row.get(field)) is int and row[field] == relative,
                "original storage charges differ",
            )
        for field, expected in (
            ("completed_updates", relative),
            ("attempted_updates", relative + int(failed)),
            ("completed_training_examples", 9 * relative),
            ("attempted_training_examples", 9 * (relative + int(failed))),
        ):
            _require(
                type(raw.get(field)) is int and raw[field] == expected,
                "original paid prefix differs",
            )
        expected_reason = (
            "ValueError: memory batch exceeds entry or logical payload capacity" if failed else None
        )
        _require(
            row.get("reason") == raw.get("reason") == expected_reason,
            "source refusal reason differs",
        )
        for field, expected in (
            ("native_training_examples_before", 9 * point),
            ("fifo_retirement_metadata_bytes", 16 if method.endswith("_fifo") else 0),
        ):
            _require(
                type(row.get(field)) is int and row[field] == expected,
                "source native or retirement accounting differs",
            )
        initial: Any = initial_states[point // 128]
        if method == "observation_memory_fifo":
            initial = convert_action_memory_to_fifo(initial)
        end = _saved_endpoint(row.get("end_snapshot"), method, point + relative)
        for state, label, source_status in ((initial, 0, "complete"), (end, relative, status)):
            inputs.append(
                dict(
                    source_updates=point,
                    relative_updates=label,
                    source_status=source_status,
                    method=method,
                    state=state,
                    original_cohort_status=status,
                    original_cohort_reason=row["reason"],
                )
            )
    return inputs


def _contrasts(readouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for before, after in zip(readouts[::2], readouts[1::2], strict=True):
        if before["status"] != "complete" or after["status"] != "complete":
            continue
        row = dict(source_updates=before["source_updates"], method=before["method"])
        for name, contexts in (("A", ("A",)), ("E", ("E-blue", "E-red"))):
            counts = [
                sum(slot["success"] for slot in report["episodes"] if slot["context"] in contexts)
                for report in (before, after)
            ]
            row[name] = dict(
                initial_successes=counts[0],
                end_successes=counts[1],
                end_minus_initial=counts[1] - counts[0],
                episodes_per_endpoint=2,
            )
        results.append(row)
    return results


def run_readouts(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [
        (point, method, relative)
        for point, method in describe_protocol()["cohort_order"]
        for relative in (0, 1 if point == 128 and method == "observation_memory" else 16)
    ]
    _require(
        type(inputs) is list
        and len(inputs) == 8
        and [(item["source_updates"], item["method"], item["relative_updates"]) for item in inputs]
        == expected,
        "readout requires exactly the fixed eight owned endpoint inputs",
    )
    from src.adapters.action_memory_learner import ActionMemoryLearner
    from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
    from src.adapters.minigrid_policy_evaluation import (
        MiniGridEvaluationFailed,
        _return_budget_guard,
        evaluate_memory_return_checkpoint,
    )
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.shared.process_memory import ProcessRssSampler

    # Why: eight bounded input records and at most eight returned reports need
    # full checks across callbacks. Fresh serialization preserves types/extra
    # fields without the earlier recursive Python scalar-comparison cost. No
    # untrusted pickle is loaded, and no success-history chain is retained.
    input_bytes = pickle.dumps(inputs)
    bindings = [(item, item["state"]) for item in inputs]
    finished: list[tuple[dict[str, Any], bytes, int, dict[str, Any]]] = []
    slots = [
        dict(
            status="not_run",
            source_updates=item["source_updates"],
            relative_updates=item["relative_updates"],
            method=item["method"],
            source_status=item["source_status"],
            original_cohort_status=item["original_cohort_status"],
            original_cohort_reason=item["original_cohort_reason"],
            reason="not reached",
            episodes=[],
            scores=[],
            work=None,
        )
        for item in inputs
    ]
    report: dict[str, Any] = dict(
        protocol=describe_protocol(),
        status="running",
        reason=None,
        source_experiment_status="incomplete",
        readouts=slots,
        contrasts=[],
        training_updates=0,
        scientific_advantage_verified=False,
    )
    active: dict[str, Any] | None = None
    budget: Any = None

    def fail(error: Exception) -> None:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        if active is not None and active["status"] == "running":
            # A lost delegated receipt leaves paid work unknown, never zero.
            active.update(
                status="failed",
                reason=report["reason"],
                state_unchanged=None,
                work=dict(
                    failure_work_may_be_unobserved=True,
                    live_prediction_calls_attempted=None,
                    environment_resets_attempted=None,
                    environment_steps_attempted=None,
                ),
                delegated_report_unavailable=True,
            )
        elif active is not None and active["status"] == "complete":
            active.update(
                delegated_status="complete",
                status="failed",
                reason=report["reason"],
                state_unchanged=None,
            )
        changed = []
        for prior, frozen, index, accepted in finished:
            if report["readouts"][index] is not prior or pickle.dumps(prior) != frozen:
                changed.append(index)
            # Preserve the private paid capture even if an external callback
            # corrupted/replaced its original returned object. At most8.
            report["readouts"][index] = accepted
        report["changed_prior_readout_indices"] = changed

    started = monotonic()
    try:
        with ProcessRssSampler() as sampler:
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0, max_wall_seconds=90, max_process_rss_bytes=768 * 1024**2
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget_guard = _return_budget_guard(budget)

            def check() -> None:
                budget_guard()
                _require(
                    len(inputs) == 8
                    and all(
                        item is bound and item["state"] is state
                        for item, (bound, state) in zip(inputs, bindings, strict=True)
                    ),
                    "owned saved input binding changed",
                )
                _require(pickle.dumps(inputs) == input_bytes, "owned saved input evidence changed")
                for prior, frozen, index, _ in finished:
                    _require(
                        report["readouts"][index] is prior and pickle.dumps(prior) == frozen,
                        "preceding live readout evidence changed",
                    )
                budget_guard()

            try:
                for index, item in enumerate(inputs):
                    check()
                    active = slots[index]
                    active.update(status="running", reason=None)
                    state = item["state"]
                    kind = (
                        ActionMemoryLearner
                        if item["method"] == "observation_memory"
                        else FifoActionMemoryLearner
                    )
                    source = kind(state.policy)
                    source.restore_state(state)
                    check()
                    try:
                        result = evaluate_memory_return_checkpoint(
                            source,
                            state,
                            source_updates=item["source_updates"],
                            relative_updates=item["relative_updates"],
                            source_status=item["source_status"],
                            budget=budget,
                            check_boundary=check,
                        )
                    except MiniGridEvaluationFailed as error:
                        result = error.report
                        result.update(
                            original_cohort_status=item["original_cohort_status"],
                            original_cohort_reason=item["original_cohort_reason"],
                        )
                        report["readouts"][index] = result
                        active = result
                        raise
                    result.update(
                        original_cohort_status=item["original_cohort_status"],
                        original_cohort_reason=item["original_cohort_reason"],
                    )
                    report["readouts"][index] = result
                    active = result
                    check()
                    finished.append((result, pickle.dumps(result), index, deepcopy(result)))
                    check()
                    active = None
                budget.before_final()
                check()
                sampler.sample()
                report["observed_process_rss"] = asdict(sampler.snapshot())
                check()
                report["status"] = "complete"
            except Exception as error:
                fail(error)
            report["training_updates"] = budget.updates_completed
    except Exception as error:
        fail(error)
    if report["status"] == "complete":
        try:
            check()  # Closing resource callbacks must precede complete publication.
        except Exception as error:
            fail(error)
    # Context exit is also an external boundary. Preserve its actual permanent
    # charge observation even when the closing guard refuses publication.
    report["training_updates"] = budget.updates_completed if budget is not None else None
    # Do not derive new contrasts from corrupted or unobserved evidence.
    report["contrasts"] = _contrasts(report["readouts"]) if report["status"] == "complete" else []
    report["total_seconds"] = monotonic() - started
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--return-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.describe:
        if any(
            value is not None for value in (args.source_artifact, args.return_artifact, args.output)
        ):
            parser.error("--describe reads no artifacts and publishes no output")
        print(json.dumps(describe_protocol(), indent=2, allow_nan=False))
        return 0
    if args.source_artifact is None or args.return_artifact is None:
        parser.error("both --source-artifact and --return-artifact are required")
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("output must be new with an existing parent directory")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = monotonic()
    inputs = load_inputs(args.source_artifact, args.return_artifact)
    source_load_seconds = monotonic() - started
    report = run_readouts(inputs)
    report.update(
        source_artifact=str(args.source_artifact),
        return_artifact=str(args.return_artifact),
        source_load_seconds=source_load_seconds,
    )
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf8")
    _require(len(payload) <= MAX_OUTPUT_BYTES, "readout output exceeds32MiB; not published")
    if args.output is not None:
        with args.output.open("xb") as output:
            output.write(payload)
    sys.stdout.write(payload.decode("utf8"))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
