"""First-party frozen-recurrent TRAIN readout and target-free A games."""

import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import isfinite
from pathlib import Path
import pickle
import random
from time import monotonic

import numpy as np

from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters.minigrid_public_cue_recurrent_pilot import _typed_sources
from src.adapters.minigrid_learning_pilot import _signature, _check_signature, _batch
from src.adapters.minigrid_recurrent_readout import RecurrentReadoutLearner, copy_snapshot
from src.adapters.minigrid_policy_evaluation import (
    evaluate_recurrent_readout_checkpoint,
    evaluate_recurring_recurrent_readout_checkpoint,
    _return_budget_guard,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_return_phase import paid_report_guard
from src.core.learner_ports import TrainingUpdateFailed
from src.infra.minigrid_affine_example import example_affine_training
from src.infra.minigrid_memory import _require_minigrid
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

SHA = "5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508"
RSS = 512 * 1024**2


def describe(*, fresh_probe=False):
    if type(fresh_probe) is not bool:
        raise ValueError("fresh_probe must be an exact bool")
    protocol = dict(
        definition="recurrent_readout_train_A_native_v1",
        source="bundled original A-blue/red TRAIN9rows each; raw148/H8/K7/seed17",
        learning="two fixed float64 one-hot LS fits18rows; frozen nativeRNN0SGD",
        games="two target-free original TRAIN-A blue/red; all7argmax/seed17/max16",
        inference="bounded ordered-prefix recomputation, n(n+1)/2 recurrent forward rows per game",
        limits=dict(body_seconds=120, training_seconds=60, readout_seconds=90, rss_bytes=RSS),
        final_available=False,
        probe_available=False,
    )
    if fresh_probe:
        from src.adapters.minigrid_readout_probe import describe_readout_probe

        protocol.update(
            definition="recurrent_readout_A_then_disposable_F_A_v1",
            probe_available=True,
            disposable_probe=describe_readout_probe(),
        )
    return protocol


def _episodes():
    with example_affine_training() as path:
        return read_training_sources(path, expected_sha256=SHA, raw_frames=True)


def _budget_report(budget):
    return (
        None
        if budget is None
        else dict(
            limits=asdict(budget.budget),
            birth=budget.started_at,
            last_clock=budget.last_clock,
            updates=budget.updates_completed,
            replay=budget.replay_examples_completed,
        )
    )


def run_example(*, fresh_probe=False):
    if type(fresh_probe) is not bool:
        raise ValueError("fresh_probe must be an exact bool")
    started = monotonic()
    report = dict(
        protocol=describe(fresh_probe=True) if fresh_probe else describe(),
        status="incomplete",
        reason=None,
        training=None,
        games=None,
        failure_work=None,
        preservation=None,
        training_budget=None,
        readout_budget=None,
        readback_error=None,
    )
    if fresh_probe:
        report["probe"] = None
    train_budget = readout_budget = stream = source = None
    protocol, protocol_pin = report["protocol"], deepcopy(report["protocol"])
    with ProcessRssSampler() as memory:
        try:
            _require_minigrid()
            episodes = _episodes()
            sources = _typed_sources(episodes)[:2]
            source_pin = _signature(sources)
            torch = require_torch()
            rng = (
                torch.get_rng_state().clone(),
                pickle.dumps(np.random.get_state()),
                random.getstate(),
            )
            source = RecurrentReadoutLearner()
            initial = source.snapshot_state()
            train_limits = ToyExecutionBudget(
                max_training_updates=2, max_wall_seconds=60, max_process_rss_bytes=RSS
            )
            train_budget = ToyBudgetSession(train_limits, monotonic)
            train_budget.attach_memory(memory)
            stream = AttributionStream(
                source,
                sources,
                (),
                checkpoint_updates=(0, 2),
                label_delay=1,
                budget=train_budget,
                training_only=True,
            )
            actor = deepcopy(stream._runtime.actor.snapshot_state())

            def guard():
                if (
                    report["protocol"] is not protocol
                    or not _same_state(protocol, protocol_pin)
                    or not _check_signature(sources, source_pin)
                ):
                    raise ValueError("original protocol or released TRAIN source changed")
                if (
                    monotonic() - started >= 120
                    or memory.peak_bytes is None
                    or memory.peak_bytes > RSS
                ):
                    raise ValueError("recurrent readout whole-case time/RSS exhausted")
                stream._check_original()
                if (
                    not _same_state(initial, source.snapshot_state())
                    or not _same_state(actor, stream._runtime.actor.snapshot_state())
                    or not bool(torch.equal(rng[0], torch.get_rng_state()))
                    or rng[1] != pickle.dumps(np.random.get_state())
                    or rng[2] != random.getstate()
                ):
                    raise ValueError("original source/actor/RNG changed")

            guard()
            stream.run_until(2)
            state = copy_snapshot(stream.candidate_snapshot())
            report["training"] = stream.finish()
            training_owner = report["training"]
            training_value = deepcopy(training_owner)
            guard()
            reader = source.fork()
            reader.restore_state(state)
            readout_budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=0,
                    max_wall_seconds=90,
                    max_process_rss_bytes=RSS,
                ),
                monotonic,
            )
            readout_budget.attach_memory(memory)

            def read_guard():
                guard()
                if report["training"] is not training_owner or not _same_state(
                    training_owner, training_value
                ):
                    raise ValueError("preceding paid training report changed or was replaced")

            report["games"] = evaluate_recurrent_readout_checkpoint(
                reader, state, sources=sources, budget=readout_budget, check_boundary=read_guard
            )
            if fresh_probe:
                from src.adapters.minigrid_readout_probe import run_readout_probe

                games_owner = report["games"]
                games_check, original_games = paid_report_guard(games_owner)

                def probe_guard():
                    if report["games"] is not games_owner:
                        raise ValueError("preceding A games report was replaced")
                    games_check()
                    read_guard()
                    if report["games"] is not games_owner:
                        raise ValueError("preceding A games report was replaced")
                    games_check()
                    source._check()
                    if not _same_value(source._state, initial):
                        raise ValueError("original live source changed during a snapshot callback")

                try:
                    probe_guard()
                    slots = games_owner.get("episodes", [])
                    if (
                        games_owner.get("status") != "complete"
                        or games_owner.get("state_unchanged") is not True
                        or len(slots) != 2
                        or any(
                            slot.get("status") != "complete"
                            or slot.get("success") is not True
                            or slot.get("environment", {}).get("closed") is not True
                            for slot in slots
                        )
                    ):
                        raise ValueError(
                            "fresh probe requires both successful closed original A games"
                        )
                    report["probe"] = run_readout_probe(
                        reader, state, sources, check_boundary=probe_guard, memory=memory
                    )
                    probe_guard()
                    if report["probe"]["status"] != "complete":
                        raise ValueError(
                            "disposable readout probe incomplete; original evidence retained"
                        )
                except Exception:
                    report["games"] = original_games
                    raise
            else:
                read_guard()
            report["preservation"] = dict(
                source=True, actor=True, torch_rng=True, numpy_rng=True, python_rng=True
            )
            report["status"] = "complete"
        except Exception as error:
            report["reason"] = f"{type(error).__name__}: {error}"
            if isinstance(error, TrainingUpdateFailed):
                report["failure_work"] = dict(error.work)
            if hasattr(error, "report"):
                report["games"] = deepcopy(error.report)
            if stream is not None and report["training"] is None:
                try:
                    report["training"] = dict(
                        status="incomplete",
                        applied_updates=[asdict(r) for r in stream.applied_updates],
                    )
                except Exception as readback:
                    report["readback_error"] = f"{type(readback).__name__}: {readback}"
        finally:
            report["training_budget"] = _budget_report(train_budget)
            report["readout_budget"] = _budget_report(readout_budget)
            report["seconds"] = monotonic() - started
            report["peak_bytes"] = memory.peak_bytes
    return report


def describe_recurring():
    result = describe()
    result.update(
        definition="recurrent_readout_train_ABCA_native_v1",
        source="bundled original A/B/C_train blue/red TRAIN9rows each; raw148/H8/K7/seed17",
        learning="eight chronological float64 one-hot LS fits72row exposures/54unique TRAIN rows/18repeated A rows; frozen nativeRNN0SGD; no replay operation",
        context_sequence=["A", "B", "C_train", "A"],
        fit_points=[2, 4, 6, 8],
        readout_contexts=[["A"], ["A", "B"], ["A", "B", "C_train"], ["A", "B", "C_train"]],
        games="18target-free arrived TRAIN games; fixed order/all7argmax/seed17/max16",
        outcomes="current-task acquisition, old-task retention, return success separate; recovery on return requires initial acquisition+last pre-return failure+regain; recovery by return also reported",
    )
    return result


def _recurring_packets(sources):
    # Use the existing packet constructor, with distinct chronological arrivals.
    # Returning A is an ordinary new TRAIN phase, not a hidden replay operation.
    return tuple(
        _batch(replace(sources[index].episode, episode_id=f"recurrent-readout-arrival-{i}"))
        for i, index in enumerate((0, 1, 2, 3, 4, 5, 0, 1))
    )


def _qualified_recurring_games(point, games):
    roster = [
        (c, cue) for c in ("A", "B", "C_train")[: min(point, 6) // 2] for cue in ("blue", "red")
    ]
    episodes = games["episodes"]
    if games["status"] != "complete" or [(e["context"], e["cue"]) for e in episodes] != roster:
        raise ValueError("sequential outcomes require complete original arrived task roster")
    if games.get("state_unchanged") is not True or any(
        e.get("status") != "complete"
        or e.get("state_unchanged") is not True
        or e.get("environment", {}).get("closed") is not True
        or type(e.get("success")) is not bool
        or type(e.get("return")) not in (float, int)
        or not isfinite(e["return"])
        for e in episodes
    ):
        raise ValueError("sequential outcomes require preserved closed observed game outcomes")
    return {(e["context"], e["cue"]): e["success"] for e in episodes}


def _recurring_outcomes(records):
    if [r["updates"] for r in records] != [2, 4, 6, 8]:
        raise ValueError("sequential outcomes require every original fit point")
    by_point = {r["updates"]: _qualified_recurring_games(r["updates"], r["games"]) for r in records}
    acquisition = [
        dict(updates=point, context=context, cue=cue, success=by_point[point][context, cue])
        for point, context in ((2, "A"), (4, "B"), (6, "C_train"))
        for cue in ("blue", "red")
    ]
    retention = [
        dict(updates=point, context=context, cue=cue, success=success)
        for point, current in ((4, "B"), (6, "C_train"), (8, "A"))
        for (context, cue), success in by_point[point].items()
        if context != current
    ]
    returning = []
    for cue in ("blue", "red"):
        acquired = by_point[2]["A", cue]
        lost = any(not by_point[p]["A", cue] for p in (4, 6))
        last_lost = not by_point[6]["A", cue]
        success = by_point[8]["A", cue]
        returning.append(
            dict(
                cue=cue,
                initially_acquired=acquired,
                failed_while_absent=lost,
                failed_before_return=last_lost,
                return_success=success,
                observed_recovery_by_return=acquired and lost and success,
                observed_recovery=acquired and last_lost and success,
            )
        )
    return dict(acquisition=acquisition, retention=retention, returning_A=returning)


def _check_paid_evidence(current, owner, check, message):
    # The shared guard seals the original report, so pin its outer root too.
    if current is not owner:
        raise ValueError(message)
    try:
        check()
    except ValueError as error:
        raise ValueError(message) from error


def run_recurring_example():
    """Plain sequential learning through the existing stream; no replay or probe."""
    started = monotonic()
    report = dict(
        protocol=describe_recurring(),
        status="incomplete",
        reason=None,
        training=None,
        readouts=[],
        outcomes=None,
        failure_work=None,
        preservation=None,
        training_budget=None,
        readout_budget=None,
        readback_error=None,
    )
    train_budget = readout_budget = stream = None
    readout_guard = None
    active_point = None
    records = report["readouts"]
    prior: list[tuple[dict, Callable[[], None]]] = []
    paid_receipts: list[dict] = []
    training_evidence = None
    protocol = report["protocol"]
    protocol_guard, _ = paid_report_guard(protocol)
    with ProcessRssSampler() as memory:
        try:
            _require_minigrid()
            sources = _typed_sources(_episodes())
            packets = _recurring_packets(sources)
            signatures = [(v, _signature(v)) for v in (sources, packets)]
            torch = require_torch()
            rng = (
                torch.get_rng_state().clone(),
                pickle.dumps(np.random.get_state()),
                random.getstate(),
            )
            source = RecurrentReadoutLearner()
            initial = source.snapshot_state()
            train_budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=8, max_wall_seconds=60, max_process_rss_bytes=RSS
                ),
                monotonic,
            )
            train_budget.attach_memory(memory)
            stream = AttributionStream(
                source,
                packets,
                (),
                checkpoint_updates=(0, 8),
                label_delay=1,
                budget=train_budget,
                training_only=True,
            )
            actor = deepcopy(stream._runtime.actor.snapshot_state())

            def check_evidence():
                _check_paid_evidence(
                    report["protocol"],
                    protocol,
                    protocol_guard,
                    "original sequential protocol changed or was replaced",
                )
                if any(not _check_signature(v, pin) for v, pin in signatures):
                    raise ValueError("original sequential TRAIN source or arrivals changed")
                if report["readouts"] is not records or len(records) != len(prior):
                    raise ValueError("preceding sequential evidence changed or was replaced")
                for record, (owner, check) in zip(records, prior, strict=True):
                    _check_paid_evidence(
                        record,
                        owner,
                        check,
                        "preceding sequential evidence changed or was replaced",
                    )
                if training_evidence is not None:
                    _check_paid_evidence(
                        report["training"],
                        *training_evidence,
                        "completed sequential training evidence changed or was replaced",
                    )

            def guard():
                check_evidence()
                if (
                    monotonic() - started >= 120
                    or memory.peak_bytes is None
                    or memory.peak_bytes > RSS
                ):
                    raise ValueError("sequential whole-case time/RSS exhausted")
                stream._check_original()
                actual_receipts = [asdict(r) for r in stream.applied_updates]
                if not _same_state(actual_receipts[: len(paid_receipts)], paid_receipts):
                    raise ValueError("preceding paid TRAIN receipts changed")
                if (
                    not _same_state(initial, source.snapshot_state())
                    or not _same_state(actor, stream._runtime.actor.snapshot_state())
                    or not bool(torch.equal(rng[0], torch.get_rng_state()))
                    or rng[1] != pickle.dumps(np.random.get_state())
                    or rng[2] != random.getstate()
                ):
                    raise ValueError("original sequential source/actor/RNG changed")
                if readout_guard is not None:
                    readout_guard()
                    readout_budget.before_final()
                # Refuse evidence changed by native snapshot or clock callbacks.
                check_evidence()

            for point in (2, 4, 6, 8):
                guard()  # Validate preceding success before admitting further TRAIN work.
                active_point = point
                for update in (point - 1, point):
                    guard()
                    stream.run_until(update)
                    guard()
                state = copy_snapshot(stream.candidate_snapshot())
                guard()
                paid_receipts = [asdict(r) for r in stream.applied_updates]
                if len(paid_receipts) != point or state.completed_updates != point:
                    raise ValueError("sequential stream lost its committed fit prefix")
                guard()
                if point == 8:
                    # Seal completed TRAIN before readout-only work, retaining
                    # the original lifetime, charges and proof after closing.
                    report["training"] = stream.finish()
                    seal, _ = paid_report_guard(report["training"])
                    training_evidence = (report["training"], seal)
                    guard()
                if readout_budget is None:
                    readout_budget = ToyBudgetSession(
                        ToyExecutionBudget(
                            max_training_updates=0,
                            max_replay_examples=0,
                            max_wall_seconds=90,
                            max_process_rss_bytes=RSS,
                        ),
                        monotonic,
                    )
                    readout_budget.attach_memory(memory)
                    readout_guard = _return_budget_guard(readout_budget)
                reader = source.fork()
                reader.restore_state(state)
                games = evaluate_recurring_recurrent_readout_checkpoint(
                    reader,
                    state,
                    updates=point,
                    sources=sources[: min(point, 6)],
                    budget=readout_budget,
                    check_boundary=guard,
                )
                record = dict(updates=point, training_receipts=deepcopy(paid_receipts), games=games)
                records.append(record)
                seal, _ = paid_report_guard(record)
                prior.append((record, seal))
                _qualified_recurring_games(point, games)
                guard()
            guard()
            report["outcomes"] = _recurring_outcomes(records)
            guard()
            report["preservation"] = dict(
                source=True, actor=True, torch_rng=True, numpy_rng=True, python_rng=True
            )
            report["status"] = "complete"
        except Exception as error:
            report["reason"] = f"{type(error).__name__}: {error}"
            if isinstance(error, TrainingUpdateFailed):
                report["failure_work"] = dict(error.work)
            if hasattr(error, "report"):
                records.append(
                    dict(updates=active_point, games=deepcopy(error.report), status="incomplete")
                )
            if stream is not None and report["training"] is None:
                try:
                    report["training"] = dict(
                        status="incomplete",
                        applied_updates=[asdict(r) for r in stream.applied_updates],
                    )
                except Exception as readback:
                    report["readback_error"] = f"{type(readback).__name__}: {readback}"
        finally:
            report["training_budget"] = _budget_report(train_budget)
            report["readout_budget"] = _budget_report(readout_budget)
            report["seconds"] = monotonic() - started
            report["peak_bytes"] = memory.peak_bytes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--recurring", action="store_true", help="plain A/B/C/A TRAIN and arrived-task readouts"
    )
    mode.add_argument(
        "--fresh-probe", action="store_true", help="isolated two-fit F probe and two-fit A return"
    )
    args = parser.parse_args()
    if args.describe:
        print(
            json.dumps(
                describe_recurring()
                if args.recurring
                else describe(fresh_probe=True)
                if args.fresh_probe
                else describe(),
                allow_nan=False,
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    with args.output.open("x", encoding="utf8") as output:
        if args.recurring:
            result = run_recurring_example()
        elif args.fresh_probe:
            result = run_example(fresh_probe=True)
        else:
            result = run_example()
        encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf8")) > 2 * 1024**2:
            raise ValueError("recurrent readout report exceeds2MiB")
        output.write(encoded + "\n")
    print(json.dumps(dict(status=result["status"], reason=result["reason"])))
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
