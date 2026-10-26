"""Frozen references on an accepted learnable contextual serving recipe.

Reuse the accepted periodic case; run only missing fixed controls through its
existing runner. Readonly phase-matched utility uses the original core objective.
No policy fitting, new runtime, final/probe access or model reconstruction.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from time import monotonic
from typing import Any

from scripts import context_head_live as live
from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes
from scripts.context_head_serving import TASKS
from src.app.probe_return_phase import paid_report_guard
from src.core.controller_objective import (
    ControllerObjective,
    ControllerMeasures,
    controller_utility,
)
from src.core.reference_controller import ReferencePolicy
from src.shared.process_memory import ProcessRssSampler

PERIODIC_SHA = "c34723a5146334084f6ea26922d4980a774c6ece26830f40fcf758eedadf2ea1"
MISSING = ("no_learning", "drift", "budget_aware")
MODES = ("no_learning", "periodic", "drift", "budget_aware")


def task_window(case):
    """Validate finite known work before scoring; pending unused TRAIN is explicit."""
    if (
        case.get("complete") is not True
        or case.get("definition") != "live_context_serving_v1"
        or case.get("source_sha256") != HEAD_SOURCE_SHA
        or case.get("source_unchanged") is not True
        or any(case.get(k) != 0 for k in ("final_examples", "probe_updates", "replay_examples"))
        or json.dumps(case.get("live_plan"), sort_keys=True, allow_nan=False)
        != json.dumps(asdict(live.live_plan()), sort_keys=True, allow_nan=False)
    ):
        raise ValueError("context comparison requires a complete original TRAIN-only live case")
    observed = case["live_serving"]
    if (
        not observed["closed"]
        or observed["stopped"]
        or observed["unreturned_work_unknown"]
        or case["action_snapshot"]["progress"]["unreturned_work_unknown"]
    ):
        raise ValueError("incomplete or unknown serving/model work has no utility")
    mode = case["reference_policy"]["mode"]
    if mode not in MODES or case["reference_policy"] != asdict(ReferencePolicy(mode)):
        raise ValueError("comparison requires unchanged original fixed reference settings")
    rows = observed["records"]
    if len(rows) != 14 or [r["tick"] for r in rows] != list(range(14)):
        raise ValueError("comparison requires the original fourteen opportunities")
    for i, row in enumerate(rows):
        if (
            row["role"] != "live_service"
            or row.get("public_context") is not True
            or (row.get("policy_version"), row.get("concept")) != live.LIVE_TASKS[i]
            or row["complete"] is not True
            or row["episode"]["complete"] is not True
            or type(row["episode"]["success"]) is not bool
            or any(q.get("complete") is not True for q in row["requests"])
        ):
            raise ValueError("incomplete or changed task/role/request evidence has no utility")
        scenario = "stale_once" if i in (4, 12) else "revoke_commit" if i == 6 else "ordinary"
        if row.get("scenario") != scenario or row["episode"].get("scenario") != scenario:
            raise ValueError("common task scenario differs from frozen comparison")
    work = case["training_work"]
    updates = work["returned_updates"]
    if (
        any(
            type(work[k]) is not int
            for k in ("attempted_updates", "attempted_rows", "returned_updates")
        )
        or mode == "periodic"
        and updates != 2
        or mode == "no_learning"
        and updates != 0
        or len(case["native_calls"]) != updates
        or any(c.get("complete") is not True for c in case["native_calls"])
        or type(updates) is not int
        or not 0 <= updates <= 2
        or work["attempted_updates"] != updates
        or work["attempted_rows"] != (0, 3, 9)[updates]
        or case["budget"]["updates"] != updates
        or case["stream"]["pending_unapplied"] != 2 - updates
        or len(case["applied_updates"]) != updates
    ):
        raise ValueError(
            "training attempts/receipts/pending work differ from original bounded case"
        )
    window = rows[7:13]
    if tuple((r["policy_version"], r["concept"]) for r in window) != TASKS:
        raise ValueError("common six-task comparison window differs")
    return dict(
        old=sum(r["episode"]["success"] for r in window[:4]),
        new=sum(r["episode"]["success"] for r in window[4:]),
        updates=updates,
        rows=work["attempted_rows"],
        pending=2 - updates,
    )


def score_cases(cases):
    from scripts.run_bp_ticket_serving import _violation_count

    if type(cases) is not list or len(cases) != 4:
        raise ValueError("comparison requires all four fixed cases before scoring")
    by_mode = {case["reference_policy"]["mode"]: case for case in cases}
    if set(by_mode) != set(MODES):
        raise ValueError(
            "comparison requires distinct no_learning/periodic/drift/budget_aware cases"
        )
    outcomes = {mode: task_window(by_mode[mode]) for mode in MODES}
    base = outcomes["no_learning"]
    scored = []
    for mode in MODES:
        case, outcome = by_mode[mode], outcomes[mode]
        episodes = [r["episode"] for r in case["live_serving"]["records"]] + [
            r["episode"] for r in case["episodes"]
        ]
        measures = ControllerMeasures(
            base["new"],
            outcome["new"],
            2,
            base["old"],
            outcome["old"],
            4,
            case["elapsed_seconds"],
            case["memory"]["peak_bytes"],
            _violation_count(episodes),
            True,
        )
        scored.append(
            dict(
                mode=mode,
                outcomes=outcome,
                measures=asdict(measures),
                utility=asdict(controller_utility(ControllerObjective(), measures)),
            )
        )
    return scored


def _run_comparison(input_path, periodic_path, memory):
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="context_reference_comparison_v1",
        complete=False,
        source_sha256=HEAD_SOURCE_SHA,
        periodic_sha256=PERIODIC_SHA,
        objective=asdict(ControllerObjective()),
        cases=[],
        scores=[],
        pending_modes=list(MISSING),
        prior_training=dict(updates=12, rows=72, mistakes=16),
        reused_periodic_work=dict(updates=2, rows=9, mistakes=2),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        metric_scope="ticks7..12 six known tasks vs same no_learning window; full invocation time/RSS incl conditional rollback; development-only unequal-work proxies",
        limitations=[
            "budget_aware reserves4 against2credits and waits; structurally unsuitable default, not general inferiority",
            "periodic saved earlier; single fixed-order instrumented controls, no paired timing efficiency claim",
            "no heldout meta-generalization, inference-overlap or CPC advantage claim",
        ],
    )
    paid: list[Any] = []
    try:
        raw = _source_bytes(input_path)
        if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
            raise ValueError("original qualified source replaced")
        periodic_raw = _source_bytes(periodic_path)
        if sha256(periodic_raw).hexdigest() != PERIODIC_SHA:
            raise ValueError("accepted periodic source replaced")
        periodic = json.loads(periodic_raw)
        task_window(periodic)
        if periodic["reference_policy"]["mode"] != "periodic":
            raise ValueError("accepted periodic source mode differs")
        cases = report["cases"]
        plan_pin = asdict(live.live_plan())

        def check():
            if monotonic() - started > 360:
                raise ValueError("original three-control study deadline exhausted")
            if memory.snapshot().peak_bytes > 512 * 1024**2:
                raise ValueError("original three-control study RSS allowance exhausted")
            if _source_bytes(input_path) != raw or _source_bytes(periodic_path) != periodic_raw:
                raise ValueError("original source or accepted periodic bytes changed")
            if asdict(live.live_plan()) != plan_pin:
                raise ValueError("original comparison live plan changed")
            if report["cases"] is not cases or len(cases) != len(paid):
                report["cases"] = cases
                cases[:] = [row for row, _, _ in paid]
                raise ValueError("preceding complete paid case container replaced")
            for i, (row, expected, guard) in enumerate(paid):
                if cases[i] is not row:
                    cases[:] = [original for original, _, _ in paid]
                    raise ValueError("preceding complete paid case replaced")
                try:
                    guard()
                except ValueError:
                    row.clear()
                    row.update(deepcopy(expected))
                    raise ValueError("preceding complete paid case corrupted") from None

        def retain(case):
            task_window(case)
            guard, expected = paid_report_guard(case)
            cases.append(case)
            paid.append((case, expected, guard))

        retain(periodic)
        for mode in MISSING:
            check()
            case = live.run_case(input_path, reference_mode=mode)
            # Keep paid failed reports even when they cannot admit more work or score.
            if case.get("complete") is not True:
                cases.append(case)
                raise ValueError(f"{mode} finite case incomplete; no following control admitted")
            if case.get("reference_policy", {}).get("mode") != mode:
                cases.append(case)
                raise ValueError(
                    "returned fixed reference mode differs; no following control admitted"
                )
            try:
                retain(case)
            except Exception:
                cases.append(case)
                raise
            report["pending_modes"].remove(mode)
            check()
        report["scores"] = score_cases(cases)
        check()
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:512]
    finally:
        report["elapsed_seconds"] = monotonic() - started
    return report


def run_comparison(input_path, periodic_path):
    """Measure comparison/report/guard overhead in addition to each original case."""
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        report = _run_comparison(input_path, periodic_path, memory)
        report["study_memory"] = asdict(memory.snapshot())
    return report
