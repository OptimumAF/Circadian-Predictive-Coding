"""Compose the existing fixed probe from an owned terminal catalog candidate.

No source-file/model loader, new learning loop, promotion or comparison utility.
The caller owns the live window; this module owns only disposable diagnostic work.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any, Callable

import numpy as np

from scripts import ticket_workflow_native as native
from scripts import probe_ticket_policy_learning as recipe
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard

MAX_UPDATES = 64
MAX_ROWS = 192


def _budget_state(budget):
    # Observation only: before_final would advance the finished window's clock.
    return (
        budget_binding(budget),
        asdict(budget.budget),
        deepcopy(budget.progress),
        deepcopy(
            {
                k: v
                for k, v in vars(budget).items()
                if k not in {"budget", "clock", "progress", "process_rss_sampler"}
            }
        ),
    )


def _terminal_guard(report, row, runtime, source, control, gate, live, budget, require_source):
    if (
        report.get("complete") is not True
        or report.get("catalog_actor_unchanged") is not True
        or report.get("source_unchanged") is not True
        or report.get("numpy_rng_unchanged") is not True
        or report.get("torch_rng_unchanged") is not True
        or report["training_work"]["unreturned_work_unknown"] is not False
        or not live.snapshot()["closed"]
        or live.snapshot()["unreturned_work_unknown"]
    ):
        raise ValueError("probe requires a complete known closed private window")
    paid = {k: v for k, v in report.items() if k != "isolated_probe"}
    paid_guard, expected_window = paid_report_guard(paid)
    candidate = runtime.candidate_snapshot()
    source_pin, actor_pin = source.snapshot_state(), runtime.actor.serving_snapshot()
    owners = (runtime._candidate, runtime._inbox, gate._limits, control.record, control.replay)

    def state():
        with runtime._exclusive():
            cursor = runtime._inbox._borrow_cursor()
            entries = (*cursor.experiences, *cursor.labels, *cursor.applied)
            cursors = deepcopy((cursor, runtime._read_consolidation_cursor()))
        return (
            entries,
            cursors,
            gate.snapshot(),
            _budget_state(budget),
            _budget_state(control.replay_budget),
        )

    entries, *values = state()
    baseline = deepcopy(values)

    def guard():
        require_source()
        control.require_pin()
        live._require_original()
        if report.get("isolated_probe") is not row or report.keys() != {*paid, "isolated_probe"}:
            raise ValueError("original diagnostic/report owner replaced")
        if any(report[k] is not v for k, v in paid.items()):
            raise ValueError("original paid window record replaced")
        paid_guard()
        current_owners = (
            runtime._candidate,
            runtime._inbox,
            gate._limits,
            control.record,
            control.replay,
        )
        if any(a is not b for a, b in zip(owners, current_owners, strict=True)):
            raise ValueError("original terminal runtime owner replaced")
        current_entries, *current_values = state()
        if (
            len(current_entries) != len(entries)
            or any(a is not b for a, b in zip(entries, current_entries))
            or not _same_value(current_values, baseline)
        ):
            raise ValueError("original receipts, pending work, sharing or budgets changed")
        if not (
            native.same_state(runtime.candidate_snapshot(), candidate)
            and native.same_state(source.snapshot_state(), source_pin)
            and native.same_state(runtime.actor.serving_snapshot(), actor_pin)
        ):
            raise ValueError("original source, candidate or actor changed")

    guard()

    def retain_evidence():
        try:
            if any(report.get(k) is not v for k, v in paid.items()):
                raise ValueError("paid window replaced")
            paid_guard()
        except ValueError:
            row["paid_window_evidence_refused"] = True
            for key, value in expected_window.items():
                report[key] = deepcopy(value)

    return guard, candidate.state, retain_evidence


def run_catalog_probe(
    report, *, runtime, source, control, gate, live, budget, require_source, spec, memory
):
    """Retain partial paid work; never train through original live callbacks."""
    row = report["isolated_probe"]
    row.update(
        definition="owned_catalog_candidate_probe_v1",
        status="failed",
        complete=False,
        cases=[],
        training_work=dict(
            attempted_updates=0, returned_updates=0, attempted_rows=0, unreturned_work_unknown=False
        ),
        scope="fixed known TRAIN-derived policy probe; not unseen-task generalization, plasticity loss, final evaluation or deployed learning",
    )
    work = row["training_work"]
    cases = row["cases"]
    created_arms: list[dict[str, Any]] = []
    expected = dict(attempted_updates=0, returned_updates=0, attempted_rows=0)
    seeds: list[tuple[Any, Any]] = []
    paid: list[tuple[dict[str, Any], Callable[[], None], dict[str, Any]]] = []
    retain_window = None
    initialization = row["diagnostic_seed_owners"] = dict(attempts=0, returns=0)
    started = monotonic()
    numpy_before = deepcopy(np.random.get_state())
    torch = None
    torch_before = None
    try:
        torch = native.require_torch()
        torch_before = torch.get_rng_state().clone()
        terminal_guard, candidate, retain_window = _terminal_guard(
            report, row, runtime, source, control, gate, live, budget, require_source
        )
        cohorts = recipe.build_cohorts(spec)
        cohort_pin = deepcopy(cohorts)

        def guard():
            terminal_guard()
            if monotonic() - started > 180:
                raise ValueError("original aggregate diagnostic deadline exhausted")
            if not _same_value(np.random.get_state(), numpy_before) or not torch.equal(
                torch.get_rng_state(), torch_before
            ):
                raise ValueError("diagnostic changed caller RNG")
            if not _same_value(cohorts, cohort_pin):
                raise ValueError("original fixed diagnostic data changed")
            if row.get("training_work") is not work or any(
                type(work.get(k)) is not int or work[k] != v for k, v in expected.items()
            ):
                raise ValueError("paid diagnostic counters changed")
            if row.get("cases") is not cases or work.get("unreturned_work_unknown") is not False:
                raise ValueError("original diagnostic cases or work status replaced")
            for seed, state in seeds:
                if not native.same_state(seed.snapshot_state(), state):
                    raise ValueError("diagnostic source owner changed")
            for i, (previous, check, _) in enumerate(paid):
                if row["cases"][i] is not previous:
                    raise ValueError("preceding paid diagnostic arm replaced")
                check()

        def before(features):
            guard()
            expected["attempted_updates"] += 1
            expected["attempted_rows"] += len(features)
            work.update(expected)
            if expected["attempted_updates"] > MAX_UPDATES or expected["attempted_rows"] > MAX_ROWS:
                raise ValueError("original aggregate diagnostic work exhausted")

        def after():
            expected["returned_updates"] += 1
            work.update(expected)

        for name in ("warm", "fresh"):
            guard()
            # Why this: native forks inherit callbacks. These callbacks belong
            # only to diagnostics, never the source's live training ledger.
            initialization["attempts"] += 1
            seed = native.counted_learner("bp", 7, 6, before, after)
            initialization["returns"] += 1
            if seed is source or seed is runtime._candidate:
                raise ValueError("diagnostic seed aliases the original live learner")
            guard()
            if name == "warm":
                seed.restore_state(candidate)
            state = seed.snapshot_state()
            if name == "fresh" and (state.completed_updates != 0 or state.training_examples != 0):
                raise ValueError("fresh probe requires fixed untrained initialization")
            seeds.append((seed, deepcopy(state)))
            guard()
            arm: dict[str, Any] = {}
            created_arms.append(arm)
            cases.append(arm)
            recipe.run_arm(
                seed, state, cohorts, spec, guard, memory, name, row=arm, input_diagnostics=True
            )
            recipe._require_arm(arm)
            guard()
            check, accepted = paid_report_guard(arm)
            paid.append((arm, check, accepted))
        guard()
        row.update(complete=True, status="complete", original_state_unchanged=True)
    except Exception as error:
        row.update(complete=False, status="failed", error=f"{type(error).__name__}: {error}"[:1024])
    finally:
        if report.get("isolated_probe") is not row:
            row["paid_evidence_refused"] = True
            report["isolated_probe"] = row
        if (
            row.get("cases") is not cases
            or len(cases) != len(created_arms)
            or any(a is not b for a, b in zip(cases, created_arms))
        ):
            row["paid_evidence_refused"] = True
            cases[:] = created_arms
            row["cases"] = cases
        if retain_window is not None:
            retain_window()
        # Preserve the bounded two original accepted records after refusing an
        # edit/replacement; this restores evidence only, never model authority.
        for i, (previous, check, accepted) in enumerate(paid):
            try:
                if row["cases"][i] is not previous:
                    raise ValueError("paid arm replaced")
                check()
            except (ValueError, IndexError):
                row["paid_evidence_refused"] = True
                previous.clear()
                previous.update(deepcopy(accepted))
                if i < len(row["cases"]):
                    row["cases"][i] = previous
                else:
                    row["cases"].append(previous)
        # Counters come from the original callbacks, even if visible work was edited.
        work.update(expected)
        row["training_work"] = work
        work["unreturned_work_unknown"] = (
            expected["attempted_updates"] != expected["returned_updates"]
        )
        row["numpy_rng_unchanged"] = _same_value(np.random.get_state(), numpy_before)
        row["torch_rng_unchanged"] = (
            None if torch is None else bool(torch.equal(torch.get_rng_state(), torch_before))
        )
        initialization["unreturned_work_unknown"] = (
            initialization["attempts"] != initialization["returns"]
        )
        row["elapsed_seconds"] = monotonic() - started
