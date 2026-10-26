"""Fixed same-recipe references through the existing contextual live API.

Only three missing controls execute; accepted arrived-shift evidence is data.
No learner/policy fitting, scheduler, final/probe access or metric selection.
All cases must finish before readonly scoring. Costs are unequal and unpaired.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts import context_head_live as live
from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes
from scripts.context_head_serving import TASKS
from scripts.context_stream_comparison import _receipt_work, _same
from scripts.context_training_streams import ContextTrainingRecipe
from scripts.ticket_serving_timeline import serving_preview
from src.app.probe_return_phase import paid_report_guard
from src.core.controller_objective import (
    ControllerMeasures,
    ControllerObjective,
    controller_utility,
)
from src.core.reference_controller import ReferencePolicy
from src.shared.process_memory import ProcessRssSampler

RECIPE = ContextTrainingRecipe(
    "arrived_shift_blocks",
    ("old-v1",) * 3 + ("new-v1",) * 3 + ("old-v1",) * 2,
    2,
)
MISSING = ("no_learning", "periodic", "candidate_ack")
MODES = (*MISSING, "arrived_shift")
REUSED_SHA = "a7aeeb064a58dbbf97bde3eeac3c859ed3c3068cb6652db85c3541cf2b877d86"


def _reused_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or sha256(raw).hexdigest() != REUSED_SHA:
        raise ValueError("accepted100 shift report replaced or exceeds original bound")
    return raw


def case_outcome(case, mode, *, recipe=RECIPE):
    if type(recipe) is not ContextTrainingRecipe or len(recipe.contexts) != 8:
        raise ValueError("readonly outcome requires an explicit eight-packet recipe")
    recipe.__post_init__()
    if (
        mode not in MODES
        or case.get("complete") is not True
        or case.get("definition") != "live_context_recipe_v1"
        or case.get("source_sha256") != HEAD_SOURCE_SHA
        or case.get("source_unchanged") is not True
        or not _same(case.get("context_recipe"), asdict(recipe))
        or not _same(case.get("live_plan"), asdict(live.recipe_plan()))
        or not _same(case.get("promotion_policy"), asdict(live.policy()))
        or any(case.get(k) != 0 for k in ("final_examples", "probe_updates", "replay_examples"))
    ):
        raise ValueError("complete original eight-packet TRAIN-only recipe required")
    expected_mode = "drift" if mode in ("candidate_ack", "arrived_shift") else mode
    if case.get("reference_policy") != asdict(ReferencePolicy(expected_mode)):
        raise ValueError("original reference settings changed")
    observed = case["live_serving"]
    if (
        not observed["closed"]
        or observed["stopped"]
        or observed["unreturned_work_unknown"]
        or case["action_snapshot"]["progress"]["unreturned_work_unknown"]
    ):
        raise ValueError("unknown serving/native work has no utility")
    rows = observed["records"]
    if len(rows) != 24:
        raise ValueError("all twenty-four live opportunities required")
    for row, declared in zip(rows, serving_preview(live.recipe_plan()), strict=True):
        if (
            row.get("role") != "live_service"
            or row.get("public_context") is not True
            or any(
                row.get(k) != declared[k] for k in ("tick", "policy_version", "concept", "scenario")
            )
            or row.get("complete") is not True
            or row["episode"].get("complete") is not True
            or row["episode"].get("scenario") != declared["scenario"]
            or type(row["episode"].get("success")) is not bool
            or any(q.get("complete") is not True for q in row["requests"])
        ):
            raise ValueError("changed/incomplete task role/scenario/typed evidence has no utility")
    actual_rows, acknowledged = _receipt_work(case, recipe)
    work = case["training_work"]
    n = work["returned_updates"]
    if (
        type(n) is not int
        or not 0 <= n <= 8
        or work["attempted_updates"] != n
        or work["attempted_rows"] != actual_rows
        or case["budget"]["updates"] != n
        or mode == "no_learning"
        and n != 0
        or mode == "periodic"
        and n != 8
        or case["stream"]["pending_unapplied"] != 8 - n - acknowledged
        or case["stream"]["arrived_labels"] != 8
        or case["stream"]["opportunities"] != 24
        or case["stream"]["score_forward_calls"] != 0
        or any(c.get("complete") is not True for c in case["native_calls"])
        or case["budget"]["limits"]["max_training_updates"] != 8
    ):
        raise ValueError("training/receipt/budget work differs from original finite case")
    if mode != "candidate_ack" and (acknowledged or "candidate_feedback" in case):
        raise ValueError("unexpected acknowledgement/candidate work")
    diagnostic = (
        case.get("candidate_feedback")
        if mode == "candidate_ack"
        else case.get("arrived_shift_feedback")
    )
    if mode in ("candidate_ack", "arrived_shift"):
        if (
            diagnostic is None
            or diagnostic["attempts"] != diagnostic["returns"]
            or diagnostic["rows_attempted"] != diagnostic["rows_returned"]
            or sum(r["rows"] for r in diagnostic["records"]) != diagnostic["rows_attempted"]
            or len(diagnostic["records"]) != diagnostic["attempts"]
            or any(r.get("complete") is not True for r in diagnostic["records"])
        ):
            raise ValueError("incomplete diagnostic work has no utility")
    elif "arrived_shift_feedback" in case:
        raise ValueError("unexpected arrived-shift work")
    if mode == "candidate_ack":
        if diagnostic["attempts"] > 24 or diagnostic["rows_attempted"] > 144:
            raise ValueError("candidate diagnostic allowance exceeded")
        if case.get("receipt_progress") != dict(
            actual_updates=n,
            acknowledged=acknowledged,
            consumed=n + acknowledged,
            total=8,
            complete=n + acknowledged == 8,
        ):
            raise ValueError("acknowledgement progress invented native work")
        for row in diagnostic["records"]:
            i = row["index"]
            prior_updates = sum(u["applied_at"] < row["read_at"] for u in case["applied_updates"])
            prior_ack = sum(
                a["acknowledged_at"] < row["read_at"] for a in case["acknowledged_experiences"]
            )
            if (
                type(i) is not int
                or not 0 <= i < 8
                or row.get("role") != "train"
                or row["label_observed_at"] != i
                or row["label_arrived_at"] != i + recipe.label_delay
                or not i + recipe.label_delay <= row["read_at"] < 24
                or row["native_updates"] != 12 + row["wake_updates"]
                or row["rows"] != (3 if recipe.contexts[i] == "new-v1" else 6)
                or i != prior_updates + prior_ack
                or row["wake_updates"] != prior_updates
                or row["base_actor_version"] != "qualified-12"
                or row["candidate_version"] != "c0"
            ):
                raise ValueError("candidate feedback role/clock/version changed")
    window = rows[17:23]
    if tuple((r["policy_version"], r["concept"]) for r in window) != TASKS:
        raise ValueError("common old/new task window changed")
    return dict(
        old=sum(r["episode"]["success"] for r in window[:4]),
        new=sum(r["episode"]["success"] for r in window[4:]),
        live_successes=sum(r["episode"]["success"] for r in rows),
        updates=n,
        rows=actual_rows,
        acknowledged=acknowledged,
        pending=8 - n - acknowledged,
        diagnostic_rows=0 if diagnostic is None else diagnostic["rows_attempted"],
    )


def score_cases(cases):
    from scripts.run_bp_ticket_serving import _violation_count

    if type(cases) is not dict or set(cases) != set(MODES):
        raise ValueError("all four complete methods required before any scoring")
    outcomes = {m: case_outcome(cases[m], m) for m in MODES}
    base = outcomes["no_learning"]
    scored = []
    for mode in MODES:
        case, outcome = cases[mode], outcomes[mode]
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


def run_comparison(input_path, saved_shift_path):
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="context_fifo_reference_study_v1",
        complete=False,
        cases={},
        scores=[],
        pending_modes=list(MISSING),
        source_sha256=HEAD_SOURCE_SHA,
        reused_shift_sha256=REUSED_SHA,
        objective=asdict(ControllerObjective()),
        prior_training=dict(updates=12, rows=72, mistakes=16),
        reused_work=dict(updates=2, rows=12, diagnostic_rows=39),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        qualification="known TRAIN templates, shared opportunity/credit ceilings, unequal work; unpaired time/RSS proxies include conditional rollback",
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            raw = _source_bytes(input_path)
            if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                raise ValueError("qualified source replaced")
            saved = _reused_bytes(saved_shift_path)
            cases = report["cases"]
            guards: dict[str, Any] = {}

            def check():
                rss = memory.sample()
                if monotonic() - started > 600 or rss is None or rss > 512 * 1024**2:
                    raise ValueError("original study time/RSS budget exhausted")
                if _source_bytes(input_path) != raw or _reused_bytes(saved_shift_path) != saved:
                    raise ValueError("source or accepted100 evidence changed")
                if report["cases"] is not cases or set(cases) != set(guards):
                    report["cases"] = cases
                    cases.clear()
                    cases.update({m: value[0] for m, value in guards.items()})
                    raise ValueError("preceding paid cases changed")
                for mode, (owner, guard, expected) in guards.items():
                    if cases[mode] is not owner:
                        cases[mode] = owner
                        raise ValueError("preceding paid case replaced")
                    try:
                        guard()
                    except ValueError:
                        owner.clear()
                        owner.update(deepcopy(expected))
                        raise ValueError("preceding paid case corrupted") from None

            def retain(mode, case):
                cases[mode] = case  # Preserve a paid failed prefix before validation.
                case_outcome(case, mode)
                if mode != "arrived_shift" and (
                    not _same(case["training_stream"], cases["arrived_shift"]["training_stream"])
                    or case["budget"]["limits"] != cases["arrived_shift"]["budget"]["limits"]
                ):
                    raise ValueError("matched data/label schedule or budget limits changed")
                guard, expected = paid_report_guard(case)
                guards[mode] = (case, guard, expected)

            retain("arrived_shift", json.loads(saved))
            for mode in MISSING:
                check()
                kwargs = dict(reference_mode=mode, stream_recipe=RECIPE)
                if mode == "candidate_ack":
                    kwargs.update(
                        reference_mode="drift", candidate_feedback=True, acknowledge_correct=True
                    )
                case = live.run_case(input_path, **kwargs)
                retain(mode, case)
                report["pending_modes"].remove(mode)
                check()
            report["scores"] = score_cases(cases)
            check()
            report["complete"] = True
        except Exception as error:
            report["scores"] = []
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report["elapsed_seconds"] = monotonic() - started
            report["study_memory"] = asdict(memory.snapshot())
    return report
