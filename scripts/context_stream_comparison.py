"""Finite missing-controller applications through existing contextual ports.

Reuse limited periodic engineering evidence and a frozen scalar fit as data.
No refit, final/probe access, runtime framework or retrospective confirmation.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any

from scripts import context_head_live as live
from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes
from scripts.context_head_serving import TASKS
from scripts.context_training_streams import CONTEXT_TRAINING_RECIPES, WITHHELD_CONTEXT_RECIPES
from scripts.ticket_serving_timeline import serving_preview
from src.app.probe_return_phase import paid_report_guard
from src.core.contextual_controller import ContextualChoice, ContextualEstimate, FittedController
from src.core.controller_objective import (
    ControllerObjective,
    ControllerMeasures,
    controller_utility,
)
from src.core.reference_controller import ReferencePolicy
from src.shared.process_memory import ProcessRssSampler

PERIODIC_SHA = "92b1cb91c3eb06c1fc7234bef09a2945a0e760d81a761959f2db406830771a79"
FIT_SHA = "d96f68a5a821cb6be1e7f55a0f7dd0cedc6519600322a7b68272cb0eff39dc3b"
MISSING = ("no_learning", "drift", "budget_aware", "fitted", "candidate_drift")
MODES = ("no_learning", "periodic", "drift", "budget_aware", "fitted", "candidate_drift")
WITHHELD_MODES = (*MODES, "acknowledgement_drift")


def _same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def protocol_snapshot(*, withheld=False):
    if type(withheld) is not bool:
        raise ValueError("withheld comparison opt-in must be an exact boolean")
    result = dict(
        definition="context_stream_comparison_protocol_v1",
        complete=True,
        plan=asdict(live.recipe_plan()),
        opportunities=serving_preview(live.recipe_plan()),
        recipes=[asdict(r) for r in CONTEXT_TRAINING_RECIPES],
        missing_order=MISSING,
        common_ticks=list(range(17, 23)),
        objective=asdict(ControllerObjective()),
        qualification="periodic84 engineering evidence has incomplete prior scenario declaration; not confirmation; denial22 retained; costs unpaired",
    )
    if withheld:
        result.update(
            definition="withheld_context_orders_protocol_v1",
            recipes=[asdict(r) for r in WITHHELD_CONTEXT_RECIPES],
            missing_order=WITHHELD_MODES,
            case_count=14,
            qualification="fresh matched packet orders/delays; known TRAIN templates, denial22 retained; joint factors and serial/asymmetric costs, no sealed-task/CPC/efficiency claim",
        )
    return result


def _native_exposures(calls, receipts, row_counts, indices):
    """Reconcile cumulative work from the qualified source's 72 rows/16 errors."""
    if len(calls) != len(indices):
        raise ValueError("native calls differ from applied packet indices")
    examples, mistakes = 72, 16
    for i, (call, receipt, index) in enumerate(zip(calls, receipts, indices, strict=True)):
        rows = row_counts[index]
        delta = receipt["diagnostic"]["value"]
        if (
            type(delta) not in (int, float)
            or not isfinite(delta)
            or not 0 <= delta <= rows
            or delta != int(delta)
        ):
            raise ValueError("native mistake diagnostic is not a bounded integer")
        examples += rows
        mistakes += int(delta)
        state = call["state"]
        expected = {
            "completed_updates": 13 + i,
            "training_examples": examples,
            "mistake_updates": mistakes,
        }
        if (
            type(call["rows"]) is not int
            or call["rows"] != rows
            or any(type(state[k]) is not int or state[k] != value for k, value in expected.items())
        ):
            raise ValueError("native cumulative row/state work differs from applied packets")
    return examples - 72


def _receipt_work(case, recipe):
    """Count verified acknowledgements separately from actual native calls."""
    applied = case["applied_updates"]
    acknowledged = case.get("acknowledged_experiences", [])
    row_counts = [3 if c == "new-v1" else 6 for c in recipe.contexts]
    by_index = {}
    native_indices = []
    # Outer protocols bind a validated recipe (1..8 packets); retain their exact
    # scope while counting its declared identities rather than a fixed six.
    for kind, records in (("applied_at", applied), ("acknowledged_at", acknowledged)):
        for row in records:
            name = row["sample_id"]
            if type(name) is not str or name not in tuple(
                f"batch-{i}" for i in range(len(recipe.contexts))
            ):
                raise ValueError("unknown original receipt identity")
            index = int(name[-1])
            if (
                index in by_index
                or row["episode_id"] != "attribution"
                or row["event_id"] != f"label-{index}"
                or row["actor_version"] != "qualified-12"
                or row["learner_version"] != "c0"
                or row["observed_at"] != index
                or row["arrived_at"] != index + recipe.label_delay
                or type(row[kind]) is not int
                or not row["arrived_at"] <= row[kind] < 24
                or any(type(row[k]) is not int for k in ("observed_at", "arrived_at"))
            ):
                raise ValueError("changed original consumed receipt provenance")
            by_index[index] = (kind, row)
            if kind == "applied_at":
                native_indices.append(index)
    if set(by_index) != set(range(len(by_index))) or native_indices != sorted(native_indices):
        raise ValueError("consumption differs from original FIFO TRAIN prefix")
    native = 0
    for index in range(len(by_index)):
        kind, row = by_index[index]
        if kind == "applied_at":
            native += 1
            if type(row["update_number"]) is not int or row["update_number"] != native:
                raise ValueError("native receipt count changed")
        elif (
            type(row["wake_updates"]) is not int
            or row["wake_updates"] != native
            or type(row["candidate_revision"]) is not int
            or row["candidate_revision"] < 0
        ):
            raise ValueError("acknowledgement invented optimizer progress")
    times = [row[kind] for kind, row in (by_index[i] for i in range(len(by_index)))]
    if times != sorted(set(times)):
        raise ValueError("consumed receipts are not chronological")
    rows = _native_exposures(case["native_calls"], applied, row_counts, native_indices)
    return rows, len(acknowledged)


def _periodic_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(3 * 1024**2 + 1)
    if len(raw) > 3 * 1024**2:
        raise ValueError("periodic engineering source exceeds original bound")
    return raw


def restore_scalar_policy(record):
    if (
        record.get("complete") is not True
        or record.get("source_sha256") != HEAD_SOURCE_SHA
        or record.get("definition") != "context_controller_learning_v1"
        or record["training_state"]["completed"] != 4
        or record["training_state"]["fitted"] != record["fitted_policy"]
    ):
        raise ValueError("original complete contextual scalar fit required")
    p = record["fitted_policy"]
    policy = FittedController(
        p["learner_version"],
        p["state_digest"],
        ControllerObjective(**p["objective"]),
        p["fitted_at"],
        tuple(ContextualEstimate(**e) for e in p["estimates"]),
        ContextualChoice(**p["choices"]),
    )
    if policy.learner_version != "qualified-12" or policy.objective != ControllerObjective():
        raise ValueError("fit version/objective differs from frozen source")
    return policy


def case_outcomes(case, recipe, mode, fitted, *, withheld=False):
    """Known complete execution, exact roles/scenarios and costs before scoring."""
    if type(withheld) is not bool:
        raise ValueError("withheld comparison opt-in must be an exact boolean")
    if (
        mode not in (WITHHELD_MODES if withheld else MODES)
        or withheld
        and recipe not in WITHHELD_CONTEXT_RECIPES
        or case.get("complete") is not True
        or case.get("definition") != "live_context_recipe_v1"
        or case.get("source_sha256") != HEAD_SOURCE_SHA
        or case.get("source_unchanged") is not True
        or not _same(case.get("context_recipe"), asdict(recipe))
        or not _same(case.get("live_plan"), asdict(live.recipe_plan()))
        or not _same(case.get("promotion_policy"), asdict(live.policy()))
        or any(case.get(k) != 0 for k in ("final_examples", "probe_updates", "replay_examples"))
    ):
        raise ValueError("complete original TRAIN-only contextual recipe required")
    if mode == "fitted":
        if not _same(case.get("contextual_policy"), asdict(fitted)):
            raise ValueError("fitted policy changed")
    elif case.get("reference_policy") != asdict(
        ReferencePolicy("drift" if mode in ("candidate_drift", "acknowledgement_drift") else mode)
    ):
        raise ValueError("original fixed reference settings changed")
    diagnostic = case.get("candidate_feedback")
    if mode in ("candidate_drift", "acknowledgement_drift"):
        if (
            diagnostic is None
            or diagnostic.get("row_limit") != 144
            or diagnostic["attempts"] != diagnostic["returns"]
            or diagnostic["rows_attempted"] != diagnostic["rows_returned"]
            or not 0 <= diagnostic["rows_attempted"] <= 144
            or not all(p.get("complete") is True for p in diagnostic["records"])
        ):
            raise ValueError("incomplete/changed candidate diagnostic has no utility")
        records = diagnostic["records"]
        if (
            len(records) != diagnostic["attempts"]
            or sum(p["rows"] for p in records) != diagnostic["rows_attempted"]
        ):
            raise ValueError("candidate diagnostic charges differ from returned records")
        for p in records:
            if (
                p.get("role") != "train"
                or type(p["index"]) is not int
                or not 0 <= p["index"] < 6
                or p["label_observed_at"] != p["index"]
                or p["label_arrived_at"] != p["index"] + (recipe.label_delay if withheld else 2)
                or not p["label_arrived_at"] <= p["read_at"] < 24
                or p["rows"] != (3 if recipe.contexts[p["index"]] == "new-v1" else 6)
                or p["base_actor_version"] != "qualified-12"
                or p["candidate_version"] != "c0"
                or p["native_updates"] != 12 + p["wake_updates"]
            ):
                raise ValueError("candidate diagnostic has changed role/causal source provenance")
            if withheld:
                n = sum(u["applied_at"] < p["read_at"] for u in case["applied_updates"])
                a = sum(
                    u["acknowledged_at"] < p["read_at"]
                    for u in case.get("acknowledged_experiences", [])
                )
                if (
                    type(p.get("mistakes")) is not int
                    or not 0 <= p["mistakes"] <= p["rows"]
                    or p["wake_updates"] != n
                    or p["index"] != n + a
                ):
                    raise ValueError(
                        "candidate diagnostic differs from preceding consumed progress"
                    )
        if (
            withheld
            and mode == "acknowledgement_drift"
            and any(
                not any(
                    p["index"] == int(ack["sample_id"][-1])
                    and p["mistakes"] == 0
                    and p["read_at"] == ack["acknowledged_at"]
                    and p["wake_updates"] == ack["wake_updates"]
                    for p in records
                )
                for ack in case.get("acknowledged_experiences", [])
            )
        ):
            raise ValueError("acknowledgement has no complete zero-mistake TRAIN diagnostic")
    elif diagnostic is not None:
        raise ValueError("unexpected candidate diagnostic")
    if withheld and case.get("task_feedback") is not None:
        raise ValueError("withheld classifier study cannot substitute task-return units")
    noise = case["synthetic_label_noise"]
    if len(noise) != (3 if recipe.noisy_old_invoice else 0) or any(
        p.get("role") != "train"
        or p["batch"] != index
        or p["semantic_row"] != "v1:invoice:ready"
        or p["source_row_id"] != "update-1-1"
        or p["row_id"] != f"{recipe.recipe_id}:batch{index}:v1:invoice:ready"
        or p["clean_action"] != "route"
        or p["corrupted_action"] != "defer"
        for p, index in zip(noise, (1, 3, 5), strict=False)
    ):
        raise ValueError("declared synthetic TRAIN noise provenance changed")
    stream = case["training_stream"]
    if (
        stream["role"] != "train"
        or stream["row_counts"]
        != ([3 if c == "new-v1" else 6 for c in recipe.contexts] if withheld else [3, 6] * 3)
        or stream["label_delay"] != (recipe.label_delay if withheld else 2)
        or withheld
        and stream["contexts"] != [f"{recipe.recipe_id}:{c}" for c in recipe.contexts]
    ):
        raise ValueError("original TRAIN receipt recipe changed")
    observed = case["live_serving"]
    if (
        not observed["closed"]
        or observed["stopped"]
        or observed["unreturned_work_unknown"]
        or case["action_snapshot"]["progress"]["unreturned_work_unknown"]
    ):
        raise ValueError("unknown live/native work has no utility")
    rows = observed["records"]
    preview = serving_preview(live.recipe_plan())
    if len(rows) != 24:
        raise ValueError("all24original opportunities required")
    for row, task in zip(rows, preview, strict=True):
        if (
            row["role"] != "live_service"
            or row.get("public_context") is not True
            or any(row.get(k) != task[k] for k in ("tick", "policy_version", "concept", "scenario"))
            or row.get("complete") is not True
            or row["episode"].get("complete") is not True
            or row["episode"].get("scenario") != task["scenario"]
            or type(row["episode"].get("success")) is not bool
            or any(q.get("complete") is not True for q in row["requests"])
        ):
            raise ValueError("changed/incomplete task/scenario/role/typed evidence has no utility")
    work = case["training_work"]
    n = work["returned_updates"]
    actual_rows, acknowledgements = (
        _receipt_work(case, recipe)
        if withheld
        else ((0, 3, 9, 12, 18, 21, 27)[n] if type(n) is int and 0 <= n <= 6 else -1, 0)
    )
    consumed = n + acknowledgements
    if withheld:
        if mode != "acknowledgement_drift" and (acknowledgements or "receipt_progress" in case):
            raise ValueError("unexpected TRAIN acknowledgement")
        if mode == "acknowledgement_drift" and case.get("receipt_progress") != dict(
            actual_updates=n,
            acknowledged=acknowledgements,
            consumed=consumed,
            total=6,
            complete=consumed == 6,
        ):
            raise ValueError("acknowledged progress differs from actual native work")
    if (
        type(n) is not int
        or not 0 <= n <= 6
        or work["attempted_updates"] != n
        or type(work["attempted_updates"]) is not int
        or type(work["attempted_rows"]) is not int
        or work["attempted_rows"] != actual_rows
        or case["budget"]["updates"] != n
        or len(case["applied_updates"]) != n
        or len(case["native_calls"]) != n
        or any(c.get("complete") is not True for c in case["native_calls"])
        or case["stream"]["pending_unapplied"] != 6 - consumed
        or case["stream"]["arrived_labels"] != 6
        or case["stream"]["opportunities"] != 24
        or case["budget"]["limits"]["max_training_updates"] != 6
        or case["budget"]["limits"]["max_wall_seconds"] != 60
        or mode == "periodic"
        and n != 6
        or mode == "no_learning"
        and n != 0
    ):
        raise ValueError("incomplete/changed original native work or envelope")
    if consumed < 6 and any(
        k in case for k in ("guard_report", "promotion_receipt", "drain_publication")
    ):
        raise ValueError("undrained recipe cannot publish")
    if (
        type(case["elapsed_seconds"]) not in (int, float)
        or not isfinite(case["elapsed_seconds"])
        or not 0 <= case["elapsed_seconds"] <= 180
        or type(case["memory"]["peak_bytes"]) is not int
        or not 0 < case["memory"]["peak_bytes"] <= 512 * 1024**2
    ):
        raise ValueError("unknown/invalid case cost has no utility")
    window = rows[17:23]
    if tuple((r["policy_version"], r["concept"]) for r in window) != TASKS:
        raise ValueError("original common task window changed")
    return dict(
        old=sum(r["episode"]["success"] for r in window[:4]),
        new=sum(r["episode"]["success"] for r in window[4:]),
        updates=n,
        rows=work["attempted_rows"],
        pending=6 - consumed,
    )


def score_cases(cases, fitted, *, withheld=False):
    if type(withheld) is not bool:
        raise ValueError("withheld comparison opt-in must be an exact boolean")
    from scripts.run_bp_ticket_serving import _violation_count

    recipes = WITHHELD_CONTEXT_RECIPES if withheld else CONTEXT_TRAINING_RECIPES
    modes = WITHHELD_MODES if withheld else MODES
    count = len(recipes) * len(modes)
    if type(cases) is not list or len(cases) != count:
        raise ValueError(f"all{count}distinct complete cases required before any utility")
    expected = {(r.recipe_id, m) for r in recipes for m in modes}
    indexed = {(c["recipe_id"], c["mode"]): c["result"] for c in cases}
    if set(indexed) != expected:
        raise ValueError(f"all{count}distinct complete cases required before any utility")
    outcomes = {
        key: case_outcomes(
            value,
            next(r for r in recipes if r.recipe_id == key[0]),
            key[1],
            fitted,
            **(dict(withheld=True) if withheld else {}),
        )
        for key, value in indexed.items()
    }
    scored = []
    for recipe in recipes:
        base = outcomes[recipe.recipe_id, "no_learning"]
        for mode in modes:
            c = indexed[recipe.recipe_id, mode]
            o = outcomes[recipe.recipe_id, mode]
            episodes = [r["episode"] for r in c["live_serving"]["records"]] + [
                r["episode"] for r in c["episodes"]
            ]
            measures = ControllerMeasures(
                base["new"],
                o["new"],
                2,
                base["old"],
                o["old"],
                4,
                c["elapsed_seconds"],
                c["memory"]["peak_bytes"],
                _violation_count(episodes),
                True,
            )
            scored.append(
                dict(
                    recipe_id=recipe.recipe_id,
                    mode=mode,
                    outcomes=o,
                    measures=asdict(measures),
                    utility=asdict(controller_utility(ControllerObjective(), measures)),
                )
            )
    return scored


def run_comparison(input_path, periodic_path, fitted_path, *, withheld=False):
    if type(withheld) is not bool:
        raise ValueError("withheld comparison opt-in must be an exact boolean")
    if withheld and periodic_path is not None:
        raise ValueError("withheld opt-in requires fresh cases without periodic source")
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="context_stream_comparison_v1",
        complete=False,
        cases=[],
        scores=[],
        protocol=protocol_snapshot(withheld=withheld),
        source_sha256=HEAD_SOURCE_SHA,
        periodic_sha256=PERIODIC_SHA,
        fit_sha256=FIT_SHA,
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        prior_training=dict(updates=12, rows=72, mistakes=16),
        historical_fit_work=dict(updates=2, rows=9, mistakes=1),
        reused_periodic_work=dict(updates=12, rows=54, mistakes=7),
        qualifications=[
            "84scenario description incomplete; reused periodic is limited engineering evidence, not confirmation",
            "fixed-order serial retained heap/earlier periodic/asymmetric diagnostics: no efficiency claim",
            "denial22 remains in common window, not acquisitionfailure; no heldout/CPC/generalization claim",
        ],
    )
    if withheld:
        report.update(
            definition="withheld_context_orders_v1",
            qualifications=[
                "fresh order/delay validation; known teacher templates, not unseen-task generalization",
                "joint order/delay factors; serial retained heap/asymmetric diagnostics, no efficiency claim",
                "frozen classifier fit82; taskfit87 remains failed/unverified; denial22 retained",
            ],
        )
        report.pop("periodic_sha256")
        report.pop("reused_periodic_work")
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            raw = _source_bytes(input_path)
            prior_raw = b"" if withheld else _periodic_bytes(periodic_path)
            fit_raw = _source_bytes(fitted_path)
            if (
                sha256(raw).hexdigest() != HEAD_SOURCE_SHA
                or not withheld
                and sha256(prior_raw).hexdigest() != PERIODIC_SHA
                or sha256(fit_raw).hexdigest() != FIT_SHA
            ):
                raise ValueError("original source/periodic/fit bytes replaced")
            fit_record = json.loads(fit_raw)
            fitted = restore_scalar_policy(fit_record)
            fit_pin = deepcopy(fitted)
            report["historical_fit_source_costs"] = dict(
                whole_study=fit_record["elapsed_seconds"],
                memory=fit_record["memory"],
                trial_seconds=[t["result"]["elapsed_seconds"] for t in fit_record["trials"]],
                qualification="whole source includes old application; no all-in/amortized new utility claim",
            )
            prior = {} if withheld else json.loads(prior_raw)
            if not withheld and (prior.get("complete") is not True or len(prior["cases"]) != 2):
                raise ValueError("complete original engineering cases required")
            cases = report["cases"]
            paid: list[Any] = []
            pin = deepcopy(report["protocol"])

            def check():
                if monotonic() - started > 600 or memory.snapshot().peak_bytes > 512 * 1024**2:
                    raise ValueError("original missing10study allowance exhausted")
                if (
                    _source_bytes(input_path) != raw
                    or not withheld
                    and _periodic_bytes(periodic_path) != prior_raw
                    or _source_bytes(fitted_path) != fit_raw
                    or fitted != fit_pin
                    or not _same(protocol_snapshot(withheld=withheld), pin)
                ):
                    raise ValueError("original source/periodic/fit/protocol changed")
                if report["cases"] is not cases or len(cases) != len(paid):
                    report["cases"] = cases
                    cases[:] = [c for c, _, _ in paid]
                    raise ValueError("preceding paid case container replaced")
                for i, (row, expected, guard) in enumerate(paid):
                    if cases[i] is not row:
                        cases[:] = [c for c, _, _ in paid]
                        raise ValueError("preceding paid case replaced")
                    try:
                        guard()
                    except ValueError:
                        row.clear()
                        row.update(deepcopy(expected))
                        raise ValueError("preceding paid case corrupted") from None

            def retain(result, recipe, mode, origin):
                row = dict(recipe_id=recipe.recipe_id, mode=mode, origin=origin, result=result)
                cases.append(row)  # Retain paid failed/malformed returns before refusing more work.
                case_outcomes(
                    result, recipe, mode, fitted, **(dict(withheld=True) if withheld else {})
                )
                guard, expected = paid_report_guard(row)
                paid.append((row, expected, guard))

            if not withheld:
                for recipe, result in zip(CONTEXT_TRAINING_RECIPES, prior["cases"], strict=True):
                    retain(result, recipe, "periodic", "retrospective84engineering")
            check()
            for recipe in WITHHELD_CONTEXT_RECIPES if withheld else CONTEXT_TRAINING_RECIPES:
                for mode in WITHHELD_MODES if withheld else MISSING:
                    check()
                    kwargs: dict[str, Any] = dict(stream_recipe=recipe)
                    if mode == "fitted":
                        kwargs["contextual_policy"] = fitted
                    else:
                        kwargs["reference_mode"] = (
                            "drift"
                            if mode in ("candidate_drift", "acknowledgement_drift")
                            else mode
                        )
                    if mode in ("candidate_drift", "acknowledgement_drift"):
                        kwargs["candidate_feedback"] = True
                    if mode == "acknowledgement_drift":
                        kwargs["acknowledge_correct"] = True
                    result = live.run_case(input_path, **kwargs)
                    retain(result, recipe, mode, "prospectively_declared_missing_application")
                    check()
            report["scores"] = score_cases(
                cases, fitted, **(dict(withheld=True) if withheld else {})
            )
            check()
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report.update(elapsed_seconds=monotonic() - started, memory=asdict(memory.snapshot()))
    return report
