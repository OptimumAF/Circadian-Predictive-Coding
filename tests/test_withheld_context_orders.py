"""Saved/scalar metadata fixtures only; fabricated cases are not study results."""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import pytest

from scripts import context_stream_comparison as study
from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as cli
from scripts.context_training_streams import WITHHELD_CONTEXT_RECIPES as RECIPES
from src.core.reference_controller import ReferencePolicy
from test_context_stream_comparison import SOURCE, PERIODIC, FIT


@pytest.fixture(scope="module")
def saved():
    base = json.loads(PERIODIC.read_bytes())["cases"][0]
    fit = study.restore_scalar_policy(json.loads(FIT.read_bytes()))
    return base, fit


def metadata_case(base, recipe, mode, fit):
    """Deliberately scripted input to admission; no native execution or estimates."""
    case = deepcopy(base)
    n = 0 if mode == "no_learning" else 6 if mode == "periodic" else 2
    ack = 4 if mode == "acknowledgement_drift" else 0
    counts = [3 if c == "new-v1" else 6 for c in recipe.contexts]
    case["context_recipe"] = asdict(recipe)
    case["synthetic_label_noise"] = []
    case["training_stream"].update(
        contexts=[f"{recipe.recipe_id}:{c}" for c in recipe.contexts],
        row_counts=counts,
        label_delay=recipe.label_delay,
    )
    if mode == "fitted":
        case.pop("reference_policy", None)
        case["contextual_policy"] = asdict(fit)
    else:
        case["reference_policy"] = asdict(
            ReferencePolicy("drift" if mode.endswith("_drift") else mode)
        )
    case["training_work"] = dict(
        attempted_updates=n, attempted_rows=sum(counts[:n]), returned_updates=n
    )
    case["budget"]["updates"] = n
    case["native_calls"] = case["native_calls"][:n]
    case["applied_updates"] = case["applied_updates"][:n]
    examples, mistakes = 72, 16
    for i, (call, receipt) in enumerate(zip(case["native_calls"], case["applied_updates"])):
        call["rows"] = counts[i]
        call["state"]["completed_updates"] = 13 + i
        examples += counts[i]
        mistakes += int(receipt["diagnostic"]["value"])
        call["state"]["training_examples"] = examples
        call["state"]["mistake_updates"] = mistakes
        receipt.update(
            observed_at=i, arrived_at=i + recipe.label_delay, applied_at=i + recipe.label_delay + 2
        )
    case["stream"].update(completed_updates=n, pending_unapplied=6 - n - ack)
    if n + ack < 6:
        for key in ("guard_report", "drain_publication", "promotion_receipt"):
            case.pop(key, None)
    if mode in ("candidate_drift", "acknowledgement_drift"):
        records = [
            dict(
                complete=True,
                role="train",
                index=i,
                label_observed_at=i,
                label_arrived_at=i + recipe.label_delay,
                read_at=i + recipe.label_delay + 2,
                rows=counts[i],
                base_actor_version="qualified-12",
                candidate_version="c0",
                wake_updates=min(i, n),
                native_updates=12 + min(i, n),
                mistakes=int(i < n),
            )
            for i in range(n + ack)
        ]
        rows = sum(p["rows"] for p in records)
        case["candidate_feedback"] = dict(
            row_limit=144,
            attempts=len(records),
            returns=len(records),
            rows_attempted=rows,
            rows_returned=rows,
            records=records,
        )
    if ack:
        case["acknowledged_experiences"] = [
            dict(
                sample_id=f"batch-{i}",
                episode_id="attribution",
                event_id=f"label-{i}",
                actor_version="qualified-12",
                learner_version="c0",
                observed_at=i,
                arrived_at=i + recipe.label_delay,
                acknowledged_at=i + recipe.label_delay + 2,
                wake_updates=n,
                candidate_revision=i + 1,
            )
            for i in range(n, 6)
        ]
        case["receipt_progress"] = dict(
            actual_updates=n, acknowledged=ack, consumed=6, total=6, complete=True
        )
    return case


def test_should_freeze_complementary_orders_delays_and_keep_original_protocol():
    original = study.protocol_snapshot()
    p = study.protocol_snapshot(withheld=True)
    assert (
        len(original["recipes"]) == 2
        and original["definition"] == "context_stream_comparison_protocol_v1"
    )
    assert tuple(RECIPES[0].contexts) == (
        "old-v1",
        "new-v1",
        "new-v1",
        "old-v1",
        "old-v1",
        "new-v1",
    )
    assert tuple(RECIPES[1].contexts) == (
        "new-v1",
        "old-v1",
        "old-v1",
        "new-v1",
        "new-v1",
        "old-v1",
    )
    assert [r.label_delay for r in RECIPES] == [1, 2]
    assert p["case_count"] == 14 and p["missing_order"] == study.WITHHELD_MODES
    assert p["opportunities"][22]["scenario"] == "revoke_commit"
    assert study.protocol_snapshot() == original


@pytest.mark.parametrize("mode", study.WITHHELD_MODES)
@pytest.mark.parametrize("recipe", RECIPES)
def test_should_admit_dynamic_rows_and_separate_acknowledgements(saved, recipe, mode):
    base, fit = saved
    case = metadata_case(base, recipe, mode, fit)
    result = study.case_outcomes(case, recipe, mode, fit, withheld=True)
    assert result["rows"] == sum(p["rows"] for p in case["native_calls"])
    assert result["pending"] == (0 if mode == "acknowledgement_drift" else 6 - result["updates"])


@pytest.mark.parametrize(
    "fault",
    [
        "clock",
        "id",
        "double",
        "work",
        "native",
        "exposure",
        "mistakes",
        "ack_count",
        "wake",
        "proof",
        "diagnostic_clock",
        "role",
        "units",
    ],
)
def test_should_refuse_changed_receipt_work_or_unsupported_feedback_before_utility(saved, fault):
    base, fit = saved
    recipe = RECIPES[0]
    case = metadata_case(base, recipe, "acknowledgement_drift", fit)
    row = case["acknowledged_experiences"][0]
    if fault == "clock":
        row["arrived_at"] += 1
    if fault == "id":
        row["event_id"] = "other"
    if fault == "double":
        row["sample_id"] = "batch-0"
    if fault == "work":
        case["training_work"]["attempted_rows"] = 27
    if fault == "native":
        case["native_calls"][0]["state"]["completed_updates"] = 99
    if fault == "exposure":
        case["native_calls"][0]["state"]["training_examples"] = 75
    if fault == "mistakes":
        case["native_calls"][0]["state"]["mistake_updates"] += 1
    if fault == "ack_count":
        case["receipt_progress"]["actual_updates"] = 6
    if fault == "wake":
        row["wake_updates"] = 6
    if fault == "proof":
        case["candidate_feedback"]["records"][2]["mistakes"] = 1
    if fault == "diagnostic_clock":
        case["candidate_feedback"]["records"][2]["read_at"] = 2
    if fault == "role":
        case["candidate_feedback"]["records"][0]["role"] = "final_test"
    if fault == "units":
        case["task_feedback"] = object()
    with pytest.raises(ValueError):
        study.case_outcomes(case, recipe, "acknowledgement_drift", fit, withheld=True)


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_should_refuse_non_boolean_opt_in_before_sources_or_scoring(flag):
    with pytest.raises(ValueError, match="exact boolean"):
        study.protocol_snapshot(withheld=flag)
    with pytest.raises(ValueError, match="exact boolean"):
        study.score_cases(None, None, withheld=flag)
    with pytest.raises(ValueError, match="exact boolean"):
        study.run_comparison(None, None, None, withheld=flag)


def test_should_require_all_fourteen_before_scoring(saved):
    base, fit = saved
    rows = [
        dict(recipe_id=r.recipe_id, mode=m, result=metadata_case(base, r, m, fit))
        for r in RECIPES
        for m in study.WITHHELD_MODES
    ]
    assert len(study.score_cases(rows, fit, withheld=True)) == 14
    with pytest.raises(ValueError, match="all14"):
        study.score_cases(rows[:-1], fit, withheld=True)
    rows[-1]["result"]["complete"] = False
    with pytest.raises(ValueError):
        study.score_cases(rows, fit, withheld=True)


def test_should_run_all_fresh_and_never_open_old_periodic_or_task_fit(saved, monkeypatch):
    base, fit = saved
    seen = []
    monkeypatch.setattr(study, "_periodic_bytes", lambda p: pytest.fail("old periodic opened"))

    def case(path, **kwargs):
        mode = (
            "fitted"
            if "contextual_policy" in kwargs
            else "acknowledgement_drift"
            if kwargs.get("acknowledge_correct")
            else "candidate_drift"
            if kwargs.get("candidate_feedback")
            else kwargs["reference_mode"]
        )
        seen.append((kwargs["stream_recipe"].recipe_id, mode))
        return metadata_case(base, kwargs["stream_recipe"], mode, fit)

    monkeypatch.setattr(live, "run_case", case)
    result = study.run_comparison(SOURCE, None, FIT, withheld=True)
    assert result["complete"], result.get("error")
    assert seen == [(r.recipe_id, m) for r in RECIPES for m in study.WITHHELD_MODES]
    assert len(result["scores"]) == len(result["cases"]) == 14
    assert "periodic_sha256" not in result and "reused_periodic_work" not in result


@pytest.mark.parametrize("replace", [False, True])
def test_should_refuse_prior_paid_changes_and_stop_without_score(saved, replace, monkeypatch):
    base, fit = saved
    retained: list[dict[str, Any]] = []
    seal = study.paid_report_guard

    def guard(row):
        retained.append(row)
        return seal(row)

    monkeypatch.setattr(study, "paid_report_guard", guard)
    seen = []

    def case(path, **kwargs):
        seen.append(1)
        if retained:
            if replace:
                retained[0]["result"]["training_work"] = deepcopy(
                    retained[0]["result"]["training_work"]
                )
            else:
                retained[0]["result"]["training_work"]["attempted_rows"] = 999
        return metadata_case(base, kwargs["stream_recipe"], kwargs["reference_mode"], fit)

    monkeypatch.setattr(live, "run_case", case)
    result = study.run_comparison(SOURCE, None, FIT, withheld=True)
    assert not result["complete"] and len(seen) == 2 and result["scores"] == []
    assert result["cases"][0]["result"]["training_work"]["attempted_rows"] == 0


def test_should_stop_after_first_failed_fresh_case(monkeypatch):
    seen = []

    def case(*args, **kwargs):
        seen.append(1)
        return dict(complete=False, error="lost return")

    monkeypatch.setattr(live, "run_case", case)
    result = study.run_comparison(SOURCE, None, FIT, withheld=True)
    assert not result["complete"] and len(seen) == len(result["cases"]) == 1
    assert result["scores"] == []


def test_should_preview_and_dispatch_explicit_cli_without_periodic_source(tmp_path, monkeypatch):
    output = tmp_path / "preview.json"
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--context-stream-protocol",
                "--context-withheld-orders",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_bytes())["case_count"] == 14
    seen = []

    def comparison(*paths, **kwargs):
        seen.append((paths, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(study, "run_comparison", comparison)
    out = tmp_path / "case.json"
    assert (
        cli.main(
            [
                "--source",
                str(SOURCE),
                "--context-stream-comparison",
                "--context-withheld-orders",
                "--fitted-report",
                str(FIT),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert seen == [((SOURCE, None, FIT), dict(withheld=True))]
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--source",
                "unused",
                "--context-withheld-orders",
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
