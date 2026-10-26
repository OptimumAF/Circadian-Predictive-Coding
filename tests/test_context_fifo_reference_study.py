"""Fixed analysis/procedure on scalar reports; no real models or teachers."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import context_fifo_reference_study as study
from scripts.context_stream_comparison import _receipt_work
from scripts.context_training_streams import CONTEXT_TRAINING_RECIPES
from scripts.ticket_serving_timeline import serving_preview
from src.core.controller_objective import ControllerObjective
from src.core.reference_controller import ReferencePolicy


def receipt_data(recipe, applied=None):
    applied = set(range(len(recipe.contexts))) if applied is None else set(applied)
    calls, updates, acknowledged = [], [], []
    examples, native = 72, 0
    for i, context in enumerate(recipe.contexts):
        row = dict(
            sample_id=f"batch-{i}",
            episode_id="attribution",
            event_id=f"label-{i}",
            actor_version="qualified-12",
            learner_version="c0",
            observed_at=i,
            arrived_at=i + recipe.label_delay,
        )
        if i in applied:
            native += 1
            n = 3 if context == "new-v1" else 6
            examples += n
            updates.append(
                dict(
                    **row,
                    applied_at=i + recipe.label_delay,
                    update_number=native,
                    diagnostic=dict(value=0.0),
                )
            )
            calls.append(
                dict(
                    rows=n,
                    complete=True,
                    state=dict(
                        completed_updates=12 + native,
                        training_examples=examples,
                        mistake_updates=16,
                    ),
                )
            )
        else:
            acknowledged.append(
                dict(
                    **row,
                    acknowledged_at=i + recipe.label_delay,
                    wake_updates=native,
                    candidate_revision=i,
                )
            )
    return calls, updates, acknowledged


def case(mode):
    rows = []
    for task in serving_preview(study.live.recipe_plan()):
        success = task["scenario"] != "revoke_commit" and (
            task["policy_version"],
            task["concept"],
        ) != ("v1", "legacy")
        rows.append(
            dict(
                **task,
                role="live_service",
                complete=True,
                requests=[dict(complete=True)],
                episode=dict(complete=True, success=success, scenario=task["scenario"], steps=[]),
            )
        )
    applied = (
        []
        if mode == "no_learning"
        else list(range(8))
        if mode == "periodic"
        else [3, 6]
        if mode == "candidate_ack"
        else [0, 1]
    )
    calls, updates, acknowledged = receipt_data(study.RECIPE, applied)
    if mode != "candidate_ack":
        acknowledged = []
    n, a = len(updates), len(acknowledged)
    result = dict(
        complete=True,
        definition="live_context_recipe_v1",
        source_sha256=study.HEAD_SOURCE_SHA,
        source_unchanged=True,
        context_recipe=asdict(study.RECIPE),
        live_plan=asdict(study.live.recipe_plan()),
        promotion_policy=asdict(study.live.policy()),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        reference_policy=asdict(
            ReferencePolicy("drift" if mode in ("candidate_ack", "arrived_shift") else mode)
        ),
        live_serving=dict(records=rows, closed=True, stopped=False, unreturned_work_unknown=False),
        action_snapshot=dict(progress=dict(unreturned_work_unknown=False)),
        training_work=dict(
            attempted_updates=n, returned_updates=n, attempted_rows=sum(c["rows"] for c in calls)
        ),
        native_calls=calls,
        applied_updates=updates,
        budget=dict(updates=n, limits=dict(max_training_updates=8)),
        stream=dict(
            pending_unapplied=8 - n - a, arrived_labels=8, opportunities=24, score_forward_calls=0
        ),
        training_stream=dict(
            contexts=list(study.RECIPE.contexts), label_delay=2, row_counts=[6, 6, 6, 3, 3, 3, 6, 6]
        ),
        episodes=[],
        elapsed_seconds=1.0,
        memory=dict(peak_bytes=1024),
    )
    if mode in ("candidate_ack", "arrived_shift"):
        records = []
        for i, c in enumerate(study.RECIPE.contexts):
            prior = sum(x < i for x in applied)
            records.append(
                dict(
                    index=i,
                    role="train",
                    rows=3 if c == "new-v1" else 6,
                    complete=True,
                    label_observed_at=i,
                    label_arrived_at=i + 2,
                    read_at=i + 2,
                    wake_updates=prior,
                    native_updates=12 + prior,
                    base_actor_version="qualified-12",
                    candidate_version="c0",
                )
            )
        result["candidate_feedback" if mode == "candidate_ack" else "arrived_shift_feedback"] = (
            dict(attempts=8, returns=8, rows_attempted=39, rows_returned=39, records=records)
        )
    if mode == "candidate_ack":
        result["acknowledged_experiences"] = acknowledged
        result["receipt_progress"] = dict(
            actual_updates=n, acknowledged=a, consumed=n + a, total=8, complete=True
        )
    return result


def test_should_reuse_accepted100_metadata_without_model_reconstruction():
    saved = json.loads(
        Path("artifacts/runs/arrived-candidate-shift-20261023/actual.json").read_text(
            encoding="utf8"
        )
    )
    outcome = study.case_outcome(saved, "arrived_shift")
    assert outcome["updates"] == 2 and outcome["rows"] == 12 and outcome["pending"] == 6
    assert outcome["live_successes"] == 15 and outcome["old"] == 4 and outcome["new"] == 0


def test_should_score_all_methods_with_original_window_weights_and_directions():
    cases = {m: case(m) for m in study.MODES}
    cases["periodic"]["live_serving"]["records"][21]["episode"]["success"] = True
    scores = study.score_cases(cases)
    assert [r["mode"] for r in scores] == list(study.MODES)
    assert asdict(ControllerObjective())["new_gain_weight"] == 1
    assert scores[1]["utility"]["normalized_new_gain"] == 0.5
    assert scores[1]["outcomes"]["new"] == 1  # Revoked commit22 remains a failure.
    assert scores[1]["utility"]["value"] > scores[0]["utility"]["value"]
    assert scores[2]["outcomes"]["acknowledged"] == 6
    assert scores[2]["outcomes"]["rows"] == 9


@pytest.mark.parametrize(
    "fault", ["partial", "role", "scenario", "request", "unknown", "recipe", "source"]
)
def test_should_refuse_changed_or_incomplete_task_evidence_before_utility(fault):
    c = case("no_learning")
    row = c["live_serving"]["records"][22]
    if fault == "partial":
        c["complete"] = False
    if fault == "role":
        row["role"] = "final_test"
    if fault == "scenario":
        row["scenario"] = "ordinary"
    if fault == "request":
        row["requests"][0]["complete"] = False
    if fault == "unknown":
        c["action_snapshot"]["progress"]["unreturned_work_unknown"] = True
    if fault == "recipe":
        c["context_recipe"]["label_delay"] = 1
    if fault == "source":
        c["source_sha256"] = "changed"
    with pytest.raises(ValueError):
        study.case_outcome(c, "no_learning")


@pytest.mark.parametrize("recipe", [CONTEXT_TRAINING_RECIPES[0], study.RECIPE])
def test_should_bound_receipt_ids_by_declared_six_or_eight_packet_recipe(recipe):
    calls, updates, _ = receipt_data(recipe)
    c = dict(native_calls=calls, applied_updates=updates)
    rows, a = _receipt_work(c, recipe)
    assert rows == sum(3 if v == "new-v1" else 6 for v in recipe.contexts) and a == 0
    c["applied_updates"][-1]["sample_id"] = f"batch-{len(recipe.contexts)}"
    with pytest.raises(ValueError, match="identity"):
        _receipt_work(c, recipe)


@pytest.mark.parametrize(
    "fault",
    [
        "counter",
        "prefix",
        "duplicate",
        "native",
        "ack_progress",
        "diagnostic_clock",
        "diagnostic_rows",
    ],
)
def test_should_refuse_invented_native_or_acknowledgement_work(fault):
    c = case("candidate_ack")
    if fault == "counter":
        c["training_work"]["attempted_rows"] += 1
    if fault == "prefix":
        c["acknowledged_experiences"].pop(0)
    if fault == "duplicate":
        c["acknowledged_experiences"].append(deepcopy(c["acknowledged_experiences"][0]))
    if fault == "native":
        c["native_calls"][0]["state"]["completed_updates"] += 1
    if fault == "ack_progress":
        c["receipt_progress"]["actual_updates"] += 1
    if fault == "diagnostic_clock":
        c["candidate_feedback"]["records"][0].update(wake_updates=1, native_updates=13)
    if fault == "diagnostic_rows":
        c["candidate_feedback"]["records"][0]["rows"] += 1
        c["candidate_feedback"]["rows_attempted"] += 1
        c["candidate_feedback"]["rows_returned"] += 1
    with pytest.raises(ValueError):
        study.case_outcome(c, "candidate_ack")


@pytest.mark.parametrize("missing", study.MODES)
def test_should_require_all_methods_before_any_scoring(missing):
    with pytest.raises(ValueError, match="all four"):
        study.score_cases({m: case(m) for m in study.MODES if m != missing})


def bind_stubs(tmp_path, monkeypatch, callback):
    raw = json.dumps(case("arrived_shift")).encode()
    path = tmp_path / "saved.json"
    path.write_bytes(raw)
    monkeypatch.setattr(study, "REUSED_SHA", sha256(raw).hexdigest())
    monkeypatch.setattr(study.live, "run_case", callback)
    return Path("examples/context_head_source/source.json"), path


def test_should_finish_three_missing_controls_before_scoring_and_reuse_shift(tmp_path, monkeypatch):
    events = []

    def run(path, **kwargs):
        mode = "candidate_ack" if kwargs.get("acknowledge_correct") else kwargs["reference_mode"]
        events.append(mode)
        assert kwargs["stream_recipe"] == study.RECIPE
        return case(mode)

    source, saved = bind_stubs(tmp_path, monkeypatch, run)
    original = study.score_cases

    def score(cases):
        assert events == list(study.MISSING)
        return original(cases)

    monkeypatch.setattr(study, "score_cases", score)
    r = study.run_comparison(source, saved)
    assert r["complete"] and len(r["scores"]) == 4 and not r["pending_modes"]
    assert events == list(study.MISSING)


@pytest.mark.parametrize("failure", study.MISSING)
def test_should_preserve_failed_paid_prefix_and_stop_all_following_work(
    tmp_path, monkeypatch, failure
):
    events = []

    def run(path, **kwargs):
        mode = "candidate_ack" if kwargs.get("acknowledge_correct") else kwargs["reference_mode"]
        events.append(mode)
        c = case(mode)
        if mode == failure:
            c["complete"] = False
        return c

    source, saved = bind_stubs(tmp_path, monkeypatch, run)
    r = study.run_comparison(source, saved)
    assert not r["complete"] and not r["scores"]
    assert events == list(study.MISSING[: study.MISSING.index(failure) + 1])
    assert failure in r["cases"] and not r["cases"][failure]["complete"]


def test_should_restore_corrupted_preceding_paid_evidence_and_stop(tmp_path, monkeypatch):
    owned: list[dict[str, Any]] = []

    def run(path, **kwargs):
        c = case(kwargs["reference_mode"])
        if owned:
            owned[0]["elapsed_seconds"] = 999
        owned.append(c)
        return c

    source, saved = bind_stubs(tmp_path, monkeypatch, run)
    r = study.run_comparison(source, saved)
    assert not r["complete"] and not r["scores"] and len(owned) == 2
    assert r["cases"]["no_learning"]["elapsed_seconds"] == 1.0


def test_should_refuse_replaced_saved_evidence_before_any_new_case(tmp_path, monkeypatch):
    source, saved = bind_stubs(tmp_path, monkeypatch, lambda *a, **k: pytest.fail("model entered"))
    saved.write_text("{}", encoding="utf8")
    r = study.run_comparison(source, saved)
    assert not r["complete"] and not r["cases"] and not r["scores"]
