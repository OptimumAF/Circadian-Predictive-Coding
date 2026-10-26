"""Literal recorded comparisons; no trained models or physical outcomes produced."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from src.app import minigrid_private_affine_inspector as private
from src.app.key_door_inspector import project_key_door_report
from src.infra import key_door_inspector_files as files
from test_affine_workflow_inspector import complete_report as normal_report, episode
from test_minigrid_inspector_projection import pointer

ROOT = Path(__file__).resolve().parents[1]
ACTUAL = ROOT / "artifacts/runs/portable-affine-adaptation-20261024/actual.json"


def budget(updates, replay, wall=90):
    limits = dict(
        max_training_updates=updates,
        max_replay_examples=replay,
        max_wall_seconds=wall,
        max_process_rss_bytes=512 * 1024**2,
    )
    return dict(
        initial_limits=deepcopy(limits),
        limits=limits,
        updates_completed=updates,
        replay_examples_completed=replay,
        started_at=10.0,
        observed_budget_started_at=10.0,
        ended_at=11.0,
    )


def native_episode(context, cue, role):
    row = episode("B", cue)
    row.update(context=context, source_role=role)
    row["environment"].update(
        context=context,
        role=role,
        episode_id=f"live-{private.METHOD}-8-{context}-{cue}",
        layout=dict(
            context=context,
            rotation=0,
            branch_row=2 if context.startswith("F-") else 3,
            swap_keys=False,
            wall_color="grey",
        ),
    )
    return row


def report():
    original = normal_report()
    arms, games = [], []
    for index, name in enumerate(private.ARMS):
        replay = 54 * index
        old = []
        for update in range(3):
            correct = 9 if index else 9 - update
            old.append(
                dict(
                    updates=update,
                    contexts={
                        identity: dict(
                            rows=9,
                            correct=correct,
                            ce=1.0 + update / 10,
                            actions=[0] * correct + [1] * (9 - correct),
                            expected=[0] * 9,
                        )
                        for identity, _, _ in private.SOURCES
                    },
                )
            )
        arms.append(
            dict(
                arm=name,
                status="complete",
                reason=None,
                source_fork_calls=1,
                source_native_state_unchanged=True,
                budget=budget(2, replay),
                updates_accounted_by_study=2,
                committed_old_TRAIN_reuse_rows=replay,
                committed_extra_F_rows=54 - replay,
                attempts=[
                    dict(
                        update=n,
                        fit_rows_attempted=36,
                        old_TRAIN_reuse_charged=27 * index,
                        extra_F_exposures_charged=27 * (1 - index),
                    )
                    for n in (1, 2)
                ],
                plan=dict(input_array_bytes=171072, storage_scope="fit matrices only"),
                native_endpoint=dict(
                    completed_updates=8,
                    training_examples=180,
                    failed_work=None,
                    heads=[dict(fit_rows=36), dict(fit_rows=36)],
                ),
                old_task_readouts=old,
                probe=dict(
                    completed_updates=2,
                    completed_training_examples=72,
                    checkpoints=[
                        dict(
                            updates=n,
                            scores={
                                f"F-{cue}": dict(ce=1.5 - n / 10, accuracy=(6 + n) / 9)
                                for cue in ("blue", "red")
                            },
                        )
                        for n in range(3)
                    ],
                ),
            )
        )
        phase = deepcopy(original["games"])
        phase.update(
            role="private_F_then_returned_A",
            rehearsal_arm=name,
            updates=8,
            source_fit_calls=8,
            source_fit_rows=180,
            episodes=[native_episode(c, q, r) for c, q, r in private.SLOTS],
        )
        games.append(phase)
    return dict(
        definition=private.DEFINITION,
        status="complete",
        reason=None,
        protocol=dict(new_fit_calls=10, new_fit_rows=252, historical_source_reads=0),
        TRAIN_reference_unchanged=True,
        training=original["training"],
        collection=[
            dict(
                role=role,
                cue=cue,
                context=f"F-{cue}",
                status="complete",
                closed=True,
                environment_steps=1,
            )
            for role in ("probe_train", "probe_test")
            for cue in ("blue", "red")
        ],
        private_comparison=dict(
            definition="minigrid_private_affine_rehearsal_comparison_v1",
            status="complete",
            reason=None,
            protocol=dict(
                upstream_paid_fits=6,
                upstream_paid_rows=108,
                source_native_fits=6,
                source_native_rows=108,
            ),
            arms=arms,
            budget=budget(4, 54),
        ),
        games=games,
        readout_budget=budget(0, 0, 150),
        workflow_budget=dict(
            max_wall_seconds=180,
            work_deadline_seconds=170,
            max_process_rss_bytes=512 * 1024**2,
            elapsed_seconds=2.0,
            sampled_peak_rss_bytes=123456,
            scope="workflow only, external imports excluded",
        ),
    )


def partial_report():
    r = report()
    r.update(status="failed", reason="170s deadline")
    phase = r["games"][1]
    phase.update(status="failed", reason="deadline", state_unchanged=False)
    for index in range(1, 4):
        c, q, role = private.SLOTS[index]
        phase["episodes"][index] = dict(
            context=c,
            cue=q,
            source_role=role,
            status="failed" if index == 1 else "not_run",
            reason="deadline" if index == 1 else "not reached",
            predictions=[],
            prediction_calls_attempted=0,
            prediction_calls_returned=0,
            environment=None,
            success=None,
            state_unchanged=False if index == 1 else None,
            **{"return": None},
        )
    phase["work"].update(
        live_prediction_calls_attempted=1,
        live_prediction_calls_returned=1,
        environment_steps_attempted=1,
        environment_steps_returned=1,
        environment_resets_attempted=1,
        environment_resets_completed=1,
        failure_work_may_be_unobserved=True,
    )
    return r


def test_should_present_both_arms_old_new_curves_costs_and_original_source_pointers():
    r = report()
    before = deepcopy(r)
    view = project_key_door_report(r, source_label="literal.json")
    assert view["schema_id"] == "minigrid_private_affine_inspector_v1"
    assert view["source"] == r == before
    assert [c["key"] for c in view["conditions"]] == ["TRAIN reference", *private.LABELS]
    assert all(len(c["curves"]) == 24 for c in view["conditions"])
    for c in view["conditions"]:
        for row in c["curves"]:
            saved = pointer(view["source"], row["pointer"])
            assert row["ce"] == saved["ce"]
            assert row["accuracy"] == (
                saved["correct"] / 9 if row.get("derived_accuracy") else saved["accuracy"]
            )
        for item in c["costs"] + c["proofs"]:
            assert pointer(view["source"], item["pointer"]) == item["value"]
        for game in c["episodes"]:
            assert pointer(view["source"], game["pointer"]) == game["record"]
            assert len(game["frames"]) == 2 and game["frames"][-1]["decision_pointer"] is None
    for item in view["global_costs"]:
        assert pointer(view["source"], item["pointer"]) == item["value"]
    r["private_comparison"]["arms"][0]["attempts"][0]["fit_rows_attempted"] = 1
    assert view["source"] == before


@pytest.mark.parametrize("omitted_outcomes", [False, True])
def test_should_preserve_partial_failed_and_unrun_slots_and_false_invariance(omitted_outcomes):
    recorded = partial_report()
    if omitted_outcomes:
        for key in ("success", "return"):
            recorded["games"][1]["episodes"][1].pop(key)
    view = project_key_door_report(recorded, source_label="partial.json")
    joint = view["conditions"][2]
    assert joint["status"] == "complete" and joint["native_status"] == "failed"
    assert [e["record"]["status"] for e in joint["episodes"]] == [
        "complete",
        "failed",
        "not_run",
        "not_run",
    ]
    assert [len(e["frames"]) for e in joint["episodes"]] == [2, 0, 0, 0]
    assert all(e["record"].get("success") is None for e in joint["episodes"][1:])
    if omitted_outcomes:
        assert "success" not in joint["episodes"][1]["record"]
        assert "return" not in joint["episodes"][1]["record"]
    assert any(
        p["value"] is False and p["pointer"] == "/games/1/state_unchanged" for p in joint["proofs"]
    )
    assert any(
        c["value"] is True and c["pointer"].endswith("failure_work_may_be_unobserved")
        for c in joint["costs"]
    )


def test_should_keep_failed_fit_charges_and_later_control_unknown():
    r = partial_report()
    r["games"] = None
    comparison = r["private_comparison"]
    comparison["status"] = "failed"
    comparison["arms"] = comparison["arms"][:1]
    arm = comparison["arms"][0]
    arm["status"] = "failed"
    arm["probe"]["checkpoints"] = []
    arm["old_task_readouts"] = []
    view = project_key_door_report(r, source_label="paid-fit-failure.json")
    first, second = view["conditions"][1:]
    assert first["status"] == "failed" and second["status"] == "not_run"
    assert all(row["accuracy"] is None for row in first["curves"] + second["curves"])
    charges = [c for c in first["costs"] if c["pointer"].endswith("extra_F_exposures_charged")]
    assert [c["value"] for c in charges] == [27, 27]
    assert all(e["pointer"] is None for e in second["episodes"])


def test_should_preserve_worse_confidence_without_selecting_a_winner():
    r = partial_report()
    r["private_comparison"]["arms"][1]["probe"]["checkpoints"][2]["scores"]["F-blue"]["ce"] = 2.0
    view = project_key_door_report(r, source_label="tradeoff.json")
    values = [
        next(c["ce"] for c in arm["curves"] if c["context"] == "F-blue" and c["updates"] == 2)
        for arm in view["conditions"][1:]
    ]
    assert values == [1.3, 2.0] and len(view["conditions"]) == 3


def test_should_keep_missing_controls_and_costs_unknown_without_fabricated_sources():
    view = project_key_door_report(
        dict(definition=private.DEFINITION, status="failed", reason="missing runtime"),
        source_label="empty.json",
    )
    assert len(view["conditions"]) == 3 and view["global_costs"] == []
    for c in view["conditions"]:
        assert c["status"] == "not_run"
        assert c["pointer"] is None
        assert all(v["accuracy"] is v["ce"] is v["pointer"] is None for v in c["curves"])
        assert all(e["derived"] and e["pointer"] is None and not e["frames"] for e in c["episodes"])


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "geometry",
        "action",
        "after_failed",
        "clock",
        "attempt",
        "head",
        "lineage",
        "order",
        "point",
        "correct",
        "training",
        "complete",
        "resource",
        "study_clock",
        "collection",
        "point_failure",
    ],
)
def test_should_refuse_inconsistent_or_relabelled_saved_evidence(fault):
    r = partial_report()
    a = r["private_comparison"]["arms"][0]
    e = r["games"][0]["episodes"][0]
    options: dict[str, Any] = {}
    if fault == "role":
        e["source_role"] = "train"
    elif fault == "geometry":
        e["environment"]["layout"]["branch_row"] = 4
    elif fault == "action":
        e["predictions"][0]["action"] = 6
    elif fault == "after_failed":
        r["games"][1]["episodes"][2] = deepcopy(r["games"][0]["episodes"][2])
    elif fault == "clock":
        a["budget"]["observed_budget_started_at"] += 1
    elif fault == "attempt":
        a["attempts"][0]["extra_F_exposures_charged"] = 0
    elif fault == "head":
        a["native_endpoint"]["heads"][0]["fit_rows"] = 9
    elif fault == "lineage":
        r["private_comparison"]["protocol"]["upstream_paid_fits"] = 12
    elif fault == "order":
        r["private_comparison"]["arms"].reverse()
    elif fault == "point":
        a["probe"]["checkpoints"][1]["updates"] = 2
    elif fault == "correct":
        a["old_task_readouts"][0]["contexts"][private.SOURCES[0][0]]["correct"] = 0
    elif fault == "training":
        r["training"]["status"] = "failed"
    elif fault == "complete":
        r["status"] = "complete"
    elif fault == "study_clock":
        a["budget"]["started_at"] = a["budget"]["observed_budget_started_at"] = 10.5
    elif fault == "collection":
        r["collection"][0]["closed"] = False
    elif fault == "point_failure":
        a["probe"]["checkpoints"][0]["status"] = "failed"
    else:
        options = dict(resources={}, resource_label="unverified")
    with pytest.raises(ValueError):
        project_key_door_report(r, source_label="bad.json", **options)


def test_should_render_exclusive_inert_html_with_all_private_controls(tmp_path):
    r = partial_report()
    r["reason"] = "</script><script>bad()</script>"
    source = tmp_path / "input.json"
    output = tmp_path / "view.html"
    source.write_text(json.dumps(r), encoding="utf8")
    original = source.read_bytes()
    files.write_key_door_inspector(source, output)
    page = output.read_text(encoding="utf8")
    assert (
        "minigrid_private_affine_inspector_v1" in page
        and "Private learning, retention and return" in page
    )
    assert "</script><script>bad()" not in page
    assert all(name in page for name in private.LABELS)
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, output)


def test_should_project_original_saved_partial_result_without_models_or_physics():
    if not ACTUAL.is_file():
        pytest.skip("local120 qualification report not distributed")
    raw = ACTUAL.read_bytes()
    view = project_key_door_report(json.loads(raw), source_label=ACTUAL.name)
    assert view["source"]["status"] == "failed" and ACTUAL.read_bytes() == raw
    assert len(view["conditions"][1]["episodes"]) + len(view["conditions"][2]["episodes"]) == 8
    assert [len(e["frames"]) for e in view["conditions"][2]["episodes"]] == [10, 0, 0, 0]
    assert [e["record"]["success"] for e in view["conditions"][1]["episodes"]] == [
        True,
        True,
        False,
        False,
    ]
    assert view["conditions"][2]["native_status"] == "failed"
    assert "success" not in view["conditions"][2]["episodes"][1]["record"]
    assert "return" not in view["conditions"][2]["episodes"][1]["record"]
