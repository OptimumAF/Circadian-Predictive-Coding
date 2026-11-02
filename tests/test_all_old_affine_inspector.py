"""Literal all-old reports: metadata/trace consistency, never learned outcomes."""

from copy import deepcopy
import json
from typing import Any

import pytest

from src.app import minigrid_private_affine_inspector as private
from src.app.key_door_inspector import project_key_door_report
from src.app.minigrid_affine_inspector import LAYOUTS, _native_costs
from src.infra import key_door_inspector_files as files
from test_minigrid_inspector_projection import pointer
from test_private_affine_inspector import native_episode, report as old_report


def work(phase):
    rows = phase["episodes"]
    envs = [r["environment"] for r in rows if r.get("environment") is not None]
    phase["work"].update(
        live_prediction_calls_attempted=sum(r.get("prediction_calls_attempted", 0) for r in rows),
        live_prediction_calls_returned=sum(r.get("prediction_calls_returned", 0) for r in rows),
        environment_steps_attempted=sum(e["environment_steps_attempted"] for e in envs),
        environment_steps_returned=sum(e["environment_steps"] for e in envs),
        environment_resets_attempted=len(envs),
        environment_resets_completed=len(envs),
    )


def report():
    r = old_report()
    r["definition"] = private.ALL_OLD_DEFINITION
    r["protocol"].update(definition=private.ALL_OLD_DEFINITION, all_old_tasks=True)
    r["protocol"]["games"] = dict(
        definition="minigrid_private_rehearsal_F_then_all_ABC_games_v1",
        phases=list(private.ARMS),
        contexts=[context for context, _, _ in private.ALL_OLD_SLOTS],
        roles=[role for _, _, role in private.ALL_OLD_SLOTS],
        games_per_phase=8,
        seed=17,
        max_steps_per_phase=128,
    )
    for phase in r["games"]:
        phase.update(role="private_F_then_all_old_ABC", include_old_tasks=True)
        for context, cue, role in private.ALL_OLD_SLOTS[4:]:
            row = native_episode(context, cue, role)
            row["environment"]["layout"] = deepcopy(LAYOUTS[context])
            phase["episodes"].append(row)
        work(phase)
    return r


def test_should_show_both_full_rosters_exact_layouts_pointers_and_detached_source():
    r = report()
    before = deepcopy(r)
    view = project_key_door_report(r, source_label="all-old.json")
    assert view["all_old_tasks"] is True and view["source"] == r == before
    for arm in view["conditions"][1:]:
        episodes = arm["episodes"]
        assert [(e["context"], e["cue"], e["record"]["source_role"]) for e in episodes] == list(
            private.ALL_OLD_SLOTS
        )
        assert [e["phase"] for e in episodes] == [
            "Fresh F",
            "Fresh F",
            "Old A",
            "Old A",
            "Old B",
            "Old B",
            "Old C",
            "Old C",
        ]
        for game in episodes:
            assert pointer(view["source"], game["pointer"]) == game["record"]
            for frame in game["frames"]:
                if frame["decision_pointer"] is not None:
                    assert pointer(view["source"], frame["decision_pointer"]) is not None
        for item in arm["costs"] + arm["proofs"]:
            assert pointer(view["source"], item["pointer"]) == item["value"]
    r["games"][0]["episodes"][4]["environment"]["layout"]["rotation"] = 3
    assert view["source"] == before


def test_should_keep_old_four_game_shape_and_omit_new_marker():
    r = old_report()
    view = project_key_door_report(r, source_label="old.json")
    assert "all_old_tasks" not in view
    assert all(len(c["episodes"]) == 4 for c in view["conditions"][1:])
    assert [e["phase"] for e in view["conditions"][1]["episodes"]] == [
        "Fresh F",
        "Fresh F",
        "Returning A",
        "Returning A",
    ]


@pytest.mark.parametrize("omit_outcomes", [False, True])
def test_should_preserve_failed_B_and_unobserved_C_without_invented_proofs(omit_outcomes):
    r = report()
    r.update(status="failed", TRAIN_reference_unchanged=None)
    phase = r["games"][1]
    phase.update(status="failed", reason="readback failed", state_unchanged=False)
    for i in range(4, 8):
        context, cue, role = private.ALL_OLD_SLOTS[i]
        row: dict[str, Any] = dict(
            context=context,
            cue=cue,
            source_role=role,
            status="failed" if i == 4 else "not_run",
            predictions=[],
            environment=None,
            prediction_calls_attempted=0,
            prediction_calls_returned=0,
            state_unchanged=False if i == 4 else None,
            success=None,
            **{"return": None},
        )
        if omit_outcomes:
            row.pop("success")
            row.pop("return")
        phase["episodes"][i] = row
    work(phase)
    phase["work"]["failure_work_may_be_unobserved"] = True
    view = project_key_door_report(r, source_label="partial.json")
    games = view["conditions"][2]["episodes"]
    assert [g["record"]["status"] for g in games[4:]] == ["failed", "not_run", "not_run", "not_run"]
    assert all(not g["frames"] and g["record"].get("success") is None for g in games[4:])
    assert view["source"]["TRAIN_reference_unchanged"] is None
    assert any(p["value"] is False for p in view["conditions"][2]["proofs"])
    if omit_outcomes:
        assert all("success" not in g["record"] and "return" not in g["record"] for g in games[4:])


@pytest.mark.parametrize(
    "fault",
    [
        "old_definition",
        "phase_flag",
        "phase_flag_type",
        "phase_role",
        "protocol_flag",
        "protocol_definition",
        "missing",
        "extra",
        "order",
        "context",
        "role",
        "cue",
        "rotation",
        "swap",
        "wall",
        "branch",
        "reset_count",
        "steps",
        "fit_rows",
        "after_failed",
        "false_complete_proof",
        "plan_roster",
        "plan_roles",
        "plan_seed",
        "plan_definition",
        "plan_count",
    ],
)
def test_should_refuse_mixed_rosters_substituted_contexts_and_corrupt_evidence(fault):
    r = report()
    phase = r["games"][0]
    row = phase["episodes"][4]
    if fault == "old_definition":
        r["definition"] = private.DEFINITION
    elif fault == "phase_flag":
        phase["include_old_tasks"] = False
    elif fault == "phase_flag_type":
        phase["include_old_tasks"] = 1
    elif fault == "phase_role":
        phase["role"] = "private_F_then_returned_A"
    elif fault == "protocol_flag":
        r["protocol"]["all_old_tasks"] = False
    elif fault == "protocol_definition":
        r["protocol"]["definition"] = private.DEFINITION
    elif fault.startswith("plan_"):
        key, value = {
            "plan_roster": ("contexts", ["F-blue"] * 8),
            "plan_roles": ("roles", ["train"] * 8),
            "plan_seed": ("seed", 18),
            "plan_definition": ("definition", "wrong"),
            "plan_count": ("games_per_phase", 4),
        }[fault]
        r["protocol"]["games"][key] = value
    elif fault == "missing":
        phase["episodes"].pop()
    elif fault == "extra":
        phase["episodes"].append(deepcopy(row))
    elif fault == "order":
        phase["episodes"][4:6] = reversed(phase["episodes"][4:6])
    elif fault == "context":
        row["context"] = "A"
    elif fault == "role":
        row["source_role"] = "probe_test"
    elif fault == "cue":
        row["cue"] = "red"
    elif fault in ("rotation", "swap", "wall", "branch"):
        key, value = {
            "rotation": ("rotation", 0),
            "swap": ("swap_keys", False),
            "wall": ("wall_color", "grey"),
            "branch": ("branch_row", 3),
        }[fault]
        row["environment"]["layout"][key] = value
    elif fault == "reset_count":
        phase["work"]["environment_resets_completed"] = 4
    elif fault == "steps":
        phase["work"]["environment_steps_attempted"] = 129
    elif fault == "fit_rows":
        phase["source_fit_rows"] = 126
    elif fault == "after_failed":
        r["status"] = phase["status"] = "failed"
        phase["episodes"][3]["status"] = "failed"
    else:
        phase["state_unchanged"] = False
    with pytest.raises(ValueError):
        project_key_door_report(r, source_label="bad.json")


@pytest.mark.parametrize("count", [True, 0, 5, 16])
def test_should_not_offer_unbounded_roster_cost_admission(count):
    with pytest.raises(ValueError, match="roster"):
        _native_costs({}, [], episode_limit=count)


@pytest.mark.parametrize("count", [4, 8])
def test_should_keep_each_rosters_original_step_ceiling_on_partial_work(count):
    phase = (old_report() if count == 4 else report())["games"][0]
    phase["status"] = "failed"
    phase["work"]["failure_work_may_be_unobserved"] = True
    phase["work"]["environment_steps_attempted"] = 16 * count

    def costs():
        if count == 4:
            return _native_costs(phase, phase["episodes"])
        return _native_costs(phase, phase["episodes"], episode_limit=8)

    assert costs()
    phase["work"]["environment_steps_attempted"] += 1
    with pytest.raises(ValueError):
        costs()


def test_should_keep_missing_all_old_phases_unknown():
    view = project_key_door_report(
        dict(definition=private.ALL_OLD_DEFINITION, status="failed"),
        source_label="unrun.json",
    )
    for arm in view["conditions"][1:]:
        assert len(arm["episodes"]) == 8
        assert all(
            e["pointer"] is None and e["derived"] and not e["frames"] for e in arm["episodes"]
        )


def test_should_write_the_new_mode_through_existing_inert_exclusive_export(tmp_path):
    source, output = tmp_path / "source.json", tmp_path / "all-old.html"
    r = report()
    source.write_text(json.dumps(r), encoding="utf8")
    original = source.read_bytes()
    files.write_key_door_inspector(source, output)
    page = output.read_text(encoding="utf8")
    assert '"all_old_tasks":true' in page and "Private learning and old-room retention" in page
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, output)
