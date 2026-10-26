"""Saved-only projection checks; no environment execution or learner imports."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from src.app.key_door_inspector import project_key_door_report


@pytest.fixture(scope="module")
def accepted():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/r21-return-fresh-task-probes-20261016/native.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def success(accepted):
    return deepcopy(
        next(
            episode
            for episode in accepted["conditions"]["off"]["policy_evaluation"]["episodes"]
            if episode["outcome"] == "success"
        )
    )


def single(episode):
    return {
        "status": "complete",
        "protocol": {"protocol_id": "saved_only_fixture"},
        "policy_evaluation": {"episodes": [episode]},
    }


def episode_projection(report):
    return project_key_door_report(report, source_label="saved.json")["conditions"][0]["episodes"][
        0
    ]


def pointer(document, reference):
    result = document
    for segment in reference.split("/")[1:]:
        result = result[int(segment)] if type(result) is list else result[segment]
    return result


def test_should_preserve_all_accepted_conditions_episodes_and_exact_origins(accepted):
    before = deepcopy(accepted)
    result = project_key_door_report(accepted, source_label="accepted paired report")
    assert result["schema_id"] == "key_door_inspector_v1"
    assert result["origin"] == "recorded_artifact" and result["source"] == accepted
    assert [condition["key"] for condition in result["conditions"]] == ["off", "on"]
    for condition in result["conditions"]:
        assert condition["pointer"] == "/conditions/" + condition["key"]
        assert condition["unknown"] == {"topology": None, "lineage": None, "sleep_proposals": None}
        assert condition["map_origin"] == "derived_from_recorded_layout_and_applied_actions"
        assert len(condition["episodes"]) == 32
        assert condition["episodes"][0]["record"]["success"] is False
        for episode in condition["episodes"]:
            assert pointer(result["source"], episode["pointer"]) == episode["record"]
            assert len(episode["frames"]) == len(episode["record"]["transitions"]) + 1
    assert accepted == before
    result["source"]["conditions"]["off"]["status"] = "changed"
    result["conditions"][0]["episodes"][0]["record"]["status"] = "changed"
    assert accepted == before


def test_should_derive_recorded_success_position_cue_key_and_door_changes(success):
    report = single(success)
    result = episode_projection(report)
    assert result["pointer"] == "/policy_evaluation/episodes/0"
    assert result["frames"][0]["position"] == [2, 4]
    assert result["frames"][0]["visible_cue"] == 1
    assert result["frames"][1]["visible_cue"] == 0
    assert [frame["held_color"] for frame in result["frames"]] == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert [frame["door_open"] for frame in result["frames"]] == [False] * 7 + [True, True]
    assert result["frames"][-1]["position"] == [2, 0]
    assert result["frames"][-1]["remaining_keys"] == [{"position": [3, 2], "color": -1}]
    assert project_key_door_report(report, source_label="x")["conditions"][0]["pointer"] == ""


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_should_rotate_saved_external_actions_and_screen_positions_without_environment_steps(
    success, turns
):
    success["layout"]["rotation"] = turns
    for transition in success["transitions"]:
        for key in ("observation", "next_observation"):
            original = transition[key][:4]
            transition[key][:4] = [original[(index - turns) % 4] for index in range(4)]
        original = transition["probabilities"][:4]
        transition["probabilities"][:4] = [original[(index - turns) % 4] for index in range(4)]
        transition["action"] = (transition["action"] + turns) % 4 if transition["action"] < 4 else 4
        transition["legal_actions"] = sorted(
            (action + turns) % 4 if action < 4 else 4 for action in transition["legal_actions"]
        )
    result = episode_projection(single(success))
    assert result["frames"][0]["position"] == {1: [0, 2], 2: [2, 0], 3: [4, 2]}[turns]
    assert result["frames"][-1]["position"] == {1: [4, 2], 2: [2, 4], 3: [0, 2]}[turns]


def test_should_preserve_wrong_key_terminal_and_negative_return(success):
    success.update(
        cue=-1,
        outcome="wrong_key",
        success=False,
        terminal=True,
        steps_executed=7,
        prediction_calls_attempted=7,
        prediction_calls_returned=7,
    )
    success["return"] = -1.0
    success["transitions"] = success["transitions"][:7]
    success["transitions"][0]["observation"][6] = -1.0
    final = success["transitions"][-1]
    final.update(reward=-1.0, terminal=True, next_observation=final["observation"].copy())
    result = episode_projection(single(success))
    assert result["record"]["return"] == -1
    assert result["frames"][-1]["position"] == [2, 1]
    assert result["frames"][-1]["door_open"] is False


def test_should_preserve_unresolved_prediction_and_no_layout_failure_slots(accepted):
    partial = deepcopy(accepted["conditions"]["off"]["policy_evaluation"]["episodes"][0])
    partial.update(
        status="failed",
        reason="prediction failed",
        outcome=None,
        state_unchanged=False,
        prediction_calls_returned=0,
    )
    partial["transitions"][0].update(action=None, probabilities=None)
    projected = episode_projection(single(partial))
    assert projected["record"]["outcome"] is None
    assert projected["frames"][0]["position"] == projected["frames"][1]["position"]
    assert projected["frames"][1]["visible_cue"] == 1
    absent = {
        key: deepcopy(partial[key])
        for key in (
            "checkpoint_updates",
            "context",
            "cue",
            "layout_partition",
            "status",
            "reason",
            "state_unchanged",
        )
    }
    absent["layout"] = None
    assert episode_projection(single(absent))["frames"] == []
    absent["status"], absent["reason"], absent["state_unchanged"] = "not_run", "not reached", None
    assert episode_projection(single(absent))["record"]["state_unchanged"] is None


def test_should_preserve_failed_condition_without_inventing_evaluation():
    source = {
        "status": "failed",
        "protocol": {"protocol_id": "failed_before_evaluation"},
        "policy_evaluation": None,
    }
    condition = project_key_door_report(source, source_label="partial.json")["conditions"][0]
    assert condition["episodes"] == [] and condition["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_updates", True),
        ("cue", 0),
        ("context", " A"),
        ("layout_partition", "unknown"),
        ("status", []),
        ("num_classes", True),
        ("observation_dim", 7),
        ("steps_executed", 9),
        ("prediction_calls_attempted", 7),
        ("prediction_calls_returned", 6),
        ("return", 0),
        ("terminal", False),
        ("outcome", "illegal_action"),
        ("success", False),
        ("max_steps", 7),
    ],
)
def test_should_refuse_corrupt_episode_metadata_counts_and_outcomes(success, field, value):
    success[field] = value
    with pytest.raises(ValueError):
        episode_projection(single(success))


@pytest.mark.parametrize(
    "kind",
    [
        "observation",
        "next_observation",
        "legal",
        "bool_legal",
        "action",
        "tie",
        "probability",
        "sum",
        "applied",
        "reward",
        "terminal",
        "continuation",
        "extra",
    ],
)
def test_should_refuse_corrupt_saved_transition_arithmetic_and_decisions(success, kind):
    first = success["transitions"][0]
    if kind in ("observation", "next_observation"):
        first[kind][0] = -1
    elif kind == "legal":
        first["legal_actions"] = [0, 1]
    elif kind == "bool_legal":
        first["legal_actions"] = [False]
    elif kind == "action":
        first["action"] = 1
    elif kind == "tie":
        first["probabilities"] = [0.4, 0.4, 0.1, 0.1, 0]
        first["action"] = 1
    elif kind == "probability":
        first["probabilities"][0] = -0.1
    elif kind == "sum":
        first["probabilities"] = [0.9] * 5
    elif kind == "applied":
        first["applied"] = False
    elif kind == "reward":
        first["reward"] = 1
    elif kind == "terminal":
        first["terminal"] = True
    elif kind == "continuation":
        success["transitions"].append(deepcopy(success["transitions"][-1]))
    else:
        first["extra"] = 0
    with pytest.raises(ValueError):
        episode_projection(single(success))


@pytest.mark.parametrize(
    "kind", ["seed", "point", "partition", "origin", "rotation", "layout_context", "missing_layout"]
)
def test_should_refuse_identity_layout_and_origin_corruption(success, kind):
    if kind == "seed":
        success["identity"]["seed"] = True
    elif kind == "point":
        success["identity"]["checkpoint_updates"] = 64
    elif kind == "partition":
        success["identity"]["task_sequence"] = ["B"]
    elif kind == "origin":
        success["execution_origin"] = "new_simulation"
    elif kind == "rotation":
        success["layout"]["rotation"] = 4
    elif kind == "layout_context":
        success["layout"]["layout_id"] = "B"
    else:
        success.pop("layout")
    with pytest.raises(ValueError):
        episode_projection(single(success))


@pytest.mark.parametrize(
    "kind",
    ["nonfinite", "nonjson", "conditions", "slots", "transitions", "complete_without_evaluation"],
)
def test_should_refuse_nonfinite_nonjson_and_oversized_structures(success, kind):
    report: Any = single(success)
    if kind == "nonfinite":
        report["unrelated_saved_cost"] = float("nan")
    elif kind == "nonjson":
        report["unrelated_saved_cost"] = object()
    elif kind == "conditions":
        report = {
            "status": "complete",
            "conditions": {"off": report, "on": report, "third": report},
        }
    elif kind == "slots":
        report["policy_evaluation"]["episodes"] *= 129
    elif kind == "transitions":
        success["transitions"] *= 5
    else:
        report["policy_evaluation"] = None
    with pytest.raises(ValueError):
        project_key_door_report(report, source_label="bad.json")


@pytest.mark.parametrize("label", ["", " padded ", "x" * 1025, None])
def test_should_require_a_bounded_explicit_source_label(success, label):
    with pytest.raises(ValueError):
        project_key_door_report(single(success), source_label=label)


@pytest.mark.parametrize(
    "kind",
    [
        "probes",
        "probe_report",
        "probe_checkpoints",
        "probe_scores",
        "probe_metric",
        "diagnostics",
        "decisions",
        "correct",
        "training",
        "training_metric",
        "clocks",
        "outcomes",
        "by_layout",
        "outcome_metric",
        "work",
        "cost",
        "replay",
        "events",
        "selected",
    ],
)
def test_should_refuse_malformed_present_frontend_containers_and_arithmetic(success, kind):
    report = single(success)
    malformed_array: Any = []
    if kind == "probes":
        report["probes"] = {}
    elif kind.startswith("probe_") or kind in ("diagnostics", "decisions", "correct"):
        point: Any = {"updates": 0, "scores": {"E": {"ce": 1, "accuracy": 0.5}}}
        probe: Any = {"checkpoint_updates": 0, "report": {"checkpoints": [point]}}
        report["probes"] = [probe]
        if kind == "probe_report":
            probe["report"] = []
        elif kind == "probe_checkpoints":
            probe["report"]["checkpoints"] = {}
        elif kind == "probe_scores":
            point["scores"] = []
        elif kind == "probe_metric":
            point["scores"]["E"]["ce"] = "1"
        elif kind == "diagnostics":
            point["decision_diagnostics"] = []
        else:
            point["decision_diagnostics"] = {
                "E": {"decisions": "not an array" if kind == "decisions" else [{"correct": 1}]}
            }
    elif kind in ("training", "training_metric", "clocks"):
        report["checkpoints"] = (
            {}
            if kind == "training"
            else [
                {"mean_accuracy": "0.5"}
                if kind == "training_metric"
                else {"clocks": malformed_array}
            ]
        )
    elif kind in ("outcomes", "by_layout", "outcome_metric"):
        report["policy_evaluation"]["outcomes"] = (
            {}
            if kind == "outcomes"
            else [
                {"by_layout": malformed_array}
                if kind == "by_layout"
                else {"by_layout": {"A": {"success_rate": 2}}}
            ]
        )
    elif kind == "work":
        report["work"] = []
    elif kind == "cost":
        report["work"] = {"score_forwards_completed": "10"}
    elif kind == "replay":
        report["replay"] = []
    elif kind == "events":
        report["replay"] = {"rehearsal": {"events": {}}}
    else:
        report["replay"] = {"rehearsal": {"events": [{"selected_episodes": 1}]}}
    with pytest.raises(ValueError):
        project_key_door_report(report, source_label="malformed.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps_executed", True),
        ("prediction_calls_attempted", "1"),
        ("prediction_calls_returned", -1),
        ("return", "0"),
        ("num_classes", True),
        ("terminal", 0),
        ("transitions", {}),
        ("transitions", [{"probabilities": [2]}]),
    ],
)
def test_should_validate_present_fields_even_when_layout_is_unknown(success, field, value):
    absent: Any = {
        "checkpoint_updates": 192,
        "context": "A",
        "cue": 1,
        "layout_partition": "known_layout",
        "status": "failed",
        "layout": None,
    }
    absent[field] = value
    with pytest.raises(ValueError):
        episode_projection(single(absent))


def test_should_preserve_nullable_partial_readouts_without_imputing_numbers(success):
    report = single(success)
    report.update(
        probes=[
            {
                "checkpoint_updates": 0,
                "charged_updates": None,
                "report": {
                    "checkpoints": [
                        {"updates": 0, "scores": {"E": None}, "decision_diagnostics": {"E": None}}
                    ]
                },
            }
        ],
        checkpoints=[{"updates": 0, "mean_ce": None}],
        work=None,
        replay=None,
    )
    before = deepcopy(report)
    result = project_key_door_report(report, source_label="unknown.json")
    assert result["source"] == before and report == before
    assert result["source"]["probes"][0]["charged_updates"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps_executed", 1),
        ("prediction_calls_attempted", 1),
        ("prediction_calls_returned", 1),
        ("return", 1),
        ("success", True),
        ("terminal", True),
        ("outcome", "success"),
    ],
)
def test_should_refuse_positive_work_or_success_claims_in_unobserved_slots(field, value):
    absent: Any = {
        "checkpoint_updates": 192,
        "context": "A",
        "cue": 1,
        "layout_partition": "known_layout",
        "status": "failed",
        "layout": None,
    }
    absent[field] = value
    with pytest.raises(ValueError, match="unobserved slot"):
        episode_projection(single(absent))
