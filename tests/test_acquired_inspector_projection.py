"""Saved acquired-task TRAIN projection; no environment or learner execution."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.core.context_return_metrics import summarize_acquired_return


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "off": "artifacts/runs/r21-acquired-task-return-20261016/pilot.json",
    "recent48": "artifacts/runs/r21-acquired-replay-20261016/recent48/pilot.json",
    "coverage": "artifacts/runs/r21-acquired-replay-20261016/coverage/pilot.json",
}


@pytest.fixture(scope="module")
def saved():
    return {
        control: json.loads((ROOT / path).read_text(encoding="utf-8"))
        for control, path in PATHS.items()
    }


def project(report):
    return project_key_door_report(report, source_label="saved TRAIN artifact")


@pytest.mark.parametrize("control", list(PATHS))
def test_should_preserve_original_train_pointers_records_frames_and_unavailable_off_clocks(
    saved, control
):
    source = deepcopy(saved[control])
    before = deepcopy(source)
    result = project(source)
    condition = result["conditions"][0]
    assert condition["key"] == "single" and condition["pointer"] == ""
    assert (
        condition["readout_key"] == "acquired_return_diagnostics"
        and condition["source_role"] == "train"
    )
    assert condition["unknown"] == dict(topology=None, lineage=None, sleep_proposals=None)
    assert result["source"] == before and source == before and len(condition["episodes"]) == 12
    for index, episode in enumerate(condition["episodes"]):
        assert episode["pointer"] == f"/acquired_return_diagnostics/episodes/{index}"
        assert episode["record"] == before["acquired_return_diagnostics"]["episodes"][index]
        assert len(episode["frames"]) == len(episode["record"]["transitions"]) + 1
        assert (
            episode["record"]["phase"] == "evaluation"
            and episode["record"]["source_role"] == "train"
        )
        assert episode["frames"][0]["position"] == (
            [2, 4] if episode["record"]["context"] == "A" else [2, 0]
        )
    if control == "off":
        assert "checkpoint_clocks" not in result["source"]["acquired_return_diagnostics"]
    else:
        assert [
            clock["native_updates"]
            for clock in result["source"]["acquired_return_diagnostics"]["checkpoint_clocks"]
        ] == [288, 384, 480]
    result["source"]["status"] = "changed"
    condition["episodes"][0]["record"]["success"] = "changed"
    condition["episodes"][0]["frames"][0]["position"][0] = 99
    assert source == before


def test_should_keep_legacy_paired_projection_schema_and_episode_pointers():
    source = json.loads(
        (ROOT / "artifacts/runs/r21-return-fresh-task-probes-20261016/native.json").read_text(
            encoding="utf-8"
        )
    )
    result = project(source)
    for condition in result["conditions"]:
        assert set(condition) == {
            "key",
            "pointer",
            "status",
            "protocol_id",
            "episodes",
            "map_origin",
            "unknown",
        }
        assert all(
            episode["pointer"].startswith(condition["pointer"] + "/policy_evaluation/episodes/")
            for episode in condition["episodes"]
        )


@pytest.mark.parametrize(
    "fault",
    [
        "final_selector",
        "paired",
        "wrong_protocol",
        "renamed_selector",
        "top_role",
        "protocol_role",
        "helper_role",
        "episode_role",
        "identity_role",
        "phase",
        "slot_point",
        "law_marker",
    ],
)
def test_should_refuse_mixed_or_mislabeled_train_selectors_roles_and_identity(saved, fault):
    source = deepcopy(saved["coverage"])
    readout = source["acquired_return_diagnostics"]
    if fault == "final_selector":
        source["policy_evaluation"] = None
    elif fault == "paired":
        source = dict(status="complete", conditions={"off": source})
    elif fault == "wrong_protocol":
        source["protocol"]["protocol_id"] = "saved_final_policy_v1"
    elif fault == "renamed_selector":
        source["policy_evaluation"] = source.pop("acquired_return_diagnostics")
    elif fault == "top_role":
        source["source_role"] = "final_test"
    elif fault == "protocol_role":
        source["protocol"]["diagnostic_data_role"] = "final_test"
    elif fault == "helper_role":
        readout["source_role"] = "final_test"
    elif fault == "episode_role":
        readout["episodes"][0]["source_role"] = "final_test"
    elif fault == "identity_role":
        readout["episodes"][0]["identity"]["task_sequence"] = ["B"]
    elif fault == "phase":
        readout["episodes"][0]["phase"] = "train"
    elif fault == "slot_point":
        readout["episodes"][0]["checkpoint_updates"] = 193
    else:
        readout["episodes"][0]["action_conditioning"] = "raw_argmax"
    with pytest.raises(ValueError):
        project(source)


@pytest.mark.parametrize(
    "fault",
    [
        "protocol_control",
        "top_control",
        "helper_control",
        "replay_flag",
        "identity_flag",
        "clock_missing",
        "clock_native",
        "clock_rows",
        "clock_float",
        "raw_clock",
        "candidate_count",
        "charges",
        "off_fabricated_clock",
    ],
)
def test_should_refuse_control_or_recorded_clock_disagreement(saved, fault):
    source = deepcopy(saved["off" if fault == "off_fabricated_clock" else "coverage"])
    readout = source["acquired_return_diagnostics"]
    if fault == "protocol_control":
        source["protocol"]["replay_control"] = "recent48"
    elif fault == "top_control":
        source["replay_control"] = "recent48"
    elif fault == "helper_control":
        readout["replay_control"] = "recent48"
    elif fault == "replay_flag":
        readout["replay_enabled"] = False
    elif fault == "identity_flag":
        readout["episodes"][0]["identity"]["replay_enabled"] = False
    elif fault == "clock_missing":
        del readout["checkpoint_clocks"]
    elif fault == "clock_native":
        readout["checkpoint_clocks"][0]["native_updates"] = 192
    elif fault == "clock_rows":
        readout["checkpoint_clocks"][0]["native_training_examples"] = 2303
    elif fault == "clock_float":
        readout["checkpoint_clocks"][0]["wake_updates"] = 192.0
    elif fault == "raw_clock":
        source["checkpoints"][12]["clocks"]["native_updates"] = 192
    elif fault == "candidate_count":
        source["partial_stream"]["native_candidate_updates"] = 320
    elif fault == "charges":
        source["charged_updates"] = 480
    else:
        readout["checkpoint_clocks"] = deepcopy(
            saved["coverage"]["acquired_return_diagnostics"]["checkpoint_clocks"]
        )
    with pytest.raises(ValueError):
        project(source)


@pytest.mark.parametrize(
    "fault", ["raw_sum", "conditional", "illegal_mass", "argmax", "missing_raw", "support"]
)
def test_should_validate_recorded_raw_to_legal_conditional_law_before_playback(saved, fault):
    source = deepcopy(saved["off"])
    transition = source["acquired_return_diagnostics"]["episodes"][0]["transitions"][0]
    if fault == "raw_sum":
        transition["raw_probabilities"] = [0.0] * 5
    elif fault == "conditional":
        transition["probabilities"] = [0.5, 0.5, 0, 0, 0]
    elif fault == "illegal_mass":
        transition["probabilities"][1] = 1e-13
    elif fault == "argmax":
        transition["action"] = 1
    elif fault == "missing_raw":
        del transition["raw_probabilities"]
    else:
        transition["legal_actions"] = [0, 2]
    with pytest.raises(ValueError):
        project(source)


@pytest.mark.parametrize(
    "fault",
    [
        "outcomes",
        "signed_live",
        "initial_live",
        "initial_condition",
        "C_condition",
        "top_interpretation",
        "raw_metric",
        "raw_endpoint",
        "raw_range",
        "raw_source_range",
        "forged_raw_sequence",
        "forged_raw_ce",
        "work",
    ],
)
def test_should_refuse_finite_derived_claims_that_contradict_saved_evidence(saved, fault):
    source = deepcopy(saved["coverage"])
    readout = source["acquired_return_diagnostics"]
    if fault == "outcomes":
        readout["outcomes"][0]["by_layout"]["A"]["success_rate"] = 0.5
    elif fault == "signed_live":
        readout["metrics"]["success_rate"]["old_A_loss"] = -1
    elif fault == "initial_live":
        readout["initial_A_acquired"] = False
    elif fault == "initial_condition":
        source["acquired_initial_condition_met"] = False
    elif fault == "C_condition":
        source["acquired_new_task_condition_met"] = False
    elif fault == "top_interpretation":
        source["acquired_interpretation"] = "inconclusive_initial_A_not_acquired"
    elif fault == "raw_metric":
        source["acquired_return"]["metrics"]["ce"]["return_gap"] = 123
    elif fault == "raw_endpoint":
        source["acquired_return"]["endpoint_values"][0]["updates"] = 191
    elif fault == "raw_range":
        source["acquired_return"]["endpoint_values"][0]["A"]["accuracy"] = 2
    elif fault == "raw_source_range":
        source["checkpoints"][0]["scores"]["A-red"]["accuracy"] = 2
    elif fault in ("forged_raw_sequence", "forged_raw_ce"):
        endpoints = source["acquired_return"]["endpoint_values"]
        if fault == "forged_raw_sequence":
            endpoints[0]["C_train"]["exact_sequence_accuracy"] = 1.0
        else:
            endpoints[0]["A"]["ce"] = 0.5
        source["acquired_return"] = summarize_acquired_return(tuple(endpoints))
    else:
        readout["work"]["live_callback_returns"] += 1
    with pytest.raises(ValueError):
        project(source)


def test_should_preserve_partial_unobserved_train_slots_and_unknown_metrics(saved):
    source = deepcopy(saved["off"])
    source.update(
        status="incomplete",
        acquired_return=None,
        acquired_initial_condition_met=None,
        acquired_interpretation="not_reached",
    )
    readout = source["acquired_return_diagnostics"]
    readout.update(
        status="not_run",
        outcomes=None,
        metrics=None,
        initial_A_acquired=None,
        interpretation="not reached",
        live_sessions_started=0,
        work={},
    )
    for index, episode in enumerate(readout["episodes"]):
        readout["episodes"][index] = {
            key: deepcopy(episode[key])
            for key in ("checkpoint_updates", "context", "cue", "layout_partition", "source_role")
        }
        readout["episodes"][index].update(
            status="not_run", reason="not reached", state_unchanged=None, layout=None
        )
    result = project(source)
    assert len(result["conditions"][0]["episodes"]) == 12
    assert all(episode["frames"] == [] for episode in result["conditions"][0]["episodes"])
    assert result["source"]["acquired_return_diagnostics"]["outcomes"] is None
