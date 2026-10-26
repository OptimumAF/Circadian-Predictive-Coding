"""Saved FIFO36 learning curves remain inspectable without invoking a learner."""

import builtins
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.minigrid_saved_fifo_inspector import (
    PROTOCOL_ID,
    project_minigrid_saved_fifo_report,
)
from test_minigrid_inspector_projection import pointer


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/saved-fifo-fresh-return-20261019/native.json"
)
METHOD = "public_invariant_observation_memory_fifo"


@pytest.fixture(scope="module")
def recorded_fifo():
    return json.loads(ARTIFACT.read_text(encoding="utf8"))


def lost_helper_report(recorded):
    """A failed helper can lose its report while its permanent charges survive."""
    report = deepcopy(recorded)
    report.update(status="incomplete", reason="helper report unavailable after paid work")
    cohort = report["cohorts"][1]
    cohort.update(status="failed", reason=report["reason"], report=None)
    cohort["outcomes"] = dict.fromkeys(cohort["outcomes"])
    report["work"]["probe_totals_are_unknown"] = True
    return report


def test_should_project_actual_saved_report_as_detached_sixty_four_observed_curves(recorded_fifo):
    before = deepcopy(recorded_fifo)
    projected = project_key_door_report(recorded_fifo, source_label=ARTIFACT.name)
    assert PROTOCOL_ID == recorded_fifo["protocol"]["protocol_id"]
    assert projected["schema_id"] == "minigrid_saved_fifo_inspector_v1"
    assert projected["origin"] == "recorded_artifact"
    assert projected["source_label"] == ARTIFACT.name
    assert projected["source"] == before == recorded_fifo
    assert projected["source"] is not recorded_fifo
    condition = projected["conditions"][0]
    assert len(projected["conditions"]) == 1
    assert condition["key"] == METHOD and condition["pointer"] == ""
    assert condition["source_role"] == "train_and_probe_test"
    assert condition["status"] == "complete" and condition["episodes"] == []
    curves = condition["curves"]
    assert len(curves) == 64
    identity = {
        (row["source_updates"], row["phase"], row["updates"], row["context"], row["cue"])
        for row in curves
    }
    assert len(identity) == 64
    assert {row["source_updates"] for row in curves} == {0, 128}
    assert {row["phase"] for row in curves} == {"F learning", "A return"}
    assert {row["updates"] for row in curves} == {0, 4, 8, 16}
    for row in curves:
        score = pointer(projected["source"], row["pointer"])
        assert row["status"] == "complete"
        assert row["ce"] == score["ce"] and row["accuracy"] == score["accuracy"]
        assert row["exact_sequence_accuracy"] == float(score["accuracy"] == 1)
        assert row["role"] == ("train" if row["context"] == "A" else "probe_test")
        assert row["cue"] in ("blue", "red")
    projected["source"]["cohorts"][0]["report"]["checkpoints"][0]["scores"]["F-blue"]["ce"] = 99
    assert recorded_fifo == before


def test_should_preserve_cold_acquisition_mature_retention_and_exact_signed_source_values(
    recorded_fifo,
):
    p = project_minigrid_saved_fifo_report(recorded_fifo, source_label="saved.json")
    condition = p["conditions"][0]
    curves = condition["curves"]
    for row in curves:
        if row["source_updates"] == 128 and row["context"] == "A":
            assert row["accuracy"] == row["exact_sequence_accuracy"] == 1
        if row["context"].startswith("F-") and row["updates"] == 16:
            assert row["accuracy"] == 1
    metrics = condition["signed_metrics"]
    assert len(metrics) == 8
    for row in metrics:
        assert row["value"] == pointer(p["source"], row["pointer"])
        expected_direction = (
            "positive_gain_better" if row["label"].endswith("gain") else "positive_loss_worse"
        )
        assert row["direction"] == expected_direction
    assert p["source"]["scientific_advantage_verified"] is False
    assert p["source"]["origin"]["status"] == "incomplete"
    assert p["source"]["origin"]["budget_status"] == "spent; not restored"
    assert p["source"]["origin"]["censored_games"] == [[64, "B", "blue"]]


def test_should_keep_paid_failed_cohort_measurements_unknown_instead_of_zero(recorded_fifo):
    report = lost_helper_report(recorded_fifo)
    before = deepcopy(report)
    p = project_minigrid_saved_fifo_report(report, source_label="partial.json")
    assert p["source"] == before == report
    curves = p["conditions"][0]["curves"]
    assert len(curves) == 64
    unknown = [row for row in curves if row["source_updates"] == 128]
    assert len(unknown) == 32
    for row in unknown:
        assert row["status"] == "not_run" and row["derived"] is True
        assert (
            row["pointer"] is row["ce"] is row["accuracy"] is row["exact_sequence_accuracy"] is None
        )
    assert p["source"]["cohorts"][1]["charged_updates"] == 32
    assert p["source"]["cohorts"][1]["phase_charges"] == {"F": 16, "A_return": 16}
    assert all(
        row["value"] is None
        for row in p["conditions"][0]["signed_metrics"]
        if row["source_updates"] == 128
    )


def test_should_preserve_false_proofs_and_exact_cost_pointers_without_acceptance_claim(
    recorded_fifo,
):
    report = deepcopy(recorded_fifo)
    report["cohorts"][1]["source_unchanged"] = False
    p = project_minigrid_saved_fifo_report(report, source_label="unproven.json")
    condition = p["conditions"][0]
    proofs = condition["proofs"]
    assert any(
        row["pointer"] == "/cohorts/1/source_unchanged" and row["value"] is False for row in proofs
    )
    assert condition["costs"]
    for row in proofs + condition["costs"]:
        assert row["value"] == pointer(p["source"], row["pointer"])
    assert any(
        row["pointer"] == "/work/completed_storage_updates" and row["value"] == 64
        for row in condition["costs"]
    )


@pytest.mark.parametrize(
    "kind",
    [
        "protocol",
        "seed",
        "width",
        "classes",
        "capacity",
        "payload",
        "retirement",
        "source_clock",
        "rows_before",
        "end_clock",
        "end_rows",
        "snapshot_policy",
        "F_charge",
        "A_charge",
        "total_charge",
        "observed_charge",
        "phase_role",
        "source_role",
        "storage_role",
        "monitor_role",
        "final",
        "advantage",
        "negative_cost",
        "bool_cost",
        "nan_cost",
        "accuracy",
        "negative_ce",
        "diagnostic",
        "invented_gain",
        "origin_budget",
        "origin_rows",
        "origin_censored",
    ],
)
def test_should_refuse_contradictory_roles_clocks_charges_metrics_and_final_claims(
    recorded_fifo, kind
):
    r = deepcopy(recorded_fifo)
    c = r["cohorts"][1]
    policy = r["protocol"]["policy"]
    if kind == "protocol":
        r["protocol"]["protocol_id"] += "-different"
    elif kind == "seed":
        r["protocol"]["seed"] = 18
    elif kind in ("width", "classes", "capacity", "payload"):
        key = {
            "width": "feature_dim",
            "classes": "num_classes",
            "capacity": "max_entries",
            "payload": "max_payload_bytes",
        }[kind]
        policy[key] += 1
    elif kind == "retirement":
        r["protocol"]["retirement_metadata_bytes"] = 0
    elif kind == "source_clock":
        c["source_updates"] = 127
    elif kind == "rows_before":
        c["native_training_examples_before"] -= 9
    elif kind in ("end_clock", "end_rows"):
        key = "completed_updates" if kind == "end_clock" else "training_examples"
        c["end_snapshot"]["native_state"][key] += 1
    elif kind == "snapshot_policy":
        c["end_snapshot"]["policy"]["native_policy"]["max_entries"] = 64
    elif kind in ("F_charge", "A_charge"):
        c["phase_charges"]["F" if kind == "F_charge" else "A_return"] = 15
    elif kind == "total_charge":
        c["charged_updates"] = 31
    elif kind == "observed_charge":
        c["budget_updates_observed"] = 31
    elif kind == "phase_role":
        r["protocol"]["phases"][0]["role"] = "final_test"
    elif kind == "source_role":
        r["source_reports"][0]["role"] = "final_test"
    elif kind == "storage_role":
        r["probe_data"][0]["role"] = "probe_test"
    elif kind == "monitor_role":
        c["report"]["training_readouts"]["checkpoints"][0]["decision_diagnostics"][
            "minigrid-source-A-blue:saved-monitor"
        ]["role"] = "final_test"
    elif kind == "final":
        r["final_evaluation"] = {"status": "complete", "accuracy": 1.0}
    elif kind == "advantage":
        r["scientific_advantage_verified"] = True
    elif kind in ("negative_cost", "bool_cost", "nan_cost"):
        r["work"]["completed_storage_updates"] = {
            "negative_cost": -1,
            "bool_cost": True,
            "nan_cost": float("nan"),
        }[kind]
    elif kind in ("accuracy", "negative_ce"):
        c["report"]["checkpoints"][0]["scores"]["F-blue"][
            "accuracy" if kind == "accuracy" else "ce"
        ] = 2 if kind == "accuracy" else -1
    elif kind == "diagnostic":
        c["report"]["checkpoints"][0]["decision_diagnostics"]["F-blue"]["confusion"][0][0] += 1
    elif kind == "invented_gain":
        c["outcomes"]["new_F_gain"] += 1
    elif kind == "origin_budget":
        r["origin"]["budget_status"] = "renewed"
    elif kind == "origin_rows":
        r["origin"]["training_rows"] = 1143
    else:
        r["origin"]["censored_games"] = [[64, "B", "red"]]
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(r, source_label="corrupt.json")


def test_should_project_through_existing_dispatcher_without_importing_model_or_environment(
    recorded_fifo, monkeypatch
):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "numpy", "gymnasium", "minigrid")
        assert not name.startswith("src.adapters")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    p = project_key_door_report(recorded_fifo, source_label="native.json")
    assert p["schema_id"] == "minigrid_saved_fifo_inspector_v1"


def test_should_refuse_checkpoint_after_a_failed_phase_returned_no_updates(recorded_fifo):
    report = deepcopy(recorded_fifo)
    report.update(status="incomplete", reason="update failed before returning")
    cohort = report["cohorts"][0]
    cohort.update(status="failed", reason=report["reason"])
    cohort["outcomes"] = dict.fromkeys(cohort["outcomes"])
    helper = cohort["report"]
    helper.update(
        status="failed",
        reason=report["reason"],
        completed_updates=0,
        completed_training_examples=0,
        diagnostics=[],
        return_phase=None,
    )
    report["work"]["probe_totals_are_unknown"] = True
    # Paid attempts can survive failure, but a checkpoint at16 cannot predate
    # the first completed update. Keep the original observed checkpoint payload.
    assert helper["checkpoints"][-1]["updates"] == 16
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(report, source_label="impossible-prefix.json")


@pytest.mark.parametrize(
    "field",
    [
        "attempted_storage_updates",
        "completed_storage_updates",
        "attempted_training_rows",
        "completed_training_rows",
        "score_calls_attempted",
        "score_calls_completed",
        "score_rows_completed",
        "observed_charged_updates_lower_bound",
    ],
)
def test_should_refuse_known_aggregate_work_that_disagrees_with_saved_helpers(recorded_fifo, field):
    report = deepcopy(recorded_fifo)
    report["work"][field] -= 1
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(report, source_label="inconsistent-totals.json")


@pytest.mark.parametrize("location", ["protocol", "outer_snapshot", "native_snapshot"])
@pytest.mark.parametrize(
    "field", ["feature_dim", "num_classes", "max_entries", "max_payload_bytes"]
)
def test_should_refuse_equal_valued_float_policy_in_every_recorded_policy(
    recorded_fifo, location, field
):
    report = deepcopy(recorded_fifo)
    policy = {
        "protocol": report["protocol"]["policy"],
        "outer_snapshot": report["cohorts"][0]["end_snapshot"]["policy"]["native_policy"],
        "native_snapshot": report["cohorts"][0]["end_snapshot"]["native_state"]["policy"],
    }[location]
    policy[field] = float(policy[field])
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(report, source_label="floating-policy.json")


@pytest.mark.parametrize(
    "field", ["feature_dim", "num_classes", "max_entries", "max_payload_bytes"]
)
def test_should_refuse_boolean_policy_dimensions(recorded_fifo, field):
    report = deepcopy(recorded_fifo)
    report["protocol"]["policy"][field] = True
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(report, source_label="boolean-policy.json")


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("F", "role", "train"),
        ("F", "context", "A-blue"),
        ("A", "role", "probe_test"),
        ("A", "context", "F-blue"),
    ],
)
def test_should_refuse_optional_diagnostic_role_or_context_relabeling(
    recorded_fifo, kind, field, value
):
    report = deepcopy(recorded_fifo)
    helper = report["cohorts"][0]["report"]
    checkpoint = (
        helper["checkpoints"][0] if kind == "F" else helper["training_readouts"]["checkpoints"][0]
    )
    key = "F-blue" if kind == "F" else "minigrid-source-A-blue:saved-monitor"
    checkpoint["decision_diagnostics"][key][field] = value
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(report, source_label="relabelled-diagnostic.json")


@pytest.mark.parametrize("phase", ["F", "A_return"])
def test_should_refuse_complete_cohort_with_failed_training_monitor(recorded_fifo, phase):
    report = deepcopy(recorded_fifo)
    helper = report["cohorts"][0]["report"]
    if phase == "A_return":
        helper = helper["return_phase"]
    helper["training_readouts"].update(status="failed", reason="saved monitor failed")
    assert report["cohorts"][0]["status"] == "complete"
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(report, source_label="false-complete-monitor.json")


@pytest.mark.parametrize("phase", ["F", "A_return"])
@pytest.mark.parametrize("readout", ["probe", "training_monitor"])
def test_should_refuse_complete_cohort_with_explicit_failed_observed_checkpoint(
    recorded_fifo, phase, readout
):
    report = deepcopy(recorded_fifo)
    helper = report["cohorts"][0]["report"]
    if phase == "A_return":
        helper = helper["return_phase"]
    if readout == "training_monitor":
        helper = helper["training_readouts"]
    # Ordinary native points need no explicit status; an explicit failed status
    # cannot be promoted to complete merely because score dictionaries exist.
    helper["checkpoints"][0].update(status="failed", reason="recorded checkpoint failed")
    with pytest.raises(ValueError):
        project_minigrid_saved_fifo_report(report, source_label="false-complete-point.json")
