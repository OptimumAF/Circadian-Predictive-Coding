"""Real quota/receipt and disposable probe runners with fake neural arithmetic."""

from copy import deepcopy
import json
import pickle
import sys

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import key_door_quota_probes as pilot
from src.adapters import key_door_reward_pilot as recipe
from src.adapters import key_door_pilot as shared
import test_acquired_quota_prefix as prefix_fixture
from test_recurrent_key_door_pilot import Learner as ArrayLearner
from test_torch_feature_rehearsal_pilot import forbidden


def environment(monkeypatch, *, short=True):
    original_collect = shared._collect
    if short:
        monkeypatch.setattr(prefix_fixture, "PREFIX_UPDATES", 2)
        monkeypatch.setattr(pilot.acquired, "PREFIX", 2)
        monkeypatch.setattr(recipe, "ACQUIRED_PREFIX_UPDATES", 2)
    evidence = prefix_fixture.environment(monkeypatch)
    prefix_collect = shared._collect
    evidence.update(probe_calls=[], probe_owners=[], packets=[])

    def collect(identity, layout, cue, role):
        assert role != "final_test", "probe-only route constructed final data"
        if role in ("probe_train", "probe_test"):
            result = original_collect(identity, layout, cue, role)
            evidence["packets"].append(result)
            return result
        return prefix_collect(identity, layout, cue, role)

    def bridge(state, *, check_boundary):
        check_boundary()
        owner = ArrayLearner(None)
        owner.restore_state(state)
        evidence["probe_owners"].append(owner)
        check_boundary()
        return owner

    original_probe = pilot.run_fresh_task_probe

    def probe(source, state, *data, **options):
        evidence["probe_calls"].append((deepcopy(state), options["budget"]))
        return original_probe(source, state, *data, **options)

    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(recipe, "_evaluate_arm", forbidden)
    monkeypatch.setattr(recipe, "evaluate_frozen_physical_checkpoints", forbidden)
    monkeypatch.setattr(pilot, "recurrent_probe_source", bridge)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    return evidence


def test_should_run_fixed192_prefix_own96_steps_and_independent_e16_probes_without_finals(
    monkeypatch,
):
    evidence = environment(monkeypatch, short=False)
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert report["status"] == "complete", (row["reason"], report["global_errors"])
    assert row["wake_updates"] == len(evidence["updates"]) == len(row["training"])
    assert 200 <= row["wake_updates"] <= 208
    assert len(evidence["demonstrations"]) == 192
    assert evidence["physical"] == {"reward_prefix": 96}
    assert [slot["checkpoint_updates"] for slot in row["probes"]] == [0, row["wake_updates"]]
    assert report["probe_updates_charged"] == 32
    assert len(evidence["budgets"]) == 1
    wake = evidence["budgets"][0]
    assert wake.budget.max_training_updates == 208 and wake.updates_completed == row["wake_updates"]
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["stream_result"] is None and row["episodes"] == []
    assert not evidence["final_packets"] and not evidence["final_started"]
    assert report["final_data_constructed"] is False and report["reserved_data_episodes"] == 18
    assert len(evidence["probe_calls"]) == 2
    for slot, (state, budget), owner in zip(
        row["probes"], evidence["probe_calls"], evidence["probe_owners"], strict=True
    ):
        assert budget is not wake and budget.budget.max_training_updates == 16
        assert budget.updates_completed == slot["charged_updates"] == 16
        assert pickle.dumps(owner.snapshot_state()) == pickle.dumps(state)
        assert (
            slot["source_unchanged"]
            is slot["state_unchanged"]
            is slot["budget_binding_preserved"]
            is True
        )
        assert slot["report"]["completed_training_examples"] == 128
        assert slot["report"]["score_forward_calls_completed"] == 8
        assert slot["report"]["decision_diagnostics_enabled"] is True
        assert [p["updates"] for p in slot["report"]["checkpoints"]] == [0, 4, 8, 16]
    assert all(packet.role in ("probe_train", "probe_test") for packet in evidence["packets"])


@pytest.mark.parametrize("mutation", ["budget", "future_probe", "checkpoint"])
def test_should_refuse_changed_initial_bindings_before_any_benchmark_training(
    monkeypatch, mutation
):
    evidence = environment(monkeypatch)
    original = pilot.run_fresh_task_probe

    def corrupt(source, state, *data, **options):
        # Initial probe owns no benchmark updates yet.
        row = captured["row"]
        if mutation == "budget":
            evidence["budgets"][0].last_clock += 1.0
        elif mutation == "future_probe":
            row["probes"][1] = deepcopy(row["probes"][1])
        else:
            row["probes"][0]["checkpoint_updates"] = False
        return original(source, state, *data, **options)

    captured = {}
    arm = recipe._run_arm

    def run(row, *args, **options):
        captured["row"] = row
        return arm(row, *args, **options)

    monkeypatch.setattr(recipe, "_run_arm", run)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", corrupt)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert report["runs"][0]["wake_updates"] == 0
    assert not evidence["updates"] and not evidence["demonstrations"]
    assert report["probe_updates_charged"] == 0


def test_should_charge_failed_probe_and_leave_end_probe_and_benchmark_unreached(monkeypatch):
    evidence = environment(monkeypatch)

    def failed(source, state, *data, **options):
        budget = options["budget"]
        budget.before_update()
        budget.record_update()
        return {
            "status": "failed",
            "reason": "declared failed probe",
            "completed_updates": 1,
            "completed_training_examples": 8,
            "score_forward_calls_completed": 0,
        }

    monkeypatch.setattr(pilot, "run_fresh_task_probe", failed)
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert report["probe_updates_charged"] == 1 and row["wake_updates"] == 0
    assert row["probes"][0]["status"] == "failed" and row["probes"][1]["status"] == "not_run"
    assert not evidence["updates"]


@pytest.mark.parametrize("when", ["train", "end_probe"])
def test_should_refuse_changed_prior_probe_evidence_before_more_work(monkeypatch, when):
    evidence = environment(monkeypatch)

    def corrupt_row(row):
        row["probes"][0]["report"]["completed_updates"] = 15

    if when == "train":
        evidence["after_train"] = lambda owner, update, row: corrupt_row(row)
    else:
        original = pilot.run_fresh_task_probe
        arm = recipe._run_arm
        captured = {}

        def run(row, *args, **options):
            captured["row"] = row
            return arm(row, *args, **options)

        def probe(source, state, *data, **options):
            if state.completed_updates:
                corrupt_row(captured["row"])
            return original(source, state, *data, **options)

        monkeypatch.setattr(recipe, "_run_arm", run)
        monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert row["probes"][0]["report"]["completed_updates"] == 15
    assert row["wake_updates"] == (1 if when == "train" else len(row["training"]))
    assert report["probe_updates_charged"] == 16


@pytest.mark.parametrize("describe", [True, False])
def test_should_dispatch_probe_only_route_without_user_tuning_options(
    monkeypatch, capsys, describe
):
    selected = "describe_quota_probe_protocol" if describe else "run_quota_probe_pilot"
    calls = []

    def selected_pilot():
        calls.append(True)
        return {"status": "complete"}

    monkeypatch.setattr(cli, selected, selected_pilot)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-quota-probe"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["status"] == "complete"


def test_should_describe_without_constructing_models_budgets_or_probe_data(monkeypatch):
    monkeypatch.setattr(recipe, "require_torch", forbidden)
    monkeypatch.setattr(shared, "build_probe_stream", forbidden)
    monkeypatch.setattr(recipe, "ProcessRssSampler", forbidden)
    protocol = pilot.describe_protocol()
    assert protocol["prefix"]["updates"] == 192
    assert protocol["benchmark"]["maximum_native_updates"] == 208
    assert protocol["probes"]["total_updates"] == 32
    assert protocol["final_evaluation"].startswith("unavailable")


def test_should_refuse_last_readback_mutation_after_both_probes_without_erasing_charges(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    original = recipe._readback_arm

    def readback(row, source, runtime, *args, **options):
        original(row, source, runtime, *args, **options)
        if row["status"] == "complete":
            runtime._candidate.state.completed_updates += 1

    monkeypatch.setattr(recipe, "_readback_arm", readback)
    report = pilot.run_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert "probe changed original quota" in row["reason"]
    assert report["probe_updates_charged"] == 32
    assert row["wake_updates"] == len(evidence["updates"])
    assert row["work"]["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize(
    "options,final",
    [
        ({"probe_only": True}, None),
        ({"probe_only": 1, "checkpoint_probe": forbidden}, None),
        ({"checkpoint_probe": forbidden}, None),
        ({"probe_only": True, "checkpoint_probe": forbidden}, forbidden),
    ],
)
def test_should_refuse_invalid_probe_mode_before_constructing_native_owner(
    monkeypatch, options, final
):
    environment(monkeypatch)
    monkeypatch.setattr(recipe, "TorchDemonstrationRewardLearner", forbidden)
    row = pilot.acquired._row(False)
    recipe._run_arm(row, final, float("inf"), None, **options)
    assert row["status"] == "failed" and row["wake_updates"] == 0
    assert "probe-only" in row["reason"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--source-artifact", "unused.json"],
        ["--environment-seed", "1"],
        ["--probe-decision-diagnostics"],
        ["--seed", "5"],
    ],
)
def test_should_refuse_unfrozen_probe_cli_inputs_before_work(monkeypatch, flags):
    monkeypatch.setattr(cli, "run_quota_probe_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--scenario", "torch-key-door-quota-probe", *flags])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_preserve_unknown_partial_reserved_collection_cost_instead_of_reporting_zero(
    monkeypatch,
):
    environment(monkeypatch)

    def failed_collection():
        raise RuntimeError("partial reserved collection cost unknown")

    monkeypatch.setattr(shared, "build_probe_stream", failed_collection)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["probe_updates_charged"] == 0
    assert report["reserved_data_episodes"] is None
    assert report["reserved_data_work_may_be_unobserved"] is True
