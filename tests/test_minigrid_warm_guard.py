"""Completed warm guards cannot weaken the enclosing reward evidence proof.

Only real runtime authority and owned fake boundary callbacks are exercised.
No native learner, environment, experiment or extra update dose is created.
"""

import pytest

from src.adapters import minigrid_reward_pilot as pilot
from src.adapters import minigrid_warm_readouts as diagnostics
from test_minigrid_warm_reward import Native, ports as ports, prepare_fixture
from test_minigrid_warm_readout_pilot import (
    install_scoring,
    readout_case as readout_case,
)
from test_torch_feature_rehearsal_pilot import Sampler


def arm_second_collection(work):
    def arm(index, call):
        if index == 1:
            work["armed"] = True

    work["during_collection"] = arm


def assert_one_paid_update(report, work):
    assert report["status"] == "incomplete" and report["wake_updates"] == 1
    assert report["total_new_native_updates"] == 129
    assert len(work["reward_calls"]) == 1 and len(work["collections"]) == 2
    assert "report" not in work["collections"][1]
    assert report["training"][0]["receipt"]["update_number"] == 1
    assert report["training"][0]["collection"]["work"]["sampling_draws"] == 2
    assert all(slot["status"] == "not_run" for slot in report["training"][2:])


@pytest.mark.parametrize("target", ["collection", "receipt", "protocol"])
def test_should_recheck_full_preceding_evidence_after_completed_warm_callback(
    monkeypatch, ports, target
):
    prepare_fixture(monkeypatch, ports)
    work = ports.work
    changed: list[bool] = []
    arm_second_collection(work)

    def corrupt():
        if not work.get("armed") or changed:
            return
        changed.append(True)
        report = work["reports"][0]
        if target == "collection":
            report["training"][0]["collection"]["work"]["sampling_draws"] = 99
        elif target == "receipt":
            report["training"][0]["receipt"]["update_number"] = 99
        else:
            report["protocol"]["learning_rate"] = 0.04

    work["warm_guard_callback"] = corrupt
    report = pilot.run_pilot(warm_start=True)
    assert changed == [True]
    assert_one_paid_update(report, work)


def test_should_recheck_completed_diagnostic_payload_after_warm_callback(monkeypatch, readout_case):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)
    changed: list[bool] = []

    def arm(index, call):
        if index == 1:
            case.work["armed"] = True

    def corrupt():
        if case.work.get("armed") and not changed:
            changed.append(True)
            case.work["reports"][0]["train_diagnostics"]["readouts"][0]["scores"][0]["accuracy"] = (
                1.0
            )

    case.work["during_readout"] = arm
    case.work["warm_guard_callback"] = corrupt
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    result = report["train_diagnostics"]
    assert changed == [True] and report["status"] == "incomplete"
    assert report["wake_updates"] == 8 and report["total_new_native_updates"] == 136
    assert result["readouts"][0]["status"] == "complete"
    assert result["readouts"][0]["scores"][0]["accuracy"] == 0.0
    assert result["readouts"][1]["status"] != "complete"
    assert result["work"]["raw_score_forwards_completed"] == 6
    assert len(case.work["reward_calls"]) == 8
    assert result["endpoint_changes"] is result["endpoint_conditions"] is None


@pytest.mark.parametrize("target", ["source", "actor", "candidate"])
def test_should_recheck_prior_record_after_late_native_snapshot_callback(
    monkeypatch, ports, target
):
    prepare_fixture(monkeypatch, ports)
    work = ports.work
    changed: list[bool] = []
    arm_second_collection(work)
    snapshot = Native.snapshot_state

    def read(owner):
        state = snapshot(owner)
        if work.get("armed") and not changed:
            runtime = work["runtimes"][0]
            selected = (
                work["warm_source"]
                if target == "source"
                else runtime.actor._learner
                if target == "actor"
                else runtime._candidate
            )
            if owner is selected._learner:
                changed.append(True)
                work["reports"][0]["training"][0]["collection"]["work"]["sampling_draws"] = 99
        return state

    monkeypatch.setattr(Native, "snapshot_state", read)
    report = pilot.run_pilot(warm_start=True)
    assert changed == [True]
    assert_one_paid_update(report, work)


def test_should_restore_full_boundary_checks_after_completed_warm_guard_raises(monkeypatch, ports):
    prepare_fixture(monkeypatch, ports)
    work = ports.work
    prepare, exit_sampler = pilot.prepare_warm_start, Sampler.__exit__
    raised: list[bool] = []
    boundary_refusals: list[str] = []
    arm_second_collection(work)

    def prepared(report, memory, deadline, check_boundary):
        work["warm_boundary"] = check_boundary
        return prepare(report, memory, deadline, check_boundary)

    def reject_once():
        if work.get("armed") and not raised:
            raised.append(True)
            raise RuntimeError("completed warm guard rejected the continuation")

    def sampler_exit(owner, *args):
        result = exit_sampler(owner, *args)
        if raised and not boundary_refusals:
            work["reports"][0]["training"][0]["collection"]["work"]["sampling_draws"] = 99
            try:
                work["warm_boundary"]()
            except ValueError as error:
                boundary_refusals.append(str(error))
        return result

    monkeypatch.setattr(pilot, "prepare_warm_start", prepared)
    monkeypatch.setattr(Sampler, "__exit__", sampler_exit)
    work["warm_guard_callback"] = reject_once
    report = pilot.run_pilot(warm_start=True)
    assert raised == [True] and len(boundary_refusals) == 1
    assert "preceding reward evidence" in boundary_refusals[0]
    assert "completed warm guard rejected" in report["reason"]
    assert_one_paid_update(report, work)


def test_should_keep_full_pending_payload_checks_during_initial_warm_preparation(
    monkeypatch, ports
):
    def prepare(report, memory, deadline, check_boundary):
        ports.work["reports"][0]["training"][7]["reason"] = "unobserved pending record replacement"
        check_boundary()
        pytest.fail("initial warm preparation must stop before native work")

    monkeypatch.setattr(pilot, "prepare_warm_start", prepare)
    report = pilot.run_pilot(warm_start=True)
    assert report["status"] == "incomplete" and report["wake_updates"] == 0
    assert report["total_new_native_updates"] == 0
    assert ports.work["runtimes"] == [] and ports.work["collections"] == []
    assert not ports.work.get("reward_calls")
