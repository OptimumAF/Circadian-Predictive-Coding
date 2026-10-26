"""An explicit engineering allowance never renews completed training budgets.

These tests reuse fake ports and a small saved protocol, not historical model
execution. Scientific data, dose, seed, metrics and directions remain fixed.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_minigrid_reward_smoke as cli
from src.adapters import minigrid_reward_pilot as pilot
from src.adapters import minigrid_warm_readouts as diagnostics
import src.shared.torch_runtime as torch_runtime
from test_minigrid_momentum_pilot import paired as paired
from test_minigrid_warm_readout_pilot import readout_case as readout_case
from test_minigrid_warm_reward import ports as ports
from test_torch_feature_rehearsal_pilot import forbidden


OPTIONS = dict(warm_start=True, train_readouts=True, momentum_control=True)
FLAGS = ["--warm-start", "--train-readouts", "--momentum-control"]


@pytest.fixture
def revised(monkeypatch, paired):
    prepare = pilot.prepare_warm_start

    def legacy_fake(report, memory, deadline, check_boundary):
        assert deadline == 310.0
        # Adapt only the old fake's assertion; real caller/control/readout
        # guards still receive their original newly declared310 deadline.
        return prepare(report, memory, 190.0, check_boundary)

    monkeypatch.setattr(pilot, "prepare_warm_start", legacy_fake)
    return paired


def run():
    return pilot.run_pilot(**OPTIONS, extended_control_readouts=True)


def helper_call(flag, *, control_state=None):
    return diagnostics.run_train_diagnostics(
        {},
        object(),
        ((128, object()), (136, object())),
        (),
        object(),
        310.0,
        training_phase={},
        check_boundary=forbidden,
        control_state=control_state,
        extended_control_readouts=flag,
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_revision_before_any_resources(monkeypatch, value):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**OPTIONS, extended_control_readouts=value)
    with pytest.raises(ValueError):
        pilot.run_pilot(**OPTIONS, extended_control_readouts=value)
    with pytest.raises(ValueError):
        helper_call(value, control_state=object())


@pytest.mark.parametrize(
    "options", [{}, {"warm_start": True}, {"warm_start": True, "train_readouts": True}]
)
def test_should_require_existing_full_control_route_before_resources(monkeypatch, options):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options, extended_control_readouts=True)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options, extended_control_readouts=True)


def test_should_require_owned_control_endpoint_before_extended_helper_work(monkeypatch):
    monkeypatch.setattr(diagnostics, "evaluate_checkpoint", forbidden)
    with pytest.raises(ValueError, match="control endpoint"):
        helper_call(True)


def test_should_preserve_saved_v1_protocol_and_all_default_descriptors(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/r21-minigrid-momentum-compose-20261016/describe.json"
    )
    payload = path.read_bytes()
    assert len(payload) < 64 * 1024
    receipt = json.loads(payload)
    assert receipt["code"] == 0 and receipt["stderr"] == ""
    accepted = json.loads(receipt["stdout"])
    current = pilot.describe_protocol(**OPTIONS)
    assert current == accepted
    assert current == pilot.describe_protocol(**OPTIONS, extended_control_readouts=False)
    assert current["limits"]["train_readout_phase_seconds"] == 90.0
    assert current["limits"]["body_seconds"] == 180.0 and current["limits"]["parent_seconds"] == 210
    for options in ({}, {"warm_start": True}, {"warm_start": True, "train_readouts": True}):
        assert pilot.describe_protocol(**options) == pilot.describe_protocol(
            **options, extended_control_readouts=False
        )
    assert path.read_bytes() == payload


def test_should_change_only_version_limits_and_explicit_engineering_allowance(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    before = pilot.describe_protocol(**OPTIONS)
    revised = pilot.describe_protocol(**OPTIONS, extended_control_readouts=True)
    assert revised["protocol_id"] == "minigrid_warm_reward_momentum_control_v2"
    assert type(revised["engineering_allowance"]) is str and revised["engineering_allowance"]
    normalized = deepcopy(revised)
    normalized.pop("engineering_allowance")
    normalized["protocol_id"] = before["protocol_id"]
    normalized["limits"] = before["limits"]
    assert normalized == before
    expected_limits = dict(
        before["limits"], body_seconds=300.0, parent_seconds=330, train_readout_phase_seconds=210.0
    )
    assert revised["limits"] == expected_limits
    assert (
        revised["limits"]["reward_phase_seconds"]
        == revised["limits"]["control_phase_seconds"]
        == 45
    )
    assert revised["planned_work"] == before["planned_work"]


def test_should_finish_same_training_and_readouts_beyond_old_90_seconds_with_frozen_phase_clocks(
    revised,
):
    case, work = revised.case, revised.work
    work["during_paired_readout"] = lambda index: case.clock.__setitem__(0, 110.0)
    report = run()
    assert report["status"] == "complete", report["reason"]
    assert report["protocol"]["protocol_id"] == "minigrid_warm_reward_momentum_control_v2"
    assert len(work["controls"]) == len(work["reward_calls"]) == 8
    assert len(work["readouts"]) == 3
    assert [readout["method"] for readout in work["readouts"]] == [
        "recurrent_warm_reward",
        "recurrent_warm_reward",
        "recurrent_warm_momentum_control",
    ]
    assert [readout["state"].completed_updates for readout in work["readouts"]] == [128, 136, 136]
    assert [readout["state"].training_examples for readout in work["readouts"]] == [
        1152,
        1168,
        1168,
    ]
    budget = work["readouts"][0]["budget"]
    assert all(readout["budget"] is budget for readout in work["readouts"])
    assert (
        budget.budget.max_wall_seconds == 210
        and budget.started_at == 10.0
        and budget.last_clock == 110.0
    )
    reward, control = work["budget"], work["control_runtimes"][0]._budget
    assert reward.budget.max_wall_seconds == control.budget.max_wall_seconds == 45
    assert reward.updates_completed == control.updates_completed == 8
    assert (
        reward.started_at == reward.last_clock == control.started_at == control.last_clock == 10.0
    )
    assert report["total_new_native_updates"] == 144 and report["total_new_training_rows"] == 1184
    assert report["train_diagnostics"]["work"]["raw_score_rows_completed"] == 162
    assert report["train_diagnostics"]["control_minus_reward"] is not None


def test_should_cap_extended_readout_by_remaining_original_whole_deadline(monkeypatch, revised):
    prepare = pilot.prepare_warm_start

    def late_prefix(*args):
        result = prepare(*args)
        revised.case.clock[0] = 200.0
        return result

    monkeypatch.setattr(pilot, "prepare_warm_start", late_prefix)
    report = run()
    assert report["status"] == "complete", report["reason"]
    budget = revised.work["readouts"][0]["budget"]
    assert budget.started_at == 200.0 and budget.budget.max_wall_seconds == 110.0
    assert revised.work["budget"].last_clock == 200.0
    assert revised.work["control_runtimes"][0]._budget.last_clock == 200.0
    assert report["total_new_native_updates"] == 144


@pytest.mark.parametrize("clock", [221.0, 311.0])
def test_should_stop_without_renewal_when_readout_or_whole_v2_deadline_expires(revised, clock):
    def expire(index):
        if index == 2:
            revised.case.clock[0] = clock

    revised.work["during_paired_readout"] = expire
    report = run()
    result = report["train_diagnostics"]
    assert report["status"] == result["status"] == "incomplete"
    assert report["total_new_native_updates"] == 144 and report["wake_updates"] == 8
    assert len(revised.work["reward_calls"]) == len(revised.work["controls"]) == 8
    assert all(record["status"] == "complete" for record in result["readouts"][:2])
    assert result["readouts"][2]["status"] != "complete"
    assert result["work"]["raw_score_rows_completed"] == 108
    assert result["control_minus_reward"] is None and result["control_endpoint_changes"] is None
    assert (
        revised.work["budget"].last_clock
        == revised.work["control_runtimes"][0]._budget.last_clock
        == 10.0
    )


@pytest.mark.parametrize("flags", [[], ["--warm-start"], ["--warm-start", "--train-readouts"]])
def test_should_refuse_cli_revision_without_full_control_route_before_native_import(
    monkeypatch, flags
):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main([*flags, "--extended-control-readouts"])
    assert error.value.code == 2


def test_should_describe_v2_cli_without_native_resources(monkeypatch, capsys):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    assert cli.main([*FLAGS, "--extended-control-readouts", "--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == pilot.describe_protocol(
        **OPTIONS, extended_control_readouts=True
    )


@pytest.mark.parametrize("extended", [False, True])
def test_should_forward_only_explicit_revision_cli_option(monkeypatch, capsys, extended):
    calls: list[dict] = []

    def run_stub(**options):
        calls.append(options)
        return {"status": "complete"}

    torch = SimpleNamespace(
        set_num_threads=lambda count: None, set_num_interop_threads=lambda count: None
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    monkeypatch.setattr(pilot, "run_pilot", run_stub)
    flags = [*FLAGS, "--extended-control-readouts"] if extended else FLAGS
    assert cli.main(flags) == 0
    expected = dict(OPTIONS, extended_control_readouts=True) if extended else OPTIONS
    assert calls == [expected] and json.loads(capsys.readouterr().out)["status"] == "complete"
