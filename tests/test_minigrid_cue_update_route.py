"""Optional finite-update composition preserves accepted warm and raw evidence."""

import importlib
import json
from types import SimpleNamespace

import pytest

import scripts.run_minigrid_gradient_smoke as cli
from src.adapters import minigrid_warm_gradient_pilot as pilot
import src.shared.torch_runtime as torch_runtime
from test_minigrid_gradient_route import install_port
from test_minigrid_warm_gradient_pilot import case as case, ports as ports
from test_minigrid_warm_reward import Native
from test_torch_feature_rehearsal_pilot import forbidden


def install_cues(monkeypatch, case, *, fail=False, after=None):
    calls = []

    def run(report, source, state, sources, baseline, budget, *, check_boundary):
        assert source is case.work["prepared"]["source"]
        assert state.completed_updates == 128 and state.training_examples == 1152
        assert sources is case.sources and baseline["status"] == "complete"
        assert len(baseline["scores"]) == 6 and len(baseline["cue_gradient_geometry"]) == 3
        assert budget.budget.max_training_updates == 0 and budget.updates_completed == 0
        calls.append(dict(source=source, state=state, baseline=baseline))
        report.update(status="running", arms=[])
        check_boundary()
        report.update(
            status="incomplete" if fail else "complete",
            reason="second update refused" if fail else None,
            arms=[dict(cue="blue", status="complete")],
            work=dict(
                native_updates_committed=1 if fail else 2,
                committed_training_rows=9 if fail else 18,
                failure_work_may_be_unobserved=fail,
            ),
        )
        check_boundary()
        if after is not None:
            after()

    component = importlib.import_module("src.adapters.minigrid_cue_updates")
    monkeypatch.setattr(component, "run_cue_updates", run)
    return calls


def test_should_run_two_cue_arms_only_after_accepted_base_and_charge_separately(monkeypatch, case):
    install_port(monkeypatch)
    calls = install_cues(monkeypatch, case)
    report = pilot.run_pilot(cue_updates=True)
    assert report["status"] == "complete", report["reason"]
    assert len(calls) == 1
    assert report["protocol"]["protocol_id"] == "minigrid_warm_a128_isolated_cue_updates_v1"
    assert report["work"]["native_train_calls"] == 130
    assert report["work"]["training_rows"] == 1170
    assert report["diagnostic_budget"]["updates_completed"] == 0
    assert report["diagnostic_budget"]["limits"]["max_wall_seconds"] == 20.0
    assert case.work["warm_budget"].updates_completed == 128
    assert case.work["warm_deadline"] == 55.0


def test_should_retain_partial_cue_work_and_full_paid_baseline_on_refusal(monkeypatch, case):
    install_port(monkeypatch)
    install_cues(monkeypatch, case, fail=True)
    report = pilot.run_pilot(cue_updates=True)
    assert report["status"] == "incomplete"
    assert report["cue_update_readout"]["reason"] == "second update refused"
    assert report["work"]["native_train_calls"] == 129
    assert report["work"]["training_rows"] == 1161
    assert report["work"]["raw_scores"]["score_rows_completed"] == 54
    assert report["work"]["cue_gradient_geometry"]["forward_rows_completed"] == 54
    assert report["work"]["failure_work_may_be_unobserved"] is True


def test_should_refuse_late_cue_evidence_tamper_without_adopting_forged_charges(monkeypatch, case):
    install_port(monkeypatch)
    armed = [False]
    install_cues(monkeypatch, case, after=lambda: armed.__setitem__(0, True))
    snapshot = Native.snapshot_state

    def mutate(owner):
        state = snapshot(owner)
        if armed[0] and owner is case.work["warm_source"]._learner:
            armed[0] = False
            case.work["pilot_reports"][0]["cue_update_readout"]["work"][
                "native_updates_committed"
            ] = 1000
        return state

    monkeypatch.setattr(Native, "snapshot_state", mutate)
    report = pilot.run_pilot(cue_updates=True)
    assert report["status"] == "incomplete"
    assert report["work"]["native_train_calls"] == 130
    assert report["cue_update_readout"]["work"]["native_updates_committed"] == 2


def test_should_leave_default_route_without_cue_work(monkeypatch, case):
    install_port(monkeypatch)
    component = importlib.import_module("src.adapters.minigrid_cue_updates")
    monkeypatch.setattr(component, "run_cue_updates", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert "cue_update_readout" not in report
    assert "isolated_cue_updates" not in report["work"]
    assert report["work"]["native_train_calls"] == 128


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_should_refuse_non_boolean_opt_in_before_work(monkeypatch, value):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError, match="exact bool"):
        pilot.run_pilot(cue_updates=value)
    with pytest.raises(ValueError, match="exact bool"):
        pilot.describe_protocol(cue_updates=value)


def test_should_describe_both_arm_doses_and_costs_without_native_setup(monkeypatch, capsys):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    assert cli.main(["--describe", "--cue-update-check"]) == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol == pilot.describe_protocol(cue_updates=True)
    assert protocol["planned_work"]["native_train_calls"] == 130
    assert protocol["planned_work"]["training_rows"] == 1170
    assert protocol["planned_work"]["isolated_raw_rows"] == 108
    assert protocol["planned_work"]["isolated_update_seconds"] == 20.0


def test_should_route_actual_cli_flag_to_explicit_pilot_option(monkeypatch, capsys):
    torch = SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None)
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(status="complete", work={"native_train_calls": 130})

    monkeypatch.setattr(pilot, "run_pilot", run)
    assert cli.main(["--cue-update-check"]) == 0
    assert calls == [dict(cue_updates=True)]
    assert json.loads(capsys.readouterr().out)["work"]["native_train_calls"] == 130
