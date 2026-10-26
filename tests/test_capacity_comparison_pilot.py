"""Complete frozen cohort recipe, isolation and its existing CLI composition."""

from dataclasses import asdict
import sys
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import capacity_comparison_pilot as pilot
from src.adapters.numpy_capacity import measure_numpy_capacity
from src.app import toy_execution_budget as budget_module
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.local_pilot_budget import PilotRequest, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSegment


def test_should_describe_every_frozen_cohort_and_work_limit_without_native_construction(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "build_learner", lambda *a: pytest.fail("native construction"))
    monkeypatch.setattr(pilot, "ProcessRssSampler", lambda: pytest.fail("physical RSS"))
    description = pilot.describe_protocol()
    rows = pilot._rows()
    assert len(rows) == len({(r["method"], r["mode"]) for r in rows}) == 23
    assert all(r["status"] == "not_run" for r in rows)
    assert description["planned_work"]["native_updates"] == 414
    assert description["planned_work"]["training_rows"] == 3312
    assert description["planned_work"]["final_score_forwards"] == 184
    assert sum(r["initialization_work"]["splits"] for r in rows) == 15
    assert sum(r["initialization_work"]["prunes"] for r in rows) == 4
    assert description["removed_units"] == [2] and description["pruned_width"] == 3
    assert "differs" in description["initialization"]
    assert (
        asdict(validate_local_pilot_request(PilotRequest(pilot.RESOURCES)).resources)
        == description["resources"]
    )


def test_should_reserve_validation_independently_from_training_and_final_data():
    validation = pilot.build_validation()
    training, final = pilot.build_stream("recurring")
    assert [p.role for p in validation] == ["validation", "validation"]
    assert [len(p.features) for p in validation] == [4, 4]
    reserved = {i for p in validation for i in p.row_ids}
    assert not reserved.intersection(i for p in (*training, *final) for i in p.row_ids)
    for axis, batch in enumerate(validation):
        np.testing.assert_array_equal(batch.targets, (batch.features[:, [axis]] >= 0).astype(float))
        for other in (*training, *final):
            assert not np.shares_memory(batch.features, other.features)
            assert not np.shares_memory(batch.targets, other.targets)
    again = pilot.build_validation()
    np.testing.assert_array_equal(validation[0].features, again[0].features)
    validation[0].features.fill(99)
    assert not np.any(again[0].features == 99)


@pytest.mark.parametrize(
    "method,mode,parameters,auxiliary",
    [
        ("bp", "fixed_pruned_width", 13, 0),
        ("pc", "ordinary_fixed", 17, 0),
        ("gated_cpc", "bounded_renewal", 17, 160),
    ],
)
def test_should_compose_existing_native_ports_with_complete_declared_payloads(
    method, mode, parameters, auxiliary
):
    learner = pilot.build_learner(method, mode)
    state = learner.snapshot_state()
    usage = measure_numpy_capacity(state)
    assert usage.parameters == parameters and usage.auxiliary_bytes == auxiliary
    assert (type(state) is CoupledStructuralState) == (mode in pilot.RAW_MODES)
    if mode == "bounded_renewal":
        assert state.unit_history.updates == 0
        assert state.renewal_policy.max_importance == 0.001
        assert state.wake_state.wake_updates == 0


@pytest.mark.parametrize(
    "method,mode", [("other", "fixed_wide"), ("bp", "bounded_renewal"), ("pc", "unknown")]
)
def test_should_refuse_unsupported_native_cases_before_factory_work(monkeypatch, method, mode):
    monkeypatch.setattr(
        pilot.capacity_pilot, "build_learner", lambda *a: pytest.fail("native factory")
    )
    with pytest.raises(ValueError):
        pilot.build_learner(method, mode)


class FakeSampler:
    start_bytes = 1

    def sample(self):
        return 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def snapshot(self):
        return ProcessRssSegment(1, 1, 1, 1, 0.005)


def test_should_execute_the_actual_eighteen_wake_pruning_cohort_without_tuning_the_gate(
    monkeypatch,
):
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    row = next(r for r in pilot._rows() if r["method"] == "pc" and r["mode"] == "split_prune")
    pilot._run_one(row, pilot.monotonic() + 30, cast(Any, FakeSampler()))
    assert row["status"] == "complete", row.get("reason")
    assert row["source_learner_unchanged"] and row["committed_updates"] == 18
    assert row["committed_training_rows"] == 144
    widths = [p["width"] for p in row["native_checkpoints"]]
    assert widths[:2] == [2, 4] and widths[2] in (3, 4) and widths[3] == widths[2]
    assert row["capacity"]["completed_native_calls"] == 18
    assert row["capacity"]["peak_parameters"] == 17
    assert row["capacity"]["mean_parameters"] == (13 if widths[2] == 3 else 43 / 3)
    event = row["capacity"]["pruning"]["events"][0]
    assert event["status"] in ("complete", "rejected")
    assert event["decision_status"] in ("accepted", "rejected")
    assert row["stream_result"]["score_forward_calls"] == 8
    assert set(row["metrics"]) == {"new_B_gain", "retention_loss", "return_gain"}


def test_should_keep_every_unreached_cohort_when_the_sampler_fails(monkeypatch):
    monkeypatch.setattr(
        pilot, "ProcessRssSampler", lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
    )
    monkeypatch.setattr(pilot, "build_learner", lambda *a: pytest.fail("native factory"))
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(report["runs"]) == 23
    assert report["wake_updates"] == 0
    assert all(r["status"] == "not_run" for r in report["runs"])
    assert all(r["status"] == "unavailable" and r["metrics"] is None for r in report["comparisons"])
    assert report["full_R26_complete"] is report["scientific_advantage_verified"] is False


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_route_only_the_requested_complete_capacity_recipe(monkeypatch, capsys, action):
    import scripts.run_stream_rehearsal_pilot as cli

    expected = {"status": "complete", "protocol": "complete frozen recipe"}
    selected = (
        "describe_capacity_comparison_protocol"
        if action == "describe"
        else "run_capacity_comparison_pilot"
    )
    monkeypatch.setattr(cli, selected, lambda: expected)
    monkeypatch.setattr(cli, "run_capacity_pilot", lambda: pytest.fail("old native case"))
    monkeypatch.setattr(cli, "run_bp_pilot", lambda: pytest.fail("old BP case"))
    monkeypatch.setattr(sys, "argv", ["pilot", "--" + action, "--scenario", "capacity-comparison"])
    assert cli.main() == 0
    assert "complete frozen recipe" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--recent-only", "--diagnose-learnability"])
def test_should_refuse_unfrozen_cli_flags_before_native_work(monkeypatch, flag):
    import scripts.run_stream_rehearsal_pilot as cli

    monkeypatch.setattr(cli, "run_capacity_comparison_pilot", lambda: pytest.fail("native work"))
    monkeypatch.setattr(sys, "argv", ["pilot", "--run", "--scenario", "capacity-comparison", flag])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
