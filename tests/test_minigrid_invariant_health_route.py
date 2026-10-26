"""Bound TRAIN health composes with paid scores and isolated probes, without learning."""

from dataclasses import replace
import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_invariant_learnability import case as case, train_sources
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_pc_qualification import FLAGS, METHOD, environment
from test_torch_feature_rehearsal_pilot import forbidden


def hidden(state, values):
    result = np.zeros((len(values), 8), dtype=np.float64)
    result[:, 0] = 1
    return result


@pytest.mark.parametrize("singleton,calls,ff,rnn", [(True, 30, 30, 0), (False, 150, 120, 270)])
def test_should_describe_selected_train_health_costs_without_resources(
    monkeypatch, singleton, calls, ff, rnn
):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    options = dict(public_history=True, invariant_history=True, pc_qualification=singleton)
    original = pilot.describe_protocol(**options)
    report = pilot.describe_protocol(**options, activation_health=True)
    assert (
        report["protocol_id"]
        == original["protocol_id"].removesuffix("_v1") + "_native_activation_health_v1"
    )
    assert report["methods"] == original["methods"]
    for key, value in original["planned_work"].items():
        assert report["planned_work"][key] == value + (
            rnn if key == "recurrent_unroll_steps" else 0
        )
    work = report["planned_work"]
    assert work["activation_calls"] == work["activation_svd_calls"] == calls
    assert work["activation_rows"] == 9 * calls and work["activation_array_bytes"] == 576 * calls
    assert work["activation_feedforward_hidden_affines"] == ff
    assert work["activation_recurrent_hidden_steps"] == rnn
    assert work["activation_projection_rows"] == 9 * ff
    assert work["activation_projection_array_bytes"] == 2368 * 9 * ff
    assert "no probe/final" in report["activation_health"]["scope"]


@pytest.mark.parametrize("describe", [True, False])
def test_should_dispatch_health_with_bound_singleton(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--pc-qualification",
            "--native-activation-health",
        ]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [dict(**FLAGS, activation_health=True)]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_should_finish_guarded_health_before_separate_probes(monkeypatch, evaluator):
    fixture = environment(monkeypatch, evaluator)
    calls = []

    def read(state, values):
        assert len(fixture.captured["finished"]) == 1
        assert not fixture.work["probes"]
        assert values.shape == (9, 296)
        calls.append(state.completed_updates)
        return hidden(state, values)

    monkeypatch.setattr(pilot, "read_native_activations", read)
    report = pilot.run_pilot(**FLAGS, activation_health=True)
    assert report["status"] == "complete", report["reason"]
    assert calls == [point for point in (0, 32, 64, 96, 128) for _ in range(6)]
    row = report["methods"][0]
    assert all(
        score["activation_health"]["status"] == "complete"
        for slot in row["readouts"]
        for score in slot["scores"]
    )
    assert all("activation_health" not in slot["report"] for slot in row["probes"])
    health = report["work"]["activation_health"]
    assert health["activation_calls_completed"] == health["activation_svd_calls_completed"] == 30
    assert (
        health["activation_rows_completed"] == health["activation_projection_rows_returned"] == 270
    )
    assert health["activation_array_bytes"] == 17280
    assert health["activation_projection_array_bytes_returned"] == 639360
    assert report["work"]["native_train_calls"] == 160  # simulated TRAIN/probe updates
    assert report["work"]["known_raw_score_forwards_completed"] == 30
    assert report["work"]["probe_score_forwards_completed"] == 16
    assert [call["budget"].updates_completed for call in fixture.work["probes"]] == [16, 16]
    assert row["source_unchanged"] and row["actor_unchanged"]


@pytest.mark.parametrize(
    "fault", ["reader", "shape", "source", "svd", "prior_score", "cost", "cost_type"]
)
def test_should_preserve_paid_bound_score_and_stop_on_health_fault(monkeypatch, case, fault):
    _, clock = case
    source = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
    )
    state = source.snapshot_state()
    sources = train_sources()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=16, max_wall_seconds=60), clock
    )
    reads = []

    def read(snapshot, values):
        reads.append(True)
        if fault == "reader":
            raise RuntimeError("reader failed")
        if fault == "source":
            changed = source.snapshot_state()
            changed.native_state.head_state["bias_hidden"][0] += 1
            source.restore_state(changed)
        return (
            hidden(snapshot, values)[:, :7].copy() if fault == "shape" else hidden(snapshot, values)
        )

    def svd(rows):
        raise RuntimeError("SVD failed")

    monkeypatch.setattr(pilot, "read_native_activations", read)
    if fault == "svd":
        monkeypatch.setattr(pilot, "diagnose_activations", svd)

    # A boundary callback may corrupt preceding published evidence. Access that
    # evidence through the function frame only in this adversarial test.
    def boundary():
        if not reads or fault not in ("prior_score", "cost", "cost_type"):
            return
        report = sys._getframe(2).f_locals["report"]
        if fault == "prior_score":
            report["scores"][0]["ce"] += 1
        else:
            report["work"]["activation_calls_completed"] = True if fault == "cost_type" else 900

    # Ordinary source/SVD/reader failures exercise the same public helper.
    report = pilot._score_checkpoint(
        source,
        state,
        method=METHOD,
        updates=0,
        sources=sources,
        budget=budget,
        check_boundary=boundary,
        activation_health=True,
    )
    assert report["status"] == "failed", report
    assert reads == [True]
    assert len(report["scores"]) == 1 and budget.updates_completed == 0
    work = report["work"]
    assert work["score_forward_calls_completed"] == work["activation_calls_attempted"] == 1
    assert work["activation_calls_completed"] == (0 if fault == "reader" else 1)
    assert work["activation_projection_rows_returned"] == (0 if fault == "reader" else 9)
    assert work["failure_work_may_be_unobserved"] is (fault in ("reader", "svd"))
    json.dumps(report, allow_nan=False)


def test_should_reject_final_input_before_health_or_score(monkeypatch, case):
    _, clock = case
    source = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
    )
    sources = train_sources()
    first = replace(
        sources[0], role="final_test", episode=replace(sources[0].episode, role="final_test")
    )
    monkeypatch.setattr(pilot, "read_native_activations", forbidden)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "predict", forbidden)
    with pytest.raises(ValueError):
        pilot._score_checkpoint(
            source,
            source.snapshot_state(),
            method=METHOD,
            updates=0,
            sources=(first, *sources[1:]),
            budget=ToyBudgetSession(
                ToyExecutionBudget(max_training_updates=16, max_wall_seconds=60), clock
            ),
            check_boundary=lambda: None,
            activation_health=True,
        )


@pytest.mark.parametrize("fault", ["value", "record", "list"])
def test_should_reject_reader_time_changes_before_sealing_health(monkeypatch, case, fault):
    from copy import deepcopy

    _, clock = case
    source = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
    )
    reads = []
    original_score = []

    def read(state, values):
        report = sys._getframe(1).f_locals["report"]
        reads.append(True)
        if len(reads) == 2:
            original_score.append(deepcopy(report["scores"][0]))
            if fault == "value":
                report["scores"][0]["ce"] += 1
            elif fault == "record":
                report["scores"][0] = deepcopy(report["scores"][0])
            else:
                report["scores"] = list(report["scores"])
        return hidden(state, values)

    monkeypatch.setattr(pilot, "read_native_activations", read)
    report = pilot._score_checkpoint(
        source,
        source.snapshot_state(),
        method=METHOD,
        updates=0,
        sources=train_sources(),
        budget=ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=16, max_wall_seconds=60), clock
        ),
        check_boundary=lambda: None,
        activation_health=True,
    )
    assert report["status"] == "failed" and len(reads) == 2
    assert report["scores"][0] == original_score[0]
    assert len(report["scores"]) == 2 and "activation_health" not in report["scores"][1]
    work = report["work"]
    assert work["score_forward_calls_completed"] == work["activation_calls_completed"] == 2
    assert work["activation_svd_calls_completed"] == 1
    assert work["activation_feedforward_hidden_affines_completed"] == 2
    assert work["activation_projection_rows_returned"] == 18
    assert not work["failure_work_may_be_unobserved"]


def test_should_execute_real_zero_update_train_health_with_bound_reader(monkeypatch, case):
    from src.adapters.torch_activation_diagnostics import read_native_activations
    from src.adapters.torch_feature_rehearsal_pilot import _same_state

    _, clock = case
    source = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
    )
    state = source.snapshot_state()
    sources = train_sources()
    monkeypatch.setattr(pilot, "read_native_activations", read_native_activations)
    report = pilot._score_checkpoint(
        source,
        state,
        method=METHOD,
        updates=0,
        sources=sources,
        budget=ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=16, max_wall_seconds=60), clock
        ),
        check_boundary=lambda: None,
        activation_health=True,
    )
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] and report["state_unchanged"]
    assert _same_state(source.snapshot_state(), state)
    assert source.completed_updates == source.training_examples == 0
    work = report["work"]
    assert (
        work["activation_calls_completed"]
        == work["activation_svd_calls_completed"]
        == work["activation_feedforward_hidden_affines_completed"]
        == 6
    )
    assert work["activation_rows_completed"] == work["activation_projection_rows_returned"] == 54
    assert work["activation_projection_array_bytes_returned"] == 127872
    for packet, score in zip(sources, report["scores"], strict=True):
        np.testing.assert_array_equal(
            np.array(score["activation_health"]["activations"]),
            read_native_activations(state, packet.features),
        )
