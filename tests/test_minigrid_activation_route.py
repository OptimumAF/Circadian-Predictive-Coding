"""TRAIN-only native health composes with the frozen history/probe route."""

from dataclasses import replace
import json
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from test_minigrid_history_route import history_environment
from test_minigrid_learning_pilot import (
    Learner,
    METHODS,
    POINTS,
    evaluator as evaluator,
    same_state,
)
from test_torch_feature_rehearsal_pilot import forbidden


def hidden_rows(state, values):
    result = np.zeros((len(values), 8), dtype=np.float64)
    result[:, 0] = 1.0
    result[:, 1] = state.completed_updates / 256
    return result


def test_should_add_health_only_to_known_train_cells_without_changing_training_or_probes(
    monkeypatch, evaluator
):
    case = history_environment(monkeypatch, evaluator)
    activations: list[dict[str, Any]] = []

    def read(state, values):
        assert len(case.captured["finished"]) == 5
        assert (
            state.completed_updates in POINTS
            and state.training_examples == 9 * state.completed_updates
        )
        assert values.dtype == np.float64 and values.shape == (9, state.policy.feature_dim)
        # The reader receives neither released source arrays nor any probe data.
        for _, _, packets, _ in case.captured["streams"]:
            assert not any(np.shares_memory(values, packet.features) for packet in packets)
        result = hidden_rows(state, values)
        activations.append(dict(state=state, values=values, result=result))
        return result

    monkeypatch.setattr(pilot, "read_native_activations", read)
    report = pilot.run_pilot(public_history=True, activation_health=True)
    assert report["status"] == "complete", report["reason"]
    assert len(case.work["trains"]) == 800 and len(case.work["predicts"]) == 230
    assert len(case.raw) == 25 and len(case.probes) == 10 and len(activations) == 150
    assert not case.work.get("sessions") and not case.work.get("environments")
    assert len(case.work["probe_environments"]) == 4
    assert report["comparisons"] is not None and report["scientific_advantage_verified"] is False
    call = 0
    for row, (_, source, _, budget) in zip(
        report["methods"], case.captured["streams"], strict=True
    ):
        assert row["method"] in METHODS
        assert budget.updates_completed == row["charged_updates"] == 128
        assert source.state.completed_updates == 0
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert row["final_evaluation"]["status"] == "not_run"
        for slot in row["readouts"]:
            assert slot["role"] == "train" and slot["status"] == "complete"
            assert slot["work"]["score_forward_calls_completed"] == 6
            assert slot["work"]["activation_calls_completed"] == 6
            assert slot["work"]["activation_rows_completed"] == 54
            assert slot["work"]["activation_array_bytes"] == 3456
            assert slot["work"]["activation_svd_calls_completed"] == 6
            for score in slot["scores"]:
                assert score["context"] in ("A", "B", "C_train")
                assert score["ce"] == pytest.approx(float(np.log(2)))
                assert score["accuracy"] == score["exact_sequence_accuracy"] == 0
                health = score["activation_health"]
                assert health["status"] == "complete"
                np.testing.assert_array_equal(health["activations"], activations[call]["result"])
                assert activations[call]["state"].completed_updates == slot["updates"]
                metrics = health["metrics"][0]
                assert metrics["rows"] == 9 and metrics["units"] == 8
                assert metrics["dead_fraction"] == (7 / 8 if slot["updates"] == 0 else 6 / 8)
                assert metrics["saturated_fraction"] == 1 / 8
                assert metrics["mean_absolute_activity"] == pytest.approx(
                    (1 + slot["updates"] / 256) / 8
                )
                assert metrics["feature_effective_rank"] == 0
                call += 1
        assert all("activation_health" not in slot["report"] for slot in row["probes"])
    work = report["work"]
    assert work["native_train_calls"] == 800 and work["training_rows"] == 7200
    assert work["known_raw_score_forwards_completed"] == 150
    assert work["probe_score_forwards_completed"] == 80
    health_work = work["activation_health"]
    assert (
        health_work["activation_calls_attempted"]
        == health_work["activation_calls_completed"]
        == 150
    )
    assert health_work["activation_rows_completed"] == 1350
    assert health_work["activation_array_bytes"] == 86400
    assert health_work["activation_svd_calls_completed"] == 150
    assert health_work["activation_feedforward_hidden_affines_completed"] == 120
    assert health_work["activation_recurrent_hidden_steps_completed"] == 270
    assert health_work["activation_recurrent_output_projections_completed"] == 270
    # Published lists own values even after a reader-owned result changes.
    activations[0]["result"][:] = -1
    assert (
        report["methods"][0]["readouts"][0]["scores"][0]["activation_health"]["activations"][0][0]
        == 1
    )


def raw_case(monkeypatch, evaluator):
    monkeypatch.setattr(pilot, "_same_state", same_state)
    source = Learner(TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
    sources = pilot._history_packets(evaluator.sources)
    monkeypatch.setattr(pilot, "read_native_activations", hidden_rows)
    return source, sources


@pytest.mark.parametrize(
    "fault", ["native", "shape", "dtype", "nonfinite", "range", "source", "boundary", "svd"]
)
def test_should_preserve_paid_prediction_and_stop_after_first_health_fault(
    monkeypatch, evaluator, fault
):
    source, sources = raw_case(monkeypatch, evaluator)
    calls: list[bool] = []

    def read(state, values):
        calls.append(True)
        if fault == "native":
            raise RuntimeError("activation failed")
        result = hidden_rows(state, values)
        if fault == "shape":
            return result[:, :-1].copy()
        if fault == "dtype":
            return result.astype(np.float32)
        if fault == "nonfinite":
            result[0, 0] = np.nan
        if fault == "range":
            result[0, 0] = 1.1
        if fault == "source":
            source.state.parameters["weight"] += 1
        return result

    def guard():
        if calls and fault == "boundary":
            raise ValueError("completed evidence changed")

    def failed_svd(layers):
        raise RuntimeError("SVD failed")

    monkeypatch.setattr(pilot, "read_native_activations", read)
    if fault == "svd":
        monkeypatch.setattr(pilot, "diagnose_activations", failed_svd)
    report = pilot._score_checkpoint(
        source,
        source.snapshot_state(),
        method="bp",
        updates=0,
        sources=sources,
        budget=evaluator.budget,
        check_boundary=guard,
        activation_health=True,
    )
    assert report["status"] == "failed" and report["reason"] and calls == [True]
    assert len(evaluator.work["predicts"]) == 1 and not evaluator.work.get("trains")
    assert len(report["scores"]) == 1
    paid_score = report["scores"][0]
    assert paid_score["ce"] == pytest.approx(float(np.log(2)))
    if fault in ("native", "shape", "dtype", "nonfinite", "range"):
        assert "activation_health" not in paid_score
    else:
        assert paid_score["activation_health"]["status"] == "read"
        assert paid_score["activation_health"]["metrics"] is None
        assert len(paid_score["activation_health"]["activations"]) == 9
    assert evaluator.budget.updates_completed == 0
    work = report["work"]
    assert work["score_forward_calls_attempted"] == work["score_forward_calls_completed"] == 1
    assert work["score_rows_completed"] == 9 and work["activation_calls_attempted"] == 1
    assert work["activation_rows_attempted"] == 9
    assert work["activation_calls_completed"] == (0 if fault == "native" else 1)
    if fault == "svd":
        assert work["activation_svd_calls_attempted"] == 1
        assert work["activation_svd_calls_completed"] == 0
    if fault in ("native", "svd"):
        assert work["failure_work_may_be_unobserved"] is True
    json.dumps(report, allow_nan=False)


def test_should_refuse_final_role_before_any_prediction_or_activation(monkeypatch, evaluator):
    source, sources = raw_case(monkeypatch, evaluator)
    first = replace(
        sources[0], role="final_test", episode=replace(sources[0].episode, role="final_test")
    )
    monkeypatch.setattr(pilot, "read_native_activations", forbidden)
    report = pilot._score_checkpoint(
        source,
        source.snapshot_state(),
        method="bp",
        updates=0,
        sources=(first, *sources[1:]),
        budget=evaluator.budget,
        check_boundary=lambda: None,
        activation_health=True,
    )
    assert report["status"] == "failed" and "TRAIN" in report["reason"]
    assert not evaluator.work.get("predicts") and report["work"]["activation_calls_attempted"] == 0


def test_should_preserve_old_raw_schema_when_health_is_not_requested(monkeypatch, evaluator):
    source, sources = raw_case(monkeypatch, evaluator)
    monkeypatch.setattr(pilot, "read_native_activations", forbidden)
    report = pilot._score_checkpoint(
        source,
        source.snapshot_state(),
        method="bp",
        updates=0,
        sources=sources,
        budget=evaluator.budget,
        check_boundary=lambda: None,
    )
    assert report["status"] == "complete" and len(report["scores"]) == 6
    assert all("activation_health" not in score for score in report["scores"])
    assert not any(key.startswith("activation_") for key in report["work"])


def test_should_describe_health_without_resources_and_preserve_explicit_false(monkeypatch):
    for name in (
        "require_torch",
        "ProcessRssSampler",
        "MiniGridKeyDoor",
        "read_native_activations",
        "diagnose_activations",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    original = pilot.describe_protocol(public_history=True)
    assert original == pilot.describe_protocol(public_history=True, activation_health=False)
    assert "activation_health" not in original
    protocol = pilot.describe_protocol(public_history=True, activation_health=True)
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["planned_work"]["native_train_calls"] == 800
    assert protocol["planned_work"]["activation_calls"] == 150
    assert protocol["planned_work"]["activation_array_bytes"] == 86400
    assert protocol["planned_work"]["activation_recurrent_output_projections"] == 270


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_health_before_resources(monkeypatch, invalid):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(public_history=True, activation_health=invalid)
    with pytest.raises(ValueError):
        pilot.run_pilot(public_history=True, activation_health=invalid)


def test_should_require_history_for_native_health_before_resources(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    with pytest.raises(ValueError, match="public_history"):
        pilot.describe_protocol(activation_health=True)
    with pytest.raises(ValueError, match="public_history"):
        pilot.run_pilot(activation_health=True)


@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_health_only_on_the_explicit_history_cli(monkeypatch, capsys, describe):
    args = [
        "pilot",
        "--scenario",
        "torch-minigrid-history-learnability",
        "--native-activation-health",
    ]
    monkeypatch.setattr(sys, "argv", args + (["--describe"] if describe else []))
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    cli.main()
    assert calls == [dict(public_history=True, activation_health=True)]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize("scenario", ["torch-minigrid-learnability", "torch-minigrid-cue-key-door"])
def test_should_refuse_health_on_other_cli_scenarios_before_dispatch(monkeypatch, scenario):
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario, "--native-activation-health"]
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_pilot", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
