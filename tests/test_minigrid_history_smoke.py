"""Resource-free CLI qualification for the fixed trained-history live case."""

import json
import sys
from typing import Any
from copy import deepcopy

import pytest

import scripts.run_minigrid_history_evaluation_smoke as cli
from test_torch_feature_rehearsal_pilot import forbidden
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub
from test_minigrid_learning_pilot import Learner, evaluator as evaluator, same_state
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
import src.app.toy_execution_budget as budgets


def test_should_describe_frozen_history_live_costs_without_constructing_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "evaluate_checkpoint"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(cli.shared, "AttributionStream", forbidden)
    monkeypatch.setattr(cli.shared, "_collect_sources", forbidden)
    protocol = cli.describe_protocol()
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc", "recurrent_bc"]
    assert protocol["feature_dim_by_method"] == {
        method: 148 if method == "recurrent_bc" else 296 for method in protocol["methods"]
    }
    assert protocol["checkpoints"] == [0, 32] and protocol["label_delay"] == 1
    assert protocol["seed"] == 17 and protocol["role"] == "train"
    assert protocol["final_episodes"] == protocol["probe_updates"] == 0
    assert protocol["planned_work"] == {
        "native_training_updates": 160,
        "training_rows": 1440,
        "raw_score_forwards": 30,
        "raw_score_rows": 270,
        "live_episodes": 30,
        "max_live_predictions": 480,
        "max_live_physics": 480,
        "teacher_episodes": 6,
        "expected_teacher_physics": 54,
        "max_teacher_physics": 96,
        "max_total_physics": 576,
        "max_environment_resets": 36,
        "recurrent_training_rows": 288,
        "recurrent_raw_rows": 54,
        "max_recurrent_live_steps": 96,
    }
    assert protocol["limits"]["body_seconds"] == 90
    assert protocol["limits"]["parent_seconds"] == 120
    assert protocol["limits"]["cpu_threads"] == protocol["limits"]["interop_threads"] == 1
    assert protocol["limits"]["gpu"] == 0


def test_should_route_describe_to_protocol_only_and_emit_finite_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["history-live", "--describe"])
    monkeypatch.setattr(cli, "run_smoke", forbidden)
    protocol = {"protocol_id": "fixed", "native_training_updates": 160}
    monkeypatch.setattr(cli, "describe_protocol", lambda: protocol)
    cli.main()
    assert json.loads(capsys.readouterr().out) == protocol


@pytest.mark.parametrize("status", ["complete", "incomplete"])
def test_should_preserve_run_evidence_and_signal_incomplete_case_exit(monkeypatch, capsys, status):
    monkeypatch.setattr(sys, "argv", ["history-live"])
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    result = {
        "status": status,
        "reason": None if status == "complete" else "original budget refused",
        "methods": [{"charged_updates": 1}],
        "evaluations": [],
    }
    calls = []

    def run():
        calls.append(True)
        return result

    monkeypatch.setattr(cli, "run_smoke", run)
    if status == "complete":
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
    assert calls == [True] and json.loads(capsys.readouterr().out) == result


def test_should_refuse_nonfinite_output_before_printing_or_claiming_success(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["history-live"])
    monkeypatch.setattr(cli, "run_smoke", lambda: {"status": "complete", "value": float("nan")})
    with pytest.raises(ValueError):
        cli.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("argument", ["--scenario", "--public-history", "--seed"])
def test_should_refuse_unfrozen_cli_controls_before_case_execution(monkeypatch, argument):
    monkeypatch.setattr(sys, "argv", ["history-live", argument])
    monkeypatch.setattr(cli, "run_smoke", forbidden)
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("fault", ["none", "prior_result", "source_evidence", "prior_cost"])
def test_should_run_existing_guarded_a32_streams_before_any_live_readout(
    monkeypatch, evaluator, fault
):
    monkeypatch.setattr(cli, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(cli, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(cli.shared, "_same_state", same_state)
    collection, training_rows = [], []
    previous: list[dict[str, Any]] = []

    def collect(check, reports):
        reports.append(dict(environment_steps=9))
        collection.append(reports)
        return deepcopy(evaluator.sources)

    monkeypatch.setattr(cli.shared, "_collect_sources", collect)
    train = cli.shared._train_arm

    def train_arm(row, *args, **kwargs):
        value = train(row, *args, **kwargs)
        training_rows.append(row)
        return value

    monkeypatch.setattr(cli.shared, "_train_arm", train_arm)

    def learner(method):
        policy = (
            TorchRecurrentPolicy(148, 8, 7, 17)
            if method == "recurrent_bc"
            else TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
        )
        return Learner(policy)

    monkeypatch.setattr(cli, "_history_learner", learner)
    evaluate = cli.evaluate_checkpoint

    def readout(*args, **kwargs):
        assert len(evaluator.work.get("trains", [])) == 160
        assert kwargs["updates"] == 32 and kwargs["public_history"] is True
        if previous:
            if fault == "prior_result":
                previous[0]["work"]["raw_score_rows_completed"] = 0
            elif fault == "source_evidence":
                collection[0][0]["environment_steps"] = 999
            elif fault == "prior_cost":
                training_rows[0]["charged_updates"] = 0
        value = evaluate(*args, **kwargs)
        previous.append(value)
        return value

    monkeypatch.setattr(cli, "evaluate_checkpoint", readout)
    report = cli.run_smoke()
    assert all(row["charged_updates"] is not None for row in report["methods"]), report["reason"]
    assert sum(row["charged_updates"] for row in report["methods"]) == 160
    if fault != "none":
        assert report["status"] == "incomplete" and report["reason"]
        assert report["evaluations"][0]["work"]["raw_score_rows_completed"] == 54
        assert report["source_reports"][0]["environment_steps"] == 9
        assert len(evaluator.work["environments"]) == 6
        return
    assert report["status"] == "complete", report["reason"]
    assert len(report["evaluations"]) == len(report["methods"]) == 5
    assert all(row["checkpoint_updates"] == [0, 32] for row in report["methods"])
    assert all(row["source_unchanged"] and row["actor_unchanged"] for row in report["methods"])
    assert sum(row["work"]["live_prediction_calls_returned"] for row in report["evaluations"]) == 60


@pytest.mark.parametrize("points", [(False, 32), (0, True), [0, 32], (0, 16), (0, 32, 64)])
def test_should_refuse_undeclared_training_checkpoints_before_learner_construction(
    monkeypatch, points
):
    monkeypatch.setattr(cli.shared, "_learner", forbidden)
    with pytest.raises(ValueError):
        cli.shared._train_arm({}, (), None, 1.0, forbidden, checkpoint_updates=points)
