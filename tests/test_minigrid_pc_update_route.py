"""Scripted PC checkpoint composition; no learner updates or real environments."""

from copy import deepcopy
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_gradient_readouts as gradients
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, forbidden
from test_minigrid_learning_pilot import Learner, same_state
from test_minigrid_prior_gradient_readouts import sources


OPTIONS: dict[str, Any] = {**FLAGS, "qualification_method": "pc", "joint_catalogue": True}
CLI_OPTIONS = ["pc" if argument == "bp" else argument for argument in CLI_FLAGS] + [
    "--joint-catalogue"
]


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_describe_two_whole_catalogue_reads_without_changing_original_doses(
    no_resources, method
):
    options = {**OPTIONS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    assert original == pilot.describe_protocol(**options, pc_update_diagnostics=False)
    result = pilot.describe_protocol(**options, pc_update_diagnostics=True)
    assert result["methods"] == original["methods"] == [f"public_invariant_{method}"]
    assert result["probe"] == original["probe"]
    assert result["joint_catalogue"] == original["joint_catalogue"]
    assert result["pc_update_diagnostics"]["source_updates"] == [0, 128]
    for key, value in original["planned_work"].items():
        assert result["planned_work"][key] == value
    work = result["planned_work"]["pc_update_diagnostics"]
    assert work["catalogue_calls"] == 2
    assert work["projection_rows"] == 108
    assert work["projection_array_bytes"] == 255744
    assert work["prior_forward_calls"] == 4 and work["prior_forward_rows"] == 216
    assert work["prior_backward_calls"] == 2 and work["prior_backward_rows"] == 108
    assert work["relaxation_forward_calls"] == 6 and work["relaxation_forward_rows"] == 324
    assert work["native_train_calls"] == work["optimizer_steps"] == work["published_updates"] == 0


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_nonprimitive_flags_before_any_resource(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError, match="pc_update_diagnostics"):
            operation(**OPTIONS, pc_update_diagnostics=flag)


@pytest.mark.parametrize(
    "changed",
    [
        {"joint_catalogue": False},
        {"qualification_method": None},
        {"qualification_method": "bp"},
        {"qualification_method": "gated_cpc"},
        {"train_action_margins": True},
        {"public_history": False},
        {"invariant_history": False},
        {"distinct_probe": False},
        {"old_task_readouts": False},
        {"return_to_a": False},
    ],
)
def test_should_refuse_unsupported_recipe_before_resources(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**OPTIONS, **changed}, pc_update_diagnostics=True)


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_should_forward_only_explicit_cli_diagnostic_flag(monkeypatch, capsys, describe, enabled):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *CLI_OPTIONS]
        + (["--pc-update-diagnostics"] if enabled else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**OPTIONS, **({"pc_update_diagnostics": True} if enabled else {})}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        [item for item in CLI_OPTIONS if item != "--joint-catalogue"],
        ["bp" if item == "pc" else item for item in CLI_OPTIONS],
        ["gated_cpc" if item == "pc" else item for item in CLI_OPTIONS],
        [*CLI_OPTIONS, "--train-action-margins"],
    ],
)
def test_should_reject_invalid_cli_before_dispatch(monkeypatch, arguments):
    for name in ("describe_minigrid_learnability_protocol", "run_minigrid_learnability_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, "--pc-update-diagnostics", "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def native_payload(*, complete=True):
    work = dict.fromkeys(gradients.PC_NATIVE_WORK_KEYS, 0)
    work.update(
        snapshot_validations_attempted=1,
        snapshot_validations_completed=1,
        projection_calls_attempted=6,
        projection_calls_completed=6,
        projection_rows_attempted=54,
        projection_rows_completed=54,
        projection_array_bytes_returned=127872,
        prior_forward_calls_attempted=2,
        prior_forward_calls_completed=2 if complete else 1,
        prior_forward_rows_attempted=108,
        prior_forward_rows_completed=108 if complete else 54,
        prior_backward_calls_attempted=1,
        prior_backward_calls_completed=1,
        prior_backward_rows_attempted=54,
        prior_backward_rows_completed=54,
        relaxation_forward_calls_attempted=3,
        relaxation_forward_calls_completed=3,
        relaxation_forward_rows_attempted=162,
        relaxation_forward_rows_completed=162,
        candidate_parameter_sets_completed=1,
        failure_work_may_be_unobserved=not complete,
    )
    return dict(
        status="complete" if complete else "incomplete",
        reason=None if complete else "scripted final prior refusal",
        definition="torch_invariant_ungated_pc_realized_update_v1",
        rows=54,
        source_episode_ids=[packet.episode.episode_id for packet in sources()],
        work=work,
        prior_mean_ce=1.0,
        counterfactual_prior_mean_ce=1.1,
        prior_ce_change=0.1,
        gradient_delta_geometry={"gradient_dot_delta": 0.2},
    )


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(gradients, "_same_state", same_state)
    return dict(
        state={"weight": [1.0], "gradient": None},
        sources=sources(),
        report=dict(
            scores=[{"episode": index, "ce": 0.5} for index in range(6)],
            pc_update_diagnostics=[],
            work=dict(
                score_forward_calls_completed=6,
                score_rows_completed=54,
                pc_update_diagnostics=gradients.empty_pc_work(),
            ),
        ),
    )


def install_reader(monkeypatch, callback=None):
    calls = []

    def read(state, episodes):
        assert type(episodes) is tuple and len(episodes) == 6
        assert [episode.context for episode in episodes] == [
            "A",
            "A",
            "B",
            "B",
            "C_train",
            "C_train",
        ]
        assert all(
            episode.role == "train" and len(episode.transitions) == 9 for episode in episodes
        )
        call = {"state": state, "episodes": episodes, "payload": native_payload()}
        calls.append(call)
        if callback is not None:
            result = callback(call)
            if result is not None:
                return result
        return call["payload"]

    monkeypatch.setattr(gradients, "diagnose_pc_update", read)
    return calls


def run(case, boundary=lambda: None):
    gradients.run_pc_update_readouts(
        case["report"], case["state"], case["sources"], check_boundary=boundary
    )


def test_should_read_one_complete_catalogue_with_owned_payload_and_separate_costs(
    monkeypatch, case
):
    calls = install_reader(monkeypatch)
    before = pickle.dumps((case["state"], case["sources"]))
    run(case)
    result = case["report"]
    records = result["pc_update_diagnostics"]
    assert len(calls) == len(records) == 1
    assert records[0]["context"] == "joint_train_catalogue"
    assert records[0]["episode_ids"] == [p.episode.episode_id for p in case["sources"]]
    assert calls[0]["state"] is not case["state"]
    assert pickle.dumps((case["state"], case["sources"])) == before
    work = result["work"]["pc_update_diagnostics"]
    assert work["catalogue_calls_attempted"] == work["catalogue_calls_completed"] == 1
    assert work["train_rows_admitted"] == 54
    assert work["input_array_equivalent_bytes"] == 128304
    assert work["projection_rows_completed"] == 54
    assert work["prior_forward_rows_completed"] == 108
    assert work["prior_backward_rows_completed"] == 54
    assert work["relaxation_forward_rows_completed"] == 162
    assert work["native_train_calls"] == work["optimizer_steps"] == work["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is False
    assert result["work"]["score_rows_completed"] == 54
    calls[0]["payload"]["prior_mean_ce"] = 99.0
    assert records[0]["diagnostic"]["prior_mean_ce"] == 1.0


@pytest.mark.parametrize("failure", ["native", "incomplete", "generic"])
def test_should_preserve_all_paid_raw_scores_and_partial_native_evidence(
    monkeypatch, case, failure
):
    original = deepcopy(case["report"]["scores"])

    def fail(call):
        if failure == "generic":
            raise RuntimeError("unobserved return")
        partial = native_payload(complete=False)
        if failure == "native":
            raise GradientDiagnosticFailed(partial)
        return partial

    calls = install_reader(monkeypatch, fail)
    with pytest.raises((GradientDiagnosticFailed, ValueError, RuntimeError)):
        run(case)
    assert len(calls) == 1 and case["report"]["scores"] == original
    row = case["report"]["pc_update_diagnostics"][0]
    assert row["context"] == "joint_train_catalogue"
    if failure == "generic":
        assert row["diagnostic"] is None
    else:
        assert row["diagnostic"]["status"] == "incomplete"
    work = case["report"]["work"]["pc_update_diagnostics"]
    assert work["catalogue_calls_attempted"] == 1
    assert work["prior_forward_rows_completed"] == (0 if failure == "generic" else 54)
    assert work["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "state",
        "diagnostic_state",
        "sources",
        "scores",
        "score_owner",
        "work",
        "work_owner",
        "work_type",
        "record",
        "record_owner",
        "record_type",
        "boundary",
    ],
)
def test_should_restore_original_paid_evidence_when_callback_corrupts_it(monkeypatch, case, fault):
    result = case["report"]
    original = deepcopy(result["scores"])

    def corrupt(call):
        if fault == "state":
            case["state"]["weight"][0] = 9.0
        elif fault == "diagnostic_state":
            call["state"]["weight"][0] = 9.0
        elif fault == "sources":
            case["sources"][0].features[0, 0] = -0.8
        elif fault == "scores":
            result["scores"][0]["ce"] = 9.0
        elif fault == "score_owner":
            result["scores"] = deepcopy(result["scores"])
        elif fault == "work":
            result["work"]["pc_update_diagnostics"]["prior_forward_rows_completed"] = 999
        elif fault == "work_owner":
            result["work"] = deepcopy(result["work"])
        elif fault == "work_type":
            result["work"]["pc_update_diagnostics"]["catalogue_calls_attempted"] = 1.0
        elif fault == "record":
            result["pc_update_diagnostics"][0]["context"] = "other"
        elif fault == "record_owner":
            result["pc_update_diagnostics"][0] = deepcopy(result["pc_update_diagnostics"][0])
        elif fault == "record_type":
            result["pc_update_diagnostics"][0]["episode_ids"] = tuple(
                result["pc_update_diagnostics"][0]["episode_ids"]
            )

    calls = install_reader(monkeypatch, corrupt)

    def boundary():
        if calls and fault == "boundary":
            raise ValueError("preceding evidence changed")

    with pytest.raises(ValueError, match="evidence changed"):
        run(case, boundary)
    assert len(calls) == 1 and result["scores"] == original
    assert result["pc_update_diagnostics"][0]["context"] == "joint_train_catalogue"
    assert type(result["pc_update_diagnostics"][0]["episode_ids"]) is list
    assert result["pc_update_diagnostics"][0]["diagnostic"]["status"] == "complete"
    work = result["work"]["pc_update_diagnostics"]
    assert type(work["catalogue_calls_attempted"]) is int
    assert work["catalogue_calls_completed"] == 1 and work["prior_forward_rows_completed"] == 108
    assert work["failure_work_may_be_unobserved"] is True


def test_should_sum_pc_costs_without_changing_original_diagnostic_schema():
    first, second = gradients.empty_pc_work(), gradients.empty_pc_work()
    first.update(catalogue_calls_completed=1, prior_forward_rows_completed=108, seconds=0.25)
    second.update(catalogue_calls_attempted=1, seconds=0.5, failure_work_may_be_unobserved=True)
    result = gradients.sum_pc_work([first, second])
    assert result["catalogue_calls_completed"] == result["catalogue_calls_attempted"] == 1
    assert result["prior_forward_rows_completed"] == 108 and result["seconds"] == 0.75
    assert result["failure_work_may_be_unobserved"] is True
    assert "catalogue_calls_attempted" not in gradients.empty_work()
    assert "catalogue_calls_attempted" not in gradients.empty_prior_work()


@pytest.mark.parametrize("fault", ["payload_owner", "payload_type", "cost_owner", "cost_type"])
def test_should_refuse_late_replacement_of_completed_diagnostic_before_readmission(
    monkeypatch, case, fault
):
    calls = install_reader(monkeypatch)
    changed = False

    def boundary():
        nonlocal changed
        records = case["report"]["pc_update_diagnostics"]
        if changed or not records or records[0]["diagnostic"] is None:
            return
        changed = True
        record = records[0]
        if fault == "payload_owner":
            record["diagnostic"] = deepcopy(record["diagnostic"])
        elif fault == "payload_type":
            record["diagnostic"]["rows"] = 54.0
        elif fault == "cost_owner":
            record["diagnostic"]["work"] = deepcopy(record["diagnostic"]["work"])
        else:
            record["diagnostic"]["work"]["native_train_calls"] = False

    with pytest.raises(ValueError, match="evidence changed"):
        run(case, boundary)
    assert changed and len(calls) == 1
    diagnostic = case["report"]["pc_update_diagnostics"][0]["diagnostic"]
    assert type(diagnostic["rows"]) is int and diagnostic["rows"] == 54
    assert type(diagnostic["work"]["native_train_calls"]) is int
    assert diagnostic["work"]["native_train_calls"] == 0
    assert case["report"]["work"]["pc_update_diagnostics"]["catalogue_calls_completed"] == 1
    assert len(case["report"]["scores"]) == 6


@pytest.mark.parametrize("point", [0, 128])
@pytest.mark.parametrize("failure", [False, True])
def test_should_preserve_six_raw_scores_in_real_checkpoint_composition_with_scripted_reader(
    monkeypatch, point, failure
):
    monkeypatch.setattr(Learner, "work", {})
    monkeypatch.setattr(pilot, "_same_state", same_state)
    monkeypatch.setattr(gradients, "_same_state", same_state)
    monkeypatch.setattr(
        pilot, "_invariant_snapshot", lambda state, *args, **kwargs: deepcopy(state)
    )
    source = Learner(SimpleNamespace(method="pc"))
    state = source.snapshot_state()
    state.completed_updates, state.training_examples = point, 54 * point

    def read(call):
        if failure:
            raise GradientDiagnosticFailed(native_payload(complete=False))
        return None

    calls = install_reader(monkeypatch, read)
    result = pilot._score_checkpoint(
        source,
        state,
        method="public_invariant_pc",
        updates=point,
        sources=sources(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0),
        check_boundary=lambda: None,
        pc_update_diagnostics=True,
        catalogue_rows=54,
    )
    assert result["status"] == ("failed" if failure else "complete"), result["reason"]
    assert len(calls) == 1 and len(result["scores"]) == 6
    assert result["work"]["score_rows_completed"] == 54
    assert result["work"]["pc_update_diagnostics"]["prior_forward_rows_completed"] == (
        54 if failure else 108
    )
    if not failure:
        assert result["source_unchanged"] is result["state_unchanged"] is True
    else:
        assert result["pc_update_diagnostics"][0]["diagnostic"]["status"] == "incomplete"
        assert result["work"]["failure_work_may_be_unobserved"] is True
    assert not Learner.work.get("trains")
