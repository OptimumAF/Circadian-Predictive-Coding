"""Weighted joint PC checkpoint diagnostics using scripted, model-free ports."""

from copy import deepcopy
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_gradient_readouts as gradients
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_joint_catalogue_route import forbidden
from test_minigrid_learning_pilot import Learner, same_state
from test_minigrid_pc_update_route import CLI_OPTIONS, OPTIONS, native_payload
from test_minigrid_prior_gradient_readouts import sources


WEIGHTED_OPTIONS = {**OPTIONS, "weighted_catalogue": True}
WEIGHTED_CLI = [*CLI_OPTIONS, "--weighted-catalogue"]
EXPECTED_WEIGHTS = (9 / 14, 9 / 14, *(9 / 7 for _ in range(5)))
EXTRA_COUNTERS = (
    "proposal_calls",
    "proposal_rows",
    "proposal_nominal_latent_row_iterations",
    "fixed_state_energy_calls",
    "fixed_state_energy_rows",
)


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_describe_weighted_reads_with_original_benchmark_and_probe_doses(
    no_resources, method
):
    options = {**WEIGHTED_OPTIONS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    result = pilot.describe_protocol(**options, pc_update_diagnostics=True)
    assert original == pilot.describe_protocol(**options, pc_update_diagnostics=False)
    assert result["methods"] == original["methods"] == [f"public_invariant_{method}"]
    assert result["probe"] == original["probe"]
    assert result["return_phase"] == original["return_phase"]
    assert result["joint_catalogue"] == original["joint_catalogue"]
    assert result["weighted_catalogue"] == original["weighted_catalogue"]
    assert result["pc_update_diagnostics"]["source_updates"] == [0, 128]
    assert (
        result["pc_update_diagnostics"]["definition"]
        == "torch_invariant_weighted_pc_realized_update_v1"
    )
    for key, value in original["planned_work"].items():
        assert result["planned_work"][key] == value
    work = result["planned_work"]["pc_update_diagnostics"]
    for key, value in dict(
        catalogue_calls=2,
        projection_rows=108,
        proposal_calls=2,
        proposal_rows=108,
        proposal_nominal_latent_row_iterations=216,
        fixed_state_energy_calls=2,
        fixed_state_energy_rows=108,
        weight_derivations=2,
        weight_rows=108,
        relaxation_forward_calls=0,
        relaxation_forward_rows=0,
        native_train_calls=0,
        optimizer_steps=0,
        published_updates=0,
    ).items():
        assert work[key] == value, key
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_nonprimitive_weighted_flags_before_resources(no_resources, value):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(
                **{**WEIGHTED_OPTIONS, "weighted_catalogue": value}, pc_update_diagnostics=True
            )


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
        {"weighted_bp": True},
        {"weighted_pc": True},
    ],
)
def test_should_refuse_incompatible_weighted_recipe_before_models(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**WEIGHTED_OPTIONS, **changed}, pc_update_diagnostics=True)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_both_explicit_cli_options(monkeypatch, capsys, method, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    arguments = [method if item == "pc" else item for item in WEIGHTED_CLI]
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments, "--pc-update-diagnostics"] + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [
        {**WEIGHTED_OPTIONS, "qualification_method": method, "pc_update_diagnostics": True}
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "arguments",
    [
        [item for item in WEIGHTED_CLI if item != "--joint-catalogue"],
        ["bp" if item == "pc" else item for item in WEIGHTED_CLI],
        ["gated_cpc" if item == "pc" else item for item in WEIGHTED_CLI],
        [*WEIGHTED_CLI, "--weighted-pc"],
        [*WEIGHTED_CLI, "--weighted-bp"],
        [*WEIGHTED_CLI, "--train-action-margins"],
    ],
)
def test_should_refuse_invalid_cli_without_dispatch(monkeypatch, arguments):
    for name in ("describe_minigrid_learnability_protocol", "run_minigrid_learnability_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, "--pc-update-diagnostics", "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def weighted_payload(*, complete=True):
    payload = native_payload(complete=complete)
    payload.update(
        definition="torch_invariant_weighted_pc_realized_update_v1",
        class_weights=list(EXPECTED_WEIGHTS),
        proposal=dict(
            weighted_energy=1.2, nominal_latent_row_iterations=108, hidden_displacement_l2=0.3
        ),
        prior_weighted_mean_ce=1.0,
        counterfactual_prior_weighted_mean_ce=1.1,
        weighted_prior_ce_change=0.1,
    )
    work = payload["work"]
    for name in tuple(work):
        if name.startswith("relaxation_forward_"):
            work[name] = 0
    for group, amount in zip(EXTRA_COUNTERS, (1, 54, 108, 1, 54), strict=True):
        work[f"{group}_attempted"] = amount
        work[f"{group}_completed"] = amount if complete else 0
    return payload


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
                pc_update_diagnostics=gradients.empty_pc_work(weighted=True),
            ),
        ),
    )


def install_reader(monkeypatch, callback=None):
    calls = []

    def read(state, episodes, *, class_weights=None):
        assert type(episodes) is tuple and len(episodes) == 6
        assert type(class_weights) is tuple and len(class_weights) == 7
        assert class_weights == EXPECTED_WEIGHTS
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
        call = dict(
            state=state, episodes=episodes, class_weights=class_weights, payload=weighted_payload()
        )
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
        case["report"], case["state"], case["sources"], check_boundary=boundary, weighted=True
    )


def test_should_derive_aggregate_train_weights_and_keep_separate_paid_work(monkeypatch, case):
    calls = install_reader(monkeypatch)
    before = pickle.dumps((case["state"], case["sources"]))
    run(case)
    assert len(calls) == 1
    assert calls[0]["state"] is not case["state"]
    assert pickle.dumps((case["state"], case["sources"])) == before
    record = case["report"]["pc_update_diagnostics"][0]
    assert record["context"] == "joint_train_catalogue"
    assert record["diagnostic"]["class_weights"] == list(EXPECTED_WEIGHTS)
    work = case["report"]["work"]["pc_update_diagnostics"]
    for group, amount in zip(
        (*EXTRA_COUNTERS, "weight_derivations", "weight_rows"),
        (1, 54, 108, 1, 54, 1, 54),
        strict=True,
    ):
        assert work[f"{group}_attempted"] == work[f"{group}_completed"] == amount
    assert work["catalogue_calls_completed"] == 1 and work["train_rows_admitted"] == 54
    assert work["native_train_calls"] == work["optimizer_steps"] == work["published_updates"] == 0
    assert case["report"]["work"]["score_rows_completed"] == 54
    calls[0]["payload"]["class_weights"][0] = 99.0
    assert record["diagnostic"]["class_weights"] == list(EXPECTED_WEIGHTS)


@pytest.mark.parametrize("failure", ["native", "incomplete", "generic"])
def test_should_retain_scores_derivation_and_partial_proposal_costs(monkeypatch, case, failure):
    original = deepcopy(case["report"]["scores"])

    def fail(call):
        if failure == "generic":
            raise RuntimeError("unobserved proposal return")
        partial = weighted_payload(complete=False)
        if failure == "native":
            raise GradientDiagnosticFailed(partial)
        return partial

    calls = install_reader(monkeypatch, fail)
    with pytest.raises((GradientDiagnosticFailed, ValueError, RuntimeError)):
        run(case)
    assert len(calls) == 1 and case["report"]["scores"] == original
    work = case["report"]["work"]["pc_update_diagnostics"]
    assert work["weight_derivations_completed"] == 1 and work["weight_rows_completed"] == 54
    assert work["proposal_calls_attempted"] == (0 if failure == "generic" else 1)
    assert work["proposal_calls_completed"] == 0
    assert work["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "state",
        "diagnostic_state",
        "sources",
        "scores",
        "score_owner",
        "work_owner",
        "new_counter_type",
    ],
)
def test_should_refuse_callback_changes_and_preserve_paid_evidence(monkeypatch, case, fault):
    original = deepcopy(case["report"]["scores"])

    def corrupt(call):
        result = case["report"]
        if fault == "state":
            case["state"]["weight"][0] = 99.0
        elif fault == "diagnostic_state":
            call["state"]["weight"][0] = 99.0
        elif fault == "sources":
            case["sources"][0].features[0, 0] = -0.8
        elif fault == "scores":
            result["scores"][0]["ce"] = 99.0
        elif fault == "score_owner":
            result["scores"] = deepcopy(result["scores"])
        elif fault == "work_owner":
            result["work"] = deepcopy(result["work"])
        else:
            result["work"]["pc_update_diagnostics"]["weight_derivations_completed"] = 1.0

    calls = install_reader(monkeypatch, corrupt)
    with pytest.raises(ValueError, match="evidence changed"):
        run(case)
    assert len(calls) == 1 and case["report"]["scores"] == original
    assert case["report"]["pc_update_diagnostics"][0]["diagnostic"]["status"] == "complete"
    work = case["report"]["work"]["pc_update_diagnostics"]
    assert type(work["weight_derivations_completed"]) is int
    assert work["weight_derivations_completed"] == work["proposal_calls_completed"] == 1
    assert work["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "weights_owner",
        "weights_type",
        "proposal_owner",
        "proposal_type",
        "proposal_counter_type",
        "payload_owner",
    ],
)
def test_should_refuse_replacement_of_completed_weighted_payload_before_readmission(
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
        payload = records[0]["diagnostic"]
        if fault == "weights_owner":
            payload["class_weights"] = list(payload["class_weights"])
        elif fault == "weights_type":
            payload["class_weights"] = tuple(payload["class_weights"])
        elif fault == "proposal_owner":
            payload["proposal"] = deepcopy(payload["proposal"])
        elif fault == "proposal_type":
            payload["proposal"]["nominal_latent_row_iterations"] = 108.0
        elif fault == "proposal_counter_type":
            payload["work"]["proposal_calls_completed"] = 1.0
        else:
            records[0]["diagnostic"] = deepcopy(payload)

    with pytest.raises(ValueError, match="evidence changed"):
        run(case, boundary)
    assert changed and len(calls) == 1
    payload = case["report"]["pc_update_diagnostics"][0]["diagnostic"]
    assert type(payload["class_weights"]) is list
    assert type(payload["proposal"]["nominal_latent_row_iterations"]) is int
    assert type(payload["work"]["proposal_calls_completed"]) is int
    assert case["report"]["work"]["pc_update_diagnostics"]["proposal_calls_completed"] == 1


def test_should_sum_weighted_costs_without_extending_original_schema():
    first, second = gradients.empty_pc_work(weighted=True), gradients.empty_pc_work(weighted=True)
    first.update(weight_derivations_completed=1, proposal_rows_completed=54, seconds=0.25)
    second.update(
        weight_derivations_attempted=1,
        proposal_rows_attempted=54,
        seconds=0.5,
        failure_work_may_be_unobserved=True,
    )
    total = gradients.sum_pc_work([first, second], weighted=True)
    assert total["weight_derivations_completed"] == total["weight_derivations_attempted"] == 1
    assert total["proposal_rows_completed"] == total["proposal_rows_attempted"] == 54
    assert total["seconds"] == 0.75 and total["failure_work_may_be_unobserved"] is True
    assert "proposal_rows_completed" not in gradients.empty_pc_work()
    assert gradients.empty_pc_work() == gradients.empty_pc_work(weighted=False)


@pytest.mark.parametrize("point", [0, 128])
@pytest.mark.parametrize("failure", [False, True])
def test_should_compose_weighted_checkpoint_after_all_six_paid_scores(monkeypatch, point, failure):
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
            raise GradientDiagnosticFailed(weighted_payload(complete=False))
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
        weighted_catalogue=True,
    )
    assert result["status"] == ("failed" if failure else "complete"), result["reason"]
    assert len(calls) == 1 and len(result["scores"]) == 6
    assert result["work"]["score_rows_completed"] == 54
    assert result["work"]["pc_update_diagnostics"]["weight_rows_completed"] == 54
    assert result["work"]["pc_update_diagnostics"]["proposal_calls_completed"] == (
        0 if failure else 1
    )
    assert not Learner.work.get("trains")


@pytest.mark.parametrize("port", ["prepare_train_catalogue", "catalogue_class_weights"])
def test_should_charge_failed_weight_derivation_before_any_numerical_call(monkeypatch, case, port):
    calls = install_reader(monkeypatch)
    original = deepcopy(case["report"]["scores"])

    def refuse(*args):
        raise ValueError("scripted weight derivation refused")

    monkeypatch.setattr(gradients, port, refuse)
    with pytest.raises(GradientDiagnosticFailed):
        run(case)
    assert not calls
    assert case["report"]["scores"] == original
    work = case["report"]["work"]["pc_update_diagnostics"]
    assert work["weight_derivations_attempted"] == 1
    assert work["weight_rows_attempted"] == 54
    assert work["weight_derivations_completed"] == work["weight_rows_completed"] == 0
    assert work["proposal_calls_attempted"] == work["proposal_calls_completed"] == 0
    assert work["failure_work_may_be_unobserved"] is True
    assert case["report"]["pc_update_diagnostics"][0]["diagnostic"]["status"] == "incomplete"
