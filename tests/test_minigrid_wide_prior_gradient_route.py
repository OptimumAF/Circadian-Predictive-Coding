"""Scripted H32 TRAIN prior-gradient composition, without models or physics.

Reuse the existing scripted H32 snapshot port and original six TRAIN sources.
These checks prove route composition and paid-work isolation, not tensor math,
acquisition, causal interference or a circadian advantage.
"""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import random
import sys

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_gradient_readouts as gradients
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, METHODS, forbidden
from test_minigrid_learning_pilot import Learner
from test_minigrid_prior_gradient_readouts import install_reader, payload
from test_minigrid_wide_pc_diagnostics_route import case as case, prepare


OPTIONS = {**FLAGS, "joint_catalogue": True, "wide_neural": True}


@pytest.mark.parametrize("method", METHODS)
def test_should_admit_actual_public_runtime_before_any_native_work(
    monkeypatch, no_resources, method
):
    calls = []

    def reached_resource_boundary():
        calls.append(method)
        raise RuntimeError("scripted runtime resource boundary reached")

    monkeypatch.setattr(pilot, "require_torch", reached_resource_boundary)
    result = pilot.run_pilot(
        **{**OPTIONS, "qualification_method": method, "prior_gradient_geometry": True}
    )
    assert calls == [method]
    assert result["status"] == "incomplete"
    assert "scripted runtime resource boundary reached" in result["reason"]
    assert result["work"]["native_train_calls"] == 0


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


def run(case, boundary=lambda: None, budget=None, **changed):
    return pilot._score_checkpoint(
        case["source"],
        case["state"],
        **{
            "method": case["method"],
            "updates": case["state"].completed_updates,
            "sources": case["sources"],
            "budget": budget
            or ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0),
            "check_boundary": boundary,
            "prior_gradient_geometry": True,
            "catalogue_rows": 54,
            "wide_neural": True,
            **changed,
        },
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_extra_train_costs_without_changing_h32_learning_recipe(
    no_resources, method
):
    options = {**OPTIONS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    assert original == pilot.describe_protocol(**options, prior_gradient_geometry=False)
    result = pilot.describe_protocol(**options, prior_gradient_geometry=True)
    for key in (
        "methods",
        "hidden_dim",
        "seed",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "checkpoint_updates",
        "probe",
        "joint_catalogue",
        "wide_neural",
        "trainable_parameters",
        "primary_comparisons",
    ):
        assert result[key] == original[key]
    for key, value in original["planned_work"].items():
        assert result["planned_work"][key] == value
    diagnostic = result["prior_gradient_geometry"]
    assert diagnostic["methods"] == [f"public_invariant_{method}"]
    assert "not native PC" in diagnostic["objective"]
    assert "descriptive" in diagnostic["interpretation"]
    work = result["planned_work"]["prior_gradient_geometry"]
    assert work["pair_calls"] == work["snapshot_validations"] == 15
    assert work["forward_calls"] == work["backward_calls"] == work["projection_calls"] == 30
    assert work["forward_rows"] == work["backward_rows"] == work["projection_rows"] == 270
    assert work["projection_array_bytes"] == 639360
    assert work["geometry_reductions"] == 75
    assert work["optimizer_steps"] == work["published_updates"] == 0


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_should_admit_and_forward_explicit_wide_prior_cli(monkeypatch, capsys, describe, method):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    arguments = [method if argument == "bp" else argument for argument in CLI_FLAGS]
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments, "--joint-catalogue", "--wide-neural", "--prior-gradient-geometry"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**OPTIONS, "qualification_method": method, "prior_gradient_geometry": True}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "changed",
    [
        {"wide_neural": False},
        {"qualification_method": "pc_relax64"},
        {"qualification_method": "neutral_cpc_relax64"},
        {"activation_health": True},
        {"gradient_geometry": True},
        {"pc_update_diagnostics": True},
        {"bp_update_diagnostics": True},
        {"train_action_margins": True},
        {"train_fit_bounds": True},
        {"old_a_rehearsal": True},
        {"extra_f_exposure": True},
        {"probe_rollouts": True},
        {"deferred_probe_rollouts": True},
        {"train_a_rollout": True},
        {"weighted_catalogue": True},
        {"probe_inference": "fixed-weighted"},
    ],
)
def test_should_refuse_h8_joint_and_additional_wide_combinations_before_resources(
    no_resources, changed
):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**OPTIONS, "prior_gradient_geometry": True, **changed})


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_should_refuse_nonboolean_prior_option_before_resources(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**OPTIONS, prior_gradient_geometry=flag)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("point", [0, 32, 64, 96, 128])
def test_should_score_all_six_train_sources_before_three_owned_prior_pairs(
    monkeypatch, case, method, point
):
    prepare(case, method, point)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0)
    before = pickle.dumps((case["source"].snapshot_state(), case["state"], case["sources"]))
    binding = (
        id(budget),
        id(budget.budget),
        id(budget.clock),
        budget.updates_completed,
        budget.replay_examples_completed,
    )

    def observe(call, index):
        assert len(Learner.work["predicts"]) == 6
        assert call["state"] is not case["state"]
        assert call["state"].policy.hidden_dim == 32
        assert call["state"].completed_updates == point
        assert call["state"].training_examples == 54 * point
        assert not Learner.work.get("trains")

    calls = install_reader(monkeypatch, callback=observe)
    result = run(case, budget=budget)
    assert result["status"] == "complete", result["reason"]
    assert [call["pair"][0].context for call in calls] == ["A", "B", "C_train"]
    assert len({id(call["state"]) for call in calls}) == 3
    assert len(result["scores"]) == 6 and result["work"]["score_rows_completed"] == 54
    assert result["expected_training_examples"] == 54 * point
    assert result["source_unchanged"] is result["state_unchanged"] is True
    work = result["work"]["prior_gradient_geometry"]
    assert work["pair_calls_attempted"] == work["pair_calls_completed"] == 3
    assert work["forward_rows_completed"] == work["backward_rows_completed"] == 54
    assert work["projection_rows_completed"] == work["train_rows_admitted"] == 54
    assert work["projection_array_bytes_returned"] == 127872
    assert work["geometry_reductions_completed"] == 15
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is False
    assert binding == (
        id(budget),
        id(budget.budget),
        id(budget.clock),
        budget.updates_completed,
        budget.replay_examples_completed,
    )
    assert pickle.dumps((case["source"].snapshot_state(), case["state"], case["sources"])) == before
    assert (point, 54 * point) in case["admissions"]
    assert not Learner.work.get("trains")
    calls[0]["payload"]["joint_geometry"]["cosine"] = 999
    assert result["prior_gradient_geometry"][0]["diagnostic"]["joint_geometry"]["cosine"] == -0.5


@pytest.mark.parametrize("failure", ["native", "incomplete", "generic"])
def test_should_preserve_paid_first_pair_and_partial_second_pair_after_failure(
    monkeypatch, case, failure
):
    prepare(case, point=128)

    def fail(call, index):
        if index == 1:
            if failure == "generic":
                raise RuntimeError("scripted unobserved diagnostic return")
            partial = payload(complete=False)
            if failure == "native":
                raise GradientDiagnosticFailed(partial)
            return partial

    calls = install_reader(monkeypatch, callback=fail)
    result = run(case)
    assert result["status"] == "failed" and len(calls) == 2
    assert len(result["scores"]) == 6 and result["work"]["score_rows_completed"] == 54
    records = result["prior_gradient_geometry"]
    assert [record["context"] for record in records] == ["A", "B"]
    assert records[0]["diagnostic"]["status"] == "complete"
    assert (records[1]["diagnostic"] is None) is (failure == "generic")
    if failure != "generic":
        assert records[1]["diagnostic"]["status"] == "incomplete"
    work = result["work"]["prior_gradient_geometry"]
    assert work["pair_calls_attempted"] == 2
    assert (
        work["forward_rows_completed"]
        == work["projection_rows_completed"]
        == (18 if failure == "generic" else 27)
    )
    assert work["geometry_reductions_completed"] == 5
    assert work["failure_work_may_be_unobserved"] is True
    assert result["work"]["failure_work_may_be_unobserved"] is True
    assert not Learner.work.get("trains")


@pytest.mark.parametrize(
    "fault", ["source", "checkpoint", "diagnostic_state", "input", "boundary", "clock"]
)
def test_should_refuse_mutated_state_inputs_or_guards_after_preserving_paid_pair(
    monkeypatch, case, fault
):
    prepare(case)
    now = [1.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_wall_seconds=2), lambda: now[0]
    )

    def corrupt(call, index):
        if fault == "source":
            case["source"].state.parameters["weight"][0, 0] = 9.0
        elif fault == "checkpoint":
            case["state"].parameters["weight"][0, 0] = 9.0
        elif fault == "diagnostic_state":
            call["state"].parameters["weight"][0, 0] = 9.0
        elif fault == "input":
            case["sources"][0].features[0, 0] = -0.8
        elif fault == "clock":
            now[0] = 4.0

    calls = install_reader(monkeypatch, callback=corrupt)

    def boundary():
        if calls and fault == "boundary":
            raise ValueError("original stream evidence changed")

    result = run(case, boundary, budget)
    assert result["status"] == "failed" and len(calls) == 1
    assert len(result["scores"]) == 6
    assert result["prior_gradient_geometry"][0]["diagnostic"]["status"] == "complete"
    assert result["work"]["prior_gradient_geometry"]["pair_calls_completed"] == 1
    assert budget.updates_completed == 0 and not Learner.work.get("trains")


@pytest.mark.parametrize("fault", ["late_cost", "late_payload", "late_source_ids"])
def test_should_refuse_late_corruption_without_renewing_paid_work(monkeypatch, case, fault):
    prepare(case)
    calls = install_reader(monkeypatch)
    original = gradients.run_prior_gradient_readouts
    holder = {}

    def observe(report, *args, **options):
        holder["report"] = report
        return original(report, *args, **options)

    monkeypatch.setattr(gradients, "run_prior_gradient_readouts", observe)
    changed = False

    def boundary():
        nonlocal changed
        report = holder.get("report")
        if changed or report is None or not report["prior_gradient_geometry"]:
            return
        record = report["prior_gradient_geometry"][0]
        if record["diagnostic"] is None:
            return
        changed = True
        if fault == "late_cost":
            report["work"]["prior_gradient_geometry"]["forward_rows_completed"] = 999
        elif fault == "late_payload":
            record["diagnostic"]["joint_geometry"]["cosine"] = 999
        else:
            record["episode_ids"][0] = "final-test-replacement"

    result = run(case, boundary)
    assert changed and result["status"] == "failed" and "evidence changed" in result["reason"]
    assert len(calls) == 1 and len(result["scores"]) == 6
    record = result["prior_gradient_geometry"][0]
    assert record["diagnostic"]["joint_geometry"]["cosine"] == -0.5
    assert record["episode_ids"][0] == "minigrid-source-A-blue"
    assert result["work"]["prior_gradient_geometry"]["forward_rows_completed"] == 18


def test_should_preserve_original_rng_boundary_guard(monkeypatch, case):
    prepare(case)
    rng = random.Random(17)
    initial = rng.getstate()
    calls = install_reader(monkeypatch, callback=lambda call, index: (rng.random(), None)[1])

    def boundary():
        if rng.getstate() != initial:
            raise ValueError("original RNG witness changed")

    result = run(case, boundary)
    assert result["status"] == "failed" and "RNG witness changed" in result["reason"]
    assert len(calls) == 1 and len(result["scores"]) == 6
    assert result["work"]["prior_gradient_geometry"]["pair_calls_completed"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "value",
        "list_owner",
        "row_owner",
        "diagnostic_owner",
        "cost_owner",
        "cost_value",
        "cost_type",
        "native_cost_owner",
    ],
)
def test_should_refuse_replaced_completed_prior_evidence_during_publication_tail(
    monkeypatch, case, fault
):
    prepare(case)
    calls = install_reader(monkeypatch)
    original = gradients.run_prior_gradient_readouts
    holder = {}

    def observe(report, *args, **options):
        original(report, *args, **options)
        holder["report"] = report

    monkeypatch.setattr(gradients, "run_prior_gradient_readouts", observe)
    changed = False

    def boundary():
        nonlocal changed
        if changed or not holder:
            return
        changed = True
        report = holder["report"]
        records = report["prior_gradient_geometry"]
        if fault == "value":
            records[0]["diagnostic"]["joint_geometry"]["cosine"] = 999
        elif fault == "list_owner":
            report["prior_gradient_geometry"] = deepcopy(records)
        elif fault == "row_owner":
            records[0] = deepcopy(records[0])
        elif fault == "diagnostic_owner":
            records[0]["diagnostic"] = deepcopy(records[0]["diagnostic"])
        elif fault == "cost_owner":
            report["work"]["prior_gradient_geometry"] = deepcopy(
                report["work"]["prior_gradient_geometry"]
            )
        elif fault == "cost_type":
            report["work"]["prior_gradient_geometry"]["forward_rows_completed"] = 54.0
        elif fault == "native_cost_owner":
            records[0]["diagnostic"]["work"] = deepcopy(records[0]["diagnostic"]["work"])
        else:
            report["work"]["prior_gradient_geometry"]["forward_rows_completed"] = 999

    result = run(case, boundary)
    assert changed and result["status"] == "failed"
    assert "evidence" in result["reason"] or "paid probe report" in result["reason"]
    assert len(calls) == 3 and len(result["scores"]) == 6
    records = result["prior_gradient_geometry"]
    assert [record["context"] for record in records] == ["A", "B", "C_train"]
    assert all(record["diagnostic"]["joint_geometry"]["cosine"] == -0.5 for record in records)
    assert result["work"]["prior_gradient_geometry"]["forward_rows_completed"] == 54
    assert type(result["work"]["prior_gradient_geometry"]["forward_rows_completed"]) is int
    assert result["work"]["prior_gradient_geometry"]["pair_calls_completed"] == 3
    assert not Learner.work.get("trains")


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_nontrain_source_before_any_scoring_or_diagnostic(monkeypatch, case, role):
    prepare(case)
    first = case["sources"][0]
    case["sources"] = (
        replace(first, role=role, episode=replace(first.episode, role=role)),
        *case["sources"][1:],
    )
    calls = install_reader(monkeypatch)
    with pytest.raises(ValueError):
        run(case)
    assert not calls and not Learner.work.get("predicts")


@pytest.mark.parametrize("fault", ["rows", "updates"])
def test_should_refuse_wrong_h32_native_clock_before_any_read(monkeypatch, case, fault):
    prepare(case)
    calls = install_reader(monkeypatch)
    if fault == "rows":
        case["state"].training_examples = 1
    else:
        case["state"].completed_updates = True
    with pytest.raises(ValueError):
        run(case)
    assert not calls and not Learner.work.get("predicts")


@pytest.mark.parametrize(
    "changed",
    [
        {"updates": 1},
        {"updates": True},
        {"catalogue_rows": 9},
        {"catalogue_rows": 54.0},
        {"wide_neural": 1},
        {"prior_gradient_geometry": 1},
        {"activation_health": True},
        {"gradient_geometry": True},
        {"pc_update_diagnostics": True},
        {"weighted_catalogue": True},
        {"train_action_margins": True},
    ],
)
def test_should_refuse_mixed_private_route_before_snapshot_or_work(monkeypatch, case, changed):
    prepare(case)
    calls = install_reader(monkeypatch)
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    with pytest.raises(ValueError):
        run(case, **changed)
    assert not calls and not Learner.work.get("predicts")
