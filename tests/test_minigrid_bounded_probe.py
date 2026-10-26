"""Public weighted probe recipes and original isolated F/old-A/A native routing."""

from copy import deepcopy
from dataclasses import asdict
import json
import pickle
import random
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_bounded_probe as recipes
from src.adapters import minigrid_learnability as pilot
from src.adapters import torch_pc_inference as backend
from src.adapters import torch_train_weighted_learner as scheduled
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core import categorical_pc_inference as core
from src.core.learner_ports import TrainingUpdateFailed
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import case as case
from test_minigrid_relax64_native import training_sources
from test_torch_feature_learner import assert_tree
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


MODES = ("fixed-weighted", "bounded-weighted")
FLAGS = dict(
    public_history=True,
    invariant_history=True,
    qualification_method="pc",
    joint_catalogue=True,
    distinct_probe=True,
    old_task_readouts=True,
    return_to_a=True,
)
CLI_FLAGS = [
    "--scenario",
    "torch-minigrid-history-learnability",
    "--invariant-history",
    "--qualification-method",
    "pc",
    "--joint-catalogue",
    "--distinct-probe",
    "--old-task-readouts",
    "--return-to-a",
]


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        core_attempts=0,
        core_rows=0,
        configured_latent_row_iterations=0,
        attempted_latent_row_iterations=0,
        transition_attempts=0,
        transition_completed=0,
        prediction_calls=0,
        prediction_rows=0,
    )
    proposal = backend.bounded_class_weighted_pc_update

    def propose(features, targets, parameters, **options):
        work["core_attempts"] += 1
        work["core_rows"] += len(features)
        work["configured_latent_row_iterations"] += len(features) * options["inference_steps"]
        try:
            result = proposal(features, targets, parameters, **options)
        except TrainingUpdateFailed as error:
            work["attempted_latent_row_iterations"] += error.work.get(
                "attempted_latent_row_iterations", 0
            )
            raise
        work["attempted_latent_row_iterations"] += (
            result.diagnostics.attempted_latent_row_iterations
        )
        return result

    monkeypatch.setattr(backend, "bounded_class_weighted_pc_update", propose)
    predict = InvariantTorchFeatureLearner.predict

    def predictions(owner, features):
        work["prediction_calls"] += 1
        work["prediction_rows"] += len(features)
        return predict(owner, features)

    monkeypatch.setattr(InvariantTorchFeatureLearner, "predict", predictions)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        original = kind._apply_training_gradients

        def transition(head, *args, _original=original, **options):
            work["transition_attempts"] += 1
            result = _original(head, *args, **options)
            work["transition_completed"] += 1
            return result

        monkeypatch.setattr(kind, "_apply_training_gradients", transition)
    yield work
    for key, value in work.items():
        record_property(key, value)


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "pc_relax64", "neutral_cpc_relax64"])
@pytest.mark.parametrize("weighted_catalogue", [False, True])
def test_should_preserve_exact_legacy_descriptor_when_option_is_none(
    no_resources, method, weighted_catalogue
):
    flags = {**FLAGS, "qualification_method": method, "weighted_catalogue": weighted_catalogue}
    assert pilot.describe_protocol(**flags) == pilot.describe_protocol(
        **flags, probe_inference=None
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cap", [2, 64])
def test_should_freeze_matched_objective_recipes_without_tolerance_fitting(mode, cap):
    assert recipes.MODES == MODES
    settings = recipes.policy(mode, cap)
    assert asdict(settings) == dict(
        min_steps=cap if mode == "fixed-weighted" else 1,
        gradient_tolerance=None if mode == "fixed-weighted" else 1e-6,
        relative_energy_tolerance=None,
    )
    assert recipes.policy(mode, cap) is not settings


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "pc_relax64", "neutral_cpc_relax64"])
@pytest.mark.parametrize("weighted_catalogue", [False, True])
def test_should_describe_explicit_recipes_before_runtime_resources(
    no_resources, mode, method, weighted_catalogue
):
    flags = {**FLAGS, "qualification_method": method, "weighted_catalogue": weighted_catalogue}
    baseline = pilot.describe_protocol(**flags)
    actual = pilot.describe_protocol(**flags, probe_inference=mode)
    assert actual["seed"] == baseline["seed"]
    assert actual["methods"] == baseline["methods"]
    assert actual["inference_steps"] == baseline["inference_steps"]
    assert (
        actual["planned_work"]["native_train_calls"]
        == baseline["planned_work"]["native_train_calls"]
    )
    assert actual["planned_work"]["training_rows"] == baseline["planned_work"]["training_rows"]
    assert actual["protocol_id"] != baseline["protocol_id"]
    for key in (
        "learning_rate",
        "momentum",
        "inference_learning_rate",
        "metric_direction",
        "checkpoint_updates",
        "joint_catalogue",
        "probe_support",
    ):
        assert actual[key] == baseline[key], key
    assert actual["probe_inference"]["mode"] == mode
    assert actual["probe_inference"]["policy"] == asdict(
        recipes.policy(mode, baseline["inference_steps"])
    )
    assert actual["probe_inference"]["max_steps"] == baseline["inference_steps"]
    assert (
        actual["probe"]["objective"]
        == actual["return_phase"]["objective"]
        == backend.TORCH_BOUNDED_WEIGHTED_PC_ID
    )
    if weighted_catalogue:
        excluded = {"probe_objective", "isolation", "diagnostic"}
        assert {
            key: value for key, value in actual["weighted_catalogue"].items() if key not in excluded
        } == {
            key: value
            for key, value in baseline["weighted_catalogue"].items()
            if key not in excluded
        }


@pytest.mark.parametrize("mode", [True, False, 1, "adaptive", [], {}])
def test_should_reject_invalid_option_values_before_resource_construction(no_resources, mode):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(
                public_history=True,
                invariant_history=True,
                qualification_method="pc",
                joint_catalogue=True,
                distinct_probe=True,
                old_task_readouts=True,
                return_to_a=True,
                probe_inference=mode,
            )


@pytest.mark.parametrize(
    "changed",
    [
        dict(public_history=False),
        dict(invariant_history=False),
        dict(joint_catalogue=False),
        dict(qualification_method=None),
        dict(qualification_method="bp"),
        dict(qualification_method="gated_cpc"),
        dict(distinct_probe=False),
        dict(old_task_readouts=False),
        dict(return_to_a=False),
        dict(memory_control=True),
        dict(pc_qualification=True),
        dict(activation_health=True),
        dict(gradient_geometry=True),
        dict(prior_gradient_geometry=True),
        dict(pc_update_diagnostics=True),
        dict(probe_rollouts=True),
        dict(deferred_probe_rollouts=True),
        dict(train_action_margins=True),
    ],
)
def test_should_reject_incompatible_recipes_before_models_or_sources(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**FLAGS, **changed}, probe_inference="bounded-weighted")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_explicit_cli_recipe_to_the_existing_entry_point(
    monkeypatch, capsys, mode, describe
):
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
        ["pilot", *CLI_FLAGS, "--probe-inference", mode] + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**FLAGS, "probe_inference": mode}]
    assert json.loads(capsys.readouterr().out) == dict(status="complete")


@pytest.mark.parametrize(
    "arguments",
    [
        [item for item in CLI_FLAGS if item != "--joint-catalogue"],
        ["bp" if item == "pc" else item for item in CLI_FLAGS],
        [*CLI_FLAGS, "--probe-rollouts"],
    ],
)
def test_should_reject_invalid_cli_composition_without_dispatch(monkeypatch, arguments):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *arguments, "--probe-inference", "bounded-weighted", "--describe"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def probe_case(clock, method):
    source = pilot._history_learner(method)
    state = source.snapshot_state()
    sources = pilot._history_packets(training_sources())
    readouts = tuple(item for item in sources if item.context == "A")
    raw = raw_probe_data()
    data = tuple(pilot._history_packets(items) for items in raw)
    benchmark = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock)
    row = pilot._new_row(method)
    return source, state, readouts, data, benchmark, row


def invoke_probe(slot, source, state, readouts, data, benchmark, row, mode):
    return pilot._run_probe(
        slot,
        dict(source=source, row=row, budget=benchmark),
        state,
        data,
        sampler=Sampler(),
        deadline=100.0,
        check_boundary=lambda: None,
        training_readouts=readouts,
        return_to_a=True,
        catalogue_rows=54,
        probe_inference=mode,
    )


@pytest.mark.parametrize(
    "method",
    [
        "public_invariant_pc",
        "public_invariant_neutral_cpc",
        "public_invariant_pc_relax64",
        "public_invariant_neutral_cpc_relax64",
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_should_execute_original_f16_old_a_readouts_and_a16_without_changing_the_benchmark(
    case, monkeypatch, method, mode, work_accounting
):
    torch, clock = case
    source, state, readouts, data, benchmark, row = probe_case(clock, method)
    original = source.snapshot_state()
    inputs = pickle.dumps((readouts, data))
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    captured = []
    run = pilot.run_fresh_task_probe

    def observe(owner, supplied_state, training, evaluation, **options):
        captured.append((owner, options["budget"], options["return_budget"]))
        assert owner is not source and owner.source_role == "probe_train"
        assert_tree(torch, owner.snapshot_state(), state)
        return run(owner, supplied_state, training, evaluation, **options)

    monkeypatch.setattr(pilot, "run_fresh_task_probe", observe)
    slot = row["probes"][0]
    completed = invoke_probe(slot, source, state, readouts, data, benchmark, row, mode)
    completed()
    assert slot["status"] == slot["report"]["status"] == "complete"
    assert slot["charged_updates"] == slot["return_charged_updates"] == 16
    assert (
        slot["source_unchanged"]
        is slot["state_unchanged"]
        is slot["budget_binding_preserved"]
        is True
    )
    report, returned = slot["report"], slot["report"]["return_phase"]
    assert returned["status"] == "complete"
    assert report["completed_training_examples"] == returned["completed_training_examples"] == 144
    assert len(report["diagnostics"]) == len(returned["diagnostics"]) == 16
    cap = state.policy.native_policy.inference_steps
    for diagnostic in (*report["diagnostics"], *returned["diagnostics"]):
        assert len(diagnostic["native_work"]) == 32 and len(diagnostic["stream_work"]) == 7
        assert diagnostic["inference_policy"] == asdict(recipes.policy(mode, cap))
        assert (
            0
            <= diagnostic["inference"]["accepted_steps"]
            <= diagnostic["inference"]["attempted_steps"]
            <= cap
        )
        if mode == "fixed-weighted":
            assert diagnostic["inference"]["accepted_steps"] == cap
    assert len(captured) == 1
    template, fresh_budget, return_budget = captured[0]
    assert fresh_budget is not benchmark and return_budget is not fresh_budget
    assert fresh_budget.updates_completed == return_budget.updates_completed == 16
    assert benchmark.updates_completed == 0
    assert_tree(torch, template.snapshot_state(), state)
    assert_tree(torch, source.snapshot_state(), original)
    assert_tree(torch, state, original)
    assert pickle.dumps((readouts, data)) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    assert work_accounting["core_attempts"] == work_accounting["transition_completed"] == 32
    observed = recipes.sum_work([row])
    assert observed["successful_receipts"] == 32 and observed["failed_receipts"] == 0
    assert (
        observed["native_work"]["published_updates"]
        == observed["scheduling_work"]["published_updates"]
        == 32
    )
    original_work = report["diagnostics"][0]["native_work"]
    report["diagnostics"][0]["native_work"] = ()
    with pytest.raises(ValueError):
        completed()
    report["diagnostics"][0]["native_work"] = original_work
    completed()


def test_should_keep_failed_candidate_work_and_original_zero_budget_charge(
    case, monkeypatch, work_accounting
):
    torch, clock = case
    source, state, readouts, data, benchmark, row = probe_case(clock, "public_invariant_pc")
    slot = row["probes"][0]

    def fail(*args, **options):
        raise OSError("explicit probe energy interrupted")

    monkeypatch.setattr(core, "_energy", fail)
    with pytest.raises(ValueError):
        invoke_probe(slot, source, state, readouts, data, benchmark, row, "bounded-weighted")
    assert slot["status"] == "failed" and slot["charged_updates"] == 0
    paid = slot["report"]["failed_native_work"]
    assert len(paid["work"]) == 32 and len(paid["scheduling_work"]) == 7
    assert paid["work"]["energy_evaluations_attempted"] == 1
    assert paid["work"]["energy_evaluations_completed"] == 0
    assert paid["work"]["failure_work_may_be_unobserved"] is True
    assert paid["work"]["published_updates"] == paid["scheduling_work"]["published_updates"] == 0
    assert benchmark.updates_completed == 0 and work_accounting["core_attempts"] == 1
    assert_tree(torch, source.snapshot_state(), state)


def test_should_preserve_foreign_benchmark_changes_after_a_paid_probe_update(case, monkeypatch):
    torch, clock = case
    source, state, readouts, data, benchmark, row = probe_case(clock, "public_invariant_pc")
    slot = row["probes"][0]
    original = scheduled.train_bounded_probe_episode

    def change(candidate, packet, weights, **options):
        result = original(candidate, packet, weights, **options)
        with torch.no_grad():
            source._native._head.bias_output[0, 0] += 0.125
        return result

    monkeypatch.setattr(scheduled, "train_bounded_probe_episode", change)
    with pytest.raises(ValueError):
        invoke_probe(slot, source, state, readouts, data, benchmark, row, "bounded-weighted")
    expected = deepcopy(state)
    expected.native_state.head_state["bias_output"][0, 0] += 0.125
    assert_tree(torch, source.snapshot_state(), expected)
    assert slot["status"] == "failed" and slot["charged_updates"] == 1
    assert slot["report"]["completed_updates"] == 1 and benchmark.updates_completed == 0


def test_should_keep_partial_unknown_native_costs_separate_from_paid_schedule_costs():
    failure: dict[str, dict[str, int | bool]] = dict(
        work={"failure_work_may_be_unobserved": True},
        scheduling_work={"candidate_forks_attempted": 1, "published_updates": 0},
    )
    rows = [
        dict(
            probes=[
                dict(
                    report=dict(
                        attempted_updates=1, completed_updates=0, failed_native_work=failure
                    )
                )
            ]
        )
    ]
    result = recipes.sum_work(rows)
    assert result["successful_receipts"] == 0 and result["failed_receipts"] == 1
    assert result["native_work"] == failure["work"]
    assert result["scheduling_work"] == failure["scheduling_work"]
    assert result["failure_work_may_be_unobserved"] is True
    result["native_work"]["invented"] = 999
    assert "invented" not in failure["work"]
