"""Checkpoint128 integration: scalar failure boundaries and one native public case.

Native case195calls7515rows/P94x846, diagnostic CE15x135/backward9x81.
Synthetic collection only; no teacher, physics, final data or scientific claim.
"""

from copy import deepcopy
import json
from inspect import getclosurevars
from pathlib import Path
import sys

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app import toy_execution_budget as budgets
from src.shared.torch_runtime import require_torch
from test_minigrid_bp_update_route_work import FLAGS, record
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode, train_sources
from test_minigrid_joint_catalogue_route import CLI_FLAGS
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


def raw_sources():
    return tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )


def complete_record():
    value = record()
    value.update(
        reason=None,
        snapshot_unchanged=True,
        inputs_unchanged=True,
        rng_unchanged=True,
        steps=[dict(native_before_updates=n, native_after_updates=n + 1) for n in (128, 129, 130)],
    )
    return value


@pytest.fixture
def scalar_case(monkeypatch):
    monkeypatch.setattr(pilot, "monotonic", lambda: 1.0)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot.shared, "_invariant_feature_counter", lambda *a, **kw: None)
    observed = []

    def diagnose(state, wake, replay, *, budget, check_boundary):
        assert wake.episode_id == "minigrid-source-B-blue"
        assert [e.episode_id for e in replay] == ["minigrid-source-A-blue", "minigrid-source-A-red"]
        assert budget.updates_completed == budget.replay_examples_completed == 0
        assert budget.budget.max_training_updates == 3 and budget.budget.max_replay_examples == 18
        assert budget.budget.max_wall_seconds == 9.0
        observed.append(budget)
        for index in range(3):
            check_boundary()
            budget.before_update()
            if index:
                budget.record_replay(9)
            budget.record_update()
        return complete_record()

    monkeypatch.setattr(pilot, "diagnose_bp_updates", diagnose)
    return observed


def readout():
    return pilot._run_bp_checkpoint_readout(
        dict(row=dict(method="public_invariant_bp"), states=[(128, object())]),
        train_sources(),
        sampler=Sampler(),
        deadline=10.0,
        check_boundary=lambda: None,
    )


def test_should_keep_original_description_and_declare_separate_cycle(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    original = pilot.describe_protocol(**FLAGS)
    assert original == pilot.describe_protocol(**FLAGS, bp_update_diagnostics=False)
    selected = pilot.describe_protocol(**FLAGS, bp_update_diagnostics=True)
    for key in ("methods", "seed", "hidden_dim", "probe", "return_phase", "primary_comparisons"):
        assert original[key] == selected[key]
    work = selected["planned_work"]
    assert work["native_train_calls"] == 195 and work["training_rows"] == 7515
    assert work["replay_rows"] == 18 and work["final_episodes"] == 0


@pytest.mark.parametrize("key", list(FLAGS))
def test_should_refuse_partial_recipe_before_allocating(monkeypatch, key):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**FLAGS, key: False}, bp_update_diagnostics=True)


@pytest.mark.parametrize(
    "mutation", ["limits", "clock", "updates", "replay", "last_clock", "sampler"]
)
def test_should_reject_changed_completed_allowance(scalar_case, mutation):
    value, guard = readout()
    assert value["status"] == "complete", value["reason"]
    assert value["observed_budget_updates"] == 3 and value["observed_replay_rows"] == 18
    guard()
    budget = scalar_case[0]
    if mutation == "limits":
        budget.budget = deepcopy(budget.budget)
    elif mutation == "clock":
        budget.clock = lambda: 1.0
    elif mutation == "sampler":
        budget.process_rss_sampler = Sampler()
    else:
        setattr(
            budget,
            {"updates": "updates_completed", "replay": "replay_examples_completed"}.get(
                mutation, mutation
            ),
            0,
        )
    with pytest.raises(ValueError, match="allowance or charged prefix changed"):
        guard()


def test_should_retain_paid_incomplete_child_prefix(monkeypatch, scalar_case):
    def fail(state, wake, replay, *, budget, **kwargs):
        budget.record_update()
        value = complete_record()
        value.update(status="incomplete", reason="injected after first charged attempt")
        value["work"].update(
            native_train_calls_attempted=1,
            native_train_calls_returned=0,
            native_training_rows_attempted=9,
            native_training_rows_returned=0,
            charged_updates=1,
            charged_replay_rows=0,
            failure_work_may_be_unobserved=True,
        )
        raise pilot.GradientDiagnosticFailed(value)

    monkeypatch.setattr(pilot, "diagnose_bp_updates", fail)
    value, _ = readout()
    assert value["status"] == "incomplete"
    assert value["observed_budget_updates"] == 1
    assert value["work"]["native_train_calls_attempted"] == 1
    assert value["work"]["native_train_calls_returned"] == 0


def test_should_not_accept_replaced_allowance_during_enrollment(monkeypatch, scalar_case):
    attach = budgets.ToyBudgetSession.attach_memory

    def replaced(budget, sampler):
        attach(budget, sampler)
        budget.budget = deepcopy(budget.budget)

    monkeypatch.setattr(budgets.ToyBudgetSession, "attach_memory", replaced)
    monkeypatch.setattr(pilot, "diagnose_bp_updates", forbidden)
    value, _ = readout()
    assert value["status"] == "incomplete" and "enrollment" in value["reason"]
    assert value["budget_binding_preserved"] is False
    assert value["work"] is None


@pytest.mark.parametrize(
    "fault",
    ["child_failure", "late_budget_change", "prior_arm_change", "report_replace", "report_corrupt"],
)
def test_should_stop_before_next_work_and_preserve_observed_child(monkeypatch, scalar_case, fault):
    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda *a: raw_sources())
    monkeypatch.setattr(pilot.shared, "_readback", lambda *a: None)
    admitted, collected = [], []

    def train(row, *args, **kwargs):
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=6912,
            wake_calls_attempted=128,
            wake_calls_returned=128,
        )

        def guard():
            if admitted and fault == "prior_arm_change":
                raise ValueError("injected preceding benchmark changed")

        return dict(row=row, states=[(128, object())], guard=guard)

    diagnose = pilot.diagnose_bp_updates

    def observe(*args, **kwargs):
        value = diagnose(*args, **kwargs)
        admitted.append(value)
        if fault == "child_failure":
            value.update(status="incomplete", reason="injected paid child failure")
            value["work"]["failure_work_may_be_unobserved"] = True
            raise pilot.GradientDiagnosticFailed(value)
        return value

    def collect(boundary, *args, **kwargs):
        collected.append(True)
        if fault == "late_budget_change":
            scalar_case[0].updates_completed = 2
        else:
            # Read the existing parent's public evidence slot at the external
            # collection boundary; no new production accessor is warranted.
            common = getclosurevars(boundary).nonlocals["common_guard"]
            public = getclosurevars(common).nonlocals["report"]
            if fault == "report_replace":
                public["bp_update_diagnostics"] = deepcopy(public["bp_update_diagnostics"])
            else:
                public["bp_update_diagnostics"]["work"]["charged_updates"] = 0
        boundary()
        pytest.fail("changed completed observer admitted reserved F data")

    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "diagnose_bp_updates", observe)
    monkeypatch.setattr(pilot, "_collect_probe_sources", collect)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    # Main pilot has180s remaining; keep scalar fake's independent budget assertion.
    monkeypatch.setattr(pilot, "BODY_SECONDS", 9.0)
    result = pilot.run_pilot(**FLAGS, bp_update_diagnostics=True)
    assert result["status"] == "incomplete"
    assert len(admitted) == 1
    assert collected == ([] if fault in ("child_failure", "prior_arm_change") else [True])
    assert result["bp_update_diagnostics"]["work"]["native_train_calls_attempted"] == 3
    assert result["bp_update_diagnostics"]["work"]["charged_updates"] == 3
    assert result["work"]["native_train_calls"] == 131
    assert result["work"]["training_rows"] == 6939
    assert result["work"]["replay_rows"] == 18
    assert result["methods"][0]["charged_updates"] == 128


def test_should_run_native_public_cli_with_owned_checkpoint128_and_exact_costs(
    monkeypatch,
    capsys,
    record_property,
):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda *a: raw_sources())
    work = dict(
        updates=0,
        training_rows=0,
        predictions=0,
        prediction_rows=0,
        forwards=0,
        forward_rows=0,
        backwards=0,
    )
    train, predict = TorchFeatureLearner.train_batch, TorchFeatureLearner.predict
    ce, grad, backward = (
        torch.nn.functional.cross_entropy,
        torch.autograd.grad,
        torch.Tensor.backward,
    )

    def update(owner, features, targets):
        work["updates"] += 1
        work["training_rows"] += len(features)
        assert work["updates"] <= 195 and work["training_rows"] <= 7515
        return train(owner, features, targets)

    def score(owner, features):
        work["predictions"] += 1
        work["prediction_rows"] += len(features)
        assert work["predictions"] <= 94 and work["prediction_rows"] <= 846
        return predict(owner, features)

    def loss(logits, *args, **kwargs):
        work["forwards"] += 1
        work["forward_rows"] += len(logits)
        assert work["forwards"] <= 210 and work["forward_rows"] <= 7650
        return ce(logits, *args, **kwargs)

    def differentiation(original, *args, **kwargs):
        work["backwards"] += 1
        assert work["backwards"] <= 204
        return original(*args, **kwargs)

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", update)
    monkeypatch.setattr(TorchFeatureLearner, "predict", score)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", loss)
    monkeypatch.setattr(torch.autograd, "grad", lambda *a, **kw: differentiation(grad, *a, **kw))
    monkeypatch.setattr(
        torch.Tensor, "backward", lambda *a, **kw: differentiation(backward, *a, **kw)
    )
    arms, snapshots, observer = [], [], []
    native_arm, diagnose = pilot.shared._train_arm, pilot.diagnose_bp_updates

    def arm(*args, **kwargs):
        value = native_arm(*args, **kwargs)
        arms.append(value)
        snapshots.append(deepcopy(value["source"].snapshot_state()))
        return value

    def observe(*args, **kwargs):
        assert not observer
        assert args[0] is arms[0]["states"][-1][1]
        assert args[0].completed_updates == 128
        value = diagnose(*args, **kwargs)
        observer.append(value)
        return value

    def collect(boundary, *args, **kwargs):
        boundary()
        assert len(observer) == 1 and observer[0]["status"] == "complete"
        return raw_probe_data()

    monkeypatch.setattr(pilot.shared, "_train_arm", arm)
    monkeypatch.setattr(pilot, "diagnose_bp_updates", observe)
    monkeypatch.setattr(pilot, "_collect_probe_sources", collect)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *CLI_FLAGS, "--joint-catalogue", "--bp-update-diagnostics"]
    )
    try:
        cli.main()
        report = json.loads(capsys.readouterr().out)
        stage = Path("artifacts/runs/minigrid-bp-update-route-20261018")
        (stage / "native-public-report.json").write_text(
            json.dumps(report, indent=2), encoding="utf8"
        )
        assert report["status"] == "complete", report["reason"]
        extra = report["bp_update_diagnostics"]
        assert extra["status"] == "complete"
        assert extra["checkpoint_updates"] == 128 and extra["benchmark_training_examples"] == 6912
        assert extra["observed_budget_updates"] == 3 and extra["observed_replay_rows"] == 18
        actual = report["work"]
        assert actual["native_train_calls"] == 195 and actual["training_rows"] == 7515
        assert actual["replay_rows"] == 18 and not actual["observed_work_is_lower_bound"]
        assert (
            actual["total_score_forwards_completed"] == 94
            and actual["total_score_rows_completed"] == 846
        )
        assert actual["bp_update_diagnostics"]["forward_calls_completed"] == 15
        assert actual["bp_update_diagnostics"]["backward_calls_completed"] == 9
        row = report["methods"][0]
        assert row["charged_updates"] == 128 and row["native_training_examples"] == 6912
        assert row["work"]["native_train_calls"] == 195
        assert row["final_evaluation"]["status"] == "not_run"
        assert report["scientific_advantage_verified"] is False and report["comparisons"] is None
        assert _same_state(arms[0]["source"].snapshot_state(), snapshots[0])
        arms[0]["guard"]()
        assert work == dict(
            updates=195,
            training_rows=7515,
            predictions=94,
            prediction_rows=846,
            forwards=210,
            forward_rows=7650,
            backwards=204,
        )
    finally:
        for key, value in work.items():
            record_property(f"bp_route_native_{key}", value)
