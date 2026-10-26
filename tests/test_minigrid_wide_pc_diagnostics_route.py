"""Scripted H32 composition guards, not native tensor validation or learning.

The snapshot port checks the declared H32 policy and exact clocks. Accepted native
snapshot validation is reused separately; these tests instantiate no Torch model.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
import random

import pytest

from src.adapters import minigrid_gradient_readouts as gradients
from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_joint_catalogue_route import forbidden
from test_minigrid_learning_pilot import Learner, same_state
from test_minigrid_pc_update_route import install_reader, native_payload, sources


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(Learner, "work", {})
    monkeypatch.setattr(pilot, "_same_state", same_state)
    monkeypatch.setattr(gradients, "_same_state", same_state)
    for name in ("require_torch", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    admissions = []

    def snapshot(value, point, method, *, catalogue_rows=None, wide_neural=False):
        assert wide_neural is True and catalogue_rows == 54
        assert type(value.policy) is InvariantTorchFeaturePolicy
        assert value.policy.native_policy == TorchFeaturePolicy(
            method.removeprefix("public_invariant_"), 296, 32, 17, num_classes=7
        )
        if type(value.completed_updates) is not int or value.completed_updates != point:
            raise ValueError("scripted native update clock differs")
        if type(value.training_examples) is not int or value.training_examples != 54 * point:
            raise ValueError("scripted native row clock differs")
        admissions.append((point, value.training_examples))
        return deepcopy(value)

    monkeypatch.setattr(pilot, "_invariant_snapshot", snapshot)
    return {"admissions": admissions}


def prepare(case, method="pc", point=0):
    source = Learner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 32, 17, num_classes=7))
    )
    state = source.snapshot_state()
    state.completed_updates, state.training_examples = point, 54 * point
    case.update(source=source, state=state, sources=sources(), method=f"public_invariant_{method}")
    return case


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
            "pc_update_diagnostics": True,
            "catalogue_rows": 54,
            "wide_neural": True,
            **changed,
        },
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("point", [0, 128])
def test_should_route_whole_original_train_catalogue_on_owned_h32_checkpoint(
    monkeypatch, case, method, point
):
    prepare(case, method, point)
    calls = install_reader(monkeypatch)
    before = pickle.dumps((case["source"].snapshot_state(), case["state"], case["sources"]))
    result = run(case)
    assert result["status"] == "complete", result["reason"]
    assert len(calls) == 1
    assert calls[0]["state"].policy.hidden_dim == 32
    assert calls[0]["state"].completed_updates == point
    assert calls[0]["state"].training_examples == 54 * point
    assert calls[0]["state"] is not case["state"]
    assert result["source_unchanged"] is result["state_unchanged"] is True
    assert len(result["scores"]) == 6 and result["work"]["score_rows_completed"] == 54
    work = result["work"]["pc_update_diagnostics"]
    assert work["catalogue_calls_completed"] == 1 and work["projection_rows_completed"] == 54
    assert work["native_train_calls"] == work["published_updates"] == work["optimizer_steps"] == 0
    assert (point, 54 * point) in case["admissions"]
    assert (0, 0) in case["admissions"]
    assert pickle.dumps((case["source"].snapshot_state(), case["state"], case["sources"])) == before
    assert not Learner.work.get("trains")


@pytest.mark.parametrize(
    "changed",
    [
        {"updates": 32},
        {"updates": True},
        {"catalogue_rows": 9},
        {"catalogue_rows": 54.0},
        {"method": "public_invariant_bp"},
        {"method": "public_invariant_gated_cpc"},
        {"wide_neural": 1},
        {"pc_update_diagnostics": 1},
        {"weighted_catalogue": True},
        {"prior_gradient_geometry": True},
        {"train_action_margins": True},
    ],
)
def test_should_refuse_private_mixed_routes_before_snapshot_or_read(monkeypatch, case, changed):
    prepare(case)
    calls = install_reader(monkeypatch)
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    with pytest.raises(ValueError):
        run(case, **changed)
    assert not calls


@pytest.mark.parametrize("failure", ["native", "incomplete", "generic"])
def test_should_preserve_paid_scores_and_diagnostic_prefix_after_failure(
    monkeypatch, case, failure
):
    prepare(case, point=128)

    def fail(call):
        if failure == "generic":
            raise RuntimeError("scripted unobserved diagnostic return")
        partial = native_payload(complete=False)
        if failure == "native":
            raise GradientDiagnosticFailed(partial)
        return partial

    calls = install_reader(monkeypatch, fail)
    result = run(case)
    assert result["status"] == "failed" and len(calls) == 1
    assert len(result["scores"]) == 6 and result["work"]["score_rows_completed"] == 54
    work = result["work"]["pc_update_diagnostics"]
    assert work["catalogue_calls_attempted"] == 1
    assert work["prior_forward_rows_completed"] == (0 if failure == "generic" else 54)
    assert work["failure_work_may_be_unobserved"] is True
    assert result["work"]["failure_work_may_be_unobserved"] is True
    assert not Learner.work.get("trains")


@pytest.mark.parametrize(
    "fault", ["source", "checkpoint", "diagnostic_state", "input", "boundary", "clock"]
)
def test_should_refuse_changed_source_inputs_or_outer_guards_after_paid_diagnostic(
    monkeypatch, case, fault
):
    prepare(case)
    now = [1.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_wall_seconds=2), lambda: now[0]
    )

    def corrupt(call):
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

    calls = install_reader(monkeypatch, corrupt)

    def boundary():
        if calls and fault == "boundary":
            raise ValueError("original outer evidence guard refused")

    result = run(case, boundary, budget)
    assert result["status"] == "failed" and len(calls) == 1
    assert len(result["scores"]) == 6
    assert result["work"]["pc_update_diagnostics"]["catalogue_calls_completed"] == 1
    assert result["pc_update_diagnostics"][0]["diagnostic"]["status"] == "complete"
    assert budget.updates_completed == 0 and not Learner.work.get("trains")


@pytest.mark.parametrize("fault", ["work_owner", "payload_owner", "rows_type", "ids_owner"])
def test_should_refuse_replaced_paid_evidence_after_helper_returns(monkeypatch, case, fault):
    prepare(case)
    calls = install_reader(monkeypatch)
    original = gradients.run_pc_update_readouts
    published = []

    def observe(report, *args, **options):
        original(report, *args, **options)
        published.append(report)

    monkeypatch.setattr(gradients, "run_pc_update_readouts", observe)

    def boundary():
        if not published:
            return
        report = published.pop()
        record = report["pc_update_diagnostics"][0]
        if fault == "work_owner":
            report["work"]["pc_update_diagnostics"] = deepcopy(
                report["work"]["pc_update_diagnostics"]
            )
        elif fault == "payload_owner":
            record["diagnostic"] = deepcopy(record["diagnostic"])
        elif fault == "rows_type":
            record["diagnostic"]["rows"] = 54.0
        else:
            record["diagnostic"]["source_episode_ids"] = list(
                record["diagnostic"]["source_episode_ids"]
            )

    result = run(case, boundary)
    assert result["status"] == "failed" and "evidence changed" in result["reason"]
    assert len(calls) == 1 and len(result["scores"]) == 6
    assert type(result["pc_update_diagnostics"][0]["diagnostic"]["rows"]) is int
    assert result["work"]["pc_update_diagnostics"]["catalogue_calls_completed"] == 1


def test_should_keep_original_global_rng_guard_active(monkeypatch, case):
    prepare(case)
    rng = random.Random(17)
    initial = rng.getstate()
    calls = install_reader(monkeypatch, lambda call: (rng.random(), None)[1])

    def boundary():
        if rng.getstate() != initial:
            raise ValueError("scripted original global RNG witness changed")

    result = run(case, boundary)
    assert result["status"] == "failed" and "RNG witness changed" in result["reason"]
    assert len(calls) == 1 and len(result["scores"]) == 6
    assert result["work"]["pc_update_diagnostics"]["catalogue_calls_completed"] == 1


@pytest.mark.parametrize("fault", ["rows", "updates"])
def test_should_refuse_scripted_native_clock_changes_before_read(monkeypatch, case, fault):
    prepare(case)
    calls = install_reader(monkeypatch)
    if fault == "rows":
        case["state"].training_examples = 1
    else:
        case["state"].completed_updates = True
    with pytest.raises(ValueError):
        run(case)
    assert not calls


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_nontrain_packets_before_diagnostic(monkeypatch, case, role):
    prepare(case)
    calls = install_reader(monkeypatch)
    packet = case["sources"][0]
    case["sources"] = (
        replace(packet, role=role, episode=replace(packet.episode, role=role)),
        *case["sources"][1:],
    )
    with pytest.raises(ValueError):
        run(case)
    assert not calls
