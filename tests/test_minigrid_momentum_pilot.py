"""Existing warm runner opt-in control; real inbox/runtime with fake native IO."""

from copy import deepcopy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_minigrid_reward_smoke as cli
from src.adapters import minigrid_reward_pilot as pilot
from src.adapters import minigrid_momentum_control as control
from src.adapters import minigrid_warm_readouts as diagnostics
from src.adapters.torch_recurrent_zero_loss import TORCH_RECURRENT_ZERO_LOSS_ID
from src.core.learner_ports import TrainingDiagnostic
from test_minigrid_warm_readout_pilot import readout_case as readout_case, scored
from test_minigrid_warm_reward import Native, ports as ports
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.fixture
def paired(monkeypatch, readout_case):
    case = readout_case
    work = case.work
    work.update(controls=[], control_runtimes=[])
    monkeypatch.setattr(control, "monotonic", lambda: case.clock[0])
    monkeypatch.setattr(control, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    original_runtime = control.ActorShadowRuntime

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        work["control_runtimes"].append(result)
        return result

    def zero(owner, episode):
        assert len(work["reward_calls"]) == 8
        before = owner.snapshot_state()
        work["controls"].append((deepcopy(episode), before))
        if work.get("fail_control") == len(work["controls"]):
            raise RuntimeError("native control refused")
        for i, (key, parameter) in enumerate(owner.state.parameters.items()):
            buffer = owner.state.optimizer_state["state"][i]["momentum_buffer"] * 0.9
            owner.state.optimizer_state["state"][i]["momentum_buffer"] = buffer
            owner.state.parameters[key] = parameter - 0.03 * buffer
            owner.state.gradients[key] = np.zeros_like(parameter)
        owner.state.completed_updates += 1
        owner.state.training_examples += len(episode.transitions)
        callback = work.get("during_control")
        if callback:
            callback(len(work["controls"]) - 1)
        return TrainingDiagnostic(TORCH_RECURRENT_ZERO_LOSS_ID, 0.0)

    def evaluate(
        source,
        state,
        *,
        method,
        updates,
        native_training_examples,
        layouts,
        role,
        sources,
        budget,
        check_boundary,
    ):
        assert len(work["controls"]) == 8 and len(work["reward_calls"]) == 8
        assert role == "train" and sources is case.sources and len(layouts) == 3
        assert source is work["runtimes"][0]._actor._learner or source.completed_updates == 128
        assert method in ("recurrent_warm_reward", "recurrent_warm_momentum_control")
        assert (
            state.completed_updates == updates
            and state.training_examples == native_training_examples
        )
        assert budget.budget.max_training_updates == 0
        check_boundary()
        work["readouts"].append(dict(method=method, state=deepcopy(state), budget=budget))
        current = scored(updates, native_training_examples)
        current["method"] = method
        for score in current["scores"]:
            score["ce"] = 0.25 if method.endswith("momentum_control") else 0.5
        if work.get("malformed_control_readout") and method.endswith("momentum_control"):
            current["scores"] = current["scores"][:-1]
        callback = work.get("during_paired_readout")
        if callback:
            callback(len(work["readouts"]) - 1)
        check_boundary()
        return current

    monkeypatch.setattr(control, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(Native, "train_zero_loss_episode", zero, raising=False)
    monkeypatch.setattr(diagnostics, "evaluate_checkpoint", evaluate)
    return SimpleNamespace(case=case, work=work)


def run():
    return pilot.run_pilot(warm_start=True, train_readouts=True, momentum_control=True)


def test_should_complete_shared_warm_control_after_reward_before_distinct_train_readouts(paired):
    work = paired.work
    result = run()
    assert result["status"] == "complete", result["reason"]
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert len(work["reward_calls"]) == len(work["controls"]) == 8
    for (episode, before), reward in zip(work["controls"], work["reward_calls"], strict=True):
        assert pickle.dumps(episode) == pickle.dumps(reward["episode"])
        assert before.completed_updates == reward["before"].completed_updates
        assert before.training_examples == reward["before"].training_examples
    assert work["control_runtimes"][0]._candidate is not work["runtimes"][0]._candidate
    assert [r["method"] for r in work["readouts"]] == [
        "recurrent_warm_reward",
        "recurrent_warm_reward",
        "recurrent_warm_momentum_control",
    ]
    assert [r["state"].completed_updates for r in work["readouts"]] == [128, 136, 136]
    assert [r["state"].training_examples for r in work["readouts"]] == [1152, 1168, 1168]
    assert result["wake_updates"] == 8 and result["total_new_native_updates"] == 144
    assert result["total_new_training_rows"] == 1184
    assert result["total_new_work"]["control_repeated_observation_exposure_rows"] == 16
    assert result["total_new_work"]["environment_steps"] == 60
    assert all(
        v["ce_gain"] == 0.25 for v in result["train_diagnostics"]["control_minus_reward"].values()
    )
    reward_budget = work["budget"]
    assert reward_budget.updates_completed == 8 and reward_budget.last_clock == 10.0
    assert work["control_runtimes"][0]._budget is not reward_budget


def test_should_preserve_paid_reward_and_control_prefix_and_skip_readouts_after_control_failure(
    paired,
):
    paired.work["fail_control"] = 3
    result = run()
    assert result["status"] == "incomplete" and "native control refused" in result["reason"]
    assert result["wake_updates"] == 8
    assert result["momentum_control"]["status"] == "incomplete"
    assert result["total_new_native_updates"] == 138
    assert result["total_new_training_rows"] == 1172
    assert result["total_new_work"]["control_repeated_observation_exposure_rows"] == 4
    assert result["total_new_work"]["failure_work_may_be_unobserved"]
    assert not paired.work["readouts"]


def test_should_refuse_mutated_reward_evidence_during_control_before_next_update(paired):
    def corrupt(index):
        if index == 0:
            paired.work["reports"][0]["training"][0]["receipt"]["update_number"] = 99

    paired.work["during_control"] = corrupt
    result = run()
    assert result["status"] == "incomplete" and len(paired.work["controls"]) == 1
    assert not paired.work["readouts"]
    assert result["training"][0]["receipt"]["update_number"] == 1


def test_should_refuse_completed_control_evidence_corruption_during_first_readout(paired):
    def corrupt(index):
        if index == 0:
            paired.work["reports"][0]["momentum_control"]["status"] = "replaced"

    paired.work["during_paired_readout"] = corrupt
    result = run()
    assert result["status"] == "incomplete" and len(paired.work["readouts"]) == 1
    assert result["momentum_control"]["status"] == "complete"


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_bool_control_before_resources(monkeypatch, value):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.run_pilot(warm_start=True, train_readouts=True, momentum_control=value)


@pytest.mark.parametrize("warm,readouts", [(False, False), (True, False), (False, True)])
def test_should_require_warm_train_readout_route_before_resources(monkeypatch, warm, readouts):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(warm_start=warm, train_readouts=readouts, momentum_control=True)


def test_should_describe_new_control_resource_free_and_preserve_old_default(monkeypatch, capsys):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    assert pilot.describe_protocol(warm_start=True, train_readouts=True) == pilot.describe_protocol(
        warm_start=True, train_readouts=True, momentum_control=False
    )
    assert cli.main(["--warm-start", "--train-readouts", "--momentum-control", "--describe"]) == 0
    descriptor = json.loads(capsys.readouterr().out)
    assert descriptor["planned_work"]["native_train_calls"] == 144
    assert descriptor["planned_work"]["maximum_control_repeated_observation_rows"] == 128
    assert (
        descriptor["limits"]["body_seconds"] == 180
        and descriptor["limits"]["control_phase_seconds"] == 45
    )
    assert descriptor["planned_work"]["final_episodes"] == 0
    assert descriptor["limits"]["train_readout_phase_seconds"] == 90


def test_should_keep_paid_readouts_but_refuse_malformed_control_comparison(paired):
    paired.work["malformed_control_readout"] = True
    report = run()
    assert report["status"] == "incomplete"
    assert len(paired.work["readouts"]) == 3
    diagnostics = report["train_diagnostics"]
    assert diagnostics["status"] == "incomplete"
    assert diagnostics["endpoint_changes"] is None
    assert diagnostics["control_endpoint_changes"] is None
    assert diagnostics["control_minus_reward"] is None
    assert diagnostics["work"]["raw_score_rows_completed"] == 162


def test_should_prove_reward_evidence_after_completed_control_snapshot_callback(
    monkeypatch, paired
):
    work = paired.work
    snapshot = Native.snapshot_state
    armed: list[int] = []
    corrupted: list[bool] = []

    def last_control_callback(owner):
        result = snapshot(owner)
        if armed and not corrupted and owner is work["control_runtimes"][0]._candidate._learner:
            corrupted.append(True)
            work["reports"][0]["training"][0]["receipt"]["update_number"] = 99
        return result

    work["during_paired_readout"] = lambda index: armed.append(index)
    monkeypatch.setattr(Native, "snapshot_state", last_control_callback)
    result = run()
    assert corrupted == [True] and result["status"] == "incomplete"
    assert len(work["controls"]) == 8 and len(work["readouts"]) == 1
    assert result["training"][0]["receipt"]["update_number"] == 1
