"""Complete coupled state through the real delayed-label runtime and CLI."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock


def same_state(actual: CoupledSnapshot, expected: CoupledSnapshot) -> None:
    np.testing.assert_equal(asdict(actual), asdict(expected))


@pytest.mark.parametrize("circadian", [False, True])
def test_should_continue_complete_native_state_only_after_arrival_without_actor_mutation(circadian):
    factory = (
        CoupledCircadianPredictiveCodingNetwork if circadian else CoupledPredictiveCodingNetwork
    )
    model = factory(2, (3, 2), 23)
    features = np.array([[0.3, -0.2], [-0.5, 0.4]])
    targets = np.array([[1.0], [0.0]])
    model.train_epoch(features, targets, 0.03, 2, 0.1)
    original = model.snapshot_state()
    direct = factory(2, (3, 2), 0)
    direct.restore_state(original)
    learner = CoupledLearner(
        model, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
    )
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    runtime = ActorShadowRuntime(
        learner,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    actor_before = runtime.actor.snapshot_state()
    prediction = runtime.actor.predict(features)
    runtime.record_experience(
        Experience("s0", "e0", 0, "a0", features, "train", ExperiencePermissions(training=True))
    )
    with pytest.raises(ValueError, match="role"):
        runtime.record_label(LabelArrival("bad", "s0", "e0", 2, "a0", targets, "final_test"))
    runtime.record_label(LabelArrival("label0", "s0", "e0", 2, "a0", targets))
    clock.advance_to(1)
    assert runtime.train_ready() == ()
    same_state(runtime.candidate_snapshot().state, original)
    assert budget.updates_completed == 0
    clock.advance_to(2)
    applied = runtime.train_ready()
    expected = direct.train_epoch(features, targets, 0.03, 2, 0.1)
    assert len(applied) == 1 and applied[0].diagnostic.value == expected.energy
    assert applied[0].diagnostic.definition == expected.energy_definition
    assert budget.updates_completed == 1 and runtime.train_ready() == ()
    same_state(runtime.candidate_snapshot().state, direct.snapshot_state())
    same_state(runtime.actor.snapshot_state().state, actor_before.state)
    same_state(model.snapshot_state(), original)
    same_state(learner.snapshot_state(), original)
    np.testing.assert_array_equal(runtime.actor.predict(features).prediction, prediction.prediction)
    assert runtime.actor.version == "a0"
    if circadian:
        wake = runtime.candidate_snapshot().state.wake_state
        assert wake is not None and wake.wake_updates == 2
        assert runtime.candidate_snapshot().wake_updates == 1


def test_should_run_coupled_neutral_cli_with_real_delayed_runtime():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_native_stream_demo", "--scenario", "coupled-neutral"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["protocol_id"] == "coupled_native_adapter_stream_v1"
    assert report["scope"] == "integration_smoke" and report["seed"] == 23
    pc, cpc = report["methods"]
    assert [pc["method"], cpc["method"]] == ["coupled_pc", "coupled_cpc"]
    for method in (pc, cpc):
        assert method["experiment_state"]["completed_updates"] == 4
        assert [e["applied_sample_ids"] for e in method["events"]] == [
            [],
            [],
            ["s0"],
            ["s1"],
            ["s2"],
            ["s3"],
        ]
        assert len(method["diagnostics"]) == 4
        assert {d["definition"] for d in method["diagnostics"]} == {
            "numpy_coupled_binary_energy_v1"
        }
    assert pc["events"] == cpc["events"]
    assert pc["diagnostics"] == cpc["diagnostics"]
    assert pc["final_candidate_predictions"] == cpc["final_candidate_predictions"]
    assert pc["native_adaptation"] is None
    assert cpc["native_adaptation"]["wake_updates"] == 4
    assert any(value > 0 for value in cpc["native_adaptation"]["chemistry"])
    assert cpc["native_adaptation"]["gates"] == [1.0] * 4
