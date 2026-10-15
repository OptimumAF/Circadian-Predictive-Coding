"""Owned matched BP ports, state continuation and real label/actor boundaries."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.matched_backprop import MatchedBackpropNetwork, MatchedBackpropSnapshot


def same_state(actual: MatchedBackpropSnapshot, expected: MatchedBackpropSnapshot) -> None:
    np.testing.assert_equal(asdict(actual), asdict(expected))


@pytest.mark.parametrize("widths", [(3,), (3, 2), (3, 2, 3, 2)])
def test_should_match_native_updates_and_preserve_owned_forks_and_restore(widths):
    parameters = CoupledPredictiveCodingNetwork(2, widths, 23).snapshot_parameters()
    native = MatchedBackpropNetwork(parameters)
    x, y = np.array([[0.3, -0.2], [-0.5, 0.4]]), np.array([[1.0], [0.0]])
    native.train_epoch(x, y, 0.03)
    original = native.snapshot_state()
    direct = MatchedBackpropNetwork(original.parameters)
    learner = MatchedBackpropLearner(native, learning_rate=0.03)
    fork = learner.fork()
    saved = learner.snapshot_state()
    saved.parameters.output_weight.fill(999)
    for _ in range(3):
        expected = direct.train_epoch(x, y, 0.03)
        observed = learner.train_batch(x, y)
        assert observed.value == expected.loss
        assert observed.definition == expected.loss_definition
        same_state(learner.snapshot_state(), direct.snapshot_state())
    same_state(native.snapshot_state(), original)
    same_state(fork.snapshot_state(), original)
    learner.restore_state(original)
    learner.train_batch(x, y)
    fork.train_batch(x, y)
    same_state(learner.snapshot_state(), fork.snapshot_state())
    prediction = learner.predict(x)
    prediction.fill(999)
    np.testing.assert_array_equal(learner.predict(x), fork.predict(x))


def test_should_keep_serving_actor_and_source_unchanged_until_allowed_candidate_update():
    parameters = CoupledPredictiveCodingNetwork(2, (3, 2), 23).snapshot_parameters()
    native = MatchedBackpropNetwork(parameters)
    learner = MatchedBackpropLearner(native, learning_rate=0.03)
    original = native.snapshot_state()
    x, y = np.array([[0.3, -0.2]]), np.array([[1.0]])
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
    runtime.record_experience(
        Experience("s0", "e0", 0, "a0", x, "train", ExperiencePermissions(training=True))
    )
    with pytest.raises(ValueError, match="role"):
        runtime.record_label(LabelArrival("bad", "s0", "e0", 2, "a0", y, "final_test"))
    runtime.record_label(LabelArrival("label", "s0", "e0", 2, "a0", y))
    clock.advance_to(1)
    assert runtime.train_ready() == () and budget.updates_completed == 0
    same_state(runtime.candidate_snapshot().state, original)
    clock.advance_to(2)
    applied = runtime.train_ready()
    direct = MatchedBackpropNetwork(parameters)
    expected = direct.train_epoch(x, y, 0.03)
    assert len(applied) == 1 and applied[0].diagnostic.value == expected.loss
    assert budget.updates_completed == 1 and runtime.train_ready() == ()
    same_state(runtime.candidate_snapshot().state, direct.snapshot_state())
    same_state(runtime.actor.snapshot_state().state, original)
    same_state(native.snapshot_state(), original)
    same_state(learner.snapshot_state(), original)
    assert runtime.actor.version == "a0"


@pytest.mark.parametrize("rate", [True, 0, float("inf"), 10**1000])
def test_should_refuse_invalid_adapter_rate(rate: Any):
    parameters = CoupledPredictiveCodingNetwork(2, (3,), 23).snapshot_parameters()
    with pytest.raises(ValueError):
        MatchedBackpropLearner(MatchedBackpropNetwork(parameters), learning_rate=rate)


def test_should_refuse_foreign_model_and_snapshot_without_changing_state():
    pc: Any = CoupledPredictiveCodingNetwork(2, (3,), 23)
    with pytest.raises(ValueError, match="exact native"):
        MatchedBackpropLearner(pc, learning_rate=0.03)
    learner = MatchedBackpropLearner(
        MatchedBackpropNetwork(pc.snapshot_parameters()), learning_rate=0.03
    )
    original = learner.snapshot_state()
    with pytest.raises(ValueError):
        learner.restore_state(pc.snapshot_state())
    same_state(learner.snapshot_state(), original)


def test_should_run_all_matched_methods_through_actual_delayed_cli():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_native_stream_demo", "--scenario", "matched-baselines"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["protocol_id"] == "matched_binary_adapter_stream_v1"
    assert report["scope"] == "integration_smoke" and report["seed"] == 23
    assert report["settings"]["hidden_dims"] == [3, 2]
    assert report["settings"]["latent_steps_per_update"] == {"bp": 0, "pc": 2, "cpc": 2}
    bp, pc, cpc = report["methods"]
    assert [m["method"] for m in (bp, pc, cpc)] == ["matched_bp", "coupled_pc", "coupled_cpc"]
    assert bp["initial_candidate_predictions"] == pc["initial_candidate_predictions"]
    assert pc["initial_candidate_predictions"] == cpc["initial_candidate_predictions"]
    source = CoupledPredictiveCodingNetwork(2, (3, 2), 23).snapshot_parameters()
    np.testing.assert_equal(
        report["initial_parameters"],
        {
            "hidden_weights": [w.tolist() for w in source.hidden_weights],
            "hidden_biases": [b.tolist() for b in source.hidden_biases],
            "output_weight": source.output_weight.tolist(),
            "output_bias": source.output_bias.tolist(),
        },
    )
    for method in (bp, pc, cpc):
        assert method["experiment_state"]["completed_updates"] == 4
        assert [e["applied_sample_ids"] for e in method["events"]] == [
            [],
            [],
            ["s0"],
            ["s1"],
            ["s2"],
            ["s3"],
        ]
        assert all(
            e["actor_predictions"] == bp["initial_candidate_predictions"] for e in method["events"]
        )
    assert {d["definition"] for d in bp["diagnostics"]} == {"numpy_matched_binary_bce_preupdate_v1"}
    assert pc["diagnostics"] == cpc["diagnostics"]
    assert pc["final_candidate_predictions"] == cpc["final_candidate_predictions"]
