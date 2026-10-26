"""TRAIN margins reuse paid scores and preserve the existing runner/state ports.

Four native cases perform48 score forwards/432rows and zero updates in total.
All full-run/failure cases use existing scripted learner and teacher fixtures.
"""

import pickle
import random
from dataclasses import replace

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.core.categorical_probability_margins import probability_margins
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_learnability import bound, train_sources
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_pc_qualification import environment, FLAGS
from test_torch_feature_learner import assert_tree


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_add_native_train_margins_without_more_forwards_or_state_changes(method):
    torch = require_torch()
    source = bound(method)
    initial = source.snapshot_state()
    packets = train_sources()
    inputs = pickle.dumps(packets)
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    clock = lambda: 1.0
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock)
    binding = (budget.budget, budget.clock, budget.started_at, budget.last_clock)
    options = dict(
        method=f"public_invariant_{method}",
        updates=0,
        sources=packets,
        budget=budget,
        check_boundary=lambda: None,
    )
    old = pilot._score_checkpoint(source, initial, **options)
    new = pilot._score_checkpoint(source, initial, **options, train_action_margins=True)
    assert old["status"] == new["status"] == "complete"
    assert len(old["scores"]) == len(new["scores"]) == 6
    for plain, extended, packet in zip(old["scores"], new["scores"], packets, strict=True):
        assert {key: value for key, value in extended.items() if key != "action_margins"} == plain
        assert extended["action_margins"] == probability_margins(
            np.array(plain["probabilities"], dtype=np.float64), packet.targets, 7
        )
    assert {
        key: value for key, value in new["work"].items() if not key.startswith("action_margin_")
    } == old["work"]
    assert new["work"]["score_forward_calls_completed"] == 6
    assert new["work"]["score_rows_completed"] == 54
    assert (
        new["work"]["action_margin_calls_attempted"]
        == new["work"]["action_margin_calls_completed"]
        == 6
    )
    assert new["work"]["action_margin_rows_completed"] == 54
    assert new["work"]["action_margin_input_array_bytes"] == 54 * 64
    assert new["work"]["action_margin_seconds"] >= 0
    assert budget.updates_completed == 0
    assert binding == (budget.budget, budget.clock, budget.started_at, budget.last_clock)
    assert_tree(torch, source.snapshot_state(), initial)
    assert pickle.dumps(packets) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]


@pytest.mark.parametrize("fault", [None, "raise", "source", "detached_inputs"])
def test_should_keep_paid_train_scores_separate_from_probes_on_success_or_failure(
    monkeypatch, evaluator, fault
):
    fixture = environment(monkeypatch, evaluator)
    calls = []
    original = pilot.probability_margins

    def reduce(predictions, targets, count):
        calls.append((predictions.copy(), targets.copy()))
        if fault == "raise":
            raise ValueError("injected margin failure")
        result = original(predictions, targets, count)
        if fault == "source":
            fixture.work["arms"][0]["source"].state.parameters["weight"] += 9.0
        if fault == "detached_inputs":
            predictions[:] = 0.0
            targets[:] = 6.0
        return result

    monkeypatch.setattr(pilot, "probability_margins", reduce)
    report = pilot.run_pilot(**FLAGS, train_action_margins=True)
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    cost = report["work"]["train_action_margins"]
    if fault in ("raise", "source"):
        assert report["status"] != "complete"
        assert len(calls) == 1 and not fixture.work["probes"]
        raw = row["readouts"][0]
        assert raw["status"] == "failed" and len(raw["scores"]) == 1
        assert "action_margins" not in raw["scores"][0]
        assert raw["work"]["score_forward_calls_completed"] == 1
        assert cost["action_margin_calls_attempted"] == 1
        assert cost["action_margin_calls_completed"] == (fault == "source")
        if fault == "source":
            assert fixture.work["arms"][0]["source"].state.parameters["weight"][0, 0] == 9.0
    else:
        assert report["status"] == "complete"
        assert len(calls) == cost["action_margin_calls_completed"] == 30
        assert cost["action_margin_rows_completed"] == 270
        assert cost["action_margin_input_array_bytes"] == 270 * 64
        assert cost["additional_native_forwards"] == cost["optimizer_steps"] == 0
        assert len(fixture.work["probes"]) == 2
        for slot in row["probes"]:
            assert "action_margins" not in slot
        assert report["work"]["known_raw_score_forwards_completed"] == 30
        for raw in row["readouts"]:
            assert len(raw["scores"]) == 6
            for score in raw["scores"]:
                expected = probability_margins(
                    np.array(score["probabilities"], dtype=np.float64),
                    np.array(score["target_actions"], dtype=np.float64).reshape(-1, 1),
                    7,
                )
                assert score["action_margins"] == expected
    assert row["work"]["train_action_margins"] == cost


@pytest.mark.parametrize("role", ["final", "probe_train", "probe_test"])
def test_should_refuse_non_train_roles_before_any_learner_or_callback(role):
    packets = train_sources()
    invalid = (replace(packets[0], role=role), *packets[1:])

    def forbidden():
        pytest.fail("invalid TRAIN admission reached a callback")

    with pytest.raises(ValueError):
        pilot._score_checkpoint(
            object(),
            object(),
            method="public_invariant_bp",
            updates=0,
            sources=invalid,
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0),
            check_boundary=forbidden,
            train_action_margins=True,
        )
