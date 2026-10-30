"""Existing disposable probe with LS heads; no scientific comparison/capture."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import context_head_probe as head
from scripts import probe_ticket_policy_learning as recipe
from scripts.ticket_linear_probe import bind_readouts
from src.adapters.context_linear_fit_learner import (
    ContextLinearFitLearner,
    CONTEXT_LINEAR_RESIDUAL_ID,
)
from src.adapters.context_linear_fit_decision import ContextLinearFitDecisionPredictor
from src.adapters import linear_fit_learner as linear
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.context_linear_state import ContextLinearFitPolicy


@pytest.fixture(scope="module")
def inputs():
    spec = head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)
    return spec, head.build_head_cohorts(spec)


def source_for(inputs):
    spec, cohorts = inputs
    owner = ContextLinearFitLearner(
        ContextLinearFitPolicy(
            9, 6, (spec.vocabulary.index("policy_v1"), spec.vocabulary.index("policy_v2"))
        )
    )
    # Two bounded correctness-fixture TRAIN fits, separate from private probe dose.
    for batch in cohorts[2]:
        owner.train_batch(batch.features, batch.targets)
    return owner


def test_should_use_existing_probe_return_and_keep_source_actor_rng_inputs(inputs, monkeypatch):
    source = source_for(inputs)
    original, actor = source.snapshot_state(), source.fork()
    cohorts = inputs[1]
    pin = deepcopy(cohorts)
    rng = deepcopy(np.random.get_state())
    sessions = [ToyBudgetSession(recipe.HEAD_LIMITS, recipe.monotonic) for _ in range(2)]
    observed = []
    train = ContextLinearFitLearner.train_batch

    def train_private(owner, x, y):
        assert owner is not source and owner is not actor
        observed.append(len(x))
        assert any(
            np.array_equal(x, b.features) and np.array_equal(y, b.targets)
            for group in (cohorts[0], cohorts[3])
            for b in group
        )
        return train(owner, x, y)

    monkeypatch.setattr(ContextLinearFitLearner, "train_batch", train_private)
    result = run_fresh_task_probe(
        source,
        original,
        cohorts[0],
        cohorts[1],
        budget=sessions[0],
        checkpoint_updates=recipe.CHECKPOINTS,
        training_readouts=cohorts[2],
        return_training=cohorts[3],
        return_budget=sessions[1],
        decision_diagnostics=True,
    )
    assert result["status"] == result["return_phase"]["status"] == "complete", result
    assert observed == [3] * 16 + [6] * 16
    for row, rows in ((result, 48), (result["return_phase"], 96)):
        assert row["completed_updates"] == 16 and row["completed_training_examples"] == rows
        assert [p["updates"] for p in row["checkpoints"]] == [0, 4, 8, 16]
        assert all(d["definition"] == CONTEXT_LINEAR_RESIDUAL_ID for d in row["diagnostics"])
        assert row["training_readouts"]["status"] == "complete"
    assert source.snapshot_state() == actor.snapshot_state() == original
    assert recipe._same_value(cohorts, pin) and recipe._same_value(np.random.get_state(), rng)
    assert [s.updates_completed for s in sessions] == [16, 16]
    assert sessions[0].budget is recipe.HEAD_LIMITS and sessions[1].budget is recipe.HEAD_LIMITS


@pytest.mark.parametrize("role", ["train", "final_test", "outer_selection", "probe_test"])
def test_should_refuse_wrong_probe_training_role_before_fork(inputs, role, monkeypatch):
    source = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
    cohorts = inputs[1]
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("fork before role refusal"))
    with pytest.raises(ValueError, match="probe_train"):
        run_fresh_task_probe(
            source,
            source.snapshot_state(),
            (replace(cohorts[0][0], role=role), *cohorts[0][1:]),
            cohorts[1],
            budget=ToyBudgetSession(recipe.HEAD_LIMITS, recipe.monotonic),
            checkpoint_updates=recipe.CHECKPOINTS,
        )


def test_should_run_existing_ticket_arm_with_linear_initial_and_both_endpoints(inputs):
    source = source_for(inputs)
    state = source.snapshot_state()
    row = recipe.run_arm(
        source,
        state,
        inputs[1],
        inputs[0],
        lambda: None,
        None,
        "fixture",
        separate_context_heads=True,
        linear_fit=True,
    )
    recipe._require_arm(row, separate_context_heads=True, linear_fit=True)
    assert source.snapshot_state() == state
    assert len(row["episodes"]) == 18
    assert [(r["native_updates"], r["native_rows"]) for r in row["readout_work"]] == [
        (2, 12),
        (18, 60),
        (34, 156),
    ]
    assert [s["updates"] for s in row["sessions"]] == [16, 16]
    assert all(s["limits"] == recipe.asdict(recipe.HEAD_LIMITS) for s in row["sessions"])
    for reading in row["readout_work"]:
        work = reading["work"]
        assert work["decision_returns"] == work["routing_rows"]
        assert work["token_visits"] == 2 * work["routing_token_visits"]
        assert work["nominal_affine_terms"] == 48 * work["rows_attempted"]
    assert all(
        e["role"] == "probe_test"
        and e["episode"]["complete"]
        and e["episode"]["training_updates"] == 0
        for e in row["episodes"]
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(linear_fit=1),
        dict(separate_context_heads=False),
        dict(input_diagnostics=True),
        dict(head_tasks=recipe.TRANSFER_HEAD_TASKS),
    ],
)
def test_should_refuse_incompatible_linear_arm_before_fork(inputs, options, monkeypatch):
    source = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("fork"))
    kwargs = dict(separate_context_heads=True, linear_fit=True)
    kwargs.update(options)
    with pytest.raises(ValueError, match="linear_fit"):
        recipe.run_arm(
            source,
            source.snapshot_state(),
            inputs[1],
            inputs[0],
            lambda: None,
            None,
            "fixture",
            **kwargs,
        )


@pytest.mark.parametrize("fault", ["lost", "container", "pending_work", "source"])
def test_should_stop_after_first_paid_read_and_keep_original_costs(inputs, fault, monkeypatch):
    source = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
    row: dict[str, Any] = {}
    state, calls = source.snapshot_state(), []
    old_owner = source._state
    predict = ContextLinearFitDecisionPredictor.__call__

    def corrupt(reader, request):
        result = predict(reader, request)
        calls.append(True)
        if fault == "lost":
            raise RuntimeError("lost probe read")
        if fault == "container":
            row["readout_work"] = []
        elif fault == "pending_work":
            row["readout_work"][-1]["work"]["affine_attempts"] = 999
        else:
            source._state = deepcopy(source._state)
        return result

    monkeypatch.setattr(ContextLinearFitDecisionPredictor, "__call__", corrupt)
    monkeypatch.setattr(
        linear, "fit_categorical_linear", lambda *a: pytest.fail("fit after failed read")
    )
    with pytest.raises(ValueError):
        recipe.run_arm(
            source,
            state,
            inputs[1],
            inputs[0],
            lambda: None,
            None,
            "fixture",
            row=row,
            separate_context_heads=True,
            linear_fit=True,
        )
    assert calls == [True] and len(row["readout_work"]) == 1
    work = row["readout_work"][0]["work"]
    assert work["decision_attempts"] == work["decision_returns"] == work["unfitted_returns"] == 1
    assert work["affine_attempts"] == 0 and "probe" not in row
    assert [s["updates"] for s in row["sessions"]] == [0, 0]
    assert source.snapshot_state() == state
    assert (source._state is old_owner) == (fault != "source")


def test_should_keep_failed_private_fit_work_and_close_return_admission(inputs, monkeypatch):
    source = source_for(inputs)
    row: dict[str, Any] = {}
    state, calls = source.snapshot_state(), []
    fit = linear.fit_categorical_linear

    def lost(*args):
        calls.append(True)
        fit(*args)
        raise RuntimeError("lost private probe solve")

    monkeypatch.setattr(linear, "fit_categorical_linear", lost)
    with pytest.raises(ValueError, match="original paid session"):
        recipe.run_arm(
            source,
            state,
            inputs[1],
            inputs[0],
            lambda: None,
            None,
            "fixture",
            row=row,
            separate_context_heads=True,
            linear_fit=True,
        )
    assert calls == [True] and source.snapshot_state() == state
    assert row["probe"]["status"] == "failed"
    assert row["probe"]["attempted_updates"] == 1 and row["probe"]["completed_updates"] == 0
    assert [s["updates"] for s in row["sessions"]] == [0, 0]
    assert len(row["readout_work"]) == 1
    failure = row["probe"]["failed_native_work"]
    assert failure["work"]["fit_rows_attempted"] == 3 and failure["work"]["candidate_commits"] == 0


@pytest.mark.parametrize("fault", ["value", "replace", "header"])
def test_should_refuse_prior_episode_change_after_one_paid_private_update(
    inputs, fault, monkeypatch
):
    source = source_for(inputs)
    state, calls = source.snapshot_state(), []
    row: dict[str, Any] = {}
    train = ContextLinearFitLearner.train_batch
    original_episodes: list[Any] = []

    def corrupt(owner, x, y):
        result = train(owner, x, y)
        calls.append(True)
        original_episodes.extend((e, deepcopy(e)) for e in row["episodes"])
        if fault == "value":
            row["episodes"][0]["episode"]["success"] = not row["episodes"][0]["episode"]["success"]
        elif fault == "replace":
            row["episodes"] = deepcopy(row["episodes"])
        else:
            row["prior_native_updates"] += 1
        return result

    monkeypatch.setattr(ContextLinearFitLearner, "train_batch", corrupt)
    with pytest.raises(ValueError):
        recipe.run_arm(
            source,
            state,
            inputs[1],
            inputs[0],
            lambda: None,
            None,
            "fixture",
            row=row,
            separate_context_heads=True,
            linear_fit=True,
        )
    assert calls == [True] and source.snapshot_state() == state
    assert row["prior_native_updates"] == 2
    assert len(row["episodes"]) == 6
    assert all(
        a is old and recipe._same_value(a, pin)
        for a, (old, pin) in zip(row["episodes"], original_episodes, strict=True)
    )
    assert row["probe"]["status"] == "failed"
    assert row["probe"]["completed_updates"] == 1 and row["sessions"][0]["updates"] == 1
    assert row["sessions"][1]["updates"] == 0


def test_should_keep_default_arm_call_and_two_endpoint_contract(inputs, monkeypatch):
    source = ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
    calls = []

    def probe(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "literal"}

    monkeypatch.setattr(recipe, "run_fresh_task_probe", probe)
    monkeypatch.setattr(recipe, "_require_arm", lambda *a, **kw: None)
    r = recipe.run_arm(
        source, source.snapshot_state(), inputs[1], inputs[0], lambda: None, None, "fixture"
    )
    assert r["probe"] == {"status": "literal"}
    assert r["episodes"] == r["readout_work"] == []
    assert len(calls) == 1 and set(calls[0]) == {
        "budget",
        "checkpoint_updates",
        "training_readouts",
        "return_training",
        "return_budget",
        "decision_diagnostics",
        "check_boundary",
        "endpoint_observer",
    }


@pytest.mark.parametrize("fault", ["episode", "container", "header", "readout"])
@pytest.mark.parametrize("throws", [False, True])
def test_should_recheck_paid_evidence_after_returning_or_throwing_callback(fault, throws):
    episode, reading = {"success": True}, {"work": 3}
    row: dict[str, Any] = dict(
        mode="fixture",
        prior_native_updates=2,
        prior_native_rows=12,
        episodes=[episode],
        readout_work=[reading],
    )
    owners = row["episodes"], row["readout_work"]
    calls = []

    def mutate():
        calls.append(True)
        if fault == "episode":
            episode["success"] = False
        elif fault == "container":
            row["episodes"] = deepcopy(row["episodes"])
        elif fault == "header":
            row["prior_native_rows"] = 0
        else:
            reading["work"] = 0
        if throws:
            raise RuntimeError("throw after mutation")

    guard, episode_pins = bind_readouts(row, [(reading, deepcopy(reading))], mutate)
    episode_pins.append((episode, deepcopy(episode)))
    with pytest.raises(ValueError, match="paid endpoint"):
        guard()
    assert calls == [True]
    assert row["episodes"] is owners[0] and row["readout_work"] is owners[1]
    assert row["episodes"][0] is episode and row["readout_work"][0] is reading
    assert episode["success"] is True and reading["work"] == 3 and row["prior_native_rows"] == 12


def test_should_allow_unchanged_callback_and_preserve_original_exception():
    row = dict(
        mode="fixture", prior_native_updates=0, prior_native_rows=0, episodes=[], readout_work=[]
    )
    guard, _ = bind_readouts(row, [], lambda: None)
    guard()
    failure = RuntimeError("original callback failure")

    def fail():
        raise failure

    guard, _ = bind_readouts(row, [], fail)
    with pytest.raises(RuntimeError) as result:
        guard()
    assert result.value is failure
