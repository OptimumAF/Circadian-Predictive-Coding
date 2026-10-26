"""Mixed-objective disposable probes: one cached source, no benchmark training.

Prospective full module: one H2/D2/A2 initializer (8 normal values), three SGD
returns/6 rows including the source warmup, 19 predictions/38 rows, and five
gradient calls/10 rows (two are physical read-only diagnostics). These imply
24 native forwards/48 rows. No fit, head solve, SDK or adapted learner export.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as task
from src.core.continuous_demonstration import continuous_training_pairs
from test_continuous_probe import assert_state, episode


def forbidden(*args, **kwargs):
    pytest.fail("mixed probe admitted forbidden fit, head solve or private work")


def mixed_episode(name="A"):
    spec = replace(episode().spec, action_units=("m", "normalized_gripper"))
    value = episode(name, rows=2, declared=spec)
    return replace(
        value,
        transitions=tuple(
            replace(row, action=(row.action[0], -1.0 if index == 0 else 1.0))
            for index, row in enumerate(value.transitions)
        ),
    )


def mixed_inputs():
    old = (
        mixed_episode(),
        mixed_episode("B"),
        replace(mixed_episode("B"), episode_id="synthetic-C", context="context-C"),
    )
    plan = probe.ContinuousProbePlan(
        mixed_episode("Ftrain"), mixed_episode("Ftest"), diagnostics=True
    )
    return plan, old


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        SGD_attempts=0,
        SGD_returns=0,
        SGD_rows_attempted=0,
        SGD_rows_returned=0,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows_attempted=0,
        prediction_rows_returned=0,
        gradient_attempts=0,
        gradient_returns=0,
        gradient_rows_attempted=0,
        gradient_rows_returned=0,
        forward_attempts=0,
        forward_returns=0,
        forward_rows_attempted=0,
        forward_rows_returned=0,
    )
    initialize, train = (
        bp.NumpyContinuousBCLearner.__init__,
        bp.NumpyContinuousBCLearner.train_batch,
    )
    predict, gradient, forward = (
        bp.NumpyContinuousBCLearner.predict,
        task.continuous_bc_loss_and_gradients,
        task._forward,
    )

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def counted(original, kind):
        def call(*args, **options):
            values = args[1] if kind in ("SGD", "prediction") else args[0]
            counts[kind + "_attempts"] += 1
            counts[kind + "_rows_attempted"] += len(values)
            returned = original(*args, **options)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += len(values)
            return returned

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", counted(train, "SGD"))
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", counted(predict, "prediction"))
        patch.setattr(task, "continuous_bc_loss_and_gradients", counted(gradient, "gradient"))
        patch.setattr(task, "_forward", counted(forward, "forward"))
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("REMAINING_PROBE_LEDGER"):
        with Path(os.environ["REMAINING_PROBE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 8
    assert counts["SGD_attempts"] == counts["SGD_returns"] == 3
    assert counts["SGD_rows_attempted"] == counts["SGD_rows_returned"] == 6
    assert counts["prediction_attempts"] == counts["prediction_returns"] == 19
    assert counts["prediction_rows_returned"] == 38
    assert counts["gradient_attempts"] == counts["gradient_returns"] == 5
    assert counts["gradient_rows_returned"] == 10
    assert counts["forward_attempts"] == counts["forward_returns"] == 24
    assert counts["forward_rows_returned"] == 48


@pytest.fixture(scope="module")
def source(ledger):
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(mixed_episode().spec, hidden_dim=2, seed=17, binary_gripper_axis=1)
    )
    features, targets = continuous_training_pairs(mixed_episode())
    owner.train_batch(np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    return owner


@pytest.fixture(scope="module")
def successful(source):
    before, (plan, old) = source.snapshot_state(), mixed_inputs()
    input_seal = asdict(plan), tuple(asdict(item) for item in old)
    observed: dict[str, Any] = {}
    predictions: list[Any] = []
    training: list[Any] = []
    factory, predict, train = (
        probe.ActorShadowRuntime,
        bp.NumpyContinuousBCLearner.predict,
        bp.NumpyContinuousBCLearner.train_batch,
    )

    def runtime(*args, **options):
        made = factory(*args, **options)
        observed.update(runtime=made, budget=options["budget"])
        return made

    def predicted(owner, values):
        result = predict(owner, values)
        predictions.append((values.tolist(), result.tolist()))
        return result

    def trained(owner, values, targets):
        training.append((values.tolist(), targets.tolist()))
        return train(owner, values, targets)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(probe, "ActorShadowRuntime", runtime)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained)
        report = probe.run_continuous_probe(source, plan, training_readouts=old)
    return report, before, plan, old, input_seal, observed, predictions, training


def test_should_update_only_disposable_mixed_probe_and_preserve_warm_source_actor_and_inputs(
    source, successful
):
    report, before, plan, old, seal, observed, _, training = successful
    assert report["complete"] is True, report.get("error")
    assert_state(source.snapshot_state(), before)
    assert before.completed_updates == 1 and before.training_examples == 2
    assert any(np.any(value != 0) for value in before.velocity)
    assert_state(observed["runtime"].actor.snapshot_state().state, before)
    candidate = observed["runtime"].candidate_snapshot().state
    assert candidate.policy == before.policy and candidate.policy.binary_gripper_axis == 1
    assert (candidate.completed_updates, candidate.training_examples) == (2, 4)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(candidate.parameters, before.parameters, strict=True)
    )
    assert all(
        not np.shares_memory(a, b)
        for a, b in zip(
            candidate.parameters + candidate.velocity,
            before.parameters + before.velocity,
            strict=True,
        )
    )
    assert training == [
        (
            [list(row.observation) for row in plan.training.transitions],
            [list(row.action) for row in plan.training.transitions],
        )
    ]
    assert asdict(plan) == seal[0] and tuple(asdict(item) for item in old) == seal[1]
    assert report["native_updates"][0]["diagnostic"]["definition"] == task.MIXED_OBJECTIVE
    assert report["applied_updates"][0]["diagnostic"] == report["native_updates"][0]["diagnostic"]
    assert report["events"][0]["deferred_exact"] is True
    assert report["events"][0]["success_validated"] is True
    assert observed["budget"].updates_completed == 1
    assert "learner" not in report and "candidate" not in report
    json.dumps(report, allow_nan=False)


def test_should_score_three_old_train_tasks_and_probe_test_without_using_their_targets_for_updates(
    successful,
):
    report, _, plan, old, _, _, predictions, _ = successful
    episodes = (plan.evaluation,) + old + (plan.training,)
    names = ("fresh", "old_0", "old_1", "old_2", "fresh_training")
    assert len(predictions) == 10
    for phase, checkpoint in enumerate(report["checkpoints"]):
        assert checkpoint["complete"] is True and tuple(checkpoint["scores"]) == names
        for index, (name, item) in enumerate(zip(names, episodes, strict=True)):
            features, returned = predictions[5 * phase + index]
            assert features == [list(row.observation) for row in item.transitions]
            squared = [
                ((value[axis] - row.action[axis]) / scale) ** 2
                for value, row in zip(returned, item.transitions, strict=True)
                for axis, scale in enumerate((4.0, 1.0))
            ]
            score = checkpoint["scores"][name]
            assert score["role"] == item.role and score["episode_id"] == item.episode_id
            assert score["residual"]["overall"]["normalized_half_mse"] == pytest.approx(
                math.fsum(squared) / (2 * len(squared)), rel=1e-12, abs=1e-15
            )
    loss = lambda phase, name: report["checkpoints"][phase]["scores"][name]["residual"]["overall"][
        "normalized_half_mse"
    ]
    assert report["metrics"] == dict(
        fresh_acquisition_gain=loss(0, "fresh") - loss(1, "fresh"),
        fresh_training_gain=loss(0, "fresh_training") - loss(1, "fresh_training"),
        old_retention_loss={name: loss(1, name) - loss(0, name) for name in names[1:4]},
    )
    work = report["work"]
    assert work["score_rows_returned_by_role"] == dict(probe_test=4, train=12, probe_train=4)
    assert work["prediction_rows_returned"] == 20 and work["training_rows_completed"] == 2
    assert work["old_readout_input_numeric_bytes"] == 192
    assert work["reserved_numeric_bytes"] == 128
    assert work["step_diagnostics"]["gradient_rows_returned"] == 2
    # This diagnostic is physical MSE, distinct from the native mixed objective.
    assert report["update_diagnostics"][0]["result"]["objective_definition"] == task.OBJECTIVE


@pytest.mark.parametrize("target", [0.0, -0.5, 0.5])
def test_should_refuse_nonbinary_probe_training_target_before_fork(source, monkeypatch, target):
    plan, old = mixed_inputs()
    rows = plan.training.transitions
    training = replace(plan.training, transitions=(replace(rows[0], action=(2.0, target)), rows[1]))
    invalid = replace(plan, training=training)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(probe, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError, match="endpoint"):
        probe.run_continuous_probe(source, invalid, training_readouts=old)


@pytest.mark.parametrize("role", ["probe_test", "final_test"])
def test_should_refuse_heldout_third_readout_before_private_work(source, monkeypatch, role):
    plan, old = mixed_inputs()
    bad = old[:2] + (replace(old[2], role=role),)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(probe, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        probe.run_continuous_probe(source, plan, training_readouts=bad)


def test_should_retain_paid_mixed_update_when_final_third_train_score_fails(source, monkeypatch):
    before, (plan, old) = source.snapshot_state(), mixed_inputs()
    predict = bp.NumpyContinuousBCLearner.predict
    calls = 0

    def fail_after_return(owner, values):
        nonlocal calls
        result = predict(owner, values)
        calls += 1
        if calls == 9:
            raise RuntimeError("paid third TRAIN prediction returned before observer failure")
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", fail_after_return)
    report = probe.run_continuous_probe(source, plan, training_readouts=old)
    assert report["complete"] is False and "third TRAIN" in report["error"]
    assert_state(source.snapshot_state(), before)
    assert calls == 9 and len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["events"][0]["success_validated"] is True
    assert report["native_updates"][0]["diagnostic"]["definition"] == task.MIXED_OBJECTIVE
    assert report["work"]["training_rows_completed"] == 2
    assert report["work"]["prediction_attempts"] == 9 and report["work"]["prediction_returns"] == 8
    assert (
        report["checkpoints"][0]["complete"] is True
        and report["checkpoints"][1]["complete"] is False
    )
    assert tuple(report["checkpoints"][1]["scores"]) == ("fresh", "old_0", "old_1")
    assert "metrics" not in report and "candidate" not in report
